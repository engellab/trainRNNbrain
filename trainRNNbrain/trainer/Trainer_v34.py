'''
Class which accepts RNN_torch and a task and has a mode to train RNN
'''
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torcheval.metrics.functional import r2_score, mean_squared_error
from tqdm.auto import tqdm
import json
from trainRNNbrain.utils import jsonify
from pathlib import Path

def iqr_scale(x: torch.Tensor, q_low: float = 0.25, q_high: float = 0.75, eps: float = 1e-12) -> torch.Tensor:
    v = x.reshape(-1)
    scale = torch.quantile(v, q_high) - torch.quantile(v, q_low)
    return torch.clamp(scale, min=eps)

def multi_iqr_scale(
    x: torch.Tensor,
    pairs=((0.1, 0.9), (0.25, 0.75), (0.4, 0.6), (0.1, 0.6), (0.4, 0.9)),
    eps: float = 1e-12,
) -> torch.Tensor:
    v = x.reshape(-1)
    if v.numel() == 0:
        return torch.tensor(float("nan"), device=v.device, dtype=v.dtype)
    v_sorted, _ = torch.sort(v)
    n = v_sorted.numel()
    idx = lambda q: min(max(int(round(q * (n - 1))), 0), n - 1)
    diffs = [v_sorted[idx(hi)] - v_sorted[idx(lo)] for lo, hi in pairs]
    return torch.clamp(torch.stack(diffs).mean(), min=eps)

def mad_scale(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    v = x.reshape(-1)
    scale = 1.4826 * torch.abs(torch.median(v - torch.median(v)))
    return torch.clamp(scale, min=eps)


class Trainer():
    def __init__(self, RNN, Task, optimizer,
                 max_iter=1000, tol=1e-12,
                 lambda_orth=0.3, orth_input_only=True,
                 lambda_sm=0.001, lambda_si=0.1, lambda_hi=0.1,
                 lambda_htvar=0.05, lambda_hlvar=0.9, lambda_cl=0.05,
                 inequality_method='hhi',
                 dropout=False, drop_rate=0.3, p=2):
        # CHG: removed 'criterion' entirely (unused)
        # CHG: fixed keys -> 'h_time_variance','h_local_variance'
        self.RNN = RNN
        self.Task = Task
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.tol =  tol
        self.lambda_orth, self.orth_input_only = lambda_orth, orth_input_only
        self.lambda_sm = lambda_sm
        self.lambda_si = lambda_si
        self.lambda_hi = lambda_hi
        self.lambda_hlvar = lambda_hlvar
        self.lambda_htvar = lambda_htvar
        self.lambda_cl = lambda_cl
        self.penalty_map = {
            "channel_overlap"  : (self.channel_overlap_penalty,    self.lambda_orth),
            "s_magnitude"      : (self.s_magnitude_penalty,        self.lambda_sm),
            "s_inequality"     : (self.s_inequality_penalty,       self.lambda_si),
            "h_inequality"     : (self.h_inequality_penalty,       self.lambda_hi),
            "h_time_variance"  : (self.h_time_variance_penalty,    self.lambda_htvar),
            "h_local_variance" : (self.h_local_variance_penalty,   self.lambda_hlvar),
            "clustering"       : (self.clustering_penalty,         self.lambda_cl)
        }
        self.loss_monitor = {k: [] for k in self.penalty_map}; self.loss_monitor["behavior"] = []
        self.indicators_monitor = {f"grad_norm_{k}": [] for k in self.penalty_map}; self.indicators_monitor["grad_norm_behavior"] = []
        self.p, self.dropout, self.drop_rate = p, dropout, drop_rate
        self.inequality_method = inequality_method
        self.participation = (1e-6 * torch.ones(self.RNN.N, device=self.RNN.device)) if self.dropout else None

    @staticmethod
    def to_item(x):
        return x.detach() if torch.is_tensor(x) else torch.as_tensor(x)

    @staticmethod
    def print_iteration_info(
            iter,
            train_loss,
            min_train_loss,
            val_loss=None,
            min_val_loss=None,
            train_direction='min',
            val_direction='min'
    ):
        """
        Print training and validation metrics with green highlight if improved.

        Args:
            iter: iteration number
            train_loss: current training loss or metric
            min_train_loss: best training value so far
            val_loss: current validation loss or metric (optional)
            min_val_loss: best validation value so far (optional)
            train_direction: 'min' or 'max' — defines if lower or higher is better for training
            val_direction: 'min' or 'max' — defines if lower or higher is better for validation
        """
        gr_prfx = '\033[92m'
        gr_sfx = '\033[0m'

        def is_improved(current, best, direction):
            if direction == 'min':
                return current <= best
            elif direction == 'max':
                return current >= best
            else:
                raise ValueError("Direction must be 'min' or 'max'")

        # Evaluate improvement
        train_improved = is_improved(train_loss, min_train_loss, train_direction)
        train_prfx = gr_prfx if train_improved else ''
        train_sfx = gr_sfx if train_improved else ''

        if val_loss is not None and min_val_loss is not None:
            val_improved = is_improved(val_loss, min_val_loss, val_direction)
            val_prfx = gr_prfx if val_improved else ''
            val_sfx = gr_sfx if val_improved else ''
            print(f"iteration {iter},"
                  f" train: {train_prfx}{np.round(train_loss, 6)}{train_sfx},"
                  f" val: {val_prfx}{np.round(val_loss, 6)}{val_sfx}")
        else:
            print(f"iteration {iter},"
                  f" train: {train_prfx}{np.round(train_loss, 6)}{train_sfx}")

    @staticmethod
    def normalized_concentration(C, mask, eps=1e-8):
        X = C[:, mask]
        P = X / (X.sum(1, keepdim=True) + eps)
        hhi = (P * P).sum(1)
        base = 1.0 / X.size(1)
        return ((hhi - base) / (1.0 - base)).clamp(0, 1)

    @staticmethod
    def gini_penalty(
            x,
            eps: float = 1e-8,
            tau: float = 0.0,
            detach_stats: bool = True,
            max_z: float = 8.0,  # clamp exponent for numeric stability
    ):
        v = x.reshape(-1)
        if v.numel() <= 1:
            return torch.zeros((), dtype=v.dtype, device=v.device)
        # Robust scale
        center = torch.mean(v)
        scale = multi_iqr_scale(v)

        if detach_stats:
            scale = scale.detach()
            center = center.detach()
        z = (v - center) / (scale + eps)
        if max_z is not None: z = torch.clamp(z, -max_z, max_z)
        u = torch.exp(z)

        mu = torch.mean(u)
        if torch.abs(mu) < eps:
            return torch.zeros((), dtype=v.dtype, device=v.device)
        d = u.unsqueeze(0) - u.unsqueeze(1)
        diffs = torch.sqrt(d * d + tau * tau) if tau > 0 else d.abs()
        # Scale-invariant form: multiplying u by c cancels in numerator/denominator
        return diffs.mean() / (2.0 * mu + eps)

    @staticmethod
    def hhi_penalty(
            x,
            eps: float = 1e-8,
            detach_stats: bool = True,
            max_z: float = 8.0,
    ):
        v = x.reshape(-1)
        center = torch.mean(v)
        scale = multi_iqr_scale(v)
        if detach_stats:
            center = center.detach()
            scale = scale.detach()
        z = (v - center) / (scale + eps)
        if max_z is not None: z = torch.clamp(z, -max_z, max_z)
        u = torch.exp(z)

        p = u / (u.sum() + eps)
        hhi = (p * p).sum()
        n = p.numel()
        return (n * hhi - 1.0) / (n - 1.0 + eps)

    @staticmethod
    def behavior_penalty(output, target, mask):
        return ((output[:, mask, :] - target[:, mask, :]) ** 2).mean()

    def channel_overlap_penalty(self, states, input):
        '''
        Encourages input channels to be non overlapping: so that a neuron receives at most one input channel
        '''
        b = self.RNN.W_inp if self.orth_input_only else torch.cat((self.RNN.W_inp, self.RNN.W_out.T), dim=1)
        b = b / (torch.linalg.vector_norm(b, dim=0) + 1e-8)
        G = torch.tril(b.T @ b, diagonal=-1)
        lower_tri_mask = torch.tril(torch.ones_like(G), diagonal=-1)
        return torch.sqrt(torch.mean(G[lower_tri_mask == 1.0] ** 2))

    def s_inequality_penalty(self, states, input):
        activity = torch.mean(torch.abs(states), dim=(1, 2))  # (N,)
        if self.inequality_method == 'gini':
            return self.gini_penalty(activity)
        elif self.inequality_method == 'hhi':
            return self.hhi_penalty(activity)

    def s_magnitude_penalty(self, states, input):
        return torch.sqrt(torch.tensor(self.RNN.N)).to(self.RNN.device) * torch.mean(torch.abs(states) ** 2)

    def h_inequality_penalty(self, states, input, method='hhi'):
        h = (torch.einsum('ij,jkl->ikl', self.RNN.W_rec, states) +
             torch.einsum('ij,jkl->ikl', self.RNN.W_inp, input))
        mean_h = torch.mean(h, dim=(1, 2))  # (N,)
        if self.inequality_method == 'gini':
            return self.gini_penalty(-mean_h)
        elif self.inequality_method == 'hhi':
            return self.hhi_penalty(-mean_h)

    def h_time_variance_penalty(self, states, input, eps=1e-8):
        h = (torch.einsum('ij,jkl->ikl', self.RNN.W_rec, states)
             + torch.einsum('ij,jkl->ikl', self.RNN.W_inp, input))
        mean_t = h.mean((0, 2))
        var_between = mean_t.var(unbiased=False)
        var_within = h.var((0, 2), unbiased=False).mean()
        denom = (var_between + var_within).detach() + eps
        return var_between / denom

    def h_local_variance_penalty(self, states, input):
        device, dtype = states.device, states.dtype
        mean_s = states.mean(dim=(1, 2))  # (N,)
        contrib = (self.RNN.W_rec * mean_s.unsqueeze(0)).abs()  # (N, N) rows=i (post), cols=j (pre)
        dale_cols = self.RNN.dale_mask.to(device)  # (N,)
        conc_E = self.normalized_concentration(contrib, (dale_cols == 1))
        conc_I = self.normalized_concentration(contrib, (dale_cols == -1))
        v = conc_E + conc_I
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v.mean()

    def clustering_penalty(self, states, input,
                           attract_margin=0.1,
                           repell_margin=0.3,
                           diameter_quantile=0.9,
                           beta=20.0, eps=1e-8):
        X = states.view(states.shape[0], -1)
        loss = torch.tensor(0.0, device=states.device, dtype=states.dtype)
        if getattr(self.RNN, 'dale_mask', None) is None:
            dale_mask = torch.ones(X.shape[0], device=states.device, dtype=states.dtype)
        else:
            dale_mask = self.RNN.dale_mask

        for nrn_sign in [1, -1]:
            X_subpop = X[dale_mask == nrn_sign, :]
            if X_subpop.shape[0] <= 1:
                continue
            X_norm_sq = (X_subpop ** 2).sum(dim=1, keepdim=True)
            D2 = X_norm_sq + X_norm_sq.T - 2 * X_subpop @ X_subpop.T
            D = D2.clamp(min=eps).sqrt()
            # Remove diagonal (self-distances)
            i, j = torch.triu_indices(D.shape[0], D.shape[1], offset=1)
            dists = D[i, j]

            diameter = torch.quantile(dists, diameter_quantile).detach()
            attract_thresh = attract_margin * diameter
            repel_thresh = repell_margin * diameter

            # Triangle penalty (fully smooth)
            term1 = F.softplus(dists / (attract_thresh + eps), beta=beta)
            term2 = F.softplus((dists - attract_thresh) / (attract_thresh + eps), beta=beta)
            term3 = F.softplus((dists - attract_thresh) / ((repel_thresh - attract_thresh) + eps), beta=beta)
            triangle = F.softplus(term1 - term2 - term3, beta=beta)
            loss += triangle.mean()
        return loss

    def train_step(self, input, target_output, mask):
        def z(): return torch.zeros((), device=self.RNN.device)
        params = [p for p in self.RNN.parameters() if p.requires_grad]  # CHG: moved earlier

        def grad_norm_of(scalar):
            if scalar is None or (not getattr(scalar, "requires_grad", False)): return z()
            grads = torch.autograd.grad(scalar, params, retain_graph=True, create_graph=False, allow_unused=True)
            s = z()
            for g in grads:
                if g is not None: s = s + (g.detach() ** 2).sum()
            return torch.sqrt(s)

        states_full, predicted_output_full = self.RNN(input, w_noise=True, dropout=False, drop_rate=None)

        if self.dropout:
            self.participation = (states_full.abs().mean((1,2)) + states_full.std((1,2), unbiased=False)).detach()
            _, predicted_output_do = self.RNN(input, w_noise=True, dropout=True,
                                              drop_rate=self.drop_rate, participation=self.participation)
            behavior_mismatch_penalty = self.behavior_penalty(predicted_output_do, target_output, mask)
        else:
            behavior_mismatch_penalty = self.behavior_penalty(predicted_output_full, target_output, mask)

        penalty_dict_raw = {k: (fn(states_full, input) if lam != 0 else None) for k, (fn, lam) in self.penalty_map.items()}
        penalty_dict_scaled = {k: (lam * penalty_dict_raw[k] if lam != 0 else z()) for k, (fn, lam) in self.penalty_map.items()}
        penalty_dict_scaled["behavior"] = behavior_mismatch_penalty

        grads = {f"grad_norm_{k}": grad_norm_of(v) for k,v in penalty_dict_raw.items()}
        grads["grad_norm_behavior"] = grad_norm_of(behavior_mismatch_penalty)

        loss = sum(penalty_dict_scaled.values())

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        for k in self.penalty_map:
            self.loss_monitor[k].append(self.to_item(penalty_dict_scaled[k]))
            self.indicators_monitor[f"grad_norm_{k}"].append(self.to_item(grads[f"grad_norm_{k}"]))
        self.loss_monitor["behavior"].append(self.to_item(behavior_mismatch_penalty))
        self.indicators_monitor["grad_norm_behavior"].append(self.to_item(grads["grad_norm_behavior"]))
        return loss.item()

    def eval_step(self, input, target_output, mask, metric="R2"):
        with torch.no_grad():
            self.RNN.eval()
            states, predicted_output_val = self.RNN(input, w_noise=False, dropout=False)
            if metric=='R2':
                val_loss = r2_score(target_output[:, mask, :], predicted_output_val[:, mask, :])
            elif metric == 'MSE':
                val_loss = mean_squared_error(target_output[:, mask, :], predicted_output_val[:, mask, :])
            return float(val_loss.cpu().numpy())

    def run_training(self, train_mask, same_batch=False, shuffle=False):
        # logs = {}
        train_losses = []
        val_losses = []
        self.RNN.train()  # puts the RNN into training mode (sets update_grad = True)
        min_train_loss = np.inf
        best_net_params = deepcopy(self.RNN.get_params())
        if same_batch:
            input_batch, target_batch, conditions_batch = self.Task.get_batch(shuffle=shuffle)
            input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.RNN.device)
            target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.RNN.device)

        for iter in tqdm(range(self.max_iter)):
            if not same_batch:
                input_batch, target_batch, conditions_batch = self.Task.get_batch(shuffle=shuffle)
                input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.RNN.device)
                target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.RNN.device)

            train_loss = self.train_step(input=input_batch,
                                         target_output=target_batch,
                                         mask=train_mask)
            eps = 1e-8
            # positivity of entries of W_inp and W_out
            self.RNN.W_inp.data = torch.maximum(self.RNN.W_inp.data, torch.tensor(eps))
            self.RNN.W_out.data = torch.maximum(self.RNN.W_out.data, torch.tensor(eps))

            if self.RNN.constrained:
                self.enforce_dale(eps)

            # if iter % 25 == 0:
            #     logs["params"] = deepcopy(self.RNN.get_params())
            #     logs["params"]["activation_slope"] = logs["params"]["activation_slope"].item()
            #     logs["dropout_mask"] = self.RNN.last_dropout_mask.detach().cpu().numpy().flatten()
            #     logs["drop_probs"] = self.RNN.last_drop_probs.detach().cpu().numpy().flatten()
            #     logs["participation"] = self.participation.detach().cpu().numpy().flatten()
            #     logs["iter"] = iter
            #     data = jsonify(logs)
            #     Path("../../log").mkdir(parents=True, exist_ok=True)
            #     Path(f"../../log/drop_{iter:06d}.json").write_text(json.dumps(data, indent=4))


            self.print_iteration_info(iter, train_loss, min_train_loss)
            train_losses.append(train_loss)
            if train_loss <= min_train_loss:
                min_train_loss = train_loss
                best_net_params = deepcopy(self.RNN.get_params())
            if train_loss <= self.tol:
                print("Reached tolerance!")
                self.RNN.set_params(best_net_params)

                return self.RNN, train_losses, val_losses, best_net_params, logs

        self.RNN.set_params(best_net_params)
        return self.RNN, train_losses, val_losses, best_net_params

    def enforce_dale(self, eps=1e-8):
        with torch.no_grad():
            # W_rec
            W_rec = self.RNN.W_rec
            dale_mask_expanded_rec = self.RNN.dale_mask.unsqueeze(0).repeat(W_rec.shape[0], 1)
            abberant_mask_rec = (W_rec * dale_mask_expanded_rec < 0)
            corrected_rec = W_rec.clone()
            corrected_rec[abberant_mask_rec] = eps * dale_mask_expanded_rec[abberant_mask_rec]
            self.RNN.W_rec.copy_(corrected_rec)

            # W_out
            W_out = self.RNN.W_out
            dale_mask_expanded_out = self.RNN.dale_mask.unsqueeze(0).repeat(W_out.shape[0], 1)
            abberant_mask_out = (W_out * dale_mask_expanded_out < 0)
            corrected_out = W_out.clone()
            corrected_out[abberant_mask_out] = eps * dale_mask_expanded_out[abberant_mask_out]
            self.RNN.W_out.copy_(corrected_out)
            return None





