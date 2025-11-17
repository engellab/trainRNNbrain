'''
Class which accepts RNN_torch and a task and has a mode to train RNN
'''
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torcheval.metrics.functional import r2_score, mean_squared_error
import time
import math
from tqdm.auto import tqdm
import json
from trainRNNbrain.utils import jsonify
from pathlib import Path

# TODO: figure out better sparsity penalty
# encourange W_out redundancy
# figure out how to reconcile sparsity with zero-stickiness
# spatially embedded networks instead? to maintain sparsity
# participating redistributions (how to encourage non participants to participate?)

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
    def __init__(self,
                 RNN, Task, optimizer,
                 max_iter=1000,
                 tol=1e-12,
                 lambda_iwm=0.1,
                 lambda_rwm=0.1,
                 lambda_ow=0.1,
                 lambda_rws=0.1,
                 lambda_tv=0.1,
                 lambda_orth=0.3,
                 orth_input_only=True,
                 lambda_sm=0.001,
                 lambda_si=0.1,
                 lambda_hi=0.1,
                 lambda_htvar=0.05,
                 lambda_hlvar=0.9,
                 lambda_cl=0.05,
                 inequality_method='hhi',
                 dropout=False, drop_rate=0.3,
                 p=2):
        self.RNN = RNN
        self.max_sigma_rec = self.RNN.sigma_rec
        self.max_sigma_inp = self.RNN.sigma_inp
        self.Task = Task
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_iwm = lambda_iwm
        self.lambda_rwm = lambda_rwm
        self.lambda_ow = lambda_ow
        self.lambda_rws = lambda_rws
        self.lambda_tv = lambda_tv
        self.lambda_orth, self.orth_input_only = lambda_orth, orth_input_only
        self.lambda_sm = lambda_sm
        self.lambda_si = lambda_si
        self.lambda_hi = lambda_hi
        self.lambda_hlvar = lambda_hlvar
        self.lambda_htvar = lambda_htvar
        self.lambda_cl = lambda_cl
        self.penalty_map = {
            "inp_weights_magnitude": (self.inp_weights_magnitude_penalty, self.lambda_iwm),
            "rec_weights_magnitude": (self.rec_weights_magnitude_penalty, self.lambda_rwm),
            "out_weights": (self.out_weights_penalty, self.lambda_ow),
            "rec_weights_sparsity": (self.rec_weights_sparsity_penalty, self.lambda_rws),
            "output_var": (self.trial_output_var_penalty, self.lambda_tv),
            "channel_overlap": (self.channel_overlap_penalty, self.lambda_orth),
            "s_magnitude": (self.s_magnitude_penalty, self.lambda_sm),
            "s_inequality": (self.s_inequality_penalty, self.lambda_si),
            "h_inequality": (self.h_inequality_penalty, self.lambda_hi),
            "h_time_variance": (self.h_time_variance_penalty, self.lambda_htvar),
            "h_local_variance": (self.h_local_variance_penalty, self.lambda_hlvar),
            "clustering": (self.clustering_penalty, self.lambda_cl),
        }
        self.loss_monitor = {'behavior': [], **{k: [] for k in self.penalty_map}}
        self.gradients_monitor = {'g_behavior': [], **{f"g_{k}": [] for k in self.penalty_map}}
        self.scaled_gradients_monitor = {'sg_behavior': [], **{f"sg_{k}": [] for k in self.penalty_map}}
        self.p, self.dropout, self.drop_rate = p, dropout, drop_rate
        self.inequality_method = inequality_method
        self.participation = (1e-6 * torch.ones(self.RNN.N, device=self.RNN.device)) if self.dropout else None
        self.iter_n = 0

    @staticmethod
    def to_item(x):
        return x.detach().cpu() if torch.is_tensor(x) else torch.as_tensor(x)

    @staticmethod
    def print_iteration_info(
            iter,
            max_iter,
            train_loss,
            min_train_loss,
            r2,
            elapsed_t,
            eta,
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
            print(f"iteration {iter}/{max_iter},"
                  f" train: {train_prfx}{np.round(train_loss, 6)}{train_sfx},"
                  f" r2: {train_prfx}{np.round(r2, 6)}{train_sfx},"
                  f" val_score: {val_prfx}{np.round(val_loss, 6)}{val_sfx};"
                  f" elapsed: {elapsed_t}, remaining ~ {eta}")
        else:
            print(f"iteration {iter}/{max_iter},"
                  f" train: {train_prfx}{np.round(train_loss, 6)}{train_sfx},"
                  f" r2: {train_prfx}{np.round(r2, 6)}{train_sfx};"
                  f" elapsed: {elapsed_t}, remaining ~ {eta}")

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

    def trial_output_var_penalty(self, states, input, eps=1e-8):
        # states: (N, T, K), W_out: (n_out, N)
        y = torch.einsum('on,ntk->otk', self.RNN.W_out, states)  # (n_out, T, K)
        yc = y - y.mean(dim=2, keepdim=True)  # center across trials (K)
        num = (yc * yc).mean()  # E_{o,t,k}[(y - ⟨y⟩_trial)^2]
        den = (y * y).mean().clamp_min(eps)  # normalize by overall power
        return num / den

    def inp_weights_magnitude_penalty(self, states, input, cap=0.05):
        return torch.expm1(torch.relu(self.RNN.W_inp.abs() - cap) / cap).mean()

    def out_weights_penalty(self, states, input, c=2.0, alpha=10.0, gamma=1.0, eps=1e-12):
        W = self.RNN.W_out  # (D, N)
        N = self.RNN.N
        A = W.abs()
        p_l1 = (A.sum(1) - c).pow(2).mean()  # total row gain ≈ c (size-invariant)
        cap = alpha / N
        p_cap = torch.expm1(torch.relu(A - cap) / cap).mean()
        P = A / (A.sum(1, keepdim=True) + eps)  # rowwise probs
        hhi = (P * P).sum(1)  # ∈[1/n,1]
        n = A.size(1)
        p_hhi = ((n * hhi - 1.0) / (n - 1.0 + eps)).mean()  # ∈[0,1]
        return p_l1 + p_cap + gamma * p_hhi

    def rec_weights_magnitude_penalty(self, states, input, cap=0.05, N_ref=100, k_ref=20):
        W = self.RNN.W_rec
        exc, inh = self.RNN.dale_mask > 0, self.RNN.dale_mask < 0
        cap_e, cap_i = cap, cap * self.RNN.exc2inhR
        ex = torch.relu(W[:, exc].abs() - cap_e) / cap_e
        ix = torch.relu(W[:, inh].abs() - cap_i) / cap_i
        excess = torch.cat([ex.reshape(-1), ix.reshape(-1)])
        return torch.expm1(excess).sum() / (N_ref * k_ref)

    def rec_weights_sparsity_penalty(self, states, ipnut, k_tgt=20, eps=1e-8):
        W = self.RNN.W_rec  # (N, N)
        l1 = W.abs().sum(dim=1)
        l2 = (W.square().sum(dim=1) + eps).sqrt()
        S = (l1 * l1) / (l2 * l2)  # effective support per row
        over = torch.relu(S - k_tgt)
        return (over * over).mean() / (k_tgt * k_tgt)

    def s_magnitude_penalty(self, states, input, cap=0.2, q=0.9, beta=50.0, eps=1e-12):
        x = states.abs().view(states.size(0), -1)  # flatten over time/trials
        s_q = torch.quantile(x + eps, q, dim=1)  # (N,)
        over = s_q - cap
        return torch.nn.functional.softplus(beta * over).mean() / beta

    def channel_overlap_penalty(self, states, input, eps=1e-8):
        B = self.RNN.W_inp if self.orth_input_only else torch.cat((self.RNN.W_inp, self.RNN.W_out.T), dim=1)  # (N,M)
        B = B / (torch.linalg.vector_norm(B, dim=0, keepdim=True) + eps)  # col unit-norm
        G = B.T @ B  # (M,M)
        M = G.shape[0]
        if M <= 1:
            return torch.zeros((), device=G.device, dtype=G.dtype)
        i, j = torch.tril_indices(M, M, offset=-1, device=G.device)
        cos_ij = G[i, j]
        return torch.sqrt((cos_ij * cos_ij).mean())

    def s_inequality_penalty(self, states, input):
        activity = torch.mean(torch.abs(states), dim=(1, 2))  # (N,)
        if self.inequality_method == 'gini':
            return self.gini_penalty(activity)
        elif self.inequality_method == 'hhi':
            return self.hhi_penalty(activity)

    def h_inequality_penalty(self, states, input, method='hhi'):
        h = (torch.einsum('ij,jkl->ikl', self.RNN.W_rec, states) +
             torch.einsum('ij,jkl->ikl', self.RNN.W_inp, input))
        mean_h = torch.mean(h, dim=(1, 2))  # (N,)
        if self.inequality_method == 'gini':
            return self.gini_penalty(mean_h)
        elif self.inequality_method == 'hhi':
            return self.hhi_penalty(mean_h)

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
        def z():
            return torch.zeros((), device=self.RNN.device)

        params = [p for p in self.RNN.parameters() if p.requires_grad]

        def grad_norm_of(scalar):
            if scalar is None or not getattr(scalar, "requires_grad", False): return z()
            grads = torch.autograd.grad(scalar, params, retain_graph=True, create_graph=False, allow_unused=True)
            s = z()
            for g in grads:
                if g is not None: s = s + (g.detach() ** 2).sum()
            return s.sqrt()

        # noise schedule
        scale = torch.as_tensor(self.max_iter / 12, dtype=torch.float32, device=self.RNN.device)
        center = torch.as_tensor(self.max_iter / 3, dtype=torch.float32, device=self.RNN.device)
        mult = 1.0 / (1.0 + torch.exp(-(self.iter_n - center) / scale))
        self.RNN.sigma_rec = self.max_sigma_rec * mult
        self.RNN.sigma_inp = self.max_sigma_inp * mult

        states_full, predicted_output_full = self.RNN(input, w_noise=True, dropout=False, drop_rate=None)

        if self.dropout:
            self.participation = (states_full.abs().mean((1, 2)) + states_full.std((1, 2), unbiased=False)).detach()
            _, predicted_output_do = self.RNN(input,
                                              w_noise=True,
                                              dropout=True,
                                              drop_rate=self.drop_rate,
                                              participation=self.participation)
            behavior_mismatch_penalty = self.behavior_penalty(predicted_output_do, target_output, mask)
        else:
            behavior_mismatch_penalty = self.behavior_penalty(predicted_output_full, target_output, mask)

        penalty_dict_raw = {k: (fn(states_full, input) if lam != 0 else None) for k, (fn, lam) in
                            self.penalty_map.items()}
        penalty_dict_scaled = {k: (lam * penalty_dict_raw[k] if lam != 0 else z()) for k, (fn, lam) in
                               self.penalty_map.items()}
        penalty_dict_scaled["behavior"] = behavior_mismatch_penalty

        grads = {f"g_{k}": grad_norm_of(v) for k, v in penalty_dict_raw.items()}
        grads["g_behavior"] = grad_norm_of(behavior_mismatch_penalty)

        loss = sum(penalty_dict_scaled.values())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.loss_monitor["behavior"].append(self.to_item(behavior_mismatch_penalty))
        self.gradients_monitor["g_behavior"].append(self.to_item(grads["g_behavior"]))
        self.scaled_gradients_monitor["sg_behavior"].append(self.to_item(grads["g_behavior"]))
        for k in self.penalty_map:
            # self.loss_monitor[k].append(self.to_item(penalty_dict_raw[k] if self.penalty_map[k][1] != 0 else z()))
            self.loss_monitor[k].append(self.to_item(penalty_dict_scaled[k] if self.penalty_map[k][1] != 0 else z()))
            self.gradients_monitor[f"g_{k}"].append(self.to_item(grads[f"g_{k}"]))
            self.scaled_gradients_monitor[f"sg_{k}"].append(self.to_item(self.penalty_map[k][1] * self.to_item(grads[f"g_{k}"])))

        self.iter_n += 1
        with torch.no_grad():
            y = predicted_output_full[:, mask, :]
            t = target_output[:, mask, :]
            mse = (y - t).pow(2).mean()
            sst = (t - t.mean()).pow(2).mean().clamp_min(1e-12)
            r2 = 1.0 - mse / sst
        return float(loss.item()), float(r2.item())

    def eval_step(self, inp, tgt, mask, noise=False, dropout=False, drop_rate=None, seed=None):
        if seed is not None: torch.manual_seed(seed)
        self.RNN.eval()
        dt = next(self.RNN.parameters()).dtype
        dev = self.RNN.device
        with torch.no_grad():
            srec, sinp = float(self.RNN.sigma_rec), float(self.RNN.sigma_inp)
            if not noise: self.RNN.sigma_rec = self.RNN.sigma_inp = 0.0
            inp, tgt = inp.to(dev, dt), tgt.to(dev, dt)
            mask = mask if isinstance(mask, slice) else torch.as_tensor(mask, device=dev)
            _, y = self.RNN(inp, w_noise=noise, dropout=dropout, drop_rate=drop_rate)
            y, t = y[:, mask, :], tgt[:, mask, :]
            r2 = 1 - (y - t).pow(2).mean() / ((t - t.mean()).pow(2).mean().clamp_min(1e-12))
            self.RNN.sigma_rec, self.RNN.sigma_inp = srec, sinp
        return float(r2)

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

        tic = time.perf_counter()
        for iter in range(self.max_iter):

            if not same_batch:
                input_batch, target_batch, conditions_batch = self.Task.get_batch(shuffle=shuffle)
                input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.RNN.device)
                target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.RNN.device)

            train_loss, r2 = self.train_step(input=input_batch,
                                         target_output=target_batch,
                                         mask=train_mask)
            eps = 1e-8
            # positivity of entries of W_inp and W_out
            eps_t = torch.tensor(eps, device=self.RNN.W_inp.device, dtype=self.RNN.W_inp.dtype)
            self.RNN.W_inp.data = torch.maximum(self.RNN.W_inp.data, eps_t)
            eps_t = torch.tensor(eps, device=self.RNN.W_out.device, dtype=self.RNN.W_out.dtype)
            self.RNN.W_out.data = torch.maximum(self.RNN.W_out.data, eps_t)
            # self.RNN.W_inp.data.clamp_(min=0.0)
            # self.RNN.W_out.data.clamp_(min=0.0)

            if self.RNN.constrained:
                self.enforce_dale(eps)
            # sr_old = self.project_recurrent_spectral_radius(r_star=0.95)


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

            toc = time.perf_counter()
            delta = toc - tic
            elapsed_t = time.strftime("%H:%M:%S", time.gmtime(delta))
            proj_total = (delta / (iter + 1)) * self.max_iter
            remaining = proj_total - delta
            eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
            self.print_iteration_info(iter + 1, self.max_iter, train_loss, min_train_loss, r2, elapsed_t, eta)
            train_losses.append(train_loss)
            if train_loss <= min_train_loss:
                min_train_loss = train_loss
                best_net_params = deepcopy(self.RNN.get_params())
        last_net_params = deepcopy(self.RNN.get_params())
        self.RNN.set_params(last_net_params) # assuming that the more training it went through - the better.
        return self.RNN, train_losses, val_losses, best_net_params, last_net_params

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





