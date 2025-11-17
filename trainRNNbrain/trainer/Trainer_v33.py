'''
Class which accepts RNN_torch and a task and has a mode to train RNN
'''
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torcheval.metrics.functional import r2_score, mean_squared_error
from tqdm.auto import tqdm

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
    def __init__(self, RNN, Task, criterion, optimizer,
                 max_iter=1000,
                 tol=1e-12,
                 lambda_orth=0.3,
                 orth_input_only=True,
                 lambda_rm=0.001,
                 lambda_rvar=0.1,
                 lambda_hm=0.1,
                 lambda_hvar=0.05,
                 lambda_cat=0.05,
                 dropout=False,
                 drop_rate=0.3,
                 p=2):
        '''
        :param RNN: pytorch RNN (specific template class)
        :param Task: task (specific template class)
        :param max_iter: maximum number of iterations
        :param tol: float, such that if the cost function reaches tol the optimization terminates
        :param criterion: function to evaluate loss
        :param optimizer: pytorch optimizer (Adam, SGD, etc.)
        :param lambda_ort: float, regularization softly imposing a pair-wise orthogonality
         on columns of W_inp and rows of W_out
        :param orth_input_only: bool, if impose penalties only on the input columns,
         or extend the penalty onto the output rows as well
        :param lambda_r: float, regularization of the mean firing rates during the trial
        '''
        self.RNN = RNN
        self.Task = Task
        self.max_iter = max_iter
        self.tol = tol
        self.criterion = criterion
        self.optimizer = optimizer
        self.lambda_orth = lambda_orth
        self.orth_input_only = orth_input_only
        self.lambda_rm = lambda_rm
        self.lambda_rvar = lambda_rvar
        self.lambda_hm = lambda_hm
        self.lambda_hvar = lambda_hvar
        self.lambda_cat = lambda_cat
        self.loss_monitor = {"behavior": [],
                             "channel overlap": [], #lambda_orth
                             "activity_magnitude": [], #lambda_rm
                             "activity_variability": [],  # lambda_rvar
                             "net inputs variability": [],  # lambda_hvar
                             "net inputs mean": [], #, #lambda_hm
                             "categorical": []} #lambda_cat
        self.indicators_monitor = {"grad norm behavior": [],
                                   "grad norm activity magnitude": [],
                                   "grad norm activity variability": [],
                                   "grad norm channel overlap": [],
                                   "grad norm net inputs variability": [],
                                   "grad norm net inputs mean": [],
                                   "grad norm categorical": []}
                                   # "grad norm categorical": []}
        self.p = p
        self.dropout = dropout
        self.drop_rate = drop_rate
        self.counter = 0
        self.participation = 1e-6 * torch.ones(self.RNN.N) if self.dropout else None
        self.p_alpha = 0.8

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
    def to_item(x):
        return x.detach() if torch.is_tensor(x) else torch.tensor(x)

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

    def channel_overlap_penalty(self):
        '''
        Encourages input channels to be non overlapping: so that a neuron receives at most one input channel
        '''
        b = self.RNN.W_inp if self.orth_input_only else torch.cat((self.RNN.W_inp, self.RNN.W_out.T), dim=1)
        b = b / (torch.linalg.vector_norm(b, dim=0) + 1e-8)
        G = torch.tril(b.T @ b, diagonal=-1)
        lower_tri_mask = torch.tril(torch.ones_like(G), diagonal=-1)
        return torch.sqrt(torch.mean(G[lower_tri_mask == 1.0] ** 2))

    def activity_variability_penalty(self, states):
        activity = torch.mean(torch.abs(states), dim=(1, 2))  # (N,)
        return self.gini_penalty(activity)

    def activity_magnitude_penalty(self, states):
        activity = torch.mean(torch.abs(states), dim=(1, 2))  # (N,)
        return torch.sqrt(torch.tensor(self.RNN.N)) * torch.mean(torch.abs(activity))

    def net_inputs_penalty(self, states, inp):
        h = (torch.einsum('ij,jkl->ikl', self.RNN.W_rec, states) +
             torch.einsum('ij,jkl->ikl', self.RNN.W_inp, inp))
        mean_h = torch.mean(h, dim=(1, 2))  # (N,)
        return self.gini_penalty(-mean_h) # minus sign to penalize overly inhibited neurons

    def net_inputs_variability_penalty(self, states, inp, eps: float = 1e-8):
        # h: (N, T, B)
        h = (torch.einsum('ij,jkl->ikl', self.RNN.W_rec, states) +
             torch.einsum('ij,jkl->ikl', self.RNN.W_inp, inp))
        std_t = torch.std(h, dim=1, unbiased=False)  # (N, B): variability over time, per trial
        std_b = torch.std(h, dim=2, unbiased=False)  # (N, T): variability over batch, per time
        x = std_t.mean()  # mean over neurons & trials
        y = std_b.mean()  # mean over neurons & time
        denom = (x + y).detach() + eps  # detach to avoid gaming the denominator
        return x / denom

    def categorical_penalty(self, states, attract_margin=0.1, repell_margin=0.3, diameter_quantile=0.9, beta=20.0):
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
            D = D2.clamp(min=1e-8).sqrt()
            # Remove diagonal (self-distances)
            i, j = torch.triu_indices(D.shape[0], D.shape[1], offset=1)
            dists = D[i, j]

            diameter = torch.quantile(dists, diameter_quantile).detach()
            attract_thresh = attract_margin * diameter
            repel_thresh = repell_margin * diameter

            # Triangle penalty (fully smooth)
            term1 = F.softplus(dists / (attract_thresh + 1e-8), beta=beta)
            term2 = F.softplus((dists - attract_thresh) / (attract_thresh + 1e-8), beta=beta)
            term3 = F.softplus((dists - attract_thresh) / ((repel_thresh - attract_thresh) + 1e-8), beta=beta)
            triangle = F.softplus(term1 - term2 - term3, beta=beta)
            loss += triangle.mean()
        return loss

    def train_step(self, input, target_output, mask):
        # Forward pass
        states, predicted_output = self.RNN(input, w_noise=True,
                                            dropout=self.dropout,
                                            drop_rate=self.drop_rate,
                                            participation=self.participation)

        if self.dropout:
            with torch.inference_mode():
                s, _ = self.RNN(input, w_noise=False, dropout=False)
                self.participation = torch.mean(torch.abs(s), dim=(1, 2)) + torch.std(s, dim=(1, 2), unbiased=False)

        behavior_mismatch_penalty = self.criterion(target_output[:, mask, :], predicted_output[:, mask, :])
        # Helper: scalar zero on the right device/dtype
        def z():
            return torch.zeros((), device=states.device, dtype=behavior_mismatch_penalty.dtype)

        # ---- Compute RAW penalties first (unscaled) ----
        channel_overlap_raw = self.channel_overlap_penalty() if self.lambda_orth != 0 else None
        activity_variability_raw = self.activity_variability_penalty(states) if self.lambda_rvar != 0 else None
        activity_magnitude_raw = self.activity_magnitude_penalty(states) if self.lambda_rm != 0 else None
        net_inputs_raw = self.net_inputs_penalty(states, input) if self.lambda_hm != 0 else None
        net_inputs_var_raw = self.net_inputs_variability_penalty(states, input) if self.lambda_hvar != 0 else None
        categorical_raw = self.categorical_penalty(states) if self.lambda_cat != 0 else None

        # ---- Scaled terms (Torch zeros when off) ----
        channel_overlap_penalty = self.lambda_orth * channel_overlap_raw if channel_overlap_raw is not None else z()
        activity_variability_penalty = self.lambda_rvar * activity_variability_raw if activity_variability_raw is not None else z()
        activity_magnitude_penalty = self.lambda_rm * activity_magnitude_raw if activity_magnitude_raw is not None else z()
        net_inputs_penalty = self.lambda_hm * net_inputs_raw if net_inputs_raw is not None else z()
        net_inputs_variability_penalty = self.lambda_hvar * net_inputs_var_raw if net_inputs_var_raw is not None else z()
        categorical_penalty = self.lambda_cat * categorical_raw if categorical_raw is not None else z()

        # ---- Gradient norms (read-only) for each term ----
        params = [p for p in self.RNN.parameters() if p.requires_grad]

        def grad_norm_of(scalar):
            """L2 norm of grads of `scalar` w.r.t. params, returns scalar tensor on device."""
            if scalar is None or (not getattr(scalar, "requires_grad", False)):
                return z()
            grads = torch.autograd.grad(
                outputs=scalar,
                inputs=params,
                retain_graph=True,  # keep graph for the main backward below
                create_graph=False,
                allow_unused=True,
            )
            total = z()
            for g in grads:
                if g is not None:
                    total = total + (g.detach() ** 2).sum()
            return torch.sqrt(total)

        # Compute norms for the *raw* penalties and behavior loss
        gnorm_behavior = grad_norm_of(behavior_mismatch_penalty)
        gnorm_activity_magnitude = grad_norm_of(activity_magnitude_penalty)
        gnorm_activity_variability = grad_norm_of(activity_variability_penalty)
        gnorm_channel_overlap = grad_norm_of(channel_overlap_raw)
        gnorm_net_inputs_var = grad_norm_of(net_inputs_var_raw)
        gnorm_net_inputs_mean = grad_norm_of(net_inputs_raw)
        gnorm_categorical = grad_norm_of(categorical_raw)

        # ---- Total loss ----
        loss = (behavior_mismatch_penalty
                + channel_overlap_penalty
                + activity_variability_penalty
                + activity_magnitude_penalty
                + net_inputs_penalty
                + net_inputs_variability_penalty
                + categorical_penalty)

        # Backprop + step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # ---- Logging ----
        self.loss_monitor["behavior"].append(self.to_item(behavior_mismatch_penalty))
        self.loss_monitor["channel overlap"].append(self.to_item(channel_overlap_penalty))
        self.loss_monitor["activity_magnitude"].append(self.to_item(activity_magnitude_penalty))
        self.loss_monitor["activity_variability"].append(self.to_item(activity_variability_penalty))
        self.loss_monitor["net inputs mean"].append(self.to_item(net_inputs_penalty))
        self.loss_monitor["net inputs variability"].append(self.to_item(net_inputs_variability_penalty))
        self.loss_monitor["categorical"].append(self.to_item(categorical_penalty))

        self.indicators_monitor["grad norm behavior"].append(self.to_item(gnorm_behavior))
        self.indicators_monitor["grad norm activity magnitude"].append(self.to_item(gnorm_activity_magnitude))
        self.indicators_monitor["grad norm activity variability"].append(self.to_item(gnorm_activity_variability))
        self.indicators_monitor["grad norm channel overlap"].append(self.to_item(gnorm_channel_overlap))
        self.indicators_monitor["grad norm net inputs variability"].append(self.to_item(gnorm_net_inputs_var))
        self.indicators_monitor["grad norm net inputs mean"].append(self.to_item(gnorm_net_inputs_mean))
        self.indicators_monitor["grad norm categorical"].append(self.to_item(gnorm_categorical))

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

            self.print_iteration_info(iter, train_loss, min_train_loss)
            train_losses.append(train_loss)
            if train_loss <= min_train_loss:
                min_train_loss = train_loss
                best_net_params = deepcopy(self.RNN.get_params())
            if train_loss <= self.tol:
                print("Reached tolerance!")
                self.RNN.set_params(best_net_params)
                return self.RNN, train_losses, val_losses, best_net_params

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





