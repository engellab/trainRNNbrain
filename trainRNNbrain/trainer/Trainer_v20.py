'''
Class which accepts RNN_torch and a task and has a mode to train RNN
'''
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torcheval.metrics.functional import r2_score, mean_squared_error
from tqdm.auto import tqdm

class Trainer():
    def __init__(self, RNN, Task, criterion, optimizer,
                 max_iter=1000,
                 tol=1e-12,
                 lambda_orth=0.3,
                 orth_input_only=True,
                 lambda_r=0.05,
                 lambda_iz=0.05,
                 lambda_cat=0.05,
                 lambda_h=0.1,
                 lambda_hvar=0.05,
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
        self.lambda_r = lambda_r
        self.lambda_h = lambda_h
        self.lambda_hvar = lambda_hvar
        self.lambda_cat = lambda_cat
        self.lambda_iz = lambda_iz
        self.loss_monitor = {"behavior": [],
                             "channel overlap": [], #lambda_orth
                             "activity": [], #lambda_r
                             "net inputs variability": [],  # lambda_hvar
                             "net inputs mean": []}#, #lambda_h
                             # "isolation": [], # lambda_iz
                             # "categorical": [], #lambda_cat
                             # }
        self.indicators_monitor = {"gini participation": [],
                                   "grad norm behavior": [],
                                   "grad norm activity": [],
                                   "grad norm channel overlap": [],
                                   "grad norm net inputs variability": [],
                                   "grad norm net inputs mean": []}#,
                                   # "grad norm isolation": [],
                                   # "grad norm categorical": []}
        self.p = p
        self.dropout = dropout
        self.drop_rate = drop_rate
        self.counter = 0


    @staticmethod
    def side_penalty(values, q=0.75, eps=1e-6):
        v = values.flatten()
        vmin = v.min().detach()
        vmax = v.max().detach()
        if (vmax - vmin) <= eps:
            return torch.zeros((), dtype=v.dtype, device=v.device)

        t = torch.quantile(v, q=q).detach()  # threshold (detach)
        med = torch.median(v).detach()
        # Robust scale: IQR or MAD (both detached)
        q25 = torch.quantile(v, 0.25).detach()
        q75 = torch.quantile(v, 0.75).detach()
        iqr = (q75 - q25).abs()
        mad = (v - med).abs().median().detach()
        spread = torch.clamp(torch.max(iqr, mad), min=eps).detach()

        # One normalization using robust spread (bounded grad)
        x = torch.clamp((v - t) / spread, min=0.0)
        x = torch.clamp(x, max=1.0)  # optional cap for extra safety

        s = x.sum()
        if s <= eps:
            return torch.zeros((), dtype=v.dtype, device=v.device)

        w = (x / (s + eps)).detach()
        return (1.0 - q) * (x * w).sum()

    @staticmethod
    def one_sided_kurtosis_penalty(values, q=0.75, eps=1e-6):
        """
        Scale-invariant one-sided (right) kurtosis-like penalty.

        Definition (empirical expectation over all samples):
            kappa_+(q) = E[ ((x - μ)_+)^4 * 1{x >= t_q} ] / (Var[x])^2
        where μ is the mean (detached) and t_q is the q-quantile (detached).

        - Gate: only values in the right tail (x >= t_q) contribute.
        - Center: deviations measured from the mean μ (classic kurtosis style).
        - Normalization: divide by (Var[x])^2 => scale-free.
        - Edge cases:
            * Nearly-constant vector -> 0
            * Empty tail (all x < t_q) -> 0
        """
        v = values.flatten()
        # Quick constant check (bounded grads)
        vmin = v.min().detach()
        vmax = v.max().detach()
        if (vmax - vmin) <= eps:
            return torch.zeros((), dtype=v.dtype, device=v.device)

        # Location & scale (detached for stable normalization)
        mu = v.mean().detach()
        var = ((v - mu) ** 2).mean().detach()
        if var <= eps:
            return torch.zeros((), dtype=v.dtype, device=v.device)
        denom = (var + eps) ** 2  # σ^4 with epsilon guard
        # Right-tail gate at quantile q (detached threshold)
        t = torch.quantile(v, q=q).detach()
        gate = (v >= t).to(v.dtype)  # indicator as float
        # One-sided fourth central moment (expectation over all samples)
        pos_dev = torch.clamp(v - mu, min=0.0)  # (x - μ)_+
        m4_plus = ((pos_dev ** 4) * gate).mean()
        # Scale-invariant penalty
        return m4_plus / denom

    @staticmethod
    def gini_evenness(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Returns an evenness score in [0,1], where 1 = perfectly even, 0 = maximally uneven.
        Assumes w is 1D or will be flattened; negative entries are allowed but ignored via clamp_min(0).
        """
        w = w.reshape(-1).clamp_min(0)  # ensure nonnegative "participation"
        N = w.numel()
        if N == 0:
            return torch.zeros((), device=w.device, dtype=w.dtype)
        s = w.sum()
        if torch.isnan(s) or s <= 0:
            return torch.zeros((), device=w.device, dtype=w.dtype)
        ws, _ = torch.sort(w)  # ascending
        i = torch.arange(1, N + 1, device=w.device, dtype=ws.dtype)
        # Lorenz/Gini formula: sum((2i - N - 1) * w_i) / (N * sum w)
        G = ((2 * i - (N + 1)) * ws).sum() / (N * (s + eps))
        G = G.abs()  # numerical safety
        Gmax = (N - 1) / N  # scalar float
        evenness = 1.0 - (G / Gmax)
        return evenness.clamp(0.0, 1.0)

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

    def activity_penalty(self, states, high_percent=0.75, eps=1e-6):
        """
        Penalizes excessive activity of units (per-neuron mean |activity|)
        states: (N, T, B)
        """
        activity = torch.mean(torch.abs(states), dim=(1, 2))  # (N,)
        return self.side_penalty(activity, q=high_percent, eps=eps)

    def net_inputs_penalty(self, states, inp, low_percent=0.25, high_percent=0.75, eps=1e-6):
        """
        states: (N, T, B)
        inp:    (M, T, B)
        W_rec:  (N, N)
        W_inp:  (N, M)
        """
        # --- validation ---
        if not (0.0 <= low_percent <= 1.0 and 0.0 <= high_percent <= 1.0):
            raise ValueError(f"low_percent and high_percent must be in [0,1]; got {low_percent}, {high_percent}.")
        if low_percent > high_percent:
            raise ValueError(f"low_percent ({low_percent}) cannot be greater than high_percent ({high_percent}).")

        h = (torch.einsum('ij,jkl->ikl', self.RNN.W_rec, states) +
             torch.einsum('ij,jkl->ikl', self.RNN.W_inp, inp))
        mean_h = torch.mean(h, dim=(1, 2))  # (N,)
        low_penalty = self.side_penalty(-mean_h, q=1 - low_percent, eps=eps)
        high_penalty = self.side_penalty(mean_h, q=high_percent, eps=eps)
        return low_penalty + high_penalty

    def net_inputs_variability_penalty(self, states, inp, percent=0.25, eps=1e-8):
        """
        Encourages variability of net inputs across trials:
          - Compute per-time std across trials (dim=2), then mean over time (dim=1).
          - Penalize neurons whose variability is in the lower (percent) tail.
        shapes:
          states: (N, T, B)
          inp:    (M, T, B)
          W_rec:  (N, N), W_inp: (N, M)
        """
        # --- validation ---
        if not (0.0 <= percent <= 1.0):
            raise ValueError(f"`percent` must be in [0,1]; got {percent}.")

        # Net inputs
        h = (torch.einsum('ij,jkl->ikl', self.RNN.W_rec, states) +
             torch.einsum('ij,jkl->ikl', self.RNN.W_inp, inp))  # (N, T, B)

        # If there is only one trial, std across trials is identically zero.
        if h.size(2) == 1:
            return torch.zeros((), dtype=h.dtype, device=h.device)

        # Timewise std across trials, then mean over time
        # Use population std so the scale doesn't depend on B
        std_across_trials = torch.std(h, dim=2, unbiased=False)  # (N, T)
        neural_trial_variability = torch.mean(std_across_trials, dim=1)  # (N,)

        # Low-tail penalty (flip sign to target small variability)
        penalty = self.side_penalty(-neural_trial_variability, q=1.0 - percent, eps=eps)
        return penalty

    # def categorical_penalty(self, states, attract_margin=0.1, repell_margin=0.3, diameter_quantile=0.9, beta=20.0):
    #     X = states.view(states.shape[0], -1)
    #     loss = torch.tensor(0.0, device=states.device, dtype=states.dtype)
    #     if getattr(self.RNN, 'dale_mask', None) is None:
    #         dale_mask = torch.ones(X.shape[0], device=states.device, dtype=states.dtype)
    #     else:
    #         dale_mask = self.RNN.dale_mask
    #
    #     for nrn_sign in [1, -1]:
    #         X_subpop = X[dale_mask == nrn_sign, :]
    #         if X_subpop.shape[0] <= 1:
    #             continue
    #         X_norm_sq = (X_subpop ** 2).sum(dim=1, keepdim=True)
    #         D2 = X_norm_sq + X_norm_sq.T - 2 * X_subpop @ X_subpop.T
    #         D = D2.clamp(min=1e-8).sqrt()
    #         # Remove diagonal (self-distances)
    #         i, j = torch.triu_indices(D.shape[0], D.shape[1], offset=1)
    #         dists = D[i, j]
    #
    #         diameter = torch.quantile(dists, diameter_quantile).detach()
    #         attract_thresh = attract_margin * diameter
    #         repel_thresh = repell_margin * diameter
    #
    #         # Triangle penalty (fully smooth)
    #         term1 = F.softplus(dists / (attract_thresh + 1e-8), beta=beta)
    #         term2 = F.softplus((dists - attract_thresh) / (attract_thresh + 1e-8), beta=beta)
    #         term3 = F.softplus((dists - attract_thresh) / ((repel_thresh - attract_thresh) + 1e-8), beta=beta)
    #         triangle = F.softplus(term1 - term2 - term3, beta=beta)
    #         loss += triangle.mean()
    #     return loss

    # def isolation_penalty(self, percent=0.5, min_threshold=0.01, beta=20.0):
    #     '''
    #     If average (excitatory) incoming connections are below the threshold (either hard or dynamic)
    #     the unit is penalized depending on how far below threshold its connectedness lies
    #     '''
    #     if not self.RNN.dale_mask is None:
    #         exc_inds = torch.where(self.RNN.dale_mask == 1.0)[0]
    #         incoming_weights = self.RNN.W_rec[:, exc_inds].abs().sum(dim=1)
    #     else:
    #         incoming_weights = self.RNN.W_rec.abs().sum(dim=1)
    #     min_threshold = torch.tensor(min_threshold, device=self.RNN.device)
    #     threshold = torch.maximum(torch.quantile(incoming_weights, percent).detach(), min_threshold)
    #     incoming_penalty = torch.mean((F.softplus(threshold - incoming_weights, beta=beta) / threshold) ** 2)
    #     return incoming_penalty

    # def afferents_variability_penalty(self, percent=0.5, min_threshold=0.02, beta=20.0):
    #     '''
    #     Encourages incoming weight variability (seen in the units with the highest participation)
    #     If the incoming (excitatory) weights variability is below a certain threshold, it is penalized for it.
    #     '''
    #     if not self.RNN.dale_mask is None:
    #         exc_inds = torch.where(self.RNN.dale_mask == 1.0)[0]
    #         incoming_weights_varaibility = self.RNN.W_rec[:, exc_inds].abs().std(dim=1)
    #     else:
    #         incoming_weights_varaibility = self.RNN.W_rec.abs().std(dim=1)
    #     min_threshold = torch.tensor(min_threshold, device=self.RNN.device)
    #     threshold = torch.maximum(torch.quantile(incoming_weights_varaibility, percent).detach(), min_threshold)
    #     weights_variability_penalty = torch.mean((F.softplus(threshold - incoming_weights_varaibility, beta=beta) / threshold) ** 2)
    #     return weights_variability_penalty

    def train_step(self, input, target_output, mask):
        # Forward pass
        states, predicted_output = self.RNN(input, w_noise=True, dropout=self.dropout, drop_rate=self.drop_rate)

        behavior_mismatch_penalty = self.criterion(target_output[:, mask, :], predicted_output[:, mask, :])

        # Helper: scalar zero on the right device/dtype
        def z():
            return torch.zeros((), device=states.device, dtype=behavior_mismatch_penalty.dtype)

        # ---- Compute RAW penalties first (unscaled) ----
        channel_overlap_raw = self.channel_overlap_penalty() if self.lambda_orth != 0 else None
        activity_raw = self.activity_penalty(states) if self.lambda_r != 0 else None
        net_inputs_raw = self.net_inputs_penalty(states, input) if self.lambda_h != 0 else None
        net_inputs_var_raw = self.net_inputs_variability_penalty(states, input) if self.lambda_hvar != 0 else None
        # isolation_raw = self.isolation_penalty() if self.lambda_iz != 0 else None
        # categorical_raw = self.categorical_penalty(states) if self.lambda_cat != 0 else None

        # ---- Scaled terms (Torch zeros when off) ----
        channel_overlap_penalty = self.lambda_orth * channel_overlap_raw if channel_overlap_raw is not None else z()
        activity_penalty = self.lambda_r * activity_raw if activity_raw is not None else z()
        net_inputs_penalty = self.lambda_h * net_inputs_raw if net_inputs_raw is not None else z()
        net_inputs_variability_penalty = self.lambda_hvar * net_inputs_var_raw if net_inputs_var_raw is not None else z()
        # isolation_penalty = self.lambda_iz * isolation_raw if isolation_raw is not None else z()
        # categorical_penalty = self.lambda_cat * categorical_raw if categorical_raw is not None else z()

        # Diagnostics: participation & Gini
        participation = torch.std(states, dim=(1, 2)) + torch.mean(states, dim=(1, 2))
        gini_participation = self.gini_evenness(participation)

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
        gnorm_activity = grad_norm_of(activity_penalty)
        gnorm_channel_overlap = grad_norm_of(channel_overlap_raw)
        gnorm_net_inputs_var = grad_norm_of(net_inputs_var_raw)
        gnorm_net_inputs_mean = grad_norm_of(net_inputs_raw)
        # gnorm_categorical = grad_norm_of(categorical_raw)
        # gnorm_isolation = grad_norm_of(isolation_raw)
        # (activity grad norm available if you add a key; see note below)
        # gnorm_activity       = grad_norm_of(activity_raw)

        # ---- Total loss ----
        loss = (behavior_mismatch_penalty
                + channel_overlap_penalty
                + activity_penalty
                + net_inputs_penalty
                + net_inputs_variability_penalty)
                # + isolation_penalty
                # + categorical_penalty)

        # Backprop + step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # ---- Logging ----
        self.loss_monitor["behavior"].append(self.to_item(behavior_mismatch_penalty))
        self.loss_monitor["channel overlap"].append(self.to_item(channel_overlap_penalty))
        self.loss_monitor["activity"].append(self.to_item(activity_penalty))
        self.loss_monitor["net inputs mean"].append(self.to_item(net_inputs_penalty))
        self.loss_monitor["net inputs variability"].append(self.to_item(net_inputs_variability_penalty))
        # self.loss_monitor["isolation"].append(self.to_item(isolation_penalty))
        # self.loss_monitor["categorical"].append(self.to_item(categorical_penalty))

        self.indicators_monitor["gini participation"].append(self.to_item(gini_participation))
        self.indicators_monitor["grad norm behavior"].append(self.to_item(gnorm_behavior))
        self.indicators_monitor["grad norm activity"].append(self.to_item(gnorm_activity))
        self.indicators_monitor["grad norm channel overlap"].append(self.to_item(gnorm_channel_overlap))
        self.indicators_monitor["grad norm net inputs variability"].append(self.to_item(gnorm_net_inputs_var))
        self.indicators_monitor["grad norm net inputs mean"].append(self.to_item(gnorm_net_inputs_mean))
        # self.indicators_monitor["grad norm isolation"].append(self.to_item(gnorm_isolation))
        # self.indicators_monitor["grad norm categorical"].append(self.to_item(gnorm_categorical))

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