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
from trainRNNbrain.training.training_utils import multi_iqr_scale
from trainRNNbrain.utils import jsonify
from pathlib import Path

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
                 lambda_met = 0.1,
                 lambda_si=0.1,
                 lambda_hi=0.1,
                 lambda_htvar=0.05,
                 lambda_hlvar=0.9,
                 lambda_cl=0.05,
                 cap_low = 0.05,
                 cap_high = 0.7,
                 penalize_low = False,
                 inequality_method='hhi',
                 dropout=False,
                 drop_rate=0.05,
                 p=2):
        self.UpV = 100 # units per unit of volume, hard constant!
        self.RNN = RNN
        self.max_sigma_rec = self.RNN.sigma_rec
        self.max_sigma_inp = self.RNN.sigma_inp
        self.Task = Task
        self.optimizer = optimizer
        # make sure masks exist, on the right device, and don’t require grad
        for name in ['recurrent_mask','input_mask','output_mask']:
            m = getattr(self.RNN, name, None)
            if m is not None:
                m.requires_grad_(False)
                if m.device != self.RNN.W_rec.device:
                    setattr(self.RNN, name, m.to(self.RNN.W_rec.device))
        self.register_mask_hooks()

        self.max_iter = max_iter
        self.tol = tol
        # name of the penalty, it's scale and dictionary of arguments to be passed
        self.penalty_map = {
            "inp_weights_magnitude": (self.inp_weights_magnitude_penalty, lambda_iwm, {"cap100":0.5}),
            "rec_weights_magnitude": (self.rec_weights_magnitude_penalty, lambda_rwm, {"cap100":0.07, "N_ref":100, "k_ref":20}),
            "out_weights": (self.out_weights_penalty, lambda_ow, {"c":2.0, "cap100":0.3, "gamma":1.0}),
            "rec_weights_sparsity": (self.rec_weights_sparsity_penalty, lambda_rws, {"tg_deg":20}),
            "output_var": (self.trial_output_var_penalty, lambda_tv, {}),
            "channel_overlap": (self.channel_overlap_penalty, lambda_orth, {"orth_input_only":orth_input_only}),
            "s_magnitude": (self.s_magnitude_penalty, lambda_sm, {"cap_low": cap_low, "cap_high": cap_high, "q":0.9, "alpha":5.0, "penalize_low": penalize_low}),
            "metabolic": (self.metabolic_penalty, lambda_met, {}),
            "s_inequality": (self.s_inequality_penalty, lambda_si, {"method":inequality_method}),
            "h_inequality": (self.h_inequality_penalty, lambda_hi, {"method":inequality_method}),
            "h_time_variance": (self.h_time_variance_penalty, lambda_htvar, {}),
            "h_local_variance": (self.h_local_variance_penalty, lambda_hlvar, {}),
            "clustering": (self.clustering_penalty, lambda_cl, {}),
        }
        self.loss_monitor = {'behavior': [], **{k: [] for k in self.penalty_map}}
        self.gradients_monitor = {'g_behavior': [], **{f"g_{k}": [] for k in self.penalty_map}}
        self.scaled_gradients_monitor = {'sg_behavior': [], **{f"sg_{k}": [] for k in self.penalty_map}}
        self.p, self.dropout, self.drop_rate = p, dropout, drop_rate
        self.participation = (1e-6 * torch.ones(self.RNN.N, device=self.RNN.device)) if self.dropout else None
        self.iter_n = 0

    def register_mask_hooks(self):
        """
        Optional one-time setup: attach grad hooks so masked coords never receive grads.
        Still keep enforce_masks() post-step for hard zeros and state cleanup.
        """
        R = self.RNN
        if getattr(self.RNN, 'recurrent_mask', None) is not None:
            R.W_rec.register_hook(lambda g: g * R.recurrent_mask)
        if getattr(self.RNN, 'input_mask', None) is not None:
            R.W_inp.register_hook(lambda g: g * R.input_mask)
        if getattr(self.RNN, 'output_mask', None) is not None:
            R.W_out.register_hook(lambda g: g * R.output_mask)

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

    def inp_weights_magnitude_penalty(self, states, input, cap100=0.5, alpha=5.0, beta=1.0, eps=1e-12):
        dev, dt = self.RNN.device, states.dtype
        N, U = self.RNN.N, self.UpV
        scale = torch.log1p(torch.as_tensor(N, device=dev, dtype=dt)) / torch.log1p(
            torch.as_tensor(U, device=dev, dtype=dt))
        cap = torch.as_tensor(cap100, device=dev, dtype=dt) * (U / N) * scale
        A = self.RNN.W_inp.abs()
        e = torch.log2((A + eps) / (cap + eps))
        under = torch.relu(-e).pow(2)
        over = torch.pow(2.0, alpha * torch.relu(e)) - 1.0
        return (under + beta * over).mean()

    def out_weights_penalty(self, states, input, c=2.0, cap100=0.3, gamma=1.0, alpha=5.0, beta=1.0, eps=1e-12):
        dev, dt = self.RNN.device, states.dtype
        N, U = self.RNN.N, self.UpV
        scale = torch.log1p(torch.as_tensor(N, device=dev, dtype=dt)) / torch.log1p(
            torch.as_tensor(U, device=dev, dtype=dt))
        cap = torch.as_tensor(cap100, device=dev, dtype=dt) * (U / N) * scale
        A = self.RNN.W_out.abs()
        p_l1 = (A.sum(1) - torch.as_tensor(c, device=dev, dtype=dt)).pow(2).mean()
        e = torch.log2((A + eps) / (cap + eps))
        under = torch.relu(-e).pow(2)
        over = torch.pow(2.0, alpha * torch.relu(e)) - 1.0
        p_cap = (under + beta * over).mean()
        P = A / (A.sum(1, keepdim=True) + eps)
        n = A.size(1)
        hhi = (P * P).sum(1)
        p_hhi = ((n * hhi - 1.0) / (n - 1.0 + eps)).mean()
        return p_l1 + p_cap + gamma * p_hhi

    def rec_weights_magnitude_penalty(self, states, input, cap100=0.07, N_ref=100, k_ref=20, alpha=5.0, beta=1.0, eps=1e-12):
        R, dev, dt = self.RNN, self.RNN.device, states.dtype
        N = R.N
        scale = torch.log1p(torch.as_tensor(self.UpV, device=dev, dtype=dt)) / torch.log1p(torch.as_tensor(N, device=dev, dtype=dt))
        cap = torch.as_tensor(cap100, device=dev, dtype=dt) * scale
        cap_e, cap_i = cap, cap * torch.as_tensor(R.exc2inhR, device=dev, dtype=dt)
        W = R.W_rec.abs()
        exc, inh = (R.dale_mask > 0), (R.dale_mask < 0)
        eE = torch.log2((W[:, exc] + eps) / (cap_e + eps))
        eI = torch.log2((W[:, inh] + eps) / (cap_i + eps))
        pE = (torch.relu(-eE).pow(2) + beta * (torch.pow(2.0, alpha * torch.relu(eE)) - 1.0)).mean()
        pI = (torch.relu(-eI).pow(2) + beta * (torch.pow(2.0, alpha * torch.relu(eI)) - 1.0)).mean()
        return (pE + pI) * (N / (N_ref * k_ref))

    def rec_weights_sparsity_penalty(self, states, ipnut, tg_deg=20, eps=1e-8):
        W = self.RNN.W_rec  # (N, N)
        l1 = W.abs().sum(dim=1)
        l2 = (W.square().sum(dim=1) + eps).sqrt()
        S = (l1 * l1) / (l2 * l2)  # effective support per row
        over = torch.relu(S - tg_deg)
        return (over * over).mean() / (tg_deg * tg_deg)

    def s_magnitude_penalty(self, states, input, cap_low=0.05, cap_high=0.7, q=0.9, alpha=5.0, eps=1e-12,
                            penalize_low=False):
        x = states.abs().view(states.size(0), -1)
        dev, dt = states.device, states.dtype
        sN = torch.log1p(torch.as_tensor(self.UpV, device=dev, dtype=dt)) / \
             torch.log1p(torch.as_tensor(self.RNN.N, device=dev, dtype=dt))  # ~ O(1/log N)
        clo = torch.as_tensor(cap_low, device=dev, dtype=dt) * sN
        chi = torch.as_tensor(cap_high, device=dev, dtype=dt) * sN
        eps = torch.as_tensor(eps, device=dev, dtype=dt)

        s_q = torch.quantile(x + eps, q, dim=1)  # per-neuron summary

        if penalize_low:
            e_lo = torch.log((s_q + eps) / (clo + eps))  # <0 if below lower cap
            under = torch.pow(torch.relu(-e_lo), 2)  # quadratic push from zero
        else:
            under = 0

        e_hi = torch.log((s_q + eps) / (chi + eps))  # >0 if above upper cap

        over = torch.expm1(alpha * torch.relu(e_hi))  # steep over-cap
        return (under + over).mean()

    def metabolic_penalty(self, states, input):
        return torch.mean(states ** 2)

    def channel_overlap_penalty(self, states, input, orth_input_only, eps=1e-8, tiny=1e-12):
        R = self.RNN
        B = R.W_inp if orth_input_only else torch.cat((R.W_inp, R.W_out.T), dim=1)
        denom = torch.linalg.vector_norm(B, dim=0, keepdim=True).clamp_min(eps)
        B = B / denom
        G = B.T @ B
        M = G.size(0)
        if M <= 1: return torch.zeros((), device=G.device, dtype=G.dtype)
        i, j = torch.tril_indices(M, M, offset=-1, device=G.device)
        x = (G[i, j] * G[i, j]).mean()
        return torch.sqrt(x + tiny)

    def s_inequality_penalty(self, states, input, method='hhi'):
        activity = torch.mean(torch.abs(states), dim=(1, 2))  # (N,)
        if method == 'gini':
            method_fn = self.gini_penalty
        elif method == 'hhi':
            method_fn = self.hhi_penalty
        else:
            raise NotImplementedError
        return method_fn(activity)

    def h_inequality_penalty(self, states, input, method='hhi'):
        h = (torch.einsum('ij,jkl->ikl', self.RNN.W_rec, states) +
             torch.einsum('ij,jkl->ikl', self.RNN.W_inp, input))
        mean_h = torch.mean(h, dim=(1, 2))  # (N,)
        if method == 'gini':
            method_fn = self.gini_penalty
        elif method == 'hhi':
            method_fn = self.hhi_penalty
        else:
            raise NotImplementedError
        return method_fn(mean_h)

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

    def combine_grads(self, loss_main, penalty_raw, params):
        """
        Behavior-safe gradient composition.
        - Computes g_main = ∂loss_main/∂params.
        - For each penalty k with λ_k, forms g_k = ∂(λ_k * L_k)/∂params.
        - If ⟨g_k, g_main⟩ < 0, projects g_k to remove the *negative* component along g_main.
        - Returns summed list of tensors matching params.
        """
        def fg(L):
            g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
            return [gi if gi is not None else torch.zeros_like(p) for gi, p in zip(g, params)]

        dot = lambda a, b: sum((ai * bi).sum() for ai, bi in zip(a, b))

        g_main = fg(loss_main)
        denom = dot(g_main, g_main) + 1e-20
        g_tot = [gm.clone() for gm in g_main]

        for k, (fn, lam, kwargs) in self.penalty_map.items():
            if lam == 0: continue
            Lk = penalty_raw[k]
            gk = fg(lam * Lk)
            s = dot(gk, g_main) / denom
            if s < 0:
                gk = [gi - s * gm for gi, gm in zip(gk, g_main)]
            g_tot = [gt + gi for gt, gi in zip(g_tot, gk)]
        return g_tot

    def mask_param_state(self, p, m):
        """Zero Adam moments at masked entries so momentum/variance can’t resurrect zeros."""
        st = self.optimizer.state.get(p, None)
        if st is None: return
        t = st.get('exp_avg', None)
        if t is not None: t.mul_(m)
        t = st.get('exp_avg_sq', None)
        if t is not None: t.mul_(m)

    def mask_grads(self):
        """
        Apply structural masks to parameter *gradients* (pre-step).
        Covers recurrent, input, and output matrices if masks exist on self.RNN.
        """
        R = self.RNN
        if getattr(R, 'recurrent_mask', None) is not None and getattr(R.W_rec, 'grad', None) is not None:
            R.W_rec.grad.mul_(R.recurrent_mask)
        if getattr(R, 'input_mask', None) is not None and getattr(R.W_inp, 'grad', None) is not None:
            R.W_inp.grad.mul_(R.input_mask)
        if getattr(R, 'output_mask', None) is not None and getattr(R.W_out, 'grad', None) is not None:
            R.W_out.grad.mul_(R.output_mask)

    def enforce_masks(self):
        """Post-step: hard-zero masked weights + Adam moments."""
        with torch.no_grad():
            for w_name, m_name in (('W_rec', 'recurrent_mask'),
                                   ('W_inp', 'input_mask'),
                                   ('W_out', 'output_mask')):
                W = getattr(self.RNN, w_name, None)
                M = getattr(self.RNN, m_name, None)
                if W is None or M is None: continue
                # optional if masks might live on wrong device/dtype:
                # if M.device != W.device or M.dtype != W.dtype: M = M.to(W.device).type_as(W)
                W.mul_(M)  # hard zeros on weights
                self.mask_param_state(W, M)  # zero Adam moments at masked coords
        return None

    def enforce_io_nonnegativity(self):
        with torch.no_grad():
            self.RNN.W_inp.clamp_min_(1e-12)
            self.RNN.W_out.clamp_min_(1e-12)
        return None

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

    def log_step(self, pred_full, tgt, mask, behavior_loss, penalty_raw, grads_diag):
        """Compute scalar losses (detached) and append precomputed grad norms."""
        with torch.no_grad():
            loss_log = behavior_loss + sum(lam * penalty_raw[k] for k, (_, lam, kwargs) in self.penalty_map.items() if lam != 0)
            loss_val = float(loss_log.detach().cpu().item())
            y = pred_full[:, mask, :]
            t = tgt[:, mask, :]
            mse = (y - t).pow(2).mean()
            sst = (t - t.mean()).pow(2).mean().clamp_min(1e-12)
            r2_val = float((1.0 - mse / sst).item())

        # monitors (use precomputed grads_diag; no autograd here)
        self.loss_monitor["behavior"].append(float(behavior_loss.detach().cpu().item()))
        self.gradients_monitor["g_behavior"].append(float(grads_diag["g_behavior"]))
        self.scaled_gradients_monitor["sg_behavior"].append(float(grads_diag["g_behavior"]))
        for k in self.penalty_map:
            val = self.penalty_map[k][1] * penalty_raw[k] if self.penalty_map[k][1] != 0 else torch.zeros((), device=self.RNN.device)
            self.loss_monitor[k].append(float(val.detach().cpu().item()))
            self.gradients_monitor[f"g_{k}"].append(float(grads_diag[f"g_{k}"]))
            self.scaled_gradients_monitor[f"sg_{k}"].append(self.penalty_map[k][1] * self.gradients_monitor[f"g_{k}"][-1])
        return loss_val, r2_val

    def train_step(self, input, target_output, mask):

        # noise schedule
        scale = torch.as_tensor(self.max_iter / 12, dtype=torch.float32, device=self.RNN.device)
        center = torch.as_tensor(self.max_iter / 3, dtype=torch.float32, device=self.RNN.device)
        mult = 1.0 / (1.0 + torch.exp(-(self.iter_n - center) / scale))
        self.RNN.sigma_rec = self.max_sigma_rec * mult
        self.RNN.sigma_inp = self.max_sigma_inp * mult

        # forward passes
        states_full, pred_full = self.RNN(input, w_noise=True, dropout=False, drop_rate=None)
        if self.dropout:
            self.participation = (states_full.abs().quantile((1, 2), q=0.9) + states_full.std((1, 2), unbiased=False)).detach()
            _, pred_do = self.RNN(input, w_noise=True, dropout=True, drop_rate=self.drop_rate, participation=self.participation)
            behavior_loss = self.behavior_penalty(pred_do, target_output, mask)
        else:
            behavior_loss = self.behavior_penalty(pred_full, target_output, mask)

        # penalties
        penalty_raw = {k: (fn(states_full, input, **kwargs) if lam != 0 else None) for k, (fn, lam, kwargs) in self.penalty_map.items()}

        # PRE-STEP grad diagnostics
        params = [p for p in self.RNN.parameters() if p.requires_grad]

        def grad_norm(x):
            if x is None or not getattr(x, "requires_grad", False): return 0.0
            g = torch.autograd.grad(x, params, retain_graph=True, create_graph=False, allow_unused=True)
            s = torch.zeros((), device=self.RNN.device)
            for gi in g:
                if gi is not None: s = s + (gi.detach() ** 2).sum()
            return float(s.sqrt().detach().cpu().item())

        grads_diag = {f"g_{k}": grad_norm(penalty_raw[k]) for k in self.penalty_map}
        grads_diag["g_behavior"] = grad_norm(behavior_loss)

        # log before mutations
        loss_val, r2_val = self.log_step(pred_full, target_output, mask, behavior_loss, penalty_raw, grads_diag)

        # compose & step (hooks will mask grads automatically)
        g_tot = self.combine_grads(behavior_loss, penalty_raw, params)
        self.optimizer.zero_grad(set_to_none=True)
        for p, g in zip(params, g_tot): p.grad = g
        # self.mask_grads()  # ← remove when hooks are registered
        self.optimizer.step()

        # post-step constraints: apply transforms first, then structural masks last
        self.enforce_dale()
        self.enforce_io_nonnegativity()
        self.enforce_masks()

        self.iter_n += 1
        return loss_val, r2_val

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

    def run_training(self, train_mask, same_batch=False, shuffle=False, val_every=None):
        R, dev = self.RNN, self.RNN.device
        to_tensor = lambda x: torch.from_numpy(x.astype('float32')).to(dev)

        def get_batch():
            inp, tgt, cond = self.Task.get_batch(shuffle=shuffle)
            return to_tensor(inp), to_tensor(tgt)

        train_losses, val_losses = [], []
        R.train()
        best_loss = float('inf')
        best_params = deepcopy(R.get_params())

        if same_batch:
            input_batch, target_batch = get_batch()

        tic = time.perf_counter()
        for it in range(1, self.max_iter + 1):
            if not same_batch:
                input_batch, target_batch = get_batch()

            train_loss, r2 = self.train_step(input=input_batch, target_output=target_batch, mask=train_mask)
            train_losses.append(train_loss)

            # track best
            if train_loss <= best_loss:
                best_loss = train_loss
                best_params = deepcopy(R.get_params())

            # optional validation
            if val_every and (it % val_every == 0):
                R.eval()
                with torch.no_grad():
                    v_inp, v_tgt = get_batch()
                    # simple val metric: reuse train_step forward bits (or expose a val_step)
                    states_full, pred_full = R(v_inp, w_noise=False, dropout=False, drop_rate=None)
                    y = pred_full[:, train_mask, :]
                    t = v_tgt[:, train_mask, :]
                    mse = (y - t).pow(2).mean()
                    sst = (t - t.mean()).pow(2).mean().clamp_min(1e-12)
                    val_r2 = float((1.0 - mse / sst).item())
                    val_losses.append(float(mse.detach().cpu().item()))
                R.train()

            # timing (CUDA-safe if needed)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            dt = time.perf_counter() - tic
            elapsed = time.strftime("%H:%M:%S", time.gmtime(dt))
            eta = time.strftime("%H:%M:%S", time.gmtime((dt / it) * self.max_iter - dt))
            self.print_iteration_info(it, self.max_iter, train_loss, best_loss, r2, elapsed, eta)

        last_params = deepcopy(R.get_params())
        return R, train_losses, val_losses, best_params, last_params






