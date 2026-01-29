'''
Class which accepts RNN_torch and a task and has a mode to train RNN
'''
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import time
from trainRNNbrain.training.training_utils import multi_iqr_scale
from dataclasses import dataclass

@dataclass
class Penalties:
    '''Collection of penalty methods for RNN training.'''
    def __init__(self, RNN):
        self.RNN = RNN
        self.UpV = 100 # N units per unit of volume, hard constant!

    def task_penalty(self, states, input, output, target, mask):
        return ((output[:, mask, :] - target[:, mask, :]) ** 2).mean()
    
    def inp_weights_magnitude_penalty(self, states, input=None, output=None, target=None, mask=None, cap100=0.5, gamma=5.0, eps=1e-12):
        dev, dt = states.device, states.dtype
        N, U = states.size(0), states.size(1)
        scale = torch.log1p(torch.as_tensor(N, device=dev, dtype=dt)) / torch.log1p(
            torch.as_tensor(U, device=dev, dtype=dt))
        cap = torch.as_tensor(cap100, device=dev, dtype=dt) * (U / N) * scale
        A = self.RNN.W_inp.abs()
        r = (A + eps) / (cap + eps)
        over = torch.pow(torch.relu(r - 1) + 1.0, gamma) - 1.0
        return over.mean()

    def out_weights_penalty(self, states, input=None, output=None, target=None, mask=None, c=2.0, cap100=0.3, alpha=1.0, gamma=5.0, eps=1e-12):
        dev, dt = states.device, states.dtype
        N, U = states.size(0), states.size(1)
        scale = torch.log1p(torch.as_tensor(N, device=dev, dtype=dt)) / torch.log1p(
            torch.as_tensor(U, device=dev, dtype=dt))
        cap = torch.as_tensor(cap100, device=dev, dtype=dt) * (U / N) * scale
        A = self.RNN.W_out.abs()
        p_l1 = (A.sum(1) - torch.as_tensor(c, device=dev, dtype=dt)).pow(2).mean()
        r = (A + eps) / (cap + eps)
        over = torch.pow(torch.relu(r - 1) + 1.0, gamma) - 1.0
        p_cap = over.mean()
        P = A / (A.sum(1, keepdim=True) + eps)
        n = A.size(1)
        hhi = (P * P).sum(1)
        p_hhi = ((n * hhi - 1.0) / (n - 1.0 + eps)).mean()
        return p_l1 + p_cap + alpha * p_hhi

    def rec_weights_magnitude_penalty(self, states, input=None, output=None, target=None, mask=None,
                                       cap100=0.07, N_ref=100, k_ref=20, gamma=5.0, eps=1e-12):
        R, dev, dt = self.RNN, states.device, states.dtype
        N = R.N
        scale = torch.log1p(torch.as_tensor(self.UpV, device=dev, dtype=dt)) / torch.log1p(torch.as_tensor(N, device=dev, dtype=dt))
        cap = torch.as_tensor(cap100, device=dev, dtype=dt) * scale
        cap_e, cap_i = cap, cap * torch.as_tensor(R.exc2inhR, device=dev, dtype=dt)
        W = R.W_rec.abs()
        exc, inh = (R.dale_mask > 0), (R.dale_mask < 0)
        rE = (W[:, exc] + eps) / (cap_e + eps)
        rI = (W[:, inh] + eps) / (cap_i + eps)
        pE = (torch.pow(torch.relu(rE - 1.0) + 1.0, gamma) - 1.0)
        pI = (torch.pow(torch.relu(rI - 1.0) + 1.0, gamma) - 1.0)
        return (pE + pI).mean() * (N / (N_ref * k_ref))

    def rec_weights_sparsity_penalty(self, states, input=None, output=None, target=None, mask=None, tg_deg=20, eps=1e-12):
        W = self.RNN.W_rec  # (N, N)
        l1 = W.abs().sum(dim=1)
        l2 = (W.square().sum(dim=1) + eps).sqrt()
        S = (l1 * l1) / (l2 * l2)  # effective support per row
        over = torch.relu(S - tg_deg)
        return (over ** 2).mean() / (tg_deg ** 2)

    def s_magnitude_penalty(self, states, input=None, output=None, target=None, mask=None,
                            cap_s=0.3,
                            quantile_kind='hard',
                            q=0.9, p=15, tau=0.1,
                            penalty_type="additive",
                            g_top=5.0, g_bot=5.0,
                            alpha=1.0, beta=1.0, eps=1e-12):
        '''
        exponential penalty on the up side, quadratic penalty on the down side of log(activity ratio)
        '''
        x = states.abs().view(states.size(0), -1)  # (N, T*B)
        dev, dt = states.device, states.dtype
        cap_s = torch.as_tensor(cap_s, device=dev, dtype=dt)

        scale = torch.log1p(torch.as_tensor(self.UpV, device=dev, dtype=dt)) / \
                torch.log1p(torch.as_tensor(self.RNN.N, device=dev, dtype=dt))
        cap = cap_s * scale  # scales as O(1 / log(N))

        eps = torch.as_tensor(eps, device=dev, dtype=dt)
        if quantile_kind == 'hard':
            activity = torch.quantile(x + eps, q, dim=1)  # per-neuron summary
        elif quantile_kind == 'power_mean':
            activity = torch.mean((x + eps).pow(p), dim=1).pow(1.0 / p)
        elif quantile_kind == 'logsumexp':
            activity = a = x / tau
            activity = tau * (torch.logsumexp(a, dim=1) - torch.log(torch.as_tensor(a.size(1), device=a.device, dtype=a.dtype)))

        if penalty_type == "multiplicative":
            e = torch.log((activity + eps) / (cap + eps))  # >0 over, <0 under
            p_under = (torch.expm1(g_bot * torch.relu(-e)))  # quadratic penalty for under activitated units
            p_over = (torch.expm1(g_top * torch.relu(e)))  # exponential penalty for over activated units
        elif penalty_type == "additive":
            over = torch.relu(activity - cap)
            under = torch.relu(cap - activity)
            p_over = torch.pow(over / (cap + eps), g_top)
            p_under = torch.pow(under / (cap + eps), g_bot)
        return (alpha * p_under + beta * p_over).mean()
    
    def h_magnitude_penalty(self, states, input=None, output=None, target=None, mask=None,
                            h_thr=-0.3,
                            quantile_kind='logsumexp',
                            q=0.9, p=15, tau=0.1,
                            penalty_type="additive",
                            eps=1e-12):
        '''
        exponential penalty on the up side, quadratic penalty on the down side of log(activity ratio)
        '''
        x = states.abs().view(states.size(0), -1)  # (N, T*B)
        dev, dt = states.device, states.dtype
        h_thr = torch.as_tensor(h_thr, device=dev, dtype=dt)

        scale = torch.log1p(torch.as_tensor(self.UpV, device=dev, dtype=dt)) / \
                torch.log1p(torch.as_tensor(self.RNN.N, device=dev, dtype=dt))
        cap = h_thr * scale  # scales as O(1 / log(N))

        eps = torch.as_tensor(eps, device=dev, dtype=dt)
        if quantile_kind == 'hard':
            activity = torch.quantile(x + eps, q, dim=1)  # per-neuron summary
        elif quantile_kind == 'power_mean':
            activity = torch.mean((x + eps).pow(p), dim=1).pow(1.0 / p)
        elif quantile_kind == 'logsumexp':
            activity = a = x / tau
            activity = tau * (torch.logsumexp(a, dim=1) - torch.log(torch.as_tensor(a.size(1), device=a.device, dtype=a.dtype)))

        if penalty_type == "multiplicative":
            e = torch.log((activity + eps) / (cap + eps))  # >0 over, <0 under
            p_under = (torch.expm1(2 * torch.relu(-e)))  # quadratic penalty for under activitated units
        elif penalty_type == "additive":
            under = torch.relu(cap - activity)
            p_under = torch.pow(under / (cap + eps), 2)
        return (p_under).mean()

    def metabolic_penalty(self, states, input=None, output=None, target=None, mask=None):
        return torch.mean(states ** 2)

    def channel_overlap_penalty(self, states=None, input=None, output=None, target=None, mask=None, orth_input_only=True, eps=1e-8):
        B = self.RNN.W_inp if orth_input_only else torch.cat((self.RNN.W_inp, self.RNN.W_out.T), dim=1)
        B = B / (torch.linalg.vector_norm(B, dim=0, keepdim=True) + eps)  # col unit-norm
        G = B.T @ B  # (M, M)
        M = G.shape[0]
        if M <= 1:
            return torch.zeros((), device=G.device, dtype=G.dtype)
        i, j = torch.tril_indices(M, M, offset=-1, device=G.device)
        return torch.sqrt((G[i, j]**2).mean())

    def gini_penalty_(
            self,
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

    def hhi_penalty_(
            self,
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

    def trial_output_var_penalty(self, states, input=None, output=None, target=None, mask=None, eps=1e-12):
        # states: (N, T, K), W_out: (n_out, N)
        yc = output - output.mean(dim=2, keepdim=True)  # center across trials (K)
        num = (yc * yc).mean()  # E_{o,t,k}[(y - ⟨y⟩_trial)^2]
        den = (output * output).mean().clamp_min(eps)  # normalize by overall power
        return num / den

    def s_inequality_penalty(self, states, input=None, output=None, target=None, mask=None, method='hhi'):
        activity = torch.mean(torch.abs(states), dim=(1, 2))  # (N,)
        if method == 'gini':
            method_fn = self.gini_penalty_
        elif method == 'hhi':
            method_fn = self.hhi_penalty_
        else:
            raise NotImplementedError
        return method_fn(activity)

    def h_inequality_penalty(self, states, input, output=None, target=None, mask=None, method='hhi'):
        h = (torch.einsum('ij,jkl->ikl', self.RNN.W_rec, states) +
             torch.einsum('ij,jkl->ikl', self.RNN.W_inp, input))
        mean_h = torch.mean(h, dim=(1, 2))  # (N,)
        if method == 'gini':
            method_fn = self.gini_penalty_
        elif method == 'hhi':
            method_fn = self.hhi_penalty_
        else:
            raise NotImplementedError
        return method_fn(mean_h)

    def h_time_variance_penalty(self, states, input, output=None, target=None, mask=None, eps=1e-8):
        h = (torch.einsum('ij,jkl->ikl', self.RNN.W_rec, states)
             + torch.einsum('ij,jkl->ikl', self.RNN.W_inp, input))
        mean_t = h.mean((0, 2))
        var_between = mean_t.var(unbiased=False)
        var_within = h.var((0, 2), unbiased=False).mean()
        denom = (var_between + var_within).detach() + eps
        return var_between / denom
    
    @staticmethod
    def normalized_concentration_(C, mask, eps=1e-8):
        X = C[:, mask]
        P = X / (X.sum(1, keepdim=True) + eps)
        hhi = (P * P).sum(1)
        base = 1.0 / X.size(1)
        return ((hhi - base) / (1.0 - base)).clamp(0, 1)

    def h_local_variance_penalty(self, states, input=None, output=None, target=None, mask=None):
        device, dtype = states.device, states.dtype
        mean_s = states.mean(dim=(1, 2))  # (N,)
        contrib = (self.RNN.W_rec * mean_s.unsqueeze(0)).abs()  # (N, N) rows=i (post), cols=j (pre)
        dale_cols = self.RNN.dale_mask.to(device)  # (N,)
        conc_E = self.normalized_concentration_(contrib, (dale_cols == 1))
        conc_I = self.normalized_concentration_(contrib, (dale_cols == -1))
        v = conc_E + conc_I
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v.mean()

    def clustering_penalty(self, states, input, output, target, mask,
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
    
    def eff_dim_tail_energy_penalty(self, states, input=None, output=None, target=None, mask=None, k=6, eps=1e-8):
        # states: (N, T, K)  ->  X: (N, D), D = T*K
        X = states.reshape(states.shape[0], -1)
        X = X - torch.mean(X, dim=1, keepdim=True)

        D = X.shape[1]
        C = (X @ X.T) / (D + eps)                      # (N, N) covariance-like (PSD)
        e = torch.linalg.eigvalsh(C).flip(0)           # descending eigenvalues

        k = int(k)
        tail = torch.sum(e[k:]) if k < e.numel() else torch.zeros((), device=e.device, dtype=e.dtype)
        return tail / (torch.sum(e).detach() + eps)    # scale-invariant tail energy


class Trainer():
    def __init__(self,
                 RNN, Task, optimizer,
                 max_iter=1000,
                 anneal_noise=True,
                 lambda_iwm=0.0,
                 iwm_args=None,
                 lambda_rwm=0.0,
                 rwm_args=None,
                 lambda_ow=0.0,
                 ow_args=None,
                 lambda_rws=0.05,
                 rws_args=None,
                 lambda_tv=0.0,
                 tv_args=None,
                 lambda_orth=0.3,
                 orth_args={"orth_input_only": True},
                 lambda_sm=0.005,
                 sm_args=None,
                 lambda_hm=0.0,
                 hm_args=None,
                 lambda_met = 0.0,
                 met_args=None,
                 lambda_si=0.0,
                 si_args=None,
                 lambda_hi=0.0,
                 hi_args=None,
                 lambda_htvar=0.0,
                 htvar_args=None,
                 lambda_hlvar=0.0,
                 hlvar_args=None,
                 lambda_cl=0.0,
                 cl_args = None,
                 lambda_effdim=0.0,
                 effdim_args=None,
                 dropout=False,
                 dropout_args=None,
                 monitor=True,
                 max_grad_norm=10.0):
        self.RNN = RNN
        self.Penalties = Penalties(RNN=self.RNN) # dataclass containing all the penalty methods
        self.max_sigma_rec = self.RNN.sigma_rec
        self.max_sigma_inp = self.RNN.sigma_inp
        self.Task = Task
        self.optimizer = optimizer
        self.monitor = monitor
        self.max_iter = max_iter
        self.max_grad_norm = max_grad_norm
        self.anneal_noise = anneal_noise
        
        # make sure masks exist, on the right device, and don’t require grad
        for name in ['recurrent_mask', 'input_mask', 'output_mask']:
            m = getattr(self.RNN, name, None)
            if m is not None:
                m.requires_grad_(False)
                if m.device != self.RNN.W_rec.device:
                    setattr(self.RNN, name, m.to(self.RNN.W_rec.device))
        
        # name of the penalty, it's scale (lambda) and dictionary of arguments to be passed
        self.penalty_map = {
            "task": (self.Penalties.task_penalty, 1.0, {}),
            "inp_weights_magnitude": (self.Penalties.inp_weights_magnitude_penalty, lambda_iwm, iwm_args),
            "rec_weights_magnitude": (self.Penalties.rec_weights_magnitude_penalty, lambda_rwm, rwm_args),
            "out_weights": (self.Penalties.out_weights_penalty, lambda_ow, ow_args),
            "rec_weights_sparsity": (self.Penalties.rec_weights_sparsity_penalty, lambda_rws, rws_args),
            "output_var": (self.Penalties.trial_output_var_penalty, lambda_tv, tv_args),
            "channel_overlap": (self.Penalties.channel_overlap_penalty, lambda_orth, orth_args),
            "s_magnitude": (self.Penalties.s_magnitude_penalty, lambda_sm, sm_args),
            "h_magnitude": (self.Penalties.h_magnitude_penalty, lambda_hm, hm_args),
            "metabolic": (self.Penalties.metabolic_penalty, lambda_met, met_args),
            "s_inequality": (self.Penalties.s_inequality_penalty, lambda_si, si_args),
            "h_inequality": (self.Penalties.h_inequality_penalty, lambda_hi, hi_args),
            "h_time_variance": (self.Penalties.h_time_variance_penalty, lambda_htvar, htvar_args),
            "h_local_variance": (self.Penalties.h_local_variance_penalty, lambda_hlvar, hlvar_args),
            "clustering": (self.Penalties.clustering_penalty, lambda_cl, cl_args),
            "eff_dim_tail_energy": (self.Penalties.eff_dim_tail_energy_penalty, lambda_effdim, effdim_args),
        }
        if monitor:
            self.loss_monitor = {**{k: [] for k in self.penalty_map}}
            self.gradients_monitor = {**{f"g_{k}": [] for k in self.penalty_map}}
            self.scaled_gradients_monitor = {**{f"sg_{k}": [] for k in self.penalty_map}}
        
        self.dropout = dropout
        self.dropout_args = dropout_args if dropout_args is not None else {"dropout_kind": None, "sampling_method": None, "drop_rate": 0.0, "dropout_beta": 1.0}
        self.participation = (1e-6 * torch.ones(self.RNN.N, device=self.RNN.device)) if self.dropout else None
        self.iter_n = 0


    @staticmethod
    def to_item_(x):
        return x.detach().cpu() if torch.is_tensor(x) else torch.as_tensor(x)
    
    @staticmethod
    def zero_(device):
        return torch.zeros((), device=device)

    def mask_param_state_(self, p, m):
        """Zero Adam moments at masked entries so momentum/variance can’t resurrect zeros."""
        st = self.optimizer.state.get(p, None)
        if st is None: return
        t = st.get('exp_avg', None)
        if t is not None: t.mul_(m)
        t = st.get('exp_avg_sq', None)
        if t is not None: t.mul_(m)
        return None

    @staticmethod
    def flat_grad_(loss, params, retain_graph=False, allow_unused=True):
        grads = torch.autograd.grad(
            loss, params,
            retain_graph=retain_graph,
            allow_unused=allow_unused
        )
        return [g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, params)]

    @staticmethod
    def dot_grads_(a, b):
        return sum(torch.sum(ga * gb) for ga, gb in zip(a, b))

    @staticmethod
    def get_task_safe_gradients_(params, penalty_map, penalty_dict_raw,
                                task_key="task", allow_unused=True):
        """Get gradients where extra penalties are projected to not hurt task performance."""
        task_loss = penalty_dict_raw[task_key]

        g_task = Trainer.flat_grad_(
            task_loss, params,
            retain_graph=True,
            allow_unused=allow_unused
        )
        nt = Trainer.dot_grads_(g_task, g_task) + 1e-20

        # Build total penalty tensor or None if nothing active
        tot_penalty = None
        for k, (_, L, _) in penalty_map.items():
            if k == task_key or L == 0:
                continue
            term = L * penalty_dict_raw[k]
            tot_penalty = term if tot_penalty is None else tot_penalty + term

        if tot_penalty is None:
            return [g.clone() for g in g_task]

        g_pen = Trainer.flat_grad_(
            tot_penalty, params,
            retain_graph=False,
            allow_unused=allow_unused
        )

        # Projection step is purely algebra on gradients; no need to track a graph here
        with torch.no_grad():
            s = Trainer.dot_grads_(g_pen, g_task) / nt
            if s < 0:
                g_pen = [gp - s * gb for gp, gb in zip(g_pen, g_task)]
            g_tot = [gb + gp for gb, gp in zip(g_task, g_pen)]
        return g_tot


    @staticmethod
    def grad_norm_of_(scalar, params, device):
        if scalar is None or not getattr(scalar, "requires_grad", False):
            return Trainer.zero_(device)
        grads = torch.autograd.grad(scalar, params, retain_graph=True, create_graph=False, allow_unused=True)
        s = Trainer.zero_(device)
        for g in grads:
            if g is not None:
                s = s + (g.detach() ** 2).sum()
        return s.sqrt()

    def enforce_masks_(self):
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
                self.mask_param_state_(W, M)  # zero Adam moments at masked coords
        return None

    def enforce_io_nonnegativity_(self):
        with torch.no_grad():
            self.RNN.W_inp.clamp_min_(1e-12)
            self.RNN.W_out.clamp_min_(1e-12)
        return None

    def enforce_dale_(self, eps=1e-12):
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

    def anneal_noise_levels_(self):
        # noise schedule
        scale = torch.as_tensor(self.max_iter / 12, dtype=torch.float32, device=self.RNN.device)
        center = torch.as_tensor(self.max_iter / 3, dtype=torch.float32, device=self.RNN.device)
        mult = 1.0 / (1.0 + torch.exp(-(self.iter_n - center) / scale))
        self.RNN.sigma_rec = self.max_sigma_rec * mult
        self.RNN.sigma_inp = self.max_sigma_inp * mult
        return None
    
    @staticmethod
    def r2_score(output, target, mask):
        y = output[:, mask, :]
        t = target[:, mask, :]
        r2 = 1.0 - (y - t).pow(2).mean() / (t - t.mean()).pow(2).mean().clamp_min(1e-12)
        r2_val = float(r2.item())
        return r2_val
    
    @staticmethod
    def get_participation_(states, q=0.9, eps=1e-8):
        x = states.abs().view(states.size(0), -1)  # (N, T*B)
        activity = torch.quantile(x + eps, q, dim=1)  # per-neuron summary
        activity_std = x.std(dim=1, unbiased=False)
        participation = activity + activity_std
        return participation
    
    def train_step(self, input, target_output, mask):
        if self.anneal_noise:
            self.anneal_noise_levels_()

        params = [p for p in self.RNN.parameters() if p.requires_grad]

        states, output_full = self.RNN(input, w_noise=True, dropout=False, dropout_args={})
        output_do = output_full
        if self.dropout:
            if self.dropout_args["sampling_method"] == "participation" and self.participation is None:
                self.participation = 1e-6 * torch.ones(self.RNN.N, device=states.device)
            part = self.participation if self.dropout_args["sampling_method"] == "participation" else None
            eta = self.dropout_args.get("eta", 0.0)
            _, output_do = self.RNN(input, w_noise=True, dropout=True, dropout_args=self.dropout_args, participation=part)

            if self.dropout_args["sampling_method"] == "participation":
                new_part = self.get_participation_(states, q=self.dropout_args["activity_q"], eps=1e-12).detach()
                self.participation = (1 - eta) * self.participation + eta * new_part

        penalty_dict_raw = {
            k: (fn(states, input, (output_full if (self.dropout and k=='task') else output_do), target_output, mask, **kwargs) if L != 0 else None)
            for k, (fn, L, kwargs) in self.penalty_map.items()
        }
        
        # --- 1) gradient norms for monitoring (uses autograd.grad) ---
        if self.monitor:
            grads = {
                f"g_{k}": Trainer.grad_norm_of_(penalty_dict_raw[k], params, self.RNN.device)
                for k in self.penalty_map
            }

        # --- 2) behavior/task-safe combined gradient (also uses autograd.grad) ---
        g_tot = Trainer.get_task_safe_gradients_(
            params, self.penalty_map, penalty_dict_raw,
            task_key="task", allow_unused=True
        )

        # --- 3) apply combined gradient ---
        self.optimizer.zero_grad(set_to_none=True)
        for p, g in zip(params, g_tot):
            p.grad = g
        
        torch.nn.utils.clip_grad_norm_(params, max_norm=self.max_grad_norm)
        self.optimizer.step()

        # --- 4) now it's safe to mutate weights in-place ---
        self.enforce_masks_()
        self.enforce_io_nonnegativity_()
        self.enforce_dale_()

        # --- 5) compute total loss and r2 for reporting ---
        loss_val = Trainer.zero_(self.RNN.device)
        for k, (_, L, _) in self.penalty_map.items():
            loss_k = penalty_dict_raw[k] if L != 0 else Trainer.zero_(self.RNN.device)
            loss_val += L * loss_k
        loss_val = float(loss_val.detach().cpu().item())
        r2_val = Trainer.r2_score(output_full, target_output, mask)
        
        # additional monitoring
        if self.monitor:
            penalty_dict_scaled = {
                k: (L * penalty_dict_raw[k] if L != 0 else Trainer.zero_(self.RNN.device))
                for k, (_, L, _) in self.penalty_map.items()
            }
            with torch.no_grad():
                Z = Trainer.zero_(self.RNN.device)
                for k, (_, L, _) in self.penalty_map.items():
                    loss_k = penalty_dict_scaled[k] if L != 0 else Z
                    grad_val = self.to_item_(grads[f"g_{k}"])
                    self.loss_monitor[k].append(self.to_item_(loss_k))
                    self.gradients_monitor[f"g_{k}"].append(grad_val)
                    self.scaled_gradients_monitor[f"sg_{k}"].append(L * grad_val)
        

        self.iter_n += 1
        return loss_val, r2_val

    def eval_step(self, inp, tgt, mask, noise=False, dropout=False, dropout_args=None, seed=None):
        if seed is not None: torch.manual_seed(seed)
        self.RNN.eval()
        dt = next(self.RNN.parameters()).dtype
        dev = self.RNN.device
        with torch.no_grad():
            srec, sinp = float(self.RNN.sigma_rec), float(self.RNN.sigma_inp)
            if not noise: self.RNN.sigma_rec = self.RNN.sigma_inp = 0.0
            inp, tgt = inp.to(dev, dt), tgt.to(dev, dt)
            mask = mask if isinstance(mask, slice) else torch.as_tensor(mask, device=dev)
            _, y = self.RNN(inp, w_noise=noise, dropout=dropout, dropout_args=dropout_args)
            r2 = Trainer.r2_score(y, tgt, mask)
            self.RNN.sigma_rec, self.RNN.sigma_inp = srec, sinp
        return float(r2)

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

        tic = time.perf_counter()
        # torch.autograd.set_detect_anomaly(True)
        for iter in range(self.max_iter):

            if not same_batch:
                input_batch, target_batch, conditions_batch = self.Task.get_batch(shuffle=shuffle)
                input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.RNN.device)
                target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.RNN.device)

            train_loss, r2 = self.train_step(input=input_batch,
                                         target_output=target_batch,
                                         mask=train_mask)

            toc = time.perf_counter()
            elapsed_t, eta = self.get_eta_(tic, toc, iter, self.max_iter)
            self.print_iteration_info_(iter + 1, self.max_iter, train_loss, min_train_loss, r2, elapsed_t, eta)
            train_losses.append(train_loss)
            if train_loss <= min_train_loss:
                min_train_loss = train_loss
                best_net_params = deepcopy(self.RNN.get_params())
        last_net_params = deepcopy(self.RNN.get_params())
        self.RNN.set_params(last_net_params) # assuming that the more training it went through - the better.
        return self.RNN, train_losses, val_losses, best_net_params, last_net_params
    
    @staticmethod
    def get_eta_(tic, toc, iter, max_iter):
        delta = toc - tic
        elapsed_t = time.strftime("%H:%M:%S", time.gmtime(delta))
        proj_total = (delta / (iter + 1)) * max_iter
        remaining = proj_total - delta
        eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
        return elapsed_t, eta
    
    @staticmethod
    def print_iteration_info_(
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
