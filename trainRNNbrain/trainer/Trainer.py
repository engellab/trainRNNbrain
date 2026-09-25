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
from collections import defaultdict, deque

def scored_(x, mask):
    """The scored entries of a (channels, T, B) tensor, as (channels, n_scored).

    `mask` is either the usual time index (1-d LongTensor / slice / ndarray: every trial scored at
    the same steps) or a per-trial boolean (T, B) tensor (a batch mixing tasks of different length
    and scoring window, see tasks/TaskMultiRule.py). Both reduce to the same mean over scored
    entries, so every loss and r2 below is unchanged for the time-index case.
    """
    if torch.is_tensor(mask) and mask.dtype == torch.bool and mask.dim() == 2:
        return x[:, mask]
    return x[:, mask, :].reshape(x.shape[0], -1)


@dataclass
class Penalties:
    '''Collection of penalty methods for RNN training.'''
    def __init__(self, RNN):
        self.RNN = RNN
        self.UpV = 100 # N units per unit of volume, hard constant!

    def task_penalty(self, states, input, output, target, mask):
        return ((scored_(output, mask) - scored_(target, mask)) ** 2).mean()
    
    def inp_weights_magnitude_penalty(self, states, input=None, output=None, target=None, mask=None, cap100=0.5, gamma=5.0, eps=1e-12):
        """Soft cap on |W_inp|, hinged at gamma. Cap scales with N only, as its siblings do.

        FIXED 2026-09-22. It read `N, U = states.size(0), states.size(1)`, so U was the TRIAL
        LENGTH, and the cap came out as cap100 * (T/N) * log1p(N)/log1p(T). Two things were wrong:
        the cap depended on how long a trial is, which is not a property of the network and should
        never enter a weight scale; and the log ratio was inverted relative to
        out_weights_magnitude_penalty, so the cap scaled the wrong way with N. As coded, the same
        1000-unit network got a cap of 0.182 on a T=300 task and 0.278 on a T=500 one, and going
        from N=1000 to N=4000 shrank it 3.3x where the intended form shrinks it 1.2x.

        `U` was almost certainly meant to be `self.UpV`, the same hard constant 100 that
        out_weights_magnitude_penalty and fr_magnitude_penalty use. That is what it now is.

        No result is affected: lambda_iwm is 0 in every config and launcher, so this penalty has
        never been applied to a run on disk.

        Args:
            states: (N, T, B), used only for device/dtype and N; cap100: cap at the N=UpV
            reference; gamma: hinge sharpness; eps: division guard.
        Returns:
            scalar penalty, mean over W_inp entries of ((relu(|W|/cap - 1) + 1)^gamma - 1).
        """
        dev, dt = states.device, states.dtype
        N = states.size(0)
        scale = torch.log1p(torch.as_tensor(self.UpV, device=dev, dtype=dt)) / torch.log1p(
            torch.as_tensor(N, device=dev, dtype=dt))
        cap = torch.as_tensor(cap100, device=dev, dtype=dt) * scale
        A = self.RNN.W_inp.abs()
        r = (A + eps) / (cap + eps)
        over = torch.pow(torch.relu(r - 1) + 1.0, gamma) - 1.0
        return over.mean()

    def out_weights_magnitude_penalty(self, states, input=None, output=None, target=None, mask=None, cap100=0.03, gamma=5.0, eps=1e-12):
        R, dev, dt = self.RNN, states.device, states.dtype
        N = R.N
        scale = torch.log1p(torch.as_tensor(self.UpV, device=dev, dtype=dt)) / torch.log1p(torch.as_tensor(N, device=dev, dtype=dt))
        cap = torch.as_tensor(cap100, device=dev, dtype=dt) * scale
        W = R.W_out.abs()
        r = (W + eps) / (cap + eps)
        over = torch.pow(torch.relu(r - 1.0) + 1.0, gamma) - 1.0
        return over.mean()

    # def rec_weights_magnitude_penalty(self, states, input=None, output=None, target=None, mask=None,
    #                                    cap100=0.07, N_ref=100, k_ref=20, gamma=5.0, eps=1e-12):
    #     R, dev, dt = self.RNN, states.device, states.dtype
    #     N = R.N
    #     scale = torch.log1p(torch.as_tensor(self.UpV, device=dev, dtype=dt)) / torch.log1p(torch.as_tensor(N, device=dev, dtype=dt))
    #     cap = torch.as_tensor(cap100, device=dev, dtype=dt) * scale
    #     cap_e, cap_i = cap, cap * torch.as_tensor(R.exc2inhR, device=dev, dtype=dt)
    #     W = R.W_rec.abs()
    #     exc, inh = (R.dale_mask > 0), (R.dale_mask < 0)
    #     rE = (W[:, exc] + eps) / (cap_e + eps)
    #     rI = (W[:, inh] + eps) / (cap_i + eps)
    #     pE = (torch.pow(torch.relu(rE - 1.0) + 1.0, gamma) - 1.0)
    #     pI = (torch.pow(torch.relu(rI - 1.0) + 1.0, gamma) - 1.0)
    #     return (pE.mean() + pI.mean()) * (N / (N_ref * k_ref))

    def rec_weights_magnitude_penalty(self, states, input=None, output=None, target=None, mask=None, account4dale=True,
                                       cap100=0.07, N_ref=100, k_ref=20, gamma=5.0, eps=1e-12):
        R, dev, dt = self.RNN, states.device, states.dtype
        N = R.N
        scale = torch.log1p(torch.as_tensor(self.UpV, device=dev, dtype=dt)) / torch.log1p(torch.as_tensor(N, device=dev, dtype=dt))
        cap = torch.as_tensor(cap100, device=dev, dtype=dt) * scale
        W = R.W_rec.abs()
        # account4dale needs a dale_mask; without Dale's law there is no E/I split to account for,
        # so fall back to the single-cap branch instead of indexing a None mask.
        if account4dale and getattr(R, "dale_mask", None) is not None:
            cap_e, cap_i = cap, cap * torch.as_tensor(R.exc2inhR, device=dev, dtype=dt)
            exc, inh = (R.dale_mask > 0), (R.dale_mask < 0)
            rE = (W[:, exc] + eps) / (cap_e + eps)
            rI = (W[:, inh] + eps) / (cap_i + eps)
            pE = (torch.pow(torch.relu(rE - 1.0) + 1.0, gamma) - 1.0)
            pI = (torch.pow(torch.relu(rI - 1.0) + 1.0, gamma) - 1.0)
            return (pE.mean() + pI.mean()) * (N / (N_ref * k_ref))
        else:
            r = (W + eps) / (cap + eps)
            p = (torch.pow(torch.relu(r - 1.0) + 1.0, gamma) - 1.0)
            return p.mean() * (N / (N_ref * k_ref))

    def rec_weights_sparsity_penalty(self, states, input=None, output=None, target=None, mask=None, tg_deg=20, eps=1e-12):
        W = self.RNN.W_rec  # (N, N)
        l1 = W.abs().sum(dim=1)
        l2 = (W.square().sum(dim=1) + eps).sqrt()
        S = (l1 * l1) / (l2 * l2)  # effective support per row
        over = torch.relu(S - tg_deg)
        return (over ** 2).mean() / (tg_deg ** 2)

    def fr_magnitude_penalty(self, states, input=None, output=None, target=None, mask=None,
                            cap_fr=0.3,
                            tau=0.1,
                            g_top=5.0, g_bot=5.0,
                            alpha=1.0, beta=1.0, eps=1e-12,
                            aggregation='mean',
                            tau_n=None):
        '''
        Firing rate magnitude penalty: MSE from the desired cap_fr (scaled by log(N)).

        aggregation controls how penalties are combined across neurons:
          'mean'      — simple mean over neurons (default; original behaviour).
          'logsumexp' — log-sum-exp aggregation, sensitive to outlier units.
                        Requires tau_n: small tau_n (→ 0) ≈ max; large (→ ∞) ≈ mean.
                        Formula: tau_n * (logsumexp(p / tau_n) - log(N)).
        '''
        x = states.view(states.size(0), -1)  # (N, T*B)
        if self.RNN.equation_type == "h":
            x = self.RNN.activation(x)
        # Do NOT take abs(x): for signed activations (GELU/tanh) abs lets a unit satisfy the target by
        # sitting negative (|-0.2| = 0.2 = cap). Using the SIGNED firing rate drives the (soft-max over
        # time) activity toward +cap, keeping units genuinely positive/active. No-op for r>=0 activations.
        dev, dt = states.device, states.dtype
        cap_fr = torch.as_tensor(cap_fr, device=dev, dtype=dt)

        scale = torch.log1p(torch.as_tensor(self.UpV, device=dev, dtype=dt)) / \
                torch.log1p(torch.as_tensor(self.RNN.N, device=dev, dtype=dt))
        cap = cap_fr * scale  # scales as O(1 / log(N))

        eps = torch.as_tensor(eps, device=dev, dtype=dt)
        activity = a = x / tau
        activity = tau * (torch.logsumexp(a, dim=1) - torch.log(torch.as_tensor(a.size(1), device=a.device, dtype=a.dtype)))

        over = torch.relu(activity - cap)
        under = torch.relu(cap - activity)
        # float64 for the powers ONLY. `over`/`under` are shape (N,) - the reduction over the
        # T*B axis already happened - so this costs N doubles, not the (N, T*B) tensor. The
        # function is unchanged; it is evaluated without overflowing. In fp32, (over/cap)^3
        # overflows at over/cap ~ 7e12, which an effective recurrent gain of only 1.10 reaches
        # after T=300 steps; the reported loss then goes inf while the GRADIENT is still finite,
        # so a non-finite-gradient guard never sees it. fp64 moves that wall to gain ~1.6.
        cap64 = cap.double() if torch.is_tensor(cap) else cap
        p_over  = torch.pow(over.double()  / (cap64 + eps), g_top)
        p_under = torch.pow(under.double() / (cap64 + eps), g_bot)

        if aggregation == 'logsumexp' and tau_n is not None:
            tau_n = torch.as_tensor(tau_n, device=dev, dtype=dt)
            log_N = torch.log(torch.as_tensor(float(p_over.shape[0]), device=dev, dtype=dt))
            agg_over  = tau_n * (torch.logsumexp(p_over  / tau_n, dim=0) - log_N)
            agg_under = tau_n * (torch.logsumexp(p_under / tau_n, dim=0) - log_N)
            return alpha * agg_under + beta * agg_over
        # default: mean aggregation
        return (alpha * p_under + beta * p_over).mean()
    
    def h_magnitude_penalty(self, states, input=None, output=None, target=None, mask=None,
                            h_thr=-0.1,
                            tau=0.1,
                            g_top=5.0, g_bot=5.0,
                            alpha=1.0, beta=1.0):
        '''
        
        '''
        x = states.abs().view(states.size(0), -1)  # (N, T*B)
        dev, dt = states.device, states.dtype
        h_thr = torch.as_tensor(h_thr, device=dev, dtype=dt)

        scale = torch.log1p(torch.as_tensor(self.UpV, device=dev, dtype=dt)) / \
                torch.log1p(torch.as_tensor(self.RNN.N, device=dev, dtype=dt))
        cap = h_thr * scale  # scales as O(1 / log(N))

        activity = a = x / tau
        activity = tau * (torch.logsumexp(a, dim=1) - torch.log(torch.as_tensor(a.size(1), device=a.device, dtype=a.dtype)))
        over = torch.relu(activity - cap)
        under = torch.relu(cap - activity)
        p_over = torch.pow(over, g_top)
        p_under = torch.pow(under, g_bot)
        return (alpha * p_under + beta * p_over).mean()

    def metabolic_penalty(self, states, input=None, output=None, target=None, mask=None):
        '''Metabolic cost: mean squared firing rate.'''
        if self.RNN.equation_type == "h":
            fr = self.RNN.activation(states)
        elif self.RNN.equation_type == "s":
            fr = states
        return torch.mean(fr ** 2)

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

    def fr_inequality_penalty(self, states, input=None, output=None, target=None, mask=None, method='hhi'):
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
    
    def clustering_penalty(self, states, input, output, target, mask,
                           attract_margin=0.1,
                           repell_margin=0.3,
                           diameter_quantile=0.9,
                           beta=20.0, eps=1e-8):
        X = states.view(states.shape[0], -1)
        if self.RNN.equation_type == 'h':
            X = self.RNN.activation(X)
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
        if self.RNN.equation_type == 'h':
            X = self.RNN.activation(X)
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
                 lambda_owm=0.0,
                 owm_args=None,
                 lambda_rws=0.05,
                 rws_args=None,
                 lambda_tv=0.0,
                 tv_args=None,
                 lambda_orth=0.3,
                 orth_args={"orth_input_only": True},
                 lambda_frm=0.005,
                 frm_args=None,
                 lambda_hm=0.0,
                 hm_args=None,
                 lambda_met = 0.0,
                 met_args=None,
                 lambda_fri=0.0,
                 fri_args=None,
                 lambda_hi=0.0,
                 hi_args=None,
                 lambda_htvar=0.0,
                 htvar_args=None,
                 lambda_cl=0.0,
                 cl_args = None,
                 lambda_effdim=0.0,
                 effdim_args=None,
                 synaptic_scaling=False,
                 scaling_args=None,
                 prune_reinit=False,
                 prune_args=None,
                 dropout=False,
                 dropout_args=None,
                 monitor=True,
                 task_safe_gradients=True,
                 track_participation=False,
                 track_every=10,
                 store_participation_every=None,
                 log_silent_every=None,
                 track_drift=False,
                 drift_lags=(100, 1000, 10000),
                 valid_batch=None,
                 track_valid_every=50,
                 max_grad_norm=10.0,
                 spike_factor=1e6,
                 gnorm_window=1000,
                 gnorm_min_samples=100,
                 snapshot_every=1000,
                 restore_after=500):
        self.RNN = RNN
        self.Penalties = Penalties(RNN=self.RNN) # dataclass containing all the penalty methods
        self.max_sigma_rec = self.RNN.sigma_rec
        self.max_sigma_inp = self.RNN.sigma_inp
        self.Task = Task
        # (inputs, target) held-out batch for the noise-free validation probe, or None. Supplied by
        # run_experiment so the Trainer stays task-agnostic.
        self.valid_batch = valid_batch
        # The held-out batch is nearly as large as the training one, so evaluating it at the
        # participation cadence costs ~18% of runtime. It changes far too slowly to need that
        # resolution, hence its own coarser cadence.
        self.track_valid_every = track_valid_every
        self.optimizer = optimizer
        self.monitor = monitor
        self.max_iter = max_iter
        self.max_grad_norm = max_grad_norm
        # --- gradient-spike defences (see train_step) -------------------------------------------
        # An update whose gradient norm exceeds spike_factor x the running scale is DROPPED rather
        # than clipped. Clipping is near-useless here: Adam normalises by sqrt(v), so tightening
        # max_grad_norm from 50 to 0.1 was measured to cut the damage of one spike only from 287
        # to 233 normal-sized steps. Skipping costs 0 - Adam's moments never see the spike.
        # ⚠️ spike_factor must be LARGE. At 100 the guard fired on 6-18% of ordinary gradients
        # once their distribution was heavy-tailed (measured on lognormal streams with sigma 3-5,
        # which is what an frm run near instability actually produces). A real catastrophe is
        # ~1e30x the normal norm - the observed non-finite cases came from gradients of ~1e27
        # against a normal ~1e-3 - so 1e6 still catches 50/50 injected catastrophes while its
        # false-positive rate stays at 0.00-0.30%. See tests/test_spike_guard_no_deadlock.py.
        self.spike_factor = spike_factor
        # ⚠️ The reference MUST be a rolling median over EVERY step, accepted or skipped. The first
        # version used an EMA updated only on ACCEPTED steps and it deadlocked: once the reference
        # was seeded low, every gradient exceeded spike_factor x it, so every step was skipped, so
        # the reference never updated. After restore_after skips the rollback set it to None, one
        # step was accepted, it re-seeded from that single sample, and the trap closed again -
        # a 1-in-(restore_after+1) acceptance ratio. Measured: 798 rollbacks x 500 skips = 99.8%
        # of 400000 iterations skipped, across 11 runs that all finished with negative r2.
        # A median over all observed norms cannot go stale, and is robust to the spikes themselves.
        self.gnorm_window = deque(maxlen=gnorm_window)
        self.gnorm_min_samples = gnorm_min_samples
        self.loss_window = deque(maxlen=gnorm_window)
        self.snapshot_every = snapshot_every      # accepted steps between last-good snapshots
        self.restore_after = restore_after        # consecutive skips before rolling back
        self.consecutive_skips = 0
        self.n_skipped = 0                        # reported at the end of training
        self.n_restored = 0
        self._last_good = None
        self._accepted = 0
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
            "out_weights_magnitude": (self.Penalties.out_weights_magnitude_penalty, lambda_owm, owm_args),
            "rec_weights_sparsity": (self.Penalties.rec_weights_sparsity_penalty, lambda_rws, rws_args),
            "output_var": (self.Penalties.trial_output_var_penalty, lambda_tv, tv_args),
            "channel_overlap": (self.Penalties.channel_overlap_penalty, lambda_orth, orth_args),
            "fr_magnitude": (self.Penalties.fr_magnitude_penalty, lambda_frm, frm_args),
            "h_magnitude": (self.Penalties.h_magnitude_penalty, lambda_hm, hm_args),
            "metabolic": (self.Penalties.metabolic_penalty, lambda_met, met_args),
            "fr_inequality": (self.Penalties.fr_inequality_penalty, lambda_fri, fri_args),
            "h_inequality": (self.Penalties.h_inequality_penalty, lambda_hi, hi_args),
            "h_time_variance": (self.Penalties.h_time_variance_penalty, lambda_htvar, htvar_args),
            "clustering": (self.Penalties.clustering_penalty, lambda_cl, cl_args),
            "eff_dim_tail_energy": (self.Penalties.eff_dim_tail_energy_penalty, lambda_effdim, effdim_args),
        }
        if monitor:
            self.loss_monitor = {**{k: [] for k in self.penalty_map}}
            self.gradients_monitor = {**{f"g_{k}": [] for k in self.penalty_map}}
            self.scaled_gradients_monitor = {**{f"sg_{k}": [] for k in self.penalty_map}}
        
        # True (default, legacy): penalty-gradient components opposing the task gradient are projected
        # out, so a penalty can never hurt task performance. False: plain multi-objective descent on
        # task + sum(lambda_k * penalty_k), i.e. the standard way of combining losses.
        self.task_safe_gradients = task_safe_gradients

        self.dropout = dropout
        self.dropout_args = dropout_args if dropout_args is not None else {"dropout_kind": None, "sampling_method": None, "drop_rate": 0.0, "dropout_beta": 1.0}
        self.participation = (1e-6 * torch.ones(self.RNN.N, device=self.RNN.device)) if self.dropout else None

        # --- prune-and-reinitialise ("recycle a dead unit") ---
        self.prune_reinit = prune_reinit
        self.prune_args = prune_args if prune_args is not None else {
            "check_every": 100, "patience": 5, "active_rel": 0.05}
        # strikes[i] = consecutive checks unit i has been silent for. Reset on any active check.
        self._reinit_strikes = torch.zeros(self.RNN.N, device=self.RNN.device)
        self._n_reinit_events = 0                                    # total redraws, counts repeats
        self._reinit_ever = torch.zeros(self.RNN.N, dtype=torch.bool, device=self.RNN.device)
        # running contribution utility, and the iteration each unit was last replaced (for maturity)
        self._unit_utility = torch.zeros(self.RNN.N, device=self.RNN.device)
        self._last_replaced = torch.full((self.RNN.N,), -1e9, device=self.RNN.device)
        # cumulative multiplicative boost applied by reinit_mode="rescale", one entry per unit.
        # Capped, so a unit the rule can never revive cannot be inflated without bound -- the
        # failure mode that killed synaptic_scaling_.
        self._rescale_cum = torch.ones(self.RNN.N, device=self.RNN.device)

        # --- homeostatic multiplicative scaling (excitation up / inhibition down) ---
        self.synaptic_scaling = synaptic_scaling
        self.scaling_args = scaling_args if scaling_args is not None else {
            "every": 100, "eta": 0.05, "scale_q": 0.5}
        self.iter_n = 0

        # per-unit participation logged every `track_every` iterations during training
        self.track_participation = track_participation
        self.track_every = int(track_every)
        # "metrics": scalar series aligned to "iters" (NaN where a lag was not due this probe).
        # "participation": the per-unit matrix, on its own coarser cadence. Nothing about the
        # weights is ever written to disk — everything is reduced to scalars during training.
        self.participation_monitor = ({"iters": [], "participation": [], "temporal_pr": [],
                                       "participation_iters": [],
                                       "metrics": defaultdict(list)}
                                      if track_participation else None)
        # Weight-drift bookkeeping. Reference snapshots at several lags, because a single short lag
        # cannot tell a settled-but-jittering network from one that is still systematically drifting:
        # with noise injected every step the weights random-walk forever, so the distance plateaus at
        # a noise floor rather than reaching zero. Diffusion grows as sqrt(lag), systematic drift as
        # lag, so comparing lags separates them; the cosine between consecutive displacements is the
        # direct test (≈0 = random walk, >0 = still marching one way).
        self.track_drift = bool(track_drift)
        self.drift_lags = tuple(int(l) for l in drift_lags) if drift_lags else ()
        self.DRIFT_MATS = ("W_inp", "W_rec", "W_out")   # bias excluded: normally not trained here
        self.store_participation_every = int(store_participation_every or track_every)
        # Print the silent count to stdout on this cadence (None = never). The trace above is only
        # written to disk on completion, so this is the one way to watch silencing DURING a run.
        self.log_silent_every = int(log_silent_every) if log_silent_every else None
        self._drift_refs = {}     # lag -> (iteration, {name: cpu weight copy})
        self._part_refs = {}      # lag -> (iteration, participation vector)
        self._prev_w = {}         # weights at the previous probe
        self._prev_disp = {}      # displacement at the previous probe


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
        eps = float(getattr(self.RNN, "weight_boundary_eps", 1e-12))
        with torch.no_grad():
            self.RNN.W_inp.clamp_min_(eps)
            self.RNN.W_out.clamp_min_(eps)
        return None

    def enforce_dale_(self, eps=1e-12):
        eps = float(getattr(self.RNN, "weight_boundary_eps", eps))
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


    def prune_and_reinit_(self, states):
        """Reinitialise units that have been silent for `patience` consecutive checks.

        WHY THIS AND NOT A PENALTY. A silent ReLU unit is not merely quiet, it is FROZEN: with
        r_i = 0 at every timestep, dL/dW_rec[i,j] ~ relu'(h_i) * r_j = 0 and dL/dW_rec[j,i] ~ r_i = 0,
        so every weight into and out of the unit has exactly zero gradient. No penalty on the loss
        can revive it, because a penalty acts through the same vanished gradient. The only escape
        is to overwrite the weights directly. This is the "recycle a useless neuron" mechanism:
        the unit is not punished, it is replaced.

        WHAT IS REDRAWN. Incoming weights only - W_rec[i, :] and W_inp[i, :] - from the same
        N(0, 1/sqrt(N)) the network was initialised from. Incoming weights are what decide whether
        the unit fires; a zero-mean draw gives positive drive on roughly half of timesteps, which is
        enough to unfreeze the gradient. Outgoing weights are left alone: they received no gradient
        while the unit was dead, so they still hold whatever training last left there, and redrawing
        them would discard learned structure for units that died late.
        # ponytail: incoming-only. Add outgoing redraw if revived units are found to shock the
        # readout - the co-primary r2 read-out is what would show that.

        The draw happens BEFORE the Dale / mask / non-negativity projections in train_step, so the
        fresh row is cleaned up by the existing machinery in the same step rather than needing its
        own sign handling.

        Adam's moments for the redrawn entries are zeroed. Without that, stale momentum from before
        the unit died immediately pushes the new weights back toward the dead configuration.

        Args:
            states: (N, T, B) tensor from the training forward pass. Noisy (w_noise=True) - this
                    reuses the pass that already happened rather than paying for a clean probe;
                    the silence threshold is far too coarse for the noise to matter.

        Returns:
            None; mutates self.RNN.W_rec / W_inp and self._reinit_strikes in place.
        """
        args = self.prune_args
        if self.iter_n % int(args["check_every"]) != 0:
            return None

        p = self.participation_from_states_(states).detach()
        silent = p < float(args["active_rel"]) * torch.quantile(p, 0.95)

        # CONTRIBUTION UTILITY, from Dohare et al., Nature 632:768-774 (2024). Participation alone
        # scores how loudly a unit fires; it says nothing about whether that firing reaches the
        # rest of the network. A unit driving weights that go nowhere is healthy by participation
        # and useless in fact. Multiplying the rate by the total outgoing weight fixes that, and
        # the running average over checks stops a single quiet batch from condemning a unit.
        #   u <- decay * u + (1 - decay) * |r| * sum|W_out|
        with torch.no_grad():
            out_w = self.RNN.W_rec.abs().sum(dim=0) + self.RNN.W_out.abs().sum(dim=0)
            d = float(args.get("utility_decay", 0.99))
            self._unit_utility = d * self._unit_utility + (1.0 - d) * p * out_w
        score = (self._unit_utility if args.get("utility", "participation") == "contribution"
                 else p)

        self._reinit_strikes = torch.where(silent, self._reinit_strikes + 1,
                                           torch.zeros_like(self._reinit_strikes))

        # MATURITY, also from Dohare et al. A unit that was just replaced is protected for
        # `maturity` iterations NO MATTER WHAT IT DOES. Their reason is mechanical: they zero a new
        # unit's outgoing weights, so its utility starts at zero and it would be the very next unit
        # chosen. Ours is measured: the strike counter only protects a unit that starts firing, so
        # a replacement that stays quiet is replaced again, and at N=500 that produced 54.6
        # replacements per unit with the active count unmoved.
        mature = (self.iter_n - self._last_replaced) >= int(args.get("maturity", 0))
        doomed = (self._reinit_strikes >= int(args["patience"])) & mature
        n = int(doomed.sum())
        if n == 0:
            return None

        # REPLACEMENT RATE CAP. Dohare et al. replace on the order of 1e-5 of units per step; our
        # unbounded rule ran at 4.7e-4 on CDDM at N=1000 (stable) and 1.4e-3 on the flip-flop at
        # N=500 (29-44% of gradient updates discarded). When more units qualify than the cap
        # allows, the LOWEST-utility ones go first, which is their selection rule.
        cap_frac = float(args.get("max_replace_frac", 1.0))
        cap = max(1, int(round(cap_frac * self.RNN.N)))
        if n > cap:
            cand = torch.nonzero(doomed, as_tuple=True)[0]
            keep = cand[torch.argsort(score[cand])[:cap]]
            doomed = torch.zeros_like(doomed)
            doomed[keep] = True
            n = cap

        std = 1.0 / np.sqrt(self.RNN.N)
        mode = args.get("reinit_mode", "random")
        with torch.no_grad():
            idx = torch.nonzero(doomed, as_tuple=True)[0]

            if mode == "copy":
                # FUNCTION-PRESERVING DUPLICATION (the Net2Net construction). A randomly redrawn
                # unit has no function, so the task gradient has no reason to keep it, and the
                # screen of 2026-09-22 measured exactly that: 26,012 redraws over 591 units, 44
                # deaths each, with the active count unmoved. A copy of a working unit has a
                # function by construction, which removes "the new unit was useless" as the
                # explanation for its death.
                #
                # Copying the incoming weights alone would DOUBLE the donor's contribution to every
                # downstream unit and jolt the loss. So the donor's OUTGOING weights are split in
                # half between donor and copy: for any downstream unit k the pair then contributes
                # (W[k,donor]/2)*r + (W[k,copy]/2)*r = W[k,donor]_old * r, unchanged.
                #
                # THE DIAGONAL NEEDS EXPLICIT HANDLING and its absence is not a detail. With
                # self_connections=False the W_rec diagonal is masked out of the forward pass, so a
                # unit's self-weight contributes nothing. Duplicating the donor's row would move
                # that self-weight into the off-diagonal entry (copy, donor), where it is NOT
                # masked, inventing a self-excitation loop the donor never had. Both cross terms
                # are therefore zeroed, which is also what exact preservation requires: the donor
                # received nothing from itself, so the copy must receive nothing from the donor.
                #
                # Noise on the copy's incoming weights breaks the symmetry. Without it the two
                # units take identical drive, emit identical output and receive identical
                # gradients, so they stay identical forever and the pair does exactly what the
                # donor alone did.
                #
                # Done one unit at a time. n is a handful per event, and a vectorised version has
                # to reason about two copies drawn from the same donor and about copies writing
                # into each other's rows. The loop is obviously correct; the speed is irrelevant.
                live_idx = torch.nonzero(~silent, as_tuple=True)[0]
                if live_idx.numel() == 0:
                    return None
                # Donors are drawn in proportion to participation: the busiest units are the ones
                # most clearly carrying a function worth copying.
                weights = p[live_idx].clamp_min(1e-12)
                donors = live_idx[torch.multinomial(weights, n, replacement=True,
                                                    generator=self.RNN.random_generator)]
                jitter = float(args.get("copy_noise", 0.05))
                # RESCALE THE JITTERED ROW BACK TO THE DONOR ROW'S LENGTH. Multiplicative jitter
                # grows the row norm by sqrt(1 + jitter^2) -- 3.2x at jitter = 3 -- so a sweep over
                # the noise level would vary the copy's weight MAGNITUDE and its DIRECTION at once,
                # and magnitude is the axis that has already destabilised this network (the
                # uncapped duplication runs, and synaptic scaling in every configuration). With the
                # length pinned, the noise level sets one thing: how far the copy's weight vector is
                # rotated off the donor's, at expected cosine 1 / sqrt(1 + jitter^2).
                norm_preserve = bool(args.get("copy_norm_preserve", False))

                for copy_i, donor_j in zip(idx.tolist(), donors.tolist()):
                    row_rec = self.RNN.W_rec[donor_j, :].clone()
                    row_inp = self.RNN.W_inp[donor_j, :].clone()
                    # The donor's SELF-weight, kept aside because the 2x2 block spanning the pair
                    # is set explicitly below; it has to be read before the outgoing column is
                    # halved, which halves W[j,j] along with the rest of that column.
                    self_w = self.RNN.W_rec[donor_j, donor_j].clone()
                    if bool(args.get("copy_iid", False)):
                        # THE DONOR'S OUTGOING COLUMN, NOBODY'S INCOMING WEIGHTS. The jitter sweep
                        # destroyed the donor's weight VALUES and recruitment held (752 units at
                        # jitter 3.0 against 685 at jitter 0); the permutation destroyed their
                        # PLACEMENT and 81% of the effect held (606). What every recruiting cell
                        # still shares, and a plain random redraw does not, is the halved donor
                        # outgoing column. This cell keeps that and throws the incoming row away
                        # entirely, so it asks whether the outgoing projection is the whole story.
                        row_rec = torch.randn(self.RNN.N, device=self.RNN.device,
                                              generator=self.RNN.random_generator) * std
                        row_inp = torch.randn(self.RNN.W_inp.shape[1], device=self.RNN.device,
                                              generator=self.RNN.random_generator) * std
                    if bool(args.get("copy_permute", False)):
                        # THE SAME WEIGHTS, IN THE WRONG PLACES. A row of W_rec is indexed by
                        # source unit, so permuting it keeps the donor's exact multiset of weights
                        # -- every magnitude, every sign, the same sparsity -- while destroying
                        # which units the new unit listens to. Set against the jitter sweep it
                        # separates the two things a copy carries: jitter keeps the donor's wiring
                        # POSITIONS and scrambles the VALUES (at 3.0 it flips 37% of signs and
                        # still recruits 752 units), a permutation keeps the values and scrambles
                        # the positions. The two positions spanning the pair are left out, because
                        # the 2x2 self-weight block below owns them.
                        keep = torch.ones(self.RNN.N, dtype=torch.bool, device=self.RNN.device)
                        keep[copy_i] = False
                        keep[donor_j] = False
                        slots = torch.nonzero(keep, as_tuple=True)[0]
                        shuffled = slots[torch.randperm(slots.numel(), device=self.RNN.device,
                                                        generator=self.RNN.random_generator)]
                        row_rec[slots] = row_rec[shuffled].clone()
                        row_inp = row_inp[torch.randperm(
                            row_inp.numel(), device=self.RNN.device,
                            generator=self.RNN.random_generator)].clone()
                    len_rec, len_inp = row_rec.norm(), row_inp.norm()
                    # split the donor's outgoing weights between donor and copy
                    self.RNN.W_rec[:, donor_j] *= 0.5
                    self.RNN.W_out[:, donor_j] *= 0.5
                    self.RNN.W_rec[:, copy_i] = self.RNN.W_rec[:, donor_j]
                    self.RNN.W_out[:, copy_i] = self.RNN.W_out[:, donor_j]
                    # give the copy the donor's inputs
                    if jitter > 0:
                        row_rec = row_rec * (1.0 + jitter * torch.randn(
                            row_rec.shape, device=row_rec.device,
                            generator=self.RNN.random_generator))
                        row_inp = row_inp * (1.0 + jitter * torch.randn(
                            row_inp.shape, device=row_inp.device,
                            generator=self.RNN.random_generator))
                        if norm_preserve:
                            row_rec = row_rec * (len_rec / row_rec.norm().clamp_min(1e-12))
                            row_inp = row_inp * (len_inp / row_inp.norm().clamp_min(1e-12))
                    self.RNN.W_rec[copy_i, :] = row_rec
                    self.RNN.W_inp[copy_i, :] = row_inp
                    # THE 2x2 BLOCK SPANNING THE PAIR, which the literal row copy gets wrong.
                    #
                    # A row of W_rec is indexed by SOURCE unit, so position k means "from unit k"
                    # in anybody's row and the transplant needs no shifting -- except at the two
                    # positions where the copy's identity differs from the donor's. The donor's
                    # self-weight is the awkward one, because it sits in the donor's row AND in the
                    # donor's outgoing column, so halving that column halves the self-weight too.
                    #
                    # With self_connections=true, which every experiment here uses, the donor drew
                    # self_w * r of its own drive from itself, and the pair has to keep drawing
                    # exactly that once both twins fire alike. Splitting self_w in half across all
                    # four entries does it: each twin then receives (self_w/2)*r from itself and
                    # (self_w/2)*r from the other, summing to self_w * r. The block is also neutral
                    # on the DIFFERENCE between the twins -- it adds the same amount to both -- so
                    # it preserves the function without gluing the pair together.
                    #
                    # Until 2026-09-24 all three of these entries were zeroed instead, which left
                    # the copy with no self-loop at all and the donor with half of its own (the
                    # half the column halving took). Both twins were detuned, not just the copy.
                    #
                    # With self_connections=false the donor genuinely had no self-loop -- the
                    # diagonal is not a free weight -- so the whole block must be zero, and
                    # zeroing it is also what stops the donor's masked self-weight from reappearing
                    # as a live connection at the unmasked entry (copy, donor).
                    if self.RNN.self_connections:
                        half = 0.5 * self_w
                        self.RNN.W_rec[copy_i, copy_i] = half
                        self.RNN.W_rec[copy_i, donor_j] = half
                        self.RNN.W_rec[donor_j, copy_i] = half
                        # W[donor_j, donor_j] is already self_w/2, from the column halving
                    else:
                        self.RNN.W_rec[copy_i, donor_j] = 0.0
                        self.RNN.W_rec[donor_j, copy_i] = 0.0
                        self.RNN.W_rec[copy_i, copy_i] = 0.0
            elif mode == "orth":
                # A DIRECTION THE POPULATION IS NOT ALREADY USING. Measured on 2026-09-24,
                # duplication raises the active count 2.5x at N=1000 (277 -> 705) while activity
                # dimensionality rises only 1.4x (6.3 -> 9.1), so dimensions per active unit FALL
                # from 0.020 to 0.013: the copies largely ride directions the network already had.
                # `dead` dropout does the opposite, 32.2 dimensions on 913 units, 0.034 each.
                # The bar is therefore adding DIMENSIONS, not units, and a copy cannot add one by
                # construction. This draws a random incoming row and removes everything the live
                # units' rows already span, so the new unit reads a combination of inputs no
                # existing unit reads. The projection is solved ONCE per event for all n dead units
                # together; per-unit least squares would be ~10,000 solves over a run.
                live_idx = torch.nonzero(~silent, as_tuple=True)[0]
                if live_idx.numel() == 0:
                    return None
                # ORDER MATTERS AND THE COMPLEMENT MUST EXIST.
                # Zeroing the revived units' outgoing columns AFTER orthogonalising perturbs the
                # new row at exactly those entries (it left cosine 0.155 against a live row), so
                # the columns are zeroed FIRST and the projection then runs over the updated rows.
                # Restricting the projection to the surviving columns instead does not work: with
                # n_live live units and only n_keep surviving columns, the live rows span the whole
                # restricted space whenever n_live >= n_keep, the residual is numerically zero, and
                # normalising it turns rounding error into a full-size weight row. Measured at
                # N=40: 23 live rows, 23 surviving columns, cosine 0.244.
                # A complement exists only while the live population does not already span R^N.
                if int(live_idx.numel()) >= self.RNN.N - 1:
                    self.RNN.W_rec[idx, :] = torch.randn(
                        n, self.RNN.N, device=self.RNN.device,
                        generator=self.RNN.random_generator) * std
                else:
                    self.RNN.W_rec[:, idx] = 0.0                    # outgoing first
                    self.RNN.W_out[:, idx] = 0.0
                    L = self.RNN.W_rec[live_idx, :]                 # (n_live, N), already updated
                    V = torch.randn(self.RNN.N, n, device=self.RNN.device,
                                    generator=self.RNN.random_generator)
                    # QR, not lstsq: lstsq defaults to a driver assuming full column rank, and a
                    # live population's rows are routinely rank-deficient.
                    Q, _ = torch.linalg.qr(L.T, mode="reduced")
                    V = V - Q @ (Q.T @ V)
                    V = V / V.norm(dim=0, keepdim=True).clamp_min(1e-8) * (
                        std * float(np.sqrt(self.RNN.N)))
                    self.RNN.W_rec[idx, :] = V.T
                self.RNN.W_inp[idx, :] = torch.randn(
                    n, self.RNN.W_inp.shape[1], device=self.RNN.device,
                    generator=self.RNN.random_generator) * std

            elif mode == "mix":
                # A BLEND OF SEVERAL WORKING UNITS, which is not a copy of any of them. Keeps the
                # property that made duplication work -- the new unit sits where the network is
                # already active, so it fires and the task gradient has something to hold on to --
                # while removing the exact twin that makes a copy redundant by construction.
                live_idx = torch.nonzero(~silent, as_tuple=True)[0]
                if live_idx.numel() < 2:
                    return None
                k = min(int(args.get("mix_k", 4)), int(live_idx.numel()))
                w = p[live_idx].clamp_min(1e-12)
                for row, copy_i in enumerate(idx.tolist()):
                    pick = live_idx[torch.multinomial(w, k, replacement=False,
                                                      generator=self.RNN.random_generator)]
                    # Dirichlet(1,...,1) over the k donors: a uniformly random blend, so no single
                    # donor dominates and the result is a twin of nobody.
                    a = -torch.log(torch.rand(k, device=self.RNN.device,
                                              generator=self.RNN.random_generator).clamp_min(1e-12))
                    a = a / a.sum()
                    self.RNN.W_rec[copy_i, :] = (a.unsqueeze(1) * self.RNN.W_rec[pick, :]).sum(0)
                    self.RNN.W_inp[copy_i, :] = (a.unsqueeze(1) * self.RNN.W_inp[pick, :]).sum(0)
                    self.RNN.W_rec[copy_i, copy_i] = 0.0
                self.RNN.W_rec[:, idx] = 0.0
                self.RNN.W_out[:, idx] = 0.0

            elif mode == "bias_kick":
                # NO DONOR AT ALL. A dead ReLU unit is frozen because relu'(h) = 0, so every weight
                # into and out of it has exactly zero gradient. Writing a bias offset directly --
                # not through the optimiser, which cannot reach it either -- lifts h above zero on
                # some timesteps and unfreezes the unit, leaving its weights untouched so it must
                # find its own function from where it already sits.
                # The offset is measured, not guessed: -median(h_i) makes the unit cross threshold
                # on about half of its timesteps.
                # THE PREDICTION, recorded before the run: Pavel expects this to fail, because the
                # recurrent input that silenced the unit is still there and still training, so the
                # network should simply re-suppress it. If it treadmills where duplication does not,
                # the donor's FUNCTION is what makes duplication stick, not merely escaping the
                # frozen state.
                h = states.detach()[idx]                            # (n, T, B) pre-activations
                offs = -torch.quantile(h.reshape(n, -1), 0.5, dim=1)
                b = self.RNN.bias
                if isinstance(b, torch.nn.Parameter):
                    b.data[idx] = offs
                else:
                    b[idx] = offs

            elif mode == "rescale":
                # GRADUAL UNSUPPRESSION, no donor and no redraw of what the unit knows. The unit's
                # OWN trained incoming weights are kept and nudged: excitation up by alpha,
                # inhibition down by alpha, a little every step, until it fires. That is the
                # property the measured arms say matters -- every rule that threw a unit's trained
                # incoming weights away landed between 262 and 333 active units at N=1000, while
                # the two that kept trained weights reached 373 (bias_kick) and 702 (copy).
                #
                # WHY THIS IS NOT THE OLD synaptic_scaling_, WHICH BLEW UP. That rule scaled every
                # unit toward a population set-point at every event and never stopped, so a unit it
                # could not rescue stayed below the set-point and was scaled at all 400 events. The
                # inflation it produced sits on exactly those units: in the one configuration that
                # survived, the rows of units it FAILED to revive ended at twice the control's while
                # the rows of units it rescued were only 15% above. Three things differ here:
                #   - it stops the moment the unit passes the activity criterion;
                #   - it is one-sided, so an active unit is never scaled down;
                #   - the cumulative boost per unit is capped, so "until it revives" is bounded in
                #     the case where it never does.
                #
                # WHY ALPHA IS THIS SMALL. Applied every step, the rule pushes the same direction
                # every step, unlike gradient steps which partly cancel, so matching a gradient
                # step's size (about 0.8% relative) would boost a unit 148-fold inside 1000 steps.
                # The step is set from the timescale instead: measured on the controls, the median
                # silent unit needs alpha ~ 1.25 in total and the 90th percentile ~ 3-5, so at
                # 1.0005 per step the median revives in ~450 steps and the tail in ~3200.
                #
                # THE OUTGOING WEIGHTS ARE REDRAWN, NOT ZEROED. Zeroing makes a firing unit
                # invisible to the loss, so its incoming weights get no gradient until the outgoing
                # ones have grown back -- which is the likeliest reason `zero_out` did nothing here.
                # Redrawing costs nothing while the unit is silent (it emits ~0, so its outgoing
                # column carries ~0 whatever its weights are) and leaves it with a working,
                # nonzero column the moment it revives. The redraw also overwrites the diagonal,
                # which is how the self-connection stays out of the multiplicative boost -- boosting
                # a self-weight is direct positive feedback, and trained self-weights here run about
                # 33x the median off-diagonal weight.
                a = float(args.get("rescale_alpha", 1.0005))
                cum_cap = float(args.get("rescale_cap", 8.0))
                act = idx[self._rescale_cum[idx] < cum_cap]
                if act.numel() > 0:
                    normalize = bool(args.get("rescale_normalize", True))
                    for W in (self.RNN.W_rec, self.RNN.W_inp):
                        blk = W[act, :]
                        before = blk.norm(dim=1, keepdim=True)
                        blk = torch.where(blk > 0, blk * a, blk / a)
                        if normalize:
                            # Holds the row's total synaptic weight fixed, making the event a pure
                            # redistribution from inhibition to excitation. Without it the row
                            # changes by exactly sqrt((alpha^2 E + I/alpha^2)/(E + I)) for summed
                            # squares E of its excitatory and I of its inhibitory weights. That
                            # exceeds 1 only where the two are COMPARABLE, which is the pump that
                            # killed the first two attempts; an inhibition-dominated row, which is
                            # what a silenced unit has, shrinks under the same rule (measured:
                            # 0.667x at alpha 1.5 for I/E = 8e4, against 1.158x at I/E = 1).
                            blk = blk * (before / blk.norm(dim=1, keepdim=True).clamp_min(1e-12))
                        W[act, :] = blk
                    self.RNN.W_rec[:, act] = torch.randn(
                        self.RNN.N, act.numel(), device=self.RNN.device,
                        generator=self.RNN.random_generator) * std
                    self.RNN.W_out[:, act] = torch.randn(
                        self.RNN.W_out.shape[0], act.numel(), device=self.RNN.device,
                        generator=self.RNN.random_generator) * std
                    self._rescale_cum[act] *= a

            else:
                self.RNN.W_rec[idx, :] = torch.randn(n, self.RNN.N, device=self.RNN.device,
                                                     generator=self.RNN.random_generator) * std
                self.RNN.W_inp[idx, :] = torch.randn(n, self.RNN.W_inp.shape[1],
                                                     device=self.RNN.device,
                                                     generator=self.RNN.random_generator) * std
                if mode == "zero_out":
                    # Dohare et al.'s rule: resample the incoming weights, ZERO the outgoing ones.
                    # The new unit then cannot disturb anything the network has already learned,
                    # which is function preservation in its simplest form - no donor, no bookkeeping
                    # and no copy-cascade. It has to grow its output back through the gradient,
                    # which is why their maturity threshold exists.
                    # NOTE our own "random" mode is NOT this: it leaves the outgoing weights at
                    # whatever training last put there. The measured failure of `random` (44 deaths
                    # per unit, ~1% survival) therefore says nothing about their published method.
                    self.RNN.W_rec[:, idx] = 0.0
                    self.RNN.W_out[:, idx] = 0.0

            # Adam carries per-entry moments; stale ones would undo the redraw within a few steps.
            # NOT for `rescale`, whose incoming rows are nudged by 0.05% rather than redrawn: their
            # gradient statistics are still valid, and zeroing them every step would reset the
            # optimiser for every silent unit on every step.
            if mode != "rescale":
                for prm in (self.RNN.W_rec, self.RNN.W_inp):
                    st = self.optimizer.state.get(prm, None)
                    if st:
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in st:
                                st[key][idx, :] = 0.0
            if mode in ("copy", "rescale"):
                for prm, cols in ((self.RNN.W_rec, idx), (self.RNN.W_out, idx)):
                    st = self.optimizer.state.get(prm, None)
                    if st:
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in st:
                                st[key][:, cols] = 0.0

        self._reinit_strikes[doomed] = 0
        self._last_replaced[doomed] = float(self.iter_n)
        self._n_reinit_events += n
        self._reinit_ever[doomed] = True
        return None

    def synaptic_scaling_(self, states):
        """Homeostatic multiplicative scaling of INCOMING weights, excitation up / inhibition down.

        WHY THE SIGN SPLIT. Naive multiplicative scaling has the wrong sign of effect on the case
        that matters. If unit i is silent because its net drive is NEGATIVE -- other units inhibit
        it -- then multiplying its whole incoming row by alpha > 1 makes the drive MORE negative and
        buries the unit deeper. Biology does not do this: Turrigiano-style synaptic scaling acts on
        EXCITATORY synapses, while inhibitory synapses scale the opposite way under the same
        activity deprivation. So the rule here is

            W_ij <- alpha_i * W_ij   where W_ij > 0
            W_ij <- W_ij / alpha_i   where W_ij < 0

        which raises h_i from both directions when alpha_i > 1. Relative weights are preserved
        WITHIN the excitatory set and within the inhibitory set separately, which is the property
        that keeps scaling from destroying learned selectivity -- and the property a magnitude
        penalty does not have. Signs are preserved, so this is Dale-safe and mask-safe (zeros stay
        zero) without any special handling.

        WHAT IT CANNOT DO. A unit with no excitatory input left cannot be rescued: shrinking its
        inhibition drives h_i towards 0 from below but never across. Scaling rescues the weakly
        driven; `prune_and_reinit_` rescues the structurally disconnected. They are complementary,
        which is a testable prediction -- the pair should beat either alone.

        The set-point is the `scale_q` quantile of participation over the LIVE pool, so it is
        scale-free and self-calibrating rather than another absolute constant to guess. Scaling is
        two-sided: over-active units are scaled DOWN, which is what drives the population towards
        homogeneity instead of merely lifting a floor.

        Args:
            states: (N, T, B) tensor from the training forward pass, used to score participation.

        Returns:
            None; mutates self.RNN.W_rec and self.RNN.W_inp in place.
        """
        args = self.scaling_args
        if self.iter_n % int(args["every"]) != 0:
            return None

        eta = float(args["eta"])
        p = self.participation_from_states_(states).detach()
        live = p >= 0.05 * torch.quantile(p, 0.95)
        if live.sum() < 2:
            return None

        if args.get("scale_target", "median") == "lognormal":
            # RANK-MATCHED TARGET. Every unit is pulled toward the place a LOGNORMAL population of
            # this size would put a unit of its rank, not toward a single set-point. Two properties
            # follow, and both matter:
            #   - it cannot homogenise. A busy unit's target sits high in the lognormal, so it is
            #     not dragged to the middle; a single set-point drags everything to the middle,
            #     which is how frm ends up with only 0.8 decades of spread where cortex has ~2.
            #   - the bottom-ranked units get a target that is SMALL BUT NONZERO (for N=1000 and
            #     sigma_log=1.2 the lowest rank asks for ~2% of the median), so a silent unit is
            #     asked to rejoin the tail, not to become average.
            # A lognormal has support on (0, inf), so "match a lognormal" already entails "no unit
            # at exactly zero" -- the two goals are the same requirement, not competing ones.
            sigma = float(args.get("sigma_log", 1.2))
            N = p.numel()
            # median of the live pool sets the scale, so the target tracks the network's own
            # operating point instead of imposing an absolute rate.
            mu = torch.log(torch.quantile(p[live], 0.5).clamp_min(1e-8))
            ranks = torch.argsort(torch.argsort(p)).to(p.dtype)      # 0 = quietest
            u = (ranks + 0.5) / N
            z = torch.erfinv(2.0 * u - 1.0) * float(np.sqrt(2.0))    # standard normal quantiles
            target = torch.exp(mu + sigma * z).clamp_min(1e-8)
        else:
            target = torch.quantile(p[live], float(args["scale_q"])).clamp_min(1e-8)

        # alpha_i = 1 + eta * (target - p_i)/target, clipped to [1/(1+eta), 1+eta] so the step size
        # is bounded by the single parameter eta rather than needing its own ceiling.
        alpha = 1.0 + eta * (target - p) / target
        alpha = alpha.clamp(1.0 / (1.0 + eta), 1.0 + eta)

        alpha = alpha.unsqueeze(1)

        with torch.no_grad():
            for W in (self.RNN.W_rec, self.RNN.W_inp):
                before = W.norm(dim=1, keepdim=True)
                pos = W > 0
                neg = W < 0
                W[pos] = (W * alpha)[pos]
                W[neg] = (W / alpha)[neg]
                if bool(args.get("preserve_row_norm", True)):
                    # WHY THIS IS REQUIRED, not optional. The sign split multiplies a unit's
                    # excitatory weights by alpha and divides its inhibitory ones by alpha, so the
                    # row's magnitude changes by exactly sqrt((alpha^2 E + I/alpha^2)/(E + I)) for
                    # summed squares E of the excitatory and I of the inhibitory weights. That
                    # factor exceeds 1 wherever the two are COMPARABLE, which is where this rule
                    # keeps every unit, so each event pumps magnitude in whichever direction it
                    # scales and 300 events compound it. (An earlier version of this comment said
                    # the factor is alpha + 1/alpha > 2 for every alpha, which is too strong: a
                    # strongly inhibition-dominated row shrinks instead.) That is what destroyed the first two attempts: all three seeds of the
                    # median-set-point arm discarded 19-27% of their gradient updates and the
                    # rank-matched arm 31-55%, every one ending at r2 of NaN or -inf, with the
                    # median activity of one run climbing 0.087 -> 0.207 -> 0.990 -> 1.7e10 while
                    # a control held 0.084 -> 0.099 throughout.
                    #
                    # Rescaling each row back to the norm it had makes the event a pure
                    # REDISTRIBUTION between a unit's excitatory and inhibitory input at fixed
                    # total synaptic weight. The drive on a silenced unit still rises, because
                    # moving weight from inhibition to excitation raises W.r at constant ||W||,
                    # which is the whole mechanism. What can no longer happen is the population
                    # growing without bound.
                    #
                    # Normalising the mean of alpha instead does NOT fix this and was tried first:
                    # the pump is per-row and survives any constraint on alpha's average.
                    after = W.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    W.mul_(before / after)
        return None

    def enforce_inp_cap_(self):
        """Clamp |W_inp| to the model's inp_weight_cap after an optimiser step. No-op if unset.

        WHY A CLAMP AND NOT A PENALTY. `inp_weights_magnitude_penalty` exists and is wired in as
        lambda_iwm, but its gamma=5 hinge is unusable as a soft cap here: at initialisation every
        |W_inp| is ~0.03, well under any sensible cap, so the penalty is EXACTLY zero and exerts no
        gradient at all; on a trained unpenalised network the largest weight is ~41x a cap of 0.36,
        so the term reaches ~1e8. A lambda sized for one end is inert or explosive at the other,
        and this project already carries spike_factor/restore_after machinery because frm misbehaved
        far more mildly than that. A clamp has one parameter with an obvious meaning and cannot
        spike.

        WHAT IT TESTS. The unpenalised network concentrates input drive into ~6% of weights - the
        99th percentile of |W_inp| is 8.57 against a median of 0.002 - and a unit's input row norm
        predicts its participation at r = +0.65. Under frm+rws that correlation inverts to -0.61 and
        the row-norm CV falls from 2.28 to 0.74. The hypothesis this clamp tests is the causal
        direction: does forcing input drive to spread RECRUIT units, or does recruiting units merely
        happen to spread the drive?

        Returns:
            None; clamps self.RNN.W_inp in place.
        """
        cap = getattr(self.RNN, "inp_weight_cap", None)
        if cap is None:
            return None
        with torch.no_grad():
            self.RNN.W_inp.clamp_(min=-cap, max=cap)
        return None

    def enforce_bias_range_(self):
        if not getattr(self.RNN, "bias_trainable", False):
            return None
        br = getattr(self.RNN, "bias_range", None)
        if br is None:
            return None
        b_low, b_high = torch.unbind(torch.as_tensor(br, device=self.RNN.bias.device, dtype=self.RNN.bias.dtype))
        with torch.no_grad():
            self.RNN.bias.clamp_(min=b_low, max=b_high)
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
        y = scored_(output, mask)
        t = scored_(target, mask)
        r2 = 1.0 - (y - t).pow(2).mean() / (t - t.mean()).pow(2).mean().clamp_min(1e-12)
        r2_val = float(r2.item())
        return r2_val
    
    def participation_from_states_(self, states, q=0.9, chunk=512):
        '''
        Per-unit participation of a firing-rate tensor: std(fr) + q-quantile(fr), pooled over
        (time, trials), matching the offline readout PerformanceAnalyzer.plot_participation.

        THE SINGLE DEFINITION OF PARTICIPATION. It feeds both the logged trace and the dropout
        sampler. Until 2026-09-21 the sampler had its OWN scorer, `get_participation_`, which read
        the RAW states - for equation_type "h" the pre-activations - and scored q(|h|) + std(|h|).
        A unit held far below threshold on every trial has a large |h| and so scored highly, so the
        sampler could not distinguish a permanently silent unit from a busy one (Spearman rho = 0.24
        between the two scores; 51 +- 4% of the drop mass landed on units that were already silent
        and dropping them is a no-op). That scorer is deleted and this one is used instead: dropout
        now samples on the firing rate, which is the only quantity that can be dropped. Runs before
        that commit used the old scorer and are pinned by their commit hash.

        Args:
            states: (N, T, B) tensor of network states as returned by RNN_torch.forward.
                    For equation_type "h" these are pre-activations and the activation is applied here;
                    for "s" they are already firing rates.
        Args (cont.):
            q: quantile of the FIRING RATE entering the score. The participation trace logs q=0.9
               (the project-wide definition of participation, unchanged); the dropout sampler passes
               dropout_args["activity_q"].
            chunk: number of UNITS processed at a time. Bounds peak memory independently of N.

        Returns:
            (N,) tensor of participation values, one per unit.

        ⚠️ CHUNKED BECAUSE THE UNCHUNKED FORM RUNS OUT OF GPU MEMORY AT LARGE N. `torch.quantile`
        sorts its input, so it allocates roughly twice the tensor, and the old code additionally
        materialised the full activation `fr` alongside `states`. At N=4000 with T=300 and batch 1024
        that is a 13.7 GiB allocation on top of 37 GiB already resident, and the job dies with
        torch.OutOfMemoryError inside this function. Measured: N=3000 completed, N=4000 and N=5000
        both OOMed on a 44 GiB GPU.

        Chunking over the unit axis is exact in the sense that matters: std and quantile are computed
        per row, so rows are independent. It is NOT bitwise identical in float64 - `std` is a
        reduction whose summation order depends on the block shape, giving last-bit differences of
        4e-16 to 3e-15. In float32, the dtype actually used, the difference measured over 6000 units
        of a realistic bimodal population is exactly 0.0, and ZERO units change silence class under
        the hard (1e-6), task-calibrated absolute (4e-2), or scale-free rule. Applying the activation
        per chunk also avoids ever holding a second full-size copy of the states.
        '''
        N = states.size(0)
        out = torch.empty(N, device=states.device, dtype=states.dtype)
        for i in range(0, N, chunk):
            blk = states[i:i + chunk].reshape(min(chunk, N - i), -1)   # view; (chunk, T*B)
            if self.RNN.equation_type == "h":
                blk = self.RNN.activation(blk)
            out[i:i + chunk] = (blk.std(dim=1, unbiased=False)
                                + torch.quantile(blk.abs(), q, dim=1))
        return out

    def track_participation_(self, input_batch, iter, target_batch=None, mask=None):
        """
        Probe the network and record training diagnostics, reducing everything to scalars on the fly.

        Recorded at every probe (into monitor["metrics"], aligned to monitor["iters"]):
          silent_1em6                  number of units with participation < 1e-6
          temporal_pr                  per-unit (sum r)^2/sum r^2 over the probe's (time, trial)
                                       samples, stored on the store_participation_every cadence
                                       alongside `participation` and sharing its iteration index
          loss_clean_train             masked MSE of the SAME noise-free probe on the training batch
          loss_clean_valid             the same on a held-out batch, if valid_batch was supplied.
                                       Recorded on its own coarser cadence (track_valid_every), so
                                       it is NOT aligned with monitor["iters"] - it has its own
                                       index, spaced track_valid_every apart.
          dp_lag<L>                    ||p(t) - p(t-L)|| / ||p(t)||, the participation vector's
                                       relative change over lag L
          norm_<W>                     ||W(t)||_F, so a rising drift_<W> can be attributed to the
                                       numerator or the denominator
          drift_<W>_lag<L>             ||W(t) - W(t-L)||_F / ||W(t)||_F for W_inp, W_rec, W_out
          cos_<W>                      cosine between consecutive displacements of that matrix
                                       (~0 = jitter, >0 = still drifting systematically)
        Lagged entries are NaN on probes where that lag is not yet due. The full per-unit
        participation vector is stored separately, on the coarser store_participation_every cadence,
        because it is the only bulky item.

        Args:
            input_batch: (n_inputs, T, B) inputs the probe is measured on (the training batch).
            iter: current training iteration.
        Returns:
            None; mutates self.participation_monitor.
        """
        mon = self.participation_monitor
        with torch.no_grad():
            # w_noise=False so the trace is comparable to the offline (noise-free) analysis
            states, out_clean = self.RNN(input_batch, w_noise=False, dropout=False, dropout_args=None)
            p = self.participation_from_states_(states)

        mon["iters"].append(int(iter))
        met = mon["metrics"]
        met["silent_1em6"].append(float((p < 1e-6).sum()))
        # Prune-and-reinit diagnostics. `events` counts every redraw INCLUDING repeats of the same
        # unit, `ever` counts distinct units touched. Both are needed to detect a treadmill: a
        # mechanism that redraws units which promptly re-die shows events >> ever with no gain in
        # the active count, and that is a failure, not a partial success.
        if self.prune_reinit:
            met["reinit_events"].append(float(self._n_reinit_events))
            met["reinit_units_ever"].append(float(self._reinit_ever.sum()))
        if self.log_silent_every and iter % self.log_silent_every == 0:
            q95 = torch.quantile(p, 0.95)
            print(f"[silence] iter {iter}: hard(p<1e-6) {int((p < 1e-6).sum())}/{p.numel()}  "
                  f"scale-free(p<0.05*q95) {int((p < 0.05 * q95).sum())}/{p.numel()}  "
                  f"flipflop(p<4e-2) {int((p < 4e-2).sum())}/{p.numel()}  q95(p)={float(q95):.4f}", flush=True)

        # Deterministic loss, free: this forward pass already happened for the participation probe.
        # The loss recorded every iteration during training is NOISY (train_step uses w_noise=True),
        # so its minimum is a noise lottery and it cannot separate "learned" from "got a good draw".
        # The clean loss can. A held-out batch is evaluated too when one was supplied, because with
        # same_batch=True nothing else in the pipeline distinguishes learning from memorising the
        # 450 fixed trials.
        if target_batch is not None and mask is not None:
            with torch.no_grad():
                met["loss_clean_train"].append(
                    float(((scored_(out_clean, mask) - scored_(target_batch, mask)) ** 2).mean()))
                if self.valid_batch is not None and iter % self.track_valid_every == 0:
                    vi, vt = self.valid_batch
                    _, vout = self.RNN(vi, w_noise=False, dropout=False, dropout_args=None)
                    met["loss_clean_valid"].append(
                        float(((vout[:, mask, :] - vt[:, mask, :]) ** 2).mean()))

        if iter % self.store_participation_every == 0:
            mon["participation"].append(p.cpu().numpy().astype("float32"))
            # Per-unit TEMPORAL participation ratio (sum r)^2 / sum r^2 over the pooled
            # (time, trial) samples of this same noise-free probe: the effective number of samples
            # each unit is active for, i.e. PR along TIME rather than along units. Divided by the
            # sample count it is the Treves-Rolls lifetime sparseness.
            #
            # WHY IT IS TRACKED. frm pins each unit's soft-max (peak-ish) activity and succeeds at
            # it - CV across live units is 0.125 under frm and 0.112 under frm+rws - while temporal
            # PR differs by 2.2x between those conditions. Occupancy is the axis the penalties
            # actually separate on, and it CANNOT be recovered from the stored participation vector
            # (std + q90), which is why it has to be computed here, from the rates, before they are
            # discarded. Costs one reduction over a tensor the probe already materialised.
            with torch.no_grad():
                rr = states if self.RNN.equation_type != "h" else self.RNN.activation(states)
                rr = rr.reshape(rr.size(0), -1).double()
                num = rr.sum(dim=1) ** 2
                den = (rr * rr).sum(dim=1)
                tpr = torch.where(den > 0, num / den.clamp_min(1e-300),
                                  torch.zeros_like(den))       # 0 for an all-silent unit
            mon["temporal_pr"].append(tpr.cpu().numpy().astype("float32"))
            mon["participation_iters"].append(int(iter))

        nan = float("nan")
        if not self.track_drift:
            return None

        with torch.no_grad():
            cur = {n: getattr(self.RNN, n).detach() for n in self.DRIFT_MATS
                   if getattr(self.RNN, n, None) is not None}
            # Raw Frobenius norms. drift_* is normalised by ||W(t)||, so a rising drift is ambiguous
            # between a growing numerator and a shrinking denominator; these disambiguate it.
            for n in cur:
                met[f"norm_{n}"].append(float(cur[n].norm()))
            for lag in self.drift_lags:
                ref = self._drift_refs.get(lag)
                pref = self._part_refs.get(lag)
                # Emit only when the reference is a full `lag` old, then refresh. Refreshing as soon
                # as a reference ages out would make the actual separation oscillate between 0 and
                # lag, and different lags would coincide.
                due = ref is not None and (iter - ref[0]) >= lag
                for n in cur:
                    met[f"drift_{n}_lag{lag}"].append(
                        float((cur[n] - ref[1][n].to(cur[n].device)).norm()
                              / cur[n].norm().clamp_min(1e-12)) if due else nan)
                met[f"dp_lag{lag}"].append(
                    float((p - pref[1].to(p.device)).norm() / p.norm().clamp_min(1e-12))
                    if due and pref is not None else nan)
                if ref is None or due:
                    self._drift_refs[lag] = (int(iter), {n: cur[n].to("cpu", copy=True) for n in cur})
                    self._part_refs[lag] = (int(iter), p.to("cpu", copy=True))

            # directional persistence, measured on consecutive probes
            for n in cur:
                prev = self._prev_w.get(n)
                if prev is None:
                    met[f"cos_{n}"].append(nan)
                else:
                    disp = (cur[n] - prev.to(cur[n].device)).flatten()
                    pd = self._prev_disp.get(n)
                    met[f"cos_{n}"].append(
                        float(torch.dot(disp, pd.to(disp.device))
                              / (disp.norm() * pd.norm()).clamp_min(1e-30)) if pd is not None else nan)
                    self._prev_disp[n] = disp.to("cpu", copy=True)
                self._prev_w[n] = cur[n].to("cpu", copy=True)
        return None

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
            # 0.5 (the value every shipped config sets), NOT 0.0: at eta=0 the EMA would freeze
            # at its 1e-6 initialisation, the softmax would be uniform, and participation dropout
            # would silently degrade to UNIFORM dropout with the config still reading "participation".
            eta = self.dropout_args.get("eta", 0.5)
            _, output_do = self.RNN(input, w_noise=True, dropout=True, dropout_args=self.dropout_args, participation=part)

            if self.dropout_args["sampling_method"] == "participation":
                new_part = self.participation_from_states_(
                    states, q=self.dropout_args["activity_q"]).detach()
                self.participation = (1 - eta) * self.participation + eta * new_part

        # The TASK loss is scored on the dropout pass (output_do), the penalties on the full pass
        # (states / output_full), so a penalty keeps its usual meaning while the task gradient sees
        # the ablated or muted network. Until 2026-09-15 the two were the other way round: the task
        # loss read output_full and output_do reached only the penalties, none of which use the
        # output - so dropout changed NOTHING in the gradient (verified: 'dead' and 'mute' runs with
        # the same seeds were bit-identical, and both differed from dropout=False only through the
        # extra forward pass consuming the noise generator). Every earlier dropout run is void.
        penalty_dict_raw = {
            k: (fn(states, input, (output_do if k == 'task' else output_full), target_output, mask, **kwargs) if L != 0 else None)
            for k, (fn, L, kwargs) in self.penalty_map.items()
        }
        
        # --- 1) gradient norms for monitoring (uses autograd.grad) ---
        if self.monitor:
            grads = {
                f"g_{k}": Trainer.grad_norm_of_(penalty_dict_raw[k], params, self.RNN.device)
                for k in self.penalty_map
            }

        # --- 2) combined gradient: task-safe projection, or plain summed loss ---
        if self.task_safe_gradients:
            g_tot = Trainer.get_task_safe_gradients_(
                params, self.penalty_map, penalty_dict_raw,
                task_key="task", allow_unused=True
            )
        else:
            total_loss = None
            for k, (_, L, _) in self.penalty_map.items():
                if L == 0:
                    continue
                term = L * penalty_dict_raw[k]
                total_loss = term if total_loss is None else total_loss + term
            g_tot = Trainer.flat_grad_(total_loss, params, retain_graph=False, allow_unused=True)

        # --- 3) apply combined gradient ---
        self.optimizer.zero_grad(set_to_none=True)
        for p, g in zip(params, g_tot):
            p.grad = g

        # DROP an anomalous update rather than clip it.
        #
        # Two failure modes, one test. (1) A non-finite gradient: clip_grad_norm_ does NOT guard
        # inf/nan (error_if_nonfinite defaults to False), so total_norm=inf makes clip_coef =
        # max_norm/inf = 0 and the offending grad becomes inf*0 = nan, which the optimizer writes
        # into the weights; the next forward pass spreads it and the run trains on nan to
        # completion without ever exiting non-zero. (2) A finite but enormous gradient: clipping
        # bounds its norm but NOT its effect, because Adam divides by sqrt(v) and is therefore
        # scale-invariant - measured, one spike drags the weights 287 normal-sized steps at
        # max_grad_norm=50 and still 233 at 0.1. Skipping is the only thing that costs zero:
        # Adam's m and v never see the spike at all.
        #
        # clip_grad_norm_ returns the PRE-clip norm, so the test is free.
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=self.max_grad_norm)
        n_now = float(total_norm)
        # Record EVERY finite norm, skipped or not, so the reference can never go stale.
        if np.isfinite(n_now):
            self.gnorm_window.append(n_now)
        ref = (float(np.median(self.gnorm_window))
               if len(self.gnorm_window) >= self.gnorm_min_samples else None)
        spike = ref is not None and ref > 0 and n_now > self.spike_factor * ref
        if (not torch.isfinite(total_norm)) or spike:
            self.optimizer.zero_grad(set_to_none=True)
            self.n_skipped += 1
            self.consecutive_skips += 1
            # Sustained skipping means the weights are parked in a region where every batch
            # overflows. Because a skip freezes the weights, nothing can move them back out on
            # its own - the run would spin to its wall-clock limit doing nothing (18 jobs did).
            # Roll back to the last good snapshot and reset Adam, whose moments are stale.
            if self._last_good is not None and self.consecutive_skips >= self.restore_after:
                with torch.no_grad():
                    for prm, good in zip(params, self._last_good):
                        prm.copy_(good)
                self.optimizer.state.clear()
                self.consecutive_skips = 0
                self.n_restored += 1
        else:
            self.optimizer.step()
            self.consecutive_skips = 0
            self._accepted += 1
            # snapshot the FIRST accepted step too, else a run that diverges before
            # snapshot_every has nothing to roll back to and spins to its wall-clock limit.
            # ⚠️ Only snapshot a state that is actually GOOD. Gating on an accepted-step counter
            # alone lets a snapshot be taken while the network is already broken, and rollback
            # then restores the broken state - it can never climb back out.
            healthy = (len(self.loss_window) < self.gnorm_min_samples
                       or self.loss_window[-1] <= float(np.median(self.loss_window)))
            if self._last_good is None or (self._accepted % self.snapshot_every == 0 and healthy):
                self._last_good = [prm.detach().clone() for prm in params]

        # --- 4) now it's safe to mutate weights in-place ---
        # For weight_boundary="reflective" the constraints are baked into the forward pass
        # (effective weight = |param|*sign*mask), so the post-step projections are skipped;
        # for "sticky" (legacy/default) they are applied as before.
        # Redraw dead units BEFORE the projections below, so Dale / masks / non-negativity clean
        # up the fresh rows in this same step instead of needing their own sign handling.
        if self.prune_reinit:
            self.prune_and_reinit_(states)
        if self.synaptic_scaling:
            self.synaptic_scaling_(states)

        if getattr(self.RNN, "weight_boundary", "sticky") == "sticky":
            self.enforce_masks_()   # structural (zero diagonal / masked entries), independent of Dale
            if getattr(self.RNN, "io_nonnegativity", True):
                self.enforce_io_nonnegativity_()
            if getattr(self.RNN, "dale", True):
                self.enforce_dale_()
        self.enforce_bias_range_()
        self.enforce_inp_cap_()

        # --- 5) compute total loss and r2 for reporting ---
        loss_val = Trainer.zero_(self.RNN.device)
        for k, (_, L, _) in self.penalty_map.items():
            loss_k = penalty_dict_raw[k] if L != 0 else Trainer.zero_(self.RNN.device)
            loss_val += L * loss_k
        loss_val = float(loss_val.detach().cpu().item())
        if np.isfinite(loss_val):
            self.loss_window.append(loss_val)      # feeds the snapshot health gate above
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
        # A task that scores its trials differently (TaskMultiRule) supplies a per-trial (T, B) mask
        # with every batch; otherwise the global time mask applies to every trial.
        per_trial = hasattr(self.Task, "batch_mask")
        mask_of = lambda conds: (torch.from_numpy(self.Task.batch_mask(conds)).to(self.RNN.device)
                                 if per_trial else train_mask)
        if same_batch:
            input_batch, target_batch, conditions_batch = self.Task.get_batch(shuffle=shuffle)
            input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.RNN.device)
            target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.RNN.device)
            batch_mask = mask_of(conditions_batch)

        tic = time.perf_counter()
        # torch.autograd.set_detect_anomaly(True)
        for iter in range(self.max_iter):

            if not same_batch:
                input_batch, target_batch, conditions_batch = self.Task.get_batch(shuffle=shuffle)
                input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.RNN.device)
                target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.RNN.device)
                batch_mask = mask_of(conditions_batch)

            if self.track_participation and (iter % self.track_every == 0):
                self.track_participation_(input_batch, iter,
                                          target_batch=target_batch, mask=batch_mask)

            train_loss, r2 = self.train_step(input=input_batch,
                                         target_output=target_batch,
                                         mask=batch_mask)

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
    def format_duration_(seconds):
        """Format a duration as [D-]HH:MM:SS, without wrapping at 24 hours.

        The obvious spelling, time.strftime("%H:%M:%S", time.gmtime(seconds)), is wrong for
        durations: gmtime maps the value onto a time OF DAY, and %H is an hour-of-day field, so it
        wraps at 24 h. A 50-hour run printed "02:00:00" and every ETA past a day was unusable -
        precisely the long runs where an ETA is worth having. Formatted arithmetically here, with a
        leading day count in the same D-HH:MM:SS form squeue uses.

        Args:
            seconds: duration in seconds; may be negative or non-integral. A projected remaining
                time goes slightly negative on the final iteration (which is what produced the
                "remaining ~ 23:59:59" seen at the end of completed runs), so negatives clamp to 0.
        Returns:
            str: "07:12:33" under a day, "2-03:20:00" beyond one.
        """
        s = max(0, int(seconds))
        d, s = divmod(s, 86400)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def get_eta_(tic, toc, iter, max_iter):
        """Elapsed wall time and projected time remaining, both as [D-]HH:MM:SS strings.

        Args:
            tic, toc: perf_counter timestamps at the start of training and now.
            iter: zero-based index of the iteration just completed.
            max_iter: total iterations the run will perform.
        Returns:
            (elapsed, eta) formatted by format_duration_.
        """
        delta = toc - tic
        proj_total = (delta / (iter + 1)) * max_iter
        return (Trainer.format_duration_(delta),
                Trainer.format_duration_(proj_total - delta))
    
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


if __name__ == "__main__":
    # Self-check for the duration formatter. The 24-hour wrap it replaces was reported three times
    # from SLURM logs before being fixed, so the >24 h cases are the ones that matter here.
    _f = Trainer.format_duration_
    assert _f(0) == "00:00:00"
    assert _f(59.9) == "00:00:59"                 # truncates, never rounds up to :60
    assert _f(3661) == "01:01:01"
    assert _f(86399) == "23:59:59"                # last value the old version got right
    assert _f(86400) == "1-00:00:00"              # old version wrapped this to 00:00:00
    assert _f(178571) == "2-01:36:11"             # ~49.6 h, the Della N=10000 case
    assert _f(-5) == "00:00:00"                   # negative remaining on the final iteration
    print("format_duration_ self-check passed")
