#!/usr/bin/env python3
"""
Build the cache behind manuscript Figure 2: four measures for every network of the three remedies.

RUNS ON THE CLUSTER, where the trained networks live; the figure script reads the cache it writes.
The four arms are matched on everything but the intervention -- gamma = 0, N = 1000, 3-bit
flip-flop, 40,000 iterations, lr 1e-3, weight decay 1e-6, sigma_rec = sigma_inp = 0.05, batch 1024.
Cells trained with a different gamma are deliberately NOT included: gamma is cubic saturation in
the dynamics, so it changes the base network and a cross-gamma comparison is not like for like.
(That rules out NBitFlipFlop_ff_revive_g01_fix, the only other corrected duplication sweep.)

⚠️ THE DUPLICATION CELL IS THE CORRECTED CONSTRUCTION. Before 2026-09-24 duplication zeroed the
2x2 weight block spanning donor and copy, which left the copy with no self-connection and the donor
with half of its own, so a single duplication moved the output by up to 8.9e-03 against an output
scale of 0.27. Cells carrying that bug are named `__DETUNED_SELFWEIGHT` on disk and are excluded.

THE FOUR MEASURES, per network:
  r2            recomputed on a fresh held-out batch WITH the network's own noise, which is the
                quantity stored in the folder name -- so it can be checked against it. The
                noise-free value is recorded too and printed in the table.
  active units  scale-free rule, p_i >= 0.05 * q_95(p), on p_i = std(r_i) + q_0.9(|r_i|).
  dimensionality  participation ratio (sum ev)^2 / sum ev^2 of the noise-free rate covariance over
                the ACTIVE units: the number of directions the population uses, 1 if every unit does
                the same thing. The count of components reaching 95% of the variance is recorded
                beside it, because participation ratio and a variance threshold can disagree.
  weight distribution  histogram of log10|W_rec| over the nonzero entries, plus the lognormal shape
                statistics of ln|W_rec| (sd, skew, and the log10 q99/q01 range). Recorded for the
                whole matrix and for the incoming rows of the active units only, since duplication
                and rescale both act on incoming rows.

⚠️ GATE. Every network is rebuilt from ITS OWN saved config and scored; anything whose recomputed r2
misses the stored value by more than R2_TOL is dropped, because a wrong forward pass still yields a
plausible-looking dimensionality and a perfectly plausible weight histogram.

Usage (on the cluster, under a repo new enough for every feature these runs used):
    python f2_remedies_cache.py                     # writes ~/fig_paper_F2_cache.npz
    python f2_remedies_cache.py OUT.npz
    F2_REPO=~/other_worktree python f2_remedies_cache.py
Then copy the file to data/fig_paper_F2_cache.npz beside the figure script.
"""

import glob
import os
import re
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

# The repo the networks are REBUILT from must have every model feature they were trained with.
# `_cperturb` predates `sigma_w` and raises TypeError on the whole synaptic-noise arm, so the
# default is a worktree new enough to carry it. Older cells still pass the r2 gate from here,
# which is the check that the task class has not moved under them.
REPO = os.environ.get("F2_REPO", "/home/pt1290/trainRNNbrain_sizeser")
sys.path.insert(0, REPO)
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask

D = os.environ.get("F2_DATA", "/home/pt1290/trainRNNbrain/data/trained_RNNs")
R2_TOL = 0.03
TRIALS = 128                 # 300 timesteps x 128 trials = 38400 samples against 1000 units, far
                             # above what a covariance over at most 1000 units needs. 256 trials
                             # holds two 1000x76800 rate matrices at once and is OOM-killed.
SILENT_REL = 0.05            # the scale-free silence rule used everywhere in this project
ZERO_TOL = 1e-12
# log10|W| bin edges. The range has to cover the WIDEST arm, not the control: at [-8, 0.5] the
# target-based rescale cells lost up to 4.2% of their weights off the right edge, which is the
# tail that panel (e) exists to show. The lower edge matches ZERO_TOL, so nothing falls off the
# left; the check below asserts the loss is negligible rather than trusting the range.
LOG_BINS = np.linspace(-12.0, 4.0, 321)

# (arm, label, N, cell path). Every cell is gamma = 0, 3-bit flip-flop, 40,000 iterations, lr 1e-3,
# weight decay 1e-6, sigma_rec = sigma_inp = 0.05, batch 1024 - checked config by config, not assumed
# from the folder name. N = 1000 carries the four-way comparison; the other sizes carry the dropout
# size series, whose N = 4000 control comes from `paper_grid` because `ff_revive` never ran one.
#
# Every rescale cell that is matched is measured, not just one: the rule was developed over five
# sweeps and its early alpha-only form recruits nothing, so quoting that form as "rescale" reports
# the rule at its weakest. Synaptic noise is a grid in the same way. Which cell the FIGURE draws is
# chosen in fig_paper_F2.py and the whole grid is printed beneath it.
RESCALE_CELLS = [
    # alpha only, no activity target: the first form of the rule (2026-09-25)
    "NBitFlipFlop_rescale_revive/EqType=h_k=3_N=1000_norm=t_a=1.0005",
    "NBitFlipFlop_rescale_revive/EqType=h_k=3_N=1000_norm=t_a=1.002",
    "NBitFlipFlop_rescale_revive/EqType=h_k=3_N=1000_norm=f_a=1.0005",
    "NBitFlipFlop_rescale_revive/EqType=h_k=3_N=1000_norm=f_a=1.002",
    # a fixed activity target, so a boosted unit is held until it fires and then released
    "NBitFlipFlop_rescale_states/EqType=h_k=3_N=1000_norm=t_a=1.002_tgt=0.2",
    "NBitFlipFlop_rescale_states/EqType=h_k=3_N=1000_norm=t_a=1.002_tgt=0.5",
    "NBitFlipFlop_rescale_active/EqType=h_k=3_N=1000_norm=t_a=1.002_tgt=2.5_act=true",
    "NBitFlipFlop_rescale_active/EqType=h_k=3_N=1000_norm=f_a=1.002_tgt=2.5_act=true",
    "NBitFlipFlop_revive_ops2/EqType=h_k=3_N=1000_op=rescale_novel=false_tgt=2.5_step=0.0005_prot=0.35",
    "NBitFlipFlop_revive_ops2/EqType=h_k=3_N=1000_op=rescale_novel=false_tgt=4.0_step=0.0005_prot=0.20",
    "NBitFlipFlop_revive_ops2/EqType=h_k=3_N=1000_op=rescale_novel=false_tgt=6.0_step=0.0005_prot=0.20",
    # ... and released with a refractory tail rather than immediately
    "NBitFlipFlop_revive_refr/EqType=h_k=3_N=1000_op=rescale_novel=false_tgt=2.5_step=0.0005_prot=0.20_refr=1000",
    "NBitFlipFlop_revive_refr/EqType=h_k=3_N=1000_op=rescale_novel=false_tgt=2.5_step=0.0005_prot=0.20_refr=4000",
]

# Synaptic noise: the recurrent weight matrix is redrawn around its mean at EVERY timestep, with
# per-synapse sd sigma_w * |W_ij|. sw is that multiplier, so 1.0 means a synapse fluctuates by as
# much as its own strength. This is not a rule that acts on silent units at all - it is a property
# of the dynamics - which is why it is the one arm here that changes no update rule.
SYNNOISE_CELLS = [
    "NBitFlipFlop_synnoise/EqType=h_k=3_N=1000_sw=0.1",
    "NBitFlipFlop_synnoise/EqType=h_k=3_N=1000_sw=0.3",
    "NBitFlipFlop_synnoise/EqType=h_k=3_N=1000_sw=1.0",
    "NBitFlipFlop_synnoise_ext/EqType=h_N=1000_sw=2.0_iters=40000",
    "NBitFlipFlop_synnoise_ext/EqType=h_N=1000_sw=3.0_iters=40000",
]

# The dropout size series, and the control at each size it is measured against.
SIZE_CELLS = [
    ("control", 500, "NBitFlipFlop_ff_revive/EqType=h_k=3_N=500_pen=none_arm=none"),
    ("control", 2000, "NBitFlipFlop_ff_revive/EqType=h_k=3_N=2000_pen=none_arm=none"),
    ("control", 4000, "NBitFlipFlop_paper_grid/EqType=h_N=4000_arm=control"),
    ("mute", 500, "NBitFlipFlop_dropout_sizes/EqType=h_k=3_N=500_pen=none_do=mute_rate=0.20_beta=4"),
    ("mute", 2000, "NBitFlipFlop_dropout_sizes/EqType=h_k=3_N=2000_pen=none_do=mute_rate=0.20_beta=4"),
    ("mute", 4000, "NBitFlipFlop_dropout_sizes/EqType=h_k=3_N=4000_pen=none_do=mute_rate=0.20_beta=4"),
]

CELLS = [
    ("control", "no intervention", 1000,
     "NBitFlipFlop_ff_revive/EqType=h_k=3_N=1000_pen=none_arm=none"),
    ("mute", "dropout: mute", 1000,
     "NBitFlipFlop_dropout_sizes/EqType=h_k=3_N=1000_pen=none_do=mute_rate=0.20_beta=4"),
    ("duplicate", "prune + duplicate", 1000,
     "NBitFlipFlop_copy_perturb/EqType=h_k=3_N=1000_cn=0"),
] + [("rescale", "rescale", 1000, c) for c in RESCALE_CELLS] \
  + [("synnoise", "synaptic noise", 1000, c) for c in SYNNOISE_CELLS] \
  + [(arm, "size series", n, c) for arm, n, c in SIZE_CELLS]


def participation_ratio(x):
    """Number of directions a population uses: (sum ev)^2 / sum ev^2 of its covariance.

    Args:
        x: array (n_units, n_samples) of firing rates.
    Returns:
        float, between 1 and n_units.
    """
    ev = np.linalg.eigvalsh(np.cov(np.asarray(x, dtype=np.float64)))
    ev = ev[ev > 1e-12]
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def n_comp_95(x):
    """How many principal components carry 95% of a population's variance.

    Args:
        x: array (n_units, n_samples) of firing rates.
    Returns:
        int.
    """
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.asarray(x, dtype=np.float64))))[::-1]
    ev = ev[ev > 1e-12]
    return int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1)


def weight_shape(w):
    """Histogram and lognormal shape of a set of weight magnitudes.

    Args:
        w: array of weights of any shape; zeros and structural zeros are dropped.
    Returns:
        dict with 'hist' (counts over LOG_BINS of log10|w|), 'n_nonzero', 'sigma_log' (sd of
        ln|w|), 'skew_log' (skewness of ln|w|, 0 for an exact lognormal) and 'spread'
        (log10 of the q99/q01 magnitude ratio), or None if fewer than 100 nonzeros.
    """
    m = np.abs(np.asarray(w, dtype=np.float64).ravel())
    m = m[m > ZERO_TOL]
    if m.size < 100:
        return None
    lg = np.log(m)
    z = (lg - lg.mean()) / lg.std(ddof=1)
    return dict(hist=np.histogram(np.log10(m), bins=LOG_BINS)[0].astype(np.int64),
                n_nonzero=int(m.size),
                sigma_log=float(lg.std(ddof=1)),
                skew_log=float((z ** 3).mean()),
                spread=float(np.log10(np.quantile(m, 0.99) / np.quantile(m, 0.01))))


def load_net(net_dir):
    """Rebuild a trained network from the config saved beside its weights.

    Rebuilding from a reconstructed config instead of the saved one substitutes the wrong noise
    levels and drops the trained bias, and then fails the r2 gate on every network.

    Args:
        net_dir: path to one trained-network folder.
    Returns:
        (RNN_torch, task, training mask, stored r2 read off the folder name, weight dict).
    """
    cfg = OmegaConf.load(glob.glob(os.path.join(net_dir, "*_config.yaml"))[0])
    raw = dict(np.load(glob.glob(os.path.join(net_dir, "*LastParams*.npz"))[0], allow_pickle=True))
    d = {k: (v.item() if isinstance(v, np.ndarray) and v.dtype == object and v.shape == () else v)
         for k, v in raw.items()}
    m = cfg.model
    # activation_args comes from the CONFIG, not the npz. Different sweeps saved it differently -
    # the cperturb runs stored the dict, the CDDM/flip-flop penalty runs stored only its KEYS as a
    # string array, which dict() turns into a ValueError - and the config is the authoritative
    # record either way.
    rnn = RNN_torch(N=int(m.N), activation_args=OmegaConf.to_container(m.activation_args),
                    equation_type=str(m.equation_type), dale=bool(m.dale),
                    io_nonnegativity=bool(m.io_nonnegativity),
                    self_connections=bool(m.self_connections), bias_range=list(m.bias_range),
                    gamma=float(m.gamma), dt=float(m.dt), tau=float(m.tau),
                    sigma_rec=float(m.sigma_rec), sigma_inp=float(m.sigma_inp),
                    sigma_out=float(m.sigma_out),
                    # ⚠️ sigma_w IS PART OF THE TRAINED CONDITION. The synaptic-noise arm redraws
                    # W_rec around its mean at every timestep; rebuilding without it scores the
                    # network under dynamics it never trained in. Left out, the sw=2.0 and sw=3.0
                    # cells recomputed at r2 0.66-0.83 against a stored 0.92-0.93 and failed the
                    # gate, and the quieter cells passed while still being scored wrongly.
                    sigma_w=float(getattr(m, "sigma_w", 0.0)),
                    n_inputs=int(cfg.task.n_inputs),
                    n_outputs=int(cfg.task.n_outputs), seed=0)
    with torch.no_grad():
        for k in ("W_rec", "W_inp", "W_out"):
            getattr(rnn, k).copy_(torch.tensor(np.asarray(d[k], dtype=np.float32)))
        yi = torch.tensor(np.asarray(d["y_init"], dtype=np.float32))
        if isinstance(rnn.y_init, torch.nn.Parameter):
            rnn.y_init.copy_(yi)
        else:
            rnn.y_init = yi
        if d.get("bias") is not None and np.asarray(d["bias"]).size == rnn.N:
            rnn.bias.copy_(torch.tensor(np.asarray(d["bias"], dtype=np.float32)))
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
    stored = float(re.match(r"(-?[0-9.]+|nan)_", os.path.basename(net_dir)).group(1))
    return rnn, task, mask, stored, d


def analyse(net_dir):
    """All four measures for one trained network.

    Args:
        net_dir: path to one trained-network folder.
    Returns:
        dict of scalars plus the two weight histograms.
    """
    rnn, task, mask, stored, d = load_net(net_dir)
    n_from_cfg = int(rnn.N)
    bi, bt, _ = task.get_batch()
    bi = torch.tensor(bi[:, :, :TRIALS], dtype=torch.float32)
    bt = torch.tensor(bt[:, :, :TRIALS], dtype=torch.float32)
    with torch.no_grad():
        states, out = rnn(bi, w_noise=True)
        r2_noisy = float(Trainer.r2_score(out, bt, mask))
        r_noisy = torch.relu(states).numpy().reshape(rnn.N, -1)   # float32, on purpose
        srec, sinp, sw = float(rnn.sigma_rec), float(rnn.sigma_inp), float(rnn.sigma_w)
        rnn.sigma_rec = rnn.sigma_inp = rnn.sigma_w = 0.0
        states_c, out_c = rnn(bi, w_noise=False)
        rnn.sigma_rec, rnn.sigma_inp, rnn.sigma_w = srec, sinp, sw
        r2_clean = float(Trainer.r2_score(out_c, bt, mask))
        r_clean = torch.relu(states_c).numpy().reshape(rnn.N, -1)

    # active units on the NOISY run, which is the condition the network trained in and the one every
    # other active-unit count in this project is measured under
    p = r_noisy.std(axis=1) + np.quantile(np.abs(r_noisy), 0.9, axis=1)
    live = p >= SILENT_REL * np.quantile(p, 0.95)

    W = np.asarray(d["W_rec"], dtype=np.float64)
    whole = weight_shape(W)
    rows = weight_shape(W[live]) if live.sum() > 5 else None
    inp = weight_shape(np.asarray(d["W_inp"], dtype=np.float64))

    if whole is not None:
        outside = 1.0 - whole["hist"].sum() / whole["n_nonzero"]
        assert outside < 1e-4, (f"{outside:.2%} of |W_rec| falls outside the histogram range "
                               f"{LOG_BINS[0]:.0f}..{LOG_BINS[-1]:.0f}; widen LOG_BINS")
    out_d = dict(N_cfg=n_from_cfg, stored=stored, r2=r2_noisy, r2_clean=r2_clean,
                 n_active=int(live.sum()),
                 dims=float("nan"), dims95=float("nan"),
                 w_inp_sigma=float("nan") if inp is None else inp["sigma_log"])
    if live.sum() >= 2:
        out_d["dims"] = participation_ratio(r_clean[live])
        out_d["dims95"] = float(n_comp_95(r_clean[live]))
    for tag, sh in (("w", whole), ("wact", rows)):
        for k in ("hist", "n_nonzero", "sigma_log", "skew_log", "spread"):
            out_d[f"{tag}_{k}"] = (np.zeros(len(LOG_BINS) - 1, np.int64) if k == "hist" else
                                   float("nan")) if sh is None else sh[k]
    return out_d


def main(out_path):
    """Score every network of every cell and write the cache. Returns the output path."""
    recs, fields = [], None
    for arm, label, n_units, pat in CELLS:
        for nd in sorted(glob.glob(os.path.join(D, pat, "*"))):
            if not os.path.isdir(nd):
                continue
            try:
                r = analyse(nd)
            except Exception as e:
                print(f"  SKIP {arm:>10s} {os.path.basename(nd)[:12]}: {type(e).__name__}: {e}",
                      flush=True)
                continue
            if r["N_cfg"] != n_units:
                print(f"  SKIP {arm:>10s} {os.path.basename(nd)[:12]}: cell says N={n_units} but "
                      f"the saved config says N={r['N_cfg']}", flush=True)
                continue
            gate = abs(r["stored"] - r["r2"]) < R2_TOL
            print(f"  {arm:>10s} N={n_units:5d} stored {r['stored']:7.4f} recomp {r['r2']:7.4f} "
                  f"{'PASS' if gate else 'FAIL':>4s}  active {r['n_active']:4d}  "
                  f"dims {r['dims']:6.2f}  sigma_log {r['w_sigma_log']:5.2f}", flush=True)
            if not gate:
                continue
            r.update(arm=arm, label=label, N=n_units, cell="/".join(pat.split("/")[-2:]))
            recs.append(r)
            fields = fields or sorted(r)
    if not recs:
        raise SystemExit("no network passed the gate")
    cache = {k: np.array([r[k] for r in recs]) for k in fields}
    cache["log_bins"] = LOG_BINS
    np.savez_compressed(out_path, **cache)
    print(f"\nwrote {out_path}: {len(recs)} networks, "
          f"{ {a: sum(r['arm'] == a for r in recs) for a in dict.fromkeys(r['arm'] for r in recs)} }")
    return out_path


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/fig_paper_F2_cache.npz"))
