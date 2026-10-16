#!/usr/bin/env python3
"""Is the FIRING-RATE distribution lognormal, and which way does it err? One row per network.

The companion of the weight-distribution cache. Slide 20 used to ask this of the recurrent weights;
the question the literature actually poses is about rates - the across-neuron firing-rate
distribution in cortex is reported as lognormal (Buzsaki & Mizuseki 2014), and the weights are a
separate claim with a separate reference. This script measures the rate version.

WHAT IS MEASURED, and the three choices it rests on:

  1. ACROSS UNITS, not across time. One number per unit, then the distribution of those numbers over
     the population. That is the shape the lognormal claim is about.
  2. THE STATISTIC PER UNIT IS THE MEAN RATE over time and trials, on the noise-free pass. Two
     others are recorded beside it so the answer can be checked against the choice: the soft maximum
     over time that frm uses, and the project's participation statistic std(r) + q_0.9(|r|). A claim
     that holds on one and not the others is a claim about the statistic, not about the network.
  3. OVER ACTIVE UNITS ONLY. log(0) is undefined and a silent unit's rate is numerically zero, so a
     distribution "over all N" cannot be put on a log axis at all. The active set is the project's
     scale-free rule, p_i >= 0.05 * q_95(p). ⚠️ THIS MAKES THE COMPARISON CONDITIONAL: an arm that
     revives 600 units is described over 600 units and the control over 300, so a difference in
     shape can come from which units are in the set. The count is recorded per row so that is
     visible, and the same statistics are also computed over each network's top-K loudest units
     for a FIXED K (200, below the smallest control's active count) as a control for exactly this:
     every arm is then described over the same NUMBER of units, so a difference in shape cannot be
     a difference in how many units are in the set.

FOR EACH NETWORK:
  n_active       units passing the scale-free rule on the noise-free pass
  r_hist         histogram of log10(rate) over the active units, on fixed bins
  r_sigma_log    sd of ln(rate) - the lognormal width parameter
  r_ks, r_ks_ref KS distance of ln(rate) to its own best-fit normal, and the same for a same-size
                 sample drawn FROM that fitted normal, which is the sampling scale the first number
                 has to beat before any departure is real
  r_skew_log     skewness of ln(rate), 0 for an exact lognormal; negative = an excess of very quiet
                 units, positive = an excess of very loud ones
  r_kurt_log     excess kurtosis of ln(rate), 0 for an exact lognormal
  r_spread       log10 of the ratio q99/q01 of the rate, so "1,000-fold" is readable off it
  rk_*           the same six over the 200 loudest units of that network, every arm matched on count

Usage (where the trained networks live):
    python f2_rate_lognormal.py                      # writes ~/f2_rate_lognormal.npz
    python f2_rate_lognormal.py OUT.npz
    F2_REPO=~/worktree F2_DATA=~/trained_RNNs python f2_rate_lognormal.py
Then copy beside the figure script as data/f2_rate_lognormal.npz.
"""

import glob
import os
import sys

import numpy as np
from scipy import stats
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f2_remedies_cache as C   # the loader, the gate and the arm classification, reused verbatim
from common import active_mask

import torch

OUT_DEFAULT = os.path.expanduser("~/f2_rate_lognormal.npz")
# the slide this feeds is one task at one size; set either to None to sweep everything
ONLY_TASK = os.environ.get("F2R_TASK", "NBitFlipFlop") or None
ONLY_N = int(os.environ.get("F2R_N", "1000") or 0) or None
LOG_BINS = np.linspace(-6.0, 1.0, 141)   # log10(rate); covers the quietest active unit to the top
KS_REF_DRAWS = 8                         # samples averaged for the sampling-scale reference
SHARED_K = 200                           # units per network for the count-matched comparison;
                                         # below the smallest control active count (257)
ZERO = 1e-12


def rate_stats(r, idx):
    """Lognormal diagnostics of a per-unit rate vector over a chosen subset of units.

    Args:
        r: (N,) float array of per-unit rate statistics, >= 0;
        idx: index array or boolean mask selecting the units to describe.
    Returns:
        dict with hist, sigma_log, ks, ks_ref, skew_log, kurt_log, spread, n. Every field is NaN
        and the histogram zero if fewer than 8 units survive, because none of these statistics
        means anything on a handful of points.
    """
    v = np.asarray(r, float)[idx]
    v = v[np.isfinite(v) & (v > ZERO)]
    out = {"hist": np.zeros(len(LOG_BINS) - 1, float), "n": len(v)}
    if len(v) < 8:
        out.update({k: np.nan for k in
                    ("sigma_log", "ks", "ks_ref", "skew_log", "kurt_log", "spread")})
        return out
    lg = np.log(v)
    out["hist"] = np.histogram(np.log10(v), bins=LOG_BINS)[0].astype(float)
    out["sigma_log"] = float(lg.std(ddof=1))
    out["skew_log"] = float(stats.skew(lg))
    out["kurt_log"] = float(stats.kurtosis(lg))      # Fisher: 0 for a normal
    mu, sd = lg.mean(), lg.std(ddof=1)
    out["ks"] = float(stats.kstest(lg, "norm", args=(mu, sd)).statistic)
    rng = np.random.default_rng(0)
    out["ks_ref"] = float(np.mean([
        stats.kstest(rng.normal(mu, sd, len(lg)), "norm", args=(mu, sd)).statistic
        for _ in range(KS_REF_DRAWS)]))
    q01, q99 = np.quantile(v, [0.01, 0.99])
    out["spread"] = float(np.log10(q99 / max(q01, ZERO)))
    return out


def per_unit_rates(net_dir):
    """Three per-unit rate statistics of one network, on the noise-free pass.

    Args:
        net_dir: path to one trained-network folder.
    Returns:
        (mean_rate, softmax_rate, participation, stored, n_units) - three (N,) arrays plus the
        folder's stored r2 and the network size. The pass is `w_noise=False`, which also switches
        off sigma_w, so every arm is described in one condition.
    """
    rnn, task, _mask, stored, _d = C.load_net(net_dir)
    bi_np, _bt_np, _ = task.get_batch()
    n_trials = np.asarray(bi_np).shape[2]
    sub = np.arange(0, n_trials, max(1, n_trials // C.TRIALS))[:C.TRIALS]
    bi = torch.tensor(np.asarray(bi_np)[:, :, sub], dtype=torch.float32)
    with torch.no_grad():
        states, _ = rnn(bi, w_noise=False)
        r = torch.relu(states).numpy()                      # (N, T, B)
    flat = r.reshape(r.shape[0], -1)
    mean_rate = flat.mean(axis=1)
    # frm's own statistic: a soft maximum over time, taken per trial then averaged over trials.
    # ⚠️ VIA logsumexp, NOT exp-then-sum: at tau = 0.1 a rate of 8 already overflows float64 in
    # exp(r/tau), and the first version of this line did that silently on the louder arms.
    tau = 0.1
    sm = tau * logsumexp(r / tau, axis=1) - tau * np.log(r.shape[1])
    softmax_rate = sm.mean(axis=1)
    participation = flat.std(axis=1) + np.quantile(np.abs(flat), 0.9, axis=1)
    return mean_rate, softmax_rate, participation, stored, int(rnn.N)


def main(out_path=OUT_DEFAULT):
    """Walk every cell the Figure 2 cache would accept and write one row per network.

    Args:
        out_path: where to write the npz.
    Returns:
        the number of networks recorded.
    """
    cells = C.discover(C.D)
    if ONLY_TASK:
        cells = [c for c in cells if c[1] == ONLY_TASK]
    if ONLY_N:
        cells = [c for c in cells if c[2] == ONLY_N]
    print(f"{len(cells)} matched cells under {C.D}")
    rows = []
    for arm, task, n_cell, pat in cells:
        for net in sorted(glob.glob(os.path.join(C.D, pat, "*"))):
            if not os.path.isdir(net):
                continue
            try:
                mean_rate, softmax_rate, part, stored, n_units = per_unit_rates(net)
            except Exception as exc:
                print(f"  SKIP {arm} {os.path.basename(net)[:9]}: {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            active = active_mask(part)
            row = {"cell": pat, "arm": arm, "N": n_units, "stored": stored, "task": task,
                   "n_active": int(active.sum())}
            # the top-K set is taken on the SAME statistic being described, so "the 200 loudest"
            # means loudest by that statistic rather than by a different one
            for tag, v in (("", mean_rate), ("sm", softmax_rate), ("pt", part)):
                pre = f"r_{tag}_" if tag else "r_"
                for st, sel in ((rate_stats(v, active), pre),
                                (rate_stats(v, np.argsort(v)[-SHARED_K:]), pre.replace("r", "rk", 1))):
                    for k, val in st.items():
                        row[f"{sel}{k}"] = val
            rows.append(row)
            print(f"  {row['task'][:5]:5s} {arm:10s} N={n_units:5d} active {row['n_active']:4d} "
                  f"sd ln r {row['r_sigma_log']:.2f}  skew {row['r_skew_log']:+.2f}  "
                  f"ks {row['r_ks']:.3f} (ref {row['r_ks_ref']:.3f})", flush=True)
    if not rows:
        print("no networks found")
        return 0
    keys = sorted({k for r in rows for k in r})
    np.savez(out_path, log_bins=LOG_BINS,
             **{k: np.array([r.get(k, np.nan) for r in rows],
                            dtype=object if isinstance(rows[0].get(k), str) else float)
                for k in keys})
    print(f"\nwrote {out_path}: {len(rows)} networks")
    return len(rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT)
