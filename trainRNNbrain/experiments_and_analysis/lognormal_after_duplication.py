"""Does duplication with jitter leave the recurrent weights lognormal?

WHY IT MATTERS. Cortical synaptic strengths among connected pairs are lognormal, spanning roughly
two orders of magnitude (Song et al. 2005; Lefort et al. 2009; Buzsaki & Mizuseki 2014). An
intervention that recruits units by manufacturing a weight distribution biology does not produce
has bought its units with an artifact. Duplication with multiplicative jitter is the intervention
most likely to distort the shape, because it multiplies a whole row by (1 + jitter * eps).

WHAT THE JITTER SHOULD DO, predicted before measuring. Multiplying a magnitude by |1 + jitter*eps|
adds log|1 + jitter*eps| to its log-magnitude. For SMALL jitter that increment is close to normal,
so a lognormal stays lognormal and only widens: sigma_log should go to
sqrt(sigma_log^2 + jitter^2). For LARGE jitter it is not: 1 + 3*eps lands near zero whenever eps is
near -1/3, and log of a near-zero number runs to minus infinity, so jitter 3.0 should add a heavy
LEFT tail and drive the skewness of log|W| negative. Skewness is 0 for an exact lognormal, which
makes it the statistic that separates the two regimes.

STATISTICS, all descriptive. sigma_log is the sd of log|W| over the nonzero weights, the lognormal
shape parameter. KS is the Kolmogorov-Smirnov DISTANCE between log|W| and the best-fit normal --
the parameters are fitted on the same data, so it is not a calibrated test, and with ~10^6 weights
any real distribution would reject at any p-value, which is why a same-size normal reference sample
is printed beside it to give the number a scale. skew is the skewness of log|W|. spread is
log10(q99/q01) of |W|, an order-of-magnitude range rather than a "decade" count.

Usage:
    python lognormal_after_duplication.py            (reads the cell list below)
"""
import glob
import os
import sys

import numpy as np
from scipy import stats

D = "/home/pt1290/trainRNNbrain/data/trained_RNNs"
ZERO_TOL = 1e-12
rng = np.random.default_rng(0)

CELLS = [
    ("control",        "NBitFlipFlop_ff_revive/EqType=h_k=3_N=1000_pen=none_arm=none"),
    ("copy iid",       "NBitFlipFlop_copy_iid/EqType=h_k=3_N=1000_cn=0"),
    ("copy permuted",  "NBitFlipFlop_copy_permute/EqType=h_k=3_N=1000_cn=0"),
    ("jitter 0",       "NBitFlipFlop_copy_perturb/EqType=h_k=3_N=1000_cn=0"),
    ("jitter 0.05",    "NBitFlipFlop_copy_perturb/EqType=h_k=3_N=1000_cn=0.05"),
    ("jitter 0.3",     "NBitFlipFlop_copy_perturb/EqType=h_k=3_N=1000_cn=0.3"),
    ("jitter 1.0",     "NBitFlipFlop_copy_perturb/EqType=h_k=3_N=1000_cn=1.0"),
    ("jitter 3.0",     "NBitFlipFlop_copy_perturb/EqType=h_k=3_N=1000_cn=3.0"),
    ("mute dropout",   "NBitFlipFlop_dropout_sizes/EqType=h_k=3_N=1000_pen=none_do=mute_rate=0.20_beta=4"),
    ("dead dropout",   "NBitFlipFlop_dropout_sizes/EqType=h_k=3_N=1000_pen=none_do=dead_rate=0.20_beta=4"),
]


def shape(x):
    """How lognormal a set of magnitudes is.

    Args:
        x: 1-D array of non-negative magnitudes.
    Returns: dict with sigma_log, ks, ks_ref, skew and spread, or None if too few nonzeros.
    """
    nz = np.asarray(x, dtype=np.float64).ravel()
    nz = nz[nz > ZERO_TOL]
    if nz.size < 100:
        return None
    lg = np.log(nz)
    mu, sd = lg.mean(), lg.std(ddof=1)
    return dict(
        sigma_log=float(sd),
        ks=float(stats.kstest(lg, "norm", args=(mu, sd)).statistic),
        ks_ref=float(stats.kstest(rng.normal(mu, sd, nz.size), "norm", args=(mu, sd)).statistic),
        skew=float(stats.skew(lg)),
        spread=float(np.log10(np.quantile(nz, 0.99) / np.quantile(nz, 0.01))))


def net_shapes(net_dir):
    """Lognormal shape of one network's recurrent weights, whole matrix and active rows.

    Args:
        net_dir: path to one trained-network folder.
    Returns: (whole-matrix dict, active-rows dict), either may be None.
    """
    import pickle
    W = np.asarray(np.load(glob.glob(net_dir + "/*LastParams*.npz")[0],
                           allow_pickle=True)["W_rec"], dtype=float)
    if not np.isfinite(W).all():
        return None, None
    tp = glob.glob(net_dir + "/*ParticipationTrace.pkl")
    live = None
    if tp:
        p = np.asarray(pickle.load(open(tp[0], "rb"))["participation"][-1], dtype=float)
        live = p >= 0.05 * np.quantile(p, 0.95)
    return shape(np.abs(W)), (shape(np.abs(W[live])) if live is not None and live.sum() > 5
                              else None)


def mean_of(dicts, key):
    """Mean of one field across per-network dicts, ignoring the ones that came back empty."""
    vals = [d[key] for d in dicts if d]
    return float(np.mean(vals)) if vals else float("nan")


if __name__ == "__main__":
    print("WHOLE RECURRENT MATRIX")
    print(f"{'cell':>15s} {'n':>2s} {'sigma_log':>10s} {'KS':>7s} {'KS ref':>7s} "
          f"{'skew':>7s} {'spread':>7s}")
    print("-" * 62)
    rows_active = {}
    for name, pat in CELLS:
        whole, act = [], []
        for nd in sorted(glob.glob(os.path.join(D, pat, "*"))):
            if not os.path.isdir(nd):
                continue
            w, a = net_shapes(nd)
            if w:
                whole.append(w)
            if a:
                act.append(a)
        if not whole:
            continue
        rows_active[name] = act
        print(f"{name:>15s} {len(whole):2d} {mean_of(whole,'sigma_log'):10.2f} "
              f"{mean_of(whole,'ks'):7.3f} {mean_of(whole,'ks_ref'):7.3f} "
              f"{mean_of(whole,'skew'):7.2f} {mean_of(whole,'spread'):7.1f}")

    print("\nROWS OF ACTIVE UNITS ONLY")
    print(f"{'cell':>15s} {'n':>2s} {'sigma_log':>10s} {'KS':>7s} {'KS ref':>7s} "
          f"{'skew':>7s} {'spread':>7s}")
    print("-" * 62)
    for name, act in rows_active.items():
        if not act:
            continue
        print(f"{name:>15s} {len(act):2d} {mean_of(act,'sigma_log'):10.2f} "
              f"{mean_of(act,'ks'):7.3f} {mean_of(act,'ks_ref'):7.3f} "
              f"{mean_of(act,'skew'):7.2f} {mean_of(act,'spread'):7.1f}")
