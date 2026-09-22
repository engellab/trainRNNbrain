"""Does a hard cap pile weight mass AT the bound, and how lognormal are the distributions anyway?

TWO QUESTIONS, ONE READ-OUT.

(1) THE PILE-UP CHECK. `enforce_inp_cap_` is a HARD clamp. In the plasticity literature, additive
    rules with hard bounds produce BIMODAL weight distributions with mass stacked against the
    bound (van Rossum/Bi/Turrigiano 2000; Rubin/Lee/Sompolinsky 2001; Gutig et al. 2003), and the
    unimodal lognormal seen experimentally was historically used as evidence AGAINST hard bounds.
    Cortex has a physical ceiling -- PSD area, receptor slots, spine volume -- but approaches it as
    a TAIL, not a wall. So if a substantial fraction of |W_inp| sits exactly at the cap, the clamp
    has manufactured a distribution biology does not produce, and any units it recruited came from
    that artifact rather than from the mechanism the experiment was meant to test.

    PRE-REGISTERED THRESHOLD, fixed before running: pile-up is CONFIRMED if more than 1% of the
    |W_inp| entries sit within 1e-6 of the cap. The 1% bar is not arbitrary -- at initialisation
    |W_inp| ~ half-normal with sd 1/sqrt(N) = 0.032 at N=1000, so a cap of 0.3 is ~9.4 sd away and
    essentially NO mass starts near it. Any appreciable mass at the bound is training-induced.

(2) THE LOGNORMAL BASELINE. Cortical EPSP amplitudes among connected pairs are lognormal, spanning
    roughly two orders of magnitude, with a large atom at exactly zero from unconnected pairs
    (Song et al. 2005; Lefort et al. 2009; Buzsaki & Mizuseki 2014). This reports how far the
    trained networks sit from that shape, for weights AND for firing-rate participation. A
    lognormal has support on (0, inf) and therefore NO atom at zero -- so "make the distribution
    lognormal" and "leave no unit silent" are the same requirement, not competing ones.

Usage:
    python trainRNNbrain/experiments_and_analysis/weight_distribution_shape.py <net_dir_glob> [cap]
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
from scipy import stats

ZERO_TOL = 1e-12      # below this a weight counts as structurally absent, not small
CAP_TOL = 1e-6        # within this of the cap counts as sitting ON the bound
PILEUP_BAR = 0.01     # pre-registered: >1% of entries on the bound = pile-up confirmed
rng = np.random.default_rng(0)   # only for the normal reference sample in lognormal_report


def load_net(net_dir):
    """Load one trained network's weights and its final participation vector.

    Args:
        net_dir: directory holding *LastParams_*.json and *ParticipationTrace.pkl.
    Returns:
        dict with W_inp, W_rec, W_out (np arrays) and participation (np array or None).
    """
    pf = glob.glob(os.path.join(net_dir, "*LastParams_*.json"))
    if not pf:
        return None
    d = json.load(open(pf[0]))
    out = {k: np.asarray(d[k], dtype=np.float64) for k in ("W_inp", "W_rec", "W_out")}
    tf = glob.glob(os.path.join(net_dir, "*ParticipationTrace.pkl"))
    out["participation"] = (np.asarray(pickle.load(open(tf[0], "rb"))["participation"][-1],
                                       dtype=np.float64) if tf else None)
    return out


def lognormal_report(x, label):
    """Describe how lognormal a set of non-negative magnitudes is.

    Args:
        x: 1-D array of non-negative values (weights or participations).
        label: name for the printed row.
    Returns:
        dict of summary statistics; also prints one formatted line.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    zero_frac = float((x <= ZERO_TOL).mean())
    nz = x[x > ZERO_TOL]
    if nz.size < 10:
        print(f"  {label:26s} too few nonzero values ({nz.size})")
        return {}
    lg = np.log(nz)
    # KS of log-magnitudes against the best-fit normal. Parameters are estimated from the same
    # data, so this is a descriptive goodness-of-fit, NOT a calibrated p-value -- reported as a
    # distance only, and a normal reference sample is printed alongside so the number has a scale.
    ks = stats.kstest(lg, "norm", args=(lg.mean(), lg.std(ddof=1))).statistic
    ref = stats.kstest(rng.normal(lg.mean(), lg.std(ddof=1), nz.size), "norm",
                       args=(lg.mean(), lg.std(ddof=1))).statistic
    decades = float(np.log10(np.quantile(nz, 0.99) / np.quantile(nz, 0.01)))
    print(f"  {label:26s} zero={zero_frac:6.1%}  sigma_log={lg.std(ddof=1):5.2f}  "
          f"KS={ks:.3f} (normal ref {ref:.3f})  spread={decades:4.1f} decades")
    return {"zero_frac": zero_frac, "sigma_log": lg.std(ddof=1), "ks": ks, "decades": decades}


def pileup_report(W_inp, cap):
    """Fraction of |W_inp| entries sitting exactly on a hard cap.

    Args:
        W_inp: (N, n_inputs) input weight matrix.
        cap: the clamp value used during training, or None if uncapped.
    Returns:
        fraction on the bound (float), or None if cap is None.
    """
    a = np.abs(W_inp).ravel()
    if cap is None:
        print(f"  {'pile-up':26s} no cap applied; max|W_inp|={a.max():.3f}")
        return None
    on = float((np.abs(a - cap) <= CAP_TOL).mean())
    verdict = "PILE-UP CONFIRMED" if on > PILEUP_BAR else "no pile-up"
    print(f"  {'pile-up at cap':26s} {on:6.2%} of entries within {CAP_TOL:g} of cap={cap}  "
          f"-> {verdict} (bar {PILEUP_BAR:.0%})")
    return on


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pattern = sys.argv[1]
    cap = float(sys.argv[2]) if len(sys.argv) > 2 else None
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    print(f"{len(dirs)} net directories matching {pattern}\n")
    for nd in dirs:
        net = load_net(nd)
        if net is None:
            continue
        print(os.path.basename(os.path.dirname(nd))[:70] or nd[:70])
        lognormal_report(np.abs(net["W_inp"]), "|W_inp|")
        lognormal_report(np.abs(net["W_rec"]), "|W_rec|")
        if net["participation"] is not None:
            lognormal_report(net["participation"], "participation (rates)")
        pileup_report(net["W_inp"], cap)
        print()
