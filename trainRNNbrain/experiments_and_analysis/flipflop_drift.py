#!/usr/bin/env python3
"""
Did the flip-flop networks reach diffusive drift, or are updates still biased? A stopping criterion.

WHY ASK. Matching on an absolute loss level is uncomfortable whenever the floor differs with N, since
a common level then reads different sizes at different depths relative to their own asymptotes. The
natural repair - match at a common fraction of the way to the floor - needs L_inf, which is not
always identifiable within the budget. So a criterion that does not reference the loss at all would
be worth having.

WHAT WOULD COUNT. If training has stopped making systematic progress and is only jittering, then:

  cos_W          cosine between consecutive weight displacements. ~0 = isotropic jitter,
                 > 0 = still marching in a consistent direction. This is the cleanest single test.
  alpha          lag-scaling exponent of ||W(t)-W(t-L)||_F vs lag L, measured across the three
                 logged lags. 1.0 = ballistic (pure drift), 0.5 = diffusive, < 0.5 = confined.
  dp_lag         relative change of the PARTICIPATION vector over a lag - the same question asked
                 about which units are active rather than about the weights.

THIS FAILED ON CDDM, and the failure is documented: the relative weight change decays as a power law
so a 1% criterion extrapolates to 0.5-5.6 MILLION iterations, and alpha wanders below 0.5 and back
without ever crossing stably. The flip-flop is a different enough task (no discontinuous target, a
floor near zero, much faster early learning) that it is worth re-testing rather than assumed, and the
metrics are already in the traces - this costs nothing.

The honest prior is that it fails again. Reported either way; a negative result here is what
justifies the loss-matching protocol rather than leaving it looking like an arbitrary choice.

Output: img/internal_figures/flipflop_drift.png

Usage:  python flipflop_drift.py
NOTE ON PROVENANCE. Every number that this file previously quoted from the flip-flop came from the
first sweep, which ran `same_batch=True` and therefore trained on 256 frozen trials - memorisation,
not the task. Those numbers are RETRACTED and have been stripped rather than updated; the data is
quarantined in `data/trained_RNNs/RETRACTED_samebatch_NBitFlipFlop_ksweep/`. Nothing here has yet
been run against the corrected fresh-batch sweep, so this file currently states METHOD only, with no
results. Do not reintroduce a remembered figure into these docstrings.

"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, drift_alpha, series
from flipflop_ksweep import load

LAGS = (100, 1000, 10000)
COS_BAR = 0.05      # |cos| below this counts as isotropic jitter


def series(trace, key):
    """Extract one metric as (iterations, values) with NaN padding removed.

    Drift distances are logged only once per lag on the probe grid and NaN-padded elsewhere.

    Args:
        trace: a loaded trace; key: metric name.
    Returns:
        (iters, values) float arrays.
    """
    v = np.asarray(trace["metrics"][key], dtype=float)
    it = np.asarray(trace["iters"], dtype=float)[:len(v)]
    m = ~np.isnan(v)
    return it[m], v[m]


def main():
    """Test whether any flip-flop network reaches isotropic jitter by 300k iterations."""
    by, _ = load()
    ks = sorted({k for k, _ in by})
    Ns = sorted({N for _, N in by})

    print("Has the motion become diffusive/isotropic by 300k?")
    print("%3s %6s %14s %14s %14s %12s" % ("k", "N", "cos_W_rec end", "alpha end",
                                           "drift_lag1000", "dp_lag1000"))
    rows = {}
    for N in Ns:
        for k in ks:
            cs, als, dr, dp = [], [], [], []
            for t in by.get((k, N), []):
                _, c = series(t, "cos_W_rec")
                cs.append(np.mean(c[-2000:]))
                ia, a = drift_alpha(t)
                if a.size:
                    als.append(np.mean(a[-20:]))
                _, d = series(t, "drift_W_rec_lag1000")
                dr.append(d[-1])
                _, p = series(t, "dp_lag1000")
                dp.append(p[-1])
            if not cs:
                continue
            rows[(k, N)] = (np.mean(cs), np.mean(als) if als else np.nan, np.mean(dr), np.mean(dp))
            print("%3d %6d %14.3f %14.3f %14.4f %12.4f" % (k, N, *rows[(k, N)]))
        print()

    cos_all = np.array([v[0] for v in rows.values()])
    al_all = np.array([v[1] for v in rows.values()])
    print("VERDICT")
    print("  cos_W_rec at end:  min %.3f  max %.3f   (isotropic jitter needs |cos| < %.2f)"
          % (cos_all.min(), cos_all.max(), COS_BAR))
    print("  cells reaching isotropic jitter: %d / %d"
          % (int((np.abs(cos_all) < COS_BAR).sum()), len(cos_all)))
    print("  alpha at end:      min %.2f  max %.2f   (1.0 ballistic, 0.5 diffusive, <0.5 confined)"
          % (np.nanmin(al_all), np.nanmax(al_all)))
    print("  -> %s" % ("a drift criterion is USABLE" if (np.abs(cos_all) < COS_BAR).all()
                       else "NO cell reaches isotropic jitter; a drift stopping criterion is NOT "
                            "usable, same as on CDDM"))

    fig, ax = plt.subplots(2, 3, figsize=(19, 9.5))
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(ks)))
    Nshow = Ns[-1]
    for k in ks:
        for t in by.get((k, Nshow), []):
            it, c = series(t, "cos_W_rec")
            w = 501
            ax[0][0].plot(it[w - 1:], np.convolve(c, np.ones(w) / w, mode="valid"),
                          color=cols[ks.index(k)], lw=1.3, alpha=.85)
            ia, a = drift_alpha(t)
            ax[0][1].plot(ia, a, color=cols[ks.index(k)], lw=1.3, alpha=.85)
            it2, d = series(t, "drift_W_rec_lag1000")
            ax[0][2].plot(it2, d, color=cols[ks.index(k)], lw=1.3, alpha=.85)
            it3, p = series(t, "dp_lag1000")
            ax[1][0].plot(it3, p, color=cols[ks.index(k)], lw=1.3, alpha=.85)
        ax[0][0].plot([], [], color=cols[ks.index(k)], lw=2, label=f"k={k}")

    ax[0][0].axhline(0, color="k", lw=1)
    ax[0][0].axhspan(-COS_BAR, COS_BAR, color="grey", alpha=.25, label="isotropic jitter")
    ax[0][0].set(xscale="log", xlabel="iteration", ylabel=r"$\cos$ between updates",
                 title=f"(a) directional persistence, $W_{{rec}}$, N={Nshow}\n"
                       "0 = jitter; above 0 = still marching")
    ax[0][1].axhline(0.5, color="k", ls="--", lw=1, label="diffusive")
    ax[0][1].axhline(1.0, color="grey", ls=":", lw=1, label="ballistic")
    ax[0][1].set(xscale="log", xlabel="iteration", ylabel=r"$\alpha$",
                 title=f"(b) lag-scaling exponent, N={Nshow}")
    ax[0][2].set(xscale="log", yscale="log", xlabel="iteration",
                 ylabel=r"$\|W(t)-W(t-10^3)\|_F/\|W\|_F$",
                 title=f"(c) relative weight change, N={Nshow}")
    ax[1][0].set(xscale="log", yscale="log", xlabel="iteration",
                 ylabel=r"$\|p(t)-p(t-10^3)\|/\|p\|$",
                 title=f"(d) participation drift, N={Nshow}")

    for idx, (col, lab) in enumerate([(0, r"$\cos$ at 300k"), (1, r"$\alpha$ at 300k")]):
        a = ax[1][1 + idx]
        for N in Ns:
            xs = [k for k in ks if (k, N) in rows]
            a.plot(xs, [rows[(k, N)][col] for k in xs], "o-", lw=2, ms=8, label=f"N={N}")
        if col == 0:
            a.axhspan(-COS_BAR, COS_BAR, color="grey", alpha=.25)
            a.axhline(0, color="k", lw=1)
        else:
            a.axhline(0.5, color="k", ls="--", lw=1)
        a.set(xlabel="task complexity $k$", ylabel=lab, xticks=ks,
              title=f"(e{idx}) endpoint {lab} vs $k$")

    for a in ax.ravel():
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
        a.grid(alpha=.3)
    fig.suptitle("Flip-flop: is training still making systematic progress at 300k?\n"
                 "a drift-based stopping criterion needs cos -> 0; it failed on CDDM and is "
                 "re-tested here because the metrics were already logged", fontsize=12)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "flipflop_drift.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
