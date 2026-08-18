#!/usr/bin/env python3
"""
Silent-unit fraction at the moment each network STABLY reaches a common loss threshold.

This is the matched-performance comparison in its simplest form: every size is read at its own
iteration T_N, defined as the last time its smoothed loss is above L*, so all networks are equally
good at the task when M is counted. It is licensed by the measured result that the loss floor is
common to within ~1% across a 20-fold size range (and what difference exists runs the safe way -
larger networks sit slightly HIGHER, so they are not secretly under-trained at equal loss).

"STABLY" is doing real work. The per-iteration loss comes from a NOISY forward pass, and a single
favourable draw dips below any threshold long before the network is actually there - at L=0.025 the
raw trace first touches the line ~7x earlier than the smoothed one. T_N is therefore taken from a
centred running mean over VALID windows only; zero-padded edges otherwise manufacture a crossing at
iteration 1.

Both silent-unit criteria are reported because they disagree by an order of magnitude early in
training and converge only once a network is heavily silenced:
    hard        p_i < 1e-6            a unit that has genuinely stopped firing
    scale-free  p_i < 0.05*q95(p)     negligible relative to the network's own scale

Output: img/internal_figures/silent_at_threshold.png

Usage:  python silent_at_threshold.py [L* ...]        (default: 0.023 0.025)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_drift_curves import load_traces, IMG_DIR
from plot_M_vs_N import active_count

WINDOW = 2001


def stable_crossing(L, thr, window=WINDOW):
    """Last iteration at which the smoothed loss is still above `thr`; None if never below.

    Args:
        L: per-iteration loss array; thr: threshold; window: centred running-mean width (odd).
    Returns:
        iteration index (1-based) after which the smoothed loss stays below thr, or None.
    """
    h = window // 2
    s = np.convolve(L, np.ones(window) / window, mode="valid")
    it = np.arange(h + 1, len(L) - h + 1)
    above = it[s > thr]
    if not len(above) or above[-1] >= it[-1]:
        return None
    return int(above[-1])


def main():
    """Report and plot the silent fraction at each size's stable crossing of each threshold."""
    thrs = [float(a) for a in sys.argv[1:] if not a.startswith("--")] or [0.023, 0.025]
    by = load_traces("data/trained_RNNs/CDDM_std_g0_drift")
    Ns = sorted(by)
    out = {}

    for thr in thrs:
        print(f"\n=== L* = {thr:.4f} (stable crossing) ===")
        print("%7s %16s %22s %22s" % ("N", "T_N", "silent, hard (%)", "silent, scale-free (%)"))
        for N in Ns:
            T, h, s = [], [], []
            for t in by[N]:
                x = stable_crossing(t["loss"], thr)
                P = np.array(t["participation"])
                I = np.array(t["participation_iters"])
                if x is None or x > I[-1]:
                    continue
                p = P[np.argmin(np.abs(I - x))]
                T.append(x)
                h.append(100 * (N - active_count(p, "hard")) / N)
                s.append(100 * (N - active_count(p, "scalefree")) / N)
            if not T:
                continue
            out[(thr, N)] = (np.mean(T), np.mean(h), np.std(h), np.mean(s), np.std(s), len(T))
            print("%7d %8.0f +- %-5.0f %11.1f +- %-8.1f %11.1f +- %-8.1f"
                  % (N, np.mean(T), np.std(T), np.mean(h), np.std(h), np.mean(s), np.std(s)))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    cols = plt.cm.viridis(np.linspace(0.15, 0.75, len(thrs)))
    for col, (crit, idx, ttl) in enumerate([("hard", 1, r"$p_i<10^{-6}$"),
                                            ("scalefree", 3, r"$p_i<0.05\,q_{95}(p)$")]):
        for i, thr in enumerate(thrs):
            xs = [N for N in Ns if (thr, N) in out]
            mu = [out[(thr, N)][idx] for N in xs]
            sd = [out[(thr, N)][idx + 1] for N in xs]
            ax[col].errorbar(xs, mu, yerr=sd, fmt="o-", color=cols[i], ms=8, capsize=4, lw=2,
                             label=f"$L^*$={thr:.4f}  ($T_N$={out[(thr, xs[0])][0]:.0f}"
                                   f"–{out[(thr, xs[-1])][0]:.0f})")
        ax[col].set(xscale="log", xlabel="$N$", ylabel="silent units (% of N)", ylim=(0, 100),
                    title=f"({'ab'[col]}) {ttl}")
        ax[col].legend(fontsize=9)
        ax[col].grid(alpha=.3)
    fig.suptitle("Silent-unit fraction when each size STABLY reaches a common loss threshold\n"
                 "(matched performance; T_N from a centred 2001-iteration mean, not the raw trace)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    p = os.path.join(IMG_DIR, "silent_at_threshold.png")
    fig.savefig(p, dpi=150)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
