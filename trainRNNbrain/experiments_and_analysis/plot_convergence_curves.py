#!/usr/bin/env python3
"""
Has training stopped changing the network? Two curves per network size, from the participation traces.

Left  : new silent units per 1000 iterations (trailing window). Zero = the silent population has
        stopped growing.
Right : relative change of the participation VECTOR over the same window,
        ||p(t) - p(t-1000)|| / ||p(t)||. Zero = unit activities have stopped moving at all — a
        stricter test, since the silent count can be flat while units still rearrange.

Bands are 95% CI across networks (t-interval, n=5).

Usage:  python plot_convergence_curves.py [drift_curves.npz] [out.png]
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

PAL = {"N100": "#0072B2", "N500": "#009E73", "N1000": "#E69F00"}   # CVD-validated, fixed order
T_CRIT = {5: 2.776, 4: 3.182, 3: 4.303}     # two-sided 95%, n-1 dof


def ci(a, axis=0):
    """Mean and half-width of the 95% t-interval across networks."""
    m, sd, n = a.mean(axis), a.std(axis, ddof=1), a.shape[axis]
    return m, T_CRIT.get(n, 2.776) * sd / np.sqrt(n)


def smooth(x, k=25):
    """Light moving average. mode='valid' so the ends are dropped rather than biased toward zero,
    which is what 'same' padding does and what would otherwise fake a late drop to convergence."""
    return np.convolve(x, np.ones(k) / k, mode="valid")


def main():
    npz = sys.argv[1] if len(sys.argv) > 1 else "data/trained_RNNs/drift_curves.npz"
    out = sys.argv[2] if len(sys.argv) > 2 else "img/internal_figures/convergence_curves.png"
    d = np.load(npz)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    for key in ("N100", "N500", "N1000"):
        c = d[key]                       # (nets, 5, T): iters, net_new, gross_new, gross_rec, rel_dp
        it = c[0, 0]
        for ax, row, lbl in ((axes[0], 1, "new silent units / 1000 iters"),
                             (axes[1], 4, r"$\|\Delta p\| / \|p\|$ per 1000 iters")):
            m, h = ci(c[:, row, :])
            m, h = smooth(m), smooth(h)
            k = len(it) - len(m)
            itv = it[k // 2: k // 2 + len(m)]
            ax.plot(itv, m, color=PAL[key], lw=1.6, label=key.replace("N", "N = "))
            ax.fill_between(itv, m - h, m + h, color=PAL[key], alpha=0.18, linewidth=0)
            ax.set_xlabel("training iteration")
            ax.set_ylabel(lbl)

    axes[0].axhline(0, color="0.4", lw=0.8, ls="--")
    axes[0].set_title("Silent population: still growing at 30000?", fontsize=10)
    axes[1].set_yscale("log")
    axes[1].set_title("Participation vector: still moving?", fontsize=10)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="0.92", lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Standard RNNs, no penalties — the loss is flat long before the network stops changing",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

    print(f"\n{'N':>6s} {'new silent/1000it @ end':>25s} {'|dp|/|p| @ end':>15s} {'decay exponent':>15s} "
          f"{'iters to reach 1%':>18s}")
    for key in ("N100", "N500", "N1000"):
        c = d[key]; it = c[0, 0]
        m_new, h_new = ci(c[:, 1, :]); m_rel, _ = ci(c[:, 4, :])
        # the relative change decays as a power law in iteration; fit the last two thirds
        sel = it > it[-1] / 3
        b, a = np.polyfit(np.log(it[sel]), np.log(m_rel[sel]), 1)
        reach = float(np.exp((np.log(0.01) - a) / b))
        print(f"{key[1:]:>6s} {m_new[-200:].mean():>12.2f} +/- {h_new[-200:].mean():<8.2f} "
              f"{m_rel[-200:].mean():>14.4f} {b:>15.2f} {reach:>18,.0f}")


if __name__ == "__main__":
    main()
