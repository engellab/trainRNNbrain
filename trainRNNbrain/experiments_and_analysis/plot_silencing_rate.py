#!/usr/bin/env python3
"""
Does the silencing rate decay to zero, or to a nonzero plateau?

Plots r(t) = new silent units per 1000 iterations (trailing window), mean with 95% t-interval
across networks, on log-log axes, with two candidate fits to the post-peak decay:
    power law   r = exp(a) * t^b        -> reaches zero, but only asymptotically
    plateau     r = c + A exp(-t/tau)   -> silencing never stops, at rate c

Distinguishing them matters: under the plateau model, standard weight decay keeps silencing units
forever; under the power law it eventually stops. The 30000-iteration data cannot separate them
cleanly, which is exactly why the 100000-iteration horizon runs were needed.

Usage:  python plot_silencing_rate.py [drift_curves.npz] [out.png]
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

PAL = {"N100": "#0072B2", "N500": "#009E73", "N1000": "#E69F00"}
T95 = {5: 2.776}


def smooth(x, k=51):
    """Moving average with the ends dropped (never padded — padding fakes a decay to zero)."""
    return np.convolve(x, np.ones(k) / k, "valid")


def main():
    npz = sys.argv[1] if len(sys.argv) > 1 else "data/trained_RNNs/drift_curves.npz"
    out = sys.argv[2] if len(sys.argv) > 2 else "img/internal_figures/silencing_rate_decay.png"
    d = np.load(npz)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    for ax, key in zip(axes, ("N100", "N500", "N1000")):
        c = d[key]
        it, R = c[0, 0], c[:, 1, :]
        m, h = R.mean(0), T95[5] * R.std(0, ddof=1) / np.sqrt(R.shape[0])
        k = 51
        sm, sh = smooth(m, k), smooth(h, k)
        iv = it[k // 2: k // 2 + len(sm)]

        ax.fill_between(iv, sm - sh, sm + sh, color=PAL[key], alpha=0.18, linewidth=0)
        ax.plot(iv, sm, color=PAL[key], lw=1.6, label="measured")
        ax.axhline(0, color="0.4", lw=0.8, ls="--")

        ipk = int(np.argmax(sm))
        sel = (iv > iv[ipk]) & (sm > 0)
        if sel.sum() > 20:
            b, a = np.polyfit(np.log(iv[sel]), np.log(sm[sel]), 1)
            tt = np.linspace(iv[ipk], 1e5, 300)
            ax.plot(tt, np.exp(a) * tt ** b, color="0.35", lw=1.2, ls="--",
                    label=f"power law  b={b:.2f}")
            try:
                (cc, A, tau), _ = curve_fit(lambda t, c0, A0, t0: c0 + A0 * np.exp(-t / t0),
                                            iv[sel], sm[sel],
                                            p0=[max(sm[-1], 0.1), sm[ipk], 1e4], maxfev=20000)
                if cc > 0:      # a negative plateau is unphysical; do not draw a nonsense fit
                    ax.plot(tt, cc + A * np.exp(-tt / tau), color="0.55", lw=1.2, ls=":",
                            label=f"plateau  c={cc:.2f}")
            except Exception:
                pass
        ax.set_xscale("log")
        ax.set_xlim(1e3, 1e5)
        ax.set_xlabel("training iteration")
        ax.set_title(key.replace("N", "N = "), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="0.92", lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7.5)
    axes[0].set_ylabel("new silent units per 1000 iterations")
    fig.suptitle("Does silencing stop? Rate of new silent units, with 95% CI "
                 "(weight_decay = 1e-6, dashed region beyond 30000 is extrapolation)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
