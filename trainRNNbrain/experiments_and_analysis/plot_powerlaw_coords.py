#!/usr/bin/env python3
"""
Plot the training loss in coordinates where a power-law approach becomes a straight line.

The model is L(t) = L_inf + A t^(-gamma). The obvious linearising coordinate is log(L - L_inf)
against log t, and it does give a straight line — but it needs L_inf, which is a FITTED parameter
that this project has shown is not well determined (its estimate is still rising with the length of
the fit window). Subtracting a parameter chosen to make the power law fit, and then displaying the
resulting straightness as evidence for the power law, is circular.

There is a coordinate that avoids this entirely. Differentiating with respect to log t kills the
constant:

    -dL/d(log t) = A*gamma*t^(-gamma)      =>      log(-dL/dlog t) = log(A*gamma) - gamma * log t

So plotting -dL/d(log t) against t on log-log axes is a straight line if and only if the loss follows
a power-law approach to SOME asymptote, and its slope is -gamma directly. No L_inf is used, assumed,
or fitted. This is the honest test of the functional form, and the gamma it returns is an independent
check on the gamma from the three-parameter fit.

Panels:
  (a) raw loss on log-log — visibly curved, because of the floor; this is what motivates the rest
  (b) -dL/d(log t) on log-log — the L_inf-free linearisation; slope = -gamma
  (c) L - L_inf on log-log — the conventional linearisation, shown for comparison and labelled as
      partly circular

Output: img/internal_figures/powerlaw_coords.png

Usage:  python plot_powerlaw_coords.py [SWEEP_FOLDER]
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_loss_fit import load_losses, fit_loss, logbin, IMG_DIR, FIT_STARTS


def log_derivative(t, y):
    """Local derivative -dy/d(log t), on the grid of t.

    Args:
        t, y: equal-length arrays, t strictly increasing and positive.
    Returns:
        array of -dy/dlog(t); positive while y is decreasing.
    """
    return -np.gradient(y, np.log(t))


def doubling_difference(L, ts, frac=0.02):
    """L(t/2) - L(t): the loss removed over the last doubling of the budget.

    This is the same information as -dL/dlog t but measured across a whole factor of two rather than
    between adjacent points, so it is far less sensitive to the noise in the loss trace. For a
    power-law approach it scales as A*t^(-gamma)*(2^gamma - 1), i.e. still exactly t^(-gamma), and
    still involves no L_inf.

    Args:
        L: loss array indexed from iteration 1; ts: iterations at which to evaluate;
        frac: half-width of the averaging window, as a fraction of t.
    Returns:
        (ts_used, D) arrays, restricted to positive differences that fit inside the trace.
    """
    out_t, out_d = [], []
    for t in ts:
        if t > len(L):
            continue
        w = max(int(frac * t), 50)
        lo = L[max(int(t / 2) - w, 0):int(t / 2) + w].mean()
        hi = L[max(int(t) - w, 0):int(t)].mean()
        if lo - hi > 0:
            out_t.append(float(t))
            out_d.append(float(lo - hi))
    return np.array(out_t), np.array(out_d)


def gamma_from_doubling(L, t_min, frac=0.02):
    """Estimate gamma as minus the log-log slope of the doubling difference. No L_inf is used.

    Args:
        L: loss array; t_min: smallest iteration to include; frac: averaging half-width fraction.
    Returns:
        (gamma, t_used, D_used).
    """
    ts = np.unique(np.round(np.logspace(np.log10(2 * t_min), np.log10(len(L)), 30)).astype(int))
    td, dd = doubling_difference(L, ts, frac)
    if len(td) < 4:
        return float("nan"), td, dd
    return -np.polyfit(np.log(td), np.log(dd), 1)[0], td, dd


def main():
    """Draw the three coordinate systems and compare gamma estimated with and without L_inf."""
    sweep = ([a for a in sys.argv[1:] if not a.startswith("--")] or
             ["data/trained_RNNs/CDDM_std_g0_drift"])[0]
    by = load_losses(sweep)
    Ns = sorted(by)
    cols = plt.cm.plasma(np.linspace(0.1, 0.72, len(Ns)))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    rows = []

    for k, N in enumerate(Ns):
        for j, (tag, L) in enumerate(by[N]):
            t = np.arange(1, len(L) + 1, dtype=float)
            m = t >= FIT_STARTS[0]
            tb, yb = logbin(t[m], L[m], nbins=45)
            lbl = f"N={N}" if j == 0 else None

            ax[0].plot(tb, yb, "-", color=cols[k], lw=1.4, alpha=.85, label=lbl)

            g_d, td, dd = gamma_from_doubling(L, FIT_STARTS[0])
            ax[1].plot(td, dd, "o", color=cols[k], ms=4, alpha=.75, label=lbl)
            if np.isfinite(g_d):
                c = np.polyfit(np.log(td), np.log(dd), 1)[1]
                ax[1].plot(td, np.exp(c) * td ** (-g_d), "-", color=cols[k], lw=1.8)

            Li, A, g_f = fit_loss(L, FIT_STARTS[0])
            ax[2].plot(tb, np.maximum(yb - Li, 1e-9), "-", color=cols[k], lw=1.4, alpha=.85,
                       label=lbl)
            rows.append((N, tag, g_d, g_f))

    ax[0].set(xscale="log", yscale="log", xlabel="iteration", ylabel="$L(t)$")
    ax[0].set_title("(a) raw loss, log-log\nCURVED — the floor is not a power law")
    ax[1].set(xscale="log", yscale="log", xlabel="iteration",
              ylabel=r"$L(t/2)-L(t)$   (loss removed per doubling)")
    ax[1].set_title(r"(b) $L_\infty$-FREE linearisation"
                    "\nstraight = power law; slope $=-\\gamma$")
    ax[2].set(xscale="log", yscale="log", xlabel="iteration", ylabel=r"$L(t)-L_\infty$")
    ax[2].set_title("(c) conventional linearisation\nstraight partly BY CONSTRUCTION")
    for a in ax:
        a.legend(fontsize=9)
        a.grid(alpha=.25)
    fig.suptitle(r"Loss in power-law coordinates. $L=L_\infty+At^{-\gamma}$ "
                 r"$\Rightarrow$ $-\mathrm{d}L/\mathrm{d}\log t = A\gamma\,t^{-\gamma}$, "
                 "which needs no asymptote", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    out = os.path.join(IMG_DIR, "powerlaw_coords.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}\n")

    print("gamma: from the L_inf-FREE doubling difference vs from the 3-parameter fit")
    print("%6s %11s %14s %14s %9s" % ("N", "seed", "gamma (doubling)", "gamma (fit)", "diff"))
    for N, tag, gd, gf in rows:
        print("%6d %11s %14.3f %14.3f %9.3f" % (N, tag, gd, gf, gd - gf))
    for N in Ns:
        r = [x for x in rows if x[0] == N]
        print("  N=%5d  mean gamma: doubling %.3f +- %.3f   fit %.3f +- %.3f"
              % (N, np.mean([x[2] for x in r]), np.std([x[2] for x in r]),
                 np.mean([x[3] for x in r]), np.std([x[3] for x in r])))


if __name__ == "__main__":
    main()
