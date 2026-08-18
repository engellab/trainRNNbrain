#!/usr/bin/env python3
"""
Model comparison for the shape of the training-loss decay: power law vs exponential vs stretched.

The three candidates make physically different claims:

  power law      L - L_inf = A t^(-gamma)            no characteristic timescale; scale-free decay,
                                                     the behaviour of a broad spectrum of curvatures
                                                     all still relaxing
  exponential    L - L_inf = A exp(-t/tau)           ONE dominant timescale tau; convergence is
                                                     predictable and effectively complete after a few
                                                     tau, which is what a single slowest mode gives
  stretched      L - L_inf = A exp(-(t/tau)^beta)    a broad DISTRIBUTION of timescales; interpolates
                                                     between the two, beta=1 recovers the exponential

Two independent fits are run, because each has a different weakness:

  (A) fit L(t) directly, with L_inf free. Weakness: L_inf is poorly determined here, so a model can
      win by exploiting that freedom rather than by describing the shape.
  (B) fit D(t) = L(t/2) - L(t), which contains NO L_inf at all. Weakness: differencing amplifies
      noise. Under each model D(t) is
        power law    D = A' t^(-gamma)
        exponential  D = A [exp(-t/(2 tau)) - exp(-t/tau)]
        stretched    D = A [exp(-(t/(2 tau))^beta) - exp(-(t/tau)^beta)]

A model that wins BOTH ways is believed; a model that wins only in (A) is suspected of exploiting
L_inf.

CRITERION, FIXED BEFORE RUNNING: models are compared by AIC computed on log-residuals (the noise in
the loss is multiplicative), with the same data points for every model. Delta-AIC > 10 = decisive,
4-10 = substantial, < 4 = inconclusive. A model is declared the winner only if it wins by > 10 in
BOTH fits and in a majority of seeds.

PREDICTION, STATED BEFORE RUNNING: the pure exponential will lose badly, because it has a single
timescale and cannot cover 1.7 decades; the pure power law will lose at late times because the
measured local exponent is not constant; the stretched exponential will win, with beta < 1.

Output: img/internal_figures/decay_model_comparison.png

Usage:  python compare_decay_models.py [SWEEP_FOLDER]
"""

import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_loss_fit import load_losses, logbin, IMG_DIR
from plot_local_exponent import doubling_curve

T_MIN = 4000


def aic(logresid, k):
    """AIC for a least-squares fit on log-residuals, assuming Gaussian errors in log space.

    Args:
        logresid: array of (log observed - log predicted); k: number of fitted parameters.
    Returns:
        AIC value; lower is better.
    """
    n = len(logresid)
    rss = float(np.sum(logresid ** 2))
    return n * np.log(rss / n) + 2 * k


# ---- models for L(t), with L_inf free (3 or 4 parameters) --------------------------------------
def L_power(t, Li, A, g):
    """Power-law approach: L_inf + A t^(-gamma)."""
    return Li + A * t ** (-g)


def L_exp(t, Li, A, tau):
    """Single-timescale exponential approach: L_inf + A exp(-t/tau)."""
    return Li + A * np.exp(-t / tau)


def L_stretch(t, Li, A, tau, b):
    """Stretched exponential approach: L_inf + A exp(-(t/tau)^beta)."""
    return Li + A * np.exp(-np.power(np.clip(t / tau, 1e-12, None), b))


# ---- models for D(t) = L(t/2) - L(t), which contain no L_inf (2 or 3 parameters) ---------------
def D_power(t, A, g):
    """Doubling difference under a power law: proportional to t^(-gamma)."""
    return A * t ** (-g)


def D_exp(t, A, tau):
    """Doubling difference under a single exponential."""
    return A * (np.exp(-t / (2 * tau)) - np.exp(-t / tau))


def D_stretch(t, A, tau, b):
    """Doubling difference under a stretched exponential."""
    z = np.clip(t / tau, 1e-12, None)
    return A * (np.exp(-np.power(z / 2, b)) - np.exp(-np.power(z, b)))


def fit_log(fn, x, y, p0, bounds):
    """Least-squares fit of fn to y in LOG space, returning params, AIC and log-residuals.

    Args:
        fn: model callable fn(x, *params); x, y: data (y > 0);
        p0: initial parameters; bounds: (lower, upper) sequences for curve_fit.
    Returns:
        (params, aic_value, logresid) or (None, inf, None) if the fit fails.
    """
    try:
        p, _ = curve_fit(lambda xx, *pp: np.log(np.clip(fn(xx, *pp), 1e-300, None)),
                         x, np.log(y), p0=p0, bounds=bounds, maxfev=80000)
        r = np.log(y) - np.log(np.clip(fn(x, *p), 1e-300, None))
        return p, aic(r, len(p)), r
    except Exception:
        return None, float("inf"), None


def main():
    """Run both model comparisons on every seed and report AIC differences."""
    sweep = ([a for a in sys.argv[1:] if not a.startswith("--")] or
             ["data/trained_RNNs/CDDM_std_g0_drift"])[0]
    by = load_losses(sweep)
    Ns = sorted(by)
    rowsA, rowsD = [], []
    fig, ax = plt.subplots(2, len(Ns), figsize=(5.6 * len(Ns), 9), squeeze=False)
    mcol = {"power": "#1f77b4", "exponential": "#d62728", "stretched": "#2ca02c"}

    for k, N in enumerate(Ns):
        for j, (tag, L) in enumerate(by[N]):
            t = np.arange(1, len(L) + 1, dtype=float)
            m = t >= T_MIN
            tb, yb = logbin(t[m], L[m], nbins=50)

            # ---- (A) direct fit of L(t), L_inf free
            lo, hi = yb.min() * 0.5, yb.min() * 1.02
            fits = {
                "power": fit_log(L_power, tb, yb, [yb.min() * 0.95, 1.0, 0.5],
                                 ([lo, 1e-6, 0.01], [hi, 1e6, 3.0])),
                "exponential": fit_log(L_exp, tb, yb, [yb.min() * 0.95, 0.01, 3e4],
                                       ([lo, 1e-9, 1e2], [hi, 1e3, 1e8])),
                "stretched": fit_log(L_stretch, tb, yb, [yb.min() * 0.95, 0.05, 1e4, 0.4],
                                     ([lo, 1e-9, 1e-2, 0.02], [hi, 1e3, 1e9, 1.5])),
            }
            best = min(fits, key=lambda z: fits[z][1])
            rowsA.append((N, tag, {z: fits[z][1] for z in fits}, best,
                          fits["stretched"][0][3] if fits["stretched"][0] is not None else np.nan))

            # ---- (B) fit of D(t), no L_inf
            td, dd = doubling_curve(L)
            fitsD = {
                "power": fit_log(D_power, td, dd, [1.0, 0.5], ([1e-12, 0.01], [1e8, 3.0])),
                "exponential": fit_log(D_exp, td, dd, [0.01, 3e4], ([1e-9, 1e2], [1e3, 1e8])),
                "stretched": fit_log(D_stretch, td, dd, [0.05, 1e4, 0.4],
                                     ([1e-9, 1e-2, 0.02], [1e3, 1e9, 1.5])),
            }
            bestD = min(fitsD, key=lambda z: fitsD[z][1])
            rowsD.append((N, tag, {z: fitsD[z][1] for z in fitsD}, bestD,
                          fitsD["stretched"][0][2] if fitsD["stretched"][0] is not None else np.nan))

            if j == 0:
                ax[0][k].plot(tb, yb, "o", color="k", ms=3, label="data")
                for z in fits:
                    if fits[z][0] is not None:
                        ax[0][k].plot(tb, globals()["L_" + ("power" if z == "power" else
                                                            "exp" if z == "exponential" else
                                                            "stretch")](tb, *fits[z][0]),
                                      "-", color=mcol[z], lw=1.8, label=z)
                ax[1][k].plot(td, dd, "o", color="k", ms=3, label="data")
                for z in fitsD:
                    if fitsD[z][0] is not None:
                        ax[1][k].plot(td, globals()["D_" + ("power" if z == "power" else
                                                            "exp" if z == "exponential" else
                                                            "stretch")](td, *fitsD[z][0]),
                                      "-", color=mcol[z], lw=1.8, label=z)
        ax[0][k].set(xscale="log", xlabel="iteration", ylabel="$L(t)$",
                     title=f"(A) N={N}: fit $L(t)$, $L_\\infty$ free")
        ax[1][k].set(xscale="log", yscale="log", xlabel="iteration",
                     ylabel="$L(t/2)-L(t)$",
                     title=f"(B) N={N}: fit $D(t)$, no $L_\\infty$")
        for r in (0, 1):
            ax[r][k].legend(fontsize=8)
            ax[r][k].grid(alpha=.25)
    fig.suptitle("Power law vs exponential vs stretched exponential (seed 1 shown; AIC over all seeds)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(IMG_DIR, "decay_model_comparison.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}\n")

    for name, rows in (("(A) fit L(t), L_inf free", rowsA), ("(B) fit D(t), no L_inf", rowsD)):
        print(f"=== {name} ===   AIC (lower better); dAIC = AIC(model) - AIC(best)")
        print("%6s %11s %12s %12s %12s   %-12s %s"
              % ("N", "seed", "power", "exponential", "stretched", "best", "extra"))
        for N, tag, a, best, extra in rows:
            b = min(a.values())
            print("%6d %11s %12.1f %12.1f %12.1f   %-12s %s"
                  % (N, tag, a["power"] - b, a["exponential"] - b, a["stretched"] - b, best,
                     ("beta=%.2f" % extra) if np.isfinite(extra) else ""))
        wins = {}
        for _, _, _, best, _ in rows:
            wins[best] = wins.get(best, 0) + 1
        print("   wins:", wins, "\n")


if __name__ == "__main__":
    main()
