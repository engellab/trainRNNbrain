#!/usr/bin/env python3
"""
Decay-fit parameters against network size, with matched fit ranges and an extrapolation test.

TWO METHODOLOGICAL POINTS THIS EXISTS TO ENFORCE.

1. MATCHED FIT RANGE. gamma drifts upward with the length of the window it is fitted on (one seed
   gave 0.35 / 0.47 / 0.50 / 0.53 for t_max = 25k / 50k / 100k / 200k). N=2000 ran to 300k while the
   others ran to 200k, so comparing their raw fitted parameters compares fit ranges, not sizes.
   Everything here is fitted on t in [2000, T_MATCH] for every size.

2. IS THE STRETCHED EXPONENTIAL ONLY WINNING BECAUSE IT HAS FOUR PARAMETERS? AICc already charges for
   the extra one, but the decisive test is out-of-sample: fit both families on the FIRST part of the
   run and score them on the part they never saw. A family that wins only by flexibility overfits and
   loses there; one that wins by having the right shape keeps winning.

Output: img/internal_figures/fit_params_vs_N.png

Usage:  python fit_params_vs_N.py
"""

import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_losses, IMG_DIR
from plot_loss_curves import logbin_stats, f_pow_floor, f_stretch1

FIT_FROM = 2000
T_MATCH = 200000          # every size fitted on the same window
T_SPLIT = 50000           # extrapolation test: fit below this, score above it


def fit(fn, t, y, p0, bounds):
    """Least-squares fit of fn to log(y). Returns params or None."""
    try:
        p, _ = curve_fit(lambda x, *q: np.log(np.clip(fn(x, *q), 1e-300, None)),
                         t, np.log(y), p0=p0, bounds=bounds, maxfev=200000)
        return p
    except Exception:
        return None


P0 = {"power": (lambda y: [y.min() * .95, 1.0, .5], ([0, 1e-9, .01], [1, 1e6, 3])),
      "stretched": (lambda y: [y.min() * .95, .3, .2], ([0, 1e-9, .02], [1, 1e3, 1.5]))}
FN = {"power": f_pow_floor, "stretched": f_stretch1}


def main():
    """Fit both families per seed at matched range, test extrapolation, plot parameters vs N."""
    by = load_losses("data/trained_RNNs/CDDM_std_g0_drift")
    Ns = sorted(by)
    res = {k: {N: [] for N in Ns} for k in FN}
    extrap = {k: {N: [] for N in Ns} for k in FN}

    for N in Ns:
        for _, L in by[N]:
            c, m, _, _ = logbin_stats(L)
            sel = (c >= FIT_FROM) & (c <= T_MATCH)
            for k in FN:
                p = fit(FN[k], c[sel], m[sel], P0[k][0](m[sel]), P0[k][1])
                if p is not None:
                    res[k][N].append(p)
            # extrapolation: fit early, score late
            e = (c >= FIT_FROM) & (c <= T_SPLIT)
            l = (c > T_SPLIT) & (c <= T_MATCH)
            if e.sum() > 8 and l.sum() > 8:
                for k in FN:
                    p = fit(FN[k], c[e], m[e], P0[k][0](m[e]), P0[k][1])
                    if p is not None:
                        r = np.log(m[l]) - np.log(np.clip(FN[k](c[l], *p), 1e-300, None))
                        extrap[k][N].append(float(np.sqrt(np.mean(r ** 2))))

    print(f"Fitted on t in [{FIT_FROM}, {T_MATCH}] for EVERY size (N=2000 truncated from 300k)\n")
    print("POWER LAW + FLOOR:  L = L_inf + A t^-gamma")
    print("%7s %20s %18s %18s %4s" % ("N", "L_inf", "A", "gamma", "n"))
    for N in Ns:
        a = np.array(res["power"][N])
        print("%7d %11.5f+-%.5f %10.3f+-%.3f %10.3f+-%.3f %4d"
              % (N, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std(),
                 a[:, 2].mean(), a[:, 2].std(), len(a)))
    print("\nSTRETCHED EXPONENTIAL, tau FIXED AT 1:  L = L_inf + A exp(-t^beta)")
    print("%7s %20s %18s %18s %4s" % ("N", "L_inf", "A", "beta", "n"))
    for N in Ns:
        a = np.array(res["stretched"][N])
        print("%7d %11.5f+-%.5f %10.3f+-%.3f %10.4f+-%.4f %4d"
              % (N, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std(),
                 a[:, 2].mean(), a[:, 2].std(), len(a)))

    print(f"\nEXTRAPOLATION TEST: fit on t<={T_SPLIT}, score RMS log-error on {T_SPLIT}<t<={T_MATCH}")
    print("%7s %18s %18s %12s" % ("N", "power law", "stretched", "winner"))
    for N in Ns:
        pw, st = np.array(extrap["power"][N]), np.array(extrap["stretched"][N])
        if not len(pw) or not len(st):
            continue
        print("%7d %9.5f+-%.5f %9.5f+-%.5f %12s"
              % (N, pw.mean(), pw.std(), st.mean(), st.std(),
                 "stretched" if st.mean() < pw.mean() else "power law"))

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    SPECS = [
        ("stretched", 0, r"$L_\infty$", "C4", "s", r"(a) STRETCHED: floor $L_\infty$"),
        ("stretched", 1, r"$A$", "C4", "s", r"(b) STRETCHED: amplitude $A$"),
        ("stretched", 2, r"$\beta$", "C4", "s", r"(c) STRETCHED: exponent $\beta$"),
        ("power", 0, r"$L_\infty$", "C3", "o", r"(d) POWER LAW: floor $L_\infty$"),
        ("power", 1, r"$A$", "C3", "o", r"(e) POWER LAW: amplitude $A$"),
        ("power", 2, r"$\gamma$", "C3", "o", r"(f) POWER LAW: exponent $\gamma$"),
    ]
    for k, (fam, idx, ylab, col, mk, ttl) in enumerate(SPECS):
        a = ax[k // 3, k % 3]
        mu = [np.mean([p[idx] for p in res[fam][N]]) for N in Ns]
        sd = [np.std([p[idx] for p in res[fam][N]]) for N in Ns]
        a.errorbar(Ns, mu, yerr=sd, fmt=mk + "-", color=col, ms=8, capsize=4, lw=2)
        for N in Ns:
            a.plot([N] * len(res[fam][N]), [p[idx] for p in res[fam][N]], mk, color=col,
                   ms=4, alpha=.35)
        # a relative spread annotation, so "flat" is quantified rather than eyeballed
        rng = 100 * (max(mu) - min(mu)) / np.mean(mu)
        a.set(xscale="log", xlabel="$N$", ylabel=ylab, title=f"{ttl}\nspread {rng:.1f}% over 20x in $N$")
        a.grid(alpha=.3)
    fig.suptitle(r"Decay-fit parameters vs network size.   "
                 r"TOP $L=L_\infty+A\,e^{-t^{\beta}}$ ($\tau$ fixed at 1, unidentifiable)   "
                 r"|   BOTTOM $L=L_\infty+A\,t^{-\gamma}$" "\n"
                 f"every size fitted on t $\\in$ [{FIT_FROM}, {T_MATCH}]; "
                 "faint points = individual seeds (N=2000 has n=1)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(IMG_DIR, "fit_params_vs_N.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
