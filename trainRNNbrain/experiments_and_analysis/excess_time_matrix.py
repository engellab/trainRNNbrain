#!/usr/bin/env python3
"""
WHEN does each network reach 1.10x its own loss floor? The read-out time over the (N, k) grid.

Everything else measures what the networks look like AT that point. This measures the point itself:
how long a condition takes to get essentially all the way down to the best it can do. It is the
loss-based twin of the drift settling time, and it is worth its own figure because the read-out time
is an OUTPUT of the analysis, not a knob - the whole `excess` criterion rests on it behaving sensibly.

T is fitted as  T = A * N^beta * k^gamma:
    gamma > 0   harder tasks take longer, as convergence requires
    beta  < 0   LARGER networks converge FASTER - repeatedly measured on this task (the drift
                settling law gave beta = -0.16, and the stretched-exponential half-lives gave
                t_half = 6569 / 5182 / 4234 at N = 500 / 1000 / 2000 for k=2)

⚠️ EACH RUN'S FLOOR IS FITTED OVER ITS OWN BUDGET, not a common range. Range-matching is only a proxy
for "the floor is estimated correctly", and forcing every condition onto a common 150k is what made
frm's floor invalid, since frm needs ~400k to converge.

⚠️ THE CONDITIONS HAVE DIFFERENT BUDGETS (none 400-500k, rws 150k, frm/both 400k). That does not bias
T here - T is typically 15-40k, far inside every budget - but a cell whose T approached its budget
would be censored, so the fraction T/budget is printed as a guard.

Output: img/internal_figures/excess_time_matrix.png

Usage:  python excess_time_matrix.py [EXCESS_DELTA]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR
import plotstyle as ps
import pr_matrix as P


def fit_time_law(runs, pen):
    """Fit T = A N^beta k^gamma for one penalty, with bootstrap CIs."""
    K, NN, T = [], [], []
    for r in runs:
        if r["pen"] == pen and np.isfinite(r["T"]) and r["T"] > 0:
            K.append(r["k"]); NN.append(r["N"]); T.append(r["T"])
    if len(T) < 8 or len(set(NN)) < 2 or len(set(K)) < 3:
        return None
    K, NN, T = np.array(K, float), np.array(NN, float), np.array(T, float)
    y = np.log(T)
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(y), np.log(NN), np.log(K)]), y,
                               rcond=None)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(K[i])) < 3 or len(np.unique(NN[i])) < 2:
            continue
        bs.append(np.linalg.lstsq(np.column_stack([np.ones(len(y)), np.log(NN[i]), np.log(K[i])]),
                                  y[i], rcond=None)[0])
    bs = np.array(bs)
    q = lambda j: (float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5)))
    return dict(A=float(np.exp(beta[0])), beta=float(beta[1]), gamma=float(beta[2]),
                beta_ci=q(1), gamma_ci=q(2), n=len(T))


def cells_T(runs, pen, ks, Ns):
    """(mean, sd, n, mean T/budget) grids of the read-out time over (N, k)."""
    Z = np.full((len(Ns), len(ks)), np.nan)
    S = np.full((len(Ns), len(ks)), np.nan)
    C = np.zeros((len(Ns), len(ks)), dtype=int)
    F = np.full((len(Ns), len(ks)), np.nan)
    box, frac = {}, {}
    for r in runs:
        if r["pen"] != pen or not np.isfinite(r["T"]):
            continue
        box.setdefault((r["k"], r["N"]), []).append(r["T"])
        frac.setdefault((r["k"], r["N"]), []).append(r["T"] / r["budget"])
    for (k, N), v in box.items():
        if k in ks and N in Ns:
            i, j = Ns.index(N), ks.index(k)
            Z[i, j], S[i, j], C[i, j] = np.mean(v), np.std(v), len(v)
            F[i, j] = np.mean(frac[(k, N)])
    return Z, S, C, F


def main():
    """Compute the excess read-out time and plot it over the (N, k) grid, per penalty."""
    if len(sys.argv) > 1:
        P.EXCESS_DELTA = float(sys.argv[1])
    ps.setup()
    runs = P.load()
    for r in runs:
        r["floor"] = P.fit_floor(r["loss"], r["budget"])
        r["T"] = P.excess_time(r["loss"], r["floor"], P.EXCESS_DELTA)
    runs = [r for r in runs if np.isfinite(r["T"])]
    ks = sorted({r["k"] for r in runs})
    Ns = sorted({r["N"] for r in runs})
    d = P.EXCESS_DELTA

    print(f"T = iteration where the noise-free loss first reaches {1+d:.2f} x that run's OWN floor\n")
    print("fitted  T = A N^beta k^gamma")
    print("%-6s %5s %24s %26s %12s" % ("pen", "n", "beta (size)", "gamma (complexity)", "T(k8)/T(k1)"))
    laws = {}
    for pen in P.PENS:
        f = fit_time_law(runs, pen)
        laws[pen] = f
        if not f:
            n = sum(1 for r in runs if r["pen"] == pen)
            print("%-6s %5d   not fittable (needs k>=3 at >=2 sizes)" % (pen, n)); continue
        print("%-6s %5d   %+.3f [%+.3f, %+.3f]     %+.3f [%+.3f, %+.3f] %11.2fx"
              % (pen, f["n"], f["beta"], f["beta_ci"][0], f["beta_ci"][1],
                 f["gamma"], f["gamma_ci"][0], f["gamma_ci"][1], 8.0 ** f["gamma"]))

    print("\ncensoring guard — mean T / budget per condition (near 1.0 would mean T is cut off):")
    for pen in P.PENS:
        v = [r["T"] / r["budget"] for r in runs if r["pen"] == pen]
        if v:
            print("  %-6s %.3f   (max %.3f)" % (pen, np.mean(v), max(v)))

    # ---- figure --------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, len(P.PENS), figsize=(4.3 * len(P.PENS), 8.4), squeeze=False)
    allZ = [cells_T(runs, p, ks, Ns)[0] for p in P.PENS]
    finite = np.concatenate([z[np.isfinite(z)] for z in allZ if np.isfinite(z).any()])
    vmin, vmax = np.percentile(finite, 2), np.percentile(finite, 98)
    for c_i, pen in enumerate(P.PENS):
        Z, S, C, F = cells_T(runs, pen, ks, Ns)
        a = ax[0][c_i]
        if not np.isfinite(Z).any():
            a.text(.5, .5, f"no {pen} data yet", ha="center", va="center", transform=a.transAxes,
                   color="0.5", fontsize=11)
            a.set_xticks([]); a.set_yticks([])
        else:
            im = a.imshow(Z / 1000, cmap="viridis", vmin=vmin / 1000, vmax=vmax / 1000,
                          aspect="auto")
            for i in range(len(Ns)):
                for j in range(len(ks)):
                    if np.isfinite(Z[i, j]):
                        col = "white" if Z[i, j] < (vmin + vmax) / 2 else "black"
                        a.text(j, i, f"{Z[i, j]/1000:.0f}k", ha="center", va="bottom", fontsize=7.5,
                               color=col)
                        a.text(j, i, f"±{S[i, j]/1000:.0f}k", ha="center", va="top", fontsize=5.6,
                               color=col, alpha=.85)
                    else:
                        a.text(j, i, "·", ha="center", va="center", color="0.6", fontsize=9)
            a.set(xticks=range(len(ks)), xticklabels=ks, yticks=range(len(Ns)),
                  yticklabels=[str(n) for n in Ns])
            cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
            cb.set_label("read-out iteration (thousands)", fontsize=8)
        f = laws.get(pen)
        sub = (f"$T = {f['A']:.0f}\\,N^{{{f['beta']:+.2f}}}k^{{{f['gamma']:+.2f}}}$" if f
               else "law not fittable")
        a.set_title(f"{pen}\nwhen loss reaches {1+d:.2f}x its own floor\n{sub}",
                    fontsize=10.5, fontweight="bold")
        if c_i == 0:
            a.set_ylabel("N (units)")

        b = ax[1][c_i]
        if not np.isfinite(Z).any():
            b.text(.5, .5, "—", ha="center", va="center", transform=b.transAxes, color="0.6")
            b.set_xticks([]); b.set_yticks([]); continue
        for i, N in enumerate(Ns):
            if np.isfinite(Z[i]).any():
                ps.band(b, ks, Z[i] / 1000, S[i] / 1000, ps.col_n(N), label=f"N={N}")
        if f:
            kk = np.linspace(min(ks), max(ks), 100)
            for i, N in enumerate(Ns):
                if np.isfinite(Z[i]).any():
                    b.plot(kk, f["A"] * N ** f["beta"] * kk ** f["gamma"] / 1000, "--",
                           color=ps.col_n(N), lw=1.1, alpha=.75)
            txt = (f"$\\beta$={f['beta']:+.2f} [{f['beta_ci'][0]:+.2f},{f['beta_ci'][1]:+.2f}]\n"
                   f"$\\gamma$={f['gamma']:+.2f} [{f['gamma_ci'][0]:+.2f},{f['gamma_ci'][1]:+.2f}]\n"
                   f"$k$=1→8: {8.0**f['gamma']:.2f}x longer")
            b.text(.03, .96, txt, transform=b.transAxes, fontsize=6.8, va="top", ha="left",
                   bbox=dict(fc="white", ec="0.7", alpha=.85, boxstyle="round,pad=0.3"))
        b.set(xlabel="k (bits)", ylabel="read-out iteration (thousands)", xticks=ks,
              title="read-out time vs k\ndashed = fitted law")
        b.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"When does each network reach {1+d:.2f}x its own loss floor?\n"
                 "the read-out time behind every `excess` comparison  ·  "
                 r"fitted as $T = A\,N^{\beta}k^{\gamma}$", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return ps.save(fig, "excess_time_matrix", tight=False)


if __name__ == "__main__":
    main()
