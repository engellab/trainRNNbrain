#!/usr/bin/env python3
"""
THE DECISIVE FIGURE. Matched on W_inp diffusion, active units depend on N and not on k.

WHY W_inp. Of the variables whose weight motion can be tracked, the INPUT weights keep moving
directionally the longest - they settle at ~46k-73k iterations where W_rec settles at ~19k-41k and
W_out freezes by ~5k. Matching networks at the point where the LAST thing still moving has stopped
moving is the most conservative dynamical read-out available: every other variable has already
settled, so nothing is being compared mid-flight.

This needs no threshold on the loss, no fitted floor, and no assumption that the task is converged.
It asks only: read each network once its updates have stopped carrying it anywhere new.

THE CLAIM, and how each panel tests it:

  (a) M vs N, coloured by k        if M depends only on N, every k lies on ONE line.
  (b) M vs k, one line per N       if M does not depend on k, every line is FLAT.
  (c) M / N^b vs k                 the decisive panel: divide out the size dependence, then look for
                                   any k trend in what is left. Flat = none.

The top row repeats all three at a FIXED COMPUTE read-out, where the same networks DO show a k
dependence - so the figure shows both the effect and its absence side by side, and the difference
between the rows is only WHEN the networks were read.

Output: img/internal_figures/flipflop_decisive.png

Usage:  python flipflop_decisive.py [ALPHA_THRESHOLD]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, diffusive_onset, drift_alpha
import plotstyle as ps
import flipflop_figures as F

DRIFT_VAR = "W_inp"      # the last variable still moving; see module docstring
ALPHA_TH = 0.60          # gamma and c are unchanged over 0.50-0.80; see flipflop_diffusion.py
N_BOOT = 2000


def fit_law(K, NN, M, n_boot=N_BOOT):
    """Fit M = A N^b k^c in log space with bootstrap CIs on b and c.

    Args:
        K, NN, M: per-run complexity, size and active-unit count; n_boot: resamples.
    Returns:
        dict with A, b, c and 95% CIs for b and c.
    """
    K, NN, M = np.asarray(K, float), np.asarray(NN, float), np.asarray(M, float)
    y = np.log(M)
    X = np.column_stack([np.ones_like(y), np.log(NN), np.log(K)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(K[i])) < 3 or len(np.unique(NN[i])) < 2:
            continue
        Xi = np.column_stack([np.ones(len(y)), np.log(NN[i]), np.log(K[i])])
        try:
            bs.append(np.linalg.lstsq(Xi, y[i], rcond=None)[0])
        except np.linalg.LinAlgError:
            continue
    bs = np.array(bs)
    ci = lambda j: (float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5)))
    return dict(A=float(np.exp(beta[0])), b=float(beta[1]), c=float(beta[2]),
                b_ci=ci(1), c_ci=ci(2))


def gather(runs, getT):
    """Per-run (k, N, M) at a given read-out, plus per-cell means and sds.

    Args:
        runs: run records; getT: callable run -> read-out iteration.
    Returns:
        (K, NN, M, cells) with cells = {(k, N): (mean, sd)}.
    """
    K, NN, M = [], [], []
    for r in runs:
        m = F.M_at(r, getT(r), "scalefree")
        if np.isfinite(m) and m > 0:
            K.append(r["k"]); NN.append(r["N"]); M.append(m)
    cells = {}
    for k, n, m in zip(K, NN, M):
        cells.setdefault((k, n), []).append(m)
    cells = {g: (float(np.mean(v)), float(np.std(v))) for g, v in cells.items()}
    return np.array(K, float), np.array(NN, float), np.array(M, float), cells


def main():
    """Build the two-row decisive figure and print the numbers behind it."""
    global ALPHA_TH
    if len(sys.argv) > 1:
        ALPHA_TH = float(sys.argv[1])
    ps.setup()
    runs, _ = F.load()
    for r in runs:
        f = F.fit_run(r["loss"], F.T_ITER)
        r["L_inf"] = f[0] if f else None
    worst = max(r["L_inf"] for r in runs if r["L_inf"])
    F.readout_times(runs, F.LOSS_MARGIN * worst)
    for r in runs:
        t = diffusive_onset(r["trace"], DRIFT_VAR, thresh=ALPHA_TH)
        r["T_diff"] = t if (np.isfinite(t) and t <= r["budget"]) else np.nan

    ks = sorted({r["k"] for r in runs})
    Ns = sorted({r["N"] for r in runs})
    ok = [r for r in runs if np.isfinite(r["T_diff"])]
    print(f"{len(ok)}/{len(runs)} runs reach the {DRIFT_VAR} diffusive point at alpha < {ALPHA_TH}")
    print(f"read-out iteration: mean {np.mean([r['T_diff'] for r in ok]):.0f}, "
          f"range {min(r['T_diff'] for r in ok):.0f}-{max(r['T_diff'] for r in ok):.0f}\n")

    rows = [("FIXED COMPUTE\n(all read at t = 150,000)", lambda r: float(F.T_ITER)),
            (f"MATCHED {DRIFT_VAR} DIFFUSION\n(each read where $\\alpha<{ALPHA_TH}$)",
             lambda r: r["T_diff"])]

    fig, ax = plt.subplots(2, 3, figsize=(16.2, 9.6))
    for i, (title, getT) in enumerate(rows):
        K, NN, M, cells = gather(runs, getT)
        law = fit_law(K, NN, M)
        print(f"{title.splitlines()[0]:34s} b = {law['b']:.3f} "
              f"[{law['b_ci'][0]:.3f}, {law['b_ci'][1]:.3f}]   "
              f"c = {law['c']:.3f} [{law['c_ci'][0]:.3f}, {law['c_ci'][1]:.3f}]")

        # (a) M vs N, one colour per k. One line if M depends on N alone.
        for k in ks:
            xs = [N for N in Ns if (k, N) in cells]
            if not xs:
                continue
            ps.band(ax[i][0], xs, [cells[(k, N)][0] for N in xs],
                    [cells[(k, N)][1] for N in xs], ps.col_k(k, ks), alpha=0.10)
        xx = np.array(Ns, float)
        ax[i][0].plot(xx, law["A"] * xx ** law["b"] * np.mean(ks) ** law["c"], "k-", lw=2, alpha=.75)
        ax[i][0].set(xscale="log", yscale="log", xlabel="N (units)", ylabel="active units $M$",
                     xticks=Ns, xticklabels=[str(n) for n in Ns],
                     title=f"(a) $M$ vs $N$ — one curve per $k$\n"
                           f"$b = {law['b']:.2f}$ [{law['b_ci'][0]:.2f}, {law['b_ci'][1]:.2f}]")
        if i == 0:
            ps.legend_k(ax[i][0], ks, loc="upper left")

        # (b) M vs k, one line per N. Flat lines if M does not depend on k.
        for N in Ns:
            xs = [k for k in ks if (k, N) in cells]
            ps.band(ax[i][1], xs, [cells[(k, N)][0] for k in xs],
                    [cells[(k, N)][1] for k in xs], ps.col_n(N), label=f"N={N}")
        ax[i][1].set(xlabel="k (bits)", ylabel="active units $M$", xticks=ks, yscale="log",
                     title=f"(b) $M$ vs $k$ — one curve per $N$\n"
                           f"$c = {law['c']:.3f}$ "
                           f"[{law['c_ci'][0]:.3f}, {law['c_ci'][1]:.3f}]")
        ax[i][1].legend(loc="best")

        # (c) THE DECISIVE PANEL: divide out N, then look for any k trend left.
        resid = M / NN ** law["b"]
        for N in Ns:
            m = NN == N
            ax[i][2].scatter(K[m] + 0.06 * (Ns.index(N) - 1), resid[m], s=26,
                             color=ps.col_n(N), alpha=.85, label=f"N={N}")
        med = [np.median(resid[K == k]) for k in ks]
        ax[i][2].plot(ks, med, "k-", lw=2, alpha=.8, label="median")
        ax[i][2].axhline(np.median(resid), color="k", ls="--", lw=1.2, alpha=.6)
        lo8, hi8 = 8.0 ** np.array(law["c_ci"])
        ax[i][2].set(xlabel="k (bits)", ylabel="$M / N^{b}$  (size dependence divided out)",
                     xticks=ks,
                     title=f"(c) DECISIVE: any $k$ trend left after removing $N$?\n"
                           f"$M(k{{=}}8)/M(k{{=}}1) = {8.0**law['c']:.2f}\\times$ "
                           f"[{lo8:.2f}, {hi8:.2f}]")
        ax[i][2].legend(loc="best", fontsize=7)

        for a in ax[i]:
            a.set_ylabel(a.get_ylabel(), fontsize=9)
        ax[i][0].text(-0.28, 0.5, title, transform=ax[i][0].transAxes, rotation=90,
                      va="center", ha="center", fontsize=11, fontweight="bold")

    fig.suptitle("Active units depend on network SIZE, not on task COMPLEXITY — "
                 "once networks are compared at the same dynamical state\n"
                 f"Top: read at a fixed budget, where a $k$ effect appears. "
                 f"Bottom: read where {DRIFT_VAR} stops moving directionally, where it does not. "
                 f"Same networks, same measure; only the read-out time differs.", fontsize=11.5)
    fig.tight_layout(rect=[0.02, 0, 1, 0.93])
    return ps.save(fig, "flipflop_decisive", tight=False)


if __name__ == "__main__":
    main()
