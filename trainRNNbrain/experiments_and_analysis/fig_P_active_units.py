#!/usr/bin/env python3
"""
The problem in one figure (abstract claims P1-P4): how many units of a trained RNN are active, how
that number scales with network size N and task complexity k, and what it costs to get 1000 of them.

Two rows x two panels, unpenalised ReLU networks, each panel under BOTH silence criteria of its task:

  columns  left  CDDM, N = 100..5000 (CDDM_std_g0_drift). Criteria: hard p < 1e-6 and scale-free
                 p < 0.05 q95(p). Fit M = A N^b, bootstrap CI
                 over seeds.
           right k-bit flip-flop, N = 500..4000, k = 1..8 (pr_matrix roots). Criteria: task-calibrated
                 absolute p < 4e-2 and scale-free. Fit M = A N^b k^c over the whole grid, bootstrap CI
                 (pr_matrix.fit_power_law).
  rows     top   MATCHED STATE: CDDM at matched performance (the deepest loss level every seed reaches,
                 plot_M_vs_N.ladder); flip-flop at 1.10x each run's own fitted loss floor.
           bottom MATCHED COMPUTE: every network read at iteration FIXED_ITER. 100k is the largest
                 budget every cell reaches (CDDM N = 5000 and flip-flop N = 4000 both trained 100k), so
                 that is the value; large networks are read EARLIER in their silencing than small ones
                 would be at their own end, which is why the k exponent is positive here (convergence
                 depth, not task demand - project_trajectory 2026-08-25).

Each fitted law is extrapolated to M = 1000 (flip-flop at k = 3) and the required N is printed on the
panel. Subclaims: most units are silent (points far below M = N); M grows as N^b with b between 1/3
and 1/2; k barely moves M at matched state; 1000 active units cost N ~ 10^4.

Usage:  python fig_P_active_units.py
Output: img/internal_figures/fig_P_active_units.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP, T_at_loss, active_count, load_traces
import plotstyle as ps
from plot_M_vs_N import ladder
import pr_matrix

CDDM_SWEEP = "data/trained_RNNs/CDDM_std_g0_drift"
MIN_N = 500                      # N = 100 is excluded everywhere (Pavel, 2026-09-11): it is fully active and only one condition has it
CDDM_FIT_MIN_N = MIN_N
TARGET_M = 1000
EXTRAP_K = 3
FIXED_ITER = 100_000
CDDM_CRIT = [("scalefree", "scale-free  $p<0.05\\,q_{95}(p)$", "-", "o"), ("hard", "hard  $p<10^{-6}$", "--", "s")]
FF_CRIT = [("scalefree", "scale-free  $p<0.05\\,q_{95}(p)$", "-", "o"), (SILENT_FLIPFLOP, "absolute  $p<4\\times10^{-2}$", "--", "s")]
N_BOOT = 2000
CDDM_COL = "#1f77b4"


def cddm_points(by, fixed_iter=None):
    """Active count per seed, CDDM unpenalised, both criteria, at matched performance or a fixed iteration.

    Args:
        by: load_traces output (dict N -> traces); fixed_iter: read at this iteration, or None for
            matched performance at the deepest shared loss level.
    Returns:
        (dict criterion -> dict N -> list of M over seeds, read-out label).
    """
    Lstar = None if fixed_iter else ladder(by)[0]
    out = {c[0]: {} for c in CDDM_CRIT}
    for N in sorted(by):
        if N < MIN_N:
            continue
        for t in by[N]:
            I = np.asarray(t["participation_iters"])
            T = fixed_iter if fixed_iter else T_at_loss(t["loss"], Lstar)
            if T is None or T > I[-1] + (I[1] - I[0]):     # tolerate one probe step past the last sample
                continue
            p = t["participation"][np.argmin(abs(I - T))]
            for crit, *_ in CDDM_CRIT:
                out[crit].setdefault(N, []).append(active_count(p, crit))
    return out, (f"iteration {fixed_iter // 1000}k" if fixed_iter else f"matched performance ($L^*$ = {Lstar:.4f})")


def fit_cddm(byN, min_N):
    """M = A N^b over seed-resampled cells, with a bootstrap CI on b.

    Args:
        byN: dict N -> list of M; min_N: smallest N in the fit.
    Returns:
        dict A, b, b_ci.
    """
    Ns = [N for N in sorted(byN) if N >= min_N]
    rng = np.random.default_rng(0)
    def fit(sample):
        b, a = np.polyfit(np.log(Ns), np.log([np.mean(sample[N]) for N in Ns]), 1)
        return float(np.exp(a)), float(b)
    A, b = fit(byN)
    bs = [fit({N: rng.choice(byN[N], len(byN[N])) for N in Ns})[1] for _ in range(N_BOOT)]
    return dict(A=A, b=b, b_ci=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))))


def flipflop_runs(runs, fixed_iter=None):
    """Set each unpenalised flip-flop run's read-out iteration.

    Args:
        runs: pr_matrix.load() output; fixed_iter: read at this iteration (dropped if beyond the
            run's budget), or None for the excess criterion 1.10x own floor.
    Returns:
        list of runs with 'T' finite.
    """
    for r in runs:
        if fixed_iter:
            r["T"] = fixed_iter if fixed_iter <= r["budget"] else float("nan")
        else:
            r["floor"] = pr_matrix.fit_floor(r["loss"], r["budget"])
            r["T"] = pr_matrix.excess_time(r["loss"], r["floor"], pr_matrix.EXCESS_DELTA)
    return [r for r in runs if np.isfinite(r["T"])]


def n_for(M, A, b, k=None, c=0.0):
    """N at which the fitted law reaches M active units (at complexity k if the law has one)."""
    return (M / (A * (k ** c if k else 1.0))) ** (1.0 / b)


def panel_cddm(ax, cd, label, tag):
    """CDDM panel: seeds, means, fits, extrapolation, for both criteria."""
    lines, needs = [], []
    for crit, lab, ls, mk in CDDM_CRIT:
        byN = cd[crit]
        Ns = sorted(byN)
        f = fit_cddm(byN, CDDM_FIT_MIN_N)
        need = n_for(TARGET_M, f["A"], f["b"]); needs.append(need)
        ax.plot(Ns, [np.mean(byN[N]) for N in Ns], ls, marker=mk, color=CDDM_COL, label=lab)
        for N in Ns:
            ax.scatter([N] * len(byN[N]), byN[N], color=CDDM_COL, s=12, alpha=.4, zorder=3)
        xx = np.array([CDDM_FIT_MIN_N, need])
        ax.plot(xx, f["A"] * xx ** f["b"], ls, color=CDDM_COL, lw=0.9, alpha=.5)
        ax.scatter([need], [TARGET_M], marker="x", color="k", zorder=5)
        lines.append(f"{lab.split('  ')[0]}: $b$ = {f['b']:.2f} [{f['b_ci'][0]:.2f}, {f['b_ci'][1]:.2f}]  →  {TARGET_M} active at N ≈ {need:.1e}")
        print(f"CDDM {tag} {crit}: " + "  ".join(f"N={N}: {np.mean(byN[N]):.0f}±{np.std(byN[N]):.0f}" for N in Ns)
              + f"   b={f['b']:.3f} [{f['b_ci'][0]:.3f}, {f['b_ci'][1]:.3f}]  N({TARGET_M})={need:.0f}")
    full = np.array([min(Ns), max(needs)])
    ax.plot(full, full, ":", color="0.4", lw=1, label="$M=N$")
    ax.axhline(TARGET_M, color="0.7", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel="network size N", ylabel="active units M", title=f"CDDM, {label}")
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8, color="0.25")
    ax.legend(loc="lower right")


def panel_flipflop(ax, runs, label, tag):
    """Flip-flop panel: one line per k, both criteria, joint-law fit and extrapolation at k = EXTRAP_K."""
    ks = sorted({r["k"] for r in runs})
    lines = []
    for crit, lab, ls, mk in FF_CRIT:
        fn = lambda p, c=crit: active_count(p, c)
        law = pr_matrix.fit_law(runs, "none", fn)
        need = n_for(TARGET_M, law["A"], law["b"], EXTRAP_K, law["c"])
        for k in ks:
            cell = {}
            for r in runs:
                if r["k"] == k:
                    v = pr_matrix.measure(r, fn)
                    if np.isfinite(v):
                        cell.setdefault(r["N"], []).append(v)
            Ns = sorted(cell)
            ax.plot(Ns, [np.mean(cell[N]) for N in Ns], ls, marker=mk, ms=4, lw=1.1, color=ps.col_k(k, ks),
                    label=f"k = {k}" if crit == "scalefree" else None)
        xx = np.array([500, need])
        ax.plot(xx, law["A"] * xx ** law["b"] * EXTRAP_K ** law["c"], ls, color=ps.col_k(EXTRAP_K, ks), lw=0.9, alpha=.6)
        ax.scatter([need], [TARGET_M], marker="x", color="k", zorder=5)
        lines.append(f"{lab.split('  ')[0]}: $b$ = {law['b']:.2f} [{law['b_ci'][0]:.2f}, {law['b_ci'][1]:.2f}], "
                     f"$c$ = {law['c']:+.2f} [{law['c_ci'][0]:+.2f}, {law['c_ci'][1]:+.2f}]  →  k = {EXTRAP_K}: {TARGET_M} active at N ≈ {need:.1e}")
        print(f"flip-flop {tag} {crit}: b={law['b']:.3f} [{law['b_ci'][0]:.3f}, {law['b_ci'][1]:.3f}]  c={law['c']:+.3f} "
              f"[{law['c_ci'][0]:+.3f}, {law['c_ci'][1]:+.3f}]  n={law['n']}  N({TARGET_M}, k={EXTRAP_K})={need:.0f}")
    full = np.array([500, 2e4])
    ax.plot(full, full, ":", color="0.4", lw=1, label="$M=N$")
    ax.axhline(TARGET_M, color="0.7", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel="network size N", ylabel="active units M",
           title=f"k-bit flip-flop, {label};  solid = scale-free, dashed = absolute")
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8, color="0.25")
    ax.legend(loc="lower right", ncol=2)


def main():
    """Draw the two rows and write fig_P_active_units.png."""
    ps.setup()
    fig, ax = plt.subplots(2, 2, figsize=(12, 9.4))
    by = load_traces(CDDM_SWEEP)
    runs = pr_matrix.load()
    runs = [r for r in runs if r["pen"] == "none"]

    cd, label = cddm_points(by)
    panel_cddm(ax[0, 0], cd, label, "matched")
    panel_flipflop(ax[0, 1], flipflop_runs(runs), "read at 1.10× own loss floor", "matched")

    cd, label = cddm_points(by, FIXED_ITER)
    panel_cddm(ax[1, 0], cd, label, f"iter{FIXED_ITER // 1000}k")
    panel_flipflop(ax[1, 1], flipflop_runs(runs, FIXED_ITER), f"read at iteration {FIXED_ITER // 1000}k", f"iter{FIXED_ITER // 1000}k")

    fig.suptitle("Trained RNNs keep few units active: $M \\propto N^{b}$ with $b$ between 1/3 and 1/2, almost independent of k; "
                 "1000 active units cost N ~ 10$^4$\ntop: matched state   ·   bottom: matched compute", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return ps.save(fig, "fig_P_active_units", tight=False)


if __name__ == "__main__":
    main()
