#!/usr/bin/env python3
"""
Iterations needed to reach (1 + delta) x each run's OWN fitted loss floor, as T(k) curves per N.

Same read-out as `pr_matrix.py --excess`, but plotting the READ-OUT TIME itself rather than what is
measured there: how long does each condition take to get within 1% and 3% of the best loss it will
ever achieve? One panel per penalty, one curve per N, delta on separate rows.

⚠️ THESE NUMBERS ARE CENSORED, NOT MISSING AT RANDOM. A run that never reaches (1+delta) x floor
inside its budget contributes no point, and the runs that fail are systematically the SLOW ones - so
every plotted mean is biased LOW wherever coverage is below 100%. Coverage is printed per cell and
any cell missing a seed is ringed on the curve; a cell with fewer than 2 reaching seeds is dropped
entirely rather than plotted as a single-seed point with no error bar.

⚠️ BUDGETS ARE NOT EQUAL ACROSS THE GRID. N=4000 runs a 100k budget against 400-500k elsewhere, so
its censoring is far more severe at delta=0.01 and its curve must not be read as "faster".

Floors are fitted per run over that run's own budget, never a common range (see pr_matrix).

Output: img/internal_figures/time_to_floor.png

Usage:  python flipflop_time_to_floor.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle as ps
from pr_matrix import PENS, load, fit_floor, excess_time, fit_power_law

DELTAS = [0.01, 0.03]
MIN_SEEDS = 2          # cells with fewer reaching seeds are dropped, not plotted bare


def read_times(runs, delta):
    """Read-out iteration for every run at one delta.

    Args:
        runs: list of run dicts from pr_matrix.load(), each already carrying a fitted "floor".
        delta: excess above the floor, e.g. 0.01.
    Returns:
        list of (pen, k, N, T) with T possibly nan (never reached inside the budget).
    """
    return [(r["pen"], r["k"], r["N"], excess_time(r["loss"], r["floor"], delta)) for r in runs]


def cells(rows, pen, ks, Ns):
    """(mean, sd, n_reached, n_total) grids of read-out time over (N, k) for one penalty.

    Args:
        rows: output of read_times(); pen: penalty; ks, Ns: sorted axis values.
    Returns:
        four (len(Ns), len(ks)) arrays.
    """
    Z = np.full((len(Ns), len(ks)), np.nan)
    S = np.full((len(Ns), len(ks)), np.nan)
    R = np.zeros((len(Ns), len(ks)), dtype=int)
    C = np.zeros((len(Ns), len(ks)), dtype=int)
    for i, N in enumerate(Ns):
        for j, k in enumerate(ks):
            v = np.array([t for p, kk, nn, t in rows if p == pen and kk == k and nn == N])
            got = v[np.isfinite(v)]
            R[i, j], C[i, j] = got.size, v.size
            if got.size >= MIN_SEEDS:
                Z[i, j], S[i, j] = got.mean(), got.std()
    return Z, S, R, C


def main():
    """Plot and tabulate time-to-floor at each delta, one panel per penalty."""
    ps.setup()
    runs = load()
    for r in runs:
        r["floor"] = fit_floor(r["loss"], r["budget"])
    ks = sorted({r["k"] for r in runs})
    Ns = sorted({r["N"] for r in runs})

    fig, ax = plt.subplots(len(DELTAS), len(PENS), figsize=(4.3 * len(PENS), 4.8 * len(DELTAS)),
                           squeeze=False, sharey=True)
    for r_i, delta in enumerate(DELTAS):
        rows = read_times(runs, delta)
        print(f"\n{'='*96}\niterations to reach {1+delta:.2f} x own floor "
              f"(mean +- sd over seeds; 'n/N' = seeds reaching / seeds present)\n{'='*96}")
        for c_i, pen in enumerate(PENS):
            Z, S, R, C = cells(rows, pen, ks, Ns)
            a = ax[r_i][c_i]
            print(f"\n  {pen}")
            print("    N     " + "".join(f"{k:>14}" for k in ks))
            for i, N in enumerate(Ns):
                if not C[i].any():
                    continue
                line = f"    {N:<6}"
                for j in range(len(ks)):
                    line += (f"{Z[i,j]/1000:>8.0f}k {R[i,j]}/{C[i,j]}" if np.isfinite(Z[i, j])
                             else f"{'-':>8} {R[i,j]}/{C[i,j]}")
                print(line)
                if np.isfinite(Z[i]).any():
                    ps.band(a, ks, Z[i], S[i], ps.col_n(N), label=f"N={N}")
                    part = np.isfinite(Z[i]) & (R[i] < C[i])       # censored cells
                    if part.any():
                        a.plot(np.array(ks)[part], Z[i][part], "o", mfc="none", ms=11,
                               mec=ps.col_n(N), mew=1.6)
            f = fit_power_law([k for p, k, n, t in rows if p == pen and np.isfinite(t)],
                              [n for p, k, n, t in rows if p == pen and np.isfinite(t)],
                              [t for p, k, n, t in rows if p == pen and np.isfinite(t)])
            if f:
                print(f"    law  T = {f['A']:.3g} N^{f['b']:+.2f} k^{f['c']:+.2f}   "
                      f"b [{f['b_ci'][0]:+.2f},{f['b_ci'][1]:+.2f}]  "
                      f"c [{f['c_ci'][0]:+.2f},{f['c_ci'][1]:+.2f}]")
                a.text(.03, .97, f"$T = {f['A']:.2g}N^{{{f['b']:+.2f}}}k^{{{f['c']:+.2f}}}$\n"
                                 f"$b$={f['b']:+.2f} [{f['b_ci'][0]:+.2f},{f['b_ci'][1]:+.2f}]   "
                                 f"$c$={f['c']:+.2f} [{f['c_ci'][0]:+.2f},{f['c_ci'][1]:+.2f}]",
                       transform=a.transAxes, fontsize=6.8, va="top",
                       bbox=dict(fc="white", ec="0.7", alpha=.85, boxstyle="round,pad=0.3"))
            a.set_yscale("log")
            a.set(xlabel="k (bits)", xticks=ks,
                  ylabel=f"iterations to {1+delta:.2f}× floor" if c_i == 0 else "",
                  title=(f"{pen}\n" if r_i == 0 else "") +
                        f"time to {1+delta:.2f}× floor\nopen ring = some seeds never reached it")
            a.legend(fontsize=7, loc="lower right")
            a.grid(alpha=.25, which="both")
    fig.suptitle("Iterations to reach (1+δ)× each run's OWN fitted loss floor\n"
                 "⚠️ censored: runs that never reach the level contribute nothing, so ringed points "
                 "are biased LOW  ·  N=4000 has a 100k budget against 400-500k elsewhere",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return ps.save(fig, "time_to_floor", tight=False)


if __name__ == "__main__":
    main()
