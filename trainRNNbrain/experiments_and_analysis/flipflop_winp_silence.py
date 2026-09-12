#!/usr/bin/env python3
"""
Does the initial scale of W_inp change how many units go silent? (T5 read-out)

Unpenalised ReLU flip-flop networks whose W_inp rows were all set to norm s at initialisation
(NBitFlipFlop_std_winp, s in {0.5, 2, 5, 20}) against the default-init ksweep networks (row norm
0.039) at the same k and N. Everything is measured by flipflop_sigmoid_silence.analyse():
silence under the scale-free, modulation and absolute criteria; participation Hoyer and 1/HHI; the
four axes; the participation trajectory (silent fraction and Hoyer vs iteration).

Also reports the FINAL W_inp row norms of live and silent units and ||W_inp||_F per s, since the
hypothesis is about whether the network keeps or sheds the input weight it was given.

Decision rule (pre-registered, project_trajectory.md 2026-09-11): silence < 20% at any s with task
R2 >= 0.9 -> the init scale is a cause; unchanged (within seed spread of the baseline at matched
iteration) -> excluded; intermediate -> reported as a dependence on s.

Usage:  python flipflop_winp_silence.py [WINP_ROOT] [--no-baseline]
Output: img/internal_figures/winp_silence.png
"""

import os
import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle as ps
from common import participation, SILENT_FLIPFLOP
from flipflop_sigmoid_silence import analyse, rates_targets
from flipflop_dimensionality import run_folders

WINP_ROOT = "data/trained_RNNs/NBitFlipFlop_std_winp"
BASELINE_S = 0.039          # default draw at N=2000; the label for the ksweep networks
CMAP = plt.get_cmap("viridis")


def row_norm_stats(folder):
    """Final W_inp row norms of live and silent units, and the total norm.

    Args:
        folder: run folder.
    Returns:
        dict(live_row, silent_row, fro) medians / total; silent_row is nan if no unit is silent.
    """
    W = np.load(glob.glob(os.path.join(folder, "*LastParams*.npz"))[0])["W_inp"]
    r, *_ = rates_targets(folder)
    live = participation(r) >= SILENT_FLIPFLOP
    rn = np.linalg.norm(W, axis=1)
    return dict(live_row=float(np.median(rn[live])) if live.any() else np.nan,
                silent_row=float(np.median(rn[~live])) if (~live).any() else np.nan,
                fro=float(np.linalg.norm(W)))


def main():
    """Tabulate silence and the four axes per (s, N); plot trajectories and end-of-training values."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = args[0] if args else WINP_ROOT
    ps.setup()
    runs = []
    for f in sorted(glob.glob(os.path.join(root, "*", "*", "*LastParams*.npz"))):
        m = re.search(r"_N=(\d+)_s=([\d.]+)", f)
        runs.append((float(m.group(2)), int(m.group(1)), os.path.dirname(f)))
    if "--no-baseline" not in sys.argv:
        Ns = {N for _, N, _ in runs} or {500, 1000}
        runs += [(BASELINE_S, N, f) for f, pen, k, N in run_folders() if k == 3 and pen == "none" and N in Ns]
    rows = []
    for s, N, f in runs:
        m = analyse(f); m.update(row_norm_stats(f)); m["s"] = s; m["N"] = N; rows.append(m)
        print(f"  s={s:<6} N={N:<5} r2={m['r2']:.3f} scalefree={m['scalefree']:.3f} unmod={m['unmodulated']:.3f} "
              f"partHoyer={m['part_hoyer']:.3f} 1/HHI={m['inv_hhi']:.0f} | live row {m['live_row']:.2f} silent row "
              f"{m['silent_row']:.3f} ||W||_F {m['fro']:.1f} | sel={m['sel']:.2f} temp={m['temp']:.2f} D_PR={m['d_pr']:.1f} "
              f"| sf@150k={m.get('sf_at_150k', np.nan):.3f} max 1k-jump={m.get('max_jump_1k', np.nan):.3f} at {m.get('jump_iter', -1)}", flush=True)

    print(f"\n{'s':>7}{'N':>6}{'n':>3}{'r2':>6}{'scalefree':>11}{'sf@150k':>9}{'unmodulated':>13}{'part Hoyer':>12}{'1/HHI':>7}{'live row':>9}{'silent row':>11}{'||W||_F':>9}{'sel':>6}{'temp':>6}{'D_PR':>6}{'max jump/1k':>13}")
    for s in sorted({r["s"] for r in rows}):
        for N in sorted({r["N"] for r in rows if r["s"] == s}):
            rr = [r for r in rows if r["s"] == s and r["N"] == N]
            f = lambda k, fmt: fmt.format(np.nanmean([r[k] for r in rr]), np.nanstd([r[k] for r in rr]))
            print(f"{s:>7}{N:>6}{len(rr):>3}{f('r2', '{:.2f}'):>6}{f('scalefree', '{:.2f}±{:.2f}'):>11}{f('sf_at_150k', '{:.2f}'):>9}{f('unmodulated', '{:.2f}±{:.2f}'):>13}"
                  f"{f('part_hoyer', '{:.2f}±{:.2f}'):>12}{f('inv_hhi', '{:.0f}'):>7}{f('live_row', '{:.2f}'):>9}{f('silent_row', '{:.3f}'):>11}"
                  f"{f('fro', '{:.0f}'):>9}{f('sel', '{:.2f}'):>6}{f('temp', '{:.2f}'):>6}{f('d_pr', '{:.1f}'):>6}{f('max_jump_1k', '{:.2f}±{:.2f}'):>13}")

    ss = sorted({r["s"] for r in rows}); col = {s: CMAP(i / max(len(ss) - 1, 1)) for i, s in enumerate(ss)}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for r in rows:
        if "traj_iters" not in r:
            continue
        ls = "-" if r["N"] == max(x["N"] for x in rows) else "--"
        ax[0].plot(r["traj_iters"], r["traj_silent"], ls, color=col[r["s"]], alpha=.8, label=f"s={r['s']} N={r['N']}")
        ax[1].plot(r["traj_iters"], r["traj_hoyer"], ls, color=col[r["s"]], alpha=.8, label=f"s={r['s']} N={r['N']}")
    ax[0].set(xlabel="iteration", ylabel="silent fraction (p < 0.05 q95)", title="silent fraction along training", xscale="log")
    ax[1].set(xlabel="iteration", ylabel="participation Hoyer sparsity", title="concentration along training", xscale="log", ylim=(0, 1))
    for a in ax[:2]:
        h, l = a.get_legend_handles_labels(); u = dict(zip(l, h)); a.legend(u.values(), u.keys(), fontsize=6.5); a.grid(alpha=.25)
    for N, mk in zip(sorted({r["N"] for r in rows}), ("o", "s")):
        rr = [r for r in rows if r["N"] == N]
        ax[2].scatter([r["s"] for r in rr], [r["unmodulated"] for r in rr], marker=mk, color="k", alpha=.7, label=f"N={N}: unmodulated")
        ax[2].scatter([r["s"] for r in rr], [r["part_hoyer"] for r in rr], marker=mk, facecolors="none", edgecolors="k", alpha=.7, label=f"N={N}: participation Hoyer")
    ax[2].set(xscale="log", xlabel="initial W_inp row norm s (0.039 = default draw)", ylabel="fraction / sparsity", title="end of training", ylim=(0, 1)); ax[2].grid(alpha=.25)
    h, l = ax[2].get_legend_handles_labels(); u = dict(zip(l, h)); ax[2].legend(u.values(), u.keys(), fontsize=7)
    fig.suptitle("W_inp initial row norm vs silence, unpenalised 3-bit flip-flop (T5)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return ps.save(fig, "winp_silence", tight=False)


if __name__ == "__main__":
    main()
