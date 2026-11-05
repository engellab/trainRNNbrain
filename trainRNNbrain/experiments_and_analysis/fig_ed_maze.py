#!/usr/bin/env python3
"""Extended Data figure: the maze task (reinforcement learning) shows the same loss of participation.

QUESTION. Does a network trained by PPO on the maze (issue engellab/trainRNNbrain#3; paper package
ptolmachev/dead-relu-paper#182) leave units inactive the way the supervised tasks do: fewer active
units than N once the task is solved, a share of active units that falls with N, a count that keeps
falling after the task is learned, and the same loss under tanh and the shifted sigmoid? Figure 1
has no room for a fourth ablation sub-panel, so the maze gets this figure.

  (a) the clean probe's success rate against PPO updates, one thin line per ReLU run of the size
      series, colour by N, a filled mark at each run's "learned" probe (the first probe from which
      the success stays at or above the dashed 0.9 for MAZE_LEARNED_HOLD probes);
  (b) active units against PPO updates, the same runs, one sub-panel per N with y from 0 to that N.
      A shared 0 to 2,000 axis would put the N = 250 curve in its bottom eighth, where a change of
      20 units is 0.3 mm; one axis per size shows every curve at the same fraction of its network;
  (c) active units at "learned" (filled) and at the end of training (open) against N, one marker per
      run, the seed mean as a bar, and a power law fitted to the size means of the end counts. The
      exponent b is reported as in Results 2 (fig_paper_F2_size.power_fit: a least-squares line
      through log mean count against log N, a bootstrap 95% interval over runs within each size).
      The supervised fit starts at N = 500 because nearly every unit is active at N = 100; the maze
      suite has no N = 100 cell, so every size with a learned run is fitted;
  (d) the ablation (rl_maze_ablate.py): success rate under the training noise as units are removed,
      quietest first (solid), busiest first (dotted) and in random order (dashed, band: range over
      the random orders), x as a share of N so that networks of different sizes overlay, a vertical
      line at each network's inactive count and that count, with N, at the top;
  (e) the activations at N = 1,000: active units at the end of training under the participation rule
      (circles) and under the modulation rule (squares; std alone, saved as active_mod, the count
      that detects a saturated unit), per activation, with the mean success rate under each name.

READ-OUT POINTS (pre-registered on issue #3, in common): "learned" is the first probe from which the
clean probe's success stays at or above 0.9 for 5 probes (common.maze_learned_index), the maze
counterpart of the floor crossing; "end" is the last probe (update 2,000 in the suite). A run whose
end success is below 0.9 is drawn with a cross in (c) and excluded from the means and the fit, as
the pre-registration excludes a size that does not learn. Marker fill follows the style sheet
(paperstyle, dead-relu-paper#83): filled = the run's own convergence point, open = a fixed
iteration (the end of the budget); the two counting rules of (e) differ by shape, both open.

COLOUR. One task, so one hue: TASK_MAZE, with N as a lightness ramp (N_SHADE), as Figure 2f does for
the 3BFF sizes. plotstyle.col_n is not used: its tab10 blue and green are the 3BFF and DMTS hues,
and it has no entry for N = 250. Text is INK or MUTED.

DATA. --root (default data/trained_RNNs_rl_maze/Maze_suite): N = 250, 500, 1000, 2000 x seeds 1-3,
ReLU, and tanh, sigmoid (shifted), softplus (beta 25), leaky_relu at N = 1000, seeds 1-2 (the
launcher slurm/rl_maze_suite_della.slurm). Each run folder holds a saved config, a *Progress.json and
a *Participation.npz (common.maze_load_runs reads the layout of run_rl.py and the earlier one of
trainRNNbrain/rl/run_maze_ppo.py). Runs with the goal cue on (the "cue" arm of the first
two arrays) are dropped unless --keep-cue. --ablation (default data/rl_maze_ablation): the JSONs of
rl_maze_ablate.py. The suite lands cell by cell: every panel draws what exists and the script
prints which expected cells are missing.

OUTPUT. img/internal_figures/fig_ed_maze.{pdf,svg} (paperstyle.save, vector only, 165.1 mm wide,
FIG_H_MM tall) and, on stdout, one row per run (N, activation, seed, learned update, active at
learned, active at end, modulation count at end, success at end), the fit, the ablation numbers
and the text-overlap check.

Run:  python trainRNNbrain/experiments_and_analysis/fig_ed_maze.py [--root DIR] [--ablation DIR]
      [--keep-cue]
Dev:  --root data/trained_RNNs_rl_maze/Maze_second (8 ReLU runs, N = 500 and 1000, two arms)
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator

import paperstyle as ps
from common import MAZE_LEARNED_HOLD, MAZE_LEARNED_SUCCESS, maze_learned_index, maze_load_runs

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ROOT = os.path.join(REPO, "data", "trained_RNNs_rl_maze", "Maze_suite")
ABLATION_DIR = os.path.join(REPO, "data", "rl_maze_ablation")

# the suite's grid (slurm/rl_maze_suite_della.slurm): the ReLU size series and the activations
SIZES = (250, 500, 1000, 2000)
SEEDS = (1, 2, 3)
ACT_N = 1000                      # the activations are compared at this size
ACTS = ("relu", "leaky_relu", "softplus", "tanh", "sigmoid")   # saved names, drawing order
ACT_SEEDS = (1, 2)
ACT_LABEL = {"relu": "ReLU", "leaky_relu": "leaky\nReLU", "softplus": "softplus", "tanh": "tanh",
             "sigmoid": "shifted\nsigmoid"}   # at most two lines, under 9 mm wide at TICK_PT
EXPECTED = [(N, "relu", s) for N in SIZES for s in SEEDS] + \
           [(ACT_N, a, s) for a in ACTS if a != "relu" for s in ACT_SEEDS]

# N as a lightness ramp of the maze hue (ps.TASK_MAZE = #941862 is N = 2,000): light to dark
N_SHADE = {250: "#d29ebd", 500: "#bf74a1", 1000: "#a94681", 2000: ps.TASK_MAZE, 4000: "#681145"}

N_BOOT = 2000                     # bootstrap resamples for the exponent's interval (panel c)
BOOT_SEED = 20261104
X_OFF = 1.07                      # panel c: "learned" at N / X_OFF, "end" at N * X_OFF (log axis)
BAR_HALF = 1.035                  # panel c: a mean bar spans x / BAR_HALF to x * BAR_HALF
D_YLIM = (0.0, 1.14)              # panel d: room above success = 1 for the inactive-count labels

# layout in millimetres from the canvas's top-left corner: (left, top, width, height)
FIG_H_MM = 107.0
B_STEP_MM = 27.0                  # panel b: one sub-panel every this many mm (19 wide, 8 gap)
LAYOUT_MM = {
    "a":  (12.5, 8.0, 32.5, 32.0),
    "b0": (58.0, 8.0, 19.0, 32.0),      # the first of the four sub-panels of (b)
    "c":  (12.5, 60.0, 34.5, 34.0),      # 12.5: the "2,000" tick labels and the y label fit left of it
    "d":  (59.0, 60.0, 46.0, 34.0),
    "d_key": (57.0, 50.5, 50.0, 7.0),   # the legend of (d), above it and off the data
    "e":  (118.5, 60.0, 43.0, 34.0),
}


def shade(N):
    """Colour of a network size: a tint of the maze hue, or MUTED for a size outside the suite.

    Args:
        N: int, the network size.
    Returns:
        str, a matplotlib colour.
    """
    return N_SHADE.get(int(N), ps.MUTED)


def ax_mm(fig, left, top, width, height):
    """Add an axes placed in millimetres from the canvas's top-left corner (as fig_paper_F1 does).

    Args:
        fig: the figure (its size fixes the conversion); left, top, width, height: mm.
    Returns:
        the new Axes.
    """
    W, H = fig.get_figwidth() / ps.MM, fig.get_figheight() / ps.MM
    return fig.add_axes([left / W, 1.0 - (top + height) / H, width / W, height / H])


def read_points(run):
    """The two read-out points of one run.

    Args:
        run: dict from common.maze_load_runs.
    Returns:
        dict: learned_i (int probe index or None), learned (update there or None), active_learned
        (count there or None), active_end, success_end (last probe), mod_end (modulation count at
        the last probe, or None if the run did not save active_mod), mod_learned (same, at learned).
    """
    pr = run["probes"]
    li = maze_learned_index(pr["probe_success"])
    mod = pr.get("active_mod")
    return dict(learned_i=li,
                learned=None if li is None else int(pr["update"][li]),
                active_learned=None if li is None else int(pr["active"][li]),
                active_end=int(pr["active"][-1]),
                success_end=float(pr["probe_success"][-1]),
                mod_end=None if mod is None else int(mod[-1]),
                mod_learned=None if (mod is None or li is None) else int(mod[li]))


def summarise(runs):
    """One row per run, printed as a table: every number the figure draws.

    Args:
        runs: list from common.maze_load_runs, each with its read_points under "pt".
    Returns:
        list of dicts: N, act, seed, cue, updates (last update), learned, active_learned,
        active_end, mod_end, success_end, learned_ok (bool, success_end >= MAZE_LEARNED_SUCCESS).
    """
    rows = []
    for r in runs:
        p = r["pt"]
        rows.append(dict(N=r["N"], act=r["act"], seed=r["seed"], cue=r["cue"],
                         updates=int(r["progress"][-1]["update"]), learned=p["learned"],
                         active_learned=p["active_learned"], active_end=p["active_end"],
                         mod_end=p["mod_end"], success_end=p["success_end"],
                         learned_ok=p["success_end"] >= MAZE_LEARNED_SUCCESS))
    fmt = lambda v: "-" if v is None else str(v)
    print(f"{'N':>5} {'act':>10} {'seed':>4} {'cue':>5} {'updates':>7} {'learned':>7} {'act@learn':>9} "
          f"{'act@end':>7} {'share%':>6} {'mod@end':>7} {'succ@end':>8}")
    for w in rows:
        print(f"{w['N']:>5} {w['act']:>10} {w['seed']:>4} {str(w['cue']):>5} {w['updates']:>7} "
              f"{fmt(w['learned']):>7} {fmt(w['active_learned']):>9} {w['active_end']:>7} "
              f"{100 * w['active_end'] / w['N']:>6.1f} {fmt(w['mod_end']):>7} {w['success_end']:>8.2f}")
    return rows


def missing_cells(runs):
    """Print and return the expected suite cells with no run under the root.

    Args:
        runs: list from common.maze_load_runs (after the cue filter).
    Returns:
        list of (N, act, seed) tuples from EXPECTED that no run matches.
    """
    have = {(r["N"], r["act"], r["seed"]) for r in runs}
    miss = [c for c in EXPECTED if c not in have]
    if miss:
        print(f"missing {len(miss)} of {len(EXPECTED)} suite cells: "
              + ", ".join(f"N={N} {a} s{s}" for N, a, s in miss))
    else:
        print("every suite cell is present")
    return miss


def load_ablation(folder):
    """Every ablation JSON of rl_maze_ablate.py under a folder.

    Args:
        folder: str.
    Returns:
        list of dicts (N, n_active, ks, score_quiet, score_busy, score_random (list of lists),
        score_full, clean_full, label, run_dir), sorted by N then label.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(path) as fh:
            out.append(json.load(fh))
    return sorted(out, key=lambda a: (a["N"], a["label"]))


def power_fit(cells):
    """Least-squares line through log(mean active count) against log N, over every size given.

    Args:
        cells: dict N -> array of active counts (at least two sizes).
    Returns:
        (exponent b, prefactor A) with mean count ~ A N^b.
    """
    Ns = np.array(sorted(cells), float)
    ys = np.array([cells[N].mean() for N in sorted(cells)])
    b, loga = np.polyfit(np.log(Ns), np.log(ys), 1)
    return float(b), float(np.exp(loga))


def boot_exponent(cells, rng):
    """Bootstrap 95% interval of the fitted exponent, resampling runs within each size.

    Args:
        cells: dict N -> array of active counts; rng: numpy Generator.
    Returns:
        (2.5th, 97.5th percentile) of the exponent over N_BOOT resamples.
    """
    bs = [power_fit({N: rng.choice(v, len(v)) for N, v in cells.items()})[0] for _ in range(N_BOOT)]
    return tuple(float(q) for q in np.percentile(bs, [2.5, 97.5]))


def time_axis(ax, x_max):
    """Ticks and limits of a PPO-update axis: 0, half, x_max, formatted as counts.

    Args:
        ax: axes; x_max: int, the last update drawn.
    Returns:
        None.
    """
    ax.set_xlim(0, x_max)
    ax.set_xticks([0, x_max / 2, x_max])
    ps.count_ticks(ax.xaxis)


def panel_success(ax, series, x_max):
    """Panel (a): the clean probe's success rate against PPO updates, one line per run.

    Args:
        ax: axes; series: the ReLU runs (with "pt"); x_max: int, the x limit.
    Returns:
        None.
    """
    for r in series:
        col, pr, li = shade(r["N"]), r["probes"], r["pt"]["learned_i"]
        ax.plot(pr["update"], pr["probe_success"], "-", lw=ps.THIN_LW, color=col, zorder=3)
        if li is not None:
            ax.plot([pr["update"][li]], [pr["probe_success"][li]], "o", ms=ps.MARKER_PT, zorder=5,
                    **ps.mark_floor(col))
    ax.axhline(MAZE_LEARNED_SUCCESS, ls=(0, (4, 3)), lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax.text(0.01 * x_max, MAZE_LEARNED_SUCCESS + 0.015, f"{MAZE_LEARNED_SUCCESS:g}",
            fontsize=ps.TICK_PT, color=ps.MUTED, ha="left", va="bottom")
    time_axis(ax, x_max)
    ax.set(ylim=(0, 1.02), xlabel="PPO update", ylabel="success rate (clean probe)")
    ax.set_yticks([0, 0.5, 1.0])
    sizes = sorted({r["N"] for r in series})
    handles = [Line2D([], [], color=shade(N), lw=1.2, label=f"$N$ = {ps.fmt_count(N)}") for N in sizes]
    handles.append(Line2D([], [], ls="none", marker="o", ms=ps.MARKER_PT, **ps.mark_floor(ps.MUTED),
                          label="learned"))
    # lower right: a run that has learned sits at success 1, above the dashed line and the legend
    ax.legend(handles=handles, loc="lower right", borderaxespad=0.3)
    ps.ygrid(ax)


def panel_active(axes, series, x_max):
    """Panel (b): active units against PPO updates, one sub-panel per size, y from 0 to N.

    A filled mark at the "learned" probe and an open mark at the last probe, on every run.

    Args:
        axes: one Axes per entry of SIZES, left to right; series: the ReLU runs (with "pt");
        x_max: int, the x limit.
    Returns:
        list of the sizes with no run.
    """
    empty = []
    for ax, N in zip(axes, SIZES):
        runs = [r for r in series if r["N"] == N]
        col = shade(N)
        for r in runs:
            pr, li = r["probes"], r["pt"]["learned_i"]
            ax.plot(pr["update"], pr["active"], "-", lw=ps.THIN_LW, color=col, zorder=3)
            if li is not None:
                ax.plot([pr["update"][li]], [pr["active"][li]], "o", ms=ps.MARKER_PT, zorder=5,
                        **ps.mark_floor(col))
            ax.plot([pr["update"][-1]], [pr["active"][-1]], "o", ms=ps.MARKER_PT, ls="none",
                    zorder=5, clip_on=False, **ps.mark_fixed(col))
        if not runs:
            empty.append(N)
            ax.text(0.5, 0.5, "no runs", transform=ax.transAxes, ha="center", va="center",
                    fontsize=ps.TICK_PT, color=ps.MUTED)
        time_axis(ax, x_max)
        ax.set(ylim=(0, N), title=f"$N$ = {ps.fmt_count(N)}")
        ax.set_yticks([0, N / 2, N])
        ps.count_ticks(ax.yaxis)
        ps.ygrid(ax)
    axes[0].set_ylabel(ps.LABELS["active"])
    return empty


def panel_size(ax, rows):
    """Panel (c): active units at "learned" and at the end against N, with the power-law fit.

    One marker per ReLU run: filled circle at N / X_OFF for the count at "learned", open circle at
    N * X_OFF for the count at the end; a run whose end success is below MAZE_LEARNED_SUCCESS is a
    cross at the end position and enters no mean and no fit. A bar at each size's mean. The fit is a
    line through the size means of the end counts over every size with at least one learned run
    (power_fit); its exponent is in the legend. The diagonal marks "every unit active".

    Args:
        ax: axes; rows: the summarise rows.
    Returns:
        dict with b, lo, hi (exponent and its bootstrap 95% interval; None without two sizes),
        cells (N -> array of end counts of the learned runs), failed (list of (N, seed) not learned).
    """
    rel = [w for w in rows if w["act"] == "relu"]
    cells, cells_l, failed = {}, {}, []
    for w in rel:
        col, N = shade(w["N"]), w["N"]
        if w["learned_ok"]:
            cells.setdefault(N, []).append(w["active_end"])
            ax.plot([N * X_OFF], [w["active_end"]], "o", ms=ps.MARKER_PT, ls="none", zorder=5,
                    **ps.mark_fixed(col))
        else:
            failed.append((N, w["seed"]))
            ax.plot([N * X_OFF], [w["active_end"]], "x", ms=ps.MARKER_PT, mec=col, mew=0.8, zorder=5)
        if w["active_learned"] is not None:
            cells_l.setdefault(N, []).append(w["active_learned"])
            ax.plot([N / X_OFF], [w["active_learned"]], "o", ms=ps.MARKER_PT, zorder=5,
                    **ps.mark_floor(col))
    cells = {N: np.array(v, float) for N, v in cells.items()}
    cells_l = {N: np.array(v, float) for N, v in cells_l.items()}
    for cc, x_fac in ((cells_l, 1 / X_OFF), (cells, X_OFF)):
        for N, v in cc.items():
            x = N * x_fac
            ax.plot([x / BAR_HALF, x * BAR_HALF], [v.mean()] * 2, "-", lw=1.4, color=shade(N),
                    zorder=4, solid_capstyle="butt")
    fit = dict(b=None, lo=None, hi=None, cells=cells, failed=failed)
    handles = [Line2D([], [], ls="none", marker="o", ms=ps.MARKER_PT, **ps.mark_floor(ps.MUTED),
                      label="learned"),
               Line2D([], [], ls="none", marker="o", ms=ps.MARKER_PT, **ps.mark_fixed(ps.MUTED),
                      label="end of training")]
    if failed:
        handles.append(Line2D([], [], ls="none", marker="x", ms=ps.MARKER_PT, mec=ps.MUTED, mew=0.8,
                              label="not learned"))
    if len(cells) >= 2:
        b, A = power_fit(cells)
        lo, hi = boot_exponent(cells, np.random.default_rng(BOOT_SEED))
        fit.update(b=b, lo=lo, hi=hi)
        xf = np.array([min(cells) * 0.8, max(cells) * 1.25])
        ax.plot(xf, A * xf ** b, "-", lw=0.8, color=ps.INK, zorder=3)
        handles.append(Line2D([], [], color=ps.INK, lw=0.8, label=f"fit, $b$ = {b:.2f}"))
    x_lo, x_hi = SIZES[0] / 1.4, SIZES[-1] * 1.4
    counts = [w["active_end"] for w in rel] + [w["active_learned"] for w in rel
                                               if w["active_learned"] is not None]
    y_lo = 0.6 * min(counts) if counts else 50
    ax.set(xscale="log", yscale="log", xlim=(x_lo, x_hi), ylim=(y_lo, SIZES[-1] * 1.5))
    nn = np.array([x_lo, x_hi])
    ax.plot(nn, nn, "-", lw=ps.THIN_LW, color=ps.FAINT, zorder=2)
    ax.text(SIZES[0], SIZES[0] * 1.08, "every unit active", rotation=45, transform_rotates_text=True,
            fontsize=ps.TICK_PT, color=ps.MUTED, ha="left", va="bottom", rotation_mode="anchor")
    ax.set_xticks(SIZES)
    ax.xaxis.set_minor_locator(NullLocator())
    ps.count_ticks(ax.xaxis)
    y_ticks = [t for t in (50, 100, 200, 500, 1000, 2000) if y_lo <= t <= SIZES[-1] * 1.5]
    ax.set_yticks(y_ticks)
    ax.yaxis.set_minor_locator(NullLocator())
    ps.count_ticks(ax.yaxis)
    ax.set(xlabel=ps.LABELS["size"], ylabel=ps.LABELS["active"])
    ax.legend(handles=handles, loc="upper left", borderaxespad=0.3, handlelength=1.6)
    ps.ygrid(ax)
    return fit


def panel_ablation(ax, ax_key, abl):
    """Panel (d): success rate as units are removed, every ablated ReLU network overlaid.

    Per network, in its size's shade: quietest first (solid), busiest first (dotted), the mean over
    the random orders (dashed) with their range as a band; a thin vertical line at its inactive
    count (the counts are printed and belong in the caption: with three sizes within 36 to 57
    percent, labels on the axes overlapped). The legend sits in `ax_key`, above the panel and off the data, because
    a network with more than half its units inactive runs its quietest-first curve at success 1
    across the top of the axes.

    Args:
        ax: axes; ax_key: a blank axes above it for the legend; abl: list from load_ablation.
    Returns:
        list of dicts per drawn network: label, N, n_inactive, full, quiet_at_inactive,
        random_at_inactive, busy_at_inactive (success rates with 0 and n_inactive units removed).
    """
    rows = []
    drawn = [a for a in abl if "act=relu" in a["label"]]
    for a in abl:
        if a not in drawn:
            print(f"  panel d: {a['label']} is not a ReLU network, not drawn")
    for a in drawn:
        N, col = a["N"], shade(a["N"])
        x = 100.0 * np.asarray(a["ks"], float) / N
        rand = np.asarray(a["score_random"], float)
        n_in = N - a["n_active"]
        ax.fill_between(x, rand.min(0), rand.max(0), color=col, alpha=0.15, lw=0, zorder=2)
        ax.plot(x, a["score_quiet"], "-", lw=ps.DATA_LW, color=col, zorder=5)
        ax.plot(x, rand.mean(0), "--", lw=0.9, color=col, zorder=4)
        ax.plot(x, a["score_busy"], ":", lw=1.1, color=col, zorder=4)
        ax.plot([100.0 * n_in / N] * 2, [0, 1.0], "-", lw=ps.THIN_LW, color=col, zorder=3)
        i = int(np.argmin(np.abs(np.asarray(a["ks"]) - n_in)))
        rows.append(dict(label=a["label"], N=N, n_inactive=n_in, full=a["score_full"],
                         quiet_at_inactive=a["score_quiet"][i], random_at_inactive=float(rand.mean(0)[i]),
                         busy_at_inactive=a["score_busy"][i]))
    if not drawn:
        ax.text(0.5, 0.5, "no ablation results", transform=ax.transAxes, ha="center", va="center",
                fontsize=ps.TICK_PT, color=ps.MUTED)
    ax.set(xlim=(0, 100), ylim=D_YLIM, xlabel="units removed (% of the network)",
           ylabel="success rate (training noise)")
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 0.5, 1.0])
    ps.blank(ax_key)
    # the random order's band and its mean line share one entry (a tuple handle draws both)
    ax_key.legend(handles=[Line2D([], [], color=ps.MUTED, lw=ps.DATA_LW),
                           (Patch(color=ps.MUTED, alpha=0.2, lw=0),
                            Line2D([], [], color=ps.MUTED, lw=0.9, ls="--")),
                           Line2D([], [], color=ps.MUTED, lw=1.1, ls=":"),
                           Line2D([], [], color=ps.MUTED, lw=ps.THIN_LW)],
                  labels=["quietest first", "random order (band: range)",
                          "busiest first", "inactive units"],
                  loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2, handlelength=1.6,
                  columnspacing=1.0, borderaxespad=0.0, borderpad=0.0)
    ps.ygrid(ax)
    return rows


def panel_activations(ax, rows):
    """Panel (e): active units at the end of training per activation at N = ACT_N.

    Per run, an open circle for the participation-rule count and an open square for the modulation
    count (both read at the end of training), a bar at each mean; under every activation its name
    and the mean success rate at the end (the number of runs is printed, not drawn).

    Args:
        ax: axes; rows: the summarise rows.
    Returns:
        dict act -> (p-rule counts, modulation counts, success rates) over its runs.
    """
    col, out, ticks = shade(ACT_N), {}, []
    for k, act in enumerate(ACTS):
        mine = [w for w in rows if w["N"] == ACT_N and w["act"] == act]
        p = np.array([w["active_end"] for w in mine], float)
        m = np.array([w["mod_end"] for w in mine if w["mod_end"] is not None], float)
        s = np.array([w["success_end"] for w in mine], float)
        out[act] = (p, m, s)
        for dx, v, marker in ((-0.17, p, "o"), (0.17, m, "s")):
            if not len(v):
                continue
            xs = k + dx + np.linspace(-0.06, 0.06, len(v)) if len(v) > 1 else [k + dx]
            ax.plot(xs, v, marker, ms=ps.MARKER_PT, ls="none", zorder=5, **ps.mark_fixed(col))
            ax.plot([k + dx - 0.13, k + dx + 0.13], [v.mean()] * 2, "-", lw=1.4, color=col, zorder=4,
                    solid_capstyle="butt")
        # every name ends on the second line, so the numbers line up on the third
        name = ACT_LABEL[act] if "\n" in ACT_LABEL[act] else "\n" + ACT_LABEL[act]
        ticks.append(f"{name}\n{s.mean():.2f}" if mine else f"{name}\nno runs")
    ax.set(xlim=(-0.5, len(ACTS) - 0.5), ylim=(0, ACT_N), xlabel="activation, success at the end",
           ylabel=ps.of_n(ps.LABELS["active"], ACT_N))
    ax.set_xticks(range(len(ACTS)))
    ax.set_xticklabels(ticks, fontsize=ps.TICK_PT, linespacing=1.25)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks([0, ACT_N / 4, ACT_N / 2, 3 * ACT_N / 4, ACT_N])
    ps.count_ticks(ax.yaxis)
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=ps.MARKER_PT, **ps.mark_fixed(ps.MUTED),
                              label="participation rule"),
                       Line2D([], [], ls="none", marker="s", ms=ps.MARKER_PT, **ps.mark_fixed(ps.MUTED),
                              label="modulation rule")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.0,
              columnspacing=1.0)
    ps.ygrid(ax)
    return out


def check_overlaps(fig):
    """Print every pair of drawn text artists whose rendered boxes overlap.

    Collected per axes: free text, titles, axis labels, legend entries and the tick labels whose
    tick lies inside the view; plus the figure's own texts.

    Args:
        fig: the finished figure.
    Returns:
        list of (text a, text b) pairs that overlap.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        leg = ax.get_legend()
        if leg is not None:
            texts += list(leg.get_texts())
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            for tick in axis.get_major_ticks():
                if lo <= tick.get_loc() <= hi and tick.label1.get_visible():
                    texts.append(tick.label1)
    texts = [t for t in texts if t.get_visible() and t.get_text().strip()]
    boxes = [(t, t.get_window_extent(rend)) for t in texts]
    hits = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(boxes) for b, bb in boxes[i + 1:]
            if ba.overlaps(bb) and ba.width > 0 and bb.width > 0]
    for a, b in hits:
        print(f"  !! text overlap: {a[:40]!r} <-> {b[:40]!r}")
    if not hits:
        print(f"  .. no overlapping text ({len(boxes)} text artists checked)")
    return hits


def draw(runs, rows, abl):
    """Lay out the five panels, draw them, check the text and save the figure.

    Args:
        runs: list from common.maze_load_runs with "pt" (read_points) on each; rows: the summarise
        rows; abl: list from load_ablation.
    Returns:
        str, the path of the PDF.
    """
    ps.setup()
    fig = ps.figure(FIG_H_MM)
    ax_a = ax_mm(fig, *LAYOUT_MM["a"])
    left, top, width, height = LAYOUT_MM["b0"]
    axes_b = [ax_mm(fig, left + k * B_STEP_MM, top, width, height) for k in range(len(SIZES))]
    ax_c, ax_d, ax_key, ax_e = (ax_mm(fig, *LAYOUT_MM[k]) for k in ("c", "d", "d_key", "e"))

    series = [r for r in runs if r["act"] == "relu"]
    x_max = int(max((r["probes"]["update"][-1] for r in series), default=2000))
    x_max = int(np.ceil(x_max / 500.0) * 500)
    panel_success(ax_a, series, x_max)
    empty_b = panel_active(axes_b, series, x_max)
    W, H = fig.get_figwidth() / ps.MM, fig.get_figheight() / ps.MM
    x_mid = (left + (len(SIZES) - 1) * B_STEP_MM + width + left) / 2.0
    fig.text(x_mid / W, 1.0 - (top + height + 7.5) / H, "PPO update", ha="center", va="top",
             fontsize=ps.LABEL_PT, color=ps.INK)
    fit = panel_size(ax_c, rows)
    abl_rows = panel_ablation(ax_d, ax_key, abl)
    acts = panel_activations(ax_e, rows)
    for ax, letter in ((ax_a, "a"), (axes_b[0], "b"), (ax_c, "c"), (ax_d, "d"), (ax_e, "e")):
        ps.panel_letter(ax, letter)

    print("Panel b: sizes with no run:", empty_b or "none")
    if fit["b"] is None:
        print("Panel c: no fit (fewer than two sizes with a learned run)")
    else:
        print(f"Panel c: end count ~ N^b, b = {fit['b']:.3f}, bootstrap 95% [{fit['lo']:.3f}, {fit['hi']:.3f}], "
              + "; ".join(f"N = {N}: mean {v.mean():.0f} (n = {len(v)})" for N, v in sorted(fit["cells"].items())))
    if fit["failed"]:
        print("Panel c: not learned (cross, excluded from means and fit):",
              ", ".join(f"N={N} s{s}" for N, s in fit["failed"]))
    print("Panel d:", "no ablation results" if not abl_rows else "")
    for w in abl_rows:
        print(f"  {w['label']}: {w['n_inactive']} of {w['N']} inactive; success intact {w['full']:.2f}; "
              f"with the inactive units removed: quietest first {w['quiet_at_inactive']:.2f}, "
              f"random {w['random_at_inactive']:.2f}, busiest first {w['busy_at_inactive']:.2f}")
    print("Panel e (N = %d):" % ACT_N)
    for act, (p, m, s) in acts.items():
        print(f"  {act:>10}: " + ("no runs" if not len(p) else
              f"p rule {', '.join(f'{v:.0f}' for v in p)}; modulation "
              f"{', '.join(f'{v:.0f}' for v in m) or '-'}; success {', '.join(f'{v:.2f}' for v in s)}"))
    check_overlaps(fig)
    return ps.save(fig, "fig_ed_maze")


def main():
    """Parse arguments, load the runs and the ablation results, print the table, draw the figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT, help="folder of <cell>/s<seed>/ maze run folders")
    ap.add_argument("--ablation", default=ABLATION_DIR, help="folder of rl_maze_ablate.py JSONs")
    ap.add_argument("--keep-cue", action="store_true",
                    help="also draw the runs with env.goal_cue true (dropped by default)")
    args = ap.parse_args()
    runs = maze_load_runs(args.root)
    if not args.keep_cue:
        n_cue = sum(r["cue"] for r in runs)
        runs = [r for r in runs if not r["cue"]]
        if n_cue:
            print(f"dropped {n_cue} goal-cue runs (--keep-cue draws them)")
    if not runs:
        raise SystemExit(f"no runs under {args.root}")
    for r in runs:
        r["pt"] = read_points(r)
    rows = summarise(runs)
    missing_cells(runs)
    abl = load_ablation(args.ablation)
    print(f"{len(abl)} ablation results under {args.ablation}")
    draw(runs, rows, abl)


if __name__ == "__main__":
    main()
