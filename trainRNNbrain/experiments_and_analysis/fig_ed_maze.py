#!/usr/bin/env python3
"""Extended Data figure: the maze task (reinforcement learning) shows the same loss of participation.

ROUND 19 REVIEW (PR #207, items 7a, 7b): in panel b a probe that reads 0 active units at success >= 0.5
(ZERO_READ_SUCCESS; N = 4,000 seed 3 at update 7,390, success 0.98) is drawn as a gap and printed
(masked_active); panel a's '0.9' sits outside the right spine.

QUESTION. Does a network trained by PPO on the maze (issue engellab/trainRNNbrain#3; paper package
ptolmachev/dead-relu-paper#182) leave units inactive the way the supervised tasks do: fewer active
units than N once the task is solved, a share of active units that falls with N, a count that keeps
falling after the task is learned, and the same loss under other activation functions? Figure 1
has no room for a fourth ablation sub-panel, so the maze gets this figure.

ROUND 19 (dead-relu-paper#201, #206; entries 2026-11-08_0915, 2026-11-08_2346, 2026-11-09_0213,
2026-11-09_1341). Every Maze network trains in 2,000-update stages until it is learned, cap 8,000,
with one budget per size: 4,000 updates at N = 250 to 2,000, 8,000 at N = 4,000. N = 4,000 seed 4
stopped at update 5,940 with non-finite weights (MAZE_STOPPED); its record is drawn to that update
and marked, and it enters no mean and no fit. The size series is every ReLU run of the suite: four
at N = 250, three at 500, six at 1,000 (the uniform seeds 1-3 and the G12 seeds 4-6), three at 2,000
and five at 4,000 (four learned). Issue #203: no two open markers of one axes overprint. Markers of
one group are spread sideways (`spread_x`) by at least SPREAD_MM between centres, every mean is an
ink tick beside its group, and fit lines run under the white insides of the open markers.

  (a) the clean probe's success rate against PPO updates, one thin line per ReLU run of the size
      series, colour by N, a filled mark at each run's "learned" probe (the first probe from which
      the success stays at or above the dashed 0.9 for MAZE_LEARNED_HOLD probes), a cross at the
      last finite probe of a stopped run; the key sits in its own axes below the panel;
  (b) active units against PPO updates, the same runs, one sub-panel per N with y from 0 to that N
      and x from 0 to that size's budget (4,000 or 8,000). The end-of-training markers of one size
      are spread to the left of the budget's end where they would overprint;
  (c) active units at "learned" (filled) and at the end of training (open) against N, one marker per
      learned run, the size mean as an ink tick beside each group, and a power law fitted to the size
      means of the end counts at N >= FIT_MIN_N (fig_paper_F2_size.power_fit's rule: a least-squares
      line through log mean count against log N; the bootstrap 95% interval resamples runs within
      each fitted size, as round19_size_exponent.py does). The end of training is each size's own
      final budget, so the budgets are unequal (4,000 updates at N <= 2,000, 8,000 at N = 4,000);
      the fit on equal 4,000-update budgets and the fits over every size are printed;
  (d) the ablation (rl_maze_ablate.py): success rate under the training noise as units are removed,
      quietest first (solid), busiest first (dotted) and in random order (dashed, band: range over
      the random orders), x as a share of N, a vertical line at each network's inactive count. An
      ablation is drawn when its run folder is a folder of a learned run's chain and the run was
      learned on its record cut at the last update of that folder (the networks of the round 18
      figure: N = 250 s1, 500 s1 and 1,000 s1, s2 at update 2,000; 2,000 s1-s3 and 4,000 s1 at
      update 4,000);
  (e) the activations at N = 1,000: active units under the participation rule (circles) and under
      the modulation rule (squares; std alone, saved as active_mod, the count that detects a
      saturated unit), per activation, with the read-out update and the mean success rate there
      under each name. ReLU (seeds 1-6), leaky ReLU and softplus are read at update 4,000, the
      equal budget of the N = 1,000 group; tanh and the shifted sigmoid at the standard settings
      were not continued past update 2,000 and are read there. Set apart, under "initial spectral
      radius 0.45": the shifted sigmoid (full runs, seeds 3-5) and ReLU (seeds 3-5; seed 5 was
      stopped by the rollback guard near update 1,700 and is a cross at 0) of RADIUS_ROOT, read at
      update 4,000 (entry 2026-11-09_0213), a per-activation change of the protocol.

READ-OUT POINTS (pre-registered on issue #3, in common): "learned" is the first probe from which the
clean probe's success stays at or above 0.9 for 5 probes, in a run whose last probe is also at or
above 0.9 (common.maze_learned_index), the maze counterpart of the floor crossing; "end" is the
chain's last probe. Marker fill follows the style sheet (paperstyle, dead-relu-paper#83): filled =
the run's own convergence point, open = a fixed iteration (the end of the budget); the two counting
rules of (e) differ by shape, both open.

COLOUR. One task, so one hue: TASK_MAZE, with N as a lightness ramp (N_SHADE). Text is INK or MUTED.

DATA. --root (default data/trained_RNNs_rl_maze/Maze_suite): the ReLU size series and tanh, sigmoid
(shifted), softplus (beta 25) and leaky_relu at N = 1,000 (slurm/rl_maze_suite_*.slurm);
common.maze_load_runs joins each run with its continuations. --radius-root (default
data/trained_RNNs_rl_maze/Maze_r19full): the initial-spectral-radius-0.45 runs. --ablation (default
data/rl_maze_ablation): the JSONs of rl_maze_ablate.py; a relative run_dir is resolved from the
repository root (#195).

OUTPUT. img/internal_figures/fig_ed_maze.{pdf,svg} (paperstyle.save, vector only, 165.1 mm wide,
FIG_H_MM tall) and, on stdout, one row per run, the fits, the ablation numbers, panel e's numbers
and the text-overlap check.

Run:  python trainRNNbrain/experiments_and_analysis/fig_ed_maze.py [--root DIR] [--radius-root DIR]
      [--ablation DIR] [--keep-cue]
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
RADIUS_ROOT = os.path.join(REPO, "data", "trained_RNNs_rl_maze", "Maze_r19full")   # radius 0.45 (round 19)
ABLATION_DIR = os.path.join(REPO, "data", "rl_maze_ablation")

# the suite's grid (slurm/rl_maze_suite_*.slurm): the ReLU size series and the activations
SIZES = (250, 500, 1000, 2000, 4000)
SEEDS = (1, 2, 3)
SIZE_SEEDS = {250: (1, 2, 3, 4), 500: (1, 2, 3), 1000: (1, 2, 3, 4, 5, 6), 2000: (1, 2, 3),
              4000: (1, 2, 3, 4, 5)}      # every ReLU run of the size series since round 19
ACT_N = 1000                      # the activations are compared at this size
ACTS = ("relu", "leaky_relu", "softplus", "tanh", "sigmoid")   # saved names, drawing order
ACT_SEEDS = (1, 2, 3)
ACT_LABEL = {"relu": "ReLU", "leaky_relu": "leaky\nReLU", "softplus": "softplus", "tanh": "tanh",
             "sigmoid": "shifted\nsigmoid"}   # at most two lines
EXPECTED = [(N, "relu", s) for N in SIZES for s in SIZE_SEEDS[N]] + \
           [(ACT_N, a, s) for a in ACTS if a != "relu" for s in ACT_SEEDS]
ACT_READ = 4000                   # panel e: the N = 1,000 group's equal budget (round 19 protocol)
ACT_READ_EARLY = 2000             # panel e: tanh and the shifted sigmoid were not continued past this
RADIUS_ACTS = ("sigmoid", "relu")  # panel e's right block: the runs of RADIUS_ROOT, drawing order
RADIUS = 0.45                     # their initial spectral radius (model.spectral_rad)

# Runs that the rollback guard stopped, or whose weights became non-finite, with the last update of
# their usable record. Entry 2026-11-08_2346: N = 4,000 seed 4 has non-finite weights after every
# update from 5,941 (probe success 0 and no active unit from 5,950); it is "stopped at update 5,940".
# A stopped run is drawn to that update and marked; it enters no mean and no fit.
MAZE_STOPPED = {(4000, "relu", 4): 5940}
# a probe with 0 active units at this success or above is a read-out artefact, masked in panel b
ZERO_READ_SUCCESS = 0.5

# N as a lightness ramp of the maze hue (ps.TASK_MAZE = #941862 is N = 2,000): light to dark
N_SHADE = {250: "#d29ebd", 500: "#bf74a1", 1000: "#a94681", 2000: ps.TASK_MAZE, 4000: "#681145"}

N_BOOT = 2000                     # bootstrap resamples for the exponent's interval (panel c)
FIT_MIN_N = 500                   # fig_paper_F2_size.FIT_MIN_N: the smallest size fitted (round 18, #201)
BOOT_SEED = 20261031              # fig_paper_F2_size.BOOT_SEED: one seed for every Maze fit
X_OFF = 1.17                      # panel c: the "learned" group centred at N / X_OFF, the "end" group at
                                  # N * X_OFF (log axis), so the two groups of one size do not meet
SPREAD_MM = 1.15                  # least distance between two marker centres of one group (#203):
                                  # a marker's outer rim lies 0.7 mm from its centre (MARKER_PT 3 pt
                                  # plus the ring and the rim), and the centre's pixel block needs
                                  # 0.15 mm of paper beyond it
TICK_MM = 1.6                     # the length of a mean tick beside a group
D_YLIM = (0.0, 1.14)              # panel d: room above success = 1
C_YLIM = (90, 3000)               # panel c: counts from 125 (N = 250) to 851 (N = 4,000)

# layout in millimetres from the canvas's top-left corner: (left, top, width, height)
FIG_H_MM = 155.0
B_STEP_MM = 21.6                  # panel b: one sub-panel every this many mm (14.5 wide, 7.1 gap)
LAYOUT_MM = {
    "a":  (12.5, 8.0, 32.5, 32.0),
    "a_key": (8.0, 49.5, 46.0, 9.0),    # the key of (a), below its x label and off the traces
    "b0": (57.0, 8.0, 14.5, 32.0),      # the first of the five sub-panels of (b)
    "c":  (12.5, 63.0, 52.0, 34.0),
    "d":  (79.0, 63.0, 46.0, 34.0),
    "d_key": (129.0, 63.0, 34.0, 20.0),  # the legend of (d), right of it and off the data
    "e":  (12.5, 114.0, 150.0, 27.0),
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


def cut_at(run, update):
    """Cut a run's joined record at an update: the probes and progress records up to it.

    Args:
        run: dict from common.maze_load_runs; update: int, the last update kept.
    Returns:
        a shallow copy of the run with "probes" (every array cut, p and p_update on their own grid)
        and "progress" cut at `update`.
    """
    pr = run["probes"]
    m = np.asarray(pr["update"]) <= update
    mp = np.asarray(pr["p_update"]) <= update if "p_update" in pr else None
    out = dict(run)
    # the probe arrays share the probe grid; the stored participation vectors (p, with their
    # updates p_update) have their own, sparser grid
    out["probes"] = {k: np.asarray(v)[mp if k in ("p", "p_update") and mp is not None else m]
                     for k, v in pr.items()}
    out["progress"] = [rec for rec in run["progress"] if rec["update"] <= update]
    return out


def mark_stopped(runs, stopped=MAZE_STOPPED):
    """Cut every stopped run at the last update of its usable record and flag it.

    Args:
        runs: list from common.maze_load_runs; stopped: dict (N, act, seed) -> last usable update.
    Returns:
        list of runs, each with "stopped" (the last usable update, or None); a stopped run's record
        ends at that update.
    """
    out = []
    for r in runs:
        u = stopped.get((r["N"], r["act"], r["seed"]))
        r = cut_at(r, u) if u is not None else dict(r)
        r["stopped"] = u
        out.append(r)
    return out


def summarise(runs):
    """One row per run, printed as a table: every number the figure draws.

    Args:
        runs: list from common.maze_load_runs, each with its read_points under "pt" and "stopped"
            (mark_stopped).
    Returns:
        list of dicts: N, act, seed, cue, updates (last update), learned, active_learned,
        active_end, mod_end, success_end, stopped (last usable update or None), learned_ok (bool:
        not stopped and learned by common.maze_learned_index, whose rule includes a last probe at or
        above MAZE_LEARNED_SUCCESS).
    """
    rows = []
    for r in runs:
        p = r["pt"]
        rows.append(dict(N=r["N"], act=r["act"], seed=r["seed"], cue=r["cue"],
                         updates=int(r["progress"][-1]["update"]), learned=p["learned"],
                         active_learned=p["active_learned"], active_end=p["active_end"],
                         mod_end=p["mod_end"], success_end=p["success_end"], stopped=r.get("stopped"),
                         learned_ok=r.get("stopped") is None and p["learned_i"] is not None))
    fmt = lambda v: "-" if v is None else str(v)
    print(f"{'N':>5} {'act':>10} {'seed':>4} {'cue':>5} {'updates':>7} {'learned':>7} {'act@learn':>9} "
          f"{'act@end':>7} {'share%':>6} {'mod@end':>7} {'succ@end':>8}  status")
    for w in rows:
        status = (f"stopped at update {w['stopped']:,}" if w["stopped"] is not None
                  else ("learned" if w["learned_ok"] else "not learned"))
        print(f"{w['N']:>5} {w['act']:>10} {w['seed']:>4} {str(w['cue']):>5} {w['updates']:>7} "
              f"{fmt(w['learned']):>7} {fmt(w['active_learned']):>9} {w['active_end']:>7} "
              f"{100 * w['active_end'] / w['N']:>6.1f} {fmt(w['mod_end']):>7} {w['success_end']:>8.2f}  {status}")
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


def power_fit(cells, min_n=FIT_MIN_N):
    """Least-squares line through log(mean active count) against log N, over the sizes N >= min_n.

    Args:
        cells: dict N -> array of active counts (at least two sizes N >= min_n); min_n: int, the
            smallest size fitted (FIT_MIN_N; 0 fits every size, the second reading).
    Returns:
        (exponent b, prefactor A) with mean count ~ A N^b.
    """
    Ns = np.array([N for N in sorted(cells) if N >= min_n], float)
    ys = np.array([cells[N].mean() for N in sorted(cells) if N >= min_n])
    b, loga = np.polyfit(np.log(Ns), np.log(ys), 1)
    return float(b), float(np.exp(loga))


def boot_exponent(cells, rng, min_n=FIT_MIN_N):
    """Bootstrap 95% interval of the fitted exponent, resampling runs within each fitted size.

    Only the sizes N >= min_n are resampled, as the round 18 and 19 read-outs do
    (round19_size_exponent.fit), so the interval equals the notebook's for the same seed.

    Args:
        cells: dict N -> array of active counts; rng: numpy Generator; min_n: as power_fit.
    Returns:
        (2.5th, 97.5th percentile) of the exponent over N_BOOT resamples.
    """
    fitted = {N: v for N, v in cells.items() if N >= min_n}
    bs = [power_fit({N: rng.choice(v, len(v)) for N, v in fitted.items()}, min_n)[0] for _ in range(N_BOOT)]
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


def spread_x(ax, x0, ys, placed, d_mm=SPREAD_MM, side="both", max_slots=12):
    """Horizontal drawing positions for one group of markers, so that no two centres are closer than d_mm.

    The markers are placed in order of value. Each takes the first slot, in page millimetres from
    x0, of 0, -d, +d, -2d, +2d, ... (side "both") or 0, -d, -2d, ... (side "left") at which its
    centre lies at least d_mm from every centre in `placed` (markers already drawn in the same axes,
    of any group). The axes' position, scales and limits must be final. The values are not moved.

    Args:
        ax: the Axes; x0: float, the group's centre in data units; ys: sequence of floats (data
        units); placed: list of (x, y) display points, extended in place with this group's centres;
        d_mm: float; side: "both" or "left"; max_slots: int, slots tried per side.
    Returns:
        numpy array of x positions in data units, in the order of `ys`.
    """
    fig = ax.figure
    d_px = d_mm * fig.dpi / 25.4
    tr = ax.transData
    ys = np.asarray(ys, float)
    xs = np.full(len(ys), float(x0))
    steps = [0] + [k for j in range(1, max_slots) for k in ((-j, j) if side == "both" else (-j,))]
    for i in np.argsort(ys, kind="stable"):
        px0, py = tr.transform((x0, ys[i]))
        for k in steps:
            px = px0 + k * d_px
            if all(np.hypot(px - qx, py - qy) >= d_px - 1e-9 for qx, qy in placed):
                break
        placed.append((px, py))
        xs[i] = tr.inverted().transform((px, py))[0]
    return xs


def mean_tick(ax, xs, value, col=None):
    """Draw a group's mean as a short ink tick right of its rightmost marker, under every marker.

    Args:
        ax: the Axes; xs: the group's x positions (data units); value: the mean (data units);
        col: tick colour (INK by default).
    Returns:
        None.
    """
    fig = ax.figure
    tr = ax.transData
    px = max(tr.transform((x, value))[0] for x in xs)
    py = tr.transform((xs[0], value))[1]
    gap, length = 0.9 * fig.dpi / 25.4, TICK_MM * fig.dpi / 25.4
    x_a = tr.inverted().transform((px + gap, py))[0]
    x_b = tr.inverted().transform((px + gap + length, py))[0]
    ax.plot([x_a, x_b], [value, value], "-", lw=0.9, color=col or ps.INK, zorder=4,
            solid_capstyle="butt")


def stopped_mark(ax, x, y, col, **kw):
    """Draw the cross that marks a stopped run (the rollback guard, or non-finite weights).

    Args:
        ax: the Axes; x, y: the mark's position (data units); col: its colour; kw: passed to plot.
    Returns:
        the Line2D drawn.
    """
    return ax.plot([x], [y], "x", ms=ps.MARKER_PT + 1.0, mec=col, mew=1.0, ls="none", zorder=7, **kw)[0]


def panel_success(ax, ax_key, series, x_max):
    """Panel (a): the clean probe's success rate against PPO updates, one line per run.

    A filled mark at each learned run's learned probe; a stopped run (MAZE_STOPPED) is drawn to its
    last usable update and ends in a cross.

    Args:
        ax: axes; ax_key: a blank axes below it for the key (one entry per size, one for the
        learned mark and one for a stopped run when there is one); series: the ReLU runs (with "pt"
        and "stopped"); x_max: int, the x limit.
    Returns:
        None.
    """
    any_stopped = False
    for r in series:
        col, pr, li = shade(r["N"]), r["probes"], r["pt"]["learned_i"]
        ax.plot(pr["update"], pr["probe_success"], "-", lw=ps.THIN_LW, color=col, zorder=3)
        if r.get("stopped") is not None:
            any_stopped = True
            stopped_mark(ax, pr["update"][-1], pr["probe_success"][-1], ps.INK)
        elif li is not None:
            ax.plot([pr["update"][li]], [pr["probe_success"][li]], "o", ms=ps.MARKER_PT, zorder=5,
                    **ps.mark_floor(col))
    ax.axhline(MAZE_LEARNED_SUCCESS, ls=(0, (4, 3)), lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    # round 20 (dead-relu-paper #212, Reviewer S, figures item 4): the level is named by its own tick label,
    # so no text sits beside the 21 traces (until round 20 a "0.9" just outside the right spine, PR #207
    # review, item 7b); and the probe by the paper's name for it, the noise-free probe (item 7)
    time_axis(ax, x_max)
    ax.set(ylim=(0, 1.02), xlabel="PPO update", ylabel="success rate (noise-free probe)")
    ax.set_yticks([0, 0.5, MAZE_LEARNED_SUCCESS, 1.0])
    ax.set_yticklabels(["0", "0.5", f"{MAZE_LEARNED_SUCCESS:g}", ""])    # "1" would touch "0.9"
    sizes = sorted({r["N"] for r in series})
    handles = [Line2D([], [], color=shade(N), lw=1.2, label=f"$N$ = {ps.fmt_count(N)}") for N in sizes]
    handles.append(Line2D([], [], ls="none", marker="o", ms=ps.MARKER_PT, **ps.mark_floor(ps.MUTED),
                          label="learned"))
    if any_stopped:
        handles.append(Line2D([], [], ls="none", marker="x", ms=ps.MARKER_PT + 1.0, mec=ps.INK, mew=1.0,
                              label="stopped (non-finite weights)"))
    # the key sits below the panel: with 21 traces, every corner inside the axes has data
    ps.blank(ax_key)
    ax_key.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3,
                  handlelength=1.6, columnspacing=1.0, borderaxespad=0.0, borderpad=0.0)
    ps.ygrid(ax)


def masked_active(run):
    """A run's probe active counts for drawing, with impossible zero reads masked.

    A probe that reports 0 active units while the agent still succeeds (success at or above
    ZERO_READ_SUCCESS) cannot be a real state of the network: with no active unit the read-out is
    zero. Such a probe is drawn as a gap and printed (PR #207 review, item 7a: N = 4,000 seed 3 has
    one, at update 7,390, success 0.98). A stopped run's zero reads at success 0 are not masked.

    Args:
        run: dict from common.maze_load_runs (with "N" and "seed").
    Returns:
        np.ndarray of float, the probe active counts with the masked probes set to NaN.
    """
    pr = run["probes"]
    act = np.asarray(pr["active"], dtype=float)
    bad = (act == 0) & (np.asarray(pr["probe_success"], dtype=float) >= ZERO_READ_SUCCESS)
    for i in np.flatnonzero(bad):
        print(f"  panel b: masked probe N = {run['N']} seed {run.get('seed')} update {int(pr['update'][i])}: "
              f"0 active at success {float(pr['probe_success'][i]):.2f}")
    act[bad] = np.nan
    return act


def panel_active(axes, series):
    """Panel (b): active units against PPO updates, one sub-panel per size, y from 0 to N.

    Each sub-panel's x axis runs to its size's budget (the longest record of the size, rounded up to
    500). A filled mark at the "learned" probe and an open mark at the last probe of every run that
    was not stopped; a stopped run ends in a cross. The open marks of one size are spread to the left
    (spread_x) where they would overprint (#203).

    Args:
        axes: one Axes per entry of SIZES, left to right; series: the ReLU runs (with "pt" and
        "stopped").
    Returns:
        (list of the sizes with no run, dict N -> x limit).
    """
    empty, x_lims = [], {}
    for ax, N in zip(axes, SIZES):
        runs = [r for r in series if r["N"] == N]
        col = shade(N)
        x_max = int(np.ceil(max((r["probes"]["update"][-1] for r in runs), default=2000) / 500.0) * 500)
        x_lims[N] = x_max
        time_axis(ax, x_max)
        ax.set(ylim=(0, N), title=f"$N$ = {ps.fmt_count(N)}")
        placed, ends = [], []
        for r in runs:
            pr, li = r["probes"], r["pt"]["learned_i"]
            ax.plot(pr["update"], masked_active(r), "-", lw=ps.THIN_LW, color=col, zorder=3)
            if r.get("stopped") is not None:
                stopped_mark(ax, pr["update"][-1], pr["active"][-1], ps.INK, clip_on=False)
                continue
            if li is not None:
                ax.plot([pr["update"][li]], [pr["active"][li]], "o", ms=ps.MARKER_PT, zorder=5,
                        **ps.mark_floor(col))
                placed.append(tuple(ax.transData.transform((pr["update"][li], pr["active"][li]))))
            ends.append((pr["update"][-1], pr["active"][-1]))
        for x_end in sorted({e[0] for e in ends}):
            ys = [y for x, y in ends if x == x_end]
            xs = spread_x(ax, x_end, ys, placed, side="left")
            ax.plot(xs, ys, "o", ms=ps.MARKER_PT, ls="none", zorder=5, clip_on=False, **ps.mark_fixed(col))
        if not runs:
            empty.append(N)
            ax.text(0.5, 0.5, "no runs", transform=ax.transAxes, ha="center", va="center",
                    fontsize=ps.TICK_PT, color=ps.MUTED)
        ax.set_yticks([0, N / 2, N])
        ps.count_ticks(ax.yaxis)
        ps.ygrid(ax)
    axes[0].set_ylabel(ps.LABELS["active"])
    return empty, x_lims


def counts_at(run, update):
    """A run's state on its record cut at an update (the equal-budget read).

    Args:
        run: dict from common.maze_load_runs; update: int.
    Returns:
        (learned on the cut record, active count at its last probe), or (False, None) when the
        record does not reach `update`.
    """
    pr = run["probes"]
    if pr["update"][-1] < update:
        return False, None
    m = pr["update"] <= update
    return maze_learned_index(pr["probe_success"][m]) is not None, int(pr["active"][m][-1])


def panel_size(ax, rows, runs):
    """Panel (c): active units at "learned" and at the end against N, with the power-law fit.

    One marker per learned ReLU run: a filled circle in the group centred at N / X_OFF for the
    count at "learned", an open circle in the group centred at N * X_OFF for the count at the end of
    training (the size's own final budget); the markers of a group are spread sideways (spread_x)
    and each group's mean is an ink tick beside it (mean_tick). A stopped run is left out (it is
    marked in a and b). The fit is a line through the size means of the end counts over N >=
    FIT_MIN_N (power_fit), drawn under the markers; its exponent is in the legend. The diagonal
    marks "every unit active". The equal-budget fit (every learned run's count at update 4,000, on
    its record cut there) and the fits over every size are computed and returned, not drawn.

    Args:
        ax: axes; rows: the summarise rows; runs: the runs (with "stopped"), for the cut at 4,000.
    Returns:
        dict: b, lo, hi (end-of-training fit and its bootstrap 95% interval; None without two fitted
        sizes), b_all, lo_all, hi_all (the same over every size), cells (N -> end counts), cells_l
        (N -> learned-point counts), learned_fit (b, lo, hi at the learned point, N >= FIT_MIN_N),
        u4000 (b, lo, hi, cells of the equal-budget read at update 4,000), stopped (list of (N, seed)).
    """
    rel = [w for w in rows if w["act"] == "relu" and w["learned_ok"]]
    stopped = [(w["N"], w["seed"]) for w in rows if w["act"] == "relu" and w["stopped"] is not None]
    x_lo, x_hi = SIZES[0] / 1.45, SIZES[-1] * 1.45
    ax.set(xscale="log", yscale="log", xlim=(x_lo, x_hi), ylim=C_YLIM)
    cells, cells_l = {}, {}
    for w in rel:
        cells.setdefault(w["N"], []).append(w["active_end"])
        cells_l.setdefault(w["N"], []).append(w["active_learned"])
    cells = {N: np.array(v, float) for N, v in sorted(cells.items())}
    cells_l = {N: np.array(v, float) for N, v in sorted(cells_l.items())}
    placed = []
    for N in sorted(cells):
        col = shade(N)
        xs = spread_x(ax, N / X_OFF, cells_l[N], placed)
        ax.plot(xs, cells_l[N], "o", ms=ps.MARKER_PT, ls="none", zorder=5, **ps.mark_floor(col))
        mean_tick(ax, xs, cells_l[N].mean())
        xs = spread_x(ax, N * X_OFF, cells[N], placed)
        ax.plot(xs, cells[N], "o", ms=ps.MARKER_PT, ls="none", zorder=5, **ps.mark_fixed(col))
        mean_tick(ax, xs, cells[N].mean())
    fit = dict(b=None, lo=None, hi=None, cells=cells, cells_l=cells_l, stopped=stopped)
    handles = [Line2D([], [], ls="none", marker="o", ms=ps.MARKER_PT, **ps.mark_floor(ps.MUTED),
                      label="learned"),
               Line2D([], [], ls="none", marker="o", ms=ps.MARKER_PT, **ps.mark_fixed(ps.MUTED),
                      label="end of training"),
               Line2D([], [], color=ps.INK, lw=0.9, label="mean")]
    fitted = [N for N in cells if N >= FIT_MIN_N]
    if len(fitted) >= 2:
        b, A = power_fit(cells)
        lo, hi = boot_exponent(cells, np.random.default_rng(BOOT_SEED))
        b_all = power_fit(cells, 0)[0]
        lo_all, hi_all = boot_exponent(cells, np.random.default_rng(BOOT_SEED), 0)
        bl = power_fit(cells_l)[0]
        ll, hl = boot_exponent(cells_l, np.random.default_rng(BOOT_SEED))
        fit.update(b=b, lo=lo, hi=hi, b_all=b_all, lo_all=lo_all, hi_all=hi_all, learned_fit=(bl, ll, hl))
        xf = np.array([min(fitted) * 0.8, max(fitted) * 1.25])
        # under the markers: the white insides of the open markers keep their centres paper (#203)
        ax.plot(xf, A * xf ** b, "-", lw=0.8, color=ps.INK, zorder=3)
        handles.append(Line2D([], [], color=ps.INK, lw=0.8, label=f"fit to the end, $b$ = {b:.2f}"))
    # the equal-budget read: every ReLU run learned on its record cut at update 4,000
    c4 = {}
    for r in runs:
        if r["act"] != "relu" or r.get("stopped") is not None:
            continue
        ok, a = counts_at(r, 4000)
        if ok:
            c4.setdefault(r["N"], []).append(a)
    c4 = {N: np.array(v, float) for N, v in sorted(c4.items())}
    if sum(N >= FIT_MIN_N for N in c4) >= 2:
        fit["u4000"] = (power_fit(c4)[0], *boot_exponent(c4, np.random.default_rng(BOOT_SEED)), c4)
    nn = np.array([x_lo, x_hi])
    ax.plot(nn, nn, "-", lw=ps.THIN_LW, color=ps.FAINT, zorder=2)
    x_lab = 280.0
    ax.text(x_lab, x_lab * 1.1, "every unit active", rotation=45, transform_rotates_text=True,
            fontsize=ps.TICK_PT, color=ps.MUTED, ha="left", va="bottom", rotation_mode="anchor")
    ax.set_xticks(SIZES)
    ax.xaxis.set_minor_locator(NullLocator())
    ps.count_ticks(ax.xaxis)
    ax.set_yticks([t for t in (100, 200, 500, 1000, 2000) if C_YLIM[0] <= t <= C_YLIM[1]])
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
    percent, labels on the axes overlapped). The legend sits in `ax_key`, right of the panel and off
    the data, because a network with more than half its units inactive runs its quietest-first curve
    at success 1 across the top of the axes.

    Args:
        ax: axes; ax_key: a blank axes right of it for the legend; abl: list from load_ablation.
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
                  labels=["least to most participating", "random orders (band: range)",
                          "most to least participating", "inactive units"],
                  loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1, handlelength=1.6,
                  borderaxespad=0.0, borderpad=0.0)
    ps.ygrid(ax)
    return rows


def act_columns(suite, radius_runs):
    """The columns of panel (e): one per activation and read point, the radius-0.45 block last.

    Args:
        suite: the suite's runs (with "stopped"); radius_runs: the runs of RADIUS_ROOT.
    Returns:
        list of dicts: key, label (tick text, two name lines), block ("default" or "radius"), read
        (the update read), runs (finished runs read at that update), stopped (number of runs of the
        column stopped before it), p, m (participation-rule and modulation-rule counts), s (success).
    """
    cols = []

    def make(key, name, block, runs, read):
        """One column from its runs, each read at the last probe at or before `read`."""
        keep, n_stop = [], 0
        for r in runs:
            if r.get("stopped") is not None or r["probes"]["update"][-1] < read:
                n_stop += 1
                continue
            keep.append(cut_at(r, read))
        pts = [read_points(r) for r in keep]
        cols.append(dict(key=key, label=name, block=block, read=read, runs=keep, stopped=n_stop,
                         p=np.array([q["active_end"] for q in pts], float),
                         m=np.array([q["mod_end"] for q in pts if q["mod_end"] is not None], float),
                         s=np.array([q["success_end"] for q in pts], float)))

    for act in ACTS:
        runs = [r for r in suite if r["N"] == ACT_N and r["act"] == act]
        read = ACT_READ if act in ("relu", "leaky_relu", "softplus") else ACT_READ_EARLY
        make(act, ACT_LABEL[act], "default", runs, read)
    for act in RADIUS_ACTS:
        runs = [r for r in radius_runs if r["N"] == ACT_N and r["act"] == act]
        make(f"{act}_rad", ACT_LABEL[act], "radius", runs, ACT_READ)
    return cols


def panel_activations(ax, cols):
    """Panel (e): active units per activation at N = ACT_N, two counting rules.

    Per column, open circles (participation rule) left of the column's centre and open squares
    (modulation rule) right of it, each group spread sideways (spread_x) with its mean as an ink tick
    beside it; a run stopped before the read is a cross at 0 on the column's centre. Under every
    column, its name, the update read and the mean success rate there. The columns of RADIUS_ROOT
    (initial spectral radius 0.45) are set apart right of a thin separator, under their own header.

    Args:
        ax: axes; cols: list from act_columns.
    Returns:
        list of the columns with their x positions (key "x").
    """
    col = shade(ACT_N)
    xs_col, x = [], 0.0
    for k, c in enumerate(cols):
        if k and c["block"] != cols[k - 1]["block"]:
            x += 0.7                                  # the gap between the two blocks
        xs_col.append(x)
        x += 1.0
    ax.set(xlim=(-0.6, xs_col[-1] + 0.6), ylim=(0, 1.08 * ACT_N))
    placed, ticks = [], []
    for c, xc in zip(cols, xs_col):
        c["x"] = xc
        for dx, v, marker in ((-0.2, c["p"], "o"), (0.2, c["m"], "s")):
            if not len(v):
                continue
            xs = spread_x(ax, xc + dx, v, placed)
            ax.plot(xs, v, marker, ms=ps.MARKER_PT, ls="none", zorder=5, **ps.mark_fixed(col))
            mean_tick(ax, xs, v.mean())
        if c["stopped"]:
            stopped_mark(ax, xc, 0, ps.INK, clip_on=False)
        stop_txt = f", {c['stopped']} stopped" if c["stopped"] else ""
        succ = f"{c['s'].mean():.2f}" if len(c["s"]) else "-"
        name = c["label"] if "\n" in c["label"] else c["label"] + "\n"
        ticks.append(f"{name}\nupdate {ps.fmt_count(c['read'])}\nsuccess {succ}{stop_txt}")
    sep = [(xs_col[k - 1] + xs_col[k]) / 2 for k in range(1, len(cols)) if cols[k]["block"] != cols[k - 1]["block"]]
    for xsep in sep:
        ax.axvline(xsep, color=ps.FAINT, lw=ps.THIN_LW, zorder=1)
    blocks = [("default", "initial spectral radius 1.2 (default)"),
              ("radius", f"initial spectral radius {RADIUS:g}")]
    for blk, text in blocks:
        xb = [c["x"] for c in cols if c["block"] == blk]
        if xb:
            ax.text((min(xb) + max(xb)) / 2, 1.0, text, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=ps.TICK_PT, color=ps.MUTED)
    ax.set_xticks(xs_col)
    ax.set_xticklabels(ticks, fontsize=ps.TICK_PT, linespacing=1.2)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks([0, ACT_N / 2, ACT_N])
    ps.count_ticks(ax.yaxis)
    ax.set_ylabel(ps.of_n(ps.LABELS["active"], ACT_N))
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=ps.MARKER_PT, **ps.mark_fixed(ps.MUTED),
                              label="participation rule"),
                       Line2D([], [], ls="none", marker="s", ms=ps.MARKER_PT, **ps.mark_fixed(ps.MUTED),
                              label="modulation rule"),
                       Line2D([], [], color=ps.INK, lw=0.9, label="mean"),
                       Line2D([], [], ls="none", marker="x", ms=ps.MARKER_PT + 1.0, mec=ps.INK, mew=1.0,
                              label="stopped")],
              loc="lower right", bbox_to_anchor=(1.0, 1.09), ncol=4, borderaxespad=0.0,
              columnspacing=1.2, handlelength=1.4)
    ps.ygrid(ax)
    return cols


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


def folder_last_update(folder):
    """The last update recorded in one run folder (its own Progress file, not the chain's).

    Args:
        folder: str, a run folder.
    Returns:
        int, the update of the folder's last progress record.
    """
    with open(sorted(glob.glob(os.path.join(folder, "*Progress.json")))[0]) as fh:
        return int(json.load(fh)[-1]["update"])


def ablations_to_draw(abl, runs):
    """The ablations of panel (d): of a learned run, at a point where that run was learned.

    An ablation is drawn when its run folder (run_dir, resolved from the repository root when
    relative, #195) is a folder of a run's chain and the run, on its record cut at the last update
    of that folder, is learned (common.maze_learned_index) and not stopped. The others are listed.

    Args:
        abl: list from load_ablation; runs: the suite's runs (with "stopped").
    Returns:
        list of the ablation dicts drawn, each with "update" (the last update of its folder) added.
    """
    in_chain = {os.path.normpath(d): r for r in runs for d in r["chain"]}
    kept = []
    for a in abl:
        d = a["run_dir"] if os.path.isabs(a["run_dir"]) else os.path.join(REPO, a["run_dir"])
        d = os.path.normpath(d)
        r = in_chain.get(d)
        if r is None:
            print(f"  panel d: {a['label']} is not in a run's chain under --root; not drawn")
            continue
        u = folder_last_update(d)
        ok, _ = counts_at(r, u)
        if r.get("stopped") is not None or not ok:
            print(f"  panel d: {a['label']} (update {u:,}) was not learned at that update; not drawn")
            continue
        kept.append(dict(a, update=u))
    return kept


def draw(runs, rows, abl, radius_runs):
    """Lay out the five panels, draw them, check the text and save the figure.

    Args:
        runs: list from common.maze_load_runs with "pt" (read_points) and "stopped" on each; rows:
        the summarise rows; abl: the ablations to draw (ablations_to_draw); radius_runs: the runs of
        RADIUS_ROOT (with "stopped").
    Returns:
        str, the path of the PDF.
    """
    ps.setup()
    fig = ps.figure(FIG_H_MM)
    ax_a, ax_a_key = ax_mm(fig, *LAYOUT_MM["a"]), ax_mm(fig, *LAYOUT_MM["a_key"])
    left, top, width, height = LAYOUT_MM["b0"]
    axes_b = [ax_mm(fig, left + k * B_STEP_MM, top, width, height) for k in range(len(SIZES))]
    ax_c, ax_d, ax_key, ax_e = (ax_mm(fig, *LAYOUT_MM[k]) for k in ("c", "d", "d_key", "e"))

    series = [r for r in runs if r["act"] == "relu"]
    x_max = int(max((r["probes"]["update"][-1] for r in series), default=2000))
    x_max = int(np.ceil(x_max / 500.0) * 500)
    panel_success(ax_a, ax_a_key, series, x_max)
    empty_b, x_lims = panel_active(axes_b, series)
    W, H = fig.get_figwidth() / ps.MM, fig.get_figheight() / ps.MM
    x_mid = (left + (len(SIZES) - 1) * B_STEP_MM + width + left) / 2.0
    fig.text(x_mid / W, 1.0 - (top + height + 7.5) / H, "PPO update", ha="center", va="top",
             fontsize=ps.LABEL_PT, color=ps.INK)
    fit = panel_size(ax_c, rows, runs)
    abl_rows = panel_ablation(ax_d, ax_key, abl)
    cols = panel_activations(ax_e, act_columns(runs, radius_runs))
    for ax, letter in ((ax_a, "a"), (axes_b[0], "b"), (ax_c, "c"), (ax_d, "d"), (ax_e, "e")):
        ps.panel_letter(ax, letter)

    print("Panel b: sizes with no run:", empty_b or "none", "; x limits:", x_lims)
    fmt_cells = lambda cc: "; ".join(f"N = {N}: mean {v.mean():.1f} (n = {len(v)}; {', '.join(f'{x:.0f}' for x in v)})"
                                     for N, v in sorted(cc.items()))
    print("Panel c, learned point:", fmt_cells(fit["cells_l"]))
    print("Panel c, end of training (each size's final budget):", fmt_cells(fit["cells"]))
    if fit["b"] is not None:
        bl, ll, hl = fit["learned_fit"]
        print(f"Panel c: learned-point fit at N >= {FIT_MIN_N}: b = {bl:.3f} [{ll:.3f}, {hl:.3f}] (Fig. 2a's; not drawn here)")
        print(f"Panel c: end count ~ N^b over N >= {FIT_MIN_N} (drawn; unequal budgets), b = {fit['b']:.3f}, "
              f"bootstrap 95% [{fit['lo']:.3f}, {fit['hi']:.3f}]; over every size, b = {fit['b_all']:.3f} "
              f"[{fit['lo_all']:.3f}, {fit['hi_all']:.3f}]")
    if "u4000" in fit:
        b4, l4, h4, c4 = fit["u4000"]
        print(f"Panel c (printed, not drawn): update 4,000 on equal budgets, learned there, N >= {FIT_MIN_N}: "
              f"b = {b4:.3f} [{l4:.3f}, {h4:.3f}]; " + fmt_cells(c4))
    if fit["stopped"]:
        print("Panel c: stopped (left out):", ", ".join(f"N={N} s{s}" for N, s in fit["stopped"]))
    print("Panel d:", "no ablation results" if not abl_rows else "")
    for w, a in zip(abl_rows, abl):
        print(f"  {w['label']} (update {a['update']:,}): {w['n_inactive']} of {w['N']} inactive; success intact {w['full']:.2f}; "
              f"with the inactive units removed: quietest first {w['quiet_at_inactive']:.2f}, "
              f"random {w['random_at_inactive']:.2f}, busiest first {w['busy_at_inactive']:.2f}")
    print("Panel e (N = %d):" % ACT_N)
    for c in cols:
        seeds = ", ".join(str(r["seed"]) for r in c["runs"])
        print(f"  {c['block']:>7} {c['key']:>11} at update {c['read']:,} (seeds {seeds}; {c['stopped']} stopped): "
              f"p rule {', '.join(f'{v:.0f}' for v in c['p']) or '-'} (mean {c['p'].mean() if len(c['p']) else float('nan'):.0f}); "
              f"modulation {', '.join(f'{v:.0f}' for v in c['m']) or '-'}; "
              f"success {', '.join(f'{v:.2f}' for v in c['s'])} (mean {c['s'].mean() if len(c['s']) else float('nan'):.3f})")
    check_overlaps(fig)
    return ps.save(fig, "fig_ed_maze")


def load_suite(root, keep_cue=False):
    """The maze runs of a root, with the goal-cue runs dropped, stopped runs cut, and read points.

    Args:
        root: str; keep_cue: bool, keep the runs with env.goal_cue true.
    Returns:
        list of runs (common.maze_load_runs) with "stopped" and "pt" (read_points).
    """
    runs = maze_load_runs(root)
    if not keep_cue:
        n_cue = sum(r["cue"] for r in runs)
        runs = [r for r in runs if not r["cue"]]
        if n_cue:
            print(f"dropped {n_cue} goal-cue runs under {root} (--keep-cue draws them)")
    runs = mark_stopped(runs)
    for r in runs:
        r["pt"] = read_points(r)
    return runs


def mark_unfinished(runs):
    """Flag a radius-0.45 run whose record ends before its stage's budget as stopped (the guard).

    Every stage is 2,000 updates, so a record that ends below a multiple of 2,000 was stopped by the
    rollback guard (entry 2026-11-09_0213: ReLU seed 5 near update 1,700).

    Args:
        runs: list from load_suite.
    Returns:
        the same list, "stopped" set to the last update for such runs.
    """
    for r in runs:
        last = int(r["progress"][-1]["update"])
        if last % 2000:
            r["stopped"] = last
    return runs


def main():
    """Parse arguments, load the runs and the ablation results, print the table, draw the figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT, help="folder of <cell>/s<seed>/ maze run folders")
    ap.add_argument("--radius-root", default=RADIUS_ROOT, help="the initial-spectral-radius-0.45 runs")
    ap.add_argument("--ablation", default=ABLATION_DIR, help="folder of rl_maze_ablate.py JSONs")
    ap.add_argument("--keep-cue", action="store_true",
                    help="also draw the runs with env.goal_cue true (dropped by default)")
    args = ap.parse_args()
    runs = load_suite(args.root, args.keep_cue)
    if not runs:
        raise SystemExit(f"no runs under {args.root}")
    rows = summarise(runs)
    missing_cells(runs)
    radius_runs = mark_unfinished(load_suite(args.radius_root)) if os.path.isdir(args.radius_root) else []
    print(f"{len(radius_runs)} runs under {args.radius_root}")
    for r in radius_runs:
        print(f"  {r['act']} s{r['seed']}: last update {r['progress'][-1]['update']:,}"
              + (f", stopped" if r.get("stopped") is not None else ""))
    abl = load_ablation(args.ablation)
    print(f"{len(abl)} ablation results under {args.ablation}")
    draw(runs, rows, ablations_to_draw(abl, runs), radius_runs)


if __name__ == "__main__":
    main()
