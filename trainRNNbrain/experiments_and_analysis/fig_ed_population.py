#!/usr/bin/env python3
"""
Extended Data Fig. 2 (dead-relu-paper #56): population statistics over all units against the same
statistics over the active units only, every network drawn.

Draws data/population_stats.json, written by population_stats.py (notebook entries
2026-10-31_2212_population-stats-rule.md and 2026-10-31_2220_population-stats-results.md), and, for
panel a, data/connectivity_confirm.json, written by connectivity_confirm.py (notebook entries
2026-11-02_2218_connectivity-confirm-rule.md and 2026-11-02_2232_connectivity-confirm-result.md).
Nothing is computed here beyond reading those files.

  (a) s.d. of the recurrent weights (W1) of the 39 networks of the confirmatory test, over all units
      against over the active units (the active-active submatrix), with the range of the same
      statistic over 200 random sets of as many units as are active (round 12, r12-weights; this
      panel replaced the participation ratio of the rate covariance)
  (b) participation ratio after dividing each unit's rates by their standard deviation
  (c) selective (task marker) and mixed-selective (x) unit counts. Every marker is filled or a
      cross, never open: the weights are the floor-crossing snapshots, and fill means the floor
      crossing in every figure (dead-relu-paper#83), so selective and mixed-selective differ by shape
  (d) the same as shares: count / 1,000 against count / active count
  (e) k-means class count k* chosen by silhouette
  (f) class shares of matched classes, one point per matched pair

Round 20 (dead-relu-paper #212, Reviewer S, structure item 5 and figures item 5): the Maze networks join
every panel, so the figure uses the paper's three tasks; DMTS stays as a fourth series, labelled as the fourth
task. Maze, panels b-f: the two learned networks of data/population_stats_maze.json (rl_maze_population_stats.py;
the Maze networks of Figure 1e, seeds 1 and 2 of N = 1,000 at update 2,000, the learned agents only as Figure
1e draws them); panel a: the same two networks in data/round18/g13_maze_weight_spread.json (notebook entry
2026-11-07_0235, G13), drawn as one marker at their mean with "2" beside it, because their values lie within a
quarter of a marker of each other. Maze markers are OPEN diamonds (update 2,000, a fixed budget). The figure is
drawn at its print size, 165.1 x FIG_H_MM mm, so its text prints at the house sizes (6 and 6.5 pt; until round
20 it was drawn 112 mm tall and shrunk to 0.78 by the float, which printed its text at 4.7 pt).

Usage:  python fig_ed_population.py
Output: img/internal_figures/fig_ed_population.{pdf,svg}
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps                                   # noqa: E402
import fig_paper_F5_drive as F5D                          # noqa: E402  (open-marker groups, as Figs. 3 and 5 draw them)

# written by population_stats.py (its OUT); a path here, so drawing does not load torch and hydra
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data",
                                    "population_stats.json"))
# written by connectivity_confirm.py (its OUT)
CONFIRM = os.path.join(os.path.dirname(OUT), "connectivity_confirm.json")
# written by rl_maze_population_stats.py (its OUT): the Maze networks of Figure 1e
MAZE_OUT = os.path.join(os.path.dirname(OUT), "population_stats_maze.json")
# the spread of the recurrent weights of the same Maze networks (round 18, G13; notebook 2026-11-07_0235)
MAZE_W1 = os.path.join(os.path.dirname(OUT), "round18", "g13_maze_weight_spread.json")
FIG_H_MM = 126.0               # drawn at print size (112 mm shrunk to 0.78 until round 20)
W1_LIM = (0.003, 0.3)          # panel a, both axes (log)
# Every marker has a thin ink edge (Pavel's item 54): ps.mark_floor / ps.mark_fixed set it (MARKER_EW).

TASKS = ("3BFF", "CDDM", "Maze", "DMTS")
MARKER = {"3BFF": "o", "CDDM": "s", "Maze": "D", "DMTS": "^"}
TASK_LABEL = {"3BFF": "3BFF", "CDDM": "CDDM", "Maze": "Maze", "DMTS": "DMTS (fourth task)"}
MIXED_MARKER = "x"             # mixed-selective units in panels c and d, in the task colour
# panel e: each network is shifted ALONG the diagonal (the same amount in x and y), so equal values
# stay on the identity line; tasks get separate stretches of the diagonal, networks 0.1 apart, at most 0.3 from the true value
K_SHIFT = {"3BFF": (-0.3, -0.2, -0.1), "CDDM": (0.1, 0.2, 0.3), "DMTS": (-0.1, 0.0, 0.1), "Maze": (0.25, 0.25)}


def load_confirm():
    """The per-network results of connectivity_confirm.py (the 39 networks of the main test)."""
    with open(CONFIRM) as fh:
        return json.load(fh)["results"]


def load():
    """The per-network results of population_stats.py and the learned Maze networks of rl_maze_population_stats.py.

    Returns:
        list of result dicts, sorted by task (TASKS order) then seed; a Maze entry carries task "Maze".
    """
    with open(OUT) as fh:
        res = json.load(fh)["results"]
    with open(MAZE_OUT) as fh:
        res += [r for r in json.load(fh)["results"] if r["learned"]]
    return sorted(res, key=lambda s: (TASKS.index(s["task"]), str(s["seed"])))


def load_maze_w1():
    """The spread of the recurrent weights of the learned Maze networks of population_stats_maze.json.

    Returns:
        list of dicts (s_all, s_act, subset_min, subset_max) of data/round18/g13_maze_weight_spread.json, the rows
        whose run folder is a learned network of MAZE_OUT.
    """
    with open(MAZE_OUT) as fh:
        learned = {os.path.basename(r["run_dir"]) for r in json.load(fh)["results"] if r["learned"]}
    with open(MAZE_W1) as fh:
        rows = json.load(fh)["rows"]
    return [r for r in rows if os.path.basename(r["folder"]) in learned]


def identity(ax, lo, hi):
    """Draw the line y = x from lo to hi behind the data, and set both axes to [lo, hi]."""
    ax.plot([lo, hi], [lo, hi], color=ps.MUTED, lw=ps.THIN_LW, ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")


def dots(ax, x, y, task, mixed=False, ms=ps.MARKER_PT + 0.6):
    """One network's marker: task colour, and the task's shape or, for mixed-selective units in
    panels c and d, a cross (shape, not fill, carries that distinction; dead-relu-paper#83). A 3BFF,
    CDDM or DMTS network is measured at its floor crossing (filled); a Maze network at update 2,000,
    a fixed budget (open).

    Args:
        ax: axes; x, y: coordinates; task: "3BFF", "CDDM", "Maze" or "DMTS"; mixed: draw the cross;
        ms: marker size in points.
    """
    col = ps.TASK_COLORS[task]
    if mixed:
        ax.plot(x, y, MIXED_MARKER, ms=ms + 0.4, mec=col, mew=0.8, zorder=3, clip_on=False)
    else:
        kw = ps.mark_fixed(col) if task == "Maze" else ps.mark_floor(col)
        ax.plot(x, y, MARKER[task], ms=ms, **kw, zorder=4 if task == "Maze" else 3, clip_on=False)


def scalar_panel(ax, res, key, lo, hi, title, ticks):
    """Panel b: one statistic over all units against over the active units.

    Args:
        ax: axes; res: results; key: "S1" or "S2"; lo, hi: axis range; title: panel title;
        ticks: tick positions for both axes.
    """
    identity(ax, lo, hi)
    for s in res:
        dots(ax, s[key]["act"], s[key]["all"], s["task"])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_title(title)


def weight_panel(ax, conf, maze):
    """Panel a: s.d. of the recurrent weights over all units against over the active units, one
    marker per network of the confirmatory test, and a line spanning the range over 200 random sets
    of as many units as are active (at the network's value over all units).

    Round 20: open markers of one task whose markers would overlap are drawn as one open marker at their
    mean, with a thin bar over their range in y and their number beside it (fig_paper_F5_drive.open_groups
    and draw_groups, the convention of Figures 3 and 5; tests/scan_open_markers.py failed on this panel's
    overlapping open markers until then). Filled markers stay one per network.

    Args:
        ax: axes; conf: records of data/connectivity_confirm.json; maze: load_maze_w1() rows (open markers,
            update 2,000), with their random-set ranges.
    """
    lo, hi = W1_LIM
    ax.set_xscale("log")
    ax.set_yscale("log")
    identity(ax, lo, hi)
    for r in conf:
        y = r["all"]["W1"]
        ax.plot([r["sub_min"]["W1"], r["sub_max"]["W1"]], [y, y], color=ps.FAINT, lw=ps.THIN_LW,
                solid_capstyle="butt", zorder=2)
    for r in maze:
        ax.plot([r["subset_min"], r["subset_max"]], [r["s_all"]] * 2, color=ps.FAINT, lw=ps.THIN_LW,
                solid_capstyle="butt", zorder=2)
    s_open = (ps.MARKER_PT + 0.2) ** 2
    groups = []
    for task in ("3BFF", "CDDM", "DMTS"):
        col = ps.TASK_COLORS[task]
        rows = [r for r in conf if r["task"] == task]
        # filled = weights at the floor crossing (a kept snapshot within 1,000 iterations after it);
        # open = weights saved at a fixed iteration (Methods)
        at_floor = [r["t_star"] == r["t_star"] and 0 <= r["read_iter"] - r["t_star"] <= 1000 for r in rows]
        for r, f in zip(rows, at_floor):
            if f:
                ax.plot(r["act"]["W1"], r["all"]["W1"], MARKER[task], ms=ps.MARKER_PT + 0.2, ls="none",
                        zorder=3, clip_on=False, **ps.mark_floor(col))
        fixed = [r for r, f in zip(rows, at_floor) if not f]
        groups += F5D.open_groups(ax, [r["act"]["W1"] for r in fixed], [r["all"]["W1"] for r in fixed], col,
                                  MARKER[task], s_open)
    maze_groups = F5D.open_groups(ax, [r["s_act"] for r in maze], [r["s_all"] for r in maze],
                                  ps.TASK_COLORS["Maze"], MARKER["Maze"], s_open)
    for g in maze_groups:
        g["side"] = "below"       # beside it the label sat on the 3BFF networks' random-set lines, above it on the identity line
    groups += maze_groups
    F5D.draw_groups(ax, F5D.see_through(ax, groups))
    ticks = [0.003, 0.01, 0.03, 0.1, 0.3]
    labels = ["0.003", "0.01", "0.03", "0.1", "0.3"]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.minorticks_off()
    handles = [ax.plot([], [], "o", ms=ps.MARKER_PT + 0.2, ls="none", **ps.mark_fixed(ps.MUTED))[0],
               ax.plot([], [], "o", ms=ps.MARKER_PT + 0.2, ls="none", **ps.mark_floor(ps.MUTED))[0],
               ax.plot([], [], color=ps.FAINT, lw=ps.THIN_LW)[0]]
    ax.legend(handles, ["fixed iteration", "floor crossing", "random unit sets"], loc="upper left",
              frameon=False, handlelength=1.2, handletextpad=0.4, borderaxespad=0.2, labelspacing=0.25)
    ax.set_title("s.d. of recurrent weights")


def selectivity_panel(ax, res, share):
    """Panels c and d: selective (task marker) and mixed-selective (cross) units, as counts or shares.

    Args:
        ax: axes; res: results; share: False for counts, True for count / set size.
    """
    if share:
        identity(ax, 0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    else:
        identity(ax, 0, 600)
        ax.set_xticks([0, 200, 400, 600])
        ax.set_yticks([0, 200, 400, 600])
    for s in res:
        for key, mixed in (("S3a", False), ("S3b", True)):
            v = s[key]
            x, y = (v["act"], v["all"]) if share else (v["n_act"], v["n_all"])
            dots(ax, x, y, s["task"], mixed=mixed)
    ax.set_title("fraction of units selective" if share else "number of selective units")


def k_panel(ax, res):
    """Panel e: k* over all units against k* over the active units. Each network is shifted along
    the diagonal (K_SHIFT), so coinciding networks stay visible and equal values stay on y = x."""
    identity(ax, 1.5, 8.5)
    for task in TASKS:
        rows = [s for s in res if s["task"] == task]
        for s, d in zip(rows, K_SHIFT[task]):
            dots(ax, s["S4"]["k_act"] + d, s["S4"]["k_all"] + d, task, ms=ps.SMALL_MARKER_PT + 0.4)
    ax.set_xticks(range(2, 9))
    ax.set_yticks(range(2, 9))
    ax.set_title("number of classes")


def share_panel(ax, res):
    """Panel f: each matched pair of classes, its share over active units against over all units."""
    identity(ax, 0, 1)
    for s in res:
        for _ia, _ib, sh_all, sh_act in s["S4"]["pairs"]:
            dots(ax, sh_act, sh_all, s["task"], ms=ps.MARKER_PT)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_title("fraction of units per matched class")


def check_overlaps(fig):
    """Print every pair of text boxes that overlap, and any text outside the canvas.

    Returns:
        number of problems found.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    texts = [t for t in fig.findobj(lambda a: hasattr(a, "get_text") and a.get_visible())
             if getattr(t, "get_text", lambda: "")() and t.get_text().strip()]
    texts = list({id(t): t for t in texts}.values())      # findobj can return an artist twice
    boxes = [(t.get_text(), t.get_window_extent(r)) for t in texts]
    bad = 0
    for i, (ti, bi) in enumerate(boxes):
        if bi.x0 < -0.5 or bi.y0 < -0.5 or bi.x1 > W + 0.5 or bi.y1 > H + 0.5:
            print(f"  !! outside the canvas: {ti!r}")
            bad += 1
        for tj, bj in boxes[i + 1:]:
            same = ti == tj and tuple(bi.bounds) == tuple(bj.bounds)   # one legend text seen twice
            if bi.overlaps(bj) and not same:
                print(f"  !! overlap: {ti!r} / {tj!r}")
                bad += 1
    return bad


def main():
    """Draw the figure from the results files and save it as PDF and SVG."""
    res = load()
    conf = load_confirm()
    maze_w1 = load_maze_w1()
    ps.setup()
    fig = ps.figure(h_mm=FIG_H_MM)
    # margins set explicitly (mm on a 165.1 x FIG_H_MM canvas): the key row on top, the six square panels
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, bottom=0.07, top=0.875, wspace=0.38, hspace=0.42)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    weight_panel(axes[0], conf, maze_w1)
    scalar_panel(axes[1], res, "S2", 0, 50, "dimensionality of normalised rates", [0, 10, 20, 30, 40, 50])
    selectivity_panel(axes[2], res, share=False)
    selectivity_panel(axes[3], res, share=True)
    k_panel(axes[4], res)
    share_panel(axes[5], res)
    for ax, letter in zip(axes, "abcdef"):
        ax.set_xlabel("over active units")
        ax.set_ylabel("over all units")
        ps.panel_letter(ax, letter)
    # legends above the panels, outside the data: tasks (shape and colour; Maze open, at update 2,000), and the
    # shape key of c, d
    handles = [axes[0].plot([], [], MARKER[t], ms=ps.MARKER_PT + 0.6, ls="none",
                            **(ps.mark_fixed if t == "Maze" else ps.mark_floor)(ps.TASK_COLORS[t]))[0] for t in TASKS]
    fig.legend(handles, [TASK_LABEL[t] for t in TASKS], loc="upper left", bbox_to_anchor=(0.06, 0.995), ncol=4,
               frameon=False, handletextpad=0.3, columnspacing=1.2)
    fill = [axes[0].plot([], [], "o", ms=ps.MARKER_PT + 0.6, ls="none", **ps.mark_floor(ps.INK))[0],
            axes[0].plot([], [], MIXED_MARKER, ms=ps.MARKER_PT + 1.0, mec=ps.INK, mew=0.8,
                         ls="none")[0]]
    fig.legend(fill, ["selective (task marker)", "mixed-selective"], loc="upper right", bbox_to_anchor=(0.985, 0.995),
               ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.2, title="panels c, d:",
               alignment="left")
    n = check_overlaps(fig)
    print(f"text overlaps or overflows: {n}")
    for s in res:
        print(f"  {s['task']:>5} {str(s['seed']):>11}: S2 {s['S2']['all']:.1f} / {s['S2']['act']:.1f}; selective "
              f"{s['S3a']['n_all']} / {s['S3a']['n_act']} ({s['S3a']['n_inactive']} inactive); fractions "
              f"{s['S3a']['all']:.3f} / {s['S3a']['act']:.3f}; classes {s['S4']['k_all']} / {s['S4']['k_act']}")
    for r in maze_w1:
        print(f"  Maze {os.path.basename(r['folder'])}: s.d. of recurrent weights {r['s_all']:.4f} over all units, "
              f"{r['s_act']:.4f} over the active units; random sets {r['subset_min']:.4f} to {r['subset_max']:.4f}")
    print(ps.save(fig, "fig_ed_population"))


if __name__ == "__main__":
    main()
