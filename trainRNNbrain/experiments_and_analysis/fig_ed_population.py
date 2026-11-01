#!/usr/bin/env python3
"""
Extended Data Fig. 2 (dead-relu-paper #56): population statistics over all units against the same
statistics over the active units only, every network drawn.

Draws data/population_stats.json, written by population_stats.py (notebook entries
2026-10-31_2212_population-stats-rule.md and 2026-10-31_2220_population-stats-results.md). Nothing
is computed here beyond reading that file.

  (a) participation ratio of the rate covariance
  (b) participation ratio after dividing each unit's rates by their standard deviation
  (c) selective (task marker) and mixed-selective (x) unit counts. Every marker is filled or a
      cross, never open: the weights are the floor-crossing snapshots, and fill means the floor
      crossing in every figure (dead-relu-paper#83), so selective and mixed-selective differ by shape
  (d) the same as shares: count / 1,000 against count / active count
  (e) k-means class count k* chosen by silhouette
  (f) class shares of matched classes, one point per matched pair

Usage:  python fig_ed_population.py
Output: img/internal_figures/fig_ed_population.{pdf,svg}
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps                                   # noqa: E402

# written by population_stats.py (its OUT); a path here, so drawing does not load torch and hydra
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data",
                                    "population_stats.json"))

TASKS = ("3BFF", "CDDM", "DMTS")
MARKER = {"3BFF": "o", "CDDM": "s", "DMTS": "^"}
MIXED_MARKER = "x"             # mixed-selective units in panels c and d, in the task colour
# panel e: each network is shifted ALONG the diagonal (the same amount in x and y), so equal values
# stay on the identity line; tasks get separate stretches of the diagonal, networks 0.1 apart, at most 0.3 from the true value
K_SHIFT = {"3BFF": (-0.3, -0.2, -0.1), "CDDM": (0.1, 0.2, 0.3), "DMTS": (-0.1, 0.0, 0.1)}


def load():
    """The per-network results of population_stats.py, sorted by task then seed."""
    with open(OUT) as fh:
        res = json.load(fh)["results"]
    return sorted(res, key=lambda s: (TASKS.index(s["task"]), s["seed"]))


def identity(ax, lo, hi):
    """Draw the line y = x from lo to hi behind the data, and set both axes to [lo, hi]."""
    ax.plot([lo, hi], [lo, hi], color=ps.MUTED, lw=ps.THIN_LW, ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")


def dots(ax, x, y, task, mixed=False, ms=ps.MARKER_PT + 0.6):
    """One network's marker: task colour, and the task's filled shape or, for mixed-selective units in
    panels c and d, a cross (shape, not fill, carries that distinction; dead-relu-paper#83).

    Args:
        ax: axes; x, y: coordinates; task: "3BFF", "CDDM" or "DMTS"; mixed: draw the cross;
        ms: marker size in points.
    """
    col = ps.TASK_COLORS[task]
    if mixed:
        ax.plot(x, y, MIXED_MARKER, ms=ms + 0.4, mec=col, mew=0.8, zorder=3, clip_on=False)
    else:
        ax.plot(x, y, MARKER[task], ms=ms, mew=0.7, **ps.mark_floor(col), zorder=3, clip_on=False)


def scalar_panel(ax, res, key, lo, hi, title, ticks):
    """Panels a and b: one statistic over all units against over the active units.

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
    ax.set_title("selective units, share" if share else "selective units, count")


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
    ax.set_title("class shares, matched classes")


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
    """Draw the figure from the results file and save it as PDF and SVG."""
    res = load()
    ps.setup()
    fig = ps.figure(h_mm=112)
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, bottom=0.075, top=0.86,
                          wspace=0.42, hspace=0.52)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    scalar_panel(axes[0], res, "S1", 0, 6, "dimensionality of rates", [0, 2, 4, 6])
    scalar_panel(axes[1], res, "S2", 0, 30, "dimensionality of normalised rates", [0, 10, 20, 30])
    selectivity_panel(axes[2], res, share=False)
    selectivity_panel(axes[3], res, share=True)
    k_panel(axes[4], res)
    share_panel(axes[5], res)
    for ax, letter in zip(axes, "abcdef"):
        ax.set_xlabel("over active units")
        ax.set_ylabel("over all units")
        ps.panel_letter(ax, letter)
    # legends above the panels, outside the data: tasks (shape and colour), and the shape key of c, d
    handles = [axes[0].plot([], [], MARKER[t], ms=ps.MARKER_PT + 0.6, mfc=ps.TASK_COLORS[t],
                            mec=ps.TASK_COLORS[t], ls="none")[0] for t in TASKS]
    fig.legend(handles, TASKS, loc="upper left", bbox_to_anchor=(0.06, 0.995), ncol=3,
               frameon=False, handletextpad=0.3, columnspacing=1.2)
    fill = [axes[0].plot([], [], "o", ms=ps.MARKER_PT + 0.6, mfc=ps.INK, mec=ps.INK, ls="none")[0],
            axes[0].plot([], [], MIXED_MARKER, ms=ps.MARKER_PT + 1.0, mec=ps.INK, mew=0.8,
                         ls="none")[0]]
    fig.legend(fill, ["selective (task marker)", "mixed-selective"], loc="upper right", bbox_to_anchor=(0.985, 0.995),
               ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.2, title="panels c, d:",
               alignment="left")
    n = check_overlaps(fig)
    print(f"text overlaps or overflows: {n}")
    print(ps.save(fig, "fig_ed_population"))


if __name__ == "__main__":
    main()
