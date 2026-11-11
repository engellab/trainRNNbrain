"""No text of a manuscript figure lies on another panel's axes or past the canvas edge (dead-relu-paper #232).

The check that every figure script runs (fig_ed_maze.check_overlaps) tests text against text and
against the canvas edge. It missed Fig. 5d's rotated y label, which ran about 0.5 mm into the
frame of Fig. 5c's CDDM panels (round-22 committee, #231, Reviewer S figures 1). This scan adds
the missing test.

Runs a figure script as __main__ (with its command-line arguments) with `paperstyle.save` wrapped,
so each finished figure is kept after it is written (tests/scan_marker_edges.run_and_capture).
For every visible text of the figure (free text, titles, axis labels, tick labels whose tick lies
inside the view, legend texts and figure texts) it takes the rendered box, with the axis-label pads
floored as save() floors them.

Pass rules, fixed before the first scan:
  1. no text box overlaps the frame box (the axes patch, spines included) of an axes other than the
     one the text belongs to; an axes without a visible frame and patch (an axes used only to hold
     a key or text) is skipped as a target;
  2. no text box reaches past the canvas edge.
Each overlap is printed with its depth in millimetres.

Run (repository root):
  python tests/scan_text_on_axes.py fig_paper_F5_tasks fig_paper_F4_noreturn:draw ...
A script name may carry its arguments after a colon, separated by commas. Exits 1 if any rule fails.
"""
import os
import sys

from matplotlib.transforms import Bbox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scan_marker_edges import run_and_capture  # noqa: E402  (puts the figure scripts' folder on the path)
import paperstyle as ps  # noqa: E402  (needs the path above)


def texts_of(fig):
    """Every visible, non-empty text of a figure, with the axes it belongs to.

    Args:
        fig: a finished matplotlib Figure.
    Returns:
        list of (Text, Axes or None); None for a figure-level text.
    """
    out = [(t, None) for t in fig.texts]
    for ax in fig.axes:
        own = list(ax.texts) + [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label]
        leg = ax.get_legend()
        if leg is not None:
            own += list(leg.get_texts())
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            for tick in axis.get_major_ticks():
                if lo <= tick.get_loc() <= hi:
                    own += [lab for lab in (tick.label1, tick.label2) if lab.get_visible()]
        out += [(t, ax) for t in own]
    return [(t, ax) for t, ax in out if t.get_visible() and t.get_text().strip()]


def framed(ax):
    """True if an axes draws a frame or a face (a visible spine, or a patch that is not transparent)."""
    spines = any(s.get_visible() for s in ax.spines.values())
    face = ax.patch.get_visible() and ax.patch.get_facecolor()[3] > 0 and ax.axison
    return ax.get_visible() and ax.axison and (spines or face)


def scan(fig):
    """Apply rules 1 and 2 to one figure.

    Args:
        fig: a finished matplotlib Figure.
    Returns:
        (number of texts checked, list of failure strings).
    """
    ps.floor_label_pads(fig)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    to_mm = 25.4 / fig.dpi
    frames = [(ax, ax.patch.get_window_extent(rend)) for ax in fig.axes if framed(ax)]
    W, H = fig.bbox.width, fig.bbox.height
    texts = texts_of(fig)
    fails = []
    for t, owner in texts:
        b = t.get_window_extent(rend)
        if b.width <= 0:
            continue
        name = t.get_text().replace("\n", " ")[:50]
        for ax, f in frames:
            if ax is owner:
                continue
            inter = Bbox.intersection(b, f)
            if inter is not None and inter.width > 0 and inter.height > 0:
                fails.append(f"text {name!r} lies on the frame of another axes "
                             f"({inter.width * to_mm:.2f} x {inter.height * to_mm:.2f} mm)")
        if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H:
            fails.append(f"text {name!r} reaches past the canvas edge")
    return len(texts), fails


def main():
    """Scan every script named on the command line; exit 1 if any text fails a rule."""
    bad = 0
    for spec in sys.argv[1:]:
        script, _, args = spec.partition(":")
        for fig, name in run_and_capture(script, [a for a in args.split(",") if a]):
            n, fails = scan(fig)
            bad += len(fails)
            print(f"== {name}: {n} texts checked, {len(fails)} fail")
            for f in fails:
                print(f"   !! {f}")
    print("PASS" if bad == 0 else f"FAIL: {bad} texts")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
