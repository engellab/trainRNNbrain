"""Every marker of a manuscript figure has a thin black edge (dead-relu-paper round 12, Pavel's item 54).

Runs a figure script as __main__ (with its command-line arguments) with `paperstyle.save` wrapped, so
each finished matplotlib figure is kept after it is written, and classifies every marker artist of
it: the markers of Line2D artists (ax.plot, ax.errorbar, legend handles) and the paths of
PathCollections (ax.scatter). It reads the artists, not the PDF; tests/scan_open_markers.py checks
the printed page.

Pass rules, fixed before the first scan:
  1. filled marker (a face that is neither transparent nor white): edge colour ink (every RGB
     channel at most 0.2) and edge width between 0.3 and 0.4 pt;
  2. open marker (a white or transparent face): white face, an edge in its series colour of
     paperstyle.OPEN_MARKER_EW, and the ink rim of paperstyle.open_rim (a path-effect Stroke in ink
     whose width is OPEN_MARKER_EW plus twice MARKER_EW) drawn round it;
  3. no artist without a marker carries the ink rim (an error bar or a data line drawn with the
     rim would be outlined in black).
A marker that is a stroke by its shape ('x', '+', '|', '_', '1'-'4': no inside, so no edge round it,
error-bar caps included) gets no edge. Rule 4, added after the review of PR #177: a stroke glyph
drawn with a stroke width equal to MARKER_EW fails, because that width came from paperstyle's
rcParam for marker edges and not from the script (Fig. 2c caps and the Fig. 4 legend 'x' were
thinned that way). Each figure's count of glyphs is printed. Neither is a ring: an ink circle with no face,
at least RING_MIN_PT across, drawn round a marker as an annotation (Fig. 5's "R^2 from weights inside
a loss burst"); rings are counted with the glyphs.

Run (repository root):
  python tests/scan_marker_edges.py fig_paper_F1 fig_paper_F4_noreturn:draw ...
A script name may carry its arguments after a colon, separated by commas. Prints one line per
failing artist group and a count per figure; exits 1 if any rule fails.
"""
import os
import runpy
import sys

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EA = os.path.join(REPO, "trainRNNbrain", "experiments_and_analysis")
sys.path.insert(0, EA)

import paperstyle as ps  # noqa: E402  (needs the path above)

INK_MAX = 0.2               # rule 1: every RGB channel of an ink edge at most this
EW_RANGE = (0.3, 0.4)       # rule 1: edge width of a filled marker, pt
NO_MARKER = (None, "None", "none", "", " ")
GLYPHS = ("x", "+", "|", "_", "1", "2", "3", "4")   # a stroke, not a shape: no inside, no edge
RING_MIN_PT = 5.0           # an ink circle with no face at least this wide is an annotation ring
# the paths ax.scatter stores for the stroke glyphs (marker path in its own transform)
GLYPH_PATHS = [MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform()).vertices
               for m in GLYPHS]


def is_glyph_path(path):
    """True if a scatter path is one of the stroke glyphs (GLYPH_PATHS)."""
    v = path.vertices
    return any(g.shape == v.shape and np.allclose(g, v) for g in GLYPH_PATHS)


def is_ring(face, edge, d_pt):
    """True for an annotation ring: no face, an ink edge, at least RING_MIN_PT across."""
    return face[3] == 0 and max(edge[:3]) <= INK_MAX and d_pt >= RING_MIN_PT


def run_and_capture(script, args):
    """Run one figure script as __main__ with `args` and keep every figure it saves.

    Args:
        script: str, file stem of a script in trainRNNbrain/experiments_and_analysis;
        args: list of str, its command-line arguments.
    Returns:
        list of (matplotlib Figure, figure name), one per paperstyle.save call.
    """
    captured = []
    original = ps.save

    def keep(fig, name, *a, **kw):
        """paperstyle.save, remembering the figure and its name."""
        path = original(fig, name, *a, **kw)
        captured.append((fig, name))
        return path

    ps.save = keep
    argv = sys.argv
    sys.argv = [os.path.join(EA, f"{script}.py")] + list(args)
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        ps.save, sys.argv = original, argv
    return captured


def has_rim(artist):
    """True if the artist carries an ink path-effect Stroke (paperstyle.open_rim)."""
    for eff in artist.get_path_effects() or []:
        if isinstance(eff, pe.Stroke):
            fg = eff._gc.get("foreground")
            if fg is not None and max(to_rgba(fg)[:3]) <= INK_MAX:
                return True
    return False


def rim_width(artist):
    """Line width (pt) of the artist's ink path-effect Stroke, or nan."""
    for eff in artist.get_path_effects() or []:
        if isinstance(eff, pe.Stroke):
            return float(eff._gc.get("linewidth", np.nan))
    return float("nan")


def classify(face, edge, ew, rim, rimw):
    """The verdict for one marker.

    Args:
        face, edge: RGBA tuples; ew: edge width in pt; rim: bool, carries the ink rim;
        rimw: the rim stroke's width in pt (nan without a rim).
    Returns:
        (kind, ok, why): kind 'filled' or 'open'; ok bool; why a short reason when not ok.
    """
    is_open = face[3] == 0 or min(face[:3]) > 0.95
    if not is_open:
        if max(edge[:3]) > INK_MAX or edge[3] == 0:
            return "filled", False, "edge not ink"
        if not EW_RANGE[0] - 1e-6 <= ew <= EW_RANGE[1] + 1e-6:
            return "filled", False, f"edge {ew:.2f} pt"
        if rim:
            return "filled", False, "filled marker with a rim"
        return "filled", True, ""
    if face[3] == 0:
        return "open", False, "transparent face (the rim would show inside)"
    if not rim:
        return "open", False, "no ink rim"
    if abs(ew - ps.OPEN_MARKER_EW) > 1e-6 or abs(rimw - (ps.OPEN_MARKER_EW + 2 * ps.MARKER_EW)) > 1e-6:
        return "open", False, f"ring {ew:.2f} pt, rim stroke {rimw:.2f} pt"
    return "open", True, ""


def axes_name(fig, ax):
    """A short name of an axes: its index in the figure and its title or y label."""
    lab = (ax.get_title() or ax.get_ylabel() or "").replace("\n", " ")[:28] if ax is not None else ""
    idx = fig.axes.index(ax) if ax in fig.axes else -1
    return f"ax{idx} {lab!r}"


def scan(fig):
    """Classify every marker artist of one figure.

    Args:
        fig: a finished matplotlib Figure.
    Returns:
        (n_ok, n_glyph, failures): n_glyph counts stroke markers (GLYPHS) and rings, not checked; failures is a dict (axes name, artist kind, marker, reason, face hex,
        edge hex) -> count of markers.
    """
    n_ok, n_glyph, fails = 0, 0, {}
    for art in fig.findobj(lambda a: isinstance(a, (Line2D, PathCollection))):
        if not art.get_visible():
            continue
        ax = art.axes
        where = axes_name(fig, ax) if ax is not None else "legend"
        if isinstance(art, Line2D):
            mk = art.get_marker()
            if isinstance(mk, (int, np.integer)):      # an axis tick (TICKLEFT, TICKDOWN, ...)
                continue
            if mk in NO_MARKER or art.get_markersize() == 0 or len(art.get_xdata()) == 0:
                if has_rim(art) and art.get_linestyle() not in ("None", "none", ""):
                    key = (where, "Line2D", "-", "line carries the ink rim", "", "")
                    fails[key] = fails.get(key, 0) + 1
                continue
            if mk in GLYPHS:
                n = max(len(np.atleast_1d(art.get_xdata())), 1)
                if abs(art.get_markeredgewidth() - ps.MARKER_EW) < 1e-6:
                    key = (where, "Line2D", str(mk), "stroke glyph at the rcParam edge width", "", "")
                    fails[key] = fails.get(key, 0) + n
                else:
                    n_glyph += n
                continue
            face = to_rgba(art.get_markerfacecolor())
            edge = to_rgba(art.get_markeredgecolor())
            if is_ring(face, edge, art.get_markersize()):
                n_glyph += max(len(np.atleast_1d(art.get_xdata())), 1)
                continue
            kind, ok, why = classify(face, edge, art.get_markeredgewidth(), has_rim(art), rim_width(art))
            if ok and has_rim(art) and art.get_linestyle() not in ("None", "none", ""):
                ok, why = False, "line carries the ink rim"
            n = max(len(np.atleast_1d(art.get_xdata())), 1)
            if ok:
                n_ok += n
            else:
                key = (where, "Line2D", str(mk), why, ps_hex(face), ps_hex(edge))
                fails[key] = fails.get(key, 0) + n
        else:
            offs = art.get_offsets()
            if len(offs) == 0:
                continue
            if art.get_paths() and is_glyph_path(art.get_paths()[0]):
                n_glyph += len(offs)
                continue
            fcs, ecs = art.get_facecolors(), art.get_edgecolors()
            lws = np.atleast_1d(art.get_linewidths())
            sizes = np.atleast_1d(art.get_sizes()) if len(art.get_sizes()) else np.array([0.0])
            for i in range(len(offs)):
                face = tuple(fcs[i % len(fcs)]) if len(fcs) else (0, 0, 0, 0)
                edge = tuple(ecs[i % len(ecs)]) if len(ecs) else (0, 0, 0, 0)
                if is_ring(face, edge, np.sqrt(sizes[i % len(sizes)])):
                    n_glyph += 1
                    continue
                kind, ok, why = classify(face, edge, float(lws[i % len(lws)]), has_rim(art),
                                         rim_width(art))
                if ok:
                    n_ok += 1
                else:
                    key = (where, "scatter", "path", why, ps_hex(face), ps_hex(edge))
                    fails[key] = fails.get(key, 0) + 1
    return n_ok, n_glyph, fails


def ps_hex(c):
    """RGBA tuple -> '#rrggbb' (alpha appended as '@a' when below 1)."""
    s = "#" + "".join(f"{int(round(255 * v)):02x}" for v in c[:3])
    return s + (f"@{c[3]:.2f}" if c[3] < 1 else "")


def main():
    """Scan every script named on the command line; exit 1 if any marker fails a rule."""
    bad = 0
    for spec in sys.argv[1:]:
        script, _, args = spec.partition(":")
        for fig, name in run_and_capture(script, [a for a in args.split(",") if a]):
            n_ok, n_glyph, fails = scan(fig)
            n_bad = sum(fails.values())
            bad += n_bad
            print(f"== {name}: {n_ok} markers pass, {n_bad} fail, {n_glyph} stroke glyphs and rings not checked")
            for (where, kind, mk, why, face, edge), n in sorted(fails.items()):
                print(f"   !! {where} {kind} marker {mk!r}: {why} (face {face}, edge {edge}) x{n}")
    print("PASS" if bad == 0 else f"FAIL: {bad} markers")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
