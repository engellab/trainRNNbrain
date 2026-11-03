"""Every open marker of a manuscript figure reads as open on the printed page (dead-relu-paper #104).

Runs a figure script as __main__ with `paperstyle.save` wrapped, so the finished matplotlib figure
is kept after it is written. From that figure it takes the centre of every drawn open marker (a
Line2D or scatter marker whose face is transparent or white) and of every ring (an open ink circle
of Figure 5's ring size, RING_S), in printed millimetres. Then it renders the PDF the script wrote
with PyMuPDF at 300 dpi, which shares no code with matplotlib's drawing, and reads the pixels there.

Pass rules, fixed before the first scan (#104):
  1. centre: the darkest channel of the 3 x 3 pixel block at the centre of every open marker's
     inside is at least 200/255 (the inside is paper, not ink or another mark). The centre of a
     triangle's inside is its incentre (`inside_centre`); this was set after the first scan, which
     showed matplotlib's anchor of a 3-4 pt open triangle lying on its anti-aliased edge stroke.
     The table also gives the darkest value at the anchor itself;
  2. overlap: no two open markers of one axes have centres closer than half the larger marker's
     diameter (they overlap by more than half a marker width). Added after the PR #118 review,
     which ruled out moving a marker off its measured value: a pair whose faces are both
     half-transparent white (fig_paper_F5_drive.see_through, Fig. 5e) is a deliberate overlap at
     the measured values; it is printed as "see-through" and not counted;
  3. rings: no two rings of one axes have centres closer than 1.0 mm.
Rings on top (dead-relu-paper #176, round 13). Extended Data Fig. 4 keeps every open marker at its
measured value and redraws each marker's coloured ring above all the white insides
(fig_paper_F2_fails.ring_overlay; the redrawn ring carries the gid RING_OVERLAY and is read as the
marker it redraws, not as another marker). Two open markers whose centres are closer than one
diameter then cross: each one's ring runs over or near the other's centre, and no drawing that keeps
both at their measured values avoids that. For two markers that both have a ring on top:
  1'. centre: a centre that reads dark is printed as "ring crossing" and not counted if the outline
      of another such marker of the same axes passes within CROSS_MM of it (half the ring and its
      ink rim, plus the 3 x 3 pixel block and anti-aliasing). Nothing else can darken such a
      centre: lines and filled markers lie under the white insides. Any other dark centre fails;
  2'. overlap: a pair closer than rule 2's distance is printed as "rings on top" and not counted
      if the two differ in colour, so the two rings tell the two values apart. A difference in
      shape alone does not excuse a pair (review of PR #179): concentric rings of one colour print
      as one symbol;
  2''. nested: a pair is printed as "nested" and not counted if the smaller marker's outline lies
      wholly inside the larger one's, at least NEST_GAP_MM from it everywhere (the two half rings,
      the inner marker's rim and 0.1 mm of paper), so the two rings stay apart with paper between
      them (fig_paper_F2_fails.nested_sizes). Rule 1 still applies to both centres.
A marker without a ring on top is scanned exactly as before.
A check of the coordinate mapping, not a pass rule: the centre of nearly every filled marker must
read dark (darkest channel under 200); if it does not, the centres are in the wrong place and the
scan proves nothing.

Run (repository root): python tests/scan_open_markers.py fig_paper_F2_size fig_paper_F5_drive
Prints one line per failing marker, a table per axes, and exits 1 if any rule fails.
"""
import io
import os
import runpy
import sys

import numpy as np
import pymupdf
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path as MplPath

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EA = os.path.join(REPO, "trainRNNbrain", "experiments_and_analysis")
sys.path.insert(0, EA)

import paperstyle as ps  # noqa: E402  (needs the path above)

DPI = 300
MM_PER_PT = 25.4 / 72
CENTRE_MIN = 200            # rule 1, darkest channel at an open-marker centre (0-255)
RING_MIN_MM = 1.0           # rule 3
RING_SIZE_MIN = 30          # scatter size from which an open ink circle is a ring (RING_S = 34)
RING_OVERLAY = "ring-overlay"   # gid of a ring redrawn over its own open marker (fig_paper_F2_fails)
# rule 1': half the ring (OPEN_MARKER_EW / 2) and the ink rim (MARKER_EW) in mm, plus half the 3 x 3
# block at DPI and 0.05 mm of anti-aliasing
CROSS_MM = (ps.OPEN_MARKER_EW / 2 + ps.MARKER_EW) * 25.4 / 72 + 1.5 * 25.4 / DPI + 0.05
# rule 2'': the outer ring's inner half, the inner ring's outer half and rim, and 0.1 mm of paper
NEST_GAP_MM = (ps.OPEN_MARKER_EW + ps.MARKER_EW) * 25.4 / 72 + 0.1


def run_and_capture(script):
    """Run one figure script as __main__ and keep every figure it saves.

    Args:
        script: str, file stem of a script in trainRNNbrain/experiments_and_analysis.
    Returns:
        list of (matplotlib Figure, path of the PDF written), one per paperstyle.save call.
    """
    captured = []
    original = ps.save

    def keep(fig, name, *args, **kwargs):
        """paperstyle.save, remembering the figure and the PDF path it returns."""
        path = original(fig, name, *args, **kwargs)
        captured.append((fig, path))
        return path

    ps.save = keep
    argv = sys.argv
    sys.argv = [os.path.join(EA, f"{script}.py")]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        ps.save, sys.argv = original, argv
    return captured


def is_white_or_none(rgba):
    """True if an RGBA face is transparent or white/paper (every channel above 0.95)."""
    return rgba[3] == 0 or min(rgba[:3]) > 0.95


def see_through(rgba):
    """True if an RGBA face is white at partial opacity (a deliberate overlap, rule 2)."""
    return 0 < rgba[3] < 1 and min(rgba[:3]) > 0.95


def legend_artists(fig):
    """Ids of every artist that belongs to a legend of the figure (legend keys are not data)."""
    out = set()
    for leg in fig.findobj(Legend):
        out |= {id(a) for a in leg.findobj()}
    return out


def inside_axes(artist, xy):
    """Boolean mask of display points that are drawn: inside the axes box unless clipping is off.

    Args:
        artist: the artist; xy: (n, 2) display coordinates.
    Returns:
        (n,) bool array.
    """
    ax = artist.axes
    if ax is None or not artist.get_clip_on():
        return np.ones(len(xy), bool)
    bb = ax.bbox
    tol = 0.5
    return ((xy[:, 0] >= bb.x0 - tol) & (xy[:, 0] <= bb.x1 + tol) &
            (xy[:, 1] >= bb.y0 - tol) & (xy[:, 1] <= bb.y1 + tol))


def inside_centre(verts_pt):
    """Offset (pt, y up) from a marker's anchor to the centre of its inside.

    A triangle's inside is centred on its incentre; matplotlib anchors a triangle at half its height,
    0.45 half-heights from its slanted edges, so in an open 3-4 pt triangle the anchor lies under
    0.5 pt from the edge stroke. Every other house marker (circle, square, diamond) is symmetric
    about its anchor.

    Args:
        verts_pt: (n, 2) array, the marker path's vertices in pt around its anchor.
    Returns:
        (dx, dy) in pt.
    """
    uniq = np.unique(np.round(verts_pt, 6), axis=0)
    if len(uniq) != 3:
        return 0.0, 0.0
    A, B, C = uniq
    a, b, c = np.linalg.norm(B - C), np.linalg.norm(C - A), np.linalg.norm(A - B)
    centre = (a * A + b * B + c * C) / (a + b + c)
    return float(centre[0]), float(centre[1])


def markers_of(fig):
    """Every drawn marker of a figure, classified as open, filled or ring.

    Args:
        fig: a drawn matplotlib Figure.
    Returns:
        list of dicts with kind ("open", "filled", "ring"), x_mm and y_mm (from the top-left corner
        of the page), d_mm (marker diameter), axes (the Axes or None), legend (bool), edge (hex).
    """
    # Lay the figure out exactly as the PDF was written: the PDF backend draws at 72 dpi, and a
    # legend's position depends on the renderer's text widths, so an Agg draw would move it.
    fig.set_dpi(72)
    fig.savefig(io.BytesIO(), format="pdf", transparent=True, bbox_inches=None)
    in_legend = legend_artists(fig)
    h_px = fig.bbox.height
    px_to_mm = 25.4 / fig.dpi
    out = []

    overlays = []                         # (axes, x mm, y mm) of every ring redrawn on top
    for col in fig.findobj(PathCollection):
        if col.get_visible() and col.get_gid() == RING_OVERLAY and len(col.get_offsets()):
            for x, y in col.get_offset_transform().transform(col.get_offsets()):
                overlays.append((col.axes, x * px_to_mm, (h_px - y) * px_to_mm))

    def add(artist, xy, kind, d_pt, edge, verts_pt, thin=False, outline_pt=None):
        """Append the drawn points of one artist to `out` (anchor and centre of the inside).

        outline_pt: (n, 2) polygon of the marker's outline in pt round its anchor (y up), kept for
        rules 1' and 2'' with `on_top` (a ring redrawn over this marker).
        """
        xy = np.atleast_2d(np.asarray(xy, float))
        if not len(xy):
            return
        dx, dy = inside_centre(verts_pt)
        ok = inside_axes(artist, xy) & np.all(np.isfinite(xy), axis=1)
        for (x, y) in xy[ok]:
            x_mm, y_mm = x * px_to_mm, (h_px - y) * px_to_mm
            on_top = any(a is artist.axes and abs(ox - x_mm) < 0.01 and abs(oy - y_mm) < 0.01
                         for a, ox, oy in overlays)
            outline = None
            if outline_pt is not None:
                outline = np.c_[x_mm + outline_pt[:, 0] * MM_PER_PT, y_mm - outline_pt[:, 1] * MM_PER_PT]
            out.append(dict(kind=kind, x_mm=x_mm, y_mm=y_mm, cx_mm=x_mm + dx * MM_PER_PT,
                            cy_mm=y_mm - dy * MM_PER_PT, d_mm=d_pt * MM_PER_PT, axes=artist.axes,
                            legend=id(artist) in in_legend, edge=edge, thin=thin, on_top=on_top,
                            outline=outline))

    ticks = {id(a) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis) for a in axis.findobj()}
    for ln in fig.findobj(Line2D):
        mk = ln.get_marker()
        if id(ln) in ticks or not ln.get_visible() or mk in (None, "None", "none", "", " ") or \
                (isinstance(mk, str) and (mk.startswith("$") or mk in ("_", "|"))) or \
                ln.get_markersize() <= 0:          # error-bar caps are lines, not markers
            continue
        face = to_rgba(ln.get_markerfacecolor())
        kind = "open" if (ln.get_fillstyle() == "none" or is_white_or_none(face)) else "filled"
        xy = ln.get_transform().transform(ln.get_xydata())
        every = ln.get_markevery()          # a legend key draws its marker at one of three points
        if every is not None:
            xy = xy[every] if isinstance(every, (slice, list, tuple, np.ndarray)) else xy[::every]
        style = MarkerStyle(mk)
        verts = style.get_transform().transform(style.get_path().vertices) * ln.get_markersize()
        add(ln, xy, kind, ln.get_markersize(), ps_hex(ln.get_markeredgecolor()), verts,
            see_through(face))
    for col in fig.findobj(PathCollection):
        if not col.get_visible() or not len(col.get_offsets()) or col.get_gid() == RING_OVERLAY:
            continue
        faces = col.get_facecolor()
        edges = col.get_edgecolor()
        sizes = col.get_sizes()
        open_ = len(faces) == 0 or all(is_white_or_none(f) for f in faces)
        edge = ps_hex(edges[0]) if len(edges) else "none"
        s = float(sizes[0]) if len(sizes) else 36.0
        kind = "filled"
        if open_:
            kind = "ring" if (s >= RING_SIZE_MIN and edge == ps_hex(ps.INK)) else "open"
        xy = col.get_offset_transform().transform(col.get_offsets())
        polys = col.get_paths()[0].to_polygons(closed_only=False)
        add(col, xy, kind, np.sqrt(s), edge, col.get_paths()[0].vertices * np.sqrt(s),
            len(faces) > 0 and all(see_through(f) for f in faces),
            np.vstack(polys) * np.sqrt(s) if polys else None)
    return out


def outline_distance(outline, x, y):
    """Shortest distance (mm) from a point to a closed polygon outline (rule 1').

    Args:
        outline: (n, 2) polygon vertices in page mm; x, y: the point in page mm.
    Returns:
        float.
    """
    a = outline
    b = np.roll(outline, -1, axis=0)
    ab = b - a
    t = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) /
                np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
    px, py = a[:, 0] + t * ab[:, 0], a[:, 1] + t * ab[:, 1]
    return float(np.min(np.hypot(px - x, py - y)))


def nested(a, b):
    """Rule 2'': True if the smaller of two markers lies wholly inside the larger, NEST_GAP_MM clear.

    Args:
        a, b: marker dicts from markers_of with "outline" polygons (page mm).
    Returns:
        bool.
    """
    if a["outline"] is None or b["outline"] is None:
        return False
    small, big = (a, b) if a["d_mm"] < b["d_mm"] else (b, a)
    if not np.all(MplPath(big["outline"]).contains_points(small["outline"])):
        return False
    return min(outline_distance(big["outline"], x, y) for x, y in small["outline"]) >= NEST_GAP_MM


def ps_hex(c):
    """A colour as lower-case #rrggbb, or "none" if it is transparent."""
    rgba = to_rgba(c)
    if rgba[3] == 0:
        return "none"
    return "#" + "".join(f"{round(255 * v):02x}" for v in rgba[:3])


def darkest_at(img, x_mm, y_mm):
    """Darkest channel of the 3 x 3 pixel block at a page position of a 300 dpi render.

    Args:
        img: (H, W, 3) uint8 array; x_mm, y_mm: position from the top-left corner (mm).
    Returns:
        int, 0-255.
    """
    cx, cy = int(round(x_mm / 25.4 * DPI)), int(round(y_mm / 25.4 * DPI))
    blk = img[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2]
    return int(blk.min())


def render(pdf):
    """Render page 1 of a PDF at DPI on white, as an (H, W, 3) uint8 array."""
    page = pymupdf.open(pdf)[0]
    pix = page.get_pixmap(dpi=DPI, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]


def axes_name(ax):
    """A readable name for an axes: its title, else its y label, else its x label."""
    if ax is None:
        return "(figure)"
    for t in (ax.get_title(), ax.get_ylabel(), ax.get_xlabel()):
        if t.strip():
            return " ".join(t.split())[:40]
    return f"axes at x={ax.get_position().x0:.2f}, y={ax.get_position().y0:.2f}"


def scan(fig, pdf):
    """Apply the three pass rules to one figure and print the result.

    Args:
        fig: the figure as saved; pdf: path of its PDF.
    Returns:
        (rows, n_fail): rows, one per axes, (name, open markers, centre fails, darkest centre value,
        overlapping pairs, rings, closest ring pair mm); n_fail, the number of rule failures.
    """
    img = render(pdf)
    marks = markers_of(fig)
    for m in marks:
        m["dark"] = darkest_at(img, m["cx_mm"], m["cy_mm"])
        m["dark_anchor"] = darkest_at(img, m["x_mm"], m["y_mm"])
    filled = [m for m in marks if m["kind"] == "filled" and not m["legend"]]
    dark_ok = sum(m["dark"] < CENTRE_MIN for m in filled)
    print(f"\n{os.path.basename(pdf)}: mapping check, {dark_ok} of {len(filled)} filled-marker "
          "centres read dark")
    for m in filled:
        if m["dark"] >= CENTRE_MIN:
            print(f"  light filled centre: {axes_name(m['axes'])}: {m['edge']} at ({m['x_mm']:.1f}, "
                  f"{m['y_mm']:.1f}) mm, darkest channel {m['dark']}")
    by_ax = {}
    for m in marks:
        by_ax.setdefault((id(m["axes"]), m["legend"]), (m["axes"], m["legend"], []))[2].append(m)
    rows, n_fail = [], 0
    for ax, leg, ms in by_ax.values():
        op = [m for m in ms if m["kind"] == "open"]
        rg = [m for m in ms if m["kind"] == "ring"]
        if not op and not rg:
            continue
        bad = []
        for m in op:
            if m["dark"] >= CENTRE_MIN:
                continue
            crossing = [o for o in op if o is not m and m["on_top"] and o["on_top"] and
                        o["outline"] is not None and
                        outline_distance(o["outline"], m["cx_mm"], m["cy_mm"]) <= CROSS_MM]
            if crossing:
                print(f"  ring crossing (not counted): {axes_name(ax)}: {m['edge']} marker at "
                      f"({m['x_mm']:.1f}, {m['y_mm']:.1f}) mm, darkest channel {m['dark']}, under the "
                      f"ring of {', '.join(o['edge'] for o in crossing)}")
                continue
            bad.append(m)
        pairs = 0
        if not leg:
            for i, a in enumerate(op):
                for b in op[i + 1:]:
                    dist = np.hypot(a["x_mm"] - b["x_mm"], a["y_mm"] - b["y_mm"])
                    if dist < 0.5 * max(a["d_mm"], b["d_mm"]):
                        if a["thin"] and b["thin"]:
                            print(f"  see-through (not counted): {axes_name(ax)}: {a['edge']} and "
                                  f"{b['edge']} at ({a['x_mm']:.1f}, {a['y_mm']:.1f}) mm, "
                                  f"{dist:.2f} mm apart")
                            continue
                        if a["on_top"] and b["on_top"] and nested(a, b):
                            print(f"  nested (not counted): {axes_name(ax)}: {a['edge']} and "
                                  f"{b['edge']} at ({a['x_mm']:.1f}, {a['y_mm']:.1f}) mm, "
                                  f"{dist:.2f} mm apart")
                            continue
                        if a["on_top"] and b["on_top"] and a["edge"] != b["edge"]:
                            print(f"  rings on top (not counted): {axes_name(ax)}: {a['edge']} and "
                                  f"{b['edge']} at ({a['x_mm']:.1f}, {a['y_mm']:.1f}) mm, "
                                  f"{dist:.2f} mm apart")
                            continue
                        pairs += 1
                        print(f"  overlap: {axes_name(ax)}: {a['edge']} at ({a['x_mm']:.1f}, "
                              f"{a['y_mm']:.1f}) mm and {b['edge']} at ({b['x_mm']:.1f}, "
                              f"{b['y_mm']:.1f}) mm, {dist:.2f} mm apart")
        ring_d = [np.hypot(a["x_mm"] - b["x_mm"], a["y_mm"] - b["y_mm"])
                  for i, a in enumerate(rg) for b in rg[i + 1:]]
        close = min(ring_d) if ring_d else np.nan
        ring_fail = sum(d < RING_MIN_MM for d in ring_d)
        for m in bad:
            print(f"  centre: {axes_name(ax)}{' (legend)' if leg else ''}: {m['edge']} marker at "
                  f"({m['x_mm']:.1f}, {m['y_mm']:.1f}) mm, darkest channel {m['dark']}")
        n_fail += len(bad) + pairs + ring_fail
        rows.append((axes_name(ax) + (" (legend)" if leg else ""), len(op), len(bad),
                     min((m["dark"] for m in op), default=255),
                     min((m["dark_anchor"] for m in op), default=255), pairs, len(rg), close,
                     ring_fail))
    print(f"{'axes':44s} {'open':>5s} {'centre<200':>10s} {'darkest':>7s} {'anchor':>6s} "
          f"{'overlaps':>8s} {'rings':>5s} {'closest ring mm':>15s} {'rings<1mm':>9s}")
    for r in rows:
        print(f"{r[0]:44s} {r[1]:5d} {r[2]:10d} {r[3]:7d} {r[4]:6d} {r[5]:8d} {r[6]:5d} "
              f"{r[7]:15.2f} {r[8]:9d}")
    return rows, n_fail


def main():
    """Scan every figure that the named scripts save; exit 1 if any pass rule fails."""
    total = 0
    for script in sys.argv[1:] or ["fig_paper_F2_size", "fig_paper_F5_drive"]:
        for fig, pdf in run_and_capture(script):
            _, n_fail = scan(fig, pdf)
            total += n_fail
    print(f"\n{'FAIL' if total else 'PASS'}: {total} rule failures")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
