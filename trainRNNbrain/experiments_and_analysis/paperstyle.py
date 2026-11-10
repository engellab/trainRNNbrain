#!/usr/bin/env python3
"""
House style for the MANUSCRIPT figures (fig_paper_*.py, fig_supp_*.py, fig_ed_*.py): the style sheet
every figure follows, the fonts, the palettes, the number format, and the schematic primitives.

Distinct from `plotstyle.py` on purpose: `plotstyle.py` is the screen-sized, grid-on style for
internal diagnostics. The reader's copy of this file is the paper repo's `docs/figure_style.md`.

STYLE SHEET (dead-relu-paper issues #40 and #15; reference: Tolmachev & Engel 2025, NMI)
==========================================================================================

Print size. The paper is `article`, 11 pt, letter paper, 1 in margins, so \\textwidth is 6.5 in =
165.1 mm and \\textheight is 9 in = 228.6 mm (measured with \\the\\textwidth on 31 Oct 2026). A figure
prints at 100% only if its PDF is exactly as wide as the space LaTeX gives it. So:
  - draw a full-width figure at W_FULL_MM = 165.1 mm, a half-width one at W_HALF_MM = 80 mm (two
    side by side leave a 5.1 mm gutter); make it with `figure(h_mm)` and save it with `save()`,
    which writes the canvas exactly, without cropping, when the width is one of these two;
  - keep a main figure at most MAX_H_MM = 141 mm tall: the main floats cap height at
    0.62 \\textheight (141.7 mm), and a taller figure is shrunk to fit;
  - include it at its own size, `\\includegraphics{fig.pdf}` (or width=\\textwidth, the same thing
    for a 165.1 mm canvas).
  The Nature journal widths (89 mm single, 183 mm double column) apply after conversion to the
  journal class; a 165.1 mm figure fits inside 183 mm unscaled. W1/W2 (88/180 mm) are the old
  widths: a 180 mm figure is printed at 92% (165.1/180), so its 7 pt text prints at 6.4 pt.

Type. One family for every glyph, maths included: Helvetica (macOS), else Arial, else Liberation
Sans, else DejaVu Sans. `setup()` registers the regular, bold, oblique and bold-oblique faces
explicitly (matplotlib reads only the first face of macOS's Helvetica.ttc, which is why panel
letters rendered regular: #15), and sets mathtext to a custom font set of the same family, so
$R^2$, $N$ and $\\tau$ are Helvetica too. `pdffonts` on a figure lists that one family only.
  LABEL_PT          6.5 pt   axis labels, panel titles, legend titles, in-figure text
  TICK_PT           6 pt     tick labels, legend entries, small annotations
  PANEL_LETTER_PT   8.5 pt   bold lower-case panel letters a, b, c, top-left of each panel
  MIN_PT            5 pt     no base text smaller, anywhere
  MIN_SCRIPT_PT     4.2 pt   mathtext sets sub- and superscripts at 70% of their base, so the
                             'rec' of $\\sigma_{rec}$ in 6 pt text is 4.2 pt; nothing below that
Strokes and marks, in points:
  AXIS_LW 0.6 (spines, ticks; ticks 2.4 long), DATA_LW 1.0 (a data line), THIN_LW 0.5 (per-run
  traces, guides, connectors), MARKER_PT 3.0 (marker diameter), SMALL_MARKER_PT 2.0 (one dot per run),
  MARKER_EW 0.35 (the ink edge of every marker), OPEN_MARKER_EW 0.7 (the coloured ring of an open
  marker).
Spacing: LABEL_PAD_PT 4.0 pt between an axis label and its tick labels (matplotlib's own default).
  `save()` raises any smaller pad to it, so a script's `labelpad=1` no longer reaches the page;
  measured between PyMuPDF text boxes, a y-label then sits 1.36 mm from its tick labels
  (0.3 mm at labelpad=1; tests/test_paperstyle.py fails under 1 mm; dead-relu-paper #63).

Colour: one colour per meaning, everywhere, and a colour is never reused for another meaning.
  TASK_COLORS   3BFF blue, CDDM burnt orange, DMTS green, Maze raspberry. Reserved: no rule,
                setting, unit class or annotation uses these hues.
  RULE_COLORS   the training rules, in three hue families so that a family reads as one idea:
                  activity penalties (violet): rate penalty, sparsity penalty, rate + sparsity
                  noise during training (ochre): dropout, synaptic noise
                  prune and reinitialise (magenta): duplicate, rescale
  SETTING       one teal for "a standard training setting changed from its default"; SETTING_RAMP
                (light to dark teal) for the ordered values of one setting (a ladder of weight
                decays, SGD learning rates). Two categorical interventions in one panel take
                SETTING_RAMP[1] and SETTING_RAMP[2] with direct labels.
  BASELINE      light grey: the default or untreated reference every rule is compared against.
  ACTIVE        near-black, INACTIVE light grey: unit classes (a raster, a unit grid, a count).
  Text is always INK/MUTED, never a series colour. Status marks (a severed connection, a limit
  line) are drawn in ink with a word and a line style; there is no red "bad" colour (BAD = INK).
  Validated with the dataviz validator (`validate_palette.js --pairs all`, white surface) on the
  sets that share a panel: the four tasks, worst colour-blind dE 9.7 (DMTS and CDDM; pass,
  target 8), normal-vision worst 19.7 (Maze and CDDM; pass, floor 15); the Figure 5 set
  (baseline, ACTIVE, rate, rate + sparsity, dropout, synaptic noise, duplicate, rescale), worst
  8.0 (pass), normal-vision worst 15.5 (pass, floor 15); the Figure 3 set
  (baseline, ACTIVE, the three penalties, SETTING_RAMP[1] and [2]), worst 8.9, normal-vision 15.9.
  Pairs that fail the checks, and the limitation each one carries (dead-relu-paper#63). Every pair
  of the 18 series colours above (tasks, rules, the four teals, BASELINE, ACTIVE, INACTIVE) was
  run; every pair under 8 in a colour-blind simulation or under 15 in normal vision is listed here.
  The validator and an independent Machado-2009 / OKLab script agree on every value.
  - Tasks and rules are never colour-coded in the same panel: DMTS and rescale are 2.7 apart under
    deuteranopia; 3BFF and sparsity 2.9 under protanopia (11.1 under deuteranopia); 3BFF and rate
    penalty 7.2 under deuteranopia and 14.8 in normal vision; CDDM and dropout 13.2 in normal vision;
    Maze and duplicate 10.5 in normal vision (9.5 under protanopia); Maze and rescale 11.0 in both.
    A panel that crosses tasks with rules gives tasks a facet or a marker and rules the colour.
  - Tasks and the setting teals are never colour-coded in the same panel: DMTS and SETTING_RAMP[1]
    are 7.7 apart under deuteranopia and 8.3 in normal vision; 3BFF and SETTING_RAMP[1] 12.8 in
    normal vision; Maze and SETTING_RAMP[2] 3.3 under deuteranopia, Maze and SETTING_RAMP[3] 5.5
    under protanopia.
  - The setting teals never share a panel with rescale, duplicate or synaptic noise: SETTING
    (SETTING_RAMP[2]) and rescale are 3.7 apart under protanopia; SETTING_RAMP[1] and rescale 6.4
    under deuteranopia; SETTING_RAMP[3] and duplicate 5.2 under protanopia; SETTING and synaptic
    noise 13.7 in normal vision. With the violet penalties and with dropout every teal passes.
  - INACTIVE never shares a panel with BASELINE (8.6 apart in normal vision and under both
    simulations) or with the setting teals (SETTING_RAMP[0]: 3.8 under protanopia, 10.0 in normal
    vision). ACTIVE and SETTING_RAMP[3] (12.4 in normal vision) never share a panel.
  - BASELINE and SETTING_RAMP[0] are 4.4 apart under deuteranopia and 8.3 in normal vision. A sweep
    that uses both draws the default with its own marker shape and names it in the legend (the
    metabolic-penalty and input-weight-scale panels of the Extended Data figure of every
    standard-setting sweep); a marker edge carries no meaning (see Markers).
  - SETTING_RAMP[2] and SETTING_RAMP[3] are 10.7 apart in normal vision. The teals are the ordered
    values of one setting, read by lightness; every value is named in a legend or a direct label.
  - BASELINE has a contrast of 2.35:1 on white, under the validator's 3:1 for a mark. A BASELINE
    series is always named directly (a label beside it, a legend entry, or the caption's "grey,
    ...") and never carries a value that only its colour identifies.
  - SYNAPTIC_NOISE (chroma 0.089) and SETTING (0.07) are below the chroma floor of 0.10, so they
    read close to grey. Neither is the only cue to identity: each comes with its own marker shape
    or a direct label (Fig. 5 gives every rule its own marker).
  - The dark family tints fail the validator's lightness band on purpose: lightness is what tells
    tints apart.
  The figures on paper main of 1 Nov 2026 meet these limitations; checked from which colours each
  figure script draws in each panel, and on a render of the Extended Data figure of every
  standard-setting sweep.

Words and numbers.
  - Axis labels are lower case, a noun phrase with the unit or symbol last: LABELS["active"]
    "active units", LABELS["size"] "network size $N$", LABELS["r2"] "held-out $R^2$",
    LABELS["iteration"] "iteration". Panel titles follow the same case. Symbols are mathtext.
  - Counts of units and iterations use one format, a comma every three digits, in labels, ticks,
    legends and text alike: "1,000", "4,000", "75,000", "200,000" (`fmt_count`, `count_ticks`).
    The paper's prose writes iterations "75{,}000 iterations"; ticks never say "1k" or
    "(thousands)". Numbers below 1,000 have no separator.
  - Panel letters: lower-case bold, PANEL_LETTER_PT, placed by `panel_letter(ax, "a")` a fixed
    distance from the axes' top-left corner (PANEL_LETTER_X_MM left of the left spine,
    PANEL_LETTER_Y_MM above the top), so every letter sits the same way on the page.
  - Output is vector only: `save()` writes PDF and SVG, never PNG.

Markers (dead-relu-paper #83). One convention in every figure: a FILLED marker is a value read at
the floor crossing (the first iteration after which the run's smoothed noisy training loss stays
at or below 1.05 times its loss floor), an OPEN marker a value read at a fixed iteration (75,000
iterations, or the end of a fixed training budget), with that iteration named in the caption. Fill
carries no other meaning anywhere; any other distinction (measured against estimated, one size
against another, one rule against another) uses marker shape, colour or words. Draw them with
`mark_floor(col)` and `mark_fixed(col)` (alias `mark_75k`); name them in a legend with
MARK_FLOOR_LABEL and MARK_75K_LABEL, or f"{fmt_count(it)} iterations" for another fixed iteration.
Edges (Pavel's item 54, round 12): every marker has a thin ink edge, MARKER_EW = 0.35 pt, and the
edge carries no meaning. A filled marker is its series colour inside an ink edge. An open marker is
a white inside, a ring of its series colour (OPEN_MARKER_EW) and an ink rim round the ring
(`open_rim`, a path-effect stroke), so it still reads as open. setup() makes the ink edge the
default of ax.plot / ax.errorbar markers and of ax.scatter; ax.scatter's edge width is not an
rcParam, so pass `linewidths=MARKER_EW` (mark_floor(col, scatter=True) does). Draw an open marker
apart from its error bar (errorbar fmt="none", then ax.plot): the rim would outline the bar too.
Check a figure with `python tests/scan_marker_edges.py <script>`.

Self-checks:
  python trainRNNbrain/experiments_and_analysis/paperstyle.py   swatches + primitives sheet
  python tests/test_paperstyle.py [figure.pdf ...]             fonts, bold face, sizes, page width
"""

import logging
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from fontTools.ttLib import TTCollection
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.transforms import ScaledTranslation

from common import IMG_DIR

# --------------------------------------------------------------------------------------------
# print size
# --------------------------------------------------------------------------------------------

MM = 1.0 / 25.4                 # millimetres -> inches
TEXTWIDTH_MM = 165.1            # \textwidth of main.tex (article, letter, 1 in margins) = 6.5 in
TEXTHEIGHT_MM = 228.6           # \textheight = 9 in
W_FULL_MM = TEXTWIDTH_MM        # a full-width figure prints at 100% at exactly this width
W_HALF_MM = 80.0                # two half-width figures side by side, 5.1 mm gutter
MAX_H_MM = 141.0                # main floats cap height at 0.62 \textheight = 141.7 mm
PRINT_WIDTHS_MM = (W_FULL_MM, W_HALF_MM)
W_FULL, W_HALF = W_FULL_MM * MM, W_HALF_MM * MM     # the same, in inches (figsize units)
# Legacy widths (Nature Communications, 88/180 mm). LaTeX scales a 180 mm figure to 165.1 mm (92%).
W1, W2 = 88 * MM, 180 * MM

# --------------------------------------------------------------------------------------------
# type, strokes, marks (points)
# --------------------------------------------------------------------------------------------

LABEL_PT = 6.5                  # axis labels, panel titles, in-figure text
TICK_PT = 6.0                   # tick labels, legend entries, small annotations
PANEL_LETTER_PT = 8.5           # bold panel letters
MIN_PT = 5.0                    # the smallest base text allowed anywhere
MIN_SCRIPT_PT = 0.7 * TICK_PT   # a sub/superscript of tick-size text (mathtext's fixed 70%)
PANEL_LETTER_X_MM = 7.0         # letter's left edge this far left of the axes' left spine
PANEL_LETTER_Y_MM = 2.0         # letter's baseline this far above the axes' top edge
LABEL_PAD_PT = 4.0              # axis label to tick labels; save() raises any smaller pad to this

AXIS_LW = 0.6                   # spines and ticks
TICK_LEN = 2.4                  # tick length
DATA_LW = 1.0                   # a data line
THIN_LW = 0.5                   # per-run traces, guides, connectors
MARKER_PT = 3.0                 # marker diameter (matplotlib markersize)
SMALL_MARKER_PT = 2.0           # one dot per run in a crowded strip
MARKER_EW = 0.35                # the ink edge of every marker (Pavel's item 54)
OPEN_MARKER_EW = 0.7            # the coloured ring of an open marker

# Font families tried in order; the first with regular, bold and oblique faces wins (`setup`).
FONT_CANDIDATES = ("Helvetica", "Arial", "Liberation Sans", "DejaVu Sans")
HELVETICA_TTC = "/System/Library/Fonts/Helvetica.ttc"   # macOS; faces 0-3 regular/bold/obl./bold-obl.
FONT_CACHE_DIR = os.path.join(matplotlib.get_cachedir(), "paperstyle_fonts")
FONT_FAMILY = None              # set by setup() to the family actually used

# --------------------------------------------------------------------------------------------
# colour, named by meaning
# --------------------------------------------------------------------------------------------

# Neutral ink: text, axes, guides.
INK, MUTED, FAINT, GRID = "#0b0b0b", "#6b6963", "#a8a69e", "#e1e0d9"
PAPER = "#fcfcfb"               # drawing surface; save() exports transparent

# Unit classes and the reference condition.
ACTIVE = "#1a1a1a"              # an active unit
INACTIVE = "#c4c4c4"            # an inactive unit (drawn hollow or light, never omitted)
BASELINE = "#a9a9a9"            # the default / untreated reference; never a series colour

# Tasks. Reserved: these four hues mean a task and nothing else.
TASK_3BFF = "#2a78d6"           # 3-bit flip-flop, blue
TASK_CDDM = "#b35806"           # context-dependent decision making, burnt orange
TASK_DMTS = "#12916a"           # delayed match-to-sample, green
TASK_MAZE = "#941862"           # maze running (PPO), raspberry: the one hue that clears every check
                                # against the other three (validator, 4 Nov 2026); see the limitations
TASK_COLORS = {"3BFF": TASK_3BFF, "CDDM": TASK_CDDM, "DMTS": TASK_DMTS, "Maze": TASK_MAZE}

# Training rules, three hue families. Within a family the members differ in lightness.
RATE_PENALTY = "#7442b9"        # violet: the firing-rate penalty (frm)
SPARSITY_PENALTY = "#ba6de0"    # light violet: the recurrent-weight sparsity penalty (rws)
RATE_SPARSITY = "#4b0083"       # dark violet: both penalties (frm + rws)
DROPOUT = "#aa8d0f"             # ochre: unit dropout during training
SYNAPTIC_NOISE = "#664d00"      # dark ochre: synaptic noise during training
DUPLICATE = "#701034"           # dark magenta: prune a silent unit, duplicate a working one
RESCALE = "#bc3e79"             # magenta: rescale a silent unit's incoming weights
RULE_COLORS = {
    "rate penalty": RATE_PENALTY,
    "sparsity penalty": SPARSITY_PENALTY,
    "rate + sparsity": RATE_SPARSITY,
    "dropout": DROPOUT,
    "synaptic noise": SYNAPTIC_NOISE,
    "duplicate": DUPLICATE,
    "rescale": RESCALE,
}

# Standard training settings (weight decay, noise level, initial scale, optimizer, interventions).
SETTING_RAMP = ["#70c0c9", "#1894a1", "#0e5c64", "#043d43"]    # light -> dark teal, ordered values
SETTING = SETTING_RAMP[2]       # "a setting changed from its default", one colour

# Round 21 (dead-relu-paper #217, Reviewer S figures 4): one code across the paper, colour = task and
# shape = technique or setting. A figure that tells a setting's values apart by colour uses shades of the
# task colour (task_ramp), never the teal SETTING_RAMP. No shade is lighter than 0.62: a lighter filled
# marker prints near white and reads as open (dead-relu-paper PR #216 re-review).
TASK_RAMP_W = (0.62, 0.85, 1.0, 1.35)


def task_shade(col, w):
    """A shade of a colour: mixed with white for w < 1 (lighter), with black for w > 1 (darker, by w - 1).

    Args:
        col: a matplotlib colour; w: float, 1 leaves the colour unchanged.
    Returns:
        str, a hex colour.
    """
    rgb = matplotlib.colors.to_rgb(col)
    if w <= 1:
        return matplotlib.colors.to_hex(tuple(1 - (1 - c) * w for c in rgb))
    return matplotlib.colors.to_hex(tuple(c * (2 - w) for c in rgb))


def task_ramp(task):
    """Four shades of a task's colour, light to dark, for the ordered values of one setting.

    Args:
        task: "3BFF", "CDDM", "DMTS" or "Maze".
    Returns:
        list of four hex colours (TASK_RAMP_W shades of TASK_COLORS[task]).
    """
    return [task_shade(TASK_COLORS[task], w) for w in TASK_RAMP_W]


# Verdict marks are ink plus a word and a line style: no status hue may echo a task hue.
GOOD = BAD = INK

# --- legacy names, kept so every existing figure script still runs ---------------------------
BASE = BASELINE                 # the unpenalised reference
# The old five validated slots. New code uses the named colours above; SLOTS[0] and SLOTS[2] are
# the 3BFF and DMTS task colours, so using them for anything else breaks the task reservation.
SLOTS = ["#2a78d6", "#d94f2b", "#12916a", "#8b46d6", "#a67c00"]
RESCALE_TINT = RESCALE
SYNNOISE_TINT = SYNAPTIC_NOISE
# Condition keys used by the existing scripts, mapped to the meaning-named colours.
COND_COL = {
    "baseline":  BASELINE,
    "none":      BASELINE,
    "dropout":   DROPOUT,
    "mute":      DROPOUT,
    "dead":      DROPOUT,
    "duplicate": DUPLICATE,
    "rescale":   RESCALE,
    "synnoise":  SYNAPTIC_NOISE,
    "frm":       RATE_PENALTY,
    "rws":       SPARSITY_PENALTY,
    "both":      RATE_SPARSITY,
    "frm + rws": RATE_SPARSITY,
}

# --------------------------------------------------------------------------------------------
# words and numbers
# --------------------------------------------------------------------------------------------

LABELS = {
    "active": "active units",
    "inactive": "inactive units",
    "size": "network size $N$",
    "r2": "held-out $R^2$",
    "iteration": "iteration",
    "participation": "participation $p$",
    "loss": "training loss",
}


def fmt_count(n):
    """Format a count of units or iterations the one way the paper does: '1,000', '75,000', '250'.

    Args:
        n: number (int or float); non-integers keep up to 3 significant decimals ('0.5').
    Returns:
        str.
    """
    n = float(n)
    if n == round(n):
        return f"{int(round(n)):,}"
    return f"{n:,.3g}"


def of_n(label, n):
    """Label with the network size it is out of, e.g. of_n('active units', 1000) -> 'active units (of 1,000)'.

    Args:
        label: str; n: the total, a count.
    Returns:
        str.
    """
    return f"{label} (of {fmt_count(n)})"


def count_ticks(axis):
    """Make an axis label its ticks with `fmt_count` ('1,000', '200,000'); no '1k', no offset text.

    On a log axis the minor tick labels are switched off, so only the chosen major ticks are labelled.

    Args:
        axis: a matplotlib Axis, e.g. ax.xaxis.
    Returns:
        None.
    """
    axis.set_major_formatter(FuncFormatter(lambda v, _pos: fmt_count(v)))
    if axis.get_scale() == "log":
        axis.set_minor_formatter(NullFormatter())


# --------------------------------------------------------------------------------------------
# markers: the measurement point (dead-relu-paper #83)
# --------------------------------------------------------------------------------------------

MARK_FLOOR_LABEL = "floor crossing"         # legend words for a filled marker
MARK_75K_LABEL = "75,000 iterations"        # legend words for an open marker read at 75,000;
                                            # another fixed iteration: f"{fmt_count(it)} iterations"


def open_rim():
    """The ink rim drawn round an open marker's coloured ring, as matplotlib path effects.

    An ink stroke OPEN_MARKER_EW + 2 MARKER_EW wide is drawn first; the marker is then drawn over
    it with its white face and its coloured ring, which leaves MARKER_EW of ink outside the ring.
    The white face covers the stroke's inner half, so the inside stays white.

    Returns:
        list of path effects, for the `path_effects` keyword of ax.plot, Line2D or ax.scatter.
    """
    return [patheffects.Stroke(linewidth=OPEN_MARKER_EW + 2 * MARKER_EW, foreground=INK),
            patheffects.Normal()]


def mark_floor(col, scatter=False):
    """Marker colours for a value read at the floor crossing: FILLED, in every figure.

    Fill means the measurement point and nothing else; the caller picks the marker shape and size.
    The edge is ink, MARKER_EW wide, as on every marker.

    Args:
        col: str, the series colour.
        scatter: bool, False for keyword names of ax.plot / ax.errorbar / Line2D (mfc, mec, mew),
            True for ax.scatter (facecolors, edgecolors, linewidths).
    Returns:
        dict of keyword arguments, e.g. ax.plot(x, y, "o", **mark_floor(ps.RATE_PENALTY)).
    """
    if scatter:
        return dict(facecolors=col, edgecolors=INK, linewidths=MARKER_EW)
    return dict(mfc=col, mec=INK, mew=MARKER_EW)


def mark_fixed(col, scatter=False):
    """Marker colours for a value read at a fixed iteration: OPEN, in every figure.

    The fixed iteration is 75,000 iterations or the end of a fixed training budget; the caption
    names which (dead-relu-paper #83, decision update). The marker is a white inside, a ring of the
    series colour OPEN_MARKER_EW wide, and an ink rim MARKER_EW wide round the ring (`open_rim`).
    Use it on markers only (ls="none" or fmt="none" bars drawn separately): the rim outlines any
    line of the same artist.

    Args:
        col: str, the series colour (drawn as the ring).
        scatter: bool, False for ax.plot / ax.errorbar / Line2D keywords (mfc, mec, mew,
            path_effects), True for ax.scatter (facecolors, edgecolors, linewidths, path_effects).
    Returns:
        dict of keyword arguments.
    """
    if scatter:
        return dict(facecolors="white", edgecolors=col, linewidths=OPEN_MARKER_EW,
                    path_effects=open_rim())
    return dict(mfc="white", mec=col, mew=OPEN_MARKER_EW, path_effects=open_rim())


mark_75k = mark_fixed           # the first name of `mark_fixed`, kept for scripts that use it


def open_key(col, marker="o", ms=MARKER_PT, ls="-", lw=DATA_LW, face="white"):
    """A legend key of a line through an open marker, as a (line, marker) pair of artists.

    One Line2D with both a line and `mark_fixed` keywords would draw the ink rim round the line too;
    the pair is drawn by matplotlib's default HandlerTuple, line first, marker on top. Pass the label
    through ax.legend(handles=[...], labels=[...]).

    Args:
        col: str, the series colour; marker: matplotlib marker; ms: marker size (pt);
        ls, lw: line style and width of the key's line; face: the open marker's inside colour.
    Returns:
        tuple (Line2D line, Line2D marker).
    """
    line = Line2D([], [], color=col, ls=ls, lw=lw)
    mark = Line2D([], [], ls="none", marker=marker, ms=ms, **dict(mark_fixed(col), mfc=face))
    return line, mark


# --------------------------------------------------------------------------------------------
# fonts and rcParams
# --------------------------------------------------------------------------------------------

def _weight_of(entry):
    """Numeric weight (100-900) of a matplotlib FontEntry, whose weight may be an int or a name."""
    w = entry.weight
    return w if isinstance(w, int) else font_manager.weight_dict.get(w, 400)


def _helvetica_faces():
    """Write the regular, bold, oblique and bold-oblique faces of macOS Helvetica.ttc as single TTFs.

    matplotlib reads only face 0 (regular) of a .ttc, so a bold request silently falls back to the
    regular face (#15). The four faces are extracted once into FONT_CACHE_DIR (matplotlib's cache,
    outside the repository) and reused.

    Returns:
        list of str, paths of the four .ttf files; empty if Helvetica.ttc is not on this machine.
    """
    if not os.path.exists(HELVETICA_TTC):
        return []
    names = ["Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"]
    paths = [os.path.join(FONT_CACHE_DIR, f"{n}.ttf") for n in names]
    if all(os.path.exists(p) for p in paths):
        return paths
    os.makedirs(FONT_CACHE_DIR, exist_ok=True)
    faces = TTCollection(HELVETICA_TTC).fonts
    for face in faces:
        ps_name = face["name"].getDebugName(6)
        if ps_name in names:
            face.save(paths[names.index(ps_name)])
    return [p for p in paths if os.path.exists(p)]


def _has_three_faces(family):
    """True if `family` resolves to a regular, a bold (>= 600) and an oblique/italic face of itself.

    Args:
        family: str, a font family name.
    Returns:
        bool.
    """
    entries = [f for f in font_manager.fontManager.ttflist if f.name == family]
    regular = any(_weight_of(f) < 600 and f.style == "normal" for f in entries)
    bold = any(_weight_of(f) >= 600 and f.style == "normal" for f in entries)
    oblique = any(f.style in ("italic", "oblique") for f in entries)
    return regular and bold and oblique


def register_fonts():
    """Register the house typeface's faces with matplotlib and return the family used.

    Tries FONT_CANDIDATES in order; Helvetica's faces come from `_helvetica_faces`. Idempotent: the
    faces are added once per process.

    Returns:
        str, the family name (also stored in FONT_FAMILY).
    """
    global FONT_FAMILY
    if FONT_FAMILY is not None:
        return FONT_FAMILY
    # fontTools logs benign notes about Helvetica's tables ('extra bytes in post', 'created
    # timestamp') when extracting the faces and when matplotlib subsets them into a PDF.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    for path in _helvetica_faces():
        font_manager.fontManager.addfont(path)
    for family in FONT_CANDIDATES:
        if _has_three_faces(family):
            FONT_FAMILY = family
            break
    else:
        FONT_FAMILY = "DejaVu Sans"
    if FONT_FAMILY != "Helvetica":
        warnings.warn(f"paperstyle: Helvetica with a bold face not found; using {FONT_FAMILY}")
    return FONT_FAMILY


def setup():
    """Apply the manuscript rcParams. Call once at the top of every figure script.

    Registers the typeface (`register_fonts`), sets every text size, stroke and mark of the style
    sheet, and points mathtext at the same family so no second typeface is embedded. A glyph the
    family lacks (an arrow, $\\propto$, $\\sim$, angle brackets) falls back to STIX, which `pdffonts`
    lists as a second family (STIXGeneral-Regular): write the word, or draw the arrow with `arrow()`.

    Returns:
        str, the font family in use.
    """
    family = register_fonts()
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 400,                # only for raster artists embedded in the vector file
        "savefig.facecolor": PAPER,
        "figure.facecolor": PAPER,
        "font.family": "sans-serif",
        "font.sans-serif": [family, "Arial", "DejaVu Sans"],   # later entries: per-glyph fallback
        "font.size": LABEL_PT,
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.sf": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "mathtext.default": "it",
        "mathtext.fallback": "stixsans",
        "axes.titlesize": LABEL_PT,
        "axes.titleweight": "normal",
        "axes.labelsize": LABEL_PT,
        "axes.labelpad": LABEL_PAD_PT,
        "axes.linewidth": AXIS_LW,
        "axes.edgecolor": "#3a3a36",
        "axes.labelcolor": INK,
        "axes.facecolor": PAPER,
        "axes.grid": False,                # a manuscript axes earns its ink
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.formatter.useoffset": False,
        "axes.unicode_minus": True,        # U+2212, which Helvetica has
        "grid.alpha": 0.22,
        "grid.linewidth": THIN_LW,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": AXIS_LW,
        "ytick.major.width": AXIS_LW,
        "xtick.major.size": TICK_LEN,
        "ytick.major.size": TICK_LEN,
        "xtick.minor.width": AXIS_LW * 0.75,
        "ytick.minor.width": AXIS_LW * 0.75,
        "xtick.minor.size": TICK_LEN * 0.6,
        "ytick.minor.size": TICK_LEN * 0.6,
        "legend.fontsize": TICK_PT,
        "legend.title_fontsize": LABEL_PT,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "lines.linewidth": DATA_LW,
        "lines.markersize": MARKER_PT,
        "lines.markeredgecolor": INK,      # every marker has a thin ink edge (item 54)
        "lines.markeredgewidth": MARKER_EW,
        "scatter.edgecolors": INK,         # scatter's edge width is not an rcParam: pass MARKER_EW
        "errorbar.capsize": 1.8,
        "pdf.fonttype": 42,                # embed TrueType so editors can select the text
        "ps.fonttype": 42,
        "svg.fonttype": "none",            # SVG text stays text, not outlines
    })
    return family


def figure(h_mm, w_mm=W_FULL_MM):
    """A new figure at its printed size, to be laid out with explicit GridSpec margins.

    Args:
        h_mm: printed height in mm (a main figure: at most MAX_H_MM); w_mm: printed width in mm,
            W_FULL_MM (default) or W_HALF_MM.
    Returns:
        matplotlib Figure.
    """
    if h_mm > MAX_H_MM:
        warnings.warn(f"paperstyle.figure: {h_mm} mm is taller than MAX_H_MM = {MAX_H_MM} mm; "
                      "a main float will shrink it")
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))


def panel_letter(ax, letter, dx=None, dy=None, size=PANEL_LETTER_PT, x_mm=PANEL_LETTER_X_MM,
                 y_mm=PANEL_LETTER_Y_MM):
    """Put a bold lower-case panel letter at the top-left of an axes, outside the plot box.

    Default placement is in millimetres from the axes' top-left corner, so every letter sits at the
    same distance from its panel whatever the panel's size. Legacy calls that pass `dx`/`dy` (axes
    fractions) keep their old placement.

    Args:
        ax: axes; letter: the label, e.g. 'a';
        dx, dy: legacy position in axes coordinates (either given -> axes-fraction placement);
        size: font size in points (PANEL_LETTER_PT);
        x_mm: distance of the letter's left edge left of the left spine, mm;
        y_mm: distance of the letter's baseline above the top edge, mm.
    Returns:
        the Text artist.
    """
    if dx is not None or dy is not None:
        x, y = (-0.085 if dx is None else dx), (1.045 if dy is None else dy)
        transform, va = ax.transAxes, "bottom"
    else:
        x, y = 0.0, 1.0
        transform = ax.transAxes + ScaledTranslation(-x_mm * MM, y_mm * MM, ax.figure.dpi_scale_trans)
        va = "baseline"
    return ax.text(x, y, letter, transform=transform, fontsize=size, fontweight="bold",
                   va=va, ha="left", color=INK)


def despine(ax, keep=("left", "bottom")):
    """Hide every spine not named in `keep`.

    Args:
        ax: axes; keep: spine names to leave visible.
    Returns:
        None.
    """
    for side, sp in ax.spines.items():
        sp.set_visible(side in keep)


def ygrid(ax, alpha=0.22):
    """Turn on a recessive horizontal-only grid, behind the data.

    Horizontal only because every quantitative panel in this paper reads a value off the y axis;
    vertical lines would add ink without adding a reading.

    Args:
        ax: axes; alpha: grid opacity.
    Returns:
        None.
    """
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=alpha, linewidth=0.5, color=GRID)
    ax.xaxis.grid(False)


# --------------------------------------------------------------------------------------------
# schematic primitives - the reason this module exists
# --------------------------------------------------------------------------------------------

def unit_grid(ax, x0, y0, n_on, n_total=100, col=ACTIVE, off_col=INACTIVE,
              side=None, pitch=1.0, s=7.0, lw=0.55, zorder=3):
    """Draw a square grid of unit glyphs with `n_on` of them filled - a pictogram of a live fraction.

    This is the paper's recurring visual idiom: the network as 100 dots, filled to the measured
    active fraction. It puts a number the reader must otherwise decode from an axis into a shape
    they read at a glance, and it is the same picture as the quantitative panel below it.

    `pitch` takes a pair because these grids are drawn on schematic axes whose x and y ranges are
    unrelated (a blank axes is usually 0..1 in x and 0..1 in y over a panel that is far from
    square). A single pitch in data units then renders the "square" grid as a row of tall columns,
    which is what happened the first time this was used. Pass (dx, dy) chosen so that dx/dy matches
    the panel's data-to-display aspect, or use `square_pitch` to compute it.

    Args:
        ax: axes; x0, y0: top-left corner in data coordinates; n_on: filled dots;
        n_total: dots in the grid (100 gives one dot per percent); col: fill colour for live units;
        off_col: edge colour for silent units (they are drawn hollow, never omitted - a silent unit
            is present in the network and that is the whole point);
        side: dots per row, default sqrt(n_total); pitch: spacing in data units, a scalar or
            (dx, dy); s: marker area; lw: edge width; zorder: draw order.
    Returns:
        (width, height) of the drawn grid in data units.
    """
    side = int(round(n_total ** 0.5)) if side is None else side
    dx, dy = (pitch, pitch) if np.isscalar(pitch) else pitch
    n_on = int(np.clip(round(n_on), 0, n_total))
    for d in range(n_total):
        r, c = divmod(d, side)
        on = d < n_on
        ax.scatter(x0 + c * dx, y0 - r * dy, s=s,
                   color=col if on else "none",
                   edgecolor=col if on else off_col, linewidth=lw, zorder=zorder)
    rows = int(np.ceil(n_total / side))
    return (side - 1) * dx, (rows - 1) * dy


def square_pitch(ax, dx):
    """The y pitch that makes a grid of x-pitch `dx` look square on this axes.

    Args:
        ax: axes, already given its final xlim/ylim and position; dx: pitch along x in data units.
    Returns:
        dy in data units such that dx and dy span the same number of display points.
    """
    bb = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    px_per_x = bb.width / max(abs(x1 - x0), 1e-12)
    px_per_y = bb.height / max(abs(y1 - y0), 1e-12)
    return dx * px_per_x / max(px_per_y, 1e-12)


def arrow(ax, xy_from, xy_to, col=MUTED, lw=0.9, style="-|>", rad=0.0, ls="-", zorder=2,
          mutation_scale=7, shrink=1.5):
    """Draw an annotated connector between two points in data coordinates.

    CURVATURE. `rad` bows the arc to the LEFT of the direction of travel, so for a pair of opposite
    connectors between the same two nodes one sign gives two arcs on opposite sides (what you want)
    and the endpoints must be the node CENTRES. Offsetting the endpoints perpendicular to the line
    and then bowing produces a pinched bowtie whose heads land on the nodes - which is what the
    first version of Figure 2a did. Give `shrink` in points, large enough to clear the node glyph.

    Args:
        ax: axes; xy_from, xy_to: (x, y) endpoints; col: colour; lw: line width;
        style: arrow style; rad: curvature (0 straight; bows left of travel); ls: line style;
        zorder: draw order; mutation_scale: arrow-head size; shrink: points trimmed from BOTH ends,
            so a connector between node centres stops clear of the glyphs.
    Returns:
        the FancyArrowPatch.
    """
    p = FancyArrowPatch(xy_from, xy_to, arrowstyle=style, lw=lw, color=col, zorder=zorder,
                        linestyle=ls, mutation_scale=mutation_scale,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=shrink, shrinkB=shrink)
    ax.add_patch(p)
    return p


def box(ax, x, y, w, h, label=None, col=MUTED, face="none", lw=0.7, fs=6.5, pad=0.02,
        text_col=None, zorder=2, ls="-"):
    """Draw a rounded box, optionally with a centred label - the unit of a circuit schematic.

    Args:
        ax: axes; x, y: lower-left corner; w, h: size; label: centred text or None;
        col: edge colour; face: fill colour; lw: edge width; fs: label font size;
        pad: corner rounding; text_col: label colour, defaults to `col`; zorder: draw order;
        ls: edge line style.
    Returns:
        the FancyBboxPatch.
    """
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={pad}",
                       linewidth=lw, edgecolor=col, facecolor=face, zorder=zorder, linestyle=ls)
    ax.add_patch(p)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
                color=text_col or col, zorder=zorder + 1)
    return p


def blank(ax):
    """Turn an axes into a bare drawing surface for a schematic: no ticks, no spines, no grid."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    return ax


def strip(ax, xs, groups, cols, width=0.24, jitter=0.055, rng=None, mean_lw=1.6, ms=3.0,
          alpha=0.85, zorder=3):
    """Per-seed dots with a mean bar - the project's standard for "n seeds per condition".

    Every seed is drawn. A bar chart of means over n = 3..7 hides exactly the thing a reader needs
    to judge (the spread, and whether one outlier carries the effect), and this project has already
    retracted one claim that rested on the worst of seven seeds being quoted as the result.

    Args:
        ax: axes; xs: one x position per group; groups: list of 1-D arrays of per-seed values;
        cols: one colour per group; width: half-width of the mean bar; jitter: x jitter sd;
        rng: np.random.Generator for reproducible jitter (seeded by the caller); mean_lw: mean bar
        width; ms: marker size; alpha: dot opacity; zorder: draw order.
    Returns:
        list of (mean, sd, n) per group.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    out = []
    for x, g, c in zip(xs, groups, cols):
        g = np.asarray(g, float)
        g = g[np.isfinite(g)]
        if not len(g):
            out.append((np.nan, np.nan, 0))
            continue
        ax.plot(x + rng.normal(0, jitter, len(g)), g, "o", ms=ms, color=c, alpha=alpha,
                mec=INK, mew=MARKER_EW, zorder=zorder, clip_on=False)
        ax.plot([x - width, x + width], [g.mean()] * 2, "-", lw=mean_lw, color=c,
                zorder=zorder + 1, solid_capstyle="butt")
        out.append((float(g.mean()), float(g.std(ddof=1)) if len(g) > 1 else 0.0, len(g)))
    return out




def _is_print_width(fig):
    """True if the figure is W_FULL_MM or W_HALF_MM wide (within 0.05 mm)."""
    w_mm = fig.get_figwidth() / MM
    return any(abs(w_mm - w) < 0.05 for w in PRINT_WIDTHS_MM)


def floor_label_pads(fig, pad_pt=LABEL_PAD_PT):
    """Raise every axis label's pad in a figure to at least `pad_pt`, so no label touches its ticks.

    A script that passes `labelpad=1` gets a label 0.15-0.3 mm from its tick labels on the page
    (dead-relu-paper #63); this floor overrides any pad below `pad_pt` and leaves larger pads alone.

    Args:
        fig: matplotlib Figure; pad_pt: float, the smallest pad in points (at print size for a
            print-width figure).
    Returns:
        list of str, the text of every axis label whose pad was raised.
    """
    raised = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if axis.label.get_text() and axis.labelpad < pad_pt:
                axis.labelpad = pad_pt
                raised.append(axis.label.get_text())
    return raised


def save(fig, name, w_mm=None, h_mm=None, tight=None, out_dir=IMG_DIR):
    """Save a manuscript figure as PDF + SVG into img/internal_figures. VECTOR ONLY.

    Every panel here is line art, so a raster export only loses: it fixes the resolution at export
    time and cannot be re-typeset. The PDF is the submission asset; the SVG sits beside it because
    the Markdown working documents render an inline <img>, which a PDF is not.

    Canvas. A figure drawn at a print width (W_FULL_MM or W_HALF_MM) is written at exactly its canvas
    size, uncropped, so it prints at 100%; anything drawn outside the canvas would be clipped, and a
    warning names how far it overflows. Any other width keeps the old behaviour, a tight crop with
    0.02 in padding, whose printed scale then depends on the crop.

    Label pads. Before drawing, `floor_label_pads` raises every axis label's pad to LABEL_PAD_PT,
    whatever `labelpad` the script passed, so a GridSpec must leave room for it; a warning gives
    the count and names the labels raised.

    Args:
        fig: the figure; name: file stem;
        w_mm, h_mm: resize the canvas to this printed size first (mm); None keeps the current size;
        tight: True crops to the drawn content, False writes the canvas; None (default) chooses
            False for a print-width figure and True otherwise;
        out_dir: output directory (default img/internal_figures).
    Returns:
        the path of the PDF.
    """
    if w_mm is not None or h_mm is not None:
        fig.set_size_inches((w_mm * MM) if w_mm is not None else fig.get_figwidth(),
                            (h_mm * MM) if h_mm is not None else fig.get_figheight())
    if tight is None:
        tight = not _is_print_width(fig)
    raised = floor_label_pads(fig)
    if raised:
        names = ", ".join(sorted({" ".join(t.split()) for t in raised}))
        warnings.warn(f"paperstyle.save({name}): raised {len(raised)} axis label pads to "
                      f"{LABEL_PAD_PT} pt: {names}")
    if not tight:
        renderer = fig.canvas.get_renderer()
        bb = fig.get_tightbbox(renderer)            # inches
        over_mm = max(-bb.x0, -bb.y0, bb.x1 - fig.get_figwidth(), bb.y1 - fig.get_figheight()) / MM
        if over_mm > 0.2:
            warnings.warn(f"paperstyle.save({name}): content extends {over_mm:.1f} mm outside the "
                          "canvas and will be clipped; adjust the GridSpec margins")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}.pdf")
    # transparent=True, not the PAPER surface: an exported panel carrying the #fcfcfb fill reads as
    # a grey block on the white of a LaTeX page or a slide. The page supplies the surface.
    kw = dict(bbox_inches="tight", pad_inches=0.02) if tight else dict(bbox_inches=None)
    for path in (out, os.path.join(out_dir, f"{name}.svg")):
        fig.savefig(path, transparent=True, **kw)
    plt.close(fig)
    print(f"wrote {out} (+ .svg)")
    return out


def demo_figure():
    """A full-width demo figure that uses every text role of the style sheet (for the self-checks).

    Returns:
        matplotlib Figure, W_FULL_MM x 50 mm, two panels with labels, maths, ticks, a legend and
        panel letters.
    """
    setup()
    fig = figure(50)
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.98, bottom=0.2, top=0.88, wspace=0.3)
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    sizes = np.array([250, 500, 1000, 2000, 4000])
    # one made-up power law per task, so the three task colours are drawn apart
    for (task, col), b in zip(TASK_COLORS.items(), (0.48, 0.44, 0.55)):
        ax_a.plot(sizes, 10 * sizes ** b, "o-", color=col, label=task)
    ax_a.set_xscale("log")
    ax_a.set_xticks(sizes)
    count_ticks(ax_a.xaxis)
    count_ticks(ax_a.yaxis)
    ax_a.set(xlabel=LABELS["size"], ylabel=LABELS["active"])
    ax_a.legend(loc="upper left")
    it = np.linspace(0, 200000, 50)
    ax_b.plot(it, 1 - np.exp(-it / 40000), color=RATE_PENALTY, label="rate penalty")
    ax_b.plot(it, 1 - np.exp(-it / 60000), color=BASELINE, label="no rule")
    ax_b.set_xticks([0, 100000, 200000])
    count_ticks(ax_b.xaxis)
    ax_b.set(xlabel=LABELS["iteration"], ylabel=LABELS["r2"])
    ax_b.text(150000, 0.2, r"$\tau = 2$, $\sigma_{rec}$ = 0.05, $-1$", fontsize=TICK_PT, color=MUTED)
    panel_letter(ax_a, "a")
    panel_letter(ax_b, "b")
    return fig


def _self_check():
    """Draw the palettes and every primitive once and assert the palette invariants. Raises on failure."""
    task_hues = set(TASK_COLORS.values())
    others = list(RULE_COLORS.values()) + SETTING_RAMP + [ACTIVE, INACTIVE, BASELINE, INK, MUTED]
    assert not task_hues & set(others), "a task colour is reused for another meaning"
    assert len(set(RULE_COLORS.values())) == len(RULE_COLORS), "two rules share a colour"
    assert set(COND_COL.values()) <= set(RULE_COLORS.values()) | {BASELINE}, \
        "a legacy condition key points outside the rule palette"
    assert BASE == BASELINE and BASELINE not in RULE_COLORS.values(), "baseline must stay neutral"

    setup()
    fig = figure(60)
    gs = fig.add_gridspec(1, 3, left=0.02, right=0.98, bottom=0.04, top=0.86, wspace=0.08)
    axes = [fig.add_subplot(gs[k]) for k in range(3)]

    ax = blank(axes[0])
    rows = ([(k, c) for k, c in TASK_COLORS.items()] + [(k, c) for k, c in RULE_COLORS.items()]
            + [(f"setting {i}", c) for i, c in enumerate(SETTING_RAMP)]
            + [("baseline", BASELINE), ("active", ACTIVE), ("inactive", INACTIVE)])
    for i, (k, c) in enumerate(rows):
        ax.add_patch(Rectangle((0, -i), 0.25, 0.8, color=c))
        ax.text(0.3, -i + 0.4, f"{k}  {c}", va="center", fontsize=TICK_PT, color=INK)
    ax.set(xlim=(0, 1.4), ylim=(-len(rows), 1.0))
    ax.set_title("colours by meaning", loc="left")

    ax = blank(axes[1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_title("unit_grid: 26 vs 97 of 100 active", loc="left")
    fig.canvas.draw()                                  # square_pitch needs a laid-out axes
    dx = 0.035
    dy = square_pitch(ax, dx)
    assert dy > 0, "square_pitch must return a positive pitch"
    w, _ = unit_grid(ax, 0.04, 0.92, 26, 100, pitch=(dx, dy))
    unit_grid(ax, 0.04 + w + 0.12, 0.92, 97, 100, pitch=(dx, dy))

    ax = blank(axes[2])
    box(ax, 0.05, 0.55, 0.3, 0.3, "unit", col=INK)
    box(ax, 0.65, 0.55, 0.3, 0.3, "read-out", col=MUTED)
    arrow(ax, (0.36, 0.70), (0.64, 0.70), col=INK)
    arrow(ax, (0.36, 0.60), (0.64, 0.50), col=BAD, ls=":", rad=-0.3)
    ax.set(xlim=(0, 1), ylim=(0.2, 1.05))
    ax.set_title("box / arrow", loc="left")

    p = save(fig, "paperstyle_selfcheck")
    assert os.path.exists(p)

    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(W_HALF, 40 * MM))
    res = strip(ax, [0, 1], [rng.normal(263, 13, 7), rng.normal(362, 8, 7)], [BASELINE, RATE_PENALTY])
    assert res[0][2] == 7 and np.isfinite(res[0][0]), "strip must report mean, sd and n"
    plt.close(fig)

    # strip must survive an empty group rather than raising - some cells are still training
    fig, ax = plt.subplots()
    res = strip(ax, [0, 1], [np.array([]), np.array([1.0, 2.0])], [BASELINE, RATE_PENALTY])
    assert res[0][2] == 0 and res[1][2] == 2
    plt.close(fig)

    assert fmt_count(1000) == "1,000" and fmt_count(75000) == "75,000" and fmt_count(250) == "250"
    print(f"paperstyle.py self-check passed (font family: {FONT_FAMILY})")


if __name__ == "__main__":
    _self_check()
