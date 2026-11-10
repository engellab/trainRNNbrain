#!/usr/bin/env python3
"""Extended Data figure: the standard training settings and the training techniques on the maze task (PPO), N = 1,000.

QUESTION. Figure 3 and Figure 5 draw one marker per value; this figure draws every Maze network behind them,
at the stated read (update 2,000) and, for the arms continued to update 4,000 in round 19, at update 4,000
as well (dead-relu-paper#201; entries 2026-11-07_1111, 2026-11-08_0915, 2026-11-09_1221). Issues
engellab/trainRNNbrain#5 (the techniques) and #6 (the settings) asked for the same settings and techniques on
every task.

ROUND 19 (this version). The data come from maze_arms_r19.py, the reader Figure 3 shares, which cuts every
joined record at update 2,000 and at update 4,000 and cross-checks every read against
data/round19/round19_results.json and fz_reruns.json (printed):
  - settings at 2,000 compare with the five controls learned at update 2,000 (uniform seeds 1, 2 and the G12
    seeds 4-6, 394-449 active; entry 2026-11-08_0859); uniform seed 3 (415, success 0.73) is the cross in the
    control column;
  - settings and techniques at 4,000 compare with ReLU N = 1,000 seeds 1-6 at update 4,000, all learned
    (314-404 active, success 1.00);
  - techniques at 2,000 compare with the three uniform-series controls, mean 430 active, success 0.898, bar
    449 (issue #5 as written; decision #194);
  - the rate-penalty, rate + sparsity and metabolic-penalty arms and prune-and-duplicate are their reruns with
    trainer.freeze_after_end=true (data/trained_RNNs_rl_maze_fz); the first runs of those arms are withdrawn
    (entries 2026-11-08_0940, _0943) and are not drawn. Read-out dropout, noisy synapses, rescaling, redraw
    and zero-out keep their first runs;
  - the round 18 parity values (entry 2026-11-07_0136: noisy synapses sigma_w 0.3 and 0.1, read-out dropout at
    rate 0.05, rescaling at target 4) are drawn beside the paper's values of their techniques;
  - the panel of removal curves on the rate-penalty networks (panel h until round 18) is removed: it drew the
    withdrawn networks, and Figure 5d carries the Maze removal.

PANELS (165.1 x FIG_H_MM mm, bold letters).
  (a, b) settings, block 1, at update 2,000: active units (a) and the clean probe's success (b), one OPEN marker
      per network that learned by 2,000, a cross for a network that did not, a cross at 0 with "stopped" for a
      network the rollback guard stopped before 2,000; a bar at the mean over the networks that learned (the
      networks the settings rule reads). The grey band is the learned controls' range of counts, the grey line
      their mean; in (b) the grey line is their mean success and the dashed line the learned level 0.9.
      Weight decay, the input-weight row norm, activity noise, the initial spectral radius, the sparsity
      penalty.
  (c, d) block 2, the same marks: the input-weight cap, the recurrent-weight magnitude penalty, the metabolic
      penalty, AdamW, SGD at four learning rates, re-initialisation of inactive units (redraw, zero-out).
  (e, f) the setting arms continued to update 4,000 (every seed of an arm with a seed not learned at 2,000),
      read at update 4,000 against the six controls there; the same marks. Titles staggered on two rows with
      a thin rule under each, as Figure 3.
  key  one entry per technique value (colour, marker) and a row of marker meanings.
  (g) the techniques at update 2,000: top, active units as a fold change over the three controls' mean (430),
      one open marker per network (a cross for a network without a learned point, a cross at 0 for one
      stopped), a bar at the mean over every network read (the mean the criterion judges), the grey band the
      three controls' range in the same units; bottom, success minus the controls' mean success (0.898), the
      same marks, the tolerance -0.05 dashed. Columns grouped under their technique.
  (h) active units against PPO updates for every technique network and the six controls (grey), one thin line
      per network to the end of its record, a filled marker at its learned point.
  (i) the technique arms continued to update 4,000, as (g) at update 4,000: the fold change over the six
      controls' mean at 4,000 (350.7) and success minus their mean (1.00).

COLOURS. Settings: the controls BASELINE grey, a setting's values on the teal SETTING_RAMP light to dark in the
ladder's order (Figure 3's convention). Techniques: Figure 5's code, the Maze colour with shades per value and the technique shapes (round 20),
a parity value as a lighter (or, for rescaling at target 4, darker) shade of its technique's colour. Rate +
sparsity is a left triangle, as in fig_paper_F5_tasks: Figure 5's earlier filled plus has no white inside once it
is drawn open (tests/scan_open_markers.py, rule 1).

OUTPUT. img/internal_figures/fig_ed_maze_settings_techniques.{pdf,svg} (paperstyle.save, vector only) and, on
stdout: the reader's cross-check, the references, one line per slot of every block, one line per technique
value at each read with its verdict and failing clauses, and the text-overlap check (fig_ed_maze.check_overlaps).

Run:  PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/fig_ed_maze_settings.py
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D

import paperstyle as ps
from common import MAZE_LEARNED_SUCCESS, maze_learned_index
from fig_ed_maze import ax_mm, check_overlaps, time_axis
from fig_ed_maze_interventions import TRI_SCALE
import maze_arms_r19 as M19

N = M19.N
# The settings blocks at update 2,000: (column title, [(arm key, value label), ...]) per setting. The keys are
# the stage-1 folder names maze_arms_r19 reads; the labels are the values as Figure 3 prints them.
BLOCKS = [
    [("control", [("control", "control")]),
     ("weight decay\n(default $10^{-6}$)", [("wd0", "0"), ("wd1e-5", "$10^{-5}$"), ("wd1e-4", "$10^{-4}$")]),
     ("input-weight\nnorm", [("inorm0.5", "0.5"), ("inorm2", "2"), ("inorm5", "5"), ("inorm20", "20")]),
     ("activity noise\n(default 0.05)", [("srec0", "0"), ("srec0.01", "0.01"), ("srec0.1", "0.1")]),
     ("initial spectral\nradius (default 1.2)", [("srad0.5", "0.5"), ("srad1.5", "1.5")]),
     ("sparsity\npenalty", [("rws0.05", "0.05")])],
    [("control", [("control", "control")]),
     ("input-\nweight cap", [("cap0.3", "0.3")]),
     ("magnitude\npenalty", [("rwm0.05", "0.05"), ("rwm0.2", "0.2")]),
     ("metabolic\npenalty", [("met0.01", "0.01"), ("met0.1", "0.1"), ("met1", "1"), ("met10", "10")]),
     ("AdamW", [("adamw", "")]),
     ("SGD,\nlearning rate", [("sgd0.03", "0.03"), ("sgd0.1", "0.1"), ("sgd0.3", "0.3"), ("sgd1", "1")]),
     ("re-initialise\ninactive units", [("redraw", "redraw"), ("zeroout", "zero-out")])],
]
# block 3 (e, f): the same settings, at update 4,000, for the arms with a continued seed (one-line titles,
# staggered on two rows); key prefix -> title
TITLE_4000 = {"wd": "weight decay", "inorm": "input-weight norm", "srec": "activity noise",
              "srad": "initial spectral radius", "rws": "sparsity penalty", "rwm": "magnitude penalty",
              "met": "metabolic penalty", "adamw": "AdamW", "sgd": "SGD, learning rate"}


def shade(col, w):
    """A colour mixed with white (w < 1: lighter) or with black (w > 1: darker, by w - 1).

    Args:
        col: a matplotlib colour; w: float, 1 leaves the colour unchanged.
    Returns:
        str, a hex colour.
    """
    rgb = matplotlib.colors.to_rgb(col)
    if w <= 1:
        return matplotlib.colors.to_hex(tuple(1 - (1 - c) * w for c in rgb))
    return matplotlib.colors.to_hex(tuple(c * (2 - w) for c in rgb))


# The technique values (panels g-i), in drawing order: arm key -> (key-row label, colour, marker, column label in
# panel g, group heading in panel g or None)
# Round 20 (dead-relu-paper #212, Reviewer S figures item 6; PR #216 review): panels g-i use Figure 5's code, the
# task's colour (Maze raspberry) with the technique's marker shape; shades of the raspberry tell a technique's
# values apart. Until round 20 each technique had its own hue family.
MAZE = ps.TASK_MAZE
MAZE_RAMP = [shade(MAZE, 0.62), shade(MAZE, 0.85), MAZE, shade(MAZE, 1.35)]   # a setting's values, light to dark
TECH = {
    "frm0.03":           ("rate penalty 0.03",            shade(MAZE, 0.62),            "o", "0.03", "rate penalty"),
    "frm0.1":            ("rate penalty 0.1",             shade(MAZE, 0.8),             "o", "0.1", "rate penalty"),
    "frm0.3":            ("rate penalty 0.3",             MAZE,             "o", "0.3", "rate penalty"),
    "frm0.1_rws0.05":    ("rate 0.1 + sparsity 0.05",     shade(MAZE, 1.45),  "<", "rate 0.1\n+ sparsity", None),
    "dropout0.2_beta4":  ("read-out dropout 0.2",         MAZE,         "v", "0.2", "read-out\ndropout"),
    "dropout0.05_beta4": ("read-out dropout 0.05",        shade(MAZE, 0.7), "v", "0.05", "read-out\ndropout"),
    "sigmaw1.0":         ("noisy synapses $\\sigma_w$ 1.0", MAZE,      "s", "1.0", "noisy\nsynapses"),
    "sigmaw0.3":         ("noisy synapses $\\sigma_w$ 0.3", shade(MAZE, 0.8), "s", "0.3", "noisy\nsynapses"),
    "sigmaw0.1":         ("noisy synapses $\\sigma_w$ 0.1", shade(MAZE, 0.62), "s", "0.1", "noisy\nsynapses"),
    "rescale":           ("rescaling, $m$ = 10",          MAZE,    "D", "$m$ = 10", "rescaling"),
    "rescale_mild":      ("rescaling, smaller factor", shade(MAZE, 0.8), "D", "smaller\nfactor", "rescaling"),
    "rescale_few":       ("rescaling, fewer units",  shade(MAZE, 0.62), "D", "fewer\nunits", "rescaling"),
    "rescale_tgt4":      ("rescaling, $m$ = 4",           shade(MAZE, 1.4),    "D", "$m$ = 4", "rescaling"),
    "duplicate":         ("prune-and-duplicate",          shade(MAZE, 1.5),       "^", "prune-and-\nduplicate", None),
}
# panel i's column labels (a heading per technique does not fit one column)
LABEL_4000 = {"frm0.3": "rate\n0.3", "frm0.1_rws0.05": "rate 0.1\n+ sparsity", "dropout0.2_beta4": "dropout\n0.2",
              "sigmaw1.0": "$\\sigma_w$\n1.0", "rescale_mild": "smaller\nfactor",
              "rescale_few": "fewer\nunits"}

# layout in millimetres from the canvas's top-left corner
LEFT_MM = 15.0                    # the rows' two-line y labels fit left of it
PITCH_MM = 8.5                    # one value's column in blocks 1 and 2
GAP_MM = 3.5                      # between two settings
PITCH4_MM, GAP4_MM = 7.0, 2.6     # block 3
# round 20 (dead-relu-paper #212, Reviewer S, structure 4): the figure 216 -> 202 mm, so that it and its caption fit
# one page at a print scale of at least 0.9 (rows 14.5/9.5/3.0 mm and tops 9.5/48/90.5 mm until then)
TOP_MM = (9.5, 44.5, 83.5)        # the top of each block's count axes
COUNT_H_MM, SUCCESS_H_MM, ROW_GAP_MM = 12.5, 8.5, 2.5
TITLE4_ROWS_MM = (2.2, 8.8)       # block 3: the titles' baselines above its count axes, staggered (7.4 until round 21, when the lower row wrapped)
TITLE4_DX_MM = {}                 # block 3: a title moved off its slots' centre ("initial spectral radius" -1.75 in round 20)
# Round 21 (dead-relu-paper #217, Reviewer S figures 5): on the lower row of block 3 the titles "input-weight norm",
# "initial spectral radius" and "magnitude penalty" ran together; they are written on two lines each
TITLE4_WRAP = {"input-weight norm": "input-weight\nnorm", "initial spectral radius": "initial spectral\nradius",
               "magnitude penalty": "magnitude\npenalty"}
TITLE_RULE_MM = 0.9               # a thin rule this far under each staggered title
SEED_DX_MM = 1.5                  # the seeds' markers sit this far apart in a settings slot: an open marker's
                                  # ring and rim must not reach its neighbour's centre (tests/scan_open_markers.py)
TRIANGLES = ("^", "v", "<")       # drawn TRI_SCALE larger, as in Figure 5
KEY_MM = (0.0, 113.0, 165.1, 14.5)        # (0, 123.5, 165.1, 13.5) until round 20
G_MM = dict(top=(17.0, 132.5, 145.0, 11.0), bot=(17.0, 146.5, 145.0, 10.0))
H_MM = (17.0, 174.0, 66.0, 19.0)
I_MM = dict(top=(103.5, 174.0, 57.0, 9.5), bot=(103.5, 186.0, 57.0, 7.5))
FIG_H_MM = 202.0
HEAD_MM = -6.6                    # panel g: the group headings' rule, this far below the bottom axes
TECH_DX = 0.22                    # panels g, i: the seeds' markers sit this far apart (column units)


# --------------------------------------------------------------------------------------------
# settings blocks
# --------------------------------------------------------------------------------------------

def slot_positions(groups, pitch=PITCH_MM, gap=GAP_MM):
    """The x position (mm from the axes' left edge) of every value slot and the span of every group.

    Args:
        groups: list of (title, [(key, label), ...]); pitch, gap: mm.
    Returns:
        (slots, spans, width): slots a list of (key, label, x); spans a list of (title, x_centre, x_left,
        x_right) per group; width the axes width in mm.
    """
    slots, spans, x = [], [], pitch / 2
    for g, (title, values) in enumerate(groups):
        if g:
            x += gap
        xs = []
        for key, label in values:
            slots.append((key, label, x))
            xs.append(x)
            x += pitch
        spans.append((title, float(np.mean(xs)), xs[0] - pitch / 2, xs[-1] + pitch / 2))
    return slots, spans, x - pitch / 2


def colour_of(groups):
    """Each arm key's colour: the controls grey, a setting's values in shades of the Maze colour, light to dark.

    Round 20 (dead-relu-paper PR #216 re-review): one Maze colour throughout the figure, as in Figures 1-3 and 5;
    until round 20 the setting values were on the teal ramp. No shade is lighter than 0.62 (a lighter filled
    marker prints near white).

    Args:
        groups: list of (title, [(key, label), ...]).
    Returns:
        dict key -> colour.
    """
    out = {}
    for _, values in groups:
        for k, (key, _) in enumerate(values):
            out[key] = ps.BASELINE if key == "control" else MAZE_RAMP[min(k, len(MAZE_RAMP) - 1)]
    return out


def draw_slot(ax_n, ax_s, x, col, seeds, U, pitch):
    """One value's marks at update U in both rows: a marker per network, crosses, the mean bar.

    Args:
        ax_n, ax_s: the count and success axes (x in mm); x: the slot's centre; col: its colour; seeds: the
        arm's maze_arms_r19 seed rows; U: int, the read; pitch: mm, the slot width.
    Returns:
        str, what was drawn, for the printout.
    """
    xs = x + (np.arange(len(seeds)) - (len(seeds) - 1) / 2) * SEED_DX_MM
    what, learned, n_stop = [], [], 0
    for xi, s in zip(xs, seeds):
        p = s[f"at{U}"]
        if p["reached"]:
            for ax, y in ((ax_n, p["active"]), (ax_s, p["success"])):
                if p["learned"]:
                    ax.plot([xi], [y], "o", ms=ps.MARKER_PT, ls="none", zorder=5, clip_on=False, **ps.mark_fixed(col))
                else:
                    ax.plot([xi], [y], "x", ms=ps.MARKER_PT, mec=col, mew=0.8, ls="none", zorder=5, clip_on=False)
            if p["learned"]:
                learned.append(p)
        else:
            n_stop += 1
            for ax in (ax_n, ax_s):
                ax.plot([xi], [0], "x", ms=ps.MARKER_PT, mec=col, mew=0.8, ls="none", zorder=5, clip_on=False)
        what.append(f"s{s['seed']} {M19.fmt_seed(s, U)}")
    if learned:
        for ax, v in ((ax_n, np.mean([p["active"] for p in learned])), (ax_s, np.mean([p["success"] for p in learned]))):
            ax.plot([x - 0.36 * pitch, x + 0.36 * pitch], [v] * 2, "-", lw=1.4, color=col, zorder=4,
                    solid_capstyle="butt")
    if n_stop:
        ax_n.text(x, 0.08 * N, "stopped", fontsize=ps.TICK_PT, color=ps.MUTED, ha="center", va="bottom")
    return "; ".join(what)


def axes_rows(fig, top, width, U, ref, letters):
    """The count and success axes of one block, with the reference band and lines and the axis labels.

    Args:
        fig: the figure; top: mm; width: mm; U: int, the read; ref: maze_arms_r19.settings_reference at U;
        letters: (count letter, success letter).
    Returns:
        (ax_n, ax_s).
    """
    ax_n = ax_mm(fig, LEFT_MM, top, width, COUNT_H_MM)
    ax_s = ax_mm(fig, LEFT_MM, top + COUNT_H_MM + ROW_GAP_MM, width, SUCCESS_H_MM)
    ax_n.axhspan(ref["active_min"], ref["active_max"], color=ps.FAINT, alpha=0.3, lw=0, zorder=1)
    ax_n.axhline(ref["active"], lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax_s.axhline(ref["success"], lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax_s.axhline(MAZE_LEARNED_SUCCESS, ls=(0, (4, 3)), lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    for ax in (ax_n, ax_s):
        ax.set_xlim(0, width)
        ps.ygrid(ax)
    ax_n.set_ylim(0, N)
    ax_n.set_yticks([0, N / 2, N])
    ps.count_ticks(ax_n.yaxis)
    ax_n.set_ylabel("active\nunits")
    ax_n.text(0.004, 0.97, f"update {U:,}", transform=ax_n.transAxes, fontsize=ps.LABEL_PT, color=ps.INK,
              ha="left", va="top", fontweight="bold")
    ax_n.tick_params(axis="x", length=0, labelbottom=False)
    ax_s.set_ylim(0, 1.03)
    # round 20 (dead-relu-paper #212, Reviewer S, figures item 4): the learned level is named by its tick
    # label; the "0.9" written inside the axes overlapped the control markers
    ax_s.set_yticks([0, 0.5, MAZE_LEARNED_SUCCESS])
    ax_s.set_yticklabels(["0", "0.5", f"{MAZE_LEARNED_SUCCESS:g}"])
    ax_s.set_ylabel("success\nrate")
    ax_s.tick_params(axis="x", length=0, pad=3)
    ps.panel_letter(ax_n, letters[0])
    ps.panel_letter(ax_s, letters[1])
    return ax_n, ax_s


def draw_block(fig, top, groups, controls, arms, U, ref, letters, pitch=PITCH_MM, gap=GAP_MM, stagger=False):
    """One settings block at update U: the count row and the success row.

    Args:
        fig: the figure; top: mm; groups: list of (title, [(key, label), ...]); controls, arms: maze_arms_r19
        output; U: int, the read; ref: settings_reference at U; letters: (count, success) letters; pitch,
        gap: mm; stagger: bool, the titles on two rows with a rule under each (block 3), else one row.
    Returns:
        list of (key, what was drawn, direction at U) per slot.
    """
    slots, spans, width = slot_positions(groups, pitch, gap)
    cols = colour_of(groups)
    ax_n, ax_s = axes_rows(fig, top, width, U, ref, letters)
    out = []
    for key, label, x in slots:
        seeds = controls if key == "control" else arms[key]["seeds"]
        what = draw_slot(ax_n, ax_s, x, cols[key], seeds, U, pitch)
        d = "reference" if key == "control" else M19.setting_direction(seeds, ref, U)
        out.append((key, what, d))
    for i, (title, x, x0, x1) in enumerate(spans):
        if stagger:
            y_mm = TITLE4_ROWS_MM[i % 2]
            ax_n.text(x + TITLE4_DX_MM.get(title, 0.0), 1.0 + y_mm / COUNT_H_MM, TITLE4_WRAP.get(title, title),
                      transform=ax_n.get_xaxis_transform(), fontsize=ps.LABEL_PT, color=ps.INK, ha="center",
                      va="bottom", linespacing=1.0)
            ax_n.plot([x0, x1], [1.0 + (y_mm - TITLE_RULE_MM) / COUNT_H_MM] * 2, transform=ax_n.get_xaxis_transform(),
                      color=ps.MUTED, lw=ps.THIN_LW, clip_on=False, zorder=1)
        else:
            ax_n.text(x, 1.04, title, transform=ax_n.get_xaxis_transform(), fontsize=ps.LABEL_PT,
                      color=ps.INK, ha="center", va="bottom", linespacing=1.15)
    for ax in (ax_n, ax_s):
        ax.set_xticks([x for _, _, x in slots])
    ax_s.set_xticklabels([label for _, label, _ in slots], fontsize=ps.TICK_PT, linespacing=1.25)
    return out


def block_4000(arms):
    """The groups of block 3: every settings arm of BLOCKS with a seed continued past update 2,000.

    Args:
        arms: maze_arms_r19.load_arms output.
    Returns:
        list of (title, [(key, label), ...]), the control first, in BLOCKS' order.
    """
    groups = [("control", [("control", "control")])]
    for block in BLOCKS:
        for _, values in block:
            keep = [(k, lab) for k, lab in values if k != "control" and k in arms
                    and any(s["continued"] for s in arms[k]["seeds"])]
            if not keep:
                continue
            stem = next(t for t in TITLE_4000 if keep[0][0].startswith(t))
            groups.append((TITLE_4000[stem], [(k, lab if lab else "") for k, lab in keep]))
    return groups


# --------------------------------------------------------------------------------------------
# technique panels
# --------------------------------------------------------------------------------------------

def tech_marker(ax, x, y, key, learned):
    """One technique network's marker: open in its value's shape if learned, a cross if not.

    Args:
        ax: axes; x, y: data coordinates; key: a TECH key; learned: bool.
    Returns:
        None.
    """
    _, col, marker = TECH[key][:3]
    if learned:
        ax.plot([x], [y], marker, ms=ps.MARKER_PT * (TRI_SCALE if marker in TRIANGLES else 1.0), ls="none",
                zorder=5, clip_on=False, **ps.mark_fixed(col))
    else:
        ax.plot([x], [y], "x", ms=ps.MARKER_PT, mec=col, mew=0.8, ls="none", zorder=5, clip_on=False)


def panel_fold(ax_t, ax_b, keys, arms, tref, U, labels, headings):
    """Panels g and i: fold change over the controls' mean (top) and each network's success rate (bottom).

    Round 20 (dead-relu-paper #212, Reviewer W, figures item 1, as in Figure 5b): the bottom row draws the
    success rate itself, with the learned level 0.9 dashed and the controls' mean as a grey line, in place of
    the success rate minus the controls' mean against the tolerance 0.05, because the 0.9 condition decides
    every Maze verdict the success rate decides. The printout keeps the difference.

    Args:
        ax_t, ax_b: the two axes; keys: the TECH keys, one column each; arms: maze_arms_r19 output; tref:
        maze_arms_r19.technique_reference at U; U: int, the read; labels: list of column labels; headings:
        bool, draw each technique's heading under its columns.
    Returns:
        list of (key, verdict, clauses, per-seed text, mean fold, mean success difference) per column.
    """
    rows = []
    folds_all = [1.0]
    for i, key in enumerate(keys):
        seeds = arms[key]["seeds"] if key in arms else []
        col = TECH[key][1]
        read = [s for s in seeds if s[f"at{U}"]["reached"]]
        n_stop = len(seeds) - len(read) if seeds else len(M19.SEEDS)
        n = len(read) + n_stop
        xs = [i + (j - (n - 1) / 2) * TECH_DX for j in range(n)]
        fold = [s[f"at{U}"]["active"] / tref["active"] for s in read]
        succ = [s[f"at{U}"]["success"] for s in read]
        cost = [v - tref["success"] for v in succ]
        for xj, s, f, c in zip(xs, read, fold, succ):
            tech_marker(ax_t, xj, f, key, s[f"at{U}"]["learned"])
            tech_marker(ax_b, xj, c, key, s[f"at{U}"]["learned"])
        for xj in xs[len(read):]:
            ax_t.plot([xj], [0], "x", ms=ps.MARKER_PT, mec=col, mew=0.8, ls="none", zorder=5, clip_on=False)
        if n_stop and not read:
            # on a paper patch, so the grey line at a fold change of 1 does not run through the word (round 20)
            ax_t.text(i, 0.12, "stopped", fontsize=ps.TICK_PT, color=ps.MUTED, ha="center", va="bottom",
                      transform=ax_t.get_xaxis_transform(), rotation=90, zorder=3,
                      bbox=dict(boxstyle="square,pad=0.1", fc=ps.PAPER, ec="none"))
        if read:
            for ax, v in ((ax_t, np.mean(fold)), (ax_b, np.mean(succ))):
                ax.plot([i - 0.34, i + 0.34], [v] * 2, "-", lw=1.4, color=col, zorder=4, solid_capstyle="butt")
            folds_all += fold
        v, cl = M19.technique_verdict(seeds, tref, U) if seeds else ("no verdict", [])
        txt = "; ".join(f"s{s['seed']} {M19.fmt_seed(s, U)}" for s in seeds) or "3 stopped before a checkpoint"
        rows.append((key, v, cl, txt, float(np.mean(fold)) if fold else None, float(np.mean(cost)) if cost else None))
    ax_t.axhspan(tref["active_min"] / tref["active"], tref["active_max"] / tref["active"], color=ps.FAINT,
                 alpha=0.3, lw=0, zorder=1)
    ax_t.axhline(1, lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax_t.set_ylim(0, np.ceil(max(folds_all) * 1.12 * 2) / 2)
    ax_t.set_yticks(np.arange(0, ax_t.get_ylim()[1] + 1e-9, 1.0))
    ax_t.set_ylabel("fold\nchange")
    ax_t.tick_params(axis="x", length=0, labelbottom=False)
    ax_b.set_ylim(0, 1.03)
    ax_b.axhline(tref["success"], lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax_b.axhline(MAZE_LEARNED_SUCCESS, ls=(0, (4, 3)), lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax_b.set_yticks([0, 0.5, MAZE_LEARNED_SUCCESS])
    ax_b.set_yticklabels(["0", "0.5", f"{MAZE_LEARNED_SUCCESS:g}"])
    ax_b.set_ylabel("success\nrate")
    ax_b.set_xticks(range(len(keys)))
    ax_b.set_xticklabels(labels, fontsize=ps.TICK_PT, linespacing=1.2)
    ax_b.tick_params(axis="x", length=0, pad=3)
    for ax in (ax_t, ax_b):
        ax.set_xlim(-0.6, len(keys) - 0.4)
        ps.ygrid(ax)
    if headings:
        height_mm = ax_b.get_position().height * ax_b.figure.get_figheight() / ps.MM
        y_rule = HEAD_MM / height_mm
        i = 0
        while i < len(keys):
            h, j = TECH[keys[i]][4], i
            while j + 1 < len(keys) and TECH[keys[j + 1]][4] == h:
                j += 1
            if h is not None:
                ax_b.plot([i - 0.4, j + 0.4], [y_rule] * 2, transform=ax_b.get_xaxis_transform(), color=ps.MUTED,
                          lw=ps.THIN_LW, clip_on=False, solid_capstyle="butt")
                ax_b.text((i + j) / 2, y_rule - 0.4 / height_mm, h, transform=ax_b.get_xaxis_transform(),
                          fontsize=ps.TICK_PT, color=ps.INK, ha="center", va="top", linespacing=1.1)
            i = j + 1
    return rows


def panel_traces(ax, keys, arms, controls, x_max):
    """Panel h: active units against PPO updates, one line per network, a filled marker at the learned point.

    Args:
        ax: axes; keys: the TECH keys drawn; arms, controls: maze_arms_r19 output; x_max: int, the x limit.
    Returns:
        int, the number of networks drawn.
    """
    n = 0
    lines = [(c, ps.BASELINE, "o", 2) for c in controls]
    lines += [(s, TECH[k][1], TECH[k][2], 3) for k in keys if k in arms for s in arms[k]["seeds"]]
    for s, col, marker, z in lines:
        r = s["run"]
        if r is None:
            continue
        pr = r["probes"]
        ax.plot(pr["update"], pr["active"], "-", lw=ps.THIN_LW, color=col, zorder=z)
        li = maze_learned_index(pr["probe_success"])
        if li is not None:
            ax.plot([pr["update"][li]], [pr["active"][li]], marker, ls="none", zorder=5,
                    ms=ps.MARKER_PT * (TRI_SCALE if marker in TRIANGLES else 1.0), **ps.mark_floor(col))
        n += 1
    time_axis(ax, x_max)
    ax.set(ylim=(0, N), xlabel="PPO update", ylabel=ps.of_n(ps.LABELS["active"], N))
    ax.set_yticks([0, N / 2, N])
    ps.count_ticks(ax.yaxis)
    ps.ygrid(ax)
    return n


def panel_key(ax, keys):
    """The key row: one entry per technique value (colour, marker) and a row of marker meanings.

    Args:
        ax: a blank axes spanning the figure width; keys: the TECH keys.
    Returns:
        None.
    """
    ps.blank(ax)
    h = [Line2D([], [], marker="o", color=ps.BASELINE, ls="-", lw=ps.DATA_LW, ms=3.4, mec=ps.INK,
                mew=ps.MARKER_EW, label="control")]
    h += [Line2D([], [], marker=TECH[k][2], color=TECH[k][1], ls="-", lw=ps.DATA_LW, ms=3.4, mec=ps.INK,
                 mew=ps.MARKER_EW, label=TECH[k][0]) for k in keys]
    leg = ax.legend(handles=h, loc="upper center", ncol=5, frameon=False, fontsize=ps.TICK_PT, handlelength=1.5,
                    columnspacing=1.1, handletextpad=0.35, borderaxespad=0)
    ax.add_artist(leg)
    h2 = [Line2D([], [], marker="o", ls="none", ms=ps.MARKER_PT, **ps.mark_floor(ps.INK), label="learned point (h)"),
          Line2D([], [], marker="o", ls="none", ms=ps.MARKER_PT, **ps.mark_fixed(ps.INK),
                 label="learned by the update read"),
          Line2D([], [], marker="x", ls="none", ms=ps.MARKER_PT, mec=ps.INK, mew=0.8,
                 label="not learned by the update read; at 0, stopped")]
    ax.legend(handles=h2, loc="lower center", ncol=len(h2), frameon=False, fontsize=ps.TICK_PT,
              handlelength=1.0, columnspacing=1.6, handletextpad=0.35, borderaxespad=0)


def main():
    """Load every network, draw the three settings blocks and the technique panels, print every number, save."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    controls, arms = M19.load_controls(), M19.load_arms()
    M19.crosscheck(controls, arms)
    sref = {U: M19.settings_reference(controls, U) for U in M19.READS}
    tref = {U: M19.technique_reference(controls, U) for U in M19.READS}
    for U in M19.READS:
        print(f"settings reference at {U}: {sref[U]}\ntechnique reference at {U}: {tref[U]}")
    ps.setup()
    fig = ps.figure(FIG_H_MM)
    groups4 = block_4000(arms)
    for top, groups, letters, U, kw in ((TOP_MM[0], BLOCKS[0], ("a", "b"), 2000, {}),
                                        (TOP_MM[1], BLOCKS[1], ("c", "d"), 2000, {}),
                                        (TOP_MM[2], groups4, ("e", "f"), 4000,
                                         dict(pitch=PITCH4_MM, gap=GAP4_MM, stagger=True))):
        print(f"\n--- settings, panels {letters[0]} and {letters[1]}, update {U:,} (direction against "
              f"{sref[U]['active_min']}-{sref[U]['active_max']}, learned seeds only) ---")
        for key, what, d in draw_block(fig, top, groups, controls, arms, U, sref[U], letters, **kw):
            seeds = controls if key == "control" else arms[key]["seeds"]
            read = [s[f"at{U}"] for s in seeds if s[f"at{U}"]["reached"]]
            dperf = (np.mean([p["success"] for p in read]) - sref[U]["success"]) if read else float("nan")
            src = "" if key == "control" else f" [{arms[key]['source']}]"
            print(f"  {key:>14}{src}: {d}; success minus the learned controls' mean {dperf:+.3f}; {what}")
    keys2 = list(TECH)
    keys4 = [k for k in TECH if k in arms and any(s["continued"] for s in arms[k]["seeds"])]
    panel_key(ax_mm(fig, *KEY_MM), keys2)
    ax_gt, ax_gb = ax_mm(fig, *G_MM["top"]), ax_mm(fig, *G_MM["bot"])
    rows2 = panel_fold(ax_gt, ax_gb, keys2, arms, tref[2000], 2000, [TECH[k][3] for k in keys2], headings=True)
    ps.panel_letter(ax_gt, "g")
    ax_gt.text(0.0, 1.06, "techniques at update 2,000", transform=ax_gt.transAxes, fontsize=ps.LABEL_PT,
               color=ps.INK, ha="left", va="bottom")
    ax_h = ax_mm(fig, *H_MM)
    n_h = panel_traces(ax_h, keys2, arms, controls, 4000)
    ps.panel_letter(ax_h, "h")
    ax_it, ax_ib = ax_mm(fig, *I_MM["top"]), ax_mm(fig, *I_MM["bot"])
    rows4 = panel_fold(ax_it, ax_ib, keys4, arms, tref[4000], 4000, [LABEL_4000[k] for k in keys4], headings=False)
    ps.panel_letter(ax_it, "i")
    ax_it.text(0.0, 1.08, "techniques at update 4,000", transform=ax_it.transAxes, fontsize=ps.LABEL_PT,
               color=ps.INK, ha="left", va="bottom")
    for U, rows in ((2000, rows2), (4000, rows4)):
        print(f"\n--- techniques at update {U:,}: fold over {tref[U]['active']:.1f} (controls {tref[U]['seeds']}, bar "
              f"{tref[U]['active_max']}), success minus {tref[U]['success']:.3f} ---")
        for key, v, cl, txt, mf, mc in rows:
            src = f" [{arms[key]['source']}]" if key in arms else ""
            print(f"  {key:>18}{src}: {v}{' (' + ','.join(cl) + ')' if cl else ''}; mean fold "
                  f"{'-' if mf is None else f'{mf:.2f}'}; success difference {'-' if mc is None else f'{mc:+.3f}'}; {txt}")
    print(f"\npanel h: {n_h} networks drawn")
    check_overlaps(fig)
    ps.save(fig, "fig_ed_maze_settings_techniques")


if __name__ == "__main__":
    main()
