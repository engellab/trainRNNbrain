#!/usr/bin/env python3
"""Extended Data figure: the standard training settings on the maze task (PPO) at N = 1,000.

QUESTION. Figure 3 shows, on CDDM and the 3-bit flip-flop, that changing a standard training
setting moves the number of active units but no setting stops its fall. Issue
engellab/trainRNNbrain#6 (Pavel, 5 Nov) asks for the same settings on every task, so every value
of Figure 3's ladders was run on the maze (slurm/rl_maze_interventions_spock.slurm; the inventory in
docs/interventions_matrix.md). This is the maze counterpart of Figure 3a: one column per setting,
its values along the column, the controls beside them.

PANELS (165.1 x FIG_H_MM mm, bold letters). Two blocks, each of two rows that share their columns:
  (a, c) active units at the end of training (update 2,000), one OPEN marker per run (paperstyle,
      dead-relu-paper#83: an open marker is a fixed iteration) and a bar at the value's mean over
      its finished runs; a run that did not learn (common.maze_learned_index) is a cross in the
      value's colour, as in fig_ed_maze; a run the rollback guard stopped is a cross at 0 with
      "stopped" printed once in its column; a value with no finished run says "unfinished"; a value
      with some runs still training says how many finished. The learned controls' range of counts
      is a band across the row, their mean a line (the uniform series' N = 1,000 ReLU runs; the
      pre-registered exclusion of a run that does not learn applies to the controls);
  (b, d) the clean probe's success rate at the end, the same marks; the controls' mean success as a
      line and the "learned" level (0.9) dashed.
  Block 1 (a, b): weight decay, input-weight row norm, activity noise, initial spectral radius, the
      sparsity penalty alone. Block 2 (c, d): the input-weight cap, the recurrent-weight magnitude
      penalty, the metabolic penalty, AdamW, SGD at four learning rates, the re-initialisation of
      inactive units (random redraw; zero-out of the outgoing weights as well). Each block starts
      with the controls. Colours: the controls BASELINE grey; a setting's values on the teal
      SETTING_RAMP, light to dark in the ladder's order (Figure 3's convention); no task colour.

ARMS ARE DERIVED FROM EACH RUN'S SAVED CONFIG (fig_ed_maze_interventions.arm_of), never from the
folder name; load_records reports a folder whose name does not match. The verdicts on these arms
(a setting "changes the count" when all three seeds lie outside the learned controls' range on one
side) are read_maze_arms.py's; this figure draws the numbers behind them.

DATA. --root (default data/trained_RNNs_rl_maze/Maze_interventions), --control-root (default
data/trained_RNNs_rl_maze/Maze_suite), as fig_ed_maze_interventions.py.

OUTPUT. img/internal_figures/fig_ed_maze_settings.{pdf,svg} (paperstyle.save, vector only) and,
on stdout, one line per column (the value, its runs' counts and success rates, what is drawn) and
the text-overlap check (fig_ed_maze.check_overlaps).

Run:  python trainRNNbrain/experiments_and_analysis/fig_ed_maze_settings.py [--root DIR]
      [--control-root DIR]
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")

import paperstyle as ps
from common import MAZE_LEARNED_SUCCESS
from fig_ed_maze import ax_mm, check_overlaps
from fig_ed_maze_interventions import CONTROL_ROOT, N, ROOT, SEEDS, control_reference, load_records

# The two blocks: (column title, [(arm key, value label), ...]) per setting, in drawing order. The
# keys are fig_ed_maze_interventions.arm_of's; the labels are the values as Figure 3 prints them.
BLOCKS = [
    [("control", [("control", "control")]),
     ("weight decay\n(default $10^{-6}$)", [("wd0", "0"), ("wd1e-5", "$10^{-5}$"), ("wd1e-4", "$10^{-4}$")]),
     ("input-weight\nrow norm", [("inorm0.5", "0.5"), ("inorm2", "2"), ("inorm5", "5"), ("inorm20", "20")]),
     ("activity noise\n(default 0.05)", [("srec0", "0"), ("srec0.01", "0.01"), ("srec0.1", "0.1")]),
     ("initial spectral\nradius (default 1.2)", [("srad0.5", "0.5"), ("srad1.5", "1.5")]),
     ("sparsity\npenalty", [("rws0.05", "0.05")])],
    [("control", [("control", "control")]),
     ("input-\nweight cap", [("cap0.3", "0.3")]),
     ("recurrent-weight\npenalty", [("rwm0.05", "0.05"), ("rwm0.2", "0.2")]),
     ("metabolic\npenalty", [("met0.01", "0.01"), ("met0.1", "0.1"), ("met1", "1"), ("met10", "10")]),
     ("AdamW", [("adamw", "")]),
     ("SGD,\nlearning rate", [("sgd0.03", "0.03"), ("sgd0.1", "0.1"), ("sgd0.3", "0.3"), ("sgd1", "1")]),
     ("re-initialise\ninactive units", [("redraw", "redraw"), ("zeroout", "zero-out")])],
]

# layout in millimetres: one slot per value, a gap between settings, two rows per block
LEFT_MM = 14.0                    # the rows' y labels (two lines) fit left of it
PITCH_MM = 8.5                    # one value's column
GAP_MM = 3.5                      # between two settings
TOP_MM = (14.0, 79.0)             # the top of each block's count axes
COUNT_H_MM, SUCCESS_H_MM, ROW_GAP_MM = 25.0, 18.0, 2.0
FIG_H_MM = 133.0
SEED_DX_MM = 0.9                  # the three seeds' markers sit this far apart


def slot_positions(groups):
    """The x position (mm from the axes' left edge) of every value slot and of every group's centre.

    Args:
        groups: one block of BLOCKS.
    Returns:
        (slots, centres, width): slots is a list of (key, label, x) in order; centres a list of
        (title, x) per group; width the axes width in mm.
    """
    slots, centres, x = [], [], PITCH_MM / 2
    for g, (title, values) in enumerate(groups):
        if g:
            x += GAP_MM
        xs = []
        for key, label in values:
            slots.append((key, label, x))
            xs.append(x)
            x += PITCH_MM
        centres.append((title, float(np.mean(xs))))
    return slots, centres, x - PITCH_MM / 2


def colour_of(groups):
    """Each arm key's colour: the controls grey, a setting's values on the teal ramp light to dark.

    Args:
        groups: one block of BLOCKS.
    Returns:
        dict key -> colour.
    """
    out = {}
    for _, values in groups:
        for k, (key, _) in enumerate(values):
            out[key] = ps.BASELINE if key == "control" else ps.SETTING_RAMP[min(k, len(ps.SETTING_RAMP) - 1)]
    return out


def runs_of(records, key):
    """The records of one arm: the controls from the series, every other arm from the ladder root.

    Args:
        records: list from fig_ed_maze_interventions.load_records; key: an arm key.
    Returns:
        (fin, running, stopped): finished records (by seed), records still training, records the
        guard stopped.
    """
    source = "control" if key == "control" else "ladder"
    mine = sorted([r for r in records if r["key"] == key and r["source"] == source], key=lambda r: r["seed"])
    fin = [r for r in mine if r["finished"]]
    stopped = [r for r in mine if not r["finished"] and r["stopped_at"] is not None]
    running = [r for r in mine if not r["finished"] and r["stopped_at"] is None]
    return fin, running, stopped


def draw_slot(ax_n, ax_s, x, key, col, fin, running, stopped, n_no_files):
    """One value's marks in both rows: a marker per finished run, crosses for the stopped, a mean bar.

    Args:
        ax_n, ax_s: the count and success axes (x in mm); x: the slot's centre; key: the arm key;
        col: its colour; fin, running, stopped: from runs_of; n_no_files: int, the arm's folders
        that hold no run files (stopped before their first checkpoint).
    Returns:
        str, what was drawn, for the printout.
    """
    marks = fin + stopped
    n_marks = len(marks) + n_no_files
    xs = x + (np.arange(n_marks) - (n_marks - 1) / 2) * SEED_DX_MM
    what = []
    for xi, r in zip(xs, marks):
        if r["finished"]:
            y_n, y_s = r["pt"]["active_end"], r["pt"]["success_end"]
            if r["pt"]["learned_i"] is None:
                for ax, y in ((ax_n, y_n), (ax_s, y_s)):
                    ax.plot([xi], [y], "x", ms=ps.MARKER_PT, mec=col, mew=0.8, ls="none", zorder=5, clip_on=False)
            else:
                for ax, y in ((ax_n, y_n), (ax_s, y_s)):
                    ax.plot([xi], [y], "o", ms=ps.MARKER_PT, ls="none", zorder=5, clip_on=False, **ps.mark_fixed(col))
            what.append(f"s{r['seed']} {y_n} {y_s:.2f}" + ("" if r["pt"]["learned_i"] is not None else " (not learned)"))
        else:
            for ax in (ax_n, ax_s):
                ax.plot([xi], [0], "x", ms=ps.MARKER_PT, mec=col, mew=0.8, ls="none", zorder=5, clip_on=False)
            what.append(f"s{r['seed']} stopped@{r['stopped_at']}")
    for xi in xs[len(marks):]:
        for ax in (ax_n, ax_s):
            ax.plot([xi], [0], "x", ms=ps.MARKER_PT, mec=col, mew=0.8, ls="none", zorder=5, clip_on=False)
    if n_no_files:
        what.append(f"{n_no_files} stopped before a checkpoint")
    if len(fin) == len(SEEDS):
        for ax, v in ((ax_n, np.mean([r["pt"]["active_end"] for r in fin])),
                      (ax_s, np.mean([r["pt"]["success_end"] for r in fin]))):
            ax.plot([x - 0.36 * PITCH_MM, x + 0.36 * PITCH_MM], [v] * 2, "-", lw=1.4, color=col,
                    zorder=4, solid_capstyle="butt")
    if stopped or n_no_files:
        ax_n.text(x, 0.08 * N, "stopped", fontsize=ps.TICK_PT, color=ps.MUTED, ha="center", va="bottom")
    if running and not fin:
        ax_n.text(x, 0.5, "unfinished", rotation=90, transform=ax_n.get_xaxis_transform(),
                  fontsize=ps.TICK_PT, color=ps.MUTED, ha="center", va="center")
        what.append(f"{len(running)} training (at updates {', '.join(str(r['last_update']) for r in running)})")
    elif running:
        ax_n.text(x, 0.96, f"{len(fin)} of {len(SEEDS)}", transform=ax_n.get_xaxis_transform(),
                  fontsize=ps.TICK_PT, color=ps.MUTED, ha="center", va="top")
        what.append(f"{len(running)} training")
    if not marks and not running and not n_no_files:
        ax_n.text(x, 0.5, "absent", rotation=90, transform=ax_n.get_xaxis_transform(),
                  fontsize=ps.TICK_PT, color=ps.MUTED, ha="center", va="center")
        what.append("absent")
    return "; ".join(what)


def draw_block(fig, top, groups, records, no_files, ref, letters):
    """One block: the count row and the success row of a set of settings.

    Args:
        fig: the figure; top: mm, the top of the count axes; groups: one block of BLOCKS;
        records: from load_records; no_files: dict folder name -> number of run folders without
        run files; ref: control_reference; letters: (count row letter, success row letter).
    Returns:
        list of (key, what was drawn) per slot.
    """
    slots, centres, width = slot_positions(groups)
    cols = colour_of(groups)
    ax_n = ax_mm(fig, LEFT_MM, top, width, COUNT_H_MM)
    ax_s = ax_mm(fig, LEFT_MM, top + COUNT_H_MM + ROW_GAP_MM, width, SUCCESS_H_MM)
    if ref is not None:
        ax_n.axhspan(ref["active_min"], ref["active_max"], color=ps.FAINT, alpha=0.3, lw=0, zorder=1)
        ax_n.axhline(ref["active"], lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
        ax_s.axhline(ref["success"], lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax_s.axhline(MAZE_LEARNED_SUCCESS, ls=(0, (4, 3)), lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    out = []
    for key, label, x in slots:
        fin, running, stopped = runs_of(records, key)
        from_folder = no_files.get(key, 0)
        out.append((key, draw_slot(ax_n, ax_s, x, key, cols[key], fin, running, stopped, from_folder)))
    for title, x in centres:
        ax_n.text(x, 1.04, title, transform=ax_n.get_xaxis_transform(), fontsize=ps.LABEL_PT,
                  color=ps.INK, ha="center", va="bottom", linespacing=1.15)
    for ax in (ax_n, ax_s):
        ax.set_xlim(0, width)
        ax.set_xticks([x for _, _, x in slots])
        ps.ygrid(ax)
    ax_n.set_ylim(0, N)
    ax_n.set_yticks([0, N / 2, N])
    ps.count_ticks(ax_n.yaxis)
    ax_n.set_ylabel(f"active units\nat the end (of {ps.fmt_count(N)})")
    ax_n.tick_params(axis="x", length=0, labelbottom=False)
    ax_s.set_ylim(0, 1.03)
    ax_s.set_yticks([0, 0.5, 1.0])
    ax_s.set_ylabel("success\nat the end")
    ax_s.set_xticklabels([label for _, label, _ in slots], fontsize=ps.TICK_PT, linespacing=1.25)
    ax_s.tick_params(axis="x", length=0, pad=3)
    ax_s.text(0.3, MAZE_LEARNED_SUCCESS - 0.03, f"{MAZE_LEARNED_SUCCESS:g}", fontsize=ps.TICK_PT,
              color=ps.MUTED, ha="left", va="top")
    ps.panel_letter(ax_n, letters[0])
    ps.panel_letter(ax_s, letters[1])
    return out


def main():
    """Parse arguments, load the runs, draw both blocks, print every column, save the figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT, help="the arms: <arm>/s<seed>/ maze run folders")
    ap.add_argument("--control-root", default=CONTROL_ROOT, help="the uniform series' run folders")
    args = ap.parse_args()
    records, stopped_folders = load_records(args.root, args.control_root)
    ref = control_reference(records)
    no_files = {}
    for s in stopped_folders:
        no_files[s["folder_arm"]] = no_files.get(s["folder_arm"], 0) + 1
    ps.setup()
    fig = ps.figure(FIG_H_MM)
    rows = []
    for top, groups, letters in zip(TOP_MM, BLOCKS, (("a", "b"), ("c", "d"))):
        rows += draw_block(fig, top, groups, records, no_files, ref, letters)
    if ref is None:
        print("no learned control: no band and no reference line")
    else:
        print(f"controls' reference: {ref['n']} learned, active {ref['active']:.1f} "
              f"({ref['active_min']}-{ref['active_max']}), success {ref['success']:.3f}")
    for key, what in rows:
        print(f"  {key:>10}: {what}")
    check_overlaps(fig)
    ps.save(fig, "fig_ed_maze_settings")


if __name__ == "__main__":
    main()
