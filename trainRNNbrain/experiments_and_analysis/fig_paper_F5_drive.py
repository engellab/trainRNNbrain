#!/usr/bin/env python3
"""
Manuscript Figure 5 - TRAINING RULES THAT ACT ON INDIVIDUAL UNITS (per-unit rules) on 3BFF.

Every per-unit rule raises the active count above that of its control networks; the fold change and
the cost in held-out R^2 and in training stability differ by rule; the added units carry task signal;
and the combination test is inconclusive (dead-relu-paper#17). Drawn at its printed size,
165.1 x 140 mm (round 2, dead-relu-paper#55; style: paperstyle.py, paper docs/figure_style.md).

  key  one row above the panels: the marker and colour of the control networks and of each rule,
       and the ring that marks an R^2 computed from weights saved inside a loss burst or a stretch of
       non-finite loss (dead-relu-paper#18; rule fixed in notebook 2026-11-01_0745).
  (a)  N = 1,000: active units at every participation probe over 200,000 iterations, median over
       each rule's networks and over the control networks (traces collected by fig5_burst_check.py);
       each rule's marker every 25,000 iterations.
  (b)  N = 1,000 at 75,000 iterations: active units against held-out R^2, every network (the intact
       scores of the removal analysis, a3_remedy_ablation.py, 9 noise draws).
  (c)  top: fold change (a network's active count over the control networks' mean, same N, same
       measurement point) against N = 500-4,000; bottom: R^2 minus the control networks' mean.
       Measured at the floor crossing (read_remedies.py); rescaling and read-out dropout have no
       floor crossing at N >= 2,000, so their whole series is drawn at 75,000 iterations, open
       markers, dashed (dead-relu-paper#16).
  (d)  the rate penalty alone at N = 1,000: noisy training loss (top) and active units (bottom) of
       its three networks over training; grey band, the control networks at their floor crossing.
  (e)  left: R^2 after removing a network's k added units (k = its active count minus the control
       networks' mean; its k least participating active units) against removing k random active
       units; N = 1,000 filled, 4,000 open. Right: median R^2 after removing a random 10, 30 or 50%
       of all units, N = 1,000.
  (f)  the combination test, N = 1,000: rate penalty (lambda 0.03), noisy synapses (sigma_w 1) and
       rescaling, each present or absent (dot matrix), three networks per combination; active
       units, R^2 and participation ratio at the floor crossing (filled) and at 75,000 iterations
       (open).

DATA. Read outputs copied from Della into data/remedy_reads_31oct/ (gitignored):
  remedies_catalogue.json  ~/rm_read/     (read_remedies.py 2d32b3b, Della array 16553347 + 16555725)
  a3_N<N>.json             ~/a3_read/     (a3_remedy_ablation.py 30efe65, Della jobs 16553440-43)
  read_b3.json             ~/b3b4_read/   (read_b3_b4_4000.py, Della job 16553903)
  rm_diag.json             ~/rm_diag/     (rm_exclusion_diag.py cb36d82, Della job 16554963)
  fig5r2_read/             ~/fig5r2_read/ (fig5_burst_check.py e36f638, Della job 16592651):
                           burst_check.json, traces_N1000.json
and the rate-penalty runs' loss and participation traces (TrainLosses.json, ParticipationTrace.pkl,
config) under data/trained_RNNs/NBitFlipFlop_remedy_grid/. Panel (d) re-reads them with
common.read_run. Notebook: docs/project_trajectory/2026-10-31_1532, _1732, _1750, _1953 (the reads)
and 2026-11-01_0745 and its follow-up (the burst check and this figure).

Usage:  python fig_paper_F5_drive.py
Output: img/internal_figures/fig_paper_F5_drive.pdf (+ .svg), and a printout of every number the
        caption and Results 5 quote.
"""

import collections
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from common import DATA_DIR, load_run, read_run

READS = os.path.join(DATA_DIR, "..", "remedy_reads_31oct")
B2_DIR = os.path.join(DATA_DIR, "NBitFlipFlop_remedy_grid")
SIZES = (500, 1000, 2000, 4000)
FIG_H_MM = 141.0                 # paperstyle MAX_H_MM

# The per-unit rules in the main figure, in key order. Rate + sparsity appears only in the Extended
# Data and Supplementary Table 2.
MAIN = ("frm", "synnoise", "rescale", "duplicate", "dropout")
NAME = {"control": "control networks", "frm": "rate penalty", "synnoise": "noisy synapses",
        "rescale": "rescaling", "duplicate": "prune-and-duplicate", "dropout": "read-out dropout",
        "frm+rws": "rate + sparsity"}
COL = {"control": ps.BASELINE, "frm": ps.RATE_PENALTY, "synnoise": ps.SYNAPTIC_NOISE,
       "rescale": ps.RESCALE, "duplicate": ps.DUPLICATE, "dropout": ps.DROPOUT,
       "frm+rws": ps.RATE_SPARSITY}
MARK = {"control": "o", "frm": "o", "synnoise": "s", "rescale": "D", "duplicate": "^",
        "dropout": "v", "frm+rws": "P"}
# Rules whose size series is drawn at 75,000 iterations: no floor crossing at N >= 2,000
# (dead-relu-paper#16).
AT_75K = ("rescale", "dropout")

B3_CELLS = ("none", "rate", "noise", "rescale", "rate+noise", "rate+rescale", "noise+rescale",
            "rate+noise+rescale")
B3_PARTS = (("rate", "frm"), ("noise", "synnoise"), ("rescale", "rescale"))
RING_S = 34                    # scatter size of the ring that marks an R^2 inside a burst
# The three rate-penalty networks of panel d and Extended Data Fig. 8: the rate violet and two
# lighter steps of it, none darker (a darker violet is RATE_SPARSITY's; dead-relu-paper#84 item 4).
RATE_SHADES = (ps.RATE_PENALTY, "#9d7fd3", "#c3b0e8")
# Row labels of panel f, in words (dead-relu-paper#84 item 1); the rules are those of B3_PARTS.
B3_LABEL = {"none": "none", "rate": "rate", "noise": "noise", "rescale": "rescaling",
            "rate+noise": "rate + noise", "rate+rescale": "rate + rescaling",
            "noise+rescale": "noise + rescaling", "rate+noise+rescale": "all three"}


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------

def load_catalogue():
    """The per-unit-rule catalogue (read_remedies.py output), one dict per cell.

    Returns:
        list of dicts with keys cell, role, family, signature, N, n_runs, n_read, excluded, active,
        r2, r2_measured, t_star, active_d10, active_d20, active_75k, r2_75k, mult_*, dr2, dr2_75k.
        Arrays over runs: `active`, `r2`, `active_d10/d20` cover the runs with a floor crossing at
        1.05x; `active_75k` and `r2_75k` cover every run with weights at 75,000 iterations.
    """
    with open(os.path.join(READS, "remedies_catalogue.json")) as fh:
        return json.load(fh)


def size_series(cat):
    """Each family's size series: the one (family, signature) present at three or more sizes.

    The same selection a3_remedy_ablation.py uses, so panels (c) and (e) draw the same cells.

    Args:
        cat: load_catalogue() list.
    Returns:
        dict family -> {N: cell dict}; "control" maps to the role == "control" cells (Figure 1's).
    Raises:
        AssertionError if a family has more than one signature at three or more sizes, or a size
        appears twice.
    """
    groups = collections.defaultdict(list)
    for c in cat:
        if c["role"] in ("topup", "b2"):
            groups[(c["family"], c["signature"])].append(c)
    out = {"control": {c["N"]: c for c in cat if c["role"] == "control"}}
    for (fam, _), cells in groups.items():
        if len({c["N"] for c in cells}) < 3:
            continue
        assert fam not in out, f"two size series for {fam}"
        assert len({c["N"] for c in cells}) == len(cells), f"duplicate size in {fam}"
        out[fam] = {c["N"]: c for c in cells}
    return out


def load_a3():
    """The removal-analysis records (a3_remedy_ablation.py output) for every N.

    Returns:
        dict N -> list of per-network dicts (run_dir, family, cell, n_active, r2_full,
        random_10/30/50, surplus, r2_recruited, r2_random_active, r2_keep_ctrl, r2_no_inactive).
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(READS, "a3_N*.json"))):
        with open(f) as fh:
            recs = json.load(fh)
        out[recs[0]["N"]] = recs
    return out


def load_b3():
    """The combination-test read (read_b3_b4_4000.py output): runs, stats, n_found and verdicts."""
    with open(os.path.join(READS, "read_b3.json")) as fh:
        return json.load(fh)


def load_b2(N):
    """Every rate-penalty-alone network (lambda 0.1) at size N, re-read with common.read_run.

    Args:
        N: network size (500, 1000 or 2000 are on the Mac).
    Returns:
        list of read_run dicts (it_loss, loss, it_act, active_series, reason, floor, t_star,
        bursts, ...), sorted by folder name.
    """
    runs = []
    for d in sorted(glob.glob(os.path.join(B2_DIR, f"EqType=h_N={N}_arm=frm0.1", "*", ""))):
        rec = load_run(d)
        runs.append(read_run(d, run=rec))
    return runs


def load_marks():
    """Which R^2 values come from weights inside a loss burst or a non-finite stretch.

    Source: fig5_burst_check.py's burst_check.json (rule: notebook 2026-11-01_0745).

    Returns:
        dict with
          cat: {(cell, "floor" or "75k"): list of affected R^2 values} for the catalogue;
          a3: {run_dir: where} for the removal analysis (every score uses the 75,000 weights);
          b3: {(cell, name, "floor" or "fixed"): where} for the combination test;
          where is "calm", "burst" or "non-finite".
    """
    with open(os.path.join(READS, "fig5r2_read", "burst_check.json")) as fh:
        res = json.load(fh)
    cat, a3, b3 = collections.defaultdict(list), {}, {}
    for r in res:
        if "error" in r:
            continue
        for v in r["values"]:
            if r["source"] == "catalogue" and v["where"] != "calm":
                ro = "75k" if v["what"] == "75k" else "floor"
                cat[(r["cell"], ro)].append(v["r2"])
            elif r["source"] == "a3" and v["what"] == "intact":
                a3[r["run_dir"]] = v["where"]
            elif r["source"] == "b3":
                b3[(r["cell"], r["key"], v["what"])] = v["where"]
    return dict(cat=cat, a3=a3, b3=b3)


def load_traces():
    """Active-count traces of the N = 1,000 catalogue networks (fig5_burst_check.py).

    Returns:
        dict run_dir -> (it_act array, active array).
    """
    with open(os.path.join(READS, "fig5r2_read", "traces_N1000.json")) as fh:
        raw = json.load(fh)
    return {d: (np.asarray(v["it_act"], float), np.asarray(v["active"], float))
            for d, v in raw.items()}


# --------------------------------------------------------------------------------------------
# derived quantities and drawing helpers
# --------------------------------------------------------------------------------------------

def per_run(series, fam, N, readout, marks=None):
    """One family's networks at one size and one measurement point, as fold changes and R^2 changes.

    Args:
        series: size_series() dict; fam: family key; N: size;
        readout: "floor" (the floor crossing at 1.05x) or "75k";
        marks: load_marks() dict, or None (then nothing is flagged).
    Returns:
        dict with active (networks' counts), mult (fold change: count / control mean, same
        measurement point), r2, dr2 (R^2 minus the control networks' mean R^2; every network kept,
        non-finite R^2 included), burst (bool per R^2 value: from weights inside a loss burst or a
        non-finite stretch), n_read, n_runs. Empty arrays when the cell is missing.
    """
    c, ctrl = series.get(fam, {}).get(N), series["control"][N]
    if c is None:
        return dict(active=np.array([]), mult=np.array([]), r2=np.array([]), dr2=np.array([]),
                    burst=np.array([], bool), n_read=0, n_runs=0)
    ka, kr = ("active", "r2") if readout == "floor" else ("active_75k", "r2_75k")
    act, r2 = np.asarray(c[ka], float), np.asarray(c[kr], float)
    base_a, base_r = np.mean(ctrl[ka]), np.mean(ctrl[kr])
    bad = marks["cat"].get((c["cell"], readout), []) if marks else []
    burst = np.array([any(v == b for b in bad) for v in r2], bool)
    return dict(active=act, mult=act / base_a, r2=r2, dr2=r2 - base_r, burst=burst,
                n_read=len(c[ka]), n_runs=c["n_runs"])


def ring(ax, x, y):
    """Draw the ink ring that marks an R^2 from weights inside a loss burst around points (x, y).

    Args:
        ax: axes; x, y: array-likes of positions (data coordinates).
    Returns:
        None.
    """
    if len(np.atleast_1d(x)):
        ax.scatter(x, y, s=RING_S, marker="o", facecolor="none", edgecolor=ps.INK, lw=0.5,
                   zorder=4, clip_on=False)


def edge_arrow(ax, x, y, direction, col, filled, marker="o", s=10):
    """An off-scale value: its own marker at the axes' edge, with a short arrow pointing at it.

    The marker keeps the series' shape and the fill of its measurement point (filled, floor
    crossing; open, 75,000 iterations); the arrow comes from inside the plot and points out of it,
    so an off-axis value is not drawn as a triangle that could read as read-out dropout's marker
    (dead-relu-paper#99 review, item 9).

    Args:
        ax: axes; x, y: where to draw the marker (data coordinates, inside the limits);
        direction: "<" (value beyond the left edge) or "v" (beyond the bottom edge);
        col: colour; filled: bool (True at the floor crossing); marker: the series' marker;
        s: scatter size.
    Returns:
        None.
    """
    kw = ps.mark_floor(col, scatter=True) if filled else ps.mark_75k(col, scatter=True)
    ax.scatter([x], [y], s=s, marker=marker, zorder=4, clip_on=False, **kw)
    off = (9, 0) if direction == "<" else (0, 9)          # points: the arrow starts inside the plot
    ax.annotate("", xy=(x, y), xytext=off, textcoords="offset points", zorder=4,
                arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", lw=0.5,
                                color=col, shrinkA=0, shrinkB=2.2))


def style_axes(ax):
    """Spines left and bottom only; tick length and width from the style sheet.

    Args:
        ax: axes.
    Returns:
        None.
    """
    ps.despine(ax)
    ax.tick_params(length=ps.TICK_LEN, width=ps.AXIS_LW)


# --------------------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------------------

def panel_key(ax):
    """The key above the panels: a row of rules (colour, shape) and a row of marker meanings.

    Fill marks the measurement point and nothing else (dead-relu-paper#83): filled, the floor
    crossing; open, 75,000 iterations. The ring marks an R^2 from weights inside a loss burst, the
    arrow a value beyond the axis, drawn at the edge.

    Args:
        ax: a blank axes spanning the figure width.
    Returns:
        None.
    """
    ps.blank(ax)
    h = [Line2D([], [], marker=MARK[f], color=COL[f], mec=COL[f], ls="-", lw=ps.DATA_LW,
                ms=3.4, label=NAME[f]) for f in ("control",) + MAIN]
    leg = ax.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, fontsize=ps.TICK_PT,
                    handlelength=1.5, columnspacing=1.2, handletextpad=0.35, borderaxespad=0)
    ax.add_artist(leg)
    h2 = [Line2D([], [], marker="o", ls="none", ms=3.4, label=ps.MARK_FLOOR_LABEL,
                 **ps.mark_floor(ps.INK)),
          Line2D([], [], marker="o", ls="none", ms=3.4, label=ps.MARK_75K_LABEL,
                 **ps.mark_75k(ps.INK)),
          Line2D([], [], marker="o", mfc="none", mec=ps.INK, mew=0.5, ls="none", ms=5.2,
                 label="$R^2$ from weights inside a loss burst"),
          Line2D([], [], marker=r"$\leftarrow$", color=ps.INK, ls="none", ms=5,
                 label="value beyond the axis (at the edge)")]
    ax.legend(handles=h2, loc="lower center", ncol=len(h2), frameon=False, fontsize=ps.TICK_PT,
              handlelength=1.0, columnspacing=1.6, handletextpad=0.35, borderaxespad=0)


def panel_traces(ax, series, traces):
    """Panel (a): active units over training at N = 1,000, median over each group's networks.

    Args:
        ax: axes; series: size_series(); traces: load_traces().
    Returns:
        dict family -> (n networks, median active count at the probe nearest 75,000, at the last
        probe), for the printout.
    """
    out = {}
    for fam in ("control",) + MAIN:
        cell = series[fam][1000]["cell"]
        runs = [v for d, v in traces.items() if f"/{cell}/" in d]
        assert len(runs) == series[fam][1000]["n_runs"], f"traces for {len(runs)} runs of {cell}"
        it = runs[0][0]
        assert all(np.array_equal(r[0], it) for r in runs), f"probe iterations differ in {cell}"
        med = np.median(np.vstack([r[1] for r in runs]), axis=0)
        ax.plot(it, med, color=COL[fam], lw=0.6 if fam == "frm" else ps.DATA_LW,
                zorder=1 if fam == "frm" else 2)
        # one marker per trace, at 75,000 iterations (open: the 75,000-iteration read-out, as in b);
        # its shape, not only its lightness, tells the two ochre rules apart
        j = int(np.argmin(np.abs(it - 75000)))
        kw = ps.mark_75k(COL[fam])
        kw["mfc"] = "white"                  # still open; white so the trace does not run through it
        ax.plot([it[j]], [med[j]], ls="none", marker=MARK[fam], ms=4.0, zorder=6, **kw)
        out[fam] = (len(runs), float(med[j]), float(med[-1]))
    ax.axvline(75000, color=ps.FAINT, lw=ps.THIN_LW, zorder=0)
    ax.set_xlim(0, 200000)
    ax.set_xticks([0, 50000, 100000, 150000, 200000])
    ps.count_ticks(ax.xaxis)
    ax.set_ylim(0, 1030)
    ax.set_yticks([0, 250, 500, 750, 1000])
    ps.count_ticks(ax.yaxis)
    ax.set_xlabel(ps.LABELS["iteration"])
    ax.set_ylabel(ps.of_n("active units", 1000))
    ax.set_title("$N$ = 1,000; median over networks", fontsize=ps.LABEL_PT, pad=3)
    style_axes(ax)
    return out


def panel_one_size(ax, a3, marks):
    """Panel (b): N = 1,000 at 75,000 iterations, active units against held-out R^2, every network.

    Args:
        ax: axes; a3: load_a3() dict; marks: load_marks().
    Returns:
        dict family -> (active list, r2 list, burst list) as drawn, for the printout.
    """
    drawn = collections.defaultdict(lambda: ([], [], []))
    for r in a3[1000]:
        fam = r["family"]
        if fam not in MAIN and fam != "control":
            continue
        drawn[fam][0].append(r["n_active"])
        drawn[fam][1].append(r["r2_full"])
        drawn[fam][2].append(marks["a3"][r["run_dir"]] != "calm")
    for fam in ("control",) + MAIN:
        a, r2, b = (np.asarray(v) for v in drawn[fam])
        b = b.astype(bool)
        # every value here is read at 75,000 iterations: open markers (dead-relu-paper#83)
        ax.scatter(a, r2, s=14, marker=MARK[fam], zorder=3, **ps.mark_75k(COL[fam], scatter=True))
        ring(ax, a[b], r2[b])
    ax.set_xlim(0, 1040)          # rate-penalty networks reach 995: whole markers, not halves
    ax.set_xticks([0, 250, 500, 750, 1000])
    ps.count_ticks(ax.xaxis)
    ax.set_ylim(0.918, 0.956)
    ax.set_yticks([0.93, 0.94, 0.95])
    ax.set_xlabel(ps.of_n("active units", 1000))
    ax.set_ylabel(ps.LABELS["r2"])
    ax.set_title("$N$ = 1,000; 75,000 iterations", fontsize=ps.LABEL_PT, pad=3)
    ax.text(1035, 0.9195, "rate penalty: count swings during training (d)", fontsize=ps.TICK_PT,
            color=ps.INK, ha="right", va="bottom")
    style_axes(ax)
    return drawn


def panel_sizes(ax_m, ax_r, series, marks):
    """Panel (c): fold change (top) and R^2 change (bottom) against N, per rule.

    Floor crossings filled with solid lines; the AT_75K families at 75,000 iterations, open markers,
    dashed lines. A size with no floor crossing is left out of that family's line. An R^2 change
    below the axis is drawn as an arrowhead at the bottom edge.

    Args:
        ax_m, ax_r: the two axes (shared x); series: size_series() dict; marks: load_marks().
    Returns:
        dict family -> {N: per_run dict} as drawn, for the printout.
    """
    offs = dict(zip(MAIN, np.linspace(-0.06, 0.06, len(MAIN))))
    lo_r, hi_r = -0.064, 0.008
    y_edge = lo_r + 0.003
    drawn = {}
    for fam in MAIN:
        ro = "75k" if fam in AT_75K else "floor"
        drawn[fam] = {}
        xs, mm, rr = [], [], []
        for N in SIZES:
            d = per_run(series, fam, N, ro, marks)
            drawn[fam][N] = d
            if not len(d["mult"]):
                continue
            x = N * 10 ** offs[fam]
            kw = ps.mark_75k(COL[fam], scatter=True) if ro == "75k" else \
                ps.mark_floor(COL[fam], scatter=True)
            ax_m.scatter(np.full(len(d["mult"]), x), d["mult"], s=7, marker=MARK[fam], zorder=3, **kw)
            ok = d["dr2"] > y_edge
            ax_r.scatter(np.full(ok.sum(), x), d["dr2"][ok], s=7, marker=MARK[fam], zorder=3, **kw)
            for _ in range(int((~ok).sum())):
                edge_arrow(ax_r, x, y_edge, "v", COL[fam], ro != "75k", MARK[fam], s=7)
            yb = np.where(ok, d["dr2"], y_edge)[d["burst"]]
            ring(ax_r, np.full(len(yb), x), yb)
            xs.append(x)
            mm.append(d["mult"].mean())
            rr.append(np.median(d["dr2"]))
        ls = "--" if fam in AT_75K else "-"
        ax_m.plot(xs, mm, ls=ls, color=COL[fam], lw=ps.DATA_LW, zorder=2)
        ax_r.plot(xs, rr, ls=ls, color=COL[fam], lw=ps.DATA_LW, zorder=2)
    for ax in (ax_m, ax_r):
        ax.set_xscale("log")
        ax.set_xticks(SIZES)
        ax.set_xticklabels([ps.fmt_count(n) for n in SIZES])
        ax.minorticks_off()
        ax.set_xlim(400, 5000)
        style_axes(ax)
    ax_m.axhline(1, color=ps.FAINT, lw=ps.THIN_LW, zorder=1)
    ax_r.axhline(0, color=ps.FAINT, lw=ps.THIN_LW, zorder=1)
    ax_m.set_ylim(0.6, 7.0)
    ax_m.set_yticks([1, 3, 5, 7])
    ax_m.set_ylabel("fold change")
    ax_r.set_ylim(lo_r, hi_r)
    ax_r.set_yticks([-0.06, -0.03, 0])
    ax_r.set_ylabel("$R^2$ change")
    ax_r.set_xlabel(ps.LABELS["size"])
    plt.setp(ax_m.get_xticklabels(), visible=False)
    ax_m.text(430, 3.6, "rate penalty: no floor\ncrossing at $N$ $\\leq$ 1,000", fontsize=ps.TICK_PT,
              color=ps.INK, ha="left", va="center")
    h = [Line2D([], [], ls="-", marker="o", ms=3, lw=0.9, color=ps.MUTED,
                label=ps.MARK_FLOOR_LABEL, **ps.mark_floor(ps.MUTED)),
         Line2D([], [], ls="--", marker="o", ms=3, lw=0.9, color=ps.MUTED,
                label=ps.MARK_75K_LABEL, **ps.mark_75k(ps.MUTED))]
    ax_m.legend(handles=h, loc="upper left", fontsize=ps.TICK_PT, handlelength=2.2,
                borderaxespad=0.1, frameon=False)
    return drawn


def panel_rate_alone(ax_l, ax_a, runs, ctrl_floor):
    """Panel (d): the rate penalty alone at N = 1,000 - noisy loss (top) and active units (bottom).

    Args:
        ax_l, ax_a: the two axes (shared x); runs: load_b2(1000);
        ctrl_floor: the N = 1,000 control networks' active counts at the floor crossing (catalogue).
    Returns:
        list of dicts per network (reason, burst count, active range after 3,000 iterations, raw
        loss maximum), for the printout.
    """
    info = []
    for r, col in zip(runs, RATE_SHADES):
        it, L = np.asarray(r["it_loss"]), np.asarray(r["loss"])
        # vector-friendly: the per-bin maximum (the bursts) and median (the trend), 250-iteration bins
        nb = len(L) // 250
        Lb = L[:nb * 250].reshape(nb, 250)
        xb = it[:nb * 250].reshape(nb, 250)[:, 0]
        ax_l.plot(xb, np.clip(Lb.max(1), None, 1e3), color=col, lw=0.3, alpha=0.6)
        ax_l.plot(xb, np.median(Lb, 1), color=col, lw=0.8)
        ia, act = np.asarray(r["it_act"]), np.asarray(r["active_series"])
        ax_a.plot(ia, act, color=col, lw=0.4)
        late = act[ia >= 3000]
        info.append(dict(name=r["name"], reason=r["reason"], n_bursts=len(r["bursts"]),
                         lo=int(late.min()), hi=int(late.max()), loss_max=float(np.nanmax(L))))
    ax_a.axhspan(min(ctrl_floor), max(ctrl_floor), color=ps.BASELINE, alpha=0.35, lw=0, zorder=0)
    ax_a.text(197000, min(ctrl_floor) - 25, "control networks at their floor crossing",
              fontsize=ps.TICK_PT, color=ps.INK, ha="right", va="top")
    ax_l.set_yscale("log")
    ax_l.set_ylim(0.03, 1e4)                 # headroom: the maxima are clipped at 1,000
    ax_l.set_yticks([0.1, 10, 1000])
    ax_l.set_yticklabels(["0.1", "10", "1,000"])
    ax_l.minorticks_off()
    ax_l.set_ylabel("noisy loss")
    ax_a.set_ylim(0, 1030)
    ax_a.set_yticks([0, 500, 1000])
    ps.count_ticks(ax_a.yaxis)
    ax_a.set_ylabel("active units")
    ax_a.set_xlabel(ps.LABELS["iteration"])
    for ax in (ax_l, ax_a):
        ax.set_xlim(0, 200000)
        ax.set_xticks([0, 50000, 100000, 150000, 200000])
        ps.count_ticks(ax.xaxis)
        style_axes(ax)
    plt.setp(ax_l.get_xticklabels(), visible=False)
    ax_l.set_title("rate penalty alone, $N$ = 1,000, three networks", fontsize=ps.LABEL_PT, pad=3)
    return info


def panel_removal(axs, a3, marks):
    """Panel (e): R^2 after removing each network's k added units against k random active units.

    One sub-panel per size (N = 1,000 and 4,000), every network at 75,000 iterations, so every
    marker is open (dead-relu-paper#83); a value below the axis is an arrowhead at the edge.

    Args:
        axs: two axes (N = 1,000, N = 4,000), the second sharing the first's y-axis;
        a3: load_a3() dict; marks: load_marks().
    Returns:
        dict (N, family) -> list of (median R^2 after k random removals, R^2 after removing the
        added units, where) per network, for the printout.
    """
    lo, hi = -0.05, 0.8
    out = {}
    for ax, N in zip(axs, (1000, 4000)):
        ax.plot([lo, 0.45], [lo, 0.45], color=ps.FAINT, lw=ps.THIN_LW, zorder=1)
        for r in a3[N]:
            fam = r["family"]
            if fam not in MAIN:
                continue
            x, y = float(np.median(r["r2_random_active"])), float(r["r2_recruited"])
            if x < lo:
                xd, yd = lo + 0.01, y
                edge_arrow(ax, xd, yd, "<", COL[fam], False, MARK[fam], s=11)
            elif y < lo:
                xd, yd = x, lo + 0.012
                edge_arrow(ax, xd, yd, "v", COL[fam], False, MARK[fam], s=11)
            else:
                xd, yd = x, y
                ax.scatter([x], [y], s=11, marker=MARK[fam], zorder=3,
                           **ps.mark_75k(COL[fam], scatter=True))
            where = marks["a3"][r["run_dir"]]
            if where != "calm":
                ring(ax, [xd], [yd])
            out.setdefault((N, fam), []).append((x, y, where))
        ax.set_xlim(lo - 0.012, 0.45)
        ax.set_ylim(lo - 0.015, hi)
        ax.set_xticks([0, 0.2, 0.4])
        ax.set_yticks([0, 0.4, 0.8])
        ax.set_title(f"$N$ = {ps.fmt_count(N)}", fontsize=ps.LABEL_PT, pad=3)
        style_axes(ax)
    axs[0].set_ylabel("$R^2$, $k$ added\nunits removed")
    axs[1].tick_params(labelleft=False)
    axs[0].text(0.44, 0.47, "equal\ncost", fontsize=ps.TICK_PT, color=ps.INK, ha="right",
                va="bottom")
    # one x-label centred under both sub-panels
    x0, x1 = axs[0].get_position().x0, axs[1].get_position().x1
    axs[0].figure.text((x0 + x1) / 2, axs[0].get_position().y0 - 0.05,
                       "$R^2$, $k$ random active units removed", fontsize=ps.LABEL_PT,
                       color=ps.INK, ha="center", va="top")
    return out


def panel_combination(ax_m, axs, b3, marks):
    """Panel (f): the combination test, N = 1,000, per network, at the floor crossing and at 75,000.

    Args:
        ax_m: axes for the row labels (which rules each combination has, in words); axs: three
        axes sharing y (active units, held-out R^2, participation ratio); b3: load_b3();
        marks: load_marks().
    Returns:
        list of (cell, measurement point, active, r2, pr, where) rows drawn, for the printout.
    """
    rows = []
    ypos = {c: len(B3_CELLS) - 1 - i for i, c in enumerate(B3_CELLS)}
    lims = [(200, 1000), (0.83, 0.96), (0, 14)]
    ps.blank(ax_m)
    ax_m.set_xlim(0, 1)
    for cell in B3_CELLS:
        ax_m.text(1.0, ypos[cell], B3_LABEL[cell], fontsize=ps.TICK_PT, color=ps.INK,
                  ha="right", va="center")
    ax_m.text(1.0, len(B3_CELLS) - 0.35, "rules present", fontsize=ps.TICK_PT, color=ps.MUTED,
              ha="right", va="bottom")
    for cell in B3_CELLS:
        for ro, dy in (("floor", 0.15), ("fixed", -0.15)):
            for r in b3["runs"][cell]:
                rec = r[ro]
                if rec is None:
                    continue
                where = marks["b3"].get((cell, r["name"], ro), "calm")
                vals = [rec["active"], rec["r2"], rec["pr"]]
                rows.append((cell, ro, *vals, where))
                kw = ps.mark_floor(ps.INK, scatter=True) if ro == "floor" else \
                    ps.mark_75k(ps.INK, scatter=True)
                for k, (ax, v, (lo, hi)) in enumerate(zip(axs, vals, lims)):
                    v = np.nan if v is None else float(v)
                    y = ypos[cell] + dy
                    if not np.isfinite(v) or v < lo:
                        xr = lo + 0.015 * (hi - lo)
                        edge_arrow(ax, xr, y, "<", ps.INK, ro == "floor", "o", s=8)
                    else:
                        xr = v
                        ax.scatter([v], [y], s=8, marker="o", zorder=3, **kw)
                    if k == 1 and where != "calm":
                        ring(ax, [xr], [y])
    for ax, (lo, hi), lab in zip(axs, lims, ("active units", ps.LABELS["r2"],
                                             "participation\nratio")):
        ax.set_xlim(lo, hi)
        ax.set_xlabel(lab)
        style_axes(ax)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    for ax in [ax_m] + list(axs):
        ax.set_ylim(-0.6, len(B3_CELLS) - 0.4)
    for ax in axs:
        for y in range(len(B3_CELLS)):
            ax.axhline(y, color=ps.GRID, lw=0.4, zorder=0)
    # tick labels kept inside each column so neighbouring columns' labels stay >= 3 mm apart
    axs[0].set_xticks([300, 600, 900])
    axs[1].set_xticks([0.85, 0.95])
    axs[2].set_xticks([5, 10])
    return rows


# --------------------------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------------------------

def build():
    """Draw Figure 5 and print every number its caption and Results 5 quote.

    Returns:
        the path of the PDF written by paperstyle.save.
    """
    ps.setup()
    cat = load_catalogue()
    series = size_series(cat)
    a3, b3 = load_a3(), load_b3()
    marks, traces = load_marks(), load_traces()
    b2 = load_b2(1000)

    fig = ps.figure(h_mm=FIG_H_MM)

    def y(mm):
        """Figure fraction of a height given in mm from the figure's top edge."""
        return 1.0 - mm / FIG_H_MM

    L, W = 0.075, 0.475                    # left column: left and right edges (figure fractions)
    M, R = 0.585, 0.965                    # right column

    ax_key = fig.add_axes([0.0, y(8.5), 1.0, y(0.8) - y(8.5)])
    panel_key(ax_key)

    # row 1 (a, b): axes 15-41 mm from the top
    ax_a = fig.add_axes([L, y(41), W - L, y(15) - y(41)])
    ax_b = fig.add_axes([M, y(41), R - M, y(15) - y(41)])
    info_a = panel_traces(ax_a, series, traces)
    drawn_b = panel_one_size(ax_b, a3, marks)

    # row 2 (c, d): 57.5-91 mm
    gc = fig.add_gridspec(2, 1, left=L, right=W, top=y(57.5), bottom=y(91), hspace=0.12,
                          height_ratios=[1.45, 1])
    ax_cm, ax_cr = fig.add_subplot(gc[0]), fig.add_subplot(gc[1])
    drawn_c = panel_sizes(ax_cm, ax_cr, series, marks)

    gd = fig.add_gridspec(2, 1, left=M, right=R, top=y(57.5), bottom=y(91), hspace=0.12,
                          height_ratios=[1, 1.45])
    ax_dl, ax_da = fig.add_subplot(gd[0]), fig.add_subplot(gd[1])
    info_d = panel_rate_alone(ax_dl, ax_da, b2, series["control"][1000]["active"])

    # row 3 (e, f): 107-127.5 mm; e takes less width than its column so that f's labels fit
    ge = fig.add_gridspec(1, 2, left=L, right=0.385, top=y(107), bottom=y(127.5), wspace=0.12)
    ax_e1 = fig.add_subplot(ge[0])
    ax_e2 = fig.add_subplot(ge[1], sharey=ax_e1)
    drawn_e = panel_removal([ax_e1, ax_e2], a3, marks)

    gf = fig.add_gridspec(1, 4, left=0.445, right=R, top=y(107), bottom=y(127.5), wspace=0.22,
                          width_ratios=[0.62, 1, 1, 1])
    ax_fm = fig.add_subplot(gf[0])
    axs_f = [fig.add_subplot(gf[i]) for i in (1, 2, 3)]
    rows_f = panel_combination(ax_fm, axs_f, b3, marks)

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_cm, "c"), (ax_dl, "d"), (ax_e1, "e")):
        ps.panel_letter(ax, letter, x_mm=10.5)
    ps.panel_letter(ax_fm, "f", x_mm=6.0)

    report(series, info_a, drawn_b, drawn_c, info_d, drawn_e, rows_f, b3)
    return ps.save(fig, "fig_paper_F5_drive")


def report(series, info_a, drawn_b, drawn_c, info_d, drawn_e, rows_f, b3):
    """Print the numbers the caption and Results 5 quote, from exactly what was drawn.

    Args:
        series: size_series(); info_a, drawn_b, drawn_c, info_d, drawn_e, rows_f: the panel
        functions' returns; b3: load_b3().
    Returns:
        None.
    """
    print("\n(a) N = 1,000 median active count over networks: (n, at 75k, at the end)")
    for fam, v in info_a.items():
        print(f"  {fam:9s} n={v[0]} {v[1]:.0f} {v[2]:.0f}")
    print("\n(b) N = 1,000, 75k (removal analysis intact scores):")
    ctrl_mean = np.mean(drawn_b["control"][1])
    for fam, (a, r2, b) in drawn_b.items():
        print(f"  {fam:9s} n={len(a)} active {min(a)}-{max(a)} (mean {np.mean(a):.0f}, fold "
              f"{np.mean(a) / np.mean(drawn_b['control'][0]):.2f})  R2 {min(r2):.4f}-{max(r2):.4f}"
              f"  R2 - control mean {min(r2) - ctrl_mean:+.4f} to {max(r2) - ctrl_mean:+.4f}"
              f"  in a burst {sum(b)}")
    print("\n(c) fold change and R2 change (network - control mean), by N:")
    for N in SIZES:
        c = series["control"][N]
        print(f"  control N={N}: floor active {c['active']} R2 {np.round(c['r2'], 3).tolist()};"
              f" 75k active {c['active_75k']}")
    for fam, byN in drawn_c.items():
        for N, d in byN.items():
            if not len(d["mult"]):
                print(f"  {fam:9s} N={N}: no floor crossing ({d['n_runs']} runs)")
                continue
            print(f"  {fam:9s} N={N}: {d['n_read']}/{series[fam][N]['n_runs']} runs, fold "
                  f"{d['mult'].mean():.2f} ({d['mult'].min():.2f}-{d['mult'].max():.2f}), dR2 median "
                  f"{np.median(d['dr2']):+.3f} (range {d['dr2'].min():+.3f} to {d['dr2'].max():+.3f})"
                  f"; in a burst {int(d['burst'].sum())}")
    print("\n(d) rate penalty alone, N = 1,000:")
    for i in info_d:
        print(f"  {i['name'][:40]}: {i['reason']}; {i['n_bursts']} bursts; active after 3k "
              f"{i['lo']}-{i['hi']}; raw loss max {i['loss_max']:.3g}")
    print("\n(e) removal, (median random-k, added) per network:")
    for k, v in drawn_e.items():
        rec = [round(y, 2) for _, y, _ in v]
        rnd = [round(x, 2) for x, _, _ in v]
        nb = sum(w != "calm" for _, _, w in v)
        print(f"  N={k[0]} {k[1]:9s} added {rec}  random-k {rnd}  medians {np.median(rec):.2f} vs "
              f"{np.median(rnd):.2f}; random lower in {sum(x < y for x, y, _ in v)}/{len(v)};"
              f" in a burst {nb}")
    print("\n(f) combination test outcomes:", b3["verdict"]["floor"]["verdict"], "(floor);",
          b3["verdict"]["fixed"]["verdict"], "(75k)")
    for cell in B3_CELLS:
        for ro in ("floor", "fixed"):
            rr = [r for r in rows_f if r[0] == cell and r[1] == ro]
            print(f"  {cell:20s} {ro:5s} n={len(rr)} active {[r[2] for r in rr]} "
                  f"R2 {[None if r[3] is None else round(r[3], 3) for r in rr]} "
                  f"PR {[round(r[4], 1) for r in rr]} where {[r[5] for r in rr]}")


if __name__ == "__main__":
    build()
