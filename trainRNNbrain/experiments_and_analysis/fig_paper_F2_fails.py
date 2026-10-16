#!/usr/bin/env python3
"""Manuscript Figure 2 - WHAT FAILS. Every standard training knob, and the tricks that revive silent
units, on the standard network (ReLU, bias 0, N = 1000). Figure 1 is the phenomenon; Figure 3 what
worked. The old `fig_paper_F2.py` is kept because other code imports it.

WHY TWO VIEWS PER KNOB. Each run is read where its noisy training loss first reaches 1.05x its own
fitted floor (`common.noisy_floor`), the project's rule. Its blind spot: a knob that speeds
convergence is read EARLIER, while more units are still active. CDDM input scale 2 reads ~700
active at its floor (crossing ~4,500) against the control's ~430 (crossing ~25,000), yet has FEWER
active units at every matched iteration from 25,000 on. So every ladder is drawn as a PAIR of panels
sharing one active-units axis:
  left   held-out R^2 at each run's floor against its active units there (the trade-off scatter:
         dose ladders coloured in dose order, categorical ones by marker shape, the standard network
         ringed in black; the axes are transposed from `fig_knobs_at_floor --tradeoff` so that the
         active-units axis is shared with the traces);
  right  active units against training iteration for every run, each run's floor crossing marked
         with the same symbol, so a point on the left and its crossing on the right sit at the
         same height. A run with no usable crossing is dashed. The dotted vertical line is the
         ladder's common budget.
Panels a-i are the nine ladders of `fig_knobs_at_floor.LADDERS`; j and k the revival screens
(placeholders until their top-ups land; picked up automatically on the next --collect); l the
bottom line: every rung's mean at its own floor (open) and at its ladder's common budget (filled),
so a knob that only reached its floor sooner shows up as a filled symbol LEFT of its reference.

BUDGETS DIFFER between ladders and are never compared across them: weight decay, recurrent noise and
activation 200k; metabolic, spectral radius, W_rec penalty, rate/sparsity and the CDDM revival
screen 75k (the metabolic ladder's 200k control is read at 75k via `end_at`); CDDM input scale 50k
(its 100k controls read at 50k); flip-flop input scale 150k; flip-flop revival 200k.

Floor counts and R^2 come from `heldout_r2.read_rung` unchanged (R^2 measured on kept
snapshots where they exist - only the CDDM input-scale ladder - otherwise estimated from the run's
end; on CDDM an R^2 difference below ~0.01 is scoring noise). Traces from
`common.runs_with_loss`.

Usage (repository root):
    python trainRNNbrain/experiments_and_analysis/fig_paper_F2_fails.py --collect   # Della CPU node:
                       read every run (R^2 from data/heldout_r2_cache.json), write the cache, draw
    python trainRNNbrain/experiments_and_analysis/fig_paper_F2_fails.py --probe     # per-run table
    python trainRNNbrain/experiments_and_analysis/fig_paper_F2_fails.py             # draw from the
                       cache (data/fig_paper_F2_fails_cache.pkl; copy it from Della to draw locally)
Output: img/internal_figures/fig_paper_F2_fails.{pdf,svg} (vector only, paperstyle.save), with a
layout check printed (legends covering data, overlapping text) and the bottom-line numbers.
"""

import argparse
import os
import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fig_knobs_at_floor as K
import paperstyle as ps
from common import DATA_DIR, logbin, noisy_floor, runs_with_loss
from heldout_r2 import load_cache, read_rung, save_cache

CACHE = os.path.normpath(os.path.join(DATA_DIR, "..", "fig_paper_F2_fails_cache.pkl"))

# The revival screens, continued to a budget where they can reach their floor (trajectory
# 2026-10-13: top-ups A and B). Each cell is read only when its folder holds finished runs; until
# then the panel is a placeholder. The CDDM control is the screen's own in-sweep control; the
# flip-flop screen has none, and its config differs from the claim-1 flip-flop control
# (`ff_revive_c200000`, arm none) only in the prune keys, so that control is its reference.
REVIVAL = [
    dict(stem="revival_cddm", knob="reviving inactive units", ref=0, categorical=True,
         task="CDDM, $N$ = 1000, 75,000 iterations",
         rungs=[("none", f"{DATA_DIR}/CDDM_screen3_c75000/EqType=h_N=1000_pen=none_arm=none"),
                ("input-weight\ncap 0.3", f"{DATA_DIR}/CDDM_screen3_c75000/EqType=h_N=1000_pen=none_arm=cap"),
                ("re-initialise\ninactive units", f"{DATA_DIR}/CDDM_screen3_c75000/EqType=h_N=1000_pen=none_arm=prune")]),
    dict(stem="revival_ff", knob="reviving inactive units", ref=0, categorical=True,
         task="3-bit flip-flop, $N$ = 1000, 200,000 iterations",
         rungs=[("none", f"{DATA_DIR}/NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=1000_pen=none_arm=none"),
                ("zero-out\ninactive units", f"{DATA_DIR}/NBitFlipFlop_replace_variants_c200000/EqType=h_k=3_N=1000_mode=zero_out_rate=0.025_mat=1000"),
                ("random\nredraw", f"{DATA_DIR}/NBitFlipFlop_replace_variants_c200000/EqType=h_k=3_N=1000_mode=random_rate=0.025_mat=1000")]),
]


def collect_spec(spec, r2cache):
    """Read one ladder: the floor read-out of every rung, and every run's participation trace.

    The floor counts and R^2 come from `heldout_r2.read_rung`, unchanged. The traces come
    from `common.runs_with_loss`, and each run's crossing is recomputed with the same
    `common.noisy_floor` call so it can be marked on its own trace; the two must agree, and an
    AssertionError says so if they do not.

    Args:
        spec: one entry of `fig_knobs_at_floor.LADDERS` or REVIVAL; r2cache: open heldout_r2 cache.
    Returns:
        dict(spec=spec, rungs=[dict(label, path, read, runs)]), with `read` read_rung's dict and
        `runs` one dict per run: name, it_act (iterations of the participation probes), active
        (active units at each probe), t_star (floor crossing, nan when the run has no usable
        crossing) and reason (why not, or None).
    """
    rungs = []
    for lab, path in spec["rungs"]:
        if not os.path.isdir(path):
            rungs.append(dict(label=lab, path=path, read=None, runs=[]))
            continue
        read = read_rung(path, spec.get("end_at"), r2cache)
        runs = []
        for r in runs_with_loss(path):
            it_a = np.asarray(r["it_act"], float)
            f = noisy_floor(r["it_loss"], r["loss"], task=r["task"])
            t, why = f["t_star"], f["reason"]
            if why is None and not it_a[0] <= t <= it_a[-1]:
                t, why = float("nan"), "participation trace does not cover the crossing"
            if why is not None:
                t = float("nan")
            runs.append(dict(name=r["name"], it_act=it_a, active=np.asarray(r["active"], float),
                             t_star=float(t), reason=why))
        mine = sorted(r["t_star"] for r in runs if np.isfinite(r["t_star"]))
        assert np.allclose(mine, sorted(read["cross"])), f"crossings disagree in {path}"
        rungs.append(dict(label=lab, path=path, read=read, runs=runs))
    return dict(spec=spec, rungs=rungs)


def collect():
    """Read every ladder and revival screen and pickle the result to CACHE. Returns CACHE."""
    r2cache = load_cache()
    out = dict(ladders=[collect_spec(s, r2cache) for s in K.LADDERS],
               revival=[collect_spec(s, r2cache) for s in REVIVAL])
    save_cache(r2cache)
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh)
    print(f"wrote {CACHE}")
    return CACHE


def active_at(run, t):
    """A run's active-unit count at the probe nearest iteration t, or nan if t is outside its trace.

    Args:
        run: one run dict from collect_spec; t: an absolute iteration.
    Returns:
        float.
    """
    it = run["it_act"]
    if not it[0] - 500 <= t <= it[-1] + 500:
        return float("nan")
    return float(run["active"][int(np.argmin(np.abs(it - t)))])


def probe(data):
    """Print, per run, its trace span, crossing, and counts at the floor and at fixed iterations."""
    for group in ("ladders", "revival"):
        for lad in data[group]:
            spec = lad["spec"]
            print(f"\n=== {spec['knob']}  ({spec['task']})  end_at={spec.get('end_at')}")
            for rg in lad["rungs"]:
                lab = rg["label"].replace("\n", " ")
                if rg["read"] is None:
                    print(f"  {lab:<26} no folder")
                    continue
                print(f"  {lab:<26} n={rg['read']['n']}  floor {rg['read']['floor'].astype(int).tolist()}"
                      f"  R2 {np.round(rg['read']['r2'], 3).tolist()}"
                      f"  measured {rg['read']['measured'].tolist()}")
                for r in rg["runs"]:
                    at = " ".join(f"{active_at(r, t):5.0f}" for t in (5e3, 12.5e3, 25e3, 50e3, 75e3, 150e3, 200e3))
                    print(f"     {r['name'][:9]:<10} trace {r['it_act'][0]:>7.0f}-{r['it_act'][-1]:<7.0f}"
                          f" ({len(r['it_act'])} probes)  t*={r['t_star']:>7.0f}"
                          f"  @5k,12.5k,25k,50k,75k,150k,200k: {at}"
                          f"  {r['reason'] or ''}")


# ------------------------------------------------------------------------------------------------
# drawing
# ------------------------------------------------------------------------------------------------

# Panel order, by `stem`: CDDM dose ladders first, then the two categorical CDDM ladders and the
# input scale on both tasks, so the two input-scale ladders sit side by side.
ORDER = ["slide_k_weightdecay", "slide_k_recnoise", "slide_k_metabolic",
         "slide_k_recgain", "slide_k_rwm", "slide_k_frm_rws",
         "slide_k_activation", "slide_k_inputscale_cddm", "slide_k_inputscale"]
N_UNITS = 1000
Y_MAX = 1040                # room for markers at exactly 1000 (the rate penalty)
X_MIN = 2000                # first iteration drawn on the log axis of the traces. Below ~2,500 a log
                            # bin holds fewer than three probes (one every 100 iterations), and no
                            # run reaches its floor before 4,400.
K_TICKS = [2e3, 1e4, 5e4, 2e5]      # labelled iterations; 5k, 20k and 100k get unlabelled ticks
K_MINOR = [5e3, 2e4, 1e5]
TRACE_BINS = 40             # log-spaced bins the drawn traces are median-reduced into (see panel_pair)
# Rung labels shortened for the legends of this figure, where a long label covers data; the
# parameters dropped here (leak 0.01, beta 25, the sigmoid's gain and offset) go in the caption.
DISPLAY = {"leaky ReLU\nleak 0.01": "leaky ReLU", "softplus\n$\\beta$ = 25": "softplus",
           "sigmoid\n7.5(x$-$0.3)": "sigmoid", "default\ndraw": "default",
           "rate\n$\\lambda$ = 0.2": "rate 0.2", "sparsity\n$\\lambda$ = 0.05": "sparsity 0.05"}
# A ladder whose traces leave no room for a legend borrows the legend of a ladder with the same
# rungs and colours; its dashed traces still mark the runs with no floor read-out.
BORROW_LEGEND = {"slide_k_inputscale": "slide_k_inputscale_cddm"}
# Short row labels for the bottom-line panel, by stem.
SHORT = {"slide_k_weightdecay": "weight decay", "slide_k_recnoise": "recurrent noise",
         "slide_k_metabolic": "metabolic penalty", "slide_k_recgain": "spectral radius",
         "slide_k_rwm": "$W_{rec}$ penalty", "slide_k_frm_rws": "rate / sparsity pen.",
         "slide_k_activation": "activation", "slide_k_inputscale_cddm": "input scale",
         "slide_k_inputscale": "input scale (FF)", "revival_cddm": "revival", "revival_ff": "revival (FF)"}


def rung_style(spec, i, n):
    """Colour and marker of rung i, the same rule as `fig_knobs_at_floor.tradeoff`.

    Args:
        spec: a ladder; i: rung index; n: number of rungs.
    Returns:
        (colour, marker).
    """
    if spec.get("categorical", False):
        return ps.SLOTS[i], "osD^v"[i]
    return plt.get_cmap("viridis")(np.linspace(0.05, 0.85, n))[i], "o"


def budget(lad):
    """A ladder's common budget: `end_at` when set, else the shortest trace among its runs.

    Args:
        lad: one collected ladder.
    Returns:
        the iteration (float), or nan if no rung has runs.
    """
    if lad["spec"].get("end_at"):
        return float(lad["spec"]["end_at"])
    ends = [r["it_act"][-1] for rg in lad["rungs"] for r in rg["runs"]]
    return float(min(ends)) if ends else float("nan")


def has_runs(lad):
    """True when every rung of a collected ladder has at least one finished run."""
    return all(len(rg["runs"]) for rg in lad["rungs"])


def panel_pair(ax_r2, ax_tr, lad):
    """One ladder as two panels sharing the active-units axis.

    Left (narrow): held-out R^2 at each run's floor (x) against active units there (y), one point
    per run. Right: active units over training for every run (log iteration axis), each run's floor
    crossing marked with the same symbol as its point on the left, so a point and its crossing sit
    at the same height. A run with no usable crossing is drawn dashed, without a symbol.

    Each trace is drawn as medians in TRACE_BINS log-spaced bins (common.logbin). Probe-to-probe
    noise is small on both tasks (median step 1-4 units), but a flip-flop count drops by 100-350
    units within one probe 4-10 times per run on the input-scale ladder, and drawn raw that
    ladder's 15 traces cannot be told apart; the medians keep the level a trace stays at and
    lose the one-probe spikes. The crossing symbol is the RAW count at the probe nearest the
    crossing, the number the left panel and fig_knobs_at_floor report, so on the flip-flop it can
    sit off its line.

    Args:
        ax_r2, ax_tr: the two axes; lad: one collected ladder.
    Returns:
        list of legend handles (one per rung).
    """
    spec = lad["spec"]
    n = len(lad["rungs"])
    handles = []
    for i, rg in enumerate(lad["rungs"]):
        col, mk = rung_style(spec, i, n)
        ref = i == spec["ref"]
        rd = rg["read"]
        if rd is not None and len(rd["floor"]):
            ax_r2.scatter(rd["r2"], rd["floor"], s=16 if ref else 11, marker=mk, color=col,
                          edgecolors="k" if ref else "none", linewidths=0.7, zorder=3 + ref)
        for r in rg["runs"]:
            keep = (r["it_act"] >= X_MIN) & (r["it_act"] <= budget(lad))
            tb, yb = logbin(r["it_act"][keep], r["active"][keep], nbins=TRACE_BINS)
            ax_tr.plot(tb, yb, color=col,
                       lw=0.9 if ref else 0.65, alpha=0.9,
                       ls="-" if np.isfinite(r["t_star"]) else (0, (2, 1.2)), zorder=2 + ref)
        for r in rg["runs"]:
            if np.isfinite(r["t_star"]):
                ax_tr.scatter([r["t_star"]], [active_at(r, r["t_star"])], s=16 if ref else 11,
                              marker=mk, color=col, edgecolors="k" if ref else "none",
                              linewidths=0.7, zorder=5 + ref)
        k = len(rd["floor"]) if rd is not None else 0
        nr = len(rg["runs"])
        lab = DISPLAY.get(rg["label"], rg["label"]).replace("\n", " ") + ("" if k == nr else f" ({k}/{nr})")
        handles.append(Line2D([], [], color=col, marker=mk, ms=3.6 if ref else 3.0, lw=0.8,
                              mec="k" if ref else col, mew=0.6 if ref else 0, label=lab))
    B = budget(lad)
    ax_tr.set_xscale("log")
    ax_tr.set_xlim(X_MIN, B * 1.25)
    ax_tr.set_xticks([t for t in K_TICKS if t <= B * 1.25])
    ax_tr.set_xticklabels([f"{t / 1e3:.0f}k" for t in K_TICKS if t <= B * 1.25])
    ax_tr.set_xticks([t for t in K_MINOR if t <= B * 1.25], minor=True)
    ax_tr.set_xticklabels([], minor=True)
    ax_tr.axvline(B, color=ps.FAINT, lw=0.6, ls=":", zorder=1)
    for ax in (ax_r2, ax_tr):
        ax.set_ylim(0, Y_MAX)
        ax.axhline(N_UNITS, color=ps.GRID, lw=0.6, zorder=0)
    ax_tr.tick_params(labelleft=False)
    return handles


def bottom_line(ax, lads):
    """Every ladder on one active-units axis: each rung at its own floor and at the ladder's
    common budget.

    One row per ladder. On the row, filled symbols are each rung's mean active count at the end of
    the ladder's common budget (every run of the ladder read at the same iteration); the open
    symbols just above are the same rungs at their own floor, joined to their filled symbol by a
    thin line. A knob that read high at its floor only because it got there sooner ends left of its
    reference.

    Args:
        ax: the axes; lads: collected ladders, top to bottom.
    Returns:
        list of (ladder, budget, [(rung, floor mean, end mean)]) rows, for the caption.
    """
    rows = []
    for j, lad in enumerate(lads):
        spec = lad["spec"]
        B = budget(lad)
        y = -j
        out = []
        n = len(lad["rungs"])
        for i, rg in enumerate(lad["rungs"]):
            col, mk = rung_style(spec, i, n)
            ref = i == spec["ref"]
            ends = [active_at(r, B) for r in rg["runs"]]
            ends = [v for v in ends if np.isfinite(v)]
            fl = rg["read"]["floor"] if rg["read"] is not None else []
            m_end = float(np.mean(ends)) if ends else float("nan")
            m_fl = float(np.mean(fl)) if len(fl) else float("nan")
            ax.plot([m_fl, m_end], [y + 0.38, y], color=col, lw=0.5, alpha=0.6, zorder=2)
            ax.scatter([m_end], [y], s=16 if ref else 11, marker=mk, color=col,
                       edgecolors="k" if ref else "none", linewidths=0.7, zorder=4 + ref)
            ax.scatter([m_fl], [y + 0.38], s=9 if ref else 7, marker=mk, facecolors="none",
                       edgecolors=col, linewidths=0.6, zorder=3)
            out.append((rg["label"].replace("\n", " "), m_fl, m_end))
        ax.text(-0.03, y + 0.19, f"{SHORT[spec['stem']]}, {B / 1e3:.0f}k",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=5.6,
                color=ps.INK)
        rows.append((SHORT[spec["stem"]], B, out))
    ax.set_ylim(-len(lads) + 0.5, 0.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, Y_MAX)
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_xlabel("active units (of 1000)", labelpad=1)
    ax.axvline(N_UNITS, color=ps.GRID, lw=0.6, zorder=0)
    return rows


def title_text(lad, pending=False):
    """Two-line pair title: the knob, then the task, where the floors fell, and the budget.

    Args:
        lad: one collected ladder; pending: True for a revival screen still training.
    Returns:
        the title string.
    """
    spec = lad["spec"]
    task = "flip-flop" if "flip" in spec["task"] else "CDDM"
    if pending:
        return f"{spec['knob']}\n{task}: pending"
    cross = [r["t_star"] for rg in lad["rungs"] for r in rg["runs"] if np.isfinite(r["t_star"])]
    where = f"floors at {min(cross) / 1e3:.0f}-{max(cross) / 1e3:.0f}k" if cross else "no floor read"
    knob = "initial spectral radius" if spec["stem"] == "slide_k_recgain" else spec["knob"]
    return f"{knob}\n{task}, {where}, budget {budget(lad) / 1e3:.0f}k"


def covered(ax, leg):
    """Number of drawn trace vertices and symbols that fall inside a legend's box; a legend that
    reaches outside its axes counts as covering everything.

    Args:
        ax: a traces axes, or a tuple of axes for a legend spanning a panel pair (then every axes'
            data is tested and the outside-the-axes rule does not apply); leg: its legend.
    Returns:
        int.
    """
    axes = ax if isinstance(ax, tuple) else (ax,)
    fig = axes[0].figure
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer()).padded(2)
    box = axes[0].get_window_extent()
    if len(axes) == 1 and (bb.x0 < box.x0 - 1 or bb.x1 > box.x1 + 1 or bb.y0 < box.y0 - 1 or bb.y1 > box.y1 + 1):
        return 10 ** 6
    pts = []
    for a in axes:
        pts += [a.transData.transform(ln.get_xydata()) for ln in a.lines if len(ln.get_xydata()) > 2]
        pts += [a.transData.transform(c.get_offsets()) for c in a.collections]
    pts = np.concatenate(pts) if pts else np.zeros((0, 2))
    return int(((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) &
                (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).sum())


def place_legend(ax, handles, alt=None):
    """Put the rung legend where it covers no data: inside the traces axes, else inside `alt`.

    The legend's size is measured once per column count; candidate positions are then tested
    arithmetically against every drawn vertex and symbol: the four corners first (one column, then
    two, then three), then a grid over the whole axes. When the traces axes has no free spot (dense
    traces), the same search runs on `alt`, the paired R^2 panel, whose points sit in one column.
    Falls back to the traces axes' lower-left corner, which `check_layout` then reports.

    Args:
        ax: the traces axes; handles: legend handles; alt: a second axes to try, or None.
    Returns:
        (the axes holding the legend - a tuple (ax, alt) for a legend spanning both - , the legend).
    """
    kw = dict(handles=handles, fontsize=5.0, handlelength=1.6, borderaxespad=0.0,
              labelspacing=0.2, handletextpad=0.4, columnspacing=0.8, loc="lower left")
    fig = ax.figure
    rend = fig.canvas.get_renderer()
    m = 0.015                                   # margin from the axes edge, in axes fractions

    def search(a):
        """The legend placed on axes `a` covering no data, or None (no legend left on `a`)."""
        fig.canvas.draw()
        pts = [a.transData.transform(ln.get_xydata()) for ln in a.lines if len(ln.get_xydata()) > 2]
        pts += [a.transData.transform(c.get_offsets()) for c in a.collections]
        pts = np.concatenate(pts) if pts else np.zeros((0, 2))
        box = a.get_window_extent()
        for grid in (False, True):
            for ncol in (1, 2, 3):
                leg = a.legend(ncol=ncol, bbox_to_anchor=(0, 0), **kw)
                fig.canvas.draw()
                bb = leg.get_window_extent(rend)
                w, h = (bb.width + 4) / box.width, (bb.height + 4) / box.height
                if w > 1 - 2 * m or h > 1 - 2 * m:
                    continue
                if grid:
                    cand = [(x, y) for y in np.arange(m, 1 - h - m, 0.025)
                            for x in np.arange(m, 1 - w - m, 0.025)]
                else:
                    cand = [(m, m), (1 - w - m, 1 - h - m), (1 - w - m, m), (m, 1 - h - m)]
                for x, y in cand:
                    x0, y0 = box.x0 + x * box.width, box.y0 + y * box.height
                    x1, y1 = x0 + w * box.width, y0 + h * box.height
                    hit = ((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)).any()
                    if not hit:
                        return a.legend(ncol=ncol, bbox_to_anchor=(x + 2 / box.width, y + 2 / box.height), **kw)
        if a.get_legend() is not None:
            a.get_legend().remove()
        return None

    for a in (ax, alt):
        if a is not None:
            leg = search(a)
            if leg is not None:
                return a, leg
    if alt is not None:
        # neither panel alone has room (the R^2 panel is narrower than the legend): search the
        # rectangle spanning both panels, testing the data of both, with the legend in figure units
        fig.canvas.draw()
        pts = []
        for a in (ax, alt):
            pts += [a.transData.transform(ln.get_xydata()) for ln in a.lines if len(ln.get_xydata()) > 2]
            pts += [a.transData.transform(c.get_offsets()) for c in a.collections]
        b1, b2 = alt.get_window_extent(), ax.get_window_extent()
        # the panels' left and bottom axis lines are obstacles too: a legend must not straddle them
        for b in (b1, b2):
            ys = np.linspace(b.y0, b.y1, 200)
            xs = np.linspace(b.x0, b.x1, 200)
            pts += [np.c_[np.full_like(ys, b.x0), ys], np.c_[xs, np.full_like(xs, b.y0)]]
        pts = np.concatenate(pts)
        X0, X1, Y0, Y1 = min(b1.x0, b2.x0), max(b1.x1, b2.x1), max(b1.y0, b2.y0), min(b1.y1, b2.y1)
        W, H = fig.bbox.width, fig.bbox.height
        for ncol in (1, 2, 3):
            leg = fig.legend(ncol=ncol, bbox_to_anchor=(0, 0), **kw)
            fig.canvas.draw()
            bb = leg.get_window_extent(rend)
            w, h = bb.width + 4, bb.height + 4
            leg.remove()
            for y0 in np.arange(Y1 - h - 3, Y0 + 3, -0.02 * (Y1 - Y0)):
                for x0 in np.arange(X0 + 3, X1 - w - 3, 0.02 * (X1 - X0)):
                    hit = ((pts[:, 0] > x0) & (pts[:, 0] < x0 + w) & (pts[:, 1] > y0) & (pts[:, 1] < y0 + h)).any()
                    if not hit:
                        return (ax, alt), fig.legend(ncol=ncol, bbox_to_anchor=((x0 + 2) / W, (y0 + 2) / H), **kw)
    return ax, ax.legend(ncol=1, bbox_to_anchor=(m, m), **kw)


def draw(data):
    """Assemble Figure 2 from the collected data and write it (PDF + SVG).

    Layout, 180 mm wide: four rows of three slots. Each of the nine knob ladders and the two revival
    screens takes one slot as a PAIR of panels that share the active-units axis (the narrow R^2
    panel and the traces panel, see panel_pair), so a run's point on the trade-off and its floor
    crossing on its trace sit at the same height. The last slot is the bottom line across ladders.

    Args:
        data: the dict `collect` writes.
    Returns:
        (output path, bottom-line rows).
    """
    # the specs (titles, labels) come from this file, not from the cache, so a wording change
    # needs no re-collect; the rung folders must still be the ones the cache was read from
    current = {s["stem"]: s for s in K.LADDERS + REVIVAL}
    for lad in data["ladders"] + data["revival"]:
        spec = current[lad["spec"]["stem"]]
        # compared below the data root: the cache may have been written on Della
        assert ([p.split("trained_RNNs/")[-1] for _, p in spec["rungs"]]
                == [rg["path"].split("trained_RNNs/")[-1] for rg in lad["rungs"]]), \
            f"{spec['stem']}: rungs changed since the cache was collected; re-run --collect"
        lad["spec"] = spec
    by_stem = {lad["spec"]["stem"]: lad for lad in data["ladders"]}
    lads = [by_stem[s] for s in ORDER]
    rev = data["revival"]

    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 200 * ps.MM))
    gs = GridSpec(4, 3, figure=fig, left=0.065, right=0.985, top=0.955, bottom=0.05,
                  hspace=0.72, wspace=0.30)
    slots = [gs[i // 3, i % 3] for i in range(9)] + [gs[3, 0], gs[3, 1]]
    letters = iter("abcdefghijkl")
    pairs = []
    for slot, lad in zip(slots, lads + rev):
        sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=slot, width_ratios=[0.42, 1.0], wspace=0.20)
        ax_r2, ax_tr = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])
        ps.panel_letter(ax_r2, next(letters), dx=-0.62, dy=1.16)
        if lad in rev and not any(len(rg["runs"]) for rg in lad["rungs"][1:]):
            # the slot for a revival screen whose treated cells have no finished run yet
            for ax in (ax_r2, ax_tr):
                ps.blank(ax)
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                       ls=(0, (3, 2)), lw=0.6, ec=ps.FAINT))
            names = "\nand ".join(rg["label"].replace("\n", " ") for rg in lad["rungs"][1:])
            ax_tr.text(0.5, 0.5, f"{names}:\ntop-ups to the floor\nstill training; drawn\n"
                       "here when they land", transform=ax_tr.transAxes, ha="center",
                       va="center", fontsize=6, color=ps.MUTED, linespacing=1.3)
            pairs.append((ax_r2, ax_tr, lad, None))
            continue
        handles = panel_pair(ax_r2, ax_tr, lad)
        ax_r2.set_xlabel("$R^2$ at floor", labelpad=1)
        ax_tr.set_xlabel("training iteration", labelpad=1)
        pairs.append((ax_r2, ax_tr, lad, handles))

    # one R^2 range for every CDDM pair so they compare by eye; the flip-flop gets its own
    drawn = [p for p in pairs if p[3] is not None]
    for task_ff, ticks in ((False, [0.80, 0.83]), (True, [0.949, 0.951])):  # inside the range: an edge tick ran into the next axis's "2k"
        group = [p for p in drawn if ("flip" in p[2]["spec"]["task"]) == task_ff]
        if not group:
            continue
        r2 = np.concatenate([rg["read"]["r2"] for p in group for rg in p[2]["rungs"]
                             if rg["read"] is not None])
        pad = 0.08 * (r2.max() - r2.min())
        lo, hi = r2.min() - pad, r2.max() + pad
        for p in group:
            p[0].set_xlim(lo, hi)
            p[0].set_xticks(ticks)
    for i, (ax_r2, ax_tr, lad, handles) in enumerate(pairs):
        pos_l, pos_r = ax_r2.get_position(), ax_tr.get_position()
        fig.text((pos_l.x0 + pos_r.x1) / 2, pos_l.y1 + 0.006, title_text(lad, handles is None),
                 ha="center", va="bottom", fontsize=6.2, color=ps.INK, linespacing=1.15)
        if i % 3 == 0:
            ax_r2.set_ylabel("active units (of 1000)")
    # legends last, once every axis has its final limits, so the coverage test reads the real layout
    legends = []
    for ax_r2, ax_tr, lad, handles in pairs:
        if handles is None:
            continue
        lender = BORROW_LEGEND.get(lad["spec"]["stem"])
        if lender:
            handles = [Line2D([], [], ls="none",
                              label=f"rungs as in {'abcdefghi'[ORDER.index(lender)]}")]
        holder, leg = place_legend(ax_tr, handles, alt=ax_r2)
        legends.append((holder, lad, leg))

    # the bottom line sits where a traces panel would, so its axis lines up with the panels above
    # and its row labels take the place of an R^2 panel
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3, 2], width_ratios=[0.42, 1.0], wspace=0.20)
    ax_b = fig.add_subplot(sub[0, 1])
    rows = bottom_line(ax_b, [lad for lad in lads + rev if has_runs(lad)])
    pos = ax_b.get_position()
    ps.panel_letter(ax_b, "l", dx=-0.62 * 0.42 - 0.20 * 0.71, dy=1.16)
    fig.text(pos.x0 + pos.width * 0.35, pos.y1 + 0.006,
             "all ladders: mean at own floor (open)\nand at common budget (filled)",
             ha="center", va="bottom", fontsize=6.2, color=ps.INK, linespacing=1.15)
    check_layout(fig, legends)
    return ps.save(fig, "fig_paper_F2_fails", w_mm=180), rows


def drawn_texts(fig):
    """Every text artist that is actually drawn: titles, labels, legend entries, free text, and the
    tick labels whose tick lies inside its axis' view limits.

    Args:
        fig: the figure.
    Returns:
        list of Text artists.
    """
    out = [t for t in fig.texts if t.get_text().strip()]
    for leg in fig.legends:                      # legends spanning a panel pair live on the figure
        out += list(leg.get_texts())
    for ax in fig.axes:
        out += [t for t in ax.texts if t.get_text().strip()]
        out += [t for t in (ax.title, ax.xaxis.label, ax.yaxis.label) if t.get_text().strip()]
        leg = ax.get_legend()
        if leg is not None:
            out += list(leg.get_texts())
        for axis, (lo, hi) in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lo, hi), max(lo, hi)
            for tk in axis.get_major_ticks():
                if lo <= tk.get_loc() <= hi and tk.label1.get_visible() and tk.label1.get_text():
                    out.append(tk.label1)
    return out


def check_layout(fig, legends):
    """Print every legend that covers data and every pair of drawn texts whose tight bounding boxes
    intersect, plus any text outside the canvas. Read the output; do not trust the eye.

    Args:
        fig: the figure, fully drawn; legends: (traces axes, ladder, legend) tuples.
    Returns:
        number of problems found.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bad = 0
    for ax, lad, leg in legends:
        c = covered(ax, leg)
        if c:
            bad += 1
            print(f"  OVERLAP: legend of {lad['spec']['stem']} covers {c} data points")
    texts = drawn_texts(fig)
    boxes = [(t, t.get_window_extent(rend)) for t in texts]
    W, H = fig.bbox.width, fig.bbox.height
    for t, b in boxes:
        if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H:
            bad += 1
            print(f"  OUTSIDE: '{t.get_text()[:40]}'")
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            if boxes[a][1].overlaps(boxes[b][1]):
                bad += 1
                print(f"  OVERLAP: '{boxes[a][0].get_text()[:30]}' x '{boxes[b][0].get_text()[:30]}'")
    print(f"layout check: {len(boxes)} texts, {len(legends)} legends, {bad} problem(s)")
    return bad


def main():
    """Collect, probe, and draw Figure 2."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect", action="store_true", help="read every run and write the cache")
    ap.add_argument("--probe", action="store_true", help="print the per-run table from the cache")
    a = ap.parse_args()
    if a.collect:
        collect()
    with open(CACHE, "rb") as fh:
        data = pickle.load(fh)
    if a.probe:
        probe(data)
        return
    out, rows = draw(data)
    print("\n--- bottom line: mean active units at own floor -> at the common budget ---")
    for knob, B, rr in rows:
        print(f"  {knob} ({B / 1e3:.0f}k): " + "; ".join(f"{lab} {f:.0f}->{e:.0f}" for lab, f, e in rr))


if __name__ == "__main__":
    main()
