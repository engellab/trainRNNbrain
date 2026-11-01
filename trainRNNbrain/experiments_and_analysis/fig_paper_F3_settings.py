#!/usr/bin/env python3
"""Manuscript Figure 3 - no standard training setting keeps units active - and Extended Data Fig. 5,
the optimizer control (B5). Both are same-size comparisons (N = 1000), so every run is read at a
fixed 75,000 iterations (Pavel, 16 Oct), not at its loss floor; the floor read of the same ladders is
Extended Data Fig. 4 (`fig_paper_F2_fails.py`).

  a  every rung of every ladder and intervention screen: active units at 75k (left) and held-out R^2 at
     75k minus the ladder default's mean (right), one point per run, default ringed. The CDDM
     input-scale ladder ends at 50k and is read there.
  b  weight decay, CDDM: active units against iteration to 200k, floor crossings open
  c  initial input-weight scale, 3-bit flip-flop: the same to 150k
  d  optimizer (B5), 3-bit flip-flop: active units against held-out R^2, both at 75k
  e  interventions designed to raise the active count: active units against iteration, CDDM
     screen (75k) and 3-bit flip-flop (200k)

Extended Data Fig. 5 (`fig_ed_optimizer`): every B5 arm at its floor, at 75k and at 200k.

Inputs, no reading of runs here:
  data/fig_paper_F2_fails_cache.pkl   `fig_paper_F2_fails.py --collect` (Della): every ladder and
                                      revival run, its participation trace, its floor read and its
                                      75k read (`fig_paper_F2_fails.read_fixed`)
  data/b5_read_cache.pkl              `b5_read.py` (Mac): every B5 run at its floor, 75k and 200k

Usage (repository root):  python trainRNNbrain/experiments_and_analysis/fig_paper_F3_settings.py
Output: img/internal_figures/fig_paper_F3_settings.{pdf,svg} and fig_ed_optimizer.{pdf,svg}
(vector only, paperstyle.save), a layout check for each, and every number drawn.
"""

import os
import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fig_knobs_at_floor as K
import fig_paper_F2_fails as F2F
import paperstyle as ps
from b5_read import CACHE as B5_CACHE, SGD_COLOR

N_UNITS = 1000
FRM_COL = ps.COND_COL["frm"]          # rungs that include the rate penalty, a participation drive
ARM_COL = ps.SLOTS[0]                 # every other non-default rung in panel a
BIN = 2500                            # iterations per median bin of the drawn traces (b, c, e)

# Panel a: one block per ladder, top to bottom; header text and one short label per rung.
BLOCKS = [
    ("slide_k_weightdecay", "weight decay (CDDM)", ["0", "10\u207b\u2076 (default)", "10\u207b\u2075", "10\u207b\u2074"]),
    ("slide_k_recnoise", "recurrent noise $\\sigma_{rec}$ (CDDM)", ["0", "0.01", "0.05 (default)", "0.1"]),
    ("slide_k_metabolic", "metabolic penalty $\\lambda$ (CDDM)", ["0 (default)", "0.01", "0.1", "1", "10"]),
    ("slide_k_recgain", "initial spectral radius (CDDM)", ["0.5", "1.2 (default)", "2.0"]),
    ("slide_k_rwm", "$W_{rec}$ magnitude penalty $\\lambda$ (CDDM)", ["0 (default)", "0.05", "0.2"]),
    ("slide_k_activation", "activation (CDDM)", ["ReLU (default)", "leaky ReLU", "softplus", "sigmoid"]),
    ("slide_k_frm_rws", "rate and sparsity penalties (CDDM)", ["none (default)", "rate", "sparsity", "both"]),
    ("slide_k_inputscale_cddm", "input-weight scale (CDDM, at 50k)", ["default", "0.5", "2", "5", "20"]),
    ("slide_k_inputscale", "input-weight scale (3BFF)", ["default", "0.5", "2", "5", "20"]),
    ("revival_cddm", "interventions (CDDM)", ["none", "input-weight cap", "re-initialise"]),
    ("revival_ff", "interventions (3BFF)", ["none", "zero-out", "random redraw"]),
]
FRM_RUNGS = {("slide_k_frm_rws", 1), ("slide_k_frm_rws", 3)}     # rate, both
# SGD learning rate -> marker, on top of its colour (SGD_COLOR): three dark oranges alone are hard
# to tell apart in print. "v" is kept for the runs drawn on the bottom edge.
SGD_MARKER = {"0.03": "^", "0.1": ">", "0.3": "D", "1": "P", "3": "X"}


def load():
    """Both caches, with each ladder's spec refreshed from the code (labels may have changed).

    Returns:
        (by_stem: {stem: collected ladder}, b5: list of ((opt, wd, lr), rows) from b5_read).
    """
    with open(F2F.CACHE, "rb") as fh:
        data = pickle.load(fh)
    current = {s["stem"]: s for s in K.LADDERS + F2F.REVIVAL}
    by_stem = {}
    for lad in data["ladders"] + data["revival"]:
        spec = current[lad["spec"]["stem"]]
        assert ([p.split("trained_RNNs/")[-1] for _, p in spec["rungs"]]
                == [rg["path"].split("trained_RNNs/")[-1] for rg in lad["rungs"]]), \
            f"{spec['stem']}: rungs changed since the cache was collected"
        assert all("fixed" in r for rg in lad["rungs"] for r in rg["runs"]), \
            "cache has no 75k read: re-run fig_paper_F2_fails.py --collect"
        lad["spec"] = spec
        by_stem[spec["stem"]] = lad
    with open(B5_CACHE, "rb") as fh:
        b5 = pickle.load(fh)
    return by_stem, b5


def linbin(t, y, width=BIN):
    """Median of y in consecutive bins of `width` iterations, for drawing a trace on a linear axis.

    Args:
        t, y: equal-length arrays (iterations, values); width: bin width in iterations.
    Returns:
        (bin centres, bin medians), bins holding fewer than 3 points dropped.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    idx = np.floor(t / width).astype(int)
    tb, yb = [], []
    for b in np.unique(idx):
        m = idx == b
        if m.sum() >= 3:
            tb.append((b + 0.5) * width)
            yb.append(np.median(y[m]))
    return np.array(tb), np.array(yb)


def panel_a(ax_n, ax_r, by_stem):
    """Every rung at 75k: active units (ax_n) and held-out R^2 minus the default's (ax_r).

    Args:
        ax_n, ax_r: two axes sharing rows; by_stem: the collected ladders.
    Returns:
        list of (block header, rung label, [active], [R^2], [how R^2 was got], t_read) per rung.
    """
    rows, y = [], 0.0
    yt = []
    for stem, head, labels in BLOCKS:
        lad = by_stem[stem]
        ref = lad["spec"]["ref"]
        r2_ref = np.mean([r["fixed"]["r2"] for r in lad["rungs"][ref]["runs"]])
        n_ref = np.mean([r["fixed"]["active"] for r in lad["rungs"][ref]["runs"]])
        ax_n.text(-0.60, y, head, transform=ax_n.get_yaxis_transform(), ha="left", va="center",
                  fontsize=5.6, fontweight="bold", color=ps.INK)
        y0 = y - 1
        for i, (rg, lab) in enumerate(zip(lad["rungs"], labels)):
            y -= 1
            is_ref = i == ref
            col = ps.BASE if is_ref else (FRM_COL if (stem, i) in FRM_RUNGS else ARM_COL)
            fx = [r["fixed"] for r in rg["runs"]]
            act = np.array([f["active"] for f in fx])
            r2 = np.array([f["r2"] for f in fx])
            meas = np.array([f["r2_how"] != "estimate" for f in fx])
            ax_n.scatter(act, np.full(len(act), y), s=9, color=col, zorder=4,
                         edgecolors="k" if is_ref else "none", linewidths=0.6)
            ax_n.plot([act.mean()] * 2, [y - 0.32, y + 0.32], color=col, lw=0.9, zorder=3)
            for m, fill in ((meas, True), (~meas, False)):
                if m.any():
                    ax_r.scatter(r2[m] - r2_ref, np.full(m.sum(), y), s=9, zorder=4,
                                 facecolors=col if fill else "none", edgecolors="k" if is_ref else col,
                                 linewidths=0.6)
            ax_n.text(-0.02, y, f"{lab} ({len(act)})", transform=ax_n.get_yaxis_transform(),
                      ha="right", va="center", fontsize=5.3, color=ps.INK)
            yt.append(y)
            rows.append((head, lab, act.tolist(), r2.tolist(), [f["r2_how"] for f in fx],
                         fx[0]["t_read"]))
        # the default's mean as a guide through the block
        ax_n.plot([n_ref] * 2, [y - 0.45, y0 + 0.45], color=ps.FAINT, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
        y -= 1.25
    for ax in (ax_n, ax_r):
        ax.set_ylim(y + 0.5, 0.6)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    ax_n.set_xlim(0, 1040)
    ax_n.set_xticks([0, 250, 500, 750, 1000])
    ax_n.axvline(N_UNITS, color=ps.GRID, lw=0.6, zorder=0)
    ax_n.set_xlabel("active units at 75k (of 1000)", labelpad=1)
    ax_r.axvline(0, color=ps.FAINT, lw=0.6, zorder=0)
    ax_r.set_xlabel("$R^2$ minus default", labelpad=1)
    return rows


def traces(ax, lad, x_max, labels, legend_loc):
    """Active units against iteration for every run of a ladder (linear axis), binned medians.

    The floor crossing of each run with a floor read is an open symbol at its raw count; the 75k
    read is a filled symbol; a dashed vertical line marks 75,000 iterations.

    Args:
        ax: axes; lad: one collected ladder; x_max: last iteration drawn; labels: legend text per
        rung; legend_loc: matplotlib legend location, or None for no legend.
    Returns:
        list of (label, [floor crossings], [floor counts], [75k counts], [counts at x_max]).
    """
    spec = lad["spec"]
    n = len(lad["rungs"])
    out, handles = [], []
    for i, (rg, lab) in enumerate(zip(lad["rungs"], labels)):
        col, mk = F2F.rung_style(spec, i, n)
        is_ref = i == spec["ref"]
        cross, at_floor, at_end = [], [], []
        for r in rg["runs"]:
            keep = r["it_act"] <= x_max
            tb, yb = linbin(r["it_act"][keep], r["active"][keep])
            ax.plot(tb, yb, color=col, lw=0.9 if is_ref else 0.65, alpha=0.9, zorder=2 + is_ref)
            if np.isfinite(r["t_star"]):
                cross.append(r["t_star"])
                at_floor.append(F2F.active_at(r, r["t_star"]))
                ax.scatter([r["t_star"]], [at_floor[-1]], s=10, marker=mk, facecolors="none",
                           edgecolors=col, linewidths=0.6, zorder=5)
            ax.scatter([r["fixed"]["probe"]], [r["fixed"]["active"]], s=11, marker=mk, color=col,
                       edgecolors="k" if is_ref else "none", linewidths=0.6, zorder=6)
            at_end.append(F2F.active_at(r, x_max))
        out.append((lab, cross, at_floor, [r["fixed"]["active"] for r in rg["runs"]], at_end))
        handles.append(Line2D([], [], color=col, marker=mk, ms=3.0, lw=0.8,
                              mec="k" if is_ref else col, mew=0.5 if is_ref else 0, label=lab))
    ax.axvline(F2F.FIXED_IT, color=ps.FAINT, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(0, x_max * 1.02)
    ticks = np.arange(0, x_max + 1, 50_000 if x_max > 100_000 else 25_000)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t / 1e3:.0f}k" for t in ticks])
    ax.set_ylim(0, 1040)
    ax.set_yticks([0, 500, 1000])
    ax.axhline(N_UNITS, color=ps.GRID, lw=0.6, zorder=0)
    ax.set_xlabel("training iteration", labelpad=1)
    if legend_loc:
        ax.legend(handles=handles, loc=legend_loc, fontsize=5.5, ncol=2, handlelength=1.4,
                  columnspacing=0.8, labelspacing=0.2, borderaxespad=0.2)
    return out


def b5_style(opt, wd, lr):
    """Colour, marker and fill of one B5 arm, and its legend label.

    Args:
        opt, wd, lr: the arm, as strings ('adam', '1e-6', '1e-3').
    Returns:
        dict(color, marker, filled, label).
    """
    if opt == "adam" and wd == "1e-6":
        return dict(color=ps.BASE, marker="o", filled=True, label="Adam (default)")
    if opt == "adam":
        return dict(color=ps.SLOTS[0], marker="o", filled=True, label="Adam, wd 0")
    if opt == "adamw":
        return dict(color=ps.SLOTS[2], marker="s", filled=True, label="AdamW")
    return dict(color=SGD_COLOR[lr], marker=SGD_MARKER[lr], filled=wd != "0", label=f"SGD lr {lr}")


def b5_scatter(ax, b5, key, y_lo, legend):
    """Every B5 run on one axes: active units against held-out R^2 at one read-out.

    A run below y_lo is drawn on the bottom edge as a downward triangle.

    Args:
        ax: axes; b5: b5_read rows; key: "floor", "fixed" or "end"; y_lo: lower R^2 limit;
        legend: True to add the legend.
    Returns:
        list of (arm, [active], [R^2]) for the runs with a read at this read-out.
    """
    out, handles = [], []
    for (opt, wd, lr), rows in b5:
        st = b5_style(opt, wd, lr)
        if key == "floor":
            pts = [(r["act_floor"], r["r2_floor"]) for r in rows if np.isfinite(r["r2_floor"])]
        elif key == "fixed":
            pts = [(r["fixed"]["active"], r["fixed"]["r2"]) for r in rows]
        else:
            pts = [(r["act_end"], r["r2_end"]) for r in rows]
        pts = [(a, v) for a, v in pts if np.isfinite(a) and np.isfinite(v)]
        out.append(((opt, wd, lr), [a for a, _ in pts], [v for _, v in pts]))
        is_ref = opt == "adam" and wd == "1e-6"
        for a, v in pts:
            low = v < y_lo
            ax.scatter([a], [y_lo if low else v], s=12, marker="v" if low else st["marker"],
                       facecolors=st["color"] if st["filled"] else "none", edgecolors=
                       "k" if is_ref else st["color"], linewidths=0.6, zorder=4 + is_ref, clip_on=False)
        if wd != "0" or opt != "sgd":
            handles.append(Line2D([], [], ls="none", marker=st["marker"], ms=3.2, color=st["color"],
                                  mec="k" if is_ref else st["color"], mew=0.5, label=st["label"]))
    ax.set_xlim(0, 1000)
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_ylim(y_lo, 0.97)
    ax.set_xlabel("active units (of 1000)", labelpad=1)
    if legend:
        handles.append(Line2D([], [], ls="none", marker="D", ms=3.0, mfc="none", mec=ps.MUTED,
                              mew=0.6, label="open: SGD, wd 0"))
        ax.legend(handles=handles, loc="upper right", fontsize=5.5, ncol=2, handlelength=1.0,
                  columnspacing=0.6, labelspacing=0.2, borderaxespad=0.2)
    return out


def check_layout(fig, name):
    """Print every pair of drawn texts whose bounding boxes overlap and any text off the canvas.

    Args:
        fig: the figure, fully drawn; name: figure name for the printout.
    Returns:
        number of problems.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = F2F.drawn_texts(fig)
    boxes = [(t, t.get_window_extent(rend)) for t in texts]
    W, H = fig.bbox.width, fig.bbox.height
    bad = 0
    for t, b in boxes:
        if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H:
            bad += 1
            print(f"  OUTSIDE: '{t.get_text()[:40]}'")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                bad += 1
                print(f"  OVERLAP: '{boxes[i][0].get_text()[:30]}' x '{boxes[j][0].get_text()[:30]}'")
    # legends must not cover data
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            c = F2F.covered(ax, leg)
            if c:
                bad += 1
                print(f"  LEGEND covers {c} data points")
    print(f"{name} layout check: {len(boxes)} texts, {bad} problem(s)")
    return bad


def figure3(by_stem, b5):
    """Assemble Figure 3 and write it (PDF + SVG).

    Args:
        by_stem: collected ladders; b5: B5 rows.
    Returns:
        dict of every number drawn, by panel.
    """
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 165 * ps.MM))
    outer = GridSpec(1, 2, figure=fig, left=0.205, right=0.985, top=0.975, bottom=0.05,
                     width_ratios=[1.0, 1.05], wspace=0.30)
    left = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], width_ratios=[1.0, 0.42], wspace=0.10)
    ax_n, ax_r = fig.add_subplot(left[0, 0]), fig.add_subplot(left[0, 1])
    nums = dict(a=panel_a(ax_n, ax_r, by_stem))
    ps.panel_letter(ax_n, "a", dx=-0.62, dy=1.0)

    right = GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[0, 1], hspace=0.62)
    ax_b = fig.add_subplot(right[0, 0])
    nums["b"] = traces(ax_b, by_stem["slide_k_weightdecay"], 200_000,
                       ["wd 0", "$10^{-6}$ (default)", "$10^{-5}$", "$10^{-4}$"], "upper right")
    ax_b.set_title("weight decay, CDDM", fontsize=6.5, pad=2)
    ax_c = fig.add_subplot(right[1, 0])
    nums["c"] = traces(ax_c, by_stem["slide_k_inputscale"], 150_000,
                       ["default", "scale 0.5", "2", "5", "20"], "upper right")
    ax_c.set_title("initial input-weight scale, 3BFF", fontsize=6.5, pad=2)
    ax_d = fig.add_subplot(right[2, 0])
    nums["d"] = b5_scatter(ax_d, b5, "fixed", 0.86, legend=True)
    ax_d.set_ylabel("held-out $R^2$")
    ax_d.set_title("optimizer, 3BFF, at 75k", fontsize=6.5, pad=2)
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=right[3, 0], wspace=0.12, width_ratios=[1, 1.6])
    ax_e1, ax_e2 = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])
    nums["e_cddm"] = traces(ax_e1, by_stem["revival_cddm"], 75_000, ["none", "cap", "re-init."], "upper right")
    nums["e_ff"] = traces(ax_e2, by_stem["revival_ff"], 200_000, ["none", "zero-out", "redraw"], "upper right")
    ax_e2.tick_params(labelleft=False)
    ax_e1.set_title("interventions: CDDM", fontsize=6.5, pad=2, loc="left")
    ax_e2.set_title("3BFF", fontsize=6.5, pad=2)
    for ax in (ax_b, ax_c, ax_e1):
        ax.set_ylabel("active units")
    for ax, l in ((ax_b, "b"), (ax_c, "c"), (ax_d, "d"), (ax_e1, "e")):
        ps.panel_letter(ax, l, dx=-0.17 if ax is not ax_e1 else -0.42, dy=1.02)
    check_layout(fig, "fig_paper_F3_settings")
    ps.save(fig, "fig_paper_F3_settings", w_mm=180)
    return nums


def figure_ed5(b5):
    """Extended Data Fig. 5: every B5 run at its floor, at 75k and at 200k (PDF + SVG).

    Args:
        b5: B5 rows.
    Returns:
        dict of the points drawn, by read-out.
    """
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 62 * ps.MM))
    gs = GridSpec(1, 3, figure=fig, left=0.065, right=0.975, top=0.88, bottom=0.17, wspace=0.12)
    nums = {}
    for k, (key, title) in enumerate((("floor", "at each run's loss floor"), ("fixed", "at 75,000 iterations"),
                                      ("end", "at 200,000 iterations (end)"))):
        ax = fig.add_subplot(gs[0, k])
        nums[key] = b5_scatter(ax, b5, key, 0.86, legend=k == 2)
        ax.set_title(title, fontsize=6.5, pad=2)
        if k:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("held-out $R^2$")
        ps.panel_letter(ax, "abc"[k], dx=-0.14 if k == 0 else -0.05, dy=1.02)
    check_layout(fig, "fig_ed_optimizer")
    ps.save(fig, "fig_ed_optimizer", w_mm=180)
    return nums


def report(nums3, nums5):
    """Print every number drawn, per panel, as ranges over runs.

    Args:
        nums3: figure3's return; nums5: figure_ed5's return.
    Returns:
        None.
    """
    def rng(v, fmt="{:.0f}"):
        """'lo-hi' of a list (or the one value), formatted."""
        v = [x for x in v if np.isfinite(x)]
        if not v:
            return "-"
        lo, hi = min(v), max(v)
        return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}-{fmt.format(hi)}"

    print("\n--- Figure 3a: active units at 75k (R^2 at 75k; how) ---")
    for head, lab, act, r2, how, t in nums3["a"]:
        print(f"  {head:<42} {lab:<22} n={len(act)} active {rng(act):<10} mean {np.mean(act):6.1f}"
              f"  R2 {rng(r2, '{:.3f}'):<12} [{','.join(sorted(set(how)))}] read at {t / 1e3:.0f}k")
    for p in ("b", "c", "e_cddm", "e_ff"):
        print(f"\n--- Figure 3{p}: crossing (k) / at floor / at 75k / at end of axis ---")
        for lab, cross, fl, f75, end in nums3[p]:
            print(f"  {lab:<22} t* {rng([c / 1e3 for c in cross])}k  floor {rng(fl)}  75k {rng(f75)}  end {rng(end)}")
    for key in ("floor", "fixed", "end"):
        print(f"\n--- B5 ({key}) ---")
        for (opt, wd, lr), act, r2 in nums5[key]:
            mean = f"{np.mean(act):6.1f}" if act else "     -"
            print(f"  {opt:<6} wd {wd:<5} lr {lr:<5} n={len(act)} active {rng(act):<10} mean {mean}"
                  f"  R2 {rng(r2, '{:.3f}')}")


def main():
    """Draw Figure 3 and Extended Data Fig. 5 from the two caches and print every number."""
    by_stem, b5 = load()
    nums3 = figure3(by_stem, b5)
    nums5 = figure_ed5(b5)
    report(nums3, nums5)


if __name__ == "__main__":
    main()
