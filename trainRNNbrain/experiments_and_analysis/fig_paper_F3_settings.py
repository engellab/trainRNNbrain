#!/usr/bin/env python3
"""Manuscript Figure 3 - standard training settings move the active count but none stops its fall.

Same-size comparisons (N = 1000), so every network is measured at a fixed 75,000 iterations
(Pavel, 16 Oct), not at its floor crossing; both measurement points of every sweep are in Extended
Data Fig. 4 (`fig_paper_F2_fails.py`), together with the optimizer comparison at the floor crossing
and at 200,000 iterations.

ROUND 2 (dead-relu-paper#54). Drawn at the printed width (165.1 mm) in the house style of
`paperstyle.py`; panel a holds only the rows Results 3 discusses (the full set is Extended Data
Fig. 4); colours by meaning: default grey, other setting values on the teal ramp, the rate penalty in
the violet rule family; no task colour.

  a  active units at 75,000 iterations (left) and held-out R^2 minus the sweep's default (right), one
     point per network: CDDM default; weight decay 0 and 1e-4 (the highest and lowest means); the
     metabolic penalty 10 (the largest R^2 loss); the means of the 15 other CDDM values; the rate
     penalty alone and with the sparsity penalty (a per-unit rule, Results 5); 3BFF default and
     initial input-weight scale 20
  b  weight decay, CDDM: active units against iteration to 200,000
  c  initial input-weight scale, 3BFF: the same to 150,000
  d  optimizer comparison, 3BFF: held-out R^2 against active units at 75,000 iterations
  e  revival methods: input-weight cap and re-initialisation (CDDM, 75,000) and re-initialisation
     with outgoing weights zeroed or kept (3BFF, 200,000)

Inputs, no reading of runs here:
  data/fig_paper_F2_fails_cache.pkl   `fig_paper_F2_fails.py --collect` (Della): every sweep and
                                      revival run, its participation trace, its floor crossing and its
                                      75k measurement (`fig_paper_F2_fails.read_fixed`)
  data/b5_read_cache.pkl              `b5_read.py` (Mac): every optimizer run at its floor crossing,
                                      75k and 200k

Usage (repository root):  python trainRNNbrain/experiments_and_analysis/fig_paper_F3_settings.py
Output: img/internal_figures/fig_paper_F3_settings.{pdf,svg} (vector only, paperstyle.save), a layout
check, and every number drawn.
"""

import os
import pickle
import sys

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fig_knobs_at_floor as K
import fig_paper_F2_fails as F2F
import paperstyle as ps

N_UNITS = 1000
BIN = 2500                            # iterations per median bin of the drawn traces (b, c, e)

# Panel a, top to bottom: (block, row label, stem, value index or OTHERS).
OTHERS = "others"
ROWS_A = [
    ("CDDM", "default", "slide_k_weightdecay", 1),
    ("CDDM", "weight decay 0", "slide_k_weightdecay", 0),
    ("CDDM", "weight decay $10^{-4}$", "slide_k_weightdecay", 3),
    ("CDDM", "metabolic penalty 10", "slide_k_metabolic", 4),
    ("CDDM", "15 other values (means)", None, OTHERS),
    ("CDDM", "rate penalty", "slide_k_frm_rws", 1),
    ("CDDM", "rate and sparsity penalties", "slide_k_frm_rws", 3),
    ("3BFF", "default", "slide_k_inputscale", 0),
    ("3BFF", "input-weight scale 20", "slide_k_inputscale", 4),
]
# The CDDM sweeps of panel a (the CDDM input-weight scale ends at 50,000 iterations and is left to
# Extended Data Fig. 4h; the revival methods are panel e).
CDDM_SWEEPS = ["slide_k_weightdecay", "slide_k_recnoise", "slide_k_metabolic", "slide_k_recgain",
               "slide_k_rwm", "slide_k_activation", "slide_k_frm_rws"]
RATE_RUNGS = {("slide_k_frm_rws", 1), ("slide_k_frm_rws", 3)}    # per-unit rule, not a setting
SHOWN = {(stem, i) for _, _, stem, i in ROWS_A if i != OTHERS}
# Optimizer panel: SGD learning rate 3 does not learn the task and is left to Extended Data Fig. 4l.
SKIP_LR = ("3",)
# Direct labels of the SGD clusters in panel d: (text, active units, R^2) at the label's left end.
SGD_LABELS = [("SGD, 0.03", 690, 0.882), ("0.1", 612, 0.903), ("0.3 and 1", 300, 0.882)]
# Layout, in millimetres on the 165.1 mm x H_MM canvas: (x0, y0, width, height) of every axes.
H_MM = 128
POS = {"a_n": (42, 75, 45, 47), "a_r": (91, 75, 20, 47), "b": (123, 75, 37, 47),
       "c": (12, 12, 36, 43), "d": (59, 12, 42, 43), "e1": (109, 12, 22, 43), "e2": (137, 12, 24, 43)}


def load():
    """Both caches, with each sweep's spec refreshed from the code (labels may have changed).

    Returns:
        (by_stem: {stem: collected sweep}, b5: list of ((opt, wd, lr), rows) from b5_read).
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
            "cache has no 75k measurement: re-run fig_paper_F2_fails.py --collect"
        lad["spec"] = spec
        by_stem[spec["stem"]] = lad
    with open(F2F.B5_CACHE, "rb") as fh:
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


def fixed_of(lad, i):
    """Active counts, held-out R^2 and how R^2 was got, at 75,000 iterations, for value i of a sweep.

    Args:
        lad: one collected sweep; i: value index.
    Returns:
        (active array, R^2 array, list of 'final'/'parent'/'snapshot'/'estimate').
    """
    fx = [r["fixed"] for r in lad["rungs"][i]["runs"]]
    return (np.array([f["active"] for f in fx]), np.array([f["r2"] for f in fx]),
            [f["r2_how"] for f in fx])


def panel_a(ax_n, ax_r, by_stem):
    """The rows Results 3 discusses, at 75,000 iterations: active units (ax_n) and R^2 minus default (ax_r).

    One point per network. R^2 is taken relative to the mean of the same sweep's default networks;
    filled points are measured on weights saved at 75,000 iterations, open ones estimated by the loss
    ratio. The "15 other values" row draws each remaining CDDM value's mean (open teal diamonds); its
    networks are in Extended Data Fig. 4.

    Args:
        ax_n, ax_r: two axes sharing rows; by_stem: the collected sweeps.
    Returns:
        list of (block, row label, [active], [R^2 minus default], [how R^2 was got]) per row.
    """
    rows, y, block = [], 0.0, None
    guide, row_y = {}, []
    for blk, lab, stem, i in ROWS_A:
        if blk != block:
            if block is not None:
                y -= 0.6
            ax_n.text(-0.02, y, blk, transform=ax_n.get_yaxis_transform(), ha="right", va="center",
                      fontsize=ps.LABEL_PT, fontweight="bold", color=ps.INK)
            y -= 1
            block = blk
        if i == OTHERS:
            act, dr2 = [], []
            for s in CDDM_SWEEPS:
                lad = by_stem[s]
                ref = lad["spec"]["ref"]
                r2_ref = fixed_of(lad, ref)[1].mean()
                for k in range(len(lad["rungs"])):
                    if k == ref or (s, k) in SHOWN or (s, k) in RATE_RUNGS:
                        continue
                    a, r2, _ = fixed_of(lad, k)
                    act.append(a.mean())
                    dr2.append(r2.mean() - r2_ref)
            act, dr2 = np.array(act), np.array(dr2)
            assert len(act) == 15, f"expected 15 other CDDM values, found {len(act)}"
            for ax, x in ((ax_n, act), (ax_r, dr2)):     # diamonds: open circles mean "estimated"
                ax.scatter(x, np.full(len(x), y), s=9, marker="D", facecolors="none",
                           edgecolors=ps.SETTING, linewidths=0.6, zorder=4)
            rows.append((blk, lab, act.tolist(), dr2.tolist(), ["mean of value"] * len(act)))
        else:
            lad = by_stem[stem]
            ref = lad["spec"]["ref"]
            col, _ = F2F.rung_style(lad["spec"], i, len(lad["rungs"]))
            act, r2, how = fixed_of(lad, i)
            dr2 = r2 - fixed_of(lad, ref)[1].mean()
            is_ref = i == ref
            edge = ps.INK if is_ref else "none"
            ax_n.scatter(act, np.full(len(act), y), s=10, color=col, edgecolors=edge, linewidths=0.5,
                         zorder=4)
            ax_n.plot([act.mean()] * 2, [y - 0.3, y + 0.3], color=col, lw=ps.DATA_LW, zorder=3)
            meas = np.array([h != "estimate" for h in how])
            for m, fill in ((meas, True), (~meas, False)):
                if m.any():
                    ax_r.scatter(dr2[m], np.full(m.sum(), y), s=10, zorder=4,
                                 facecolors=col if fill else "none",
                                 edgecolors=(ps.INK if is_ref else col) if fill or is_ref else col,
                                 linewidths=0.6)
            if is_ref:
                guide[blk] = [act.mean(), y, y]
            rows.append((blk, lab, act.tolist(), dr2.tolist(), how))
        if blk in guide:
            guide[blk][2] = y
        ax_n.text(-0.02, y, lab, transform=ax_n.get_yaxis_transform(), ha="right", va="center",
                  fontsize=ps.LABEL_PT, color=ps.INK)
        row_y.append(y)
        y -= 1
    # the two rate-penalty rows are a per-unit rule (Results 5), not a standard setting: say so in ink
    y_rate = [yy for (blk, lab, stem, i), yy in zip(ROWS_A, row_y) if (stem, i) in RATE_RUNGS]
    ax_n.text(960, np.mean(y_rate), "per-unit rule", ha="right", va="center", fontsize=ps.TICK_PT,
              color=ps.MUTED)
    for x, y0, y1 in guide.values():           # the block default's mean, through its block
        ax_n.plot([x, x], [y1 - 0.45, y0 + 0.45], color=ps.FAINT, lw=ps.THIN_LW, ls=(0, (2, 1.5)),
                  zorder=1)
    for ax in (ax_n, ax_r):
        ax.set_ylim(y + 0.4, 0.6)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    ax_n.set_xlim(0, 1040)
    ax_n.set_xticks([0, 250, 500, 750, 1000])
    ps.count_ticks(ax_n.xaxis)
    ax_n.axvline(N_UNITS, color=ps.GRID, lw=0.6, zorder=0)
    ax_n.set_xlabel("active units at 75,000 iterations", labelpad=1)
    ax_r.axvline(0, color=ps.FAINT, lw=ps.THIN_LW, zorder=0)
    ax_r.set_xlim(-0.055, 0.012)
    ax_r.set_xticks([-0.04, -0.02, 0])
    ax_r.set_xticklabels(["−0.04", "−0.02", "0"])
    ax_r.set_xlabel("held-out $R^2$ minus default", labelpad=1)
    return rows


def traces(ax, lad, x_max, ticks, legend_loc, ncol=1):
    """Active units against iteration for every run of a sweep (linear axis), binned medians.

    The floor crossing of each run that has one is an open symbol at its raw count; the 75,000
    iteration measurement is a filled symbol; a dashed vertical line marks 75,000 iterations. Colours
    and legend text come from `fig_paper_F2_fails` (rung_style, VALUE_LABELS).

    Args:
        ax: axes; lad: one collected sweep; x_max: last iteration drawn; ticks: x tick positions;
        legend_loc: matplotlib legend location, or None for no legend; ncol: legend columns.
    Returns:
        (list of (label, [floor crossings], [counts at floor crossing], [counts at 75k],
        [counts at x_max]), the legend or None).
    """
    spec = lad["spec"]
    n = len(lad["rungs"])
    labels = F2F.VALUE_LABELS[spec["stem"]]
    out, handles = [], []
    for i, (rg, lab) in enumerate(zip(lad["rungs"], labels)):
        col, mk = F2F.rung_style(spec, i, n)
        is_ref = i == spec["ref"]
        cross, at_floor, at_end = [], [], []
        for r in rg["runs"]:
            keep = r["it_act"] <= x_max
            tb, yb = linbin(r["it_act"][keep], r["active"][keep])
            ax.plot(tb, yb, color=col, lw=0.8 if is_ref else 0.6, alpha=0.9, zorder=2 + is_ref)
            if np.isfinite(r["t_star"]):
                cross.append(r["t_star"])
                at_floor.append(F2F.active_at(r, r["t_star"]))
                ax.scatter([r["t_star"]], [at_floor[-1]], s=9, marker=mk, facecolors="none",
                           edgecolors=col, linewidths=0.6, zorder=5)
            ax.scatter([r["fixed"]["probe"]], [r["fixed"]["active"]], s=10, marker=mk, color=col,
                       edgecolors=ps.INK if is_ref else "none", linewidths=0.5, zorder=6)
            at_end.append(F2F.active_at(r, x_max))
        out.append((lab, cross, at_floor, [r["fixed"]["active"] for r in rg["runs"]], at_end))
        handles.append(Line2D([], [], color=col, marker=mk, ms=3.0, lw=0.8,
                              mec=ps.INK if is_ref else col, mew=0.5 if is_ref else 0, label=lab))
    ax.axvline(F2F.FIXED_IT, color=ps.FAINT, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(0, x_max * 1.02)
    ax.set_xticks(ticks)
    ps.count_ticks(ax.xaxis)
    ax.set_ylim(0, 1040)
    ax.set_yticks([0, 500, 1000])
    ps.count_ticks(ax.yaxis)
    ax.axhline(N_UNITS, color=ps.GRID, lw=0.6, zorder=0)
    ax.set_xlabel(ps.LABELS["iteration"], labelpad=1)
    leg = None
    if legend_loc:
        leg = ax.legend(handles=handles, loc=legend_loc, fontsize=ps.TICK_PT, ncol=ncol,
                        handlelength=1.0, columnspacing=0.8, labelspacing=0.15, borderaxespad=0.3,
                        handletextpad=0.3)
    return out, leg


def panel_d(ax, b5):
    """Optimizer comparison on 3BFF at 75,000 iterations: held-out R^2 against active units.

    Markers and colours from `fig_paper_F2_fails.b5_style` (shared with Extended Data Fig. 4l). The
    legend names the optimizers; the SGD learning rates are labelled at their clusters.

    Args:
        ax: axes; b5: b5_read rows.
    Returns:
        (per-arm points, the legend).
    """
    pts, _ = F2F.b5_scatter(ax, b5, "fixed", skip_lr=SKIP_LR)
    ax.set_xticks([0, 250, 500, 750, 1000])
    ps.count_ticks(ax.xaxis)
    ax.set_yticks([0.87, 0.90, 0.93, 0.96])
    handles = []
    for opt, wd, lr, lab in (("adam", "1e-6", "1e-3", "Adam (default)"),
                             ("adam", "0", "1e-3", "Adam, no weight decay"),
                             ("adamw", "1e-6", "1e-3", "AdamW"),
                             ("sgd", "1e-6", "0.1", "SGD (learning rate)")):
        st = F2F.b5_style(opt, wd, lr)
        handles.append(Line2D([], [], ls="none", marker=st["marker"], ms=3.0, color=st["color"],
                              mec=ps.INK if lab.endswith("(default)") else st["color"], mew=0.5,
                              label=lab))
    for text, x, y in SGD_LABELS:
        ax.text(x, y, text, fontsize=ps.TICK_PT, color=ps.INK, va="center", ha="left")
    ax.set_xlabel(ps.LABELS["active"], labelpad=1)
    ax.set_ylabel(ps.LABELS["r2"], labelpad=1)
    # between the Adam runs (R^2 above 0.95) and the SGD runs (below 0.92), right of every run
    leg = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.79), fontsize=ps.TICK_PT,
                    handlelength=0.8, labelspacing=0.15, borderaxespad=0.3, handletextpad=0.3)
    return pts, leg


def check_layout(fig, name, legends):
    """Print every pair of drawn texts whose bounding boxes overlap, any text off the canvas, and any
    legend that covers data.

    Args:
        fig: the figure, fully drawn; name: figure name for the printout; legends: (axes, legend) pairs.
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
    for ax, leg in legends:
        c = F2F.covered(ax, leg)
        if c:
            bad += 1
            print(f"  LEGEND covers {c} data points: {[t.get_text() for t in leg.get_texts()][:2]}")
    print(f"{name} layout check: {len(boxes)} texts, {bad} problem(s)")
    return bad


def figure3(by_stem, b5):
    """Assemble Figure 3 at its printed size and write it (PDF + SVG).

    Args:
        by_stem: collected sweeps; b5: optimizer comparison rows.
    Returns:
        dict of every number drawn, by panel.
    """
    ps.setup()
    fig = ps.figure(H_MM)

    def axes_mm(x0, y0, w, h):
        """A new axes at (x0, y0) with size (w, h), all in millimetres from the canvas' lower left."""
        return fig.add_axes([x0 / ps.W_FULL_MM, y0 / H_MM, w / ps.W_FULL_MM, h / H_MM])

    legends = []
    a_n, a_r = axes_mm(*POS["a_n"]), axes_mm(*POS["a_r"])
    nums = dict(a=panel_a(a_n, a_r, by_stem))
    ps.panel_letter(a_n, "a", x_mm=POS["a_n"][0] - 1.0)

    ax_b = axes_mm(*POS["b"])
    nums["b"], leg = traces(ax_b, by_stem["slide_k_weightdecay"], 200_000, [0, 100_000, 200_000],
                            "upper right", ncol=2)
    legends.append((ax_b, leg))
    ax_b.set_title("weight decay, CDDM", fontsize=ps.LABEL_PT, pad=2)

    ax_c, ax_d, ax_e1, ax_e2 = (axes_mm(*POS[k]) for k in ("c", "d", "e1", "e2"))
    nums["c"], leg = traces(ax_c, by_stem["slide_k_inputscale"], 150_000, [0, 50_000, 100_000, 150_000],
                            "upper right", ncol=2)
    legends.append((ax_c, leg))
    ax_c.set_title("initial input-weight scale, 3BFF", fontsize=ps.LABEL_PT, pad=2)
    nums["d"], leg = panel_d(ax_d, b5)
    legends.append((ax_d, leg))
    ax_d.set_title("optimizer, 3BFF", fontsize=ps.LABEL_PT, pad=2)
    nums["e_cddm"], leg = traces(ax_e1, by_stem["revival_cddm"], 75_000, [0, 75_000], "upper right")
    legends.append((ax_e1, leg))
    nums["e_ff"], leg = traces(ax_e2, by_stem["revival_ff"], 200_000, [0, 200_000], "upper right")
    legends.append((ax_e2, leg))
    ax_e2.tick_params(labelleft=False)
    ax_e1.set_title("revival methods, CDDM", fontsize=ps.LABEL_PT, pad=2)
    ax_e2.set_title("re-initialising, 3BFF", fontsize=ps.LABEL_PT, pad=2)
    for ax in (ax_b, ax_c, ax_e1):
        ax.set_ylabel(ps.LABELS["active"], labelpad=1)
    for ax, letter in ((ax_b, "b"), (ax_c, "c"), (ax_d, "d"), (ax_e1, "e")):
        ps.panel_letter(ax, letter, x_mm=9.0)
    check_layout(fig, "fig_paper_F3_settings", legends)
    ps.save(fig, "fig_paper_F3_settings")
    return nums


def report(nums):
    """Print every number drawn, per panel, as ranges over runs and means.

    Args:
        nums: figure3's return.
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

    print("\n--- Figure 3a: active units at 75k; held-out R^2 minus the sweep's default (how) ---")
    for blk, lab, act, dr2, how in nums["a"]:
        print(f"  {blk:<5} {lab:<30} n={len(act):<3} active {rng(act):<10} mean {np.mean(act):6.1f}"
              f"  dR2 {rng(dr2, '{:+.4f}'):<16} mean {np.mean(dr2):+.4f} [{','.join(sorted(set(how)))}]")
    for p in ("b", "c", "e_cddm", "e_ff"):
        print(f"\n--- Figure 3{p}: crossing (k) / at floor / at 75k / at end of axis (mean) ---")
        for lab, cross, fl, f75, end in nums[p]:
            print(f"  {lab:<22} t* {rng([c / 1e3 for c in cross])}k  floor {rng(fl)}  75k {rng(f75)}"
                  f" (mean {np.mean(f75):.1f})  end {rng(end)} (mean {np.mean(end):.1f})")
    print("\n--- Figure 3d: optimizer at 75k ---")
    for (opt, wd, lr), act, r2 in nums["d"]:
        mean = f"{np.mean(act):6.1f}" if act else "     -"
        print(f"  {opt:<6} wd {wd:<5} lr {lr:<5} n={len(act)} active {rng(act):<10} mean {mean}"
              f"  R2 {rng(r2, '{:.3f}')}")


def main():
    """Draw Figure 3 from the two caches and print every number."""
    by_stem, b5 = load()
    report(figure3(by_stem, b5))


if __name__ == "__main__":
    main()
