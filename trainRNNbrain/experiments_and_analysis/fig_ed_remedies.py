#!/usr/bin/env python3
"""
Extended Data for Figure 5: one figure of the training techniques across network sizes on 3BFF
(paper float ED9_remedies_size; fig_ed_techniques_sizes: the fold change at three margins, a-c,
over the rate penalty alone, d-g), and the rows of Supplementary Table 2 (float EDT3_remedies,
printed as LaTeX). Until round 17 the two halves were two figures, fig_ed_remedies_size and
fig_ed_rate_alone; they were merged for dead-relu-paper#190 (Reviewer S, structure item 7: at most
ten Extended Data figures). Round 2 (dead-relu-paper#55): drawn at the printed width, 165.1 mm, in
the house style.

  d-g         the rate penalty alone (lambda_frm 0.1) on 3BFF.
              d-f, N = 500, 1,000, 2,000: the noisy training loss (per-250-iteration maximum and
              median) and the active count at every participation probe, every network, with each
              network's own floor crossing (1.05x) where one exists (common.read_run).
              g, held-out R^2 at the floor crossing against N, rate penalty and control networks,
              circles where measured on kept weights and crosses where estimated (every control
              value is estimated), with the test's fail threshold (the control networks' mean minus
              0.02). Every value is at the floor crossing, so the circles are filled
              (dead-relu-paper#83: fill marks the measurement point only).
  a-c         fold change against N for every size-series rule, at the crossings of 1.05x, 1.10x
              and 1.20x each network's own fitted floor. A network with no crossing at 1.05x because
              its loss is still falling at 200,000 iterations ("never stays under the bar") is
              counted at a looser margin when it has a crossing there (rm_exclusion_diag.py output);
              networks excluded for any other reason are excluded at every margin. Fold change =
              the network's active count over the control networks' mean at the same N and margin.
  Supp. T. 2  printed rows, one block per rule: N, floor crossings / networks, loss bursts per network
              (range over every network trained), active units at the floor crossing, fold change and
              R^2 change at the floor crossing and at 75,000 iterations, how many floor-crossing R^2
              values are measured on kept weights, how many 75,000-iteration R^2 values come from
              weights inside a loss burst or a non-finite stretch (fig5_burst_check.py, rule in
              notebook 2026-11-01_0745), and the removal analysis at 75,000 iterations (median R^2
              after removing the k added units and after removing k random active units).

Data and provenance: as fig_paper_F5_drive.py (data/remedy_reads_31oct/, the rate-penalty runs
under data/trained_RNNs/NBitFlipFlop_remedy_grid/), plus the per-network records of the same
catalogue read (Della ~/rm_read/runs_*.json, copied to data/remedy_reads_31oct/rm_read_runs/),
which hold each network's burst count.

Usage:  python fig_ed_remedies.py
Output: img/internal_figures/fig_ed_techniques_sizes.pdf (+ .svg), and the
        LaTeX rows of Supplementary Table 2 on stdout.
"""

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
from fig_paper_F5_drive import (COL, MARK, NAME, RATE_SHADES, READS, SIZES, load_a3, load_b2,
                                load_catalogue, load_marks, per_run, style_axes, size_series)

ED9_FAMS = ("frm", "synnoise", "rescale", "duplicate", "dropout", "frm+rws")
MARGINS = (("active", "1.05"), ("active_d10", "1.10"), ("active_d20", "1.20"))
FAIL_R2 = 0.02           # the rate-penalty size test fails if R^2 at the floor crossing is lower by more
TABLE_NAME = {"frm": "Rate penalty", "synnoise": "Noisy synapses", "rescale": "Rescaling",
              "duplicate": "Prune-and-duplicate", "dropout": "Read-out dropout",
              "frm+rws": "Rate + sparsity"}


def load_diag():
    """The rm_exclusion_diag.py records for the networks that never stay under the 1.05x bar.

    Returns:
        dict cell -> list of records (active_d10, active_d20: counts at the 1.10x / 1.20x
        crossing, None where there is none).
    """
    out = {}
    for f in ("rm_diag.json", "rm_diag33.json"):
        with open(os.path.join(READS, f)) as fh:
            for r in json.load(fh):
                out.setdefault(r["cell"], []).append(r)
    return out


def load_bursts():
    """Loss-burst counts of every network in the catalogue read (read_remedies.py per-run records).

    A burst is common.read_run's: a stretch where the 51-iteration running median of the noisy loss
    exceeds twice a wide reference curve, counted from the fit start (Methods).

    Returns:
        dict cell path -> list of int burst counts, one per network that has one (a diverged
        network, whose folder is named nan, has none).
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(READS, "rm_read_runs", "runs_*.json"))):
        with open(f) as fh:
            for r in json.load(fh):
                if "n_bursts" in r:
                    out.setdefault(r["cell"], []).append(int(r["n_bursts"]))
    return out


def load_where_75k():
    """Whether each network's weights at 75,000 iterations lie inside a loss burst or non-finite stretch.

    Pools the catalogue's 75,000-iteration values and the removal analysis's intact scores from
    fig5_burst_check.py (the same weights, so the same verdict; checked: no network differs).

    Returns:
        dict cell -> {network folder name: "calm", "burst" or "non-finite"}.
    """
    with open(os.path.join(READS, "fig5r2_read", "burst_check.json")) as fh:
        res = json.load(fh)
    out = {}
    for r in res:
        if "error" in r:
            continue
        for v in r["values"]:
            if (r["source"], v["what"]) in (("catalogue", "75k"), ("a3", "intact")):
                name = os.path.basename(r["run_dir"].rstrip("/"))
                prev = out.setdefault(r["cell"], {}).setdefault(name, v["where"])
                assert prev == v["where"], f"two verdicts for {name}"
    return out


def fmt_range(v):
    """'57--60' for a list of ints, '57' when min equals max, '--' when empty."""
    if not len(v):
        return "--"
    lo, hi = int(min(v)), int(max(v))
    return str(lo) if lo == hi else f"{lo}--{hi}"


def margin_counts(cell, key, diag):
    """Every network of a cell with a crossing at one margin, and its active count there.

    Args:
        cell: catalogue cell dict; key: "active", "active_d10" or "active_d20"; diag: load_diag().
    Returns:
        (counts array, number of networks found in the cell).
    """
    vals = list(cell[key])
    if key != "active":
        vals += [r[key] for r in diag.get(cell["cell"], []) if r[key] is not None]
    return np.asarray(vals, float), cell["n_runs"]


# --------------------------------------------------------------------------------------------
# The Extended Data figure of the techniques across sizes (until round 17: ED Fig. 7 and ED Fig. 8)
# --------------------------------------------------------------------------------------------

SIZES_H_MM = 128.0       # the merged figure: the margins row (a-c) over the rate penalty alone (d-g)


def y_frac(mm):
    """Figure fraction of a height given in mm from the top edge of a SIZES_H_MM-tall figure.

    Args:
        mm: float, millimetres from the top edge.
    Returns:
        float, the matplotlib figure fraction.
    """
    return 1.0 - mm / SIZES_H_MM


def panel_rate_alone(fig, series, top_mm, bottom_mm):
    """Panels d-g of the sizes figure: the rate penalty alone on 3BFF (the previous fig_ed_rate_alone).

    d-f, N = 500, 1,000, 2,000: the noisy loss (top) and the number of active units (bottom) over
    training, every network, its floor crossing marked where it has one; g, held-out R^2 at the
    floor crossing against N for the rate-penalty and control networks, circles measured and
    crosses estimated, with the rate-penalty size test's fail threshold.

    Args:
        fig: the figure; series: fig_paper_F5_drive.size_series(load_catalogue()); top_mm,
            bottom_mm: the band of the figure (mm from its top edge) the panels occupy; the loss
            axes take the upper 36% of it and the active-units axes the part below 40%.
    Returns:
        None.
    """
    h = bottom_mm - top_mm
    y_l1, y_l0 = y_frac(top_mm), y_frac(top_mm + 0.36 * h)
    y_a1, y_a0 = y_frac(top_mm + 0.40 * h), y_frac(bottom_mm)
    shades = RATE_SHADES                 # the rate violet and two lighter steps (dead-relu-paper#84)
    lefts = (0.075, 0.315, 0.555)
    width, x_d = 0.155, 0.80             # d is wide enough for horizontal size labels (#88)
    first = []
    for j, N in enumerate((500, 1000, 2000)):
        ax_l = fig.add_axes([lefts[j], y_l0, width, y_l1 - y_l0])
        ax_a = fig.add_axes([lefts[j], y_a0, width, y_a1 - y_a0])
        first.append(ax_l)
        ctrl = series["control"][N]["active"]
        ax_a.axhspan(min(ctrl), max(ctrl), color=ps.BASELINE, alpha=0.35, lw=0, zorder=0)
        runs = load_b2(N)
        for r, col in zip(runs, shades):
            it, L = np.asarray(r["it_loss"]), np.asarray(r["loss"])
            nb = len(L) // 250
            Lb = L[:nb * 250].reshape(nb, 250)
            xb = it[:nb * 250].reshape(nb, 250)[:, 0]
            ax_l.plot(xb, np.clip(Lb.max(1), None, 1e3), color=col, lw=0.3, alpha=0.6)
            ax_l.plot(xb, np.median(Lb, 1), color=col, lw=0.8)
            ia, act = np.asarray(r["it_act"]), np.asarray(r["active_series"])
            ax_a.plot(ia, act, color=col, lw=0.4)
            if r["reason"] is None:
                ax_a.plot([r["t_star"]], [N * 1.05], marker="v", ms=3.2, color=col, clip_on=False)
            print(f"rate alone (d-f) N={N} {r['name'][:28]}: {r['reason']}; bursts {len(r['bursts'])};"
                  f" t* {r['t_star']:.0f}; active at t* {r['active']}; end {r['active_end']}")
        ax_l.set_yscale("log")
        ax_l.set_ylim(0.03, 1e4)                 # headroom: the maxima are clipped at 1,000
        ax_l.set_yticks([0.1, 10, 1000])
        ax_l.set_yticklabels(["0.1", "10", "1,000"])
        ax_l.minorticks_off()
        ax_a.set_ylim(0, N * 1.1)
        ax_a.set_yticks([0, N // 2, N])
        ps.count_ticks(ax_a.yaxis)
        for ax in (ax_l, ax_a):
            ax.set_xlim(0, 200000)
            ax.set_xticks([0, 100000, 200000])
            ps.count_ticks(ax.xaxis)
            style_axes(ax)
        plt.setp(ax_l.get_xticklabels(), visible=False)
        ax_a.set_xlabel(ps.LABELS["iteration"])
        ax_l.set_title(f"$N$ = {ps.fmt_count(N)}", fontsize=ps.LABEL_PT, pad=3)
        if j == 0:
            ax_l.set_ylabel("noisy loss")
            ax_a.set_ylabel("active units")
            ax_a.text(197000, min(ctrl) - 12, "control networks", fontsize=ps.TICK_PT,
                      color=ps.INK, ha="right", va="top")

    ax = fig.add_axes([x_d, y_a0, 0.985 - x_d, y_l1 - y_a0])
    for N in SIZES:
        c = series["control"][N]
        cm = np.asarray(c["r2_measured"], bool)          # every control R^2 here is estimated
        cr = np.asarray(c["r2"], float)
        ax.scatter(np.full(cm.sum(), N * 0.94), cr[cm], s=8, marker="o", zorder=3,
                   **ps.mark_floor(ps.BASELINE, scatter=True))
        ax.scatter(np.full((~cm).sum(), N * 0.94), cr[~cm], s=10, marker="x", color=ps.BASELINE,
                   lw=0.7, zorder=3)
        print(f"panel g N={N}: control R2 measured {cm.tolist()}")
        f = series["frm"].get(N)
        if f is not None and len(f["r2"]):
            meas = np.asarray(f["r2_measured"], bool)
            r2 = np.asarray(f["r2"], float)
            ax.scatter(np.full(meas.sum(), N * 1.06), r2[meas], s=10, marker="o", zorder=3,
                       **ps.mark_floor(COL["frm"], scatter=True))
            ax.scatter(np.full((~meas).sum(), N * 1.06), r2[~meas], s=12, marker="x",
                       color=COL["frm"], lw=0.7, zorder=3)
            print(f"panel g N={N}: rate R2 {np.round(r2, 3).tolist()} measured {meas.tolist()};"
                  f" control mean {np.mean(c['r2']):.3f}")
        ax.plot([N * 0.82, N * 1.2], [np.mean(c["r2"]) - FAIL_R2] * 2, color=ps.INK, lw=0.6,
                ls=":")
    ax.set_xscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([ps.fmt_count(n) for n in SIZES])
    ax.minorticks_off()
    ax.set_xlim(380, 5300)
    ax.set_ylim(0.905, 0.975)
    ax.set_yticks([0.91, 0.93, 0.95])
    ax.set_xlabel(ps.LABELS["size"])
    ax.set_ylabel("held-out $R^2$ at the floor crossing")
    style_axes(ax)
    h = [Line2D([], [], color=ps.BASELINE, lw=2.0, label="control"),
         Line2D([], [], color=COL["frm"], lw=2.0, label="rate penalty"),
         Line2D([], [], marker="o", color=ps.INK, ls="none", ms=3, label="measured"),
         Line2D([], [], marker="x", color=ps.INK, ls="none", ms=3.4, mew=0.7, label="estimated"),
         Line2D([], [], color=ps.INK, ls=":", lw=0.6, label="fail threshold")]
    ax.legend(handles=h, loc="upper center", fontsize=ps.TICK_PT, handletextpad=0.3,
              borderaxespad=0.1, labelspacing=0.25, handlelength=1.3, frameon=False, ncol=1)
    for a in first:
        ps.panel_letter(a, "def"[first.index(a)], x_mm=9.0)
    ps.panel_letter(ax, "g", x_mm=11.0)


def panel_margins(fig, series, diag, top_mm, bottom_mm):
    """Panels a-c of the sizes figure: fold change against N at three margins (the previous fig_ed_remedies_size).

    Args:
        fig: the figure; series: size_series(); diag: load_diag(); top_mm, bottom_mm: the band of
            the figure (mm from its top edge) the three axes occupy.
    Returns:
        None.
    """
    gs = fig.add_gridspec(1, 3, left=0.07, right=0.80, top=y_frac(top_mm), bottom=y_frac(bottom_mm), wspace=0.12)
    offs = dict(zip(ED9_FAMS, np.linspace(-0.07, 0.07, len(ED9_FAMS))))
    axes = []
    for j, (key, lab) in enumerate(MARGINS):
        ax = fig.add_subplot(gs[j], sharey=axes[0] if axes else None)
        axes.append(ax)
        for fam in ED9_FAMS:
            xs, ms = [], []
            for N in SIZES:
                c = series[fam].get(N)
                if c is None:
                    continue
                vals, n_runs = margin_counts(c, key, diag)
                base, _ = margin_counts(series["control"][N], key, diag)
                x = N * 10 ** offs[fam]
                if not len(vals):
                    print(f"margins (a-c) {lab}x {fam:9s} N={N}: no crossing (0/{n_runs})")
                    xs.append(x)
                    ms.append(np.nan)        # a size with no crossing breaks the line there
                    continue
                m = vals / base.mean()
                ax.scatter(np.full(len(m), x), m, s=6, marker=MARK[fam], zorder=3,
                           **ps.mark_floor(COL[fam], scatter=True))
                xs.append(x)
                ms.append(m.mean())
                print(f"margins (a-c) {lab}x {fam:9s} N={N}: {len(vals)}/{n_runs} networks, fold "
                      f"{m.mean():.2f}")
            ax.plot(xs, ms, color=COL[fam], lw=ps.DATA_LW, zorder=2)
        ax.axhline(1, color=ps.FAINT, lw=ps.THIN_LW, zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(SIZES)
        ax.set_xticklabels([ps.fmt_count(n) for n in SIZES])
        ax.minorticks_off()
        ax.set_xlim(400, 5000)
        ax.set_title(f"{lab} $\\times$ the loss floor", fontsize=ps.LABEL_PT, pad=3)
        ax.set_xlabel(ps.LABELS["size"])
        style_axes(ax)
        if j:
            plt.setp(ax.get_yticklabels(), visible=False)
        ps.panel_letter(ax, "abc"[j], x_mm=9.0 if j == 0 else 4.0)
    axes[0].set_ylabel("fold change")
    axes[0].set_ylim(0.6, 7.5)
    axes[0].set_yticks([1, 3, 5, 7])
    h = [Line2D([], [], marker=MARK[f], color=COL[f], ls="-", lw=ps.DATA_LW, ms=3.2, label=NAME[f])
         for f in ED9_FAMS]
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.815, (y_frac(top_mm) + y_frac(bottom_mm)) / 2),
               fontsize=ps.TICK_PT,
               frameon=False)


def ed_sizes(series, diag):
    """Draw the Extended Data figure of the techniques across network sizes on 3BFF and save it.

    One figure for what were two until round 17 (dead-relu-paper#190, Reviewer S, structure item
    7): the fold change at three margins (a-c, panel_margins) over the rate penalty alone (d-g,
    panel_rate_alone). Nothing is recomputed; every number drawn is printed by the two panel
    functions.

    Args:
        series: size_series(); diag: load_diag().
    Returns:
        the path of the PDF written by paperstyle.save.
    """
    fig = ps.figure(h_mm=SIZES_H_MM)
    panel_margins(fig, series, diag, 6.0, 42.0)
    panel_rate_alone(fig, series, 62.0, 118.0)
    return ps.save(fig, "fig_ed_techniques_sizes")


# --------------------------------------------------------------------------------------------
# Supplementary Table 2
# --------------------------------------------------------------------------------------------

def tex_count(n):
    """An integer count as LaTeX with a thousands separator: 1854 -> '1{,}854'."""
    return ps.fmt_count(int(n)).replace(",", "{,}")


def tex_range(v):
    """'1{,}854--1{,}893' for a list of counts, one value when there is one, '--' when empty."""
    if not len(v):
        return "--"
    lo, hi = int(min(v)), int(max(v))
    return tex_count(lo) if lo == hi else f"{tex_count(lo)}--{tex_count(hi)}"


def fmt_mult(d):
    """A cell's mean fold change with 2 decimals, '--' when it has no value."""
    return f"{d['mult'].mean():.2f}" if len(d["mult"]) else "--"


def tex_num(v, sign=False):
    """A float as LaTeX with a true minus sign: 3 decimals, '+' kept when sign is True."""
    s = f"{v:+.3f}" if sign else f"{v:.3f}"
    return s.replace("-", "$-$")


def fmt_dr2(d):
    """Mean R^2 change from the control networks over the networks with R^2 > 0, 3 decimals.

    A network scored on weights taken inside a loss burst can give R^2 far below zero (down to
    -14,336 here); such a network is left out of the mean and marked with a dagger.

    Args:
        d: per_run() dict.
    Returns:
        LaTeX string, '--' when no network has R^2 > 0.
    """
    ok = np.isfinite(d["r2"]) & (d["r2"] > 0) if len(d["r2"]) else np.zeros(0, bool)
    if not ok.any():
        return "--"
    return tex_num(d["dr2"][ok].mean(), sign=True) + ("$^\\dagger$" if (~ok).any() else "")


def removal_medians(a3, fam, N):
    """Median over networks of R^2 after removing the k added units and k random active units.

    Args:
        a3: load_a3(); fam: family key; N: size.
    Returns:
        (added, random) medians as strings with 2 decimals, '--' when the cell is absent.
    """
    nets = [r for r in a3.get(N, []) if r["family"] == fam]
    if not nets:
        return "--", "--"
    added = np.median([r["r2_recruited"] for r in nets])
    rnd = np.median([np.median(r["r2_random_active"]) for r in nets])
    return f"{added:.2f}".replace("-", "$-$"), f"{rnd:.2f}".replace("-", "$-$")


def a3_dr2(a3, fam, N):
    """R^2 change at 75,000 iterations from the removal analysis's intact scores (9 noise draws).

    For the rate penalty, which has no 36-draw score at 75,000 iterations (dead-relu-paper#92): the
    mean intact R^2 of the rule's networks minus the mean intact R^2 of the control networks that
    the same analysis scored, over networks with R^2 > 0 (as fmt_dr2).

    Args:
        a3: load_a3(); fam: family key; N: size.
    Returns:
        LaTeX string marked with a pilcrow (9 draws), '--' when absent.
    """
    rule = np.asarray([r["r2_full"] for r in a3.get(N, []) if r["family"] == fam], float)
    ctrl = [r["r2_full"] for r in a3.get(N, []) if r["family"] == "control"]
    ok = np.isfinite(rule) & (rule > 0)
    if not ok.any() or not ctrl:
        return "--"
    print(f"% a3 R2 at 75k, {fam} N={N}: {np.round(rule, 3).tolist()} against control mean "
          f"{np.mean(ctrl):.4f}")
    return (tex_num(rule[ok].mean() - np.mean(ctrl), sign=True) + "$^\\P$"
            + ("$^\\dagger$" if (~ok).any() else ""))


def table_supp2(cat, series, bursts, a3, marks, where75):
    """Print the LaTeX rows of Supplementary Table 2: a block per rule, a row per size.

    Args:
        cat: load_catalogue(); series: size_series(cat); bursts: load_bursts(); a3: load_a3();
        marks: load_marks(); where75: load_where_75k().
    Returns:
        None.
    """
    print("\n% control networks: active at the floor crossing (mean), loss bursts per network, by N:")
    for N in SIZES:
        c = series["control"][N]
        print(f"%   N={N}: active mean {np.mean(c['active']):.0f}; bursts "
              f"{fmt_range(bursts.get(c['cell'], []))} ({len(bursts.get(c['cell'], []))} networks);"
              f" R2 mean {np.mean(c['r2']):.3f}")
    print("\n% Supplementary Table 2 rows (fig_ed_remedies.py)")
    n_marked = 0
    for fam in ED9_FAMS:
        print(f"\\multicolumn{{11}}{{l}}{{\\emph{{{TABLE_NAME[fam]}}}}} \\\\")
        for N in SIZES:
            c = series[fam].get(N)
            if c is None:
                continue
            fl, fx = per_run(series, fam, N, "floor", marks), per_run(series, fam, N, "75k", marks)
            act = tex_range(c["active"])
            meas = f"{sum(c['r2_measured'])}/{len(c['r2_measured'])}" if c["r2_measured"] else "--"
            b = bursts.get(c["cell"], [])
            assert len(b) == c["n_runs"], f"burst counts for {len(b)} of {c['n_runs']} runs: {c['cell']}"
            assert not fl["burst"].any(), f"a floor-crossing R2 inside a burst: {c['cell']}"
            w = where75.get(c["cell"], {})
            n_in = sum(v != "calm" for v in w.values())
            nb75 = f"{n_in}/{len(w)}" if w else "--"
            n_marked += n_in
            added, rnd = removal_medians(a3, fam, N)
            print(f"{tex_count(N)} & {c['n_read']}/{c['n_runs']} & "
                  f"{fmt_range(b)} & {act} & {fmt_mult(fl)} & {fmt_dr2(fl)} & {meas} & {fmt_mult(fx)} & "
                  f"{fmt_dr2(fx) if len(fx['r2']) else a3_dr2(a3, fam, N)} & {nb75} & "
                  f"{added} / {rnd} \\\\")
    print(f"% networks whose 75,000-iteration weights lie inside a burst, in these rows: {n_marked}")
    for c in cat:
        if c["family"] == "redraw":
            name = ("Revival method: incoming weights redrawn, outgoing kept" if "mode=random" in
                    c["cell"] else "Revival method: incoming weights redrawn, outgoing zeroed")
            print(f"\\multicolumn{{11}}{{l}}{{\\emph{{{name}}}}} \\\\")
            ctrl = series["control"][1000]
            mult = np.mean(c["active"]) / np.mean(ctrl["active"])
            dr2 = np.mean(c["r2"]) - np.mean(ctrl["r2"])
            print(f"% {c['cell']}")
            print(f"1{{,}}000 & {c['n_read']}/{c['n_runs']} & "
                  f"{fmt_range(bursts.get(c['cell'], []))} & {tex_range(c['active'])} & "
                  f"{mult:.2f} & {tex_num(dr2, sign=True)} & "
                  f"{sum(c['r2_measured'])}/{len(c['r2_measured'])} & -- & -- & -- & -- \\\\")


if __name__ == "__main__":
    ps.setup()
    cat = load_catalogue()
    series = size_series(cat)
    ed_sizes(series, load_diag())
    table_supp2(cat, series, load_bursts(), load_a3(), load_marks(), load_where_75k())
