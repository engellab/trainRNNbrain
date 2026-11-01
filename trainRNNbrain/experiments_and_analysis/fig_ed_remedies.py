#!/usr/bin/env python3
"""
Extended Data for Figure 5: the rate penalty alone (ED Fig. 8), the remedies across sizes at three
floor margins (ED Fig. 9), and the rows of the remedies catalogue (ED Table 3, printed as LaTeX).

  ED Fig. 8   B2, the rate penalty alone (lambda_frm 0.1) on the 3-bit flip-flop.
              a-c, N = 500, 1000, 2000: the noisy training loss (per-250-iteration maximum and
              median) and the active count at every participation probe, every run, with each
              run's own 1.05x floor crossing where one exists (common.read_run).
              d, held-out R^2 at the floor against N, rate penalty and control, with B2's bar
              (the control's mean minus 0.02).
  ED Fig. 9   The multiplier against N for every size-series remedy, read at 1.05x, 1.10x and
              1.20x each run's own fitted floor. A run with no crossing at 1.05x because its loss
              is still falling at 200k ("never stays under the bar") is read at a looser margin
              when it has a crossing there (rm_exclusion_diag.py output); runs excluded for any
              other reason are excluded at every margin. Multiplier = the run's active count over
              the control's mean active count at the same N and margin.
  ED Table 3  printed rows: remedy, N, runs read / found, loss bursts per run (range over every run
              trained, common.read_run's count as stored by read_remedies.py), active at the
              floor, multiplier and R^2 difference at the floor and at 75k, R^2 measured /
              estimated.

Data and provenance: as fig_paper_F5_drive.py (data/remedy_reads_31oct/, the B2 runs under
data/trained_RNNs/NBitFlipFlop_remedy_grid/), plus the per-run records of the same RMtop200 read
(Della ~/rm_read/runs_*.json, copied to data/remedy_reads_31oct/rm_read_runs/), which hold each
run's burst count.

Usage:  python fig_ed_remedies.py
Output: img/internal_figures/fig_ed_rate_alone.pdf, fig_ed_remedies_size.pdf (+ .svg), and the
        LaTeX rows of ED Table 3 on stdout.
"""

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from fig_paper_F5_drive import (COL, MARK, NAME, READS, SIZES, load_b2, load_catalogue,
                                per_run, size_series)

ED9_FAMS = ("frm", "synnoise", "rescale", "duplicate", "dropout", "frm+rws")
MARGINS = (("active", "1.05"), ("active_d10", "1.10"), ("active_d20", "1.20"))
RULE_BAR = 0.02          # B2: R^2 within 0.02 of the control's at the floor


def load_diag():
    """The rm_exclusion_diag.py records for the runs that never stay under the 1.05x bar.

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
    """Loss-burst counts of every run in the RMtop200 read (read_remedies.py per-run records).

    A burst is common.read_run's: a stretch where the 51-iteration running median of the noisy loss
    exceeds twice a wide reference curve, counted from the fit start (Methods, read-out).

    Returns:
        dict cell path -> list of int burst counts, one per run that has one (a diverged run, whose
        folder is named nan, has none).
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(READS, "rm_read_runs", "runs_*.json"))):
        with open(f) as fh:
            for r in json.load(fh):
                if "n_bursts" in r:
                    out.setdefault(r["cell"], []).append(int(r["n_bursts"]))
    return out


def fmt_range(v):
    """'57--60' for a list of ints, '57' when min equals max, '--' when empty."""
    if not len(v):
        return "--"
    lo, hi = int(min(v)), int(max(v))
    return str(lo) if lo == hi else f"{lo}--{hi}"


def margin_counts(cell, key, diag):
    """Every run of a cell with a crossing at one margin, and its active count there.

    Args:
        cell: catalogue cell dict; key: "active", "active_d10" or "active_d20"; diag: load_diag().
    Returns:
        (counts array, number of runs found in the cell).
    """
    vals = list(cell[key])
    if key != "active":
        vals += [r[key] for r in diag.get(cell["cell"], []) if r[key] is not None]
    return np.asarray(vals, float), cell["n_runs"]


# --------------------------------------------------------------------------------------------
# ED Fig. 8
# --------------------------------------------------------------------------------------------

def ed_rate_alone(series):
    """Draw Extended Data Fig. 8 (B2, the rate penalty alone).

    Args:
        series: fig_paper_F5_drive.size_series(load_catalogue()).
    Returns:
        the path of the PDF.
    """
    fig = plt.figure(figsize=(180 * ps.MM, 92 * ps.MM))
    gs = GridSpec(1, 4, figure=fig, left=0.065, right=0.99, top=0.93, bottom=0.13, wspace=0.42,
                  width_ratios=[1, 1, 1, 0.8])
    shades = [COL["frm"], "#a8341b", "#f08a6a"]
    first = []
    for j, N in enumerate((500, 1000, 2000)):
        g = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[j], hspace=0.10, height_ratios=[1, 1.4])
        ax_l, ax_a = fig.add_subplot(g[0]), fig.add_subplot(g[1])
        first.append(ax_l)
        ctrl = series["control"][N]["active"]
        ax_a.axhspan(min(ctrl), max(ctrl), color=ps.BASE, alpha=0.25, lw=0, zorder=0)
        runs = load_b2(N)
        for r, col in zip(runs, shades):
            it, L = np.asarray(r["it_loss"]), np.asarray(r["loss"])
            nb = len(L) // 250
            Lb = L[:nb * 250].reshape(nb, 250)
            xb = it[:nb * 250].reshape(nb, 250)[:, 0] / 1000
            ax_l.plot(xb, np.clip(Lb.max(1), None, 1e3), color=col, lw=0.3, alpha=0.55)
            ax_l.plot(xb, np.median(Lb, 1), color=col, lw=0.8)
            ia, act = np.asarray(r["it_act"]), np.asarray(r["active_series"])
            ax_a.plot(ia / 1000, act, color=col, lw=0.4)
            if r["reason"] is None:
                t = r["t_star"] / 1000
                ax_a.plot([t], [N * 1.035], marker="v", ms=3.5, color=col, clip_on=False)
            print(f"ED8 N={N} {r['name'][:28]}: {r['reason']}; bursts {len(r['bursts'])};"
                  f" t* {r['t_star']:.0f}; active at t* {r['active']}; end {r['active_end']}")
        ax_l.set_yscale("log")
        ax_l.set_ylim(0.03, 1e3)
        ax_l.set_yticks([0.1, 10, 1000])
        ax_a.set_ylim(0, N * 1.02)
        for ax in (ax_l, ax_a):
            ax.set_xlim(0, 200)
        plt.setp(ax_l.get_xticklabels(), visible=False)
        ax_a.set_xlabel("Iteration (thousands)")
        n_read = sum(r["reason"] is None for r in runs)
        ax_l.set_title(f"$N$ = {N}: floor read in {n_read} of {len(runs)} runs", fontsize=6.8,
                       pad=3)
        if j == 0:
            ax_l.set_ylabel("Noisy loss")
            ax_a.set_ylabel("Active units")
            ax_a.text(198, max(ctrl) + 8, "control at\nits floor", fontsize=5.6, color=ps.MUTED,
                      ha="right", va="bottom")

    ax = fig.add_subplot(gs[3])
    for N in SIZES:
        c = series["control"][N]
        ax.scatter(np.full(len(c["r2"]), N * 0.96), c["r2"], s=9, color=ps.BASE, zorder=3)
        f = series["frm"].get(N)
        if f is not None and len(f["r2"]):
            meas = np.asarray(f["r2_measured"], bool)
            r2 = np.asarray(f["r2"], float)
            ax.scatter(np.full(meas.sum(), N * 1.04), r2[meas], s=11, color=COL["frm"], zorder=3)
            ax.scatter(np.full((~meas).sum(), N * 1.04), r2[~meas], s=11, facecolor="none",
                       edgecolor=COL["frm"], lw=0.7, zorder=3)
        ax.plot([N * 0.85, N * 1.15], [np.mean(c["r2"]) - RULE_BAR] * 2, color=ps.MUTED, lw=0.7,
                ls=":")
    ax.set_xscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(n) for n in SIZES], fontsize=6)
    ax.minorticks_off()
    ax.set_xlim(400, 5000)
    ax.set_ylim(0.905, 0.990)    # headroom above the data for the legend, inside the axes
    ax.set_xlabel("Network size $N$")
    ax.set_ylabel("Held-out $R^2$ at the floor")
    ax.text(450, 0.9075, "no floor read\nat $N \\leq$ 1000", fontsize=5.6, color=COL["frm"])
    h = [Line2D([], [], marker="o", color=ps.BASE, ls="none", ms=3, label="control"),
         Line2D([], [], marker="o", color=COL["frm"], ls="none", ms=3, label="rate, measured"),
         Line2D([], [], marker="o", mfc="none", color=COL["frm"], ls="none", ms=3,
                label="rate, estimated"),
         Line2D([], [], color=ps.MUTED, ls=":", lw=0.7, label="control $-$ 0.02")]
    ax.set_yticks([0.91, 0.93, 0.95, 0.97])
    ax.legend(handles=h, loc="upper right", fontsize=5.4, handletextpad=0.3, borderaxespad=0.15,
              labelspacing=0.2, handlelength=1.2)
    for a, letter in zip(first + [ax], "abcd"):
        ps.panel_letter(a, letter, dx=-0.30 if a is not ax else -0.40, dy=1.10 if a is not ax else 1.02)
    return ps.save(fig, "fig_ed_rate_alone", w_mm=180)


# --------------------------------------------------------------------------------------------
# ED Fig. 9
# --------------------------------------------------------------------------------------------

def ed_margins(series, diag):
    """Draw Extended Data Fig. 9 (multiplier against N at three floor margins).

    Args:
        series: size_series(); diag: load_diag().
    Returns:
        the path of the PDF.
    """
    fig = plt.figure(figsize=(180 * ps.MM, 62 * ps.MM))
    gs = GridSpec(1, 3, figure=fig, left=0.06, right=0.80, top=0.90, bottom=0.18, wspace=0.18)
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
                if not len(vals):
                    print(f"ED9 {lab}x {fam:9s} N={N}: no read (0/{n_runs})")
                    xs.append(N * 10 ** offs[fam])
                    ms.append(np.nan)        # a size with no read breaks the line there
                    continue
                m = vals / base.mean()
                x = N * 10 ** offs[fam]
                ax.scatter(np.full(len(m), x), m, s=7, marker=MARK[fam], color=COL[fam], lw=0.5,
                           alpha=0.75, zorder=3)
                xs.append(x)
                ms.append(m.mean())
                print(f"ED9 {lab}x {fam:9s} N={N}: {len(vals)}/{n_runs} runs, mult {m.mean():.2f}")
            ax.plot(xs, ms, color=COL[fam], lw=1.0, zorder=2)
        ax.axhline(1, color=ps.FAINT, lw=0.6, zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(SIZES)
        ax.set_xticklabels([str(n) for n in SIZES])
        ax.minorticks_off()
        ax.set_xlim(400, 5000)
        ax.set_title(f"{lab}$\\times$ the fitted floor", fontsize=7, pad=3)
        ax.set_xlabel("Network size $N$")
        if j:
            plt.setp(ax.get_yticklabels(), visible=False)
        ps.panel_letter(ax, "abc"[j], dx=-0.12 if j else -0.20, dy=1.04)
    axes[0].set_ylabel("Multiplier (active / control's)")
    axes[0].set_ylim(0.8, 7.5)
    h = [Line2D([], [], marker=MARK[f], color=COL[f], ls="-", lw=1.0, ms=3.5, label=NAME[f])
         for f in ED9_FAMS]
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.81, 0.55), fontsize=6.3,
               frameon=False)
    return ps.save(fig, "fig_ed_remedies_size", w_mm=180)


# --------------------------------------------------------------------------------------------
# ED Table 3
# --------------------------------------------------------------------------------------------

def fmt_mult(d):
    """'2.03' for a cell's mean multiplier, '--' when it has no read."""
    return f"{d['mult'].mean():.2f}" if len(d["mult"]) else "--"


def fmt_dr2(d):
    """Mean R^2 difference from the control over the runs with R^2 > 0, 3 decimals.

    A network scored on weights taken inside a loss burst can give R^2 far below zero (down to
    -14,336 here); such a run is left out of the mean and marked with a dagger.

    Args:
        d: per_run() dict.
    Returns:
        LaTeX string, '--' when no run has R^2 > 0.
    """
    ok = np.isfinite(d["r2"]) & (d["r2"] > 0) if len(d["r2"]) else np.zeros(0, bool)
    if not ok.any():
        return "--"
    s = f"{d['dr2'][ok].mean():+.3f}".replace("-", "$-$")
    return s + ("$^\\dagger$" if (~ok).any() else "")


def table_ed3(cat, series, bursts):
    """Print the LaTeX rows of ED Table 3 (one row per remedy and size, plus the redraw cells).

    Args:
        cat: load_catalogue(); series: size_series(cat); bursts: load_bursts().
    Returns:
        None.
    """
    print("\n% control loss bursts per run, by N (ED Table 3 caption):")
    for N in SIZES:
        c = series["control"][N]
        print(f"%   N={N}: {fmt_range(bursts.get(c['cell'], []))} ({len(bursts.get(c['cell'], []))} runs)")
    print("\n% ED Table 3 rows (fig_ed_remedies.py)")
    for fam in ED9_FAMS:
        for N in SIZES:
            c = series[fam].get(N)
            if c is None:
                continue
            fl, fx = per_run(series, fam, N, "floor"), per_run(series, fam, N, "75k")
            act = (f"{int(min(c['active']))}--{int(max(c['active']))}" if len(c["active"]) > 1
                   else (str(int(c["active"][0])) if len(c["active"]) else "--"))
            meas = f"{sum(c['r2_measured'])}/{len(c['r2_measured'])}" if c["r2_measured"] else "--"
            b = bursts.get(c["cell"], [])
            assert len(b) == c["n_runs"], f"burst counts for {len(b)} of {c['n_runs']} runs: {c['cell']}"
            print(f"{NAME[fam]} & {N} & {c['n_read']}/{c['n_runs']} & {fmt_range(b)} & {act} & {fmt_mult(fl)} & "
                  f"{fmt_dr2(fl)} & {meas} & {fmt_mult(fx)} & {fmt_dr2(fx)} \\\\")
    for c in cat:
        if c["family"] == "redraw":
            mode = "random redraw" if "mode=random" in c["cell"] else "zero-out redraw"
            ctrl = series["control"][1000]
            mult = np.mean(c["active"]) / np.mean(ctrl["active"])
            dr2 = np.mean(c["r2"]) - np.mean(ctrl["r2"])
            print(f"Redraw inactive units ({mode}) & 1000 & {c['n_read']}/{c['n_runs']} & "
                  f"{fmt_range(bursts.get(c['cell'], []))} & {min(c['active'])}--{max(c['active'])} & {mult:.2f} & {dr2:+.3f} & "
                  f"{sum(c['r2_measured'])}/{len(c['r2_measured'])} & -- & -- \\\\")


if __name__ == "__main__":
    ps.setup()
    cat = load_catalogue()
    series = size_series(cat)
    ed_rate_alone(series)
    ed_margins(series, load_diag())
    table_ed3(cat, series, load_bursts())
