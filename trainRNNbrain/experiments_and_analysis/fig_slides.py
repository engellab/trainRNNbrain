#!/usr/bin/env python3
"""
Presentation figures: one claim per figure, one panel per figure, no composites.

A manuscript figure packs six panels onto a page because the reader controls the pace. A talk is the
opposite: the speaker controls the pace, so a slide carrying six panels is six slides the audience
cannot be pointed at one at a time. This script re-emits the manuscript panels as standalone figures
and builds one figure per failed intervention, so `docs/presentation.md` can show them in order.

NOTHING HERE RE-ANALYSES ANYTHING. The panels are the manuscript's own panel functions called on a
one-axes figure, and the intervention figures read their cells from `fig_paper_F1`'s own family
tables through its own loaders. If a number here disagrees with the paper, that is a bug in this
file, not a second opinion - which is the point: a talk that quotes different numbers from the paper
it is about has two problems.

THE INTERVENTION FIGURES are the expansion of Figure 1d. That panel plots every knob we varied as a
change from its own reference, which is the right summary and the wrong slide: it asks an audience
to read twenty intervals at once. Here each family gets its own figure, drawn as COUNTS rather than
changes so that the reference is visible on the same axis, with every seed shown.

Usage:  python fig_slides.py            # writes every slide
        python fig_slides.py --list     # names only
Output: img/internal_figures/slide_*.pdf (+ .svg, which is what the Markdown deck links)
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
import fig_paper_F1 as F1
import fig_paper_F2 as F2
import drift_matrix as DM

W = 110 * ps.MM             # a single-panel slide: wider than a journal column, same 7 pt type
H = 72 * ps.MM


def _family_values(family, trace=True):
    """Active-unit counts for a family's reference and each of its items.

    Args:
        family: an entry of F1.TRACE_FAMILIES, or F1.ARCHIVE_FAMILY / F1.NOISE_FAMILY;
        trace: True for the trace families (glob patterns), False for the CSV/noise families.
    Returns:
        (reference array, [(label, values array, group), ...]) with empty arrays where a cell is
        missing, so a slide shows the gap rather than silently dropping a condition.
    """
    if trace:
        title, ref_pat, cap, items = family
        ref = (F1.live_matched(ref_pat, cap) or (np.array([]),))[0]
        out = []
        for lab, pat, grp in items:
            v = np.array([]) if pat is None else (F1.live_matched(pat, cap) or (np.array([]),))[0]
            out.append((lab, np.asarray(v, float), grp))
        return np.asarray(ref, float), out
    title, ref_spec, items = family
    if isinstance(ref_spec, tuple):
        ref = F1.csv_active(ref_spec[0], **ref_spec[1])
        out = [(lab, np.asarray(F1.csv_active(s[0], **s[1]), float), grp) for lab, s, grp in items]
    else:
        # ⚠️ THE NOISE SWEEP SAVED NO PER-NETWORK ROWS, only a per-condition mean, sd and n, and it
        # uses a peak-rate silence rule rather than participation. Its slide therefore draws mean
        # and a 95% interval where the others draw every seed, and says so on the figure.
        ref = F1.noise_active(ref_spec)
        out = [(lab, (float("nan"), float("nan"), 0) if s is None else F1.noise_active(s), grp)
               for lab, s, grp in items]
    return ref, out


def collect():
    """Every intervention in Figure 1d, keyed by (family title, group).

    Returns:
        dict {(family title, group): (reference array, [(label, values), ...])}.
    """
    out = {}
    fams = [(f, True) for f in F1.TRACE_FAMILIES]
    fams += [(F1.ARCHIVE_FAMILY, False), (F1.NOISE_FAMILY, False)]
    for fam, trace in fams:
        title = fam[0]
        ref, items = _family_values(fam, trace)
        for lab, vals, grp in items:
            if grp == "reference":
                continue
            out.setdefault((title, grp), (ref, []))[1].append((lab, vals))
    return out


def summary_slide(name, title, ref_label, entries, ref, col, note=None, ref_at=0):
    """A family that saved only per-condition summaries: mean and a 95% interval, no seeds.

    Args:
        name: output stem; title: the claim; ref_label: the reference rung's name;
        entries: [(label, (mean, sd, n)), ...]; ref: the reference's (mean, sd, n);
        col: the family's colour; note: a second title line.
    Returns:
        the output path.
    """
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, H))
    rows = [(lab, v, col) for lab, v in entries]
    rows.insert(ref_at, (ref_label, ref, ps.BASE))
    xs = np.arange(len(rows), dtype=float)
    for x, (lab, (m, sd, n), c) in zip(xs, rows):
        if not n:
            continue
        ax.errorbar(x, m, yerr=1.96 * sd / np.sqrt(n), fmt="o", ms=4.0, color=c, mec="white",
                    mew=0.6, lw=1.1, capsize=2.0, zorder=4)
        ax.annotate(f"{m:.0f}", (x, m + 1.96 * sd / np.sqrt(n)), textcoords="offset points",
                    xytext=(0, 5), ha="center", va="bottom", fontsize=6.2, color=ps.INK)
    ax.axhline(ref[0], color=ps.BASE, lw=0.7, ls=":", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=6.4, rotation=22, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylabel("active units")
    ax.set_title(title + ("\n" + note if note else ""), fontsize=7.4, color=ps.INK,
                 linespacing=1.35, pad=6)
    ps.ygrid(ax)
    return ps.save(fig, name)


def dose_slide(name, title, ref_label, entries, ref, col, ylabel="active units", note=None,
               ref_at=0):
    """One intervention family as counts: the reference, then each setting, every seed drawn.

    Counts rather than changes, because a slide has no room for a reader to hold a reference in
    their head - it has to be on the axis.

    Args:
        name: output file stem; title: the claim, drawn above the axes;
        ref_label: what the reference rung is called; entries: [(label, values array), ...];
        ref: the reference's values; col: the family's colour; ylabel: y axis label;
        note: an optional second line under the title; ref_at: where the reference sits among the
        entries - on a dose ladder it is a rung, not a preamble, and putting it first makes the
        ladder read out of order.
    Returns:
        the output path.
    """
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, H))
    labels = [lab for lab, _ in entries]
    groups = [np.asarray(v, float) for _, v in entries]
    cols = [col] * len(entries)
    labels.insert(ref_at, ref_label)
    groups.insert(ref_at, np.asarray(ref, float))
    cols.insert(ref_at, ps.BASE)
    xs = np.arange(len(groups), dtype=float)
    res = ps.strip(ax, xs, groups, cols, rng=np.random.default_rng(0))
    if len(ref):
        ax.axhline(np.mean(ref), color=ps.BASE, lw=0.7, ls=":", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.4, rotation=22, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(ylabel)
    drawn = [g for g in groups if len(g)]
    if drawn:
        lo = min(g.min() for g in drawn)
        hi = max(g.max() for g in drawn)
        pad = max(0.08 * (hi - lo), 5.0)
        ax.set_ylim(max(0, lo - pad), hi + pad * 1.5)
    for x, g, (m, sd, n) in zip(xs, groups, res):
        if not n:
            ax.text(x, ax.get_ylim()[0] + 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    "no data", ha="center", fontsize=5.6, color=ps.FAINT, rotation=90)
            continue
        ax.annotate(f"{m:.0f}", (x, g.max()), textcoords="offset points", xytext=(0, 5),
                    ha="center", va="bottom", fontsize=6.2, color=ps.INK)
    ax.set_title(title + ("\n" + note if note else ""), fontsize=7.4, color=ps.INK,
                 linespacing=1.35, pad=6)
    ps.ygrid(ax)
    return ps.save(fig, name)


def panel_slide(name, fn, width=W, height=H, **kw):
    """One manuscript panel on its own figure.

    Args:
        name: output stem; fn: a callable taking the axes (and anything in kw);
        width, height: figure size in inches; kw: passed to fn.
    Returns:
        the output path.
    """
    ps.setup()
    fig, ax = plt.subplots(figsize=(width, height))
    fn(ax, **kw)
    return ps.save(fig, name)


# (slide stem, title, family title, group, reference label, reference rung, note)
INTERVENTIONS = [
    ("slide_x_activation_cddm", "A different activation does not help",
     "CDDM, 200k", "activation", "ReLU (default)", 0,
     "CDDM, N = 1000, read at 200k. Every seed drawn."),
    ("slide_x_weightdecay", "Weight decay makes it monotonically worse",
     "CDDM, 200k", "weight decay", "W.D. 10$^{-6}$ (default)", 1,
     "CDDM, N = 1000. The default is a rung of this ladder, not a separate condition."),
    ("slide_x_activation_ff", "Nor on the other task",
     "3-bit flip-flop, 150k", "activation", "ReLU (default)", 0,
     "3-bit flip-flop, N = 1000, read at 150k."),
    ("slide_x_inputscale", "Scaling the input weights does not help either",
     "3-bit flip-flop, 150k", "input scale", "input w. $\\times$1 (default)", 1,
     "3-bit flip-flop, N = 1000."),
    ("slide_x_metabolic", "The field-standard metabolic penalty moves nothing beyond seed scatter",
     "CDDM, 30k (archived)", "metabolic", "$\\lambda$ = 0 (default)", 0,
     "CDDM, N = 1000, four decades of $\\lambda$."),
    ("slide_x_architecture", "Nor does the equation form, nor a trainable bias",
     "CDDM, 30k (archived)", "architecture", "standard", 0,
     "CDDM, N = 1000."),
    ("slide_x_recnoise", "Removing recurrent noise is the largest effect we found — and it is negative",
     "CDDM, 30k (peak-rate criterion)", "noise", "$\\sigma$ = 0.05 (default)", 2,
     "CDDM, N = 1000, peak-rate criterion. Mean and 95% interval: this sweep saved no per-seed rows."),
]


def drift_slides():
    """One heatmap per weight matrix, unpenalised networks only: does training ever settle?

    `drift_matrix.py` draws three weight matrices against four penalty conditions on one sheet,
    which is the right audit and the wrong slide - twelve heatmaps cannot be pointed at one at a
    time. These are the `none` column, one matrix per figure, read at each run's own end.

    alpha is the exponent in |W(t+L) - W(t)| / |W(t)| ~ L^alpha: 1.0 ballistic (still travelling),
    0.5 diffusive (jittering in place), below 0.5 confined (mean-reverting in a basin).

    Returns:
        list of output paths.
    """
    data, buds, _dropped = DM.load()
    ks = sorted({k for _, k, _ in data})
    Ns = sorted({N for _, _, N in data})
    out = []
    for var in DM.VARS:
        Z, S, C = DM.grid(data, "none", var, ks, Ns, None)
        ps.setup()
        fig, ax = plt.subplots(figsize=(88 * ps.MM, 64 * ps.MM))
        # the scale spans the whole regime range, 0 to 1: clipped at 0.3 every cell fell into the
        # bottom colour, hiding that these networks sit at the SETTLED end of it
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        for i in range(len(Ns)):
            for j in range(len(ks)):
                if np.isfinite(Z[i, j]):
                    ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=6.0,
                            color="white" if Z[i, j] < 0.72 else "#111111")
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_yticks(range(len(Ns)), [str(n) for n in Ns])
        ax.set(xlabel="task complexity $k$", ylabel="network size $N$")
        # the verdict is computed from this figure's own median, not asserted: an earlier version
        # titled these "the weights never stop moving", which every cell on them contradicted
        med = float(np.nanmedian(Z))
        verdict = ("still travelling" if med > 0.8 else
                   "jittering in place" if med > 0.45 else "settled, mean-reverting")
        ax.set_title(f"{var}: {verdict}" + r" (median $\alpha$ = " + f"{med:.2f})\n"
                     r"$\alpha$ in $|W(t{+}L)-W(t)|/|W(t)| \sim L^{\alpha}$: "
                     r"1.0 travelling, 0.5 jittering, below 0.5 confined",
                     fontsize=7.2, color=ps.INK, linespacing=1.35, pad=6)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.ax.tick_params(labelsize=6.0)
        cb.set_label(r"$\alpha$", fontsize=7)
        out.append(ps.save(fig, f"slide_04_drift_{var}"))
    return out


def main(list_only=False):
    """Write every slide. Returns the list of output paths."""
    if list_only:
        for stem, title, *_ in INTERVENTIONS:
            print(f"  {stem:28s} {title}")
        return []
    out = []

    # ---- the manuscript panels, one per figure ------------------------------------------------
    everything = F2.load()
    at_main = F2.restrict(everything, n_units=F2.N_MAIN)
    at_1000 = F2.restrict(everything, n_units=F2.N_MAIN,
                          chosen={a: F2.pick_cell(at_main, a)[0] for a in F2.GRID_ARMS})
    # ---- Figure 1's panels, in the order the talk needs them ----------------------------------
    rates, _, pvec = F1.example_network()
    fig = plt.figure(figsize=(W, 78 * ps.MM))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.72], hspace=0.42)
    ax_net, ax_tr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    F1.panel_a(ax_net, ax_tr, rates, pvec)
    out.append(ps.save(fig, "slide_01_schematic"))

    out.append(panel_slide("slide_02_participation", F1.panel_b, p=pvec))
    out.append(panel_slide("slide_06_scaling", F1.panel_c, width=W, height=78 * ps.MM))

    # panel (e) is one sub-panel per task, stacked; DMTS is kept even though it breaks the pattern
    fig = plt.figure(figsize=(W, 96 * ps.MM))
    gs = GridSpec(len(F1.TRAJ), 1, figure=fig, hspace=0.30)
    F1.panel_e([fig.add_subplot(gs[i]) for i in range(len(F1.TRAJ))])
    out.append(ps.save(fig, "slide_03_silencing_vs_training"))

    out += drift_slides()

    out.append(panel_slide("slide_rules", F2.panel_a, width=ps.W2, height=52 * ps.MM))
    out.append(panel_slide("slide_f2_active", F2.panel_b, c=at_1000))
    out.append(panel_slide("slide_f2_r2", F2.panel_c, c=at_1000))
    out.append(panel_slide("slide_f2_dims", F2.panel_d, c=at_1000))
    out.append(panel_slide("slide_f2_weights", F2.panel_e, c=at_1000))
    out.append(panel_slide("slide_f2_size_active", F2.panel_f, c=everything))
    out.append(panel_slide("slide_f2_size_r2", F2.panel_g, c=everything))

    # ---- one figure per failed intervention ---------------------------------------------------
    fams = collect()
    for stem, title, fam_title, group, ref_label, ref_at, note in INTERVENTIONS:
        got = fams.get((fam_title, group))
        if got is None:
            print(f"  SKIP {stem}: no cells for ({fam_title}, {group})")
            continue
        ref, entries = got
        draw = summary_slide if group == "noise" else dose_slide
        out.append(draw(stem, title, ref_label, entries, ref,
                        F1.GROUP_COL.get(group, ps.MUTED), note=note, ref_at=ref_at))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="name the slides without drawing them")
    args = ap.parse_args()
    main(list_only=args.list)
