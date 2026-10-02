#!/usr/bin/env python3
"""Slides 24 and 24b as ONE panel: every dropout run, performance against the units it bought.

The two slides it replaces each spend a whole figure on one measure against the drop rate, so the
reader has to hold 24 cell means in their head to see whether the units a setting buys are paid for.
Putting active units on x and performance on y answers that directly, and the sweep's three knobs
then ride on the marker instead of on a second panel:

    x            active units of 1000, the scale-free criterion, at end of training
    y            R^2 from the trainer's noise-free, dropout-OFF probe, so every arm - dropout and
                 control alike - is scored on the full network
    shape        targeting exponent beta: circle 1, diamond 2, filled plus 4
    fill colour  drop rate rho on viridis, a sequential map because rho is a positive dose
    ring         enclosed = `dead` (the unit stops running), bare = `mute` (it stops being heard)

Every seed is drawn, and nothing joins them: paths through each (kind, beta) in rising rho were
drawn once and removed, because 6 polylines through 72 points obscured the thing the panel is for.
The rate is already on the colour axis, so the dose trend is read from colour, not from a line.

FILLED PLUS, NOT A BARE "+": an unfilled marker takes the rate colour on its strokes rather than its
face, so beta = 4 would read as a different colour family from beta = 1 and 2 at the same rho.

⚠️ R^2 HERE IS NOISE-FREE AND SO IS NOT THE NUMBER THE FIGURE 2 CACHE REPORTS. Scored in the noise
the networks trained in, `mute` gives up about two points (0.945 -> 0.922) and the arms compress
toward each other; the duplication arm showed the same instrument split, in the other direction and
much larger. The footnote carries it; the panel does not silently mix the two.

Output: img/internal_figures/slide_24_dropout_tradeoff.{pdf,svg}
Usage:  python fig_slide24_tradeoff.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from fig_slides import BERN_BETAS, BERN_CTRL, BERN_KINDS, BERN_RATES, DATA_DIR, bern_cell, bern_read

V_TARGET = 0.724        # flip-flop target variance; r2 = 1 - loss/V (pr_matrix.target_variance)
SHAPE = {1: "o", 2: "D", 4: "P"}
SIZE = {1: 26, 2: 24, 4: 34}        # a plus reads smaller than a disc of equal area
# viridis truncated at the dark end: its first 12% is near-black, which at marker size reads as the
# `dead` ring rather than as a low rate, and the ring is a different variable.
CMAP = LinearSegmentedColormap.from_list("viridis_hi", plt.get_cmap("viridis")(np.linspace(0.12, 1.0, 256)))
NORM = Normalize(vmin=0.03, vmax=0.27)   # padded past the extreme rates so no marker is near-black


def r2_of(loss):
    """R^2 from the noise-free task loss on the 3-bit flip-flop.

    Args:
        loss: array of masked MSE values from the trainer's noise-free probe.
    Returns:
        array of R^2 values, 1 - loss / V_TARGET.
    """
    return 1.0 - np.asarray(loss, float) / V_TARGET


def load():
    """Every run of the post-fix dropout grid, plus the no-dropout control.

    Returns:
        (points, control) where points is a list of dicts with kind, rate, beta, active, r2 per
        SEED, and control is (active array, r2 array). Both empty if a cell is missing.
    """
    pts = []
    for kind, _ in BERN_KINDS:
        for rate in BERN_RATES:
            for beta in BERN_BETAS:
                act, loss = bern_read(bern_cell(kind, rate, beta))
                if not len(act):
                    print(f"  missing cell: {kind} rho={rate} beta={beta}")
                    continue
                for a, r in zip(act, r2_of(loss)):
                    pts.append(dict(kind=kind, rate=rate, beta=beta, active=float(a), r2=float(r)))
    ca, cl = bern_read(os.path.join(DATA_DIR, BERN_CTRL))
    return pts, (np.asarray(ca, float), r2_of(cl))


def tradeoff_slide(name="slide_24_dropout_tradeoff"):
    """Draw the combined panel. Returns the output path, or None if the grid is incomplete."""
    pts, (ca, cr) = load()
    if not pts or not len(ca):
        print(f"  SKIP {name}: dropout grid incomplete")
        return None
    ps.setup()
    fig, ax = plt.subplots(figsize=(150 * ps.MM, 95 * ps.MM))

    # THE CONTROL IS NOT A SERIES: a crosshair at its seed mean with its own seed spread, in the
    # neutral reference colour, so no reader takes it for a fourth rate.
    ax.axhspan(cr.mean() - cr.std(ddof=1), cr.mean() + cr.std(ddof=1),
               color=ps.BASE, alpha=0.13, lw=0, zorder=1)
    ax.axvspan(ca.mean() - ca.std(ddof=1), ca.mean() + ca.std(ddof=1),
               color=ps.BASE, alpha=0.13, lw=0, zorder=1)
    ax.plot(ca, cr, "x", color=ps.BASE, ms=5, mew=1.3, zorder=5)
    # The control is named in the legend, not annotated in the axes: every empty patch of this panel
    # is empty for only one of the two kinds, so floating text lands on somebody's cluster.

    for p in pts:
        col = CMAP(NORM(p["rate"]))
        if p["kind"] == "dead":          # enclosed: the unit stops running, not just being heard
            ax.scatter([p["active"]], [p["r2"]], s=SIZE[p["beta"]] * 4.2, marker="o",
                       facecolors="none", edgecolors=ps.INK, linewidths=0.55, zorder=3)
        ax.scatter([p["active"]], [p["r2"]], s=SIZE[p["beta"]], marker=SHAPE[p["beta"]],
                   color=col, edgecolors="none", zorder=4)

    ax.set(xlabel="active units of 1000", ylabel="$R^2$, noise-free, dropout off")
    ps.ygrid(ax)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), ax=ax, pad=0.015,
                      fraction=0.045, ticks=list(BERN_RATES))
    cb.set_label(r"drop rate $\rho$", fontsize=7)
    cb.ax.set_yticklabels([f"{r:g}" for r in BERN_RATES])
    cb.outline.set_visible(False)

    # A legend fills COLUMN-wise, so the handles are interleaved to land as two readable rows:
    # the three shapes on top, the two kinds and the control underneath.
    shapes = [Line2D([], [], ls="none", marker=SHAPE[b], ms=4.6, color=ps.MUTED,
                     label=rf"$\beta$ = {b}") for b in BERN_BETAS]
    kinds = [Line2D([], [], ls="none", marker="o", ms=4.2, mfc="none", mec=ps.INK, mew=0.7,
                    label="dead (ringed)"),
             Line2D([], [], ls="none", marker="o", ms=4.2, color=ps.MUTED, label="mute (bare)"),
             Line2D([], [], ls="none", marker="x", ms=4.6, mew=1.3, color=ps.BASE,
                    label=f"no dropout: {ca.mean():.0f} units, $R^2$ {cr.mean():.3f} "
                          f"(band = seed sd {cr.std(ddof=1):.3f})")]
    keys = [h for pair in zip(shapes, kinds) for h in pair]

    ax.legend(handles=keys, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
              fontsize=6.8, frameon=False, handletextpad=0.4, columnspacing=1.6,
              borderpad=0.2, labelspacing=0.5)

    # THE CLAIM IS COMPUTED. Each kind's own slope of R^2 on active units, over its 36 runs, with the
    # control's seed spread as the scale that slope has to beat to mean anything.
    stat = {}
    for kind, _ in BERN_KINDS:
        A = np.array([p["active"] for p in pts if p["kind"] == kind])
        R = np.array([p["r2"] for p in pts if p["kind"] == kind])
        stat[kind] = (A, R, np.polyfit(A, R, 1)[0] * 100 if A.size > 2 else float("nan"))
    # Does either kind leave the control's own seed band? Tested, not asserted: the control's sd is
    # 0.063 because two of its three seeds blow up transiently (slide 23c), which is wide enough to
    # swallow an effect, and a cell counts as outside only if its three-seed mean clears the band.
    lo, hi = cr.mean() - cr.std(ddof=1), cr.mean() + cr.std(ddof=1)
    def cell_mean(kind, rate, beta):
        """Three-seed mean R^2 of one cell, or nan if the cell is absent."""
        v = [p["r2"] for p in pts if (p["kind"], p["rate"], p["beta"]) == (kind, rate, beta)]
        return float(np.mean(v)) if v else float("nan")

    out_of_band = {k: sum(1 for r in BERN_RATES for b in BERN_BETAS
                          if np.isfinite(m := cell_mean(k, r, b)) and not (lo <= m <= hi))
                   for k, _ in BERN_KINDS}
    mu_a, mu_r, slope_m = stat["mute"]
    de_a, de_r, slope_d = stat["dead"]
    # THE FIGURE STATES WHAT IT IS, NOT WHAT IT MEANS. The claim belongs in the deck text, where it
    # can be read, argued with and changed without re-rendering; a title that interprets its own
    # panel tells the audience what to see before they have seen it.
    fig.suptitle(
        "3-bit flip-flop, $N$ = 1000, 150,000 iterations\n"
        "active units against $R^2$, one point per run, 12 cells per dropout kind",
        fontsize=8.0, color=ps.INK, linespacing=1.45, y=1.145)
    fig.text(0.5, -0.02, "Scored in the noise the networks trained in, the arms compress and "
             r"$\texttt{mute}$ gives up about two points of $R^2$ (0.945 to 0.922).".replace(
                 r"\texttt{mute}", "mute"),
             ha="center", va="top", fontsize=6.6, color=ps.MUTED)
    return ps.save(fig, name, w_mm=150)


if __name__ == "__main__":
    tradeoff_slide()
