#!/usr/bin/env python3
"""Slide 24: what each kind of dropout charges for the units it recruits.

Every dropout run as one point - the units it ended with against the score it ended at - and one
fitted line per kind of dropout. The line's slope IS the claim: how much held-out performance each
100 extra active units cost. The two kinds differ about three-fold in that exchange rate, which is
a statement about two slopes and is therefore drawn as two slopes.

TWO ENCODINGS, NOT FOUR. The earlier version of this panel put the drop rate on a viridis fill, the
targeting exponent on the marker shape and the kind of dropout on a black ring around the marker,
and named all three in a legend above the axes. At projector distance none of the shape or ring
distinctions survive, and the reader had to decode a legend before the picture meant anything. So
the rate and the exponent are gone from the panel - they are still printed below it, and slides 23b
and 23c are about them - and what remains is position and one colour per kind of dropout:

    x        active units of 1000, the scale-free criterion, at end of training
    y        held-out r2: a batch the network never trained on, run at the sigma_rec and sigma_inp
             it trained with, no synaptic noise, nine draws, dropout off - so a dropout net is
             scored on its full network exactly as the control is
    colour   ps.COND_COL["mute"] for both kinds - see ONE HUE below
    fill     filled dots and a solid fit for `mute`, hollow dots and a dashed fit for `dead`
    band     the 95% confidence band of each fit, so the slope is read with its own uncertainty
    line     each kind's least-squares fit of r2 on active units, with its slope written beside it

ONE HUE FOR ONE RULE. `dead` used to be drawn in ps.COND_COL["dead"], which is the same hex as
ps.COND_COL["duplicate"] - prune-and-duplicate's colour on five other figures in this deck - so a
listener who had learned that purple meant duplication met it again here meaning something else.
`mute` and `dead` are two variants of ONE rule, so they now share the dropout hue and differ in
fill: filled dots with a solid fit against hollow dots with a dashed fit. No new hue is invented
and no other condition's colour is borrowed. The two clouds also sit in different halves of the
panel, which is the stronger separation of the two.
(Slides 24 and 24b still draw `dead` in ps.COND_COL["dead"]. Both are cut from the deck.)

THE SLOPE IS DRAWN WITH ITS UNCERTAINTY. A fitted slope printed as a bare number over a cloud
spanning 0.75 to 0.90 at one x is a claim the picture does not support, so each fit carries its 95%
confidence band and each slope is written with its own standard error.

The two kinds are named in place, at their own lines, not in a legend: "mute" hides a unit from the
read-out and leaves it running, "dead" switches it off entirely. Slide 23d draws that difference.

THE NOISE-FREE SCORE IS GONE FROM THIS PANEL AND FROM THE PROJECT. It measured the networks in a
condition none of them operates in, and it reversed the ranking of arms rather than merely shifting
it. Everything here is the one measure, so nothing has to be qualified by which instrument produced
it.

Output: img/internal_figures/slide_24_dropout_tradeoff.{pdf,svg}
Usage:  python fig_slide24_tradeoff.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from fig_slides import BERN_BETAS, BERN_CTRL, BERN_KINDS, BERN_RATES, DATA_DIR, bern_cell, bern_read

# What each kind does, in the words the panel uses, plus where its label sits relative to its own
# fitted line. `mute` runs along the top of the cloud, so its label goes under its line; `dead`
# falls away underneath, so its label goes above.
KIND_SAYS = {"mute": ("output hidden (mute)", 0.56, -1),
             "dead": ("switched off (dead)", 0.55, +1)}

# ONE HUE, TWO FILLS - see the module docstring. `mute` is the dropout colour filled with a solid
# fit; `dead` is the same colour hollow with a dashed fit. (marker face, marker edge width, fit
# line style) per kind.
KIND_FILL = {"mute": ("solid", 0.5, "-"), "dead": ("hollow", 1.2, (0, (3.2, 1.8)))}
Z95 = 1.96          # normal quantile; with 36 runs per kind the t quantile is 2.03, a 4% change


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
                for a, r in zip(act, loss):
                    pts.append(dict(kind=kind, rate=rate, beta=beta, active=float(a), r2=float(r)))
    ca, cl = bern_read(os.path.join(DATA_DIR, BERN_CTRL))
    return pts, (np.asarray(ca, float), np.asarray(cl, float))


def exchange_rate(active, r2):
    """Least-squares cost in held-out r2 of 100 extra active units, with its uncertainty.

    Args:
        active: (n,) array of active-unit counts, one per run;
        r2: (n,) array of held-out scores, same order.
    Returns:
        (slope per 100 units, intercept at 0 units, standard error of the slope per 100 units,
        residual sd in r2, mean of active, sum of squared deviations of active) as floats, all nan
        with fewer than three runs - a slope through two points is not an exchange rate.
    """
    active, r2 = np.asarray(active, float), np.asarray(r2, float)
    if active.size < 3:
        return (float("nan"),) * 6
    b, a = np.polyfit(active, r2, 1)
    n = active.size
    resid = r2 - (a + b * active)
    s = float(np.sqrt(resid @ resid / (n - 2)))          # residual sd, in units of r2
    xbar = float(active.mean())
    sxx = float(((active - xbar) ** 2).sum())
    return float(b) * 100.0, float(a), 100.0 * s / np.sqrt(sxx), s, xbar, sxx


def fit_band(xs, slope100, intercept, s_resid, xbar, sxx, n):
    """The 95% confidence band of a fitted line, evaluated at `xs`.

    The band is for the FITTED LINE, not for a future run: it is what the slope's own uncertainty
    allows, which is the claim the panel makes. The seed scatter around it stays visible as the
    dots themselves.

    Args:
        xs: (m,) active-unit positions to evaluate at; slope100: fitted slope in r2 per 100 units;
        intercept: fitted r2 at zero active units; s_resid: residual sd in r2;
        xbar: mean active-unit count of the fitted runs; sxx: sum of squared deviations of those
        counts; n: how many runs were fitted.
    Returns:
        (lower, upper), two (m,) arrays of held-out r2.
    """
    xs = np.asarray(xs, float)
    mid = intercept + slope100 * xs / 100.0
    half = Z95 * s_resid * np.sqrt(1.0 / n + (xs - xbar) ** 2 / sxx)
    return mid - half, mid + half


def tradeoff_slide(name="slide_24_dropout_tradeoff"):
    """Draw the combined panel. Returns the output path, or None if the grid is incomplete."""
    pts, (ca, cr) = load()
    if not pts or not len(ca):
        print(f"  SKIP {name}: dropout grid incomplete")
        return None
    ps.setup()
    fig, ax = plt.subplots(figsize=(150 * ps.MM, 95 * ps.MM))

    # THE CONTROL IS NOT A SERIES: a crosshair at its seed mean with its own seed spread, in the
    # neutral reference colour, so no reader takes it for a third kind of dropout.
    ax.axhspan(cr.mean() - cr.std(ddof=1), cr.mean() + cr.std(ddof=1),
               color=ps.BASE, alpha=0.13, lw=0, zorder=1)
    ax.axvspan(ca.mean() - ca.std(ddof=1), ca.mean() + ca.std(ddof=1),
               color=ps.BASE, alpha=0.13, lw=0, zorder=1)
    ax.plot(ca, cr, "x", color=ps.BASE, ms=6, mew=1.5, zorder=5)
    ax.annotate("untouched", (ca.mean(), cr.mean()), textcoords="offset points", xytext=(4, 8),
                ha="left", va="bottom", fontsize=7.0, color=ps.BASE)

    stat = {}
    col = ps.COND_COL["mute"]                 # one hue for both variants - see the module docstring
    for kind, _ in BERN_KINDS:
        A = np.array([p["active"] for p in pts if p["kind"] == kind])
        R = np.array([p["r2"] for p in pts if p["kind"] == kind])
        face, mew, fit_ls = KIND_FILL[kind]
        ax.plot(A, R, "o", ms=4.6, mfc=col if face == "solid" else "none", mec=col,
                mew=mew, alpha=0.9, zorder=4)
        slope, intercept, se, s_resid, xbar, sxx = exchange_rate(A, R)
        stat[kind] = (A, R, slope, se)
        if not np.isfinite(slope):
            continue
        # THE LINE IS THE CLAIM AND THE BAND IS WHAT THE RUNS ALLOW IT TO BE. Drawn over the runs it
        # was fitted to, with its slope written on it rather than in the title, so the three-fold
        # difference between the two kinds is a pair of visibly different gradients and not a pair
        # of numbers to compare in the reader's head.
        xs = np.linspace(A.min(), A.max(), 60)
        lo_b, hi_b = fit_band(xs, slope, intercept, s_resid, xbar, sxx, A.size)
        ax.fill_between(xs, lo_b, hi_b, color=col, alpha=0.20, lw=0, zorder=3)
        ax.plot(xs, intercept + slope * xs / 100.0, ls=fit_ls, lw=2.0, color=col, zorder=6,
                solid_capstyle="round")
        says, at, side = KIND_SAYS[kind]
        x_lab = A.min() + at * (A.max() - A.min())
        ax.annotate(f"{says}\n{slope:+.3f} $\\pm$ {Z95 * se:.3f} per 100 units",
                    (x_lab, intercept + slope * x_lab / 100.0), textcoords="offset points",
                    xytext=(0, side * 15), ha="center", va="bottom" if side > 0 else "top",
                    fontsize=7.4, color=col, linespacing=1.35)

    ax.set(xlabel="active units of 1000", ylabel="held-out $r^2$")
    ps.ygrid(ax)

    # Does either kind leave the control's own seed band? Tested, not asserted: the control's sd is
    # 0.002, and a cell counts as outside only if its three-seed mean clears the band.
    lo, hi = cr.mean() - cr.std(ddof=1), cr.mean() + cr.std(ddof=1)

    def cell_mean(kind, rate, beta):
        """Three-seed mean r2 of one cell, or nan if the cell is absent."""
        v = [p["r2"] for p in pts if (p["kind"], p["rate"], p["beta"]) == (kind, rate, beta)]
        return float(np.mean(v)) if v else float("nan")

    out_of_band = {k: sum(1 for r in BERN_RATES for b in BERN_BETAS
                          if np.isfinite(m := cell_mean(k, r, b)) and not (lo <= m <= hi))
                   for k, _ in BERN_KINDS}
    # The rate and the targeting exponent are no longer encoded in the panel, so the full grid is
    # printed here - that is where the deck text quotes them from.
    print(f"  control {ca.mean():.0f} units, r2 {cr.mean():.4f}, seed sd {cr.std(ddof=1):.4f} "
          f"(band {lo:.4f}-{hi:.4f})")
    for kind, _ in BERN_KINDS:
        A, R, slope, se = stat[kind]
        print(f"  {kind:5s} {A.min():.0f}-{A.max():.0f} units, r2 {R.min():.4f}-{R.max():.4f}, "
              f"slope {slope:+.4f} +- {Z95 * se:.4f} per 100 units ({A.size} runs), "
              f"{out_of_band[kind]}/12 cells outside the band")
    for kind, _ in BERN_KINDS:
        for rate in BERN_RATES:
            for beta in BERN_BETAS:
                v = [p for p in pts if (p["kind"], p["rate"], p["beta"]) == (kind, rate, beta)]
                if v:
                    print(f"    {kind:5s} rho={rate:<6g} beta={beta}  "
                          f"active {np.mean([x['active'] for x in v]):4.0f}  "
                          f"r2 {np.mean([x['r2'] for x in v]):.4f}")
    # The task and the size only. The exchange rate used to be printed here; it is drawn now.
    fig.suptitle("3-bit flip-flop, $N$ = 1000", fontsize=8.0, color=ps.INK, y=1.02)
    return ps.save(fig, name, w_mm=150)


if __name__ == "__main__":
    tradeoff_slide()
