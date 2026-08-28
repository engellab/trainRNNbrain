#!/usr/bin/env python3
"""
Shared plotting style for the analysis figures.

`common.py` is deliberately free of matplotlib - importing it must stay free of side effects - so
everything about how figures LOOK lives here instead.

The point is that one variable always gets one visual channel across every figure in the project:

    N (network size)      colour from a fixed discrete map, small N dark -> large N light
    k (task complexity)   colour from viridis, low k dark -> high k yellow
    read-out criterion    line style / row of a panel grid, never colour
    seed                  never its own channel; it is an error bar or a repeated faint line

Mixing those up is how a reader concludes something about N from a figure that varied k. When both
N and k must appear on one axes, colour carries whichever is the *within-panel* variable and the
other is named in the panel title.

Self-check: `python plotstyle.py`
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import IMG_DIR

# Fixed colours for the sizes actually used in this project, so N=1000 is the same red everywhere.
COL_N = {500: "#1f77b4", 1000: "#d62728", 2000: "#2ca02c", 5000: "#9467bd", 10000: "#8c564b"}

# The read-out criteria, in the order they should always be presented: cheapest/most naive first,
# most defensible last. Keys match the `crit` strings used by the figure scripts.
CRIT_STYLE = {
    "endpoint": ("-",  "v", "endpoint (own final iteration)"),
    "iter":     ("-",  "o", "fixed iteration"),
    "loss":     ("--", "s", "fixed loss $L^*$"),
    "excess":   ("-.", "^", r"fixed excess over own $L_\infty$"),
    "drift":    (":",  "D", "onset of non-directed drift"),
}


def col_n(N):
    """Colour for a network size, stable across every figure.

    Args:
        N: network size (int).
    Returns:
        matplotlib colour string; grey for sizes not in the project's fixed set.
    """
    return COL_N.get(int(N), "#7f7f7f")


def col_k(k, ks):
    """Colour for a task complexity, dark at the lowest k and light at the highest.

    Args:
        k: this complexity; ks: the full sorted list of complexities in the figure, so the mapping is
            stable across panels that happen to hold different subsets.
    Returns:
        RGBA tuple from viridis.
    """
    ks = sorted(ks)
    lo, hi = ks[0], ks[-1]
    f = 0.0 if hi == lo else (k - lo) / (hi - lo)
    return plt.cm.viridis(0.05 + 0.80 * f)


def setup():
    """Apply the project's rcParams. Call once at the top of a figure script."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.7,
        "lines.markersize": 5,
        "errorbar.capsize": 2.5,
    })


def band(ax, x, mu, sd, color, label=None, marker="o", ls="-", alpha=0.16):
    """Mean line with a +-1 sd shaded band - the project's default for "3 seeds per cell".

    A band rather than error bars because these figures routinely overlay six or more conditions on
    one axes, where bars collide and become unreadable.

    Args:
        ax: axes; x: x values; mu, sd: mean and sd, same length as x; color: line colour;
        label: legend entry or None; marker, ls: marker and line style; alpha: band opacity.
    Returns:
        None.
    """
    x, mu, sd = np.asarray(x, float), np.asarray(mu, float), np.asarray(sd, float)
    ax.plot(x, mu, ls, marker=marker, color=color, label=label)
    ok = np.isfinite(mu) & np.isfinite(sd)
    if ok.sum() > 1:
        ax.fill_between(x[ok], (mu - sd)[ok], (mu + sd)[ok], color=color, alpha=alpha, lw=0)


def contour(ax, ks, Ns, Z, label, log_n=True, levels=12, cmap="viridis", fmt="%.4f"):
    """Filled contour of a quantity over the (k, N) grid, with the sampled cells marked.

    Cells are marked because the grid is small and partly incomplete: a smooth contour over 6x3
    points invites the reader to trust interpolation that no data supports, so where the data
    actually sits has to stay visible.

    Args:
        ax: axes; ks: complexities (x); Ns: sizes (y); Z: (len(Ns), len(ks)) values, nan where the
            cell is missing; label: colourbar label; log_n: log-scale the N axis;
        levels: contour levels; cmap: colormap; fmt: contour label format.
    Returns:
        the QuadContourSet, or None if fewer than 4 cells carry data.
    """
    Z = np.asarray(Z, float)
    if np.isfinite(Z).sum() < 4:
        ax.text(.5, .5, "not enough cells", ha="center", va="center", transform=ax.transAxes)
        return None
    K, NN = np.meshgrid(np.asarray(ks, float), np.asarray(Ns, float))
    # Contouring cannot span nans; fill them by nearest-finite so the surface is drawn, then mark
    # the real cells so the interpolated region is never mistaken for measurement.
    Zf = Z.copy()
    if np.isnan(Zf).any():
        fin = np.isfinite(Zf)
        pts = np.array([(K[i, j], NN[i, j]) for i, j in zip(*np.where(fin))])
        vals = Zf[fin]
        for i, j in zip(*np.where(~fin)):
            d = ((pts[:, 0] - K[i, j]) ** 2 + (np.log(pts[:, 1]) - np.log(NN[i, j])) ** 2)
            Zf[i, j] = vals[int(np.argmin(d))]
    cs = ax.contourf(K, NN, Zf, levels=levels, cmap=cmap)
    cl = ax.contour(K, NN, Zf, levels=levels, colors="k", linewidths=0.4, alpha=0.45)
    ax.clabel(cl, inline=True, fontsize=6, fmt=fmt)
    fin = np.isfinite(Z)
    ax.plot(K[fin], NN[fin], "o", ms=3.5, mfc="white", mec="k", mew=0.8)
    ax.plot(K[~fin], NN[~fin], "x", ms=4, color="k", alpha=.55)
    if log_n:
        ax.set_yscale("log")
        ax.set_yticks(list(Ns))
        ax.set_yticklabels([str(int(n)) for n in Ns])
    ax.set(xlabel="k (bits)", ylabel="N (units)", xticks=list(ks))
    cb = ax.figure.colorbar(cs, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(label, fontsize=8)
    return cs


def legend_n(ax, Ns, **kw):
    """Add a legend keyed by network size, using the project's fixed colours."""
    ax.legend(handles=[Line2D([], [], color=col_n(N), marker="o", label=f"N={N}") for N in Ns], **kw)


def legend_k(ax, ks, **kw):
    """Add a legend keyed by task complexity, using the project's viridis mapping."""
    ax.legend(handles=[Line2D([], [], color=col_k(k, ks), marker="o", label=f"k={k}") for k in ks],
              ncol=2, **kw)


def save(fig, name, tight=True):
    """Save a figure into img/internal_figures and print the path.

    Args:
        fig: the figure; name: file stem, no extension; tight: apply tight_layout first.
    Returns:
        the written path.
    """
    if tight:
        fig.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    out = os.path.join(IMG_DIR, f"{name}.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def _self_check():
    """Assert the style invariants that other scripts rely on. Raises on failure."""
    assert col_n(1000) == col_n(1000) and col_n(500) != col_n(1000), "N colours must be distinct"
    assert col_n(99999) == "#7f7f7f", "unknown N must fall back to grey"
    # k colouring must be monotone in k and independent of which subset is passed, given same range
    ks = [1, 2, 4, 8]
    assert col_k(1, ks) != col_k(8, ks)
    assert col_k(4, ks) == col_k(4, [1, 4, 8, 2]), "col_k must not depend on argument order"
    assert col_k(3, [3]) == col_k(3, [3]), "degenerate single-k range must not divide by zero"

    # contour must survive a grid with holes and still mark which cells were real
    fig, ax = plt.subplots()
    Z = np.array([[1.0, 2.0, 3.0], [2.0, np.nan, 4.0], [3.0, 4.0, 5.0]])
    assert contour(ax, [1, 2, 3], [500, 1000, 2000], Z, "z") is not None
    plt.close(fig)
    fig, ax = plt.subplots()
    assert contour(ax, [1, 2], [500, 1000], np.full((2, 2), np.nan), "z") is None
    plt.close(fig)

    # band tolerates nans without raising
    fig, ax = plt.subplots()
    band(ax, [1, 2, 3], [1.0, np.nan, 3.0], [0.1, 0.1, np.nan], "C0", label="x")
    plt.close(fig)
    print("plotstyle.py self-check passed")


if __name__ == "__main__":
    _self_check()
