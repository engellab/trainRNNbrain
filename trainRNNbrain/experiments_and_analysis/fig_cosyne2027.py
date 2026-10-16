#!/usr/bin/env python3
"""One-page figure for the COSYNE 2027 abstract (page 2: at most 7.5 x 6.4 in, text >= 7 pt).

  A  an example trained network and eight of its units          fig_paper_F1.panel_a
  B  participation before and after training                    fig_paper_F1.panel_b_pair
  C  active units vs network size, each run at its loss floor   fig_paper_F1.panel_c
  D  ablation at the loss floor; stars = low-participation
     units removed, then retrained (3k iterations)              fig_paper_F1.panel_d
  E  loss and low-participation units over training, with the
     floor                                                      F1.trajectory, drawn compactly
  F  everything tried at N = 1000, one axes per task (CDDM,
     3-bit flip-flop; DMTS has none): active units against R^2
     minus the R^2 of its own control                           Figure 2 cache, B5, A4, 75k cache
  G  the flip-flop remedies' and the control's active units as
     % of N vs N (G_MODE "percent"; backup "ratio": multiplier
     over the control, written as fig_cosyne2027_Gratio)        75k cache (ff75k_readout)

READ-OUTS (not re-derived here). Panels C-F read every run where its NOISY training loss first
reaches 1.05x its own fitted floor (`common.noisy_floor`, with the beta guard and the smoothed-minimum
fallback). The flip-flop remedies (dropout, duplication, rescaling, synaptic noise, rate + sparsity)
have no floor read-out yet - their 200k top-ups are still training - so in F and G they are the end
of 75,000 iterations, drawn with OPEN markers and labelled so; their R^2 is the 75k cache's held-out
score under the training noise (sigma_w = 0, 8 unseeded draws), compared only with the 75k control
scored the same way. Every other R^2 in F is `heldout_r2` (seeded) as `fig_knobs_at_floor` and
`b5_read` compute it: measured on a kept snapshot where one exists, otherwise estimated from the
loss ratio.

THREE STAGES, because the data live on two machines:
  --collect-della   on a Della CPU node, from the repository root (data/ links the trained
                    networks): runs the reused F1 panel functions once on a throwaway figure and
                    records every input they read (runs_with_loss, ablation_results, the example
                    network), plus the panel E trajectories and the 75k cache; writes
                    data/fig_cosyne2027_della.pkl.
  --collect-b5      where NBitFlipFlop_b5_optimizer is (the Mac copy): reads every B5 arm with
                    b5_read.read_arm; writes data/fig_cosyne2027_b5.pkl.
  (no flag)         draws from the two caches and data/fig_paper_F2_fails_cache.pkl: the F1 panel
                    functions are called unchanged, with their data loaders replaced by replays of
                    what the Della stage recorded. Writes img/internal_figures/fig_cosyne2027.{pdf,
                    svg} and prints a layout check and every number drawn.
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
import fig_paper_F1 as F1
import fig_paper_F2 as F2
import fig_paper_F2_fails as F2F
import ff75k_readout as FF75
import b5_read as B5
from common import DATA_DIR, logbin
from heldout_r2 import load_cache

REPO = os.path.normpath(os.path.join(DATA_DIR, "..", ".."))
DELLA_CACHE = os.path.join(REPO, "data", "fig_cosyne2027_della.pkl")
B5_CACHE = os.path.join(REPO, "data", "fig_cosyne2027_b5.pkl")
SHARDS_75K = "/scratch/gpfs/TENGEL/pt1290/f2_75k/shards_v2"   # the 75k flip-flop cache (2026-10-09)

W_IN, H_IN = 7.46, 6.36     # the canvas; paperstyle.save pads the tight box by 0.02 in a side,
                            # so the written page is at most 7.5 x 6.4 in
MIN_PT = 7.0                # smallest text on the page (printed at 100%)
LETTER_PT = 10

# panel F: one colour per family, one marker per task, filled = at the loss floor, open = end of 75k
FAM_COL = {"knob": ps.SLOTS[0], "optimizer": ps.SLOTS[3], "revival": ps.SLOTS[4],
           "drive": ps.SLOTS[1]}
FAM_LABEL = {"knob": "standard knobs", "optimizer": "optimizers (SGD, AdamW, Adam wd 0)",
             "revival": "re-initialise / zero low-\nparticipation units",
             "drive": "participation drives"}
TASK_MARK = {"CDDM": "o", "flip-flop": "^"}
F_TITLE = {"CDDM": "CDDM", "flip-flop": "3-bit flip-flop"}   # panel F: one axes per task, in order
# Figure 2 rungs that are not standard knobs (stem, label prefix) -> family
RUNG_FAMILY = {("slide_k_frm_rws", "rate"): "drive", ("slide_k_frm_rws", "both"): "drive",
               ("revival_cddm", "re-initialise"): "revival", ("revival_ff", "zero-out"): "revival",
               ("revival_ff", "random"): "revival"}
# the flip-flop remedies of the 75k cache, in fig_paper_F2.ARMS order, with their panel names
REMEDIES = [("mute", "dropout"), ("duplicate", "duplication"), ("rescale", "rescaling"),
            ("synnoise", "synaptic noise"), ("both", "rate + sparsity")]
B5_STANDARD = ("adam", "1e-6", "1e-3")
# panel G: one colour per remedy, sampled from the inferno ramp so that lightness falls in the order
# the lines end at N = 4000 (rate + sparsity highest and darkest, synaptic noise lowest and lightest):
# the stack still reads in grayscale. CIE L* 15 / 28 / 40 / 52 / 62; the lightest has 3:1 contrast
# against white, enough for 7 pt labels.
REMEDY_COL = {"rate + sparsity": "#390963", "duplication": "#721a6e", "dropout": "#a82e5f",
              "rescaling": "#d74b3f", "synaptic noise": "#f06f20"}
# panel G's y axis: "percent" (active units as % of N, control drawn as its own line) is the figure;
# "ratio" (each remedy's active count over the control's) is the backup, written beside it as
# fig_cosyne2027_Gratio. Every run of the drawing stage writes both.
G_MODE = "percent"
G_MODES = ("percent", "ratio")


# ------------------------------------------------------------------------------------------------
# collection
# ------------------------------------------------------------------------------------------------

def rel(path):
    """A data path made portable: the part after 'trained_RNNs/' (the root differs per machine).

    Args:
        path: a path or glob under DATA_DIR.
    Returns:
        str.
    """
    return path.split("trained_RNNs/")[-1]


def collect_della():
    """Record every input the reused panels read, on Della, and pickle it to DELLA_CACHE.

    F1.panel_c and F1.panel_d are run once on a throwaway figure with their loaders wrapped, so the
    record holds exactly the calls they make. The panel E trajectories (F1.trajectory, which also
    scores held-out R^2) are reduced to what is drawn: 120 log-binned medians of the noisy loss,
    the participation-probe series, and the floor crossing computed on the full series. The 75k
    flip-flop cache is merged with ff75k_readout.merge and the networks that lost the task at 40k
    or 75k (r2_common < ff75k_readout.SOLVED) are set aside, as ff75k_readout does.

    Returns:
        the path written.
    """
    out = dict(runs_with_loss={}, ablation={})
    rwl, abl = F1.runs_with_loss, F1.ablation_results

    def rec_rwl(pattern, min_r2=None):
        """runs_with_loss, recorded."""
        res = rwl(pattern, min_r2=min_r2)
        out["runs_with_loss"][(rel(pattern), min_r2)] = res
        return res

    def rec_abl(prefix, readout):
        """ablation_results, recorded."""
        res = abl(prefix, readout)
        out["ablation"][(prefix, readout)] = res
        return res

    F1.runs_with_loss, F1.ablation_results = rec_rwl, rec_abl
    try:
        fig = plt.figure()
        F1.panel_c(fig.add_subplot(4, 1, 1))
        F1.panel_d([fig.add_subplot(4, 1, i) for i in (2, 3, 4)])
        plt.close(fig)
    finally:
        F1.runs_with_loss, F1.ablation_results = rwl, abl

    rates, targets, p, p_init, iterations = F1.example_network_pair()
    out["example"] = dict(rates=rates, p=p, p_init=p_init, iterations=iterations,
                          exc_fraction=F1.excitatory_fraction())

    out["trajectory"] = {}
    for label, pat, _col, _N in F1.TRAJ:
        runs = []
        for d in sorted(glob.glob(os.path.join(pat, "*"))):
            if not os.path.isdir(d):
                continue
            r = F1.trajectory(d)
            if r is None:
                continue
            tb, yb = logbin(r["it_loss"], r["loss"], nbins=120)
            runs.append(dict(name=os.path.basename(d), tb=tb, loss_rel=yb / yb[0],
                             it_act=r["it_act"], silent=r["silent"], N=r["N"], r2=r["r2"],
                             plateau=r["plateau"], no_floor=r["no_floor"],
                             it_end=float(r["it_loss"][-1])))
        out["trajectory"][label] = runs

    c = FF75.merge(SHARDS_75K)
    bad = (c["r2_common"] < FF75.SOLVED) | (c["p40_r2_common"] < FF75.SOLVED)
    keep = [k for k in ("task", "arm", "N", "cell", "n_active", "r2_common") if k in c]
    out["ff75k"] = {k: np.asarray(c[k])[~bad] for k in keep}
    out["ff75k_set_aside"] = int(bad.sum())

    with open(DELLA_CACHE, "wb") as fh:
        pickle.dump(out, fh)
    print(f"wrote {DELLA_CACHE}")
    return DELLA_CACHE


def collect_b5():
    """Read every B5 optimizer arm at its floor (b5_read.read_arm) and pickle it to B5_CACHE.

    The held-out R^2 cache is keyed by the parameter file's path relative to the checkout that
    scored it; the Mac copy was scored from a sibling checkout ('../trainRNNbrain/data/...'). The
    same files (same size and mtime in the key) are looked up under this checkout's spelling by
    aliasing those keys in memory; nothing is written back to the shared cache.

    Returns:
        the path written.
    """
    cache = load_cache()
    for k in list(cache):
        if k.startswith("../") and "/data/trained_RNNs/" in k:
            cache.setdefault("data/trained_RNNs/" + k.split("/data/trained_RNNs/", 1)[1], cache[k])
    arms = {}
    for cell in sorted(glob.glob(os.path.join(B5.SWEEP, "*/"))):
        opt, wd, lr = B5.arm_of(os.path.basename(cell.rstrip("/")))
        if opt == "sgd" and lr in B5.SKIP_LR:
            continue
        arms[(opt, wd, lr)] = B5.read_arm(cell, cache)
    with open(B5_CACHE, "wb") as fh:
        pickle.dump(arms, fh)
    print(f"wrote {B5_CACHE}")
    return B5_CACHE


def install_replays(della):
    """Replace F1's data loaders with replays of what collect_della recorded.

    Args:
        della: the dict collect_della wrote.
    Returns:
        None. F1.runs_with_loss, F1.ablation_results and F1.excitatory_fraction now read the record.
    """
    def runs_with_loss(pattern, min_r2=None):
        """Replay of F1.runs_with_loss."""
        return della["runs_with_loss"][(rel(pattern), min_r2)]

    def ablation_results(prefix, readout):
        """Replay of F1.ablation_results."""
        return della["ablation"].get((prefix, readout), ([], []))

    F1.runs_with_loss = runs_with_loss
    F1.ablation_results = ablation_results
    F1.excitatory_fraction = lambda: della["example"]["exc_fraction"]


# ------------------------------------------------------------------------------------------------
# panels
# ------------------------------------------------------------------------------------------------

def axes_in(fig, x0, y0, w, h):
    """Add an axes at a rectangle given in inches from the canvas' lower-left corner.

    Args:
        fig: the figure; x0, y0, w, h: the rectangle in inches.
    Returns:
        the Axes.
    """
    return fig.add_axes([x0 / W_IN, y0 / H_IN, w / W_IN, h / H_IN])


def letter(fig, s, x_in, top_in):
    """Panel letter, bold uppercase, its top-left corner at (x_in, top_in) inches.

    Args:
        fig: the figure; s: the letter; x_in: left edge in inches; top_in: top edge in inches
            (one value per row, so the letters of a row line up).
    Returns:
        the Text artist.
    """
    return fig.text(x_in / W_IN, top_in / H_IN, s, fontsize=LETTER_PT, fontweight="bold",
                    ha="left", va="top", color=ps.INK)


def short_task(label):
    """Short task name for the compact panels: '3-bit flip-flop' -> 'flip-flop', DMTS without delay."""
    return {"3-bit flip-flop": "flip-flop", F1.DMTS_LABEL: "DMTS"}.get(label, label)


def panel_A(fig, ex, box):
    """Panel A: F1.panel_a unchanged, then its trace header shortened to fit 7 pt text.

    Args:
        fig: the figure; ex: the example network record; box: (x0, y0, w, h) in inches, split
            between the circuit drawing and the traces.
    Returns:
        (n_live, n_shown_live, n_conn, the drawing's axes).
    """
    x0, y0, w, h = box
    w_net = 0.52 * w
    ax_net = axes_in(fig, x0, y0, w_net, h)
    ax_tr = axes_in(fig, x0 + w_net, y0, w - w_net, h - 0.02)
    n_live, n_shown_live, n_conn = F1.panel_a(ax_net, ax_tr, ex["rates"], ex["p"])
    ax_net.set_title("trained ReLU RNN, $N$ = 1000", fontsize=MIN_PT, color=ps.INK, pad=1)
    for t in ax_tr.texts:
        if t.get_text().endswith("one trial"):
            t.set_text(f"{F1.N_SHOWN} random units, one trial")
        if t.get_text().endswith("rate"):
            t.set_text("rate")        # the scale bar's value is a.u. and carries no meaning
    return n_live, n_shown_live, n_conn, ax_net


def panel_B(fig, ex, box):
    """Panel B: F1.panel_b_pair unchanged, titles and x label shortened, task and size named.

    Args:
        fig: the figure; ex: the example network record; box: (x0, y0, w, h) in inches.
    Returns:
        ((active untrained, active trained), top axes).
    """
    x0, y0, w, h = box
    gap = 0.20
    hh = (h - gap) / 2
    top = axes_in(fig, x0, y0 + hh + gap, w, hh)
    bot = axes_in(fig, x0, y0, w, hh)
    bot.sharex(top)
    counts = F1.panel_b_pair([top, bot], ex["p_init"], ex["p"], ex["iterations"])
    top.set_title("untrained", fontsize=MIN_PT, loc="left", pad=1)
    top.set_title(f"{F1.EXAMPLE_TASK}, $N$ = {len(ex['p'])}", fontsize=MIN_PT, loc="right", pad=1)
    bot.set_title(f"after {ex['iterations'] // 1000}k iterations", fontsize=MIN_PT, loc="left",
                  pad=1)
    bot.set_xlabel("participation $p_i$", labelpad=1)
    for ax in (top, bot):
        ax.set_ylabel("")
        leg = ax.get_legend()
        handles, labels = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
        leg.remove()
        # top: left of the criterion; bottom: over the empty valley, left of the criterion label
        ax.legend(handles, labels, loc="upper left",
                  bbox_to_anchor=(0.0, 1.0) if ax is top else (0.24, 1.0),
                  fontsize=MIN_PT, handlelength=0.8, handletextpad=0.3, borderaxespad=0.1,
                  labelspacing=0.15)
    fig.text((x0 - 0.36) / W_IN, (y0 + h / 2) / H_IN, "units", rotation=90, ha="left",
             va="center", fontsize=MIN_PT)
    return counts, top


def panel_C(fig, box):
    """Panel C: F1.panel_c unchanged, legend labels shortened, ticks as plain numbers at the sizes.

    Args:
        fig: the figure; box: (x0, y0, w, h) in inches.
    Returns:
        (fits as F1.panel_c returns them, the axes).
    """
    ax = axes_in(fig, *box)
    fits = F1.panel_c(ax)
    leg = ax.get_legend()
    handles = leg.legend_handles
    labels = [t.get_text().replace("3-bit flip-flop", "flip-flop")
              .replace(F1.DMTS_LABEL, "DMTS").replace(", end of training", " (end)")
              .replace("  $", " $") for t in leg.get_texts()]
    leg.remove()
    ax.legend(handles, labels, loc="lower right", fontsize=MIN_PT, handlelength=1.6,
              handletextpad=0.4, borderaxespad=0.15, labelspacing=0.2)
    # the guide labels, which collide at 7 pt in this width, become short tags under the top edge,
    # each beside the guide it names (the diagonal leaves the axes at N = ylim, the 50% guide at
    # 2 x ylim, the 10% guide at 10 x ylim)
    y_top = ax.get_ylim()[1]
    tags = {"every unit active": ("all", 0.70), "50% active": ("50%", 1.62),
            "10% active": ("10%", 8.0)}
    for t in ax.texts:
        for old, (new, f) in tags.items():
            if t.get_text() == old:
                t.set_text(new)
                t.set_rotation(0)
                t.set_position((f * y_top, 0.80 * y_top))
                t.set_ha("right")
                t.set_va("center")
                t.set_color(ps.MUTED)
        if t.get_text() in ("all", "50%", "10%", "1,000 active units"):
            # a paper-coloured box so the guide lines do not strike through the text
            t.set_bbox(dict(facecolor=ps.PAPER, edgecolor="none", pad=0.4))
    # plain numbers at the sizes trained (the axis stays logarithmic: the fits are power laws)
    ax.set_xticks([100, 500, 1000, 2000, 4000])
    ax.set_xticklabels(["100", "500", "1000", "2000", "4000"])
    ax.set_yticks([100, 200, 500, 1000])
    ax.set_yticklabels(["100", "200", "500", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("network size $N$", labelpad=1)
    ax.set_ylabel("active units", labelpad=1)
    return fits, ax


def panel_D(fig, box):
    """Panel D: F1.panel_d unchanged on three short stacked axes, titles shortened, legend moved,
    one title over the stack.

    Args:
        fig: the figure; box: (x0, y0, w, h) in inches.
    Returns:
        (rows as F1.panel_d returns them, the top axes).
    """
    x0, y0, w, h = box
    n = len(F1.ABLATION)
    gap = 0.13
    hh = (h - gap * (n - 1)) / n
    axes = [axes_in(fig, x0, y0 + (n - 1 - i) * (hh + gap), w, hh) for i in range(n)]
    rows = F1.panel_d(axes)
    for ax, (label, _src, col) in zip(axes, F1.ABLATION):
        ax.set_title("")
        ax.text(0.99, 0.93, short_task(label), transform=ax.transAxes, ha="right", va="top",
                fontsize=MIN_PT, color=col)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "", "1"])
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_ylabel("")
    axes[-1].set_xlabel("units removed (%)", labelpad=1)
    axes[1].set_ylabel("held-out $R^2$", labelpad=1)
    fig.text((x0 + w / 2) / W_IN, (y0 + h + 0.05) / H_IN, "ablation at the loss floor",
             ha="center", va="bottom", fontsize=MIN_PT, color=ps.INK)
    return rows, axes


def legend_D(fig, box):
    """The line-style key of panel D, in its own blank axes.

    Args:
        fig: the figure; box: (x0, y0, w, h) in inches.
    Returns:
        the legend.
    """
    ax = axes_in(fig, *box)
    ps.blank(ax)
    return ax.legend(handles=[
        Line2D([], [], color=ps.INK, lw=1.2, label="quietest first"),
        Line2D([], [], color=ps.INK, lw=0.9, ls="--", label="random order"),
        Line2D([], [], color=ps.INK, lw=1.0, ls=":", label="busiest first"),
        Patch(color=ps.GRID, label="low-participation\nunits"),
        Line2D([], [], color=ps.INK, marker="*", ls="", ms=6,
               label="low-participation\nunits removed,\nretrained (3k iter.)")],
        loc="upper left", fontsize=MIN_PT, handlelength=1.6, handletextpad=0.4,
        borderaxespad=0.0, labelspacing=0.25, frameon=False)


def panel_E(fig, traj, box, tasks):
    """Panel E: noisy loss (relative to its start) and low-participation units over training.

    F1.panel_e's content on a linear iteration axis, one short axes per task, every seed: the dark
    lines are the noisy training loss (log-binned medians, left axis, log scale limited to the range
    the curves span), the coloured ones the count of units below the participation criterion (right
    axis); the dashed rule is the mean of the runs' own 1.05x-floor crossings, and the shaded span
    everything after it.

    Args:
        fig: the figure; traj: the trajectory record; box: (x0, y0, w, h) in inches;
        tasks: the TRAJ labels to draw, top to bottom.
    Returns:
        (rows of (task, N, n runs, mean crossing, silent there, silent at the end, held-out r2),
        list of (axes, twin axes)).
    """
    x0, y0, w, h = box
    cols = {lab: col for lab, _p, col, _N in F1.TRAJ}
    n = len(tasks)
    gap = 0.07
    hh = (h - gap * (n - 1)) / n
    rows, pairs = [], []
    x_max = max(max(r["it_end"] for r in traj[t]) for t in tasks)
    # loss axis: the range the drawn curves span (every curve starts at 1), shared by the tasks,
    # with a margin of a quarter on the log scale at each end
    lo = min(float(np.nanmin(r["loss_rel"])) for t in tasks for r in traj[t])
    hi = max(float(np.nanmax(r["loss_rel"])) for t in tasks for r in traj[t])
    for i, task in enumerate(tasks):
        ax = axes_in(fig, x0, y0 + (n - 1 - i) * (hh + gap), w, hh)
        axr = ax.twinx()
        runs = traj[task]
        col = cols[task]
        for r in runs:
            ax.plot(r["tb"] / 1e3, r["loss_rel"], "-", lw=0.8, color=ps.INK, alpha=0.75, zorder=4)
            axr.plot(r["it_act"][1:] / 1e3, r["silent"][1:], "-", lw=0.9, color=col, alpha=0.9,
                     zorder=5)
        at = [r for r in runs if np.isfinite(r["plateau"])]
        T = float(np.mean([r["plateau"] for r in at])) if at else float("nan")
        s_T = float(np.mean([r["silent"][np.argmin(np.abs(r["it_act"] - r["plateau"]))]
                             for r in at])) if at else float("nan")
        s_end = float(np.mean([r["silent"][-1] for r in (at or runs)]))
        if at:
            ax.axvline(T / 1e3, color=ps.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=2)
            ax.axvspan(T / 1e3, x_max / 1e3, color=ps.GRID, alpha=0.45, lw=0, zorder=0)
        N = runs[0]["N"]
        ax.set(yscale="log", ylim=(lo / 1.25, hi * 1.25), xlim=(0, x_max / 1e3))
        ax.set_yticks([0.1, 1])
        ax.set_yticklabels(["0.1", "1"])
        ax.minorticks_off()
        axr.set_ylim(0, N * 1.04)
        axr.set_yticks([0, N // 2, N])
        axr.spines[["right"]].set_visible(True)
        axr.spines[["top"]].set_visible(False)
        axr.tick_params(axis="y", colors=col)
        ax.text(0.98, 0.22, short_task(task), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=MIN_PT, color=col, zorder=6)
        if i < n - 1:
            ax.tick_params(labelbottom=False)
        rows.append((task, N, len(runs), len(at), T, s_T, s_end,
                     float(np.mean([r["r2"] for r in runs]))))
        pairs.append((ax, axr))
    pairs[-1][0].set_xlabel("training iteration (thousands)", labelpad=1)
    yc = (y0 + h / 2) / H_IN
    fig.text((x0 - 0.47) / W_IN, yc, "noisy loss / starting loss", rotation=90, ha="left",
             va="center", fontsize=MIN_PT)
    fig.text((x0 + w + 0.33) / W_IN, yc, "low-participation units", rotation=90, ha="left",
             va="center", fontsize=MIN_PT)
    return rows, pairs


def f_points(f2data, b5, ff75):
    """Every point of panel F, and the control each family is measured against.

    Figure 2 ladders and revival screens: every rung that is not its ladder's reference, one point
    per run read at its floor, y = its R^2 minus the mean R^2 of the reference rung at its floor.
    B5: every arm but the standard Adam (wd 1e-6) with a floor read-out, against that arm. The
    flip-flop remedies: the operating-point cell (fig_paper_F2.pick_cell) at N = 1000, end of 75k,
    against the 75k control at N = 1000.

    Args:
        f2data: the Figure 2 cache; b5: the B5 record; ff75: the 75k record.
    Returns:
        (points, controls, failed): points a list of dicts (family, task, readout, group, active,
        dr2); controls a list of dicts (task, readout, name, active, r2, n), one per distinct
        control; failed the B5 runs left out because they lost the task (R^2 at the floor below
        ff75k_readout.SOLVED), as (arm, active, R^2).
    """
    pts, ctrls, seen = [], [], set()
    for lad in f2data["ladders"] + f2data["revival"]:
        spec = lad["spec"]
        task = "flip-flop" if "flip" in spec["task"] else "CDDM"
        ref = lad["rungs"][spec["ref"]]
        r2_ref = float(np.mean(ref["read"]["r2"]))
        key = rel(ref["path"])
        if key not in seen:
            seen.add(key)
            ctrls.append(dict(task=task, readout="floor", name=f"{spec['stem']}: {key}",
                              active=float(np.mean(ref["read"]["floor"])), r2=r2_ref,
                              n=len(ref["read"]["floor"])))
        for i, rg in enumerate(lad["rungs"]):
            if i == spec["ref"] or rg["read"] is None:
                continue
            lab = rg["label"].replace("\n", " ")
            fam = next((f for (st, pre), f in RUNG_FAMILY.items()
                        if st == spec["stem"] and lab.startswith(pre)), "knob")
            for a, r2 in zip(rg["read"]["floor"], rg["read"]["r2"]):
                pts.append(dict(family=fam, task=task, readout="floor",
                                group=f"{spec['stem']}: {lab}", active=float(a),
                                dr2=float(r2) - r2_ref))

    std = [r for r in b5[B5_STANDARD] if np.isfinite(r["r2_floor"])]
    r2_std = float(np.mean([r["r2_floor"] for r in std]))
    ctrls.append(dict(task="flip-flop", readout="floor", name="B5: Adam wd 1e-6",
                      active=float(np.mean([r["act_floor"] for r in std])), r2=r2_std, n=len(std)))
    failed = []
    for arm, rows in b5.items():
        if arm == B5_STANDARD:
            continue
        for r in rows:
            if not np.isfinite(r["r2_floor"]):
                continue
            if r["r2_floor"] < FF75.SOLVED:      # lost the task (SGD lr 3): not a trade-off point
                failed.append(("B5: %s wd %s lr %s" % arm, float(r["act_floor"]),
                               float(r["r2_floor"])))
                continue
            pts.append(dict(family="optimizer", task="flip-flop", readout="floor",
                            group="B5: %s wd %s lr %s" % arm, active=float(r["act_floor"]),
                            dr2=float(r["r2_floor"]) - r2_std))

    at = F2.restrict(ff75, n_units=1000)
    ctrl = at["arm"] == "control"
    r2_c = float(np.mean(at["r2_common"][ctrl].astype(float)))
    ctrls.append(dict(task="flip-flop", readout="75k", name="75k control, N = 1000",
                      active=float(np.mean(at["n_active"][ctrl].astype(float))), r2=r2_c,
                      n=int(ctrl.sum())))
    for arm, name in REMEDIES:
        cell, _how = F2.pick_cell(at, arm)
        m = (at["arm"] == arm) & (at["cell"] == cell)
        for a, r2 in zip(at["n_active"][m], at["r2_common"][m]):
            pts.append(dict(family="drive", task="flip-flop", readout="75k", group=name,
                            active=float(a), dr2=float(r2) - r2_c))
    return pts, ctrls, failed


def panel_F(fig, pts, ctrls, box):
    """Panel F: every intervention at N = 1000, active units against R^2 minus its control's.

    One axes per task, side by side, CDDM left and 3-bit flip-flop right (DMTS has no knob or
    remedy runs), sharing both axes. Colour = family, marker = task (circle CDDM, triangle
    flip-flop), filled = read at the loss floor, open = end of 75k (flip-flop remedies, no floor read
    yet). Each control is a grey marker at (its mean active count, 0).

    Args:
        fig: the figure; pts, ctrls: from f_points; box: (x0, y0, w, h) in inches, the whole row
            (both axes and the gap between them).
    Returns:
        dict task -> axes, in F_TITLE order.
    """
    x0, y0, w, h = box
    gap = 0.12
    wa = (w - gap) / len(F_TITLE)
    axes = {}
    for k, (task, title) in enumerate(F_TITLE.items()):
        ax = axes_in(fig, x0 + k * (wa + gap), y0, wa, h)
        if k:
            ax.sharex(axes["CDDM"])
            ax.sharey(axes["CDDM"])
            ax.tick_params(labelleft=False)
        axes[task] = ax
        mk = TASK_MARK[task]
        ax.axhline(0, color=ps.MUTED, lw=0.6, zorder=1)
        ax.axvline(1000, color=ps.GRID, lw=0.6, zorder=0)
        for fam in ["knob", "revival", "optimizer", "drive"]:
            for ro in ("floor", "75k"):
                sel = [p for p in pts if p["family"] == fam and p["task"] == task
                       and p["readout"] == ro]
                if not sel:
                    continue
                col = FAM_COL[fam]
                ax.scatter([p["active"] for p in sel], [p["dr2"] for p in sel], marker=mk, s=14,
                           facecolors=col if ro == "floor" else "none", edgecolors=col,
                           linewidths=0.8, alpha=0.85, zorder=3)
        for c in ctrls:
            if c["task"] != task:
                continue
            ax.scatter([c["active"]], [0], marker=mk, s=26,
                       facecolors=ps.BASE if c["readout"] == "floor" else "white",
                       edgecolors="k", linewidths=0.7, zorder=5)
        ax.set_title(title, fontsize=MIN_PT, color=ps.INK, pad=2)
        ps.ygrid(ax)
    first = axes["CDDM"]
    first.set_xlim(-30, 1060)
    first.set_xticks([0, 250, 500, 750, 1000])
    first.set_ylabel("$R^2$ minus its control's", labelpad=1)
    fig.text((x0 + w / 2) / W_IN, 0.03 / H_IN, "active units (of 1000)", ha="center",
             va="bottom", fontsize=MIN_PT, color=ps.INK)
    return axes


# Where each named group's label sits on panel F, in data units: group -> (text, x, y, ha). Placed
# by hand in empty space beside the group, on the axes of the group's task (check_layout tests text
# against text; the render was read for text against points). Every label is joined to its group's
# centroid by a thin line. "rate L2 penalty" is Trainer's metabolic penalty, the mean squared firing
# rate (lambda_met); "homeostatic rate penalty" is its fr_magnitude penalty, each unit's deviation
# from a target rate (lambda_frm), the abstract's name for it.
F_LABELS = {
    "slide_k_frm_rws: rate $\\lambda$ = 0.2": ("homeostatic\nrate penalty", 960, 0.0070, "right"),
    "slide_k_frm_rws: both": ("rate + sparsity", 975, -0.0215, "right"),
    "slide_k_metabolic: 10.0": ("rate L2 penalty, $\\lambda$ = 10", 470, -0.0610, "left"),
    "rate + sparsity": ("rate + sparsity", 1010, -0.0410, "right"),
    "duplication": ("duplication", 830, -0.0130, "center"),
    "synaptic noise": ("synaptic noise", 560, -0.0330, "center"),
    "dropout": ("dropout", 450, -0.0235, "right"),
    "rescaling": ("rescaling", 330, -0.0150, "right"),
    "B5: sgd wd 1e-6 lr 0.03": ("SGD, lr 0.03", 790, -0.0640, "left"),
    "slide_k_inputscale: 20": ("input scale 20", 760, 0.0090, "center"),
}


def label_F(axes, pts):
    """Name the participation drives and a few landmark groups on panel F, each on its task's axes.

    Args:
        axes: dict task -> axes, from panel_F; pts: from f_points.
    Returns:
        list of (text, centroid x, centroid y) for the groups labelled.
    """
    groups = {}
    for p in pts:
        groups.setdefault(p["group"], []).append(p)
    out = []
    for g, (text, x, y, ha) in F_LABELS.items():
        if g not in groups:
            print(f"  !! panel F label: no group {g!r}")
            continue
        sel = groups[g]
        cx = float(np.mean([p["active"] for p in sel]))
        cy = float(np.mean([p["dr2"] for p in sel]))
        col = FAM_COL[sel[0]["family"]]
        axes[sel[0]["task"]].annotate(
            text, xy=(cx, cy), xytext=(x, y), ha=ha, va="center", fontsize=MIN_PT, color=col,
            zorder=6, arrowprops=dict(arrowstyle="-", lw=0.5, color=col, shrinkA=1, shrinkB=3))
        out.append((text, cx, cy))
    return out


def legend_F(fig, box, failed):
    """The key of panel F (family colour, read-out fill, control) in a blank axes.

    The task is named over each of F's axes, so the key does not repeat the task markers.

    Args:
        fig: the figure; box: (x0, y0, w, h) in inches; failed: f_points' failed runs, named in a
            note under the key.
    Returns:
        the legend.
    """
    ax = axes_in(fig, *box)
    ps.blank(ax)
    mk = dict(ls="", ms=4.2, mew=0.8)
    h = [Line2D([], [], marker="s", color=FAM_COL[f], label=FAM_LABEL[f], **mk)
         for f in ("knob", "revival", "optimizer", "drive")]
    h += [Line2D([], [], marker="o", color=ps.INK, label="filled: at the loss floor", **mk),
          Line2D([], [], marker="o", color=ps.INK, mfc="none",
                 label="open: flip-flop interventions,\nend of 75k (no floor read yet)", **mk),
          Line2D([], [], marker="o", mfc=ps.BASE, mec="k", label="its control, at 0", **mk)]
    leg = ax.legend(handles=h, loc="upper left", fontsize=MIN_PT, handlelength=1.0,
                    handletextpad=0.4, borderaxespad=0.0, labelspacing=0.3, frameon=False)
    if failed:
        r2 = [f[2] for f in failed]
        span = f"{min(r2):.2f}" if len(r2) == 1 else f"{min(r2):.2f}-{max(r2):.2f}"
        ax.text(0.0, 0.0, f"not drawn: SGD lr 3, which loses\nthe task ($R^2$ {span} at its floor)",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=MIN_PT, color=ps.MUTED)
    return leg


def panel_G(fig, ff75, box, mode):
    """Panel G: the flip-flop remedies' active units against network size, end of 75k.

    mode "percent": the percentage of the network that is active, 100 x mean active / N, for the
    control (grey) and each remedy. mode "ratio": each remedy's mean active count over the
    control's at the same N. Either way the remedy cell is the operating point
    (fig_paper_F2.pick_cell), each remedy has its own colour (REMEDY_COL), and its end label is
    printed in that colour.

    Args:
        fig: the figure; ff75: the 75k record; box: (x0, y0, w, h) in inches;
        mode: "percent" or "ratio".
    Returns:
        (rows of (series, N, n networks, value drawn, n control networks), the axes). The value is
        a percentage of N in "percent" mode and a multiple of the control in "ratio" mode.
    """
    ax = axes_in(fig, *box)
    rows, ends = [], []
    series = ([("control", "control")] if mode == "percent" else []) + REMEDIES
    for arm, name in series:
        xs, ys = [], []
        for N in F2.SIZES:
            at = F2.restrict(ff75, n_units=N)
            ctrl = at["arm"] == "control"
            if not ctrl.any() or not (at["arm"] == arm).any():
                continue
            if arm == "control":
                m = ctrl
            else:
                cell, _how = F2.pick_cell(at, arm)
                m = (at["arm"] == arm) & (at["cell"] == cell)
            act = float(np.mean(at["n_active"][m].astype(float)))
            if mode == "percent":
                val = 100.0 * act / N
            else:
                val = act / float(np.mean(at["n_active"][ctrl].astype(float)))
            xs.append(N)
            ys.append(val)
            rows.append((name, N, int(m.sum()), val, int(ctrl.sum())))
        col = REMEDY_COL.get(name, ps.BASE)
        ax.plot(xs, ys, "-o", color=col, mfc=col if arm == "control" else "white", mec=col,
                ms=3.4, lw=1.3 if arm == "control" else 0.9, zorder=4 if arm == "control" else 3)
        ends.append((ys[-1], name, xs[-1]))
    ax.set_xscale("log")
    ax.set_xticks(list(F2.SIZES))
    ax.set_xticklabels(["500", "1k", "2k", "4k"])
    ax.minorticks_off()
    ax.set_xlim(420, 4800)
    if mode == "percent":
        ax.set_ylim(0, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("active units (% of network)", labelpad=1)
    else:
        ax.axhline(1, color=ps.MUTED, lw=0.6, zorder=1)
        ax.set_ylim(0.6, 7.3)
        ax.set_yticks([1, 2, 4, 6])
        ax.set_ylabel("active units ÷ control's", labelpad=1)
    ax.set_xlabel("network size $N$", labelpad=1)
    ps.ygrid(ax)
    # end labels, pushed apart to one text height (in data units of this axes) from the bottom up
    fig.canvas.draw()
    (_, ya), (_, yb) = ax.transData.transform([(1000, 0), (1000, 1)])
    sep = (MIN_PT * 1.4 / 72 * fig.dpi) / (yb - ya)
    placed = []
    for y, name, x in sorted(ends):
        y_lab = max(y, placed[-1] + sep) if placed else y
        placed.append(y_lab)
        col = REMEDY_COL.get(name, ps.BASE)
        # the leader leaves the label's left-middle (relpos), not its centre, so a label pushed up
        # past its neighbours does not draw its leader through the neighbour's label
        ax.annotate(name, xy=(x, y), xytext=(x * 1.18, y_lab), ha="left", va="center",
                    fontsize=MIN_PT, color=col, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=col, shrinkA=0, shrinkB=2,
                                    relpos=(0.0, 0.5)))
    ax.set_title("flip-flop remedies, end of 75k\n(no floor read yet)", loc="left",
                 fontsize=MIN_PT, color=ps.MUTED, pad=3)
    return rows, ax


# ------------------------------------------------------------------------------------------------
# checks
# ------------------------------------------------------------------------------------------------

def enforce_min_font(fig, pt=MIN_PT):
    """Raise every text artist on the figure to at least `pt` points (the reused F1 panels use 5-6.6).

    Args:
        fig: the figure; pt: the floor.
    Returns:
        the number of artists raised.
    """
    n = 0
    for t in fig.findobj(Text):
        if t.get_text() and t.get_fontsize() < pt:
            t.set_fontsize(pt)
            n += 1
    return n


def check_layout(fig):
    """Report overlapping texts, text outside the canvas, overlapping panels, and unused canvas.

    Texts: every drawn text (fig_paper_F2_fails.drawn_texts: titles, labels, legend entries, free
    text, tick labels inside the view limits), tight boxes, pairwise. Panels: the tight boxes of
    axes that are not twins of each other. Fill: the union of every axes' tight box against the
    canvas, as the blank margin on each side.

    Args:
        fig: the drawn figure.
    Returns:
        number of problems (overlaps + texts outside the canvas).
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    bad = 0
    texts = F2F.drawn_texts(fig)
    # drawn_texts reads only the centre title and the first tick label of each tick; add the left
    # and right titles and the second tick labels (a twin's right-hand axis) that are drawn
    for ax in fig.axes:
        texts += [t for t in (ax._left_title, ax._right_title) if t.get_text().strip()]
        for axis, (lo, hi) in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lo, hi), max(lo, hi)
            texts += [tk.label2 for tk in axis.get_major_ticks()
                      if lo <= tk.get_loc() <= hi and tk.label2.get_visible()
                      and tk.label2.get_text()]
    small = [t for t in texts if t.get_fontsize() < MIN_PT - 1e-6]
    for t in small:
        bad += 1
        print(f"  SMALL ({t.get_fontsize():.1f} pt): '{t.get_text()[:40]}'")
    # Text.get_window_extent, not the artist's own: an annotation's box includes its leader line
    boxes = [(t, Text.get_window_extent(t, rend)) for t in texts]
    for t, b in boxes:
        if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H:
            bad += 1
            print(f"  OUTSIDE: '{t.get_text()[:40]}'")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                bad += 1
                bi, bj = boxes[i][1], boxes[j][1]
                print(f"  OVERLAP: '{boxes[i][0].get_text()[:30]}' x '{boxes[j][0].get_text()[:30]}'"
                      f" at ({bi.x0 / fig.dpi:.2f}, {bi.y0 / fig.dpi:.2f}) in, by "
                      f"{(min(bi.x1, bj.x1) - max(bi.x0, bj.x0)) / fig.dpi:.2f} x "
                      f"{(min(bi.y1, bj.y1) - max(bi.y0, bj.y0)) / fig.dpi:.2f} in")
    axes = [a for a in fig.axes if a.get_visible()]
    tb = {a: a.get_tightbbox(rend) for a in axes}
    for i, a in enumerate(axes):
        for b in axes[i + 1:]:
            twins = (a.get_position().bounds == b.get_position().bounds
                     or a in b.get_shared_x_axes().get_siblings(b))
            if not twins and tb[a] is not None and tb[b] is not None and tb[a].overlaps(tb[b]):
                ov = (min(tb[a].x1, tb[b].x1) - max(tb[a].x0, tb[b].x0),
                      min(tb[a].y1, tb[b].y1) - max(tb[a].y0, tb[b].y0))
                if min(ov) > 1.0:       # one pixel of touching is not an overlap
                    pa, pb = (np.round(np.array(x.get_position().bounds) * [W_IN, H_IN, W_IN, H_IN], 2)
                              for x in (a, b))
                    print(f"  PANELS TOUCH: axes at {pa.tolist()} x {pb.tolist()} in "
                          f"({ov[0] / fig.dpi:.2f} x {ov[1] / fig.dpi:.2f} in)")
    x0 = min(b.x0 for b in tb.values() if b is not None) / fig.dpi
    x1 = max(b.x1 for b in tb.values() if b is not None) / fig.dpi
    y0 = min(b.y0 for b in tb.values() if b is not None) / fig.dpi
    y1 = max(b.y1 for b in tb.values() if b is not None) / fig.dpi
    print(f"  content spans x {x0:.2f}-{x1:.2f} in of {W_IN}, y {y0:.2f}-{y1:.2f} in of {H_IN}")
    print(f"layout check: {len(texts)} texts, {bad} problem(s)")
    return bad


# ------------------------------------------------------------------------------------------------
# assembly
# ------------------------------------------------------------------------------------------------

def draw(g_mode):
    """Assemble the figure from the caches, check it, save it, and print every number drawn.

    Args:
        g_mode: panel G's y axis, "percent" or "ratio" (see G_MODE). The figure with g_mode ==
            G_MODE is saved as fig_cosyne2027, the other as fig_cosyne2027_G<g_mode>.
    Returns:
        the PDF path.
    """
    with open(DELLA_CACHE, "rb") as fh:
        della = pickle.load(fh)
    with open(B5_CACHE, "rb") as fh:
        b5 = pickle.load(fh)
    with open(F2F.CACHE, "rb") as fh:
        f2data = pickle.load(fh)
    install_replays(della)
    ex = della["example"]

    ps.setup()
    plt.rcParams.update({"font.size": MIN_PT, "axes.labelsize": MIN_PT, "axes.titlesize": MIN_PT,
                         "xtick.labelsize": MIN_PT, "ytick.labelsize": MIN_PT,
                         "legend.fontsize": MIN_PT})
    fig = plt.figure(figsize=(W_IN, H_IN))

    # row 1: A | B | C
    r1y, r1h = 4.70, 1.62
    n_live, n_shown_live, n_conn, ax_a = panel_A(fig, ex, (0.14, r1y, 2.62, r1h))
    (n_init, n_trained), ax_b = panel_B(fig, ex, (3.25, r1y + 0.04, 1.70, r1h - 0.16))
    fits, ax_c = panel_C(fig, (5.50, r1y + 0.04, 1.93, r1h - 0.16))

    # row 2: D (+ its key) | E | G
    r2y, r2h = 2.66, 1.50
    rows_d, axes_d = panel_D(fig, (0.40, r2y, 1.38, r2h))
    legend_D(fig, (1.86, r2y + 0.12, 1.00, r2h - 0.12))
    rows_e, pairs_e = panel_E(fig, della["trajectory"], (3.60, r2y, 1.38, r2h),
                              ["3-bit flip-flop", "CDDM"])
    rows_g, ax_g = panel_G(fig, della["ff75k"], (5.92, r2y, 0.70, r2h - 0.28), g_mode)

    # row 3: F (+ its key)
    pts, ctrls, failed = f_points(f2data, b5, della["ff75k"])
    axes_f = panel_F(fig, pts, ctrls, (0.52, 0.32, 4.80, 1.88))
    label_F(axes_f, pts)
    legend_F(fig, (5.45, 0.32, 1.98, 1.88), failed)

    row2_top, row3_top = r2y + r2h + 0.24, 0.32 + 1.88 + 0.22
    for s, x, top in (("A", 0.0, H_IN - 0.01), ("B", 2.86, H_IN - 0.01), ("C", 5.02, H_IN - 0.01),
                      ("D", 0.0, row2_top), ("E", 3.05, row2_top), ("G", 5.36, row2_top),
                      ("F", 0.0, row3_top)):
        letter(fig, s, x, top)

    raised = enforce_min_font(fig)
    print(f"raised {raised} text artists to {MIN_PT} pt")
    check_layout(fig)
    out = ps.save(fig, "fig_cosyne2027" if g_mode == G_MODE else f"fig_cosyne2027_G{g_mode}")
    plt.close(fig)

    print("\n--- numbers drawn ---")
    print(f"A: {n_live} of 1000 active; {n_shown_live} of {F1.N_SHOWN} drawn units active")
    print(f"B: {n_init} of 1000 active untrained, {n_trained} after {ex['iterations']:,}")
    for task, (b, A, need, n_sizes, n_max, readout) in fits.items():
        print(f"C: {task}: read at {readout}, M = {A:.2f} N^{b:.3f}, {n_sizes} sizes up to {n_max:.0f}")
    for row in rows_d:
        print("D: %s %d nets, %.0f silent, R2 intact %.3f, silent removed %.3f, same number random "
              "%.3f, active removed %.3f, read-out %s at iteration %d, retrained %.3f" % row)
    for row in rows_e:
        print("E: %s N=%d, %d runs (%d with a floor), crossing %.0f, silent there %.0f, at end %.0f,"
              " held-out r2 %.3f" % row)
    for row in rows_g:
        if g_mode == "percent":
            print("G: %s N=%d n=%d %.1f%% of units active (control n=%d)" % row)
        else:
            print("G: %s N=%d n=%d multiplier %.2f (control n=%d)" % row)
    for c in ctrls:
        print(f"F control: {c['name']}: {c['active']:.0f} active, R2 {c['r2']:.4f}, n={c['n']}, "
              f"{c['readout']}")
    groups = {}
    for p in pts:
        groups.setdefault((p["family"], p["group"], p["readout"]), []).append(p)
    for (fam, g, ro), sel in groups.items():
        a = [p["active"] for p in sel]
        d = [p["dr2"] for p in sel]
        print(f"F {fam:9s} {ro:5s} {g[:60]:60s} n={len(sel)} active {min(a):.0f}-{max(a):.0f} "
              f"dR2 {min(d):+.4f}..{max(d):+.4f}")
    for arm, a, r2 in failed:
        print(f"F not drawn (lost the task): {arm}: {a:.0f} active, R2 {r2:.3f}")
    return out


def main():
    """Run one stage: --collect-della, --collect-b5, or (no flag) draw, once per panel G mode."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collect-della", action="store_true")
    ap.add_argument("--collect-b5", action="store_true")
    a = ap.parse_args()
    if a.collect_della:
        collect_della()
    elif a.collect_b5:
        collect_b5()
    else:
        for g_mode in G_MODES:
            print(f"\n===== panel G: {g_mode} =====")
            draw(g_mode)


if __name__ == "__main__":
    main()
