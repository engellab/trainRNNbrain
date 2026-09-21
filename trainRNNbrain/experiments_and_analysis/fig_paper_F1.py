#!/usr/bin/env python3
"""
Manuscript Figure 1 - THE PROBLEM. Most units of a trained ReLU RNN never fire; the bigger the
network the worse it gets; it happens on every task; and no standard knob fixes it.

This is the motivation figure, so it has to do four things in one display item, and the first row
has to explain the measurement before the second and third rows quantify it:

  (a) WHAT IS BEING MEASURED           a trained 3-bit flip-flop network, drawn as its units, with
                                       three REAL units pulled out of it: one strongly driven, one
                                       barely above threshold, one that never leaves zero. No
                                       cartoon traces - these are simulated from a trained network.
  (b) WHY "SILENT" IS NOT A JUDGEMENT  the participation distribution of that same network on a log
                                       axis. It is bimodal with four orders of magnitude of empty
                                       valley between the modes, so the threshold is read off the
                                       data rather than chosen; the pictogram states the answer.
  (c) SIZE MAKES IT WORSE              active units vs N on two tasks. The count grows as roughly
                                       N^0.46, so the FRACTION falls: 41% of a 500-unit network,
                                       13% of a 4,000-unit one. Extrapolating, 1,000 active units
                                       would need N ~ 13,000.
  (d) IT IS NOT ONE TASK               active fraction at N = 1000 on six tasks, including the
                                       20-task multitask family. None reaches half the network.
  (e) NO KNOB FIXES IT                 every intervention we tried, as a change from its OWN matched
                                       reference. The best moves the count by ~120 units; several
                                       make it worse; weight decay is a monotone poison. For scale,
                                       the rate penalty of Figure 3 moves it by ~710.

CRITERION. Everything here is the scale-free participation criterion (a unit is silent below 5% of
its own network's 95th-percentile participation), which is the only criterion that travels across
tasks and activation functions - the absolute thresholds are calibrated per task and comparing
across them is how this project once reported a rescue that was not there. Panel (b) shows why the
scale-free rule lands in the valley rather than on a mode.

READ-OUT DISCIPLINE. Every cell is read at the largest iteration EVERY seed in that cell reaches
(matched compute), never at each run's own endpoint - reading a slow big network at its end and a
fast small one at its end confounds size with convergence depth, which is how the k-exponent in an
earlier version of this project came out positive.

Usage:  python fig_paper_F1.py [--refresh]      (--refresh re-simulates the example network)
Output: img/internal_figures/fig_paper_F1.png
"""

import argparse
import csv
import glob
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from common import DATA_DIR, SILENT_REL, participation
from flipflop_diversity import rates_and_targets

CACHE = "data/fig_paper_F1_cache.npz"
# The example network is one of the 150,000-iteration unpenalised runs, NOT one of the 500,000-
# iteration k-sweep runs, so that the live count shown in panel (a) is the same quantity panel (d)
# reports for this task (~263/1000). The longer runs sit at ~195 and would contradict the panel
# below them for no reason other than the budget they happened to be trained for.
EXAMPLE_NET = "data/trained_RNNs/NBitFlipFlop_std_dropout/EqType=h_k=3_N=1000_pen=none_do=none"
N_TRIALS = 24

# (label, glob, read-out cap). The cap is the iteration the manuscript reads that family at; the
# actual read-out is min(cap, the last iteration every seed reaches), reported on the panel.
FF = f"{DATA_DIR}/NBitFlipFlop_std_ksweep/EqType=h_k=3_N={{N}}_iters=*"
SCALING = {
    "3-bit flip-flop": ([500, 1000, 2000, 4000], {
        500:  f"{FF.format(N=500)}",
        1000: f"{FF.format(N=1000)}",
        2000: f"{FF.format(N=2000)}",
        4000: f"{DATA_DIR}/NBitFlipFlop_std_bigN/EqType=h_k=3_N=4000_pen=none_iters=*",
    }, 100_000, ps.SLOTS[0]),
    "CDDM": ([500, 1000, 2000, 5000], {
        N: f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N={N}_iters=*" for N in (500, 1000, 2000, 5000)
    }, 100_000, ps.SLOTS[1]),
}

TASKS = [
    ("3-bit flip-flop",   f"{DATA_DIR}/NBitFlipFlop_std_ksweep/EqType=h_k=3_N=1000_iters=*", 150_000),
    ("CDDM",              f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N=1000_iters=*", 150_000),
    ("DMTS (36τ delay)",  f"{DATA_DIR}/DMTS_std_delay36/EqType=h_N=1000_pen=none", 150_000),
    ("DMTS (16τ delay)",  f"{DATA_DIR}/DMTS_std_pen/EqType=h_N=1000_pen=none", 150_000),
    ("Yang, context DM",  f"{DATA_DIR}/Yang_std_30k/EqType=h_set=ctxdm_N=1000_pen=none", 150_000),
    ("Yang, 20 tasks",    f"{DATA_DIR}/Yang_std_30k/EqType=h_set=multi20_N=1000_pen=none", 150_000),
    ("hyper flip-flop",   f"{DATA_DIR}/NBitFlipFlopHyper_std_hyper/EqType=h_k=4_N=1000_pen=none", 150_000),
]

# Interventions, grouped by task family. Each family carries its OWN reference, and the panel plots
# a CHANGE from that reference, because the two families differ in task and read-out iteration and
# an absolute bar chart across them would invite a comparison the data does not support.
INTERVENTIONS = [
    ("CDDM, N = 1000", f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N=1000_iters=*", None, [
        ("leaky ReLU",             f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=leakyrelu_iters=*",  "activation"),
        ("softplus (β=25)",        f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=softplus25_iters=*", "activation"),
        ("bounded sigmoid",        f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=sigmoid_iters=*",    "activation"),
        ("weight decay 0",         f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=0_iters=*",           "weight decay"),
        ("weight decay 10⁻⁵",      f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=1e-5_iters=*",        "weight decay"),
        ("weight decay 10⁻⁴",      f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=1e-4_iters=*",        "weight decay"),
    ]),
    ("3-bit flip-flop, N = 1000", f"{DATA_DIR}/NBitFlipFlop_std_ksweep/EqType=h_k=3_N=1000_iters=*", 150_000, [
        ("bounded sigmoid",        f"{DATA_DIR}/NBitFlipFlop_std_sigmoid/EqType=h_k=3_N=1000_iters=*",   "activation"),
        ("input weights ×0.5",     f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=0.5_iters=*", "input scale"),
        ("input weights ×2",       f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=2_iters=*",   "input scale"),
        ("input weights ×5",       f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=5_iters=*",   "input scale"),
        ("input weights ×20",      f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=20_iters=*",  "input scale"),
    ]),
]

# Interventions measured on the 2026-07 architecture, where no participation trace was written, so
# they cannot join the panel above. They are read from the per-condition CSVs under a peak-rate
# scale-free rule and are reported in the Supplementary, not here. Kept so the list is not lost.
CSV_ONLY = [("recurrent noise σ_rec ∈ {0, .01, .05, .1}", "CDDM_fb2792_g0_noise"),
            ("self-connections off", "silent_stats_all.csv: sweep nodale_bias"),
            ("bias fixed at 0", "silent_stats_all.csv: sweep nodale"),
            ("metabolic penalty λ ∈ {.01, .1, 1, 10}", "silent_stats_v2.csv: sweep metabolic")]

GROUP_COL = {"activation": ps.SLOTS[3], "weight decay": ps.SLOTS[4],
             "input scale": ps.SLOTS[2], "metabolic": ps.SLOTS[1]}


def traces_of(pattern):
    """Every (participation matrix, iteration vector) pair under a run-folder glob.

    Args:
        pattern: glob matching run folders (not the pickles themselves).
    Returns:
        list of (P, iters): P is (n_probes, N) participation, iters is (n_probes,).
    """
    out = []
    for f in sorted(glob.glob(os.path.join(pattern, "*", "*ParticipationTrace.pkl"))):
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        P, it = np.asarray(d.get("participation", [])), np.asarray(d.get("participation_iters", []))
        if len(it) and P.ndim == 2:
            out.append((P, it))
    return out


def live_matched(pattern, cap=None):
    """Active units per seed at the largest iteration every seed in the cell reaches.

    Matched compute, not each run's own endpoint: a big network read at its end and a small one read
    at its end differ in convergence depth as well as size, which confounds the very comparison the
    scaling panel makes.

    Args:
        pattern: glob matching run folders; cap: read no later than this iteration, or None for the
            deepest shared probe.
    Returns:
        (counts, iteration) with counts an (n_seeds,) int array, or None if the cell is empty.
    """
    tr = traces_of(pattern)
    if not tr:
        return None
    shared = min(int(it[-1]) for _, it in tr)
    it_read = shared if cap is None else min(shared, cap)
    counts = []
    for P, it in tr:
        p = P[int(np.argmin(np.abs(it - it_read)))]
        counts.append(int((p >= SILENT_REL * np.quantile(p, 0.95)).sum()))
    return np.array(counts), it_read


def example_network(refresh=False):
    """Rates, targets and participation of one trained unpenalised flip-flop network.

    Simulated noise-free from the trained weights, then cached, because panels (a) and (b) must show
    a real network rather than an illustration and the simulation costs ~30 s.

    Args:
        refresh: re-simulate even if the cache exists.
    Returns:
        (rates, targets, p): (N, T, B) rates, (k, T, B) target bits, (N,) participation.
    """
    if os.path.exists(CACHE) and not refresh:
        z = np.load(CACHE)
        return z["rates"], z["targets"], z["p"]
    folder = sorted(glob.glob(os.path.join(EXAMPLE_NET, "*/")))[0]
    rates, targets = rates_and_targets(folder, n_trials=N_TRIALS)
    p = participation(rates)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, rates=rates.astype(np.float32),
                        targets=np.asarray(targets, np.float32), p=p)
    return rates.astype(np.float32), np.asarray(targets, np.float32), p


def panel_a(ax, rates, targets, p):
    """Panel (a): the network as its units, with three real units pulled out of it.

    Args:
        ax: a blank axes spanning the panel; rates: (N, T, B); targets: (k, T, B); p: (N,).
    Returns:
        None.
    """
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(-0.09, 1.0))
    ax.figure.canvas.draw()                    # square_pitch needs the axes already laid out
    thr = SILENT_REL * np.quantile(p, 0.95)
    n_live = int((p >= thr).sum())
    order = np.argsort(-p)

    # --- left: the task, the network, the read-out -------------------------------------------
    ps.box(ax, 0.002, 0.46, 0.112, 0.22, col=ps.MUTED, face="#f2f1ec", lw=0.6)
    ax.text(0.058, 0.605, "3 bits", ha="center", va="center", fontsize=6.0, color=ps.INK)
    ax.text(0.058, 0.515, "set / reset", ha="center", va="center", fontsize=5.0, color=ps.MUTED)
    ax.text(0.058, 0.705, "input", ha="center", fontsize=5.8, color=ps.MUTED)

    # the recurrent pool, drawn as 100 glyphs filled to the measured live fraction
    dx = 0.0235
    dy = ps.square_pitch(ax, dx)
    gx, gy = 0.155, 0.84
    w_grid, h_grid = ps.unit_grid(ax, gx, gy, round(n_live / 10), 100, col=ps.SLOTS[0],
                                  off_col="#d5d4cc", pitch=(dx, dy), s=4.6, lw=0.4)
    ax.text(gx + w_grid / 2, gy + 0.055, "recurrent pool, N = 1000",
            ha="center", fontsize=6.2, color=ps.INK)
    ax.text(gx + w_grid / 2, gy - h_grid - 0.105,
            f"{n_live} of 1000 active ({n_live / 1000:.0%})",
            ha="center", fontsize=6.2, color=ps.SLOTS[0])
    ax.text(gx + w_grid / 2, gy - h_grid - 0.175, "one glyph = 10 units",
            ha="center", fontsize=5.2, color=ps.FAINT)

    bx = gx + w_grid + 0.055
    ps.box(ax, bx, 0.46, 0.115, 0.22, col=ps.MUTED, face="#f2f1ec", lw=0.6)
    ax.text(bx + 0.0575, 0.605, "3", ha="center", va="center", fontsize=6.0, color=ps.INK)
    ax.text(bx + 0.0575, 0.515, "read-outs", ha="center", va="center", fontsize=5.3, color=ps.MUTED)
    ax.text(bx + 0.0575, 0.705, "output", ha="center", fontsize=5.8, color=ps.MUTED)

    ps.arrow(ax, (0.118, 0.57), (gx - 0.014, 0.57), col=ps.MUTED)
    ps.arrow(ax, (gx + w_grid + 0.016, 0.57), (bx - 0.008, 0.57), col=ps.MUTED)

    # --- right: three real units ---------------------------------------------------------------
    picks = [(order[0], "strongly driven", ps.SLOTS[0]),
             (order[n_live - 12], "just above threshold", ps.SLOTS[2]),
             (order[600], "never leaves zero", ps.BAD)]
    x0, xw = 0.695, 0.215
    tt = np.linspace(0, 1, rates.shape[1])
    trial = int(np.argmax(rates[order[0]].max(axis=0)))

    # the task's target bits, as context for the traces below
    yb = 0.885
    for b in range(targets.shape[0]):
        sig = np.asarray(targets[b, :, trial], float)
        ax.plot(x0 + xw * tt, yb + 0.030 * b + 0.021 * sig, lw=0.6, color=ps.FAINT, zorder=2)
    ax.text(x0 - 0.010, yb + 0.045, "target bits", ha="right", va="center", fontsize=5.5,
            color=ps.MUTED)

    for j, (u, lab, col) in enumerate(picks):
        base = 0.545 - 0.255 * j
        r = np.asarray(rates[u, :, trial], float)
        ax.plot(x0 + xw * tt, base + 0.175 * r / max(r.max(), 1e-9), lw=0.8, color=col, zorder=3)
        ax.plot([x0, x0 + xw], [base, base], lw=0.4, color=ps.FAINT, zorder=1)
        ax.text(x0 - 0.010, base + 0.075, lab, ha="right", va="center", fontsize=5.7, color=col)
        ax.text(x0 + xw + 0.010, base + 0.100, f"peak {r.max():.2g}", ha="left", va="center",
                fontsize=5.3, color=ps.FAINT)
        ax.text(x0 + xw + 0.010, base + 0.035, f"$p$ = {p[u]:.2g}", ha="left", va="center",
                fontsize=5.3, color=ps.MUTED)
    ax.text(x0 + xw / 2, -0.055, "time (one trial)", ha="center", fontsize=5.6, color=ps.MUTED)
    ax.text(x0 + xw / 2, 0.985, "three real units of that network", ha="center", fontsize=6.2,
            color=ps.INK)


def panel_b(ax, p):
    """Panel (b): the participation distribution of the same network, on a log axis.

    Args:
        ax: axes; p: (N,) participation values.
    Returns:
        None.
    """
    thr = SILENT_REL * np.quantile(p, 0.95)
    pp = np.maximum(p, 1e-6)
    bins = np.logspace(np.log10(pp.min() * 0.7), np.log10(pp.max() * 1.4), 46)
    live = pp >= thr
    ax.hist(pp[~live], bins=bins, color=ps.FAINT, edgecolor="none", label=f"silent ({(~live).sum()})")
    ax.hist(pp[live], bins=bins, color=ps.SLOTS[0], edgecolor="none", label=f"active ({live.sum()})")
    ax.axvline(thr, color=ps.INK, lw=0.9, ls="--", zorder=5)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.28)
    ax.annotate("criterion\n$p_i < 0.05\\,q_{95}(p)$", xy=(thr, top * 1.02),
                xytext=(thr * 26, top * 1.16), fontsize=5.8, color=ps.INK,
                ha="center", va="center", linespacing=1.25,
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=ps.INK, shrinkA=1, shrinkB=2))
    ax.set_xscale("log")
    ax.set(xlabel="participation  $p_i=\\mathrm{std}(r_i)+q_{0.9}(|r_i|)$", ylabel="units")
    ax.annotate("", xy=(6e-4, top * 0.26), xytext=(0.9, top * 0.26),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.55, color=ps.MUTED, mutation_scale=6))
    ax.text(0.023, top * 0.305, "3.5 decades\nof empty valley", ha="center",
            fontsize=5.6, color=ps.MUTED, linespacing=1.2)
    ax.legend(loc="upper left", fontsize=5.8, bbox_to_anchor=(0.0, 1.0))
    ps.ygrid(ax)


def panel_c(ax):
    """Panel (c): active units vs network size, both tasks, with the fitted power law.

    Returns:
        dict task -> (b, A, N_needed_for_1000) of the fit.
    """
    fits, handles = {}, []
    for task, (Ns, pats, cap, col) in SCALING.items():
        xs, ys, sds = [], [], []
        for N in Ns:
            got = live_matched(pats[N], cap)
            if got is None:
                continue
            c, _ = got
            xs.append(N)
            ys.append(c.mean())
            sds.append(c.std(ddof=1) if len(c) > 1 else 0.0)
            ax.plot([N] * len(c), c, "o", ms=2.4, color=col, alpha=0.55, mec="none", zorder=4)
        xs, ys, sds = np.array(xs, float), np.array(ys, float), np.array(sds, float)
        ax.errorbar(xs, ys, yerr=sds, fmt="o-", color=col, ms=3.4, lw=1.1, zorder=5, capsize=1.6)
        b, loga = np.polyfit(np.log(xs), np.log(ys), 1)
        fits[task] = (b, np.exp(loga), np.exp((np.log(1000) - loga) / b))
        xf = np.logspace(np.log10(xs.min() * 0.85), np.log10(2.4e4), 50)
        ax.plot(xf, np.exp(loga) * xf ** b, ls=":", lw=0.8, color=col, zorder=3)
        # built by hand: an errorbar's legend handle is a container, and letting matplotlib collect
        # handles here silently produced two entries for the same task
        handles.append(Line2D([], [], color=col, marker="o", ms=3.0, lw=1.1,
                              label=f"{task}  $\\propto N^{{{b:.2f}}}$"))

    nn = np.array([3e2, 2.6e4])
    ax.plot(nn, nn, "-", lw=0.7, color=ps.MUTED, zorder=2)
    ax.text(5.0e3, 6.4e3, "every unit active", fontsize=5.5, color=ps.MUTED, rotation=38,
            ha="center", va="bottom")
    for frac, lab in [(0.5, "50% active"), (0.1, "10% active")]:
        ax.plot(nn, frac * nn, ls=(0, (4, 3)), lw=0.55, color=ps.FAINT, zorder=1)
        ax.text(2.45e4, frac * 2.45e4, lab, fontsize=5.2, color=ps.FAINT, ha="right", va="bottom")
    ax.axhline(1000, color=ps.BAD, lw=0.7, ls="-.", zorder=2)
    need = np.mean([v[2] for v in fits.values()])
    ax.text(3.4e2, 1120, f"1,000 active units\nwould need N ≈ {round(need, -3):,.0f}",
            fontsize=5.9, color=ps.BAD, va="bottom", linespacing=1.25)
    ax.set(xscale="log", yscale="log", xlabel="network size N", ylabel="active units",
           xlim=(3.2e2, 2.7e4), ylim=(140, 3.4e4))
    ax.legend(handles=handles, loc="lower right", fontsize=5.9)
    ps.ygrid(ax)
    return fits


def panel_d(ax):
    """Panel (d): active fraction at N = 1000 across tasks. Returns the measured rows."""
    rows = []
    for label, pat, cap in TASKS:
        got = live_matched(pat, cap)
        if got is None:
            continue
        c, it = got
        rows.append((label, c / 1000.0, it))
    rows.sort(key=lambda r: r[1].mean())
    y = np.arange(len(rows))
    ax.axvspan(0.5, 1.0, color="#f2f1ec", zorder=0)
    for i, (label, f, it) in enumerate(rows):
        ax.plot(f, [i] * len(f), "o", ms=2.6, color=ps.SLOTS[0], alpha=0.5, mec="none", zorder=4)
        ax.plot([f.mean()] * 2, [i - 0.26, i + 0.26], lw=1.7, color=ps.SLOTS[0], zorder=5)
        ax.text(f.mean() + 0.040, i, f"{f.mean():.0%}", va="center", fontsize=6.0,
                color=ps.SLOTS[0])
    ax.axvline(1.0, color=ps.MUTED, lw=0.7)
    ax.text(0.99, len(rows) - 0.42, "whole\nnetwork", ha="right", va="top", fontsize=5.4,
            color=ps.MUTED, linespacing=1.2)
    ax.text(0.75, -0.80, "no task reaches half", ha="center", fontsize=5.8, color=ps.MUTED)
    ax.set(yticks=y, yticklabels=[r[0] for r in rows], xlim=(0, 1.04),
           ylim=(-1.1, len(rows) - 0.35), xlabel="fraction of units active   (N = 1000, n = 3)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ps.despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    return rows


def panel_e(ax):
    """Panel (e): every intervention as a change from its own matched reference.

    A change rather than an absolute count because the two task families are read at different
    iterations on different tasks; only the within-family contrast is meaningful. Returns the rows.
    """
    rows, ticks, labels, y = [], [], [], 0
    for fam, ref_pat, cap, items in INTERVENTIONS:
        got = live_matched(ref_pat, cap)
        if got is None:
            continue
        ref, it_ref = got
        ax.axhspan(y - 0.62, y + len(items) - 0.42, color="#f6f5f0", zorder=0)
        ax.text(0.012, y - 0.46, fam, transform=ax.get_yaxis_transform(), ha="left",
                va="bottom", fontsize=6.0, color=ps.INK, zorder=6)
        ax.text(0.988, y - 0.46, f"reference {ref.mean():.0f} active units, n = {len(ref)}",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=5.6,
                color=ps.MUTED, zorder=6)
        for label, pat, group in items:
            g = live_matched(pat, cap)
            if g is None:
                labels.append(label + "  (no data)")
                ticks.append(y)
                y += 1
                continue
            c, _ = g
            d = c.mean() - ref.mean()
            se = np.sqrt(c.var(ddof=1) / len(c) + ref.var(ddof=1) / len(ref))
            col = GROUP_COL.get(group, ps.MUTED)
            ax.plot([d - 1.96 * se, d + 1.96 * se], [y, y], lw=1.0, color=col, zorder=4,
                    solid_capstyle="round")
            ax.plot(d, y, "o", ms=3.6, color=col, zorder=5, mec="none")
            tip = d + 1.96 * se if d >= 0 else d - 1.96 * se
            ax.text(tip + (12 if d >= 0 else -12), y, f"{d:+.0f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=5.6, color=col)
            rows.append((fam, label, d, se, len(c), int(it_ref)))
            ticks.append(y)
            labels.append(label)
            y += 1
        y += 1.15

    ax.axvline(0, color=ps.INK, lw=0.8, zorder=3)
    ax.text(0, y - 0.05, "no change", ha="center", va="top", fontsize=5.8, color=ps.INK)

    # the scale that matters: what the remedy of Figure 3 does on the same axis
    ax.annotate("", xy=(708, y - 0.75), xytext=(0, y - 0.75),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=ps.SLOTS[1], mutation_scale=7))
    ax.text(354, y - 0.95, "for scale — the rate penalty of Fig. 3: +708 units",
            ha="center", fontsize=6.2, color=ps.SLOTS[1])

    ax.set(yticks=ticks, yticklabels=labels, ylim=(y - 0.4, -1.5), xlim=(-270, 790),
           xlabel="change in active units vs. that family's own reference   (mean, 95% CI, n = 3)")
    ps.despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, alpha=0.2, lw=0.5, color=ps.GRID)
    ax.set_axisbelow(True)
    return rows


def main():
    """Assemble Figure 1 and write it. Returns the output path."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-simulate the example network")
    args = ap.parse_args()

    ps.setup()
    rates, targets, p = example_network(refresh=args.refresh)

    fig = plt.figure(figsize=(ps.W2, 168 * ps.MM))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.86, 1.00, 1.16],
                  width_ratios=[1.62, 1.0], hspace=0.52, wspace=0.26)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_a(ax_a, rates, targets, p)
    ps.panel_letter(ax_a, "a", dx=-0.02, dy=0.99)
    ax_a.text(-0.02, 1.13, "What is being measured", transform=ax_a.transAxes, fontsize=7.4,
              color=ps.INK, fontweight="bold")

    ax_b = fig.add_subplot(gs[0, 1])
    panel_b(ax_b, p)
    ps.panel_letter(ax_b, "b")
    ax_b.text(-0.15, 1.20, "Why the threshold is not a judgement call",
              transform=ax_b.transAxes, fontsize=7.4, color=ps.INK, fontweight="bold")

    ax_c = fig.add_subplot(gs[1, 0])
    fits = panel_c(ax_c)
    ps.panel_letter(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    rows_d = panel_d(ax_d)
    ps.panel_letter(ax_d, "d")

    ax_e = fig.add_subplot(gs[2, :])
    rows_e = panel_e(ax_e)
    ps.panel_letter(ax_e, "e", dx=-0.075)

    out = ps.save(fig, "fig_paper_F1")

    print("\n--- numbers quoted in the caption ---")
    for task, (b, A, need) in fits.items():
        print(f"  {task:18} M = {A:.2f} N^{b:.3f}   ->  M = 1000 at N = {need:,.0f}")
    for label, f, it in rows_d:
        print(f"  {label:18} {f.mean():.1%} active (n={len(f)}, read at {it:,})")
    for fam, label, d, se, n, it in rows_e:
        print(f"  {fam:26} {label:22} {d:+7.1f} +- {1.96 * se:.1f} (n={n})")
    print("\n  not in the panel (no participation trace on that architecture):")
    for what, where in CSV_ONLY:
        print(f"    {what:44} {where}")
    return out


if __name__ == "__main__":
    main()
