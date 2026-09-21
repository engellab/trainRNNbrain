#!/usr/bin/env python3
"""
Manuscript Figure 1 - THE PROBLEM. Most units of a trained ReLU RNN never fire; the bigger the
network the worse it gets; it happens on every task; and no standard knob fixes it.

This is the motivation figure, so it has to do four things in one display item, and the first row
has to explain the measurement before the second and third rows quantify it:

  (a) THE PROBLEM, AND ONLY THAT       the trained network as a circuit - inputs, a bounded
                                       recurrent pool with three quarters of it dead, outputs -
                                       beside eight units drawn at RANDOM out of that same network
                                       on one trial, on a shared rate scale. Real simulated traces,
                                       not cartoons. The task is NAMED and not explained: its trial
                                       structure is the supplementary task figure
                                       (`fig_supp_tasks.py`), because a panel that explains a task
                                       and a pathology at once explains neither.
                                       The arrows inside the pool are a nearest-neighbour SAMPLE of
                                       the connectivity, not the connectivity: these networks are
                                       dense. Nearest neighbours because a random pair is a long
                                       chord that crosses the units in between, and thirty of those
                                       is a scribble.
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
Output: img/internal_figures/fig_paper_F1.pdf (+ .svg; vector only - see paperstyle.save)
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

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
N_UNITS = 1000            # every intervention family is measured at this size

# Panel (a). The pool is drawn as N_GLYPH units standing for all N of them, filled to the MEASURED
# live fraction, and N_SHOWN units are drawn at random from the network - at random, so that the
# proportion of them that turns out to be silent is itself the result rather than a choice. Both
# seeds are fixed so the panel is reproducible; neither was searched over.
N_GLYPH, N_SHOWN = 100, 8
N_EDGES = 30              # nearest-neighbour connections drawn; the real network is dense
R_POOL = 0.80             # pool radius inside the unit-radius boundary circle
GLYPH_SEED, TRACE_SEED = 3, 11
SILENT_GREY = "#c9c8c0"   # one grey for "silent", in the drawing and in the traces alike

# (label, glob, read-out cap). The cap is the iteration the manuscript reads that family at; the
# actual read-out is min(cap, the last iteration every seed reaches), reported on the panel.
FF = f"{DATA_DIR}/NBitFlipFlop_std_ksweep/EqType=h_k=3_N={{N}}_iters=*"
SCALING = {
    "3-bit flip-flop": ([500, 1000, 2000, 4000], {
        500:  FF.format(N=500),
        1000: FF.format(N=1000),
        2000: FF.format(N=2000),
        4000: f"{DATA_DIR}/NBitFlipFlop_std_bigN/EqType=h_k=3_N=4000_pen=none_iters=*",
    }, 100_000, ps.SLOTS[0]),
    "CDDM": ([500, 1000, 2000, 5000], {
        N: f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N={N}_iters=*" for N in (500, 1000, 2000, 5000)
    }, 100_000, ps.SLOTS[1]),
    "DMTS": ([500, 1000, 2000], {
        N: f"{DATA_DIR}/DMTS_std_pen/EqType=h_N={N}_pen=none" for N in (500, 1000, 2000)
    }, 100_000, ps.SLOTS[2]),
}

# Every intervention we ran, grouped into families that share a task, an architecture, a read-out
# iteration AND a silence criterion. The panel plots a CHANGE from each family's OWN reference,
# which is what makes it legitimate to show families side by side: no number is ever compared
# across a criterion boundary, only against a reference measured the same way.
#
# Families 3 and 4 come from summary CSVs rather than participation traces. Their raw sweeps were
# deleted (Supplementary S6) and family 4 uses a peak-rate rather than a participation criterion,
# which is exactly why they are their own blocks with their own references.
TRACE_FAMILIES = [
    ("CDDM, 200k", f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N=1000_iters=*", None, [
        ("leaky ReLU",        f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=leakyrelu_iters=*",  "activation"),
        ("softplus",          f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=softplus25_iters=*", "activation"),
        ("bounded sigmoid",   f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=sigmoid_iters=*",    "activation"),
        ("weight decay 0",    f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=0_iters=*",           "weight decay"),
        ("weight decay 10⁻⁵", f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=1e-5_iters=*",        "weight decay"),
        ("weight decay 10⁻⁴", f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=1e-4_iters=*",        "weight decay"),
    ]),
    ("3-bit flip-flop, 150k", f"{DATA_DIR}/NBitFlipFlop_std_ksweep/EqType=h_k=3_N=1000_iters=*", 150_000, [
        ("bounded sigmoid",   f"{DATA_DIR}/NBitFlipFlop_std_sigmoid/EqType=h_k=3_N=1000_iters=*",   "activation"),
        ("input weights ×0.5", f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=0.5_iters=*", "input scale"),
        ("input weights ×2",   f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=2_iters=*",   "input scale"),
        ("input weights ×5",   f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=5_iters=*",   "input scale"),
        ("input weights ×20",  f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=20_iters=*",  "input scale"),
    ]),
]

# (label, csv, row filter, group). All at CDDM N=1000, eq=h, 30k.
ARCHIVE_FAMILY = ("CDDM, 30k (archived)",
                  ("silent_stats_all.csv", dict(sweep="std", penalty="none")), [
    ("Dale's law imposed",       ("silent_stats_all.csv", dict(sweep="dale", penalty="none")), "architecture"),
    ("self-connections off",     ("silent_stats_all.csv", dict(sweep="nodale_bias", penalty="none")), "architecture"),
    ("  + bias fixed at 0",      ("silent_stats_all.csv", dict(sweep="nodale", penalty="none")), "architecture"),
    ("metabolic λ = 0.01",       ("silent_stats_v2.csv", dict(sweep="metabolic", met="0.01")), "metabolic"),
    ("metabolic λ = 0.1",        ("silent_stats_v2.csv", dict(sweep="metabolic", met="0.1")), "metabolic"),
    ("metabolic λ = 1",          ("silent_stats_v2.csv", dict(sweep="metabolic", met="1.0")), "metabolic"),
    ("metabolic λ = 10",         ("silent_stats_v2.csv", dict(sweep="metabolic", met="10.0")), "metabolic"),
])

# The noise sweep is the one family under a peak-rate rather than a participation criterion.
NOISE_FAMILY = ("CDDM, 30k (peak-rate criterion)", "0.05", [
    ("recurrent noise σ = 0",    "0.0",  "noise"),
    ("recurrent noise σ = 0.01", "0.01", "noise"),
    ("recurrent noise σ = 0.1",  "0.1",  "noise"),
])

# Knobs we did NOT vary. Panel (d) shows everything we tried; these are the obvious candidates it
# does not cover, so the text says "we did not vary" rather than implying a measured null.
NEVER_SWEPT = ["spectral radius of the initial recurrent weights",
               "connectivity density (every network here is dense)",
               "learning rate (fixed by the rule lr = 1e-3 (100/N)^(1/3))",
               "batch size"]

GROUP_COL = {"activation": ps.SLOTS[3], "weight decay": ps.SLOTS[4],
             "input scale": ps.SLOTS[2], "metabolic": ps.SLOTS[1],
             "architecture": ps.SLOTS[0], "noise": ps.COND_COL["both"]}


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


def panel_a(ax_net, ax_tr, rates, p):
    """Panel (a): the problem, and nothing else. The recurrent pool with most of it dead, and
    eight units drawn at random out of that same network.

    THE TASK IS DELIBERATELY ABSENT. The previous version of this panel drew the input and output
    ports and laid the target bits over the traces, so it was half a task diagram and half a
    problem statement and did neither: a reader cannot learn the flip-flop from three grey steps,
    and the silence is not about the flip-flop anyway - it happens on all three tasks. The trial
    structure of all three now has its own supplementary figure (`fig_supp_tasks.py`), which leaves
    this panel one job: most units of a trained ReLU RNN never leave zero.

    TWO COLOURS, AND THEY ARE THE SAME TWO ON BOTH SIDES. Filled = active, hollow grey = silent in
    the drawing; the traces repeat exactly that, so a grey trace and a grey dot are the same
    statement. Giving each trace its own hue, as this panel used to, encodes identity - which is
    not a variable the reader needs.

    RATES SHARE ONE SCALE. Every trace is drawn against the same rate axis and the same scale bar.
    The earlier version normalised each unit by its own maximum, which blew a silent unit's
    numerical dust up to the height of a driven unit's response and made the panel argue against
    itself.

    Args:
        ax_net: blank axes for the network drawing; ax_tr: blank axes for the traces;
        rates: (N, T, B) firing rates; p: (N,) participation.
    Returns:
        (n_live, n_shown_live): active units in the network, and how many of the drawn units were
        active.
    """
    N = len(p)
    thr = SILENT_REL * np.quantile(p, 0.95)
    live = p >= thr
    n_live = int(live.sum())

    # --- left: the recurrent pool as a circuit -------------------------------------------------
    ps.blank(ax_net)
    ax_net.set(xlim=(-1.78, 1.78), ylim=(-2.02, 1.56))
    ax_net.set_aspect("equal", adjustable="box")

    rng = np.random.default_rng(GLYPH_SEED)
    n_on = int(round(N_GLYPH * n_live / N))
    # Vogel's sunflower inside the boundary: an equal-area packing of the disc, so a quarter of the
    # dots covers a quarter of it and reads as a quarter of the POOL. A square lattice reads as a
    # layer, which a recurrent pool is not.
    i = np.arange(N_GLYPH)
    rad = R_POOL * np.sqrt((i + 0.5) / N_GLYPH)
    ang = i * np.pi * (3.0 - np.sqrt(5.0))
    gx, gy = rad * np.cos(ang), rad * np.sin(ang)
    on = np.zeros(N_GLYPH, bool)
    on[rng.choice(N_GLYPH, n_on, replace=False)] = True   # silence is not spatially organised

    ax_net.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor=ps.MUTED, lw=0.8, zorder=1))

    # A sample of the recurrent connectivity, drawn ONLY between nearest neighbours. The trained
    # networks are dense, so any subset is a sample either way - but a random pair is a long chord
    # that passes over the units in between, and thirty of those is a scribble. Between nearest
    # neighbours no third unit can lie on the segment, so no arrow crosses a glyph.
    d = np.hypot(gx[:, None] - gx[None, :], gy[:, None] - gy[None, :])
    np.fill_diagonal(d, np.inf)
    pairs = {tuple(sorted((a, int(b)))) for a, b in enumerate(np.argmin(d, axis=1))}
    pairs = sorted(pairs)
    for a, b in [pairs[k] for k in rng.choice(len(pairs), min(N_EDGES, len(pairs)), replace=False)]:
        src, dst = (a, b) if rng.random() < 0.5 else (b, a)      # recurrence is directed
        ps.arrow(ax_net, (gx[src], gy[src]), (gx[dst], gy[dst]), col="#b3b2aa", lw=0.4,
                 zorder=2, mutation_scale=3.2, shrink=2.6)

    ax_net.scatter(gx[~on], gy[~on], s=11, facecolor="none", edgecolor=SILENT_GREY, lw=0.5,
                   zorder=3)
    ax_net.scatter(gx[on], gy[on], s=11, color=ps.SLOTS[0], edgecolor="none", zorder=4)

    # inputs and outputs: what makes it a circuit rather than a bag of units. The task is named,
    # not explained - its trial structure is the supplementary task figure.
    for y in (0.30, 0.0, -0.30):
        xc = np.sqrt(max(1.0 - y * y, 0.0))
        ps.arrow(ax_net, (-1.60, y), (-xc - 0.03, y), col=ps.INK, lw=0.7, mutation_scale=5)
        ps.arrow(ax_net, (xc + 0.03, y), (1.60, y), col=ps.INK, lw=0.7, mutation_scale=5)
    ax_net.text(-1.30, 0.44, "inputs", ha="center", va="bottom", fontsize=6.0, color=ps.INK)
    ax_net.text(1.30, 0.44, "outputs", ha="center", va="bottom", fontsize=6.0, color=ps.INK)

    # the recurrence loop: out of the boundary and back into it, bowing away from the disc
    a0 = np.radians(118)
    ps.arrow(ax_net, (np.cos(a0), np.sin(a0)), (np.cos(np.pi - a0), np.sin(np.pi - a0)),
             col=ps.MUTED, lw=0.8, rad=-0.62, mutation_scale=6, shrink=0)
    ax_net.text(0.56, 1.24, "recurrent", ha="left", va="center", fontsize=6.0, color=ps.MUTED)

    ax_net.text(0.0, -1.18, "3-bit flip-flop task", ha="center", va="center", fontsize=6.2,
                color=ps.INK)
    ax_net.scatter([-0.80], [-1.56], s=11, color=ps.SLOTS[0], edgecolor="none", zorder=4,
                   clip_on=False)
    ax_net.text(-0.68, -1.56, f"{n_live} active", ha="left", va="center", fontsize=6.4,
                color=ps.SLOTS[0])
    ax_net.scatter([-0.80], [-1.88], s=11, facecolor="none", edgecolor=SILENT_GREY, lw=0.5,
                   zorder=4, clip_on=False)
    ax_net.text(-0.68, -1.88, f"{N - n_live} silent", ha="left", va="center", fontsize=6.4,
                color=ps.MUTED)

    # --- right: units drawn at random out of that same network ---------------------------------
    ps.blank(ax_tr)
    pick = np.random.default_rng(TRACE_SEED).choice(N, N_SHOWN, replace=False)
    pick = pick[np.argsort(-p[pick])]          # active on top, so the block itself shows a ratio
    trial = int(np.argmax(rates[int(np.argmax(p))].max(axis=0)))
    R = np.asarray(rates[pick][:, :, trial], float)
    T = R.shape[1]
    tt = np.arange(T)
    scale = float(R.max())
    amp = 0.80 / max(scale, 1e-9)              # one common rate scale for every trace

    for j, (u, r) in enumerate(zip(pick, R)):
        base = float(N_SHOWN - 1 - j)
        ax_tr.plot([0, T - 1], [base, base], lw=0.4, color=ps.GRID, zorder=1)
        ax_tr.plot(tt, base + amp * r, lw=0.75, zorder=3,
                   color=ps.SLOTS[0] if p[u] >= thr else SILENT_GREY)

    n_shown_live = int(live[pick].sum())
    for lo, hi, lab, col in [(N_SHOWN - n_shown_live, N_SHOWN - 1, "active", ps.SLOTS[0]),
                             (0, N_SHOWN - n_shown_live - 1, "silent", ps.MUTED)]:
        if hi < lo:
            continue
        ax_tr.plot([T + 16] * 2, [lo - 0.18, hi + 0.86], lw=0.8, color=col, zorder=4,
                   solid_capstyle="round")
        ax_tr.text(T + 26, (lo + hi + 0.68) / 2, lab, ha="left", va="center", fontsize=6.2,
                   color=col)

    # scale bars instead of axes: a schematic panel should not spend two spines on a quantity
    # whose absolute value carries no meaning (ReLU rates are in arbitrary units)
    v = float(f"{scale / 2:.0g}")
    y0 = float(N_SHOWN - 1)                    # beside the tallest trace, not in a corner
    ax_tr.plot([-20, -20], [y0, y0 + amp * v], lw=1.0, color=ps.INK, zorder=4,
               solid_capstyle="butt")
    ax_tr.text(-27, y0 + amp * v / 2, f"{v:g} a.u.\nrate", ha="right", va="center", fontsize=5.4,
               color=ps.MUTED, linespacing=1.3)
    ax_tr.plot([0, 100], [-0.72] * 2, lw=1.0, color=ps.INK, zorder=4, solid_capstyle="butt")
    ax_tr.text(50, -0.92, r"10 $\tau$", ha="center", va="top", fontsize=5.4, color=ps.MUTED)

    ax_tr.text(T / 2, N_SHOWN + 0.02, f"{N_SHOWN} units drawn at random, one trial",
               ha="center", va="bottom", fontsize=6.4, color=ps.INK)
    ax_tr.set(xlim=(-105, T + 80), ylim=(-1.35, N_SHOWN + 0.55))
    return n_live, n_shown_live


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
    ax.text(thr * 1.35, top * 1.14, "criterion", fontsize=5.8, color=ps.INK, ha="left",
            va="center")
    ax.set_xscale("log")
    # spelled out, because "p" on its own is the one thing a reader of this figure has to be
    # told: it is not the firing rate, it is how far the rate moves and how high it gets
    ax.set(xlabel="participation  $p_i=\\mathrm{std}(r_i)+q_{0.9}(|r_i|)$", ylabel="units")
    ax.text(0.5, -0.30, "how much unit $i$'s rate moves over a trial, and how high it gets",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=ps.MUTED)
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
        # extrapolate only the two tasks with four sizes; DMTS has three and a wide seed spread,
        # so its fitted exponent is not something to project a decade beyond the data
        hi = 2.4e4 if len(xs) >= 4 else xs.max() * 1.25
        xf = np.logspace(np.log10(xs.min() * 0.85), np.log10(hi), 50)
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
    need = np.mean([v[2] for task, v in fits.items() if task != "DMTS"])
    ax.text(3.4e2, 1120, "1,000 active units", fontsize=5.9, color=ps.BAD, va="bottom")
    ax.set(xscale="log", yscale="log", xlabel="network size N", ylabel="active units",
           xlim=(3.2e2, 2.7e4), ylim=(140, 3.4e4))
    ax.legend(handles=handles, loc="lower right", fontsize=5.9)
    ps.ygrid(ax)
    return fits


def csv_active(fname, **match):
    """Active-unit counts from an archived summary CSV, at CDDM N = 1000, eq = h.

    The CSVs record the SILENT fraction under the scale-free participation rule (`rel_5p95`), so
    the active count is N(1 - rel_5p95). These sweeps' raw networks were deleted; the rows are all
    that survives, which is why they are shown as their own block against their own reference.

    Args:
        fname: CSV under DATA_DIR; match: column -> value, compared as floats where possible.
    Returns:
        (n_nets,) array of active counts.
    """
    out = []
    for r in csv.DictReader(open(os.path.join(DATA_DIR, fname))):
        if r.get("eq") != "h" or int(r["N"]) != N_UNITS:
            continue
        ok = True
        for k, v in match.items():
            try:
                ok &= float(r[k]) == float(v)
            except ValueError:
                ok &= r[k] == v
        if ok:
            out.append(N_UNITS * (1.0 - float(r["rel_5p95"])))
    return np.array(out)


def noise_active(sigma):
    """Active units at one recurrent-noise level, from the per-condition CSV of the noise sweep.

    This sweep has no participation traces, so its silence rule is peak-rate based: a unit is
    silent below 5% of the 95th-percentile peak rate. It is therefore never compared with the
    other families except as a change from its own reference.

    Args:
        sigma: sigma_rec as it appears in the CSV.
    Returns:
        (mean active, sd, n_nets).
    """
    path = os.path.join(DATA_DIR, "CDDM_fb2792_g0_noise", "silent_units_per_condition.csv")
    for r in csv.DictReader(open(path)):
        if r["eq"] == "h" and float(r["sigma_rec"]) == float(sigma):
            return (N_UNITS - float(r["silent_rel_mean"]), float(r["silent_rel_std"]),
                    int(r["n_nets"]))
    return (float("nan"), float("nan"), 0)


def panel_d(ax):
    """Panel (d): every intervention we ran, as a change from its own family's reference.

    Four families, each internally consistent in task, architecture, read-out iteration and
    silence criterion. Plotting changes rather than counts is what lets them share an axis.

    Returns:
        list of (family, label, delta, se, n) rows.
    """
    rows, ticks, labels, y = [], [], [], 0.0
    bands = []

    def block(title, ref, items):
        """Draw one family: a shaded band, a title, and one interval per intervention."""
        nonlocal y
        start = y
        for label, vals, group in items:
            if len(vals) < 2 or len(ref) < 2:
                y += 1
                continue
            d = vals.mean() - ref.mean()
            se = np.sqrt(vals.var(ddof=1) / len(vals) + ref.var(ddof=1) / len(ref))
            col = GROUP_COL.get(group, ps.MUTED)
            ax.plot([d - 1.96 * se, d + 1.96 * se], [y, y], lw=1.0, color=col, zorder=4,
                    solid_capstyle="round")
            ax.plot(d, y, "o", ms=3.4, color=col, zorder=5, mec="none")
            tip = d + 1.96 * se if d >= 0 else d - 1.96 * se
            ax.text(tip + (14 if d >= 0 else -14), y, f"{d:+.0f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=5.5, color=col)
            rows.append((title, label, float(d), float(se), len(vals)))
            ticks.append(y)
            labels.append(label)
            y += 1
        bands.append((start - 0.6, y - 0.4, title, ref.mean(), len(ref)))
        y += 1.3

    for title, ref_pat, cap, items in TRACE_FAMILIES:
        got = live_matched(ref_pat, cap)
        if got is None:
            continue
        ref, _ = got
        block(title, ref, [(lab, (live_matched(pat, cap) or (np.array([]),))[0], grp)
                           for lab, pat, grp in items])

    title, (rf, rm), items = ARCHIVE_FAMILY
    block(title, csv_active(rf, **rm),
          [(lab, csv_active(f, **m), grp) for lab, (f, m), grp in items])

    title, ref_sigma, items = NOISE_FAMILY
    rmean, rsd, rn = noise_active(ref_sigma)
    start = y
    for label, sigma, group in items:
        m, sd, n = noise_active(sigma)
        d = m - rmean
        se = np.sqrt(sd ** 2 / max(n, 1) + rsd ** 2 / max(rn, 1))
        col = GROUP_COL.get(group, ps.MUTED)
        ax.plot([d - 1.96 * se, d + 1.96 * se], [y, y], lw=1.0, color=col, zorder=4,
                solid_capstyle="round")
        ax.plot(d, y, "o", ms=3.4, color=col, zorder=5, mec="none")
        # a very negative value would put its label under the row label, so flip it inside
        if d < -250:
            ax.text(d + 1.96 * se + 16, y, f"{d:+.0f}", va="center", ha="left",
                    fontsize=5.5, color=col)
        else:
            tip = d + 1.96 * se if d >= 0 else d - 1.96 * se
            ax.text(tip + (14 if d >= 0 else -14), y, f"{d:+.0f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=5.5, color=col)
        rows.append((title, label, float(d), float(se), n))
        ticks.append(y)
        labels.append(label)
        y += 1
    bands.append((start - 0.6, y - 0.4, title, rmean, rn))

    for i, (lo, hi, title, refm, refn) in enumerate(bands):
        if i % 2 == 0:
            ax.axhspan(lo, hi, color="#f6f5f0", zorder=0)
        ax.text(0.012, lo + 0.04, f"{title}   ({refm:.0f} active)",
                transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.7,
                color=ps.INK, zorder=6)

    ax.axvline(0, color=ps.INK, lw=0.8, zorder=3)
    top = y + 0.2
    ax.annotate("", xy=(708, top), xytext=(0, top),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=ps.SLOTS[1], mutation_scale=7))
    ax.text(354, top + 0.45, "rate penalty (Fig. 3)", ha="center", fontsize=6.0,
            color=ps.SLOTS[1])
    ax.set(yticks=ticks, yticklabels=labels, ylim=(top + 1.0, -1.0), xlim=(-430, 800),
           xlabel="change in active units")
    ps.despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0, labelsize=5.8)
    ax.xaxis.grid(True, alpha=0.2, lw=0.5, color=ps.GRID)
    ax.set_axisbelow(True)
    return rows



def main():
    """Assemble Figure 1 and write it. Returns the output path."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-simulate the example network")
    args = ap.parse_args()

    ps.setup()
    rates, _, p = example_network(refresh=args.refresh)

    fig = plt.figure(figsize=(ps.W2, 176 * ps.MM))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.74, 1.55],
                  width_ratios=[1.30, 1.0], hspace=0.36, wspace=0.44)

    gs_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], width_ratios=[1.05, 1.0],
                                   wspace=0.02)
    ax_net = fig.add_subplot(gs_a[0, 0])
    ax_tr = fig.add_subplot(gs_a[0, 1])
    n_live, n_shown_live = panel_a(ax_net, ax_tr, rates, p)
    ax_net.set_title("trained ReLU RNN", fontsize=6.6, color=ps.INK, pad=2)
    ps.panel_letter(ax_net, "a", dx=-0.13, dy=1.02)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_b(ax_b, p)
    ps.panel_letter(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    fits = panel_c(ax_c)
    ps.panel_letter(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    rows_d = panel_d(ax_d)
    ps.panel_letter(ax_d, "d", dx=-0.32)

    out = ps.save(fig, "fig_paper_F1")

    print("\n--- numbers quoted in the caption ---")
    print(f"  panel a: {n_live} of {N_UNITS} units active; {n_shown_live} of {N_SHOWN} randomly "
          f"drawn units active")
    for task, (b, A, need) in fits.items():
        print(f"  {task:18} M = {A:.2f} N^{b:.3f}   ->  M = 1000 at N = {need:,.0f}")
    for fam, label, d, se, n in rows_d:
        print(f"  {fam:30} {label:24} {d:+7.1f} +- {1.96 * se:5.1f} (n={n})")
    print("\n  NEVER SWEPT, so the panel must not be read as covering them:")
    for what in NEVER_SWEPT:
        print(f"    {what}")
    return out


if __name__ == "__main__":
    main()
