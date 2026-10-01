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
                                       The arrows inside the pool are a SAMPLE of the connectivity,
                                       not the connectivity: these networks are dense. Only pairs
                                       with an empty corridor between them are joined, so no arrow
                                       crosses a unit. Red arrowheads are excitatory connections and
                                       blue bars inhibitory ones, in the proportion MEASURED in this
                                       network's recurrent weights - these networks are not
                                       sign-constrained, so that is a property of connections, not
                                       of units.
  (b) WHY "SILENT" IS NOT A JUDGEMENT  the participation distribution of that same network on a log
                                       axis. It is bimodal with four orders of magnitude of empty
                                       valley between the modes, so the threshold is read off the
                                       data rather than chosen; the pictogram states the answer.
  (c) SIZE MAKES IT WORSE,             active units vs N on all four tasks with participation
      AND IT IS NOT ONE TASK           traces at more than one size: the 3-bit and 6-bit
                                       flip-flops, CDDM, and DMTS at a delay of 7 tau. The count
                                       grows as roughly N^0.46 on three of them, so the FRACTION
                                       falls: 41% of a 500-unit network, 13% of a 4,000-unit one.
                                       Extrapolating, 1,000 active units would need N ~ 14,000.
                                       DMTS is the exception at N^0.87, from three sizes -- its
                                       projection is marked in the caption output and must not be
                                       quoted on its own.
                                       NOT HERE, and why: MemoryAntiAngle was run before
                                       participation tracking existed and has no traces at all,
                                       and the Walsh flip-flop has one seed per cell.
  (d) NO KNOB FIXES IT                 every intervention we tried, as a change from its OWN matched
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
import json
import os
import pickle
import re
import sys

import numpy as np
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from common import DATA_DIR, SILENT_REL, participation
from flipflop_diversity import rates_and_targets
from flipflop_fixedpoints import load_net

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
# The pool is drawn at N_GLYPH units, not 100. The panel is ~36 mm wide, so 100 glyphs inside the
# boundary sit 4.2 pt apart with a 3.7 pt dot in each: the dots touch, and an arrow between two of
# them has no visible length at all. At 48 the gap is 4.3 pt, a third wider than a dot, and an
# arrow keeps 4-7 pt of shaft after clearing both glyphs. The count is arbitrary either way - the
# FRACTION filled is the measurement, and it survives any count.
N_GLYPH, N_SHOWN = 48, 8
N_EDGES = 65              # connections wanted; the layout keeps as many as fit without crossing
ARC_SAMPLES = 26          # points per drawn arc, for the clearance and crossing tests
BEND = -0.20              # arc3 curvature. Negative bows LEFT of travel, so an arrow running left
                          # to right bows up and one running right to left bows down; flip the sign
                          # to swap that.
R_POOL = 0.90             # pool radius inside the unit-radius boundary circle
DOT_S = 10.0              # unit glyph area in pt^2 (3.6 pt across), on a 7.5 pt lattice
GLYPH_SEED, TRACE_SEED = 3, 11
SILENT_GREY = "#c9c8c0"   # one grey for "silent", in the drawing and in the traces alike
ACTIVE_COL = ps.SLOTS[1]  # red: active units, their traces, and their count
EXC_COL, INH_COL = ps.SLOTS[1], ps.SLOTS[0]   # excitatory / inhibitory connections

# A series listed here is drawn, with every seed, but NOT fitted. The exponent would be quoted, and
# an exponent one seed can move by 0.3 is not a measurement. 8-bit at N = 2000: two seeds sit at
# 403/440 active at the 100k read-out and the third at 1051, because it silences on a much slower
# schedule - by the end of training the three agree (297/306/387) and the series gives N^0.45, in
# the same band as every other task. At 100k it gives N^0.65. The read-out rule is not changed for
# one series, so the points are shown and the line is not.
NO_FIT = {}

# (label, glob, read-out cap). The cap is the iteration the manuscript reads that family at; the
# actual read-out is min(cap, the last iteration every seed reaches), reported on the panel.
FF = f"{DATA_DIR}/NBitFlipFlop_std_ksweep/EqType=h_k=3_N={{N}}_iters=*"
# k = 6, not 8. The k-sweep runs to k = 8, but the 8-bit N = 2000 cell reads 403 / 440 / 1051 at the
# matched read-out - one seed silences on a much slower schedule and moves the exponent from 0.45 to
# 0.65 on its own. k = 6 is the same demand in kind (64 stable states against 8) with the tightest
# cells in the sweep: the worst coefficient of variation over its three sizes is 0.076, against
# 0.576 at k = 8, so it can be fitted like the others.
FF6 = f"{DATA_DIR}/NBitFlipFlop_std_ksweep/EqType=h_k=6_N={{N}}_iters=*"
SCALING = {
    "3-bit flip-flop": ([500, 1000, 2000, 4000], {
        500:  FF.format(N=500),
        1000: FF.format(N=1000),
        2000: FF.format(N=2000),
        4000: f"{DATA_DIR}/NBitFlipFlop_std_bigN/EqType=h_k=3_N=4000_pen=none_iters=*",
    }, 100_000, ps.SLOTS[0]),
    # N=4000 comes from the big-N sweep rather than the k-sweep, exactly as the k=3 series does.
    # Verified from the two cells' own saved configs that they are the same experiment: task
    # (T, mu, n_flip_steps, batch_size, k), model (equation type, gamma, spectral radius, all three
    # noise levels, bias range) and trainer (lr rule, weight decay, anneal_noise, every lambda,
    # dropout) are identical; only N differs.
    "6-bit flip-flop": ([500, 1000, 2000, 4000], {
        500:  FF6.format(N=500),
        1000: FF6.format(N=1000),
        2000: FF6.format(N=2000),
        4000: f"{DATA_DIR}/NBitFlipFlop_std_bigN/EqType=h_k=6_N=4000_pen=none_iters=*",
    }, 100_000, ps.SLOTS[3]),
    "CDDM": ([500, 1000, 2000, 5000], {
        N: f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N={N}_iters=*" for N in (500, 1000, 2000, 5000)
    }, 100_000, ps.SLOTS[1]),
    # DMTS_d7_pen, at a delay of exactly 7 tau (70 steps of dt = 1 with tau = 10, from the sample
    # going off at t = 40 to the match arriving at t = 110). The series named DMTS_v2_pen that this
    # entry used to point at is on no disk we have; the panel silently dropped DMTS for as long as
    # that reference stood, which is why the dangling glob is recorded here rather than deleted.
    # No N = 4000 cell was ever run: at 150,000 iterations it needs ~45 h per seed.
    "DMTS, 7$\\tau$ delay": ([500, 1000, 2000], {
        N: f"{DATA_DIR}/DMTS_d7_pen/EqType=h_N={N}_pen=none" for N in (500, 1000, 2000)
    }, 150_000, ps.SLOTS[2]),
}

# ⚠️ A NETWORK THAT NEVER LEARNED THE TASK IS NOT EVIDENCE ABOUT HOW MANY UNITS THE TASK NEEDS.
# Unpenalised DMTS at this delay is bimodal per seed: it either solves the task or sits at the
# constant-output solution. Of the nine seeds, two land at r2 = 0.4275 and 0.4278 -- the same
# number to three decimals, which is what a constant output scores -- and the other seven at
# 0.9837 to 0.9995. Nothing lies between 0.43 and 0.98, so the cut is read off an empty valley
# rather than chosen; 0.8 is stated here because it sits in that valley, and it was set after
# seeing the split, not before. The excluded seeds are N = 500 (one of three) and N = 2000 (one
# of three), leaving those two cells at n = 2.
#
# No other series needs this. The flip-flop and CDDM sweeps have no chance-level runs, and
# diverged runs are already dropped by name in traces_of.
TASK_MIN_R2 = {"DMTS, 7$\\tau$ delay": 0.8}

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
        # Each activation carries its own shape parameter in the label: "a different activation" is
        # not one condition, and a reader cannot tell a leak of 0.01 from one of 0.3, or softplus at
        # beta=25 (a floor of log(2)/25 = 0.028, nearly a ReLU) from beta=1 (a floor of 0.69, nothing
        # like one). The values are read from the trained nets' own saved configs.
        ("leaky ReLU, leak 0.01", f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=leakyrelu_iters=*",  "activation"),
        ("softplus, $\\beta$ = 25", f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=softplus25_iters=*", "activation"),
        ("sigmoid, 7.5(x$-$0.3)", f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=sigmoid_iters=*",    "activation"),
        ("W.D. 0",            f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=0_iters=*",           "weight decay"),
        # 1e-6 is the default in configs/trainer/trainer.yaml, so every run in this family's
        # reference carries WD=1e-06 - the reference IS this rung, and saying so turns three
        # scattered points into a monotone dose-response
        ("W.D. 10⁻⁶*", None,                                                             "reference"),
        ("W.D. 10⁻⁵",         f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=1e-5_iters=*",        "weight decay"),
        ("W.D. 10⁻⁴",         f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=1e-4_iters=*",        "weight decay"),
    ]),
    ("3-bit flip-flop, 150k", f"{DATA_DIR}/NBitFlipFlop_std_ksweep/EqType=h_k=3_N=1000_iters=*", 150_000, [
        ("leaky ReLU, leak 0.01", f"{DATA_DIR}/NBitFlipFlop_std_activations/EqType=h_k=3_N=1000_act=leakyrelu_iters=150000",  "activation"),
        ("softplus, $\\beta$ = 25", f"{DATA_DIR}/NBitFlipFlop_std_activations/EqType=h_k=3_N=1000_act=softplus25_iters=150000", "activation"),
        ("sigmoid, 7.5(x$-$0.3)", f"{DATA_DIR}/NBitFlipFlop_std_sigmoid/EqType=h_k=3_N=1000_iters=*",   "activation"),
        # ⚠️ THESE WERE LABELLED "input w. ×0.5 … ×20" until 2026-10-01, which read as multiples of
        # the default. They are not. `model.input_row_norm=s` sets every W_inp row to the ABSOLUTE
        # norm s at init, and the default draw's rows sit at √(n_inputs/N) = 0.050 at N = 1000, so
        # the four rungs are 10×, 40×, 100× and 400× the default and the reference is BELOW all of
        # them, not between 0.5 and 2. The old labels put the reference in the middle of its own
        # ladder and made a single-peaked curve look non-monotone.
        ("row norm 0.5", f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=0.5_iters=*", "input scale"),
        ("row norm 2",   f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=2_iters=*",   "input scale"),
        ("row norm 5",   f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=5_iters=*",   "input scale"),
        ("row norm 20",  f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=20_iters=*",  "input scale"),
    ]),
]

# (label, csv, row filter, group). All at CDDM N=1000, eq=h, 30k.
ARCHIVE_FAMILY = ("CDDM, 30k (archived)",
                  ("silent_stats_all.csv", dict(sweep="std", penalty="none")), [
    ("eq. s instead of h",       ("silent_stats_all.csv", dict(sweep="std", penalty="none", eq="s")), "architecture"),
    # The `dale` sweep (Dale's law with non-negative input and output weights) was a row here until
    # 2026-09-30. Its label was the longest on the panel and cost every other panel width, and
    # Dale-constrained networks have a supplementary section of their own (S4), so it was dropped
    # from the figure rather than shortened into something unreadable.
    #
    # ⚠️ THESE ROWS WERE MISLABELLED until 2026-09-21. The CSV's sweep names were read as
    # "self-connections off" (nodale_bias) and "bias fixed at 0" (nodale). Neither is what they are.
    # Matching each sweep's per-net counts against the participation traces still on disk identifies
    # them exactly: `dale` is CDDM_ptrack_g0 (Dale + non-negative I/O), `nodale` is
    # CDDM_ptrack_g0_nodale (both constraints off, bias fixed at 0) and `nodale_bias` is
    # CDDM_ptrack_g0_nodale_trainablebias (both off, bias TRAINABLE) - the five counts agree
    # element for element in all three cases.
    #   - self-connections were never varied: `self_connections=False` is the model default on
    #     every path, including the reference, so no row can be about them;
    #   - "bias fixed at 0" is the reference's own setting, so it cannot be an intervention. The
    #     intervention is making the bias trainable, which is what `nodale_bias` does;
    #   - `nodale` differs from the `std` reference only in being a different sweep of the same
    #     unconstrained architecture, so it is a sweep-to-sweep replicate, not a knob. Dropped.
    ("trainable bias",           ("silent_stats_all.csv", dict(sweep="nodale_bias", penalty="none")), "architecture"),
    ("metabolic λ = 0.01",       ("silent_stats_v2.csv", dict(sweep="metabolic", met="0.01")), "metabolic"),
    ("metabolic λ = 0.1",        ("silent_stats_v2.csv", dict(sweep="metabolic", met="0.1")), "metabolic"),
    ("metabolic λ = 1",          ("silent_stats_v2.csv", dict(sweep="metabolic", met="1.0")), "metabolic"),
    ("metabolic λ = 10",         ("silent_stats_v2.csv", dict(sweep="metabolic", met="10.0")), "metabolic"),
])

# The noise sweep saved no participation traces, so it was the one family under a peak-rate rather
# than a participation criterion; cddm_noise_participation.py re-scores it from the trained weights
# and it now shares the rule with every other family (see noise_active).
# sigma_rec = 0.05 is the default in every model config, so as with weight decay the reference is a
# rung of this ladder rather than something outside it. Listed in ascending order with the rest.
NOISE_FAMILY = ("CDDM, 30k", "0.05", [
    ("rec. noise σ = 0",    "0.0",  "noise"),
    ("rec. noise σ = 0.01", "0.01", "noise"),
    ("rec. noise σ = 0.05*", None, "reference"),
    ("rec. noise σ = 0.1",  "0.1",  "noise"),
])

# Tried, but with no read-out that can sit on this axis. These sweeps saved no participation traces
# and have no row under the scale-free rule, so there is no honest way to put a number on them - but
# leaving them off entirely would let the panel read as "this is everything", which it is not. Their
# trained weights are still on disk, so each could be re-read with a re-simulation pass.
TRIED_NOT_PLOTTED = [
    ("cubic term $\\gamma$", "CDDM_4a031e (on) vs CDDM_4a031e_g0 (off)"),
    ("weight boundary, sticky vs reflective", "CDDM_2bc3c1_g0_reflective"),
]
# I/O positivity is a special case: it is never varied ALONE in any sweep on disk - every `nodale`
# arm switches `dale` and `io_nonnegativity` off together - so the Dale row below is a joint
# contrast and there is no I/O-positivity-only number to report.

# Knobs we did NOT vary. Panel (d) shows everything we tried; these are the obvious candidates it
# does not cover, so the text says "we did not vary" rather than implying a measured null.
NEVER_SWEPT = ["spectral radius of the initial recurrent weights",
               "connectivity density (every network here is dense)",
               "learning rate (fixed by the rule lr = 1e-3 (100/N)^(1/3))",
               "batch size"]

GROUP_COL = {"activation": ps.SLOTS[3], "weight decay": ps.SLOTS[4],
             "input scale": ps.SLOTS[2], "metabolic": ps.SLOTS[1],
             "architecture": ps.SLOTS[0], "noise": ps.COND_COL["both"]}


# Sizes that exist on disk for a scaling series but are deliberately NOT plotted. Every entry needs
# a reason, because the audit below will otherwise shout about it on every run. An empty reason is
# not allowed: if a size is excluded, that is a decision and it gets written down.
SCALING_EXCLUSIONS = {
    ("CDDM", 100):   "near the ceiling (76% of units active) and only one condition has it; "
                     "excluded from the fit in Methods",
    ("CDDM", 10000): "only reaches 80,000 iterations, so it cannot join a 100,000-iteration "
                     "matched-compute read-out; quoted separately in Methods",
}
COVERAGE_CACHE = "data/fig_paper_F1_coverage.pkl"


def _signature(cfg):
    """The experiment identity of a run, everything except network size.

    Two cells with the same signature are the same experiment at different N and belong on the same
    scaling curve, whichever sweep folder they happen to live in.

    Args:
        cfg: an OmegaConf config loaded from a run's saved *_config.yaml.
    Returns:
        a hashable tuple, or None if the config is missing a field we need.
    """
    try:
        t, m, tr = cfg.task, cfg.model, cfg.trainer
        return (str(t.taskname), int(t.n_inputs), int(t.n_outputs), str(m.equation_type),
                str(m.activation_args.name), bool(m.dale), float(m.gamma), float(m.spectral_rad),
                float(tr.lambda_frm), float(tr.lambda_rws), float(tr.lambda_met),
                bool(tr.dropout))
    except Exception:
        return None


def _cell_index(refresh=False):
    """Index every trained cell on disk by (signature -> {N: cell path}).

    Reads ONE saved config per cell, not per run, and caches the result. ~2 s cold over ~550 cells.

    Args:
        refresh: rebuild the cache even if it exists.
    Returns:
        dict mapping signature tuple -> {int N: cell directory path}.
    """
    if os.path.exists(COVERAGE_CACHE) and not refresh:
        with open(COVERAGE_CACHE, "rb") as fh:
            return pickle.load(fh)
    index = {}
    for cell in sorted(glob.glob(os.path.join(DATA_DIR, "*", "*", ""))):
        cfgs = glob.glob(os.path.join(cell, "*", "*_config.yaml"))
        if not cfgs:
            continue
        try:
            cfg = OmegaConf.load(cfgs[0])
        except Exception:
            continue
        sig = _signature(cfg)
        if sig is None:
            continue
        index.setdefault(sig, {})[int(cfg.model.N)] = cell
    with open(COVERAGE_CACHE, "wb") as fh:
        pickle.dump(index, fh)
    return index


def audit_scaling_coverage(refresh=False):
    """Shout if a scaling series has usable data on disk that SCALING does not plot.

    THE BUG THIS EXISTS TO CATCH. SCALING hard-codes each series' sizes, so a cell living in a
    different sweep folder is silently ignored. The 6-bit flip-flop was fitted on three points for
    weeks while NBitFlipFlop_std_bigN/EqType=h_k=6_N=4000_pen=none sat on disk with three seeds --
    the k-sweep glob could never have matched it, because the folder name is different. Matching on
    the configured path would therefore not have found it either. This audit instead reads every
    cell's OWN saved config and groups by experiment signature, so where a run was filed is
    irrelevant.

    Args:
        refresh: rebuild the on-disk cell index.
    Returns:
        dict series -> sorted list of unplotted sizes (empty when a series is fully covered).
    """
    index = _cell_index(refresh=refresh)
    missed = {}
    print("\n--- scaling coverage audit ---")
    for task, (sizes, cells, _it, _col) in SCALING.items():
        ref = None
        for N in sizes:
            cfgs = glob.glob(os.path.join(cells[N], "*", "*_config.yaml")) or \
                   glob.glob(os.path.join(cells[N], "*", "*", "*_config.yaml"))
            if cfgs:
                ref = _signature(OmegaConf.load(cfgs[0]))
                break
        if ref is None:
            print(f"  {task:18} no data on disk - cannot audit")
            continue
        on_disk = set(index.get(ref, {}))
        extra = sorted(n for n in on_disk - set(sizes)
                       if (task, n) not in SCALING_EXCLUSIONS)
        known = sorted(n for n in on_disk - set(sizes) if (task, n) in SCALING_EXCLUSIONS)
        note = f"  (excluded on purpose: {known})" if known else ""
        if extra:
            missed[task] = extra
            print(f"  {task:18} !! UNPLOTTED DATA AT N = {extra} -- {index[ref][extra[0]]}{note}")
        else:
            print(f"  {task:18} {len(sizes)} sizes plotted, none missed{note}")
    if missed:
        print("  ^^ add these to SCALING, or give each a reason in SCALING_EXCLUSIONS.")
    return missed


def traces_of(pattern, min_r2=None):
    """Every (participation matrix, iteration vector) pair under a run-folder glob.

    DIVERGED RUNS ARE DROPPED. A run folder is named `<score>_<task>;...`, and a run whose loss went
    to NaN is saved as `nan_...`. Its participation trace is all zeros from the divergence onward, so
    it reports ZERO active units and drags the cell mean down as if the network had silenced
    completely - which is the opposite of what happened. There are 59 such folders on disk; one of
    them sits in the 8-bit flip-flop cell at N = 2000 and was pulling its mean down by ~160 units.

    RUNS THAT NEVER LEARNED THE TASK ARE DROPPED TOO, when `min_r2` is given. A network sitting at
    the constant-output solution has whatever activity its initialisation left it, which says
    nothing about how many units the task needs -- see TASK_MIN_R2 for the one series that needs
    this and why its threshold lands where it does.

    Args:
        pattern: glob matching run folders (not the pickles themselves).
        min_r2: float, drop any run whose score prefix is below this, or None to keep every run
            that did not diverge.
    Returns:
        list of (P, iters): P is (n_probes, N) participation, iters is (n_probes,).
    """
    out = []
    for f in sorted(glob.glob(os.path.join(pattern, "*", "*ParticipationTrace.pkl"))):
        head = os.path.basename(os.path.dirname(f)).split("_")[0]
        if head == "nan":
            continue
        if min_r2 is not None:
            try:
                if float(head) < min_r2:
                    continue
            except ValueError:
                continue
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        P, it = np.asarray(d.get("participation", [])), np.asarray(d.get("participation_iters", []))
        if len(it) and P.ndim == 2:
            out.append((P, it))
    return out


def live_matched(pattern, cap=None, min_r2=None):
    """Active units per seed at the largest iteration every seed in the cell reaches.

    Matched compute, not each run's own endpoint: a big network read at its end and a small one read
    at its end differ in convergence depth as well as size, which confounds the very comparison the
    scaling panel makes.

    Args:
        pattern: glob matching run folders; cap: read no later than this iteration, or None for the
            deepest shared probe; min_r2: drop runs scoring below this, passed to traces_of.
    Returns:
        (counts, iteration) with counts an (n_seeds,) int array, or None if the cell is empty.
    """
    tr = traces_of(pattern, min_r2=min_r2)
    if not tr:
        return None
    shared = min(int(it[-1]) for _, it in tr)
    it_read = shared if cap is None else min(shared, cap)
    counts = []
    for P, it in tr:
        p = P[int(np.argmin(np.abs(it - it_read)))]
        counts.append(int((p >= SILENT_REL * np.quantile(p, 0.95)).sum()))
    return np.array(counts), it_read


PENALTY_KEYS = ("lambda_frm", "lambda_rws", "lambda_met", "lambda_orth")
CONV_MARGIN = 1.07   # "converged" = clean loss within this factor of the task's absolute bar
                     # (7%, matching pr_matrix.EXCESS_DELTA; see the note there)


def clean_loss_series(run_dir, trace):
    """The dropout-off training loss of one run, and the iterations it was sampled at.

    Prefers metrics["loss_clean_train"] inside the participation trace, which is recorded beside
    the participation probes and is the clean loss by construction. Falls back to TrainLosses.json
    ONLY where the run's own config shows dropout off and every penalty at zero, since the
    training-pass loss is then the same quantity; otherwise returns nothing rather than mixing two
    definitions. CDDM's runs predate the metric and come through the fallback.

    Args:
        run_dir: path to one trained-network folder; trace: its loaded ParticipationTrace dict.
    Returns:
        (iterations, loss) float arrays, or (None, None).
    """
    m = trace.get("metrics", {})
    if "loss_clean_train" in m:
        return np.asarray(trace["iters"], float), np.asarray(m["loss_clean_train"], float)
    cfg = glob.glob(os.path.join(run_dir, "*config.yaml"))
    jf = glob.glob(os.path.join(run_dir, "*TrainLosses.json"))
    if not (cfg and jf):
        return None, None
    t = OmegaConf.load(cfg[0]).trainer
    if bool(t.get("dropout")) or any(float(t.get(k, 0) or 0) for k in PENALTY_KEYS):
        return None, None
    L = np.asarray(json.load(open(jf[0])).get("train_losses", []), float)
    return (np.arange(len(L), dtype=float), L) if len(L) else (None, None)


def runs_with_loss(pattern, min_r2=None):
    """Every non-diverged run under a glob, with its loss series and active-unit series.

    Args:
        pattern: glob matching run folders; min_r2: drop runs scoring below this, or None.
    Returns:
        list of dicts with it_loss/loss, it_act/active, and the run's folder name.
    """
    out = []
    for d in sorted(glob.glob(os.path.join(pattern, "*"))):
        if not os.path.isdir(d):
            continue
        head = os.path.basename(d).split("_")[0]
        if head == "nan":
            continue
        if min_r2 is not None:
            try:
                if float(head) < min_r2:
                    continue
            except ValueError:
                continue
        tp = glob.glob(os.path.join(d, "*ParticipationTrace.pkl"))
        if not tp:
            continue
        try:
            tr = pickle.load(open(tp[0], "rb"))
        except Exception:
            continue
        it_l, L = clean_loss_series(d, tr)
        if L is None or not len(L):
            continue
        P = np.asarray(tr["participation"])
        it_a = np.asarray(tr["participation_iters"], float)
        if P.ndim != 2 or not len(it_a):
            continue
        act = np.array([(p >= SILENT_REL * np.quantile(p, 0.95)).sum() for p in P], float)
        out.append(dict(it_loss=it_l, loss=L, it_act=it_a, active=act, name=os.path.basename(d)))
    return out


def matched_after_convergence(cells):
    """Active units per size, read the same number of iterations after each network CONVERGED.

    ⚠️ WHY NOT A FIXED ITERATION CAP. Reading every size at the same iteration assumes every size
    reaches its final performance at the same point, and on DMTS that is false: the mean crossing
    runs 650, 31,620 and 83,040 iterations at N = 500, 1000 and 2000, so at a 150,000-iteration
    read-out the largest networks have spent a fifth as long in the phase where units go silent.
    That alone put the DMTS exponent at 0.87 against 0.31-0.47 for every other task. On the
    flip-flops and CDDM the crossing does NOT move with size (24,307/26,340/25,777 and
    705/946/1,007), which is why they were unaffected and why the artefact went unnoticed.

    THE BAR IS ABSOLUTE PER TASK, not per run: CONV_MARGIN times the worst final clean loss among
    that task's runs, so every run reaches it and none sets its own. A per-run floor is
    backward-looking -- a network ending at a higher loss gets a looser bar and crosses earlier,
    and training the same network longer moves its bar and so its crossing.

    K IS BOUNDED BY THE WORST INDIVIDUAL RUN, not by a cell mean. Bounding it by the mean lets a
    late-converging seed's target fall past its last probe, where a nearest-probe lookup silently
    clamps it to the endpoint -- reading it exactly the way this function exists to avoid. Measured
    on DMTS, that mistake reported an exponent of 0.56 where the correct value is 0.33.

    Args:
        cells: dict N -> list of run dicts from runs_with_loss.
    Returns:
        (dict N -> (counts array, mean crossing), bar, K).
    """
    runs = [r for v in cells.values() for r in v]
    bar = max(r["loss"][-1] for r in runs) * CONV_MARGIN
    for r in runs:
        r["cross"] = float(r["it_loss"][int(np.argmax(r["loss"] <= bar))])
    K = min(r["it_act"][-1] - r["cross"] for r in runs)
    out = {}
    for N, v in cells.items():
        counts = []
        for r in v:
            want = r["cross"] + K
            if want > r["it_act"][-1] + 50:            # never read past where it was measured
                raise AssertionError(f"N={N} {r['name'][:14]}: target {want:,.0f} past last probe")
            counts.append(r["active"][int(np.argmin(np.abs(r["it_act"] - want)))])
        out[N] = (np.array(counts, float), float(np.mean([r["cross"] for r in v])))
    return out, bar, K


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


def excitatory_fraction():
    """Fraction of the example network's off-diagonal recurrent weights that are positive.

    The drawn connections are a schematic, but the MIX of excitatory and inhibitory ones does not
    have to be invented: it is read from the trained weight matrix, so a reader counting arrowheads
    on the panel is counting the right proportion. These networks are not sign-constrained, so this
    is a property of connections, not of units - a unit both excites and inhibits.

    Returns:
        float in [0, 1].
    """
    folder = sorted(glob.glob(os.path.join(EXAMPLE_NET, "*/")))[0]
    net, _ = load_net(folder)
    W = np.asarray(net.W_rec)
    return float((W[~np.eye(W.shape[0], dtype=bool)] > 0).mean())


def arc_points(p0, p1, rad, n=ARC_SAMPLES):
    """Sample matplotlib's `arc3` connector as a polyline.

    The collision tests below have to run on the curve that is actually DRAWN, not on the chord, so
    this reproduces `matplotlib.patches.ConnectionStyle.Arc3` exactly: a quadratic Bezier whose
    control point is the midpoint displaced by `rad` times the chord rotated -90 degrees. That
    rotation is why a positive `rad` bows to the RIGHT of travel.

    Args:
        p0, p1: (x, y) endpoints; rad: arc3 curvature; n: samples along the curve.
    Returns:
        (n, 2) array of points from p0 to p1.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    c = 0.5 * (p0 + p1) + rad * np.array([d[1], -d[0]])
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1


def inked_span(pts, trim):
    """The part of an arc that is actually drawn: the samples clear of BOTH of its own glyphs.

    Two arrows leaving the same unit share that endpoint exactly, so a crossing test run on the
    full curves calls every such pair a crossing, and the panel ends up with at most one connection
    per unit - which is what capped the first version at twenty arrows for forty-eight units. The
    drawn arrow is shrunk clear of its glyphs anyway, so the test belongs on the shrunk curve.

    Args:
        pts: (n, 2) sampled arc; trim: distance from each endpoint to drop.
    Returns:
        (m, 2) array, or None if nothing survives.
    """
    keep = ((np.linalg.norm(pts - pts[0], axis=1) > trim)
            & (np.linalg.norm(pts - pts[-1], axis=1) > trim))
    return pts[keep] if keep.sum() >= 2 else None


def plan_connections(gx, gy, rng, n_want, lo, hi, clear, bend, trim, r_max=0.99):
    """Choose as many drawable connections as fit, greedily, and return them with their geometry.

    Three things have to hold at once, and they interact, which is why this is a search rather than
    a formula: an arrow may not pass within `clear` of a unit that is not one of its endpoints, no
    two arrows may cross, and every arrow must stay inside the boundary circle. All three are tested
    on the sampled CURVE, so bending is not cosmetic - a bowed arrow can go around a unit that
    blocks the straight chord, which is what lets the panel carry several times more connections
    than the straight-line version could.

    The bend follows the direction of travel: every arrow bows to the same side of its own
    direction, so left-to-right arrows bow one way and right-to-left arrows the other. `bend` sets
    the magnitude and which way.

    Args:
        gx, gy: (n,) unit positions; rng: seeded Generator, for which candidates are tried first;
        n_want: stop once this many are accepted; lo, hi: allowed endpoint separation; clear: the
            corridor half-width that must be free of other units; bend: signed arc3 curvature;
        trim: the un-inked length at each end, used for the crossing test; r_max: arrows must stay
            within this radius of the origin.
    Returns:
        list of (src, dst, rad, points) - points being the (n, 2) sampled curve.
    """
    P = np.column_stack([gx, gy])
    cand = [(a, b) for a in range(len(P)) for b in range(len(P)) if a != b
            and lo <= np.hypot(*(P[b] - P[a])) <= hi]
    rng.shuffle(cand)

    out, taken, used = [], [], set()
    for a, b in cand:
        if len(out) >= n_want:
            break
        if (a, b) in used or (b, a) in used:
            continue                                   # one arrow per pair of units
        pts = arc_points(P[a], P[b], bend)
        if np.hypot(pts[:, 0], pts[:, 1]).max() > r_max:
            continue                                   # would leave the boundary
        d = np.linalg.norm(pts[:, None, :] - P[None, :, :], axis=2)   # (samples, units)
        d[:, [a, b]] = np.inf
        if d.min() < clear:
            continue                                   # passes over a unit
        seg = inked_span(pts, trim)
        if seg is None:
            continue
        path = Path(seg)
        if any(path.intersects_path(q, filled=False) for q in taken):
            continue                                   # crosses an arrow already placed
        out.append((a, b, bend, pts))
        taken.append(path)
        used.add((a, b))
    return out


def check_connections(conns, gx, gy, clear, trim):
    """Assert the invariants `plan_connections` promises. Raises AssertionError on failure.

    A drawing whose whole point is "no arrow crosses anything" should fail loudly rather than ship
    a scribble, so this runs on every build.

    Args:
        conns: the output of `plan_connections`; gx, gy: unit positions; clear: corridor
            half-width; trim: the un-inked length at each end, as passed to the planner.
    Returns:
        None.
    """
    P = np.column_stack([gx, gy])
    paths = [Path(inked_span(pts, trim)) for _, _, _, pts in conns]
    for k, (a, b, _, pts) in enumerate(conns):
        d = np.linalg.norm(pts[:, None, :] - P[None, :, :], axis=2)
        d[:, [a, b]] = np.inf
        assert d.min() >= clear, f"connection {a}->{b} passes within {d.min():.4f} of a unit"
        for q in paths[k + 1:]:
            assert not paths[k].intersects_path(q, filled=False), f"connection {a}->{b} crosses"


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
        (n_live, n_shown_live, n_conn): active units in the network, how many of the drawn units
        were active, and how many connections the layout managed to place.
    """
    N = len(p)
    thr = SILENT_REL * np.quantile(p, 0.95)
    live = p >= thr
    n_live = int(live.sum())

    # --- left: the recurrent pool as a circuit -------------------------------------------------
    ps.blank(ax_net)
    ax_net.set(xlim=(-1.78, 1.78), ylim=(-2.12, 1.16))
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

    # A sample of the recurrent connectivity - the trained networks are dense, so any subset is a
    # sample. The pairs are chosen to be near each other AND to have an empty corridor between
    # them, so no arrow crosses a unit; the excitatory fraction is read off the trained weights.
    nn = 2.0 * R_POOL / np.sqrt(N_GLYPH)                  # typical nearest-neighbour separation
    clear, trim = 0.40 * nn, 0.30 * nn        # clearance 3.0 pt against a 1.8 pt glyph radius
    conns = plan_connections(gx, gy, rng, N_EDGES, 1.02 * nn, 2.4 * nn, clear, BEND, trim)
    check_connections(conns, gx, gy, clear, trim)
    p_exc = excitatory_fraction()
    for src, dst, rad, _ in conns:
        exc = rng.random() < p_exc
        ps.arrow(ax_net, (gx[src], gy[src]), (gx[dst], gy[dst]), rad=rad,
                 col=EXC_COL if exc else INH_COL, lw=0.5, zorder=2, mutation_scale=4.0,
                 shrink=1.9, style="-|>" if exc else "-[,widthB=0.32,lengthB=0.0")

    ax_net.scatter(gx[~on], gy[~on], s=DOT_S, facecolor="none", edgecolor=SILENT_GREY, lw=0.5,
                   zorder=3)
    ax_net.scatter(gx[on], gy[on], s=DOT_S, color=ACTIVE_COL, edgecolor="none", zorder=4)

    # inputs and outputs: what makes it a circuit rather than a bag of units. The task is named,
    # not explained - its trial structure is the supplementary task figure.
    for y in (0.30, 0.0, -0.30):
        xc = np.sqrt(max(1.0 - y * y, 0.0))
        ps.arrow(ax_net, (-1.60, y), (-xc - 0.03, y), col=ps.INK, lw=0.7, mutation_scale=5)
        ps.arrow(ax_net, (xc + 0.03, y), (1.60, y), col=ps.INK, lw=0.7, mutation_scale=5)
    ax_net.text(-1.30, 0.44, "inputs", ha="center", va="bottom", fontsize=6.0, color=ps.INK)
    ax_net.text(1.30, 0.44, "outputs", ha="center", va="bottom", fontsize=6.0, color=ps.INK)

    ax_net.text(0.0, -1.20, "3-bit flip-flop task", ha="center", va="center", fontsize=6.2,
                color=ps.INK)
    ax_net.scatter([-1.62], [-1.58], s=DOT_S, color=ACTIVE_COL, edgecolor="none", zorder=4,
                   clip_on=False)
    ax_net.text(-1.50, -1.58, f"{n_live} active", ha="left", va="center", fontsize=6.0,
                color=ACTIVE_COL)
    ax_net.scatter([0.16], [-1.58], s=DOT_S, facecolor="none", edgecolor=SILENT_GREY, lw=0.5,
                   zorder=4, clip_on=False)
    ax_net.text(0.28, -1.58, f"{N - n_live} silent", ha="left", va="center", fontsize=6.0,
                color=ps.MUTED)
    ps.arrow(ax_net, (-1.70, -1.94), (-1.50, -1.94), col=EXC_COL, lw=0.5, mutation_scale=4.0,
             shrink=0, style="-|>")
    ax_net.text(-1.44, -1.94, "excitatory", ha="left", va="center", fontsize=6.0, color=EXC_COL)
    ps.arrow(ax_net, (0.08, -1.94), (0.28, -1.94), col=INH_COL, lw=0.5, mutation_scale=4.0,
             shrink=0, style="-[,widthB=0.30,lengthB=0.0")
    ax_net.text(0.34, -1.94, "inhibitory", ha="left", va="center", fontsize=6.0, color=INH_COL)

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
                   color=ACTIVE_COL if p[u] >= thr else SILENT_GREY)

    n_shown_live = int(live[pick].sum())
    for lo, hi, lab, col in [(N_SHOWN - n_shown_live, N_SHOWN - 1, "active", ACTIVE_COL),
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
    return n_live, n_shown_live, len(conns)


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
    ax.hist(pp[live], bins=bins, color=ACTIVE_COL, edgecolor="none", label=f"active ({live.sum()})")
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
    """Panel (c): active units vs network size, read a matched time AFTER each network converged.

    Not at a fixed iteration. See matched_after_convergence for why: on DMTS the crossing moves
    from 650 to 83,040 iterations between N = 500 and N = 2000, so a fixed read-out compares
    networks that have spent very different amounts of time in the phase where units go silent,
    and reported that task at N^0.87 against N^0.31-0.47 for every other one. Under the matched
    read-out all four lie between 0.32 and 0.45 (3-bit flip-flop 0.45, 6-bit 0.32, CDDM 0.43,
    DMTS 0.37), and they move by at most 0.003 if CONV_MARGIN is changed from 1.07 to 1.10.

    Returns:
        dict task -> (b, A, N_needed_for_1000, n_sizes, largest N) of the fit.
    """
    fits, handles = {}, []
    for task, (Ns, pats, cap, col) in SCALING.items():
        min_r2 = TASK_MIN_R2.get(task)
        cells = {}
        for N in Ns:
            rs = runs_with_loss(pats[N], min_r2=min_r2)
            total = len(runs_with_loss(pats[N]))
            if min_r2 is not None and len(rs) < total:
                print(f"  .. {task} N={N}: {total - len(rs)} of {total} seed(s) below "
                      f"r2 {min_r2} dropped, {len(rs)} kept")
            if rs:
                cells[N] = rs
        if len(cells) < 2:
            print(f"  !! {task}: {len(cells)} size(s) with a usable clean loss - series omitted")
            continue
        read, bar, K = matched_after_convergence(cells)
        print(f"  .. {task}: converged bar {bar:.5f}, read {K:,.0f} iterations after each "
              f"network crossed it; mean crossing per size "
              + ", ".join(f"N={N}:{read[N][1]:,.0f}" for N in sorted(read)))
        xs, ys, sds = [], [], []
        for N in sorted(read):
            c = read[N][0]
            xs.append(N)
            ys.append(c.mean())
            sds.append(c.std(ddof=1) if len(c) > 1 else 0.0)
            ax.plot([N] * len(c), c, "o", ms=2.4, color=col, alpha=0.55, mec="none", zorder=4)
        xs, ys, sds = np.array(xs, float), np.array(ys, float), np.array(sds, float)
        ax.errorbar(xs, ys, yerr=sds, fmt="o-", color=col, ms=3.4, lw=1.1, zorder=5, capsize=1.6)
        if task in NO_FIT:
            print(f"  !! {task}: not fitted - {NO_FIT[task]}")
            handles.append(Line2D([], [], color=col, marker="o", ms=3.0, lw=1.1, label=task))
            continue
        b, loga = np.polyfit(np.log(xs), np.log(ys), 1)
        fits[task] = (b, np.exp(loga), np.exp((np.log(1000) - loga) / b), len(xs),
                      float(xs.max()))
        # extrapolate only the tasks with four sizes; a three-size fit with a wide seed spread is
        # not something to project a decade beyond the data
        hi = 2.4e4 if len(xs) >= 4 else xs.max() * 1.25
        xf = np.logspace(np.log10(xs.min() * 0.85), np.log10(hi), 50)
        ax.plot(xf, np.exp(loga) * xf ** b, ls=":", lw=0.8, color=col, zorder=3)
        # built by hand: an errorbar's legend handle is a container, and letting matplotlib collect
        # handles here silently produced two entries for the same task
        handles.append(Line2D([], [], color=col, marker="o", ms=3.0, lw=1.1,
                              label=f"{task}  $\\propto N^{{{b:.2f}}}$"))

    # The y axis stops just above the 1,000-unit line rather than at the top of the "every unit
    # active" diagonal: three of the four decades the full diagonal spans hold no data at all and
    # squashed the points into the bottom fifth. The FLOOR is 100, not the 150 that fitted the
    # flip-flop and CDDM series alone -- unpenalised DMTS at N = 500 sits at 114 and 134 active
    # units, and at a floor of 150 both seeds and their whole cell were clipped off the panel
    # without any warning that they had been.
    nn = np.array([3e2, 2.6e4])
    ax.plot(nn, nn, "-", lw=0.7, color=ps.MUTED, zorder=2)
    ax.text(1.55e3, 1.72e3, "every unit active", fontsize=5.5, color=ps.MUTED, ha="left",
            va="center")
    for frac, lab, xl in [(0.5, "50% active", 3.6e3), (0.1, "10% active", 1.75e4)]:
        ax.plot(nn, frac * nn, ls=(0, (4, 3)), lw=0.55, color=ps.FAINT, zorder=1)
        ax.text(xl, frac * xl * 1.12, lab, fontsize=5.2, color=ps.FAINT, ha="center", va="bottom")
    ax.axhline(1000, color=ps.BAD, lw=0.7, ls="-.", zorder=2)
    ax.text(3.4e2, 1045, "1,000 active units", fontsize=5.9, color=ps.BAD, va="bottom")
    ax.set(xscale="log", yscale="log", xlabel="network size N",
           ylabel="active units\n(read a matched time\nafter convergence)",
           xlim=(3.2e2, 2.7e4), ylim=(100, 2.3e3))
    # Lower right is the only corner the guides, the data and the extrapolations all leave empty.
    # It used to be held inboard of the right edge, away from panel d's longest row labels across
    # the gutter; panel d now sits BELOW this panel, so the corner is free and the legend goes
    # flush - held inboard in a panel this narrow it sat on top of the curves it was labelling.
    ax.legend(handles=handles, loc="lower right", fontsize=5.9)
    ps.ygrid(ax)
    return fits


# (label, glob, colour, N) for the trajectory panels, one per task. Two sources of the clean
# (dropout-off) loss are accepted, because the loss is the same quantity in both: metrics
# ["loss_clean_train"] inside the participation trace, recorded every 10 iterations, and
# TrainLosses.json for runs that predate that metric. The second is only used for a run with
# dropout off and every penalty at zero, where the training-pass loss IS the clean loss; the
# loader checks that against the run's own config rather than assuming it.
TRAJ = [
    ("3-bit flip-flop", f"{DATA_DIR}/NBitFlipFlop_std_ksweep/EqType=h_k=3_N=1000_iters=500000",
     ps.SLOTS[0], 1000),
    ("CDDM", f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N=1000_iters=200000", ps.SLOTS[1], 1000),
    ("DMTS, 7$\\tau$ delay", f"{DATA_DIR}/DMTS_d7_pen/EqType=h_N=1000_pen=none", ps.SLOTS[2], 1000),
]
# how far left of the e stack the panel letter of that column sits, as a fraction of the stack's
# width; panel a's letter is placed to match it
E_LETTER_DX = 0.13
PLATEAU_TOL = 0.07        # "performance has plateaued" = clean loss within this of its final
                          # value (7%, matching pr_matrix.EXCESS_DELTA)
PENALTY_KEYS = ("lambda_frm", "lambda_rws", "lambda_met", "lambda_orth")


def clean_loss(run_dir, trace):
    """The dropout-off training loss of one run, and the iterations it was sampled at.

    Prefers metrics["loss_clean_train"] in the participation trace, which is recorded beside the
    participation probes and is the clean loss by construction. Falls back to TrainLosses.json
    ONLY when the run's own config shows dropout off and every penalty at zero, since the
    training-pass loss is then the same quantity; otherwise it is not, and the run is skipped
    rather than silently plotted against a different definition.

    Args:
        run_dir: path to one trained-network folder; trace: its loaded ParticipationTrace dict.
    Returns:
        (iterations, loss) arrays, or (None, None).
    """
    m = trace.get("metrics", {})
    if "loss_clean_train" in m:
        return np.asarray(trace["iters"], float), np.asarray(m["loss_clean_train"], float)
    cfg = glob.glob(os.path.join(run_dir, "*config.yaml"))
    jf = glob.glob(os.path.join(run_dir, "*TrainLosses.json"))
    if not (cfg and jf):
        return None, None
    t = OmegaConf.load(cfg[0]).trainer
    if bool(t.get("dropout")) or any(float(t.get(k, 0) or 0) for k in PENALTY_KEYS):
        return None, None
    L = np.asarray(json.load(open(jf[0])).get("train_losses", []), float)
    return (np.arange(len(L), dtype=float), L) if len(L) else (None, None)


def trajectory(run_dir):
    """Silent-unit count and clean training loss over training, for one run.

    Args:
        run_dir: path to one trained-network folder.
    Returns:
        dict with the loss series, the silent-unit series, the network size, the final r2 from the
        folder name, and the iteration at which the loss first comes within PLATEAU_TOL of its
        final value; or None if the run carries no usable clean loss.
    """
    tp = glob.glob(os.path.join(run_dir, "*ParticipationTrace.pkl"))
    if not tp:
        return None
    t = pickle.load(open(tp[0], "rb"))
    it_L, L = clean_loss(run_dir, t)
    if L is None or not len(L):
        return None
    P, it_P = np.asarray(t["participation"]), np.asarray(t["participation_iters"], float)
    live = np.array([(p >= SILENT_REL * np.quantile(p, 0.95)).sum() for p in P], float)
    m = re.match(r"(-?[0-9.]+)_", os.path.basename(run_dir))
    return dict(it_loss=it_L, loss=L, it_act=it_P, silent=P.shape[1] - live, N=P.shape[1],
                r2=float(m.group(1)) if m else float("nan"),
                plateau=float(it_L[int(np.argmax(L <= L[-1] * (1.0 + PLATEAU_TOL)))]))


def running_median(y, frac=0.03, w_max=401):
    """Median over a sliding window whose width GROWS with the index, for display only.

    A fixed window is wrong on a log x axis. The clean loss falls by most of its total inside the
    first ~2,000 iterations, and a window 301 samples wide flattens exactly that part: normalised
    by its own smoothed start, a 28-fold drop is drawn as a 5-fold one. A window proportional to
    the index is narrow where the curve is steep and wide where it is only noisy, which is what a
    log axis asks for.

    The PLATEAU ITERATION is computed on the raw series, never on this.

    Args:
        y: 1-D array; frac: window half-width as a fraction of the index; w_max: cap in samples.

    ⚠️ CHANGING frac DOES ALMOST NOTHING, AND NOT BECAUSE THE DATA IS SMOOTH. frac only sets the
    window before w_max binds - at 0.03 that is the first 6,667 of a 50,000-sample series - so 87%
    of the curve is identical at any frac. What that hides is how much smoothing happens at all: the
    raw clean loss steps by 2.1% of its mean from one sample to the next, and the smoothed curve by
    0.4%, so 99.6% of the jaggedness is gone. A narrower cap keeps more (w_max 21 keeps 7.9%) and an
    uncapped proportional window keeps less (0.08%), but nothing in this range makes the drawn line
    look like the data. That is why panel (e) draws the raw series underneath it.
    Returns: array of the same length.
    """
    n = len(y)
    out = np.empty(n)
    for i in range(n):
        h = min(w_max // 2, int(frac * (i + 1)))
        out[i] = np.median(y[max(0, i - h):min(n, i + h + 1)])
    return out


def panel_e(axes):
    """Panel (e): every task trains to a good solution, and units keep going silent afterwards.

    One sub-panel per task, each showing EVERY seed rather than a mean. Left axis, dark: the
    dropout-off training loss, which is what "trained to a good level" has to be read off. Right
    axis, in the task's colour from panel c: the number of silent units, which keeps climbing after
    the loss has stopped moving. The vertical rule marks where the loss first comes within 10% of
    its final value.

    The sub-panels are stacked vertically, so they share one x range and one x label.

    Args:
        axes: a list of one Axes per entry of TRAJ, top to bottom.
    Returns:
        list of (task, N, plateau iteration, silent there, silent at end, mean final r2) rows.
    """
    rows, drawn, twins, x_end = [], [], [], []
    for ax, (label, pat, col, _) in zip(axes, TRAJ):
        runs = [trajectory(d) for d in sorted(glob.glob(os.path.join(pat, "*"))) if os.path.isdir(d)]
        runs = [r for r in runs if r]
        if not runs:
            print(f"  !! {label}: no run carries a usable clean loss - sub-panel left empty")
            continue
        axr = ax.twinx()
        for r in runs:
            # the loss is drawn RELATIVE TO ITS OWN START so all three sub-panels share one axis.
            # The three tasks finish 0.026, 0.022 and 1e-4, so on an absolute axis each sub-panel
            # would need its own range and a left label that only one of them carries would be
            # telling the reader something untrue about the other two.
            # normalised by the RAW first loss, so the drawn drop is the true one; the growing
            # window leaves the first samples essentially unsmoothed, so the curve still starts at 1
            # THE RAW SERIES GOES UNDERNEATH. The smoothed line removes 99.6% of the step-to-step
            # wiggle (measured: 2.1% of the mean per step raw, 0.4% smoothed), so on its own it
            # shows a clean descent and hides that the loss is in fact very jagged. No setting of
            # the window fixes that - a wider one smooths more, an uncapped one smooths more still.
            # Drawing both is the only honest version: the trend is readable and the noise is there.
            ax.plot(r["it_loss"][1:], (np.asarray(r["loss"], float) / r["loss"][0])[1:], "-",
                    lw=0.35, color=ps.INK, alpha=0.16, zorder=3)
            ax.plot(r["it_loss"][1:], (running_median(r["loss"]) / r["loss"][0])[1:], "-",
                    lw=0.85, color=ps.INK, alpha=0.75, zorder=4)
            axr.plot(r["it_act"][1:], r["silent"][1:], "-", lw=0.9, color=col, alpha=0.85, zorder=5)
        plateau = float(np.mean([r["plateau"] for r in runs]))
        s_plat = float(np.mean([r["silent"][np.argmin(np.abs(r["it_act"] - plateau))] for r in runs]))
        s_end = float(np.mean([r["silent"][-1] for r in runs]))
        r2 = float(np.mean([r["r2"] for r in runs]))
        N = runs[0]["N"]
        ax.axvline(plateau, color=ps.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=2)

        # x starts at 8, not 90: most of the loss drop happens inside the first hundred iterations
        # and a panel that begins at 90 shows a curve already a third of the way down while its
        # axis label says "relative to its start".
        ax.set(xscale="log", yscale="log", ylim=(4e-4, 2.2))
        axr.set(ylim=(0, N * 1.04))
        axr.spines[["top"]].set_visible(False)
        # the house style hides every right spine; this axis needs its own, or the silent-units
        # ticks hang off nothing and the reader cannot tell which panel edge they belong to
        axr.spines[["right"]].set_visible(True)
        # each sub-panel now carries its own right-hand ticks, so each set is coloured by its own
        # task rather than all three by the last one
        axr.tick_params(axis="y", colors=col)
        ax.set_title(f"{label},  $r^2 = {r2:.3f}$, $N = {N}$", fontsize=5.8, color=ps.INK, pad=3)
        drawn.append(ax)
        twins.append(axr)
        x_end.append(max(r["it_loss"][-1] for r in runs))
        rows.append((label, N, plateau, s_plat, s_end, r2))

    if not drawn:
        return rows
    # one x range for the stack, so a vertical read across the three sub-panels compares the same
    # iteration; the label and its tick labels go under the bottom sub-panel only
    for ax in drawn:
        ax.set_xlim(8, max(x_end) * 1.6)
    for ax in drawn[:-1]:
        ax.tick_params(labelbottom=False)
    drawn[-1].set_xlabel("training iteration")
    # the axis labels sit on the middle sub-panel, where each one labels all three
    mid = len(drawn) // 2
    drawn[mid].set_ylabel("clean loss, relative to its start")
    twins[mid].set_ylabel("silent units")
    return rows


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
    eq = match.pop("eq", "h")
    out = []
    for r in csv.DictReader(open(os.path.join(DATA_DIR, fname))):
        if r.get("eq") != eq or int(r["N"]) != N_UNITS:
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
    """Active units at one recurrent-noise level, under the participation rule where available.

    THIS SWEEP SAVED NO PARTICIPATION TRACES, so its original number came from a peak-rate rule - a
    unit silent below 5% of the 95th-percentile peak rate - which put its sigma = 0.05 reference at
    524 active where the metabolic reference, same task and size at the same 30,000-iteration budget,
    reads 414. That gap was the measuring stick, not the networks.

    The trained weights are on disk, so `cddm_noise_participation.py` rebuilds each net, runs it
    noise-free and scores it with the same scale-free participation rule as every other family. When
    that CSV is present it is used and the reference reads 443, beside the metabolic sweep's 414.
    The peak-rate CSV remains the fallback, so the panel still draws if the re-score has not been run.

    Args:
        sigma: sigma_rec as it appears in the CSV.
    Returns:
        (mean active, sd, n_nets).
    """
    root = os.path.join(DATA_DIR, "CDDM_fb2792_g0_noise")
    rescored = os.path.join(root, "silent_units_per_condition_participation.csv")
    if os.path.exists(rescored):
        for r in csv.DictReader(open(rescored)):
            if r["eq"] == "h" and float(r["sigma_rec"]) == float(sigma):
                return (float(r["active_mean"]), float(r["active_std"]), int(r["n_nets"]))
    for r in csv.DictReader(open(os.path.join(root, "silent_units_per_condition.csv"))):
        if r["eq"] == "h" and float(r["sigma_rec"]) == float(sigma):
            return (N_UNITS - float(r["silent_rel_mean"]), float(r["silent_rel_std"]),
                    int(r["n_nets"]))
    return (float("nan"), float("nan"), 0)


def noise_counts(sigma):
    """Per-seed active-unit counts at one recurrent-noise level, where the re-score supplies them.

    The original sweep kept only a per-condition mean, sd and n, which is why its slide drew an
    interval where every other family draws its individual networks. `cddm_noise_participation.py`
    re-scores the trained weights and records each net, so the seeds are available again.

    Args:
        sigma: sigma_rec as it appears in the CSV.
    Returns:
        list of ints, one per net; empty when the re-score has not been run.
    """
    path = os.path.join(DATA_DIR, "CDDM_fb2792_g0_noise",
                        "silent_units_per_condition_participation.csv")
    if not os.path.exists(path):
        return []
    for r in csv.DictReader(open(path)):
        if r["eq"] == "h" and float(r["sigma_rec"]) == float(sigma):
            return [int(x) for x in r.get("active_counts", "").split(";") if x]
    return []


def panel_d(ax):
    """Panel (d): every intervention we ran, as a change from its own family's reference.

    Four families, each internally consistent in task, architecture, read-out iteration and
    silence criterion. Plotting changes rather than counts is what lets them share an axis.

    Returns:
        list of (family, label, delta, se, n) rows.
    """
    rows, ticks, labels, refs, y = [], [], [], [], 0.0
    bands = []

    def block(title, ref, items):
        """Draw one family: a shaded band, a title, and one interval per intervention."""
        nonlocal y
        start = y
        for label, vals, group in items:
            # The reference condition is a rung of the weight-decay ladder, not just its baseline:
            # without it the panel shows 0 and 10^-5 and 10^-4 with nothing between them and no
            # marker for where "no change" sits inside the ladder itself. Drawn as a point at zero
            # with no interval - a condition compared with itself has no uncertainty to show - in
            # the neutral reference ink rather than a categorical slot.
            if group == "reference":
                ax.plot(0, y, "o", ms=3.4, color=ps.BASE, zorder=5, mec="none")
                refs.append(len(labels))
                ticks.append(y)
                labels.append(label)
                y += 1
                continue
            if vals is None or len(vals) < 2 or len(ref) < 2:
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
        block(title, ref, [(lab, None if pat is None else (live_matched(pat, cap) or (np.array([]),))[0], grp)
                           for lab, pat, grp in items])

    title, (rf, rm), items = ARCHIVE_FAMILY
    block(title, csv_active(rf, **rm),
          [(lab, csv_active(f, **m), grp) for lab, (f, m), grp in items])

    title, ref_sigma, items = NOISE_FAMILY
    rmean, rsd, rn = noise_active(ref_sigma)
    start = y
    for label, sigma, group in items:
        if group == "reference":
            ax.plot(0, y, "o", ms=3.4, color=ps.BASE, zorder=5, mec="none")
            refs.append(len(labels))
            ticks.append(y)
            labels.append(label)
            y += 1
            continue
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
    # Each block's reference row is drawn in the same neutral ink as its marker dot, so the label
    # and the point it names carry one colour and the asterisk is not the only thing marking it.
    # NOT bold: this style's font stack resolves both weights to the same Helvetica.ttc face, so
    # set_fontweight("bold") sets the property and changes nothing on the page.
    for i in refs:
        ax.get_yticklabels()[i].set_color(ps.BASE)
    ax.xaxis.grid(True, alpha=0.2, lw=0.5, color=ps.GRID)
    ax.set_axisbelow(True)
    return rows



def main():
    """Assemble Figure 1 and write it. Returns the output path."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-simulate the example network")
    args = ap.parse_args()

    audit_scaling_coverage()
    ps.setup()
    rates, _, p = example_network(refresh=args.refresh)

    # The three sub-panels of panel e stack vertically in the left column, under the network
    # schematic; panels c and d stack in the right column beside them. Each e sub-panel keeps the
    # width of the whole left column, which is the axis a trajectory reads along, and the three
    # give up the height they no longer need once they share one x axis and one x label.
    # The side margins are set explicitly rather than left at the matplotlib default of
    # 0.125/0.90, which spent 25 mm of the canvas on blank edges that the tight-bbox export then
    # trimmed away: the panels were paying for whitespace nobody ever saw. Reclaiming it buys the
    # gutter its real job - panel d's row labels reach 34 mm left of d's own axis and the e stack's
    # silent-units ticks reach 11 mm right of its axis, and the two were overlapping by 11 mm.
    fig = plt.figure(figsize=(ps.W2, 205 * ps.MM))
    # Panel a sits in the left column, the same width as the e stack under it, and panel b in the
    # right column, the same width as c and d under it - so every panel edge lines up with the one
    # above or below it. The left column is narrower than the right because panel d's row labels
    # hang off the right column's left edge and c has a legend to fit.
    LEFT, RIGHT, W_RATIO, GUTTER = 0.06, 0.975, 0.86, 0.52
    A_RATIOS, A_WSPACE = (1.25, 1.0), 0.02
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.62, 2.50],
                  width_ratios=[W_RATIO, 1.0], hspace=0.26, wspace=GUTTER,
                  left=LEFT, right=RIGHT)

    # The top row is split on its own rather than inheriting the columns below it. The wide gutter
    # the bottom row needs is there to hold panel d's row labels, and nothing in the top row has
    # labels to put in it - inherited, it was 42 mm of blank paper between the traces and panel b.
    # The split is chosen so that b still starts exactly where c and d start, at 0.660.
    gs_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], width_ratios=list(A_RATIOS),
                                   wspace=A_WSPACE)
    ax_net = fig.add_subplot(gs_a[0, 0])
    ax_tr = fig.add_subplot(gs_a[0, 1])
    n_live, n_shown_live, n_conn = panel_a(ax_net, ax_tr, rates, p)
    ax_net.set_title("trained ReLU RNN", fontsize=6.6, color=ps.INK, pad=2)
    # panel_letter's offset is a fraction of its OWN axes width, and the schematic's axes is a
    # fraction of the column, so the same number would not put the two letters of this column on
    # one vertical. a's offset is derived from e's by the ratio of their widths.
    net_frac = A_RATIOS[0] / (sum(A_RATIOS) * (1.0 + A_WSPACE / 2.0))
    ps.panel_letter(ax_net, "a", dx=-E_LETTER_DX / net_frac, dy=1.02)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_b(ax_b, p)
    ps.panel_letter(ax_b, "b")

    # c and d are laid on the SAME three rows as the e stack beside them, c on the first and d on
    # the other two, so every horizontal edge in this block lines up across the figure: c's top and
    # bottom with the first e sub-panel's, d's top with the second's and d's bottom with the third's.
    gs_cd = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 1], hspace=0.30)
    ax_c = fig.add_subplot(gs_cd[0, 0])
    fits = panel_c(ax_c)
    ps.panel_letter(ax_c, "c")

    ax_d = fig.add_subplot(gs_cd[1:3, 0])
    rows_d = panel_d(ax_d)
    # the same offset b and c use, so the three letters of the right column sit on one vertical
    ps.panel_letter(ax_d, "d")

    gs_e = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 0], hspace=0.30)
    axes_e = [fig.add_subplot(gs_e[i, 0]) for i in range(3)]
    rows_e = panel_e(axes_e)
    ps.panel_letter(axes_e[0], "e", dx=-E_LETTER_DX)

    out = ps.save(fig, "fig_paper_F1")

    print("\n--- numbers quoted in the caption ---")
    print(f"  panel a: {n_live} of {N_UNITS} units active; {n_shown_live} of {N_SHOWN} randomly "
          f"drawn units active; {n_conn} connections drawn over {N_GLYPH} glyphs")
    for task, (b, A, need, n_sizes, n_max) in fits.items():
        # A series measured at three sizes is fitted but its projection is not quotable on its own:
        # the DMTS exponent of 0.87 puts 1,000 active units at N = 6,022, which is 3x beyond the
        # largest network the series contains. Panel c already stops such a series' dotted line at
        # 1.25x its largest N; the caption has to carry the same warning, or the number reads as a
        # measurement.
        note = ("" if n_sizes >= 4 else
                f"   [{n_sizes} sizes, {need / n_max:.1f}x beyond N = {n_max:,.0f} - "
                f"do not quote alone]")
        print(f"  {task:18} M = {A:.2f} N^{b:.3f}   ->  M = 1000 at N = {need:,.0f}{note}")
    for task, N, plateau, s_plat, s_end, r2 in rows_e:
        print(f"  panel e: {task:20} N={N}, final r2 {r2:.3f}; loss within {PLATEAU_TOL:.0%} of "
              f"final at iter {plateau:,.0f} with {s_plat:.0f} silent, {s_end:.0f} at the end "
              f"-> {s_end - s_plat:.0f} MORE units silenced after performance plateaued")
    for fam, label, d, se, n in rows_d:
        print(f"  {fam:30} {label:24} {d:+7.1f} +- {1.96 * se:5.1f} (n={n})")
    print("\n  TRIED but with no read-out that fits this axis (weights survive, traces do not):")
    for what, where in TRIED_NOT_PLOTTED:
        print(f"    {what:42s} {where}")
    print("\n  NEVER SWEPT, so the panel must not be read as covering them:")
    for what in NEVER_SWEPT:
        print(f"    {what}")
    return out


if __name__ == "__main__":
    main()
