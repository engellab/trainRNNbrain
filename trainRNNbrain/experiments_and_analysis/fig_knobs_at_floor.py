#!/usr/bin/env python3
"""The four knob ladders on the standard (bias-0) network, read two ways: at each run's own loss
floor, and at the end of its budget.

WHY TWO READ-OUTS IN ONE PANEL. Reading a knob ladder at 1.07x each run's own fitted loss floor --
the rule the size panels use -- says every knob does nothing. On the weight-decay ladder the four
settings land within 41 units of each other at the crossing and 328 apart at the end of training.
That is not a broken read-out; it is the talk's own thesis showing up per knob. At the crossing the
clean loss is already 0.93 to 0.98 of its final value, so these networks reach their final
performance after a few thousand iterations with roughly half their units working, and then spend
the remaining ~190,000 iterations shedding units while the loss does not improve. What a knob
changes is how fast that shedding goes, not the network you would get by stopping at convergence.

So a panel that showed only the floor read-out would say "weight decay is neutral", and a panel
that showed only the end read-out would say "weight decay is the strongest lever here". Both
sentences are true of different questions, and the pair of them is the result.

THE RULE DOES NOT APPLY TO A LOSS THAT RISES LATE, AND ON THE FLIP-FLOP IT DOES. `excess_time_at`
returns the FIRST iteration where the smoothed loss reaches 1.07 x the fitted floor. That is a
convergence point only while the loss is still falling. The flip-flop input-scale runs reach a
smoothed minimum of 0.034-0.058 and then rise to 0.139, a factor of 2.5 to 4.1, so the fitted floor
is the LATE plateau and the loss first touches 1.07 x it on the way DOWN, at iteration 107 to 307
of 150,000 -- before training has done anything, with 859-935 of 1000 units still active. That
ladder is therefore reported at the end of training only, and `floor_usable` says so per rung. The
three CDDM ladders rise by 1.04 to 1.44 after their minimum and are read at the floor as intended.

⚠️ THE CLEAN-LOSS PROBE IS NOISY AND MUST BE SMOOTHED BEFORE ANY TREND IS READ OFF IT.
`metrics["loss_clean_train"]` is one forward pass per probe, and on CDDM its spread is over 50% of
its value: a single run gives min 0.00296 against a typical 0.006. Comparing raw min to raw final
therefore measures the probe's noise, not the trace, and doing exactly that put every CDDM ladder
in the "loss rises late" class on the first pass here -- wrongly, as the independent per-iteration
series in TrainLosses.json falls smoothly over the same window. Every trend below is taken from a
201-sample running median.

DEGENERATE FLOOR FITS ARE EXCLUDED AND COUNTED. `pr_matrix.fit_floor_at` bounds its floor parameter
below at 1e-6 and returns that bound as a successful fit. On these runs 4 of 40 fits come back
pinned there, wrong by a factor of 2,000 to 11,000 against the loss the run actually reached;
`at_own_floor` then drops them, because the loss never gets within 7% of 1e-6. The drop is uneven
across arms (0 of 12 on the weight-decay ladder, 4 of 15 on the metabolic one), so it is reported
per rung rather than left to silently reweight a comparison. The fitter itself is NOT changed here:
every manuscript exponent depends on it, and that is not a change to make inside a slide script.

Usage:  python trainRNNbrain/experiments_and_analysis/fig_knobs_at_floor.py [--probe]
        (run from the repository root; --probe prints the table and writes no figures)
"""

import argparse
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fig_paper_F1 as F1
import paperstyle as ps
from common import DATA_DIR

W, H = 7.6, 2.9
FLOOR_BOUND = 1e-6          # pr_matrix.fit_floor_at's lower bound on the floor parameter
DELTA = 0.07                # read at 1.07x the floor, the project's convergence margin
# A smoothed loss that ends more than this times above its own minimum has risen late, so the
# FIRST crossing of 1.07x the late floor is on the initial descent and is not a convergence
# point. Fixed before the ladders were read, not tuned to them.
MAX_LATE_RISE = 1.5

# One ladder per knob, on the bias-0 cells. The reference rung sits in its DOSE ORDER position, not
# first: on a dose ladder the default is a rung, and putting it first makes the ladder read out of
# order. `ref` marks which rung is the untouched network, drawn in ps.BASE.
LADDERS = [
    dict(stem="slide_k_weightdecay", knob="weight decay", ref=1,
         task="CDDM, $N$ = 1000, 200,000 iterations",
         rungs=[("none", f"{DATA_DIR}/CDDM_std_g0_weightdecay_bias0/EqType=h_N=1000_wd=0_iters=200000"),
                ("default\n$10^{-6}$", f"{DATA_DIR}/CDDM_std_g0_drift_bias0/EqType=h_N=1000_iters=200000"),
                ("10$\\times$\n$10^{-5}$", f"{DATA_DIR}/CDDM_std_g0_weightdecay_bias0/EqType=h_N=1000_wd=1e-5_iters=200000"),
                ("100$\\times$\n$10^{-4}$", f"{DATA_DIR}/CDDM_std_g0_weightdecay_bias0/EqType=h_N=1000_wd=1e-4_iters=200000")]),
    dict(stem="slide_k_recnoise", knob="recurrent noise", ref=2,
         task="CDDM, $N$ = 1000, 75,000 iterations",
         rungs=[("none\n$\\sigma$ = 0", f"{DATA_DIR}/CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=0"),
                ("0.01", f"{DATA_DIR}/CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=0.01"),
                ("default\n0.05", f"{DATA_DIR}/CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=0.05"),
                ("0.1", f"{DATA_DIR}/CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=0.1")]),
    # ⚠️ EVERY RUNG AT 75,000, CONTROL INCLUDED. The first version of this ladder put the drift
    # control at 200,000 against arms at 50,000 and reported that the metabolic penalty raises the
    # count from 266 to 356-417. The count falls steadily with training, so a control given four
    # times as long to shed units manufactures most of that gap. The control here is the penalty
    # grid's own unpenalised cell, which matches the arms on lr, weight decay, N, both noise terms,
    # equation type and Dale, and differs only in carrying no penalty; and both sides read the
    # _c75000 continuations, whose traces are joined and run 0 to 74,900.
    dict(stem="slide_k_metabolic", knob="metabolic penalty", ref=0,
         task="CDDM, $N$ = 1000, 75,000 iterations",
         rungs=[("none\n$\\lambda$ = 0", f"{DATA_DIR}/CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0"),
                ("0.01", f"{DATA_DIR}/CDDM_std_g0_metabolic_bias0_c75000/EqType=h_N=1000_LmbdMet=0.01"),
                ("0.1", f"{DATA_DIR}/CDDM_std_g0_metabolic_bias0_c75000/EqType=h_N=1000_LmbdMet=0.1"),
                ("1.0", f"{DATA_DIR}/CDDM_std_g0_metabolic_bias0_c75000/EqType=h_N=1000_LmbdMet=1.0"),
                ("10.0", f"{DATA_DIR}/CDDM_std_g0_metabolic_bias0_c75000/EqType=h_N=1000_LmbdMet=10.0")]),
    dict(stem="slide_k_activation", knob="activation function", ref=0,
         task="CDDM, $N$ = 1000, 200,000 iterations",
         rungs=[("ReLU\n(default)", f"{DATA_DIR}/CDDM_std_g0_drift_bias0/EqType=h_N=1000_iters=200000"),
                ("leaky ReLU\nleak 0.01", f"{DATA_DIR}/CDDM_std_g0_activations_bias0/EqType=h_N=1000_act=leakyrelu_iters=200000"),
                ("softplus\n$\\beta$ = 25", f"{DATA_DIR}/CDDM_std_g0_activations_bias0/EqType=h_N=1000_act=softplus25_iters=200000"),
                ("sigmoid\n7.5(x$-$0.3)", f"{DATA_DIR}/CDDM_std_g0_activations_bias0/EqType=h_N=1000_act=sigmoid_iters=200000")]),
    dict(stem="slide_k_inputscale", knob="input weight scale", ref=0,
         task="3-bit flip-flop, $N$ = 1000, 150,000 iterations",
         rungs=[("default\ndraw", f"{DATA_DIR}/NBitFlipFlop_std_ksweep_bias0/EqType=h_k=3_N=1000_iters=150000"),
                ("0.5", f"{DATA_DIR}/NBitFlipFlop_std_winp_bias0/EqType=h_k=3_N=1000_s=0.5_iters=150000"),
                ("2", f"{DATA_DIR}/NBitFlipFlop_std_winp_bias0/EqType=h_k=3_N=1000_s=2_iters=150000"),
                ("5", f"{DATA_DIR}/NBitFlipFlop_std_winp_bias0/EqType=h_k=3_N=1000_s=5_iters=150000"),
                ("20", f"{DATA_DIR}/NBitFlipFlop_std_winp_bias0/EqType=h_k=3_N=1000_s=20_iters=150000")]),
]


def running_median(v, w=201):
    """Running median of a 1-D series, edge-padded, window in SAMPLES.

    The clean-loss probe is a single forward pass per sample and its spread is over half its value,
    so any min/final comparison on the raw series measures the probe rather than the training.

    Args:
        v: the series; w: window length in samples, forced odd and shrunk for a short series.
    Returns:
        an array the same length as v.
    """
    v = np.asarray(v, float)
    if len(v) < w:
        w = max(3, (len(v) // 3) * 2 + 1)
    pad = w // 2
    vp = np.pad(v, pad, mode="edge")
    return np.array([np.median(vp[i:i + w]) for i in range(len(v))])


def read_rung(path):
    """Active-unit counts for one rung, at its own loss floor and at the end of its budget.

    Args:
        path: the cell directory holding one folder per seed.
    Returns:
        dict with keys `floor` (counts at the 1.07x-floor crossing), `end` (counts at the last
        participation probe), `cross` (crossing iterations), `n` (runs read), `degenerate` (fits
        pinned at the fitter's lower bound and so excluded), `unreached` (fits that were fine but
        whose loss never reached the threshold), and `loss_ratio` (loss at the crossing divided by
        the final loss, per run -- near 1 means the crossing is a real plateau and not an early
        point on a still-falling curve).
    """
    runs = F1.runs_with_loss(path)
    at_floor, at_end, cross, ratios, rises = [], [], [], [], []
    degenerate = unreached = 0
    for r in runs:
        it_l, L = np.asarray(r["it_loss"], float), np.asarray(r["loss"], float)
        it_a = np.asarray(r["it_act"], float)
        at_end.append(r["active"][-1])
        sm = running_median(L)
        rises.append(sm[-1] / sm.min())
        floor = F1.fit_floor_at(L, it_l, float(it_l[-1]))
        if floor is None or floor <= FLOOR_BOUND * 1.01:
            degenerate += 1
            continue
        T = F1.excess_time_at(L, it_l, floor, DELTA)
        if not np.isfinite(T):
            unreached += 1
            continue
        at_floor.append(r["active"][int(np.argmin(np.abs(it_a - T)))])
        cross.append(T)
        ratios.append(sm[int(np.argmin(np.abs(it_l - T)))] / sm[-1])
    rise = float(np.mean(rises)) if rises else float("nan")
    return dict(floor=np.array(at_floor, float), end=np.array(at_end, float),
                cross=np.array(cross, float), n=len(runs), degenerate=degenerate,
                unreached=unreached, loss_ratio=np.array(ratios, float),
                rise=rise, floor_usable=bool(rise <= MAX_LATE_RISE))


def ladder_slide(spec, n_total=1000):
    """One knob ladder as two panels: at its own loss floor, and at the end of training.

    Args:
        spec: one entry of LADDERS; n_total: the network size, for the ceiling line and y limit.
    Returns:
        the output path, or None if no rung had readable runs.
    """
    reads = [(lab, read_rung(p)) for lab, p in spec["rungs"]]
    if not any(len(r["end"]) for _, r in reads):
        print(f"  SKIP {spec['stem']}: no runs with a loss+participation trace")
        return None

    # A ladder whose loss rises late has no valid floor crossing (see the module docstring), so its
    # floor panel is NOT drawn rather than drawn with a caveat: a panel showing 859-935 of 1000
    # units read at iteration 107 would be believed long before the caveat was read.
    usable = any(r["floor_usable"] and len(r["floor"]) for _, r in reads)
    panels = ([("floor", "at 1.07$\\times$ its own loss floor"),
               ("end", "at the end of training")] if usable
              else [("end", "at the end of training")])

    ps.setup()
    fig, axes = plt.subplots(1, len(panels), figsize=(W if usable else W * 0.55, H),
                             sharey=True, squeeze=False)
    axes = axes[0]
    cols = [ps.BASE if i == spec["ref"] else ps.SLOTS[0] for i in range(len(reads))]
    xs = np.arange(len(reads), dtype=float)

    for ax, (key, head) in zip(axes, panels):
        groups = [r[key] for _, r in reads]
        res = ps.strip(ax, xs, groups, cols, rng=np.random.default_rng(0))
        ref = groups[spec["ref"]]
        if len(ref):
            ax.axhline(ref.mean(), color=ps.BASE, lw=0.7, ls=":", zorder=1)
        ax.axhline(n_total, color=ps.FAINT, lw=0.7, ls=":", zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for lab, _ in reads], fontsize=6.2)
        ax.set_xlim(-0.6, len(reads) - 0.4)
        # the same 0-to-N axis both panels and every other knob panel uses, so a flat ladder reads
        # as flat and the two halves can be compared by eye
        ax.set_ylim(0, n_total * 1.08)
        ax.set_title(head, fontsize=7.2, color=ps.INK, pad=4)
        for x, (m, sd, n) in zip(xs, res):
            if n:
                ax.annotate(f"{m:.0f}", (x, m), textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=6.2, color=ps.INK)
    axes[0].set_ylabel(f"active units of {n_total:,}")
    axes[0].annotate(f"all {n_total:,}", (axes[0].get_xlim()[0], n_total),
                     textcoords="offset points", xytext=(4, 3), fontsize=6.0, color=ps.MUTED)
    # on its OWN line: a one-line title longer than the axes widens the saved canvas, and the
    # panel then sits in a half-empty figure
    sub = ("" if usable else
           "\n(no valid loss-floor crossing: the loss rises "
           f"{np.nanmean([r['rise'] for _, r in reads]):.1f}$\\times$ after its minimum)")
    fig.suptitle(f"{spec['knob']} — {spec['task']}{sub}", fontsize=7.6, color=ps.INK,
                 y=1.02 if usable else 1.06)
    fig.subplots_adjust(left=0.085 if usable else 0.15, right=0.985, top=0.80,
                        bottom=0.17, wspace=0.07)
    # a one-panel figure printed at the two-panel width is a panel with an empty half beside
    # it, which reads as a missing panel rather than as a deliberate one
    return ps.save(fig, spec["stem"], w_mm=180 if usable else 104)


def knob_ratio_spans():
    """Each ladder's span of active-unit ratios against its own reference rung, at the end of
    training.

    The talk's bottom-line panel puts every knob on one ratio axis, because the ladders differ in
    task and in budget and so cannot share an absolute one. Read AT THE END rather than at the loss
    floor: at the floor the knobs are flat by construction, and five flat rows would read as "these
    were never measured" rather than as "these act after the loss stops moving", which is the point.

    Returns:
        [(label, lo, hi, lo_rung, hi_rung), ...] -- one row per ladder, ratios to its reference.
    """
    out = []
    for spec in LADDERS:
        reads = [(lab.replace("\n", " "), read_rung(path)) for lab, path in spec["rungs"]]
        ref = reads[spec["ref"]][1]["end"]
        if not len(ref):
            continue
        vals = [(lab, float(r["end"].mean() / ref.mean()))
                for lab, r in reads if len(r["end"])]
        vals = [v for v in vals if np.isfinite(v[1])]
        if len(vals) < 2:
            continue
        lo = min(vals, key=lambda t: t[1])
        hi = max(vals, key=lambda t: t[1])
        out.append((spec["knob"], lo[1], hi[1], lo[0], hi[0]))
    return out


def probe():
    """Print every rung's two read-outs, crossing, loss ratio and exclusions."""
    for spec in LADDERS:
        print(f"\n=== {spec['knob']}  ({spec['task']})")
        print(f"{'rung':<16}{'n':>3}{'at floor':>14}{'at end':>14}{'crossing':>11}"
              f"{'Lcross/Lend':>13}{'excluded':>14}{'rise':>7}")
        for lab, path in spec["rungs"]:
            r = read_rung(path)
            lab1 = lab.replace("\n", " ")
            if not r["n"]:
                print(f"{lab1:<16}  -  no runs")
                continue
            f_s = (f"{r['floor'].mean():>6.0f} ± {r['floor'].std(ddof=0):<3.0f}"
                   if len(r["floor"]) else f"{'none read':>12}")
            e_s = f"{r['end'].mean():>6.0f} ± {r['end'].std(ddof=0):<3.0f}"
            c_s = f"{r['cross'].mean():>10,.0f}" if len(r["cross"]) else f"{'-':>10}"
            q_s = f"{r['loss_ratio'].mean():>16.2f}" if len(r["loss_ratio"]) else f"{'-':>16}"
            ex = r["degenerate"] + r["unreached"]
            ex_s = f"{ex} ({r['degenerate']} bad fit)" if ex else "0"
            flag = "" if r["floor_usable"] else "  <- floor read-out invalid"
            print(f"{lab1:<16}{r['n']:>3}{f_s:>14}{e_s:>14}{c_s}{q_s}{ex_s:>14}"
                  f"{r['rise']:>7.2f}{flag}")


def main(probe_only=False):
    """Build every ladder, or just print the table. Returns the list of paths written."""
    if probe_only:
        probe()
        return []
    out = []
    for spec in LADDERS:
        got = ladder_slide(spec)
        if got:
            out.append(got)
            print(f"  wrote {got}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true", help="print the table, write nothing")
    main(ap.parse_args().probe)
