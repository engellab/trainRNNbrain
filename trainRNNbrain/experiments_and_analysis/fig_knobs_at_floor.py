#!/usr/bin/env python3
"""The four knob ladders on the standard (bias-0) network, read two ways: at each run's own loss
floor, and at the end of its budget.

WHY TWO READ-OUTS IN ONE PANEL. The floor read-out asks what network a knob gives you once the loss
has converged; the end read-out asks what it gives after the rest of the budget, during which the
loss barely moves and units keep falling silent. A knob can change one and not the other, so both
are drawn, on the same rungs, one above the other.

THE READ-OUT (2026-10-13). Each run is read where its NOISY training loss -- the per-iteration loss
it was trained on, joined along its warm-start chain -- first reaches 1.05x its own fitted floor
(`common.noisy_floor`). Runs with no usable crossing (fit pinned at its
bound, loss never reaching the bar, or a run whose last 10% ends more than 1.5x above its floor) are
dropped and counted per rung, with the reason. The previous version read the noise-free probe loss
at 1.07x through `pr_matrix.fit_floor_at`; that loss climbs back after its minimum on the flip-flop,
which made the input-scale ladder unreadable at its floor.

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
from common import DATA_DIR, noisy_floor

W, H = 7.6, 2.9

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
         task="CDDM, $N$ = 1000, 200,000 iterations",
         rungs=[("none\n$\\sigma$ = 0", f"{DATA_DIR}/CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=0_iters=200000"),
                ("0.01", f"{DATA_DIR}/CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=0.01_iters=200000"),
                ("default\n0.05", f"{DATA_DIR}/CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=0.05_iters=200000"),
                ("0.1", f"{DATA_DIR}/CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=0.1_iters=200000")]),
    # the control ran 200k and the arms 75k; both ends are read at 75k (ba75b05 withdrew a claim that
    # compared them at different budgets)
    dict(stem="slide_k_metabolic", knob="metabolic penalty", ref=0, end_at=75000,
         task="CDDM, $N$ = 1000, 75,000 iterations",
         rungs=[("none\n$\\lambda$ = 0", f"{DATA_DIR}/CDDM_std_g0_drift_bias0/EqType=h_N=1000_iters=200000"),
                ("0.01", f"{DATA_DIR}/CDDM_std_g0_metabolic_bias0_c75000/EqType=h_N=1000_LmbdMet=0.01"),
                ("0.1", f"{DATA_DIR}/CDDM_std_g0_metabolic_bias0_c75000/EqType=h_N=1000_LmbdMet=0.1"),
                ("1.0", f"{DATA_DIR}/CDDM_std_g0_metabolic_bias0_c75000/EqType=h_N=1000_LmbdMet=1.0"),
                ("10.0", f"{DATA_DIR}/CDDM_std_g0_metabolic_bias0_c75000/EqType=h_N=1000_LmbdMet=10.0")]),
    # The recinit array (job 14919348) runs both of the next two ladders with ONE in-sweep control,
    # spectral radius 1.2 and no penalty, so both share it as their reference rung.
    dict(stem="slide_k_recgain", knob="initial recurrent gain (spectral radius)", ref=1,
         task="CDDM, $N$ = 1000, 75,000 iterations",
         rungs=[("0.5", f"{DATA_DIR}/CDDM_std_g0_recinit/EqType=h_N=1000_sr=0.5_iters=75000"),
                ("default\n1.2", f"{DATA_DIR}/CDDM_std_g0_recinit/EqType=h_N=1000_sr=1.2_iters=75000"),
                ("2.0", f"{DATA_DIR}/CDDM_std_g0_recinit/EqType=h_N=1000_sr=2.0_iters=75000")]),
    dict(stem="slide_k_rwm", knob="recurrent weight-magnitude penalty", ref=0,
         task="CDDM, $N$ = 1000, 75,000 iterations",
         rungs=[("none\n$\\lambda$ = 0", f"{DATA_DIR}/CDDM_std_g0_recinit/EqType=h_N=1000_sr=1.2_iters=75000"),
                ("0.05", f"{DATA_DIR}/CDDM_std_g0_recinit/EqType=h_N=1000_rwm=0.05_iters=75000"),
                ("0.2", f"{DATA_DIR}/CDDM_std_g0_recinit/EqType=h_N=1000_rwm=0.2_iters=75000")]),
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


def read_rung(path, end_at=None):
    """Active-unit counts for one rung, at its own noisy-loss floor and at the end of its budget.

    Args:
        path: the cell directory holding one folder per seed.
        end_at: iteration to take the "end" count at, or None for the last probe. Set when a ladder's
            control was trained longer than its arms, so both ends are read at the same budget.
    Returns:
        dict with keys `floor` (counts where the noisy loss first reaches 1.05x its fitted floor),
        `end` (counts at the last participation probe), `cross` (crossing iterations), `late`
        (end-of-run loss over the floor, for the runs read), `n` (runs found) and `dropped` (one
        reason string per run with no usable crossing).
    """
    at_floor, at_end, cross, late, dropped = [], [], [], [], []
    runs = F1.runs_with_loss(path)
    for r in runs:
        it_a = np.asarray(r["it_act"], float)
        at_end.append(r["active"][-1] if end_at is None
                      else r["active"][int(np.argmin(np.abs(it_a - end_at)))])
        f = noisy_floor(r["it_loss"], r["loss"])
        if f["reason"]:
            dropped.append(f["reason"])
            continue
        T = f["t_star"]
        # a top-up whose trace was not joined to its parent's starts late; reading its first probe
        # as "the floor" would be silently wrong
        if not it_a[0] <= T <= it_a[-1]:
            dropped.append("participation trace does not cover the crossing")
            continue
        at_floor.append(r["active"][int(np.argmin(np.abs(it_a - T)))])
        cross.append(T)
        late.append(f["late_rise"])
    return dict(floor=np.array(at_floor, float), end=np.array(at_end, float),
                cross=np.array(cross, float), late=np.array(late, float), n=len(runs),
                dropped=dropped)


def ladder_slide(spec, n_total=1000):
    """One knob ladder as two panels: at its own loss floor, and at the end of training.

    Args:
        spec: one entry of LADDERS; n_total: the network size, for the ceiling line and y limit.
    Returns:
        the output path, or None if no rung had readable runs.
    """
    reads = [(lab, read_rung(p, spec.get("end_at"))) for lab, p in spec["rungs"]]
    if not any(len(r["end"]) for _, r in reads):
        print(f"  SKIP {spec['stem']}: no runs with a loss+participation trace")
        return None

    # a ladder with no run read at its floor gets no floor panel, rather than an empty one
    usable = any(len(r["floor"]) for _, r in reads)
    panels = ([("floor", "at 1.05$\\times$ its own loss floor"),
               ("end", "at the end of training")] if usable
              else [("end", "at the end of training")])

    ps.setup()
    # ONE COLUMN: the floor read-out above the end read-out, on the same rungs, so each rung's two
    # numbers line up vertically and the drop between them reads straight down the page
    fig, axes = plt.subplots(len(panels), 1, figsize=(W * 0.55, H * len(panels) * 0.95),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
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
    for ax in axes:
        ax.set_ylabel(f"active units of {n_total:,}")
    axes[0].annotate(f"all {n_total:,}", (axes[0].get_xlim()[0], n_total),
                     textcoords="offset points", xytext=(4, 3), fontsize=6.0, color=ps.MUTED)
    # on its OWN line: a one-line title longer than the axes widens the saved canvas, and the
    # panel then sits in a half-empty figure
    sub = "" if usable else "\n(no run reaches 1.05$\\times$ its noisy-loss floor)"
    fig.suptitle(f"{spec['knob']}\n{spec['task']}{sub}", fontsize=7.2, color=ps.INK,
                 y=0.995)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.86 if usable else 0.74,
                        bottom=0.10 if usable else 0.17, hspace=0.42)
    return ps.save(fig, spec["stem"], w_mm=104)


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
        reads = [(lab.replace("\n", " "), read_rung(path, spec.get("end_at")))
                 for lab, path in spec["rungs"]]
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
    """Print every rung's per-run counts at the floor and at the end, crossings, and exclusions."""
    for spec in LADDERS:
        print(f"\n=== {spec['knob']}  ({spec['task']})")
        print(f"{'rung':<18}{'n':>3}  {'active at floor':<20}{'active at end':<20}"
              f"{'crossing (k it)':<22}{'end/floor':<12}dropped")
        for lab, path in spec["rungs"]:
            r = read_rung(path, spec.get("end_at"))
            lab1 = lab.replace("\n", " ")
            if not r["n"]:
                print(f"{lab1:<18}  -  no runs")
                continue
            ints = lambda v: " ".join(f"{x:.0f}" for x in v) or "-"
            print(f"{lab1:<18}{r['n']:>3}  {ints(r['floor']):<20}{ints(r['end']):<20}"
                  f"{ints(r['cross'] / 1e3):<22}{' '.join(f'{x:.2f}' for x in r['late']) or '-':<12}"
                  f"{'; '.join(r['dropped'])}")


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
