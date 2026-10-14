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

R^2 AT THE FLOOR (`--tradeoff`). Where a run kept weight snapshots (`trainer.checkpoint_keep`), R^2
is MEASURED: held-out R^2 of the first snapshot at or after the crossing (snapshots every 500
iterations). Otherwise no weights exist at the crossing, and it is estimated from the end of the
run: R2_floor = 1 - (L(t*) / L_end) * (1 - R2_end), with L the
noisy training loss (running median at the crossing, median of the last 10% at the end) and R2_end
the held-out score of the final weights (heldout_r2.py). Against held-out R^2 measured on weights
saved at the crossing (nine claim-1 networks, trajectory 2026-10-13): eight within 0.006, one CDDM
network off by 0.0115 because two scorers differ by 0.011 on its same final weights. So on CDDM an
R^2 difference below ~0.01 is scoring noise.

Usage:  python trainRNNbrain/experiments_and_analysis/fig_knobs_at_floor.py [--probe | --tradeoff]
        (run from the repository root, on a CPU node; --probe prints the table and writes no
        figures; --tradeoff draws R^2 against active units, both at each run's floor)
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fig_paper_F1 as F1
import paperstyle as ps
from common import DATA_DIR, END_FRAC, noisy_floor, warm_start_chain
from heldout_r2 import heldout_r2, load_cache, save_cache

W, H = 7.6, 2.9
# kept weight snapshots, by <sweep>/<cell>/s<seed>/it<absolute iteration>/
SNAP_DIR = os.path.join(DATA_DIR, "..", "c1_snapshots")
SNAP_MAX_LAG = 1000     # a snapshot more than this many iterations after the crossing is not "at" it
# noise draws per held-out R^2 in this figure. One CDDM draw has an SD of ~0.011, so the project's 9
# draws leave ~0.004 of noise and two scorers differed by 0.01-0.013 on identical weights; 36 halves it.
# Same measure (heldout_r2), more draws.
R2_DRAWS = 36

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
    dict(stem="slide_k_activation", knob="activation function", ref=0, categorical=True,
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
    # the same ladder on CDDM (2026-10-13), against the claim-1 snapshot controls; same code, same
    # launcher. The controls ran 100k and the arms 50k, so both ends are read at 50k.
    dict(stem="slide_k_inputscale_cddm", knob="input weight scale", ref=0, end_at=50000,
         task="CDDM, $N$ = 1000, 50,000 iterations",
         rungs=[("default\ndraw", f"{DATA_DIR}/CDDM_c1_snapshots/EqType=h_N=1000_arm=control"),
                ("0.5", f"{DATA_DIR}/CDDM_c1_snapshots/EqType=h_N=1000_arm=winp0.5"),
                ("2", f"{DATA_DIR}/CDDM_c1_snapshots/EqType=h_N=1000_arm=winp2"),
                ("5", f"{DATA_DIR}/CDDM_c1_snapshots/EqType=h_N=1000_arm=winp5"),
                ("20", f"{DATA_DIR}/CDDM_c1_snapshots/EqType=h_N=1000_arm=winp20")]),
]


def snapshot_at(run_dir, t):
    """The kept weight snapshot at or just after iteration t, if the run (or a run it was
    warm-started from) kept snapshots.

    A continuation's floor can fall inside its parent, whose snapshots sit under the PARENT's
    sweep, cell and seed, so every run of the warm-start chain is searched.

    Args:
        run_dir: a run folder (its name carries `seed=<seed>`); t: an absolute iteration.
    Returns:
        the snapshot folder, or None when there is none within SNAP_MAX_LAG after t.
    """
    snaps = []
    for d in warm_start_chain(run_dir):
        cell = os.path.dirname(d.rstrip("/"))
        seed = re.search(r"seed=(\d+)", os.path.basename(d.rstrip("/")))
        if seed:
            snaps += glob.glob(os.path.join(SNAP_DIR, os.path.basename(os.path.dirname(cell)),
                                            os.path.basename(cell), f"s{seed.group(1)}", "it*"))
    later = sorted((int(os.path.basename(d)[2:]), d) for d in snaps if int(os.path.basename(d)[2:]) >= t)
    return later[0][1] if later and later[0][0] - t <= SNAP_MAX_LAG else None


def floor_r2(run_dir, it, L, f, cache):
    """Held-out R^2 at a run's floor crossing: measured on a kept snapshot, else estimated.

    Args:
        run_dir: the run folder; it, L: its noisy loss series; f: noisy_floor's result for it;
        cache: an open heldout_r2 cache.
    Returns:
        (R^2, measured) - held-out R^2 of the snapshot at the crossing and True, or the estimate
        1 - (L(t*) / L_end) * (1 - R2_end) and False (see the module docstring).
    """
    snap = snapshot_at(run_dir, f["t_star"])
    if snap:
        return heldout_r2(snap, draws=R2_DRAWS, cache=cache), True
    L_star = f["smoothed"][np.flatnonzero(it >= f["t_star"])[0]]
    L_end = np.median(L[it >= it[-1] - END_FRAC * (it[-1] - it[0])])
    return 1.0 - (L_star / L_end) * (1.0 - heldout_r2(run_dir, draws=R2_DRAWS, cache=cache)), False


def read_rung(path, end_at=None, cache=None):
    """Active-unit counts for one rung, at its own noisy-loss floor and at the end of its budget.

    Args:
        path: the cell directory holding one folder per seed.
        end_at: iteration to take the "end" count at, or None for the last probe. Set when a ladder's
            control was trained longer than its arms, so both ends are read at the same budget.
        cache: an open heldout_r2 cache to also estimate R^2 at the floor, or None to skip it.
    Returns:
        dict with keys `floor` (counts where the noisy loss first reaches 1.05x its fitted floor),
        `end` (counts at the last participation probe), `cross` (crossing iterations), `late`
        (end-of-run loss over the floor, for the runs read), `r2` (held-out R^2 at the floor, one
        per `floor` entry; empty without a cache), `measured` (True where that R^2 was scored on a
        snapshot rather than estimated), `n` (runs found) and `dropped` (one reason string per run
        with no usable crossing).
    """
    at_floor, at_end, cross, late, r2, measured, dropped = [], [], [], [], [], [], []
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
        if cache is not None:
            v, m = floor_r2(os.path.join(path, r["name"]), np.asarray(r["it_loss"], float),
                            np.asarray(r["loss"], float), f, cache)
            r2.append(v)
            measured.append(m)
    return dict(floor=np.array(at_floor, float), end=np.array(at_end, float),
                cross=np.array(cross, float), late=np.array(late, float),
                r2=np.array(r2, float), measured=np.array(measured, bool), n=len(runs),
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


def tradeoff():
    """Every ladder as one scatter of held-out R^2 against active units, both at each run's floor.

    One point per run. A dose ladder is coloured along a sequential map in dose order; a categorical
    ladder (activation functions) uses one marker shape per category. The untouched network is
    ringed in black. Runs with no floor read-out are left out and the legend says how many (k/n).
    The six CDDM panels share one R^2 axis so they can be compared by eye.

    Returns:
        the output path.
    """
    cache = load_cache()
    ps.setup()
    fig, axes = plt.subplots(2, 4, figsize=(180 * ps.MM, 100 * ps.MM))
    cddm_axes = []
    for ax, spec in zip(axes.flat, LADDERS):
        reads = [(lab.replace("\n", " "), read_rung(p, spec.get("end_at"), cache))
                 for lab, p in spec["rungs"]]
        meas = np.concatenate([r["measured"] for _, r in reads])
        print(f"{spec['knob']} ({spec['task']}): R^2 measured on snapshots for {meas.sum()} of "
              f"{len(meas)} runs read, estimated for the rest")
        for lab, r in reads:
            print(f"   {lab:<24} active {r['floor'].astype(int).tolist()}  R^2 "
                  f"{np.round(r['r2'], 4).tolist()}  crossing {r['cross'].astype(int).tolist()}")
        cat = spec.get("categorical", False)
        seq = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(reads)))
        for i, (lab, r) in enumerate(reads):
            k = len(r["floor"])
            ref = i == spec["ref"]
            ax.scatter(r["floor"], r["r2"], s=22 if ref else 16,
                       marker="osD^v"[i] if cat else "o",
                       color=ps.SLOTS[i] if cat else seq[i],
                       edgecolors="k" if ref else "none", linewidths=0.8, zorder=3,
                       label=lab + ("" if k == r["n"] else f" ({k}/{r['n']})"))
        ax.set_xlim(0, 1000)
        # the title gives WHERE the runs were read - the range of their floor crossings - not the
        # training budget, which would read as the read-out point (Pavel, 2026-10-13)
        cross = np.concatenate([r["cross"] for _, r in reads]) / 1e3
        task = "CDDM" if spec["task"].startswith("CDDM") else "flip-flop"
        ax.set_title(f"{spec['knob']}\n{task}, floor reached at {cross.min():.0f}-{cross.max():.0f}k it",
                     fontsize=6.4, color=ps.INK, pad=3)
        ax.legend(fontsize=5.2, frameon=False, loc="lower left", handletextpad=0.2,
                  borderaxespad=0.2, labelspacing=0.25)
        if spec["task"].startswith("CDDM"):
            cddm_axes.append(ax)
    save_cache(cache)
    lo = min(a.get_ylim()[0] for a in cddm_axes)
    hi = max(a.get_ylim()[1] for a in cddm_axes)
    for a in cddm_axes:
        a.set_ylim(lo, hi)
    for a in axes[:, 0]:
        a.set_ylabel("held-out $R^2$ at the floor")
    for a in axes[-1, :]:
        a.set_xlabel("active units (of 1000)")
    for a in axes.flat[len(LADDERS):]:
        fig.delaxes(a)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.09, wspace=0.32, hspace=0.55)
    return ps.save(fig, "fig_f2_knob_tradeoff", w_mm=180)


def main(probe_only=False, tradeoff_only=False):
    """Build every ladder, the trade-off scatter, or just print the table. Returns paths written."""
    if probe_only:
        probe()
        return []
    if tradeoff_only:
        return [tradeoff()]
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
    ap.add_argument("--tradeoff", action="store_true", help="draw R^2 against active units")
    a = ap.parse_args()
    main(a.probe, a.tradeoff)
