#!/usr/bin/env python3
"""
Manuscript Figure 3 - THE RATE PENALTY. Why units die for free, why the field-standard remedy makes
it worse, and what a penalty with a target instead of a floor recovers.

The claim this figure has to land is mechanistic, not just empirical: silence is cheap because the
task loss cannot see a unit's activity SCALE, and the usual activity regulariser makes it cheaper
still because its minimum is at zero. A penalty whose minimum is at a non-zero target inverts that.

  (a) WHY A UNIT CAN DIE FOR FREE      the ReLU scale symmetry, drawn. Multiply everything going
                                       into a unit by a > 0 and everything coming out by 1/a and the
                                       network computes exactly the same function, so the loss is
                                       constant along that direction and a unit's activity level is
                                       not identifiable from the task. And once a unit's input is
                                       negative on every trial its gradient is exactly zero: it can
                                       never come back.
  (b) WHERE EACH PENALTY PUTS ITS MINIMUM   the three penalties as functions of one unit's activity,
                                       drawn from the implementations in Trainer.py. The metabolic
                                       cost - the standard term in this literature - is minimised at
                                       r = 0: it PAYS a unit to be silent. `frm` is two-sided about
                                       a target cap, so silence is the most expensive thing a unit
                                       can do. That is the whole mechanism.
  (c) WHAT IT RECOVERS, ON EVERY TASK  active units, five tasks x four penalty arms, every seed.
                                       `frm` restores essentially the entire network everywhere;
                                       weight sparsity alone makes it WORSE everywhere.
  (d) WHAT IT COSTS                    task performance on the same cells. The cost is small and
                                       consistent (<= 0.008 in r^2) - and on the hardest memory task
                                       the rate penalty does not cost anything, it is the only arm
                                       that ever solves it.

Usage:  python fig_paper_F3.py
Output: img/internal_figures/fig_paper_F3.png
"""

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
from common import DATA_DIR, SILENT_REL

N_UNITS = 1000
READ_AT = 150_000
DMTS36_NPZ = "data/dmts_curves_delay36.npz"

# Penalty constants, read off trainRNNbrain/trainer/Trainer.py and configs/trainer/*.yaml so the
# curves in panel (b) are the functions actually optimised, not sketches of them.
UPV = 100               # Trainer.UpV, "N units per unit of volume, hard constant"
CAP_FR, G_TOP, G_BOT = 0.3, 3.0, 3.0
LAMBDA_FRM, LAMBDA_RWS, LAMBDA_MET = 0.1, 0.05, 0.1

PENS = [("none", "no penalty", ps.BASE),
        ("rws", "sparsity only\n(rws)", ps.COND_COL["rws"]),
        ("frm", "rate penalty\n(frm)", ps.COND_COL["frm"]),
        ("both", "frm + rws", ps.COND_COL["both"])]

D = DATA_DIR
TASKS = [
    ("3-bit\nflip-flop", {
        "none": f"{D}/NBitFlipFlop_std_ksweep/EqType=h_k=3_N=1000_iters=500000",
        "rws":  f"{D}/NBitFlipFlop_std_pen/EqType=h_k=3_N=1000_pen=rws",
        "frm":  f"{D}/NBitFlipFlop_std_penlong/EqType=h_k=3_N=1000_pen=frm_iters=400000",
        "both": f"{D}/NBitFlipFlop_std_penlong/EqType=h_k=3_N=1000_pen=both_iters=400000"}),
    ("hyper\nflip-flop", {
        "none": f"{D}/NBitFlipFlopHyper_std_hyper/EqType=h_k=4_N=1000_pen=none",
        "both": f"{D}/NBitFlipFlopHyper_std_hyper/EqType=h_k=4_N=1000_pen=both"}),
    ("CDDM", {
        "none": f"{D}/CDDM_std_g0_drift/EqType=h_N=1000_iters=200000",
        "rws":  f"{D}/CDDM_std_g0_penalties/EqType=h_N=1000_pen=rws",
        "frm":  f"{D}/CDDM_std_g0_penalties/EqType=h_N=1000_pen=frm",
        "both": f"{D}/CDDM_std_g0_penalties/EqType=h_N=1000_pen=both"}),
    ("DMTS\n16τ", {
        "none": f"{D}/DMTS_std_pen/EqType=h_N=1000_pen=none",
        "rws":  f"{D}/DMTS_std_pen/EqType=h_N=1000_pen=rws",
        "frm":  f"{D}/DMTS_std_pen/EqType=h_N=1000_pen=frm",
        "both": f"{D}/DMTS_std_pen/EqType=h_N=1000_pen=both"}),
    ("DMTS\n36τ", {
        "none": f"{D}/DMTS_std_delay36/EqType=h_N=1000_pen=none",
        "frm":  f"{D}/DMTS_std_delay36/EqType=h_N=1000_pen=frm",
        "both": f"{D}/DMTS_std_delay36/EqType=h_N=1000_pen=both"}),
]


def live_at(cell, cap=READ_AT):
    """Active units per seed at the largest iteration every seed in the cell reaches.

    Args:
        cell: cell folder; cap: read no later than this iteration.
    Returns:
        (n_seeds,) int array, empty if the cell has no readable traces.
    """
    tr = []
    for f in sorted(glob.glob(os.path.join(cell, "*", "*ParticipationTrace.pkl"))):
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        P, it = np.asarray(d.get("participation", [])), np.asarray(d.get("participation_iters", []))
        if len(it) and P.ndim == 2:
            tr.append((P, it))
    if not tr:
        return np.array([])
    r = min(min(int(it[-1]) for _, it in tr), cap)
    out = []
    for P, it in tr:
        p = P[int(np.argmin(np.abs(it - r)))]
        out.append(int((p >= SILENT_REL * np.quantile(p, 0.95)).sum()))
    return np.array(out)


def dmts36_readouts():
    """Clean r^2 per seed on the 36-tau task, at matched compute AND at the best checkpoint.

    These differ, and only on this task. Everywhere else a run's best checkpoint is its last one, so
    the score in the folder name is the matched-compute read-out. At 36 tau the `frm` runs find the
    memory solution and then collapse, so the folder name records a peak the network no longer
    holds. Reporting only that would break this paper's own read-out rule, in the treatment arm
    only -- exactly the comparison a referee should refuse.

    Returns:
        dict arm -> (final array, peak array), clean r^2 per seed.
    """
    z = np.load(DMTS36_NPZ, allow_pickle=True)
    tv = float(z["target_variance"])
    out = {}
    for pen in ("none", "frm", "both"):
        seeds = sorted({k.split("_")[2] for k in z.files if k.startswith(f"1000_{pen}_")})
        fin, pk = [], []
        for sd in seeds:
            r2 = 1.0 - z[f"1000_{pen}_{sd}_loss"] / tv
            fin.append(float(np.median(r2[-500:])))
            pk.append(float(np.nanmax(r2)))
        out[pen] = (np.array(fin), np.array(pk))
    return out


def r2_at(cell):
    """Task r^2 per seed, taken from the score prefix each run folder is named with.

    Args:
        cell: cell folder.
    Returns:
        (n_seeds,) float array, empty if the cell is missing.
    """
    out = []
    for d in sorted(glob.glob(os.path.join(cell, "*", ""))):
        try:
            out.append(float(os.path.basename(d.rstrip("/")).split("_")[0]))
        except ValueError:
            pass
    return np.array(out)


def frm_penalty(a, N=N_UNITS):
    """The firing-rate-magnitude penalty as a function of one unit's activity.

    Mirrors Trainer.fr_magnitude_penalty for a single unit: the per-unit term is
    ((cap - a)_+ / cap)^g_bot + ((a - cap)_+ / cap)^g_top, with cap = cap_fr * log1p(UpV)/log1p(N).
    `a` is the unit's soft-max activity over time, not its mean rate.

    Args:
        a: activity values (array); N: network size, which sets the cap.
    Returns:
        (penalty values, cap) with penalty the same shape as `a`.
    """
    cap = CAP_FR * np.log1p(UPV) / np.log1p(N)
    over = np.maximum(np.asarray(a, float) - cap, 0.0)
    under = np.maximum(cap - np.asarray(a, float), 0.0)
    return (under / cap) ** G_BOT + (over / cap) ** G_TOP, cap


def panel_a(ax):
    """Panel (a): the objective never asks for units, and the ReLU-specific story is not the cause.

    THIS PANEL WAS REBUILT AFTER A CORRECTION (Pavel, 2026-09-20). It previously argued the whole
    mechanism from the ReLU scale symmetry and the exactly-zero gradient of a dead ReLU. That
    cannot be right: Figure 1e shows silence on softplus, leaky ReLU and a bounded sigmoid too, and
    a ReLU-specific argument cannot explain an observation that is not specific to ReLU. Worse, our
    own activation sweep points the other way -- see the table drawn at the bottom of the panel.

    What survives is weaker and general: the task loss is a function of the OUTPUT, so once the task
    is solved, a solution carried by M units scores exactly as well as one carried by N. Nothing in
    the objective rewards recruiting the rest.
    """
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))

    ax.text(0.5, 0.975, "The objective never asks for the units", ha="center", fontsize=6.8,
            color=ps.INK)

    # two networks, same output, very different numbers of active units
    ax.figure.canvas.draw()
    dx = 0.0135
    dy = ps.square_pitch(ax, dx)
    for col_i, (n_on, lab) in enumerate([(26, "260 of 1000 units active"),
                                         (100, "1000 of 1000 units active")]):
        x0 = 0.085 + col_i * 0.50
        ps.unit_grid(ax, x0, 0.845, n_on, 100, col=ps.SLOTS[0], off_col="#d9d8d1",
                     pitch=(dx, dy), s=3.6, lw=0.34)
        ax.text(x0 + 4.5 * dx, 0.885, lab, ha="center", fontsize=5.9, color=ps.INK)
        ps.box(ax, x0 + 4.5 * dx - 0.088, 0.548, 0.176, 0.056, "same output", col=ps.MUTED,
               face="#f2f1ec", lw=0.6, fs=5.2)
        ps.arrow(ax, (x0 + 4.5 * dx, 0.660), (x0 + 4.5 * dx, 0.610), col=ps.MUTED)
    ax.text(0.5, 0.735, "=", ha="center", va="center", fontsize=11, color=ps.INK)
    ax.text(0.5, 0.492, "identical task loss --- the loss is indifferent between them",
            ha="center", fontsize=6.0, color=ps.INK)
    ax.text(0.5, 0.432,
            "gradient descent recruits what it recruits early; nothing enlarges that set",
            ha="center", fontsize=5.6, color=ps.MUTED)

    # the ReLU-specific sharpenings, and the measurements that rule them out as the cause
    ps.box(ax, 0.01, 0.045, 0.98, 0.345, col=ps.MUTED, face="#f6f5f0", lw=0.6, pad=0.012)
    ax.text(0.5, 0.352, "ReLU sharpens this in two ways --- neither of which is what drives it",
            ha="center", fontsize=6.0, color=ps.INK)
    rows = [("exact scale symmetry",
             "$\\mathrm{relu}(ax)=a\\,\\mathrm{relu}(x)$, so a unit's activity level is unidentifiable",
             "but remove it (softplus, sigmoid) and silence gets WORSE, not better"),
            ("death is absorbing",
             "below threshold on every trial $\\mathrm{relu}'=0$, so the unit cannot return",
             "but remove it (leaky ReLU) and nothing changes: $+6\\pm14$ units")]
    for i, (head, what, test) in enumerate(rows):
        y = 0.292 - i * 0.128
        ax.text(0.045, y, head, fontsize=5.9, color=ps.INK, va="center", ha="left")
        ax.text(0.045, y - 0.037, what, fontsize=5.1, color=ps.MUTED, va="center", ha="left")
        ax.text(0.045, y - 0.072, test, fontsize=5.3, color=ps.BAD, va="center", ha="left")
    ax.plot([0.045, 0.955], [0.176, 0.176], lw=0.4, color="#dedcd4", zorder=1)


def panel_b(ax):
    """Panel (b): where each penalty puts its minimum, as a function of a unit's activity."""
    _, cap = frm_penalty(np.zeros(1))
    a = np.linspace(0, 2.1 * cap, 700)
    frm, _ = frm_penalty(a)

    met = (a / cap) ** 2                     # metabolic cost is mean(r^2), on the same activity axis
    ax.plot(a, met / met.max(), lw=1.4, color=ps.SLOTS[4], zorder=4,
            label="metabolic  $\\lambda\\,\\langle r^2\\rangle$")
    ax.plot(a, frm / frm.max(), lw=1.6, color=ps.COND_COL["frm"], zorder=5,
            label="rate penalty  $\\mathtt{frm}$")

    ax.axvline(cap, color=ps.MUTED, lw=0.6, ls=":", zorder=2)
    ax.text(cap * 0.97, 0.90, "target\ncap", fontsize=5.8, color=ps.MUTED, va="top",
            ha="right", linespacing=1.2)

    ax.plot(0, 0, "o", ms=5, color=ps.SLOTS[4], zorder=7, clip_on=False)
    ax.annotate("minimum at $r=0$:\nsilence is REWARDED", xy=(0.004, 0.005),
                xytext=(0.30 * cap, 0.46), fontsize=5.8, color=ps.SLOTS[4], ha="left",
                linespacing=1.3,
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=ps.SLOTS[4], mutation_scale=6))
    ax.plot(cap, 0, "o", ms=5, color=ps.COND_COL["frm"], zorder=7)
    ax.annotate("minimum at the cap:\nsilence is the most\nexpensive state a unit can be in",
                xy=(cap, 0.012), xytext=(1.03 * cap, 0.60), fontsize=5.8,
                color=ps.COND_COL["frm"], ha="left", linespacing=1.3,
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=ps.COND_COL["frm"],
                                mutation_scale=6))
    ax.set(xlabel="unit's activity  (soft-max over time)",
           ylabel="penalty paid by that unit\n(each curve scaled to its own maximum)",
           xlim=(-0.005, 2.1 * cap), ylim=(-0.03, 1.10), yticks=[])
    ax.legend(loc="upper center", fontsize=6.0, ncol=1, bbox_to_anchor=(0.52, 1.0))
    ps.ygrid(ax)
    return cap


def panel_c(ax):
    """Panel (c): active units, five tasks x four penalty arms. Returns the measured rows."""
    rows, centres = [], []
    width = 0.20
    for ti, (task, cells) in enumerate(TASKS):
        centres.append(ti)
        for pi, (pen, _, col) in enumerate(PENS):
            if pen not in cells:
                continue
            v = live_at(cells[pen])
            if not len(v):
                continue
            x = ti + (pi - 1.5) * width
            ps.strip(ax, [x], [v], [col], width=width * 0.42, jitter=0.022,
                     rng=np.random.default_rng(10 + ti * 4 + pi), ms=2.6)
            rows.append((task.replace("\n", " "), pen, v.mean(), v.std(ddof=1), len(v)))
    ax.axhline(N_UNITS, color=ps.MUTED, lw=0.7, ls="--", zorder=1)
    ax.text(len(TASKS) - 0.52, N_UNITS + 22, "every unit active", ha="right", fontsize=5.6,
            color=ps.MUTED)
    ax.set(xticks=centres, xticklabels=[t for t, _ in TASKS], ylabel="active units",
           ylim=(0, 1120), xlim=(-0.55, len(TASKS) - 0.45))
    ax.tick_params(axis="x", labelsize=6.0)
    ax.legend(handles=[Line2D([], [], color=c, marker="o", ls="", ms=3.2,
                              label=l.replace("\n", " ")) for _, l, c in
                       [(p, l, c) for p, l, c in PENS]],
              loc="center left", fontsize=5.8, ncol=2, bbox_to_anchor=(0.01, 0.62))
    ps.ygrid(ax)
    return rows


def panel_d(ax):
    """Panel (d): task r^2 relative to that task's unpenalised arm. Returns the measured rows."""
    rows, centres = [], []
    width = 0.20
    dm = dmts36_readouts() if os.path.exists(DMTS36_NPZ) else {}
    for ti, (task, cells) in enumerate(TASKS):
        centres.append(ti)
        is_dmts36 = task.startswith("DMTS") and "36" in task
        base = dm["none"][0] if (is_dmts36 and dm) else r2_at(cells["none"])
        if not len(base):
            continue
        for pi, (pen, _, col) in enumerate(PENS):
            if pen not in cells:
                continue
            v = dm[pen][0] if (is_dmts36 and dm and pen in dm) else r2_at(cells[pen])
            if not len(v):
                continue
            x = ti + (pi - 1.5) * width
            ps.strip(ax, [x], [v - base.mean()], [col], width=width * 0.42, jitter=0.022,
                     rng=np.random.default_rng(50 + ti * 4 + pi), ms=2.6)
            rows.append((task.replace("\n", " "), pen, float(v.mean() - base.mean()),
                         float(v.std(ddof=1)), len(v)))
            # the best checkpoint, drawn hollow, only where it differs from the matched-compute read
            if is_dmts36 and dm and pen in dm:
                pk = dm[pen][1].mean() - dm["none"][1].mean()
                if abs(pk - (v.mean() - base.mean())) > 0.10:
                    ax.plot(x, pk, "o", ms=4.0, mfc="none", mec=col, mew=0.9, zorder=6)
    ax.axhline(0, color=ps.INK, lw=0.8, zorder=3)
    ax.text(-0.48, 0.012, "no cost", fontsize=5.8, color=ps.INK, va="bottom")
    ax.annotate("best checkpoint: the only arm that\never solves the 36τ delay (Fig. 4d)",
                xy=(len(TASKS) - 1 + 0.5 * width, 0.37), xytext=(2.25, 0.30),
                fontsize=5.6, color=ps.COND_COL["frm"], ha="center", linespacing=1.3,
                arrowprops=dict(arrowstyle="-|>", lw=0.55, color=ps.COND_COL["frm"],
                                mutation_scale=6))
    ax.annotate("…but does not hold it: at matched\ncompute it ends below baseline",
                xy=(len(TASKS) - 1 + 0.42 * width, -0.41), xytext=(2.55, -0.155),
                fontsize=5.6, color=ps.COND_COL["frm"], ha="center", linespacing=1.3,
                arrowprops=dict(arrowstyle="-|>", lw=0.55, color=ps.COND_COL["frm"],
                                mutation_scale=6))
    ax.plot([], [], "o", ms=4.0, mfc="none", mec=ps.MUTED, mew=0.9, label="best checkpoint")
    ax.plot([], [], "o", ms=3.4, color=ps.MUTED, label="matched compute (end of training)")
    ax.legend(loc="lower left", fontsize=5.6, bbox_to_anchor=(-0.01, -0.02))
    ax.set_yscale("symlog", linthresh=0.02, linscale=1.6)
    ax.set(xticks=centres, xticklabels=[t for t, _ in TASKS],
           ylabel="task $r^2$ − unpenalised $r^2$", xlim=(-0.55, len(TASKS) - 0.45),
           ylim=(-0.72, 0.62))
    ax.set_yticks([-0.4, -0.1, -0.02, 0, 0.02, 0.1, 0.4])
    ax.set_yticklabels(["−0.4", "−0.1", "−0.02", "0", "0.02", "0.1", "0.4"])
    ax.axhspan(-0.02, 0.02, color="#f6f5f0", zorder=0)
    ax.text(len(TASKS) - 0.52, 0.021, "linear below ±0.02, log outside", ha="right",
            fontsize=5.2, color=ps.FAINT, va="bottom")
    ax.tick_params(axis="x", labelsize=6.0)
    ps.ygrid(ax)
    return rows


def main():
    """Assemble Figure 3 and write it. Returns the output path."""
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 158 * ps.MM))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.12, 0.94], width_ratios=[1.10, 1.0],
                  hspace=0.40, wspace=0.24)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_a(ax_a)
    ps.panel_letter(ax_a, "a", dx=-0.02, dy=1.0)
    ax_a.text(-0.02, 1.10, "Why units die for free", transform=ax_a.transAxes, fontsize=7.4,
              color=ps.INK, fontweight="bold")

    ax_b = fig.add_subplot(gs[0, 1])
    cap = panel_b(ax_b)
    ps.panel_letter(ax_b, "b")
    ax_b.text(-0.13, 1.10, "Why the standard remedy makes it worse",
              transform=ax_b.transAxes, fontsize=7.4, color=ps.INK, fontweight="bold")

    ax_c = fig.add_subplot(gs[1, 0])
    rows_c = panel_c(ax_c)
    ps.panel_letter(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    rows_d = panel_d(ax_d)
    ps.panel_letter(ax_d, "d")

    out = ps.save(fig, "fig_paper_F3")

    print(f"\n  frm target cap at N={N_UNITS}: {cap:.4f}")
    print("\n--- active units ---")
    for task, pen, m, sd, n in rows_c:
        print(f"  {task:18} {pen:5} {m:7.1f} ± {sd:5.1f}  ({m / 10:.1f}%)  n={n}")
    print("\n--- task r^2 relative to unpenalised ---")
    for task, pen, d, sd, n in rows_d:
        print(f"  {task:18} {pen:5} {d:+.4f} ± {sd:.4f}  n={n}")
    return out


if __name__ == "__main__":
    main()
