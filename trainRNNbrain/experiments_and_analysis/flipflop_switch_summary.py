#!/usr/bin/env python3
"""
All four arms of the penalty-switch intervention, on one page.

  A1  frm  -> frm+rws   treatment: what adding rws does
  A2  frm  -> frm       control for A1 (same parent, penalty unchanged)
  A3  both -> frm       reverse: what removing rws does
  A4  both -> frm+rws   control for A3 (same parent, penalty unchanged)

A1/A2 branch from the IDENTICAL frm parent and A3/A4 from the identical `both` parent, so each
contrast is paired and any difference is the penalty, not the seed.

⚠️ BOTH CONTROLS ARE LOAD-BEARING AND FOR DIFFERENT REASONS. A4 is inert (median +0.005,
rho(start,end) = 0.78), which is what proves warm-starting and 50k extra iterations do nothing on
their own. But A2 is NOT inert (+0.050): continued frm training moves the population by itself, so
reading A1 against zero rather than against A2 would credit rws with that drift.

⚠️ CHURN IS SAMPLING-RATE DEPENDENT AND MUST BE QUOTED WITH ITS RATE. Counting dead<->alive
crossings every 10 iterations gives 1.7 (A1) vs 3.2 (A2), only 1.9x. At every 250 iterations it is
0.35 vs 2.70, nearly 8x, because the two arms differ in KIND: the residual churn under frm+rws is
fast flicker at the finest timescale (its count falls 5.7x under coarser sampling) while frm-alone's
is slow persistent switching that survives any rate (1.09x). This script uses the 250-iteration rate
throughout and states it on the figure.

⚠️ A statistic deliberately NOT shown: corr(delta, start). It looks like the natural test of "low
starters gain most" but carries a large built-in negative bias - with an endpoint unrelated to the
start, corr(a, b-a) is already ~ -0.7 from regression to the mean - and that bias scales with the
group's variance, which differs between arms. Against each arm's own shuffled-endpoint null the
excess is +0.11 (A1) and +0.29 (A2): both LESS negative than chance, control higher. It supports
nothing.

Reads the reduced trace `switch_all4.npz` (temporal PR every 250 iterations, all units, 12 runs).

Output: img/internal_figures/switch_summary.png

Usage:  python flipflop_switch_summary.py [path/to/switch_all4.npz]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle as ps

ARMS = [("A1", "frm → frm+rws", "treatment", "#1f77b4"),
        ("A2", "frm → frm", "control", "#d62728"),
        ("A3", "both → frm", "reverse", "#ff7f0e"),
        ("A4", "both → frm+rws", "control", "#2ca02c")]
REPS = "012"


def stats(z, arm):
    """Per-seed summary statistics for one arm.

    Args:
        z: the loaded npz; arm: "A1".."A4".
    Returns:
        dict of (3,) arrays over seeds, plus the (n_time,) mean median-trajectory and dead-trajectory.
    """
    med, dead, iqr, rho, churn, never = [], [], [], [], [], []
    mt, dt = [], []
    for rep in REPS:
        T = z[f"{arm}_{rep}_tpr"].astype(float)
        a, b = T[0], T[-1]
        c = np.abs(np.diff((T == 0).astype(np.int8), axis=0)).sum(0)
        med.append((np.median(a), np.median(b)))
        dead.append(((a == 0).mean(), (b == 0).mean()))
        iqr.append(np.subtract(*np.percentile(b, [75, 25])))
        rho.append(spearmanr(a[a > 0], b[a > 0]).statistic)      # live-at-t0 only; ties at 0 bias it
        churn.append(c.mean()); never.append((c == 0).mean())
        mt.append(np.median(T, axis=1)); dt.append((T == 0).mean(1))
    return dict(med=np.array(med), dead=np.array(dead), iqr=np.array(iqr), rho=np.array(rho),
                churn=np.array(churn), never=np.array(never),
                mt=np.mean(mt, axis=0), dt=np.mean(dt, axis=0),
                it=z[f"{arm}_0_it"].astype(float))


def main():
    """Four-arm summary: occupancy trajectories, dead fraction, churn and endpoint statistics."""
    path = sys.argv[1] if len(sys.argv) > 1 else "switch_all4.npz"
    z = np.load(path)
    ps.setup()
    S = {a: stats(z, a) for a, _, _, _ in ARMS}

    print(f"{'arm':<4}{'switch':<16}{'median tPR':>16}{'dead frac':>16}{'IQR':>7}"
          f"{'rho':>7}{'churn':>8}{'never flip':>12}")
    for a, lab, _, _ in ARMS:
        s = S[a]
        print(f"{a:<4}{lab:<16}{s['med'][:,0].mean():>7.3f}→{s['med'][:,1].mean():<8.3f}"
              f"{s['dead'][:,0].mean():>7.3f}→{s['dead'][:,1].mean():<8.3f}"
              f"{s['iqr'].mean():>7.3f}{s['rho'].mean():>7.2f}{s['churn'].mean():>8.2f}"
              f"{s['never'].mean():>12.2f}")

    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))

    a = ax[0][0]
    for arm, lab, kind, col in ARMS:
        s = S[arm]
        a.plot(s["it"], s["mt"], color=col, lw=1.9, ls="--" if kind == "control" else "-",
               label=f"{arm}  {lab}" + ("  (control)" if kind == "control" else ""))
    a.set(xlabel="iterations since the switch", ylabel="median temporal PR / n",
          title="occupancy: the two arms that gain rws rise, the two that lose it fall")
    a.grid(alpha=.25); a.legend(fontsize=8)

    b = ax[0][1]
    for arm, lab, kind, col in ARMS:
        s = S[arm]
        b.plot(s["it"], s["dt"], color=col, lw=1.9, ls="--" if kind == "control" else "-", label=arm)
    b.set(xlabel="iterations since the switch", ylabel="dead fraction",
          title="dead units: rws drives them to zero and HOLDS them there\n"
                "without it the fraction sawtooths for the whole run")
    b.grid(alpha=.25); b.legend(fontsize=8)

    c = ax[1][0]
    x = np.arange(len(ARMS))
    ch = [S[a]["churn"] for a, _, _, _ in ARMS]
    c.bar(x, [v.mean() for v in ch], yerr=[v.std() for v in ch], capsize=4,
          color=[col for _, _, _, col in ARMS], alpha=.85)
    for i, v in enumerate(ch):
        c.text(i, v.mean(), f"{v.mean():.2f}", ha="center", va="bottom", fontsize=9)
    c.set(xticks=x, xticklabels=[f"{a}\n{'rws ON' if a in ('A1','A4') else 'rws OFF'}"
                                 for a, _, _, _ in ARMS],
          ylabel="dead↔alive crossings per unit",
          title="churn, sampled every 250 iterations\nordering is by whether rws is ON, not by the parent")
    c.grid(alpha=.25, axis="y")

    d = ax[1][1]
    d.axis("off")
    rows = [["arm", "switch", "median tPR", "dead", "IQR", "ρ", "churn"]]
    for a_, lab, kind, _ in ARMS:
        s = S[a_]
        rows.append([a_ + ("*" if kind == "control" else ""), lab,
                     f"{s['med'][:,0].mean():.3f}→{s['med'][:,1].mean():.3f}",
                     f"{s['dead'][:,0].mean():.3f}→{s['dead'][:,1].mean():.3f}",
                     f"{s['iqr'].mean():.3f}", f"{s['rho'].mean():.2f}", f"{s['churn'].mean():.2f}"])
    t = d.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.7)
    d.set_title("* = same-penalty control\n"
                "ρ on units live at the switch (ties at zero bias the all-unit version)",
                fontsize=9.5)
    fig.suptitle("Penalty-switch intervention, all four arms — N=2000, k=3, 3 seeds\n"
                 "A1/A2 share an frm parent, A3/A4 share a `both` parent; only the penalty after "
                 "the switch differs", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return ps.save(fig, "switch_summary", tight=False)


if __name__ == "__main__":
    main()
