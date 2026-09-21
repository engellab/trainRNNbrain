#!/usr/bin/env python3
"""
Supplementary figure S7 - the 36-tau delayed-match-to-sample task.

Moved out of the main text (Pavel, 2026-09-21): it is a different kind of comparison from the rest
of the paper. Everywhere else the unpenalised network solves the task and the question is what a
penalty costs. Here the unpenalised network never solves it at all, so "cost relative to baseline"
does not mean what it means elsewhere, and the arm that does solve it does not hold the solution.
That is worth reporting and is not worth three paragraphs of main text.

WHAT IT SHOWS. Clean r^2 = 1 - loss_clean/Var(target) against training iteration, every seed, rolling
median over 301 probes. The unpenalised network never leaves the no-memory plateau (3/3). The rate
penalty finds the memory solution in every seed, reaching r^2 = 0.9998, and then loses it, all three
collapsing before 150,000 iterations. Adding weight sparsity prevents the escape altogether (3/3).

BOTH READ-OUTS, because they differ only here. At matched compute (median clean r^2 over the last
500 probes) frm ends 0.409 BELOW the baseline; at its best checkpoint it ends 0.369 above. Quoting
only the peak would score the treatment at its maximum and the control at its plateau.

Usage:  python fig_supp_dmts36.py
Output: img/internal_figures/fig_supp_dmts36.pdf (+ .svg; vector only - see paperstyle.save)
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps

DMTS36 = "data/dmts_curves_delay36.npz"
DMTS_ARMS = [("none", "no penalty", ps.BASE),
             ("frm", "rate penalty (frm)", ps.COND_COL["frm"]),
             ("both", "frm + rws", ps.COND_COL["both"])]
ESCAPE = 0.9


def panel(ax):
    """Panel (d): the 36-tau memory task, clean r^2 per seed along training. Returns the verdicts."""
    if not os.path.exists(DMTS36):
        ax.text(0.5, 0.5, "dmts_curves_delay36.npz missing", ha="center", transform=ax.transAxes)
        return {}
    z = np.load(DMTS36, allow_pickle=True)
    tv = float(z["target_variance"])
    verdict = {}
    for pen, lab, col in DMTS_ARMS:
        seeds = sorted({k.split("_")[2] for k in z.files if k.startswith(f"1000_{pen}_")})
        best, fin = [], []
        for s in seeds:
            it = z[f"1000_{pen}_{s}_iters"]
            r2 = 1.0 - z[f"1000_{pen}_{s}_loss"] / tv
            w = 301
            n = (len(r2) // w) * w
            sm = np.median(r2[:n].reshape(-1, w), axis=1)
            tt = it[:n].reshape(-1, w)[:, w // 2]
            ax.plot(tt, sm, lw=0.85, color=col, alpha=0.9, zorder=4)
            best.append(np.nanmax(r2))
            fin.append(float(np.median(r2[-500:])))
        verdict[pen] = (np.array(best), np.array(fin), len(seeds))
    ax.axhline(ESCAPE, color=ps.MUTED, lw=0.7, ls="--", zorder=2)
    ax.text(1.45e3, ESCAPE + 0.025, "memory solved", fontsize=5.6, color=ps.MUTED)
    ax.axhline(0.605, color=ps.MUTED, lw=0.7, ls=":", zorder=2)
    ax.text(1.45e3, 0.512, "no-memory plateau", fontsize=5.6, color=ps.MUTED)
    ax.annotate("frm finds the memory\nin every seed…", xy=(1.35e4, 0.985), xytext=(1.5e3, 0.77),
                fontsize=5.8, color=ps.COND_COL["frm"], linespacing=1.25, ha="left",
                arrowprops=dict(arrowstyle="-|>", lw=0.55, color=ps.COND_COL["frm"],
                                mutation_scale=6))
    ax.annotate("…and then loses it", xy=(3.6e4, 0.03), xytext=(1.5e3, 0.20),
                fontsize=5.8, color=ps.COND_COL["frm"], ha="left",
                arrowprops=dict(arrowstyle="-|>", lw=0.55, color=ps.COND_COL["frm"],
                                mutation_scale=6))
    ax.set(xscale="log", xlabel="training iteration", ylabel="clean $r^2$ on the 36τ delay task",
           xlim=(1.2e3, 1.7e5), ylim=(-0.12, 1.06))
    ax.legend(handles=[Line2D([], [], color=c, lw=1.2, label=l) for _, l, c in DMTS_ARMS],
              loc="lower left", fontsize=5.8, bbox_to_anchor=(0.0, -0.02))
    ps.ygrid(ax)
    return verdict



def readouts():
    """Matched-compute and best-checkpoint clean r^2 per seed, per arm.

    Returns:
        dict arm -> (final array, peak array).
    """
    z = np.load(DMTS36, allow_pickle=True)
    tv = float(z["target_variance"])
    out = {}
    for pen, _, _ in DMTS_ARMS:
        seeds = sorted({k.split("_")[2] for k in z.files if k.startswith(f"1000_{pen}_")})
        fin = [float(np.median((1 - z[f"1000_{pen}_{s}_loss"] / tv)[-500:])) for s in seeds]
        pk = [float(np.nanmax(1 - z[f"1000_{pen}_{s}_loss"] / tv)) for s in seeds]
        out[pen] = (np.array(fin), np.array(pk))
    return out


def main():
    """Draw the supplementary 36-tau figure and print both read-outs. Returns the output path."""
    ps.setup()
    fig, axes = plt.subplots(1, 2, figsize=(ps.W2 * 0.86, 62 * ps.MM),
                             gridspec_kw=dict(width_ratios=[1.55, 1.0]))
    panel(axes[0])
    ps.panel_letter(axes[0], "a", dx=-0.10)

    ax = axes[1]
    r = readouts()
    for i, (pen, lab, col) in enumerate(DMTS_ARMS):
        fin, pk = r[pen]
        ps.strip(ax, [i - 0.17], [fin], [col], width=0.13, jitter=0.03,
                 rng=np.random.default_rng(7 + i), ms=3.0)
        ax.plot([i + 0.17] * len(pk), pk, "o", ms=3.4, mfc="none", mec=col, mew=0.9, zorder=5)
        ax.plot([i + 0.04, i + 0.30], [pk.mean()] * 2, lw=1.4, color=col, zorder=6)
    ax.axhline(ESCAPE, color=ps.MUTED, lw=0.7, ls="--", zorder=2)
    ax.text(-0.45, ESCAPE + 0.03, "memory solved", fontsize=5.6, color=ps.MUTED)
    ax.set(xticks=range(len(DMTS_ARMS)), xticklabels=[l for _, l, _ in DMTS_ARMS],
           ylabel="clean $r^2$ at the 36τ delay", ylim=(-0.15, 1.10), xlim=(-0.55, 2.55))
    ax.tick_params(axis="x", labelsize=5.8)
    ax.legend(handles=[Line2D([], [], color=ps.MUTED, marker="o", ls="", ms=3.2,
                              label="matched compute"),
                       Line2D([], [], color=ps.MUTED, marker="o", ls="", ms=3.4, mfc="none",
                              mew=0.9, label="best checkpoint")],
              loc="lower left", fontsize=5.8)
    ps.ygrid(ax)
    ps.panel_letter(ax, "b", dx=-0.16)

    out = ps.save(fig, "fig_supp_dmts36")
    print()
    for pen, lab, _ in DMTS_ARMS:
        fin, pk = r[pen]
        print(f"  {lab:20} matched compute {fin.mean():+7.4f} ± {fin.std(ddof=1):.4f} | "
              f"best checkpoint {pk.mean():.4f}")
    return out


if __name__ == "__main__":
    main()
