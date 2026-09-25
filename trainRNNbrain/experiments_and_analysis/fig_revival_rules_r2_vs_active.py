"""Task performance against active-unit count, per network, for each unit-revival rule.

Every point is one trained network: its held-out r2 against the number of units active at the end
of training. N=1000, 3-bit flip-flop, 40,000 iterations, replacement rate 0.025, maturity 1000 --
identical for every rule, so the rule is the only difference between the arms.

The rules:
  orth       the new unit's incoming weights are drawn in the orthogonal complement of the
             surviving population, so they are novel by construction (cosine 1.9e-07 to every
             survivor)
  mix        a Dirichlet blend of four live units, so the new unit sits where the network is
             already active without being a twin of anybody
  bias_kick  no weight is touched at all; a bias offset is written directly, which lifts the unit
             above zero on some timesteps and unfreezes it
  random     a plain redraw, the floor
  zero_out   resample incoming, zero outgoing -- the rule from Dohare et al., Nature 632:768 (2024)
  copy       function-preserving duplication of a working unit

⚠️ THE `copy` POINTS ARE SUPERSEDED and drawn hollow. They come from the construction used before
2026-09-24, which zeroed the 2x2 weight block spanning donor and copy. That left the copy with no
self-connection and the donor with half of its own, so one duplication moved the network's output
by up to 8.9e-03 against an output scale of 0.27 instead of the 4.5e-08 the corrected construction
achieves. Those cells are re-running; the points are kept here only to show the scale of the
recruitment the other rules are being compared against.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito, which stays distinguishable under the common forms of colour blindness
COL = {"control": "#000000", "copy": "#999999", "orth": "#0072B2", "mix": "#009E73",
       "bias_kick": "#D55E00", "random": "#CC79A7", "zero_out": "#E69F00"}
ORDER = ["control", "random", "zero_out", "orth", "mix", "bias_kick", "copy"]
# hand-placed label offsets: several arms sit within a few units and a few thousandths of r2 of
# each other, so a uniform offset stacks their labels on top of one another
LABEL_OFFSET = {"control": (34, -22), "random": (-4, 17), "zero_out": (40, -6),
                "orth": (-32, 10), "mix": (20, 12), "bias_kick": (0, 16), "copy": (0, 18)}
DATA = "/private/tmp/claude-502/-Users-pt1290-Documents-GitHub-trainRNNbrain/1bc3843f-10a1-471f-875d-5f58458e098b/scratchpad/scatter_data.json"
OUT = "img/internal_figures/fig_revival_rules_r2_vs_active.png"

rows = json.load(open(DATA))["rules"]
ctrl = [r for r in rows if r["arm"] == "control"]
ctrl_a, ctrl_q = np.mean([r["active"] for r in ctrl]), np.mean([r["r2"] for r in ctrl])

# Two panels: duplication sits at 702 active units and squeezes the five non-copy rules into the
# left third of a single axis, so the left panel drops it and the right panel keeps it for scale.
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), sharey=True,
                         gridspec_kw={"width_ratios": [1.15, 1]})

for ax, arms, title in (
        (axes[0], [a for a in ORDER if a != "copy"], "the five non-copy rules, against the control"),
        (axes[1], ORDER, "with duplication, for scale")):
    ax.axvline(ctrl_a, color="0.8", lw=1, ls=":", zorder=1)
    ax.axhline(ctrl_q, color="0.8", lw=1, ls=":", zorder=1)
    for arm in arms:
        s_rows = [r for r in rows if r["arm"] == arm]
        if not s_rows:
            continue
        x = np.array([r["active"] for r in s_rows], float)
        y = np.array([r["r2"] for r in s_rows], float)
        superseded = arm == "copy"
        ax.scatter(x, y, s=95, zorder=3,
                   label=(f"{arm} (superseded)" if superseded else arm) if ax is axes[0] or superseded else None,
                   facecolor="none" if superseded else COL[arm],
                   edgecolor=COL[arm] if superseded else "white",
                   linewidth=1.8 if superseded else 1.2)
        if ax is axes[0]:
            ax.annotate(arm, (x.mean(), y.mean()), textcoords="offset points",
                        xytext=LABEL_OFFSET[arm], ha="center", color=COL[arm], fontsize=9.5,
                        fontweight="bold")
    ax.set_xlabel("active units at 40k")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.93", lw=0.8, zorder=0)

axes[0].set_ylabel("held-out $r^2$")
axes[0].annotate("control mean", (ctrl_a, ctrl_q), textcoords="offset points", xytext=(8, -46),
                 ha="left", color="0.6", fontsize=8)
axes[0].legend(frameon=False, loc="lower right", fontsize=8.5, ncol=2)
axes[1].legend(frameon=False, loc="lower left", fontsize=8.5)
fig.suptitle("Only bias_kick and duplication move the active count outside the control's range",
             fontsize=12.5)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=160)
print(f"wrote {OUT}")

print(f"\nN=1000, rate 0.025, maturity 1000. Control: {ctrl_a:.1f} active, r2 {ctrl_q:.4f}\n")
print(f"{'rule':>10s} {'n':>2s} {'active (per seed)':>22s} {'mean':>7s} {'vs ctrl':>8s} "
      f"{'r2':>8s} {'vs ctrl':>9s} {'separates?':>11s}")
print("-" * 86)
ctrl_lo, ctrl_hi = min(r["active"] for r in ctrl), max(r["active"] for r in ctrl)
for arm in ORDER:
    s = [r for r in rows if r["arm"] == arm]
    if not s:
        continue
    a = np.array(sorted(r["active"] for r in s), float)
    q = np.array([r["r2"] for r in s], float)
    sep = "yes" if (a.min() > ctrl_hi or a.max() < ctrl_lo) else "no"
    print(f"{arm:>10s} {len(s):2d} {', '.join(str(int(v)) for v in a):>22s} {a.mean():7.1f} "
          f"{a.mean()/ctrl_a:7.2f}x {q.mean():8.4f} {q.mean()-ctrl_q:+9.4f} "
          f"{(sep if arm != 'control' else '-'):>11s}")
