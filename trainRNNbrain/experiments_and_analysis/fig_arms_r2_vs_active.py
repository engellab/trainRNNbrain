"""Held-out r2 against active-unit count, per network, for every intervention that recruits units.

One point per trained network. 3-bit flip-flop, 40,000 iterations, three seeds per arm.

Filled markers are gamma=0, the model configuration used everywhere else in this project, so
control, mute, dead and rate-capped duplication are directly comparable within a panel. Open
markers are gamma=0.1 (cubic saturation in the dynamics), which changes the base network and
therefore carries its own control -- comparisons across the two marker styles are not like for like.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {                       # arm: (colour, filled?, label)
    "control":     ("black",   True,  "no intervention"),
    "mute":        ("#2471A3", True,  "dropout, mute"),
    "dead":        ("#C0392B", True,  "dropout, dead"),
    "duplicate":   ("#1E8449", True,  "duplication (rate-capped)"),
    "control_g":   ("black",   False, "no intervention, $\\gamma$=0.1"),
    "duplicate_g": ("#1E8449", False, "duplication, $\\gamma$=0.1"),
}
rows = json.load(open("/tmp/all_arms.json"))
OUT = "img/internal_figures/fig_arms_r2_vs_active.png"

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax, N in zip(axes, (500, 1000, 2000)):
    for arm, (col, filled, lab) in STYLE.items():
        s = [r for r in rows if r["N"] == N and r["arm"] == arm]
        if not s:
            continue
        x = [r["active"] for r in s]
        y = [r["r2"] for r in s]
        ax.scatter(x, y, s=95, zorder=3, label=lab if N == 500 else None,
                   color=col if filled else "none", edgecolor=col,
                   linewidth=1.6 if not filled else 1.1,
                   marker="o" if filled else "s")
    ax.axvline(N, color="0.8", lw=1, ls=":", zorder=1)
    ax.text(N, 0.762, f"all {N}", rotation=90, ha="right", va="bottom",
            color="0.55", fontsize=8)
    ax.set_title(f"N = {N}", fontsize=12)
    ax.set_xlabel("active units at 40k")
    ax.set_xlim(0, N * 1.1)
    ax.set_ylim(0.755, 0.965)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.93", lw=0.8, zorder=0)

axes[0].set_ylabel("held-out $r^2$")
axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
fig.suptitle("Duplication recruits units at almost no cost in performance; dropout does not",
             fontsize=13)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=160)
print(f"wrote {OUT}\n")

print(f"{'N':>5s} {'arm':>14s} {'active':>8s} {'r2':>8s} {'units gained':>13s} {'r2 cost':>9s} "
      f"{'units per r2 point':>19s}")
print("-" * 82)
for N in (500, 1000, 2000):
    base = [r for r in rows if r["N"] == N and r["arm"] == "control"]
    baseg = [r for r in rows if r["N"] == N and r["arm"] == "control_g"]
    for arm in ("mute", "dead", "duplicate", "duplicate_g"):
        s = [r for r in rows if r["N"] == N and r["arm"] == arm]
        if not s:
            continue
        ref = baseg if arm.endswith("_g") else base
        da = np.mean([r["active"] for r in s]) - np.mean([r["active"] for r in ref])
        dr = np.mean([r["r2"] for r in ref]) - np.mean([r["r2"] for r in s])
        per = f"{da / dr:>8.0f}" if dr > 1e-4 else "     free"
        print(f"{N:5d} {arm:>14s} {np.mean([r['active'] for r in s]):8.1f} "
              f"{np.mean([r['r2'] for r in s]):8.4f} {da:+13.0f} {dr:+9.4f} {per:>19s}")
