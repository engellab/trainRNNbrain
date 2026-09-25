"""Task performance against active-unit count, per network, for the two dropout kinds.

Every point is one trained network: its held-out r2 against the number of units that end up active.
3-bit flip-flop, 40,000 iterations, drop rate 0.20, beta=4, three seeds per arm (six for the N=500
control). The control has dropout off -- `dropout: False` is the trainer default, and the
dropout_args block its config carries is inert.

The aggregate tables report a mean count and a mean loss separately, which cannot show whether the
two move together. This can.

N=4000 is still training and gets a panel when it lands.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = {"control": "black", "dead": "#C0392B", "mute": "#2471A3"}
DATA = "/private/tmp/claude-502/-Users-pt1290-Documents-GitHub-trainRNNbrain/1bc3843f-10a1-471f-875d-5f58458e098b/scratchpad/scatter_data.json"
OUT = "img/internal_figures/fig_dropout_r2_vs_active.png"

rows = json.load(open(DATA))["dropout"]
sizes = sorted({r["N"] for r in rows})
fig, axes = plt.subplots(1, len(sizes), figsize=(4.6 * len(sizes), 4.6), sharey=True)
axes = np.atleast_1d(axes)

for ax, N in zip(axes, sizes):
    for arm in ("control", "mute", "dead"):
        s = [r for r in rows if r["N"] == N and r["arm"] == arm]
        if not s:
            continue
        x = [r["active"] for r in s]
        y = [r["r2"] for r in s]
        ax.scatter(x, y, s=90, color=COL[arm], label=arm if N == sizes[0] else None, zorder=3,
                   edgecolor="white", linewidth=1.2)
        # Label each cluster directly so the legend is not the only way to read identity, and put
        # the label beside the cluster rather than above it: a label above collides with the panel
        # title for the low-count arms, and `dead` at N=500 sits hard against the all-units line.
        left = np.mean(x) > 0.55 * N * 1.08
        ax.annotate(arm, (np.mean(x), np.mean(y)), textcoords="offset points",
                    xytext=(-14 if left else 14, 0), ha="right" if left else "left",
                    va="center", color=COL[arm], fontsize=9, fontweight="bold")
    ax.axvline(N, color="0.75", lw=1, ls=":", zorder=1)
    ax.set_title(f"N = {N}", fontsize=12, pad=12)
    ax.set_xlabel("active units at 40k")
    ax.set_xlim(0, N * 1.10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92", lw=0.8, zorder=0)
    ax.annotate(f"all {N} units", (N, ax.get_ylim()[1]), textcoords="offset points",
                xytext=(-5, -4), ha="right", va="top", color="0.55", fontsize=8, rotation=90)

axes[0].set_ylabel("held-out $r^2$")
axes[0].legend(frameon=False, loc="lower left", fontsize=9)
fig.suptitle("Dropout buys active units with task performance, and `dead` pays more for more",
             fontsize=12.5)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=160)
print(f"wrote {OUT}")

for N in sizes:
    print(f"\nN={N}")
    for arm in ("control", "mute", "dead"):
        s = [r for r in rows if r["N"] == N and r["arm"] == arm]
        if not s:
            continue
        a = np.array([r["active"] for r in s], float)
        q = np.array([r["r2"] for r in s], float)
        ctrl = [r for r in rows if r["N"] == N and r["arm"] == "control"]
        ca = np.mean([r["active"] for r in ctrl])
        cq = np.mean([r["r2"] for r in ctrl])
        print(f"  {arm:8s} n={len(s)}  active {a.mean():7.1f} +-{a.std(ddof=1):5.1f} "
              f"({a.mean()/ca:4.2f}x control)   r2 {q.mean():.4f} +-{q.std(ddof=1):.4f} "
              f"({q.mean()-cq:+.4f} vs control)")
