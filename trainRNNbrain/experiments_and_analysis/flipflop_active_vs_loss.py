#!/usr/bin/env python3
"""
Active-unit count against achieved loss, one trajectory per seed, at a single (N, k) cell.

Every other figure in this project reads each network at ONE iteration and then argues about which
iteration is fair. This one drops the read-out entirely: it plots the whole path in (loss, M) space,
so the question "how many units are active once the network is this good?" is answered at every loss
level simultaneously. Two networks are compared where they achieve the SAME loss, which is the
comparison the read-out criteria were only ever approximating.

Time runs LEFT TO RIGHT: the x axis is inverted, so training starts at the high-loss left edge.

X IS THE RUNNING MINIMUM OF THE LOSS - "achieved loss" in the sense of the best this network has
reached so far. Against the instantaneous loss the paths are not functions: the loss is strongly
non-monotonic (frm's gradient spikes reach 1e4), so a network revisits the same loss at many
different activity levels and every trajectory becomes a scribble. The running minimum is monotone
in t by construction, so each seed is a single readable curve. The raw (loss, M) pairs are kept as
faint dots underneath so the excursions the running minimum hides are still visible.

⚠️ TWO THRESHOLDS ARE PLOTTED AND BOTH ARE SHOWN. Row 1 uses SILENT_FLIPFLOP = 4e-2, calibrated for
this task by Otsu's method on log participation; row 2 uses the scale-free rule (5% of the 95th
percentile). They disagree, and a claim that survives only one of them is a claim about the
threshold, not about the network.

Points are sampled at `participation_iters` (the coarse cadence on which the full participation
vector is stored), and each is paired with the loss at the SAME iteration, not the nearest probe.
The loss is boxcar-smoothed first: the raw per-probe loss jitters by more than the trend, and since
it is the X coordinate here that jitter turns each path into a scribble that hides the trend.

⚠️ INDIVIDUAL NaN PROBES ARE DROPPED, NOT THE WHOLE RUN. pr_matrix.load() discards any run with a
single NaN in its loss trace; one of the three frm seeds here has 50 NaNs out of 40,000 probes from
a transient gradient spike and still finished at r2 = 0.941, so that rule would silently show two
seeds where three exist.

Output: img/internal_figures/active_vs_loss_N{N}_k{k}.png

Usage:  python flipflop_active_vs_loss.py [N] [k]        (defaults 2000 3)
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP, active_count
import plotstyle as ps
from pr_matrix import ROOTS, SKIP, PENS, PROBE, R2_MIN

SEED_COLS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
SMOOTH_W = 21          # probes; same window pr_matrix uses on the loss


def seed_of(folder):
    """Seed of a run, read from its saved config.

    File names only carry the seed for runs trained after that convention was added, so the earlier
    ksweep and std_pen folders have none; the config always does.

    Args:
        folder: run folder containing exactly one *_config.yaml.
    Returns:
        int seed, or -1 if no config is present.
    """
    cfgs = glob.glob(os.path.join(folder, "*_config.yaml"))
    return int(OmegaConf.load(cfgs[0]).seed) if cfgs else -1


def smooth_loss(L, w=SMOOTH_W):
    """Boxcar-smoothed loss with NaN probes interpolated over and edges normalised.

    Args:
        L: (n_probes,) loss trace, possibly containing NaNs.
        w: boxcar width in probes.
    Returns:
        (n_probes,) smoothed trace, same length as L.
    """
    L = np.asarray(L, dtype=float)
    bad = ~np.isfinite(L)
    if bad.all():
        return L
    if bad.any():                       # linear interpolation across the NaN spikes
        idx = np.arange(L.size)
        L = np.interp(idx, idx[~bad], L[~bad])
    kern = np.ones(w) / w
    return np.convolve(L, kern, "same") / np.convolve(np.ones(L.size), kern, "same")


def load_cell(N, k):
    """Every usable run at one (N, k) cell, grouped by penalty.

    Args:
        N: hidden size; k: number of bits.
    Returns:
        dict penalty -> list of run dicts with keys `loss` (per-probe noise-free loss),
        `part` (list of participation vectors), `piters` (iterations those were stored at),
        `seed` (int from the file name).
    """
    out = {}
    for tag, root in ROOTS.items():
        for f in sorted(glob.glob(os.path.join(root, "*", "*", "*ParticipationTrace.pkl"))):
            m = re.search(r"_k=(\d+)_N=(\d+)(?:_pen=([a-z]+))?", f)
            if not m or int(m.group(1)) != k or int(m.group(2)) != N:
                continue
            pen = m.group(3) or "none"
            if (tag, pen) in SKIP:
                continue
            base = os.path.basename(f)
            try:
                r2 = float(base.split("_")[0])
            except ValueError:
                continue
            if not (r2 >= R2_MIN):          # never solved the task; see pr_matrix.load
                continue
            with open(f, "rb") as fh:
                tr = pickle.load(fh)
            L = np.asarray(tr["metrics"].get("loss_clean_train", []), dtype=float)
            if L.size == 0 or not np.isfinite(L).any():
                continue
            out.setdefault(pen, []).append(
                dict(loss=L, part=tr["participation"],
                     piters=np.asarray(tr["participation_iters"], dtype=float),
                     seed=seed_of(os.path.dirname(f))))
    return out


def trajectory(run, criterion):
    """(loss, active count) pairs along one run's training, aligned at the stored-participation iters.

    Args:
        run: run dict from load_cell(); criterion: threshold passed to common.active_count.
    Returns:
        (xmin, y, xraw): best-so-far loss at each stored participation snapshot, the active count
        there, and the un-accumulated smoothed loss at the same snapshots.
    """
    L = smooth_loss(run["loss"])
    idx = np.round(run["piters"] / PROBE).astype(int) - 1     # piters are 1-indexed in iterations
    ok = (idx >= 0) & (idx < L.size)
    x = L[idx[ok]]
    y = np.array([active_count(run["part"][i], criterion)
                  for i in np.flatnonzero(ok)], dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[good], y[good]
    return np.minimum.accumulate(x), y, x


def main():
    """Plot active count vs achieved loss for every seed of one (N, k) cell, under two thresholds."""
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ps.setup()
    cell = load_cell(N, k)
    if not cell:
        print(f"no runs at N={N}, k={k}"); return None

    # The quantitative form of "read vertically": M/N at loss levels every condition actually
    # reaches, so no read-out criterion is involved at all.
    LEVELS = [0.30, 0.10, 0.05, 0.035, 0.030]
    print(f"N={N}, k={k}   M/N at matched achieved loss "
          f"(threshold {SILENT_FLIPFLOP:g}; mean +- sd over seeds, '-' = never reached)\n")
    print("loss   " + "".join(f"{p:>18}" for p in PENS))
    for lv in LEVELS:
        cellrow = f"{lv:<7.3f}"
        for pen in PENS:
            vals = []
            for run in cell.get(pen, []):
                x, y, _ = trajectory(run, SILENT_FLIPFLOP)
                hit = np.flatnonzero(x <= lv)
                if hit.size:
                    vals.append(y[hit[0]] / N)
            cellrow += (f"{np.mean(vals):>11.3f}±{np.std(vals):.3f}" if len(vals) == 3
                        else f"{'-' if not vals else f'{np.mean(vals):.3f}(n={len(vals)})':>18}")
        print(cellrow)
    print()

    rows = [(SILENT_FLIPFLOP, f"M  (p >= {SILENT_FLIPFLOP:g}, task-calibrated)"),
            ("scalefree", "M  (p >= 5% of the 95th pct, scale-free)")]
    fig, ax = plt.subplots(len(rows), len(PENS), figsize=(4.3 * len(PENS), 4.5 * len(rows)),
                           squeeze=False, sharex=True)
    for c_i, pen in enumerate(PENS):
        runs = sorted(cell.get(pen, []), key=lambda r: r["seed"])
        for r_i, (crit, ylab) in enumerate(rows):
            a = ax[r_i][c_i]
            if not runs:
                a.text(.5, .5, f"no {pen} runs", ha="center", va="center", transform=a.transAxes,
                       color="0.5")
                continue
            lo, hi = np.inf, -np.inf
            for s_i, run in enumerate(runs):
                x, y, xraw = trajectory(run, crit)
                col = SEED_COLS[s_i % len(SEED_COLS)]
                a.plot(xraw, y / N, ".", color=col, ms=1.6, alpha=.22)     # raw, incl. excursions
                a.plot(x, y / N, "-", color=col, lw=1.5, alpha=.95,
                       label=f"seed {run['seed']}")
                a.plot(x[0], y[0] / N, "o", color=col, ms=5, mfc="white", mew=1.3)   # start
                a.plot(x[-1], y[-1] / N, "s", color=col, ms=5)                        # end
                lo, hi = min(lo, x.min()), max(hi, x.max())
            a.set_xscale("log")
            a.set_xlim(hi * 1.6, lo / 1.3)      # framed on the running minimum, inverted
            a.set(ylim=(0, 1.05), ylabel=ylab if c_i == 0 else "",
                  xlabel="best loss achieved so far (log; training runs left to right)"
                         if r_i == len(rows) - 1 else "")
            if r_i == 0:
                a.set_title(f"{pen}", fontsize=12, fontweight="bold")
            a.legend(fontsize=7, loc="lower left")
            a.grid(alpha=.25)
    fig.suptitle(f"Active fraction vs achieved loss — N={N}, k={k}, per seed\n"
                 "line = running minimum of the loss, faint dots = raw (loss, M) pairs  ·  "
                 "open circle = start, filled square = end  ·  "
                 "read vertically to compare penalties at the SAME achieved loss",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return ps.save(fig, f"active_vs_loss_N{N}_k{k}", tight=False)


if __name__ == "__main__":
    main()
