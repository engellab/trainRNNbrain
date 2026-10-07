#!/usr/bin/env python3
"""tanh does not avoid the problem; it moves it to the other end of the curve, where the talk's
own silence criterion cannot see it.

THE CRITERION IS BUILT FOR ReLU. A unit is counted active when p_i = std_t(r_i) + q_0.9(|r_i|)
reaches 5% of the population's 95th percentile. A dormant ReLU unit sits at exactly 0, so it scores
0 and is counted out. tanh has no zero region: its quietest unit scores 65% of its own 95th
percentile against a ReLU network's median unit at 1/24,000 of its. Every tanh unit therefore
clears the bar, the count reads 1000 of 1000 at every iteration of training, and it stays 1000
while the network saturates. The instrument is blind, not the network healthy.

WHAT TO COUNT INSTEAD, FOR A BOUNDED ACTIVATION. The thing a dead ReLU unit and a saturated tanh
unit share is that neither passes gradient. So a unit is RESPONSIVE when the mean slope of its
activation over the trial set exceeds half its maximum. For tanh, slope = 1 - r^2, so the test is
mean(1 - r^2) > 0.5 -- the unit spends its time off the asymptotes.

THAT TEST NEEDS RATES AND THE TRACES STORE ONLY p, so the curve over training uses a calibrated
threshold on p. p < 1.6 reproduces the gradient test on 99.6% of units in the WORST of two very
different states -- the untrained twin, where 1000 of 1000 are responsive, and the trained network,
where 13 are. Two states that far apart is why the threshold is a calibration rather than a fit;
the agreement table is printed by `--calibrate`.

⚠️ A COUNT IS THE WRONG STATISTIC FOR tanh, AND FINDING THAT OUT IS HALF THE RESULT. A ReLU
network's units are bimodal: some sit at exactly 0 and some work, so counting the working ones is
a stable measurement and its three seeds land on 288, 288 and 290. A tanh network's units are
UNIMODAL and tightly packed -- at iteration 60,000 the population spans p = 1.52 at the 10th
percentile to 1.71 at the 90th -- so a threshold anywhere inside that band is a knife edge, and the
count of units below it swings between 0 and 1000 from one probe to the next while nothing about
the network changes. Counting units off the asymptotes gives 8, 489 and 911 across three seeds for
that reason, not because the seeds differ that much.

So the panel reports the population's saturation itself. The left half draws the distribution of p
over training, where p = 2 means a unit fully switched between -1 and +1; the whole population
slides up together, which is what "they all saturate" looks like. The right half draws the counts,
where the point is that the standard rule reports tanh as 1000 of 1000 at every probe from
initialisation to the end while its matched ReLU control falls from 775 to 289.

Usage:  python trainRNNbrain/experiments_and_analysis/fig_tanh_saturation.py [--calibrate]
        (run from the repository root)
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paperstyle as ps
from common import DATA_DIR

ROOT = f"{DATA_DIR}/CDDM_std_g0_tanh"
RELU = "EqType=h_N=1000_act=relu_iters=150000"
TANH = "EqType=h_N=1000_act=tanh_iters=150000"
N_TOTAL = 1000
# calibrated against mean(1 - r^2) > 0.5; see the module docstring and --calibrate
P_RESPONSIVE = 1.6
SILENT_REL = 0.05          # the project's scale-free bar, as a fraction of q95(p)


def curves(cell):
    """Per-run active and responsive counts over training, from the participation traces.

    Args:
        cell: a cell directory holding one folder per seed.
    Returns:
        (iterations, active (n_runs, n_probes), responsive (n_runs, n_probes)); the iteration grid
        is taken from the first run and every run in these cells shares it.
    """
    it, act, resp = None, [], []
    for d in sorted(glob.glob(os.path.join(cell, "*"))):
        f = glob.glob(os.path.join(d, "*ParticipationTrace.pkl"))
        if not os.path.isdir(d) or os.path.basename(d).startswith("nan_") or not f:
            continue
        tr = pickle.load(open(f[0], "rb"))
        P = np.asarray(tr["participation"], float)
        if it is None:
            it = np.asarray(tr["participation_iters"], float)
        act.append([(p >= SILENT_REL * np.quantile(p, 0.95)).sum() for p in P])
        resp.append([(p < P_RESPONSIVE).sum() for p in P])
    return it, np.asarray(act, float), np.asarray(resp, float)


def slide(name="slide_k_tanh_saturation"):
    """Two panels: how saturated the tanh population is, and what the standard count reports.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if either cell is unreadable.
    """
    it_t, act_t, _ = curves(os.path.join(ROOT, TANH))
    it_r, act_r, _ = curves(os.path.join(ROOT, RELU))
    bands = p_bands(os.path.join(ROOT, TANH))
    if it_t is None or it_r is None or not len(bands):
        print(f"  SKIP {name}: tanh or ReLU cell unreadable")
        return None

    # STACKED, NOT SIDE BY SIDE: both panels run over the same training axis, so putting them in a
    # column lines up every iteration between them and the reader compares down rather than across.
    ps.setup()
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.4), sharex=True)

    ax = axes[0]
    col = ps.SLOTS[1]
    for it, lo, med, hi in bands:
        x = np.maximum(it, 1)
        ax.fill_between(x, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
        ax.plot(x, med, "-", color=col, lw=1.4, zorder=3)
    ax.axhline(2.0, color=ps.FAINT, lw=0.7, ls=":", zorder=1)
    ax.annotate("fully switched between $-$1 and +1", (110, 2.0), textcoords="offset points",
                xytext=(0, 3), fontsize=6.0, color=ps.MUTED)
    ax.axhline(P_RESPONSIVE, color=ps.INK, lw=0.7, ls="--", zorder=1)
    ax.annotate("passes gradient at least half the time", (110, P_RESPONSIVE),
                textcoords="offset points", xytext=(0, -9), fontsize=6.0, color=ps.INK)
    ax.set_xscale("log")
    ax.set_xlim(100, 2.2e5)
    ax.set_ylim(0, 2.25)
    ax.set_ylabel("saturation of a tanh unit")
    ax.set_title("every tanh unit saturates, together", fontsize=7.2, color=ps.INK, pad=4)

    ax = axes[1]
    for it, y, c, ls, lab in ((it_t, act_t, ps.SLOTS[1], "--", "tanh"),
                              (it_r, act_r, ps.BASE, "-", "ReLU control")):
        x = np.maximum(it, 1)
        for j, row in enumerate(y):
            ax.plot(x, row, ls, color=c, lw=1.3, alpha=0.85, zorder=3,
                    label=lab if j == 0 else None)
        # tanh's line IS the ceiling, so its label goes below the line and ReLU's above it,
        # instead of both landing on the "all 1,000" annotation
        dy = -9 if lab == "tanh" else -2
        ax.annotate(f"{y[:, -1].mean():.0f}", (it[-1], y[:, -1].mean()),
                    textcoords="offset points", xytext=(5, dy), fontsize=6.6, color=c)
    ax.axhline(N_TOTAL, color=ps.FAINT, lw=0.7, ls=":", zorder=1)
    ax.annotate(f"all {N_TOTAL:,}", (110, N_TOTAL), textcoords="offset points", xytext=(0, 3),
                fontsize=6.0, color=ps.MUTED)
    ax.set_xscale("log")
    ax.set_xlim(100, 2.2e5)
    ax.set_ylim(0, N_TOTAL * 1.1)
    ax.set_xlabel("training iteration")
    ax.set_ylabel(f"units of {N_TOTAL:,} counted active")
    ax.set_title("and the standard rule counts them all", fontsize=7.2, color=ps.INK, pad=4)
    ax.legend(frameon=False, fontsize=6.6, loc="center left")

    fig.suptitle("CDDM, $N$ = 1000, 150,000 iterations\ntanh against its own matched ReLU control",
                 fontsize=7.6, color=ps.INK, y=0.995)
    fig.subplots_adjust(left=0.16, right=0.975, top=0.90, bottom=0.085, hspace=0.22)
    return ps.save(fig, name, w_mm=116)


def _smooth(v, w=11):
    """Running median of a 1-D series over w samples, edge-padded. w is forced odd."""
    v = np.asarray(v, float)
    w = max(3, w | 1)
    if len(v) < w:
        return v
    pad = w // 2
    vp = np.pad(v, pad, mode="edge")
    return np.array([np.median(vp[i:i + w]) for i in range(len(v))])


def p_bands(cell):
    """Per-run 10th, 50th and 90th percentile of p over training.

    A count of units past a threshold is unusable on tanh (see the module docstring); the spread of
    p is what a reader needs instead, and it is what shows the population moving as one.

    Args:
        cell: a cell directory holding one folder per seed.
    Returns:
        [(iterations, q10, median, q90), ...], one tuple per run.
    """
    out = []
    for d in sorted(glob.glob(os.path.join(cell, "*"))):
        f = glob.glob(os.path.join(d, "*ParticipationTrace.pkl"))
        if not os.path.isdir(d) or os.path.basename(d).startswith("nan_") or not f:
            continue
        tr = pickle.load(open(f[0], "rb"))
        P = np.asarray(tr["participation"], float)
        it = np.asarray(tr["participation_iters"], float)
        # SMOOTHED over 11 probes (1,100 iterations). Each probe is one noise-free forward pass
        # and its percentiles swing a few hundredths from probe to probe; the swing is the
        # measurement, and leaving it in makes a trend that spans 0.8 unreadable.
        out.append((it,) + tuple(_smooth(np.quantile(P, q, axis=1)) if q != 0.5
                                 else _smooth(np.median(P, axis=1))
                                 for q in (0.1, 0.5, 0.9)))
    return out


def calibrate():
    """Print the per-run endpoint counts both ways, so the figure's numbers can be checked."""
    for lab, cell in (("tanh", TANH), ("ReLU", RELU)):
        it, act, resp = curves(os.path.join(ROOT, cell))
        if it is None:
            continue
        print(f"{lab}: {len(act)} runs, {len(it)} probes to iteration {it[-1]:,.0f}")
        print(f"   active by the standard rule : start {act[:, 0].mean():.0f}  "
              f"end {act[:, -1].mean():.0f}")
        print(f"   off the asymptotes (p < {P_RESPONSIVE}): start {resp[:, 0].mean():.0f}  "
              f"end {resp[:, -1].mean():.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrate", action="store_true", help="print the endpoints, write nothing")
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    else:
        got = slide()
        print(f"  wrote {got}" if got else "  nothing written")
