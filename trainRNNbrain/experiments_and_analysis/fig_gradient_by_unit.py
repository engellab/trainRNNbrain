#!/usr/bin/env python3
"""Two candidate panels for the tanh slide, both drawn like the participation histogram: one network
per activation, silent units grey, the rest red, a dashed criterion.

  slide_k_unit_gradient  the gradient of the task loss on each unit's hidden state,
                         g_i = RMS_{t,trials} dL/dh_i(t), from gradient_participation.py
  slide_k_unit_slope     the mean slope of the activation at the unit's hidden state,
                         s_i = mean_{t,trials} f'(h_i(t)) -- the share of any gradient arriving at
                         the unit's rate that gets through to h_i. ReLU: the fraction of time the
                         unit is on. tanh: mean(1 - r_i^2).

THE CRITERION IS 5% OF THE UNTRAINED NETWORK'S 95th PERCENTILE, not the trained network's own. On
ReLU the two give the same count (274-319 against 269-292 across nine seeds). On tanh they do not,
and that is the point: the trained tanh population moves down together, so a bar set by its own 95th
percentile moves down with it and counts 660-1000 units in every seed. A bar fixed at the untrained
network asks whether a unit still gets the gradient a typical unit got before training.

THE GRADIENT IS DIVIDED BY sqrt(loss) BEFORE THAT COMPARISON. The trained network's loss is 20-70x
lower than the untrained one's, and the gradient scales with the residual error, so a raw comparison
would mostly measure the fall in error. The division makes it "gradient per unit of error". It is
approximate: it assumes the gradient is linear in the residual.

ONE SEED PER ACTIVATION, the one whose count is the median of nine; the nine-seed range goes in the
panel title, because the tanh seeds spread from 19 to 964.

Usage:  python fig_gradient_by_unit.py JSON_DIR   (from gradient_participation.py --out)
"""

import glob
import json
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paperstyle as ps
from common import SILENT_REL

ACTIVE_COL = ps.SLOTS[1]       # the participation histogram's red
FLOOR = 1e-5                   # gradients below this fraction of the reference share the first bin


# per-unit measures from gradient_sensitivity.py; True = a loss gradient, divided by sqrt(loss)
KEYS = {"g_state": True, "g_in": True, "s_in": False, "s_state": False}


def load(json_dir):
    """Per run: each measure, trained and untrained, divided by the untrained network's q95.

    Returns:
        {"ReLU": [run, ...], "tanh": [...]}; each run a dict with name and, per measure key,
        `key` (trained) and `key + "0"` (untrained). Loss gradients are first divided by sqrt(loss).
    """
    out = {"ReLU": [], "tanh": []}
    for f in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        r = json.load(open(f))
        run = dict(name=r["run"][:9])
        for key, per_err in KEYS.items():
            if key not in r["trained"]:
                continue
            v0, v = np.asarray(r["init"][key]), np.asarray(r["trained"][key])
            if per_err:
                v0, v = v0 / np.sqrt(r["init"]["loss"]), v / np.sqrt(r["trained"]["loss"])
            ref = np.quantile(v0, 0.95)
            run[key + "0"], run[key] = v0 / ref, v / ref
        out["tanh" if "tanh" in r["cell"] else "ReLU"].append(run)
    return out


def bar(run, key):
    """The criterion: 5% of the untrained network's 95th percentile of the same quantity."""
    return SILENT_REL * np.quantile(run[key + "0"], 0.95)


def median_run(runs, key):
    """The run whose count above the bar is the median of the set, and the set's count range."""
    n = np.array([(r[key] >= bar(r, key)).sum() for r in runs])
    return runs[int(np.argsort(n)[len(n) // 2])], n.min(), n.max()


def panel(ax, run, key, lo_n, hi_n, log, title):
    """One histogram: trained units split at the bar, the untrained network as an outline."""
    thr = bar(run, key)
    v, v0 = run[key], run[key + "0"]
    if log:
        v, v0 = np.maximum(v, FLOOR), np.maximum(v0, FLOOR)
        bins = np.logspace(np.log10(FLOOR * 0.7), np.log10(max(v.max(), v0.max()) * 1.4), 46)
    else:
        bins = np.linspace(0, 1, 41)
    live = v >= thr
    ax.hist(v[~live], bins=bins, color=ps.FAINT, edgecolor="none",
            label=f"below ({(~live).sum()})")
    ax.hist(v[live], bins=bins, color=ACTIVE_COL, edgecolor="none",
            label=f"above ({live.sum()})")
    ax.hist(v0, bins=bins, histtype="step", color=ps.INK, lw=0.8, label="same network, untrained")
    ax.axvline(thr, color=ps.INK, lw=0.9, ls="--", zorder=5)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.28)
    ax.text(thr * (1.35 if log else 1.0) + (0 if log else 0.015), top * 1.14, "criterion",
            fontsize=5.8, color=ps.INK, ha="left", va="center")
    if log:
        ax.set_xscale("log")
        ax.set_xlim(FLOOR * 0.7, 30)
    else:
        ax.set_xlim(0, 1)
    ax.set_title(f"{title}\nabove the criterion in 9 seeds: {lo_n}–{hi_n}", fontsize=6.8,
                 color=ps.INK, linespacing=1.3, pad=4)
    # the slope panel's criterion sits at the left edge, so its legend goes right
    ax.legend(loc="upper left" if log else "upper right", fontsize=5.6, handlelength=1.0)
    ps.ygrid(ax)


def figure(data, key, stem, log, xlabel, sup):
    ps.setup()
    # ONE COLUMN, like the participation histogram: the two activations share the x axis, so the
    # reader compares ReLU's split with tanh's shift straight down the page
    fig, axes = plt.subplots(2, 1, figsize=(96 * ps.MM, 100 * ps.MM), sharex=True)
    for ax, act in zip(axes, ("ReLU", "tanh")):
        run, lo_n, hi_n = median_run(data[act], key)
        panel(ax, run, key, lo_n, hi_n, log, act)
        ax.set_ylabel("units")
        print(f"  {stem} {act}: drawn seed {run['name']}, "
              f"{(run[key] >= bar(run, key)).sum()} above; 9-seed range {lo_n}-{hi_n}")
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(sup, fontsize=6.8, color=ps.INK, y=0.995)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.13, hspace=0.42)
    return ps.save(fig, stem, w_mm=96)


def seed_strip(data, stem="slide_k_unit_measures_seeds"):
    """Units above the bar in every seed, for every measure, ReLU against tanh: the comparison
    that decides which measure the talk keeps."""
    keys = [k for k in LABELS if k in data["ReLU"][0]]
    ps.setup()
    fig, ax = plt.subplots(figsize=(120 * ps.MM, 66 * ps.MM))
    xs = np.arange(len(keys), dtype=float)
    for dx, act, col in ((-0.17, "ReLU", ps.BASE), (0.17, "tanh", ACTIVE_COL)):
        groups = [np.array([(r[k] >= bar(r, k)).sum() for r in data[act]], float) for k in keys]
        ps.strip(ax, xs + dx, groups, [col] * len(keys), width=0.12,
                 rng=np.random.default_rng(0))
        ax.plot([], [], "o", color=col, label=act)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[k][1] for k in keys], fontsize=6.0)
    ax.set_ylim(0, 1080)
    ax.axhline(1000, color=ps.FAINT, lw=0.7, ls=":")
    ax.set_ylabel("units above the bar, of 1000")
    ax.legend(loc="upper right", fontsize=6.0, frameon=False)
    ax.set_title("one dot per seed; bar = 5% of the untrained network's 95th percentile",
                 fontsize=6.6, color=ps.INK)
    ps.ygrid(ax)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.27)
    return ps.save(fig, stem, w_mm=120)


# key -> (stem, short label for the strip, x label for the histogram)
LABELS = {
    "g_state": ("slide_k_unit_gradient", "A\nloss gradient\non the state",
                "loss gradient on the unit's state, per unit of error"),
    "g_in": ("slide_k_unit_gradient_in", "B\nloss gradient on\nincoming weights,\nper trial",
             "per-trial loss gradient on its incoming weights,\nper unit of error"),
    "s_in": ("slide_k_unit_sensitivity_in", "C\noutput sensitivity\nto incoming\nweights",
             "output sensitivity to the unit's incoming weights"),
    "s_state": ("slide_k_unit_sensitivity_state", "C'\noutput sensitivity\nto the state",
                "output sensitivity to the unit's state"),
}


def main(json_dir):
    data = load(json_dir)
    sup = "CDDM, $N$ = 1000, 150,000 iterations\ntrained, against the same network untrained"
    for key, (stem, _, xl) in LABELS.items():
        if key in data["ReLU"][0]:
            print(figure(data, key, stem, True,
                         xl + "\n(1 = the untrained network's 95th percentile)", sup))
    print(seed_strip(data))


if __name__ == "__main__":
    main(sys.argv[1])
