#!/usr/bin/env python3
"""
Manuscript Figure 4 - WHAT WEIGHT SPARSITY ADDS, AND WHAT IT BREAKS.

The rate penalty of Figure 3 targets a soft maximum over time, so a unit can satisfy it by firing
in a brief transient and doing nothing else. Those units are alive by every threshold in this paper
and are not doing the work. Adding a recurrent-weight-sparsity penalty removes them - and this
figure establishes that causally, in both directions, on two tasks. It also pays for it, and the
bill is in panel (d).

  (a) WHAT A BURST UNIT IS             two REAL units from the same trained network: one that fires
                                       in a transient and one that stays up. Both are "active".
                                       The temporal participation ratio separates them.
  (b) WHAT rws ACTUALLY DOES           the effective in-degree of every unit, with and without the
                                       penalty. Without it the median unit listens to ~840 of its
                                       2,000 inputs; with it every unit sits on the target of 20.
                                       The penalty does exactly what it says on the tin.
  (c) THE CAUSAL TEST                  warm-start a converged network, switch the penalty on or off,
                                       train on. Adding rws removes burst units, removing rws brings
                                       them back, matched controls barely move - and it replicates
                                       on a second task.
  (d) THE PRICE                        the 36-tau memory task, per seed, along training. The
                                       unpenalised network never finds the memory solution; the rate
                                       penalty finds it every time and then loses it; adding weight
                                       sparsity prevents the escape altogether.

Usage:  python fig_paper_F4.py [--refresh]
Output: img/internal_figures/fig_paper_F4.png
"""

import argparse
import glob
import os
import pickle
import sys

import hydra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from common import DATA_DIR, SILENT_FLIPFLOP, SILENT_REL
from flipflop_diversity import load_net, rates_and_targets
from trainRNNbrain.training.training_utils import prepare_task_arguments

CACHE = "data/fig_paper_F4_cache.npz"
N_SAMPLES = 300 * 1024          # T x batch: the trace stores the raw (sum r)^2 / sum r^2
BURST = 0.05                    # a unit up for < 5% of the probe's samples is a burst unit
TG_DEG = 20                     # rws target effective in-degree, from configs/trainer/*.yaml

FRM_CELL = f"{DATA_DIR}/NBitFlipFlop_std_penlong/EqType=h_k=3_N=2000_pen=frm_iters=400000"
BOTH_CELL = f"{DATA_DIR}/NBitFlipFlop_std_penlong/EqType=h_k=3_N=2000_pen=both_iters=400000"

SWITCH = [
    ("A1", "add\nrws", "treat", ps.COND_COL["both"]),
    ("A2", "ctrl", "ctrl", ps.BASE),
    ("A3", "remove\nrws", "treat", ps.COND_COL["frm"]),
    ("A4", "ctrl", "ctrl", ps.BASE),
]
TASKS_SWITCH = [
    ("3-bit flip-flop", f"{DATA_DIR}/NBitFlipFlop_std_switch",
     "warm_EqType=h_k=3_N=2000_arm={arm}_from=*_rep=*_iters=50000"),
    ("CDDM (replication)", f"{DATA_DIR}/CDDM_std_switch",
     "warm_EqType=h_N=2000_arm={arm}_from=*_rep=*_iters=50000"),
]

DMTS36 = "data/dmts_curves_delay36.npz"
DMTS_ARMS = [("none", "no penalty", ps.BASE),
             ("frm", "rate penalty (frm)", ps.COND_COL["frm"]),
             ("both", "frm + rws", ps.COND_COL["both"])]
ESCAPE = 0.9


def in_degree(cell, n_nets=3):
    """Effective in-degree of every unit, per network, for one penalty arm.

    S_i = (sum_j |W_ij|)^2 / sum_j W_ij^2 is the participation ratio of row i of the recurrent
    weight matrix - the number of inputs unit i effectively listens to. It is exactly the quantity
    the `rws` penalty caps, so plotting it is the most direct possible check that the penalty does
    what it claims.

    Args:
        cell: cell folder; n_nets: how many networks to read.
    Returns:
        (n_nets, N) array of effective in-degrees, or an empty array.
    """
    out = []
    for folder in sorted(glob.glob(os.path.join(cell, "*", "")))[:n_nets]:
        net, _ = load_net(folder)
        W = net.W_rec
        W = W.detach().cpu().numpy() if hasattr(W, "detach") else np.asarray(W)
        l1 = np.abs(W).sum(axis=1)
        l2sq = (W ** 2).sum(axis=1)
        out.append(l1 ** 2 / np.maximum(l2sq, 1e-30))
    return np.array(out) if out else np.array([])


def example_units(refresh=False):
    """One burst unit and one sustained unit, both ACTIVE, from the same trained `frm` network.

    Args:
        refresh: re-simulate even if the cache exists.
    Returns:
        dict with 'burst', 'sustained' (each (T,) rate traces), their tPR fractions, and the
        effective in-degrees of both penalty arms.
    """
    if os.path.exists(CACHE) and not refresh:
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    folder = sorted(glob.glob(os.path.join(FRM_CELL, "*", "")))[0]
    rates, _ = rates_and_targets(folder, n_trials=24)
    R = rates.reshape(rates.shape[0], -1)
    p = R.std(axis=1) + np.quantile(R, 0.9, axis=1)
    live = p >= SILENT_REL * np.quantile(p, 0.95)
    tpr = (R.sum(axis=1) ** 2) / np.maximum((R ** 2).sum(axis=1), 1e-30) / R.shape[1]

    idx = np.where(live)[0]
    burst_i = idx[np.argmin(tpr[idx])]
    sust_i = idx[np.argmax(tpr[idx])]
    tb = int(np.argmax(rates[burst_i].max(axis=0)))
    ts = int(np.argmax(rates[sust_i].max(axis=0)))
    out = {"burst": rates[burst_i, :, tb].astype(np.float32),
           "sustained": rates[sust_i, :, ts].astype(np.float32),
           "tpr_burst": np.float32(tpr[burst_i]), "tpr_sust": np.float32(tpr[sust_i]),
           "S_frm": in_degree(FRM_CELL).astype(np.float32),
           "S_both": in_degree(BOTH_CELL).astype(np.float32)}
    np.savez_compressed(CACHE, **out)
    return out


def burst_fractions(root, pattern):
    """Burst fraction among active units, at the first and last probe, per seed and arm.

    Args:
        root: sweep folder; pattern: cell-name pattern with an "{arm}" placeholder.
    Returns:
        dict arm -> (before, after) arrays over seeds.
    """
    res = {}
    for arm, _, _, _ in SWITCH:
        b0, b1 = [], []
        for f in sorted(glob.glob(os.path.join(root, pattern.format(arm=arm), "*",
                                               "*ParticipationTrace.pkl"))):
            try:
                d = pickle.load(open(f, "rb"))
            except Exception:
                continue
            if "temporal_pr" not in d:
                continue
            P = np.asarray(d["participation"])
            t = np.asarray(d["temporal_pr"], float) / N_SAMPLES
            l0 = P[0] >= SILENT_REL * np.quantile(P[0], 0.95)
            l1 = P[-1] >= SILENT_REL * np.quantile(P[-1], 0.95)
            b0.append((t[0][l0] < BURST).mean())
            b1.append((t[-1][l1] < BURST).mean())
        res[arm] = (np.array(b0), np.array(b1))
    return res


def panel_a(ax, c):
    """Panel (a): a burst unit and a sustained unit, both active, from the same network."""
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    tt = np.linspace(0.10, 0.94, len(c["burst"]))
    for j, (key, lab, col, tpr) in enumerate([
            ("sustained", "sustained unit", ps.COND_COL["both"], float(c["tpr_sust"])),
            ("burst", "burst unit", ps.COND_COL["frm"], float(c["tpr_burst"]))]):
        base = 0.60 - 0.36 * j
        r = np.asarray(c[key], float)
        ax.plot(tt, base + 0.26 * r / max(r.max(), 1e-9), lw=0.9, color=col, zorder=4)
        ax.plot([0.10, 0.94], [base, base], lw=0.45, color=ps.FAINT, zorder=2)
        ax.text(0.10, base + 0.30, lab, fontsize=6.4, color=col)
        ax.text(0.94, base + 0.30, f"tPR = {tpr:.2f}", fontsize=6.0, color=col, ha="right")
    ax.text(0.52, 0.035, "time (one trial)", ha="center", fontsize=5.6, color=ps.FAINT)
    ax.text(0.52, 0.965, "both units are ACTIVE by every threshold in this paper",
            ha="center", fontsize=6.2, color=ps.INK)
    ax.text(0.52, 0.125, "the rate penalty is satisfied by either one",
            ha="center", fontsize=5.8, color=ps.MUTED)


def panel_b(ax, c):
    """Panel (b): the effective in-degree distribution, with and without the sparsity penalty."""
    S_frm, S_both = np.asarray(c["S_frm"], float), np.asarray(c["S_both"], float)
    if not S_frm.size or not S_both.size:
        ax.text(0.5, 0.5, "weights not on disk", ha="center", transform=ax.transAxes)
        return {}
    bins = np.logspace(0.5, 3.6, 60)
    ax.hist(S_frm.ravel(), bins=bins, color=ps.COND_COL["frm"], alpha=0.75, edgecolor="none",
            label=f"frm  (median {np.median(S_frm):.0f})")
    ax.hist(S_both.ravel(), bins=bins, color=ps.COND_COL["both"], alpha=0.8, edgecolor="none",
            label=f"frm + rws  (median {np.median(S_both):.0f})")
    ax.axvline(TG_DEG, color=ps.INK, lw=0.9, ls="--", zorder=5)
    ax.text(TG_DEG * 1.16, ax.get_ylim()[1] * 0.93, f"target = {TG_DEG}", fontsize=5.8,
            color=ps.INK, va="top")
    ax.set(xscale="log", xlabel="effective in-degree  $S_i=(\\sum_j|W_{ij}|)^2/\\sum_j W_{ij}^2$",
           ylabel="units")
    ax.legend(loc="upper right", fontsize=5.9)
    ax.set_title("every unit is driven to the cap", fontsize=6.2, color=ps.MUTED, pad=3)
    ps.ygrid(ax)
    return {"median_frm": float(np.median(S_frm)), "median_both": float(np.median(S_both)),
            "n_frm": int(S_frm.shape[0]), "n_both": int(S_both.shape[0])}


def panel_c(ax):
    """Panel (c): the causal penalty switch, on both tasks. Returns the contrasts."""
    out, xs, labels, centres = {}, [], [], []
    x = 0.0
    for ti, (task, root, pattern) in enumerate(TASKS_SWITCH):
        res = burst_fractions(root, pattern)
        if not any(len(v[0]) for v in res.values()):
            continue
        start = x
        for arm, lab, kind, col in SWITCH:
            b0, b1 = res[arm]
            if not len(b0):
                x += 1
                continue
            d = (b1 - b0) * 100
            ps.strip(ax, [x], [d], [col], width=0.30, jitter=0.04,
                     rng=np.random.default_rng(80 + ti * 4 + len(labels)), ms=2.8)
            xs.append(x)
            labels.append(lab)
            x += 1
        centres.append((start + x - 1) / 2)
        d1 = ((res["A1"][1] - res["A1"][0]).mean() - (res["A2"][1] - res["A2"][0]).mean()) * 100
        d3 = ((res["A3"][1] - res["A3"][0]).mean() - (res["A4"][1] - res["A4"][0]).mean()) * 100
        out[task] = (d1, d3)
        ax.text(centres[-1], 44.0, task, ha="center", fontsize=6.4, color=ps.INK)
        ax.text(centres[-1], 37.5,
                f"add rws {d1:+.1f}   remove rws {d3:+.1f}\n(treatment − control)",
                ha="center", fontsize=5.6, color=ps.MUTED, linespacing=1.3)
        x += 0.9
    ax.axhline(0, color=ps.INK, lw=0.8, zorder=3)
    ax.set(xticks=xs, xticklabels=labels, ylim=(-40, 50),
           ylabel="change in burst units\n(percentage points of active units)")
    ax.tick_params(axis="x", labelsize=5.8)
    ax.set_yticks([-40, -30, -20, -10, 0, 10, 20, 30])
    ps.ygrid(ax)
    return out


def panel_d(ax):
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
    ax.text(1.5e3, ESCAPE + 0.02, "memory solved", fontsize=5.6, color=ps.MUTED)
    ax.axhline(0.605, color=ps.MUTED, lw=0.7, ls=":", zorder=2)
    ax.text(1.5e3, 0.565, "no-memory plateau", fontsize=5.6, color=ps.MUTED)
    ax.annotate("frm finds the memory\nin every seed…", xy=(2.2e4, 0.97), xytext=(3.0e3, 0.80),
                fontsize=5.8, color=ps.COND_COL["frm"], linespacing=1.25,
                arrowprops=dict(arrowstyle="-|>", lw=0.55, color=ps.COND_COL["frm"],
                                mutation_scale=6))
    ax.annotate("…and then loses it", xy=(1.1e5, 0.25), xytext=(2.6e4, 0.10),
                fontsize=5.8, color=ps.COND_COL["frm"],
                arrowprops=dict(arrowstyle="-|>", lw=0.55, color=ps.COND_COL["frm"],
                                mutation_scale=6))
    ax.set(xscale="log", xlabel="training iteration", ylabel="clean $r^2$ on the 36τ delay task",
           xlim=(1.2e3, 1.7e5), ylim=(-0.12, 1.06))
    ax.legend(handles=[Line2D([], [], color=c, lw=1.2, label=l) for _, l, c in DMTS_ARMS],
              loc="lower left", fontsize=5.8)
    ps.ygrid(ax)
    return verdict


def main():
    """Assemble Figure 4 and write it. Returns the output path."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    ps.setup()
    c = example_units(refresh=args.refresh)

    fig = plt.figure(figsize=(ps.W2, 152 * ps.MM))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.80, 1.0], width_ratios=[1.0, 1.06],
                  hspace=0.50, wspace=0.26)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_a(ax_a, c)
    ps.panel_letter(ax_a, "a", dx=-0.02, dy=1.0)
    ax_a.text(-0.02, 1.13, "How the rate penalty is gamed", transform=ax_a.transAxes,
              fontsize=7.4, color=ps.INK, fontweight="bold")

    ax_b = fig.add_subplot(gs[0, 1])
    info_b = panel_b(ax_b, c)
    ps.panel_letter(ax_b, "b")
    ax_b.text(-0.13, 1.13, "What weight sparsity does", transform=ax_b.transAxes,
              fontsize=7.4, color=ps.INK, fontweight="bold")

    ax_c = fig.add_subplot(gs[1, 0])
    contrasts = panel_c(ax_c)
    ps.panel_letter(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    verdict = panel_d(ax_d)
    ps.panel_letter(ax_d, "d")

    out = ps.save(fig, "fig_paper_F4")

    print("\n--- numbers quoted in the caption ---")
    print(f"  burst tPR {float(c['tpr_burst']):.3f} vs sustained {float(c['tpr_sust']):.3f}")
    for k, v in info_b.items():
        print(f"  {k}: {v}")
    for task, (d1, d3) in contrasts.items():
        print(f"  {task:20} adding rws {d1:+.1f} pts, removing rws {d3:+.1f} pts (treatment − control)")
    for pen, (best, fin, n) in verdict.items():
        print(f"  DMTS36 {pen:5} n={n}  best r2 {np.round(best, 3)}  final {np.round(fin, 3)}")
    return out


if __name__ == "__main__":
    main()
