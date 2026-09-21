#!/usr/bin/env python3
"""
Manuscript Figure 5 - WHY IT MATTERS. What an experimenter recording from these networks would
conclude, and how much of that conclusion is set by the optimiser rather than by the task.

This figure exists because "most units are silent" is only interesting if it changes an answer
somebody cares about. It does: the population statistics routinely used to compare model networks
with recordings - effective dimensionality, the spread of firing rates, the fraction of units tuned
to a task variable - are all computed over whichever units happened to survive training.

  (a) THE RULE NOBODY APPLIES          an experimenter never records a never-firing neuron, so every
                                       population statistic must be computed over ACTIVE UNITS ONLY.
                                       Drawn, because it is the methodological point of the figure
                                       and it is one sentence.
  (b) DIMENSIONALITY                   the participation ratio of the activity covariance, computed
                                       both ways. Over all units it is an artefact of how many units
                                       are dead; over active units only it still moves several-fold
                                       between networks that solve the same task.
  (c) RATE HETEROGENEITY               sigma of log mean rate over active units, against the
                                       cortical value of about one decade. This is where the paper's
                                       own remedy looks WORSE than the disease and we say so.
  (d) TUNED UNITS                      the fraction of active units whose rate is explained by a
                                       task variable, both ways.

EVERYTHING HERE IS RECOMPUTED FROM NETWORKS ON DISK, at the manuscript's own lambda_frm = 0.1.
It deliberately does NOT use data/trained_RNNs/population_distortion.csv, which is the source of the
corresponding numbers in docs/paper.md: that CSV is the only surviving trace of a sweep
(CDDM_std_g0) whose raw networks have been deleted, it was run at lambda_frm = 0.2, and it carries
no r^2 column, so its "networks matched for performance" premise cannot be checked. Those numbers
are not quotable; these are.

Usage:  python fig_paper_F5.py [--refresh]
Output: img/internal_figures/fig_paper_F5.pdf (+ .svg; vector only - see paperstyle.save)
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from common import DATA_DIR, SILENT_REL
from ablate_rescued_units import measure as ablation_measure
from characterize import cddm_rates

CACHE = "data/fig_paper_F5_cache.pkl"
R2_GATE = 0.15                  # a unit counts as tuned when the task design explains this much
CORTEX_SIGMA_LOG = 1.0          # log10 firing-rate spread in cortex, about one decade

ARMS = [("none", "none", ps.BASE),
        ("rws", "rws", ps.COND_COL["rws"]),
        ("frm", "frm", ps.COND_COL["frm"]),
        ("both", "frm\n+rws", ps.COND_COL["both"])]
CELLS = {
    "none": f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N=1000_iters=200000",
    "rws":  f"{DATA_DIR}/CDDM_std_g0_penalties/EqType=h_N=1000_pen=rws",
    "frm":  f"{DATA_DIR}/CDDM_std_g0_penalties/EqType=h_N=1000_pen=frm",
    "both": f"{DATA_DIR}/CDDM_std_g0_penalties/EqType=h_N=1000_pen=both",
}


def participation_ratio(X):
    """Effective dimensionality of a set of unit traces, as the covariance participation ratio.

    PR = (sum_i lambda_i)^2 / sum_i lambda_i^2 over the eigenvalues of the covariance. It is the
    standard "how many dimensions does this population use" estimator, and it is computed from the
    eigenvalues of the Gram matrix, which is the cheaper of the two equivalent routes here.

    Args:
        X: (n_units, n_samples) responses; rows are mean-centred internally.
    Returns:
        the participation ratio as a float, or nan for fewer than two units.
    """
    if X.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(axis=1, keepdims=True)
    ev = np.linalg.eigvalsh(Xc @ Xc.T)
    ev = np.clip(ev, 0, None)
    s = ev.sum()
    return float(s * s / max((ev ** 2).sum(), 1e-300)) if s > 0 else float("nan")


def tuning_r2(X, G):
    """Per-unit R^2 of a least-squares fit of each unit's response on the task design.

    Reported as a distribution rather than as a thresholded "fraction tuned". A gate needs a null
    distribution to mean anything, and at any permissive gate essentially every active unit in every
    arm passes, so the gated fraction carries no information. The median and spread of R^2 do.

    Args:
        X: (n_units, n_samples) responses; G: (n_samples, p) design with an intercept column.
    Returns:
        (n_units,) array of R^2 values, empty for an empty unit set.
    """
    if X.shape[0] == 0:
        return np.array([])
    beta, *_ = np.linalg.lstsq(G, X.T, rcond=None)
    resid = X.T - G @ beta
    ss = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return 1.0 - (resid ** 2).sum(axis=0) / np.maximum(ss, 1e-300)


def sigma_log_rate(rates_mean):
    """Spread of log10 mean firing rate, the quantity cortex puts at about one decade.

    Args:
        rates_mean: (n_units,) mean rates, strictly positive entries only are used.
    Returns:
        the standard deviation of log10 rate as a float, or nan if fewer than two units qualify.
    """
    r = np.asarray(rates_mean, float)
    r = r[r > 0]
    return float(np.std(np.log10(r))) if r.size > 1 else float("nan")


def measure_networks(refresh=False, n_nets=3):
    """Every statistic, per network, per penalty arm, computed both over all units and active only.

    Args:
        refresh: recompute even if the cache exists; n_nets: networks per arm.
    Returns:
        dict arm -> list of per-network dicts.
    """
    if os.path.exists(CACHE) and not refresh:
        return pickle.load(open(CACHE, "rb"))
    out = {}
    for pen, _, _ in ARMS:
        rows = []
        for folder in sorted(glob.glob(os.path.join(CELLS[pen], "*", "")))[:n_nets]:
            rates, G, dec_on = cddm_rates(folder)
            N = rates.shape[0]
            X = rates.reshape(N, -1).astype(np.float64)
            p = X.std(axis=1) + np.quantile(X, 0.9, axis=1)
            live = p >= SILENT_REL * np.quantile(p, 0.95)
            # per-condition decision-epoch means: the responses an experimenter would regress
            Y = rates[:, dec_on:, :].mean(axis=1).astype(np.float64)
            rows.append({
                "n_active": int(live.sum()),
                "pr_all": participation_ratio(X),
                "pr_act": participation_ratio(X[live]),
                "sigma_all": sigma_log_rate(X.mean(axis=1)),
                "sigma_act": sigma_log_rate(X[live].mean(axis=1)),
                "r2_act": tuning_r2(Y[live], G),
            })
        out[pen] = rows
        print(f"  {pen:5} n={len(rows)}")
    pickle.dump(out, open(CACHE, "wb"))
    return out


def panel_a(ax):
    """Panel (a): the active-units-only rule, drawn."""
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.figure.canvas.draw()

    dx = 0.0125
    dy = ps.square_pitch(ax, dx)
    for col_i, (title, note) in enumerate([
            ("the network", "1000 units, 290 active"),
            ("a recording", "the 290")]):
        x0 = 0.145 + col_i * 0.455
        ps.unit_grid(ax, x0, 0.66, 29, 100, col=ps.SLOTS[0],
                     off_col=("#d5d4cc" if col_i == 0 else ps.PAPER),
                     pitch=(dx, dy), s=4.2, lw=0.38)
        ax.text(x0 + 4.5 * dx, 0.78, title, ha="center", fontsize=6.2, color=ps.INK)
        ax.text(x0 + 4.5 * dx, 0.16, note, ha="center", va="top", fontsize=5.3,
                color=ps.MUTED, linespacing=1.3)
    ps.arrow(ax, (0.455, 0.46), (0.565, 0.46), col=ps.MUTED)
    ax.text(0.51, 0.505, "record", ha="center", fontsize=5.6, color=ps.MUTED)




def bars(ax, data, key_all, key_act, ylabel, title):
    """Draw one statistic for every arm, computed both ways, one dot per network.

    Args:
        ax: axes; data: measure_networks output; key_all, key_act: the two dict keys;
        ylabel: y-axis label; title: panel title.
    Returns:
        list of (arm, mean_all, mean_act, n).
    """
    rows, width = [], 0.19
    for i, (pen, lab, col) in enumerate(ARMS):
        v = data.get(pen, [])
        if not v:
            continue
        a = np.array([r[key_all] for r in v], float)
        b = np.array([r[key_act] for r in v], float)
        ps.strip(ax, [i - width], [a], ["#c9c7bf"], width=width * 0.72, jitter=0.024,
                 rng=np.random.default_rng(200 + i), ms=2.6)
        ps.strip(ax, [i + width], [b], [col], width=width * 0.72, jitter=0.024,
                 rng=np.random.default_rng(300 + i), ms=2.6)
        rows.append((pen, float(np.nanmean(a)), float(np.nanmean(b)), len(v)))
    ax.set(xticks=range(len(ARMS)), xticklabels=[l for _, l, _ in ARMS], ylabel=ylabel)
    ax.tick_params(axis="x", labelsize=5.8)
    ax.set_title(title, fontsize=6.4, color=ps.MUTED, pad=3)
    ps.ygrid(ax)
    return rows


def panel_ablation(ax):
    """Panel (e): task r^2 as the least-active units are deleted, without retraining.

    The decisive control for this paper: counting active units cannot show that the units DO
    anything. Deleting them can. Removing a unit means zeroing its column of W_rec and of W_out, so
    its influence on the rest of the network and on the read-out is gone while the survivors' own
    dynamics are untouched. Nothing is retrained, so this measures what the trained network was
    using.

    Returns:
        dict arm -> (fractions, mean r2, sd) for the least-active-first ordering.
    """
    try:
        data = ablation_measure(refresh=False)
    except Exception:
        ax.text(0.5, 0.5, "run ablate_rescued_units.py", ha="center", transform=ax.transAxes)
        return {}
    out = {}
    for pen, lab, col in [("none", "no penalty", ps.BASE), ("frm", "rate penalty", ps.COND_COL["frm"])]:
        rows = data.get(pen, [])
        if not rows:
            continue
        f = rows[0]["fracs"]
        low = np.array([r["r2_low"] for r in rows])
        m, sd = low.mean(0), low.std(0)
        ax.plot(f, m, "-o", color=col, ms=2.6, lw=1.3, label=lab, zorder=4)
        ax.fill_between(f, m - sd, m + sd, color=col, alpha=0.16, lw=0)
        out[pen] = (f, m, sd)
        frac_silent = 1 - np.mean([r["n_active"] / r["N"] for r in rows])
        if pen == "none":
            ax.axvline(frac_silent, color=ps.BASE, lw=0.7, ls=":", zorder=2)

    ax.set(xlabel="fraction of units deleted", ylabel="task $r^2$",
           xlim=(-0.02, 0.92), ylim=(-0.05, 1.0))
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.set_xticklabels(["0", "25%", "50%", "75%"])
    ax.legend(loc="lower left", fontsize=5.8)

    ps.ygrid(ax)
    return out


def main():
    """Assemble Figure 5 and write it. Returns the output path."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    ps.setup()
    data = measure_networks(refresh=args.refresh)

    fig = plt.figure(figsize=(ps.W2, 132 * ps.MM))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[0.80, 1.0], hspace=0.58, wspace=0.95)

    ax_a = fig.add_subplot(gs[0, 0:3])
    panel_a(ax_a)
    ps.panel_letter(ax_a, "a", dx=-0.02, dy=1.0)

    ax_b = fig.add_subplot(gs[1, 0:2])
    rows_b = []
    for i, (pen, lab, col) in enumerate(ARMS):
        v = data.get(pen, [])
        if not v:
            continue
        a = np.array([r["pr_act"] for r in v], float)
        ps.strip(ax_b, [i], [a], [col], width=0.30, jitter=0.035,
                 rng=np.random.default_rng(200 + i), ms=3.0)
        rows_b.append((pen, float(np.nanmean(a)), float(np.nanmean(a)), len(v)))
    ax_b.set(xticks=range(len(ARMS)), xticklabels=[l for _, l, _ in ARMS],
             ylabel="participation ratio", ylim=(0, 8))
    ax_b.tick_params(axis="x", labelsize=5.8)
    ax_b.set_title("effective dimensionality", fontsize=6.4, color=ps.MUTED, pad=3)

    ps.ygrid(ax_b)
    ps.panel_letter(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 2:4])
    rows_c = bars(ax_c, data, "sigma_all", "sigma_act", "$\\sigma$ of $\\log_{10}$ mean rate",
                  "rate heterogeneity")
    ax_c.axhline(CORTEX_SIGMA_LOG, color=ps.BAD, lw=0.9, ls="--", zorder=5)
    ax_c.text(len(ARMS) - 0.55, CORTEX_SIGMA_LOG * 1.02, "cortex", ha="right",
              va="bottom", fontsize=5.8, color=ps.BAD)
    ps.panel_letter(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 4:6])
    rows_d = []
    for i, (pen, lab, col) in enumerate(ARMS):
        v = data.get(pen, [])
        if not v:
            continue
        pooled = np.concatenate([np.asarray(r["r2_act"], float) for r in v])
        pooled = pooled[np.isfinite(pooled)]
        q1, med, q3 = np.quantile(pooled, [0.25, 0.5, 0.75])
        ax_d.plot([i, i], [q1, q3], lw=1.1, color=col, zorder=4, solid_capstyle="round")
        ax_d.plot([i - 0.22, i + 0.22], [med] * 2, lw=1.8, color=col, zorder=5)
        rows_d.append((pen, float(med), float(q1), float(q3), len(v), int(pooled.size)))
    ax_d.set(xticks=range(len(ARMS)), xticklabels=[l for _, l, _ in ARMS],
             ylabel="per-unit $R^2$ on task variables",
             ylim=(0, 1.02))
    ax_d.tick_params(axis="x", labelsize=5.8)
    ax_d.set_title("task tuning", fontsize=6.4, color=ps.MUTED, pad=3)
    ps.ygrid(ax_d)
    ps.panel_letter(ax_d, "d")

    ax_e = fig.add_subplot(gs[0, 3:6])
    abl = panel_ablation(ax_e)
    ps.panel_letter(ax_e, "e")

    out = ps.save(fig, "fig_paper_F5")
    for pen, (f, m, sd) in abl.items():
        print(f"  ablation {pen:5}: " + "  ".join(f"{int(x*100)}%:{y:.3f}" for x, y in zip(f, m)))

    print("\n--- numbers quoted in the caption (CDDM, N=1000, lambda_frm=0.1) ---")
    for pen, _, _ in ARMS:
        v = data.get(pen, [])
        if v:
            print(f"  {pen:5} n={len(v)}  active {np.mean([r['n_active'] for r in v]):6.1f}")
    print("  -- participation ratio (active units)")
    for pen, a, _, n in rows_b:
        print(f"     {pen:5} {a:8.3f}   (n={n})")
    print("  -- sigma of log10 mean rate")
    for pen, a, b, n in rows_c:
        print(f"     {pen:5} all units {a:8.3f}   active only {b:8.3f}   (n={n})")
    print("  -- per-unit task R^2 over active units (median [IQR])")
    for pen, med, q1, q3, n, m in rows_d:
        print(f"     {pen:5} {med:.3f} [{q1:.3f}, {q3:.3f}]   (n={n} nets, {m} units)")
    return out


if __name__ == "__main__":
    main()
