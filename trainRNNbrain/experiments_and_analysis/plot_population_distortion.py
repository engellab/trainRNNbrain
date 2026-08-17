#!/usr/bin/env python3
"""
Population-level statistics of networks that solve the task equally well (R^2 = 0.84-0.87).

Every statistic here is one that gets computed on model RNNs and compared against neural
recordings. If they differ between networks with indistinguishable task performance, the scientific
conclusion drawn from them depends on a training choice that methods sections do not report.

Firing-rate heterogeneity is included because it cuts the other way: cortical rate distributions are
strongly heterogeneous, so a network whose active units all fire at similar rates is LESS data-like
in that respect. It is here to be reported honestly, not because it flatters the penalty.

IMPORTANT: selectivity fractions are reported over ACTIVE UNITS ONLY. Computed over all units they
are trivially depressed wherever many units are silent, since a silent unit is non-selective by
construction — that dilution reverses the direction of the context-selectivity result. The last
panel shows the effect explicitly. Scale-free statistics (participation ratio, HHI) need no such
correction: appending zero-variance units adds zero eigenvalues and zero energy, changing neither
the numerator nor the denominator, and pr_active equals pr to the digit in every cell.

Usage:  python plot_population_distortion.py [CSV] [OUT]
"""

import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from collections import defaultdict

PENS = ["none", "rws", "frm", "both"]
EQ_COL = {"h": "#0072B2", "s": "#E69F00"}
T95 = 2.776


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/trained_RNNs/population_distortion.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "img/internal_figures/population_distortion.png"
    d = defaultdict(list)
    for r in csv.DictReader(open(src)):
        d[(r["eq"], r["penalty"])].append(r)

    def stat(eq, pen, field, scale=1.0):
        v = np.array([float(r[field]) for r in d[(eq, pen)]]) * scale
        return v.mean(), T95 * v.std(ddof=1) / np.sqrt(len(v))

    panels = [
        ("pr", "effective dimensionality\n"
               r"$PR=(\sum_i \lambda_i)^2/\sum_i \lambda_i^2$", 1.0, False),
        ("sel_ctx_act", "context-selective (%) of active\n"
                        r"$\eta^2_{\mathrm{ctx}}=\sigma^2_{\mathrm{between}}/\sigma^2_{\mathrm{total}}>0.1$", 100.0, False),
        ("sel_choice_act", "choice-selective (%) of active\n"
                           r"$\eta^2_{\mathrm{choice}}>0.1$", 100.0, False),
        ("energy", "total metabolic cost\n" r"$E=\sum_i \langle r_i^2\rangle_{t,c}$", 1.0, False),
        ("energy_hhi", "concentration of cost (log)\n"
                       r"$HHI=\sum_i p_i^2,\ p_i=\langle r_i^2\rangle/E$", 1.0, True),
        ("sigma_log", "rate heterogeneity across units\n"
                      r"$\sigma_{\log}=\mathrm{std}_i[\log_{10}\bar r_i]$", 1.0, False),
        ("within_cv", "within-trial modulation\n"
                      r"$\mathrm{med}_i\,\langle\sigma_t(r_i)\rangle_c/\bar r_i$", 1.0, False),
        ("rate_p90_p50", "rate tail\n" r"$\bar r_{(90)}/\bar r_{(50)}$", 1.0, False)]

    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    axf = axes.ravel()
    for ax, (field, label, scale, logy) in zip(axf, panels):
        w = 0.38
        for j, eq in enumerate(("h", "s")):
            m, h, xs = [], [], []
            for x, pen in enumerate(PENS):
                mu, hw = stat(eq, pen, field, scale)
                m.append(mu); h.append(hw); xs.append(x + (j - 0.5) * w)
            bars = ax.bar(xs, m, w, yerr=h, capsize=2, color=EQ_COL[eq],
                          label=f"{eq} equation", edgecolor="white", linewidth=1.2)
            for b, val, e in zip(bars, m, h):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + e,
                        f"{val:.3g}", ha="center", va="bottom", fontsize=7, color="0.25")
        ax.set_xticks(range(len(PENS))); ax.set_xticklabels(PENS)
        ax.set_xlabel("penalty"); ax.set_ylabel(label, fontsize=8)
        if logy:
            ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.92", lw=0.8); ax.set_axisbelow(True)
        if field == "sigma_log":
            # cortical rate distributions are close to lognormal with roughly a decade of spread
            ax.axhline(1.0, color="0.35", lw=1.2, ls="--")
            ax.text(3.45, 1.02, "cortex ≈ 1", fontsize=7.5, color="0.35", ha="right", va="bottom")
    axf[0].legend(frameon=False, fontsize=8)

    fig.suptitle("Networks with indistinguishable task performance (R² = 0.84–0.87) have very different "
                 "population statistics\n"
                 r"$r_i(t,c)$ = noise-free rate of unit $i$; $\bar r_i$ = mean over time and conditions; "
                 r"$\lambda_i$ = eigenvalues of the unit covariance; $\eta^2$ over the decision epoch."
                 "\n" r"All per-unit statistics are over ACTIVE units ($p_i \geq 10^{-6}$); PR and HHI need no "
                 "such correction. N = 1000, 5 nets per cell, 95% CI", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
