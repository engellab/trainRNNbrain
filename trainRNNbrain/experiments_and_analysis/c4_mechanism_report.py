#!/usr/bin/env python3
"""Read the c4_gradient_mechanism.py outputs: the pre-registered tests, per unit, and a figure.

Each test and its pass/fail bar is written in git show db10e7c:docs/notes/agent-c4.md (pre-registered before the
numbers were computed). This script prints, per run, the quantity each test reads, and draws:
  row 1 (flip-flop) and row 2 (CDDM), one column per question:
    A. g_in against the unit's on-fraction, active and silent units coloured (tests a, e)
    B. distribution of g_in, active vs silent, with the 5%-of-q95 bar (test c)
    C. reached / active as the bar moves from 0.5% to 50% of q95, every seed (test c)
    D. what 5000 iterations of real training did (c4_snapshot_drift.py): per unit class, the share
       of units active afterwards and the share whose drive at their 1% highest-h samples fell

Usage: python c4_mechanism_report.py NPZ_DIR [OUT_DIR]
  NPZ_DIR holds c4_gradient_mechanism.py's *__floor.npz and *__end.npz, and c4_snapshot_drift.py's
  *__drift.npz.
Output: OUT_DIR (default img/internal_figures)/c4_gradient_mechanism.{pdf,svg}
"""

import glob
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR

REL = 0.05
BARS = np.geomspace(0.005, 0.5, 25)
C_ACT, C_SIL, C_SR = "#1f6fb4", "#b0b0b0", "#d9541e"   # active, silent, silent-but-reached


def over(g, rel=REL):
    """Boolean mask of units with g >= rel * q95(g)."""
    return g >= rel * np.quantile(g, 0.95)


def rel_diff(a, b, floor):
    """Per-unit |a - b| / max(|b|, floor)."""
    return np.abs(a - b) / np.maximum(np.abs(b), floor)


def report(fn):
    """Print every pre-registered test for one run.

    Args:
        fn: an npz written by c4_gradient_mechanism.py.
    Returns:
        dict of per-unit arrays and masks used by the figure.
    """
    z = dict(np.load(fn, allow_pickle=True))
    task = "FF" if "FlipFlop" in fn else "CDDM"
    seed = fn.split("__s")[1].split("__")[0]
    act = over(z["ref_p"])
    gi, go = z["strict_g_in"], z["strict_g_out"]
    reach = over(gi)
    sr = reach & ~act
    nsamp = int(z["ref_T"]) * int(z["ref_B"])          # (time, trial) samples in the reference batch
    print(f"\n=== {task} seed {seed}  read={z['read']}@{z['read_iter']}  loss={float(z['strict_loss']):.5f}")
    print(f"active {act.sum()}  reached g_in {reach.sum()} ({reach.sum() / act.sum():.2f}x)  "
          f"g_out {over(go).sum()}  silent-reached {sr.sum()}  active-not-reached {(act & ~reach).sum()}")

    # oracle check against production (torch float32 autograd, clamp gate)
    for k in ("g_in", "g_out"):
        a, b = z[f"clamp_{k}"], z[f"prod_{k}"]
        keep = b >= 1e-4 * np.quantile(b, 0.95)
        d = rel_diff(a[keep], b[keep], 0)
        print(f"  ORACLE {k}: median rel diff {np.median(d):.1e}, max {d.max():.1e} over {keep.sum()} units; "
              f"counts oracle {over(a).sum()} prod {over(b).sum()}")
    print(f"  ORACLE p: max rel diff {rel_diff(z['ref_p'], z['prod_p'], 1e-12).max():.1e}; "
          f"loss {float(z['clamp_loss']):.6f} vs prod {float(z['prod_loss']):.6f}")

    # (b) exact zeros through the clamp gate
    d = rel_diff(z["clamp_g_in"], gi, 1e-300)
    print(f"  (b) strict vs clamp g_in: max rel diff {d.max():.1e} (silent units {d[~act].max():.1e}); "
          f"counts strict {reach.sum()} clamp {over(z['clamp_g_in']).sum()}; "
          f"units ever exactly 0: {(z['ref_n_zero'] > 0).sum()}")

    # (a) briefly on
    n_on = z["ref_n_on"]
    frac = n_on / nsamp
    print(f"  (a) silent-reached ever on: {(n_on[sr] > 0).sum()}/{sr.sum()}; on-fraction among them: "
          f"median {np.median(frac[sr]):.3f}, q90 {np.quantile(frac[sr], .9):.3f}, share < 10%: "
          f"{(frac[sr] < 0.1).mean():.2f}; active units' median on-fraction {np.median(frac[act]):.3f}")
    rho = spearmanr(gi[~act], n_on[~act])[0]
    print(f"      Spearman(g_in, on-count) over silent units {rho:.2f}; silent units never on: "
          f"{(n_on[~act] == 0).sum()} with g_in max {gi[~act][n_on[~act] == 0].max() if (n_on[~act] == 0).any() else float('nan'):.1e}")
    print(f"      silent-reached on in how many trials: median {np.median(z['ref_n_on_trials'][sr]):.0f} "
          f"of {int(z['ref_B'])}; on samples near an input pulse: "
          f"{z['ref_n_on_near_input'][sr].sum() / max(n_on[sr].sum(), 1):.2f} (all samples near input: "
          f"{float(z['ref_near_share']):.2f})")

    # (c) the bar
    cnt = np.array([over(gi, r).sum() for r in BARS])
    print(f"  (c) reached at 1% / 5% / 20% of q95: {over(gi, .01).sum()} / {reach.sum()} / "
          f"{over(gi, .2).sum()}  -> 1%:20% ratio {over(gi, .01).sum() / over(gi, .2).sum():.2f}; "
          f"active by p at 1%/20%: {over(z['ref_p'], .01).sum()}/{over(z['ref_p'], .2).sum()}")
    lg = np.log10(gi + 1e-300)
    print(f"      log10 g_in active [q05 {np.quantile(lg[act], .05):.2f}, median {np.median(lg[act]):.2f}]; "
          f"silent [median {np.median(lg[~act]):.2f}, q95 {np.quantile(lg[~act], .95):.2f}]; "
          f"bar {np.log10(REL * np.quantile(gi, .95)):.2f}")

    # (d) per-on-sample sensitivity
    d_on = np.sqrt(z["strict_d_on_ss"] / np.maximum(n_on, 1))
    print(f"  (d) RMS dL/dh over on samples: silent-reached median {np.median(d_on[sr]):.2e} vs active "
          f"{np.median(d_on[act]):.2e} -> ratio {np.median(d_on[sr]) / np.median(d_on[act]):.2f}")
    gr_rms = np.sqrt(z["strict_gr_ss"] / nsamp)
    print(f"      RMS dL/dr over all samples: silent-reached {np.median(gr_rms[sr]):.2e}, "
          f"silent-unreached {np.median(gr_rms[~act & ~reach]):.2e}, active {np.median(gr_rms[act]):.2e}")

    # (e) binary gate
    rmax = z["ref_r_max"]
    ratio = go / np.maximum(gi, 1e-300)
    print(f"  (e) peak rate silent-reached median {np.median(rmax[sr]):.3f} vs active {np.median(rmax[act]):.3f} "
          f"({np.median(rmax[sr]) / np.median(rmax[act]):.2f}); g_out/g_in silent-reached "
          f"{np.median(ratio[sr]):.3f} vs active {np.median(ratio[act]):.3f} "
          f"({np.median(ratio[act]) / np.median(ratio[sr]):.1f}x lower)")

    # (f) noise
    nz = np.array([over(g).sum() for g in z["noisy_g_in"]])
    print(f"  (f) reached under training noise: {nz.mean():.0f} (range {nz.min()}-{nz.max()}) vs noise-free "
          f"{reach.sum()} -> {nz.mean() / reach.sum():.2f}x; mean on-fraction of silent units noisy "
          f"{(z['noisy_n_on'][:, ~act].mean(0) / nsamp).mean():.3f} vs noise-free {frac[~act].mean():.3f}")

    # effect of one estimated Adam step
    dd = z["dd_top"]
    snr = z["snr_rec_row"]
    print(f"  STEP lr={float(z['lr']):.2e}: g_in silent-reached / active median {np.median(gi[sr]) / np.median(gi[act]):.2f}; "
          f"row SNR silent-reached {np.median(snr[sr]):.2f}, active {np.median(snr[act]):.2f}, "
          f"silent-unreached {np.median(snr[~act & ~reach]):.2f} (pure noise ~0.25)")
    for name, m in (("silent-reached", sr), ("silent-unreached", ~act & ~reach), ("active", act)):
        print(f"      {name:17s}: dd_top>0 share {(dd[m] > 0).mean():.2f}, median dd_top {np.median(dd[m]):+.2e}, "
              f"median h_q99 {np.median(z['ref_h_q99'][m]):+.3f}, median h_med {np.median(z['ref_h_med'][m]):+.3f}; "
              f"full re-run: on-count up {(z['step_n_on'][m] > n_on[m]).mean():.2f}, down "
              f"{(z['step_n_on'][m] < n_on[m]).mean():.2f}, dh_top>0 {(z['step_dh_top'][m] > 0).mean():.2f}")
    print(f"      active after the step {over(z['step_p']).sum()} (before {act.sum()})")
    if "cos_rec_row" in z:   # end of run: the saved optimizer's own next step
        for name, m in (("silent-reached", sr), ("silent-unreached", ~act & ~reach), ("active", act)):
            print(f"      TRUE Adam step, {name:17s}: cos(estimate, true) per W_rec row median "
                  f"{np.median(z['cos_rec_row'][m]):+.2f}; true dd_top>0 share {(z['true_dd_top'][m] > 0).mean():.2f}")
    return dict(task=task, seed=seed, read=str(z["read"]), act=act, reach=reach, sr=sr, gi=gi, frac=frac, dd=dd, cnt=cnt,
                bar=REL * np.quantile(gi, 0.95), hq99=z["ref_h_q99"])


def figure(runs, npz_dir, out_stem):
    """Two rows (flip-flop, CDDM) by four columns (A-D in the module docstring).

    Args:
        runs: list of report() dicts; npz_dir: folder holding the *__drift.npz files;
        out_stem: path without extension.
    """
    fig = plt.figure(figsize=(14, 6.6))
    gs = fig.add_gridspec(2, 4, left=0.06, right=0.99, bottom=0.09, top=0.92, wspace=0.32, hspace=0.42)
    for row, task in enumerate(("FF", "CDDM")):
        rs = [r for r in runs if r["task"] == task]
        if not rs:
            continue
        r0 = rs[0]
        name = "3-bit flip-flop" if task == "FF" else "CDDM"
        act, sr = r0["act"], r0["sr"]
        rest = ~act & ~sr
        # A: g_in vs on-fraction. Units never on have g_in = 0 exactly; they are drawn on a floor
        # one decade below the smallest nonzero value, at on-share 1e-6, and counted in the label.
        ax = fig.add_subplot(gs[row, 0])
        gi = r0["gi"]
        floor = gi[gi > 0].min() / 10
        never = gi == 0
        x = np.where(r0["frac"] > 0, r0["frac"], 1e-6)
        for m, c, lab in ((rest, C_SIL, "silent, not reached"), (sr, C_SR, "silent, reached"),
                          (act, C_ACT, "active")):
            ax.scatter(x[m], np.maximum(gi[m], floor), s=5, c=c, lw=0, label=f"{lab} ({m.sum()})")
        ax.axhline(r0["bar"], color="k", lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("share of samples with $h_i>0$")
        ax.set_ylabel(r"$g_{in}$")
        ax.set_title(f"A  {name}, seed {r0['seed']}", loc="left", fontsize=10)
        ax.legend(fontsize=7, loc="upper left", markerscale=2, frameon=False)
        if never.any():
            ax.annotate(f"{never.sum()} units never on:\n$g_{{in}}=0$ exactly", (1e-6, floor),
                        xytext=(3e-6, floor * 30), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
        # B: distributions (g_in = 0 drawn at the same floor as in A)
        ax = fig.add_subplot(gs[row, 1])
        lg = np.log10(np.maximum(gi, floor))
        bins = np.linspace(lg.min(), lg.max(), 50)
        ax.hist(lg[act], bins, color=C_ACT, alpha=0.7, label="active")
        ax.hist(lg[~act], bins, color=C_SIL, alpha=0.7, label="silent")
        ax.axvline(np.log10(r0["bar"]), color="k", lw=0.8, ls="--", label=r"5% of $q_{95}$")
        ax.set_xlabel(r"$\log_{10} g_{in}$"); ax.set_ylabel("units")
        ax.set_title("B  incoming-weight gradient", loc="left", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
        # C: bar dependence, every seed
        ax = fig.add_subplot(gs[row, 2])
        for r in rs:
            ax.plot(BARS * 100, r["cnt"] / r["act"].sum(), lw=1.2, label=f"seed {r['seed']} ({r['act'].sum()} active)")
        ax.axvline(REL * 100, color="k", lw=0.8, ls="--"); ax.axhline(1, color="k", lw=0.5)
        ax.set_xscale("log"); ax.set_xlabel(r"bar, % of $q_{95}(g_{in})$")
        ax.set_ylabel("units reached / active units")
        ax.set_title("C  the count depends on the bar", loc="left", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        # D: real training, 5000 iterations after the floor snapshot
        ax = fig.add_subplot(gs[row, 3])
        drift = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(npz_dir, f"{task}_s*__drift.npz")))]
        classes = (("active", C_ACT), ("silent-reached", C_SR), ("silent-unreached", C_SIL))
        for j, (k, c) in enumerate(classes):
            for q, (key, mk) in enumerate((("act_5000", "o"), ("dd_5000", "s"))):
                vals = [(d[key][d[f"cls_{k}"]] > 0).mean() if key == "act_5000" else
                        (d[key][d[f"cls_{k}"]] < 0).mean() for d in drift]
                x = q * 4 + j
                ax.bar(x, np.mean(vals), color=c, width=0.8)
                ax.scatter([x] * len(vals), vals, s=10, c="k", zorder=3)
        ax.set_xticks([1, 5]); ax.set_xticklabels(["active 5000\niterations later", "drive at its firing\nmoments fell"])
        ax.set_ylim(0, 1.05); ax.set_ylabel("share of units in the class")
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        ax.set_title("D  the next 5000 iterations", loc="left", fontsize=10)
    for ext in ("pdf", "svg"):
        fig.savefig(f"{out_stem}.{ext}")
    print("saved", out_stem)


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(sys.argv[1], "*__floor.npz")) + glob.glob(os.path.join(sys.argv[1], "*__end.npz")))
    runs = [report(f) for f in files]
    figure([r for r in runs if r["read"] == "floor"], sys.argv[1],
           os.path.join(sys.argv[2] if len(sys.argv) > 2 else IMG_DIR, "c4_gradient_mechanism"))
