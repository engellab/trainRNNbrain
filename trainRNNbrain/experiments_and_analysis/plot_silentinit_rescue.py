#!/usr/bin/env python3
"""
Prevention vs resurrection — deliberate silent-at-init experiment (CDDM_d9e0ec_g0_silentinit).

Each net was initialised with a fixed random 25% of units (set S) forced silent by over-inhibition
(model.inhibitory_boost=2.0; see calibrate_inhibitory_boost.py). S is reconstructable from the net's
seed. We reconstruct the exact perturbed init, regenerate S, load the trained net, and ask what
happened to the S units — separately for the `none` (no rescue pressure) and `frm` (rescuer) penalties.

The decisive comparison is S's fate under frm vs none:
  - if S stays silent under frm  -> frm works only by PREVENTION (cannot revive dead units);
  - if S becomes active under frm but stays silent under none -> frm can RESURRECT;
  - if S reactivates under none too -> the c=2 silencing was simply undone by training (not frm-specific).

"active/rescued" = trained peak firing rate >= 0.01 (crossed the dead line); we also report the
median trained participation of S vs non-S (are rescued S units fully in the active mode?).

Outputs:
  - data/trained_RNNs/CDDM_d9e0ec_g0_silentinit/silentinit_rescue.csv  (per-condition summary)
  - img/internal_figures/silentinit_rescue_scatter_{h,s}.png  (per-unit init->trained, S vs non-S)

Run from this directory:  python3 plot_silentinit_rescue.py
"""
import os
import glob
import json
import csv
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from trainRNNbrain.utils import unjsonify
from plot_init_vs_trained import (build_cddm_batch, peak_and_participation, rnn_numpy_from_params,
                                  init_params_from_config, SEED_OFFSET)

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = "CDDM_d9e0ec_g0_silentinit"
ROOT = os.path.join(HERE, "../../data/trained_RNNs", SWEEP)
IMG_DIR = os.path.join(HERE, "../../img/internal_figures")
FRAC = 0.25
DEAD_ABS = 0.01
PEN = {"0": ("none (control)", "#c44e52"), "0.2": ("frm=0.2 (rescuer)", "#4c72b0")}


def reconstruct_S(config_path, N):
    """Regenerate the silenced set S for one net from its saved seed (matches RNN_torch.__init__)."""
    cfg = OmegaConf.load(config_path)
    per_net_seed = int(cfg.seed) + SEED_OFFSET
    n_sil = int(round(FRAC * N))
    return np.random.default_rng(per_net_seed).choice(N, size=n_sil, replace=False)


def analyse():
    """Per net: reconstruct perturbed init + S, load trained; collect per-unit peak/participation.

    Returns:
        dict keyed by (eq, frm) -> dict of pooled arrays (init/trained peak & participation for S and
        non-S), plus per-net init-silent checks.
    """
    leaf = sorted(d for d in glob.glob(os.path.join(ROOT, "EqType=*", "*")) if os.path.isdir(d))
    batch = build_cddm_batch(glob.glob(os.path.join(leaf[0], "*_config.yaml"))[0])
    print(f"CDDM batch {batch.shape}; {len(leaf)} nets\n")
    out = {}
    for d in leaf:
        cond = os.path.basename(os.path.dirname(d))
        eq = cond.split("EqType=")[1][0]
        frm = cond.split("LmbdFR=")[1]
        cfgp = glob.glob(os.path.join(d, "*_config.yaml"))[0]
        pj = glob.glob(os.path.join(d, "*_LastParams_CDDM.json"))[0]

        ip = init_params_from_config(cfgp)               # perturbed init (config carries the boost)
        N = np.asarray(ip["W_rec"]).shape[0]
        S = reconstruct_S(cfgp, N); mask = np.zeros(N, bool); mask[S] = True
        pk_i, pa_i = peak_and_participation(rnn_numpy_from_params(ip), batch)
        with open(pj) as f:
            tp = unjsonify(json.load(f))
        pk_t, pa_t = peak_and_participation(rnn_numpy_from_params(tp), batch)

        rec = out.setdefault((eq, frm), dict(pk_i_S=[], pk_t_S=[], pk_t_nS=[],
                                             pa_i_S=[], pa_t_S=[], pa_i_nS=[], pa_t_nS=[],
                                             initS_sil=[], initnS_sil=[]))
        rec["pk_i_S"].append(pk_i[mask]);  rec["pk_t_S"].append(pk_t[mask]);  rec["pk_t_nS"].append(pk_t[~mask])
        rec["pa_i_S"].append(pa_i[mask]);  rec["pa_t_S"].append(pa_t[mask])
        rec["pa_i_nS"].append(pa_i[~mask]); rec["pa_t_nS"].append(pa_t[~mask])
        rec["initS_sil"].append(np.mean(pk_i[mask] < DEAD_ABS))
        rec["initnS_sil"].append(np.mean(pk_i[~mask] < DEAD_ABS))
        print(f"{eq} frm={frm:>3}  initS_sil={100*rec['initS_sil'][-1]:5.1f}%  "
              f"trainedS_active={100*np.mean(pk_t[mask]>=DEAD_ABS):5.1f}%  "
              f"S_med_peak={np.median(pk_t[mask]):.3f}  nonS_med_peak={np.median(pk_t[~mask]):.3f}")
    return out


def summarise(out):
    """Write the per-condition rescue table and return rows."""
    rows = []
    for (eq, frm), r in sorted(out.items()):
        nnets = len(r["initS_sil"])
        initS = 100 * np.mean(r["initS_sil"])
        pk_t_S = np.concatenate(r["pk_t_S"]); pk_t_nS = np.concatenate(r["pk_t_nS"])
        S_active = 100 * np.mean(pk_t_S >= DEAD_ABS)
        nS_active = 100 * np.mean(pk_t_nS >= DEAD_ABS)
        S_medpart = float(np.median(np.concatenate(r["pa_t_S"])))
        nS_medpart = float(np.median(np.concatenate(r["pa_t_nS"])))
        rows.append([eq, frm, nnets, round(initS, 1), round(S_active, 1), round(nS_active, 1),
                     round(S_medpart, 4), round(nS_medpart, 4)])
    out_csv = os.path.join(ROOT, "silentinit_rescue.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eq", "frm", "n_nets", "init_S_silent_pct", "trained_S_active_pct",
                    "trained_nonS_active_pct", "S_median_participation", "nonS_median_participation"])
        w.writerows(rows)
    print(f"\nwrote {os.path.normpath(out_csv)}")
    hdr = f"{'eq':>3} {'frm':>4} {'nets':>4} {'initS silent':>12} {'trainedS active':>15} {'S med part':>11} {'nonS med part':>13}"
    print(hdr); print("-" * len(hdr))
    for eq, frm, n, iS, sA, nA, sP, nP in rows:
        print(f"{eq:>3} {frm:>4} {n:>4} {iS:11.1f}% {sA:14.1f}% {sP:11.3f} {nP:13.3f}")
    return rows


def plot_scatter(out, eq, lim):
    """Per-unit init->trained participation, S (red) vs non-S (grey), none | frm panels."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    flo, yhi = 1e-4, 1.5 * lim         # log floor for plotting (S init participation ~2e-4)
    sil = 0.05
    clip = lambda v: np.clip(v, flo, yhi)
    for ax, frm in zip(axes, ("0", "0.2")):
        r = out.get((eq, frm))
        label, _ = PEN[frm]
        ax.set_title(f"{label}", fontsize=11, fontweight="bold")
        ax.plot([flo, yhi], [flo, yhi], color="0.6", lw=1, ls=":", zorder=1)
        ax.axhline(sil, color="0.7", lw=0.9, zorder=1)
        if r is not None:
            # both groups plotted at their real init participation: S starts silent (~2e-4), non-S
            # starts normal (~0.1). y = trained participation. Rescue = S climbing above the silent guide.
            xi_S = np.concatenate(r["pa_i_S"]); yt_S = np.concatenate(r["pa_t_S"])
            xi_nS = np.concatenate(r["pa_i_nS"]); yt_nS = np.concatenate(r["pa_t_nS"])
            ax.scatter(clip(xi_nS), clip(yt_nS), s=3, color="0.55", alpha=0.10,
                       edgecolor="none", zorder=2, label="non-S (active at init)")
            ax.scatter(clip(xi_S), clip(yt_S), s=6, color="#c44e52", alpha=0.25,
                       edgecolor="none", zorder=3, label="S (silenced at init)")
            rescued = 100 * np.mean(np.concatenate(r["pk_t_S"]) >= DEAD_ABS)
            ax.text(0.04, 0.96, f"S rescued (peak≥0.01): {rescued:.0f}%\nS med trained part: {np.median(yt_S):.3f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(flo, yhi); ax.set_ylim(flo, yhi)
        ax.set_xlabel("participation at INIT")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("participation after TRAINING")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(f"Silenced-at-init units (set S) through training — {eq} equation — {SWEEP}\n"
                 f"(log-log; dotted=no change; grey line=silent guide {sil})", y=1.02)
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    out_png = os.path.join(IMG_DIR, f"silentinit_rescue_scatter_{eq}.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight"); plt.close(fig)
    return out_png


def main():
    out = analyse()
    summarise(out)
    all_pa = np.concatenate([np.concatenate(r["pa_t_S"]) for r in out.values()] +
                            [np.concatenate(r["pa_t_nS"]) for r in out.values()])
    lim = float(np.percentile(all_pa[np.isfinite(all_pa)], 99.5))
    for eq in ("h", "s"):
        if any(e == eq for (e, _f) in out):
            print(f"wrote {os.path.normpath(plot_scatter(out, eq, lim))}")


if __name__ == "__main__":
    main()
