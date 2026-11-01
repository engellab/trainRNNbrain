#!/usr/bin/env python3
"""Manuscript Figure 4 - an inactive unit has no route back - with Extended Data Figs. 6 and 7.

  a  net input current x_i(t) of a silent, a low-active and an active unit on one trial, CDDM and
     3BFF (one network each, at its loss floor)
  b  incoming-weight gradient g_in against the share of time steps with x_i > 0, every unit of one
     network per task, coloured active / low-active / silent
  c  what real training did next: share of each class (active, low-active reached or not, silent)
     active 500 and 5,000 iterations after the floor snapshot, every network
  d  tanh against ReLU (CDDM, N = 1000, end of 150,000 iterations): per-unit loss gradient on the
     state, per unit of error, trained against the same network untrained
  e  B4: share of the units inactive at transfer (S0) that are active at the new task's floor, old
     and young sources, N = 1000 and 4000, with the pre-registered 10% and 30% bars
  f  B4: active units at the new task's floor, old / young / fresh, with the old networks' count at
     transfer and the gate bar; held-out R^2 under each group

Extended Data Fig. 6 (`fig_ed_mechanism`): the gradient analysis on all six networks (g_in by
class, how the reached count depends on the bar) and the convergence checks of C4b (reach over
training and after a ten-fold learning-rate cut, the batch-consistent share of the gradient, the
noisy loss after the floor). Extended Data Fig. 7 (`fig_ed_tanh`): tanh against ReLU, nine networks
each (state gradient, output sensitivity, activation slope).

DEFINITIONS (as in the paper's Methods). Active: participation p >= 0.05 q95(p) on a noise-free
batch. Silent: an inactive unit whose x_i is <= 0 on every step of every trial of the reference
batch. Low-active: every other inactive unit. Reached: g_in >= 0.05 q95(g_in) of the same network
(c4_gradient_mechanism.py, noise-free reference batch, task seed 12345).

DATA. The six C4 networks are the Figure 1d networks with kept snapshots (three CDDM, three 3BFF;
the 3BFF ones are also the B4 sources). Every input is produced elsewhere and read here:
  C4 at the current floor   `collect` below plus c4_gradient_mechanism.py and c4_snapshot_drift.py,
                            run on Della (job 16566988; notebook 2026-10-31_2137 and _2207); local
                            copy in data/fig4_noreturn/c4_floor/ (npz), traces/ and snap/; `activity`
                            below recomputes participation at t*, t* + 500 and t* + 5000 on the full
                            reference batch (c4_snapshot_drift.py uses 256 trials) into activity/
  C4 on the Fig. 1d weights `fig1d` below (Mac, dead-relu-paper#91; notebook 2026-11-01_0948): the
                            same measurement on the snapshots Figure 1d and Extended Data Fig. 2 use
                            (data/pop_stats_nets), with participation 500 and 5,500 iterations later
                            (the d0 and d5000 copies above), into c4_fig1d/, traces_fig1d/ and
                            activity_fig1d/. Panels a-c and ED6a-b draw these.
  C4b convergence           c4b_convergence.py outputs, Della agent_c4b/out (14 Oct), local copy
                            data/fig4_noreturn/out/
  tanh                      gradient_participation.py (job 15064275) and gradient_sensitivity.py (job
                            15066631) JSONs, local copy data/fig4_noreturn/gradpart/, gradsens/
  B4                        read_b2_b4.json (Della ~/b3b4_read, job 16553903; identical copy in
                            data/remedy_reads_31oct/), b4_verdict_4000.json

Usage:
  Della:  python fig_paper_F4_noreturn.py collect --c4 DIR --snap DIR --traces DIR
          (after c4_gradient_mechanism.py RUNS --out <c4> --at floor; before c4_snapshot_drift.py)
  Mac:    python trainRNNbrain/experiments_and_analysis/fig_paper_F4_noreturn.py fig1d
          python trainRNNbrain/experiments_and_analysis/fig_paper_F4_noreturn.py activity
          python trainRNNbrain/experiments_and_analysis/fig_paper_F4_noreturn.py draw
Output: img/internal_figures/fig_paper_F4_noreturn, fig_ed_mechanism, fig_ed_tanh (.pdf + .svg),
and every number drawn, printed.
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c4_gradient_mechanism as C4
import paperstyle as ps
from common import DATA_DIR, SILENT_REL, active_mask, snapshot_at

# The six C4 networks: (task, sweep, seeds). The 3BFF runs are the 200k continuations; their floor
# falls inside the parent run, whose snapshots common.snapshot_at finds along the warm-start chain.
C4_RUNS = (
    ("CDDM", "CDDM_c1_snapshots", (1423687489, 2657899762, 2670295995)),
    ("3BFF", "NBitFlipFlop_c1_snapshots_c200000", (1258717025, 2461724787, 4113307577)),
)
CELL = "EqType=h_N=1000_arm=control"
DRIFT_DTS = (0, 500, 5000)     # snapshot offsets after the floor snapshot read by c4_snapshot_drift.py
N_TRACE_TRIALS = 16            # trials of the reference batch whose full x(t) is kept for panel a
LATER = (500, 5500)            # iterations after the Figure 1d snapshot of the later weights read (fig1d)


def run_dir(sweep, seed):
    """The run folder of one C4 network.

    Args:
        sweep: sweep folder name under data/trained_RNNs; seed: int.
    Returns:
        the path of the single run folder of that seed in CELL.
    """
    hits = glob.glob(os.path.join(DATA_DIR, sweep, CELL, f"*seed={seed};*"))
    assert len(hits) == 1, (sweep, seed, hits)
    return hits[0]


def collect(c4_dir, snap_dir, trace_dir):
    """Stage the snapshots c4_snapshot_drift.py reads and keep the panel-a traces (Della).

    For each C4 network, reads the floor snapshot iteration t* that c4_gradient_mechanism.py used
    (its npz in c4_dir), links the snapshots at t*, t* + 500 and t* + 5000 and the run config into
    snap_dir under the names c4_snapshot_drift.py expects, and runs the noise-free forward pass of
    the floor weights on the first N_TRACE_TRIALS trials of the reference batch.

    Args:
        c4_dir: folder with c4_gradient_mechanism.py's *__s<seed>__floor.npz;
        snap_dir: output folder for <seed>_d{0,500,5000}.npz and <seed>_config.yaml links;
        trace_dir: output folder for <task>_s<seed>__trace.npz holding x (T, N, trials) as float32,
            the inputs u (n_in, T, trials), the targets y (n_out, T, trials) and t*.
    """
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(trace_dir, exist_ok=True)
    for task, sweep, seeds in C4_RUNS:
        for seed in seeds:
            d = run_dir(sweep, seed)
            z = np.load(glob.glob(os.path.join(c4_dir, f"*__s{seed}__floor.npz"))[0])
            assert str(z["read"]) == "floor", (seed, z["read"])
            t = int(z["read_iter"])
            for dt in DRIFT_DTS:
                snap = snapshot_at(d, t + dt)
                assert snap is not None and int(os.path.basename(snap)[2:]) == t + dt, (seed, t, dt, snap)
                link = os.path.join(snap_dir, f"{seed}_d{dt}.npz")
                if not os.path.exists(link):
                    os.symlink(glob.glob(os.path.join(snap, "*LastParams*.npz"))[0], link)
            cfg_path = glob.glob(os.path.join(d, "*_config.yaml"))[0]
            link = os.path.join(snap_dir, f"{seed}_config.yaml")
            if not os.path.exists(link):
                os.symlink(cfg_path, link)
            P = C4.load_params(os.path.join(snap_dir, f"{seed}_d0.npz"))
            u, y, _ = C4.batch(OmegaConf.load(cfg_path), C4.REF_SEED)
            u, y = u[:, :, :N_TRACE_TRIALS], y[:, :, :N_TRACE_TRIALS]
            H, _, _, _ = C4.forward(P, u)
            fn = os.path.join(trace_dir, f"{task}_s{seed}__trace.npz")
            np.savez(fn, x=H.astype(np.float32), u=u.astype(np.float32), y=y.astype(np.float32),
                     t_star=t)
            print(f"{task} s{seed}: floor snapshot {t}; drift links {DRIFT_DTS}; wrote {fn}", flush=True)


def participation(H):
    """Participation p = std(r) + q0.9(|r|) of every unit over all steps and trials.

    Args:
        H: (T, N, B) net input currents of a noise-free pass (ReLU network).
    Returns:
        (N,) participation.
    """
    R = np.maximum(H, 0.0).transpose(1, 0, 2).reshape(H.shape[1], -1)
    return R.std(1) + np.quantile(R, 0.9, axis=1)


def activity(snap_dir, out_dir):
    """Participation on the FULL reference batch at t*, t* + 500 and t* + 5000 (Mac or Della).

    c4_snapshot_drift.py recomputes participation on the first 256 trials only, where a CDDM
    network has about 80 fewer active units than on the full 450-trial batch that defines the
    classes; this pass keeps the batch identical across the three snapshots and the classes.

    Args:
        snap_dir: collect()'s snapshot links (or a copy of them); out_dir: output folder for
            <task>_s<seed>__act.npz holding p_<dt> (N,) for dt in DRIFT_DTS.
    """
    os.makedirs(out_dir, exist_ok=True)
    for task, _, seeds in C4_RUNS:
        for seed in seeds:
            cfg = OmegaConf.load(os.path.join(snap_dir, f"{seed}_config.yaml"))
            u, _, _ = C4.batch(cfg, C4.REF_SEED)
            res = {}
            for dt in DRIFT_DTS:
                P = C4.load_params(os.path.join(snap_dir, f"{seed}_d{dt}.npz"))
                H, _, _, _ = C4.forward(P, u)
                res[f"p_{dt}"] = participation(H)
                del H
            fn = os.path.join(out_dir, f"{task}_s{seed}__act.npz")
            np.savez(fn, **res)
            print(f"{task} s{seed}: active at +0 / +500 / +5000: "
                  + " / ".join(str(int(active_mask(res[f'p_{dt}']).sum())) for dt in DRIFT_DTS), flush=True)


def fig1d(pop_dir, snap_dir, c4_out, trace_out, act_out, K=10):
    """C4 on the snapshots Figure 1d and Extended Data Fig. 2 use (Mac; dead-relu-paper#91).

    For each C4 network: c4_gradient_mechanism.run_one on the population-statistics copy of the
    Figure 1d weights; the noise-free x(t) of the first N_TRACE_TRIALS reference trials (panel a);
    participation on the full reference batch at that snapshot and at the two later snapshots kept
    on the Mac, LATER[0] and LATER[1] iterations after it (the Della read's d0 and d5000 copies).
    Rule and checks: notebook 2026-11-01_0948.

    Args:
        pop_dir: data/pop_stats_nets (one folder per network with config and LastParams, and
            manifest.tsv: label, run_dir, params_file, iteration, active count of the removal file);
        snap_dir: the Della read's snapshot copies <seed>_d{0,5000}.npz and <seed>_config.yaml;
        c4_out, trace_out, act_out: output folders (npz per network, names as load_c4 reads them);
        K: number of training-noise batches.
    """
    rows = {}
    with open(os.path.join(pop_dir, "manifest.tsv")) as fh:
        for line in fh:
            label, _, _, it, n_act = line.rstrip("\n").split("\t")
            rows[label] = (int(it), int(n_act))
    for d in (c4_out, trace_out, act_out):
        os.makedirs(d, exist_ok=True)
    for task, sweep, seeds in C4_RUNS:
        for seed in seeds:
            label = f"{sweep}__{CELL}__s{seed}__floor"
            folder = os.path.join(pop_dir, label)
            it, n_act = rows[label]
            params = glob.glob(os.path.join(folder, "*LastParams*.npz"))[0]
            cfg_path = glob.glob(os.path.join(folder, "*_config.yaml"))[0]
            cfg = OmegaConf.load(cfg_path)
            # the Della copies must belong to the same run
            assert OmegaConf.to_container(cfg) == OmegaConf.to_container(
                OmegaConf.load(os.path.join(snap_dir, f"{seed}_config.yaml"))), seed
            fn = f"{label}.npz"
            if not os.path.exists(os.path.join(c4_out, fn)):
                C4.run_one(folder, c4_out, K, "floor", point=(params, "floor", it), fn=fn)
            u, y, _ = C4.batch(cfg, C4.REF_SEED)
            res = {}
            for dt, path in ((0, params), (LATER[0], os.path.join(snap_dir, f"{seed}_d0.npz")),
                             (LATER[1], os.path.join(snap_dir, f"{seed}_d5000.npz"))):
                H, _, _, _ = C4.forward(C4.load_params(path), u)
                res[f"p_{dt}"] = participation(H)
                if dt == 0:
                    k = N_TRACE_TRIALS
                    np.savez(os.path.join(trace_out, f"{task}_s{seed}__trace.npz"), x=H[:, :, :k].astype(np.float32),
                             u=u[:, :, :k].astype(np.float32), y=y[:, :, :k].astype(np.float32), t_star=it)
                del H
            np.savez(os.path.join(act_out, f"{task}_s{seed}__act.npz"), **res)
            print(f"{task} s{seed} at {it}: active {int(active_mask(res['p_0']).sum())} (removal file {n_act}); "
                  f"+{LATER[0]} {int(active_mask(res[f'p_{LATER[0]}']).sum())}; "
                  f"+{LATER[1]} {int(active_mask(res[f'p_{LATER[1]}']).sum())}", flush=True)


# --------------------------------------------------------------------------------------------
# drawing (Mac)
# --------------------------------------------------------------------------------------------

LOCAL = os.path.join(DATA_DIR, "..", "fig4_noreturn")
C4_DIR = os.path.join(LOCAL, "c4_floor")             # c4_gradient_mechanism + c4_snapshot_drift output
TRACE_DIR = os.path.join(LOCAL, "traces")            # collect() output
SNAP_LOCAL = os.path.join(LOCAL, "snap")             # collect()'s snapshot links, copied (rsync -L)
ACT_DIR = os.path.join(LOCAL, "activity")            # activity() output
POP_DIR = os.path.join(DATA_DIR, "..", "pop_stats_nets")   # the Figure 1d weights (population statistics)
FIG1D_C4 = os.path.join(LOCAL, "c4_fig1d")           # fig1d() output: what the figures draw
FIG1D_TRACES = os.path.join(LOCAL, "traces_fig1d")
FIG1D_ACT = os.path.join(LOCAL, "activity_fig1d")
C4B_DIR = os.path.join(LOCAL, "out")                 # c4b_convergence.py output (14 Oct)
GRADPART_DIR = os.path.join(LOCAL, "gradpart")       # gradient_participation.py, job 15064275
GRADSENS_DIR = os.path.join(LOCAL, "gradsens")       # gradient_sensitivity.py, job 15066631
TANH_TRACES = os.path.join(LOCAL, "tanh_traces")     # CDDM_std_g0_tanh participation traces
B4_JSON = os.path.join(DATA_DIR, "..", "remedy_reads_31oct", "read_b2_b4.json")
B4_VERDICT_4000 = os.path.join(LOCAL, "b4_verdict_4000.json")

# Colours, one meaning each (paperstyle, docs/figure_style.md). Unit classes are neutrals: active
# near-black, inactive light grey; silent units are the inactive units drawn hollow. Tasks keep
# their reserved hues and appear only where a panel compares tasks (Extended Data Fig. 6b-e).
# Networks in the transfer test are ink marks told apart by shape; the untrained (fresh)
# reference is BASELINE grey, the colour of every untreated reference in the paper.
# Fill (dead-relu-paper #83): a filled marker is a value read at the floor crossing, an open one a
# value at 75,000 iterations. Every marker here is read at or after a floor crossing, so every
# marker is filled; silent units and the network groups differ by shape.
C_ACT, C_INA = ps.ACTIVE, ps.INACTIVE
C_TRACE = ps.INK
TASK_COL = {"3BFF": ps.TASK_3BFF, "CDDM": ps.TASK_CDDM}
SRC_STYLE = {"old": dict(marker="o", mfc=ps.INK, mec=ps.INK),
             "young": dict(marker="^", mfc=ps.INK, mec=ps.INK),
             "fresh": dict(marker="s", mfc=ps.BASELINE, mec=ps.BASELINE)}
REACH_REL = SILENT_REL                               # the reach threshold, as a fraction of q95(g_in)
BARS = np.geomspace(0.005, 0.5, 25)                  # ED6b: reach thresholds swept, fraction of q95
F4_H_MM = 140.0                                      # Figure 4 height; the cap is paperstyle.MAX_H_MM


def load_c4():
    """Per C4 network, the classes and per-unit quantities read by panels a-c and ED6a-b.

    Returns:
        list of dicts (one per network, CDDM first) with task, seed, t (floor snapshot iteration),
        masks act / low / sil / reach, on-share (share of reference-batch samples with x > 0), g
        (g_in over the network's q95 of g_in), p, the arrays act_<dt> for dt in LATER (units active
        that many iterations later; full reference batch) and the trace file, if present. Read from
        fig1d()'s outputs (the Figure 1d snapshots).
    """
    out = []
    for task, _, seeds in C4_RUNS:
        for seed in seeds:
            z = np.load(glob.glob(os.path.join(FIG1D_C4, f"*__s{seed}__floor.npz"))[0])
            act = active_mask(z["ref_p"])
            n_on = z["ref_n_on"]
            sil = ~act & (n_on == 0)
            gi = z["strict_g_in"]
            r = dict(task=task, seed=seed, t=int(z["read_iter"]), act=act, sil=sil, low=~act & ~sil,
                     reach=active_mask(gi), on=n_on / (int(z["ref_T"]) * int(z["ref_B"])),
                     g=gi / np.quantile(gi, 0.95), g_raw=gi, g_torch=z["prod_g_in"], p=z["ref_p"],
                     n_on=n_on, on_trials=z["ref_n_on_trials"], B=int(z["ref_B"]),
                     near_input=z["ref_n_on_near_input"],
                     # 10 training-like batches under the run's own noise: g_in over each batch's q95
                     g_noisy_rel=z["noisy_g_in"] / np.quantile(z["noisy_g_in"], 0.95, axis=1)[:, None],
                     n_on_noisy=z["noisy_n_on"], n_samples=int(z["ref_T"]) * int(z["ref_B"]))
            fa = os.path.join(FIG1D_ACT, f"{task}_s{seed}__act.npz")
            if os.path.exists(fa):
                za = np.load(fa)
                # the +0 pass repeats c4_gradient_mechanism's reference pass: same classes expected
                r["act_0"] = active_mask(za["p_0"])
                for dt in LATER:
                    r[f"act_{dt}"] = active_mask(za[f"p_{dt}"])
                r["p_check"] = float(np.max(np.abs(za["p_0"] - z["ref_p"]) / np.maximum(z["ref_p"], 1e-12)))
            ft = os.path.join(FIG1D_TRACES, f"{task}_s{seed}__trace.npz")
            r["trace"] = ft if os.path.exists(ft) else None
            out.append(r)
    return out


def report_c4(nets):
    """Print every C4 number the caption, text and notebook use, and the pre-registered rule.

    Args:
        nets: load_c4() output.
    Returns:
        dict task -> list of per-network summary dicts.
    """
    summ = {}
    print("\n== C4 on the Figure 1d snapshots (classes on the noise-free reference batch)")
    for r in nets:
        low_r = r["low"] & r["reach"]
        s = dict(seed=r["seed"], t=r["t"], active=int(r["act"].sum()), silent=int(r["sil"].sum()),
                 low=int(r["low"].sum()), low_reached=int(low_r.sum()),
                 sil_gmax=float(r["g_raw"][r["sil"]].max()) if r["sil"].any() else float("nan"),
                 # the same, from PyTorch autograd (float32, torch.clamp): code independent of C4's
                 sil_gmax_torch=float(r["g_torch"][r["sil"]].max()) if r["sil"].any() else float("nan"),
                 on_med_low_reached=float(np.median(r["on"][low_r])) if low_r.any() else float("nan"),
                 on_max_low=float(r["on"][r["low"]].max()),
                 trials_med_low_reached=float(np.median(r["on_trials"][low_r])) if low_r.any() else float("nan"),
                 B=r["B"])
        s["near_input_share"] = float(r["near_input"][low_r].sum() / max(r["n_on"][low_r].sum(), 1))
        for dt in LATER:
            k = f"act_{dt}"
            if k in r:
                for name, m in (("active", r["act"]), ("low_reached", low_r),
                                ("low_unreached", r["low"] & ~r["reach"]), ("silent", r["sil"])):
                    s[f"{name}_{dt}"] = float(r[k][m].mean()) if m.any() else float("nan")
        if "act_0" in r:
            s["class_check"] = int((r["act_0"] != r["act"]).sum())
            print(f"  {r['task']} s{r['seed']}: full-batch re-run at t* differs from the C4 classes in "
                  f"{s['class_check']} units (largest relative p difference {r['p_check']:.1e})")
        if r["sil"].any():
            G, On = r["g_noisy_rel"][:, r["sil"]], r["n_on_noisy"][:, r["sil"]]
            s.update(noisy_on_share=float(np.median(On) / r["n_samples"]), noisy_zero=int((G == 0).sum()),
                     noisy_n=int(G.size), noisy_med=float(np.median(G)),
                     noisy_cross=int((G >= REACH_REL).any(0).sum()))
            print(f"  {r['task']} s{r['seed']}: silent units under the training noise (10 batches): x > 0 on a "
                  f"median {100 * s['noisy_on_share']:.2f}% of steps; {s['noisy_zero']} of {s['noisy_n']} "
                  f"per-batch gradients exactly 0; median g_in {s['noisy_med']:.1e} of the batch's q95 "
                  f"({REACH_REL / s['noisy_med']:.0f}x below the reach threshold); {s['noisy_cross']} units "
                  f"at or above the threshold in some batch")
        summ.setdefault(r["task"], []).append(s)
        print(f"  {r['task']} s{r['seed']} t*={r['t']}: active {s['active']}, silent {s['silent']} "
              f"(largest g_in {s['sil_gmax']:.1e}; autograd {s['sil_gmax_torch']:.1e}), low-active {s['low']} of which reached {s['low_reached']}; "
              f"reached low-active: median on-share {s['on_med_low_reached']:.3f}, on in a median "
              f"{s['trials_med_low_reached']:.0f} of {s['B']} trials, {s['near_input_share']:.2f} of their "
              f"on-steps within 2 tau of an input; largest on-share of any low-active unit {s['on_max_low']:.3f}")
        if f"active_{LATER[1]}" in s:
            print(f"     share active at t*+{LATER[0]} / t*+{LATER[1]}: " + ", ".join(
                f"{n} {s[f'{n}_{LATER[0]}']:.3f}/{s[f'{n}_{LATER[1]}']:.3f}" for n in
                ("active", "low_reached", "low_unreached", "silent")))
    ff, cd = summ["3BFF"], summ["CDDM"]
    rule = [("every 3BFF network has 0 silent units", all(s["silent"] == 0 for s in ff)),
            ("every CDDM network has >= 200 silent units with largest g_in exactly 0",
             all(s["silent"] >= 200 and s["sil_gmax"] == 0 for s in cd)),
            ("every 3BFF network has >= 100 reached low-active units", all(s["low_reached"] >= 100 for s in ff)),
            (f"in every 3BFF network < 10% of reached low-active units are active at t*+{LATER[1]}",
             all(s.get(f"low_reached_{LATER[1]}", np.nan) < 0.10 for s in ff))]
    print("  RULE (notebook 2026-10-31_2137, read as in 2026-11-01_0948): " + "; ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in rule))
    return summ


def load_b4():
    """B4 per run, from read_b2_b4.json, with the gate and verdict recomputed here.

    Returns:
        dict N -> dict with old / young / fresh run lists (each run: n_S0, revived, active0 at
        transfer, active_star at the floor, r2) and gate / verdict numbers.
    """
    d = json.load(open(B4_JSON))["b4"]
    out = {}
    print("\n== B4 (read_b2_b4.json; rule: notebook 2026-10-15_1342)")
    for N in (1000, 4000):
        g = {src: d[f"{N}_{key}"] for src, key in (("old", "src=control_age=old"),
                                                    ("young", "src=control_age=young"),
                                                    ("fresh", "src=fresh"))}
        fresh = np.array([r["active_star"] for r in g["fresh"]], float)
        old_tr = np.array([r["active0"] for r in g["old"]], float)
        old_star = np.array([r["active_star"] for r in g["old"]], float)
        share = np.array([r["revived"] / r["n_S0"] for r in g["old"]])
        f_mean, f_sd = fresh.mean(), fresh.std(ddof=1)
        gate = f_mean - old_tr.mean() > 3 * f_sd
        back = (share < 0.10).all() and f_mean - old_star.mean() > 3 * f_sd
        recruit = (share >= 0.30).sum() >= 2 or f_mean - old_star.mean() < f_sd
        verdict = ("do not come back" if back else "can be recruited" if recruit else "between the bars")
        g.update(fresh_mean=f_mean, fresh_sd=f_sd, old_tr=old_tr.mean(), old_star=old_star.mean(),
                 gate=gate, verdict=verdict if gate else "no verdict (gate fails)")
        out[N] = g
        for src in ("old", "young", "fresh"):
            for r in g[src]:
                print(f"  N={N} {src:5s}: S0 {r['n_S0']:4d}, revived {r['revived']:3d} "
                      f"({100 * r['revived'] / r['n_S0']:.1f}%), active at transfer {r['active0']:4d}, "
                      f"at the floor {r['active_star']:4d}, R2 {r['r2']:.3f} ({'measured' if r['r2_measured'] else 'estimated'}),"
                      f" t* {r['t_star']:.0f}")
        print(f"  N={N}: fresh {f_mean:.0f} +- {f_sd:.0f}; old at transfer {old_tr.mean():.0f}; old at floor "
              f"{old_star.mean():.0f}; gate {'passes' if gate else 'fails'} (gap {f_mean - old_tr.mean():.0f}, "
              f"bar {3 * f_sd:.0f}); verdict: {g['verdict']}")
    ref = json.load(open(B4_JSON))["b4_verdict_1000"], json.load(open(B4_VERDICT_4000))
    for N, v in zip((1000, 4000), ref):
        print(f"  check against the read's own b4_verdict at N={N}: {v['verdict']}, gate {v['gate']}, "
              f"fresh {v['fresh_mean']:.0f} +- {v['fresh_sd']:.0f}, old at transfer {v['old_at_transfer']:.0f}, "
              f"old at floor {v['old_star']:.0f}")
        assert v["verdict"] == out[N]["verdict"] and bool(v["gate"]) == bool(out[N]["gate"])
    return out


def load_tanh():
    """Per network of CDDM_std_g0_tanh, the per-unit measures of jobs 15064275 and 15066631.

    Each loss gradient is divided by sqrt(loss) ("per unit of error") and every measure is divided
    by the untrained network's 95th percentile of the same measure, so the bar sits at 0.05.

    Returns:
        {"ReLU": [run, ...], "tanh": [...]}; each run a dict of seed and, per measure key (g_state,
        g_in, s_in, s_state, slope), `key` (trained) and `key + "0"` (untrained).
    """
    out = {"ReLU": [], "tanh": []}
    for f in sorted(glob.glob(os.path.join(GRADPART_DIR, "*.json"))):
        r = json.load(open(f))
        act = "tanh" if "act=tanh" in os.path.basename(f) else "ReLU"
        seed = os.path.basename(f).split("seed=")[1].split(".json")[0]
        run = dict(seed=seed, slope=np.asarray(r["trained"]["slope"]), slope0=np.asarray(r["init"]["slope"]))
        fs = glob.glob(os.path.join(GRADSENS_DIR, f"*seed={seed}*.json"))
        rs = json.load(open(fs[0])) if fs else None
        for key, src, per_err in (("g_state", r, True), ("g_in_trial", rs, True), ("s_in", rs, False),
                                  ("s_state", rs, False)):
            k = "g_in" if key == "g_in_trial" else key
            if src is None or k not in src["trained"]:
                continue
            v0, v = np.asarray(src["init"][k], float), np.asarray(src["trained"][k], float)
            if per_err:
                v0, v = v0 / np.sqrt(src["init"]["loss"]), v / np.sqrt(src["trained"]["loss"])
            q = np.quantile(v0, 0.95)
            run[key + "0"], run[key] = v0 / q, v / q
        out[act].append(run)
    print("\n== tanh against ReLU (CDDM, N = 1000, end of 150,000 iterations)")
    for act, runs in out.items():
        drop = [np.median(r["g_state0"]) / np.median(r["g_state"]) for r in runs]
        above = {k: sorted(int((r[k] >= SILENT_REL).sum()) for r in runs if k in r)
                 for k in ("g_state", "g_in_trial", "s_in", "s_state")}
        print(f"  {act}: {len(runs)} networks; median state gradient per unit of error falls "
              f"{np.median(drop):.0f}-fold (range {min(drop):.0f}-{max(drop):.0f}); median slope "
              f"{np.median([np.median(r['slope0']) for r in runs]):.2f} untrained -> "
              f"{', '.join(f'{np.median(r['slope']):.2f}' for r in runs)} trained")
        for k, v in above.items():
            print(f"     units at or above 0.05 of the untrained q95, {k}: {v}")
    return out


def tanh_traces():
    """Active count over training of every CDDM_std_g0_tanh network, from its participation trace.

    Returns:
        dict "ReLU"/"tanh" -> (iterations, (n_runs, n_probes) active counts, per-run lowest p / q95(p)
        at the last probe).
    """
    out = {}
    for act, cell in (("ReLU", "EqType=h_N=1000_act=relu_iters=150000"),
                      ("tanh", "EqType=h_N=1000_act=tanh_iters=150000")):
        it, cnt, low = None, [], []
        for f in sorted(glob.glob(os.path.join(TANH_TRACES, cell, "*", "*ParticipationTrace.pkl"))):
            tr = pickle.load(open(f, "rb"))
            P = np.asarray(tr["participation"], float)
            it = np.asarray(tr["participation_iters"], float)
            cnt.append([active_mask(p).sum() for p in P])
            low.append(P[-1].min() / np.quantile(P[-1], 0.95))
        out[act] = (it, np.asarray(cnt, float), np.asarray(low))
        print(f"  {act}: {len(cnt)} traces; active at iteration {it[-1]:.0f}: "
              f"{sorted(int(c[-1]) for c in cnt)}; lowest unit's p / q95 {np.min(low):.2f}-{np.max(low):.2f}")
    return out


def pick_trace_units(r, n_candidates=20):
    """Which units and trial panel a draws for one network.

    The active unit is the one at the 75th percentile of participation among active units. The low-active
    unit is one of the n_candidates reached low-active units whose share of steps with x > 0 is
    nearest the class median; among them and the stored trials on which that unit is on at least
    once, the pair is chosen on which the active unit's largest x_i is highest, so that both
    classes are visible on one trial. The silent unit (CDDM) is the middle one by index.

    Args:
        r: one load_c4() dict with a trace file; n_candidates: low-active units considered.
    Returns:
        (x (T, N) of that trial, u (n_in, T), dict class -> unit index, trial index).
    """
    z = np.load(r["trace"])
    x = z["x"]
    pick = {}
    a = np.where(r["act"])[0]
    pick["active"] = a[np.argsort(r["p"][a])[(3 * len(a)) // 4]]
    lr = np.where(r["low"] & r["reach"])[0]
    order = lr[np.argsort(np.abs(r["on"][lr] - np.median(r["on"][lr])))][:n_candidates]
    best = None
    for i in order:
        for k in np.where((x[:, i, :] > 0).any(0))[0]:
            score = x[:, pick["active"], k].max()
            if best is None or score > best[0]:
                best = (score, i, int(k))
    _, pick["low-active"], trial = best
    if r["sil"].any():
        s = np.where(r["sil"])[0]
        pick["silent"] = s[len(s) // 2]
    return x[:, :, trial], z["u"][:, :, trial], pick, trial


def draw_traces(fig, gs_cell, nets):
    """Panel a: x_i(t) of a silent, a low-active and an active unit on one trial, CDDM and 3BFF.

    Args:
        fig: figure; gs_cell: the GridSpec cell the panel fills; nets: load_c4() output.
    Returns:
        the top-left axes (for the panel letter).
    """
    sub = gs_cell.subgridspec(3, 2, hspace=0.3, wspace=0.18)
    first = None
    for col, task in enumerate(("CDDM", "3BFF")):
        rs = [r for r in nets if r["task"] == task and r["trace"]]
        r = sorted(rs, key=lambda q: (q["low"] & q["reach"]).sum())[len(rs) // 2]
        x, u, pick, trial = pick_trace_units(r)
        t = np.arange(x.shape[0]) / 10.0                       # steps of 0.1 tau
        # input pulses: channels on for under 90% of the trial (CDDM's context cue, on throughout,
        # is left out)
        on_ch = np.abs(u) > 0
        inp = on_ch[on_ch.mean(1) < 0.9].any(0)
        for row, cls in enumerate(("silent", "low-active", "active")):
            ax = fig.add_subplot(sub[row, col])
            first = first or ax
            ax.set_xlim(0, t[-1])
            if cls in pick:
                xi = x[:, pick[cls]]
                lim = max(np.abs(xi).max() * 1.15, 1e-3)
                ax.set_ylim(-lim, lim * 1.35)
                ax.axhline(0, color=ps.MUTED, lw=ps.THIN_LW, ls=":")
                ax.plot(t, xi, color=C_TRACE, lw=ps.DATA_LW * 0.8)
                ax.fill_between(t, 0, np.maximum(xi, 0), color=C_TRACE, alpha=0.35, lw=0)
                top = ax.get_ylim()[1]
                ax.fill_between(t, top * 0.86, top, where=inp, color=ps.FAINT, lw=0, step="mid")
                print(f"  panel a {task} s{r['seed']} trial {trial}: {cls} unit {pick[cls]}, "
                      f"x {xi.min():+.3f} to {xi.max():+.3f}, positive on {int((xi > 0).sum())} of {len(xi)} steps")
            else:
                ax.set_ylim(-1, 1)
                ax.set_yticks([])
                ax.text(0.5, 0.45, "no silent unit\nin any 3BFF network", transform=ax.transAxes,
                        ha="center", va="center", fontsize=ps.TICK_PT, color=ps.MUTED)
            ax.yaxis.set_major_locator(plt.MaxNLocator(2, symmetric=True))
            if row < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"time ($\tau$)", labelpad=1)
            if col == 0:
                ax.set_ylabel(cls, labelpad=2)
            if row == 0:
                ax.set_title(f"{task}, $N$ = 1,000", pad=3)
    return first


def log_zero_axis(ax, which, zero, ticks):
    """Label a log axis with '0' at the position where exact zeros are drawn.

    Args:
        ax: axes; which: 'x' or 'y'; zero: data value standing for 0; ticks: other tick values.
    """
    labels = ["0"] + [f"$10^{{{int(np.log10(v))}}}$" if v != 1 else "1" for v in ticks]
    (ax.set_xticks if which == "x" else ax.set_yticks)([zero] + list(ticks))
    (ax.set_xticklabels if which == "x" else ax.set_yticklabels)(labels)


def draw_gradient(fig, gs_cell, nets):
    """Panel b: g_in against the share of steps with x_i > 0, every unit of one network per task.

    Silent units have both quantities exactly 0 on the noise-free batch; they are drawn at the
    axes' '0' ticks, as diamonds.

    Args:
        fig: figure; gs_cell: GridSpec cell; nets: load_c4() output.
    Returns:
        the left axes.
    """
    sub = gs_cell.subgridspec(1, 2, wspace=0.12)
    XZ, YZ = 5e-6, 1e-8          # where the exact zeros of silent units are drawn
    first = None
    for col, task in enumerate(("CDDM", "3BFF")):
        rs = [r for r in nets if r["task"] == task]
        r = sorted(rs, key=lambda q: (q["low"] & q["reach"]).sum())[len(rs) // 2]
        ax = fig.add_subplot(sub[0, col])
        first = first or ax
        rng = np.random.default_rng(0)
        ax.axhline(REACH_REL, color=ps.INK, lw=ps.THIN_LW, ls="--", zorder=1)
        for m, kw in ((r["sil"], dict(marker="D", c=C_INA, lw=0, s=3)),
                      (r["low"], dict(c=C_INA, lw=0, s=2.5)),
                      (r["act"], dict(c=C_ACT, lw=0, s=2.5))):
            if not m.any():
                continue
            xs = np.where(r["on"][m] > 0, r["on"][m], XZ * np.exp(rng.normal(0, 0.25, m.sum())))
            ys = np.where(r["g"][m] > 0, r["g"][m], YZ * np.exp(rng.normal(0, 0.25, m.sum())))
            ax.scatter(xs, ys, zorder=2, **kw)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(XZ / 2.5, 1.5)
        ax.set_ylim(YZ / 3, 30)
        log_zero_axis(ax, "x", XZ, [1e-4, 1e-2, 1])
        log_zero_axis(ax, "y", YZ, [1e-6, 1e-4, 1e-2, 1])
        if col:
            ax.set_yticklabels([])
        ax.minorticks_off()
        ax.set_xlabel("share of steps with $x_i > 0$", labelpad=1)
        if col == 0:
            ax.set_ylabel("incoming-weight gradient\n(95th percentile = 1)", labelpad=1)
        ax.set_title(f"{task}, $N$ = 1,000", pad=3)
        print(f"  panel b {task} s{r['seed']}: {r['act'].sum()} active, {r['low'].sum()} low-active "
              f"({(r['low'] & r['reach']).sum()} reached), {r['sil'].sum()} silent")
    # one legend for both networks, in the 3BFF axes' empty lower right
    hs = [Line2D([], [], ls="", marker="o", ms=3, mfc=C_ACT, mec=C_ACT, label="active"),
          Line2D([], [], ls="", marker="o", ms=3, mfc=C_INA, mec=C_INA, label="low-active"),
          Line2D([], [], ls="", marker="D", ms=2.5, mfc=C_INA, mec=C_INA, label="silent"),
          Line2D([], [], ls="--", color=ps.INK, lw=ps.THIN_LW, label="reach threshold")]
    ax.legend(handles=hs, loc="lower right", handletextpad=0.2, borderaxespad=0.2, labelspacing=0.2)
    return first


def draw_drift(ax, summ):
    """Panel c: share of each class active LATER[0] and LATER[1] iterations after the floor crossing.

    Args:
        ax: axes; summ: report_c4() output.
    """
    groups = [("CDDM", "active"), ("CDDM", "low_reached"), ("CDDM", "low_unreached"), ("CDDM", "silent"),
              ("3BFF", "active"), ("3BFF", "low_reached"), ("3BFF", "low_unreached")]
    style = {"active": dict(mfc=C_ACT, mec=C_ACT), "low_reached": dict(mfc=C_INA, mec=C_INA),
             "low_unreached": dict(mfc=C_INA, mec=C_INA), "silent": dict(mfc=C_INA, mec=C_INA)}
    xs = np.array([0, 1, 2, 3, 4.6, 5.6, 6.6])
    rng = np.random.default_rng(1)
    for x, (task, cls) in zip(xs, groups):
        for dt, dx, mk in ((LATER[0], -0.18, "^"), (LATER[1], 0.18, "o")):
            v = np.array([100 * s[f"{cls}_{dt}"] for s in summ[task]])
            ax.plot(x + dx + rng.normal(0, 0.03, len(v)), v, ls="", marker=mk, ms=3, mew=0.6,
                    clip_on=False, zorder=3, **style[cls])
            ax.plot([x + dx - 0.13, x + dx + 0.13], [v.mean()] * 2, color=ps.INK, lw=ps.DATA_LW)
    ax.set_xticks(xs)
    ax.set_xticklabels(["active", "reached", "not\nreached", "silent", "active", "reached", "not\nreached"])
    for x0, x1 in ((1, 2), (5.6, 6.6)):                 # the two low-active groups share one label
        ax.annotate("", xy=(x0 - 0.35, -0.27), xytext=(x1 + 0.35, -0.27), xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", lw=ps.THIN_LW, color=ps.MUTED))
        ax.annotate("low-active", xy=((x0 + x1) / 2, -0.3), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=ps.TICK_PT, color=ps.INK)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, 7.2)
    ax.set_ylabel("% of the class active later", labelpad=1)
    for xm, task in ((1.5, "CDDM"), (5.6, "3BFF")):
        ax.text(xm, 1.03, f"{task}, $N$ = 1,000", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=ps.LABEL_PT)
    ax.legend(handles=[Line2D([], [], marker=mk, ls="", mec=ps.INK, mfc=ps.INK, ms=3,
                              label=f"+{ps.fmt_count(dt)} iterations") for mk, dt in zip("^o", LATER)],
              loc="center right", handletextpad=0.2, bbox_to_anchor=(1.0, 0.6))
    ps.ygrid(ax)


def draw_tanh_hist(fig, gs_cell, tanh):
    """Panel d: per-unit state gradient per unit of error, trained against untrained, ReLU and tanh.

    One network per activation, the one whose count at or above the reach threshold is the
    median of nine. Outlines only: no unit-class colour, since the threshold here is set on the
    untrained network.

    Args:
        fig: figure; gs_cell: GridSpec cell; tanh: load_tanh() output.
    Returns:
        the left axes.
    """
    sub = gs_cell.subgridspec(1, 2, wspace=0.1)
    FLOOR = 1e-5
    bins = np.logspace(np.log10(FLOOR * 0.7), np.log10(40), 46)
    first = None
    for col, act in enumerate(("ReLU", "tanh")):
        runs = tanh[act]
        n = np.array([(r["g_state"] >= SILENT_REL).sum() for r in runs])
        r = runs[int(np.argsort(n)[len(n) // 2])]
        drop = [np.median(q["g_state0"]) / np.median(q["g_state"]) for q in runs]
        ax = fig.add_subplot(sub[0, col])
        first = first or ax
        v, v0 = np.maximum(r["g_state"], FLOOR), np.maximum(r["g_state0"], FLOOR)
        ax.hist(v0, bins=bins, histtype="step", color=ps.BASELINE, lw=ps.DATA_LW, label="untrained")
        ax.hist(v, bins=bins, histtype="step", color=ps.INK, lw=ps.DATA_LW, label="trained")
        ax.axvline(SILENT_REL, color=ps.INK, lw=ps.THIN_LW, ls="--", label="threshold")
        ax.set_xscale("log")
        ax.set_xlim(FLOOR * 0.7, 40)
        ax.set_xticks([1e-5, 1e-3, 1e-1, 10])
        ax.set_xticklabels(["$\\leq 10^{-5}$", "$10^{-3}$", "$10^{-1}$", "10"])
        ax.minorticks_off()
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.set_title(f"{act}, CDDM, $N$ = 1,000", pad=3)
        ax.set_xlabel("gradient on $x_i$ per unit of error\n(untrained 95th percentile = 1)", labelpad=1)
        if col == 0:
            ax.set_ylabel("units", labelpad=1)
        else:
            ax.legend(loc="upper left", handlelength=1.2, borderaxespad=0.2, labelspacing=0.2)
        print(f"  panel d {act}: drawn seed {r['seed']}, {int(n[runs.index(r)])} at or above the reach threshold, "
              f"median-unit drop {drop[runs.index(r)]:.1f}; "
              f"nine networks {sorted(int(k) for k in n)}; median-unit drop {np.median(drop):.1f} "
              f"({min(drop):.1f}-{max(drop):.1f})")
    return first


def draw_b4_revived(ax, b4):
    """Panel e: share of S0 (units inactive when the new task starts) active at its floor crossing.

    Args:
        ax: axes; b4: load_b4() output.
    """
    xs = {(1000, "old"): 0, (1000, "young"): 1, (4000, "old"): 2.6, (4000, "young"): 3.6}
    rng = np.random.default_rng(2)
    labels = []
    for (N, src), x in xs.items():
        v = np.array([100 * r["revived"] / r["n_S0"] for r in b4[N][src]])
        ax.plot(x + rng.normal(0, 0.04, len(v)), v, ls="", ms=3, mew=0.6, zorder=3, **SRC_STYLE[src])
        ax.plot([x - 0.22, x + 0.22], [v.mean()] * 2, color=ps.INK, lw=ps.DATA_LW)
        s0 = [r["n_S0"] for r in b4[N][src]]
        labels.append(f"{src}\n{ps.fmt_count(int(np.median(s0)))}")
        print(f"  panel e N={N} {src}: S0 {sorted(s0)} (median drawn), share active "
              f"{sorted(round(float(q), 1) for q in v)}%")
    ax.axhline(10, color=ps.INK, lw=ps.THIN_LW, ls="--")
    ax.axhline(30, color=ps.INK, lw=ps.THIN_LW, ls=":")
    ax.text(4.25, 11.5, "pass threshold", ha="right", va="bottom", fontsize=ps.TICK_PT)
    ax.text(4.25, 31.5, "fail threshold", ha="right", va="bottom", fontsize=ps.TICK_PT)
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels(labels)
    ax.annotate("median $S_0$", xy=(-0.03, -0.2), xycoords="axes fraction", ha="right", va="center",
                fontsize=ps.TICK_PT, color=ps.MUTED)
    for x, N in ((0.5, 1000), (3.1, 4000)):
        ax.text(x, 1.03, f"$N$ = {ps.fmt_count(N)}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=ps.LABEL_PT)
    ax.set_xlim(-0.6, 4.3)
    ax.set_ylim(0, 40)
    ax.set_ylabel("% of $S_0$ active at\nthe floor crossing", labelpad=1)
    ps.ygrid(ax)


def draw_b4_active(ax, b4):
    """Panel f: active units at the new task's floor crossing, old / young / fresh networks.

    Args:
        ax: axes; b4: load_b4() output.
    """
    rng = np.random.default_rng(3)
    ticks, labels = [], []
    for N, x0 in ((1000, 0), (4000, 3.8)):
        g = b4[N]
        for j, src in enumerate(("old", "young", "fresh")):
            x = x0 + j
            v = np.array([r["active_star"] for r in g[src]], float)
            ax.plot(x + rng.normal(0, 0.04, len(v)), v, ls="", ms=3, mew=0.6, zorder=3, **SRC_STYLE[src])
            ax.plot([x - 0.22, x + 0.22], [v.mean()] * 2, color=ps.INK, lw=ps.DATA_LW)
            if src == "old":
                v0 = np.array([r["active0"] for r in g[src]], float)
                ax.plot(x - 0.42 + rng.normal(0, 0.02, 3), v0, "x", ms=3, mew=0.7, color=ps.INK, zorder=3)
            r2 = [r["r2"] for r in g[src]]
            ticks.append(x)
            labels.append(f"{src}\n{min(r2):.2f}" + (f"–{max(r2):.2f}" if f"{max(r2):.2f}" != f"{min(r2):.2f}" else ""))
            print(f"  panel f N={N} {src}: active at the floor crossing {sorted(v.astype(int).tolist())}, "
                  f"R2 {min(r2):.3f}-{max(r2):.3f}")
        bar = g["fresh_mean"] - 3 * g["fresh_sd"]
        ax.plot([x0 - 0.65, x0 + 0.35], [bar] * 2, color=ps.INK, lw=ps.THIN_LW, ls="--")
        ax.text(x0 + 1, 1.03, f"$N$ = {ps.fmt_count(N)}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=ps.LABEL_PT)
    ax.annotate("$R^2$", xy=(-0.03, -0.2), xycoords="axes fraction", ha="right", va="center",
                fontsize=ps.TICK_PT, color=ps.MUTED)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.8, 6.3)
    ax.set_ylim(0, 1000)
    ps.count_ticks(ax.yaxis)
    ax.set_ylabel("active units at the\nfloor crossing", labelpad=1)
    ax.legend(handles=[Line2D([], [], marker="x", ls="", color=ps.INK, ms=3, label="old, at the start"),
                       Line2D([], [], ls="--", color=ps.INK, lw=ps.THIN_LW, label="pass threshold")],
              loc="upper left", handletextpad=0.3, borderaxespad=0.1, labelspacing=0.2)
    ps.ygrid(ax)


def check_layout(fig, name):
    """Print overlapping text pairs and text lying on a drawn line (display coordinates).

    A line counts as under a text if any of 60 points sampled along any of its segments falls inside
    the text's bounding box. Tick lines, spines, grid lines and legend handles are not axes lines and
    are not tested.

    Args:
        fig: a drawn figure; name: label for the printout.
    Returns:
        (number of overlapping text pairs, number of text-on-line cases).
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text().strip()]
    hidden = set()   # tick labels of ticks outside an axis's view are created but never drawn
    for ax in fig.axes:
        for lo_hi, labels, locs in ((sorted(ax.get_xlim()), ax.get_xticklabels(), ax.get_xticks()),
                                    (sorted(ax.get_ylim()), ax.get_yticklabels(), ax.get_yticks())):
            for lab, v in zip(labels, locs):
                if not lo_hi[0] - 1e-12 <= v <= lo_hi[1] + 1e-12:
                    hidden.add(id(lab))
    boxes = [(t, t.get_window_extent(rend)) for t in texts if id(t) not in hidden]
    n_text = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.overlaps(b) and a.width > 0 and b.width > 0:
                if (min(a.x1, b.x1) - max(a.x0, b.x0)) * (min(a.y1, b.y1) - max(a.y0, b.y0)) > 2.0:
                    n_text += 1
                    print(f"  OVERLAP in {name}: {boxes[i][0].get_text()[:30]!r} / {boxes[j][0].get_text()[:30]!r}")
    n_line = 0
    for ax in fig.axes:
        for ln in ax.lines:
            if not ln.get_visible() or ln.get_linestyle() in ("None", "", " "):
                continue
            P = ln.get_transform().transform(np.column_stack(ln.get_data()).astype(float))
            P = P[np.all(np.isfinite(P), 1)]
            if len(P) < 2:
                continue
            f = np.linspace(0, 1, 60)[:, None, None]
            S = (P[:-1][None] * (1 - f) + P[1:][None] * f).reshape(-1, 2)
            if ln.get_clip_on():                     # a clipped line is drawn only inside its axes
                ab = ax.bbox
                S = S[(S[:, 0] >= ab.x0) & (S[:, 0] <= ab.x1) & (S[:, 1] >= ab.y0) & (S[:, 1] <= ab.y1)]
            for t, bb in boxes:
                bb = bb.padded(0.5)
                inside = (S[:, 0] > bb.x0) & (S[:, 0] < bb.x1) & (S[:, 1] > bb.y0) & (S[:, 1] < bb.y1)
                if inside.any():
                    n_line += 1
                    print(f"  TEXT ON LINE in {name}: {t.get_text()[:30]!r} (line {ln.get_label()!r}, "
                          f"{len(P)} points)")
    W, H = fig.get_size_inches() * fig.dpi
    for t, bb in boxes:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            print(f"  OUTSIDE in {name}: {t.get_text()[:30]!r}")
    print(f"  layout {name}: {n_text} overlapping text pairs, {n_line} texts on a line")
    return n_text, n_line


def figure4(nets, summ, tanh, b4):
    """Draw Figure 4 (panels a-f in the module docstring) at the printed width.

    Args:
        nets: load_c4(); summ: report_c4(); tanh: load_tanh(); b4: load_b4().
    Returns:
        the PDF path.
    """
    fig = ps.figure(h_mm=F4_H_MM)
    r1 = fig.add_gridspec(1, 2, left=0.085, right=0.99, top=0.955, bottom=0.71, width_ratios=[1.1, 1],
                          wspace=0.28)
    r2 = fig.add_gridspec(1, 2, left=0.085, right=0.99, top=0.585, bottom=0.40, width_ratios=[1, 1.15],
                          wspace=0.24)
    r3 = fig.add_gridspec(1, 2, left=0.085, right=0.99, top=0.255, bottom=0.075, width_ratios=[1, 1.35],
                          wspace=0.25)
    ps.panel_letter(draw_traces(fig, r1[0, 0], nets), "a")
    ps.panel_letter(draw_gradient(fig, r1[0, 1], nets), "b", x_mm=11)
    ax = fig.add_subplot(r2[0, 0])
    draw_drift(ax, summ)
    ps.panel_letter(ax, "c")
    ps.panel_letter(draw_tanh_hist(fig, r2[0, 1], tanh), "d", x_mm=9)
    ax = fig.add_subplot(r3[0, 0])
    draw_b4_revived(ax, b4)
    ps.panel_letter(ax, "e")
    ax = fig.add_subplot(r3[0, 1])
    draw_b4_active(ax, b4)
    ps.panel_letter(ax, "f", x_mm=9)
    check_layout(fig, "fig_paper_F4_noreturn")
    return ps.save(fig, "fig_paper_F4_noreturn")


def ed_mechanism(nets):
    """Extended Data figure: gradient reach on all six networks and the C4b convergence checks.

    a  g_in by class on the noise-free batch, three networks pooled per task, and the silent units'
       g_in on the 10 batches under the training noise (outline); b reached / active against the
       reach threshold; c low-active units reached at fixed iterations and after 300 iterations at a
       tenth of the learning rate; d share of the per-batch gradient common to 32 batches;
       e noisy training loss over its final value.

    Args:
        nets: load_c4() output.
    Returns:
        the PDF path.
    """
    fig = ps.figure(h_mm=100)
    top = fig.add_gridspec(1, 3, left=0.075, right=0.99, top=0.93, bottom=0.62, wspace=0.34)
    bot = fig.add_gridspec(1, 3, left=0.075, right=0.99, top=0.42, bottom=0.11, wspace=0.34)
    bins = np.logspace(-7, 1.5, 40)
    for col, task in enumerate(("CDDM", "3BFF")):
        ax = fig.add_subplot(top[0, col])
        rs = [r for r in nets if r["task"] == task]
        ax.axvline(REACH_REL, color=ps.INK, lw=ps.THIN_LW, ls="--", label="reach threshold")
        for key, kw, lab in (("low", dict(color=C_INA), "low-active"), ("act", dict(color=C_ACT, alpha=0.8), "active")):
            v = np.concatenate([r["g"][r[key]] for r in rs])
            ax.hist(np.maximum(v, 1.5e-7), bins=bins, lw=0, label=lab, **kw)
        if any(r["sil"].any() for r in rs):
            v = np.concatenate([r["g_noisy_rel"][:, r["sil"]].ravel() for r in rs])
            w = np.full(len(v), 1.0 / 10)                    # per batch: 10 batches pooled
            ax.hist(np.maximum(v, 1.5e-7), bins=bins, weights=w, histtype="step", color=C_INA, lw=ps.DATA_LW,
                    ls="-", label="silent, training noise")
        ax.set_xscale("log")
        ax.set_xticks([1e-6, 1e-4, 1e-2, 1])
        ax.minorticks_off()
        ax.set_xlabel("incoming-weight gradient\n(95th percentile = 1)", labelpad=1)
        ax.set_ylabel("units" if col == 0 else "")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
        ax.set_title(f"{task}, $N$ = 1,000, 3 networks", pad=3)
        ax.legend(loc="upper left", handlelength=1.0, borderaxespad=0.1, labelspacing=0.2)
        if col == 0:
            ps.panel_letter(ax, "a")
    ax = fig.add_subplot(top[0, 2])
    for r in nets:
        cnt = np.array([active_mask(r["g_raw"], b).sum() for b in BARS])
        ax.plot(BARS * 100, cnt / r["act"].sum(), color=TASK_COL[r["task"]], lw=ps.DATA_LW * 0.8,
                label=r["task"] if r["seed"] in (1423687489, 1258717025) else None)
    ax.axvline(REACH_REL * 100, color=ps.INK, lw=ps.THIN_LW, ls="--")
    ax.axhline(1, color=ps.MUTED, lw=ps.THIN_LW, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([1, 10])
    ax.set_xticklabels(["1", "10"])
    ax.minorticks_off()
    ax.set_xlabel("reach threshold (% of the 95th percentile)", labelpad=1)
    ax.set_ylabel("units reached / active units", labelpad=1)
    ax.legend(loc="upper right")
    ps.ygrid(ax)
    ps.panel_letter(ax, "b")
    grads = {}
    for f in sorted(glob.glob(os.path.join(C4B_DIR, "grad_*.npz"))):
        _, name, it = os.path.basename(f)[:-4].split("_")
        if not name.startswith("DMTS"):
            grads[(name, int(it))] = dict(np.load(f))
    ax_c, ax_d = fig.add_subplot(bot[0, 0]), fig.add_subplot(bot[0, 1])
    print("\n== Extended Data figure (C4b, 14 Oct): low-active units reached, F_net, per measurement")
    for name in sorted({k[0] for k in grads}):
        task = "3BFF" if name.startswith("FF") else "CDDM"
        its = sorted(k[1] for k in grads if k[0] == name)
        reach = [int(grads[(name, i)]["cls_silent_reached"].sum()) for i in its]
        fnet = [float(grads[(name, i)]["F_net"]) for i in its]
        ax_c.plot(its, reach, "o-", color=TASK_COL[task], ms=2.5, lw=ps.THIN_LW * 1.5)
        ax_d.plot(its, fnet, "o-", color=TASK_COL[task], ms=2.5, lw=ps.THIN_LW * 1.5)
        line = f"  {name}: reached " + ", ".join(f"{i // 1000}k {n}" for i, n in zip(its, reach))
        fa = glob.glob(os.path.join(C4B_DIR, f"ft_{name}_*_anneal.npz"))
        if fa:
            n_after = int(np.load(fa[0])["cls_silent_reached"].sum())
            it_end = int(os.path.basename(fa[0]).split("_")[2])
            ax_c.plot([it_end + 8000], [n_after], "s", mfc=TASK_COL[task], mec=TASK_COL[task], ms=3, mew=0.7)
            line += f"; after lr/10 for 300 iterations {n_after}"
        print(line + "; F_net " + ", ".join(f"{f:.2f}" for f in fnet))
    for a in (ax_c, ax_d):
        a.set_xlim(0, 320000)
        a.set_xticks([0, 100000, 200000, 300000])
        ps.count_ticks(a.xaxis)
        a.set_xlabel(ps.LABELS["iteration"], labelpad=1)
        ps.ygrid(a)
    ax_c.set_ylabel("low-active units reached", labelpad=1)
    ax_c.set_ylim(0, 340)
    ax_c.legend(handles=[Line2D([], [], color=TASK_COL["3BFF"], marker="o", ms=2.5, lw=ps.THIN_LW * 1.5, label="3BFF"),
                         Line2D([], [], color=TASK_COL["CDDM"], marker="o", ms=2.5, lw=ps.THIN_LW * 1.5, label="CDDM"),
                         Line2D([], [], ls="", marker="s", mfc=ps.INK, mec=ps.INK, ms=3,
                                label="learning rate / 10")], loc="upper center", ncol=3,
                handletextpad=0.3, columnspacing=0.7, borderaxespad=0.1)
    ps.panel_letter(ax_c, "c")
    ax_d.set_ylabel("gradient share common\nto 32 batches", labelpad=1)
    ax_d.set_ylim(0, 1.05)
    ps.panel_letter(ax_d, "d")
    ax = fig.add_subplot(bot[0, 2])
    loss = json.load(open(os.path.join(C4B_DIR, "loss.json")))
    for name, r in loss.items():
        if name.startswith("DMTS"):
            continue
        task = "3BFF" if name.startswith("FF") else "CDDM"
        c = np.array(r["curve"])
        nb = len(c) // 5
        c5 = np.median(c[:nb * 5].reshape(nb, 5), 1)
        x5 = np.array(r["curve_it"])[:nb * 5].reshape(nb, 5).mean(1)
        ax.plot(x5, c5 / np.median(c5[-2:]), color=TASK_COL[task], lw=ps.DATA_LW * 0.8)
        print(f"  {name}: noisy-loss slope " + ", ".join(
            f"{w} {r[w]['lo'] // 1000}-{r[w]['hi'] // 1000}k {r[w]['noisy']['pct100k']:+.1f}% per 100k"
            for w in ("last", "ff200") if w in r))
    ax.set_yscale("log")
    ax.set_ylim(0.95, 2)
    ax.set_yticks([1, 1.5, 2])
    ax.set_yticklabels(["1", "1.5", "2"])
    ax.minorticks_off()
    ax.set_xlim(0, 320000)
    ax.set_xticks([0, 100000, 200000, 300000])
    ps.count_ticks(ax.xaxis)
    ax.set_xlabel(ps.LABELS["iteration"], labelpad=1)
    ax.set_ylabel("noisy training loss / final", labelpad=1)
    ps.ygrid(ax)
    ps.panel_letter(ax, "e")
    check_layout(fig, "fig_ed_mechanism")
    return ps.save(fig, "fig_ed_mechanism")


def ed_tanh(tanh, traces):
    """Extended Data figure: tanh against ReLU, CDDM, N = 1000, nine networks each.

    a  active count by the participation criterion over training; b units at or above 5% of the
    untrained network's 95th percentile, four per-unit measures; c median activation slope,
    untrained against trained. The activation function is a standard training setting, so tanh
    takes the setting colour and ReLU, the default, the baseline grey.

    Args:
        tanh: load_tanh() output; traces: tanh_traces() output.
    Returns:
        the PDF path.
    """
    fig = ps.figure(h_mm=52)
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.99, top=0.9, bottom=0.27, wspace=0.34,
                          width_ratios=[1, 1.3, 0.75])
    col = {"ReLU": ps.BASELINE, "tanh": ps.SETTING}
    ax = fig.add_subplot(gs[0, 0])
    for act in ("ReLU", "tanh"):
        it, cnt, _ = traces[act]
        for j, row in enumerate(cnt):
            ax.plot(it, row, color=col[act], lw=ps.THIN_LW, label=act if j == 0 else None)
    ax.set_xticks([0, 50000, 100000, 150000])
    ps.count_ticks(ax.xaxis)
    ps.count_ticks(ax.yaxis)
    ax.set_xlabel(ps.LABELS["iteration"], labelpad=1)
    ax.set_ylabel(ps.of_n("active units", 1000), labelpad=1)
    ax.set_ylim(0, 1100)
    ax.legend(loc="center right")
    ps.ygrid(ax)
    ps.panel_letter(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    keys = [("g_state", "gradient\non $x_i$"), ("g_in_trial", "gradient\non incoming\nweights"),
            ("s_in", "sensitivity\nto incoming\nweights"), ("s_state", "sensitivity\nto $x_i$")]
    rng = np.random.default_rng(4)
    for j, (k, _) in enumerate(keys):
        for dx, act in ((-0.18, "ReLU"), (0.18, "tanh")):
            v = np.array([int((r[k] >= SILENT_REL).sum()) for r in tanh[act] if k in r], float)
            ax.plot(j + dx + rng.normal(0, 0.03, len(v)), v, "o", ms=2.5, color=col[act], mew=0, zorder=3)
            ax.plot([j + dx - 0.12, j + dx + 0.12], [np.median(v)] * 2, color=ps.INK, lw=ps.DATA_LW)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([lab for _, lab in keys])
    ax.set_ylim(0, 1100)
    ps.count_ticks(ax.yaxis)
    ax.set_ylabel("units at or above\nthe reach threshold", labelpad=1)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=col[a], ms=3, label=a) for a in ("ReLU", "tanh")],
              loc="upper right")
    ps.ygrid(ax)
    ps.panel_letter(ax, "b")
    ax = fig.add_subplot(gs[0, 2])
    for act in ("ReLU", "tanh"):
        for r in tanh[act]:
            ax.plot([0, 1], [np.median(r["slope0"]), np.median(r["slope"])], "o-", color=col[act], ms=2.5,
                    lw=ps.THIN_LW)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["untrained", "trained"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("median activation slope", labelpad=1)
    ps.ygrid(ax)
    ps.panel_letter(ax, "c")
    check_layout(fig, "fig_ed_tanh")
    return ps.save(fig, "fig_ed_tanh")


def draw():
    """Read every input, print every number, and write the three figures."""
    ps.setup()
    nets = load_c4()
    summ = report_c4(nets)
    b4 = load_b4()
    tanh = load_tanh()
    traces = tanh_traces()
    print(figure4(nets, summ, tanh, b4))
    print(ed_mechanism(nets))
    print(ed_tanh(tanh, traces))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--c4", required=True)
    c.add_argument("--snap", required=True)
    c.add_argument("--traces", required=True)
    sub.add_parser("activity")
    sub.add_parser("fig1d")
    sub.add_parser("draw")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.c4, a.snap, a.traces)
    elif a.cmd == "activity":
        activity(SNAP_LOCAL, ACT_DIR)
    elif a.cmd == "fig1d":
        fig1d(POP_DIR, SNAP_LOCAL, FIG1D_C4, FIG1D_TRACES, FIG1D_ACT)
    else:
        draw()
