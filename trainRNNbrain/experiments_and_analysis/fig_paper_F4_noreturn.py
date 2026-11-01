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
  C4b convergence           c4b_convergence.py outputs, Della agent_c4b/out (14 Oct), local copy
                            data/fig4_noreturn/out/
  tanh                      gradient_participation.py (job 15064275) and gradient_sensitivity.py (job
                            15066631) JSONs, local copy data/fig4_noreturn/gradpart/, gradsens/
  B4                        read_b2_b4.json (Della ~/b3b4_read, job 16553903; identical copy in
                            data/remedy_reads_31oct/), b4_verdict_4000.json

Usage:
  Della:  python fig_paper_F4_noreturn.py collect --c4 DIR --snap DIR --traces DIR
          (after c4_gradient_mechanism.py RUNS --out <c4> --at floor; before c4_snapshot_drift.py)
  Mac:    python trainRNNbrain/experiments_and_analysis/fig_paper_F4_noreturn.py activity
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
from matplotlib import font_manager
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


# --------------------------------------------------------------------------------------------
# drawing (Mac)
# --------------------------------------------------------------------------------------------

LOCAL = os.path.join(DATA_DIR, "..", "fig4_noreturn")
C4_DIR = os.path.join(LOCAL, "c4_floor")             # c4_gradient_mechanism + c4_snapshot_drift output
TRACE_DIR = os.path.join(LOCAL, "traces")            # collect() output
SNAP_LOCAL = os.path.join(LOCAL, "snap")             # collect()'s snapshot links, copied (rsync -L)
ACT_DIR = os.path.join(LOCAL, "activity")            # activity() output
C4B_DIR = os.path.join(LOCAL, "out")                 # c4b_convergence.py output (14 Oct)
GRADPART_DIR = os.path.join(LOCAL, "gradpart")       # gradient_participation.py, job 15064275
GRADSENS_DIR = os.path.join(LOCAL, "gradsens")       # gradient_sensitivity.py, job 15066631
TANH_TRACES = os.path.join(LOCAL, "tanh_traces")     # CDDM_std_g0_tanh participation traces
B4_JSON = os.path.join(DATA_DIR, "..", "remedy_reads_31oct", "read_b2_b4.json")
B4_VERDICT_4000 = os.path.join(LOCAL, "b4_verdict_4000.json")

# Class colours: active as in Figure 1b; inactive units split into low-active and silent.
C_ACT, C_LOW, C_SIL = ps.SLOTS[1], ps.SLOTS[4], ps.FAINT
# B4 sources: the old and young networks are trained 3BFF networks (3BFF blue, two tints).
C_OLD, C_YOUNG, C_FRESH = ps.SLOTS[0], "#8db6ea", ps.BASE
TASK_COL = {"3BFF": ps.SLOTS[0], "CDDM": ps.SLOTS[1]}
BARS = np.geomspace(0.005, 0.5, 25)                  # ED6b: the reach bar as a fraction of q95

# Panel letters in a bold face. matplotlib registers only the regular face of the Mac's Helvetica
# (dead-relu-paper#15), so a bold letter needs a family with a registered bold file.
ARIAL_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
if os.path.exists(ARIAL_BOLD):
    font_manager.fontManager.addfont(ARIAL_BOLD)
LETTER_FAMILY = "Arial" if os.path.exists(ARIAL_BOLD) else "DejaVu Sans"


def letter(ax, s, dx=-0.085, dy=1.045):
    """A bold panel letter (paperstyle.panel_letter in a family that has a bold face).

    Args:
        ax: axes; s: the letter; dx, dy: position in axes coordinates.
    Returns:
        the Text artist.
    """
    t = ps.panel_letter(ax, s, dx=dx, dy=dy)
    t.set_fontfamily(LETTER_FAMILY)
    return t


def load_c4():
    """Per C4 network, the classes and per-unit quantities read by panels a-c and ED6a-b.

    Returns:
        list of dicts (one per network, CDDM first) with task, seed, t (floor snapshot iteration),
        masks act / low / sil / reach, on-share (share of reference-batch samples with x > 0), g
        (g_in over the network's q95 of g_in), p, the drift arrays act_500 / act_5000 (units active
        500 and 5,000 iterations later; reference batch, 256 trials) and the trace file, if present.
    """
    out = []
    for task, _, seeds in C4_RUNS:
        for seed in seeds:
            z = np.load(glob.glob(os.path.join(C4_DIR, f"*__s{seed}__floor.npz"))[0])
            act = active_mask(z["ref_p"])
            n_on = z["ref_n_on"]
            sil = ~act & (n_on == 0)
            gi = z["strict_g_in"]
            r = dict(task=task, seed=seed, t=int(z["read_iter"]), act=act, sil=sil, low=~act & ~sil,
                     reach=active_mask(gi), on=n_on / (int(z["ref_T"]) * int(z["ref_B"])),
                     g=gi / np.quantile(gi, 0.95), g_raw=gi, g_torch=z["prod_g_in"], p=z["ref_p"],
                     n_on=n_on, on_trials=z["ref_n_on_trials"], B=int(z["ref_B"]),
                     near_input=z["ref_n_on_near_input"])
            fa = os.path.join(ACT_DIR, f"{task}_s{seed}__act.npz")
            if os.path.exists(fa):
                za = np.load(fa)
                # the d0 pass repeats c4_gradient_mechanism's reference pass: same classes expected
                r["act_0"] = active_mask(za["p_0"])
                r["act_500"], r["act_5000"] = active_mask(za["p_500"]), active_mask(za["p_5000"])
                r["p_check"] = float(np.max(np.abs(za["p_0"] - z["ref_p"]) / np.maximum(z["ref_p"], 1e-12)))
            ft = os.path.join(TRACE_DIR, f"{task}_s{seed}__trace.npz")
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
    print("\n== C4 at the current floor (classes on the noise-free reference batch)")
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
        for dt in (500, 5000):
            k = f"act_{dt}"
            if k in r:
                for name, m in (("active", r["act"]), ("low_reached", low_r),
                                ("low_unreached", r["low"] & ~r["reach"]), ("silent", r["sil"])):
                    s[f"{name}_{dt}"] = float(r[k][m].mean()) if m.any() else float("nan")
        if "act_0" in r:
            s["class_check"] = int((r["act_0"] != r["act"]).sum())
            print(f"  {r['task']} s{r['seed']}: full-batch re-run at t* differs from the C4 classes in "
                  f"{s['class_check']} units (largest relative p difference {r['p_check']:.1e})")
        summ.setdefault(r["task"], []).append(s)
        print(f"  {r['task']} s{r['seed']} t*={r['t']}: active {s['active']}, silent {s['silent']} "
              f"(largest g_in {s['sil_gmax']:.1e}; autograd {s['sil_gmax_torch']:.1e}), low-active {s['low']} of which reached {s['low_reached']}; "
              f"reached low-active: median on-share {s['on_med_low_reached']:.3f}, on in a median "
              f"{s['trials_med_low_reached']:.0f} of {s['B']} trials, {s['near_input_share']:.2f} of their "
              f"on-steps within 2 tau of an input; largest on-share of any low-active unit {s['on_max_low']:.3f}")
        if "active_5000" in s:
            print("     share active at t*+500 / t*+5000: " + ", ".join(
                f"{n} {s[n + '_500']:.3f}/{s[n + '_5000']:.3f}" for n in
                ("active", "low_reached", "low_unreached", "silent")))
    ff, cd = summ["3BFF"], summ["CDDM"]
    rule = [("every 3BFF network has 0 silent units", all(s["silent"] == 0 for s in ff)),
            ("every CDDM network has >= 200 silent units with largest g_in exactly 0",
             all(s["silent"] >= 200 and s["sil_gmax"] == 0 for s in cd)),
            ("every 3BFF network has >= 100 reached low-active units", all(s["low_reached"] >= 100 for s in ff)),
            ("in every 3BFF network < 10% of reached low-active units are active at t*+5000",
             all(s.get("low_reached_5000", np.nan) < 0.10 for s in ff))]
    print("  RULE (notebook 2026-10-31_2137): " + "; ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in rule))
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


def pick_trace_units(r):
    """Which units and trial panel a draws for one network.

    The active unit is the one with the median participation among active units; the low-active
    unit is the reached low-active unit whose on-share is nearest the class median (the first such
    unit that is on in one of the stored trials); the silent unit (CDDM) is the middle one by index.
    The trial is, among the stored trials on which the low-active unit is on, the one on which the
    active unit is on for the most steps.

    Args:
        r: one load_c4() dict with a trace file.
    Returns:
        (x (T, N) of that trial, u (n_in, T), dict class -> unit index, trial index).
    """
    z = np.load(r["trace"])
    x = z["x"]
    pick = {}
    a = np.where(r["act"])[0]
    pick["active"] = a[np.argsort(r["p"][a])[len(a) // 2]]
    lr = np.where(r["low"] & r["reach"])[0]
    target = np.median(r["on"][lr])
    order = lr[np.argsort(np.abs(r["on"][lr] - target))]
    trial = None
    for i in order:
        on_trials = np.where((x[:, i, :] > 0).any(0))[0]
        if len(on_trials):
            act_on = (x[:, pick["active"], on_trials] > 0).sum(0)
            pick["low-active"], trial = i, int(on_trials[np.argmax(act_on)])
            break
    if r["sil"].any():
        s = np.where(r["sil"])[0]
        pick["silent"] = s[len(s) // 2]
    return x[:, :, trial], z["u"][:, :, trial], pick, trial


def draw_traces(fig, gs_cell, nets):
    """Panel a: x_i(t) of a silent, a low-active and an active unit, CDDM and 3BFF.

    Args:
        fig: figure; gs_cell: the GridSpec cell the panel fills; nets: load_c4() output.
    Returns:
        the top-left axes (for the panel letter).
    """
    sub = gs_cell.subgridspec(3, 2, hspace=0.25, wspace=0.12)
    first = None
    for col, task in enumerate(("CDDM", "3BFF")):
        rs = [r for r in nets if r["task"] == task and r["trace"]]
        r = sorted(rs, key=lambda q: (q["low"] & q["reach"]).sum())[len(rs) // 2]
        x, u, pick, trial = pick_trace_units(r)
        t = np.arange(x.shape[0]) / 10.0                       # steps of 0.1 tau
        # input pulses: channels that are on for under 90% of the trial (CDDM's context cue is on
        # throughout and is left out)
        on_ch = np.abs(u) > 0
        inp = on_ch[on_ch.mean(1) < 0.9].any(0)
        for row, cls, c in ((0, "silent", C_SIL), (1, "low-active", C_LOW), (2, "active", C_ACT)):
            ax = fig.add_subplot(sub[row, col])
            first = first or ax
            ax.axhline(0, color=ps.MUTED, lw=0.5, ls=":")
            if cls in pick:
                xi = x[:, pick[cls]]
                ax.plot(t, xi, color=c if cls != "silent" else ps.MUTED, lw=0.9)
                ax.fill_between(t, 0, np.maximum(xi, 0), color=c, alpha=0.5, lw=0)
                lim = np.abs(xi).max() * 1.15
                ax.set_ylim(-lim, lim)
                print(f"  panel a {task} s{r['seed']} trial {trial}: {cls} unit {pick[cls]}, "
                      f"x range {xi.min():+.3f} to {xi.max():+.3f}, on {int((xi > 0).sum())} of {len(xi)} steps")
            else:
                ax.set_ylim(-1, 1)
                ax.text(0.5, 0.5, "no silent unit\n(0 of 1000 in each of 3 networks)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=6, color=ps.MUTED)
                ax.set_yticks([])
            # input on: a tick strip along the top of each trace
            ax.fill_between(t, ax.get_ylim()[1] * 0.93, ax.get_ylim()[1], where=inp, color=ps.FAINT,
                            lw=0, step="mid")
            ax.tick_params(labelsize=5.5, length=1.8)
            ax.set_xlim(0, t[-1])
            if row < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"time ($\tau$)", labelpad=1)
            if col == 0:
                ax.set_ylabel(cls, fontsize=6.5, color=c if cls != "silent" else ps.MUTED, labelpad=1)
            if row == 0:
                ax.set_title(f"{task}, $N$ = 1000, at its loss floor", fontsize=6.8, pad=3)
    return first


def draw_gradient(fig, gs_cell, nets):
    """Panel b: g_in against the share of samples with x > 0, one network per task.

    Silent units have both quantities exactly 0; they are drawn at the axes' "0" ticks.

    Args:
        fig: figure; gs_cell: GridSpec cell; nets: load_c4() output.
    Returns:
        the left axes.
    """
    sub = gs_cell.subgridspec(1, 2, wspace=0.12)
    XZ, YZ = 5e-6, 1e-8          # where the exact zeros of silent units are drawn (below every nonzero value)
    first = None
    for col, task in enumerate(("CDDM", "3BFF")):
        rs = [r for r in nets if r["task"] == task]
        r = sorted(rs, key=lambda q: (q["low"] & q["reach"]).sum())[len(rs) // 2]
        ax = fig.add_subplot(sub[0, col])
        first = first or ax
        rng = np.random.default_rng(0)
        for m, c, lab in ((r["sil"], C_SIL, "silent"), (r["low"], C_LOW, "low-active"),
                          (r["act"], C_ACT, "active")):
            if not m.any():
                continue
            xs = np.where(r["on"][m] > 0, r["on"][m], XZ * np.exp(rng.normal(0, 0.25, m.sum())))
            ys = np.where(r["g"][m] > 0, r["g"][m], YZ * np.exp(rng.normal(0, 0.25, m.sum())))
            ax.scatter(xs, ys, s=2.2, c=c, lw=0, alpha=0.8, rasterized=False,
                       label=f"{lab} ({m.sum()})")
        ax.axhline(SILENT_REL, color=ps.INK, lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(XZ / 2.5, 1.5)
        ax.set_ylim(YZ / 3, 30)
        ax.set_xticks([XZ, 1e-4, 1e-2, 1])
        ax.set_xticklabels(["0", "$10^{-4}$", "$10^{-2}$", "1"])
        ax.set_yticks([YZ, 1e-6, 1e-4, 1e-2, 1])
        ax.set_yticklabels(["0", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "1"] if col == 0 else [])
        ax.minorticks_off()
        ax.tick_params(labelsize=5.8)
        ax.set_xlabel("share of steps with $x_i > 0$", labelpad=1)
        if col == 0:
            ax.set_ylabel("incoming-weight gradient\n(network's 95th percentile = 1)", labelpad=1)
        ax.set_title(f"{task}", fontsize=6.8, pad=3)
        ax.legend(loc="upper left", fontsize=5.6, markerscale=2.5, handletextpad=0.2, borderaxespad=0.2)
        print(f"  panel b {task} s{r['seed']}: {r['act'].sum()} active, {r['low'].sum()} low-active "
              f"({(r['low'] & r['reach']).sum()} reached), {r['sil'].sum()} silent")
    return first


def draw_drift(ax, nets, summ):
    """Panel c: share of each class active 500 and 5,000 iterations after the floor snapshot.

    Args:
        ax: axes; nets: load_c4() output; summ: report_c4() output.
    """
    groups = [("CDDM", "active", C_ACT), ("CDDM", "low_reached", C_LOW), ("CDDM", "low_unreached", C_LOW),
              ("CDDM", "silent", C_SIL), ("3BFF", "active", C_ACT), ("3BFF", "low_reached", C_LOW),
              ("3BFF", "low_unreached", C_LOW)]
    xs = np.array([0, 1, 2, 3, 4.8, 5.8, 6.8])
    rng = np.random.default_rng(1)
    for x, (task, cls, c) in zip(xs, groups):
        for dt, dx, filled in ((500, -0.17, False), (5000, 0.17, True)):
            v = np.array([100 * s[f"{cls}_{dt}"] for s in summ[task]])
            ax.plot(x + dx + rng.normal(0, 0.03, len(v)), v, "o", ms=2.6, mec=c, mew=0.7,
                    mfc=c if filled else "none", clip_on=False, zorder=3)
            ax.plot([x + dx - 0.12, x + dx + 0.12], [v.mean()] * 2, color=c, lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["active", "reached", "not\nreached", "silent", "active", "reached", "not\nreached"],
                       fontsize=5.6)
    for x0, x1 in ((1, 2), (5.8, 6.8)):                 # the two low-active groups share one label
        ax.annotate("", xy=(x0 - 0.3, -0.2), xytext=(x1 + 0.3, -0.2), xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=C_LOW))
        ax.annotate("low-active", xy=((x0 + x1) / 2, -0.23), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=5.6, color=C_LOW)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, 7.4)
    ax.set_ylabel("% of the class active later", labelpad=1)
    for x0, x1, task in ((0, 3, "CDDM"), (4.8, 6.8, "3BFF")):
        ax.text((x0 + x1) / 2, 104, task, ha="center", va="bottom", fontsize=6.6)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mec=ps.INK, mfc="none", ms=3, label="+500 iterations"),
                       Line2D([], [], marker="o", ls="", mec=ps.INK, mfc=ps.INK, ms=3, label="+5,000 iterations")],
              loc="center right", fontsize=5.6, handletextpad=0.2, bbox_to_anchor=(1.0, 0.62))
    ps.ygrid(ax)


def draw_tanh_hist(fig, gs_cell, tanh):
    """Panel d: per-unit state gradient per unit of error, trained against untrained, ReLU and tanh.

    One network per activation, the one whose count above the bar is the median of nine.

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
        hi = v >= SILENT_REL
        ax.hist(v0, bins=bins, histtype="step", color=ps.INK, lw=0.7, label="untrained")
        ax.hist(v[~hi], bins=bins, color=C_SIL, lw=0, label="trained, below the bar")
        ax.hist(v[hi], bins=bins, color=C_ACT, lw=0, label="trained, above")
        ax.axvline(SILENT_REL, color=ps.INK, lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(FLOOR * 0.7, 40)
        ax.set_xticks([1e-5, 1e-3, 1e-1, 10])
        ax.set_xticklabels(["$\\leq10^{-5}$", "$10^{-3}$", "$10^{-1}$", "10"])
        ax.minorticks_off()
        ax.tick_params(labelsize=5.8)
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.55)
        ax.set_title(f"{act} ({len(runs)} networks: {min(n)}–{max(n)} units above)", fontsize=6.4, pad=3)
        if act == "ReLU":
            ax.legend(loc="upper left", fontsize=5.4, handlelength=1.0, borderaxespad=0.2)
        else:
            ax.text(0.03, 0.96, f"median unit\n{np.median(drop):.0f}$\\times$ lower than\nuntrained "
                    f"({min(drop):.0f}–{max(drop):.0f}$\\times$\nacross networks)", transform=ax.transAxes,
                    ha="left", va="top", fontsize=5.4, linespacing=1.15)
        ax.set_xlabel("loss gradient on $x_i$ per unit of error\n(untrained 95th percentile = 1)", labelpad=1)
        if col == 0:
            ax.set_ylabel("units", labelpad=1)
        print(f"  panel d {act}: drawn seed {r['seed']}, {hi.sum()} above the bar; nine-network counts {sorted(int(k) for k in n)}")
    return first


def draw_b4_revived(ax, b4):
    """Panel e: share of S0 (units inactive at transfer) active at the new task's floor.

    Args:
        ax: axes; b4: load_b4() output.
    """
    xs = {(1000, "old"): 0, (1000, "young"): 1, (4000, "old"): 2.6, (4000, "young"): 3.6}
    rng = np.random.default_rng(2)
    labels = []
    for (N, src), x in xs.items():
        v = np.array([100 * r["revived"] / r["n_S0"] for r in b4[N][src]])
        c = C_OLD if src == "old" else C_YOUNG
        ps.strip(ax, [x], [v], [c], width=0.22, jitter=0.04, rng=rng, ms=3.0)
        s0 = [r["n_S0"] for r in b4[N][src]]
        labels.append(f"{src}\n{min(s0):,}–{max(s0):,}")
    ax.axhline(10, color=ps.INK, lw=0.7, ls="--")
    ax.axhline(30, color=ps.INK, lw=0.7, ls=":")
    ax.text(4.2, 10.8, "under 10% in every old\nnetwork: do not come back", fontsize=5.4, ha="right", va="bottom")
    ax.text(4.2, 30.8, "30% in 2 of 3: can be recruited", fontsize=5.4, ha="right", va="bottom")
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels(labels, fontsize=5.6)
    ax.annotate("$S_0$", xy=(-0.02, -0.105), xycoords="axes fraction", ha="right", va="center",
                fontsize=5.6, color=ps.MUTED)
    for x, N in ((0.5, 1000), (3.1, 4000)):
        ax.annotate(f"$N$ = {N:,}", xy=(x, -0.3), xycoords=("data", "axes fraction"), ha="center",
                    va="top", fontsize=6.4)
    ax.set_xlim(-0.6, 4.3)
    ax.set_ylim(0, 40)
    ax.set_ylabel("% of $S_0$ active at\nthe new task's floor", labelpad=1)
    ps.ygrid(ax)


def draw_b4_active(ax, b4):
    """Panel f: active units at the new task's floor, old / young / fresh, with the rule's bars.

    Args:
        ax: axes; b4: load_b4() output.
    """
    rng = np.random.default_rng(3)
    ticks, labels = [], []
    for N, x0 in ((1000, 0), (4000, 3.8)):
        g = b4[N]
        for j, (src, c) in enumerate((("old", C_OLD), ("young", C_YOUNG), ("fresh", C_FRESH))):
            x = x0 + j
            v = np.array([r["active_star"] for r in g[src]], float)
            ps.strip(ax, [x], [v], [c], width=0.22, jitter=0.04, rng=rng, ms=3.0)
            if src == "old":
                v0 = np.array([r["active0"] for r in g[src]], float)
                ax.plot(x - 0.42 + rng.normal(0, 0.02, 3), v0, "D", ms=2.4, mfc="none", mec=c, mew=0.7,
                        zorder=3)
            r2 = [r["r2"] for r in g[src]]
            ticks.append(x)
            # two decimals: every group reads 0.91 (the exact ranges are printed and in the text)
            labels.append(f"{src}\n{min(r2):.2f}" + (f"–{max(r2):.2f}" if f"{max(r2):.2f}" != f"{min(r2):.2f}" else ""))
            print(f"  panel f N={N} {src}: active at the floor {sorted(v.astype(int).tolist())}, R2 {min(r2):.3f}-{max(r2):.3f}")
        bar = g["fresh_mean"] - 3 * g["fresh_sd"]
        ax.plot([x0 - 0.65, x0 + 0.35], [bar] * 2, color=ps.INK, lw=0.8, ls="--")
        ax.annotate(f"$N$ = {N:,}", xy=(x0 + 1, -0.3), xycoords=("data", "axes fraction"), ha="center",
                    va="top", fontsize=6.4)
        ax.text(x0 + 1, 1010, "verdict: " + (g["verdict"] if g["gate"] else "none (gate fails)"),
                ha="center", va="bottom", fontsize=5.6, style="italic")
    ax.annotate("$R^2$", xy=(-0.02, -0.105), xycoords="axes fraction", ha="right", va="center",
                fontsize=5.6, color=ps.MUTED)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=5.4)
    ax.set_xlim(-0.8, 6.3)
    ax.set_ylim(0, 1000)
    ax.set_ylabel("active units at the\nnew task's floor", labelpad=1)
    ax.legend(handles=[Line2D([], [], marker="D", ls="", mfc="none", mec=C_OLD, ms=3, label="old, at transfer"),
                       Line2D([], [], ls="--", color=ps.INK, lw=0.8, label="fresh mean $-$ 3 SD")],
              loc="upper left", fontsize=5.6, handletextpad=0.3, borderaxespad=0.1)
    ps.ygrid(ax)


def check_layout(fig, name):
    """Print any pair of overlapping text artists (tight bounding boxes, display coordinates).

    Args:
        fig: a drawn figure; name: label for the printout.
    Returns:
        the number of overlapping pairs.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text().strip()]
    # tick labels of ticks outside an axis's view are created but never drawn: leave them out
    hidden = set()
    for ax in fig.axes:
        for lo_hi, labels, locs in ((sorted(ax.get_xlim()), ax.get_xticklabels(), ax.get_xticks()),
                                    (sorted(ax.get_ylim()), ax.get_yticklabels(), ax.get_yticks())):
            for lab, v in zip(labels, locs):
                if not lo_hi[0] - 1e-12 <= v <= lo_hi[1] + 1e-12:
                    hidden.add(id(lab))
    texts = [t for t in texts if id(t) not in hidden]
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    n = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.overlaps(b) and a.width > 0 and b.width > 0:
                ov = (min(a.x1, b.x1) - max(a.x0, b.x0)) * (min(a.y1, b.y1) - max(a.y0, b.y0))
                if ov > 2.0:
                    n += 1
                    print(f"  OVERLAP in {name}: {boxes[i][0].get_text()[:30]!r} / {boxes[j][0].get_text()[:30]!r}")
    W, H = fig.get_size_inches() * fig.dpi
    for t, bb in boxes:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            print(f"  OUTSIDE in {name}: {t.get_text()[:30]!r}")
    print(f"  layout {name}: {n} overlapping text pairs")
    return n


def figure4(nets, summ, tanh, b4):
    """Draw Figure 4 (panels a-f in the module docstring).

    Args:
        nets, summ: load_c4(), report_c4(); tanh: load_tanh(); b4: load_b4().
    Returns:
        the PDF path.
    """
    ps.setup()
    fig = plt.figure(figsize=(180 * ps.MM, 168 * ps.MM))
    r1 = fig.add_gridspec(1, 2, left=0.075, right=0.99, top=0.965, bottom=0.715, width_ratios=[1.15, 1],
                          wspace=0.2)
    r2 = fig.add_gridspec(1, 2, left=0.075, right=0.99, top=0.60, bottom=0.39, width_ratios=[1, 1.15],
                          wspace=0.2)
    r3 = fig.add_gridspec(1, 2, left=0.075, right=0.94, top=0.275, bottom=0.085, width_ratios=[1, 1.3],
                          wspace=0.24)
    letter(draw_traces(fig, r1[0, 0], nets), "a", dx=-0.3, dy=1.12)
    letter(draw_gradient(fig, r1[0, 1], nets), "b", dx=-0.32, dy=1.04)
    ax = fig.add_subplot(r2[0, 0])
    draw_drift(ax, nets, summ)
    letter(ax, "c", dx=-0.14, dy=1.06)
    letter(draw_tanh_hist(fig, r2[0, 1], tanh), "d", dx=-0.22, dy=1.06)
    ax = fig.add_subplot(r3[0, 0])
    draw_b4_revived(ax, b4)
    letter(ax, "e", dx=-0.2, dy=1.04)
    ax = fig.add_subplot(r3[0, 1])
    draw_b4_active(ax, b4)
    letter(ax, "f", dx=-0.16, dy=1.04)
    check_layout(fig, "fig_paper_F4_noreturn")
    return ps.save(fig, "fig_paper_F4_noreturn", w_mm=180)


def ed_mechanism(nets):
    """Extended Data Fig. 6: gradient reach on all six networks and the C4b convergence checks.

    a  g_in by class, every network (histograms, one row per task); b reached / active against the
    bar; c low-active units reached at each read point and after the ten-fold learning-rate cut;
    d batch-consistent share of the gradient (F_net); e noisy training loss over its final value.

    Args:
        nets: load_c4() output.
    Returns:
        the PDF path.
    """
    ps.setup()
    fig = plt.figure(figsize=(180 * ps.MM, 112 * ps.MM))
    top = fig.add_gridspec(1, 3, left=0.07, right=0.99, top=0.95, bottom=0.6, wspace=0.32,
                           width_ratios=[1, 1, 1])
    bot = fig.add_gridspec(1, 3, left=0.07, right=0.99, top=0.42, bottom=0.1, wspace=0.32)
    # a: g_in histograms, both tasks, all networks pooled per class
    bins = np.logspace(-7, 1.5, 45)
    for col, task in enumerate(("CDDM", "3BFF")):
        ax = fig.add_subplot(top[0, col])
        rs = [r for r in nets if r["task"] == task]
        for key, c, lab in (("sil", C_SIL, "silent"), ("low", C_LOW, "low-active"), ("act", C_ACT, "active")):
            v = np.concatenate([r["g"][r[key]] for r in rs])
            if not len(v):
                continue
            ax.hist(np.maximum(v, 1.5e-7), bins=bins, color=c, alpha=0.75, lw=0,
                    label=f"{lab} ({len(v)}, 3 networks)")
        ax.axvline(SILENT_REL, color=ps.INK, lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([1e-7, 1e-5, 1e-3, 1e-1, 10])
        ax.set_xticklabels(["$\\leq10^{-7}$", "$10^{-5}$", "$10^{-3}$", "$10^{-1}$", "10"])
        ax.minorticks_off()
        ax.set_xlabel("incoming-weight gradient\n(network's 95th percentile = 1)", labelpad=1)
        ax.set_ylabel("units" if col == 0 else "")
        ax.set_title(f"{task}, $N$ = 1000, at the loss floor", fontsize=6.8, pad=3)
        ax.legend(loc="upper left", fontsize=5.6)
        n_sil = int(sum(r["sil"].sum() for r in rs))
        if n_sil:
            # the silent units all sit in the first bin; cut that bar so the other classes stay readable
            ax.set_ylim(0, 400)
            ax.text(2.6e-7, 200, "silent bar\ncut at 400", fontsize=5.4, color=ps.MUTED, ha="left", va="center")
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        if col == 0:
            letter(ax, "a", dx=-0.2, dy=1.05)
    # b: reached / active against the bar
    ax = fig.add_subplot(top[0, 2])
    for r in nets:
        cnt = np.array([active_mask(r["g_raw"], b).sum() for b in BARS])
        ax.plot(BARS * 100, cnt / r["act"].sum(), color=TASK_COL[r["task"]], lw=0.9,
                label=r["task"] if r["seed"] in (1423687489, 1258717025) else None)
    ax.axvline(SILENT_REL * 100, color=ps.INK, lw=0.7, ls="--")
    ax.axhline(1, color=ps.MUTED, lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("reach bar (% of the 95th percentile)", labelpad=1)
    ax.set_ylabel("units reached / active units", labelpad=1)
    ax.legend(loc="upper right", fontsize=5.8)
    ps.ygrid(ax)
    letter(ax, "b", dx=-0.2, dy=1.05)
    # c, d: C4b read points
    grads = {}
    for f in sorted(glob.glob(os.path.join(C4B_DIR, "grad_*.npz"))):
        _, name, it = os.path.basename(f)[:-4].split("_")
        if name.startswith("DMTS"):
            continue
        grads[(name, int(it))] = dict(np.load(f))
    ax_c, ax_d = fig.add_subplot(bot[0, 0]), fig.add_subplot(bot[0, 1])
    print("\n== ED6 (C4b, 14 Oct): low-active units reached, F_net, per read point")
    for name in sorted({k[0] for k in grads}):
        task = "3BFF" if name.startswith("FF") else "CDDM"
        its = sorted(k[1] for k in grads if k[0] == name)
        reach = [int(grads[(name, i)]["cls_silent_reached"].sum()) for i in its]
        fnet = [float(grads[(name, i)]["F_net"]) for i in its]
        ax_c.plot(np.array(its) / 1000, reach, "o-", color=TASK_COL[task], ms=2.6, lw=0.8)
        ax_d.plot(np.array(its) / 1000, fnet, "o-", color=TASK_COL[task], ms=2.6, lw=0.8)
        line = f"  {name}: reached " + ", ".join(f"{i // 1000}k {n}" for i, n in zip(its, reach))
        fa = glob.glob(os.path.join(C4B_DIR, f"ft_{name}_*_anneal.npz"))
        if fa:
            z = np.load(fa[0])
            it_end = int(os.path.basename(fa[0]).split("_")[2])
            n_after = int(z["cls_silent_reached"].sum())
            ax_c.plot([it_end / 1000 + 6], [n_after], "s", mfc="none", mec=TASK_COL[task], ms=3, mew=0.8)
            line += f"; after lr/10 for 300 iterations {n_after}"
        print(line + "; F_net " + ", ".join(f"{f:.2f}" for f in fnet))
    ax_c.set_xlabel("training iteration (thousands)", labelpad=1)
    ax_c.set_ylabel("low-active units reached", labelpad=1)
    ax_c.legend(handles=[Line2D([], [], color=TASK_COL["3BFF"], marker="o", ms=2.6, lw=0.8, label="3BFF"),
                         Line2D([], [], color=TASK_COL["CDDM"], marker="o", ms=2.6, lw=0.8, label="CDDM"),
                         Line2D([], [], ls="", marker="s", mfc="none", mec=ps.INK, ms=3,
                                label="after learning rate / 10")], loc="upper center", fontsize=5.6, ncol=3,
                 handletextpad=0.3, columnspacing=0.8, borderaxespad=0.1)
    ax_c.set_ylim(0, 340)
    ps.ygrid(ax_c)
    letter(ax_c, "c", dx=-0.2, dy=1.05)
    ax_d.set_xlabel("training iteration (thousands)", labelpad=1)
    ax_d.set_ylabel("share of the per-batch gradient\ncommon to 32 batches", labelpad=1)
    ax_d.set_ylim(0, 1)
    ps.ygrid(ax_d)
    letter(ax_d, "d", dx=-0.2, dy=1.05)
    # e: noisy loss over its final value
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
        ax.plot(x5 / 1000, c5 / np.median(c5[-2:]), color=TASK_COL[task], lw=0.8)
        print(f"  {name}: noisy-loss slope " + ", ".join(
            f"{w} {r[w]['lo'] // 1000}-{r[w]['hi'] // 1000}k {r[w]['noisy']['pct100k']:+.1f}% per 100k"
            for w in ("last", "ff200") if w in r))
    ax.set_yscale("log")
    ax.set_ylim(0.95, 2)
    ax.set_yticks([1, 1.5, 2])
    ax.set_yticklabels(["1", "1.5", "2"])
    ax.minorticks_off()
    ax.set_xlabel("training iteration (thousands)", labelpad=1)
    ax.set_ylabel("noisy training loss / final value", labelpad=1)
    ps.ygrid(ax)
    letter(ax, "e", dx=-0.2, dy=1.05)
    check_layout(fig, "fig_ed_mechanism")
    return ps.save(fig, "fig_ed_mechanism", w_mm=180)


def ed_tanh(tanh, traces):
    """Extended Data Fig. 7: tanh against ReLU, CDDM, N = 1000, nine networks each.

    a  active count by the participation rule over training; b units at or above 5% of the
    untrained network's 95th percentile, four per-unit measures; c median activation slope,
    untrained against trained.

    Args:
        tanh: load_tanh() output; traces: tanh_traces() output.
    Returns:
        the PDF path.
    """
    ps.setup()
    fig = plt.figure(figsize=(180 * ps.MM, 58 * ps.MM))
    gs = fig.add_gridspec(1, 3, left=0.07, right=0.99, top=0.9, bottom=0.25, wspace=0.34,
                          width_ratios=[1, 1.25, 0.8])
    col = {"ReLU": ps.BASE, "tanh": ps.INK}       # no categorical slot: SLOTS[3] means duplication in Fig. 5
    ax = fig.add_subplot(gs[0, 0])
    for act in ("ReLU", "tanh"):
        it, cnt, _ = traces[act]
        for j, row in enumerate(cnt):
            ax.plot(np.maximum(it, 1) / 1000, row, color=col[act], lw=0.7, alpha=0.8,
                    label=f"{act} ({len(cnt)})" if j == 0 else None)
    ax.set_xlabel("training iteration (thousands)", labelpad=1)
    ax.set_ylabel("active units (of 1000)", labelpad=1)
    ax.set_ylim(0, 1050)
    ax.legend(loc="center right", fontsize=5.8)
    ps.ygrid(ax)
    letter(ax, "a", dx=-0.22, dy=1.05)
    ax = fig.add_subplot(gs[0, 1])
    keys = [("g_state", "gradient\non $x_i$"), ("g_in_trial", "gradient on\ninput weights"),
            ("s_in", "sensitivity to\ninput weights"), ("s_state", "sensitivity\nto $x_i$")]
    rng = np.random.default_rng(4)
    for j, (k, _) in enumerate(keys):
        for dx, act in ((-0.18, "ReLU"), (0.18, "tanh")):
            v = [int((r[k] >= SILENT_REL).sum()) for r in tanh[act] if k in r]
            ps.strip(ax, [j + dx], [np.array(v, float)], [col[act]], width=0.12, jitter=0.03, rng=rng, ms=2.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([lab for _, lab in keys], fontsize=5.6)
    ax.set_ylim(0, 1050)
    ax.set_ylabel("units at or above the bar", labelpad=1)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=col[a], ms=3, label=a) for a in ("ReLU", "tanh")],
              loc="upper right", fontsize=5.8)
    ps.ygrid(ax)
    letter(ax, "b", dx=-0.16, dy=1.05)
    ax = fig.add_subplot(gs[0, 2])
    for act in ("ReLU", "tanh"):
        s0 = [np.median(r["slope0"]) for r in tanh[act]]
        s1 = [np.median(r["slope"]) for r in tanh[act]]
        for a_, b_ in zip(s0, s1):
            ax.plot([0, 1], [a_, b_], "o-", color=col[act], ms=2.4, lw=0.6, alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["untrained", "trained"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("median activation slope", labelpad=1)
    ps.ygrid(ax)
    letter(ax, "c", dx=-0.3, dy=1.05)
    check_layout(fig, "fig_ed_tanh")
    return ps.save(fig, "fig_ed_tanh", w_mm=180)


def draw():
    """Read every input, print every number, and write the three figures."""
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
    sub.add_parser("draw")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.c4, a.snap, a.traces)
    elif a.cmd == "activity":
        activity(SNAP_LOCAL, ACT_DIR)
    else:
        draw()
