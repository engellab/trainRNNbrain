#!/usr/bin/env python3
"""Read the three cell groups no paper script has read, at N = 1,000 and 75,000 iterations (issue #6).

QUESTION. Does each of three existing interventions "work" on the task it was run on? The criterion is
the per-unit-rule criterion first written here (the matrix file holds no criterion, entry
2026-11-06_2259): at 75,000 iterations (the same-size
read-out point, docs/AGENT_GUIDE.md), the arm has more active units than every control run in all of
its seeds, AND the arm's mean held-out R^2 is within R2_TOL = 0.01 of the controls' mean. The R^2 test
is one-sided: an arm whose mean R^2 is higher than the controls' passes it.

CELLS (every run N = 1,000, the standard network: ReLU, bias fixed at 0, weight decay 1e-6, Adam,
sigma_rec = sigma_inp = 0.05; each arm's saved config is compared with its control's on the settings
no intervention changes, read_remedies.SHARED_FIELDS, and any difference is printed):
  1. 3BFF input-weight cap 0.1, 0.5, 2.0 (|W_inp| clamped after every optimiser step):
     NBitFlipFlop_std_inpcap/EqType=h_k=3_N=1000_cap=<c>_pen=none, 3 runs each, 150,000 iterations,
     final weights kept. Control: the input-scale ladder's default rung,
     NBitFlipFlop_std_ksweep_bias0/EqType=h_k=3_N=1000_iters=150000, 3 runs, 150,000 iterations. The
     control's weights are not on the Mac, so its held-out R^2 at 75k comes from the Figure 2 cache
     (data/fig_paper_F2_fails_cache.pkl, written on Della on 31 Oct by the same `read_fixed`); its
     active count is re-read from the participation trace on disk and checked against the cache.
  2. CDDM noisy synapses sigma_w 1.0: CDDM_synnoise_ext/EqType=h_N=1000_sw=1.0_iters=100000, 3 runs,
     100,000 iterations (Mac only).
  3. CDDM read-out dropout (kind mute, rate 0.2, beta 4) and prune-and-duplicate (reinit_mode copy,
     copy noise 1.0, rate 0.025, maturity 1,000): CDDM_paper_grid/EqType=h_N=1000_arm={mute,duplication},
     3 runs each, 100,000 iterations.
  The control for 2 and 3 is CDDM_paper_grid/EqType=h_N=1000_arm=control (3 runs, 100,000 iterations:
  the same launcher and budget as the arms, weights on the Mac). The paper's Figure 3a control,
  CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0 (5 runs, 50,000 + 25,000 iterations), is
  printed beside it from the Figure 2 cache as a second reference: its parent run (the first 50,000
  iterations) is not on the Mac, so its loss chain cannot be joined here. Its config says
  `same_batch: true` where the paper_grid cells say false; on CDDM the flag changes nothing, because
  TaskCDDM.get_batch returns the full condition grid on every call.

READ-OUT. `fig_paper_F2_fails.read_fixed` at FIXED_IT = 75,000: the active count (common.active_mask on
the stored participation trace, which the trainer measures on a noise-free pass) at the probe nearest
75,000, and the held-out R^2 (heldout_r2: the training sigma_rec and sigma_inp, sigma_w = 0, R2_DRAWS
seeded draws) measured on weights from 75,000 iterations where such weights exist, else estimated
from the final weights as 1 - (L(75k)/L_end)(1 - R^2_end) with the noisy training loss. None of these
runs kept weights at 75k, so every R^2 here is that estimate (r2_how "estimate"), except the Figure 3a
control's, which ends at 75k (r2_how "final"). The held-out R^2 of the final weights (R^2_end, the
input of the estimate) is printed beside it. For the noisy-synapse cell the stored score is one
noisy pass with sigma_w on and is not used; the size-series launcher (slurm/SilentReLU_size_series_della.slurm)
names the clean R^2 (w_noise=False: no activity noise and no synaptic noise) as a co-primary measure,
so `clean_r2` gives it for that cell and its control, on the final weights (one deterministic pass on
the held-out batch), beside the project's measure.

OUTPUT. One table per group on stdout, and data/read_uniform_cells.json: per run cell, name, t_read,
probe, active, r2, r2_how, r2_end (and r2_clean_end where computed); per arm the mean active count,
mean R^2 and the verdict with its two conditions; the controls' values; the criterion.

Usage (repository root, CPU; about 15 minutes the first time, then from the held-out cache):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/read_uniform_cells.py
  --draws 36 (noise draws per held-out R^2), --out <json>, --f2-cache <pkl>
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H
from common import DATA_DIR, runs_with_loss
from fig_paper_F2_fails import FIXED_IT, read_fixed
from read_remedies import SHARED_FIELDS, cfg_of, sel
from trainRNNbrain.training.training_utils import prepare_task_arguments
from trainRNNbrain.trainer.Trainer import Trainer
from trainRNNbrain.utils import import_any, filter_kwargs

F2_CACHE = os.path.normpath(os.path.join(DATA_DIR, "..", "fig_paper_F2_fails_cache.pkl"))
OUT = os.path.normpath(os.path.join(DATA_DIR, "..", "read_uniform_cells.json"))
R2_TOL = 0.01           # the arm's mean held-out R^2 may be at most this far below the controls' mean
# the settings an arm is allowed to differ from its control on (the intervention's own keys)
ARM_KEYS = ("model.inp_weight_cap", "model.sigma_w", "trainer.dropout", "trainer.dropout_args",
            "trainer.prune_reinit", "trainer.prune_args")
CHECK_FIELDS = SHARED_FIELDS + ARM_KEYS

CDDM_CTRL = "CDDM_paper_grid/EqType=h_N=1000_arm=control"
CDDM_CTRL_F3A = "CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0"
GROUPS = [
    dict(name="3BFF input-weight cap", task="3-bit flip-flop", budget=150000,
         control="NBitFlipFlop_std_ksweep_bias0/EqType=h_k=3_N=1000_iters=150000", control_r2_from="f2_cache",
         arms=[(f"cap {c}", f"NBitFlipFlop_std_inpcap/EqType=h_k=3_N=1000_cap={c}_pen=none") for c in ("0.1", "0.5", "2.0")]),
    dict(name="CDDM noisy synapses", task="CDDM", budget=100000,
         control=CDDM_CTRL, control_r2_from="local", second_control=CDDM_CTRL_F3A, clean=True,
         arms=[("sigma_w 1.0", "CDDM_synnoise_ext/EqType=h_N=1000_sw=1.0_iters=100000")]),
    dict(name="CDDM per-unit rules", task="CDDM", budget=100000,
         control=CDDM_CTRL, control_r2_from="local", second_control=CDDM_CTRL_F3A,
         arms=[("read-out dropout (mute 0.2, beta 4)", "CDDM_paper_grid/EqType=h_N=1000_arm=mute"),
               ("prune-and-duplicate (copy noise 1, rate 0.025)", "CDDM_paper_grid/EqType=h_N=1000_arm=duplication")]),
]


def seed_of(name):
    """The seed in a run folder's name (`...;seed=<n>;...`), as a string, or the name itself."""
    for part in name.split(";"):
        if part.startswith("seed="):
            return part[5:]
    return name


def clean_r2(net_dir):
    """Noise-free held-out R^2 of a run's final weights: one pass with w_noise=False.

    The network is rebuilt from its saved config as heldout_r2.score_net does, the parameters loaded
    from *LastParams*, and the held-out batch is heldout_r2.heldout_batch's. With w_noise=False the
    forward pass draws no activity noise and no synaptic noise, so the score is deterministic.

    Args:
        net_dir: a trained-network folder holding *LastParams* and *_config.yaml.
    Returns:
        float R^2, or nan when the folder has no saved parameters.
    """
    cfgs = sorted(glob.glob(os.path.join(net_dir, "*_config.yaml")))
    params, _ = H._params(net_dir)
    if not cfgs or params is None:
        return float("nan")
    cfg = OmegaConf.load(cfgs[0])
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    if "_target_" in tcfg:
        del tcfg._target_
    rnn = hydra.utils.instantiate(filter_kwargs(import_any(cfg.model._target_),
                                                OmegaConf.merge(OmegaConf.create(cfg.model), tcfg)))
    with torch.no_grad():
        for nm, prm in rnn.named_parameters():
            if nm in params:
                prm.copy_(torch.from_numpy(np.asarray(params[nm], dtype=np.float32)))
        if "y_init" in params and getattr(rnn, "y_init", None) is not None:
            rnn.y_init.copy_(torch.from_numpy(np.asarray(params["y_init"], dtype=np.float32)))
        bi, bt, mask = H.heldout_batch(cfg)
        return float(Trainer.r2_score(rnn(bi, w_noise=False)[1], bt, mask))


def read_cell(cell, cache, draws, clean=False):
    """Every run of a cell read at FIXED_IT with fig_paper_F2_fails.read_fixed.

    Args:
        cell: the cell's path relative to DATA_DIR; cache: an open heldout_r2 cache dict;
        draws: noise draws per held-out R^2; clean: also score the final weights noise-free.
    Returns:
        list of dicts, one per run (sorted by folder name): cell, name, seed, t_read, probe, active
        (int), r2, r2_how, r2_end (held-out R^2 of the final weights, `draws` draws), it_end (the last
        logged iteration) and, when `clean`, r2_clean_end.
    """
    out = []
    for r in runs_with_loss(os.path.join(DATA_DIR, cell)):
        f = read_fixed(r, FIXED_IT, cache)
        rec = dict(cell=cell, name=r["name"], seed=seed_of(r["name"]), t_read=f["t_read"], probe=f["probe"],
                   active=int(f["active"]), r2=float(f["r2"]), r2_how=f["r2_how"],
                   r2_end=float(H.heldout_r2(r["run_dir"], draws=draws, cache=cache)),
                   it_end=float(np.asarray(r["it_loss"])[-1]))
        if clean:
            rec["r2_clean_end"] = clean_r2(r["run_dir"])
        out.append(rec)
        print(f"    {cell[-52:]:52s} seed {rec['seed']:>10s}  active {rec['active']:4d}  "
              f"R2 {rec['r2']:.4f} ({rec['r2_how']})  R2_end {rec['r2_end']:.4f}"
              + (f"  clean_end {rec['r2_clean_end']:.4f}" if clean else ""), flush=True)
    return out


def read_cell_from_f2_cache(cell, f2):
    """A control cell's FIXED_IT read as the Figure 2 cache holds it, with the active count re-read.

    Args:
        cell: the cell's path relative to DATA_DIR; f2: the loaded fig_paper_F2_fails cache (dict
            with "ladders" and "revival", each rung holding `runs` with a `fixed` record per run).
    Returns:
        list of dicts as read_cell's (without r2_end), plus r2_source = "fig_paper_F2_fails_cache.pkl"
        and active_cache (the cache's count, which must equal the re-read one); empty when the cell is
        not in the cache.
    """
    rung = None
    for lad in f2["ladders"] + f2["revival"]:
        for rg in lad["rungs"]:
            if rg["path"].rstrip("/").endswith(cell):
                rung = rg
    if rung is None:
        return []
    local = {r["name"]: r for r in runs_with_loss(os.path.join(DATA_DIR, cell))}
    out = []
    for r in rung["runs"]:
        f = r["fixed"].item() if isinstance(r["fixed"], np.ndarray) else r["fixed"]
        rec = dict(cell=cell, name=r["name"], seed=seed_of(r["name"]), t_read=f["t_read"], probe=f["probe"],
                   active=int(f["active"]), active_cache=int(f["active"]), r2=float(f["r2"]), r2_how=f["r2_how"],
                   r2_source="fig_paper_F2_fails_cache.pkl")
        if r["name"] in local:               # re-read the count from the trace on disk
            lr = local[r["name"]]
            it_a = np.asarray(lr["it_act"], float)
            k = int(np.argmin(np.abs(it_a - f["t_read"])))
            rec["active"], rec["probe"] = int(lr["active"][k]), float(it_a[k])
            if rec["active"] != rec["active_cache"]:
                print(f"    MISMATCH {r['name'][:40]}: trace says {rec['active']}, cache {rec['active_cache']}")
        out.append(rec)
        print(f"    {cell[-52:]:52s} seed {rec['seed']:>10s}  active {rec['active']:4d}  "
              f"R2 {rec['r2']:.4f} ({rec['r2_how']}, F2 cache)", flush=True)
    return out


def config_differences(arm_cell, ctrl_cell):
    """The settings on which an arm's saved config differs from its control's.

    Args:
        arm_cell, ctrl_cell: cell paths relative to DATA_DIR; the first run folder of each is read.
    Returns:
        dict key -> [arm value, control value] over CHECK_FIELDS (the shared settings of
        read_remedies plus the interventions' own keys), empty when the configs agree on all of them.
    """
    def first_cfg(cell):
        """The saved config of a cell's first run folder."""
        dirs = sorted(d for d in glob.glob(os.path.join(DATA_DIR, cell, "*")) if os.path.isdir(d))
        return cfg_of(dirs[0])
    a, c = first_cfg(arm_cell), first_cfg(ctrl_cell)
    return {k: [sel(a, k), sel(c, k)] for k in CHECK_FIELDS if sel(a, k) != sel(c, k)}


def verdict(arm, ctrl):
    """The matrix file's criterion applied to one arm.

    Args:
        arm, ctrl: lists of run dicts (read_cell's) for the arm and for its control cell.
    Returns:
        dict: works (bool), count_ok (every arm run's active count exceeds the largest control
        count), r2_ok (mean arm R^2 >= mean control R^2 - R2_TOL), min_active_arm, max_active_ctrl,
        mean_active_arm, mean_active_ctrl, fold (ratio of the means), mean_r2_arm, mean_r2_ctrl,
        r2_gap (arm minus control), text (one line stating the verdict and its reasons).
    """
    aa, ca = [r["active"] for r in arm], [r["active"] for r in ctrl]
    ar, cr = [r["r2"] for r in arm], [r["r2"] for r in ctrl]
    v = dict(min_active_arm=min(aa), max_active_ctrl=max(ca), mean_active_arm=float(np.mean(aa)),
             mean_active_ctrl=float(np.mean(ca)), fold=float(np.mean(aa) / np.mean(ca)),
             mean_r2_arm=float(np.mean(ar)), mean_r2_ctrl=float(np.mean(cr)))
    v["r2_gap"] = v["mean_r2_arm"] - v["mean_r2_ctrl"]
    v["count_ok"] = bool(v["min_active_arm"] > v["max_active_ctrl"])
    v["r2_ok"] = bool(v["r2_gap"] >= -R2_TOL)
    v["works"] = v["count_ok"] and v["r2_ok"]
    why = []
    if not v["count_ok"]:
        low = [r["seed"] for r in arm if r["active"] <= v["max_active_ctrl"]]
        why.append(f"count not above the controls' largest ({v['max_active_ctrl']}) in seed(s) {', '.join(low)}")
    if not v["r2_ok"]:
        why.append(f"R2 gap {v['r2_gap']:+.3f} beyond {R2_TOL}")
    v["text"] = "works" if v["works"] else "fails: " + "; ".join(why)
    return v


def print_group(g, arms, ctrl, second, diffs):
    """Print one group's table: every run of every arm, the control runs, and each arm's verdict.

    Args:
        g: the GROUPS entry; arms: dict arm label -> run dicts; ctrl: the control's run dicts;
        second: the second control's run dicts (may be empty); diffs: dict arm label -> config
        differences from the control.
    """
    clean = g.get("clean", False)
    print(f"\n=== {g['name']} ({g['task']}, N = 1,000, read at {FIXED_IT:,} iterations; budget {g['budget']:,}) ===")
    head = "| cell | seed | iteration read (probe) | active units | held-out R2 at 75k (how) | R2 of final weights |"
    print(head + (" clean R2, final weights |" if clean else ""))
    print("|---|---|---|---|---|---|" + ("---|" if clean else ""))
    for label, rows in list(arms.items()) + [("control: " + os.path.basename(g["control"]), ctrl)] + \
            ([("second control (F2 cache): " + os.path.basename(g["second_control"]), second)] if second else []):
        for r in rows:
            line = (f"| {label} | {r['seed']} | {r['t_read']:,.0f} ({r['probe']:,.0f}) | {r['active']} | "
                    f"{r['r2']:.3f} ({r['r2_how']}{', F2 cache' if 'r2_source' in r else ''}) | "
                    f"{r['r2_end']:.3f} |" if "r2_end" in r else
                    f"| {label} | {r['seed']} | {r['t_read']:,.0f} ({r['probe']:,.0f}) | {r['active']} | "
                    f"{r['r2']:.3f} ({r['r2_how']}, F2 cache) | - |")
            if clean:
                line += f" {r['r2_clean_end']:.3f} |" if "r2_clean_end" in r else " - |"
            print(line)
    print("\n| arm | active (seeds) | mean | fold over control mean | mean R2 | R2 gap | verdict |")
    print("|---|---|---|---|---|---|---|")
    for label, rows in arms.items():
        v = verdict(rows, ctrl)
        print(f"| {label} | {', '.join(str(r['active']) for r in rows)} | {v['mean_active_arm']:.0f} | "
              f"{v['fold']:.2f} | {v['mean_r2_arm']:.3f} | {v['r2_gap']:+.3f} | {v['text']} |")
        if diffs[label]:
            print(f"    config differs from the control on: {diffs[label]}")
    print(f"    controls: active {', '.join(str(r['active']) for r in ctrl)} (mean {np.mean([r['active'] for r in ctrl]):.0f}), "
          f"mean R2 {np.mean([r['r2'] for r in ctrl]):.3f}")
    if second:
        print(f"    second control: active {', '.join(str(r['active']) for r in second)} "
              f"(mean {np.mean([r['active'] for r in second]):.0f}), mean R2 {np.mean([r['r2'] for r in second]):.3f}")


def main():
    """Read every group, print its tables and write the JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draws", type=int, default=H.R2_DRAWS, help="noise draws per held-out R^2")
    ap.add_argument("--out", default=OUT, help="JSON output path")
    ap.add_argument("--f2-cache", default=F2_CACHE, help="fig_paper_F2_fails cache (the 3BFF control's R^2)")
    a = ap.parse_args()
    t0 = time.time()
    cache = H.load_cache()
    with open(a.f2_cache, "rb") as fh:
        f2 = pickle.load(fh)
    result = dict(fixed_it=FIXED_IT, draws=a.draws, r2_tol=R2_TOL,
                  criterion=(f"works = every arm run has more active units at {FIXED_IT:,} iterations than the "
                             f"largest control count, and the arm's mean held-out R^2 is at least the controls' "
                             f"mean minus {R2_TOL}"),
                  f2_cache=dict(path=os.path.relpath(a.f2_cache), mtime=time.strftime(
                      "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(a.f2_cache)))),
                  groups=[])
    for g in GROUPS:
        print(f"\n## {g['name']}", flush=True)
        clean = g.get("clean", False)
        ctrl = (read_cell(g["control"], cache, a.draws, clean=clean) if g["control_r2_from"] == "local"
                else read_cell_from_f2_cache(g["control"], f2))
        H.save_cache(cache)
        second = read_cell_from_f2_cache(g["second_control"], f2) if g.get("second_control") else []
        arms, diffs, verdicts = {}, {}, {}
        for label, cell in g["arms"]:
            arms[label] = read_cell(cell, cache, a.draws, clean=clean)
            H.save_cache(cache)
            diffs[label] = config_differences(cell, g["control"])
            verdicts[label] = verdict(arms[label], ctrl)
        print_group(g, arms, ctrl, second, diffs)
        result["groups"].append(dict(name=g["name"], task=g["task"], budget=g["budget"],
                                     control=dict(cell=g["control"], r2_from=g["control_r2_from"], runs=ctrl),
                                     second_control=(dict(cell=g["second_control"], runs=second) if second else None),
                                     arms=[dict(label=label, cell=cell, runs=arms[label], verdict=verdicts[label],
                                                config_differs_from_control=diffs[label]) for label, cell in g["arms"]]))
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"\nwrote {os.path.relpath(a.out)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
