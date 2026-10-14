#!/usr/bin/env python3
"""Read the CDDM rate- and sparsity-penalty top-ups on the standard network (agent A4, 2026-10-13).

THE SWEEP. `CDDM_std_g0_bias0` trained 2 x 2 penalty cells (lambda_rws in {0, 0.05}, lambda_frm in
{0, 0.2}) at N = 1000 for 50k iterations, for equation types h and s; `CDDM_std_g0_bias0_c75000`
continues every run by 25k to 75k. lambda_frm is the firing-rate magnitude penalty (it pulls each
unit's soft-max rate toward a cap from both sides, so it also pushes silent units up); lambda_rws is
the recurrent-weight sparsity penalty (target in-degree 20). The `LmbdRWS=0_LmbdFR=0` cell is the
sweep's own no-penalty control at the same budget.

WHAT THIS SCRIPT DOES.
  1. Inventory and standardness: every run's saved config against the standard network (ReLU,
     equation h, zero fixed bias, gamma 0, no Dale, signed I/O, self-connections, weight decay 1e-6)
     and against the reference control's config, key by key.
  2. Floor and end read-outs of each 75k chain through `fig_knobs_at_floor.read_rung`, with held-out
     R^2 at 36 draws: estimated at the floor (no snapshots exist), measured at 50k (the parent's
     final weights) and at 75k.
  3. An independent check of the active counts: a numpy forward pass of each run's final weights
     (own integrator, own participation code) against the trace's last stored count, per unit.

The held-out R^2 cache is written to this agent's own file, not to data/heldout_r2_cache.json, which
other agents write concurrently and which save_cache would overwrite with a stale copy.

Usage:  python trainRNNbrain/experiments_and_analysis/a4_cddm_std_penalties.py OUT_DIR
        (repository root, CPU node, CUDA_VISIBLE_DEVICES= ; writes OUT_DIR/a4_cddm_std_penalties.json)
"""

import glob
import json
import os
import pickle
import sys

import numpy as np
import hydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H
import fig_knobs_at_floor as K
from common import DATA_DIR, SILENT_REL, participation, warm_start_chain
from trainRNNbrain.training.training_utils import prepare_task_arguments
from trainRNNbrain.utils import unjsonify

BASE, TOP = "CDDM_std_g0_bias0", "CDDM_std_g0_bias0_c75000"
REF_CTRL = os.path.join(DATA_DIR, "CDDM_std_g0_drift_bias0", "EqType=h_N=1000_iters=200000")
READ_CELLS = ["EqType=h_N=1000_LmbdRWS=0_LmbdFR=0", "EqType=h_N=1000_LmbdRWS=0_LmbdFR=0.2",
              "EqType=h_N=1000_LmbdRWS=0.05_LmbdFR=0", "EqType=h_N=1000_LmbdRWS=0.05_LmbdFR=0.2"]
END_AT = 75000          # the reference control ran 200k; its end is read at the arms' budget

# the standard network, as the task statement defines it
STANDARD = {"model.activation_args.name": "relu", "model.equation_type": "h",
            "model.bias_init": "zeros", "model.bias_range": [0.0, 0.0], "model.gamma": 0,
            "model.dale": False, "model.io_nonnegativity": False, "model.self_connections": True,
            "trainer.weight_decay": 1e-6}
# keys expected to differ between runs of a standard comparison (identity, budget, the penalty)
VARYING = ("seed", "experiment_tag", "paths.", "trainer.max_iter", "trainer.iter_offset",
           "trainer.lambda_rws", "trainer.lambda_frm", "trainer.checkpoint")


def flat_config(run_dir):
    """A run's saved config as a flat {dotted.key: value} dict.

    Args:
        run_dir: a trained-network folder holding *_config.yaml.
    Returns:
        dict of dotted key -> plain Python value.
    """
    cfg = OmegaConf.to_container(OmegaConf.load(sorted(glob.glob(os.path.join(run_dir, "*_config.yaml")))[0]))
    out = {}

    def walk(d, pre):
        """Flatten nested dict d into out under prefix pre."""
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{pre}{k}.")
            else:
                out[f"{pre}{k}"] = v
    walk(cfg, "")
    return out


def deviations(run_dir, ref):
    """Every way one run's config departs from the standard network or from the reference control.

    Args:
        run_dir: a trained-network folder; ref: flat_config of the reference control.
    Returns:
        list of "key: value (expected X)" strings, empty for a standard run.
    """
    c = flat_config(run_dir)
    bad = [f"{k}: {c.get(k)!r} (standard {v!r})" for k, v in STANDARD.items() if c.get(k) != v]
    for k in sorted(set(c) | set(ref)):
        if k in STANDARD or any(k.startswith(p) for p in VARYING):
            continue
        if c.get(k) != ref.get(k):
            bad.append(f"{k}: {c.get(k)!r} (control {ref.get(k)!r})")
    return bad


def inventory(ref):
    """Cells, runs, budgets, penalties and config deviations of both sweeps.

    Args:
        ref: flat_config of the reference control.
    Returns:
        list of dicts, one per cell.
    """
    rows = []
    for sweep in (BASE, TOP):
        for cell in sorted(glob.glob(os.path.join(DATA_DIR, sweep, "*"))):
            runs = sorted(d for d in glob.glob(os.path.join(cell, "*")) if os.path.isdir(d))
            cfgs = [flat_config(d) for d in runs]
            lam = sorted({tuple((k, v) for k, v in c.items() if k.startswith("trainer.lambda_") and v)
                          for c in cfgs})
            budget = sorted({c["trainer.max_iter"] + (c.get("trainer.iter_offset") or 0) for c in cfgs})
            parents = [warm_start_chain(d)[0] for d in runs]
            rows.append(dict(sweep=sweep, cell=os.path.basename(cell), n=len(runs), budget=budget,
                             penalties=[dict(x) for x in lam],
                             distinct_parents=len(set(parents)),
                             deviations=sorted({x for d in runs for x in deviations(d, ref)})))
    return rows


def oracle_active(run_dir):
    """Active units of a run's final weights, from an independent numpy forward pass.

    Integrates x <- x + (dt/tau)(-x + W_rec relu(x) + W_inp u + b) from y_init, noise-free, on the
    run's training grid of trials, scores p = std(r) + q0.9(|r|) pooled over time and trials, and
    applies p >= SILENT_REL * q95(p).

    Args:
        run_dir: a trained-network folder with *LastParams*.json, *_config.yaml, *ParticipationTrace.pkl.
    Returns:
        dict: n_oracle (active count), n_trace (trace's last stored count), agree (fraction of
        units classified the same way), trace_iter (iteration of that last probe).
    """
    cfg = OmegaConf.load(sorted(glob.glob(os.path.join(run_dir, "*_config.yaml")))[0])
    with open(sorted(glob.glob(os.path.join(run_dir, "*LastParams*.json")))[0]) as fh:
        P = {k: (np.asarray(v, float) if v is not None and not isinstance(v, (str, bool, dict)) else v)
             for k, v in unjsonify(json.load(fh)).items()}
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    u = np.asarray(task.get_batch()[0], float)                      # (n_inputs, T, B)
    W_rec, W_inp = P["W_rec"], P["W_inp"]
    b = np.zeros(W_rec.shape[0]) if P.get("bias") is None else np.ravel(P["bias"])
    alpha = cfg.model.dt / cfg.model.tau
    x = np.repeat(np.ravel(P["y_init"])[:, None], u.shape[2], axis=1)
    rates = [np.maximum(x, 0)]
    for t in range(1, u.shape[1]):
        x = x + alpha * (-x + W_rec @ np.maximum(x, 0) + W_inp @ u[:, t - 1, :] + b[:, None])
        rates.append(np.maximum(x, 0))
    p = participation(np.stack(rates, axis=1))                       # (N,)
    on = p >= SILENT_REL * np.quantile(p, 0.95)
    with open(sorted(glob.glob(os.path.join(run_dir, "*ParticipationTrace.pkl")))[0], "rb") as fh:
        tr = pickle.load(fh)
    pt = np.asarray(tr["participation"])[-1]
    on_t = pt >= SILENT_REL * np.quantile(pt, 0.95)
    return dict(n_oracle=int(on.sum()), n_trace=int(on_t.sum()), agree=float((on == on_t).mean()),
                trace_iter=int(np.asarray(tr["participation_iters"])[-1]))


def read_cell(path, cache, end_at=None):
    """Floor and end read-out of one cell, with held-out R^2 at 36 draws.

    Args:
        path: the cell directory; cache: an open heldout_r2 cache; end_at: see read_rung.
    Returns:
        read_rung's dict plus per-run lists: r2_end (final weights), r2_50k (the chain's first run's
        final weights, None when the cell is not a continuation), oracle (oracle_active), names.
    """
    r = K.read_rung(path, end_at, cache)
    runs = [d for d in sorted(glob.glob(os.path.join(path, "*"))) if os.path.isdir(d)
            and not os.path.basename(d).startswith("nan")]
    r["names"] = [os.path.basename(d) for d in runs]
    r["r2_end"] = [H.heldout_r2(d, draws=K.R2_DRAWS, cache=cache) for d in runs]
    chains = [warm_start_chain(d) for d in runs]
    r["r2_50k"] = [H.heldout_r2(c[0], draws=K.R2_DRAWS, cache=cache) if len(c) > 1 else None
                   for c in chains]
    r["oracle"] = [oracle_active(d) for d in runs]
    return r


def jsonable(v):
    """Numpy arrays and scalars in v made JSON-serialisable."""
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def main(out_dir):
    """Run the inventory, read every h cell and the reference control, print and save the result."""
    os.makedirs(out_dir, exist_ok=True)
    H.CACHE = os.path.join(out_dir, "heldout_r2_cache_a4.json")     # see the module docstring
    ref_run = sorted(d for d in glob.glob(os.path.join(REF_CTRL, "*")) if os.path.isdir(d))[0]
    ref = flat_config(ref_run)
    inv = inventory(ref)
    print("=== inventory")
    for row in inv:
        print(f"{row['sweep']:<28}{row['cell']:<40} n={row['n']} budget={row['budget']} "
              f"penalties={row['penalties']} parents={row['distinct_parents']}")
        for d in row["deviations"]:
            print(f"      DEVIATES  {d}")

    cache = H.load_cache()
    reads = {"reference control (drift, 200k; end R^2 and oracle on its 200k weights)": read_cell(REF_CTRL, cache, END_AT)}
    for c in READ_CELLS:
        reads[c] = read_cell(os.path.join(DATA_DIR, TOP, c), cache)
        H.save_cache(cache)
    H.save_cache(cache)

    print("\n=== read-outs (R^2: 36 draws; floor R^2 estimated from the end, no snapshots exist)")
    for lab, r in reads.items():
        print(f"\n{lab}: n={r['n']}  dropped={r['dropped']}")
        print(f"  floor active {r['floor'].astype(int).tolist()}  crossing {r['cross'].astype(int).tolist()}"
              f"  end/floor loss {np.round(r['late'], 2).tolist()}")
        print(f"  floor R^2 (est.) {np.round(r['r2'], 4).tolist()}")
        print(f"  end active {r['end'].astype(int).tolist()}  end R^2 {np.round(r['r2_end'], 4).tolist()}")
        print(f"  R^2 at 50k (parent) {[None if v is None else round(v, 4) for v in r['r2_50k']]}")
        print(f"  oracle vs trace {[(o['n_oracle'], o['n_trace'], round(o['agree'], 3)) for o in r['oracle']]}")
    with open(os.path.join(out_dir, "a4_cddm_std_penalties.json"), "w") as fh:
        json.dump(jsonable(dict(inventory=inv, reads=reads)), fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
