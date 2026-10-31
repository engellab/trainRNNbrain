#!/usr/bin/env python3
"""Read every flip-flop remedy top-up (RMtop200) at its floor: the remedies catalogue.

WHAT IS READ. Every cell under `NBitFlipFlop*_c200000/` (the remedy cells continued 75k -> 200k from
~/trainRNNbrain_redo at 709c17d, plus the controls continued the same way), and B2's rate-penalty-alone
cells (`NBitFlipFlop_remedy_grid/EqType=h_N=<N>_arm=frm0.1`) so the rate penalty sits in the same
table. Cells are discovered on disk and each one is classified from its saved config (`classify`),
never from a cell list.

CONTROLS. Figure 1's flip-flop controls at each N, the same ones B2 was read against
(read_b2_b4.B2_CTRL). Two more N = 1000 control cells on disk (NBitFlipFlop_c1_snapshots_c200000,
NBitFlipFlop_std_dropfix_c200000) are read too and reported beside them, as a check on the
control's spread.

PER RUN, at its own noisy-loss floor (common.read_run, read-out 79932ef):
  active units at the probe nearest the 1.05x crossing, and at the 1.10x and 1.20x crossings;
  held-out R^2 at the floor (heldout_r2.floor_r2, R2_DRAWS = 36 draws): measured on the kept snapshot
  at or within 1,000 iterations after the crossing, else estimated from the end of the run;
  also, for the same-size rule (16 Oct: same-size comparisons at a fixed 75,000 iterations), active
  units at the probe nearest 75k and held-out R^2 of the 75k weights (the parent run's final weights);
  or, when the run is not read, the reason (diverged, no trace, read_run's reason, trace not covering
  the crossing).

PER CELL (`summary`): runs found / read / excluded with reasons, mean active, multiplier = mean
active / mean active of the control at the same N (1.05x, 1.10x, 1.20x and 75k), and the R^2 gap to the
control. Every cell is also compared field by field with its control's config on the settings no
remedy changes (learning rate, weight decay, noise, bias, batch, task), and a difference is printed.

Usage (repository root, CPU node):
  python trainRNNbrain/experiments_and_analysis/read_remedies.py read OUT_DIR --shard k/n
  python trainRNNbrain/experiments_and_analysis/read_remedies.py summary OUT_DIR
`read` writes OUT_DIR/runs_<k>.json and its own held-out R^2 cache OUT_DIR/r2cache_<k>.json;
`summary` merges every runs_*.json into OUT_DIR/remedies_catalogue.json and prints the tables.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H
from common import DATA_DIR, FLOOR_DELTAS, load_run, read_run, warm_start_chain
from read_b2_b4 import B2_ARM, B2_CTRL, SIZES

TOPUP_GLOB = f"{DATA_DIR}/NBitFlipFlop*_c200000/*"
EXTRA_CTRL_1000 = (f"{DATA_DIR}/NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control",
                   f"{DATA_DIR}/NBitFlipFlop_std_dropfix_c200000/EqType=h_k=3_N=1000_pen=none_do=none")
FIXED_ITER = 75000           # the same-size read-out point (16 Oct rule)
# settings no remedy changes; a cell that differs from its control on any of them is flagged
SHARED_FIELDS = ("model.sigma_rec", "model.sigma_inp", "model.bias_range", "model.gamma",
                 "model.equation_type", "model.self_connections", "model.dale",
                 "model.activation_args.name", "trainer.lr", "trainer.weight_decay",
                 "trainer.lr_scale_exp", "trainer.max_grad_norm", "trainer.same_batch",
                 "task.n_inputs", "task.batch_size", "task.taskname")


def cfg_of(run_dir):
    """The saved config of one run folder (OmegaConf), or None when there is none."""
    hits = sorted(glob.glob(os.path.join(run_dir, "*_config.yaml")))
    return OmegaConf.load(hits[0]) if hits else None


def sel(cfg, key):
    """cfg value at a dotted key, as plain Python (lists and dicts converted), or None."""
    v = OmegaConf.select(cfg, key, default=None)
    return OmegaConf.to_container(v) if OmegaConf.is_config(v) else v


def classify(cfg):
    """Which remedy family a saved config belongs to.

    Args:
        cfg: a run's saved config.
    Returns:
        one of "control", "synnoise", "rescale", "duplicate", "redraw", "dropout", "frm",
        "frm+rws", "other". The rules follow f2_remedies_cache.classify, extended to the families it
        leaves out (rate penalty alone, redraw, other revive operators).
    """
    t = cfg.trainer
    sw = float(sel(cfg, "model.sigma_w") or 0.0)
    frm, rws = float(sel(cfg, "trainer.lambda_frm") or 0.0), float(sel(cfg, "trainer.lambda_rws") or 0.0)
    met, orth = float(sel(cfg, "trainer.lambda_met") or 0.0), float(sel(cfg, "trainer.lambda_orth") or 0.0)
    drop, prune = bool(sel(cfg, "trainer.dropout") or False), bool(sel(cfg, "trainer.prune_reinit") or False)
    on = [sw > 0, frm > 0 or rws > 0, drop, prune, met > 0 or orth > 0,
          bool(sel(cfg, "trainer.synaptic_scaling") or False)]
    if sum(on) > 1:
        return "other"                     # a combination; none is in the top-ups
    if sw > 0:
        return "synnoise"
    if frm > 0 and rws > 0:
        return "frm+rws"
    if frm > 0:
        return "frm"
    if drop:
        return "dropout" if str(sel(cfg, "trainer.dropout_args.dropout_kind")) == "mute" else "other"
    if prune:
        a = sel(cfg, "trainer.prune_args") or {}
        mode = str(a.get("reinit_mode", ""))
        if mode == "rescale":
            return "rescale" if str(a.get("revive_op", "rescale")) == "rescale" else "other"
        if mode == "copy":
            return "other" if (a.get("copy_iid") or a.get("copy_permute")) else "duplicate"
        if mode in ("random", "zero_out"):
            return "redraw"
        return "other"
    if any(on):
        return "other"
    return "control"


def signature(cfg, family):
    """The remedy's own settings as one string, so cells with the same setting group across N.

    Args:
        cfg: a run's saved config; family: classify(cfg).
    Returns:
        str, e.g. "sigma_w=1.0" or the sorted prune_args; "" for a control.
    """
    if family == "synnoise":
        return f"sigma_w={float(sel(cfg, 'model.sigma_w'))}"
    if family in ("frm", "frm+rws"):
        return json.dumps(dict(lambda_frm=sel(cfg, "trainer.lambda_frm"), lambda_rws=sel(cfg, "trainer.lambda_rws"),
                               frm_args=sel(cfg, "trainer.frm_args"), rws_args=sel(cfg, "trainer.rws_args")),
                          sort_keys=True)
    if family == "dropout":
        return json.dumps(sel(cfg, "trainer.dropout_args"), sort_keys=True)
    if family in ("rescale", "duplicate", "redraw", "other"):
        return json.dumps(sel(cfg, "trainer.prune_args"), sort_keys=True)
    return ""


def chain_member_ending_at(run_dir, it_end):
    """The run of a warm-start chain whose training ends at iteration `it_end`, or None.

    Args:
        run_dir: the last run of a chain; it_end: absolute iteration (iter_offset + max_iter).
    Returns:
        that run's folder.
    """
    for d in warm_start_chain(run_dir):
        c = cfg_of(d)
        if c is None:
            continue
        off = sel(c, "trainer.iter_offset") or 0
        if int(off) + int(c.trainer.max_iter) == it_end:
            return d
    return None


def nearest_probe(it_a, act, t):
    """(active count, probe iteration) at the participation probe nearest iteration t."""
    j = int(np.argmin(np.abs(np.asarray(it_a, float) - t)))
    return int(act[j]), float(it_a[j])


def read_one(run_dir, cache):
    """One run read at its floor and at 75k.

    Args:
        run_dir: a run folder; cache: an open held-out R^2 cache dict (filled in place).
    Returns:
        JSON-ready dict: name, seed, reason (None when read at the floor) and, where available,
        t_star, active (1.05x), active_d10 / active_d20 (1.10x / 1.20x), active_end, it_end, r2,
        r2_measured, snapshot, active_75k, probe_75k, r2_75k, late_rise, n_bursts.
    """
    name = os.path.basename(run_dir.rstrip("/"))
    seed = re.search(r"seed=(\d+)", name)
    rec = dict(name=name, seed=seed.group(1) if seed else None, run_dir=run_dir, reason=None)
    if name.split("_")[0] == "nan":
        rec["reason"] = "diverged (folder named nan)"
        return rec
    r = load_run(run_dir)
    if r is None:
        rec["reason"] = "no readable participation trace or loss"
        return rec
    it_a, act = np.asarray(r["it_act"], float), np.asarray(r["active"], float)
    rec.update(active_end=int(act[-1]), it_end=float(it_a[-1]), it_first_probe=float(it_a[0]))
    # same-size read-out: the probe nearest 75k and the weights saved at 75k
    if it_a[0] <= FIXED_ITER <= it_a[-1]:
        rec["active_75k"], rec["probe_75k"] = nearest_probe(it_a, act, FIXED_ITER)
    d75 = chain_member_ending_at(run_dir, FIXED_ITER)
    rec["r2_75k"] = H.heldout_r2(d75, draws=H.R2_DRAWS, cache=cache) if d75 else None
    rd = read_run(run_dir, run=r)
    rec.update(late_rise=float(rd["late_rise"]), n_bursts=len(rd["bursts"]))
    if rd["reason"] is not None:
        rec["reason"] = rd["reason"]
        return rec
    if not rd["covered"]:
        rec["reason"] = "participation trace does not cover the crossing"
        return rec
    rec.update(t_star=float(rd["t_star"]), active=int(rd["active"]), probe_iter=rd["probe_iter"])
    for d in FLOOR_DELTAS:
        t = rd["t_stars"].get(d, float("nan"))
        rec[f"t_d{int(round(100 * d))}"] = float(t)
        rec[f"active_d{int(round(100 * d))}"] = (nearest_probe(it_a, act, t)[0]
                                                 if np.isfinite(t) and it_a[0] <= t <= it_a[-1] else None)
    v, m = H.floor_r2(rd, cache)
    rec.update(r2=float(v), r2_measured=bool(m), snapshot=rd["snapshot"])
    return rec


def all_cells():
    """Every cell this read covers, as (cell path, role), sorted; role is "topup", "b2" or "control"."""
    cells = {c: "topup" for c in sorted(glob.glob(TOPUP_GLOB)) if os.path.isdir(c)}
    for N in SIZES:
        cells[B2_ARM.format(N=N)] = "b2"
        cells[B2_CTRL[N]] = "control"
    for c in EXTRA_CTRL_1000:
        cells[c] = "control_extra"
    return sorted(cells.items())


def do_read(out_dir, shard):
    """Read this shard's runs (every n-th run folder starting at k) and write runs_<k>.json."""
    k, n = shard
    runs = [(cell, role, d) for cell, role in all_cells()
            for d in sorted(glob.glob(os.path.join(cell, "*"))) if os.path.isdir(d)]
    mine = runs[k::n]
    print(f"{len(runs)} runs in {len(all_cells())} cells; shard {k}/{n} reads {len(mine)}", flush=True)
    cpath = os.path.join(out_dir, f"r2cache_{k}.json")
    cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
    out = []
    for cell, role, d in mine:
        rec = read_one(d, cache)
        rec.update(cell=os.path.relpath(cell, DATA_DIR), role=role)
        out.append(rec)
        json.dump(cache, open(cpath, "w"))
        print(f"  {rec['cell'][:80]:80s} {rec['name'][:10]} "
              + (f"EXCLUDED: {rec['reason']}" if rec["reason"] else
                 f"t* {rec['t_star']:7.0f} active {rec['active']:5d} R2 {rec['r2']:.4f}"
                 f"{'' if rec['r2_measured'] else ' (est)'}"), flush=True)
    json.dump(out, open(os.path.join(out_dir, f"runs_{k}.json"), "w"), indent=1)


def cell_info(cell):
    """Family, signature, N and the shared settings of a cell, from its first run's config."""
    d = sorted(x for x in glob.glob(os.path.join(DATA_DIR, cell, "*")) if os.path.isdir(x))[0]
    c = cfg_of(d)
    fam = classify(c)
    return dict(family=fam, signature=signature(c, fam), N=int(c.model.N),
                shared={k: sel(c, k) for k in SHARED_FIELDS})


def stats(vals):
    """Mean of the finite values in a list (None entries dropped), or nan."""
    v = np.array([x for x in vals if x is not None], float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def do_summary(out_dir):
    """Merge every runs_*.json, build the per-cell catalogue, print it and write remedies_catalogue.json."""
    runs = [r for f in sorted(glob.glob(os.path.join(out_dir, "runs_*.json"))) for r in json.load(open(f))]
    by_cell = {}
    for r in runs:
        by_cell.setdefault(r["cell"], []).append(r)
    info = {c: cell_info(c) for c in by_cell}
    ctrl = {N: os.path.relpath(B2_CTRL[N], DATA_DIR) for N in SIZES}
    rows = []
    for cell, rs in sorted(by_cell.items()):
        inf = info[cell]
        read = [r for r in rs if r["reason"] is None]
        row = dict(cell=cell, role=rs[0]["role"], **inf, n_runs=len(rs), n_read=len(read),
                   excluded=[f"{r['name'][:10]}: {r['reason']}" for r in rs if r["reason"]],
                   active=[r["active"] for r in read], r2=[r["r2"] for r in read],
                   r2_measured=[r["r2_measured"] for r in read], t_star=[r["t_star"] for r in read],
                   active_d10=[r.get("active_d10") for r in read], active_d20=[r.get("active_d20") for r in read],
                   active_75k=[r.get("active_75k") for r in rs if r.get("active_75k") is not None],
                   r2_75k=[r.get("r2_75k") for r in rs if r.get("r2_75k") is not None],
                   active_end=[r.get("active_end") for r in rs if r.get("active_end") is not None])
        c = ctrl.get(inf["N"])
        if c in by_cell:
            cr = [r for r in by_cell[c] if r["reason"] is None]
            ci = info[c]
            row["config_differs_from_control"] = {k: [inf["shared"][k], ci["shared"][k]]
                                                   for k in SHARED_FIELDS if inf["shared"][k] != ci["shared"][k]}
            for key in ("active", "active_d10", "active_d20"):
                row[f"mult_{key}"] = stats(row[key]) / stats([r.get(key) for r in cr])
            row["mult_75k"] = stats(row["active_75k"]) / stats([r.get("active_75k") for r in by_cell[c]])
            row["dr2"] = stats(row["r2"]) - stats([r["r2"] for r in cr])
            row["dr2_75k"] = stats(row["r2_75k"]) - stats([r.get("r2_75k") for r in by_cell[c]])
        rows.append(row)
    json.dump(rows, open(os.path.join(out_dir, "remedies_catalogue.json"), "w"), indent=1)

    print(f"{len(runs)} runs in {len(rows)} cells\n")
    print("CONTROLS")
    for row in rows:
        if row["role"].startswith("control"):
            print(f"  N={row['N']:<5d} {row['cell'][:75]:75s} read {row['n_read']}/{row['n_runs']}  "
                  f"active {row['active']}  R2 {np.round(row['r2'], 3).tolist()} measured {row['r2_measured']}  "
                  f"75k {row['active_75k']}  R2_75k {np.round(row['r2_75k'], 3).tolist()}")
    for fam in ("synnoise", "rescale", "duplicate", "dropout", "frm+rws", "frm", "redraw", "other"):
        sub = [r for r in rows if r["family"] == fam and not r["role"].startswith("control")]
        if not sub:
            continue
        print(f"\n{fam.upper()}")
        for row in sorted(sub, key=lambda x: (x["signature"], x["N"])):
            print(f"  N={row['N']:<5d} {row['cell'][:78]:78s} read {row['n_read']}/{row['n_runs']}  "
                  f"active {row['active']}  x{row.get('mult_active', np.nan):.2f} "
                  f"(1.10: x{row.get('mult_active_d10', np.nan):.2f}, 1.20: x{row.get('mult_active_d20', np.nan):.2f}, "
                  f"75k: x{row.get('mult_75k', np.nan):.2f})  R2 {np.round(row['r2'], 3).tolist()} "
                  f"meas {sum(row['r2_measured'])}/{len(row['r2'])}  dR2 {row.get('dr2', np.nan):+.3f} "
                  f"(75k {row.get('dr2_75k', np.nan):+.3f})")
            if row["excluded"]:
                print(f"         excluded: {row['excluded']}")
            if row.get("config_differs_from_control"):
                print(f"         CONFIG DIFFERS FROM CONTROL: {row['config_differs_from_control']}")


def main():
    """Parse the command line and run `read` or `summary`."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["read", "summary"])
    ap.add_argument("out_dir")
    ap.add_argument("--shard", default="0/1", help="k/n: read every n-th run starting at k")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.mode == "read":
        do_read(a.out_dir, tuple(int(x) for x in a.shard.split("/")))
    else:
        do_summary(a.out_dir)


if __name__ == "__main__":
    main()
