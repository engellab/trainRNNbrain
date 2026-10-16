#!/usr/bin/env python3
"""One figure per trained network: its noisy training loss, the read-out curve, the fitted floor and
where the network is read, with the run's path, task, size, read-out status and every hyperparameter
that differs from its task's standard control run.

WHY. The 1.05x read-out (common.noisy_floor) turns each loss trajectory into one iteration. Before
trusting a comparison built on it, every trajectory can be looked at: whether the fit follows the loss,
whether the loss still falls at the end, where bursts sit.

TWO STEPS, because the runs live on different machines and figures are drawn on the Mac (Della's fonts
differ). `extract` runs where the data is and writes one small .npz per run; `draw` turns those into PNGs.

  extract   for every run: the per-iteration noisy loss joined along its warm-start chain
            (common.noisy_loss_series), the read-out (common.noisy_floor), the run's status, the
            raw loss reduced to its 5th/50th/95th percentiles in 600 log-spaced bins, the read-out
            curve at 1,500 log-spaced iterations, and the config difference from the task's reference.
  draw      one PNG per run under OUT/<sweep>/, and OUT/index.md listing them all.

STATUS of a run (the top-up rule, notebook 2026-10-16):
  done        1.05x read-out t* exists and the run trained to >= 2 t*
  top-up      short of 2 t*; or its loss ends below its floor (noisy_floor's END_BELOW guard: still
              falling); or it has not yet stayed under the bar
  unreadable  fit pinned at its bound, loss ends > 1.5x above its floor, bursts over half of training,
              or the loss never leaves its starting level: training longer does not help

NON-STANDARD HYPERPARAMETERS are the model, trainer and task settings whose value differs from the
reference config of the run's task (REF_DIR/<taskname>.yaml, a standard control run). Bookkeeping keys
(seeds, logging, checkpoints, budget) are ignored, and so are the settings of a feature that is
switched off in both configs (frm_args when lambda_frm = 0, dropout_args when dropout is off, ...).
A key the run's config does not have (older code) is not reported.

Usage:
  python run_trajectories.py extract OUT_DIR REF_DIR RUN_DIR [RUN_DIR ...]
  python run_trajectories.py extract OUT_DIR REF_DIR --from-csv status.csv     (column `run`)
  python run_trajectories.py draw NPZ_DIR [NPZ_DIR ...] --out img/internal_figures/loss_trajectories
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
import textwrap
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C

N_BINS = 600        # log-spaced bins for the raw-loss band
N_CURVE = 1500      # log-spaced iterations at which the read-out curve is stored
IGNORE = re.compile(r"^(seed|n_nets|display_figures|experiment_tag|hydra\..*|paths\..*|task\.seed|"
                    r"task\.task_params.*|task\.mask_params.*|model\.N|model\.RNN_tag|"
                    r"trainer\.(max_iter|iter_offset|checkpoint_.*|snapshot_every|restore_after|track_.*|"
                    r"log_silent_every|store_participation_every|drift_lags.*|monitor|gnorm_.*|"
                    r"trainer_tag))$")
SWITCHES = {"trainer.frm_args": "trainer.lambda_frm", "trainer.rws_args": "trainer.lambda_rws",
            "trainer.met_args": "trainer.lambda_met", "trainer.orth_args": "trainer.lambda_orth",
            "trainer.dropout_args": "trainer.dropout", "trainer.prune_args": "trainer.prune_reinit",
            "trainer.scaling_args": "trainer.synaptic_scaling"}


def flatten(d, prefix=""):
    """Nested dict -> {dotted.key: value}; lists become tuples so they compare by value."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = tuple(v) if isinstance(v, list) else v
    return out


def config_diff(cfg, ref):
    """The run's non-standard hyperparameters (see module docstring).

    Args:
        cfg, ref: plain dicts of the run's config and its task's reference config.
    Returns:
        list of "key: reference -> run" strings, plus "warm start from <name>" when the run continues another.
    """
    a, b = flatten(cfg), flatten(ref)
    out = []
    for k, v in sorted(a.items()):
        if IGNORE.match(k) or k not in b or v == b[k]:
            continue
        switch = next((s for p, s in SWITCHES.items() if k.startswith(p + ".")), None)
        if switch and not a.get(switch) and not b.get(switch):
            continue
        out.append(f"{k}: {b[k]} -> {v}")
    for key in ("init_from", "transfer_from"):
        p = (cfg.get("paths") or {}).get(key)
        if p:
            out.append(f"{key.replace('_', ' ')}: {os.path.basename(p.rstrip('/'))}")
    return out


def chain_joins(run_dir):
    """Absolute iterations at which each continuation in the run's warm-start chain begins."""
    joins, done = [], 0
    chain = C.warm_start_chain(run_dir)
    for d in chain:
        tl = sorted(glob.glob(os.path.join(d, "*TrainLosses.json")))
        if not tl:
            break
        with open(tl[0]) as fh:
            done += len(json.load(fh).get("train_losses", []))
        joins.append(done)
    return joins[:-1]


def predicted_crossing(f, delta=0.05, t_max=1e8):
    """Iteration where the fitted curve (common.floor_curve) reaches (1+delta) x floor, or nan."""
    if f.get("fit") is None or f.get("floor_kind") != "fit":
        return float("nan")
    t = np.geomspace(1, t_max, 20000)                      # iterations since the escape
    hit = np.flatnonzero(C.floor_curve(t, f["fit"]) <= (1 + delta) * f["fit"][0])
    return float(f["escape"] + t[hit[0]]) if len(hit) else float("nan")


def status_of(f, end):
    """(status, reason, target budget) of a run under the top-up rule (module docstring)."""
    t, why = f["t_star"], f["reason"]
    if why == "loss ends below its floor":
        return "top-up", f"loss ends at {f['late_rise']:.2f}x its floor (still falling)", 2 * end
    if why is None:
        if end < 2 * t:
            return "top-up", f"trained to {end / t:.1f}x t*", 3 * t
        return "done", f"trained to {end / t:.1f}x t*", float("nan")
    if why == "never stays under the bar":
        tp = predicted_crossing(f)
        if np.isfinite(tp) and tp <= 10 * end:
            return "top-up", f"not under the bar yet; fit predicts {tp / 1e3:.0f}k", max(3 * tp, 1.5 * end)
        return "top-up", "not under the bar yet; no usable prediction", 2 * end
    return "unreadable", why, float("nan")


def extract_one(args):
    """Reduce one run to the npz that `draw` needs; returns the npz path or None.

    Args:
        args: (run_dir, out_dir, ref_dir).
    """
    run_dir, out_dir, ref_dir = args
    it, L = C.noisy_loss_series(run_dir)
    cfgs = sorted(glob.glob(os.path.join(run_dir, "*_config.yaml")))
    if L is None or not cfgs:
        return None
    ok = np.isfinite(L)
    it, L = it[ok], L[ok]
    cfg = OmegaConf.to_container(OmegaConf.load(cfgs[0]), resolve=False)
    task = cfg["task"]["taskname"]
    ref_path = os.path.join(ref_dir, f"{task}.yaml")
    ref = OmegaConf.to_container(OmegaConf.load(ref_path), resolve=False) if os.path.exists(ref_path) else {}
    f = C.noisy_floor(it, L, task=cfg["task"]["taskname"])
    end = float(it[-1])
    status, why, target = status_of(f, end)
    # raw loss as percentiles in log-spaced bins
    edges = np.unique(np.geomspace(it[0], end + 1, N_BINS + 1).astype(int))
    idx = np.searchsorted(it, edges)
    band = [(np.sqrt(edges[j] * max(edges[j + 1] - 1, edges[j])), *np.percentile(L[a:b], [5, 50, 95]))
            for j, (a, b) in enumerate(zip(idx[:-1], idx[1:])) if b > a]
    band = np.array(band)
    tc = np.unique(np.geomspace(it[0], end, N_CURVE).astype(int)).astype(float)
    curve = np.interp(tc, it, f["smoothed"]) if "smoothed" in f else np.full(len(tc), np.nan)
    sweep, cell, name = run_dir.rstrip("/").split("/")[-3:]
    seed = cfg.get("seed", "x")
    tag = hashlib.md5(name.encode()).hexdigest()[:6]
    meta = dict(run=run_dir, sweep=sweep, cell=cell, name=name, seed=seed, task=task,
                N=cfg["model"].get("N"), end=end, status=status, why=why, target=target,
                reason=f["reason"], floor=f["floor"], floor_kind=f["floor_kind"], fit=f.get("fit"),
                escape=f["escape"], t_stars={str(k): v for k, v in f["t_stars"].items()},
                late_rise=f["late_rise"], burst_from=f["burst_from"], bursts=f["bursts"],
                end_residual=f.get("end_residual"),
                joins=chain_joins(run_dir), diff=config_diff(cfg, ref) if ref else ["no reference config"],
                reference=os.path.basename(ref_path))
    path = os.path.join(out_dir, f"{sweep}__{cell}__s{seed}_{tag}.npz")
    np.savez(path, band=band, tc=tc, curve=curve, meta=json.dumps(meta, default=float))
    return path


def draw_one(args):
    """One PNG for one extracted run; returns (meta, png path)."""
    npz, out = args
    z = np.load(npz, allow_pickle=False)
    m = json.loads(str(z["meta"]))
    band, tc, curve = z["band"], z["tc"], z["curve"]
    folder = os.path.join(out, m["sweep"])
    os.makedirs(folder, exist_ok=True)
    png = os.path.join(folder, os.path.basename(npz).replace(f"{m['sweep']}__", "").replace(".npz", ".png"))

    fig = plt.figure(figsize=(15, 6.2))
    ax = fig.add_axes([0.07, 0.1, 0.52, 0.82])
    tx = fig.add_axes([0.635, 0.03, 0.36, 0.94]); tx.axis("off")
    ax.fill_between(band[:, 0], band[:, 1], band[:, 3], color="0.82", lw=0, label="raw loss, 5-95% per bin")
    ax.plot(band[:, 0], band[:, 2], color="0.55", lw=0.6, label="raw loss, median per bin")
    ax.plot(tc, curve, color="tab:red", lw=1.6, label="read-out curve (Hampel, bursts masked)")
    for b0, b1, _ in m["bursts"]:
        ax.axvspan(b0, b1, color="orange", alpha=0.3, lw=0)
    if m["fit"] is not None and np.isfinite(m["escape"]):
        t0 = m["escape"]
        tf = np.geomspace(max(C.FIT_START, t0 + 200), m["end"], 300)
        tx_ = np.geomspace(m["end"], 3 * m["end"], 60)
        ax.plot(tf, C.floor_curve(tf - t0, m["fit"]), color="tab:green", lw=1.2, label="fitted stretched exponential (end-weighted)")
        ax.plot(tx_, C.floor_curve(tx_ - t0, m["fit"]), color="tab:green", lw=1.2, ls=":",
                label="fit, extrapolated past the end")
    if m["floor"]:
        ax.axhline(m["floor"], color="tab:green", ls="--", lw=0.8, label=f"floor ({m['floor_kind']})")
        # the 1.05x and 1.10x bars are drawn strong and labelled on the axis; 1.20x faint
        for d, col, ls, lw, lab in (("0.05", "#0047ab", "-", 1.8, "1.05x floor"),
                                    ("0.1", "darkviolet", "--", 1.5, "1.10x floor"),
                                    ("0.2", "0.6", ":", 0.8, None)):
            y = (1 + float(d)) * m["floor"]
            ax.axhline(y, color=col, ls=ls, lw=lw, label=lab or "1.20x floor")
            t = m["t_stars"].get(d)
            if lab:
                ax.text(1.0, y, f" {lab}", transform=ax.get_yaxis_transform(), color=col, fontsize=8,
                        va="center", ha="left", fontweight="bold")
            if t is not None and np.isfinite(t):
                ax.axvline(t, color=col, ls=ls, lw=lw)
                if lab:
                    ax.text(t, 1.0, f"t* {lab.split()[0]}", transform=ax.get_xaxis_transform(), color=col,
                            fontsize=8, rotation=90, va="top", ha="right", fontweight="bold")
    for j in m["joins"]:
        ax.axvline(j, color="saddlebrown", ls="-.", lw=0.9)
    ax.axvline(m["end"], color="0.3", lw=0.8)
    if np.isfinite(m["escape"]):
        ax.axvline(m["escape"], color="0.4", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(max(band[0, 0], 10), 3 * m["end"])
    ok = np.isfinite(curve) & (tc > 100)
    data_lo = np.nanmin(band[:, 1])
    # a floor far below the data (a fit pinned at its bound) is left off the axis rather than squashing it
    lo = (min(m["floor"], data_lo) if m["floor"] and m["floor"] > 0.3 * data_lo else data_lo) * 0.8
    hi = (np.nanmax(curve[ok]) if ok.any() else np.nanmax(band[:, 2])) * 1.5
    ax.set_ylim(lo, hi)
    ax.set_xlabel("iteration (joined along the warm-start chain)"); ax.set_ylabel("noisy training loss")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"{m['sweep']} / {m['cell']}   seed {m['seed']}", fontsize=10)

    ts = m["t_stars"]
    fmt = lambda v: ("none" if v is None or not np.isfinite(v) else f"{v / 1e3:.1f}k" if v >= 1000 else f"{v:.0f}")
    lines = [f"STATUS: {m['status'].upper()} — {m['why']}"]
    if m["status"] == "top-up":
        lines.append(f"  top-up target: {m['target'] / 1e3:.0f}k iterations (now {m['end'] / 1e3:.0f}k)")
    lines += ["", f"task: {m['task']}    N = {m['N']}    trained {m['end'] / 1e3:.0f}k iterations",
              f"floor {m['floor']:.4g} ({m['floor_kind']})" if m["floor"] else "floor: none",
              f"read-out t*: 1.05x {fmt(ts.get('0.05'))}   1.10x {fmt(ts.get('0.1'))}   1.20x {fmt(ts.get('0.2'))}",
              (f"end of training / floor: {m['late_rise']:.3f}" if np.isfinite(m["late_rise"]) else "")
              + (f"   fit - loss at end: {m['end_residual']:+.1%}" if m.get("end_residual") is not None else ""),
              f"bursts: {len(m['bursts'])}   escape from start level: {fmt(m['escape'])}",
              "", f"NON-STANDARD vs {m['reference']} (reference -> this run):"]
    lines += [f"  {d}" for d in m["diff"]] or ["  none"]
    lines += ["", "PATH:"] + textwrap.wrap(m["run"], 66)
    lines += ["", "blue solid: 1.05x floor and its read-out t*", "violet dashed: 1.10x floor and its read-out",
              "grey dotted: 1.20x floor and its read-out", "brown dash-dot: warm-start join",
              "grey thin dotted: escape   grey solid: end   orange: burst"]
    tx.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.3)
    fig.savefig(png, dpi=90)
    plt.close(fig)
    return m, png


def main():
    """Command-line entry: `extract` or `draw` (module docstring)."""
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("out_dir"); e.add_argument("ref_dir"); e.add_argument("runs", nargs="*")
    e.add_argument("--from-csv"); e.add_argument("--workers", type=int, default=8)
    d = sub.add_parser("draw")
    d.add_argument("npz_dirs", nargs="+"); d.add_argument("--out", required=True)
    d.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "extract":
        runs = list(a.runs)
        if a.from_csv:
            runs += [r["run"] for r in csv.DictReader(open(a.from_csv))]
        os.makedirs(a.out_dir, exist_ok=True)
        with Pool(a.workers) as p:
            done = [x for x in p.map(extract_one, [(r, a.out_dir, a.ref_dir) for r in runs], chunksize=4) if x]
        print(f"extracted {len(done)} of {len(runs)} runs -> {a.out_dir}")
    else:
        npzs = sorted(f for dd in a.npz_dirs for f in glob.glob(os.path.join(dd, "*.npz")))
        with Pool(a.workers) as p:
            res = p.map(draw_one, [(f, a.out) for f in npzs], chunksize=4)
        res.sort(key=lambda x: (x[0]["sweep"], x[0]["cell"], str(x[0]["seed"])))
        with open(os.path.join(a.out, "index.md"), "w") as fh:
            fh.write("# Loss trajectories, one figure per network\n\n"
                     "Made by `trainRNNbrain/experiments_and_analysis/run_trajectories.py`; status rule and "
                     "reference configs in its docstring.\n\n"
                     "| figure | sweep | cell | seed | task | N | status | why | non-standard |\n|---|---|---|---|---|---|---|---|---|\n")
            for m, png in res:
                rel = os.path.relpath(png, a.out)
                fh.write(f"| [{os.path.basename(png)}]({rel}) | {m['sweep']} | {m['cell']} | {m['seed']} | {m['task']} | "
                         f"{m['N']} | {m['status']} | {m['why']} | {'; '.join(m['diff']) or 'none'} |\n")
        print(f"drew {len(res)} figures -> {a.out}")


if __name__ == "__main__":
    main()
