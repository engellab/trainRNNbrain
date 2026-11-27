#!/usr/bin/env python3
"""Is another optimizer a remedy for inactive units? The read, the verdict and the figure.

RULES. docs/project_trajectory/2026-11-25_1318_rules-optimizer-remedy.md, committed before any run. The cells, the
criterion and the outcomes below are copied from that entry.

MODE `read OUT_JSON --logs DIR` (Della CPU node). Every 3BFF and CDDM network of the five candidates
(<task>_optimizer_remedy/EqType=h_N=1000_arm=<arm>) and the independent-draw default networks
(<task>_independent_init; the same three seeds), at 75,000 iterations: active units at the nearest probe and held-out
R^2 of the weights saved there, 36 draws (fig_paper_F2_fails.read_fixed); the active count at every stored vector;
skipped updates (Slurm log); the differences of the saved config from the same-seed control's; the arm named by the
saved config, which must match the folder.

MODE `maze-read OUT_JSON` (Della CPU node). The Maze networks of the candidates and the matched default networks
(with their continuation), joined along the warm-start chain (common.maze_load_runs) and cut at updates 2,000 and
4,000 (entropy_setpoint_read.maze_point), with the first update's rollout (before any optimizer step) and the config
differences from the same-seed control.

MODE `units OUT_JSON --arms A [B ...]` (Della GPU node). For the 3BFF and CDDM controls and the named candidates, on the
weights at 75,000: the M09 removal (a3_remedy_ablation.ablate_net, k against the controls' mean) and the near-copy
shares (copy_stats). The Maze removal is rl_maze_ablate.py --added.

MODE `verdict READS_JSON MAZE_JSON [--fig STEM]` (Mac). Read gate, config gate, the M09 rule
(fig_paper_F5_tasks.rule) per candidate and task against the matched controls and against M09's own, the three
size-test numbers, the late-decline numbers to 75,000 (Maze: 2,000 and 4,000), the outcome of the rules entry.

Usage: PYTHONPATH=<repo> CUDA_VISIBLE_DEVICES= python optimizer_remedy_read.py read OUT.json --logs DIR
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a3_remedy_ablation as A3  # noqa: E402
import fig_paper_F5_tasks as F5T  # noqa: E402
from common import DATA_DIR, active_mask, maze_load_runs, runs_with_loss  # noqa: E402
from entropy_setpoint_read import chain_trace, flat_cfg, maze_point, window_median  # noqa: E402
from fig_paper_F2_fails import read_fixed  # noqa: E402

ARMS = ("nodecay", "adamw1e-2", "amsgrad", "sgdm0.03", "sgdm0.1")
# the optimizer keys of each arm (optimizer, weight_decay, momentum, base learning rate): the folder is checked against it
KEYS = {"control": ("adam", 1e-6, 0.0, 1e-3), "nodecay": ("adam", 0.0, 0.0, 1e-3), "adamw1e-2": ("adamw", 1e-2, 0.0, 1e-3),
        "amsgrad": ("amsgrad", 1e-6, 0.0, 1e-3), "sgdm0.03": ("sgd", 1e-6, 0.9, 0.03), "sgdm0.1": ("sgd", 1e-6, 0.9, 0.1)}
TASKS = {"3BFF": "NBitFlipFlop", "CDDM": "CDDM"}
JOB = {"3BFF": "OptRemFF", "CDDM": "OptRemCDDM"}
SEEDS = {"3BFF": ["1558264694", "254798194", "254859059"], "CDDM": ["3833537590", "1272763220", "457853"],
         "Maze": ["1", "2", "3"]}
ARM_CELL = "{prefix}_optimizer_remedy/EqType=h_N=1000_arm={arm}"
CTRL_CELL = "{prefix}_independent_init/EqType=h_N=1000_draws=independent"
MAZE_HOME = os.path.expanduser("~/trainRNNbrain/data")
MAZE_ARMS = os.path.join(MAZE_HOME, "trained_RNNs_rl_maze_optrem", "Maze_optimizer_remedy")
MAZE_CTRL = os.path.join(MAZE_HOME, "trained_RNNs_rl_maze_mctrl_N1000", "Maze_interventions")
FIXED_IT, MAZE_ENDS = 75000, (2000, 4000)
COPY_COS = 0.9                   # a near copy: incoming-weight cosine with another active unit above this
TOL, MAZE_TOL, MAZE_FLOOR = 0.01, 0.05, 0.9
# M09's own control networks (shared draws on 3BFF and CDDM; three Maze networks of the first trainer version): active
# units and mean performance (ST_verdicts, M09; 3BFF mean R^2 of the three with weights at 75,000: data/round18/g1g2.json)
M09 = {"3BFF": ((236, 288), None), "CDDM": ((333, 342), 0.8337), "Maze": ((415, 449), 0.898)}
# config keys in which an arm may differ from its same-seed control (the optimizer keys are checked by KEYS)
ALLOWED = {"experiment_tag", "trainer.optimizer", "trainer.weight_decay", "trainer.momentum", "trainer.lr",
           "trainer.max_iter", "paths.save_to", "paths.checkpoint_to", "paths.init_from", "paths.init_adam",
           "trainer.n_updates", "trainer.iter_offset"}
COLORS = {"control": "0.35", "nodecay": "#2a78d6", "adamw1e-2": "#1f9e89", "amsgrad": "#eb6834", "sgdm0.03": "#9b59b6",
          "sgdm0.1": "#c9a227"}
LABELS = {"control": "default (Adam, decay 1e-6 in the gradient)", "nodecay": "Adam, no decay (= AdamW at 0 and 1e-6)",
          "adamw1e-2": "AdamW, decay 0.01", "amsgrad": "AMSGrad", "sgdm0.03": "SGD, momentum 0.9, rate 0.03",
          "sgdm0.1": "SGD, momentum 0.9, rate 0.1"}


def arm_of(run_dir):
    """The candidate a run's saved config names, from its optimizer keys.

    Args:
        run_dir: str, a run folder holding *config.yaml.
    Returns:
        str, a key of KEYS.
    Raises:
        ValueError if no candidate has these keys.
    """
    t = OmegaConf.load(sorted(glob.glob(os.path.join(glob.escape(run_dir), "*config.yaml")))[0]).trainer
    keys = (str(t.get("optimizer", "adam")), float(t.weight_decay), float(t.get("momentum", 0.0)), float(t.lr))
    hit = [a for a, k in KEYS.items() if k[0] == keys[0] and np.allclose(k[1:], keys[1:], rtol=1e-9, atol=0)]
    if len(hit) != 1:
        raise ValueError(f"{run_dir}: optimizer keys {keys} name {hit}")
    return hit[0]


def skipped_of(run_dir, logs, job):
    """Skipped updates of one run from the Slurm log that names both its experiment tag and its seed.

    entropy_setpoint_read.skipped matches the seed alone; here the five candidates share their seeds, so that match
    returns one candidate's log for all five.

    Args:
        run_dir: str, run folder (its saved config holds experiment_tag and seed); logs: str, folder of *.out files;
        job: str, the job name the logs start with.
    Returns:
        int (0 when the log prints no `gradient spikes` line) or None when no log names this tag and seed.
    """
    c = flat_cfg(run_dir)
    key = f"{c['experiment_tag']}/s{c['seed']}/"
    for f in glob.glob(os.path.join(logs, f"{job}.*.out")):
        txt = open(f, errors="replace").read()
        if key in txt:
            m = re.search(r"gradient spikes: (\d+) updates skipped", txt)
            return int(m.group(1)) if m else 0
    return None


def differs(run_dir, ref_cfg):
    """Config keys in which a run differs from a reference, outside the allowed ones.

    Args:
        run_dir: str; ref_cfg: dict, flat_cfg of the control.
    Returns:
        (dict of unexpected differences, dict of allowed ones), each key -> [run value, control value].
    """
    c = flat_cfg(run_dir)
    d = {k: [str(c.get(k, "<absent>")), str(ref_cfg.get(k, "<absent>"))]
         for k in sorted(set(c) | set(ref_cfg)) if c.get(k, "<absent>") != ref_cfg.get(k, "<absent>")}
    return {k: v for k, v in d.items() if k not in ALLOWED}, {k: v for k, v in d.items() if k in ALLOWED}


def do_read(out_json, logs):
    """Mode `read` (module docstring).

    Args:
        out_json: str, output file; logs: str, folder of the Slurm logs.
    Returns:
        None.
    """
    cache_path = os.path.splitext(out_json)[0] + "_r2cache.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    out = {}
    for task, prefix in TASKS.items():
        ctrl_runs = {str(flat_cfg(r["run_dir"])["seed"]): r for r in
                     runs_with_loss(os.path.join(DATA_DIR, CTRL_CELL.format(prefix=prefix)))}
        ref = {s: flat_cfg(r["run_dir"]) for s, r in ctrl_runs.items()}
        out[task] = {}
        groups = [("control", list(ctrl_runs.values()))]
        for arm in ARMS:
            cell = os.path.join(DATA_DIR, ARM_CELL.format(prefix=prefix, arm=arm))
            groups.append((arm, runs_with_loss(cell) if os.path.isdir(cell) else []))
        for name, runs in groups:
            out[task][name] = []
            for r in runs:
                d = r["run_dir"]
                cfg = flat_cfg(d)
                seed = str(cfg["seed"])
                if arm_of(d) != name:
                    raise ValueError(f"{d}: config names {arm_of(d)}, folder {name}")
                f = read_fixed(r, FIXED_IT, cache)
                its, acts, _ = chain_trace(d, until=1e12)
                bad, ok = differs(d, ref.get(seed, {}))
                rec = dict(run_dir=d, seed=seed, active=int(f["active"]), probe=f["probe"], r2=float(f["r2"]),
                           r2_how=f["r2_how"], it_end=float(np.asarray(r["it_loss"])[-1]),
                           finite_loss=bool(np.isfinite(r["loss"]).all()), first_probe=[float(r["it_act"][0]), int(r["active"][0])],
                           skipped=skipped_of(d, logs, JOB[task] if name != "control" else "IndInit"),
                           draws=str(cfg.get("model.init_draws")), differs_bad=bad, differs_ok=ok, trace_it=its, trace_active=acts)
                out[task][name].append(rec)
                print(task, name, {k: v for k, v in rec.items() if not k.startswith("trace")}, flush=True)
                json.dump(cache, open(cache_path, "w"))
    json.dump(out, open(out_json, "w"), indent=1)


def do_maze_read(out_json):
    """Mode `maze-read` (module docstring).

    Args:
        out_json: str, output file.
    Returns:
        None.
    """
    runs = []
    for root, is_ctrl in ((MAZE_CTRL, True), (MAZE_ARMS, False)):
        for r in maze_load_runs(root) if os.path.isdir(root) else []:
            cell = os.path.basename(os.path.dirname(r["chain"][0]))
            if int(r["N"]) != 1000 or int(r["seed"]) not in (1, 2, 3):
                continue
            runs.append((r, "control" if is_ctrl or cell == "control" else cell))
    # a control continued into my tree is read once, from its continuation (its chain starts in the controls' tree)
    best = {}
    for r, name in runs:
        key = (name, int(r["seed"]))
        if key not in best or len(r["chain"]) > len(best[key]["chain"]):
            best[key] = r
    ref = {s: flat_cfg(r["chain"][0]) for (n, s), r in best.items() if n == "control"}
    out = {}
    for (name, seed), r in sorted(best.items()):
        if arm_of(r["chain"][0]) != name or arm_of(r["folder"]) != name:
            raise ValueError(f"{r['folder']}: config names {arm_of(r['folder'])}, folder {name}")
        bad, ok = differs(r["chain"][0], ref.get(seed, {}))
        p = {e: maze_point(r, e) for e in MAZE_ENDS}
        rec = dict(seed=str(seed), folder=r["folder"], chain=list(r["chain"]), at={str(e): p[e] for e in MAZE_ENDS},
                   update1=[r["progress"][0].get(k) for k in ("discounted_return", "success", "steps")],
                   differs_bad=bad, differs_ok=ok)
        out.setdefault(name, []).append(rec)
        print("Maze", name, seed, {e: {k: p[e][k] for k in ("last_update", "complete", "active_end", "success_end", "learned_update",
              "active_learned", "skipped_sum", "restored_sum")} for e in MAZE_ENDS}, rec["update1"], bad, flush=True)
    json.dump(out, open(out_json, "w"), indent=1)


def copy_stats(W_rec, W_inp, p, k):
    """Near-copy shares of a network's active units, by incoming-weight cosine with the nearest other active unit.

    Args:
        W_rec: (N, N) array, row i holds the weights onto unit i; W_inp: (N, n_inputs) array; p: (N,) participation;
        k: int, the number of least participating active units to report on their own (<= 0: none).
    Returns:
        dict: n_active, k, share_all and share_k (rows restricted to active sources and the inputs), full_all and
        full_k (full rows), median_cos (restricted, over active units); the _k values are None when k <= 0.
    """
    act = np.flatnonzero(active_mask(p))
    least = np.argsort(np.asarray(p)[act], kind="stable")[:max(k, 0)]
    out = dict(n_active=int(len(act)), k=int(k))
    for key, X in (("share", np.hstack([W_rec[np.ix_(act, act)], W_inp[act]])), ("full", np.hstack([W_rec[act], W_inp[act]]))):
        X = np.asarray(X, float)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
        C = X @ X.T
        np.fill_diagonal(C, -np.inf)
        near = C.max(1)
        out[f"{key}_all"] = float((near > COPY_COS).mean())
        out[f"{key}_k"] = float((near[least] > COPY_COS).mean()) if k > 0 else None
        if key == "share":
            out["median_cos"] = float(np.median(near))
    return out


def do_units(out_json, arms):
    """Mode `units` (module docstring).

    Args:
        out_json: str, output file; arms: list of str, candidates.
    Returns:
        None.
    """
    out = {}
    for task, prefix in TASKS.items():
        out[task] = {}
        n_ctrl = None
        for name in ["control"] + list(arms):
            cell = CTRL_CELL if name == "control" else ARM_CELL
            out[task][name] = []
            for d in sorted(glob.glob(os.path.join(DATA_DIR, cell.format(prefix=prefix, arm=name), "*/"))):
                d = d.rstrip("/")
                r = A3.ablate_net(d, None if name == "control" else n_ctrl)
                if r is None:
                    print(f"{task} {name} {d}: no weights at {FIXED_IT}", flush=True)
                    continue
                w = A3.A.load_params(r["params_file"])
                cfg, rnn = A3.A.build(d, w)
                with torch.no_grad():
                    p = A3.A.participation(rnn(A3.A.seeded_batches(cfg, rnn.device)["tr"][0], w_noise=False)[0]).cpu().numpy()
                # the controls have no k: their least participating third stands in (rules entry)
                k = r["n_active"] // 3 if name == "control" else r["surplus"]
                r.update(seed=str(cfg.seed), copies=copy_stats(np.asarray(w["W_rec"], float), np.asarray(w["W_inp"], float), p, k))
                out[task][name].append(r)
                print(task, name, {x: v for x, v in r.items() if x not in ("run_dir", "params_file")}, flush=True)
                json.dump(out, open(out_json, "w"), indent=1)
            if name == "control":
                n_ctrl = float(np.mean([r["n_active"] for r in out[task][name]]))


def rule(a_act, a_perf, c_act, c_perf, tol, floor=None, n_expected=3):
    """fig_paper_F5_tasks.rule, with the rules entry's incomplete-set condition.

    Args:
        a_act, a_perf, c_act, c_perf: lists for the finished networks of the arm and of the controls; tol: float;
        floor: float or None; n_expected: int, finished networks each side must have.
    Returns:
        the F5T.rule dict plus n_arm, n_ctrl; works is False when the arm is incomplete; None if a side is empty.
    """
    if not a_act or not c_act:
        return None
    v = dict(F5T.rule(a_act, a_perf, c_act, c_perf, tol, floor=floor), n_arm=len(a_act), n_ctrl=len(c_act))
    if len(a_act) < n_expected or len(c_act) < n_expected:
        v["text"] += f" [INCOMPLETE: {len(a_act)} arm, {len(c_act)} control networks]"
        v["works"] = False
    return v


def do_verdict(reads_json, maze_json, fig_stem):
    """Mode `verdict` (module docstring).

    Args:
        reads_json, maze_json: str, the files `read` and `maze-read` wrote; fig_stem: str or None.
    Returns:
        dict candidate -> {"3BFF": bool or None, "CDDM": ..., "Maze2000": ..., "Maze4000": ...}.
    """
    R, M = json.load(open(reads_json)), json.load(open(maze_json))
    res = {a: {} for a in ARMS}
    for task in TASKS:
        C = [r for r in R[task]["control"] if r["it_end"] >= FIXED_IT - 100]
        print(f"\n=== {task}: gate. controls {len(C)} of 3, seeds {sorted(r['seed'] for r in C)}")
        if sorted(r["seed"] for r in C) != sorted(SEEDS[task]):
            print("  GATE FAILS: controls incomplete; no verdict for this task")
            continue
        ca, cp = [r["active"] for r in C], [r["r2"] for r in C]
        print(f"  controls active {ca} (max {max(ca)}) R2 {np.round(cp, 4).tolist()} mean {np.mean(cp):.4f} bar {np.mean(cp) - TOL:.4f}")
        for arm in ARMS:
            rows = [r for r in R[task].get(arm, []) if r["it_end"] >= FIXED_IT - 100]
            if not R[task].get(arm):
                print(f"  {arm}: not run"); res[arm][task] = None; continue
            bad = {r["seed"]: r["differs_bad"] for r in rows if r["differs_bad"]}
            # the same-seed control's first stored count (rules, check 4)
            start = [(r["first_probe"], next(c["first_probe"] for c in C if c["seed"] == r["seed"])) for r in rows]
            act, perf = [r["active"] for r in rows], [r["r2"] for r in rows]
            v = rule(act, perf, ca, cp, TOL)
            lo, hi = M09[task][0]
            print(f"  {arm}: {v['text']}; active {act} R2 {np.round(perf, 4).tolist()} fold {v['fold']:.2f} dR2 {v['dperf']:+.4f}"
                  f"; vs M09 largest {hi}: {'above' if min(act) > hi else 'NOT above'}; "
                  f"size-test numbers: stable {sum(r['finite_loss'] and (r['skipped'] or 0) < 0.05 * FIXED_IT for r in rows)}/{len(rows)}, "
                  f"fold>=1.5 {v['fold'] >= 1.5}, R2 within 0.02 {v['dperf'] >= -0.02}; skipped {[r['skipped'] for r in rows]}")
            if bad:
                print(f"    CONFIG GATE: unexpected differences {bad}")
            if any(a != b for a, b in start):
                print(f"    START GATE: first probe differs {start}")
            res[arm][task] = bool(v["works"]) and not bad
            print("    decline (median active, 2,000 iterations before 50k -> before 75k), arm:",
                  [f"{window_median(r['trace_it'], r['trace_active'], 50000):.0f}->{window_median(r['trace_it'], r['trace_active'], 75000):.0f}" for r in rows],
                  "controls:", [f"{window_median(r['trace_it'], r['trace_active'], 50000):.0f}->{window_median(r['trace_it'], r['trace_active'], 75000):.0f}" for r in C])
    for end in MAZE_ENDS:
        C = [x["at"][str(end)] for x in M.get("control", []) if x["at"][str(end)]["complete"]]
        print(f"\n=== Maze, update {end:,}: gate. controls {len(C)} of 3")
        if len(C) < 3:
            print("  GATE FAILS: controls incomplete; no verdict at this update")
            continue
        ca, cp = [p["active_end"] for p in C], [p["success_end"] for p in C]
        print(f"  controls active {ca} success {np.round(cp, 3).tolist()} learned {[p['learned_update'] for p in C]}")
        for arm in ARMS:
            X = M.get(arm, [])
            if not X:
                continue
            fin = [x for x in X if x["at"][str(end)]["complete"]]
            act, perf = [x["at"][str(end)]["active_end"] for x in fin], [x["at"][str(end)]["success_end"] for x in fin]
            v = rule(act, perf, ca, cp, MAZE_TOL, floor=MAZE_FLOOR)
            if v is None:
                print(f"  {arm}: no finished network"); continue
            learned = [x["at"][str(end)]["learned_update"] for x in fin]
            print(f"  {arm}: {v['text']}; active {act} success {np.round(perf, 3).tolist()} learned {learned} fold {v['fold']:.2f} "
                  f"dsuccess {v['dperf']:+.3f}; vs M09 largest {M09['Maze'][0][1]}: {'above' if min(act) > M09['Maze'][0][1] else 'NOT above'}; "
                  f"size-test numbers: learned {sum(u is not None for u in learned)}/{len(learned)}, fold>=1.5 {v['fold'] >= 1.5}")
            bad = {x["seed"]: x["differs_bad"] for x in fin if x["differs_bad"]}
            if bad:
                print(f"    CONFIG GATE: {bad}")
            res[arm][f"Maze{end}"] = bool(v["works"]) and not bad
            if end == MAZE_ENDS[0]:
                print("    update-1 rollout (arm, control):", [(x["update1"], next(c["update1"] for c in M["control"] if c["seed"] == x["seed"])) for x in fin])
    print("\n=== outcome (rules entry)")
    for arm in ARMS:
        print(f"  {arm}: {res[arm]}")
    sup = [a for a in ARMS if res[a].get("3BFF") and res[a].get("CDDM")]
    full = [a for a in sup if res[a].get("Maze2000")]
    late = [a for a in sup if res[a].get("Maze4000") and not res[a].get("Maze2000")]
    print(f"  pass 3BFF and CDDM: {sup}; also Maze at 2,000: {full}; Maze only at 4,000: {late}")
    print("  " + (f"REMEDY: {full}" if full else f"MAZE ONLY WITH LONGER TRAINING (if M09's condition applies): {late}" if late
                  else "NO CANDIDATE MEETS THE CRITERION ON ALL THREE TASKS (complete cells only)"))
    if fig_stem:
        draw(R, M, fig_stem)
    return res


def draw(R, M, stem):
    """Active units against training, every network of every candidate and of the controls, one panel per task.

    Args:
        R: dict, `read` output; M: dict, `maze-read` output; stem: str, figure path without extension.
    Returns:
        None; writes stem.pdf and stem.png.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, task in zip(axes, ("3BFF", "CDDM", "Maze")):
        for arm in ("control",) + ARMS:
            if task == "Maze":
                rows = [(x["at"]["4000"]["trace_u"], x["at"]["4000"]["trace_active"], 1) for x in M.get(arm, [])]
            else:
                rows = [([u for u in r["trace_it"] if u <= FIXED_IT], r["trace_active"], 1000) for r in R[task].get(arm, [])]
            for k, (u, a, s) in enumerate(rows):
                ax.plot(np.asarray(u) / s, a[:len(u)], color=COLORS[arm], lw=1.2 if arm == "control" else 1.6, alpha=0.9,
                        label=LABELS[arm] if k == 0 and task != "Maze" else None)
        ax.set_title(f"{task}, N = 1,000", fontsize=10)
        ax.set_xlabel("PPO update" if task == "Maze" else "iteration (thousands)")
        ax.set_ylabel("active units (of 1,000)")
        ax.set_ylim(0, 1000)
        ax.grid(alpha=0.25, lw=0.5)
        if task == "Maze":
            ax.axvline(2000, color="0.7", lw=0.6)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=150)
    print(f"figure: {stem}.pdf, .png")


def main():
    """Parse the mode and run it."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("read", "maze-read", "units", "verdict"))
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--logs", default=os.path.expanduser("~/trainRNNbrain_logs"))
    ap.add_argument("--fig", default=None)
    ap.add_argument("--arms", nargs="+", default=["nodecay"])
    a = ap.parse_args()
    if a.mode == "read":
        do_read(a.paths[0], a.logs)
    elif a.mode == "maze-read":
        do_maze_read(a.paths[0])
    elif a.mode == "units":
        do_units(a.paths[0], a.arms)
    else:
        do_verdict(a.paths[0], a.paths[1], a.fig)


if __name__ == "__main__":
    main()
