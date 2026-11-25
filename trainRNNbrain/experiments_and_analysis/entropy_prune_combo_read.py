#!/usr/bin/env python3
"""Entropy set point x prune-and-duplicate, 2 x 2 at N = 1,000: the reads, the verdicts and the figure.

RULES. docs/project_trajectory/2026-11-25_1245_rules-entropy-prune-combo.md, committed before any run. The cells:
D (neither), E (entropy set point, lambda_ent 0.1), P (prune-and-duplicate), EP (both); three seeds per cell, the
same seeds in every cell of a task. Every network trains in two stages (3BFF and CDDM: 75,000 iterations, then to L;
Maze: 2,000 updates, then 2,000 more).

MODE `read` (Della CPU node). 3BFF and CDDM: per network, the active count at every stored participation vector of
both stages; held-out R^2 (36 draws) and the dimensionality of the active units' noise-free rates at 75,000 and at L;
replacements per check; the death rate; skipped updates; config differences from the same seed's E network. Adds
tasks "3BFF" and "CDDM" to OUT_JSON.
MODE `maze-read` (Della GPU node, for the replayed probe). The same for Maze, task "Maze" in OUT_JSON.
MODE `verdict` (Mac). The gate on E's recorded values, rules (b)-(e) per task, the tables, the figure (--fig).
MODE `selfcheck`. The verdict on synthetic cells, where the answer is known (rules, check 4).

Usage:
  python entropy_prune_combo_read.py read OUT_JSON --until L [--logs DIR]   (Della, repository root on PYTHONPATH)
  python entropy_prune_combo_read.py maze-read OUT_JSON --data DIR          (Della GPU; DIR holds the Maze trees)
  python entropy_prune_combo_read.py verdict OUT_JSON --fig PATH_STEM
  python entropy_prune_combo_read.py selfcheck
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H  # noqa: E402
import fig_paper_F5_tasks as F5T  # noqa: E402
from common import DATA_DIR, _maze_read_run, active_mask, maze_learned_index  # noqa: E402
from criticality_read import build_rnn, load_params  # noqa: E402
from entropy_setpoint_read import chain_trace, flat_cfg, unit_stats, window_median  # noqa: E402
from rl_maze_population_stats import probe_rates  # noqa: E402
from trainRNNbrain.trainer.TrainerPPO import load_run as load_maze  # noqa: E402

N_UNITS = 1000
CELLS = ("D", "E", "P", "EP")
ARM = {"D": "none", "E": "ent0.1", "P": "dup", "EP": "ent0.1_dup"}                        # 3BFF, CDDM folders
MAZE_ARM = {"D": "control", "E": "ent0.1", "P": "duplicate", "EP": "ent0.1_duplicate"}    # Maze folders
MAZE_STAGE1 = {"D": "trained_RNNs_rl_maze_mctrl_N1000", "E": "trained_RNNs_rl_maze_entsp_N1000",
               "P": "trained_RNNs_rl_maze_epc", "EP": "trained_RNNs_rl_maze_epc"}
MAZE_STAGE2 = "trained_RNNs_rl_maze_epc"
SEEDS = {"3BFF": (4200852863, 1256299004, 4168822465), "CDDM": (3116616303, 1299181146, 2184072010),
         "Maze": (1, 2, 3)}
PREFIX = {"3BFF": "NBitFlipFlop", "CDDM": "CDDM"}
READ = {"3BFF": 75000, "CDDM": 75000, "Maze": 2000}
TOL = {"3BFF": 0.01, "CDDM": 0.01, "Maze": 0.05}
FLOOR = {"3BFF": None, "CDDM": None, "Maze": 0.9}
WINDOW = {"3BFF": 2000, "CDDM": 2000, "Maze": 100}     # width of the median window of A(t)
PER = {"3BFF": 1000, "CDDM": 1000, "Maze": 100}        # death rate: per 100 active units per this many iterations/updates
MIN_GAP = 20                                           # units; rule (c)
CEILING = 950                                          # active units; rule (e)
COPY_CORR = 0.95
# E's recorded counts and performance (entries 2026-11-21_0755, 2026-11-24_1848, 2026-11-22_1808), by seed; the counts
# on 3BFF and CDDM are at iteration 74,900, the last vector of stage 1
RECORDED = {"3BFF": {4200852863: (384, 0.9503), 1256299004: (380, 0.9515), 4168822465: (454, 0.9516)},
            "CDDM": {3116616303: (540, 0.8340), 1299181146: (566, 0.8268), 2184072010: (539, 0.8344)},
            "Maze": {1: (634, 0.938), 2: (689, 1.000), 3: (661, 1.000)}}
ALLOWED = re.compile(r"^(experiment_tag|paths\..*|trainer\.(lambda_ent|prune_reinit)|trainer\.prune_args\."
                     r"(reinit_mode|copy_noise|copy_norm_preserve|max_replace_frac|maturity|utility))$")
ALLOWED_MAZE = re.compile(ALLOWED.pattern[:-1] + r"|^trainer\.prune_args\.(check_every|patience)$")
COLORS = {"D": "0.45", "E": "#2a78d6", "P": "#d9a400", "EP": "#2f9e44"}
NAMES = {"D": "default", "E": "entropy set point", "P": "prune-and-duplicate", "EP": "both"}


# ---------------------------------------------------------------- measurements

def deaths(t, masks, start, end):
    """Unit deaths in (start, end]: active at vector k - 1, inactive at k, k + 1 and k + 2.

    Args:
        t: times of the stored vectors, ascending; masks: list of (N,) bool active masks; start, end: the window.
    Returns:
        dict n (deaths), exposure (active units x time over the same intervals).
    """
    n, exposure = 0, 0.0
    for k in range(1, len(t) - 2):
        if start < t[k] and t[k + 2] <= end:
            n += int((masks[k - 1] & ~masks[k] & ~masks[k + 1] & ~masks[k + 2]).sum())
            exposure += float(masks[k - 1].sum()) * (t[k] - t[k - 1])
    return dict(n=n, exposure=exposure)


def count_increments(events):
    """Per-sample increments of a cumulative count that restarts at 0 at the start of stage 2.

    Args:
        events: cumulative count at every sample.
    Returns:
        (len(events),) float array.
    """
    e = np.asarray(events, float)
    d = np.diff(e, prepend=0.0)
    return np.where(d < 0, e, d)


def dimensionality(rates, active):
    """Participation ratio of the active units' rates, and the fraction of active units with a near-copy.

    Args:
        rates: (N, M) noise-free rates; active: (N,) bool.
    Returns:
        dict pr (sum ev)^2 / sum ev^2 of the covariance, near_copy (fraction with a correlation above COPY_CORR with
        another active unit), n_active.
    """
    x = np.asarray(rates[active], np.float64)
    x = x - x.mean(1, keepdims=True)
    C = x @ x.T / (x.shape[1] - 1)
    s = np.sqrt(np.diag(C))
    R = C / np.outer(s, s)
    np.fill_diagonal(R, -1)
    return dict(pr=float(np.trace(C) ** 2 / (C ** 2).sum()), near_copy=float((R.max(1) > COPY_CORR).mean()),
                n_active=int(active.sum()))


def supervised_dims(run_dir):
    """`dimensionality` of one supervised network's final weights, on the held-out batch without noise.

    Args:
        run_dir: folder holding *LastParams* and *_config.yaml.
    Returns:
        dict of `dimensionality`.
    """
    cfg = H.OmegaConf.load(sorted(glob.glob(os.path.join(run_dir, "*_config.yaml")))[0])
    rnn = build_rnn(cfg, seed=int(cfg.seed))
    load_params(rnn, H._params(run_dir)[0])
    with torch.no_grad():
        h, _ = rnn(H.heldout_batch(cfg)[0], w_noise=False)
        r = torch.relu(h).reshape(h.shape[0], -1).cpu().numpy()
    p = r.std(1) + np.quantile(np.abs(r), 0.9, axis=1)
    return dimensionality(r, active_mask(p))


def skipped_updates(logs, tag, seed):
    """Skipped updates of one 3BFF or CDDM job, from its Slurm log.

    Args:
        logs: folder of the *.out files; tag: str, the output folder the launcher prints (its last two parts);
        seed: int, the job's seed.
    Returns:
        int, or None when no log names this folder and seed.
    """
    files = [f for job in ("EntPrune", "EntSP", "EntCDDM") for f in glob.glob(os.path.join(logs, job + ".*.out"))]
    for f in files:
        with open(f, errors="replace") as fh:
            head = fh.read(20000)                                # the launcher's lines and the printed config
            txt = head + fh.read() if re.search(re.escape(tag) + r"\s*$", head, re.M) else ""
        if f"'seed': {seed}," in head and txt:
            m = re.search(r"gradient spikes: (\d+) updates skipped", txt)
            return int(m.group(1)) if m else 0
    return None


def one_dir(pattern):
    """The single folder a glob matches.

    Args:
        pattern: str, a glob.
    Returns:
        str or None when there is no match. Raises ValueError on more than one.
    """
    hit = glob.glob(pattern)
    if len(hit) > 1:
        raise ValueError(f"{pattern}: {len(hit)} folders")
    return hit[0] if hit else None


def differences(run_dir, ref_dir, allowed):
    """Config keys of run_dir that differ from ref_dir and are not allowed to.

    Args:
        run_dir, ref_dir: run folders; allowed: compiled regex of the keys that may differ.
    Returns:
        dict key -> [value, reference value].
    """
    a, b = flat_cfg(run_dir), flat_cfg(ref_dir)
    return {k: [str(a.get(k, "<absent>")), str(b.get(k, "<absent>"))] for k in sorted(set(a) | set(b))
            if a.get(k, "<absent>") != b.get(k, "<absent>") and not allowed.match(k)}


def do_read(out_json, until, logs):
    """Mode `read`: every 3BFF and CDDM network, written to out_json after each network.

    Args:
        out_json: str; until: int, L; logs: folder of the Slurm logs.
    Returns:
        None.
    """
    out = json.load(open(out_json)) if os.path.exists(out_json) else {}
    cache_path = os.path.join(os.path.dirname(os.path.abspath(out_json)), "r2cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    for task, pre in PREFIX.items():
        cells = {c: [] for c in CELLS}
        ref = {}
        for c in ("E", "D", "P", "EP"):                          # E first: the config reference
            for seed in SEEDS[task]:
                sweep1 = "entropy_setpoint" if c == "E" else "entropy_prune"
                tag = "EqType=h_N=1000_arm=" + ARM[c]
                d1 = one_dir(os.path.join(DATA_DIR, f"{pre}_{sweep1}", tag, f"*seed={seed};*"))
                d2 = one_dir(os.path.join(DATA_DIR, f"{pre}_entropy_prune_c", tag, f"*seed={seed + 1};*"))
                rec = dict(seed=seed, stage1=d1, stage2=d2, L=until, finished=False)
                cells[c].append(rec)
                if d1 is None:
                    continue
                if c == "E":
                    ref[seed] = (d1, d2)
                last = d2 or d1
                t, act, masks = chain_trace(last, until=float("inf"))
                w = {"read": H.heldout_r2(d1, draws=H.R2_DRAWS, cache=cache)}
                dims = {"read": supervised_dims(d1)}
                stats = {"read": unit_stats(d1)}
                tr = pickle.load(open(glob.glob(os.path.join(last, "*ParticipationTrace.pkl"))[0], "rb"))
                it = np.asarray(tr["iters"], float)
                inc = count_increments(tr["metrics"]["reinit_events"]) if "reinit_events" in tr["metrics"] else np.zeros(len(it))
                rec.update(t=t, act=act, differs1=differences(d1, ref[seed][0], ALLOWED),
                           skipped1=skipped_updates(logs, f"{pre}_{sweep1}/{tag}", seed),
                           repl1=float(inc[it < READ[task]].sum()) / (READ[task] / 100), r2=w, dims=dims, stats=stats)
                if d2:
                    w["end"] = H.heldout_r2(d2, draws=H.R2_DRAWS, cache=cache)
                    dims["end"], stats["end"] = supervised_dims(d2), unit_stats(d2)
                    late = it >= READ[task]
                    rec.update(finished=bool(t[-1] == until - 100 and np.isfinite(w["end"])),
                               differs2=differences(d2, ref[seed][1], ALLOWED) if ref[seed][1] else None,
                               skipped2=skipped_updates(logs, f"{pre}_entropy_prune_c/{tag}", seed + 1),
                               repl2=float(inc[late].sum()) / ((until - READ[task]) / 100),
                               repl_late_trace=[[float(a), float(b)] for a, b in zip(it[late][9::10], np.add.reduceat(
                                   inc[late], np.arange(0, late.sum(), 10)))],
                               deaths=deaths(t, masks, READ[task], until))
                print(f"{task} {c} {seed}: {act[-1]} active at {t[-1]:.0f}, R2 {w}, dims {dims}", flush=True)
                out[task] = cells
                json.dump(cache, open(cache_path, "w"))
                json.dump(out, open(out_json, "w"), indent=1)


def do_maze_read(out_json, data):
    """Mode `maze-read`: every Maze network, with the replayed probe's dimensionality. Needs a GPU.

    Args:
        out_json: str; data: folder holding the trained_RNNs_rl_maze_* trees.
    Returns:
        None.
    """
    device = torch.device("cuda")                                # a CPU replay plays other starts than the stored probe
    out = json.load(open(out_json)) if os.path.exists(out_json) else {}
    cells, ref = {c: [] for c in CELLS}, {}
    for c in ("E", "D", "P", "EP"):
        for seed in SEEDS["Maze"]:
            d1 = os.path.join(data, MAZE_STAGE1[c], "Maze_interventions", MAZE_ARM[c], f"s{seed}")
            d2 = os.path.join(data, MAZE_STAGE2, "Maze_interventions", MAZE_ARM[c] + "_cont", f"s{seed}")
            rec = dict(seed=seed, stage1=d1, stage2=d2, L=4000, finished=False)
            cells[c].append(rec)
            parts = [_maze_read_run(d) for d in (d1, d2) if glob.glob(os.path.join(d, "*Progress.json"))]
            if not parts:
                continue
            if c == "E":
                ref[seed] = (d1, d2)
            prog = [r for p in parts for r in p["progress"]]
            cat = {k: np.concatenate([p["probes"][k] for p in parts]) for k in ("update", "active", "probe_success")}
            u, act, succ = (cat[k] for k in ("update", "active", "probe_success"))
            at = {int(x): i for i, x in enumerate(u)}
            w = {k: float(succ[at[e]]) for k, e in (("read", 2000), ("end", 4000)) if e in at}
            dims = {}
            for k, d in (("read", d1), ("end", d2)):
                if k in w:
                    _, task, trainer = load_maze(d, device)
                    sim = probe_rates(task, trainer)
                    dims[k] = dict(dimensionality(sim["rates"], active_mask(sim["p"])), replay_success=sim["success"])
            li = maze_learned_index(succ)
            r1 = [r.get("copy_replaced", 0) for r in prog if r["update"] <= 2000]
            r2 = [r.get("copy_replaced", 0) for r in prog if r["update"] > 2000]
            rec.update(t=u.tolist(), act=act.astype(int).tolist(), r2=w, dims=dims, last_update=int(prog[-1]["update"]),
                       learned_update=None if li is None else int(u[li]), differs1=differences(d1, ref[seed][0], ALLOWED_MAZE),
                       skipped1=float(sum(r.get("skipped", 0) + r.get("restored", 0) for r in prog if r["update"] <= 2000)),
                       repl1=float(sum(r1)) / (8 * 2000 / 100))      # a check every 100 of 8 optimiser steps per update
            if len(parts) == 2:
                m = [active_mask(p) for p in parts[1]["probes"]["p"]]
                pu = parts[1]["probes"]["p_update"].astype(float)
                rec.update(finished="end" in w, differs2=differences(d2, ref[seed][1], ALLOWED_MAZE),
                           skipped2=float(sum(r.get("skipped", 0) + r.get("restored", 0) for r in prog if r["update"] > 2000)),
                           repl2=float(sum(r2)) / (8 * 2000 / 100), deaths=deaths(pu.tolist(), m, 2000, 4000))
            print(f"Maze {c} s{seed}: {act[-1]} active at {u[-1]}, success {w}, dims {dims}", flush=True)
            out["Maze"] = cells
            json.dump(out, open(out_json, "w"), indent=1)


# ---------------------------------------------------------------- the rules

def count_at(rec, when):
    """Active units of one network at a time that is a stored vector.

    Args:
        rec: network record; when: time.
    Returns:
        int.
    """
    return int(rec["act"][rec["t"].index(when)])


def verdict(task, cells):
    """Rules (b)-(e) on one task.

    Args:
        task: "3BFF", "CDDM" or "Maze"; cells: dict cell -> list of three network records.
    Returns:
        dict with counts, perf, fall, best, and the verdicts: stack_count (dict point -> bool), perf_ok (dict point ->
        bool), interference (dict point -> bool), ceiling (dict point -> bool), decay (outcome 1-4 as text), stacks,
        stalls_further, complete.
    """
    L = cells["EP"][0]["L"]
    when = {"read": READ[task], "end": L - 100 if task != "Maze" else L}
    done = {c: [r for r in cells[c] if r["finished"]] for c in CELLS}
    complete = {c: len(done[c]) == 3 for c in CELLS}
    at = {"read": {c: [r for r in cells[c] if np.isfinite(r.get("r2", {}).get("read", np.nan)) and when["read"] in r["t"]]
                   for c in CELLS}, "end": done}                 # the networks counted at each point
    cnt = {k: {c: [count_at(r, when[k]) for r in at[k][c]] for c in CELLS} for k in when}
    perf = {k: {c: [r["r2"][k] for r in at[k][c]] for c in CELLS} for k in when}
    mean = {k: {c: float(np.mean(cnt[k][c])) for c in CELLS} for k in when}
    pm = {k: {c: float(np.mean(perf[k][c])) for c in CELLS} for k in when}

    def within(c, k, ref):
        """Cell c's mean performance at point k is within the tolerance of `ref`, and no network below the floor."""
        return pm[k][c] >= ref - TOL[task] and (FLOOR[task] is None or all(x >= FLOOR[task] for x in perf[k][c]))

    ok = [c for c in ("E", "P") if within(c, "read", pm["read"]["D"])]
    best = max(ok or ("E", "P"), key=lambda c: mean["read"][c])
    v = dict(L=L, complete=complete, counts=cnt, perf=perf, best=best, best_qualifies=bool(ok))
    v["stack_count"] = {k: complete["EP"] and min(cnt[k]["EP"]) > max(cnt[k]["E"] + cnt[k]["P"]) for k in when}
    v["perf_ok"] = {k: within("EP", k, pm[k]["D"]) and within("EP", k, pm[k][best]) for k in when}
    v["interference"] = {k: (not complete["EP"] and complete["E"] and complete["P"])
                         or max(cnt[k]["EP"]) < min(cnt[k][best])
                         or (not within("EP", k, pm[k]["D"]) and within(best, k, pm[k]["D"])) for k in when}
    v["ceiling"] = {k: max(cnt[k]["E"] + cnt[k]["P"]) >= CEILING for k in when}
    v["not_exceeded"] = {k: [c for c in ("E", "P") if not min(cnt[k]["EP"]) > max(cnt[k][c])] for k in when}
    v["share"] = {k: (mean[k]["EP"] - mean[k][best]) / (N_UNITS - mean[k][best]) for k in when}
    v["m09"] = {k: F5T.rule(cnt[k]["EP"], perf[k]["EP"], cnt[k]["D"], perf[k]["D"], TOL[task], FLOOR[task])["text"] for k in when}

    fall = {c: [] for c in CELLS}
    for c in CELLS:
        for r in done[c]:
            a0, a1 = (window_median(r["t"], r["act"], e, WINDOW[task]) for e in (READ[task], L))
            fall[c].append((a0, a1, a0 - a1, (a0 - a1) / a0))
    F = {c: [x[2] for x in fall[c]] for c in CELLS}
    f = {c: [x[3] for x in fall[c]] for c in CELLS}
    mF = {c: float(np.mean(F[c])) for c in CELLS}
    lower = all(max(F["EP"]) < min(F[c]) and max(f["EP"]) < min(f[c]) and mF[c] - mF["EP"] >= MIN_GAP for c in ("E", "P"))
    higher = any(min(F["EP"]) > max(F[c]) and min(f["EP"]) > max(f[c]) and mF["EP"] - mF[c] >= MIN_GAP for c in ("E", "P"))
    v["fall"] = fall
    v["decay"] = ("stalled further" if lower and complete["EP"] else "falls faster" if higher else
                  "a single technique already stalls the fall" if min(mF["E"], mF["P"]) < MIN_GAP else "no difference shown")
    v["stacks"] = all(v["stack_count"].values()) and all(v["perf_ok"].values())
    v["stalls_further"] = v["decay"] == "stalled further" and v["perf_ok"]["end"]

    def rate(r):
        """Deaths per 100 active units per PER iterations (updates) of one network."""
        return r["deaths"]["n"] / r["deaths"]["exposure"] * 100 * PER[task]

    v["death_rate"] = {c: [rate(r) for r in done[c]] for c in CELLS}
    v["fewer_die"] = complete["EP"] and max(v["death_rate"]["EP"]) < min(v["death_rate"]["E"] + v["death_rate"]["P"])
    pr = {c: [r["dims"]["end"]["pr"] for r in done[c]] for c in CELLS}
    v["adds_dims"] = complete["EP"] and min(pr["EP"]) > max(pr["E"] + pr["P"])
    return v


def gate(task, cells):
    """Stop unless E reproduces its recorded counts and performance and no config differs unexpectedly (checks 1, 2).

    Args:
        task: str; cells: dict cell -> records.
    Returns:
        None. Raises SystemExit naming the failure.
    """
    for r in cells["E"]:
        a, w = RECORDED[task][r["seed"]]
        got = count_at(r, READ[task] - (100 if task != "Maze" else 0))
        if got != a or abs(r["r2"]["read"] - w) > (5e-5 if task != "Maze" else 5e-4):
            raise SystemExit(f"GATE FAILS: {task} E {r['seed']} reads {got}, {r['r2']['read']:.4f}; recorded {a}, {w}")
    for c in CELLS:
        if len(cells[c]) != 3:
            raise SystemExit(f"GATE FAILS: {task} {c} has {len(cells[c])} networks")
        for r in cells[c]:
            bad = {**(r.get("differs1") or {}), **(r.get("differs2") or {})}
            if bad:
                raise SystemExit(f"GATE FAILS: {task} {c} {r['seed']} config differs: {bad}")
    print(f"{task}: gate passes")


def do_verdict(reads_json, fig_stem):
    """Mode `verdict`: the gate, the verdicts, the tables and the figure.

    Args:
        reads_json: str, the file the read modes wrote; fig_stem: str, figure path without extension.
    Returns:
        dict task -> verdict.
    """
    R = json.load(open(reads_json))
    V = {}
    for task in [t for t in ("3BFF", "CDDM", "Maze") if t in R]:
        gate(task, R[task])
        v = V[task] = verdict(task, R[task])
        print(f"\n===== {task}, L = {v['L']}: finished {v['complete']}; best single technique {v['best']}"
              f"{'' if v['best_qualifies'] else ' (neither qualifies)'}")
        for k in ("read", "end"):
            print(f"  {k}: " + "; ".join(f"{c} {v['counts'][k][c]} (mean {np.mean(v['counts'][k][c]):.0f}, perf "
                                         f"{np.mean(v['perf'][k][c]):.4f})" for c in CELLS))
            print(f"     every EP above every E and P: {v['stack_count'][k]} (not exceeded: {v['not_exceeded'][k]}); "
                  f"performance {v['perf_ok'][k]}; interference {v['interference'][k]}; ceiling {v['ceiling'][k]}; "
                  f"share of the best's inactive units {v['share'][k]:.2f}; M09 rule: {v['m09'][k]}")
        for c in CELLS:
            print(f"  fall {c}: " + ", ".join(f"{a0:.0f} -> {a1:.0f} ({F:+.0f}, {f:+.3f})" for a0, a1, F, f in v["fall"][c]))
        print(f"  decay: {v['decay']}; STACKS {v['stacks']}; STALLS FURTHER {v['stalls_further']}")
        for c in CELLS:
            for r in R[task][c]:
                print(f"  {c:2s} {r['seed']}: finished {r['finished']}, perf {r.get('r2')}, replacements per check "
                      f"{r.get('repl1')} -> {r.get('repl2')}, deaths {r.get('deaths')}, dims {r.get('dims')}, skipped "
                      f"{r.get('skipped1')}, {r.get('skipped2')}, stats {r.get('stats')}, learned {r.get('learned_update')}")
        print(f"  death rate per 100 active units per {PER[task]}: "
              + "; ".join(f"{c} {np.round(v['death_rate'][c], 3).tolist()}" for c in CELLS))
        print(f"  fewer units die with both: {v['fewer_die']}; added units add dimensions: {v['adds_dims']}")
    print(f"\nstack on {sum(v['stacks'] for v in V.values())} of {len(V)} tasks; stall further on "
          f"{sum(v['stalls_further'] for v in V.values())} of {len(V)}")
    if fig_stem:
        draw(R, fig_stem)
    return V


def draw(R, fig_stem):
    """Active units against training, every network of every cell, one panel per task (PDF and PNG).

    Args:
        R: the reads; fig_stem: str, path without extension.
    Returns:
        None.
    """
    tasks = [t for t in ("3BFF", "CDDM", "Maze") if t in R]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 4.0), sharey=True, constrained_layout=True, squeeze=False)
    for ax, task in zip(axes[0], tasks):
        s = 1 if task == "Maze" else 1000
        for c in CELLS:
            for k, r in enumerate([r for r in R[task][c] if "t" in r]):
                t, a = np.asarray(r["t"]) / s, np.asarray(r["act"])
                w = 1 if task == "Maze" else 5                    # 3BFF, CDDM: every fifth stored vector
                ax.plot(t[::w], a[::w], color=COLORS[c], lw=1.0, alpha=0.9, label=NAMES[c] if k == 0 else None)
        ax.axvline(READ[task] / s, color="0.7", lw=0.8)
        ax.set_title(f"{task}, N = 1,000", fontsize=10)
        ax.set_xlabel("PPO update" if task == "Maze" else "iteration (thousands)")
        ax.set_xlim(0, None)
        ax.set_ylim(0, 1000)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("active units (of 1,000)")
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.savefig(fig_stem + ".pdf")
    fig.savefig(fig_stem + ".png", dpi=150)
    print(f"figure: {fig_stem}.pdf, .png")


def synthetic(task, start, end, perf):
    """Network records with a linear active-count trace, for `selfcheck`.

    Args:
        task: str; start, end: dict cell -> list of three counts at the read point and at the end;
        perf: dict cell -> performance of its networks.
    Returns:
        dict cell -> list of three records.
    """
    L = 4000 if task == "Maze" else 200000
    t = list(range(0, L + 1, 10)) if task == "Maze" else list(range(0, L, 100))
    out = {}
    for c in CELLS:
        out[c] = [dict(seed=i, L=L, finished=True, t=t, act=[int(a if x <= READ[task] else a + (b - a) * (x - READ[task])
                                                                 / (t[-1] - READ[task])) for x in t],
                       r2=dict(read=perf[c], end=perf[c]), deaths=dict(n=1, exposure=1.0),
                       dims=dict(end=dict(pr=1.0)))
                  for i, (a, b) in enumerate(zip(start[c], end[c]))]
    return out


def selfcheck():
    """The verdict on synthetic cells with a known answer (rules, check 4). Raises AssertionError on a wrong one."""
    ok = dict(D=0.95, E=0.95, P=0.95, EP=0.95)
    start = dict(D=[240, 250, 260], E=[380, 400, 450], P=[800, 810, 820], EP=[900, 910, 920])
    end = dict(D=[200, 210, 220], E=[300, 320, 350], P=[740, 750, 760], EP=[890, 900, 910])
    for task in ("3BFF", "CDDM", "Maze"):
        v = verdict(task, synthetic(task, start, end, ok))
        assert v["stacks"] and v["stalls_further"] and v["best"] == "P", (task, v)
        v = verdict(task, synthetic(task, dict(start, EP=start["P"]), dict(end, EP=end["P"]), ok))    # EP a copy of P
        assert not v["stacks"] and not v["stalls_further"] and not any(v["interference"].values()), (task, v)
        v = verdict(task, synthetic(task, dict(start, EP=[100] * 3), dict(end, EP=[90] * 3), ok))     # EP lowest
        assert not v["stacks"] and all(v["interference"].values()), (task, v)
        v = verdict(task, synthetic(task, start, end, dict(ok, EP=0.85)))                             # performance lost
        assert not v["stacks"] and all(v["interference"].values()), (task, v)
        v = verdict(task, synthetic(task, dict(start, P=[960, 970, 980], EP=[965, 975, 985]),
                                    dict(end, P=[960, 970, 980], EP=[965, 975, 985]), ok))            # ceiling
        assert all(v["ceiling"].values()) and not v["stacks"] and v["decay"].startswith("a single"), (task, v)
        v = verdict(task, synthetic(task, start, dict(end, EP=[600, 610, 620]), ok))                  # EP falls faster
        assert v["decay"] == "falls faster" and not v["stacks"], (task, v)
    m = [np.array(x, bool) for x in ([1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0])]
    assert deaths([0, 1, 2, 3, 4], m, 0, 4)["n"] == 1             # unit 1 dies at vector 1; unit 2 flickers
    assert count_increments([0, 5, 9, 0, 2]).tolist() == [0, 5, 4, 0, 2]
    x = np.random.default_rng(0).standard_normal((4, 20000))
    d = dimensionality(np.vstack([x, x[:1]]), np.ones(5, bool))
    assert d["near_copy"] == 0.4 and 3.4 < d["pr"] < 3.7          # a copy: eigenvalues 2, 1, 1, 1 -> 25 / 7
    print("selfcheck passes")


def main():
    """Command line: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("read", "maze-read", "verdict", "selfcheck"))
    ap.add_argument("out_json", nargs="?")
    ap.add_argument("--until", type=int)
    ap.add_argument("--logs", default=os.path.expanduser("~/trainRNNbrain_logs"))
    ap.add_argument("--data")
    ap.add_argument("--fig")
    a = ap.parse_args()
    if a.mode == "read":
        do_read(a.out_json, a.until, a.logs)
    elif a.mode == "maze-read":
        do_maze_read(a.out_json, a.data)
    elif a.mode == "verdict":
        do_verdict(a.out_json, a.fig)
    else:
        selfcheck()


if __name__ == "__main__":
    main()
