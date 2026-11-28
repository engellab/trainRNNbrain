#!/usr/bin/env python3
"""The entropy set point at lambda_ent = 0.1 in Figure 5: unit removal, pre-activations and the figure's data file.

RULES. docs/project_trajectory/2026-11-25_1235_rules-entropy-removal-and-preactivation.md, committed before any measurement.

MODE `removal` (Della GPU node). a3_remedy_ablation.ablate_net on the three 3BFF and three CDDM networks with
lambda_ent = 0.1 at 75,000 iterations, k against the same controls as Figure 5d (3BFF: the three controls of A3;
CDDM: the five Figure 3a controls of G3's rate_75k group), and on the gate networks of the rules (the three A3
controls, one A3 prune-and-duplicate network, one G3 rate-penalty network). Writes OUT_DIR/removal_supervised.json.
The Maze removal is rl_maze_ablate.py --added 430, one process per network: the entropy networks into
OUT_DIR/maze, the gate network (prune-and-duplicate s1) into OUT_DIR/maze_gate (slurm/entropy_fig5_della.slurm).

MODE `preact` (Della GPU node). Per network and separately over its inactive and active units, the actual
on-fraction of h, its binary entropy and the skewness of h, with Phi(mu/s) and H_i beside them, for the nine entropy
networks and their controls. Writes OUT_DIR/preactivation.json (summaries and, per unit, the values behind them in
OUT_DIR/preactivation_units.npz).

MODE `summary` (Mac). The gates, then M09's reading per task, from OUT_DIR's removal files. Writes
data/entropy_fig5/removal.json (the rows Figure 5d draws) and prints the table.

MODE `export` (Mac). The Figure 5 cells of the entropy term at lambda_ent = 0.01 and 0.1 from the three stored reads
(entropy_setpoint_reads.json, cddm_reads.json, maze_reads.json), written to data/entropy_fig5/cells.json.

Usage (repository root on PYTHONPATH):
  python entropy_fig5.py removal OUT_DIR --a3 A3_N1000_JSON
  python entropy_fig5.py preact OUT_DIR --maze-root DIR
  python entropy_fig5.py summary OUT_DIR
  python entropy_fig5.py export --reads3bff JSON --readscddm JSON --readsmaze JSON
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a3_remedy_ablation as A3  # noqa: E402
import heldout_r2 as H  # noqa: E402
from common import DATA_DIR, active_mask, warm_start_chain  # noqa: E402
from criticality_read import build_rnn, load_params  # noqa: E402
from round18_g1g2 import CTRL_CELL, state_at  # noqa: E402
from round18_removal_cddm import DEPENDS, INDEPENDENT, OUT as G3_JSON, params_file, runs, verdict  # noqa: E402
from trainRNNbrain.trainer.TrainerPPO import load_run  # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EF_DIR = os.path.join(REPO, "data", "entropy_fig5")
CELLS_JSON = os.path.join(EF_DIR, "cells.json")
REMOVAL_JSON = os.path.join(EF_DIR, "removal.json")
ARM = "EqType=h_N=1000_arm=ent0.1"
FF_ARM, CDDM_ARM = f"NBitFlipFlop_entropy_setpoint/{ARM}", f"CDDM_entropy_setpoint/{ARM}"
CDDM_CTRL_75K = "CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0"      # k on CDDM (G3, rate_75k)
CDDM_CTRL_CRIT = "CDDM_paper_grid/EqType=h_N=1000_arm=control"                     # the criterion's controls (100,000)
CDDM_GATE = "CDDM_std_g0_penalties_bias0/EqType=h_N=1000_pen=frm0.1_iters=75000"   # G3 network reproduced by the gate
MAZE_CTRL_MEAN = 430.0                  # M09: the mean of the three Maze controls
MAZE_CTRL_ACTIVE = (426, 449, 415)      # the criterion's Maze controls, seeds 1-3 (fig_paper_F5_tasks.load_maze)
MAZE_INDEPENDENT = 0.05                 # the criterion's Maze tolerance (round 18, G4)
GATE_R2, GATE_MAZE, GATE_MAZE_ACTIVE = 0.002, 0.03, 2   # the rules' environment gates
NOISE_SEED = 20261121                   # the noisy pass of entropy_setpoint_read.unit_stats
MAZE_EPISODES = 256
TARGET = 0.75
LAMBDAS = ("ent0.01", "ent0.1")


def binary_entropy(q):
    """Binary entropy in bits of on-fractions q (array in [0, 1]); 0 at 0 and 1."""
    q = np.clip(np.asarray(q, float), 1e-300, 1.0)
    p = np.clip(1.0 - np.asarray(q, float), 1e-300, 1.0)
    return -(q * np.log2(q) + p * np.log2(p))


def gauss_h(z):
    """H_i in bits from z = mu/s: the binary entropy of Phi(z), computed in log space.

    Args:
        z: torch double tensor (N,).
    Returns:
        numpy array (N,).
    """
    lp, lq = torch.special.log_ndtr(z), torch.special.log_ndtr(-z)
    return (-(lp.exp() * lp + lq.exp() * lq) / np.log(2.0)).cpu().numpy()


def unit_moments(x):
    """Per-unit pre-activation statistics of one noisy pass.

    Args:
        x: torch tensor (N, samples), the pre-activations h of every unit at every kept step.
    Returns:
        dict of (N,) numpy arrays: on (share of samples with h > 0), H_actual (its binary entropy, bits),
        skew (third standardised moment of h, population), z (mu/s), Phi (Phi(mu/s)), H_gauss (H_i, bits).
    """
    x = x.double()
    mu, s = x.mean(1), x.std(1)
    c = x - mu[:, None]
    on = (x > 0).double().mean(1).cpu().numpy()
    skew = ((c ** 3).mean(1) / (c.pow(2).mean(1)) ** 1.5).cpu().numpy()
    z = mu / s
    return dict(on=on, H_actual=binary_entropy(on), skew=skew, z=z.cpu().numpy(),
                Phi=torch.special.ndtr(z).cpu().numpy(), H_gauss=gauss_h(z))


def class_summary(u, on_clean, sel):
    """Medians (and quartiles of the three main quantities) over the units of one class.

    Args:
        u: unit_moments output; on_clean: (N,) noise-free on-fractions; sel: (N,) bool.
    Returns:
        dict of floats (empty class: only n).
    """
    out = dict(n=int(sel.sum()))
    if not sel.any():
        return out
    for k in ("on", "H_actual", "skew", "Phi", "H_gauss", "z"):
        q = np.nanpercentile(u[k][sel], [25, 50, 75])
        out[k] = float(q[1])
        out[f"{k}_q25"], out[f"{k}_q75"] = float(q[0]), float(q[2])
    out["on_clean"] = float(np.median(on_clean[sel]))
    out["never_on_noisy"] = float(np.mean(u["on"][sel] == 0))
    out["share_gauss_near"] = float(np.mean(np.abs(u["H_gauss"][sel] - TARGET) < 0.05))
    out["share_actual_near"] = float(np.mean(np.abs(u["H_actual"][sel] - TARGET) < 0.05))
    return out


def stored_mask(run_dir, it):
    """Active mask at the stored participation vector nearest iteration `it`, along the warm-start chain.

    Args:
        run_dir: str; it: int, the iteration of the weights.
    Returns:
        ((N,) bool mask, iteration of the vector).
    """
    best = None
    for d in warm_start_chain(run_dir):
        for f in glob.glob(os.path.join(d, "*ParticipationTrace.pkl")):
            with open(f, "rb") as fh:
                tr = pickle.load(fh)
            for t, p in zip(tr["participation_iters"], np.asarray(tr["participation"])):
                if best is None or abs(t - it) < abs(best[1] - it):
                    best = (active_mask(p), float(t))
    return best


def supervised_passes(state_dir, device):
    """Noisy and noise-free pre-activations of a supervised network on its held-out batch (unit_stats' pass).

    Args:
        state_dir: folder holding the config and weights; device: torch.device.
    Returns:
        (noisy, clean): tensors (N, T * trials).
    """
    cfg = OmegaConf.load(sorted(glob.glob(os.path.join(state_dir, "*_config.yaml")))[0])
    rnn = build_rnn(cfg, seed=int(cfg.seed))
    load_params(rnn, H._params(state_dir)[0])
    if hasattr(rnn, "sigma_w"):
        rnn.sigma_w = 0.0
    bi = H.heldout_batch(cfg)[0].to(rnn.device)
    rnn.random_generator.manual_seed(NOISE_SEED)
    with torch.no_grad():
        hn, _ = rnn(bi, w_noise=True)
        h0, _ = rnn(bi, w_noise=False)
    return hn.reshape(hn.shape[0], -1), h0.reshape(h0.shape[0], -1)


def maze_passes(folder, device):
    """Noisy (training rollout) and noise-free (probe) pre-activations of a Maze network, steps before the episode's end.

    Args:
        folder: run folder; device: torch.device.
    Returns:
        (noisy, clean, success): tensors (N, live steps); success of the noisy rollout (float).
    """
    _, task, tr = load_run(folder, device)
    g = torch.Generator(device=task.device)
    g.manual_seed(NOISE_SEED)
    torch.manual_seed(NOISE_SEED)
    out = tr.rollout(MAZE_EPISODES, tr.T, clean=False, generator=g)
    _, _, _, pout = tr.probe()
    res = []
    for o in (out, pout):
        with torch.no_grad():
            st = tr.sequence(o["u"], o["noise"], o["w_seed"], o["outgoing"], alive=o["alive"])[0][:, :tr.T * tr.substeps]
        res.append(st[:, tr.live_steps(o["alive"][:-1])])
    return res[0], res[1], float(out["success"].float().mean())


def do_preact(out_dir, maze_root):
    """Mode `preact`: the pre-activation statistics of the nine entropy networks and their controls.

    Args:
        out_dir: str; maze_root: str, folder that holds trained_RNNs_rl_maze* trees; its Maze_suite must be the
            Mac copy of the controls (426, 449 and 415 active units), not Della's own (455, 423 and 404).
    Returns:
        None.
    Raises:
        SystemExit if a Maze control is not the criterion's network.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recs, units = [], {}

    def one(task, group, name, noisy, clean, mask, probe, weights, extra=None):
        """Summarise one network and store it."""
        u = unit_moments(noisy)
        on_clean = (clean > 0).double().mean(1).cpu().numpy()
        r = dict(task=task, group=group, name=name, probe=probe, weights=weights, n_active=int(mask.sum()),
                 inactive=class_summary(u, on_clean, ~mask), active=class_summary(u, on_clean, mask), **(extra or {}))
        recs.append(r)
        key = f"{task}|{group}|{name}"
        units.update({f"{key}|{k}": u[k] for k in ("on", "H_actual", "skew", "Phi", "H_gauss")})
        units[f"{key}|active"] = mask
        print(f"{key}: active {r['n_active']} (vector {probe}) weights {weights}", flush=True)
        for c in ("inactive", "active"):
            s = r[c]
            if s["n"]:
                print(f"   {c:8s} n={s['n']:4d} on {s['on']:.4f} H_actual {s['H_actual']:.3f} skew {s['skew']:+.2f} "
                      f"Phi {s['Phi']:.3f} H_gauss {s['H_gauss']:.3f} near(gauss) {s['share_gauss_near']:.2f} "
                      f"near(actual) {s['share_actual_near']:.2f}", flush=True)

    groups = [("3BFF", "ent0.1", FF_ARM, 75000), ("3BFF", "control", CTRL_CELL, 75000),
              ("CDDM", "ent0.1", CDDM_ARM, 75000), ("CDDM", "control (criterion, 100,000)", CDDM_CTRL_CRIT, 100000),
              ("CDDM", "control (Fig. 3a, 75,000)", CDDM_CTRL_75K, 75000)]
    for task, group, cell, it in groups:
        for d in runs(cell):
            st = state_at(d, it) or (d if cell in (CDDM_CTRL_75K, CDDM_CTRL_CRIT) else None)
            if st is None:
                print(f"{task} {group} {os.path.basename(d)[:40]}: no weights at {it}", flush=True)
                continue
            noisy, clean = supervised_passes(st, device)
            mask, probe = stored_mask(d, it)
            one(task, group, os.path.basename(d).split(";L=")[0], noisy, clean, mask, probe, st)
    for group, pattern in (("ent0.1", "trained_RNNs_rl_maze_entsp_N1000/Maze_interventions/ent0.1/s{}"),
                           ("control", "trained_RNNs_rl_maze/Maze_suite/N=1000_act=relu/s{}")):
        for s in (1, 2, 3):
            folder = os.path.join(maze_root, pattern.format(s))
            noisy, clean, succ = maze_passes(folder, device)
            z = np.load(sorted(glob.glob(os.path.join(folder, "*Participation.npz")))[0])
            j = int(np.argmin(np.abs(z["p_update"] - 2000)))
            n = int(active_mask(z["p"][j]).sum())
            # Della's trained_RNNs_rl_maze/Maze_suite/N=1000_act=relu holds another training run of the same seeds
            if group == "control" and n != MAZE_CTRL_ACTIVE[s - 1]:
                raise SystemExit(f"{folder}: {n} active units, the criterion's control has {MAZE_CTRL_ACTIVE[s - 1]}")
            one("maze", group, f"s{s}", noisy, clean, active_mask(z["p"][j]), float(z["p_update"][j]), folder,
                dict(success_noisy_rollout=succ))
    with open(os.path.join(out_dir, "preactivation.json"), "w") as fh:
        json.dump(recs, fh, indent=1)
    np.savez_compressed(os.path.join(out_dir, "preactivation_units.npz"), **units)
    print(f"{len(recs)} networks -> {out_dir}/preactivation.json")


def do_removal(out_dir, a3_json):
    """Mode `removal`: M09's removal on the six supervised entropy networks and the gate networks.

    Args:
        out_dir: str; a3_json: str, A3's stored record (data/remedy_reads_31oct/a3_N1000.json).
    Returns:
        None.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(a3_json) as fh:
        a3 = json.load(fh)
    with open(G3_JSON) as fh:
        g3 = json.load(fh)["rate_75k"]
    rel = lambda p: p.split("trained_RNNs/")[-1]
    n_ctrl = {"3BFF": float(np.mean([r["n_active"] for r in a3 if r["family"] == "control"])), "CDDM": g3["n_ctrl"]}
    out = dict(n_ctrl=n_ctrl, gate=[], arms={})
    dup = next(r for r in a3 if r["family"] == "duplicate")
    frm = g3["arms"]["rate penalty 0.1"]["runs"][0]
    gate = [("3BFF", r, None) for r in a3 if r["family"] == "control"] + [("3BFF", dup, n_ctrl["3BFF"])]
    gate.append(("CDDM", frm, n_ctrl["CDDM"]))
    for task, ref, n in gate:
        d = os.path.join(DATA_DIR, rel(ref["run_dir"]))
        pf = params_file(d, dict(last_params=True)) if task == "CDDM" else None
        out["gate"].append(dict(task=task, ref={k: v for k, v in ref.items() if k in
                                                ("run_dir", "n_active", "r2_full", "r2_no_inactive", "r2_recruited")},
                                new=A3.ablate_net(d, n, 75000, pf)))
        print("gate", task, os.path.basename(d)[:40], "done", flush=True)
    for task, cell in (("3BFF", FF_ARM), ("CDDM", CDDM_ARM)):
        out["arms"][task] = [A3.ablate_net(d, n_ctrl[task], 75000) for d in runs(cell)]
        for r in out["arms"][task]:
            print(f"{task} active {r['n_active']} k {r['surplus']} intact {r['r2_full']:.4f} -k "
                  f"{r.get('r2_recruited', np.nan):.4f} -random {np.median(r.get('r2_random_active', [np.nan])):.4f}",
                  flush=True)
    with open(os.path.join(out_dir, "removal_supervised.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def reading(drops, independent):
    """M09's reading of one task's drops: depends, does not depend or mixed.

    Args:
        drops: list of float; independent: float, the criterion's tolerance on the task.
    Returns:
        str.
    """
    if independent == INDEPENDENT:
        return verdict(drops)
    if drops and all(x >= DEPENDS for x in drops):
        return "depends"
    return "does not depend" if drops and all(x < independent for x in drops) else "mixed"


def do_summary(out_dir):
    """Mode `summary`: check the gates, print M09's reading per task, write data/entropy_fig5/removal.json.

    Args:
        out_dir: str, the folder of removal_supervised.json and maze/.
    Returns:
        None.
    Raises:
        SystemExit if a gate fails or a network is missing.
    """
    with open(os.path.join(out_dir, "removal_supervised.json")) as fh:
        S = json.load(fh)
    bad = 0
    for g in S["gate"]:
        a, b = g["ref"], g["new"]
        diffs = {k: abs(a[k] - b[k]) for k in ("r2_full", "r2_no_inactive", "r2_recruited") if k in a and k in b}
        ok = a["n_active"] == b["n_active"] and all(v <= GATE_R2 for v in diffs.values())
        bad += not ok
        print(f"gate {g['task']} {os.path.basename(a['run_dir'])[:34]}: active {a['n_active']} vs {b['n_active']}, "
              + ", ".join(f"|d {k}| {v:.4f}" for k, v in diffs.items()) + (" ok" if ok else " FAILS"))
    with open(os.path.join(out_dir, "maze_gate", "duplicate__s1.json")) as fh:
        m = json.load(fh)
    with open(os.path.join(REPO, "data", "round19", "fz_maze_added", "duplicate__s1.json")) as fh:
        ref = json.load(fh)
    diffs = {k: abs(m[k] - ref[k]) for k in ("score_full", "score_no_inactive", "score_added")}
    ok = abs(m["n_active"] - ref["n_active"]) <= GATE_MAZE_ACTIVE and all(v <= GATE_MAZE for v in diffs.values())
    bad += not ok
    print(f"gate maze duplicate s1: active {ref['n_active']} vs {m['n_active']}, "
          + ", ".join(f"|d {k}| {v:.4f}" for k, v in diffs.items()) + (" ok" if ok else " FAILS"))
    rows = []
    for task, rs in S["arms"].items():
        for i, r in enumerate(rs, 1):
            rows.append(dict(task=task, seed=os.path.basename(r["run_dir"]).split("seed=")[1].split(";")[0],
                             name=f"entropy {task} network {i}", n_active=r["n_active"], k=r["surplus"],
                             intact=r["r2_full"], no_inactive=r["r2_no_inactive"], least=r["r2_recruited"],
                             random=float(np.median(r["r2_random_active"])), keep=r["r2_keep_ctrl"],
                             random_keep=float(np.median(r["r2_random_keep"]))))
    for s in (1, 2, 3):
        with open(os.path.join(out_dir, "maze", f"ent0.1__s{s}.json")) as fh:
            r = json.load(fh)
        rows.append(dict(task="maze", seed=s, name=f"ent0.1__s{s}", n_active=r["n_active"], k=r["k"],
                         intact=r["score_full"], no_inactive=r["score_no_inactive"], least=r["score_added"],
                         random=float(np.median(r["score_random_active"])), keep=r["score_keep_ctrl"],
                         random_keep=float(np.median(r["score_random_keep"]))))
    print(f"{'network':>22} {'act':>4} {'k':>4} {'intact':>7} {'-inact':>7} {'-k least':>8} {'-k rand':>8} {'drop':>7}")
    for w in rows:
        w["drop"] = w["intact"] - w["least"]
        print(f"{w['name']:>22} {w['n_active']:>4} {w['k']:>4} {w['intact']:7.4f} {w['no_inactive']:7.4f} "
              f"{w['least']:8.4f} {w['random']:8.4f} {w['drop']:7.4f}")
    readings = {}
    for task in ("3BFF", "CDDM", "maze"):
        mine = [w for w in rows if w["task"] == task]
        if len(mine) != 3:
            raise SystemExit(f"{task}: {len(mine)} networks, expected 3")
        drops = [w["drop"] for w in mine]
        readings[task] = dict(drops=drops, reading=reading(drops, MAZE_INDEPENDENT if task == "maze" else INDEPENDENT),
                              random_lower=sum(w["random"] < w["least"] for w in mine),
                              drops_ge_010=sum(d >= DEPENDS for d in drops))
        print(f"{task}: drops {np.round(drops, 4).tolist()} -> {readings[task]['reading']}; random lower in "
              f"{readings[task]['random_lower']} of 3")
    print(f"drops of at least {DEPENDS}: {sum(r['drops_ge_010'] for r in readings.values())} of {len(rows)}; "
          f"random lower in {sum(r['random_lower'] for r in readings.values())} of {len(rows)}")
    if bad:
        raise SystemExit(f"{bad} gate(s) FAIL: no file written")
    os.makedirs(EF_DIR, exist_ok=True)
    with open(REMOVAL_JSON, "w") as fh:
        json.dump(dict(rows=rows, readings=readings, n_ctrl=dict(S["n_ctrl"], maze=MAZE_CTRL_MEAN),
                       source="entropy_fig5.py removal and rl_maze_ablate.py --added 430"), fh, indent=1)
    print(f"-> {REMOVAL_JSON}")


def do_export(f3, fc, fm):
    """Mode `export`: the entropy term's Figure 5 cells per task and lambda, from the three stored reads.

    Args:
        f3, fc, fm: str, the 3BFF, CDDM and Maze reads JSON files.
    Returns:
        None.
    """
    R3, RC, RM = (json.load(open(f)) for f in (f3, fc, fm))
    out = dict(sources=dict(BFF3=os.path.basename(f3), CDDM=os.path.basename(fc), maze=os.path.basename(fm)), cells={})
    c3 = R3["control"]
    ctrl = {"3BFF": ([r["active_75k"] for r in c3], [r["r2_75k"] for r in c3 if r["r2_75k"] is not None], None),
            "CDDM": ([r["active_75k"] for r in RC["control"]], [r["r2_75k"] for r in RC["control"]], None),
            "maze": ([x["active_end"] for x in RM["1000"]["control"]],
                     [x["success_end"] for x in RM["1000"]["control"]], None)}
    for arm in LAMBDAS:
        lab = arm[3:]
        rows = {"3BFF": [(r["active_75k"], r["r2_75k"], True, r["run_dir"].split("seed=")[1].split(";")[0]) for r in R3[arm]],
                "CDDM": [(r["active_75k"], r["r2_75k"], True, r["seed"]) for r in RC[arm]],
                "maze": [(x["active_end"], x["success_end"], x["learned_update"] is not None, x["seed"]) for x in RM["1000"][arm]]}
        for task, rs in rows.items():
            ca, cp, _ = ctrl[task]
            out["cells"][f"{task}|{lab}"] = dict(task=task, label=lab, active=[r[0] for r in rs], perf=[r[1] for r in rs],
                                                 learned=[r[2] for r in rs], seeds=[str(r[3]) for r in rs],
                                                 ctrl_active=ca, ctrl_perf=cp, n_expected=3)
    os.makedirs(EF_DIR, exist_ok=True)
    with open(CELLS_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    for k, c in out["cells"].items():
        print(k, c["active"], np.round(c["perf"], 4).tolist(), "mean", round(float(np.mean(c["active"])), 1))
    print(f"-> {CELLS_JSON}")


def main():
    """Parse the command line and run the selected mode."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("removal", "preact", "summary", "export"))
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--a3")
    ap.add_argument("--maze-root")
    ap.add_argument("--reads3bff")
    ap.add_argument("--readscddm")
    ap.add_argument("--readsmaze")
    a = ap.parse_args()
    if a.mode == "removal":
        do_removal(a.out_dir, a.a3)
    elif a.mode == "preact":
        do_preact(a.out_dir, a.maze_root)
    elif a.mode == "summary":
        do_summary(a.out_dir)
    else:
        do_export(a.reads3bff, a.readscddm, a.readsmaze)


if __name__ == "__main__":
    main()
