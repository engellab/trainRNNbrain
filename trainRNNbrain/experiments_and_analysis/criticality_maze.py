#!/usr/bin/env python3
"""Criticality of Maze networks along training, the Maze gate, and the cross-task reading of lambda1.

RULES: docs/project_trajectory/2026-11-20_1733_rules-criticality-maze.md (the Maze form of the gate of
2026-11-20_1643_rules-criticality-projection.md). Measures from trainRNNbrain/trainer/criticality.py.

WHAT IS MEASURED, per network and weight state (init rebuilt from the saved config and seed, every kept
snapshot, the final weights of every continuation), on the trainer's probe (TrainerPPO.probe: 128 fixed
starts, noise-free, the policy mean as the action), over each episode's LIVE states only (the RNN steps of
the decisions taken before the episode ended; L_b = substeps x decisions taken):
  lyap_*         largest finite-time Lyapunov exponent per RNN step, median and quartiles over episodes
                 with at least MIN_LYAP live states (n_lyap of them)
  gain<k>_*      log10 of the read-out sensitivity at an episode's last live state pulled back k RNN steps,
                 median and quartiles over (output, episode) pairs with at least k + 1 live states
                 (n_gain<k> episodes)
  oprad_med      median spectral radius of the step Jacobian at 12 operating points inside live windows
  gate_on        fraction of units above threshold over live states after the burn-in
  rho_W, smax_W, rownorm_W   spectral radius, largest singular value, mean incoming row norm of W_rec
  active, success           the probe's number of active units and its success rate, from TrainerPPO.probe
  stored_active, stored_success   the training probe's values at the same update (kept snapshots only)
  len_med        median number of live states per episode

Usage (repository root; CPU):
  PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/criticality_maze.py read \\
      --runs <Maze_suite>/N=1000_act=relu_cont/s4 [...] --suite <Maze_suite> \\
      --snaps <checkpoints>/Maze_suite --out <json>
  ... criticality_maze.py gate --json <json> [...]
  ... criticality_maze.py cross --json <maze json> [...] --sup data/criticality/phase1_3bff.json \\
      data/criticality/phase1_cddm.json --gate-rows data/criticality/gate_rows.json
A run given to `read` is the last run of a network's warm-start chain (common.maze_run_chain, parents looked
up under --suite); the chain's first config rebuilds the init and names the snapshot folder.
"""

import argparse
import glob
import json
import math
import os
import sys

import hydra
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import _maze_read_run, maze_learned_index, maze_run_chain
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer import criticality as C
from trainRNNbrain.trainer.TrainerPPO import load_run
from trainRNNbrain.utils import filter_kwargs

LAGS = (10, 50, 100, 250, 500)
H = 500                      # the gate's horizon in RNN steps (100 decisions; rules entry)
FALLBACK_LAGS = (250, 100)   # read instead when fewer than MIN_EPISODES episodes reach H + 1 live states
MIN_EPISODES = 32
BURN_IN = 50
MIN_LYAP = BURN_IN + 100     # live states an episode needs to enter lambda1
MAX_UPDATE = 4000            # the probe record is cut here, as in the claim-gaps Maze read
TANGENT_SEED = 20261120
LAM_BAR = 0.01
G_BAR = math.log10(math.e)


def load_params(rnn, npz_path):
    """Copy a parameter file's arrays into the RNN (W_rec, W_inp, W_out, bias and the rest; y_init).

    Args:
        rnn: RNN_torch; npz_path: str, a *Params_Maze.npz written by run_rl.py.
    Returns:
        None.
    """
    p = np.load(npz_path, allow_pickle=True)
    with torch.no_grad():
        for name, prm in rnn.named_parameters():
            prm.copy_(torch.as_tensor(np.asarray(p[name], dtype=np.float32)))
    rnn.y_init = torch.as_tensor(np.asarray(p["y_init"], dtype=np.float32))


def init_rnn(cfg, seed):
    """The network a saved Maze config describes, freshly initialised (run_rl.py seeds it with the run seed).

    Args:
        cfg: a run's saved config (OmegaConf); seed: int.
    Returns:
        RNN_torch on the CPU.
    """
    args = filter_kwargs(RNN_torch, cfg.model)
    args.seed = int(seed)
    return hydra.utils.instantiate(args)


def set_weights_from(rnn, src):
    """Copy every parameter and y_init of RNN `src` into RNN `rnn` (same architecture).

    Args:
        rnn, src: RNN_torch.
    Returns:
        None.
    """
    sp = dict(src.named_parameters())
    with torch.no_grad():
        for name, prm in rnn.named_parameters():
            prm.copy_(sp[name].detach())
    rnn.y_init = src.y_init.detach().clone()


def probe_pass(trainer):
    """Play the trainer's probe and recompute its states.

    Args:
        trainer: TrainerPPO holding the weights to measure.
    Returns:
        (states, lengths, n_active, out): states (N, T * substeps, B) after every RNN step; lengths (B,)
        long, live states per episode; n_active int, the probe's own count; out, the probe rollout dict.
    """
    n_active, _, _, out = trainer.probe()
    T, k = trainer.T, trainer.substeps
    with torch.no_grad():
        states = trainer.sequence(out["u"], out["noise"], alive=out["alive"])[0][:, :T * k]
    lengths = (out["alive"][:-1].sum(dim=0) * k).round().long()
    return states, lengths, n_active, out


def quartiles(x):
    """25th, 50th and 75th percentiles of the finite entries of x, or three nans if there are none.

    Args:
        x: 1-d array.
    Returns:
        list of three floats.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, a)) for a in (0.25, 0.5, 0.75)] if len(x) else [float("nan")] * 3


def measure(trainer):
    """Every measure of the module docstring on the trainer's current weights.

    Args:
        trainer: TrainerPPO.
    Returns:
        dict of floats and ints.
    """
    rnn = trainer.RNN
    alpha = float(rnn.alpha)
    states, lengths, n_active, out = probe_pass(trainer)
    Lmax = int(lengths.max())
    with torch.no_grad():
        gates = C.relu_gates(states[:, :Lmax])
        W, Wout = rnn.W_rec.detach(), rnn.W_out.detach()[:2]       # the two action rows (turn, speed)
        g = torch.Generator().manual_seed(TANGENT_SEED)
        lam = C.lyapunov_max(W, gates, alpha, burn_in=BURN_IN, generator=g, lengths=lengths).numpy()
        lam_ok = (lengths >= MIN_LYAP).numpy()
        lags = [k for k in LAGS if k < Lmax]
        gains = C.backward_gain(W, gates, alpha, Wout, lags=lags, lengths=lengths) if lags else {}
        B = gates.shape[2]
        pts = [(int(f * (int(lengths[b]) - 1)), b) for b in (0, B // 3, (2 * B) // 3) for f in (0.25, 0.5, 0.75, 1.0)]
        orad = C.operating_radius(W, gates, alpha, pts)
        t = torch.arange(Lmax)[:, None]
        live = ((t >= BURN_IN) & (t < lengths[None, :])).to(gates.dtype)          # (Lmax, B)
        gate_on = float((gates * live[None]).sum() / (live.sum() * gates.shape[0]))
    lq = quartiles(lam[lam_ok])
    row = dict(lyap_q25=lq[0], lyap_med=lq[1], lyap_q75=lq[2],
               lyap_mean=float(np.mean(lam[lam_ok])) if lam_ok.any() else float("nan"), n_lyap=int(lam_ok.sum()),
               oprad_med=float(np.median(orad)), gate_on=gate_on, rho_W=C.spectral_radius(W),
               smax_W=float(torch.linalg.svdvals(W.double())[0]), rownorm_W=float(W.norm(dim=1).mean()),
               active=int(n_active), success=float(out["success"].float().mean()),
               len_med=float(lengths.float().median()))
    for k in LAGS:
        v = gains[k].numpy() if k in gains else np.full(2 * B, np.nan)
        row[f"gain{k}_q25"], row[f"gain{k}_med"], row[f"gain{k}_q75"] = quartiles(v)
        row[f"n_gain{k}"] = int((lengths >= k + 1).sum())
    return row


def joined_probes(chain):
    """The probe record of a warm-start chain, joined in update order and cut at MAX_UPDATE.

    Args:
        chain: list of run folders, oldest first (common.maze_run_chain).
    Returns:
        dict of arrays over probes: update, probe_success, active.
    """
    parts = [_maze_read_run(d)["probes"] for d in chain]
    pr = {k: np.concatenate([p[k] for p in parts]) for k in ("update", "probe_success", "active")}
    keep = pr["update"] <= MAX_UPDATE
    return {k: v[keep] for k, v in pr.items()}


def read_network(run_dir, suite_root, snap_root):
    """All weight states of one Maze network, measured.

    Args:
        run_dir: the last run of the network's chain; suite_root: the folder its parents are looked up
        under; snap_root: the kept-snapshot root holding <cell>/s<seed>/u<update>/.
    Returns:
        list of row dicts (network, seed, task, iteration, kind, and measure()'s keys); the init row also
        carries the learned point and the init-rebuild check.
    """
    chain = maze_run_chain(run_dir, suite_root)
    first = chain[0]
    cfg, _, trainer = load_run(first)
    seed = int(cfg.seed)
    cell = os.path.basename(os.path.dirname(first))
    pr = joined_probes(chain)
    li = maze_learned_index(pr["probe_success"])
    learned = None if li is None else int(pr["update"][li])
    snaps = sorted((int(os.path.basename(d)[1:]), d) for d in glob.glob(os.path.join(snap_root, cell, f"s{seed}", "u*")))
    after = [u for u, _ in snaps if learned is not None and u >= learned]
    base = dict(network=f"{cell}/s{seed}", seed=seed, task="Maze", N=int(cfg.model.N))
    stored = {int(u): (int(a), float(s)) for u, a, s in zip(pr["update"], pr["active"], pr["probe_success"])}

    set_weights_from(trainer.RNN, init_rnn(cfg, seed))
    W_init = trainer.RNN.W_rec.detach().clone()
    rows = [dict(base, iteration=0, kind="init", learned_update=learned,
                 learned_snapshot=after[0] if after else None, **measure(trainer))]
    print(f"{base['network']}: learned at {learned}, read at {rows[0]['learned_snapshot']}", flush=True)
    first_snap = None
    for u, d in snaps:
        load_params(trainer.RNN, glob.glob(os.path.join(d, "*Params_*.npz"))[0])
        first_snap = trainer.RNN.W_rec.detach().clone() if first_snap is None else first_snap
        sa, ss = stored.get(u, (None, None))
        rows.append(dict(base, iteration=u, kind="snapshot", stored_active=sa, stored_success=ss, **measure(trainer)))
        print(f"  u{u}: lam {rows[-1]['lyap_med']:+.4f} G500 {rows[-1]['gain500_med']:+.3f} "
              f"active {rows[-1]['active']} (stored {sa})", flush=True)
    for d in chain[1:]:
        dcfg = _maze_read_run(d)["cfg"]
        it = int(dcfg["trainer"]["iter_offset"]) + int(dcfg["trainer"]["n_updates"])
        if it > MAX_UPDATE:
            break
        load_params(trainer.RNN, glob.glob(os.path.join(d, "*Params_*.npz"))[0])
        sa, ss = stored.get(it, (None, None))
        rows.append(dict(base, iteration=it, kind="final", stored_active=sa, stored_success=ss, **measure(trainer)))
    if first_snap is not None:          # the init-rebuild check of the rules entry
        wrong = init_rnn(cfg, seed + 1).W_rec.detach()
        cos = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()))
        c_ok, c_bad = cos(W_init, first_snap), abs(cos(wrong, first_snap))
        rows[0].update(init_cos_first_snap=c_ok, init_cos_wrong_seed=c_bad, init_ok=bool(c_ok > 20 * c_bad))
    return rows


def gate_network(rows):
    """The Maze gate for one network (rules entry): init against the snapshot at or after the learned point.

    Args:
        rows: the network's rows from read_network.
    Returns:
        dict: it_read, lag (H, or the fallback lag used), lam0, lam_t, g0, g_t (G at that lag), d_lam
        (|lam_t| - |lam0|), d_g (|g_t - g0|), d_decay (|g_t| - |g0|), g100_0, g100_t, cond1, cond2,
        deteriorated, init_ok; or None if the network has no learned point.
    """
    r0 = next(r for r in rows if r["kind"] == "init")
    if r0["learned_snapshot"] is None:
        return None
    rt = next(r for r in rows if r["kind"] == "snapshot" and r["iteration"] == r0["learned_snapshot"])
    lag = next((k for k in (H,) + FALLBACK_LAGS
                if r0[f"n_gain{k}"] >= MIN_EPISODES and rt[f"n_gain{k}"] >= MIN_EPISODES), None)
    d_lam = abs(rt["lyap_med"]) - abs(r0["lyap_med"])
    g0, gt = (r0[f"gain{lag}_med"], rt[f"gain{lag}_med"]) if lag else (float("nan"), float("nan"))
    d_g = abs(gt - g0)
    return dict(it_read=rt["iteration"], lag=lag, lam0=r0["lyap_med"], lam_t=rt["lyap_med"], g0=g0, g_t=gt,
                d_lam=d_lam, d_g=d_g, d_decay=abs(gt) - abs(g0), g100_0=r0["gain100_med"],
                g100_t=rt["gain100_med"], cond1=d_lam > LAM_BAR, cond2=d_g > G_BAR,
                deteriorated=bool(d_lam > LAM_BAR and d_g > G_BAR), init_ok=r0.get("init_ok"),
                n_lyap=(r0["n_lyap"], rt["n_lyap"]), n_gain=(r0[f"n_gain{H}"], rt[f"n_gain{H}"]))


def cmd_gate(a):
    """Print one line per network and the Maze verdict."""
    rows = [r for f in a.json for r in json.load(open(f))]
    verdicts = []
    for net in sorted({r["network"] for r in rows}):
        g = gate_network([r for r in rows if r["network"] == net])
        if g is None:
            print(f"{net}: no learned point (excluded)")
            continue
        verdicts.append(g["deteriorated"])
        print(f"{net:22s} read@u{g['it_read']} lam {g['lam0']:+.4f}->{g['lam_t']:+.4f} "
              f"(d|lam| {g['d_lam']:+.4f} > {LAM_BAR}: {g['cond1']}) G{g['lag']} {g['g0']:+.3f}->{g['g_t']:+.3f} "
              f"(|dG| {g['d_g']:.3f} > {G_BAR:.3f}: {g['cond2']}; decay change {g['d_decay']:+.3f}) "
              f"G100 {g['g100_0']:+.3f}->{g['g100_t']:+.3f} n_lyap {g['n_lyap']} n_gain{H} {g['n_gain']} "
              f"deteriorated={g['deteriorated']} init_ok={g['init_ok']}")
    need = math.ceil(2 * len(verdicts) / 3)
    print(f"Maze: {sum(verdicts)} of {len(verdicts)} deteriorated (need {need}) -> gate "
          f"{'PASSES' if sum(verdicts) >= need else 'FAILS'}")


def cmd_cross(a):
    """Print lambda1 at init and at the learned point on all three tasks, and the descriptive ordering."""
    sup = [r for f in a.sup for r in json.load(open(f))]
    it_read = json.load(open(a.gate_rows))
    maze = [r for f in a.json for r in json.load(open(f))]
    vals = {}
    for net, it in sorted(it_read.items()):
        R = [r for r in sup if r["network"] == net]
        r0 = next(r for r in R if r["kind"] == "init")
        rt = next(r for r in R if r["iteration"] == it and r["kind"] != "init")
        task = {"NBitFlipFlop": "3BFF"}.get(r0["task"], r0["task"])
        vals.setdefault(task, []).append((r0["seed"], r0["lyap_med"], rt["lyap_med"], it))
    for net in sorted({r["network"] for r in maze}):
        R = [r for r in maze if r["network"] == net]
        r0 = next(r for r in R if r["kind"] == "init")
        if r0["learned_snapshot"] is None:
            continue
        rt = next(r for r in R if r["kind"] == "snapshot" and r["iteration"] == r0["learned_snapshot"])
        vals.setdefault("Maze", []).append((r0["seed"], r0["lyap_med"], rt["lyap_med"], rt["iteration"]))
    for task, v in vals.items():
        for seed, l0, lt, it in v:
            print(f"{task:5s} s{seed:<11d} lam init {l0:+.4f}  learned point ({it}) {lt:+.4f}")
    mz = [abs(x[2]) for x in vals.get("Maze", [])]
    for task in ("3BFF", "CDDM"):
        other = [abs(x[2]) for x in vals.get(task, [])]
        if not mz or not other:
            continue
        rel = ("closer to 0 than" if max(mz) < min(other) else
               "farther from 0 than" if min(mz) > max(other) else "overlaps with")
        print(f"Maze |lam| {min(mz):.4f}-{max(mz):.4f} vs {task} {min(other):.4f}-{max(other):.4f}: Maze {rel} {task}")


def main():
    """Parse the subcommand (read, gate, cross) and run it."""
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--runs", nargs="+", required=True)
    r.add_argument("--suite", required=True)
    r.add_argument("--snaps", required=True)
    r.add_argument("--out", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--json", nargs="+", required=True)
    c = sub.add_parser("cross")
    c.add_argument("--json", nargs="+", required=True)
    c.add_argument("--sup", nargs="+", required=True)
    c.add_argument("--gate-rows", required=True)
    a = ap.parse_args()
    if a.cmd == "gate":
        return cmd_gate(a)
    if a.cmd == "cross":
        return cmd_cross(a)
    torch.set_grad_enabled(False)
    rows = []
    for run in a.runs:
        rows += read_network(run, a.suite, a.snaps)
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
