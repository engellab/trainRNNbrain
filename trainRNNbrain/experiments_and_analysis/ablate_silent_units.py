#!/usr/bin/env python3
"""
CLAIM 1, SECOND HALF: the silent units of a trained network do no work.

QUESTION. A trained standard network leaves most of its units silent. Do those units carry any of the
computation, or is the task done entirely by the few that work? Remove units, re-run the network on
held-out inputs, and score it.

METHOD, one network at a time:
  1. Build the network from its own saved config and copy in only the learned tensors (the rule of
     the 2026-09-24 reload entry: the parameter file carries no architecture). Assert it is the
     standard network: ReLU, equation h, bias fixed at 0, self-connections on, no Dale, gamma 0.
  2. Order the units by participation p = std(r) + q0.9(|r|), from a noise-free pass over a full
     training-distribution batch - the same probe and the same definition as the logged trace.
     Units with p exactly 0 (CDDM, DMTS) are tie-broken by their peak pre-activation, deepest below
     threshold first. Active = p >= 0.05 q95(p).
  3. Remove a unit by zeroing its OUTGOING weights, its column of W_rec and of W_out. In the h
     dynamics a unit reaches the rest of the system only through its rate, so this is exactly the
     unit clamped to zero rate everywhere downstream, its own output noise included. Its incoming
     weights are left alone: zeroing them would still let it send relu(noise).
  4. Score held-out r2 exactly as heldout_r2.py does (unseen inputs, the training noise on, sigma_w
     = 0), averaged over DRAWS noise draws. The held-out batch is drawn once per network and the
     noise draws are seeded identically at every point, so the curves differ only by what was
     removed, not by a fresh noise sample.
  5. Three orderings, each removing nested sets of k units: quietest first, busiest first, and
     RANDOM random orders. Quietest first should hold the score until every silent unit is gone and
     then collapse; busiest first should collapse at once.

TWO READ-OUTS.
  end    the run's own end state (LastParams). Every standard control on disk has this.
  floor  the weights where the network's loss first reaches 1.05x its own fitted floor - the
         project's read-out. Only runs trained with `+trainer.checkpoint_keep=true` have weights
         there (slurm/SilentReLU_c1_snapshots_della.slurm). The floor is fitted, with
         pr_matrix.fit_floor_at / excess_time_at at delta = 0.05, on TWO series, and the snapshot
         nearest each usable crossing is ablated separately (readout floor_clean / floor_heldout):
           clean    the trace's noise-free training-batch loss, the project's standard series;
           heldout  the held-out loss of every kept snapshot under the training noise, with the
                    same seeded draws - the objective these networks actually minimise.
         A crossing is unusable when the fit sits on the 1e-6 bound, the loss never reaches the
         bar, or the smoothed loss ends more than 1.5x above its minimum. Measured on the existing
         standard controls (2026-10-12): on CDDM both series are usable, the noisy one crossing
         later (25-32k against 8-32k); on the flip-flop the clean loss climbs 3-4x after its
         minimum and the noisy one is still falling at 75k (no crossing in 9 of 16 runs); on DMTS
         both climb back in most runs.

GATES, written into every result and printed: the unablated network's score with its own noise on a
training-distribution batch against the score in its folder name (end read-out, +-0.03), and the
offline active count against the trace's count at the nearest stored probe.

Usage (on a GPU node; the Mac has no environment):
  python ablate_silent_units.py end   <run_dir>              [--out DIR]
  python ablate_silent_units.py floor <run_dir> <snap_dir>   [--out DIR]
  python ablate_silent_units.py summary                      [--out DIR]   (one row per cell)
    run_dir   a finished run folder (holds *_config.yaml, *LastParams*, *ParticipationTrace.pkl)
    snap_dir  its kept-snapshot folder: <checkpoint_to>/<Task>_<tag>/s<seed>/ holding it<iters>/
Output: one JSON per network and read-out in --out (default /scratch/gpfs/TENGEL/pt1290/c1_ablation).
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
import zlib

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import heldout_r2 as H                                    # noqa: E402  (also puts the repo on sys.path)
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask  # noqa: E402
from trainRNNbrain.trainer.Trainer import Trainer, scored_  # noqa: E402
from trainRNNbrain.utils import import_any, filter_kwargs  # noqa: E402

DRAWS = 9                    # noise draws per score, as heldout_r2.DRAWS
RANDOM_ORDERS = 5            # independent random removal orders per network
SILENT_REL = 0.05            # active = p >= SILENT_REL * q95(p), the project's scale-free rule
FLOOR_DELTA = 0.05           # read at 1.05x the fitted floor (tightened from 0.07 on 2026-10-12)
MAX_LATE_RISE = 1.5          # smoothed end / smoothed minimum above this: no usable floor
NOISE_SEED = 1000            # draw d of every score uses generator seed NOISE_SEED + d
BATCH_SEED = 20261012        # seeds the held-out and ordering batches
OUT = "/scratch/gpfs/TENGEL/pt1290/c1_ablation"


# ------------------------------------------------------------------------------------------------
# building and running a network
# ------------------------------------------------------------------------------------------------

def load_params(path):
    """Learned tensors from a parameter file (.npz or .json), as a dict of name -> array."""
    if path.endswith(".json"):
        from trainRNNbrain.utils import unjsonify
        with open(path) as fh:
            return {k: np.asarray(v) for k, v in unjsonify(json.load(fh)).items()}
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def build(run_dir, params):
    """The standard network of `run_dir`, built from its config, carrying `params`.

    Args:
        run_dir: a finished run folder (its *_config.yaml defines the architecture).
        params: dict of learned tensors (W_rec, W_inp, W_out, y_init).
    Returns:
        (cfg, rnn) with rnn in eval-ready state on its own device.
    """
    cfg = OmegaConf.load(sorted(glob.glob(os.path.join(run_dir, "*_config.yaml")))[0])
    m = cfg.model
    standard = (m.equation_type == "h" and not m.dale and not m.io_nonnegativity
                and bool(m.self_connections) and float(m.gamma) == 0
                and [float(b) for b in m.bias_range] == [0.0, 0.0]
                and m.activation_args.name == "relu")
    if not standard:
        raise SystemExit(f"FATAL: {run_dir} is not the standard network")
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    if "_target_" in tcfg:
        del tcfg._target_
    rnn = hydra.utils.instantiate(filter_kwargs(import_any(cfg.model._target_),
                                                OmegaConf.merge(OmegaConf.create(cfg.model), tcfg)))
    set_params(rnn, params)
    if hasattr(rnn, "sigma_w"):
        rnn.sigma_w = 0.0
    return cfg, rnn


def set_params(rnn, params):
    """Copy the learned tensors (and the learned initial state) into a built network, in place."""
    with torch.no_grad():
        for nm, prm in rnn.named_parameters():
            if nm in params:
                prm.copy_(torch.from_numpy(np.asarray(params[nm], dtype=np.float32)))
        if "y_init" in params and getattr(rnn, "y_init", None) is not None:
            rnn.y_init = torch.as_tensor(np.asarray(params["y_init"], dtype=np.float32),
                                         device=rnn.device)


def seeded_batches(cfg, device):
    """The held-out batch and a full training-distribution batch, both seeded.

    Returns:
        dict with 'ho' = (inputs, targets, mask) unseen by the network, and 'tr' = (inputs, targets,
        mask) from the training distribution at its training batch size.
    """
    np.random.seed(BATCH_SEED)
    torch.manual_seed(BATCH_SEED)
    bi, bt, mask = H.heldout_batch(cfg)
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    ti, tt, _ = task.get_batch()
    mask_t = torch.as_tensor(np.asarray(mask), device=device)
    to = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    return {"ho": (to(bi), to(bt), mask_t), "tr": (to(ti), to(tt), mask_t)}


def participation(states, chunk=256):
    """Per-unit p = std(r) + q0.9(|r|) over (time, trials) of noise-free states, as the trace does.

    Mirrors Trainer.participation_from_states_ (population std, chunked over units) so the offline
    ordering and the logged trace are the same quantity.
    """
    N = states.size(0)
    out = torch.empty(N, device=states.device)
    for i in range(0, N, chunk):
        blk = torch.relu(states[i:i + chunk].reshape(min(chunk, N - i), -1))
        out[i:i + chunk] = blk.std(dim=1, unbiased=False) + torch.quantile(blk.abs(), 0.9, dim=1)
    return out


class Scorer:
    """Held-out r2 (and masked MSE) of one network with any set of units removed.

    Removal zeroes the units' columns of W_rec and W_out; the intact weights are kept and restored
    before every score, so scores never compound.
    """

    def __init__(self, rnn, batch):
        self.rnn = rnn
        self.bi, self.bt, self.mask = batch
        with torch.no_grad():
            self.W_rec0 = rnn.W_rec.detach().clone()
            self.W_out0 = rnn.W_out.detach().clone()

    def __call__(self, removed=(), draws=None):
        """Mean and per-draw held-out (r2, mse) with the units in `removed` silenced."""
        draws = DRAWS if draws is None else draws
        rnn = self.rnn
        idx = torch.as_tensor(np.ascontiguousarray(removed, dtype=np.int64), device=rnn.device)
        r2s, mses = [], []
        with torch.no_grad():
            rnn.W_rec.copy_(self.W_rec0)
            rnn.W_out.copy_(self.W_out0)
            if idx.numel():
                rnn.W_rec[:, idx] = 0.0
                rnn.W_out[:, idx] = 0.0
            for d in range(draws):
                rnn.random_generator.manual_seed(NOISE_SEED + d)
                out = rnn(self.bi, w_noise=True)[1]
                r2s.append(Trainer.r2_score(out, self.bt, self.mask))
                mses.append(float(((scored_(out, self.mask) - scored_(self.bt, self.mask)) ** 2).mean()))
            rnn.W_rec.copy_(self.W_rec0)
            rnn.W_out.copy_(self.W_out0)
        return float(np.mean(r2s)), float(np.mean(mses)), r2s


# ------------------------------------------------------------------------------------------------
# the ablation
# ------------------------------------------------------------------------------------------------

QUICK = False


def k_grid(N, n_silent):
    """Removal counts: a coarse sweep of 0..N plus a fine one around the silent/active boundary."""
    if QUICK:
        return [0, n_silent, N]
    frac = np.r_[0, 0.01, 0.02, 0.03, 0.05, 0.075, np.arange(0.1, 1.0001, 0.05)]
    ks = set(np.round(frac * N).astype(int))
    for c in (n_silent, N - n_silent):
        ks |= set(np.clip(c + np.round(np.arange(-0.05, 0.0501, 0.01) * N).astype(int), 0, N))
    return sorted(int(k) for k in ks)


def ablate(rnn, batches, label, extra):
    """Run the three removal orders on one network and return a JSON-ready dict."""
    N = int(rnn.N)
    with torch.no_grad():
        ti = batches["tr"][0]
        states = rnn(ti, w_noise=False)[0]
        p = participation(states).cpu().numpy()
        hmax = states.amax(dim=(1, 2)).cpu().numpy()
        del states
    n_active = int((p >= SILENT_REL * np.quantile(p, 0.95)).sum())
    n_silent = N - n_active
    quiet_first = np.lexsort((hmax, p))                 # ascending p, ties by deepest drive
    busy_first = quiet_first[::-1].copy()
    rng = np.random.default_rng(zlib.crc32(label.encode()))
    randoms = [rng.permutation(N) for _ in range(RANDOM_ORDERS)]

    score = Scorer(rnn, batches["ho"])
    r2_full, mse_full, r2_full_draws = score()
    ks = k_grid(N, n_silent)
    curve = lambda order: [score(order[:k])[0] for k in ks]
    quiet = curve(quiet_first)
    busy = curve(busy_first)
    rand = [curve(o) for o in randoms]

    # the headline contrasts, each at an exact count
    r2_no_silent = score(quiet_first[:n_silent])[0]
    r2_rand_same = [score(o[:n_silent])[0] for o in randoms]
    r2_no_active = score(busy_first[:n_active])[0]
    print(f"  {label}: N={N} active {n_active} | r2 full {r2_full:.4f}, "
          f"all {n_silent} silent removed {r2_no_silent:.4f}, "
          f"{n_silent} random removed {np.mean(r2_rand_same):.4f}, "
          f"all {n_active} active removed {r2_no_active:.4f}")
    return dict(extra, label=label, N=N, n_active=n_active, n_silent=n_silent,
                participation=p.tolist(), r2_full=r2_full, mse_full=mse_full,
                r2_full_draws=r2_full_draws, ks=ks, quiet_first=quiet, busy_first=busy,
                random=rand, r2_no_silent=r2_no_silent, r2_random_same_count=r2_rand_same,
                r2_no_active=r2_no_active, draws=DRAWS, random_orders=RANDOM_ORDERS)


# ------------------------------------------------------------------------------------------------
# gates and read-out points
# ------------------------------------------------------------------------------------------------

def trace_of(run_dir):
    """The run's participation trace dict, or None."""
    hits = sorted(glob.glob(os.path.join(run_dir, "*ParticipationTrace.pkl")))
    if not hits:
        return None
    with open(hits[0], "rb") as fh:
        return pickle.load(fh)


def trace_active_at(trace, it):
    """The trace's active count at the stored probe nearest iteration `it`, and that probe's iteration."""
    its = np.asarray(trace["participation_iters"])
    j = int(np.argmin(np.abs(its - it)))
    p = np.asarray(trace["participation"][j], dtype=float)
    return int((p >= SILENT_REL * np.quantile(p, 0.95)).sum()), int(its[j])


def gates(rnn, batches, run_dir, iteration, check_name_r2):
    """Score with the run's own noise on a training batch, and the active count against the trace."""
    g = {}
    ti, tt, mask = batches["tr"]
    with torch.no_grad():
        rnn.random_generator.manual_seed(NOISE_SEED)
        g["r2_train_noisy"] = float(Trainer.r2_score(rnn(ti, w_noise=True)[1], tt, mask))
    if check_name_r2:
        m = re.match(r"([-\d.]+)_", os.path.basename(run_dir.rstrip("/")))
        if m:
            g["r2_folder"] = float(m.group(1))
            g["r2_gate_ok"] = abs(g["r2_train_noisy"] - g["r2_folder"]) <= 0.03
    tr = trace_of(run_dir)
    if tr is not None:
        g["trace_active"], g["trace_probe_iter"] = trace_active_at(tr, iteration)
    return g


def snapshot_series(snap_dir):
    """(iterations, parameter files) of every kept snapshot under snap_dir, in order."""
    dirs = sorted(glob.glob(os.path.join(snap_dir, "it*")))
    its, files = [], []
    for d in dirs:
        f = sorted(glob.glob(os.path.join(d, "*LastParams*.npz")))
        if f:
            its.append(int(os.path.basename(d)[2:]))
            files.append(f[0])
    return np.asarray(its), files


def crossing(L, it, smooth_w):
    """The 1.05x own-floor read-out of one loss series, with the project's guards.

    Args:
        L, it: the loss series and its iterations.
        smooth_w: samples in the box mean used for the late-rise guard.
    Returns:
        dict: floor, t_star, late_rise, degenerate, usable. `usable` needs a fitted floor above the
        1e-6 bound, a crossing, and a smoothed end no more than MAX_LATE_RISE x the smoothed minimum
        (a loss that climbs back after its minimum has a fitted floor on its late plateau, and its
        first crossing then lands on the initial descent).
    """
    from pr_matrix import fit_floor_at, excess_time_at
    ok = np.isfinite(L)
    L, it = np.asarray(L, float)[ok], np.asarray(it, float)[ok]
    floor = fit_floor_at(L, it, it[-1])
    t_star = excess_time_at(L, it, floor, FLOOR_DELTA) if floor is not None else float("nan")
    w = min(smooth_w, len(L))
    sm = np.convolve(L, np.ones(w) / w, mode="valid")
    out = dict(floor=floor, t_star=float(t_star), late_rise=float(sm[-1] / sm.min()),
               degenerate=bool(floor is not None and floor <= 1.01e-6))
    out["usable"] = bool(floor is not None and not out["degenerate"] and np.isfinite(t_star)
                         and out["late_rise"] <= MAX_LATE_RISE)
    return out


def floor_points(run_dir, snap_dir, batches_for):
    """The run's 1.05x-floor crossings on two loss series, and the kept snapshot nearest each.

    "clean"    the trace's noise-free loss on the training batch (`loss_clean_train`, every 10
               iterations) - the project's standard series.
    "heldout"  the held-out loss under the training noise, scored on every kept snapshot with the
               same seeded draws - the training objective, which these networks actually minimise.
    On CDDM both are usable; on the flip-flop and DMTS the clean loss climbs back after its minimum
    and fails the late-rise guard, which is why the second series exists.

    Args:
        run_dir: the finished run (its LastParams is the series' last point).
        snap_dir: its kept-snapshot folder.
        batches_for: callable(cfg, device) -> seeded batches.
    Returns:
        dict with the held-out series and, per series, the crossing dict plus the chosen snapshot.
    """
    its, files = snapshot_series(snap_dir)
    last = sorted(glob.glob(os.path.join(run_dir, "*LastParams*.np[zy]"))
                  or glob.glob(os.path.join(run_dir, "*LastParams*.json")))[0]
    cfg = OmegaConf.load(sorted(glob.glob(os.path.join(run_dir, "*_config.yaml")))[0])
    its = np.r_[its, int(cfg.trainer.max_iter)]
    files = files + [last]
    cfg, rnn = build(run_dir, load_params(files[0]))
    batches = batches_for(cfg, rnn.device)
    mse = []
    for f in files:
        set_params(rnn, load_params(f))
        mse.append(Scorer(rnn, batches["ho"])()[1])
    out = dict(series_iters=its.tolist(), series_heldout_mse=mse)
    out["heldout"] = crossing(np.asarray(mse), its, smooth_w=5)
    tr = trace_of(run_dir)
    if tr is not None and "loss_clean_train" in tr.get("metrics", {}):
        out["clean"] = crossing(np.asarray(tr["metrics"]["loss_clean_train"], float),
                                np.asarray(tr["iters"], float), smooth_w=201)
    for kind in ("clean", "heldout"):
        c = out.get(kind)
        if c is not None and np.isfinite(c["t_star"]):
            j = int(np.argmin(np.abs(its - c["t_star"])))
            c["snapshot_iter"], c["snapshot_file"] = int(its[j]), files[j]
    return out


def summary(out_dir):
    """Print one row per cell and read-out: means over networks of the headline contrasts.

    Args:
        out_dir: the folder of result JSONs.
    """
    rows = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        if f.endswith("_points.json"):
            continue
        with open(f) as fh:
            r = json.load(fh)
        if "r2_full" not in r:
            continue
        rows.setdefault((r["sweep"], r["cell"], r["readout"]), []).append(r)
    print(f"{'cell':62s} {'read-out':13s} n  {'silent':>11s}  {'r2 intact':>9s}  "
          f"{'-silent':>8s}  {'-random':>8s}  {'-active':>8s}  iteration")
    for (sweep, cell, readout), rs in sorted(rows.items()):
        m = lambda k: float(np.mean([r[k] for r in rs]))
        rnd = float(np.mean([np.mean(r["r2_random_same_count"]) for r in rs]))
        sil = np.array([r["n_silent"] / r["N"] for r in rs])
        print(f"{(sweep + '/' + cell)[:62]:62s} {readout:13s} {len(rs)}  "
              f"{m('n_silent'):5.0f} ({100 * sil.mean():2.0f}%)  {m('r2_full'):9.4f}  "
              f"{m('r2_no_silent'):8.4f}  {rnd:8.4f}  {m('r2_no_active'):8.4f}  "
              f"{int(m('iteration')):,}")


# ------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("readout", choices=["end", "floor", "summary"])
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("snap_dir", nargs="?")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: one noise draw, one random order, three removal counts")
    a = ap.parse_args()
    global DRAWS, RANDOM_ORDERS, QUICK
    if a.quick:
        DRAWS, RANDOM_ORDERS, QUICK = 1, 1, True
    if a.readout == "summary":
        summary(a.out)
        return
    os.makedirs(a.out, exist_ok=True)
    run_dir = a.run_dir.rstrip("/")
    cell = os.path.basename(os.path.dirname(run_dir))
    sweep = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
    seed = re.search(r"seed=(\d+)", os.path.basename(run_dir))
    label = f"{sweep}__{cell}__s{seed.group(1) if seed else 'na'}__{a.readout}"
    extra = dict(run_dir=run_dir, sweep=sweep, cell=cell, readout=a.readout)

    if a.readout == "end":
        params_file = sorted(glob.glob(os.path.join(run_dir, "*LastParams*.np[zy]"))
                             or glob.glob(os.path.join(run_dir, "*LastParams*.json")))[0]
        cfg, rnn = build(run_dir, load_params(params_file))
        tr = trace_of(run_dir)              # a top-up's trace is joined, so its last iteration is
        iteration = (int(np.asarray(tr["iters"])[-1]) if tr is not None   # the whole budget
                     else int(cfg.trainer.max_iter))
        extra.update(iteration=iteration, params_file=params_file)
        batches = seeded_batches(cfg, rnn.device)
        extra["gates"] = gates(rnn, batches, run_dir, iteration, check_name_r2=True)
    else:
        if not a.snap_dir:
            raise SystemExit("floor read-out needs the run's kept-snapshot folder")
        fp = floor_points(run_dir, a.snap_dir, seeded_batches)
        with open(os.path.join(a.out, label + "_points.json"), "w") as fh:
            json.dump(dict(extra, floor_points=fp), fh)
        for kind in ("clean", "heldout"):
            c = fp.get(kind) or {}
            tag = f"{label}_{kind}"
            print(f"  {tag}: floor {c.get('floor')}, T* {c.get('t_star')}, "
                  f"late rise {c.get('late_rise', float('nan')):.2f}, usable {c.get('usable')}")
            if not c.get("usable"):
                continue
            cfg, rnn = build(run_dir, load_params(c["snapshot_file"]))
            batches = seeded_batches(cfg, rnn.device)
            ex = dict(extra, readout=f"floor_{kind}", iteration=c["snapshot_iter"],
                      params_file=c["snapshot_file"], floor_point=c,
                      gates=gates(rnn, batches, run_dir, c["snapshot_iter"], check_name_r2=False))
            print(f"  gates: {ex['gates']}")
            res = ablate(rnn, batches, tag, ex)
            with open(os.path.join(a.out, tag + ".json"), "w") as fh:
                json.dump(res, fh)
        return
    print(f"  gates: {extra['gates']}")
    res = ablate(rnn, batches, label, extra)
    with open(os.path.join(a.out, label + ".json"), "w") as fh:
        json.dump(res, fh)
    print(f"  -> {os.path.join(a.out, label + '.json')}")


if __name__ == "__main__":
    main()
