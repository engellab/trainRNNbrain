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
  floor  the weights where the network's NOISY loss - the loss it is trained on - first reaches
         1.05x its own fitted floor (common.noisy_floor: running median over 2,001 iterations,
         fitted from the escape for a task that stalls; the rule since 2026-10-12). Only runs
         trained with `+trainer.checkpoint_keep=true` have weights there
         (slurm/SilentReLU_c1_snapshots_della.slurm). The kept snapshots are collected along the
         run's whole warm-start chain, so a continuation to 200k is read anywhere from 0 to 200k,
         and the snapshot nearest the crossing is ablated.

GATES, written into every result and printed: the unablated network's score with its own noise on a
training-distribution batch against the score in its folder name (end read-out, +-0.03), and the
offline active count against the trace's count at the nearest stored probe.

Usage (on a GPU node; the Mac has no environment):
  python ablate_silent_units.py end   <run_dir>              [--out DIR]
  python ablate_silent_units.py floor <run_dir>              [--out DIR] [--snap-root DIR]
  python ablate_silent_units.py retrain <run_dir> [--at floor] [--out DIR] (prune, then fine-tune)
  python ablate_silent_units.py summary                      [--out DIR]   (one row per cell)
    run_dir   a finished run folder (holds *_config.yaml, *LastParams*, *ParticipationTrace.pkl)
Output: one JSON per network and read-out in --out (default /scratch/gpfs/TENGEL/pt1290/c1_ablation_seeded).
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
from trainRNNbrain.utils import import_any, filter_kwargs, unjsonify  # noqa: E402
from common import active_mask, warm_start_chain, noisy_loss_series, noisy_floor, task_of  # noqa: E402

DRAWS = 9                    # noise draws per score, as heldout_r2.DRAWS
RANDOM_ORDERS = 5            # independent random removal orders per network
NOISE_SEED = H.NOISE_SEED    # draw d of every score uses generator seed NOISE_SEED + d, as heldout_r2
BATCH_SEED = 20261012        # seeds the global generators, and the fine-tune's noise generator
ORDER_SEED = 20261013        # seeds the task generator of the ordering batch (the held-out batch is
                             # seeded inside heldout_r2.heldout_batch, with H.HELDOUT_SEED)
FINETUNE_SEED = 20261014     # seeds the task generator of the fine-tune's training batches (retrain)
OUT = "/scratch/gpfs/TENGEL/pt1290/c1_ablation_seeded"   # c1_ablation holds the unseeded results


# ------------------------------------------------------------------------------------------------
# building and running a network
# ------------------------------------------------------------------------------------------------

def load_params(path):
    """Learned tensors from a parameter file (.npz or .json), as a dict of name -> array."""
    if path.endswith(".json"):
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

    np.random.seed does not reach a task's own generator (np.random.default_rng), and a sampled task
    (flip-flop, DMTS) is configured with seed None; before 2026-10-14 both batches were therefore
    fresh draws on every run. The held-out batch is now seeded in heldout_r2.heldout_batch and the
    ordering batch here, with a different seed, so the two stay distinct.

    Returns:
        dict with 'ho' = (inputs, targets, mask) unseen by the network, and 'tr' = (inputs, targets,
        mask) from the training distribution at its training batch size.
    """
    np.random.seed(BATCH_SEED)
    torch.manual_seed(BATCH_SEED)
    bi, bt, mask = H.heldout_batch(cfg)
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    tcfg.seed = ORDER_SEED
    task = hydra.utils.instantiate(tcfg)
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
    n_active = int(active_mask(p).sum())
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
    return int(active_mask(p).sum()), int(its[j])


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


SNAP_ROOT = "/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots"


def chain_snapshots(run_dir, snap_root=SNAP_ROOT):
    """Every weight state along a run's warm-start chain: kept snapshots and each segment's end.

    A run's kept snapshots live in <snap_root>/<taskname>_<experiment_tag>/s<seed>/it<iteration>/,
    named by absolute iteration; each segment's end state is its LastParams.

    Args:
        run_dir: a finished run (possibly a continuation); snap_root: where kept snapshots live.
    Returns:
        (iterations, parameter files), sorted by iteration, one file per iteration.
    """
    found = {}
    for d in warm_start_chain(run_dir):
        cfg = OmegaConf.load(sorted(glob.glob(os.path.join(d, "*_config.yaml")))[0])
        sd = os.path.join(snap_root, f"{cfg.task.taskname}_{cfg.experiment_tag}", f"s{cfg.seed}")
        for it, f in zip(*snapshot_series(sd)):
            found[int(it)] = f
        last = sorted(glob.glob(os.path.join(d, "*LastParams*.np[zy]"))
                      or glob.glob(os.path.join(d, "*LastParams*.json")))
        off = cfg.trainer.get("iter_offset", 0)
        if last:
            found[(int(off) if isinstance(off, (int, float)) else 0) + int(cfg.trainer.max_iter)] = last[0]
    its = sorted(found)
    return np.asarray(its), [found[i] for i in its]


def floor_snapshot(run_dir, snap_root=SNAP_ROOT):
    """The run's noisy-loss floor read-out and the weight state nearest its crossing.

    Returns:
        (floor dict from common.noisy_floor without its smoothed series, iteration, parameter file);
        the last two are None when the run has no read-out.
    """
    it, L = noisy_loss_series(run_dir)
    fl = noisy_floor(it, L, task=task_of(run_dir)) if L is not None else dict(reason="no noisy loss", t_star=np.nan)
    fl.pop("smoothed", None)
    if fl["reason"] is not None:
        return fl, None, None
    its, files = chain_snapshots(run_dir, snap_root)
    j = int(np.argmin(np.abs(its - fl["t_star"])))
    print(f"  floor {fl['floor']:.3e}, escape {fl['escape']:,.0f}, crossing {fl['t_star']:,.0f}; "
          f"nearest weights at {its[j]:,} ({len(its)} states on the chain)")
    return fl, int(its[j]), files[j]


RETRAIN_ITERS = 3000         # fine-tuning budget after a removal (~3% of a training run)
RETRAIN_EVERY = 250          # held-out score recorded every this many fine-tuning iterations


def retrain_curve(cfg, rnn, params, removed, batches, iters=RETRAIN_ITERS):
    """Remove `removed` and fine-tune what is left as the network was trained; score along the way.

    Separates "the silent units compute something" from "removing them perturbs a tuned system":
    if the pruned network recovers its score within a short fine-tune while held without those units,
    the task never needed them. Training matches run_experiment: Adam at the N-scaled learning rate
    with the run's weight decay, the masked task MSE under the training noise, gradient clipping at
    max_grad_norm, fresh training batches. The removed units' outgoing weights are re-zeroed after
    every step, so they stay out.

    Args:
        cfg, rnn: the run's config and its built network; params: the trained tensors to start from;
        removed: unit indices held out (empty = intact control); batches: seeded_batches output.
    Returns:
        list of (iteration, held-out r2) pairs, iteration 0 being straight after removal.
    """
    set_params(rnn, params)
    idx = torch.as_tensor(np.ascontiguousarray(removed, dtype=np.int64), device=rnn.device)
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    tcfg.seed = FINETUNE_SEED                    # the same batch sequence on every run and in every arm
    task = hydra.utils.instantiate(tcfg)
    mask = torch.as_tensor(np.asarray(get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)),
                           device=rnn.device)
    lr = float(cfg.trainer.lr) * (100.0 / float(cfg.model.N)) ** float(cfg.trainer.lr_scale_exp)
    prms = [q for q in rnn.parameters() if q.requires_grad]
    opt = torch.optim.Adam(prms, lr=lr, weight_decay=float(cfg.trainer.weight_decay))
    bi_ho, bt_ho, m_ho = batches["ho"]

    def hold_out():
        if idx.numel():
            with torch.no_grad():
                rnn.W_rec[:, idx] = 0.0
                rnn.W_out[:, idx] = 0.0

    def score():
        # ⚠️ THE GENERATOR IS SAVED AND RESTORED AROUND SCORING. It is the one the training passes
        # draw their noise from, and scoring reseeds it for its fixed draws. Without the restore,
        # training restarted from the same noise state after every score, so from iteration 251 on
        # it saw the same 250 noise realisations eleven times; on CDDM's fixed coherence grid every
        # (trial, noise) sample then recurred, the intact network memorised them by waking silent
        # units, and its held-out r2 fell 0.836 -> 0.79 (found 2026-10-13; results before the fix
        # are in /scratch/gpfs/TENGEL/pt1290/c1_retrain_bug_noise_reseed).
        state = rnn.random_generator.get_state()
        with torch.no_grad():
            vals = []
            for d in range(DRAWS):
                rnn.random_generator.manual_seed(NOISE_SEED + d)
                vals.append(Trainer.r2_score(rnn(bi_ho, w_noise=True)[1], bt_ho, m_ho))
        rnn.random_generator.set_state(state)
        return float(np.mean(vals))

    skipped = [0]
    hold_out()
    curve = [(0, score())]
    rnn.random_generator.manual_seed(BATCH_SEED)
    for it in range(1, iters + 1):
        bi, bt, _ = task.get_batch()
        bi = torch.as_tensor(np.asarray(bi), dtype=torch.float32, device=rnn.device)
        bt = torch.as_tensor(np.asarray(bt), dtype=torch.float32, device=rnn.device)
        out = rnn(bi, w_noise=True)[1]
        loss = ((scored_(out, mask) - scored_(bt, mask)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(prms, max_norm=float(cfg.trainer.max_grad_norm))
        if not torch.isfinite(gnorm):            # the Trainer skips a non-finite update; so does this
            opt.zero_grad(set_to_none=True)
            skipped[0] += 1
        else:
            opt.step()
        hold_out()
        if it % RETRAIN_EVERY == 0:
            curve.append((it, score()))
    if skipped[0]:
        print(f"    {skipped[0]} non-finite updates skipped")
    return curve


def retrain_test(run_dir, out_dir, label, extra, params_file=None):
    """Prune the silent units, or as many at random, and fine-tune; intact network fine-tuned as control.

    Args:
        params_file: the weight state to start from; the run's end state (LastParams) when None.
    """
    params_file = params_file or sorted(glob.glob(os.path.join(run_dir, "*LastParams*.np[zy]"))
                                        or glob.glob(os.path.join(run_dir, "*LastParams*.json")))[0]
    params = load_params(params_file)
    cfg, rnn = build(run_dir, params)
    batches = seeded_batches(cfg, rnn.device)
    with torch.no_grad():
        states = rnn(batches["tr"][0], w_noise=False)[0]
        p = participation(states).cpu().numpy()
        del states
    N = len(p)
    silent = np.flatnonzero(~active_mask(p))
    rnd = np.random.default_rng(zlib.crc32(label.encode())).permutation(N)[:len(silent)]
    res = dict(extra, label=label, N=N, n_silent=int(len(silent)), params_file=params_file,
               retrain_iters=RETRAIN_ITERS)
    for name, removed in (("intact", []), ("silent_removed", silent), ("random_removed", rnd)):
        res[name] = retrain_curve(cfg, rnn, params, removed, batches)
        c = res[name]
        print(f"  {label} {name:15s}: r2 {c[0][1]:.4f} after removal -> {c[-1][1]:.4f} after "
              f"{RETRAIN_ITERS} fine-tuning iterations")
    with open(os.path.join(out_dir, label + ".json"), "w") as fh:
        json.dump(res, fh)


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
    ap.add_argument("readout", choices=["end", "floor", "retrain", "summary"])
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--snap-root", default=SNAP_ROOT, help="where kept snapshots live")
    ap.add_argument("--at", choices=["end", "floor"], default="end",
                    help="retrain: start from the end state or from the weights at the floor")
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

    if a.readout == "retrain":
        pf = None
        if a.at == "floor":                      # start from the weights at the run's floor
            fl, it_j, pf = floor_snapshot(run_dir, a.snap_root)
            if pf is None:
                print(f"  {label}: no floor read-out - {fl['reason']}")
                return
            label = label.replace("__retrain", "__retrain_floor")
            extra.update(readout="retrain_floor", iteration=it_j, floor_read=fl)
        retrain_test(run_dir, a.out, label, extra, params_file=pf)
        return
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
        fl, it_j, pf = floor_snapshot(run_dir, a.snap_root)
        extra["floor_read"] = fl
        if pf is None:
            print(f"  {label}: no floor read-out - {fl['reason']}")
            with open(os.path.join(a.out, label + "_noread.json"), "w") as fh:
                json.dump(extra, fh)
            return
        cfg, rnn = build(run_dir, load_params(pf))
        extra.update(iteration=it_j, params_file=pf)
        batches = seeded_batches(cfg, rnn.device)
        extra["gates"] = gates(rnn, batches, run_dir, it_j, check_name_r2=False)
    print(f"  gates: {extra['gates']}")
    res = ablate(rnn, batches, label, extra)
    with open(os.path.join(a.out, label + ".json"), "w") as fh:
        json.dump(res, fh)
    print(f"  -> {os.path.join(a.out, label + '.json')}")


if __name__ == "__main__":
    main()
