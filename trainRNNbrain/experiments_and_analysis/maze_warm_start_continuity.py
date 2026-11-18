"""Local CPU check that a Maze run split into two halves with paths.init_adam=true continues the optimizer
exactly (the "Before submission" check of notebook entry 2026-11-17_2249_rules-maze-warm-start-test.md).

The runs go through the production entry point, trainRNNbrain/training/run_experiment.py (Hydra main ->
run_rl -> TrainerPPO), of whatever trainRNNbrain is on PYTHONPATH; for the warm-start test that is an export
of the trainer commit 8c6b36e. No trainer source is changed. In "run" mode this script wraps
TrainerPPO.run_training from outside, to (a) dump the optimizer state and learning rates the trainer starts
with, after run_rl has loaded the parent's AdamState and set the rate, (b) save every random-number state when
training ends, and (c) optionally load such a saved state before training starts.

Four runs at N = 50 with tiny episodes (4 episodes of 6 decisions, 4 probe episodes) and warmup_updates = 2,
so that the unsplit run has finished its ramp before the split:
  A    unsplit, 6 updates;
  B1   the first half, 3 updates;
  B2   the second half, 3 updates from B1 with init_adam=true (the production continuation);
  B2r  as B2, with every random-number state set to B1's at the end of its training.
Pass conditions, fixed before the first run:
  C1  the optimizer state at the start of B2 equals B1's saved AdamState file: every tensor (torch.equal)
      and every step count; and B2 starts at the base rate;
  C2  every update of B2 records lr equal to the base rate (no ramp; a ramp would give base/2 at its first);
  C3  B2's progress records updates 4-6, its probes start at 4, its saved config has iter_offset 3;
  C4  Adam's step at the end of B2 equals B1's end step + 3 x n_epochs x n_minibatches (minus skipped
      minibatches), and equals A's end step;
  C5  B2r is bitwise identical to A's updates 4-6: the saved weights, the AdamState (every tensor and
      step) and every progress record; so the only thing the split changes is the random streams;
  C6  (descriptive) the random states at the start of B2 equal those at the start of B1: each run
      re-seeds python, numpy and torch's global stream, the RNN's generator and the task's drop generator
      from the run seed, so a continuation replays the random numbers of its chain's first updates.

Usage (from the repo root; CODE is an export of the trainer commit, e.g. `git archive 8c6b36e | tar -x -C CODE`):
  PYTHONPATH=CODE python trainRNNbrain/experiments_and_analysis/maze_warm_start_continuity.py check OUTDIR
Prints one line per condition and exits 1 if any of C1-C5 fails.
"""
import copy
import glob
import json
import os
import random
import runpy
import subprocess
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

import trainRNNbrain
from trainRNNbrain.trainer.TrainerPPO import TrainerPPO

N_UPDATES_HALF = 3
SEED = 1
COMMON = ["task=Maze", "trainer=trainer_ppo", "model=rnn_relu_standard", "model.N=50", f"seed={SEED}",
          "trainer.n_episodes=4", "trainer.T=6", "trainer.probe_episodes=4", "trainer.warmup_updates=2",
          "trainer.probe_every=1", "experiment_tag=\"suite/N=50_act=relu\""]


def rng_states(trainer):
    """Every random-number state a TrainerPPO run draws from.

    Args:
        trainer: TrainerPPO, built (its Task and RNN exist).
    Returns:
        dict: python (random.getstate()), numpy (np.random.get_state()), torch (torch.get_rng_state()),
        rnn (the RNN generator's state, ByteTensor), task (the task drop generator's state, ByteTensor).
    """
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "rnn": trainer.RNN.random_generator.get_state(), "task": trainer.Task.generator.get_state()}


def set_rng_states(trainer, st):
    """Load the states saved by rng_states into every random source of a trainer run.

    Args:
        trainer: TrainerPPO; st: dict from rng_states.
    Returns:
        None.
    """
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    trainer.RNN.random_generator.set_state(st["rnn"])
    trainer.Task.generator.set_state(st["task"])


ORIGINAL_RUN_TRAINING = TrainerPPO.run_training


def wrapped_run_training(self):
    """TrainerPPO.run_training with the dumps described in the module docstring, driven by environment
    variables: WS_DUMP_START (path: optimizer state, learning rates, adam_restored, iter_offset and the
    random states at the start), WS_LOAD_RNG (path of a saved rng_states dict, loaded first) and
    WS_SAVE_RNG (path: the random states when training ends).

    Args:
        self: TrainerPPO.
    Returns:
        whatever TrainerPPO.run_training returns.
    """
    if os.environ.get("WS_LOAD_RNG"):
        set_rng_states(self, torch.load(os.environ["WS_LOAD_RNG"], weights_only=False))
    if os.environ.get("WS_DUMP_START"):
        torch.save({"opt": copy.deepcopy(self.optimizer.state_dict()),
                    "lrs": [g["lr"] for g in self.optimizer.param_groups], "base_lr": self.lr,
                    "adam_restored": self.adam_restored, "iter_offset": self.iter_offset,
                    "rng": rng_states(self)}, os.environ["WS_DUMP_START"])
    out = ORIGINAL_RUN_TRAINING(self)
    if os.environ.get("WS_SAVE_RNG"):
        torch.save(rng_states(self), os.environ["WS_SAVE_RNG"])
    return out


def run_one(outdir, name, extra, env_extra):
    """Run one training through run_experiment.py in a subprocess of this script in "run" mode.

    Args:
        outdir: str, the check's folder; name: str, the run's name (its save_to is outdir/name);
        extra: list of str, Hydra overrides beyond COMMON; env_extra: dict, the WS_* variables.
    Returns:
        str, the run folder outdir/name/Maze_suite/N=50_act=relu/s<SEED>.
    Raises:
        subprocess.CalledProcessError if the run fails (its log is outdir/name.log).
    """
    args = COMMON + [f"paths.save_to={os.path.join(outdir, name)}",
                     f"hydra.run.dir={os.path.join(outdir, name + '_hydra')}"] + extra
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", **env_extra)
    # cwd = outdir, so that a checkout in the working directory cannot shadow the trainRNNbrain on PYTHONPATH
    with open(os.path.join(outdir, name + ".log"), "w") as log:
        subprocess.run([sys.executable, os.path.abspath(__file__), "run"] + args, env=env, check=True,
                       stdout=log, stderr=subprocess.STDOUT, cwd=outdir)
    return os.path.join(outdir, name, "Maze_suite", "N=50_act=relu", f"s{SEED}")


def load_folder(folder):
    """The saved files of a maze run folder.

    Args:
        folder: str, a run folder.
    Returns:
        dict: params (dict of np.ndarray from *Params_Maze.npz), adam (the AdamState state_dict),
        progress (list of dicts), probes (dict of np.ndarray from *Participation.npz), cfg (DictConfig).
    """
    one = lambda pat: sorted(glob.glob(os.path.join(folder, pat)))[0]
    with open(one("*Progress.json")) as fh:
        progress = json.load(fh)
    params = dict(np.load(one("*Params_Maze.npz"), allow_pickle=True))
    probes = dict(np.load(one("*Participation.npz"), allow_pickle=True))
    return {"params": params, "adam": torch.load(one("*AdamState_Maze.pt"), weights_only=False),
            "progress": progress, "probes": probes, "cfg": OmegaConf.load(one("*config.yaml"))}


def adam_equal(a, b):
    """Whether two optimizer state_dicts hold identical per-parameter state (every tensor torch.equal,
    every non-tensor equal), and the list of what differs.

    Args:
        a, b: optimizer state_dicts.
    Returns:
        (bool, list of str).
    """
    diffs = []
    if sorted(a["state"]) != sorted(b["state"]):
        return False, [f"parameter keys {sorted(a['state'])} vs {sorted(b['state'])}"]
    for k in a["state"]:
        sa, sb = a["state"][k], b["state"][k]
        if sorted(sa) != sorted(sb):
            diffs.append(f"param {k}: fields {sorted(sa)} vs {sorted(sb)}")
            continue
        for f in sa:
            va, vb = sa[f], sb[f]
            same = torch.equal(va, vb) if torch.is_tensor(va) else va == vb
            if not same:
                diffs.append(f"param {k} field {f}")
    return not diffs, diffs


def steps(adam):
    """Adam's step count per parameter.

    Args:
        adam: an Adam state_dict.
    Returns:
        list of int, one per parameter.
    """
    return [int(adam["state"][k]["step"]) for k in sorted(adam["state"])]


def rng_equal(a, b):
    """Whether two rng_states dicts are identical, per source.

    Args:
        a, b: dicts from rng_states.
    Returns:
        dict of str -> bool.
    """
    out = {"python": a["python"] == b["python"],
           "numpy": all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
                        for x, y in zip(a["numpy"], b["numpy"]))}
    for k in ("torch", "rnn", "task"):
        out[k] = torch.equal(a[k], b[k])
    return out


def check(outdir):
    """Run A, B1, B2 and B2r and evaluate C1-C6 (module docstring).

    Args:
        outdir: str, an empty or new folder for the four runs and their logs.
    Returns:
        bool, True if C1-C5 all pass.
    """
    os.makedirs(outdir, exist_ok=True)
    print(f"code under test: {os.path.dirname(trainRNNbrain.__file__)}")
    p = lambda n: os.path.join(outdir, n)
    fa = run_one(outdir, "A", [f"trainer.n_updates={2 * N_UPDATES_HALF}"], {"WS_DUMP_START": p("A_start.pt")})
    fb1 = run_one(outdir, "B1", [f"trainer.n_updates={N_UPDATES_HALF}"],
                  {"WS_DUMP_START": p("B1_start.pt"), "WS_SAVE_RNG": p("B1_end_rng.pt")})
    cont = [f"trainer.n_updates={N_UPDATES_HALF}", f"paths.init_from=\"{fb1}\"", "paths.init_adam=true",
            "trainer.iter_offset=auto"]
    fb2 = run_one(outdir, "B2", cont, {"WS_DUMP_START": p("B2_start.pt")})
    fb2r = run_one(outdir, "B2r", cont, {"WS_DUMP_START": p("B2r_start.pt"), "WS_LOAD_RNG": p("B1_end_rng.pt")})
    A, B1, B2, B2r = (load_folder(f) for f in (fa, fb1, fb2, fb2r))
    s_b1, s_b2 = torch.load(p("B1_start.pt"), weights_only=False), torch.load(p("B2_start.pt"), weights_only=False)
    base = s_b2["base_lr"]
    per_update = int(A["cfg"].trainer.n_epochs) * int(A["cfg"].trainer.n_minibatches)
    results = {}

    ok, diffs = adam_equal(s_b2["opt"], B1["adam"])
    results["C1"] = (ok and s_b2["adam_restored"] and all(lr == base for lr in s_b2["lrs"]),
                     f"B2 start state == B1 saved AdamState: {ok} {diffs[:3]}; step counts at B2 start "
                     f"{steps(s_b2['opt'])} vs B1 file {steps(B1['adam'])}; adam_restored "
                     f"{s_b2['adam_restored']}; lr at start {s_b2['lrs']} (base {base:.6g})")
    lrs_b1 = [r["lr"] for r in B1["progress"]]
    lrs_b2 = [r["lr"] for r in B2["progress"]]
    lrs_a = [r["lr"] for r in A["progress"]]
    results["C2"] = (all(lr == base for lr in lrs_b2),
                     f"lr per update: B1 {lrs_b1}, B2 {lrs_b2}, A {lrs_a}")
    upd_b2 = [r["update"] for r in B2["progress"]]
    results["C3"] = (upd_b2 == list(range(N_UPDATES_HALF + 1, 2 * N_UPDATES_HALF + 1))
                     and int(B2["probes"]["update"][0]) == N_UPDATES_HALF + 1
                     and int(B2["cfg"].trainer.iter_offset) == N_UPDATES_HALF,
                     f"B2 updates {upd_b2}, probe updates {B2['probes']['update'].tolist()}, "
                     f"iter_offset {B2['cfg'].trainer.iter_offset}")
    skipped = sum(int(r.get("skipped", 0) or 0) + int(r.get("skipped_grad", 0) or 0) for r in B2["progress"])
    want = [s + N_UPDATES_HALF * per_update - skipped for s in steps(B1["adam"])]
    results["C4"] = (steps(B2["adam"]) == want and steps(B2["adam"]) == steps(A["adam"]),
                     f"steps: B1 end {steps(B1['adam'])}, B2 end {steps(B2['adam'])} (expected {want}, "
                     f"{skipped} skipped), A end {steps(A['adam'])}")
    same_params = {k: np.array_equal(A["params"][k], B2r["params"][k]) for k in A["params"]
                   if A["params"][k].dtype != object}
    ok_adam, d_adam = adam_equal(A["adam"], B2r["adam"])
    same_prog = A["progress"][N_UPDATES_HALF:] == B2r["progress"]
    same_probe = all(np.array_equal(np.asarray(A["probes"][k])[-len(B2r["probes"][k]):], B2r["probes"][k])
                     for k in ("update", "active", "active_mod", "probe_success"))
    results["C5"] = (all(same_params.values()) and ok_adam and same_prog and same_probe,
                     f"B2r vs A updates 4-6: params equal {same_params}; AdamState equal {ok_adam} {d_adam[:3]}; "
                     f"progress records equal {same_prog}; probe records equal {same_probe}")
    params_b2_vs_a = {k: bool(np.array_equal(A["params"][k], B2["params"][k])) for k in ("W_rec", "W_out")}
    ret = lambda R: [round(r["discounted_return"], 5) for r in R]
    results["C6"] = (None, f"random states at B2 start == at B1 start: {rng_equal(s_b2['rng'], s_b1['rng'])}; "
                           f"B2 (no restore) final W_rec/W_out equal to A's: {params_b2_vs_a}; discounted "
                           f"return per update: A 4-6 {ret(A['progress'][N_UPDATES_HALF:])}, B2 {ret(B2['progress'])}, "
                           f"B1 1-3 {ret(B1['progress'])}")
    for k, (passed, msg) in results.items():
        tag = "info" if passed is None else ("PASS" if passed else "FAIL")
        print(f"{k} {tag}: {msg}")
    return all(v[0] for k, v in results.items() if v[0] is not None)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        # run_experiment.py is executed as __main__ (as the launchers run it), because @hydra.main finds
        # the configs relative to the calling file only then; the patched class is the one it imports
        TrainerPPO.run_training = wrapped_run_training
        entry = os.path.join(os.path.dirname(trainRNNbrain.__file__), "training", "run_experiment.py")
        print(f"code under test: {entry}", flush=True)
        sys.argv = [entry] + sys.argv[2:]
        runpy.run_path(entry, run_name="__main__")
    elif sys.argv[1] == "check":
        sys.exit(0 if check(sys.argv[2]) else 1)
    else:
        sys.exit(f"usage: {sys.argv[0]} check OUTDIR")
