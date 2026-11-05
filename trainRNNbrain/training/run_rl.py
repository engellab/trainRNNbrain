"""The reinforcement-learning path of the training script: train the standard RNN with PPO to run
the maze to the cheese, and record how many of its units take part.

Entry point (the one training script; run_experiment.run_training dispatches here when the trainer's
_target_ is TrainerPPO):
  python trainRNNbrain/training/run_experiment.py task=Maze trainer=trainer_ppo model=rnn_relu_standard model.N=500 seed=1
Config: configs/task/Maze.yaml (the maze, the sensors, the rewards), configs/trainer/trainer_ppo.yaml
(PPO) and the project's model and paths groups, composed by run_experiment's Hydra main from
configs/base.yaml. This module reads the resolved config and overrides nothing; a launcher varies
only the keys it sweeps. Values written back into the saved config as a record of what was used,
as run_experiment does for the seed: the seed (resolved here from "random", as run_experiment
does), task.n_inputs and model.n_inputs (from the task's sensors), task.n_outputs (2 actions)
and model.n_outputs (2, or 3 with trainer.critic), so that the network can be rebuilt from the
saved config alone. Seeds: the RNN (its initial weights and its noise) is seeded with `seed`, the
task's drops with `seed + 1` (TaskMaze), so the two streams differ. One network per run
(cfg.n_nets is not used here).

Output, the standard layout under <paths.save_to>/<taskname>_<folder_tag>/s<seed>/, with
folder_tag = experiment_tag plus the trainer and RNN tags when they are not empty (as
run_experiment builds it) and the file stem s<seed>:
  s<seed>_config.yaml         the resolved config, with the values above written back
  s<seed>_Params_Maze.npz     RNN_torch.get_params(): W_rec, W_inp, W_out, y_init and the rest.
                              NOT "LastParams": the cell inventories (fig_paper_F1._cell_index,
                              the top-up launchers, common's loaders) find finished supervised runs
                              by globbing *LastParams*, and a maze run must not be counted as one.
  s<seed>_Progress.json       one dict per update: update, discounted_return (mean over the batch of
                              sum_t gamma^t r_t, the quantity PPO maximises), success (share of the
                              batch that reached the cheese), steps (mean steps to the cheese, T if
                              never), restored (rollouts discarded and weights rolled back, see
                              TrainerPPO.safe_rollout), lr (the learning rate used, after the
                              warm-up ramp) and the PPO statistics from TrainerPPO.ppo_update
                              (including skipped minibatches)
  s<seed>_AdamState_Maze.pt   the optimizer's state_dict (Adam's two moments per parameter), so the
                              run can be continued (below) rather than restarted; written as
                              run_experiment writes it for the supervised runs
  s<seed>_Participation.npz   per probe: update (the update index), active (active units under the
                              project's rule, p = std + q0.9), active_mod (active units under the
                              modulation criterion, std alone at or above 5 percent of its 95th
                              percentile: the count to read for tanh and sigmoid networks, whose
                              resting rate is not 0 and clears the p bar on its own), probe_success
                              (share of probe episodes that reached the cheese); and every
                              store_participation_every-th probe's full participation vector:
                              p (n_stored, N) with its update index p_update (n_stored,)
  progress.png                discounted return, success and active count against updates
  trajectories.png            the maze with the probe episodes' paths at the last checkpoint
The folder is written at every checkpoint (trainer.checkpoint_every updates, through the trainer's
checkpoint_fn hook, every file overwritten in place) and once more when training ends, so a run
killed by its wall clock leaves its last checkpoint readable by the analysis. The sub-folder is
s<seed>, as the supervised checkpoint tree names its run folders: utils.make_subfolder_tag puts a
validation score first in the name, and a maze run has none (its probe success is in the files).
Where a launcher points paths.save_to decides whether the maze runs sit beside the supervised cells
or in a sibling tree; the suite launchers use <save_to>_rl_maze, where every maze run so far is.
Runs written before 5 Nov 2026 by trainRNNbrain/rl/run_maze_ppo.py have the same files without the
stem (config.yaml, Params_Maze.npz, ...) plus PolicyState.pt, a copy of the weights that the
parameter file already holds; the readers (common.maze_load_runs, TrainerPPO.load_run) read both.

A CONTINUATION (since 5 Nov 2026) mirrors the supervised warm start of run_experiment.py:
`paths.init_from` names a finished maze run's folder, whose W_inp, W_rec, W_out and y_init replace
this run's fresh draw before training (load_warm_start); with `paths.init_adam: true` its AdamState
replaces the fresh optimizer state too, and the warm-up ramp is then not run again (TrainerPPO,
adam_restored; configs/trainer/trainer_ppo.yaml says why). `trainer.iter_offset` ("auto" = the
parent's last recorded update, 0 without a parent; warm_start_offset) is resolved here and written
back into the saved config, and every update index in Progress.json and Participation.npz is
iter_offset + the run's own count, so the continuation's records start where the parent's end and
common.maze_load_runs joins the chain into one run. The parent's config names the chain: a run's
`paths.init_from` is its parent. Not carried over, as in the supervised trainer: the rescale rule's
state and the dropout participation EMA, which are not saved, start afresh in the continuation.
"""
import glob
import json
import os
import random
import time
import hydra
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from trainRNNbrain.tasks.TaskMaze import TaskMaze
from trainRNNbrain.trainer.TrainerPPO import TrainerPPO
from trainRNNbrain.utils import filter_kwargs, import_any


def save_run(folder, stem, taskname, cfg, params, progress, probes, opt):
    """Write (overwrite) the five data files of the run folder listed in the module docstring.

    Args:
        folder: str; stem: str, the file prefix (s<seed>); taskname: str, "Maze";
        cfg: the DictConfig, saved resolved; params: dict, RNN_torch.get_params();
        progress: list of dicts, one per update; probes: dict of lists, one entry per probe;
        opt: the torch optimizer, whose state_dict is the AdamState file.
    """
    OmegaConf.save(cfg, os.path.join(folder, f"{stem}_config.yaml"), resolve=True)
    # mappings become 0-d object arrays, as run_experiment.storable_ does; not imported from there
    # because run_experiment imports this module (the dispatch), and the import would be circular
    arrays = {}
    for key, val in params.items():
        if val is None or isinstance(val, (str, bool)):
            continue
        if isinstance(val, DictConfig):
            val = OmegaConf.to_container(val, resolve=True)
        arrays[key] = np.array(val, dtype=object) if isinstance(val, dict) else np.asarray(val)
    # "Params", not "LastParams": the inventories glob *LastParams* for finished supervised runs
    np.savez_compressed(os.path.join(folder, f"{stem}_Params_{taskname}.npz"), **arrays)
    with open(os.path.join(folder, f"{stem}_Progress.json"), "w") as fh:
        json.dump(progress, fh)
    np.savez_compressed(os.path.join(folder, f"{stem}_Participation.npz"),
                        **{k: np.array(v) for k, v in probes.items()})
    # Adam's moment estimates, so a run can be CONTINUED (paths.init_from + paths.init_adam) rather
    # than restarted, as run_experiment saves them: two moments per parameter, and they cannot be
    # reconstructed after the fact. A failure is reported, never allowed to kill the run.
    try:
        torch.save(opt.state_dict(), os.path.join(folder, f"{stem}_AdamState_{taskname}.pt"))
    except Exception as e:
        print(f"WARNING: could not save Adam state: {e}", flush=True)


def load_warm_start(rnn, folder):
    """Copy a finished maze run's weights into a freshly built RNN: the warm start (paths.init_from).

    The counterpart of run_experiment.load_warm_start for the maze layout, whose parameter file is
    *Params_<task>.npz (never *LastParams*, see the module docstring); not imported from there
    because run_experiment imports this module. Every trainable parameter of `rnn` (W_inp, W_rec,
    W_out) is copied from the file with its shape checked, and y_init (the initial state, saved
    too and part of the dynamics; a plain attribute, not a parameter) with it. The RNN must
    already be built from this run's config: the file carries arrays, not an architecture.

    Args:
        rnn: RNN_torch, built from this run's config.
        folder: str, a maze run folder holding one *Params_*.npz.
    Returns:
        str, the path of the file read.
    Raises:
        FileNotFoundError: no *Params_*.npz in `folder`.
        KeyError: a parameter of `rnn` is missing from the file.
        ValueError: a parameter's shape differs between the file and `rnn`.
    """
    src = sorted(glob.glob(os.path.join(folder, "*Params_*.npz")))
    if not src:
        raise FileNotFoundError(f"warm start: no *Params_*.npz in {folder}")
    saved = np.load(src[0], allow_pickle=True)
    with torch.no_grad():
        for name, prm in rnn.named_parameters():
            if name not in saved.files:
                raise KeyError(f"warm start: '{name}' missing from {src[0]}; the file has {sorted(saved.files)}")
            arr = np.asarray(saved[name], dtype=np.float32)
            if tuple(arr.shape) != tuple(prm.shape):
                raise ValueError(f"warm start: '{name}' is {arr.shape} in {src[0]} but {tuple(prm.shape)} "
                                 f"in this model (N or the channel counts differ?)")
            prm.copy_(torch.from_numpy(arr).to(prm.device, prm.dtype))
    rnn.y_init = torch.as_tensor(np.asarray(saved["y_init"], dtype=np.float32),
                                 device=rnn.y_init.device, dtype=rnn.y_init.dtype)
    print(f"warm start: loaded {[n for n, _ in rnn.named_parameters()] + ['y_init']} from {src[0]}", flush=True)
    return src[0]


def warm_start_offset(init_from):
    """The absolute update at which a continuation's clock starts: its parent's last recorded update.

    Read from the parent's *Progress.json, not from its config: a run killed by its wall clock
    leaves a config whose trainer.n_updates is the budget, while its records hold the updates
    completed, and a parent's records are absolute already, so the count accumulates down a
    chain. (The supervised run_experiment.warm_start_offset reads the parent's config, whose
    max_iter its checkpoint stamps with the iterations done; the maze checkpoint overwrites the run
    folder in place and stamps nothing.)

    Args:
        init_from: str, a finished maze run's folder, or None/"" for a run starting from scratch.
    Returns:
        int: the parent's last update index, or 0 when there is no parent.
    Raises:
        FileNotFoundError: no *Progress.json in `init_from`; ValueError: it holds no update.
    """
    if not init_from:
        return 0
    src = sorted(glob.glob(os.path.join(init_from, "*Progress.json")))
    if not src:
        raise FileNotFoundError(f"warm start: no *Progress.json in {init_from}, so the update window "
                                f"this run covers cannot be established")
    with open(src[0]) as fh:
        progress = json.load(fh)
    if not progress:
        raise ValueError(f"warm start: {src[0]} records no update")
    return int(progress[-1]["update"])


def draw_progress(progress, probes, path):
    """Discounted return, success rate and active count against PPO updates, three panels, PNG.

    Args:
        progress: list of dicts with update, discounted_return, success;
        probes: dict with update, probe_success, active; path: str.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    it = [r["update"] for r in progress]
    axes[0].plot(it, [r["discounted_return"] for r in progress], lw=0.8)
    axes[0].set_ylabel("mean discounted return")
    axes[1].plot(it, [r["success"] for r in progress], lw=0.8, label="training batch")
    axes[1].plot(probes["update"], probes["probe_success"], lw=1.2, label="probe (clean, mean action)")
    axes[1].set_ylabel("success rate")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(probes["update"], probes["active"])
    axes[2].set_ylabel("active units (probe)")
    axes[2].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("PPO update")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def draw_trajectories(task, out, path, n_show=16):
    """The maze with the paths of the first n_show probe episodes: green reached the cheese, red did
    not; a dot marks each start, a star the cheese. PNG at path.

    Args:
        task: TaskMaze; out: a rollout dict (xy, steps, success); path: str; n_show: int.
    """
    wall = task.wall.cpu().numpy()
    xy, steps, success = out["xy"].cpu().numpy(), out["steps"].cpu().numpy(), out["success"].cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(wall, cmap="gray_r", origin="upper", extent=(0, wall.shape[1], wall.shape[0], 0))
    goal = task.goal_xy.cpu().numpy()
    ax.plot(goal[0], goal[1], "*", color="gold", ms=14, mec="k")
    for b in range(min(n_show, xy.shape[1])):
        k = int(steps[b]) + 1
        ax.plot(xy[:k, b, 0], xy[:k, b, 1], color="tab:green" if success[b] else "tab:red", lw=1, alpha=0.8)
        ax.plot(xy[0, b, 0], xy[0, b, 1], "o", color="k", ms=3)
    ax.set_title(f"probe: {success.mean():.2f} reached the cheese, {n_show} shown", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_rl(cfg):
    """Build the task, the RNN and the PPO trainer from the config, train, and write the run folder.

    Called by run_experiment.run_training as its first step when the trainer is TrainerPPO.

    Args:
        cfg: the composed DictConfig (task, model, trainer, paths, experiment_tag, seed).
    Returns:
        str, the run folder.
    """
    seed = cfg.seed
    if seed == "random":
        seed = time.time_ns() & 0xFFFFFFFF
    seed = int(seed)
    # Record the resolved seed back into cfg so the SAVED config + seed fully reproduce the run,
    # and seed every RNG source from it (python / numpy / torch global + cuda), as run_experiment
    # does for the supervised path.
    OmegaConf.set_struct(cfg, False)
    cfg.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    taskname = cfg.task.taskname
    trainer_tag, RNN_tag = cfg.trainer.trainer_tag, cfg.model.RNN_tag
    folder_tag = (cfg.experiment_tag + (("_" + trainer_tag) if trainer_tag != "" else "")
                  + (("_" + RNN_tag) if RNN_tag != "" else ""))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # the task block as the constructor takes it (prepare_task_arguments is not used: the maze
    # has no trial length to convert; an episode's length in decisions is the trainer's T)
    task_cls = import_any(cfg.task._target_)
    task = hydra.utils.instantiate(filter_kwargs(task_cls, cfg.task), device=device)
    # recorded, not chosen: the inputs follow from the task's sensors, the outputs from the
    # algorithm (a third read-out row is the critic's value)
    cfg.task.n_inputs, cfg.task.n_outputs = task.n_inputs, task.n_outputs
    cfg.model.n_inputs = task.n_inputs
    cfg.model.n_outputs = task.n_outputs + (1 if cfg.trainer.critic else 0)
    rnn_cls = import_any(cfg.model._target_)
    rnn_args = filter_kwargs(rnn_cls, cfg.model)
    rnn_args.seed = seed
    rnn = hydra.utils.instantiate(rnn_args)
    lr = cfg.trainer.lr * (100.0 / float(cfg.model.N)) ** cfg.trainer.lr_scale_exp   # the trainer's size rule
    opt = torch.optim.Adam(rnn.parameters(), lr=lr, weight_decay=cfg.trainer.weight_decay)
    # WARM START (paths.init_from): a finished maze run's weights replace this run's fresh draw, and
    # with paths.init_adam its Adam moments replace the fresh optimizer state, so training continues
    # from that network; the counterpart of run_experiment's warm-start block. Everything else comes
    # from THIS run's config: the trainer's options, the task, and the rate (the size rule above; a
    # restored state carries the parent's last rate and weight decay in its param_groups, and those
    # are set back to this run's values). Two parts of the parent's trainer are not saved and start
    # afresh, as in the supervised trainer: the rescale rule's state (strikes, utilities, factors,
    # the growing set, the step clock) and the dropout participation EMA.
    init_from = cfg.paths.get("init_from", None) if "paths" in cfg else None
    init_adam = bool(init_from) and bool(cfg.paths.get("init_adam", False))
    if init_from:
        load_warm_start(rnn, init_from)
        if init_adam:
            ast = sorted(glob.glob(os.path.join(init_from, "*AdamState*.pt")))
            if not ast:
                raise FileNotFoundError(f"warm start: init_adam set but no *AdamState*.pt in {init_from}")
            opt.load_state_dict(torch.load(ast[0], map_location=device))
            for group in opt.param_groups:
                group["lr"], group["weight_decay"] = lr, float(cfg.trainer.weight_decay)
            print(f"warm start: restored Adam moments from {ast[0]}", flush=True)
    # THE UPDATE CLOCK: trainer.iter_offset "auto" means "carry on counting from the parent", 0 when
    # there is no parent. Resolved here and written back, so the saved config records the absolute
    # window this run covers, as run_experiment records it for the supervised path.
    off = cfg.trainer.get("iter_offset", "auto")
    if not isinstance(off, (int, float)):
        off = warm_start_offset(init_from)
    cfg.trainer.iter_offset = int(off)
    if off:
        print(f"update clock: this run records updates {off + 1}-{off + int(cfg.trainer.n_updates)}", flush=True)
    trainer_cls = import_any(cfg.trainer._target_)
    trainer = hydra.utils.instantiate(filter_kwargs(trainer_cls, cfg.trainer),
                                      RNN=rnn, Task=task, optimizer=opt, adam_restored=init_adam,
                                      _convert_="none")

    folder = os.path.join(cfg.paths.save_to, f"{taskname}_{folder_tag}", f"s{seed}")
    stem = f"s{seed}"
    os.makedirs(folder, exist_ok=True)
    print(f"maze PPO: N={cfg.model.N} inputs={task.n_inputs} goal_cue={task.goal_cue} lr={lr:.2e} "
          f"device={device}\n -> {folder}", flush=True)

    def write_run(params):
        """Write the run folder and its two figures from the trainer's records so far.

        Args:
            params: dict, RNN_torch.get_params() at this point of training.
        """
        save_run(folder, stem, taskname, cfg, params, trainer.progress, trainer.probes, opt)
        draw_progress(trainer.progress, trainer.probes, os.path.join(folder, "progress.png"))
        draw_trajectories(task, trainer.last_probe, os.path.join(folder, "trajectories.png"))

    def checkpoint_fn(updates_done, params):
        """One checkpoint: the run folder overwritten in place.

        Args:
            updates_done: int, PPO updates completed; params: the RNN's parameters.
        """
        write_run(params)

    trainer.checkpoint_fn = checkpoint_fn
    rnn, progress, probes, last_probe = trainer.run_training()
    write_run(rnn.get_params())
    print(f"-> {folder}", flush=True)
    return folder
