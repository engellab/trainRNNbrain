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
"""
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


def save_run(folder, stem, taskname, cfg, params, progress, probes):
    """Write (overwrite) the four data files of the run folder listed in the module docstring.

    Args:
        folder: str; stem: str, the file prefix (s<seed>); taskname: str, "Maze";
        cfg: the DictConfig, saved resolved; params: dict, RNN_torch.get_params();
        progress: list of dicts, one per update; probes: dict of lists, one entry per probe.
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
    trainer_cls = import_any(cfg.trainer._target_)
    trainer = hydra.utils.instantiate(filter_kwargs(trainer_cls, cfg.trainer),
                                      RNN=rnn, Task=task, optimizer=opt, _convert_="none")

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
        save_run(folder, stem, taskname, cfg, params, trainer.progress, trainer.probes)
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
