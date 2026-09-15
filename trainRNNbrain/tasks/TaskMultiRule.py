"""Many tasks in one network, Yang-style: a rule input says which task the trial is.

Why: a network trained on one task recruits only what that task demands (~10 units per latent
function, see docs/project_trajectory.md). A cortical circuit is never doing only one task, so its
recorded population is busy for reasons the task at hand does not explain. This class trains ONE
network on a list of the repo's tasks, each trial tagged by a one-hot rule input held on for the
whole trial, so the recruitment on a single task (CDDM) can be read inside a network that also
serves fourteen others.

Shared input/output space: input = [rule one-hot (n_tasks rows) | the subtask's own inputs, padded
with zero rows to the widest subtask]; output = the subtask's outputs padded with zero rows to the
widest subtask. Trials shorter than n_steps are padded with zeros, and the padding IS scored (the
target is 0 there: return to rest after the trial). Scoring windows differ between subtasks, so a
batch carries a per-trial mask (`batch_mask`, shape (n_steps, batch), True where scored) that the
Trainer uses instead of the global time mask when the task provides one.

Every subtask is built from its own `configs/task/<name>.yaml` through the same
prepare_task_arguments / get_training_mask path as a single-task run, so its timing, channels and
scoring are exactly those of the single-task reference. `dt` must equal the model's dt.
"""
import os

import hydra
import numpy as np
from omegaconf import OmegaConf

from trainRNNbrain.tasks.TaskBase import Task
from trainRNNbrain.training.training_utils import get_training_mask, prepare_task_arguments

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "task")
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


class TaskMultiRule(Task):
    def __init__(self, n_steps, n_inputs, n_outputs, subtask_names, n_per_task, dt=1, seed=None):
        """Build the subtasks from their configs and check the shared channel space.

        Args:
            n_steps: trial length of the composite (>= the longest subtask);
            n_inputs: must equal len(subtask_names) + the widest subtask input count;
            n_outputs: must equal the widest subtask output count;
            subtask_names: list of config names under configs/task/ (without .yaml);
            n_per_task: trials drawn from each subtask per batch (batch = n_per_task * n_tasks);
            dt: integration step used to convert the subtasks' T_* fields to steps (= model.dt);
            seed: seeds this class's rng AND, through it, every subtask's rng.
        """
        Task.__init__(self, n_steps, n_inputs, n_outputs, seed)
        self.subtask_names = list(subtask_names)
        self.n_per_task = int(n_per_task)
        self.dt = dt
        self.subtasks, self.sub_T, self.sub_mask, self.sub_nin, self.sub_nout = [], [], [], [], []
        for name in self.subtask_names:
            cfg = OmegaConf.load(os.path.join(CONFIG_DIR, name + ".yaml"))
            cfg = OmegaConf.create({"task": cfg}).task           # resolve ${..T}-style references
            OmegaConf.set_struct(cfg, False)
            cfg.seed = int(self.rng.integers(2 ** 31 - 1)) if seed is not None else None
            # i.i.d.-trial tasks expose batch_size; drawing more than n_per_task is wasted (the
            # Walsh expansion alone cost 115 ms per 1024-trial batch, 55% of the composite's time)
            if "batch_size" in cfg and int(cfg.batch_size) > self.n_per_task:
                cfg.batch_size = self.n_per_task
            targs = prepare_task_arguments(cfg_task=cfg, dt=dt)
            sub = hydra.utils.instantiate(targs)
            if getattr(sub, "batch_size", 0) > self.n_per_task:      # class-default batch sizes too
                sub.batch_size = self.n_per_task
            self.subtasks.append(sub)
            self.sub_T.append(int(targs.n_steps))
            self.sub_mask.append(get_training_mask(cfg_task=cfg, dt=dt))
            self.sub_nin.append(int(cfg.n_inputs))
            self.sub_nout.append(int(cfg.n_outputs))
        self.n_tasks = len(self.subtasks)
        self.max_in, self.max_out = max(self.sub_nin), max(self.sub_nout)
        if n_inputs != self.n_tasks + self.max_in or n_outputs != self.max_out:
            raise ValueError(f"MultiRule: n_inputs must be {self.n_tasks + self.max_in} "
                             f"(= {self.n_tasks} rules + {self.max_in}) and n_outputs {self.max_out}; "
                             f"got {n_inputs}, {n_outputs}")
        if max(self.sub_T) > n_steps:
            raise ValueError(f"MultiRule: n_steps {n_steps} shorter than the longest subtask {max(self.sub_T)}")

    def embed_(self, i, inp, tgt):
        """Place one subtask's trials into the shared channel/time space.

        Args:
            i: subtask index; inp: (n_in_i, T_i, B); tgt: (n_out_i, T_i, B).
        Returns:
            (inputs (n_inputs, n_steps, B), targets (n_outputs, n_steps, B), scored (n_steps,) bool).
        """
        B = inp.shape[-1]
        X = np.zeros((self.n_inputs, self.n_steps, B))
        Y = np.zeros((self.n_outputs, self.n_steps, B))
        X[i, :, :] = 1.0                                         # rule on for the whole trial
        X[self.n_tasks:self.n_tasks + self.sub_nin[i], :self.sub_T[i], :] = inp
        Y[:self.sub_nout[i], :self.sub_T[i], :] = tgt
        scored = np.zeros(self.n_steps, dtype=bool)
        scored[self.sub_mask[i]] = True
        scored[self.sub_T[i]:] = True                            # padding: hold the outputs at 0
        return X, Y, scored

    def get_batch(self, shuffle=False):
        """A mixed batch: n_per_task trials from every subtask, grouped by task.

        Subtasks whose own batch is smaller than n_per_task are sampled with replacement (their
        trials differ through the RNN's input noise, and jitter where the task has it).

        Args:
            shuffle: permute the trials across tasks (the gradient does not depend on order).
        Returns:
            (inputs (n_inputs, n_steps, B), targets (n_outputs, n_steps, B), conditions) with
            B = n_per_task * n_tasks; each condition is {"task", "task_idx", "scored", "sub"} where
            "scored" is the trial's (n_steps,) bool scoring mask and "sub" the subtask's own dict.
        """
        Xs, Ys, conds = [], [], []
        for i, sub in enumerate(self.subtasks):
            inp, tgt, cond = sub.get_batch()
            B = inp.shape[-1]
            idx = self.rng.choice(B, self.n_per_task, replace=B < self.n_per_task)
            X, Y, scored = self.embed_(i, inp[..., idx], tgt[..., idx])
            Xs.append(X); Ys.append(Y)
            conds += [{"task": self.subtask_names[i], "task_idx": i, "scored": scored, "sub": cond[j]}
                      for j in idx]
        inputs, targets = np.concatenate(Xs, axis=2), np.concatenate(Ys, axis=2)
        if shuffle:
            perm = self.rng.permutation(inputs.shape[-1])
            inputs, targets = inputs[..., perm], targets[..., perm]
            conds = [conds[k] for k in perm]
        return inputs, targets, conds

    def task_batch(self, i):
        """The full batch of ONE subtask in the shared space, for per-task read-outs.

        Args:
            i: subtask index (or its config name).
        Returns:
            (inputs, targets, conditions) as get_batch, containing only that subtask's trials.
        """
        if isinstance(i, str):
            i = self.subtask_names.index(i)
        inp, tgt, cond = self.subtasks[i].get_batch()
        X, Y, scored = self.embed_(i, inp, tgt)
        conds = [{"task": self.subtask_names[i], "task_idx": i, "scored": scored, "sub": c} for c in cond]
        return X, Y, conds

    @staticmethod
    def batch_mask(conditions):
        """Per-trial scoring mask of a batch, (n_steps, B) bool, from the conditions get_batch returned."""
        return np.stack([c["scored"] for c in conditions], axis=1)
