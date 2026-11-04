"""Rebuild a trained maze agent from its run folder (config.yaml + PolicyState.pt)."""
import os
import hydra
import torch
from omegaconf import OmegaConf
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.utils import filter_kwargs
from trainRNNbrain.rl.maze_env import MazeEnv
from trainRNNbrain.rl.policy import MazePolicy


def load_run(folder, device=None):
    """Rebuild the env and the trained policy of a run folder.

    Args:
        folder: str, a run folder written by run_maze_ppo.py (config.yaml, PolicyState.pt).
        device: torch.device, or None for the CPU.
    Returns:
        (cfg, env, policy): the saved DictConfig, the MazeEnv, the MazePolicy with the trained
        weights. Runs written before 4 Nov 2026 (no `env.action_feedback` key; a learned log-std
        and a separate value head in the state_dict) load too: the key defaults to False and the
        extra tensors are skipped, so their clean rollout uses the RNN's read-out alone, as their
        training probe did.
    """
    device = torch.device("cpu") if device is None else device
    cfg = OmegaConf.load(os.path.join(folder, "config.yaml"))
    env_args = dict(cfg.env)
    env_args.setdefault("action_feedback", False)
    env = MazeEnv(**env_args, device=device, seed=cfg.seed + 1)
    args = filter_kwargs(RNN_torch, cfg.model)
    args.seed = cfg.seed
    rnn = hydra.utils.instantiate(args)
    rnn.to(device)
    policy = MazePolicy(rnn, cfg.ppo.get("action_std", 0.5), cfg.ppo.substeps)
    policy.load_state_dict(torch.load(os.path.join(folder, "PolicyState.pt"), map_location=device), strict=False)
    return cfg, env, policy
