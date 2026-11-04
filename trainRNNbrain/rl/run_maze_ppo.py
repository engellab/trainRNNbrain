"""Train the standard RNN with PPO to run a fixed maze to the cheese, and record how many of its
units take part.

Entry point:  python trainRNNbrain/rl/run_maze_ppo.py model.N=500 seed=1 env.goal_cue=true
Config: configs/rl/maze_ppo.yaml (the maze, the sensors, the rewards, PPO) on top of the project's
standard model config (configs/model/rnn_relu_standard.yaml) and the paths group. This script reads
the resolved config and overrides nothing; a launcher varies only the keys it sweeps. Three values
are written back into the saved config as a record of what was used: the seed (when "random"), and
model.n_inputs and model.n_outputs, which follow from the env, so that the network can be rebuilt
from the saved config alone. Seeds: the RNN (its initial weights and its noise) is seeded with
`seed`, the env's drops with `seed + 1`, so the two streams differ.

Output, under <paths.save_to>_rl_maze/Maze_<experiment_tag>/s<seed>/ (a sibling of the supervised
runs' folder, so no inventory of supervised cells ever sees these files):
  config.yaml         the resolved config, with the seed actually used written back
  Params_Maze.npz     RNN_torch.get_params(): W_rec, W_inp, W_out, y_init and the rest
  PolicyState.pt      the full policy state_dict (RNN, log_std, value head) for an exact rebuild
  Progress.json       one dict per update: update, discounted_return (mean over the batch of
                      sum_t gamma^t r_t, the quantity PPO maximises), success (share of the batch
                      that reached the cheese), steps (mean steps to the cheese, T if never),
                      log_std, and the PPO statistics from ppo_update
  Participation.npz   per probe: update (the update index), active (active units under the
                      project's rule), probe_success (share of probe episodes that reached the
                      cheese); and every store_participation_every-th probe's full participation
                      vector: p (n_stored, N) with its update index p_update (n_stored,)
  progress.png        discounted return, success and active count against updates
  trajectories.png    the maze with the probe episodes' paths at the last checkpoint
A probe is a noise-free rollout with the policy mean on a fixed batch of drops (the same drops at
every probe of every run, ppo.probe_seed), the RL counterpart of the noise-free forward pass the
supervised trainer probes with. Participation is taken over the steps before the cheese is reached.
"""
import json
import os
import time
import hydra
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.utils import filter_kwargs
from trainRNNbrain.experiments_and_analysis.common import participation, active_mask
from trainRNNbrain.rl.maze_env import MazeEnv
from trainRNNbrain.rl.policy import MazePolicy
from trainRNNbrain.rl.ppo import rollout, gae, ppo_update


def probe(env, policy, n_episodes, T, probe_seed):
    """A noise-free, mean-action rollout on the fixed probe drops.

    Args:
        env: MazeEnv; policy: MazePolicy; n_episodes, T: ints, the probe batch and episode length;
        probe_seed: int, seeds the drops (the same every call).
    Returns:
        (n_active, p, out): n_active int, active units under common.active_mask; p (N,) float
        participation over every RNN step of the probe before the cheese; out, the rollout dict.
    """
    g = torch.Generator(device=env.device)
    g.manual_seed(int(probe_seed))
    out = rollout(env, policy, n_episodes, T, clean=True, generator=g)
    k = policy.substeps
    with torch.no_grad():
        states = policy.sequence(out["u"], out["noise"])[0][:, :T * k]          # (N, T * k, B)
    rates = policy.rnn.get_firing_rates(states).cpu().numpy()
    alive = out["alive"][:-1].bool().repeat_interleave(k, dim=0).cpu().numpy()  # (T * k, B)
    p = participation(rates[:, alive])
    return int(active_mask(p).sum()), p, out


def save_run(folder, cfg, policy, progress, probes):
    """Write (overwrite) every file of the run folder listed in the module docstring.

    Args:
        folder: str; cfg: the resolved DictConfig; policy: MazePolicy;
        progress: list of dicts, one per update; probes: dict of lists, one entry per probe.
    """
    OmegaConf.save(cfg, os.path.join(folder, "config.yaml"))
    # mappings become 0-d object arrays, as run_experiment.storable_ does; not imported from there
    # because importing run_experiment pulls in the whole supervised path and registers an
    # OmegaConf resolver as a side effect
    arrays = {}
    for key, val in policy.rnn.get_params().items():
        if val is None or isinstance(val, (str, bool)):
            continue
        if isinstance(val, DictConfig):
            val = OmegaConf.to_container(val, resolve=True)
        arrays[key] = np.array(val, dtype=object) if isinstance(val, dict) else np.asarray(val)
    np.savez_compressed(os.path.join(folder, "Params_Maze.npz"), **arrays)
    torch.save(policy.state_dict(), os.path.join(folder, "PolicyState.pt"))
    with open(os.path.join(folder, "Progress.json"), "w") as fh:
        json.dump(progress, fh)
    np.savez_compressed(os.path.join(folder, "Participation.npz"),
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


def draw_trajectories(env, out, path, n_show=16):
    """The maze with the paths of the first n_show probe episodes: green reached the cheese, red did
    not; a dot marks each start, a star the cheese. PNG at path.

    Args:
        env: MazeEnv; out: a rollout dict (xy, steps, success); path: str; n_show: int.
    """
    wall = env.wall.cpu().numpy()
    xy, steps, success = out["xy"].cpu().numpy(), out["steps"].cpu().numpy(), out["success"].cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(wall, cmap="gray_r", origin="upper", extent=(0, wall.shape[1], wall.shape[0], 0))
    goal = env.goal_xy.cpu().numpy()
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


@hydra.main(version_base="1.3", config_path="../../configs", config_name="rl/maze_ppo")
def main(cfg: DictConfig) -> None:
    """Train one policy as the config says and write its run folder (module docstring)."""
    seed = (time.time_ns() & 0xFFFFFFFF) if cfg.seed == "random" else int(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg.seed = seed                       # record the seed actually used, as run_experiment does
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = cfg.ppo
    env = MazeEnv(**cfg.env, device=device, seed=seed + 1)
    cfg.model.n_inputs, cfg.model.n_outputs = env.n_inputs, env.n_actions   # recorded, not chosen
    rnn_args = filter_kwargs(RNN_torch, cfg.model)
    rnn_args.seed = seed
    rnn = hydra.utils.instantiate(rnn_args)
    policy = MazePolicy(rnn, hp.log_std_init, hp.log_std_min, hp.substeps)
    lr = hp.lr * (100.0 / float(cfg.model.N)) ** hp.lr_scale_exp      # the trainer's size rule
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=hp.weight_decay)
    folder = os.path.join(f"{cfg.paths.save_to}_rl_maze", f"Maze_{cfg.experiment_tag}", f"s{seed}")
    os.makedirs(folder, exist_ok=True)
    discount = hp.gamma ** torch.arange(hp.T, device=device, dtype=torch.float32)[:, None]
    print(f"maze PPO: N={cfg.model.N} inputs={env.n_inputs} goal_cue={env.goal_cue} lr={lr:.2e} "
          f"device={device}\n -> {folder}", flush=True)

    progress = []
    probes = {"update": [], "active": [], "probe_success": [], "p_update": [], "p": []}
    out = None
    tic = time.perf_counter()
    for it in range(1, hp.n_updates + 1):
        batch = rollout(env, policy, hp.n_episodes, hp.T)
        adv, ret = gae(batch["rewards"], batch["values"], batch["alive"], hp.gamma, hp.gae_lambda)
        stats = ppo_update(policy, optimizer, batch, adv, ret, hp.clip, hp.n_epochs,
                           hp.n_minibatches, hp.value_coef, hp.entropy_coef, hp.max_grad_norm)
        progress.append({"update": it,
                         "discounted_return": (batch["rewards"] * discount).sum(dim=0).mean().item(),
                         "success": batch["success"].float().mean().item(),
                         "steps": batch["steps"].mean().item(),
                         "log_std": [float(v) for v in policy.std().log().detach().cpu()],
                         **stats})
        if it == 1 or it % hp.probe_every == 0:
            n_active, p, out = probe(env, policy, hp.probe_episodes, hp.T, hp.probe_seed)
            probes["update"].append(it)
            probes["active"].append(n_active)
            probes["probe_success"].append(out["success"].float().mean().item())
            if (len(probes["update"]) - 1) % hp.store_participation_every == 0:
                probes["p_update"].append(it)
                probes["p"].append(p)
            r = progress[-1]
            print(f"update {it:5d}/{hp.n_updates}  disc. return {r['discounted_return']:7.3f}"
                  f"  success {r['success']:.2f}  steps {r['steps']:6.1f}"
                  f"  probe success {probes['probe_success'][-1]:.2f}"
                  f"  active {n_active}/{cfg.model.N}  kl {r['approx_kl']:.4f}  clip {r['clip_frac']:.2f}"
                  f"  {(time.perf_counter() - tic) / it:.2f} s/update", flush=True)
        if it % hp.checkpoint_every == 0 or it == hp.n_updates:
            save_run(folder, cfg, policy, progress, probes)
            draw_progress(progress, probes, os.path.join(folder, "progress.png"))
            draw_trajectories(env, out, os.path.join(folder, "trajectories.png"))
    print(f"done: {hp.n_updates} updates in {time.perf_counter() - tic:.0f} s -> {folder}", flush=True)


if __name__ == "__main__":
    main()
