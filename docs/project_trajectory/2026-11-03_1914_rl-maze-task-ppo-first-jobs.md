## 2026-11-03 19:14 — The maze task: an RL line to test dormant units under PPO (engellab/trainRNNbrain issue #2, first jobs)

**Why.** Pavel, 3 Nov: instead of DMTS, use a reinforcement-learning task. A maze with the
cheese at the centre; the agent is dropped somewhere; the network outputs left-or-right steering
and the running speed; train with PPO; then see whether units go dormant on that task too. The
infrastructure had to sit beside the supervised path without touching it, with readability second
only to training speed. Progress and the reviews of this line are posted on
engellab/trainRNNbrain issue #2, as he asked.

**The task** (trainRNNbrain/rl/maze_env.py, config configs/rl/maze_ppo.yaml). One fixed 11 x 11
ASCII maze, the cheese at the centre, reachable only through an inner ring from above or below;
the far corridor is 24 cells away by the geodesic. Each episode every agent is dropped in a random
reachable cell with a random heading. Inputs: 8 range sensors over 360 degrees, egocentric, each
1 - d/4 with d the distance to the first wall (cells); the "cue" arm adds cos and sin of the
bearing to the cheese. Outputs: turn (clipped to [-1, 1], 0.3 rad per step at 1) and speed
(clipped to [0, 1], 0.2 cells per step at 1). Reward: +5 at the cheese (within 0.5 cells; ends the
episode), -0.01 per step, -0.05 per blocked move, and potential-based shaping 0.2 (d - 0.99 d')
on the breadth-first geodesic cell distance, which leaves the optimum unchanged. Episodes last at
most 300 steps.

**The network and the training** (policy.py, ppo.py, run_maze_ppo.py). The standard RNN
(RNN_torch, configs/model/rnn_relu_standard.yaml, unchanged: ReLU, equation h, bias 0, alpha 0.1,
sigma_rec 0.05, sigma_inp 0.05, sigma_out 0.03, weight decay 1e-6, lr 1e-3 (100/N)^(1/3)). One
environment step is one RNN step through the RNN's own rhs. The policy mean is W_out r with 2
outputs; one learned log-std per output (init -0.7); a linear value head on the rates. PPO:
256 episodes per update, 4 epochs of 2 minibatches, clip 0.2, gamma 0.99, GAE lambda 0.95, value
weight 0.5, no entropy bonus, gradient norm clipped at 0.5, 1,500 updates. The RNN's noise is drawn
before the rollout and replayed in the update, so the probability ratio starts at exactly 1
(tests/test_rl_maze.py checks it). Output noise is one draw per episode, held constant, as
RNN_torch.forward does; recurrent and input noise are fresh per step.

**Read-out, fixed before the jobs ran.** Every 10 updates a probe: a noise-free rollout with the
policy mean on 128 fixed drops (probe_seed 20261103); participation p = std(r) + q0.9(|r|) per unit
over the steps before the cheese; active if p >= 0.05 q95(p) (common.active_mask). Saved per run:
the active count per probe, the full participation vector every 5th probe, the parameters
(get_params, Params_Maze.npz), the resolved config, the discounted return and the success rate per
update, a progress figure and a trajectory figure. Runs live in <save_to>_rl_maze/, a sibling of
trained_RNNs, so no inventory of supervised cells sees them.

**Pass/fail.** The task is usable if, in the cue arm, the probe success rate reaches 0.8 by the end
of training at both sizes. The blind arm (walls only) is the harder one; its result is a
measurement, not a pass condition. Dormant units: report the active count at the last probe
against N, the same way the supervised tasks are reported.

**What runs.** Della, launcher slurm/rl_maze_ppo_della.slurm from the detached worktree
~/trainRNNbrain_rlmaze pinned at 834cc50 (clean). 8 jobs: N in {500, 1000} x arms {blind, cue} x
seeds {1, 2}; 4 h cap each, under 32 GPU-h in all. Command: cd ~/trainRNNbrain_rlmaze &&
sbatch slurm/rl_maze_ppo_della.slurm. Array 16790695, tasks 1-8, submitted at 2026-11-03 19:14.
Outputs: /home/pt1290/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_first/N=<N>_cue=<bool>/s<seed>/;
logs ~/trainRNNbrain_logs/MazePPO.16790695_<i>.out. Timing on the Mac CPU at N = 500: one
rollout 1.0 s, one PPO epoch 0.6 s, so about 3.5 s per update and 1.5 h per run; the GPU should be
no slower. Spock was unreachable from this session (no open ControlMaster socket), so nothing ran
there.

**Reviews.** Two independent agents reviewed the working tree before the commit and posted on
issue #2: correctness (approve, 7 findings) and readability and conventions (7 required changes).
All were applied; the resolution is the third comment on the issue. Local checks at the commit:
tests/test_rl_maze.py 7 of 7 on the Mac and on Della's login node (torch 2.7); no function-level
imports; every function documented.

**Next.** When the array finishes: read Progress.json and Participation.npz of every run, draw
probe success and active count against updates for the four cells, and write the read-out as a
new entry; then decide whether the blind arm needs a longer budget or an easier maze.
