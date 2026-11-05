## 2026-11-04 21:27 — Maze: the paper's interventions as TrainerPPO options, mirrored from the supervised definitions (issue #5, stage 1)

Branch `wp/maze-interventions` (worktree `~/Documents/Projects/paper_push/wt/maze-interventions-code`),
from `main` at 9cc1aac. Commits 30f5be7 (trainer, config, checks) and 39aad46 (launcher). Not merged,
not pushed, nothing submitted. Issue #5 is the spec: the five interventions of Figure 5, of which four
are built here; rescaling is stage 2, prune-and-duplicate is deferred.

**What each option does and where** (`trainRNNbrain/trainer/TrainerPPO.py`; every docstring quotes
the supervised lines it mirrors; keys in `configs/trainer/trainer_ppo.yaml`, all off by default).

| option | config key | where it acts | the supervised source |
|---|---|---|---|
| rate penalty | `trainer.lambda_frm`, `frm_args` (cap_fr 0.3, tau 0.1, g_top 3, g_bot 3, eps 1e-12) | `fr_magnitude_penalty(states)` on the states `sequence` returns in every minibatch's recompute (every RNN step of every decision, the noisy rollout states, including steps after an episode ended); `loss += lambda_frm x P` in `ppo_update` | `Trainer.Penalties.fr_magnitude_penalty`: soft-max rate per unit over steps and episodes, cap 0.3 x log1p(100) / log1p(N), ((cap - a) / cap)^3 below and ((a - cap) / cap)^3 above, powers in float64, mean over units |
| sparsity penalty | `trainer.lambda_rws`, `rws_args` (tg_deg 20, eps 1e-12) | `rec_weights_sparsity_penalty()` on the raw `W_rec`; `loss += lambda_rws x P` | `Trainer.Penalties.rec_weights_sparsity_penalty`: S = (sum\|W\|)^2 / sum W^2 per row, relu(S - 20)^2 mean / 400 |
| noisy synapses | `model.sigma_w` (the RNN's own key, read as the supervised trainer reads it; the ValueError that refused it is gone) | `rollout` draws one seed per rollout from the RNN's generator; `synaptic_weights(w_seed, s)` gives W_rec x (1 + sigma_w xi) and W_inp x (1 + sigma_w xi) for step s from a generator seeded with w_seed x 2^20 + s; `advance` uses them in the rollout and in the recompute; the probe (clean) draws no seed | `RNN_torch.forward`: a fresh multiplicative draw on both matrices at every step, one draw shared across the batch, noisy passes only |
| read-out dropout | `trainer.dropout`, `dropout_args` (mute, participation, eta 0.5, active_rel 0.05, drop_rate 0.2, rescale, p_max 0.9, dropout_beta 4, activity_q 0.9; drop_rate and beta at the paper's values) | `draw_dropout_mask()` calls `RNN_torch.get_dropout_mask` on the EMA `p_bar` once per training rollout; `readout_weights(outgoing)` zeroes the dropped columns of W_out and scales the kept ones, in `heads` of the rollout and of the recompute; `participation_from_states` takes p = std + q0.9 of the rollout's states and `run_training` updates `p_bar <- (1 - eta) p_bar + eta p` after the mask of that rollout was drawn from the old value; the probe uses no mask; kind "dead" is refused | `RNN_torch.get_dropout_mask`, `drop_probabilities`, `Trainer.participation_from_states_`, the EMA in `Trainer.train_step` |

The penalty values are recorded per update in `Progress.json` under `fr_magnitude` and
`rec_weights_sparsity` (the raw P, before its lambda), only when the option is on, so a run without
the options writes the same keys as before.

**Not mirrored to the letter, and why.** (1) `forward` draws the synaptic noise from the RNN's running
generator; here each step's draw comes from its own seed (w_seed x 2^20 + s), so that a step can be
regenerated on its own. The backward pass needs that: the noisy step runs through
`torch.utils.checkpoint`, and autograd would otherwise store the perturbed W_rec and its multiplier of
every step, 2 x 4 MB x 2,505 steps = 20 GB per minibatch at N = 1,000, the very tensors the issue
says cannot be stored. The distribution of the draws is the same (independent normals per step); the
cost is the recompute's forward pass once more, so the sigma_w arm runs slower (allow about 1.5x).
(2) The participation chunk: `torch.quantile` refuses inputs above 2^24 entries, and the supervised
code's 512 units per chunk would be 328M entries at 2,505 steps x 256 episodes, so the chunk is the
largest that fits (26 units at N = 1,000); the per-unit values are the same. (3) The rate penalty
pools over every RNN step the recompute produces, including the (T + 1)-th decision's bootstrap
steps and the steps after an episode ended; the supervised penalty pools over every step of the
batch too, but the maze has no analogue of a trial's end. (4) The first dropout mask of a run is
drawn from the 1e-6 start of the EMA, with every unit in the pool at equal participation, as the
supervised trainer's first iteration is. (5) The GPU path (a CUDA generator per step) was not run
here; the Mac has no CUDA device.

**Equivalence with every option off, pass rule set first: every saved number identical.** Main's code
at 9cc1aac and this branch, both at N = 40, 3 updates of 8 episodes x 12 decisions, probe of 6
episodes, warm-up 2, seed 7, CPU (the refactor's recipe): `Progress.json` has the same keys in every
update with largest difference 0; W_rec, W_inp, W_out and y_init equal; every `Participation.npz`
array equal.

**Checks.** `tests/test_rl_maze.py` 19 of 19: the 13 earlier ones unchanged, plus the rate penalty of a
hand-made state tensor against the formula re-typed in numpy float64 (a silent unit pays exactly 1, a
unit at the cap pays nothing; relative difference under 1e-5), the sparsity penalty of a hand-made
W_rec against the re-typed formula (rows with S = 1, 20 and 32 as constructed), the synaptic-noise
replay (the recompute's log-probabilities equal the rollout's within the existing check's 1e-5, ratio
1 within 1e-4, with and without autograd; another seed or no noise changes the means; the gradient
reaches W_rec; a clean probe draws no seed), the dropout replay (dropped columns zero and kept pool
columns above 1 in the read-out both passes use; log-probabilities equal, ratio 1; the mask changes
the means; the by-hand masked read-out gives the means; the rollout's participation equals
`common.participation` on the recomputed states; a clean probe draws no mask; `run_training` moves
the EMA), the statistics keys (present and equal to the direct penalty values with one epoch and one
minibatch; absent at lambda 0) and the refusal of kind "dead". `tests/test_no_function_imports.py`
202 files ok; `tests/test_scripts_start.py` 48 scripts ok. The existing recompute is not bit-identical
to the rollout (differences of 5e-7 from the einsum on one decision's states against all at once),
which is why the replay checks use the same tolerances as the earlier ratio check.

**Smoke run, every option on.** `run_experiment.py task=Maze trainer=trainer_ppo model.N=40 seed=3
trainer.lambda_frm=0.1 trainer.lambda_rws=0.05 trainer.dropout=true model.sigma_w=1.0`, 2 updates of
16 episodes x 30 decisions, into the scratchpad: the six standard files written; update 2 statistics
`discounted_return -0.174, success 0.06, steps 29.8, policy_loss -0.0029, approx_kl -0.0011,
clip_frac 0.029, fr_magnitude 0.554, rec_weights_sparsity 0.062, skipped 0`; the saved config records
sigma_w 1.0, lambda_frm 0.1, lambda_rws 0.05, dropout true with drop_rate 0.2 and beta 4.0.

**Launcher.** `slurm/rl_maze_interventions_spock.slurm` (force-added; the suite launcher's provenance
block and Spock header): 7 arms x 3 seeds at N = 1,000, `experiment_tag="interventions/<arm>"`, arms
control, frm0.03, frm0.1, frm0.3, frm0.1_rws0.05, sigmaw1.0, dropout0.2_beta4; the command line
carries only the arm's swept keys and N. Run locally with `module`, `conda` and `srun` stubbed (the
stub appends N = 40, 2 updates) for one seed of each arm, pass rule set first: each arm's saved
config differs from the control's in exactly its swept keys plus the tag. Passed for all six arms;
the provenance line recorded 39aad46. Not submitted: about 21 x 5.5 h on Spock, the sigma_w arm
slower, per the pre-registration in issue #5.
