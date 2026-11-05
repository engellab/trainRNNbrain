## 2026-11-04 22:33 — Maze rescaling: the rule once per gradient step, on the supervised clock (issue #5, stage 2; amends 2026-11-04_2223_maze-rescale-as-trainer-option.md)

Branch `wp/maze-rescale`, commits 58401c2 (trainer, config, checks) and d4be6e6 (launcher comment),
after the coordinator's change request. Not merged, not pushed, nothing submitted. The entry this
amends described the rule as running once per update on the rollout's states; that version was
never run on a cluster.

**What changed.** `TrainerPPO.prune_and_reinit` now runs inside `ppo_update` after every
`optimizer.step()`, on the states `sequence` recomputed for that minibatch, detached (the noisy
rollout states of the minibatch's n_episodes / n_minibatches episodes under the weights before the
step, every RNN step of every decision including the (T + 1)-th decision's bootstrap steps, as the
rate penalty pools them). The order inside one optimizer step is the supervised train_step's:
forward, backward, clip, blank the growing units' incoming rows (`zero_protected_grads`, unchanged),
step, check. The clock `step_n` counts optimizer steps completed before the check (0 at the first),
so check_every, maturity and the strikes count optimizer steps, n_epochs x n_minibatches = 8 per
update, 16,000 in a 2,000-update run; rescale_alpha is a per-step tilt and the cap of 64 is reached
after log(64) / log(1.002) = 2,083 steps, about 13 percent of a run, where the per-update version
never reached it (1.002^2000 = 54.6). A minibatch whose loss is not finite is skipped without a step
and without a check (the supervised rule also runs after its spike-skip, but there the states are
finite; after a PPO skip they are not). The rollout no longer keeps its states (the 2.6 GB at
N = 1,000 is not held). Progress.json keeps the four integers per update: rescale_opened,
rescale_graduated and rescale_spent are summed over the update's checks; rescale_growing is the
number growing after the update's last check.

**Equivalence with the option off, pass rule as before (every saved number identical):** main at
9a49c9e against the branch at 58401c2, the same recipe (N = 40, seed 7, 3 updates of 8 episodes x
12 decisions, probe 6, warm-up 2, CPU): Progress.json same keys, largest difference 0; all 15 arrays
of Params_Maze.npz and all 6 of Participation.npz bitwise equal; task and model blocks equal; the
trainer block gains exactly prune_reinit and prune_args. IDENTICAL.

**Checks**, `tests/test_rl_maze.py` 23 of 23. The three-state check now feeds the rule the states
`sequence` recomputes for each fresh rollout (shape (N, T + 1, B) at substeps 1), one call per
check with no optimizer step between; its trajectory is unchanged: A opened at check 2 (patience 2),
graduated at check 5 at a soft-max rate of 0.364 against a target of 0.119; B's factor doubled to 64,
spent at check 8, re-opened at check 10 with a fresh factor and a redrawn column. The blanking check
runs the rule inside the update at patience 10^6 and target 1e-6 x the cap, so the check after the
step dooms no unit and graduates the hand-marked unit without a tilt: the step clock reads 1, the
update's counts are graduated 1, opened 0, growing 0, the incoming gradient of the unit is exactly 0
and its rows do not move while its outgoing column and other rows do. The counts check: four ints in
every record, at least one episode opened, growing never above 6 of 32, the clock at 4 x 4 x 2 = 32
after 4 updates with no skipped minibatch. `tests/test_no_function_imports.py` 202 files ok;
`tests/test_scripts_start.py` 48 scripts ok.

**Smoke run through the script**, `run_experiment.py task=Maze trainer=trainer_ppo model.N=40 seed=3
trainer.prune_reinit=true`, 3 updates of 16 episodes x 30 decisions (8 checks per update): the six
standard files; the saved config records prune_reinit true and the 15 prune_args at the paper's
values; counts per update: opened 8, 0, 0 (the ceiling round(0.2 x 40) = 8, filled at the first
check), graduated 0, 0, 0, spent 0, 0, 0, growing 8 after every update; nothing skipped, no restore.

**Launcher**, unchanged apart from its comment (alpha 1.002 per optimizer step, a check every
optimizer step). Dry run from the worktree with module, conda and srun stubbed for indices 1 and 22
(N = 40, 2 updates), the same pass rule: the two saved configs differ in exactly experiment_tag and
trainer.prune_reinit; provenance line d4be6e6. Not submitted.
