## 2026-11-20 17:33 — Rules: does training move Maze networks away from their initial dynamical regime? (criticality gate on Maze)

Written and committed before any criticality measurement on a Maze network and before any run is
submitted. Branch `wp/criticality-maze`. The question comes from Pavel, via firstmate (20 Nov), after the
criticality-projection gate failed on 3BFF and CDDM (`2026-11-20_1655_criticality-projection-phase1-result.md`):
"Could it be that for maze that would work better? I suspect that for simpler tasks you don't need
criticality." This entry adapts that gate to Maze and reuses its code
(`trainRNNbrain/trainer/criticality.py`) and its thresholds. Rules of the supervised gate:
`2026-11-20_1643_rules-criticality-projection.md`.

**Terms.**
- *Maze*: the PPO (proximal policy optimisation) navigation task, `configs/task/Maze.yaml` with
  `configs/trainer/trainer_ppo.yaml`. The RNN is the whole agent. Each decision holds the observation for
  5 RNN steps (`substeps`), and an episode has at most 500 decisions, 2,500 RNN steps.
- *Probe*: the trainer's noise-free test (`TrainerPPO.probe`): 128 episodes from fixed starts (`probe_seed`
  20261103), the policy mean as the action, run after update 1 and every 10th update. It records probe
  success (the share of the 128 episodes that reach the cheese) and the number of active units.
- *Learned point*: the first probe from which probe success stays at or above 0.9 for five probes in a row
  (50 updates), in a run whose last probe is also at or above 0.9 (`common.maze_learned_index`). It is the
  Maze counterpart of the supervised floor crossing.
- *Live steps* of a probe episode: the RNN steps of the decisions taken before the episode ended, the steps
  the probe's participation is computed over. Episode b has L_b = 5 × (decisions taken) live states.
- *Step Jacobian*, *λ1* (largest finite-time Lyapunov exponent per RNN step) and *active unit*: as in the
  supervised rules. λ1 = 0 is the edge of chaos; with every unit inactive λ1 = log 0.9 = −0.105 per step.
- *G_H*: gradient propagation over H RNN steps. The read-out's sensitivity at an episode's last live state,
  W_out[o] ⊙ 1[h > 0] for the two action outputs o (turn and speed), is pulled back H steps through J^T.
  G_H is log10 of its norm relative to the start; −1 means a tenfold decay.

### Why the open-loop Jacobian is the right one on Maze

On Maze each observation depends on the agent's earlier actions through the maze. PPO does not
differentiate through that loop: its recompute (`TrainerPPO.sequence`) replays the stored observations as
constants, so the gradient of the PPO loss travels only through the RNN state, multiplied at every step by
J_t = (1 − α)I + α W_rec D_t. λ1 and G_H are measured on that map along the probe's trajectory, with the
observations as given, the same quantity as on the supervised tasks.

### The horizon of gradient propagation: H = 500 RNN steps (100 decisions)

The supervised gate used 100 RNN steps. On Maze the gate uses G500, for three reasons fixed by the task and
the trainer, not by any measurement:
1. PPO's recompute backpropagates through the whole episode without truncation, so the PPO backprop
   horizon is the episode itself, up to 2,500 RNN steps. 100 steps would test only the first 20 decisions
   of it.
2. The return credited to each action is discounted by γ = 0.99 per decision (`critic: false`, Monte-Carlo
   returns), so its effective horizon is 1/(1 − γ) = 100 decisions = 500 RNN steps.
3. A learned episode spans about this many steps. The mean shortest route from a free cell to the cheese is
   15.9 cells, which takes at least 80 decisions (400 RNN steps) at full speed, 0.2 cells per decision; the
   longest route is 29 cells, 145 decisions (725 RNN steps).

The bar on G stays a factor of e (|ΔG500| > log10 e = 0.434), as the brief requires. Over 500 steps that bar
is met by a fifth of the per-step change it needs over 100 steps, so condition 2 below is easier to meet
than on the supervised tasks; the looser bar favours the premise. G100 is reported beside the gate with the
same bar, for comparison with the supervised tasks.

**Episodes too short for the horizon.** An episode enters G_H only if it has at least H + 1 live states. If
fewer than 32 of the 128 probe episodes qualify in either weight state of a network (init or learned
point), condition 2 for that network is read at the largest lag in {250, 100} at which at least 32
episodes qualify in both states, and the results entry says so. An episode enters λ1 only if it has at
least 150 live states (the burn-in of 50 plus 100 averaged steps).

### Phase 1. Is the deterioration there on Maze? (gate before any remedy run)

**Networks.** The default Maze networks at N = 1,000 that have both an init rebuild and a kept snapshot near
the learned point: the three G12 controls, `Maze_suite/N=1000_act=relu` seeds 4, 5 and 6 (Spock 7397702;
rules `2026-11-07_0110_round18-training-rules.md`, results `2026-11-07_0707_round18-training-interim-results.md`).
- They have the paper's default Maze configuration: ReLU standard network, `trainer_ppo`, 2,000 updates,
  `freeze_after_end` false as in every run before 8 Nov, the same as the paper's Maze controls (seeds 1-3).
  Each was continued to update 4,000 in `N=1000_act=relu_cont` (fresh Adam state and a repeated warm-up),
  as seeds 1-3 were.
- Kept snapshots every 100 updates, `u00100` to `u02000`, in
  `trained_RNNs_rl_maze_checkpoints/Maze_suite/N=1000_act=relu/s<seed>/`. Read from the copies on the Mac
  (`~/Documents/GitHub/trainRNNbrain/data/trained_RNNs_rl_maze{,_checkpoints}/Maze_suite`).
- No other default Maze network qualifies. Seeds 1-3 were trained before kept snapshots existed (7 Nov) and
  have weights only at updates 2,000 and 4,000. Della holds no Maze snapshot tree (searched today, 20 Nov,
  under `$HOME/trainRNNbrain/data` and `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data`).

**Weight states.**
- Init: the network rebuilt from the run's saved config with the run's seed (`run_rl.py` seeds the RNN with
  `seed`), checked against the first snapshot (update 100): the cosine of W_rec must exceed 20 times the
  cosine for the network rebuilt with seed + 1, as in the supervised rules.
- Every kept snapshot, updates 100 to 2,000; the final weights at update 4,000.
- The learned point is found on the run's probe record joined with its continuation and cut at update
  4,000, as in the claim-gaps Maze read. The gate reads the kept snapshot at or just after the learned
  update. Entry 2026-11-07_0707 gives learned updates 1,530, 1,180 and 1,330, so the expected snapshots are
  1,600, 1,200 and 1,400; if the reader finds other learned updates, the reader's value is used and the
  difference is reported.

**Trajectories and measures.** For each weight state, the trained agent is rebuilt with
`TrainerPPO.load_run` and plays the probe: 128 episodes, noise-free, policy mean as the action. The states
are recomputed from the probe's observations (`TrainerPPO.sequence`). Only live steps enter any measure.
- λ1 per episode: one tangent vector pushed through the transitions between live states, log growth
  averaged from step 50 on; the network's value is the median over episodes with at least 150 live states.
- G_H for H = 10, 50, 100, 250, 500: per (output, episode) pair, median over pairs with at least H + 1 live
  states. Each network's number of qualifying episodes is reported.
- Also on the same pass: the median spectral radius of J_t at 12 operating points, the fraction of units
  above threshold, ρ(W_rec), σ1(W_rec), the mean incoming row norm, the number of active units and probe
  success.
- Code: `criticality.lyapunov_max` and `criticality.backward_gain` gain an optional `lengths` argument (each
  episode's number of live states); without it they compute exactly what they computed before.

**Gate, per network.** The network has *deteriorated* if, from init to the snapshot at or just after its
learned point, both hold:
1. its λ1 moves away from 0 by more than 0.01 per step: |λ1(L)| − |λ1(0)| > 0.01;
2. its gradient propagation over 500 RNN steps changes by more than a factor of e:
   |G500(L) − G500(0)| > 0.434.

**Gate, on Maze.** The deterioration is there if at least 2 of the 3 networks deteriorated.

**Prediction.** If the idea applies to Maze, the gate passes there: training moves the Maze networks' λ1
away from 0 and changes gradient propagation over 100 decisions by more than a factor of e.
- Gate passes: Phase 2 runs (below).
- Gate fails: no projection is trained. The results entry reports the measurements and the pull request is
  opened with them.

**Reported beside the gate, not part of it:** λ1 and G_H at every snapshot and at update 4,000; condition 2
measured as a change in decay only, |G500(L)| − |G500(0)| > 0.434, because the supervised gate found G
changing toward less decay; G100 with the same bar.

### Descriptive cross-task reading (not a gate): do harder tasks sit closer to criticality?

λ1 at the learned point on all three tasks, per network, with its value at init:
- 3BFF and CDDM: the six supervised networks at the snapshot at or just after their floor crossing, from
  `data/criticality/phase1_3bff.json` and `phase1_cddm.json` (already read: 3BFF −0.006 to −0.007, CDDM
  −0.038 to −0.041 per step).
- Maze: the three networks of this entry at the learned point.

Maze sits closer to 0 than a supervised task if every Maze network has a smaller |λ1| than every network of
that task, farther if every Maze network has a larger one, and overlaps otherwise. Pavel's suspicion that
simpler tasks do not need criticality predicts that Maze, the task with the longest episodes, sits closest
to 0. The reading orders the tasks by λ1 only; it does not measure difficulty, and the trajectories differ
between tasks (closed-loop on Maze, with each observation held for 5 steps). All three tasks share
α = 0.1, so λ1 per RNN step is on one scale.

### Checks (thresholds fixed now)

1. **Init rebuild**, as above; a failure stops the read of that network until explained.
2. **The probe is reproduced.** The reader's number of active units and probe success at each kept snapshot
   on a probe update are compared with the training probe's stored values at that update. A difference of
   more than 5 units or more than 2 of 128 episodes is investigated before the verdict (the training probe
   ran on a GPU, the reader runs on the CPU, so exact equality is not required).
3. **Independent check of the gate inputs.** A script written separately from `criticality.py` and
   `RNN_torch`, kept outside the repository in `firstmate/data/criticality-maze/proof/`, numpy in float64.
   - It loads the weights from the parameter files with numpy, takes the probe's observation sequence and
     episode ends from the production rollout (the maze itself is not reimplemented), and runs its own
     forward pass of the update equation over those observations.
   - Its read-out must reproduce the production rollout's actions within 10⁻³ at every live decision, which
     ties the observations to the network that produced them.
   - It finds the learned point from the probe records with its own code (exact equality required), and
     computes λ1 with a different tangent draw and G500 and G100 with its own loop and masking.
   - Pass: |Δλ1| < 10⁻³ and |ΔG| < 0.02 at init and at the learned point in every network, and the same
     gate verdict. These are the supervised check's thresholds.
4. **Measures with lengths, known answers** (`tests/test_criticality.py`): with every length equal to T the
   result is identical to the call without lengths, and an episode cut to length L gives the result of
   running the unmasked function on its first L states.

### Phase 2 (runs only if the Maze gate passes)

**Implementation.** The hook of `Trainer.wrec_projection_` mirrored in `TrainerPPO`, off by default, with the
same four rules from `criticality.py`, applied after the optimiser step every 50 optimiser steps (one
supervised iteration is one optimiser step), η = 1.
- (a) `spectral`: ρ(W_rec) = model.spectral_rad = 1.2.
- (b) `singular` and (c) `rownorm`: the references are the singular values and row norms of W_rec at init,
  rebuilt from the chain's first config and seed. The supervised hook takes them from the start of the
  run, which at a warm start would be the parent's final weights; Maze runs in two stages, so the
  references must come from init.
- (d) `lyapunov`: λ̂1 on a noise-free pass over the live steps of the current minibatch's stored
  observations, burn-in 50, largest change 5% per event.

**Runs.** Arms `control`, `spectral`, `singular`, `rownorm`, `lyapunov`; 3 seeds each; N = 1,000; the default
Maze configuration with `freeze_after_end` true (the default for every new run since 8 Nov).
- 4,000 updates as two 2,000-update stages on Della gpu-short. Stage 2 warm-starts from stage 1 with the
  Adam state restored (`paths.init_from`, `paths.init_adam=true`), chained with `--dependency=afterok`.
- Kept snapshots every 100 updates (800 optimiser steps, a multiple of the period, so every snapshot holds
  projected weights).
- New trees only: `$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_crit` and its `_checkpoints`.
- Cost: a 2,000-update stage at N = 1,000 took about 6.7 GPU-h in the claim-gaps runs (20 GPU-h for three).
  15 runs × 2 stages × 6.7 ≈ 200 GPU-h, plus at most 10% for the projections: about 220 GPU-h, inside the
  250 GPU-h budget. 15 jobs per stage, under the 24-GPU per-user cap of gpu-short.

**Controls.**
- *Matched*: the `control` arm, same launcher and commit; only the projection differs.
- *The paper's Maze controls*: `Maze_suite/N=1000_act=relu` seeds 1-3 joined with `_cont`: 451, 517 and 428
  active units at the learned point, 360, 404 and 317 at update 4,000, probe success 1.000 at update 4,000
  (`2026-11-20_1600_claim-gaps-results.md`). Known differences: they ran with `freeze_after_end` false and a
  fresh Adam state at update 2,000.

**Read-out.** R1, the number of active units at the learned point; R2, at update 4,000; probe success at
update 4,000; stability (the run reaches update 4,000 and fewer than 5% of its optimiser steps are
skipped); λ1 at the learned point.

**Verdict, per arm** (the technique criterion of the supervised rules in its Maze form):
- *Technique criterion*: every arm network has more active units at update 4,000 than every control
  network, and the arm's mean probe success at update 4,000 is at least the controls' mean minus 0.05, with
  no arm network below 0.9 (the claim-gaps Maze tolerance).
- *Count condition at the learned point*: every arm network has a larger R1 than every matched control; a
  network with no learned point fails it.
- *Manipulation check*: every arm network has a smaller |λ1| at its learned point than every matched
  control.
- **Passes**: the technique criterion against both control sets, every arm network stable, and the count
  condition at the learned point. A pass against one control set only is "control-dependent".
- **Falsified as a remedy**: the technique criterion fails against either control set, or an arm network is
  unstable. If the manipulation check holds and the count condition fails, the arm falsifies the mechanism
  for its route; if the manipulation check fails, the arm does not test the mechanism.
- Checks before and after the runs as in the supervised rules: the control command at main and at the
  submission commit trains identically with the option off (N = 50, a few updates, CPU); each projection
  lands on its target, checked by a separate numpy script; the reads are recounted per network by a
  separate script.

No claim of the paper changes with this entry, and the paper repository is not touched.
