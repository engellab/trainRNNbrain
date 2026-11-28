## 2026-11-25 12:45 — Rules: does training with the entropy set point and prune-and-duplicate together give more active units, and a smaller late fall, than either alone?

Written and committed before any run of this test, the pilot included. Branch `wp/entropy-prune-combo`. Pavel's ask, via
firstmate (25 Nov): combine the entropy term with prune-and-duplicate and see whether the two stack, in particular
whether together they stall the late fall in the number of active units further than either does alone.

**Terms.**
- *Entropy set point* (E): the loss term λ_ent × mean_i ((H_i − 0.75)/0.75)², λ_ent = 0.1; H_i is the binary entropy, in
  bits, of Φ(mean/s.d.) of unit i's pre-activation. Entries `2026-11-21_0755`, `2026-11-22_1808`, `2026-11-24_1848`.
- *Prune-and-duplicate* (P): M09's technique (`dead-relu-paper/methods/M09_remedies.tex`): a unit inactive at three
  consecutive checks is overwritten with a noisy copy of an active unit; a check every 100 iterations (Maze: optimiser
  steps), at most 25 of 1,000 units replaced per check, 1,000 iterations (steps) of protection. Values unchanged.
- *Cells*: D (neither technique), E, P, EP (both). N = 1,000, three networks per cell and task.
- *3BFF*: the 3-bit flip-flop task. *CDDM*: the context-dependent decision-making task. *Update*: one PPO (proximal
  policy optimisation) update, 8 optimiser steps.
- *Active unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass, active if p ≥ 0.05 × the 95th percentile of p.
- *Performance*: held-out R² (`heldout_r2`, 36 noise draws, CPU) of the weights at the read point on 3BFF and CDDM; the
  success rate of the noise-free probe on Maze. *Tolerance*: 0.01 in R², 0.05 in success rate (M09).

### (a) Tasks, lengths, read points

| Task | Training | Read point | End | Late window |
|---|---|---|---|---|
| 3BFF, CDDM | 75,000 iterations, then L − 75,000 more; L = 200,000 unless the budget rule shortens it | 75,000 (stored vector at 75,000) | the last stored vector, L − 100 | 75,000 to L |
| Maze | 2,000 updates, then 2,000 more | update 2,000 | update 4,000 | 2,000 to 4,000 |

Every network of every cell trains in these two stages. Stage 2 restores the weights and Adam's moments. It does not
restore the prune-and-duplicate state (strikes, protection, running utility), the spike guard's window or the random
generator, in any cell. In P and EP networks this puts one interruption at the start of the late window: no unit is
replaced in the first 300 iterations (steps) of stage 2. On 3BFF and CDDM, stage 2 has the network's seed + 1, so the
flip-flop trials of stage 1 are not replayed; on Maze the launcher keeps the seed, as in every two-stage Maze run.

### Like-for-like cells, and what is reused

All four cells run the launchers `slurm/entropy_prune_combo_della.slurm` (new; 3BFF and CDDM) and
`slurm/rl_maze_interventions_della.slurm` (Maze; one arm added, `ent0.1_duplicate`, indices 139-141), on Della A100 cards
(`nomig`), from one clean clone `~/trainRNNbrain_epc` pinned to the commit named in the submission entry. The same three
seeds are used in every cell of a task, so a network has the same initial weights as its counterparts.

Reused for stage 1, because their training code is the same: `git diff` between their commit and this one is empty over
`trainRNNbrain/{trainer,rnns,tasks,training}` and `configs` (0ca7929, the 3BFF commit, differs in `TrainerPPO.py` and
`trainer_ppo.yaml` only, which 3BFF does not use):
- 3BFF E: `NBitFlipFlop_entropy_setpoint/..._arm=ent0.1`, seeds 4200852863, 1256299004, 4168822465 (0ca7929);
- CDDM E: `CDDM_entropy_setpoint/..._arm=ent0.1`, seeds 3116616303, 1299181146, 2184072010 (91e759a);
- Maze E: `trained_RNNs_rl_maze_entsp_N1000/Maze_interventions/ent0.1`, seeds 1-3 (e555fae);
- Maze D: `trained_RNNs_rl_maze_mctrl_N1000/Maze_interventions/control`, seeds 1-3, trained now by task
  maze-matched-controls (Della job 18374320, e555fae).

Everything else is trained: D, P and EP on 3BFF and CDDM; P and EP on Maze. The default and prune-and-duplicate networks
behind M09's numbers are not reused (older code, unseeded flip-flop trials or another trainer commit), and stage 2 is
trained for all cells.

**Check 1, configs (threshold: no other difference).** Each network's saved config must differ from the same seed's E
config of its task and stage only in `lambda_ent`, `prune_reinit`, the swept `prune_args` (`reinit_mode`, `copy_noise`,
`copy_norm_preserve`, `max_replace_frac`, `maturity`, `utility`; on Maze also `check_every`, `patience`), the tag, the
paths and, against a reused network, the Hydra keys naming them. Done before this entry: the 3BFF and CDDM
launcher, with the cluster commands stubbed, N = 50, 30 iterations on the CPU, cells D, P and EP, then stage 2 for 20
more; and the composed Maze configs of P and EP. Each differs from the saved E config only in those keys, the smoke
values, and keys Hydra resolves at run time; stage 2 restored the weights, the Adam moments and the clock and joined the
trace. Repeated on the saved configs at the read. A reused network that fails is retrained.

**What is already known.** E at the read point: 3BFF 384, 380 and 454 active units at iteration 74,900; CDDM 540, 566
and 539; Maze 634, 689 and 661. M09's P against M09's controls: 3.1, 2.1 and 1.8 times the control mean, about 810, 705
and 760 units, from other code. No network of this test's D, P or EP cells has been read, and no E network has been
trained past 75,000 iterations (N = 1,000, update 2,000).

### (b) Stacking on the number of active units

At the read point and at the end, per task: **every EP network has more active units than every E network and every P
network.** Both points must hold. (If the nine networks of E, P and EP were exchangeable, the three EP counts would be
the three largest in 1 of 84 orderings. The seeds are shared between cells, so this figure is a scale, not a test.)

### (c) Stalling the late fall

A(t) is the median number of active units over the stored vectors of the last 2,000 iterations up to t (Maze: the probes
of the last 100 updates up to t). F = A(start) − A(end) over the late window, in units, and f = F / A(start). The stored
vectors on 3BFF and CDDM are 100 iterations apart; Maze probes are 10 updates apart.

The decay is **stalled further** by both techniques on a task when every EP network has a smaller F and a smaller f than
every E network and every P network, and the mean F of EP is at least 20 units (2% of N) below the mean F of E and
below the mean F of P. Outcomes, in this order:
1. *stalled further*: as above.
2. *falls faster*: every EP network has a larger F and a larger f than every E network, or than every P network, by at
   least 20 units in the mean.
3. *a single technique already stalls the fall*: the smaller of the mean F of E and of P is under 20 units. Then there is
   no room to stall further, and the entry says so.
4. *no difference shown*: everything else.

### (d) Performance, and the best single technique

Best single technique of a task: of E and P, those whose mean performance at the read point is within the tolerance of
D's mean (and, on Maze, with no network below 0.9); of these, the one with the larger mean number of active units at the
read point. If neither qualifies, the one with the larger mean count, and the entry says so. The choice is made once, at
the read point, and kept for the end.

EP meets the performance condition at a point when its mean performance there is within the tolerance of D's mean and
within the tolerance of the best single technique's mean, and, on Maze, no EP network ends below 0.9. D networks that do
not learn the Maze count as they are, as in M09. M09's own rule for EP (every EP network above every D network, within
tolerance) is printed beside it.

**The techniques stack on a task** when (b) holds and (d) holds at both points. **They stall the late fall together
further than either alone** when outcome 1 of (c) holds and (d) holds at the end. These are two separate verdicts, given
per task, then as a number of tasks out of three. Neither is claimed in general unless it holds on all three.

### (e) What shows that they do not stack

- *No stacking*: (b) fails at either point. Reported with which single technique EP fails to exceed in every network.
- *Interference*: at either point every EP network has fewer active units than every network of the best single
  technique; or EP fails (d) where the best single technique is within tolerance of D; or an EP network does not finish
  while every E and P network does.
- *Ceiling*: if a network of E or P has 950 or more active units at a point, the comparison of counts there has fewer
  than 50 units of room. If (b) then fails, the verdict at that point is "limited by the ceiling of N units", not "no
  stacking". In every case the entry gives the share of the best single technique's inactive units that EP adds,
  (mean EP − mean best)/(1,000 − mean best). A late fall under 20 units with A(end) ≥ 950 is reported as at the ceiling.

### What the count cannot show: replacements, deaths, dimensionality

P keeps the count up by replacing inactive units, so these are recorded for every network and reported with every count:
- **Replacements per check**, from `reinit_events` (3BFF, CDDM, a cumulative count at every probe, reset to 0 at stage 2)
  and `copy_replaced` (Maze, per update), over stage 1 and over the late window.
- **Unit death rate.** A death at stored vector k: active at k − 1, inactive at k, k + 1 and k + 2. The rate is deaths per
  100 active units per 1,000 iterations, over the late window (Maze: per 100 updates, over stage 2, where every probe's
  vector is stored: `STORE_EVERY=1`, an instrument setting). A unit that is replaced has been inactive at three checks,
  about 300 iterations (37.5 updates), so its death is counted; flicker of one or two vectors is not.
- **Dimensionality**: the participation ratio (Σλ)²/Σλ² of the covariance of the active units' noise-free rates over all
  steps and trials of the held-out batch (Maze: every RNN step before the goal in the clean probe, replayed on a GPU,
  entry of PR #10), at the read point and at the end. Also the fraction of active units whose noise-free rate has a
  correlation above 0.95 with another active unit's (near-copies).

Wording fixed now, at the end point:
- "fewer units die with both techniques" only if every EP network's late-window death rate is below every E network's and
  every P network's;
- if (b) holds but not that, "the larger count is held by replacement: the units die at [rate] against [rate]";
- "the added units add dimensions" only if every EP network's participation ratio is above every E and P network's;
  otherwise "the added units add no measurable dimensions".

### Runs that do not finish, and stops

Every network that reached the read point (end) is counted there. A network stopped by the rollback guard, or with
non-finite weights, did not finish and enters no mean; its cell is then reported on the finished networks, and the
techniques cannot stack with an unfinished EP network. Skipped updates (the log's `gradient spikes` line; Maze: skipped
steps and restored rollouts) are reported per network. A job that Slurm ends with TIMEOUT or NODE_FAIL is continued from
its last `resume/` state or checkpoint to the same point (the rule of `2026-11-21_1629`). An ssh failure is not a job
failure. Any other failure needs a new entry before a rerun.

### Budget: 400 GPU-hours, the pilot included, and the rule for L

Known costs per network: 3BFF E 5.2 to 5.8 h and CDDM E 3.2 to 3.3 h for 75,000 iterations; Maze E 6.8 to 13.9 h for 2,000
updates (the slowest on a 40 GB PCIe card). At those rates and L = 200,000: 3BFF and CDDM about 250 h, Maze about 145 h,
395 in all, with no reserve. So L is fixed by this rule, before stage 2 of 3BFF and CDDM is submitted and before any count
of this test is read:
- U is the GPU-hours used (pilot, stage 1) plus the committed remainder of Maze;
- the projection is the sum over the 24 stage-2 networks of (L − 75,000) × the mean measured seconds per iteration of the
  network's own cell and task (stage 1 logs), plus 0.2 h each;
- L is the largest of 200,000, 175,000, 150,000 and 125,000 with U + 1.1 × projection ≤ 400, the same L on both tasks.

If 125,000 does not fit, or if elapsed plus projected time passes 400 at any later point, I stop and ask firstmate. No
cell is dropped. Time limits are twice the usual time: 14 h (3BFF) and 9 h (CDDM) in stage 1; up to 23 h (3BFF) and 14 h
(CDDM) in stage 2; 16 h on Maze.

**Pilot** (enters no result): 3BFF and CDDM, cells D, P and EP, one network each, 2,000 iterations; Maze, `duplicate` and
`ent0.1_duplicate`, seed 1, 40 updates. Recorded: seconds per iteration (update), peak memory, skipped updates,
replacements. If an EP pilot runs out of memory, ends with a non-finite loss or skips more than 5% of its updates, I
stop and ask firstmate.

### Order, reader and checks

Order: pilot; stage 1 of 3BFF and CDDM; stage 1 of P and EP on Maze; stage 2. The reader
(`entropy_prune_combo_read.py`) is committed before any count is read, and applies these rules in code.
2. **Gate on the reader**: on the reused networks it must reproduce E's recorded counts (3BFF 384, 380, 454 at 74,900;
   CDDM 540, 566, 539 at 74,900; Maze 634, 689, 661 at update 2,000) and, with the same held-out read, E's recorded R²
   (3BFF 0.9503, 0.9515, 0.9516; CDDM 0.8340, 0.8268, 0.8344) to four decimals.
3. **Independent recount**, outside the repository, written from this text with numpy and without `common.py` or the
   reader: active counts at both points, A(t), F, f, replacements, deaths and the verdicts of (b), (c) and (e). Pass:
   exact equality for every network and verdict.
4. **The reader can say no**: run on synthetic trees in which EP is a copy of P, and in which EP is the lowest cell, it
   must print no stacking and interference.

**Expectation, so that the read can surprise.** EP above P and E on 3BFF and CDDM at both points and on Maze; P's counts
roughly flat late and E's falling, so EP's fall close to P's and outcome 3 or 4 of (c) more likely than 1. Replacements
per check in EP below P's, because the entropy term's gradient reaches inactive units. This expectation is a guess, and
the rules do not use it.

### Out of scope

Edits to dead-relu-paper; changes to either technique or its values; other sizes (task entropy-sizes); combinations of
three. The paper's earlier combination test (`SN_combination`) is unaffected: its criteria did not name the best single
technique, which (d) does here.
