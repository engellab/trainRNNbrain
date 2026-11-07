## 2026-11-07 00:47 — Experiments across 3BFF, CDDM and Maze: what is the same, what differs, what is missing

**Why.** Pavel, 6 Nov: "Orchestrate the paper completion (3 tasks, same interventions and experiments on
all of them)". `docs/interventions_matrix.md` inventoried the training settings and techniques. No file
inventoried the experiments and analyses. This entry lists every experiment or analysis the paper reports,
with one column per task: the 3-bit flip-flop (3BFF), context-dependent decision making (CDDM) and the maze
(Maze, trained by PPO, proximal policy optimisation, a reinforcement-learning algorithm). DMTS (delayed
match-to-sample) is a fourth task in Extended Data only and is left out.

**How it was counted.** Text: the paper worktree `wt/r17-lengths-paper` at 0b402c7 (PR #197 merged),
Results, captions, Methods, Supplementary Methods and Notes. Runs: directories listed on 7 Nov on Della
(`/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs`, `.../trained_RNNs_rl_maze`), Spock
(`~/trainRNNbrain/data/trained_RNNs_rl_maze`, `~/trainRNNbrain_logs`) and the Mac (`data/`), with saved
configs read where a folder name does not give the setting. Run times: Della `sacct` and the `done:` lines
of the Spock logs. Code at trainRNNbrain e888d2a. No job was submitted and nothing was trained.

**No outcome of a missing test was computed.** Several missing cells need only seconds of CPU on files
that exist (for example `active_learned` and `active_end` are already in `data/read_maze_arms.json`). I
did not tabulate them, so that each rule can be written in a notebook entry before its number is seen.
The one calculation made here is a refit of the Maze size exponent from the rounded means printed in the
Extended Data Maze caption (finding 3).

### Terms used below

- **Read-out point**: the moment in training at which a network is measured. For 3BFF and CDDM it is
  either the **floor crossing** (the first iteration from which the smoothed noisy loss stays at or below
  1.05 times its fitted asymptote until 1.2 times that iteration) or a fixed **75k** (75,000 iterations).
  For Maze it is the **learned point** (the first probe from which the probe success stays at or above
  0.9 for five probes, in a network whose last probe is also at or above 0.9), **update 2,000**, or the
  **end** of training. One PPO update is 8 optimiser steps (4 epochs of 2 minibatches).
- **Learned**: Maze, last probe at or above 0.9; 3BFF and CDDM, held-out R² at or above 0.8.
- **k added units**: a technique network's number of active units minus the control networks' mean,
  taken as its k least participating active units.
- **GPU-h**: one GPU for one hour. Costs are for 3 networks per cell unless stated.
- **Priority 1**: a main-text sentence states the result for all tasks, or would be read that way, and
  the evidence does not cover all three; or a main-text headline depends on the cell.
  **Priority 2**: the main text names the task it covers, but a reader comparing tasks will ask for the
  others. **Priority 3**: parity only, or a version on that task would not mean anything.

### Run times used for the costs (from the logs)

| Run | Time per run |
|---|---|
| Maze N = 250, 2,000 updates (Spock 7383494_1-3) | 4.0-4.8 h |
| Maze N = 500, 2,000 updates (7383494_4-6) | 5.0-5.5 h |
| Maze N = 1,000, 2,000 updates (7383494_7-9; prune-and-duplicate 7389335_52-54) | 5.5-5.8 h |
| Maze N = 1,000, noisy synapses (7383697_16-18) | 15.3-16.1 h |
| Maze N = 2,000, per 2,000 updates (7383495, 7384301, 7389358) | 6.3-7.1 h |
| Maze N = 4,000, per 2,000 updates (Della 16883344_22, 16907630_21, 16955490_22, 16955491_23) | 14.9-22.0 h |
| Maze removal curves, one network, Mac CPU (`rl_maze_ablate.py`) | 48 min at N = 1,000; about 100 min at N = 2,000 |
| Maze retraining, one arm, 200 updates (Spock 7384815) | 40 min |
| Supervised N = 1,000, 75k (Della ActB0, WdB0, MetB0, NoiseB0) | 4.9-5.8 h |
| 3-bit hyper flip-flop transfer, 200k: N = 1,000 / N = 4,000 (B4xfer, B4x4k) | 13.4-14.7 h / 57.5-58.3 h |
| Supervised removal of units, one network (F1abl) / retraining, one network (F1ret) | 2.5-3.5 min / 22-34 min |

Supervised runs at other sizes use the guide's speeds: 75k takes about 3.6 h at N = 500, 14 h at
N = 2,000 and 49 h at N = 4,000. Maze N = 2,000 and 4,000 need 4,000 updates to learn, so about 13.6 h
and 32 h per run.

### Findings that change what the paper can say

1. **The 3BFF rate penalty at λ_frm = 0.01 was trained and never measured, and the main headline depends
   on it.** `NBitFlipFlop_remedy_grid/EqType=h_N=1000_arm=frm0.01` (Della, 3 networks, standard network,
   200k, snapshots in `c1_snapshots/NBitFlipFlop_remedy_grid`) is "not read" in Supplementary Table 3.
   The verdict table says a technique run at several values meets the criterion on a task if it meets it
   at one of them. The rate penalty meets it on CDDM and Maze, so if λ = 0.01 meets it on 3BFF (more active
   units than every control, R² within 0.01), the sentence "Prune-and-duplicate is the one technique that
   meets the criterion on all three tasks" is false. The 16 Oct pilot read it at 766 and 932 active units
   in two of three networks (entry 2026-10-16_1053); its R² at 75k has not been measured. λ = 0.03 on 3BFF
   is also missing from the verdict table, where a dash is defined as "not trained", although its three
   networks are read (724 active, R² 0.023 below the controls; Supplementary Table 3a).
2. **The one-value rule is applied to the rate penalty only.** Supplementary Table 3a lists other 3BFF
   values at N = 1,000 whose means lie inside the criterion: noisy synapses σ_w = 0.1 (1.31-fold, R² 0.001
   lower) and 0.3 (1.47-fold, 0.004 lower); read-out dropout at rate 0.05 (1.52- to 1.61-fold, 0.006 to
   0.007 lower) and rate 0.1 with β = 1 (1.59-fold, 0.009 lower). By the rule as written, noisy synapses and
   read-out dropout would meet the criterion on 3BFF, so "Read-out dropout ... in CDDM networks only, and
   noisy synapses and rescaling in none" holds only for the values shown. These lower values were never run
   on CDDM or Maze. The per-network test (every network above every control) has to be applied before the
   text changes; the table gives means.
3. **The Maze size exponent is fitted over another range and another training length than 3BFF and
   CDDM.** 3BFF and CDDM are fitted at N of 500 or more; Maze at every size from 250. Refitting the rounded
   means of the Extended Data Maze caption at N of 500 or more gives about 0.35 at the learned point
   (0.47 over every size) and about 0.41 at the end (0.52), from means only, with one learned network at
   N = 4,000 and no bootstrap. Networks of N = 2,000 and 4,000 were trained for 4,000 updates and smaller
   ones for 2,000, so "end of training" means a different training length at different sizes; in 3BFF
   (200k) and CDDM (100k) every size has the same budget. The N = 1,000 and N = 2,000 end means are
   438 and 431.
4. **CDDM networks with prune-and-duplicate and read-out dropout at N = 500, 2,000 and 4,000 exist and
   were never read.** `CDDM_paper_grid/EqType=h_N={500,1000,2000,4000}_arm={control,duplication,mute}`,
   3 networks each, 100k, standard network (bias range [0, 0] and `same_batch: false` in the saved config
   at N = 4,000). The Discussion says "Across sizes, the techniques were tested in 3BFF networks only".
   Reading these networks gives prune-and-duplicate, the technique the Discussion recommends, a size
   series on a second task at no training cost.
5. **The Maze removal of added units covers five of nine rate-penalty networks and no
   prune-and-duplicate network.** `data/rl_maze_ablation_interventions` holds rate 0.03 seeds 1-2, rate 0.3
   seeds 1-2 and rate 0.1 seed 3. Rate 0.03 seed 3 (success 0.97), rate 0.3 seed 3 (1.00) and rate 0.1
   seed 2 (0.97) learned and were not removed; neither were the three prune-and-duplicate networks (0.95 to
   0.99), the one technique that passes on Maze. Results 5 opens with "The output of a network trained with
   a technique depends on the units the technique adds, in 3BFF and Maze networks", which reads as every
   technique on Maze.
6. **Results 3 says the decrease after the read-out was checked "in the networks in which we could check
   it", but Maze networks can be checked.** Every Maze setting and technique network with a learned point
   has both counts in `data/read_maze_arms.json`. The Discussion gives "Maze networks have no loss floor"
   as the reason, while Results 2 uses the learned point for exactly this comparison in the size series.
7. **Only two Maze control networks learned the task.** The direction rule for a setting compares every
   network with every control network. The Maze range is 426-449 from two networks; the third (415) did not
   learn. Weight decay 1e-5 on Maze has 370-417, so one new learned control below 417 would remove that
   direction, and with it the claim that only weight decay changes the number on all three tasks. A third
   learned control is a falsification test of a main-text claim, not bookkeeping.

### The table: every experiment or analysis, by task

Cells: **done** (n networks, N, read-out point, script, entry), **differs** (how), or **missing**, with a
judgement of whether a version on that task means something.

| # | Experiment (where in the paper) | 3BFF | CDDM | Maze |
|---|---|---|---|---|
| 1 | Active units before and after training (Results 1, Fig. 1a,b) | **done**: n = 10, N = 1,000, iteration 0 and the last stored probe (199,900); `active_before_after.py`; entries 2026-11-02_2048, _2122 | **missing**; meaningful; the participation traces hold iteration 0 | **missing** as a number (Extended Data Maze b draws the curves from update 1); meaningful |
| 2 | Active units at N = 1,000 (Fig. 1c) | **done**: n = 4, floor crossing (59k-68k), 240-297; `fig_paper_F1.py`; 2026-10-31_1755 | **done**: n = 3, floor crossing (24k-27k), 420-437 | **differs**: 2 of 3 learned; learned point (updates 760, 1,050): 451, 517; end: 426, 449; 2026-11-04_1957 |
| 3 | Removing units without further training, N = 1,000 (Fig. 1d) | **done**: n = 3, weights within 250 iterations of the floor crossing; `ablate_silent_units.py`; 2026-10-31_1702 | **done**: n = 3, same | **differs**: n = 2 (seed 3 removed but not drawn), at the end (update 2,000), success under the training noise; no weights exist at the learned point (each checkpoint overwrites the last); `rl_maze_ablate.py`; 2026-11-05_0843 |
| 4 | Removing units across sizes (Extended Data Fig. 2; Extended Data Maze d) | **done**: n = 15, N = 500-4,000, 75k; `fig_supp_ablation.py` | **done**: n = 12, N = 500-4,000, 100k | **differs**: N = 250 (n = 1), 500 (n = 1), 1,000 (n = 2); three N = 2,000 files on disk (update 4,000) not drawn; N = 4,000 not removed |
| 5 | Retraining without the inactive units (Fig. 1d, stars) | **done**: n = 3, +3,000 iterations (4-13% of training), three arms; 2026-10-31_1702 | **done**: n = 3, same | **differs**: n = 2 (seed 3 in the notebook only), +200 updates (10%), from update 2,000; `rl_maze_retrain.py`; 2026-11-05_2226; equivalent in meaning |
| 6 | Fraction of selective units, all units against active units (Fig. 1e) | **done**: n = 3, floor-crossing weights; `population_stats.py`; 2026-10-31_2220 | **done**: n = 3, same | **differs**: n = 2, end; `rl_maze_population_stats.py`; 2026-11-05_0845. Results 1's "(... measured at the end of training)" reads as all eight networks; the six supervised ones are at the floor crossing |
| 7 | Dimensionality and k-means classes (Extended Data population b-f) | **done**: n = 3, floor crossing | **done**: n = 3 | **measured, not drawn**: n = 3, end; `data/population_stats_maze.json` |
| 8 | Spread of the recurrent weights, all against active units; weight decay as the cause (Results 1; Extended Data population a) | **done**: n = 7, 40k-200k; `connectivity_confirm.py`; 2026-11-02_2232. Cause: **missing** (a wd 0 arm exists) | **done**: n = 26; cause tested (weights between inactive units keep 0% with weight decay 1e-6, 98-99% without) | **missing**; meaningful; only seed 3 is unused by row 6, which the rule's "networks not used in b-f" requires; cause testable on the `wd0` arm |
| 9 | Size series and exponent (Fig. 2a-c) | **done**: N = 100-4,000, n = 3, 6, 4, 3, 3, floor crossing, b = 0.48 at N of 500 or more; `fig_paper_F2_size.py`; 2026-10-31_1957 | **done**: N = 100-4,000, n = 3 each, b = 0.44 | **differs**: N = 250-4,000, learned n = 3, 3, 2, 3, 1; fit over every size; budgets of 2,000 and 4,000 updates (finding 3); seeds 4 and 5 at N = 4,000 running (Della 17012367_24-25); `fig_ed_maze.py`; 2026-11-06_1723 |
| 10 | Sensitivity of the exponent: active-unit cut-off 0.025-0.10, margins 1.05/1.10/1.20 (Methods; Fig. 2c) | **done**: `threshold_sensitivity.py`; 2026-10-31_2211 | **done** | **missing**; cut-off: meaningful, from stored participation; margin: the counterpart is the success level (0.85, 0.9, 0.95) and the number of probes |
| 11 | Time course at N = 1,000; count at the end against the read-out (Fig. 2e; Results 2) | **done**: n = 4, 200k, 4 of 4 fewer at the end | **done**: n = 3, 200k, 3 of 3 | **done**: n = 3 drawn; 10 of 12 learned size-series networks fewer at the end (end means differ in length, finding 3) |
| 12 | Flat-loss test (Fig. 2f) | **done**: n = 19, N = 500-4,000, 150k to the end, thresholds 3% and 5 units; 13 of 15; `c3_flat_loss_units.py` | **missing**; meaningful; `CDDM_std_g0_drift_bias0` N = 500 and 1,000 (200k) and N = 2,000 (300k), n = 3 each, on Della | **missing**; no loss floor; counterpart below (training return flat) |
| 13 | Fewer active units at the end than at the read-out, every setting (Results 3: 126 of 127) | **done**: n = 36 (input norm, re-initialisation, optimiser); the other sweeps end at 75k; 2026-11-06_2219 | **done**: n = 91, 8 of 10 settings | **missing**; both counts in `read_maze_arms.json` (finding 6) |
| 14 | Ten standard settings at N = 1,000 (Fig. 3; Supplementary Table 2) | **done**: n = 3, 75k; `read_uniform_cells_2.py`; 2026-11-06_1830 | **differs**: input-weight norm at 50k (its training length); tanh at 150k only; noise and sparsity n = 5; spectral radius and recurrent-weight penalty now 3 networks on Della, Extended Data Fig. 4 still says 2 | **differs**: spectral radius 1.5 instead of 2.0 (2.0 is not finite in the first rollout); input-weight cap one value (3BFF four); SGD never learns; update 2,000; `read_maze_arms.py`; 2026-11-06_1723 |
| 15 | Re-initialising inactive units (Extended Data Fig. 4j,k) | **done**: redraw and zero-out, n = 3 each, 75k and floor crossing | **differs**: one method (`CDDM_screen3_c75000` arm `prune`, incoming row redrawn after 5 inactive checks), n = 3 | **done**: redraw and zero-out, n = 3 each, update 2,000 |
| 16 | Silent units (Results 4, Fig. 4a,b) | **done**: n = 3, 0 silent | **done**: n = 3, 253-310 of 1,000 | **missing**; meaningful (net input on the clean probe) |
| 17 | Incoming-weight gradient norms (Fig. 4b; Extended Data Fig. 6) | **done**: n = 3, 12 batches of 1,024 trials; `c4_gradient_mechanism.py`, `c4_reach_batches.py`; 2026-11-01_1021 | **done**: n = 3, 450 conditions, 10 noisy batches | **missing**; TrainerPPO has no fixed-trial loss; counterpart below |
| 18 | Inactive units active 500 and 5,500 iterations later (Fig. 4c) | **done**: n = 3; fewer than a tenth of the low-active units above the gradient threshold | **done**: n = 3; at most 1 silent unit per network | **missing**; per-unit participation is stored every 50 updates (`Participation.npz`, 41 vectors per 2,000 updates); weights only at the end |
| 19 | Transfer to a new task (Fig. 4d,e; Extended Data Table 5) | **done**: N = 1,000 and 4,000, 3 old, 3 young and 3 fresh networks, 200k on the 3-bit hyper flip-flop; `SilentReLU_b4_transfer_della.slurm`, `read_b2_b4.py`; 2026-10-31_1532, _1750 | **missing**; meaningful; old sources exist (200k); needs a new task with CDDM's six inputs | **missing**; meaningful, and it is the setting of the reinforcement-learning studies the Introduction cites; needs `paths.transfer_from` in `run_rl.py` and a second maze |
| 20 | Tanh: saturation and gradients (Results 4; Extended Data Fig. 7; Supplementary Note) | **differs**: saturated-unit count only, n = 3 at 75k (536, 41, 75); no gradient measure | **done**: n = 9 tanh, 9 ReLU, 150k; `gradient_sensitivity.py`, `fig_tanh_saturation.py` | **differs**: n = 3, update 2,000, 999-1,000 pass the bar, none learned (success 0.00-0.38); a gradient analysis of networks that did not learn means nothing |
| 21 | Five activation functions (Fig. 3 column; Extended Data Maze e) | **done**: n = 3, 75k | **done**: three functions at 75k, tanh at 150k | **done**: n = 3, update 2,000; also counted by the modulation rule (s.d. of the rate alone), which no supervised network has |
| 22 | Six techniques at N = 1,000 (Fig. 5a,b; Supplementary Table 2) | **done**: n = 3 (rescaling 4), 75k, rate penalty at 0.1 | **done**: n = 3 or 5, 75k (estimated from 100k weights for prune-and-duplicate, read-out dropout, noisy synapses), rate at 0.1 and 0.2 | **differs**: n = 3, update 2,000, rate at 0.03, 0.1, 0.3, rescaling in three variants |
| 23 | Several values per technique (Supplementary Table 3) | **done**: 58 value sets at N = 1,000-4,000 and 64 screens; λ = 0.01 not read (finding 1) | **differs**: one value per technique, two for the rate penalty | **differs**: one value per technique, three for the rate penalty and for rescaling |
| 24 | Active units over training under a technique (Extended Data techniques a; Extended Data Maze settings e) | **done**: n = 3-4 | **missing**; seconds from the traces | **done**: n = 29 networks |
| 25 | Techniques across sizes (Fig. 5c; Extended Data Fig. 9a-c) | **done**: six techniques, N = 500-4,000, 1-6 per size, floor crossing and 75k; `read_remedies.py`; 2026-10-31_1732 | **trained, not read** for prune-and-duplicate and read-out dropout (finding 4); the other four at N = 1,000 only | **missing**; N = 1,000 only |
| 26 | Rate-penalty size test (Extended Data Fig. 9d-g; Table 5) | **done**: λ = 0.1, N = 500-4,000, n = 3; fails; 2026-10-31_1532 | **missing** on the standard network (`CDDM_std_g0_penalties` at N = 500 and 2,000 has a trainable bias) | **missing** |
| 27 | Combination test (Extended Data techniques c-e; Supplementary Note) | **done**: 8 combinations, n = 3; inconclusive; `read_b3.json` | **missing**; not meaningful as designed: the rate penalty alone has 1,000 of 1,000 active, so no combination can exceed it | **missing**; meaningful for rate penalty with noisy synapses; rescaling fails on Maze |
| 28 | Removing the added units (Fig. 5d; Extended Data techniques b) | **done**: n = 19 at N = 1,000, and N = 4,000; the k added units with the inactive units in place, against k random active units; 75k; `a3_remedy_ablation.py`; 2026-10-31_1732 | **missing** (stated: "We did not make these removals in CDDM networks") | **differs**: n = 5 rate-penalty networks only; inactive units and then the k added units (570 in all) against 570 random units from all 1,000; update 2,000; `fig_paper_F5_tasks.maze_removal`; 2026-11-05_1230, 2026-11-07_0032 |
| 29 | Whole removal curves of technique networks (Extended Data Maze settings h) | **differs**: fixed points only (random 10, 30, 50%; every inactive unit) at 75k | **missing** | **differs**: 5 rate-penalty networks and 1 control, end |
| 30 | Floor crossing against an independent snapshot estimate (Extended Data Fig. 3) | **done**: n = 6 | **done**: n = 14 | not meaningful: the learned point is read directly on the noise-free probe |

### Maze counterparts of the loss-based analyses

**A fixed-trial loss for PPO.** The supervised gradient analyses differentiate the task loss on one fixed
noise-free batch. The PPO loss is defined on a rollout. The counterpart is a frozen rollout: play
256 episodes with a fixed seed, store the observations, the sampled actions, their log-probabilities and
the advantages (Monte-Carlo returns minus their batch mean, as the trainer computes them with
`critic: false`), and differentiate the clipped surrogate once with respect to W_rec and W_inp. At the
first pass the probability ratio is 1, so the clip is inactive and the gradient equals the policy gradient
of that batch. The noise-free probe cannot be used. On the probe each action equals the policy mean, and
the gradient of log π(a | x) with respect to the mean is (a − μ)/σ², which is exactly zero there, so every
policy-gradient term vanishes on the probe for every unit. The counterpart of "noise-free batch" is
therefore recurrent and input noise off, actions sampled with the trainer's fixed action s.d. of 0.5.
A unit whose net input is at or below zero on every step of that rollout has zero gradient on its incoming
and outgoing weights there, as on CDDM, so the silent-unit statement transfers exactly; silence has to be
defined on the same rollout, because sampled actions visit other states than the probe. The gradient
threshold (5% of the 95th percentile over units) and the class split transfer unchanged. The tanh panel's
normalisation by the square root of the loss does not: the advantages are standardised, so the surrogate
loss sits near zero. Use the ratio to the untrained network's 95th percentile instead, as Extended Data
Fig. 7d already does. A second, loss-free counterpart is the output sensitivity of
`gradient_sensitivity.py`: the Jacobian of the turn and speed outputs with respect to each unit's incoming
weights on the clean probe.

**"5,500 iterations later".** 5,500 optimiser steps are about 690 PPO updates (8 steps per update), and
500 steps about 60. With participation stored every 50 updates, the log-only version (which inactive units
are active 700 updates later) runs on every learned Maze network today. The class split into silent and
low-active units and the gradient need weights at the reference point; those exist only at the end of
each run, so the reference point is update 2,000 and the later activity needs a continuation (gap G11).

**Flat loss.** Maze has no loss floor. The quantity PPO maximises is the return of the training rollouts,
logged per update in `Progress.json` (`discounted_return`, `success`). The counterpart is: from the learned
point to the end, the change in the median training return over 100-update blocks, relative to the change
from update 1 to the learned point, with "flat" and "decreasing" thresholds written before computing. At
2,000 updates the N = 1,000 window is only 950 to 1,240 updates long, so the continuation of G11 is
needed for a window comparable with the 3BFF one.

**Transfer.** The counterpart of the hyper flip-flop is a second maze of the same size from
`make_maze.py` (another seed, the cheese elsewhere): the same ten inputs and two outputs, the recurrent and
input weights kept and new read-out weights drawn, compared with fresh networks on the new maze, with the
thresholds of Extended Data Table 5 and the learned point in place of the floor crossing. For CDDM, a task
with the same six inputs and more outputs (for example, report the motion choice and the colour choice on
separate channels whatever the cue, plus the cued choice) is the counterpart of adding outputs to 3BFF.

### Gaps: data, code, cost, priority

"Analysis" means existing networks only. Supervised analyses run on Della through `sbatch` (login nodes
kill heavy work); Maze analyses run on the Mac CPU or a Spock GPU.

| ID | Cell (row) | Data | Script and missing code | Cost | Priority |
|---|---|---|---|---|---|
| G1 | 3BFF rate penalty λ = 0.01 and 0.03 at 75k in the verdict table (22, 23) | analysis: kept snapshots at 75k | add both cells to `read_uniform_cells_2.py` and `fig_paper_F5_tasks.py`; held-out R² with 36 draws | under 0.5 CPU-h | 1 |
| G2 | Per-network criterion for every 3BFF value of Supplementary Table 3a (23) | analysis | `table_rule_settings.py` already holds per-network counts; add the per-network test and the one-value rule | under 1 CPU-h | 1 |
| G3 | Removing added units on CDDM (28) | analysis: `CDDM_std_g0_bias0_c75000` rate 0.2 and rate + sparsity (n = 5, 75k), `CDDM_std_g0_penalties_bias0` (n = 3, 75k), `CDDM_target_ladder` (75k), `CDDM_paper_grid` duplication and mute (100k weights only), `CDDM_synnoise_ext` (Mac, 100k) | extend the cell discovery of `a3_remedy_ablation.py` (3BFF only through `read_remedies.all_cells`) to these CDDM cells with their controls | about 1 GPU-h | 1 |
| G4 | Maze removal of added units for every learned technique network, and the 3BFF form of the removal (28, 29) | analysis: the learned networks not yet removed, prune-and-duplicate seeds 1-3, rate 0.03 s3, 0.1 s2, 0.3 s3, and noisy synapses s1-s3 (one ends at 0.898); the criterion counts every finished network, so say which set is removed | `rl_maze_ablate.py`: two new orders, "active units from the least participating, inactive units left in place" and "random among the active units"; score noisy-synapse networks with σ_w = 0, as 3BFF does | 9 networks x 0.8 CPU-h = 7 CPU-h, plus minutes for the k-only points on the 5 existing networks | 1 |
| G5 | 3BFF form of the Maze removal: inactive plus k added against as many random units from all N (28) | analysis: `r2_keep_ctrl` exists; the random counterpart does not | `a3_remedy_ablation.ablate_net`: one random order at n_inactive + k | under 0.5 GPU-h | 1 |
| G6 | CDDM prune-and-duplicate and read-out dropout at N = 500, 2,000, 4,000 (25) | analysis: `CDDM_paper_grid`, 27 networks | `read_uniform_cells.py` with N as a parameter; the floor crossing for the 3BFF-style panel; `fig_paper_F5_tasks.py` panel c gets a CDDM line | about 1 GPU-h | 1 |
| G7 | Flat-loss test on CDDM (12) | analysis: 9 networks, 200k-300k | `c3_flat_loss_units.py`: add the three `CDDM_std_g0_drift_bias0` cells to `CELLS`; thresholds unchanged | seconds | 1 |
| G8 | Flat-return test on Maze (12) | analysis on 12 learned networks now; longer window after G11 | new `rl_maze_flat.py` on `Progress.json` and `Participation.npz`; rule entry first | seconds | 1 |
| G9 | Learned point against the end for every Maze arm (13) | analysis | `read_maze_arms.py` prints the count; rule entry first | seconds | 1 |
| G10 | Maze exponent at N of 500 or more, with bootstrap; budgets stated (9) | analysis | `fig_ed_maze.py`, `fig_paper_F2_size.py` | seconds | 1 |
| G11 | Maze size series to one training length: N = 250, 500 and 1,000 continued by 2,000 updates (9, 11, 12, 18) | training | `rl_maze_suite_spock.slurm` with `PARENT_ROOT` and `EXTRA_UPDATES=2000` (exists); no code. The saved runs have no Adam state, so the continuation restarts Adam with the warm-up, as the N = 2,000 chains did | 3 x 4.4 + 3 x 5.3 + 3 x 5.7 = 46 GPU-h | 1 |
| G12 | Third learned Maze control at N = 1,000, with weights kept at every checkpoint (2, 3, 14, 22) | training: seeds 4-6, 2,000 updates | `run_rl.py`: keep a copy of the parameters at each checkpoint (about 15 lines; today each checkpoint overwrites the last); `TrainerPPO.load_run` reads a snapshot; then re-run `read_maze_arms.py` and `fig_paper_F3_tasks.py`, `fig_paper_F5_tasks.py` against the new control range (finding 7) | 17 GPU-h | 1 |
| G13 | Spread of the recurrent weights on Maze (8) | analysis | `connectivity_confirm.py`: a Maze loader (active set from the clean probe); state that two of the three networks are those of row 6 | seconds | 1 |
| G14 | Silent and low-active units on Maze (16) | analysis | new `rl_maze_gradients.py`, step 1: net input on the clean probe and on the frozen rollout | minutes, CPU | 1 |
| G15 | Inactive units active later on Maze, log only (18) | analysis: every learned network | new `rl_maze_noreturn.py` on the stored participation vectors, 50 and 700 updates later | seconds | 1 |
| G16 | Class split and gradient threshold for row 18 on Maze | analysis after G11 (update-2,000 weights, later participation from the continuation) | `rl_maze_gradients.py` step 2 | under 0.5 GPU-h | 1 |
| G17 | Transfer test on Maze, N = 1,000 (19) | training: 3 old sources (from G11/G12) and 3 fresh | `run_rl.py`: `paths.transfer_from` (port `load_warm_start(..., keep_fresh=("W_out",))` from `run_experiment.py`); `configs/task/Maze_transfer.yaml` with a second maze; read script after `read_b2_b4.py`; thresholds of Table 5 written first | 6 x 5.7 = 34 GPU-h | 1 |
| G18 | Transfer test on CDDM, N = 1,000 (19) | training: sources `CDDM_std_g0_drift_bias0` N = 1,000 at 200k, 3 fresh | a new task class with CDDM's six inputs and more outputs; `SilentReLU_b4_transfer_della.slurm` with a `TASK` switch; 75k (crossings at 24k-35k) | 6 x 5.3 = 31 GPU-h | 1 |
| G19 | Maze N = 4,000 seeds 4 and 5 to 4,000 updates (9) | training: first 2,000 updates running | continue with `EXTRA_UPDATES=2000` | 2 x 16 = 32 GPU-h | 1 |
| G20 | Maze incoming-weight gradients at the end, Extended Data Fig. 6a,b,d counterparts (17) | analysis | `rl_maze_gradients.py` step 2 (frozen rollout, surrogate gradient); check it against finite differences on 20 weights, written independently of `TrainerPPO.sequence` | under 0.5 GPU-h | 2 |
| G21 | Maze output sensitivity (17, 20) | analysis | port `gradient_sensitivity.py` to the clean probe | under 0.5 GPU-h | 2 |
| G22 | 3BFF tanh gradient measures (20) | analysis: 3 tanh and 3 ReLU networks at 75k | `gradient_sensitivity.py` cell list | under 0.5 GPU-h | 2 |
| G23 | Before and after training on CDDM and Maze (1) | analysis | `active_before_after.py`; Maze from `Participation.npz` (update 1, with the learning rate near zero in the warm-up) | seconds | 2 |
| G24 | Maze cut-off factor and success-level sensitivity (10) | analysis | `threshold_sensitivity.py` counterpart | seconds | 2 |
| G25 | Maze removal at N = 2,000 (drawn) and N = 4,000 seed 1 (4) | analysis: three N = 2,000 files exist | `fig_ed_maze.py` panel d; `rl_maze_ablate.py` on N = 4,000 | about 3.5 CPU-h | 2 |
| G26 | CDDM tanh at 75k (14) | analysis: the trace gives the count; R² from the loss ratio | `read_uniform_cells_2.py` | minutes | 2 |
| G27 | Maze prune-and-duplicate and rate penalty (0.03) at N = 500 and 2,000 (25, 26) | training | `rl_maze_interventions_spock.slurm` with an N override per arm; no code | 2 x (3 x 5.3 + 3 x 13.6) = 114 GPU-h | 2 |
| G28 | CDDM rate-penalty size test, λ = 0.1 at N = 500 and 2,000 (26) | training, 75k | `SilentReLU_std_penalties_spock.slurm` arm `frm` with N; controls in `CDDM_paper_grid` | 3 x 3.6 + 3 x 14 = 54 GPU-h | 2 |
| G29 | CDDM rate-penalty networks to 200k (Discussion: "whether every CDDM unit stays active in longer training is untested") | training: 3 networks, 75k to 200k | `SilentReLU_ff_topup200k_della.slurm` | 3 x 9 = 28 GPU-h | 2 |
| G30 | Lower values on CDDM and Maze for the two techniques of finding 2: σ_w = 0.3; read-out dropout rate 0.05, β = 1 (23) | training | CDDM: `SilentReLU_synnoise_ext_della.slurm` and `SilentReLU_paper_grid_della.slurm` with `TASK=cddm`; Maze: two arms in the interventions launcher | CDDM 39 + 16, Maze 47 + 17 = 119 GPU-h | 2 |
| G31 | Maze column in Extended Data population b-f (7) | analysis, numbers exist | `fig_ed_population.py` | none | 3 |
| G32 | Positive rates of inactive units on Maze (Supplementary Note) | analysis | minutes | none | 3 |
| G33 | Modulation rule on 3BFF and CDDM (21) | analysis | minutes | none | 3 |
| G34 | Weight decay as the cause of the weight spread on 3BFF and Maze (8) | analysis: wd 0 arms exist | `connectivity_confirm.py` slope test | none | 3 |
| G35 | CDDM techniques over training (24) | analysis | figure only | none | 3 |
| G36 | Maze tanh: saturation count (analysis), and 2,000 more updates in case it learns (20) | training | continuation of the three tanh runs | 17 GPU-h | 3 |
| G37 | N = 4,000: Maze prune-and-duplicate and rate penalty; CDDM rate penalty (25, 26) | training | as G27, G28, on gpu80 | 192 + 147 = 339 GPU-h | 3 |
| G38 | The other techniques across sizes: CDDM rate + sparsity, rescaling, noisy synapses at N = 500-4,000; Maze rate + sparsity, read-out dropout, noisy synapses (25) | training | launchers with N; noisy synapses at 2.5-2.8 times the time | CDDM 905, Maze 735 = 1,640 GPU-h | 3 |
| G39 | Combination test on Maze (27) | training: 4 combinations | interventions launcher; CDDM not meaningful (row 27) | 155 GPU-h | 3 |
| G40 | Value parity: rate-penalty values on every task, input-weight cap values, CDDM input norm to 75k (14, 23) | training | launchers | 77 + 95 + 28 = 200 GPU-h | 3 |
| G41 | Post-read-out decrease for the 3BFF sweeps that end at 75k (54 networks to 200k) and CDDM's optimiser comparison (15 networks) (13) | training | `SilentReLU_ff_topup200k_della.slurm` | 506 + 135 = 641 GPU-h | 3 |
| G42 | Maze N = 100 and the Maze rate penalty to 4,000 updates (9) | training | launchers | 12 + 17 = 29 GPU-h | 3 |

Not meaningful, and to be stated so where a reader would look: the floor-crossing snapshot check on Maze
(row 30); the combination test on CDDM as designed (row 27); spectral radius 2.0 on Maze (row 14);
gradient analysis of Maze tanh networks while none learns (row 20); rescaling across sizes on Maze, where
every variant lowers the number of active units at N = 1,000.

### Totals per priority (3 networks per cell)

| Priority | Training | Analysis |
|---|---|---|
| 1 | 160 GPU-h (G11 46, G12 17, G17 34, G18 31, G19 32) | about 3 GPU-h and 8 CPU-h (G1-G10, G13-G16) |
| 2 | 315 GPU-h (G27 114, G28 54, G29 28, G30 119) | under 2 GPU-h and about 4 CPU-h (G20-G26) |
| 3 | about 3,020 GPU-h (G36-G42) | minutes (G31-G35) |

Priority 1 fits one package under the 31 Oct cap of about 200 GPU-h. Priority 2 is over the cap and needs
Pavel's go-ahead or a split. Wall clock for priority 1: the Maze runs are 4-6 h each and parallel on Spock;
G17 waits for its sources from G11 or G12, so about 12 h end to end with free cards.

### Recommendation

Close before the next committee round:

1. **G1 and G2 first, before any other text change.** They can overturn the headline of the abstract,
   Results 5 and the Discussion. If λ = 0.01, σ_w = 0.1 or 0.3, or read-out dropout at rate 0.05 meets
   the criterion per network on 3BFF, either the claim becomes "at the values shown" in the abstract too,
   or the lower values are run on CDDM and Maze (G30).
2. **All priority-1 analyses** (G3-G10, G13-G16): no training, about a day of agent work, each with its
   rule written in a notebook entry before the number is computed.
3. **G12 (third Maze control)**, because the main settings claim rests on a two-network control range,
   **G11** (one Maze training length; it also gives the continuation that G8 and G16 need) and **G19**
   (already running).
4. **G17 and G18 (transfer on Maze and CDDM)**, 65 GPU-h. If they cannot finish before the round, scope
   Figure 4's title, Results 4 and the Discussion's "after training on a new task" to 3BFF instead.

State as limitations: techniques across sizes beyond prune-and-duplicate and read-out dropout on CDDM
(G27, G28, G37, G38); the combination test on 3BFF only (G39), with the reason it cannot run on CDDM;
value parity (G40); the post-read-out decrease for the 3BFF sweeps that end at 75k (G41); tanh on Maze
(G36). Priority 2 analyses (G20-G26) are cheap and can follow the priority-1 package.

**Status board.** This entry changes no claim, so `docs/paper_audit.md` is not
edited here. Findings 1, 2 and 7 can change claims; the agent that runs G1, G2 and G12 updates the board.

Issue: https://github.com/ptolmachev/dead-relu-paper/issues/201 (round-17).
