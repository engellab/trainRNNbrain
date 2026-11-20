## 2026-11-19 17:49 — Claim gaps: which task-scoped claims can be tested on all three tasks by Saturday 21 Nov 08:00 ET

**Why.** The paper revision (dead-relu-paper PR #243, merged) states every headline claim so that it holds on
3BFF, CDDM and Maze. Its claim ledger left four claims supported on only some tasks; they sit in Results as
task-scoped evidence. Pavel asked for experiments that could make those claims general, using Della and Spock
while he is away (review at the weekend). This entry ranks the four gaps and records which ones get runs.
Written before any run of this package; branch `wp/claim-gaps`.

**Terms.**
- *3BFF*: the 3-bit flip-flop task. *CDDM*: context-dependent decision making. *Maze*: the egocentric maze
  task trained by PPO (proximal policy optimisation, a reinforcement-learning algorithm); one PPO update is one
  batch of 256 episodes followed by 8 optimiser steps. *DMTS*: delayed match-to-sample, the fourth task, in
  Extended Data only.
- *Active unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass; active if p ≥ 0.05·q95(p).
- *Floor crossing*: the first iteration from which the smoothed noisy training loss stays at or below 1.05
  times its fitted floor until 1.2 times that iteration (`common.noisy_floor`). *Learned point* (Maze): the
  first probe from which probe success stays at or above 0.9 for five probes, in a run whose last probe is
  also at or above 0.9.
- *B2*: the rate-penalty size test on 3BFF (entry 2026-10-31_1532). *B4*: the transfer test on 3BFF.
  *G6*: the CDDM prune-and-duplicate size read (rules 2026-11-07_0100, result 2026-11-07_0235).
  *G17*, *G18*: the Maze and CDDM transfer tests (rules 2026-11-07_0110).
- *GPU-h*: GPU-hours.

### Where the runs can go

Measured at 20:45 ET on 19 Nov, before choosing:
- **Spock.** Every one of the 52 GPUs of partition `all` is allocated; 116 jobs from other users wait on
  priority and about 330 more on dependencies; this account's fair-share factor is 0.038. `sbatch --test-only`
  for one GPU and 30 h gave an estimated start of **27 Nov 02:02**.
- **Della.** `sbatch --test-only` for one A100 (no MIG slice): a job of 23 h or less (the gpu-short tier)
  starts at once; a 30 h job (gpu-medium) is estimated to start **21 Nov 18:00**, after the deadline. The
  Maze flat-return jobs (`MazeWarmLong`, 9 GPUs) are running; gpu-short allows 44 GPUs per user.

So every run of this package goes to **Della as jobs of 23 h or less**, in new output trees, as separate jobs
that leave the `MazeWarm*` jobs, their launcher and their trees alone. The brief asked for Spock first; with a
start estimated a week out, nothing submitted there could be read by Saturday. Every cell that needs a single
job longer than 23 h is out of reach for this deadline. That rules out every N = 4,000 cell:
- Maze at N = 4,000 runs about 32 h per 2,000 updates (audit entry 2026-11-07_0047, row G37).
- CDDM at N = 4,000 runs 2.1-2.6 s per iteration on an 80 GB A100, so 75,000 iterations take 44-54 h.

Speeds used below, measured on Della: Maze N = 2,000, 2,000 updates in 6.9-7.1 h (16866951_10-12, default
network); Maze N = 1,000 prune-and-duplicate, 2,000 updates in 6.3-8.5 h (17108178_52-54); CDDM 0.16-0.20 s per
iteration at N = 500 and 0.62-0.75 at N = 2,000 (`docs/AGENT_GUIDE.md`).

### The four gaps, ranked

| Gap | (a) Can it finish and be read by Sat 08:00? | (b) Would its outcome change a claim? | Decision |
|---|---|---|---|
| Prune-and-duplicate raises the active count across sizes (3BFF and CDDM; Maze not tested) | Yes at N = 500, 1,000, 2,000 (Maze runs of 4,000 updates as two 2,000-update jobs, 7-12 h each); no at N = 4,000 | Yes. Prune-and-duplicate is the one technique that meets the criterion on all three tasks at N = 1,000; a Maze size series either extends its size claim to all three tasks (over N = 500-2,000) or bounds it | **Run (gap A)** |
| The rate penalty fails the stable-training test across sizes (3BFF only) | Yes at N = 500, 1,000, 2,000 on CDDM (75,000 iterations, 4-18 h) and on Maze (as gap A); no at N = 4,000 | Yes, either way. If B2 fails on CDDM or Maze, the failure holds on more than one task. If it passes on both, the paper has to say that the failure is specific to 3BFF, where it now says only that 3BFF is the task tested | **Run on CDDM (gap B) and on Maze (gap C)** |
| Few inactive units become active again after training on a new task (3BFF only; CDDM and Maze gave no verdict) | Compute fits (G17 and G18 took 3-6 h per run) | No. G18's binding read at the floor crossing is "between the bars": the old CDDM networks revive 12%, 12% and 25% of their inactive units, and "do not come back" needs fewer than 10% in every old network, so more seeds cannot reach it. G17's gate fails by construction (fresh networks 437 ± 47 active at their learned point against 420 for old networks at transfer; the bar is 3 SD = 140), and 15 candidate second mazes failed the difficulty check (entry 2026-11-07_0421). A verdict would need a new test designed after these results were seen | **Skip** |
| DMTS: the pre-set test of continued decrease after the task is learned fails (1 of 9 setting networks ends with more active units) | Compute would fit | No, within this brief. DMTS is not one of the three main tasks the claims are stated on. Its pre-set test has been run and failed; new DMTS networks would be a second attempt at a decided test. The Maze form of "continued decrease" is the Maze flat-return experiment, which another task runs | **Skip** |

### What runs (rules in separate entries, each written before its jobs are submitted)

| Gap | Rules entry | New runs | Existing runs continued | GPU-h (estimate) |
|---|---|---|---|---|
| A: Maze prune-and-duplicate, N = 500, 1,000, 2,000 | `2026-11-19_1749_rules-maze-duplicate-sizes.md` | 3 at N = 500, 3 at N = 2,000, 4,000 updates each | the 3 N = 1,000 runs of `trained_RNNs_rl_maze_fz` (Della 17108178), 2,000 → 4,000 | 27 + 48 + 20 = 95 |
| B: CDDM rate penalty, N = 500, 1,000, 2,000 | `2026-11-19_1749_rules-cddm-rate-penalty-sizes.md` | 3 at N = 500, 3 at N = 2,000, 75,000 iterations | none; N = 1,000 is the existing cell (Della 16957419_4-6) | 15 + 54 = 69 |
| C: Maze rate penalty, N = 500, 1,000, 2,000 | `2026-11-19_1749_rules-maze-rate-penalty-sizes.md` | as gap A | the 3 N = 1,000 `frm0.1` runs of `trained_RNNs_rl_maze_fz` (Della 17107385), 2,000 → 4,000 | 95 |
| | | | | **about 260**, under the 400 GPU-h budget |

The N = 4,000 cells of all three gaps are not run, for the reason above; each rules entry says what its verdict
covers without them.
