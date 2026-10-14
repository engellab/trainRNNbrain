# B4, the plasticity test: code, tests, what is running

*Agent b4, night of 13-14 October 2026, branch `agent-b4`. Plan: `docs/notes/agent-bplan.md` §B4
(branch `restore-oct5`). Pavel approved new training on Spock for tonight.*

**Question.** When a trained 3-bit flip-flop network learns a new task with the same inputs, do its
silent units come back? The new task is `NBitFlipFlopHyper` at k = 3: the same 3 input channels, 7
outputs (every product of the bit states). Only the read-out is new.

## 1. Code (commit `c1396ae`, pinned from `7739a60`)

`git diff --stat 7739a60 c1396ae` shows five files, all from this agent:

| File | Change |
|---|---|
| `trainRNNbrain/training/run_experiment.py` | The warm-start loader moves into `load_warm_start(rnn, folder, keep_fresh=())`; with `keep_fresh` empty its behaviour is the old one, line for line. New key `paths.transfer_from` |
| `configs/paths/paths_DELLA.yaml`, `paths_local.yaml` | `transfer_from: null`, with its rules in a comment |
| `tests/test_transfer_warm_start.py` | 7 contracts, below |
| `slurm/SilentReLU_b4_transfer_spock.slurm` | the launcher (force-added; `slurm/*.slurm` is gitignored) |

**What a transfer does** (`paths.transfer_from=<source>` with `paths.init_from` null):
- copies `W_rec`, `W_inp`, `y_init` and a trainable bias from the source; the standard network's
  bias is fixed at 0 and comes from the config, as in every warm start;
- keeps this run's own seeded draw of `W_out`, of the new shape (7, N);
- raises if `paths.init_adam` is true (the source's moments belong to its own read-out), and raises
  if `trainer.iter_offset` is a nonzero number (a 200k source is itself a continuation, saved with
  `iter_offset: 75000`; inheriting it would date the new task's trace from 75,000);
- joins no parent trace, because `init_from` is null.

**Why a new key.** `common.warm_start_chain` follows `paths.init_from` and `noisy_loss_series`
joins the losses along that chain. With the source under `transfer_from`, the chain of a transfer
run is the run alone, so its floor is fitted on the new task's loss only. Had the source gone under
`init_from`, the floor would be fitted through the old task's loss too, and a kept-snapshot source,
which has no `TrainLosses.json`, would make `noisy_loss_series` return nothing.

**A continuation of a transfer run** (a top-up, should one be needed) sets `init_from` to the
transfer run and inherits `transfer_from` in its config. `init_from` then wins: the continuation is
an ordinary warm start of the 7-output network, and its chain is [transfer run, continuation]. The
existing top-up launcher works for it unchanged (Della); Spock has no top-up launcher yet.

**Version pin: `7739a60`, not `b524ac7`.** The young sources were trained at `b524ac7`, the old
(200k) sources' last 125k at `7739a60`. The training code of the two commits differs in one line:
a kept snapshot's folder name adds `iter_offset`, which is 0 for every transfer run, so the two
commits train a transfer identically. `7739a60` also holds the top-up launcher a continuation would
use, and every other B item trains from it. Training code (`trainer/`, `rnns/`, `tasks/`,
`training/`, `configs/`) is unchanged from `7739a60` to the head of `restore-oct5`, so the analysis
checkout rebuilds the hyper task exactly as it was trained.

## 2. Tests

All 19 test files pass on the Mac (18 existing at their baseline, plus the new one), and the new
test and `test_periodic_checkpoint.py` pass on Spock in `torch-env` (torch 2.7.0).

`tests/test_transfer_warm_start.py` runs `run_experiment.py` end to end at N = 20 from saved
configs, as the launcher does, with `trainer.lr=0` so that any difference is the copy's:
1. `keep_fresh=("W_out",)` copies W_rec, W_inp and y_init exactly; W_out equals a fresh network of
   the same seed and shape (7, N).
2. The ordinary warm start copies W_out too, still raises on a shape mismatch, and a misspelt
   `keep_fresh` name raises.
3. Through the full path: saved W_rec, W_inp, y_init and bias equal the source's bit for bit; W_out
   is the fresh draw; the trace starts at iteration 0; the config holds `transfer_from` = source,
   `init_from` null, `iter_offset` 0, the hyper task with 7 outputs.
4. `warm_start_chain` of a transfer run is the run alone; its noisy-loss series is its own 6 losses.
5. From a continuation's config: an inherited `init_from` makes it an ordinary warm start, which
   stops on W_out's shape; an inherited `iter_offset` stops on the clock; `init_adam=true` stops.
6. Continuing a transfer run carries W_out over, continues the clock, and its chain stops at the
   transfer run.
7. A kept snapshot works as a source as-is.

**Mutation check** (that the tests can fail): recording the source under `init_from` (the naive
implementation) fails contract 3; deleting the clock guard fails contract 5.

## 3. Smoke test (Mac CPU, the launcher itself)

The launcher ran locally with `module`, `conda` and `srun` stubbed, on the real N = 1000 sources
copied from Della, `ITERS=30` (old) or 12 (young, fresh), snapshots every 10:

| Mode | Source | Active at iteration 0 | r² after the run | Notes |
|---|---|---|---|---|
| transfer, old | 200k run, seed 4113307577 | 194 of 1000 (the source's own trace at 199,900: 195) | -0.09 → 0.31 in 30 iterations | |
| transfer, young | snapshot 52,500 of seed 1542248578 | 335 | 0.13 after 12 | |
| fresh | same config, no weights | 937 | 0.05 after 12 | |

Each saved config holds the hyper task (7 outputs), `iter_offset` 0, `init_from` null and the right
`transfer_from`; `warm_start_chain` returns the run alone; snapshots land in `it0000010/`, outside
the data tree; LastParams has W_out (7, 1000).

**Independent check of the iteration-0 silent set** (threshold set before running: at least 99% of
units labelled the same and active counts within 10 units). A numpy forward pass written from the
model equations and the participation spec in HANDOFF §3, on the SOURCE's saved weights and a new
batch of hyper-task inputs, against the transfer run's own trace at iteration 0. Old: 194 vs 194,
100% of units agree. Young: 334 vs 335, 99.9% agree. PASS. So the trace's first probe gives S₀,
the source's silent set on the shared inputs. Scripts: the agent's scratchpad
(`check_s0.py`, `inspect_run.py`, `floors.py`); not committed.

**One behaviour to keep in mind when reading results.** With fresh Adam moments, the first steps
move every weight with a nonzero gradient by about one learning rate (4.6e-4 at N = 1000): after 30
iterations W_rec had moved by 9.6% of its norm (old) and W_inp by 0.3%. A silent ReLU unit's
incoming weights get no task gradient, only the coupled weight-decay term, which Adam rescales to a
full-size step toward zero, the same dynamics as in the original training. S₀ is measured before
any step, at iteration 0.

## 4. Sources and decisions

| Cell | Source (Della) | Iteration |
|---|---|---|
| young | `data/c1_snapshots/NBitFlipFlop_c1_snapshots/EqType=h_N=1000_arm=control/s1542248578/it0052500`, `s2851654641/it0060000`, `s2670297180/it0067500` | 52.5k, 60k, 67.5k |
| old | `trained_RNNs/NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control/*` (seeds 2461724787, 4113307577, 1258717025, continuing 1542248578, 2851654641, 2670297180) | 200k |
| fresh | the young snapshots' configs, `MODE=fresh` (no weights loaded) | 0 |

**Decision 1, the young snapshots.** The plan named the floor crossings 27k, 39.5k and 69.5k.
Since the beta guard (trajectory, 13 Oct), two of those three fits are no floor
(`common.noisy_floor` on the 0-200k chains returns "fit pinned at its bound (beta)" for
2461724787 and 4113307577; the third crosses at 69,050). So the young source is read with one rule
for all three networks: the first kept snapshot at or after the noisy loss's first crossing of 1.05×
its smoothed minimum over 0-200k (crossings 52,159, 59,951 and 67,348, recomputed here; agent-c2ff
found the same numbers). On the one network with a valid fit the two rules agree (67k vs 69k).
The minima of the three runs agree to 0.2% (0.0339-0.0340). This choice changes only which weights
are transferred; every transfer run is still read at its own noisy-loss floor.

**Decision 2, 200k in one job instead of 100k + conditional top-up.** At 0.21 s/it on Spock (B5's
N = 1000 runs, 13 Oct) a 200k run takes ~12 h, inside one job; Spock has no top-up launcher; the
flip-flop needed 200k from scratch, so the fresh arm would likely need the second 100k; and equal
budgets keep all three arms comparable. Extra cost over the 100k plan: ~55 GPU-h.

**Decision 3, the fresh control is built from the source's own config**, so it differs from the
transfer runs only in its initial weights. The plan had suggested the remedy-grid launcher, which
does not exist on this branch.

**Decision 4, the young/old pairing** is by network, not by array index: young s1542248578 ↔ old
s2461724787; young s2851654641 ↔ old s4113307577; young s2670297180 ↔ old s1258717025.

**Copies.** The six sources (config + LastParams) were copied Della → Mac → Spock into
`~/trainRNNbrain/data/b4_sources/` on Spock; md5 sums of all 12 files match Della.

## 5. Pre-registration (written before submitting)

Read every run at its own noisy-loss floor: `common.noisy_floor(common.noisy_loss_series(run))`,
the run's chain being the run alone. A run with no valid read-out (any `reason`, including the beta
guard, the late rise above 1.5×, "never leaves its starting level") is excluded and counted.

- S₀ = units silent at iteration 0 (trace `participation[0]`, noise-free, active if
  p ≥ 0.05·q95(p)). A* = units active in the stored participation vector at the first stored
  iteration at or after the crossing (stored every 100). Revived = |S₀ ∩ A*| out of |S₀|.
- Also reported per run: newly silenced (active at 0, silent at the floor); active count at the
  floor against fresh; held-out r² at the floor (`heldout_r2.py`, first kept snapshot after the
  crossing); |S₀ ∩ A(200k)|, so a revival that does not last is visible.
- **Gate.** The test is informative at N = 1000 only if the fresh networks' mean active count at
  their floor exceeds the old networks' mean count at transfer by more than 3 fresh-network SDs.
- **Silent units do NOT come back** (supports the claim) if fewer than 10% of S₀ revive in every old
  seed, AND the old transferred networks end more than 3 fresh SDs below the fresh networks' active
  count.
- **Silent units CAN be recruited** (falsifies the claim) if 30% or more of S₀ revive in at least 2
  of 3 old seeds, OR the old transferred networks' active count at the floor comes within 1 fresh SD
  of the fresh networks'.
- Between the two bars: numbers, no verdict. A verdict needs at least 2 of 3 old runs and 2 of 3
  fresh runs with a floor read-out; otherwise the numbers are given at 200k, labelled as end of
  training, without a verdict.
- Young runs: the same numbers, read as a comparison with old (does recruitment fall with age); no
  separate verdict.
- R² is read separately: a transferred network matching fresh r² with fewer units shows lost
  recruitment without lost performance, which is not Dohare's loss of plasticity.
- Known risk to the read-out: `noisy_floor` takes the starting level from iterations 100-400 and
  needs the smoothed loss to fall below half of it. A transfer run that learns the 3 single-bit
  outputs within its first 100 iterations may start low and be rejected as "never leaves its
  starting level". If that happens it is reported, not patched.

## 6. Submitted

Submitted 14 Oct 00:32 EDT on Spock, from `~/trainRNNbrain_b4` (clean, detached at `c1396ae`),
partition `all`, 1 GPU (L40S-46G or A100-40G), `--mem=32G`, `--time=48:00:00`, 200,000 iterations,
kept snapshots every 500.

| Job | Cell | Sources (array task i = i-th in sorted order) |
|---|---|---|
| 6929407_[1-3] | `src=control_age=old` | the three 200k runs, sorted by folder name: 2461724787, 4113307577, 1258717025 |
| 6929408_[1-3] | `src=control_age=young` | snapshots s1542248578/it0052500, s2670297180/it0067500, s2851654641/it0060000 |
| 6929409_[1-3] | `src=fresh` (`MODE=fresh`) | the same three snapshot configs, no weights |

- Output: Spock `~/trainRNNbrain/data/trained_RNNs/NBitFlipFlopHyper_b4_transfer/EqType=h_N=1000_<cell>/<run>/`
- Snapshots: `/mnt/cup/labs/tengel/home/tolmach/trainRNNbrain_b4_snapshots/NBitFlipFlopHyper_b4_transfer/EqType=h_N=1000_<cell>/s<seed>/it<iters>/`
  (the lab share, 23 TB free; ~3.1 GB per run, ~28 GB in all). The first submission (6929404-6929406,
  00:30) wrote snapshots to the 95 GB home quota, which also holds the B2/B3 outputs; it was
  cancelled while pending, before any task started, and resubmitted with `SNAP_TO` on the lab share.
- Logs: Spock `~/trainRNNbrain_logs/B4xfer.<job>_<task>.out`.
- Cost: 9 runs × ~12 h (0.21 s/it, B5's measured Spock rate) ≈ 110 GPU-h. With the N = 4000 cells
  still to come (plan: ~525 GPU-h at 100k), B4 stays inside the plan's minimal budget.
- Queue at submission: every task pending on priority behind 19 B5 tasks (Slurm estimates their
  start at 17:00-21:30 on 14 Oct) and 24 B2/B3 tasks, all from the same account. Expected start late
  14 Oct to 15 Oct; finish ~12-14 h after each task starts, so about 15-16 Oct.
- Check: `squeue -u pt1290 -n B4xfer`; progress `tail -c 300 ~/trainRNNbrain_logs/B4xfer.6929407_1.out`.

## 7. Not done, and N = 4000

- **N = 4000 is not submitted.** Its old sources (`NBitFlipFlop_paper_grid_c200000/EqType=h_N=4000_arm=control`)
  are still training on Della until ~15 Oct 15:30. Spock is likely the wrong cluster for it: a
  40 GB card ran out of memory at N = 4000 on Della (1 Oct, `F2fill.14841831_9`), Spock's cards are
  A100-40G and L40S-46G, and Spock caps a job at 96 h where Della's 80 GB cards run 2.07 s/it
  (100k ≈ 58 h). Recommendation: run N = 4000 on Della with `--constraint=gpu80`, from `c1396ae`,
  using this launcher's command with the Della module lines, in 100k segments.
- Zero-cost baseline (revival on the old task between the young and old states of the same
  networks, from the claim-1 traces) and gradient reach at transfer: analysis only, not done.
- Spock → Della sync of the results: not set up; `slurm/sync_b5_spock_to_della.sh` on `b5-optimizer`
  is the template.
