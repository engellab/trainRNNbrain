# agent-c1ff: the claim-1 flip-flop networks re-read at their floor on the 300k chain

Branch `agent-c1ff` (from `restore-oct5` at a5b4f98). Della checkout `~/trainRNNbrain_c1ff`.

## 0. The three networks

The claim-1 flip-flop runs (standard 3-bit flip-flop, N = 1000, control). Each was trained 75k with
kept snapshots, continued to 200k, and two are now continued to 300k (FFtop300 15526698, 15526699).

| 200k folder seed | 75k seed (snapshots) | 300k seed | longest chain | floor read-out before the beta guard |
|---|---|---|---|---|
| 2461724787 | 1542248578 | 3194339211 | 300k | crossing 26,949, 354 active; beta = 3 (now rejected) |
| 4113307577 | 2851654641 | 3204992658 | 300k | crossing 39,142, 337 active; beta = 3 (now rejected) |
| 1258717025 | 2670297180 | none | 200k | crossing 69,050, 266 active; fit valid |

(The 300k seed for 4113307577 is read from the FFtop300 15526699 log once the run folder is written.)

## 1. Predictions and pass/fail checks, written 2026-10-14 05:45, before any 300k result was read

### 1a. The refitted floors

A floor read-out is VALID when all of these hold (thresholds fixed now):
- `common.noisy_floor` returns `reason = None` on the joined chain: floor above its lower bound,
  beta below 0.99 x 3, late_rise <= 1.5, the smoothed loss reaches 1.05x the floor.
- The loss does not end below the floor: late_rise (median of the last 10% over the floor) >= 0.95.
  `noisy_floor` does not check this side; I check it here. A fit that the loss ends below by more
  than 5% is the failure mode the beta guard was built for, so it counts as no floor even if
  beta is free.
- The weight snapshot used lies within 500 iterations of the crossing.
- tau on its bound is reported, not used to reject (the project rule).

Predictions:
- P1. Both 300k fits pass, with beta well below 3 (guess 0.3-1.5). Confidence moderate (~60%):
  the loss was still falling at 200k (ending at 0.84x and 0.91x of the rejected floors), and
  100k more iterations may not flatten it enough for a clean fit.
- P2. Both crossings land later than 27k and 39k, near the 1.05x-smoothed-minimum crossings on the
  200k chain (52k, 60k). Since the minimum can only fall with 100k more iterations, the
  1.05x-minimum crossing on the 300k chain can only move later; I expect the fitted crossings in
  50-90k. Pass: within +-15k of the 1.05x-minimum crossing recomputed on the 300k chain by
  independent code (a running median and its minimum, not the fit).
- P3. Active at the new crossings 260-345 (at 52k and 60k on the 200k chain: 341 and 292).
- P4. 1258717025 on its 200k chain reproduces the current read-out: crossing 69,050, snapshot
  69,000, 266 active. Its ablation JSON, re-run on the same device type (GPU), matches the existing
  one to within 0.002 of r2 (a mismatch means the code path or the environment changed and every
  new number is suspect until explained).

### 1b. Ablation and prune-and-retrain at the new floors

Expected from the paper's claim 1 (flip-flop at its floor): r2 intact 0.94-0.95; all silent removed
0.25-0.48; as many at random 0.01-0.09; silent pruned and retrained 3,000 iterations 0.937-0.947
(intact fine-tuned 0.943-0.950); random pruned and retrained 0.920-0.924; 266-354 active.

- P5. Intact held-out r2 0.94-0.95; removing every silent unit 0.2-0.5; as many at random < 0.15.
- P6. Pruning the silent units and retraining restores within 0.01 of the intact fine-tuned
  network; random pruning plus retraining stays at least 0.01 below the silent-pruned network.

What would change claim 1 on the flip-flop (decided now):
- C-a. For any valid network, silent-pruned-and-retrained ends more than 0.01 below the intact
  fine-tuned control: the network needs its silent units at the floor; "does not need them" fails.
- C-b. For any valid network, random-pruned-and-retrained ends within 0.005 of silent-pruned: the
  silent units are no more dispensable than any others after retraining; the contrast the paper
  draws disappears for that network.
- C-c. Removing the silent units costs nothing (r2 within 0.02 of intact): the flip-flop's
  "wired in but not needed" wording would become "not needed at all", a change of wording only.
- Otherwise claim 1 stands, and only the quoted ranges are updated.

If neither 300k fit is valid, the flip-flop's at-floor ablation rests on one network
(1258717025), and that is reported as such, not papered over with the 1.05x-minimum read-out.

## 2. Method

- Code: Della `~/trainRNNbrain_c1ff` at this branch. The only change since 7608099 (the commit that
  produced the existing claim-1 at-floor JSONs) on the ablation path is the beta guard in
  `common.noisy_floor`. `ablate_silent_units.floor_snapshot` calls `common.noisy_floor` on
  `common.noisy_loss_series` (joined along `warm_start_chain`) and takes the nearest state from
  `chain_snapshots`, which collects kept snapshots of every segment
  (`c1_snapshots/<task>_<experiment_tag>/s<segment seed>/it*`). So the 300k chain is used
  automatically once the 300k run folders exist; nothing in the code was changed.
- Floor read-out with independent cross-checks: `/scratch/gpfs/TENGEL/pt1290/c1ff/c1ff_floor_read.py`
  (not committed; scratch). It walks `paths.init_from` with plain yaml and joins the loss files by
  hand (compared against the production chain and series), recomputes the 1.05x-smoothed-minimum
  crossing with a numpy sliding-window median, checks which fit parameters sit on a bound and the
  0.95 lower bar on late_rise, and looks up the snapshot and the trace's active count.
  Check on the 200k chain (output `c1ff/read200.json`): production and hand-built chain and series
  agree for all three runs; 2461724787 and 4113307577 rejected (beta = 3.0, late_rise 0.82 and
  0.92); 1258717025 valid (floor 0.0338, A 0.113, tau 6.2k, beta 0.57, late_rise 1.006, crossing
  69,050, snapshot 69,000, 266 active). The 1.05x-minimum crossings come out at 52,159, 59,480 and
  67,348 (agent-c2ff had 52,159, 59,951, 67,348; the 59k difference is the running-median edge or
  tie handling, not investigated).
- Ablation and retrain: the existing launcher `slurm/SilentReLU_c1_ablation_della.slurm` in LIST
  mode with `REPO=~/trainRNNbrain_c1ff`, one GPU job per network and read-out (the GPU ablation
  takes ~3 min and the retrain ~35 min; on CPU the 9,000 fine-tuning iterations of N = 1000,
  T = 300, batch 1024 would take on the order of a day, so GPU). New output folders, nothing
  overwritten: `/scratch/gpfs/TENGEL/pt1290/c1_ablation_floor300` and `c1_retrain_floor300`.

## 3. Results (2026-10-14 06:40)

### 3a. The floor read-out on the longest chain

| run (200k seed) | chain | fit floor, A, tau, beta | late_rise | verdict | crossing | active |
|---|---|---|---|---|---|---|
| 2461724787 | 300k (75k + 125k + 100k, 600 states) | 0.0408, 0.776, 4.96k, **3.0** | 0.834 | rejected: beta on its bound; loss ends 17% below the floor | none (1.05x-min: 52,212, 341 active) | - |
| 4113307577 | 300k (600 states) | 0.0365, 0.250, 12.7k, **3.0** | 0.930 | rejected: beta on its bound; loss ends 7% below the floor | none (1.05x-min: 60,465, 289 active) | - |
| 1258717025 | 200k (400 states) | 0.0338, 0.113, 6.15k, 0.57 | 1.006 | valid, no parameter on a bound | 69,050 (snapshot 69,000) | 266 |

P1 FAILED: 100k more iterations did not give either run a valid fit. P2 and P3 are moot (no
crossing). The extra training did not lower the loss: the smoothed minimum is 0.03391 and 0.03386
on the 300k chain against 0.03392 and 0.03396 on the 200k chain (a change under 0.3%), and the
1.05x-minimum crossings barely move (52,159 -> 52,212; 59,480 -> 60,465). Both runs have sat at
about 0.034 since 52-60k, the same level as 1258717025 (0.0340). So these two networks are
converged; the stretched exponential does not describe their curve (a fast drop, then a flat
plateau), and with beta pinned at 3 it puts the floor 8-20% above the plateau. The rejection is a
failure of the fit, not of convergence. Whether to read such runs at 1.05x their smoothed minimum
is Pavel's call; I did not ablate them, as pre-registered.

### 3b. Ablation and prune-and-retrain at the floor (1258717025 only)

Floor ablation, same weights (69,000), four runs: the existing JSON plus my three
(`c1_ablation_floor300/`, `rep1/`, `rep2/`):

| run | active | r2 intact | all silent removed | as many at random (mean of 5) |
|---|---|---|---|---|
| existing (c1_ablation) | 266 | 0.9506 | 0.2478 | 0.014 |
| floor300 | 267 | 0.9506 | 0.2933 | 0.006 |
| rep1 | 267 | 0.9484 | 0.3077 | 0.011 |
| rep2 | 267 | 0.9513 | 0.2473 | 0.008 |

Prune and retrain 3,000 iterations (`c1_retrain_floor300/`), next to the existing run:
intact fine-tuned 0.9502 -> 0.9510 (existing 0.9494 -> 0.9501); silent pruned 0.2671 -> 0.9462
(0.2699 -> 0.9468); random pruned 0.0646 -> 0.9212 (0.0694 -> 0.9203).

Against the pre-registered checks: P5 passes; P6 passes (silent-pruned ends 0.005 below intact
fine-tuned, inside 0.01; random ends 0.025 below silent-pruned). C-a, C-b and C-c do not fire. The
numbers sit inside the paper's ranges (removed 0.25-0.48, retrained 0.937-0.947, random
0.920-0.924, active 266-354).

**Claim 1 on the flip-flop.** The conclusion stands, but under the current rule it rests on one
network: the other two of the paper's three flip-flop floor read-outs (27k and 39k) were made by
fits the beta guard now rejects, and 300k did not produce valid fits. The quoted ranges
0.25-0.48 / 0.937-0.947 / 0.920-0.924 / 266-354 come from those three read-outs and need
rewriting (one network: removed 0.25-0.31, retrained 0.946-0.947, random 0.920-0.921, 266-267
active), or the rule needs to change for plateaued runs.

### 3c. The unseeded flip-flop batch: spread and a proposed fix (not applied)

Spread, from the four ablation runs on identical weights: "all silent removed" ranges
0.247-0.308 (mean 0.274, SD 0.031); r2 intact 0.948-0.951; active 266-267 (one unit at the
boundary). The retrain endpoints repeat within 0.001 (two runs). So the "removed" number of each
flip-flop network carries about +-0.03 of batch-sampling noise, and differences between networks
of that size in the paper's 0.25-0.48 range are not meaningful.

Cause: the flip-flop config has `task.seed: None`. `prepare_task_arguments` passes it through, so
`TaskBase.__init__` builds `self.rng = np.random.default_rng()` (unseeded), and every batch the
task draws comes from that generator. `seeded_batches` calls `np.random.seed(BATCH_SEED)` and
`torch.manual_seed(BATCH_SEED)`, which do not reach a `default_rng()` instance. Both the held-out
batch (`heldout_r2.heldout_batch`) and the ordering/gate batch (`seeded_batches`) are therefore
fresh on every run. Only the noise draws (`rnn.random_generator`) are seeded. CDDM and DMTS were not
checked (CDDM's held-out inputs are a fixed coherence grid, but its trial-level draws may also use
the task's rng).

Minimal fix I would propose (two lines, ablation only):
- in `ablate_silent_units.seeded_batches`, give each task its own seeded generator right after
  instantiating it: for the held-out batch, either pass `tcfg.seed = BATCH_SEED` inside a local copy
  of `heldout_batch`, or set `task.rng = np.random.default_rng(BATCH_SEED)` before `get_batch()`;
  for the ordering batch, `task.rng = np.random.default_rng(BATCH_SEED + 1)` (a different seed, so
  the ordering and held-out batches stay distinct).
- Do not seed inside `heldout_r2.heldout_batch` itself without deciding it project-wide: every
  held-out R2 in the paper goes through it, and seeding there changes all of them (by noise-level
  amounts). Seeding it would also make held-out R2 repeatable everywhere, which is an argument
  for doing it once, deliberately.
After the fix, re-running each flip-flop ablation would give one fixed number per network; the
spread above shows how far each could move (about 0.03).

### Partial results (written at the first hand-back; superseded by §3)

- P4 FAILED as written, root-caused: re-running the ablation of 1258717025 on the same weights
  (69,000) gives a different number each time. Four runs: active 266/267/267/267, r2 intact
  0.948-0.951, all silent removed 0.247 / 0.293 / 0.308 / 0.247, as many at random 0.006-0.014.
  Cause: the flip-flop config has `task.seed: None`, so `TaskBase` uses an unseeded
  `np.random.default_rng()`; `np.random.seed(BATCH_SEED)` in `seeded_batches` does not reach it.
  The held-out batch and the ordering batch are fresh draws on every run; only the noise draws are
  seeded. So "all silent removed" on the flip-flop carries a run-to-run spread of about 0.06.
- Retrain of 1258717025 repeated (`c1_retrain_floor300`): intact 0.9502 -> 0.9510, silent
  pruned 0.2671 -> 0.9462, random 0.0646 -> 0.9212 (existing: 0.9501, 0.9468, 0.9203). The
  retrain numbers repeat within 0.001.
- Both FFtop300 runs COMPLETED (300k folders seed 3194339211 <- 2461724787, 3204992658 <-
  4113307577). Floor read job 15541920 (`c1ff/read300.json`) was still running at hand-back.
