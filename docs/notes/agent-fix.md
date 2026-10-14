# agent-fix: seeded held-out batch and the smoothed-minimum floor fallback, project-wide

Branch `agent-fix` (from `restore-oct5` at dbbd509). Della checkout `~/trainRNNbrain_fix` (worktree of
`~/trainRNNbrain_c1ana`'s repo, `data` symlinked to the scratch data). Scratch outputs:
`/scratch/gpfs/TENGEL/pt1290/agent_fix/`. Started 10:34, 2026-10-14.

## 0. The two changes (PI decisions, 2026-10-14 morning)

1. **Seeded held-out batch** (e0fc531). `heldout_r2.heldout_batch` seeds the task's generator with
   `HELDOUT_SEED = 20261012`; `ablate_silent_units.seeded_batches` seeds the ordering batch's task
   generator with `ORDER_SEED = 20261013`. The R² cache key ends with `|seeded-v1`.
   **One addition beyond the brief, needed for its own check:** `score_net` also seeds noise draw d
   with `NOISE_SEED + d` (1000 + d, the ablation code's scheme). Reason: the model config has no seed,
   and `prepare_task_arguments` passes `task.seed` (None) into the RNN constructor, so
   `rnn.random_generator` was seeded from the unseeded global numpy stream. Without this, "the same
   weights scored twice give bit-identical R²" fails on every task, CDDM included.
2. **Floor fallback** (d48789e). `common.noisy_floor`: when the fit's beta sits on its upper bound
   (3), floor = minimum of the running median over the fit window, crossing = first iteration the
   running median reaches 1.05x it, the late-rise guard still applies; `floor_kind` ("fit" or
   "smoothed_min") in the returned dict. The fit parameters stay in `fit`.

Figure 1 and the supplementary ablation figure read the new folders `c1_ablation_seeded` and
`c1_retrain_seeded` (944c038); the old ones are untouched.

## 1. Checks, written 10:40 before any run

### Stage 1 (script `agent_fix/check_stage1.py`, CPU)

- S1a. `score_net` twice on the same weights gives bit-identical R² (==), one claim-1 network per
  task (CDDM s2657899762, DMTS s3322309204, flip-flop s1258717025; end weights). Fail if any differs.
- S1b. `heldout_batch` twice gives identical arrays, every task.
- S1c. Was CDDM's batch already deterministic? Prediction from the code: yes. `TaskCDDM.get_batch`
  loops over the coherence grid and uses `self.rng` only when `shuffle=True`, which heldout_batch
  never passes. Test: the pre-fix `heldout_batch` (dbbd509) equals the new one bit for bit on CDDM.
  On DMTS (`T_random_window: 10`, offsets from `self.rng`) and the flip-flop, two pre-fix calls
  should differ (the bug); the new batch differs from any pre-fix one.
- S1d. The brief's check "on CDDM the new R² equals the old cached value to < 0.001". Prediction:
  this FAILS, and not because of the batch: the old scorer's noise draws were unseeded, so each old
  cached CDDM value is one random 9-draw average (per-draw SD ~0.011 on CDDM, so ~0.004 per 9-draw
  mean, ~0.005 for a difference of two). Test: the pre-fix scorer twice on one CDDM network (should
  differ), and new against old cached for every CDDM network cached at 9 draws (expected differences
  of SD ~0.005, centred on 0). If the batch is identical (S1c) and the differences are centred on 0
  with that spread, the batch fix leaves CDDM unchanged and the noise seeding moves each value by
  scoring noise only.
- S1e. `heldout_r2.score_net` and the ablation code's `Scorer` on the same weights, both on CPU, now
  give the same number to < 1e-6 (same batch, same seeded draws). Before the fix they differed by up
  to 0.011 (trajectory 2026-10-13, CDDM s2670295995: 0.847 against 0.836).

### Stage 2 (script `agent_fix/check_stage2.py`, CPU; old rule = `common.py` at dbbd509)

Every run of every cell in `fig_paper_F1.SCALING`, the three claim-1 snapshot cells and
`fig_knobs_at_floor.LADDERS` (bound_impact.py's cell list), old rule against new:
- S2a. Exactly 6 runs change, all from "fit pinned at its bound (beta)" to a read-out with
  `floor_kind = "smoothed_min"`: flip-flop N = 500 size-series 0.952498 (old crossing 37,744), DMTS
  paper grid N = 1000 0.9992402, claim-1 flip-flop 2461724787 and 4113307577, ksweep control
  0.9532059, input scale 20 0.9531749. A 7th changed run, or one of these 6 still without a
  read-out, fails the check. (Amended 10:45, before running: LADDERS has gained two ladders since
  bound_impact ran on 13 Oct, the A4 rate/sparsity ladder and CDDM input scale. A changed run in
  those cells is allowed only if its old reason was "fit pinned at its bound (beta)"; it is listed
  separately. Any other 7th change fails.)
- S2b. The two claim-1 flip-flop runs cross within 1,000 iterations of agent-c1ff's independent
  numpy crossings on the 200k chain (52,159 and 59,480), with ~341 and ~289 active (±15).
- S2c. Every other run: reason, t_star and floor identical to the old rule (==).

### Stage 3 (ablation and retrain, GPU; new folders)

- S3a. CDDM: its held-out and ordering batches were already deterministic and the ablation code
  always seeded its noise, and no CDDM run changes floor. So every CDDM ablation (3 at the floor, 12
  at the end) reproduces the old JSON: r2 intact, all silent removed and the random mean within
  1e-4, the same active count. A larger difference means something else changed.
- S3b. Flip-flop and DMTS at the floor: within the old run-to-run spread (flip-flop "all silent
  removed" ±0.06, i.e. 2 SD of agent-c1ff's 0.031) for 1258717025 and the DMTS runs; the two new
  flip-flop read-outs (52k, 60k) are new numbers with no old counterpart.
- S3c. The retrain contrasts claim 1 rests on, per network: silent-pruned-and-retrained within 0.01
  of the intact fine-tuned control; random-pruned-and-retrained at least 0.01 below silent-pruned
  on the flip-flop.

## 2. Results

### Stage 2: floor fallback (job 15553594, `agent_fix/check_stage2.json`) - PASS

152 runs in the cells (bound_impact's 123 plus the two ladders added since). Exactly 6 changed, all
from "fit pinned at its bound (beta)" to a read-out with `floor_kind = smoothed_min`; no run in the
two newer ladders has beta on its bound. The other 146 runs return the same reason, crossing and
floor as the old rule, compared with ==. Read-outs: 137 runs under the old rule, 143 under the new.

| run | old (fitted floor) | new floor (smoothed min) | crossing | active there | late_rise |
|---|---|---|---|---|---|
| claim-1 flip-flop 2461724787 (200k chain) | 0.0415, beta 3 | 0.03392 | 52,159 | 341 | 1.007 |
| claim-1 flip-flop 4113307577 (200k chain) | 0.0371, beta 3 | 0.03392 | 59,951 | 292 | 1.005 |
| flip-flop size series N = 500, 0.952498 (s893469585) | 0.0370 | 0.03411 | 57,118 | 237 | 1.001 |
| flip-flop ksweep control (input-scale reference), 0.9532059 | 0.0465 | 0.03286 | 78,694 | 280 | 1.035 |
| flip-flop input scale 20, 0.9531749 | 0.0326 | 0.03104 | 14,734 | 614 | 1.094 |
| DMTS paper grid N = 1000, 0.9992402 (s1167683015) | 1.07e-4 | 6.83e-5 | 91,701 | 225 | 1.395 |

S2b: the claim-1 crossings agree with agent-c1ff's independent numpy crossings on the 200k chain
(52,159: identical; 59,951 against 59,480, 471 iterations apart, inside the 1,000 bar; agent-c2ff got
59,951 with the same scipy filter as production, so the gap is the running median's edge or tie
handling). Active 341 and 292 (bar: 341 ± 15, 289 ± 15). The DMTS run sits close to the late-rise
guard (1.395 against 1.5): its smoothed minimum is a trough between bursts.

### Stage 1: seeded held-out batch (job 15553593, `agent_fix/check_stage1.json`)

| check | CDDM | DMTS | flip-flop |
|---|---|---|---|
| S1a: score_net twice, bit-identical | 0.8361642162005106 twice: PASS | 0.9994159936904907 twice: PASS | 0.9525801406966315 twice: PASS |
| S1b: new heldout_batch twice identical | PASS | PASS | PASS |
| S1c: two pre-fix batches identical | yes | no (the bug) | no (the bug) |
| S1c: pre-fix batch equals the new one | **yes, bit for bit** | no | no |
| S1e: heldout_r2 against the ablation Scorer, CPU | diff 0.0: PASS | diff 0.0: PASS | diff 0.0: PASS |

**CDDM's held-out batch was already deterministic**, as predicted from the code: the fix changes
nothing in what CDDM is scored on. **The brief's "< 0.001 against the old cached value" fails on
CDDM, and the cause is the noise, not the batch.** The pre-fix scorer twice on the same CDDM weights
gives 0.8266 and 0.8310 (0.0043 apart). Over the 61 CDDM networks cached at 9 draws, new minus old
has mean -0.0003, SD 0.0043, median |diff| 0.0029 and max 0.0116; 12 of the 61 are within 0.001. The
spread matches the predicted ~0.005 for the difference of two 9-draw means and is centred on 0, so
the old CDDM values were unbiased but each carried about ±0.004 of noise-draw lottery. The largest
move is the CDDM claim-1 control s2670295995, 0.8474 → 0.8357: the "+2.7 SD outlier" of 13 Oct and
the "heldout_r2 0.847 against the ablation code 0.836" mismatch were this lottery, and both scorers
now give 0.8357 there. Within a cell the seeded values spread far less than the old ones (recurrent
noise 0.05, five seeds: 0.8370-0.8381 against 0.8392-0.8488), so most of the old between-seed R²
spread on CDDM was scoring noise.

Tests (`python tests/<name>.py`, job 15553598): 18 of 19 pass; `test_prune_rescale_mode.py` fails
("a weight changed sign"). It imports only `RNN_torch` and `Trainer`, which this branch does not
touch (the diff from dbbd509 is limited to five analysis scripts), so the failure is not from these
changes; see §4 for the rerun.

### Stage 3: ablation and prune-and-retrain (GPU jobs 15553711-16; new folders)

Outputs: `/scratch/gpfs/TENGEL/pt1290/c1_ablation_seeded` (9 floor + 36 end-of-training JSONs) and
`c1_retrain_seeded` (9 retrain-at-floor JSONs). Old folders untouched. Code 944c038 on the Della
checkout `~/trainRNNbrain_fix`; packed runner and lists in `agent_fix/run_list.sh`, `agent_fix/lists/`;
old-vs-new table `agent_fix/compare_abl.py`.

- **S3a PASS.** All 15 CDDM ablations (3 at the floor, 12 at the end) and all 3 CDDM retrains
  reproduce the old JSONs to the 4th decimal (active count, r2 intact, all silent removed, random
  mean, every retrain endpoint). The DMTS end ablations move by at most 0.0012 (the DMTS batch is
  now fixed; before it was a fresh jitter draw).
- **S3b PASS.** Flip-flop 1258717025 at 69k: all silent removed 0.2514 (four earlier runs on the
  same weights: 0.247-0.308). DMTS at the floor moves by at most 0.003.
- **S3c: 8 of 9 pass, one at the bar.** Silent-pruned-and-retrained against the intact
  fine-tuned control: CDDM +0.0005 to +0.0008; DMTS -0.0005 to +0.003; flip-flop 2461724787 -0.003,
  4113307577 -0.0025, **1258717025 -0.0103** (bar: within 0.01). Root cause, from the 250-iteration
  curve: the pruned network dips to 0.922 at 1,750 fine-tuning iterations and is still climbing at
  3,000 (0.9366 → 0.9405 over the last 1,250); two earlier retrains of the same network ended at
  0.9468 and 0.9462 (0.003-0.005 below their intact control). The fine-tuning batches are drawn from
  the task's unseeded generator (not in this brief's scope), so the endpoint carries a run-to-run
  spread of at least 0.006 on the flip-flop. Not re-run. Random-pruned stays 0.016-0.024 below
  silent-pruned on all three flip-flop networks.
- The two new flip-flop read-outs use the nearest snapshot to the crossing: 52,000 for 2461724787
  (crossing 52,159; trace 322 active there, 341 at the 52,200 probe, so the count moves by ~20
  between neighbouring probes) and 60,000 for 4113307577 (crossing 59,951; 292).

**Claim 1 at the floor, 9 networks, old (c1_ablation, c1_retrain) → new:**

| network | read at | active | r2 intact | all silent removed | as many at random | silent pruned + retrained | intact fine-tuned |
|---|---|---|---|---|---|---|---|
| CDDM s1423687489 | 26k | 415 | 0.8319 | 0.8326 | 0.054 | 0.8300 | 0.8295 |
| CDDM s2657899762 | 25k | 426 | 0.8337 | 0.8322 | -0.140 | 0.8267 | 0.8259 |
| CDDM s2670295995 | 25k | 438 | 0.8292 | 0.8277 | -0.017 | 0.8248 | 0.8241 |
| DMTS s198893703 | 21k | 313 → 312 | 0.9989 | 0.9661 → 0.9633 | -0.081 → -0.082 | 0.9987 → 0.9988 | 0.9993 |
| DMTS s3322309204 | 17k | 441 | 0.9993 | 0.9091 → 0.9096 | -0.128 | 0.9992 → 0.9988 | 0.9991 → 0.9958 |
| DMTS s3753404946 | 42k | 291 | 0.9992 | 0.9720 → 0.9727 | -0.142 | 0.9993 | 0.9994 |
| flip-flop 1258717025 | 69k | 266 → 267 | 0.9506 → 0.9500 | 0.248 → 0.251 | 0.014 → 0.011 | 0.9468 → 0.9405 | 0.9501 → 0.9508 |
| flip-flop 2461724787 | 27k → **52k** | 354 → 323 | 0.9386 → 0.9476 | 0.421 → 0.224 | 0.090 → 0.078 | 0.9368 → 0.9477 | 0.9436 → 0.9505 |
| flip-flop 4113307577 | 39k → **60k** | 334 → 292 | 0.9464 → 0.9505 | 0.476 → 0.525 | 0.044 → 0.016 | 0.9410 → 0.9483 | 0.9465 → 0.9508 |

Random-pruned + retrained, flip-flop: 0.920-0.924 → 0.921-0.931 (1258717025 0.9210, 2461724787
0.9245, 4113307577 0.9308). DMTS s3753404946 random-pruned still returns to the constant-output
solution (0.4185). DMTS s3322309204's intact control ends at 0.9958 against 0.9991 before: the DMTS
fine-tune is also unseeded and DMTS has brief bursts back toward chance.

Claim 1 stands on all nine networks. Means (Figure 1 panel d, from the builder): flip-flop
0.949 intact → 0.334 with every silent unit removed, 0.035 at random, 0.946 retrained (706 of 1000
silent, read at 60,333 on average); CDDM 0.832 → 0.831, -0.034, -0.335, 0.827 (574 silent); DMTS
0.999 → 0.949, -0.117, -0.166, 0.999 (652 silent).

### Figures (built on Della from 944c038; PDF + SVG + PNG in `~/Documents/GitHub/trainRNNbrain/img/internal_figures/`)

The versions they replaced are kept in `/scratch/gpfs/TENGEL/pt1290/agent_fix/figs_before_fix/`.

- `fig_paper_F1` (job 15555426). Panels a, b unchanged (183 of 1000 active; 945 untrained). Panel c:
  CDDM unchanged (N^0.464); DMTS N^0.515 (N = 1000 cell now 444.3 ± 193.1, crossings 34k mean, the
  fallback run included at 91.7k with 225 active). **The flip-flop series is drawn at the END of
  200k with three sizes (N^0.40), not at the end of 75k with four as in the paper.** That is not from
  these changes: `SCALING` points at the 200k cells since 458faf0 (agent-c2ff) and the N = 4000 200k
  cell does not exist until FFtop4000 lands (~15 Oct). Read at the floor, the flip-flop gives N = 500
  201.0 ± 18.3 (6 of 6 runs; the fallback run reads 237 at 57k), N = 1000 290.5 ± 4.9 (2 of 3), N =
  2000 386.7 ± 23.5, N^0.47 over three sizes. Panel d: the table above. Panel e: 116 / 152 / **277**
  further units silenced after the floor (DMTS was 272; its N = 1000 fallback run moves the mean
  crossing).
- `fig_supp_ablation` (same job): removing every silent unit changes r2 by -0.0104 to +0.0011 on
  CDDM (12 nets), -0.0458 to -0.0012 on DMTS (8), -0.79 to -0.25 on the flip-flop (15); at random no
  network above 0.060.
- `fig_f2_knob_tradeoff` and `fig_paper_F2_fails` (job 15553722; the F2 collect cache is
  `agent_fix/fig_paper_F2_fails_cache_fix.pkl`, the one it overwrote in the shared data folder is
  `agent_fix/fig_paper_F2_fails_cache_before_fix.pkl`). Active counts at the floor are unchanged
  except the flip-flop input-scale ladder: default 298 (1 of 3 read) → 289 (2 of 3: 298, 280), scale
  20 764 (2 of 3) → 714 (3 of 3: 913, 614, 614). R² at the floor moves by scoring-noise amounts and
  spreads far less within a rung: standard CDDM 0.829-0.830 (was 0.827-0.836); metabolic λ = 10
  0.778-0.779; weight decay 1e-4 0.817-0.818 and 1e-5 0.824-0.826 against the standard's
  0.829-0.830, so the weight-decay cost is now resolved (about 0.012 at 1e-4); W_rec penalty 0.2
  0.815 and 0.825; softplus 0.826-0.829; sigmoid 0.830-0.831; flip-flop input scale 0.949-0.954.
  The builder's layout check reports three legends over data (weight decay, metabolic, activation
  trace panels); the same overlaps are in the figure it replaced (checked by eye), so they are not
  from these changes.

## 3. Paper numbers that change (old → new; the paper is not edited here)

`sections/02_problem.tex`, claim-1 paragraph:
- l.39 "removing all of them costs 0.945 → 0.38" (flip-flop) → **0.949 → 0.33**.
- l.40-41 "as many at random (0.05 and −0.12)" → **0.03** (0.035) and −0.12.
- l.42 "recovers its full score (0.942 on the flip-flop" → **0.946**; DMTS 0.999 unchanged.
- l.44 "random ... and retraining recovers less (0.920–0.924 on the flip-flop" → **0.921–0.931**.
- Unchanged: CDDM 57%, 0.832 → 0.831, −0.03, −0.34; DMTS 0.999 → 0.95; "one of three networks never
  leaves the constant-output solution"; "−0.010 to +0.001" (CDDM, end of training).
- l.49-50 "(0.94–0.95 and 0.998–0.999)", the end-of-training retrains at N = 1000: not re-run in this
  pass (not in the brief); they come from the unseeded batch.

`sections/02_problem.tex`, size and silencing paragraphs:
- l.55-56 DMTS N^0.51: 0.515 now (two decimals: 0.51 in the figure legend, 0.52 if rounded); 45.3%
  and 23.1% unchanged.
- l.67 and main.tex l.123 (Fig. 1e caption) "272 on DMTS" → **277**; 116 and 152 unchanged.
- l.81 table, DMTS "439.7 ± 201.1, loss floor ≈ 28,000" → **444.3 ± 193.1, ≈ 34,000**.
- l.56-58, l.79 and main.tex l.111-114 (flip-flop at the end of 75k: 39.5% → 13.5%, N^0.48,
  269.3 ± 24.2; "every exponent lies between 0.46 and 0.51"): a rebuilt Figure 1 no longer shows
  this (200k end, three sizes, N^0.40) for the reason given above, not because of these changes.
  Hold Figure 1 back from the paper until N = 4000 at 200k lands, or rewrite these together then.

`sections/01_intro.tex` l.58: "440 on a delayed match-to-sample task" → **444**.

`sections/02_problem.tex`, knob section:
- l.122-125 flip-flop input scale "from about 300 to 394, 488, 611 and 764 ... (0.950–0.953)" →
  **about 290** (289, two runs) ... **714** (three runs), R² **0.949–0.954**.
- Unchanged: 418 (R² 0.83), 488, 465, 519, 485, 538, 486, the metabolic 400–460 and R² 0.78 at
  λ = 10, "about 50 units", sigmoid 245, 412/266/166/86, CDDM input-scale counts.
- l.103 (LaTeX comment) "on CDDM, R² differences below ~0.01 are scoring noise": no longer true;
  with the noise draws seeded, the within-rung spread is 0.001-0.004.

`sections/S1_supplementary.tex`, Supplementary Fig. (ablation) caption:
- l.61 DMTS "−0.047 to −0.002" → **−0.046 to −0.001**.
- l.62 flip-flop "0.24–0.80" → **0.25–0.79**.
- Unchanged: CDDM −0.010 to +0.001; "no network above r² = 0.06" (max 0.060).

`sections/08_methods.tex` (wording, not numbers):
- l.320-322 "on a held-out batch drawn once per network": the held-out batch is now one fixed
  seeded draw per task, shared by every network and scorer, and the ordering batch is seeded too.
- l.421-424 Task performance: say the noise draws and the held-out batch are seeded, so a score
  repeats exactly.
- l.401-405 exclusion rules: add the fallback (a fit whose stretch exponent sits on its upper bound
  is read at 1.05x the minimum of the running median over the fit window).
- l.410-411 "on the 3-bit flip-flop at 27,000–111,000" → **52,000–111,000** (claim-1 runs 52-69k;
  size series 54-111k).

## 4. Loose ends

- `tests/test_prune_rescale_mode.py` fails ("a weight changed sign") twice in a row on Della. It
  imports only `RNN_torch` and `Trainer`, which this branch does not touch; the other 18 test
  scripts and `common.py`'s self-check pass. Not investigated further.
- The fine-tune in `retrain_curve` still draws its training batches from the task's unseeded
  generator, so flip-flop and DMTS retrain endpoints vary by ~0.006 between runs. Seeding it would
  be one line (`tcfg.seed = ...` before instantiating the task there); not done, outside the brief.
- The R² cache (`data/heldout_r2_cache.json` on scratch) now holds `|seeded-v1` entries beside the
  old ones; the old entries are never read by this code. It is written without a lock by every
  checkout that scores, so concurrent writers can drop each other's entries (recomputed, not wrong).
- DMTS N = 1000 s1167683015's fallback read sits close to the late-rise guard (1.395 against 1.5).

## 5. Follow-up (coordinator, after the merge af0c605)

**Where the test failed.** `tests/test_prune_rescale_mode.py` failed on Della (job 15553598 and
again 15554886: CPU node, `torch-env` from `anaconda3/2024.10`, `CUDA_VISIBLE_DEVICES=` empty). The
coordinator reports it passes on the merged tree on the Mac, so the failure depends on the
environment.

**Seeded fine-tune batches.** `retrain_curve` now seeds the task generator of its training batches
with `FINETUNE_SEED = 20261014`, so every arm (intact, silent pruned, random pruned) sees the same
batch sequence and a rerun sees it again. CDDM's training batch is a fixed grid, so CDDM retrains
are unaffected.

Checks, written before running (output `/scratch/gpfs/TENGEL/pt1290/c1_retrain_seeded2/`):
- R1. Flip-flop 1258717025 at its floor, three repeats: every endpoint of every arm agrees across
  repeats within 0.001. If they do, the seeded retrain is repeatable (GPU arithmetic is the only
  remaining source of difference); if not, something other than the batches still varies.
- R2. The 0.0103 question. The seeded run gives one draw from the same distribution the unseeded
  runs sampled. Together with the three unseeded runs (silent pruned minus intact: -0.0033, -0.0048,
  -0.0103), the gap is noise around the bar if the seeded gap lies inside 0.01 and the four runs
  spread by more than 0.003; it is real if the seeded gap is also beyond 0.01.
- R3. The other two flip-flop networks, once each: silent pruned within 0.01 of intact, random
  pruned at least 0.01 below silent pruned, as in §2.

Results (jobs 15559806-10, code ff994a9; `c1_retrain_seeded2/rep1..3`):

| network | silent pruned + retrained | intact fine-tuned | gap | random pruned + retrained |
|---|---|---|---|---|
| 1258717025, repeats 1/2/3 | 0.94697 / 0.94697 / 0.94697 | 0.95058 (all three) | -0.0036 | 0.92122 (all three) |
| 2461724787 | 0.94751 | 0.95057 | -0.0031 | 0.92455 |
| 4113307577 | 0.94812 | 0.95076 | -0.0026 | 0.92944 |

- R1 PASS: the three repeats agree to the 5th decimal in every arm and at every 250-iteration point.
- R2: noise around the bar. Four runs of the 1258717025 retrain have now given gaps of -0.0033,
  -0.0048, -0.0103 (unseeded) and -0.0036 (seeded), a spread of 0.007. The -0.0103 run is the one
  whose pruned network dipped at 1,750 iterations; the seeded run has no dip (0.9341 at 250, rising
  to 0.9470).
- R3 PASS: the other two networks end 0.003 below their intact control, random pruning 0.019-0.023
  below silent pruning.
- Claim-1 flip-flop retrain numbers for the paper: silent pruned 0.948 (mean), random pruned
  0.921-0.929, intact 0.951. Figure 1 panel d still draws its stars from `c1_retrain_seeded` (the
  unseeded fine-tune); point `RETRAIN_DIR` at the seeded retrains when Figure 1 is rebuilt.

Paper draft: `ptolmachev/dead-relu-paper` branch `numbers-seeded` (710de69, worktree
`~/Documents/Projects/dead_relu_paper_fix`), not merged to main; compiles with latexmk, no undefined
references. Figure 1 and its caption are untouched, so the caption's "272" (panel e) now differs from
the text's 277 until Figure 1 is synced.
