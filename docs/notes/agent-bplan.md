# Plans for B2, B3 and B4: cells, launchers, costs, and what would falsify each claim

*Written 13 October 2026, 23:40 EDT, on branch `agent-bplan` (from `restore-oct5` at 8449dc7).
Nothing here has been submitted. Every number marked "measured" was read from Della on 13 October.*

## 0. What every plan shares

**Code commit.** All three items train from `~/trainRNNbrain_c1top` on Della, which is clean at
7739a60. Its training code equals b524ac7 (the claim-1 snapshot runs) except one change: a kept
snapshot's folder carries the absolute iteration. That change matters as soon as a run is continued
in a second segment, so 7739a60 is used throughout rather than `~/trainRNNbrain_c1abl`. The training
code (`trainer/`, `rnns/`, `training/`) is unchanged between 7739a60 and the head of `restore-oct5`.

**Config keys.** Every key B2 and B3 need exists at 7739a60 (checked in `configs/trainer/trainer_ptrack_freshbatch.yaml`
and `configs/model/rnn_relu_standard.yaml`): `trainer.lambda_frm` and `frm_args`, `model.sigma_w`,
`trainer.prune_args.reinit_mode=rescale` with `rescale_target_frac`, `rescale_active_only`,
`rescale_cap`, `rescale_protect_frac`, `rescale_norm_mode`, `rescale_normalize`, `rescale_alpha`,
`check_every`, `patience`, and `trainer.checkpoint_every` with `+trainer.checkpoint_keep`.

**Smoke test (done, Della login node, 13 Oct 23:30).** A `git archive` of 7739a60 in a scratch
folder, N = 60, 30 iterations, with all three remedies on at once (`lambda_frm=0.03`,
`model.sigma_w=1.0`, the rescale-target-10 block below) and kept snapshots every 10. It trained, the
saved config carries all three settings, and two snapshots were written. The folder was deleted
afterwards. So B2 and B3 need **no code change**, only a launcher.

**The launcher for B2 and B3** is a new file, `slurm/SilentReLU_remedy_grid_della.slurm`, copied from
`SilentReLU_c1_snapshots_della.slurm` (same task flags, same tracking flags, same snapshot tree), with
four environment knobs in the style of its `INPUT_ROW_NORM`:

| Knob | Sets | Arm name gets |
|---|---|---|
| `LAMBDA_FRM=<λ>` | `trainer.lambda_frm=<λ>` (default 0) | `frm<λ>` |
| `SIGMA_W=<σ>` | `model.sigma_w=<σ>` (default 0) | `sw<σ>` |
| `RESCALE=1` | `trainer.prune_reinit=true` plus the rescale block below | `rsc10` |
| `CKPT_EVERY=<n>` | `trainer.checkpoint_every=<n>` (default 500) | — |

The knobs must *replace* the launcher's hard-coded `trainer.lambda_frm=0` and
`trainer.prune_reinit=false`, not be appended after them, so the command line never carries two
values for one key. `REPO` defaults to
`$HOME/trainRNNbrain_c1top`; submit with `--chdir=$HOME/trainRNNbrain_c1top`. The launcher is
gitignored like every `slurm/*.slurm`, so it goes in with `git add -f`, and is pulled into
`~/trainRNNbrain_c1ana` (the analysis checkout) to be submitted from there, as `c6_topups.sh` did.
`c1top` itself must stay clean.

The rescale block is the operating point already on disk (`NBitFlipFlop_size_series`,
`NBitFlipFlop_target_ladder/..._tgt=10.0`), copied from `SilentReLU_size_series_della.slurm`:

```
trainer.prune_reinit=true trainer.synaptic_scaling=false
trainer.prune_args.reinit_mode=rescale trainer.prune_args.rescale_norm_mode=self
trainer.prune_args.rescale_target_frac=10.0 trainer.prune_args.rescale_active_only=true
trainer.prune_args.rescale_normalize=true trainer.prune_args.rescale_alpha=1.002
trainer.prune_args.rescale_cap=64.0 trainer.prune_args.rescale_protect_frac=0.20
trainer.prune_args.check_every=1 trainer.prune_args.patience=1
```

**Budget.** 200,000 iterations for every flip-flop run, from scratch, in one job where the time limit
allows. Runs at N = 4000 do not fit one job and run in two segments: the remedy-grid launcher with
`ITERS_OVERRIDE=100000`, then `SilentReLU_ff_topup200k_della.slurm` with `PARENTS` set to that
output, `TARGET=200000`, the same `KEEP`, and `--dependency=afterok:<first job>`. The top-up
continues from the parent's own saved config, so every setting carries over.

⚠️ **The rescale rule's own state is not saved.** `_rescale_cum`, `_rescale_growing`,
`_rescale_episodes` and `_rescale_refract_until` live only in the `Trainer` object, so a warm start
resets every unit to "dormant, no boost used". Every rescale run on disk at 75k and 200k already
carries such resets (at 40k and 75k). A rescale run split at 100k gets one reset there. Runs at
N ≤ 2000 avoid the issue by running in one job.

**Measured speed** (seconds per iteration, Della training logs `~/trainRNNbrain_logs/*.out`,
elapsed ÷ iteration, on full-GPU nodes):

| N | control | rate + sparsity | synaptic noise σ_w 1.0 | rescale target 10 | used here for one remedy / all three |
|---|---|---|---|---|---|
| 500 | 0.16-0.19 | 0.18 | 0.19 | 0.20 | 0.18 / 0.22 |
| 1000 | 0.24 | 0.26 | 0.26 | — | 0.26 / 0.30 |
| 2000 | 0.62 | 0.71 | 0.67 | 0.75 | 0.72 / — |
| 4000 (80 GB card) | 2.07 | 2.22 | 2.23 | 2.51 | 2.25 / 2.6 |

The all-three column is an assumption (the slowest single remedy plus 5-10%). The `RMtop200` N = 1000
tasks run at 0.24-0.74 s/it because `c6_topups.sh` set no GPU constraint; the slow ones landed on
partial GPUs. Every plan below therefore asks for `--constraint=nomig` at N ≤ 2000 and
`--constraint=gpu80` at N = 4000 (a 40 GB card ran out of memory at N = 4000 on 1 October,
`F2fill.14841831_9`).

**Queue limits** (`sacctmgr show qos`, 13 Oct). The "44 GPUs" cap belongs to one queue only:

| QOS | Wall limit | Per-user cap | Ours now |
|---|---|---|---|
| gpu-short | 24 h | 44 GPUs | 44 running (42 `RMtop200`, 2 `CDwinp`), 65 pending |
| gpu-medium | 3 days | 20 GPUs, 24 jobs | 0 running, 12 `RMtop200` pending |
| gpu-long | 6 days | 16 GPUs, 10 jobs | 3 running (`FFtop4000`, until ~15 Oct 15:30), 9 `RMtop200` pending |

All 77 pending `RMtop200` tasks carry `nice=10000` (Pavel's call on 13 Oct), so a B job submitted at
the default priority starts ahead of them. The running gpu-short tasks started at ~16:50 on 13 Oct
and hit their limits by ~17:00 on 14 Oct at the latest; most finish earlier. The other session's
queued jobs (`CDwinp` ×4, `RVtop200` ×2, `S3top75` ×3) take the first freed slots. So a gpu-short
B job starts on 14 Oct, and gpu-medium and gpu-long jobs can start as soon as the cluster schedules
them. Submitting B jobs at default priority delays `RMtop200` further, which is Pavel's decision.

**Snapshot disk.** One kept snapshot (LastParams + config) is 7.7 MB at N = 1000 (measured) and
scales with N²: ~2 MB at N = 500, ~31 MB at 2000, ~123 MB at 4000. Over 200k iterations at one
snapshot per 500 that is 0.8 / 3.1 / 12 / 49 GB per run. The plans keep every 500 at N ≤ 1000, every
1000 at N = 2000 (6 GB per run) and every 5000 at N = 4000 (5 GB per run; R² at the floor then
comes from the nearest snapshot after the crossing, or from the end-of-run estimator, which agreed
within 0.0021 on the flip-flop). The TENGEL scratch fileset holds 6.6 of 8 TiB, so about 1.4 TiB is
free; everything below needs at most ~250 GB.

**Read-out, for every item.** Each network is read where its noisy training loss
(`common.noisy_loss_series`, running median over 2,001 iterations) first reaches 1.05× its own
fitted floor (`common.noisy_floor`), with the usual exclusions. Never the noise-free loss, never a
fixed iteration as the headline. Active units come from the participation trace at the crossing;
held-out R² from the kept snapshot at the crossing (`heldout_r2.py`, 9 draws under the training
noise).

⚠️ **The noisy loss of a penalised run includes the penalty.** `TrainLosses.json` stores
`Σ λ_k · penalty_k` with the task term, so a rate-penalty run's floor is the floor of the objective
it trains on. That follows the rule as written, and `RMtop200`'s rate + sparsity arm will be read the
same way. Two cells with different λ are then read on different objectives. A cheap check, offline on
the kept snapshots: score the task loss alone under training noise and confirm the crossing moves by
less than one snapshot interval's worth of active-unit change.

---

## B2. The rate penalty alone on the standard flip-flop grid

**Question.** Decision 2 recommends the rate penalty alone as the main biological remedy. It has
never been run on the standard flip-flop network.

**What is already on disk.**
- **No rate-only flip-flop run on the standard network exists.** A scan of every flip-flop config on
  Della finds `lambda_frm > 0` only in `NBitFlipFlop_ff_both40k*` (the rate + sparsity pair,
  λ_frm 0.1, λ_rws 0.05). Spock no longer holds `NBitFlipFlop_std_pen` or `_penlong` (deleted in the
  14 Sept quota cleanup).
- **`RMtop200` contains no rate-only arm.** Its 71 cells (`~/c6_cells.txt`, submitted by
  `~/c6_topups.sh`) are the cells `f2_remedies_cache.discover` finds, and `classify` returns `None`
  for a rate-only config. It does contain the pair at all four sizes
  (`ff_both40k_c75000/EqType=h_N={500,1000,2000,4000}_arm=both`), so B2 must not re-run the pair.
- **Pre-standard rate-only runs exist on the Mac only** (`data/trained_RNNs/NBitFlipFlop_std_penlong/
  EqType=h_k=3_N={500,1000,2000}_pen=frm_iters=400000`, λ_frm 0.1, trainable bias in [−1, 1]; and
  `NBitFlipFlop_std_pen/..._N=500_pen=frm`, 150k). Of the four N = 2000 runs at 400k, one ended at
  NaN. They are a guide to what to expect (979 of 1000 active at N = 1000, about a quarter of them
  transient "burst" units) and cannot enter the paper.

**Cells.** 3-bit flip-flop (`task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3`), standard network
(`model=rnn_relu_standard`, `trainer=trainer_ptrack_freshbatch`, γ 0, bias fixed at 0), arm
`LAMBDA_FRM=0.1` with the default `frm_args` (cap_fr 0.3, g_top = g_bot = 3, mean aggregation), the
rate term of the pair already on disk, so rate alone and the pair differ in one setting.
λ_rws 0, no other remedy. N = 500, 1000, 2000, 4000; 3 seeds (`seed=random`); 200k iterations.

**Controls** (all verified on Della, 13 Oct):

| N | Control at 200k | Runs |
|---|---|---|
| 500 | `NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=500_pen=none_arm=none` | 6 |
| 1000 | `NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control` (kept snapshots 0-200k, same launcher family and code) and `ff_revive_c200000/..._N=1000_...` | 3 + 3 |
| 2000 | `NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=2000_pen=none_arm=none` | 3 |
| 4000 | `NBitFlipFlop_paper_grid_c200000/EqType=h_N=4000_arm=control` | 3, still training (`FFtop4000`, due ~15 Oct 15:30); parents at 75k are complete |

The comparison arm for decision 2, rate + sparsity at 200k, lands with `RMtop200`.

**Submission** (from `~/trainRNNbrain_c1ana`, after pulling the new launcher):

```
L=slurm/SilentReLU_remedy_grid_della.slurm; C=--chdir=$HOME/trainRNNbrain_c1top
D=$HOME/trainRNNbrain/data/trained_RNNs
TASK=ff LAMBDA_FRM=0.1 ITERS_OVERRIDE=200000 N=500  sbatch $C --export=ALL --constraint=nomig --time=16:00:00 --array=1-3 $L
TASK=ff LAMBDA_FRM=0.1 ITERS_OVERRIDE=200000 N=1000 sbatch $C --export=ALL --constraint=nomig --time=24:00:00 --array=1-3 $L
TASK=ff LAMBDA_FRM=0.1 ITERS_OVERRIDE=200000 N=2000 CKPT_EVERY=1000 sbatch $C --export=ALL --constraint=nomig --time=2-12:00:00 --array=1-3 $L
TASK=ff LAMBDA_FRM=0.1 ITERS_OVERRIDE=100000 N=4000 CKPT_EVERY=5000 sbatch $C --export=ALL --constraint=gpu80 --time=3-00:00:00 --array=1-3 $L
#   then the second segment, queued behind the first:
PARENTS="$D/NBitFlipFlop_remedy_grid/EqType=h_N=4000_arm=frm0.1/*/" TARGET=200000 KEEP=1 \
  sbatch $C --export=ALL --constraint=gpu80 --time=3-00:00:00 --dependency=afterok:<id> --array=1-3 \
  slurm/SilentReLU_ff_topup200k_della.slurm      # top-up keeps every 500 when KEEP=1; set
                                                 # ++trainer.checkpoint_every=5000 there by a CKPT knob
```

The top-up launcher hard-codes 500 for `KEEP=1`; at N = 4000 it needs the same `CKPT_EVERY` knob
(a one-line launcher edit), or `KEEP=0` and the R² estimator. The output root follows the
remedy-grid launcher's tag; naming it `c1_snapshots` would mix it with the claim-1 controls, so the
new launcher should tag `remedy_grid/EqType=h_N=<N>_arm=<arm>`.

**Cost.**

| N | Hours per run | GPU-h (3 runs) | Disk |
|---|---|---|---|
| 500 | 10 | 30 | 2 GB |
| 1000 | 14 | 43 | 9 GB |
| 2000 | 40 | 120 | 18 GB |
| 4000 | 125 (2 × 63) | 375 | 15 GB |
| **all** | | **~570** | ~45 GB |

Wall time: N ≤ 1000 done by ~15 Oct; N = 2000 by ~16 Oct; N = 4000 by ~20 Oct (two 63-hour segments
in gpu-medium plus queue).

**What answers the question.** Per N: active units at the floor (mean ± SD), the multiplier against
the control read at its own floor, held-out R² at the floor, the number of runs that reach the floor,
and the share of gradient updates skipped. Side by side with rate + sparsity from `RMtop200`.

**Thresholds, set now.**
- *Supports decision 2* if at every N at least 2 of 3 runs reach their floor with fewer than 5% of
  updates skipped and a finite loss; the multiplier at the floor is at least 1.5 (rescaling, the
  weakest existing biological remedy, sits at 1.57 at N = 1000); and held-out R² at the floor is
  within 0.02 of the control's.
- *Falsifies decision 2* (rate alone cannot be the main biological remedy) if at any N ≥ 1000 fewer
  than 2 of 3 runs train stably, or the R² cost at the floor exceeds 0.02, or the multiplier falls
  below synaptic noise's at the same N read the same way (to be read from `RMtop200` at the floor;
  at 75k it was 2.08 at N = 1000 and 1.65 at N = 4000).
- A count near N (as on the pre-standard network, 979 of 1000) settles nothing about function. A3
  (ablate the recruited units) has to show that they carry R², or the count is a penalty met by
  transients.

**Minimal:** N = 500, 1000, 2000 at λ_frm 0.1, 3 seeds, 200k. ~195 GPU-h, done by ~16 Oct. Covers
the size at which the pre-standard network went unstable (N = 2000).
**Full:** add N = 4000. ~570 GPU-h, done by ~20 Oct.

---

## B3. Combination: rate penalty (reduced) × synaptic noise × rescaling

**Question.** Decision 3 allows the paper to say "several sources" only if a combination beats the
rate penalty alone. No two working rules have ever been combined.

**Why the rate penalty must be reduced.** At λ_frm 0.1 the rate penalty alone held 979 of 1000 units
on the pre-standard flip-flop, so no other factor has room to add anything; a factorial run at that
strength would read as "redundant" by ceiling. The rate factor needs a strength that leaves headroom,
picked by a rule set now (stage 0).

**Stage 0: pick λ.** Remedy-grid launcher, N = 1000, `LAMBDA_FRM` = 0.01 and 0.03, 3 seeds, 200k,
kept snapshots every 500. Rule: take the rung whose multiplier at the floor is closest to 2 (synaptic
noise's level) while its active fraction stays at or below 0.8. If both rungs exceed 0.8, add 0.003;
if both stay below 1.3×, use 0.03 anyway and say so. The chosen rung becomes cell "rate only" of the
factorial and a dose rung for B2.

**Stage 1: the 2 × 2 × 2 at N = 1000.** Factors: rate (0 or the stage-0 λ), synaptic noise (σ_w 0 or
1.0), rescaling (off or target 10). 3 seeds, 200k, one job each, kept snapshots every 500.

| Cell | Source | New runs |
|---|---|---|
| none | reuse `NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control` (same launcher family, same code, snapshots 0-200k) | 0 |
| rate | stage 0 | 0 |
| noise, rescale, rate+noise, rate+rescale, noise+rescale, all three | `SIGMA_W` / `RESCALE` / `LAMBDA_FRM` combinations | 6 × 3 = 18 |

Noise alone and rescale alone are re-run fresh rather than borrowed from the 200k top-ups in
`RMtop200`: those ran their first 40k on older commits and carry two rescale-state resets, and a
factorial needs all cells on one code path. The `RMtop200` cells then serve as a replication check.

```
TASK=ff N=1000 ITERS_OVERRIDE=200000 SIGMA_W=1.0                   sbatch $C --export=ALL --constraint=nomig --time=24:00:00 --array=1-3 $L
TASK=ff N=1000 ITERS_OVERRIDE=200000 RESCALE=1                     ...
TASK=ff N=1000 ITERS_OVERRIDE=200000 LAMBDA_FRM=<λ> SIGMA_W=1.0    ...
TASK=ff N=1000 ITERS_OVERRIDE=200000 LAMBDA_FRM=<λ> RESCALE=1      ...
TASK=ff N=1000 ITERS_OVERRIDE=200000 SIGMA_W=1.0 RESCALE=1         ...
TASK=ff N=1000 ITERS_OVERRIDE=200000 LAMBDA_FRM=<λ> SIGMA_W=1.0 RESCALE=1 ...
```

At 0.30 s/it a 200k run takes ~17 h, inside the 24-hour gpu-short limit on a full GPU. A run that
times out can be continued from its last kept snapshot, which holds LastParams and config.

**Stage 2: all three at N = 4000.** All three (3 seeds) and rate-reduced alone (3 seeds; neither the
full-strength B2 cell nor anything on disk is at the reduced λ). Both in two 100k segments,
`CKPT_EVERY=5000`. The all-three run at ~2.6 s/it needs 72 h per 100k, the whole gpu-medium limit,
so its segments go to gpu-long with `--time=4-00:00:00`. References at N = 4000: control
(`paper_grid_c200000`), synaptic noise (`size_series_c200000/..._N=4000_arm=synnoise_sw1.0`) and
rescale (`size_series_c200000/..._N=4000_arm=rescale_tgt10`), both in `RMtop200`, both pending.

**Cost.**

| Stage | Runs | GPU-h | Wall |
|---|---|---|---|
| 0, λ ladder | 6 | ~90 | ~15 h |
| 1, factorial N = 1000 | 18 | ~300 | ~17 h after stage 0 |
| 2, N = 4000 | 6 | ~810 (all three 3 × 144 h, rate alone 3 × 125 h) | ~6 days after stage 0 |

Disk: stage 0 and 1, 24 runs × 3.1 GB ≈ 75 GB; stage 2, ~30 GB.

**What answers the question.** For each cell at the floor: active units, held-out R², participation
ratio over active units, runs reaching the floor, skipped updates. Main effects and interactions are
estimated on the logit of the active fraction, so a cell near N does not read as sub-additive by
ceiling alone. At N = 4000, all three against each single remedy.

**Thresholds, set now.**
- *Supports "several sources"* if the all-three cell (or any pair) beats its best single component by
  more than 3 SD of that component on active units at the floor, with an R² cost no more than 0.01
  worse than that component's, at N = 1000; and at N = 4000 the all-three cell beats every single
  remedy read the same way. A pair or triple that matches the best single remedy on units but trains
  stably where that remedy does not (fewer than 2 of 3 runs) also counts, on stability.
- *Falsifies "several sources"* if every pair and the triple sit within 3 SD of their best single
  component on active units and on participation ratio, with no stability gain. The paper then says
  "a drive of its own", as decision 3 already allows.

**Minimal:** stages 0 and 1. ~390 GPU-h, done by ~16 Oct.
**Full:** add stage 2. ~1,200 GPU-h, done by ~22 Oct. Stage 2 is worth running only if stage 1 shows
an additive effect; if stage 1 falsifies the claim, the N = 4000 triple cannot rescue it.

---

## B4. Plasticity test (the Dohare link)

**Question.** When a trained network learns a new task, do its silent units come back?

**The new task: `NBitFlipFlopHyper` at k = 3** (`trainRNNbrain/tasks/TaskNBitFlipFlopHyper.py`,
`configs/task/NBitFlipFlopHyper.yaml`). Its inputs are exactly the 3-bit flip-flop's (3 pulse-train
channels, T = 300, μ = 4 pulses per bit, 10-step pulses); its targets are all 7 products of the bit
states (3 singles, 3 pairs, the 3-way parity). Two reasons:
- **Same input layer.** The source network's input weights and recurrent weights carry over
  unchanged; only the read-out is new. A task with other input channels (CDDM: 6, DMTS_d5: 4) would
  need fresh input weights, and fresh input weights hand every silent unit a new drive. Revival would
  then measure the new weights, not the network's plasticity.
- **It demands more units.** On the pre-standard network the hyper task recruited 1.21× the plain
  task's count at k = 2 and 1.75× at k = 4 (N = 1000, 150k; trajectory 2026-09-14). k = 3 should fall
  between, so a network that lost most of its units has a reason to recruit.

**Which networks are transferred** (3 seeds each; one transfer per source network):

| Source | N = 1000 | N = 4000 |
|---|---|---|
| Control, young | the claim-1 networks' kept snapshot at their floor crossing (27k, 39.5k, 69.5k): `c1_snapshots/NBitFlipFlop_c1_snapshots/EqType=h_N=1000_arm=control/s<seed>/it<crossing>/` | `NBitFlipFlop_paper_grid_c75000/EqType=h_N=4000_arm=control` (75k end; no snapshots exist at N = 4000, so "young" is labelled by iteration and by its distance past that run's floor) |
| Control, old | the same three networks at 200k: `NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control` | `NBitFlipFlop_paper_grid_c200000/EqType=h_N=4000_arm=control` (~15 Oct 15:30) |
| Rate penalty (full only) | B2 N = 1000: floor snapshot and 200k | B2 N = 4000 at 200k (~20 Oct) |
| Synaptic noise (full only) | `synnoise_c200000/..._N=1000_sw=1.0` (in `RMtop200` with kept snapshots, pending): floor snapshot and 200k | `size_series_c200000/..._N=4000_arm=synnoise_sw1.0` (in `RMtop200`, pending) |
| Fresh | `NBitFlipFlopHyper`, untrained standard network, remedy-grid launcher with `TASK=hyper` | same |

Using the same three N = 1000 networks for "young" and "old" pairs each young/old comparison within
one network.

**How the warm start works, and why it needs code.**
- The transfer is built from the **source's own saved config**, never from defaults: the parameter
  file carries no architecture. The topup launcher already does this
  (`--config-path <source dir> --config-name <its config>`), and a kept snapshot folder holds a config
  and LastParams, so a floor snapshot works as a source as-is. The task is swapped by three overrides
  on the saved config: `task.taskname=NBitFlipFlopHyper`,
  `task._target_=trainRNNbrain.tasks.TaskNBitFlipFlopHyper.TaskNBitFlipFlopHyper`, `task.n_outputs=7`.
  The model and trainer blocks carry over by construction, so a remedy network keeps its remedy on the
  new task. That choice is an assumption: "a network keeps its own learning rule; the task changes".
  The other reading (remedy switched off on the new task) is one extra override per arm and is left
  out of both plans.
- **Smoke test (done, 13 Oct):** this path, from a 30-iteration source, runs as far as the weight
  copy and then stops with `ValueError: warm start: 'W_out' is (3, 60) in the checkpoint but (7, 60)
  in this model`. A fresh `NBitFlipFlopHyper` run at k = 3 trains at 7739a60 with no change.
- **Code changes needed** (one commit, then a new clean checkout, e.g. `~/trainRNNbrain_b4`; none of
  them changes a non-transfer run):
  1. `run_experiment.py`: `paths.transfer` (default false) and `paths.init_skip` (default `[]`),
     added to `configs/paths/*.yaml` so the config stays the single source of truth. Parameters named
     in `init_skip` keep this run's own seeded initialisation and skip the shape check; every other
     parameter is still copied and still checked. With `transfer: true`, an `auto` iteration offset
     resolves to 0, `join_parent_trace` is skipped (the old task's trace must not be spliced onto the
     new one), and `init_adam: true` raises (the old read-out's moments do not fit). B4 uses
     `++paths.transfer=true '++paths.init_skip=[W_out]' ++paths.init_adam=false`.
  2. `common.warm_start_chain`: stop at a run whose config has `paths.transfer: true`. Without it,
     `noisy_loss_series` joins the source task's loss onto the new task's and fits one floor through
     both, and for a snapshot source it returns nothing at all (a snapshot folder has no
     `TrainLosses.json`).
  3. Tests (`tests/`): a transfer copies `W_rec`, `W_inp` and `y_init` exactly and leaves `W_out`
     equal to a fresh model of the same seed; a non-transfer warm start with a mismatched shape still
     raises; the chain of a transfer run has length 1; `tests/test_periodic_checkpoint.py` still
     passes. Then a local smoke run before any queue submission.
  4. `slurm/SilentReLU_transfer_della.slurm`, adapted from the topup launcher: `PARENTS` glob over
     run folders or snapshot folders, the overrides above, `trainer.max_iter`, `seed=random`,
     `KEEP`/`CKPT_EVERY`, tag `b4_transfer/<source cell>_age=<young|old>`. Fresh runs use the
     remedy-grid launcher with a `TASK=hyper` case (`task=NBitFlipFlopHyper task.n_inputs=3
     task.n_outputs=7`, prefix `NBitFlipFlopHyper`).
  Estimated effort: half a day including the tests. `heldout_r2.py` builds its batch from the saved
  task config generically, so it scores the hyper task without change.

**What is read, and when.**
- *At transfer (iteration 0 of the new task):* per-unit participation from the trace's
  iteration-0 probe, noise-free, 5%-of-q95 rule. S₀ = the units silent then. The network has no output
  feedback, so its activity does not depend on the new read-out, and S₀ is the source's silent set on
  the shared inputs.
- *At the floor of the new task:* where the transfer run's own noisy loss first reaches 1.05× its own
  fitted floor (the chain stops at the transfer, per code change 2). Active set A* from the trace;
  held-out R² from the kept snapshot (N = 1000) or the end-of-run estimator (N = 4000).
- *Reported per network:* revived = |S₀ ∩ A*| out of |S₀|; newly silenced = active at transfer but
  silent at the floor; active units at the floor against the fresh hyper network at its own floor;
  R² at the floor against fresh.
- *Zero-cost baseline, no new training:* on the old task, how many units silent at the young
  N = 1000 networks' floor become active again by 200k (the claim-1 chains' traces, every 100
  iterations). That number says how much revival continued training alone gives.
- Gradient reach (`gradient_participation.py`) at transfer, per unit of S₀, predicts which units can
  come back. A2 found that on the flip-flop the gradient still reaches the incoming weights of
  1.5-4.5× as many units as are active, so revival is possible in principle, and the test can fail.

**Gate, checked first.** The test needs the new task to want more units than the source has. If the
fresh hyper network's active count at its floor is not above the old control's count at transfer by
more than 3 fresh-network SDs, the test is uninformative at that N and is reported as such (no harder
task with the same inputs exists in `tasks/`; the Walsh variant has 6 inputs).

**Thresholds, set now** (the claim: a trained network's silent units do not come back when it learns
a new task).
- *Supports* if, for old controls at both N, fewer than 10% of S₀ revive in every seed, and the
  transferred network ends below the fresh network's active count by more than 3 fresh SDs.
- *Falsifies* if 30% or more of S₀ revive in at least 2 of 3 seeds, or the transferred network's
  active count at the floor comes within 1 fresh SD of the fresh network's. Results between the two
  bars are reported as numbers, without a verdict.
- R² is read separately: a transferred network that matches fresh R² with fewer units shows lost
  recruitment without lost performance, which is a different claim from Dohare's loss of plasticity
  and has to be worded as such.

**Cost** (hyper task speed taken as the plain task's; 7 output channels add nothing measurable):

| Version | N = 1000 runs | N = 4000 runs | GPU-h at 200k | GPU-h at 100k + conditional top-up |
|---|---|---|---|---|
| Minimal: control young, control old, fresh | 9 | 9 | ~125 + ~1,050 = **~1,180** | ~60 + ~525 = **~590**, plus top-ups |
| Full: + rate (young, old at 1000; old at 4000), + noise (same) | 21 | 15 | ~300 + ~1,800 = **~2,100** | ~150 + ~900 = **~1,050**, plus top-ups |

"100k + conditional top-up": every run gets 100k; a run is continued to 200k (topup launcher,
`TARGET=200000`) only if its noisy loss has not reached 1.05× a floor fit, or its last-10% median
still exceeds 1.5× the fit. The flip-flop needed 200k from scratch, but a transfer starts from a
network that already holds the 3 bits, so many transfer runs may settle earlier. The fresh hyper runs
are the likeliest to need the second 100k.

Wall time. Minimal: code and tests on 14 Oct; N = 1000 transfers and fresh runs ~15 h; N = 4000
starts when `FFtop4000` lands (~15 Oct 15:30), one 63-hour segment (or two), so done ~18-21 Oct.
Full: the rate sources wait for B2 at N = 4000 (~20 Oct) and the noise sources for `RMtop200`
(pending, deprioritised), so ~26-28 Oct.
Disk: N = 1000 kept every 500, ~3 GB per run; N = 4000 every 5000, ~5 GB per run; minimal ~75 GB.

**Minimal:** N = 1000 and 4000; control young, control old, fresh; one new task; 3 seeds (decision
5's form). ~590-1,180 GPU-h. If N = 4000 has to be cut further, drop the young N = 4000 control
first: it is a 75k state, not a floor state.
**Full:** add rate-penalty and synaptic-noise sources. ~1,050-2,100 GPU-h.

---

## Summary and recommendation

| Item | Minimal | GPU-h | Done by | Full | GPU-h | Done by | Code change |
|---|---|---|---|---|---|---|---|
| B2 | N = 500-2000, λ_frm 0.1 | ~195 | ~16 Oct | + N = 4000 | ~570 | ~20 Oct | none (new launcher) |
| B3 | λ ladder + 2×2×2 at N = 1000 | ~390 | ~16 Oct | + all three and rate-reduced at N = 4000 | ~1,200 | ~22 Oct | none (same launcher) |
| B4 | control young/old + fresh at N = 1000, 4000 | ~590-1,180 | ~18-21 Oct | + rate and noise sources | ~1,050-2,100 | ~26-28 Oct | yes: transfer warm start, chain stop, tests, launcher |

Recommendation: submit **B2 full** and **B3 minimal** together (~960 GPU-h). They share one launcher
and one code commit, B3's stage 0 doubles as B2's dose ladder, and every other remedy in Figure 3 has
an N = 4000 cell, so a missing rate-only cell there would show. Run B3's N = 4000 stage only if the
N = 1000 factorial shows an additive effect. For **B4**, make the code change first (it is the only
item that needs one) and run the minimal version with the 100k-then-conditional-top-up rule; add the
remedy sources once B2 and `RMtop200` land.
