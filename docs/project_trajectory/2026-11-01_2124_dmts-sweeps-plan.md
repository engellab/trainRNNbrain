## 2026-11-01 21:24 — DMTS setting sweeps: why DMTS had none, the design, the cost, and the pass/fail rule (set before any run)

Work package dmts-sweeps (dead-relu-paper #127). trainRNNbrain branch `wp/dmts-sweeps`, launcher
`slurm/SilentReLU_dmts_settings_della.slurm` (commit 8f81e4b). Results 3 states that standard
training settings change the number of active units but that the number still decreases after the
floor crossing, in all 116 CDDM and 3BFF networks with a crossing (entry 2026-11-01_1000). The
author asked why DMTS (delayed match-to-sample) was not tested too.

### Why DMTS had no setting sweeps

1. **A scoping decision, not a result.** On 12 Oct the status board listed "Which tasks?" as an
   open question (commit 0788f85) with the recommendation "two main tasks. CDDM carries the knobs,
   the gradient analysis and tanh; the 3-bit flip-flop carries the remedies. DMTS (short delay)
   appears only as a third series on the scaling curve." The cleanup of 15 Oct (commit 6eb901c)
   moved that recommendation, unchanged, into "Decisions taken". The notebook records no reason
   beyond the recommendation, and no entry records Pavel deciding it.
2. **DMTS training is hard to read out.** The notebook gives two reasons a DMTS comparison is more
   fragile than a CDDM one. First, DMTS networks hold a constant output for 200 to 155,000
   iterations before the loss drops (entry 2026-10-12_2035), so the floor crossing varies widely
   between networks. Second, after the drop the loss bursts back toward chance, and the floor
   guards then exclude networks (entry 2026-10-15_2218: at the 7τ delay, 1 of 3 solved N = 1000
   networks has a read-out). The 5τ delay was introduced because at 7τ 9 of 28 unpenalised networks
   never solved the task (`configs/task/DMTS_d5.yaml`).
3. **Cost was never given as a reason.** DMTS trains at the same speed as CDDM (below).

### No DMTS setting-sweep data exist

Listed on 1 Nov. Della `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs`: `DMTS_c1_snapshots`
(3 default networks, N = 1000), `DMTS_d7_pen` (7τ, rate and sparsity penalties, N = 500-4000),
`DMTS_n100`, `DMTS_paper_grid` and `DMTS_paper_grid_d5` (size series; duplication and read-out
dropout arms). Mac `data/trained_RNNs` holds the same plus `DMTS_v2_delay` (a delay ladder of
8τ, 10τ and 12τ, copied from Spock). None varies a hyperparameter or architectural setting of the
default network. Spock had no live session and could not be listed directly.

### What the existing DMTS default networks show (5τ delay, 150,000 iterations, `common.read_run`)

| N | floor crossing (iterations) | active units at the crossing → at 150,000 |
|---|---|---|
| 1000 | 18,599; 75,721; none (loss ends below its floor) | 422 → 155; 218 → 187 |
| 500 | 18,205; 62,088; 109,348 | 211 → 140; 180 → 129; 182 → 141 |
| 2000 | 51,598; 73,448; none (stall) | 386 → 309; 465 → 192 |

Two consequences for the design. A budget of 75,000 iterations, as in five of the CDDM sweeps,
would leave about a third of DMTS networks without a crossing, so every DMTS arm runs 150,000, the
DMTS default budget. The active count at 75,000 iterations is read before the crossing in some
DMTS networks, so the comparison at 75,000 iterations is reported only alongside the comparison at
the crossing.

### The full matched design and its cost (not submitted)

The CDDM setting sweeps at N = 1000 (read from the data, networks per value): weight decay 0, 1e-5,
1e-4 (3 each); recurrent noise σ_rec 0, 0.01, 0.1 (5); metabolic penalty 0.01, 0.1, 1, 10 (3);
initial spectral radius 0.5, 2.0 (2); recurrent-weight magnitude penalty 0.05, 0.2 (2); leaky ReLU,
softplus β = 25, shifted sigmoid (3); sparsity penalty 0.05 (5); initial input-weight scale 0.5, 2,
5, 20 (3, as on 3BFF). Repeating all of them on DMTS at 150,000 iterations, with a 5-network
default, needs 75 networks.

Cost per network: DMTS at N = 1000 took 6.43-6.56 h of one A100 80 GB per 150,000 iterations in all
six jobs timed (14830790_4-6, 15422708_1-3), 0.156 s per iteration. The CDDM sweeps ran at the same
speed: weight decay 73.2 GPU-h for 9 networks of 200,000 iterations (14839186, 14848710); spectral
radius and recurrent-weight penalty 31.4 GPU-h for 10 of 75,000 (14919348); CDDM input-weight scale
26.2 GPU-h for 12 of 50,000 (15502206-09), 0.15 s per iteration in each. So the full design costs
75 × 6.5 ≈ **490 GPU-h**, above the 200 GPU-h a package may submit without the editor's go-ahead.

### The design submitted: one value per setting, the one most likely to stop the decrease

For each of the eight settings, the value that gave the most active units at 75,000 iterations on
CDDM (on 3BFF for the input-weight scale), from `data/fig_paper_F2_fails_cache.pkl`. If any setting
stops the decrease on DMTS, the value that keeps the most units is the likeliest to do it, so this
design tests the claim where it is most likely to fail. The sparsity penalty has only one value.

| arm | override | CDDM mean active at 75k: value vs default |
|---|---|---|
| default | none | 337 (3BFF 275) |
| wd0 | `trainer.weight_decay=0` | 453 vs 337 |
| sigrec0.1 | `model.sigma_rec=0.1` | 397 vs 327 |
| met1 | `trainer.lambda_met=1.0` | 375 vs 337 |
| sr0.5 | `model.spectral_rad=0.5` | 361 vs 340 |
| rwm0.2 | `+trainer.lambda_rwm=0.2 +trainer.rwm_args={}` | 448 vs 340 |
| softplus25 | `model.activation_args.name=softplus +model.activation_args.beta=25.0` | 375 vs 337 |
| rws0.05 | `trainer.lambda_rws=0.05` | 265 vs 325 |
| winp20 | `model.input_row_norm=20` | 3BFF 477 vs 275 |

Every arm: 3 networks, N = 1000, task `DMTS_d5`, `rnn_relu_standard`, `trainer_ptrack_freshbatch`,
150,000 iterations, a kept snapshot every 1,000 iterations, seed drawn at random and recorded.
27 networks × 6.5 GPU-h ≈ **176 GPU-h** (`--constraint=gpu80`, where the six timed DMTS jobs ran;
one DMTS job on a 40 GB node took 11.4 h).

Softplus is set on `rnn_relu_standard`. `configs/model/rnn_softplus_beta25_standard.yaml` (and the
leaky-ReLU and sigmoid files) have no `init_draws` key, so they fall back to the code default
`"shared"` (the correlated initial draw), while `rnn_relu_standard` draws `"independent"`. A new run
made with those files would differ from the default in the activation and the initial draw.

**Local check of the launcher** (pass condition set before the diff: each arm's saved config differs
from the default arm's only in its own keys, apart from seed, tag and paths; every config has
`init_draws: independent`, `bias_range [0, 0]`, `same_batch: false`, `checkpoint_keep: true`, DMTS
with T = 190). The launcher ran with `module`/`conda`/`srun` stubbed at N = 50 for 30 iterations from
a clean worktree at 8f81e4b, one network per arm: 9 of 9 pass. In the trained weights the input
rows of `winp20` have median norm 19.98 (0.22-0.28 in the other arms), the spectral radius of
`sr0.5` is 0.53 (1.09-1.26), the saved activation of `softplus25` is softplus with β = 25, and every
loss is finite.

### Read-out and pass/fail rule (set now, before any network trains)

Read-out, as for CDDM and 3BFF in Results 3. Each network is read with `common.read_run` (1.05× its
own noisy-loss floor; the DMTS late-level floor fitted after the escape; the standard exclusions,
counted). A network that does not solve the task (held-out R² < 0.8 at the end of training) is
excluded and counted. Active units come from the participation trace (p ≥ 0.05 × q95(p)).
Held-out R² comes from `heldout_r2.py` on the kept snapshot nearest the crossing and at 75,000
iterations.

**Pass/fail for "no standard setting stops the decrease", on DMTS.** Compare, for every network
outside the rate penalty with a floor crossing, the active units at the probe nearest its crossing
with the active units at the probe nearest the end of training (150,000 iterations). Two variants
are computed as well: the probe at or before the crossing, and the median over the last 2% of
training (as in entry 2026-11-01_1000).
- **Pass:** every such network has fewer active units at the end than at its crossing, in the
  primary comparison.
- **Fail:** at least one network has as many or more at the end. The network, its arm and all
  three variants are listed, and Results 3 cannot extend its claim to DMTS without that exception.
- **Untested:** an arm with no network that has a crossing. It is reported as untested and counts
  neither for nor against.
- The claim is extended to DMTS only for the arms tested. The values not run are named as not run.

Descriptive, not part of the rule: each arm's mean active units and R² against the default at the
crossing and at 75,000 iterations (the "settings change the number" half of Results 3), and the
change per 10,000 iterations over the last third of training.

### Submission

Recorded in the entry that follows this one (command, commit, job ids).
