# Project trajectory

A running, reproducibility-first log of experiments. Each entry records *why* the experiment
was run and *exactly how* it was run (commit, config, parameters, cluster, submission command,
output layout) so it can be regenerated after all working-memory context is lost.

---

## 2026-06-26 — Silent-ReLU baseline sweep (CDDM, ReLU-Dale)

### Purpose

Establish a baseline showing that **unmodified ReLU-Dale RNNs trained on CDDM leave most units
silent** (dead ReLUs that do not contribute to the computation), and that **two activity
penalties rescue those dead units and prevent the silent-ReLU regime**:

- `lambda_rws` — recurrent-weight **sparsity** penalty (effective in-degree per unit).
- `lambda_frm` — **firing-rate magnitude** penalty (drives time-aggregated rate toward a cap).

The `lambda_rws = 0, lambda_frm = 0` cell is the *unmodified* control (expected: many silent
units). The other three cells (rws-only, frm-only, both) test whether each penalty — alone and
together — increases the number of participating units. The key per-network readout is
`participation.png` (per-unit `std(activity) + 0.9-quantile(|activity|)`); a left-heavy
distribution / large low-participation mass = many silent units.

The sweep also crosses the RNN **equation type** (`h` vs `s`) and **network size** (`N`) to
check the effect is not specific to one dynamics convention or scale.

### Grid — 120 jobs

`2 equation types × 2 lambda_rws × 2 lambda_frm × 3 sizes × 5 networks = 120`

| Axis | Config key | Values |
|---|---|---|
| Equation type | `model.equation_type` | `h`, `s` |
| Recurrent-weight sparsity | `trainer.lambda_rws` | `0`, `0.05` |
| Firing-rate magnitude | `trainer.lambda_frm` | `0`, `0.2` |
| Network size | `model.N` | `100`, `500`, `1000` |
| Networks per condition | `seed="random"`, 5 array reps | 5 |

`h` vs `s` (`trainRNNbrain/rnns/RNN_torch.py`): `h` = hidden/pre-activation state, output read
from `activation(x)`; `s` = rate/synaptic state, activation applied inside the recurrent drive.

### Fixed configuration

- **Model**: `configs/model/rnn_relu_Dale.yaml` — ReLU (`slope=1.0`), Dale-constrained,
  `dt=1`, `tau=10`, `exc2inhR=4.0`, `gamma=0.1`, `connectivity_density_rec=1.0`,
  `spectral_rad=1.2`, `sigma_rec=sigma_inp=0.05`, `sigma_out=0.03`, `bias_range=[0,0]`.
- **Task**: `configs/task/CDDM.yaml` — Mante-style context-dependent decision making,
  `T=300`, `n_inputs=6`, `n_outputs=2`, 15 coherences, loss masked on `(0,100)` and `(200,300)`.
- **Trainer**: `configs/trainer/trainer.yaml` (`BaseTrainer`) — `max_iter=30000`, `lr=0.001`
  (scaled at runtime by `(100/N)**0.333`), `weight_decay=1e-6`, `max_grad_norm=50`,
  `same_batch=True`, `monitor=True`.
- **All other penalties = 0** (`lambda_orth, lambda_rwm, lambda_owm, lambda_iwm, lambda_hm,
  lambda_met, lambda_tv, lambda_effdim, …`) — only `lambda_rws` and `lambda_frm` vary, so the
  comparison is clean.
- **Penalty args** at config defaults: `frm_args` = `{cap_fr:0.3, tau:0.1, g_top:3.0, g_bot:3.0,
  alpha:1.0, beta:1.0, aggregation:mean, tau_n:1.0}`; `rws_args` = `{tg_deg:20}`.
  This matches the regime of the existing GT (ground-truth) networks, so the unpenalized
  baseline cannot be dismissed as merely under-trained or differently-tuned.

### Code version (pinned)

- **Commit `4a031e7`** — `PerformanceAnalyzer: add task-unaware plot_unit_trial_traces`.
  Folder hash = first 6 chars = **`4a031e`**.
- This commit includes `c906118` (`plot_participation`: NaN/Inf guard for diverged networks),
  which matters here because unpenalized ReLU nets can diverge.
- On Spock the working copy is a **detached HEAD at `4a031e7`**; `trainRNNbrain` is an editable
  install (`pip install -e .`) pointing at `/usr/people/pt1290/trainRNNbrain`, so torch-env runs
  exactly this code. GitHub `origin/main` (`engellab/trainRNNbrain`) is also at `4a031e7`.
- A Spock-local uncommitted hotfix to `plot_participation` (an earlier copy of the same NaN
  guard) was `git stash`-ed before checkout; it is superseded by the committed guard.

### Compute environment (Spock / PNI)

- **Cluster**: Spock (`spockmk2-*` nodes), SLURM, partition `all` (A100-40G and L40S-46G GPUs),
  `TIMELIMIT=infinite`. `$HOME = /usr/people/pt1290`.
- **Env activation**: `module load anacondapy/2024.02 && conda activate torch-env`
  (Python 3.12.13, torch 2.7.0+cu126, CUDA available). NOT the Della stack — Spock uses a
  different home path and module set, so `paths_DELLA.yaml` does not apply here.
- **Output root** (`paths.save_to`, set via CLI override, not a committed config):
  `/usr/people/pt1290/trainRNNbrain/data/trained_RNNs`.

### Output layout

```
<save_to>/CDDM_4a031e/EqType=<h|s>_N=<100|500|1000>_LmbdRWS=<0|0.05>_LmbdFR=<0|0.2>/
    <r2score>_CDDM_relu;N=<N>;L=<lr>;MI=30000;...;Lfrm=<...>;Lrws=<...>/
        <r2>_config.yaml          # full resolved Hydra config (exact reproduction record)
        <r2>_BestParams_CDDM.json # trained weights (best val) — the network itself
        <r2>_LastParams_CDDM.json
        <r2>_LossBreakdown.{json,png}, <r2>_Grads{Raw,Scaled}.{json,png}, <r2>_TrainLoss.png
        participation.png         # <-- primary silent-unit readout
        sorted_matrices.png, avg_responses.png, intercluster_connectivity_matrices.png,
        random_trials.png
```

The leaf `EqType=...` folder is produced by `experiment_tag="4a031e/EqType=..."` plus
`trainer.trainer_tag=""` (the empty trainer tag suppresses the default `_BaseTrainer` suffix).
The 5 reps of a condition land in 5 distinct per-net subfolders, keyed by r² score.

> Animations (`animated_trajectories.mp4`, `animated_selectivity.mp4`) are **skipped** on Spock —
> torch-env has no `ffmpeg`, and `DataSaver.save_animation` degrades gracefully (warns, no crash).
> All scientific outputs are saved. To enable animations: `conda install -c conda-forge ffmpeg -n torch-env`.

### Submission

Script: [`slurm/SilentReLU_ReluDale_sweep.slurm`](../slurm/SilentReLU_ReluDale_sweep.slurm)
(self-documenting; mixed-radix index decode of `SLURM_ARRAY_TASK_ID` → the 5 grid axes).

```bash
# on Spock, repo at detached HEAD 4a031e7:
ssh spock
cd ~/trainRNNbrain
sbatch slurm/SilentReLU_ReluDale_sweep.slurm        # array 1-120 (no throttle), 12h, 1 GPU, 16G each
# smoke test of one cell (50 iters, throwaway dir):
MAXITER=50 SAVE_TO=~/trainRNNbrain/data/_smoketest sbatch --array=1-1 slurm/SilentReLU_ReluDale_sweep.slurm
```

Per-job command (one array task):

```bash
srun python trainRNNbrain/training/run_experiment.py \
  seed="random" \
  model=rnn_relu_Dale model.equation_type=<h|s> model.N=<100|500|1000> \
  trainer.trainer_tag="''" trainer.max_iter=30000 \
  trainer.lambda_rws=<0|0.05> trainer.lambda_frm=<0|0.2> \
  paths.save_to=/usr/people/pt1290/trainRNNbrain/data/trained_RNNs \
  experiment_tag="\"4a031e/EqType=<EQ>_N=<N>_LmbdRWS=<RWS>_LmbdFR=<FRM>\""
```

(`experiment_tag` is wrapped in literal double quotes so Hydra parses the `=` chars in the value
as a single quoted string.)

### Run record

- Submitted **2026-06-26 ~12:50 EDT**. SLURM array job **`5078392`** (smoke test: `5078391`,
  COMPLETED, validated full pipeline end-to-end including folder layout and graceful animation skip).
- Initially submitted at `%24` concurrency, then throttle lifted live
  (`scontrol update jobid=5078392 arraytaskthrottle=0`) — tasks now run as GPUs free,
  bounded only by Spock's shared GPU availability and fair-share priority.
- Logs: `/usr/people/pt1290/trainRNNbrain/log/SilentReLU.5078392_<task>.{out,err}`.

### Results — silent units per condition

A unit is **silent** if its peak firing rate over the noise-free CDDM validation batch is
`< 0.01` (the `dead_abs` criterion; a scale-free `< 5%`-of-p95 criterion agrees throughout).
Computed for all 120 nets by [`count_silent_units.py`](../trainRNNbrain/experiments_and_analysis/count_silent_units.py) →
`data/trained_RNNs/CDDM_4a031e/silent_units_per_condition.csv`, plotted by
[`plot_silent_units_per_condition.py`](../trainRNNbrain/experiments_and_analysis/plot_silent_units_per_condition.py).

![Silent units per condition](../img/internal_figures/silent_units_per_condition.png)

Mean silent-unit count (and % of N) over 5 nets/condition:

| eq | N | none (Lrws=0,Lfr=0) | rws only (Lrws=0.05) | fr only (Lfr=0.2) | both |
|----|----|----|----|----|----|
| h | 100  | 0 (0%)    | 15 (15%)  | 0 | 0 |
| h | 500  | 118 (24%) | 210 (42%) | 0 | 0 |
| h | 1000 | 471 (47%) | 535 (54%) | 0 | 0 |
| s | 100  | 0.2 (0%)  | 20 (20%)  | 0 | 0 |
| s | 500  | 187 (38%) | 227 (45%) | 0 | 0 |
| s | 1000 | 543 (54%) | 555 (56%) | 0 | 0 |

**Findings:**

1. **The firing-rate-magnitude penalty (`lambda_frm=0.2`) eliminates silent units entirely** —
   exactly 0 in *every* condition (both equation types, all N, with or without `lambda_rws`).
   This is the headline result: `lambda_frm` rescues/prevents dead ReLUs.
2. **Unpenalized baselines have a severe, size-scaling silent-unit problem** — negligible at
   N=100 but ~24–38% at N=500 and ~47–54% at N=1000. The dead-ReLU pathology grows with N,
   confirming the motivation for the experiment.
3. **The sparsity penalty alone (`lambda_rws=0.05`, `lambda_frm=0`) does NOT rescue — it makes
   it slightly worse** (e.g. h/N=1000: 54% vs 47% baseline; s/N=500: 45% vs 38%). As a
   recurrent-weight sparsifier it pushes *more* units toward inactivity. So the original
   "both penalties rescue" hypothesis holds for `lambda_frm` but is reversed for `lambda_rws`;
   in the combined cell, `lambda_frm` dominates and the count is still 0.
4. **Equation type (`h` vs `s`)** makes little qualitative difference; `s` has marginally more
   silent units at large N.

The per-unit **participation distribution** (`std(rate) + 0.9-quantile(|rate|)`, pooled over the
5 nets of each condition) shows the mechanism directly — reported as separate figures per
equation type (h and s are not pooled together), plotted by
[`plot_participation_histograms.py`](../trainRNNbrain/experiments_and_analysis/plot_participation_histograms.py):

![Participation histograms — h equation](../img/internal_figures/participation_histograms_h.png)

![Participation histograms — s equation](../img/internal_figures/participation_histograms_s.png)

In both equation types and at every N, the **fr-penalised** conditions (green `fr only`, blue
`both`) form a single tight mode at participation ≈ 0.4–0.6 with **no near-zero pile and no
high-participation tail** — every unit participates, none dominates (the `cap_fr=0.3` target
bounds the rates). The **unpenalised** (`none`, red) and **rws-only** (orange) conditions are
bimodal: a large spike of silent units at ≈ 0 plus a heavy tail of a few hyper-active units out
to participation > 3. The silent spike grows with N, matching the bar-chart counts above.

**Performance vs participation spread (N=1000).** The silent-unit count cannot tell `both` from
`fr only` (both are 0), so to ask whether `lambda_rws` adds anything *alongside* `lambda_frm` we
plot each N=1000 net's validation R² against **1/HHI of its participation** — the effective number
of participating units (`HHI = Σ(p_i/Σp)²`; `1/HHI = N` when participation is perfectly even,
smaller when concentrated in fewer units), on a log axis, separate panels per equation type, by
[`plot_r2_vs_hhi.py`](../trainRNNbrain/experiments_and_analysis/plot_r2_vs_hhi.py):

![R² vs 1/HHI of participation, N=1000](../img/internal_figures/r2_vs_hhi_N1000.png)

- **Two clusters, set entirely by `lambda_frm`.** fr-penalised nets (`fr only` green, `both` blue)
  sit at high 1/HHI near the even ceiling (N≈1000) — ~all units participate. `none` (red) and
  `rws only` (orange) sit at ~3–10× lower 1/HHI (~70–250 effective units) — participation
  concentrated in a minority, the rest silent. `lambda_rws` alone does **not** raise 1/HHI; it
  stays in the low-1/HHI cluster with the baseline.
- **R² is comparable across all conditions (~0.83–0.87)** — the rescue costs essentially no task
  performance: penalised nets use all units *and* solve CDDM as well as the dead-unit nets.
- **`lambda_rws` in conjunction with `lambda_frm` — present in BOTH equation types.** `both` (blue)
  sits at higher 1/HHI than `fr only` (green) in both: h 746 → 887 effective units (+19%),
  s 810 → 912 (+13%); the means are well-separated (std ≤ 24) in each. The `s` gap merely *looks*
  smaller because both clouds sit near the N ceiling on the log axis. So rws adds participation-
  evenness on top of frm regardless of equation type (marginally more in h, partly a ceiling effect
  since `s` `fr only` starts more even); it never reduces silent units below the 0 that frm already
  achieves — frm is the rescuer, rws a consistent refinement of how evenly the surviving units
  share the work.

**The worst unit, made concrete (N=1000).** Taking the *single least-participating unit* of each
N=1000 net and plotting its firing rate (x = time, y = CDDM task condition, red intensity =
activity) — 5 nets × 4 penalty conditions, separate figures per equation type, by
[`plot_least_unit_activity.py`](../trainRNNbrain/experiments_and_analysis/plot_least_unit_activity.py):

![Least-participating unit activity — h equation](../img/internal_figures/least_unit_activity_h_N1000.png)

![Least-participating unit activity — s equation](../img/internal_figures/least_unit_activity_s_N1000.png)

Under `none` and `rws only` the worst unit is uniformly blank (participation `p ≈ 0.000` — a fully
dead ReLU) in every net and both equation types. Under `fr only` the worst unit is a weak, localized
**transient** (`p ≈ 0.05–0.06`); under `both` it is stronger and more sustained (`p ≈ 0.09–0.12`).
This floor-lift by rws is essentially **identical in h and s** (5th-percentile participation of the
revived units: `fr only` ≈ 0.06 → `both` ≈ 0.13–0.14 in each equation type).

### Bottom line

**Setup.** 120 ReLU-Dale RNNs on CDDM (commit `4a031e`): equation type {h, s} × N {100, 500,
1000} × penalty {none, rws `Lrws=0.05`, fr `Lfr=0.2`, both} × 5 nets. "Silent" = peak firing
rate `< 0.01` over the noise-free validation batch.

1. **Unmodified ReLU-Dale nets waste most of their capacity, and it worsens with scale.** Silent
   fraction ≈ 0% at N=100, ~24–38% at N=500, and **~47–56% at N=1000** — both equation types.
   Roughly half a large network is dead.

2. **The firing-rate-magnitude penalty (`lambda_frm`) is the rescuer — necessary and sufficient,
   at no task cost.** `Lfr=0.2` drives silent units to **exactly 0** in every cell (both eq, all
   N, with or without rws); participation reaches the even ceiling (1/HHI ≈ N, ~all units
   contribute) and R² is unchanged (~0.83–0.87).

3. **The recurrent-weight-sparsity penalty (`lambda_rws`) does not rescue on its own** — if
   anything marginally worse than baseline (a sparsifier concentrates participation; it stays in
   the dead-unit / low-1/HHI cluster).

4. **But the FR rescue is uneven at the margin.** The *bulk* of FR-rescued units form a healthy
   participation mode (~0.4–0.6); however the *least*-participating units (`fr only`) carry only
   weak, brief, localized activations (`p ≈ 0.05`) — transients that satisfy the penalty without
   an obviously sustained role (the penalty is partly "gamed" at the margin).

5. **Adding `lambda_rws` on top of `lambda_frm` lifts and shapes that marginal floor — in BOTH
   equation types.** Under `both`, the worst units have higher participation (`p ≈ 0.09–0.12` vs
   ≈ 0.05–0.06 for `fr only`) and more sustained, time- and condition-structured responses. The
   floor-lift is essentially equal in h and s (5th-pct participation ≈ 0.06 → ≈ 0.13–0.14 in each),
   and population evenness rises in both (mean 1/HHI: h 746 → 887, s 810 → 912). So `rws` cannot
   prevent silence by itself, but in combination it consolidates the marginally-revived units from
   thin transients into more sustained responses — for h and s alike (gain marginally larger in h,
   partly a ceiling effect).

**One line.** *`lambda_frm` abolishes dead ReLUs (all N, both equation types, zero R² cost);
`lambda_rws` can't do this alone, but added to `lambda_frm` it consolidates the marginally-revived
units from thin transients into more sustained, structured responses — in both equation types
(marginally stronger for h).*

**Status of the claims / what's not yet shown.** Points 1–3 are direct measurements (silent
counts, 1/HHI, R²). Points 4–5 are interpretations of the *activity shape* of the single worst
unit, not a direct test of "computational role." To confirm: decode the task variables (context,
motion/colour evidence, choice) from the marginally-revived units and compare `fr only` vs `both`;
quantify selectivity / clustering of the revived population. A `lambda_rws` magnitude sweep would
map how much shaping it adds.

### How to reproduce from scratch

1. `git -C ~/trainRNNbrain fetch origin && git -C ~/trainRNNbrain checkout 4a031e7`
2. `module load anacondapy/2024.02 && conda activate torch-env` (or any env with torch + this
   package editable-installed).
3. `sbatch slurm/SilentReLU_ReluDale_sweep.slurm` (adjust `--partition`, `paths.save_to`,
   and the SBATCH `--output/--error` paths for a non-Spock cluster).

---

## 2026-06-29 — v2: gamma=0 ("naked" ReLU) + architectural-confound scouting

### Why

Before over-interpreting the v1 results, we scouted the model/training code for architectural and
parameter choices that could contaminate the "naked, field-standard ReLU-Dale RNN" reading. Three
matter; one is removed in this v2 run, two are flagged for follow-up.

### Confounds found (code refs: `trainRNNbrain/rnns/RNN_torch.py`, `trainRNNbrain/trainer/Trainer.py`)

1. **Cubic term `−gamma·x³` in the dynamics** (`RNN_torch.rhs`, active when `gamma > 1e-8`; config
   default `gamma=0.1`). It is a soft saturation — a *built-in activity-magnitude limiter baked into
   the dynamics* — which confounds the `lambda_frm` (firing-rate-magnitude) comparison: the v1
   baseline already had some magnitude control. **Removed in v2 via `model.gamma=0`** (config
   override, no code change; cubic skipped).

2. **Dale sign-projection is a clamp-to-eps** (`Trainer.enforce_dale_`, run after *every*
   `optimizer.step()`): a `W_rec` weight whose sign violates Dale is hard-set to `±1e-12`, i.e.
   **pinned at ~0, not reflected**. This is projected gradient descent onto the Dale orthant; a
   weight whose gradient keeps pushing it into the forbidden sign **sticks at ~0**. Over 30k steps
   this causes **emergent sparsification of `W_rec` independent of `lambda_rws`**, so part of the
   baseline's dead-unit / sparsity signature is a projection artifact rather than intrinsic to
   "naked ReLU". **Not changed in v2** (it is part of the Dale model); see follow-up below.

3. **Bias fixed at 0** (`bias_range=[0,0]`, `bias_trainable=False`): a ReLU unit with net-negative
   input across all conditions is silent with **no bias to lift it** into the active regime, which
   likely *inflates* the silent-unit count. A trainable bias is arguably more standard. **Not
   changed in v2** (kept for a clean one-variable comparison).

   Secondary (unchanged, lower expected impact): exc-only readout with `W_out ≥ 0` (inhibitory
   units never directly drive the output); no self-connections (`W_rec` diagonal zeroed each step);
   `spectral_rad=1.2` init; training noise `σ_rec=σ_inp=0.05`. Penalties are applied via a
   task-safe gradient projection (penalty-gradient components opposing the task gradient are
   dropped) — irrelevant to the no-penalty baseline.

### v2 experiment (this run)

**Single change from v1: `model.gamma=0`.** Everything else identical, **N=1000 only**.

- Grid (40 jobs) = 2 eq {h, s} × 2 `lambda_rws` {0, 0.05} × 2 `lambda_frm` {0, 0.2} × 5 nets.
- Code commit `4a031e` (gamma is a config override). Output root folder **`CDDM_4a031e_g0/`**
  (`_g0` = gamma 0), leaf `EqType=<eq>_N=1000_LmbdRWS=<rws>_LmbdFR=<frm>/`.
- Script: [`slurm/SilentReLU_ReluDale_gamma0_N1000.slurm`](../slurm/SilentReLU_ReluDale_gamma0_N1000.slurm)
  (`--mem=32G` — all N=1000; `--array=1-40`, no throttle). Submitted **2026-06-29 ~14:50 EDT**,
  SLURM array job **`5096100`** (smoke test `5095648` COMPLETED; confirmed `model.gamma=0` in the
  saved config). Re-run the v1 analysis scripts pointed at `CDDM_4a031e_g0` to compare.

#### v2 results — gamma=0 (sticky), N=1000

The full v1 figure set regenerated on `CDDM_4a031e_g0` (the analysis scripts now take a sweep-folder
arg; figures suffixed `_4a031e_g0`). gamma=0 was run for **N=1000 only**, so the N=100/500 groups in
the bar chart are empty. **Removing the cubic term barely changes the silent-unit story — the cubic
was not a confound here.** Silent units (dead, peak rate < 0.01), v1 (cubic=0.1) → v2 (gamma=0):

| eq · penalty (N=1000) | v1 (cubic=0.1) | v2 (gamma=0) |
|---|---|---|
| h · none | 47% | 44% |
| h · rws-only | 54% | 53% |
| s · none | 54% | 55% |
| s · rws-only | 56% | 56% |
| any · fr-only / both | 0% | 0% |

![Silent units per condition — gamma=0](../img/internal_figures/silent_units_per_condition_4a031e_g0.png)

![Participation histograms, h — gamma=0](../img/internal_figures/participation_histograms_h_4a031e_g0.png)

![Participation histograms, s — gamma=0](../img/internal_figures/participation_histograms_s_4a031e_g0.png)

![R² vs 1/HHI, N=1000 — gamma=0](../img/internal_figures/r2_vs_hhi_N1000_4a031e_g0.png)

![Least-participating unit activity, h — gamma=0](../img/internal_figures/least_unit_activity_h_N1000_4a031e_g0.png)

![Least-participating unit activity, s — gamma=0](../img/internal_figures/least_unit_activity_s_N1000_4a031e_g0.png)

> Repro note: the runtime `git rev-parse` folder hash scattered these 40 nets across
> `CDDM_4a031e_g0` and `CDDM_2bc3c1_g0` (Spock's checkout was advanced mid-sweep for the reflective
> code) — all gamma=0 sticky, behaviour-identical; consolidated into `CDDM_4a031e_g0` for analysis.
> Lesson: pin the hash in `experiment_tag` instead of computing it per-job at runtime.

### Planned follow-up (to discuss / next)

Make the Dale boundary behaviour an explicit, switchable config parameter rather than a hard-coded
clamp — e.g. **`weight_boundary: sticky | reflective`**:
- `sticky` = current behaviour (clamp sign-violating weights to `±eps`, pinned at 0).
- `reflective` = reflect across zero (`W ← |W| · sign_mask`) or reparametrize `W_rec = |θ| · sign_mask`,
  so weights are not pinned at the boundary.
Then re-run the gamma=0 sweep under both to quantify how much of the baseline silence is an artifact
of the `sticky` projection. (Also still open from v1: trainable-bias variant; decoding/selectivity
of the marginally-revived units.)

### v2b — `weight_boundary` implemented + reflective gamma=0 sweep

The `weight_boundary` parameter is now in the model (commits `7fdf63`, `2bc3c16`):
- **`sticky`** (default, legacy): raw `W_rec`; the Trainer clamps sign-violating entries to
  `±weight_boundary_eps` (default `1e-12`) each step. Behaviour is byte-identical to before
  (validated by `trainRNNbrain/rnns/check_weight_boundary.py`).
- **`reflective`**: the effective weight is `|param|·sign·mask`, enforced in `RNN_torch.forward`;
  the Trainer skips the post-step projections, so weights are never pinned at the boundary.
- `get_params` exports the **effective** (Dale-compliant) weights, so `RNN_numpy` and every analysis
  script are boundary-agnostic and reconstruction is correct even if the mode is unknown; `__init__`
  / `set_params` default to `sticky`/`1e-12` (legacy fallback for pre-existing nets).
- Config fields: `model.weight_boundary`, `model.weight_boundary_eps` (in `rnn_relu_Dale.yaml`).

**Reflective sweep (this run).** Same grid as the gamma=0 sweep (2 eq × 2 `lambda_rws` × 2
`lambda_frm` × 5 nets, N=1000, gamma=0) **plus `model.weight_boundary=reflective`**. Code commit
`2bc3c16`, output folder **`CDDM_2bc3c1_g0_reflective/`**. Script:
[`slurm/SilentReLU_ReluDale_gamma0_reflective_N1000.slurm`](../slurm/SilentReLU_ReluDale_gamma0_reflective_N1000.slurm).
Submitted **2026-06-29 ~15:35 EDT**, SLURM array **`5096453`**. Validated before launch by a local
20-iter run (config records `weight_boundary: reflective`, saved `W_rec` Dale-compliant, no eps-pile)
and by `check_weight_boundary` passing on the Spock L40S.

**Comparison axis:** `CDDM_4a031e_g0` (gamma=0, **sticky**) vs `CDDM_2bc3c1_g0_reflective`
(gamma=0, **reflective**). Note: because the `sticky` path is behaviour-preserving, the later jobs of
the `4a031e_g0` sweep actually executed at `7fdf63`/`2bc3c16` in sticky mode (Spock's checkout was
advanced for the reflective code) — identical results, only the recorded commit differs per job.

#### v2b results — gamma=0 reflective (N=1000)

Same figure set on `CDDM_2bc3c1_g0_reflective` (figures suffixed `_2bc3c1_g0_reflective`; N=1000
only). **The reflective boundary does NOT reduce silent units — it has as many or slightly *more*
than sticky.** So the silent-ReLU phenomenon is intrinsic to the unpenalized / rws-only ReLU-Dale
net, **not an artifact of the sticky clamp-to-eps projection.** This was a genuine falsification
test: reflective never pins weights at the boundary, so it *could* have shown far fewer silent
units — it didn't. The firing-rate-magnitude penalty still rescues to exactly 0 in every condition.

Silent units (dead, peak rate < 0.01), N=1000 — gamma=0 **sticky** vs **reflective**:

| eq · penalty | sticky | reflective |
|---|---|---|
| h · none | 44% | 46% |
| h · rws-only | 53% | 58% |
| s · none | 55% | 55% |
| s · rws-only | 56% | 63% |
| any · fr-only / both | 0% | 0% |

![Silent units — gamma=0 reflective](../img/internal_figures/silent_units_per_condition_2bc3c1_g0_reflective.png)

![Participation histograms, h — gamma=0 reflective](../img/internal_figures/participation_histograms_h_2bc3c1_g0_reflective.png)

![Participation histograms, s — gamma=0 reflective](../img/internal_figures/participation_histograms_s_2bc3c1_g0_reflective.png)

![R² vs 1/HHI, N=1000 — gamma=0 reflective](../img/internal_figures/r2_vs_hhi_N1000_2bc3c1_g0_reflective.png)

![Least-participating unit activity, h — gamma=0 reflective](../img/internal_figures/least_unit_activity_h_N1000_2bc3c1_g0_reflective.png)

![Least-participating unit activity, s — gamma=0 reflective](../img/internal_figures/least_unit_activity_s_N1000_2bc3c1_g0_reflective.png)

**Sanity check — reflective was active.** Pooled off-diagonal `|W_rec|` (h, N=1000, gamma=0):
sticky pins **19% (none) / 32% (both)** of recurrent weights at exactly the `eps=1e-12` clamp (the
spike at 1e-12); reflective has **0%** there (smallest weight ~4e-11, continuous) — confirming
`|param|·sign` was in effect with no boundary pinning. Script:
[`plot_weight_distribution.py`](../trainRNNbrain/experiments_and_analysis/plot_weight_distribution.py).

![|W_rec| distribution, sticky vs reflective](../img/internal_figures/weight_distribution_sticky_vs_reflective_h_N1000.png)

Notably, sticky pins a *large* fraction of recurrent weights at ~0, yet that does **not** produce
more silent units than reflective — so the unit-level silence is driven by the training dynamics,
not by which weights sit at the Dale boundary.

**Conclusion across the boundary controls:** neither removing the cubic term (gamma=0) nor
switching the Dale boundary (sticky → reflective) changes the picture — unpenalized / rws-only
ReLU-Dale nets are ~45–63% silent at N=1000, and the firing-rate-magnitude penalty is what rescues
them (to 0), regardless of these architectural choices. The silent-ReLU result is robust.

---

## 2026-06-29 — adversarial pre-publication checks

Goal: exclude trivial mechanisms before claiming the result. Checks run on existing data
(`tmp/check_now.py`, deliberately not committed — one-off); controls submitted as new sweeps;
remaining items in [`TODO.md`](../TODO.md).

**Now-checks (h, N=1000):**
1. **Not a code bug.** An independently-written from-scratch Euler integrator (no `RNN_numpy`/`RNN_torch`)
   reproduces the production per-unit peak rates **exactly** (max abs diff 0.0e+00, silent-set Jaccard 1.000).
2. **Not an init artifact.** Fresh untrained nets are **0% silent** at every `spectral_rad` ∈ {0.6,1,1.2,1.6}
   — the silence **emerges during training**, it is not present at initialisation.
3. **Refinement — it's a low-activity *mode*, not "dead ReLUs".** The peak-rate distribution is bimodal:
   a tight active mode (~0.3–0.5) and a broad low-activity mode (~1e-3, spanning 1e-5–1e-1). Only **~2.5%**
   are *truly* dead (peak≈0); the silent fraction is therefore threshold-dependent (13% at <1e-4, 44% at
   <1e-2, 49% at <5e-2). The principled cut is the inter-mode dip (~0.05) → ~49% in the low-activity mode.
   Penalised nets have a single tight active mode (every unit peaks >0.1). **Report the distribution, not a
   single thresholded number.**

**Controls submitted (gamma=0, N=1000, same grid):**
- **Activation:** softplus(β=25) `CDDM_f8be3e_g0_softplus25` (`5100301`), leaky-ReLU `CDDM_f8be3e_g0_leakyrelu`
  (`5100302`) — both have nonzero gradient everywhere (no dead-gradient trap). The decisive test: if they
  still show the low-activity mode, the phenomenon is "trained RNNs concentrate computation", not a hard-ReLU
  pathology; if not, the near-zero tail is ReLU-specific.
- **Noise:** sigma_rec ∈ {0,0.01,0.05,0.1}, none penalty `CDDM_f8be3e_g0_noise` (`5100303`) — is the silence
  noise-driven?

Deferred (see TODO): trainable-bias control, prevention-vs-resurrection mechanism (logged training),
second task, task-dimensionality probe.

## 2026-07-01 — control results: activation function & recurrent noise

The activation and noise controls above were re-run at commit `fb2792` (the `f8be3e` submissions were
superseded) and analysed here. All sweeps: N=1000, γ=0, no bias, scored on the **noise-free** CDDM batch
(so we measure the silence baked into the trained weights, not instantaneous noise silencing). Baseline for
comparison is `CDDM_4a031e_g0` (plain ReLU): N=1000 silent (peak<0.01) — h/none ≈47%, h/rws ≈54%, s/none ≈54%,
s/rws ≈56%; fr-only and both = 0.

**Metrics defined.** Every net is reconstructed from its saved (Dale-compliant) weights and run on the
noise-free CDDM batch, giving a firing-rate tensor `fr` of shape `(N units, T time, C conditions)`. From it:

- **peak rate** `peak_i = max over (t, c) of |fr[i]|` — unit `i`'s single most-active moment anywhere in the task.
- **dead<0.01** — fraction of units with `peak_i < 0.01`. An **absolute** floor (the rate cap/target is ~0.3–0.5),
  so "essentially never fires." Fine for ReLU (silent units hard-zero) but *unfair across activations*: softplus
  has a smooth positive floor so no unit is ever exactly 0, and this cut then reads 0% even when half the units are
  ~30× quieter than the active ones.
- **silent<5%p95** — fraction of units with `peak_i < 0.05 · p95`, where `p95` is the net's 95th-percentile peak.
  A **within-network, scale-free** cut: "is this unit's best moment still >20× quieter than the net's active
  population?" (95th percentile, not max, so one outlier unit doesn't set the scale). This is the number to compare
  across activations, because it is invariant to each net's overall activity scale.
- **participation** `p_i = std(fr[i]) + 0.9-quantile(|fr[i]|)` over (t, c) — a graded activity measure (used for the
  histograms, HHI, and least-unit pick) rather than a hard silent/active threshold.
- **HHI / 1/HHI** — Herfindahl–Hirschman index of participation: with shares `s_i = p_i / Σ p_j`, `H = Σ s_i²`.
  `H ∈ [1/N, 1]`; `H = 1/N` is perfectly even participation, larger `H` means a few units dominate. **`1/H` is the
  effective number of participating units** — the intuitive x-axis of the scatter (≈ N=1000 means "all units share
  the work", ≈ 60 means "~60 units do everything, the rest are near-silent").

Counts by
[`count_silent_units.py`](../trainRNNbrain/experiments_and_analysis/count_silent_units.py) (softplus, leaky) and
[`count_silent_units_noise.py`](../trainRNNbrain/experiments_and_analysis/count_silent_units_noise.py) (noise, adapted to the `sigrec=` dir naming).

### Activation function — `CDDM_fb2792_g0_softplus25` (38/40 nets; job 5100397), `CDDM_fb2792_g0_leakyrelu` (40/40; 5100398)

Both softplus(β=25) and leaky-ReLU (leak 0.01) have **nonzero gradient everywhere** — no dead-gradient trap.
Silent fraction at N=1000 (mean over 5 nets/condition, except softplus h/none and s/both = 4 nets — 2 jobs
short, softplus trains slower):

| eq / penalty | criterion | ReLU baseline | softplus25 | leaky-ReLU |
|----|----|----|----|----|
| h / none    | silent<5%p95 | ~47% | 40.6% | 45.2% |
| h / rws     | silent<5%p95 | ~54% | 63.6% | 56.2% |
| s / none    | silent<5%p95 | ~54% | 54.4% | 55.1% |
| s / rws     | silent<5%p95 | ~56% | 61.1% | 61.1% |
| h / none    | dead<0.01    | ~47% | **0.0%** | 32.5% |
| h / rws     | dead<0.01    | ~54% | **0.0%** | 49.5% |
| s / none    | dead<0.01    | ~54% | 52.4% | 54.0% |
| s / rws     | dead<0.01    | ~56% | 53.3% | 54.0% |
| any / fr-only, both | both | 0 | **0** | **0** |

![Participation — softplus25 h](../img/internal_figures/participation_histograms_h_fb2792_g0_softplus25.png)
![Participation — leaky h](../img/internal_figures/participation_histograms_h_fb2792_g0_leakyrelu.png)

**Finding (the key framing test): the large low-activity population is NOT a hard-ReLU / dead-gradient
artifact — it is a general property of trained CDDM RNNs.**

1. On the scale-free criterion, **all three activations concentrate ~40–64% of units into a low-activity
   mode** under `none`/`rws`, statistically indistinguishable from the ReLU baseline. Everywhere-positive
   gradient does **not** keep units active.
2. **Softplus-h is the sharpest demonstration:** `dead<0.01` = **exactly 0%** (softplus's smooth floor keeps
   every unit's peak nominally above 0.01), yet `silent<5%p95` is still 41–64%. The population doesn't
   disappear — it **reorganises from exact zeros into a soft low-activity continuum** at the same ~half-of-N
   mass. The participation histogram shows it directly: red (`none`)/orange (`rws`) still pile near zero, green
   (`fr`)/blue (`both`) collapse to one tight active bump. Softplus-s and both leaky equations keep large
   *hard*-near-zero populations (49–57%), like ReLU.
3. **The fr-magnitude penalty collapses everyone into one active mode in every activation** (fr-only, both = 0
   throughout) — the rescue is activation-independent.

**Performance vs participation spread (R² vs 1/HHI).** Each point is one N=1000 net: x = effective number of
participating units (`1/HHI`, log scale; dashed line = even N=1000), y = validation R² (the score in the net's
folder name). Plotted by
[`plot_r2_vs_hhi.py`](../trainRNNbrain/experiments_and_analysis/plot_r2_vs_hhi.py).

![R² vs 1/HHI — softplus25](../img/internal_figures/r2_vs_hhi_N1000_fb2792_g0_softplus25.png)
![R² vs 1/HHI — leaky](../img/internal_figures/r2_vs_hhi_N1000_fb2792_g0_leakyrelu.png)

The two clouds are cleanly separated on the x-axis and overlapping on the y-axis: `none`/`rws` nets solve the task
using an **effective ~60–150 units** (softplus-h: ~60–90), while `fr`/`both` nets spread the identical task across
**~700–900** — near the even line — at **equal or better R²** (~0.82–0.87 throughout). So concentrating computation
onto a small subset is **not required** for performance; it is what the network does when nothing pushes back.
Two asides visible here: for **softplus-h, `rws`-only actively hurts R²** (several nets at 0.4–0.65, the low orange
points) while barely changing the concentration — sparsifying recurrent weights degrades the solution without
redistributing activity; and the `s` equation concentrates less severely than `h` (clouds sit further right).

**The least-participating unit.** For each net we take the single lowest-participation unit and draw its firing
rate as a heatmap (x = time, y = task condition; `p` = its participation). 5 nets (rows) × 4 penalties (cols),
by [`plot_least_unit_activity.py`](../trainRNNbrain/experiments_and_analysis/plot_least_unit_activity.py).

![Least-unit activity — softplus25 h](../img/internal_figures/least_unit_activity_h_N1000_fb2792_g0_softplus25.png)

Under `none`/`rws` the **worst unit is blank** (softplus-h `p≈0.003` — nominally nonzero because of softplus's
floor, but flat and task-unrelated). Under `fr`/`both` **even the worst unit carries a clear, time- and
condition-locked activity blob** (`p≈0.05–0.12`). This is the same collapse seen in the histograms, at the
single-unit extreme: the penalty doesn't just shift the bulk — it pulls up the very tail.

### Recurrent noise — `CDDM_fb2792_g0_noise` (40/40; job 5100399)

ReLU, `none` penalty, sweeping the recurrent training noise σ_rec ∈ {0, 0.01, 0.05, 0.1}. Silent fraction at
N=1000 (mean over 5 nets/σ):

| σ_rec | h dead<0.01 | h silent<5%p95 | s dead<0.01 | s silent<5%p95 |
|----|----|----|----|----|
| 0.00 | **79.4%** | **81.6%** | 58.2% | 60.1% |
| 0.01 | 46.0% | 47.9% | 55.5% | 57.9% |
| 0.05 | 43.5% | 47.6% | 54.1% | 55.5% |
| 0.10 | 40.7% | 46.0% | 56.4% | 57.0% |

![Silent fraction vs recurrent noise](../img/internal_figures/silent_vs_noise_fb2792_g0_noise.png)

**Finding: the silence is NOT noise-driven.** Adding recurrent noise never *increases* it.

1. For **h**, σ_rec=0 (noise fully off) is a distinct pathological regime — **~80% silent**. Any nonzero noise
   (≥0.01) drops it to ~46–48%, then it is flat-to-slightly-*decreasing* with more noise: recurrent noise mildly
   **regularises against** silence rather than causing it.
2. For **s**, the silent fraction is ~54–60% and essentially **flat across all σ_rec** — noise-independent.
3. The ReLU baseline (h/none ≈47%, s/none ≈54%) matches the **σ_rec≈0.05** column (h 43.5%, s 54.1%), i.e. the
   baseline sweep trained with a nonzero default σ_rec; the σ=0 point is what removing noise entirely exposes.
4. Net: the ~45–60% low-activity population **persists across every realistic noise level** — an intrinsic
   feature of the trained solution, not a recurrent-noise artifact.

### Bottom line

Two candidate "trivial mechanism" explanations are now excluded. The low-activity population is neither a
hard-ReLU dead-gradient trap (survives softplus/leaky, everywhere-differentiable) nor noise-induced silencing
(survives noise-free, mildly reduced by noise). Combined with the earlier controls (not a code bug, emerges in
training not init, a low-activity *mode* not literal dead units), the phenomenon is best framed as
**trained CDDM RNNs concentrate computation onto a subset of units**, and the fr-magnitude penalty is what
redistributes it across the full population.

## 2026-07-01 — mechanism: prevention vs resurrection (init vs trained, per-unit)

**Question (Pavel):** how many units are silent *at initialisation* (before any training), and — for any that
start silent — are they later rescued or do they stay silent? I.e. does `frm` work by **prevention** (keep
still-active units alive) or **resurrection** (revive dead ones)?

**Exact init reconstruction (no retraining needed).** Each net in these sweeps was trained as its own job
(`n_nets=1`), so the loop index is always `i=0` and the per-net RNN generator is seeded deterministically as
`rnn_seed = cfg.seed + (0·14653 + 65537³) mod 7309 = cfg.seed + 3508` (see `run_experiment.py:60`). Weight init
draws only from that generator (`get_connectivity_Dale`), and with `bias_range=[0,0]` there is no bias draw — so
the untrained weights are a **pure function of the saved config seed**, independent of training history. We
reconstruct each net's initial weights via `RNN_torch(...).get_params()` (the same effective/Dale-compliant
export used for trained weights), score them on the identical noise-free CDDM batch, and — since units are never
reordered during training — pair **the same unit index** before and after. Reconstruction is verified
deterministic, Dale-compliant, and (independently) reproduces the earlier "0% silent at init" now-check.
Analysis: [`plot_init_vs_trained.py`](../trainRNNbrain/experiments_and_analysis/plot_init_vs_trained.py).

**Result — nothing starts silent.** Init silent fraction is **0.0% in every condition, both metrics, both
activations** (softplus25 and leakyrelu; `init_vs_trained_silent.csv`). The trained columns reproduce the counts
reported above exactly. Representative (softplus-h / leaky-h):

| sweep · eq · penalty | init dead<0.01 | init silent<5%p95 | trained dead<0.01 | trained silent<5%p95 |
|----|----|----|----|----|
| softplus · h · none | 0.0% | 0.0% | 0.0% | 40.6% |
| softplus · h · rws  | 0.0% | 0.0% | 0.0% | 63.6% |
| softplus · h · fr / both | 0.0% | 0.0% | 0.0% | 0.0% |
| leaky · h · none | 0.0% | 0.0% | 32.5% | 45.2% |
| leaky · h · rws  | 0.0% | 0.0% | 49.5% | 56.2% |

![Init vs trained participation — softplus25 h](../img/internal_figures/init_vs_trained_hist_h_fb2792_g0_softplus25.png)
![Per-unit init→trained — softplus25 h](../img/internal_figures/init_vs_trained_scatter_h_fb2792_g0_softplus25.png)

**Mechanism — training *splits* a homogeneous population; `frm` prevents the downward branch.** At init every
unit sits in one narrow participation band (~0.05–0.08; black dashed in the histogram) — no silent units, no
active mode yet. Training then bifurcates it:

- Under `none`/`rws`, ~40–64% of units are driven **below their init level** into a near-zero silent mode
  (participation ~1e-3), the rest **up** into a broad active tail. The per-unit log-log scatter shows the init
  band fanning both ways; `corr(log init, log trained) ≈ 0.3–0.5` — a unit's (tiny) init activity only weakly
  predicts its fate, i.e. **the silencing is decided during training, not preset at init**.
- Under `fr`/`both`, **every** unit is pushed **up** into a single active mode (0.15–0.6, sharp `cap_fr` cutoff);
  0% end below the silent guide.

**Answer.** There are ~no units silent at initialisation, so `frm` cannot be *resurrecting* dead units — there
are none to revive. What it does is **prevention**: on the natural init it stops training from collapsing the
~half of units that `none`/`rws` would silence. (The stronger test — deliberately force-initialising units silent
and asking whether `frm` can pull *those* back — remains open; see `TODO.md`. It only matters for the adversarial
"can it resurrect?" question; on the real init the mechanism is prevention.)

## 2026-07-01 — deliberate silent-at-init experiment: can frm RESURRECT? (submitted)

**The gap this closes.** The tracking above shows that on the *natural* init nothing is silent, so `frm`'s role is
prevention — but that leaves the sharper question unanswered: **if some units are *already* silent at init, can
`frm` resurrect them, or does it only keep still-active units alive?** Prevention vs resurrection is only
decidable if silent-at-init units actually exist. So we build an init that has them, by construction, and track
those specific units through training.

**Why we can't just "boost inhibition globally".** We first tried the intuitive knob — scale *all* inhibitory
recurrent weights up so more units go net-negative. It fails: the untrained net is nearly homogeneous (every
unit's peak firing rate is within ±6% of ~0.05), so a global inhibitory boost dims the whole network *together*
— the entire population slides through the 0.01 "silent" line as one block (0% silent at boost 1.4 → 71% at 1.6),
with no robustly-active units left (max rate ~0.013). That produces "everything is half-dead", not "a distinct
25% silent against a healthy-active 75%". Any *global scalar* preserves the init symmetry and can only dim
collectively; carving out a silent *subpopulation* requires a *unit-targeted* perturbation.
(Calibration: [`calibrate_inhibitory_boost.py`](../trainRNNbrain/experiments_and_analysis/calibrate_inhibitory_boost.py).)

**The perturbation (targeted inhibition).** Pick a fixed random 25% of units — the set **S** — and over-inhibit
*only them*: at init, after the standard Dale construction, multiply the inhibitory columns (synapses from
I-units, `dale_mask == -1`) of the **S rows** of `W_rec` by a factor **c = inhibitory_boost**. This drives each
S unit's total input net-negative → S is silent at init, while the other 75% keep their normal ~0.05 activity. S
is drawn from the net's seed (`numpy.default_rng(seed)`), so it is **exactly reconstructable from the saved
config** — we know which units were silenced and can track them. Implemented as a no-op-by-default one-liner in
`RNN_torch.__init__` (`inhibitory_boost=None, silent_init_frac=0.25`); all prior runs are unaffected.

**Calibration result.** With `|S| = 25%` fixed, the boost is a near-binary switch: `c = 1.0` → 0% of S silent,
`c ≥ 1.25` → **100% of S silent, 0% collateral on the other 75%, exactly 25% total** (robust across 4 seeds, h
and s; saturates by c≈2 with S peak ~0.003). Verified end-to-end: `boost=None` gives 0% silent, `boost=2.0`
gives exactly 25% and the silent set equals S reconstructed from the seed. **We use c = 2.0** (see the note on
what `c` means below).

**What `c` is.** `c` is *not* the silent fraction — that is fixed at 25% by `|S|` for any `c ≥ ~1.25`. `c` sets
**how hard the silencing is baked in**, i.e. the initial magnitude of the inhibitory weights onto S that training
must undo to bring an S unit back:
- `c` just above threshold (~1.25): S is *marginally* silent (input barely negative) — trivially rescuable, a
  weak test.
- larger `c` (2, 3, 6…): S is *deeply* silent (input strongly negative), the S-inhibition weights start 2–6×
  their natural value, so training must walk them much further down to re-activate an S unit — a progressively
  harder rescue.
So `c` is the **rescue-difficulty dial**. At init the *activity* saturates by c≈2 (more inhibition can't make a
floored unit more floored), but the *weight magnitude* keeps growing with `c`, which is what matters for whether
training can climb back out. We chose c = 2.0: solidly silent (unambiguously below the line, with margin above
the 1.25 threshold) yet only 2× the natural inhibition → clearly rescuable, so the test is fair and not rigged
toward "prevention" by an impossible-to-undo init. A follow-up `c ∈ {1.5, 3, 6}` sweep would trace how rescue
depends on silencing depth.

**The run (submitted — job `5103664`, commit `d9e0ec7`).** ReLU-Dale, γ=0, N=1000, c=2.0 targeted init.
Grid = 20 jobs: 2 equations (h, s) × 2 penalties × 5 seeds, with penalty being
- **`none`** (`lambda_rws=0, lambda_frm=0`) — the control: do the S units stay silent without any rescue
  pressure? (they should — this fixes the "no-rescue" reference), and
- **`frm`** (`lambda_frm=0.2`) — the rescuer: the firing-rate-magnitude penalty that collapses the low-activity
  mode in every earlier sweep.

Config `configs/model/rnn_relu_Dale_silentinit.yaml`; launcher `slurm/SilentReLU_silentinit_gamma0_N1000.slurm`.
Output → `data/trained_RNNs/CDDM_d9e0ec_g0_silentinit/` (folder uses the 6-char commit hash).

**Read-out (planned).** Reconstruct each net's init, identify S from the seed, and follow S through training with
the per-unit machinery of
[`plot_init_vs_trained.py`](../trainRNNbrain/experiments_and_analysis/plot_init_vs_trained.py) (now the
init-silent branch is populated). The decisive comparison, restricted to the S units:
- under **`none`**, S is expected to remain silent (no force pulling it up) — the prevention/no-rescue floor;
- under **`frm`**, if S **stays silent** → `frm` works purely by **prevention** (it keeps active units alive but
  cannot revive dead ones); if S **climbs into the active mode** → `frm` can **resurrect**. Hunch from the earlier
  tracking: mostly prevention, possibly partial resurrection for the `h` equation.

### Result (2026-07-02) — frm RESURRECTS, and the init silencing is not durable

All 20 nets completed (job `5103664`); analysed by
[`plot_silentinit_rescue.py`](../trainRNNbrain/experiments_and_analysis/plot_silentinit_rescue.py). Init check
passes in every net: **S = 100% silent, non-S = 0% silent** — the perturbation did exactly what it should.
Per-condition (mean over 5 nets), "active" = trained peak firing rate ≥ 0.01:

| eq | penalty | init S silent | **trained S active** | trained non-S active | S median participation | non-S median participation |
|----|----|----|----|----|----|----|
| h | none      | 100% | **57.0%** | 57.0% | 0.009 | 0.008 |
| h | frm=0.2   | 100% | **100.0%** | 100.0% | 0.134 | 0.230 |
| s | none      | 100% | **49.7%** | 47.1% | 0.000 | 0.000 |
| s | frm=0.2   | 100% | **100.0%** | 100.0% | 0.342 | 0.382 |

![Silenced-at-init units through training — h](../img/internal_figures/silentinit_rescue_scatter_h.png)

**Two conclusions:**

1. **`frm` can RESURRECT.** Under `frm`, **100% of the units that were dead at init become active** (both h and
   s), landing in the same active mode as the never-silenced units. The earlier natural-init experiment could only
   demonstrate prevention because nothing was silent at init; forcing 25% silent shows `frm`'s reach is broader —
   it drives *every* unit into the active mode regardless of whether it started active (non-S) or dead (S). So
   `frm`'s mechanism is best stated as **"make the all-units-active state the trained solution, reachable from any
   init"** — which subsumes both prevention (keep active units alive) and resurrection (revive dead ones).

2. **The init silencing is not durable — fate is decided during training.** Under `none`, the deliberately-dead S
   units end up **exactly as active as the never-silenced non-S units** (57.0% vs 57.0% for h; 49.7% vs 47.1% for
   s), i.e. S is statistically indistinguishable from the general population after training, which itself carries
   the usual ~50% silent mode. Training reshuffles which units are silent; being silenced at init confers no
   lasting disadvantage. (Consistent with the natural-init finding that init activity only weakly predicts trained
   fate, `corr ≈ 0.3–0.5`.) In the scatter this is the red (S) cloud starting at the far left (init ~2e-4) and,
   under `none`, splitting ~half above / half below the silent guide — the same split as the grey non-S cloud.

**Caveat / scope.** This used `c = 2.0` — a *moderate*, deliberately rescuable silencing (the `none` control
confirms S is not durably stuck even without `frm`). It shows `frm` *can* resurrect units that were dead at init;
it does **not** claim `frm` could revive an arbitrarily deeply-locked unit. The next experiment (master inhibitor)
tests exactly that limit.

## 2026-07-02 — master-inhibitor experiment: the clean "no gradient → no rescue" test (submitted)

**Why the inhibitory-boost result needs a follow-up — the gradient argument.** A truly-dead ReLU unit (input
negative at every timestep) has `ReLU′ = 0` throughout, so the gradient to its incoming weights is zero —
*including* from the `frm` under-penalty (which rewards sub-`cap` units for firing more, but that reward still
flows through `ReLU′`). So **no firing → no gradient → no rescue** — a fully-dead unit cannot be pulled up by its
own weight updates. Why, then, did the `inhibitory_boost` units get rescued? Two reasons, both of which we can
now see are artefacts of that construction: (i) their "silence" left a **t=1 initial-condition transient** (peak
~0.003, occurring at the first timestep before the recurrent inhibition acts — verified c-invariant from c=2 to
c=64), so they were never truly gradient-dead; and (ii) the inhibition onto them came from the **general
I-population, whose activity drifts as it trains**, so the clamp weakened and the active network could lift them.
The originally-planned `c`-sweep does not fix this — cranking `c` changes the inhibitory *weight magnitude* but
not the init silence (still that t=1 transient), so it would never produce genuinely gradient-dead units. **The
`c`-sweep is therefore replaced by this cleaner design.**

**The construction (no biases).** One inhibitory unit — the "master inhibitor" — is made a **context-locked
clamp**: it is driven ONLY by the two CDDM context cues (input channels 0 and 1, on throughout every trial) with
weight `master_ctx_drive`, receives **no recurrent input** (so the network cannot silence it), and projects deep
inhibition (`-master_inhib_strength`, default 5) onto a fixed random fraction `master_inhib_frac` of the other
units **and to no one else**. The target set is drawn from the net's seed (reconstructable). Verified at init
(`calibrate_master_inhibitor.py`): master active (peak = `ctx_drive`), **targets 100% silent, non-targets
untouched**, for every fraction. Implemented as a no-op-by-default block in `RNN_torch.__init__`.

**Why this is the clean test.** The clamp is **frozen against gradient descent**: the two weights that could
release a target — master→target, and context→master — both influence the loss only *through the dead target's
zero-derivative ReLU*, so both get ~zero gradient; and the master itself has no recurrent input to be silenced by.
The inhibition is therefore sustained across the whole trial and unreachable by training — the genuine
"no gradient → no rescue" condition, unlike the drifting, transient-leaking `inhibitory_boost`.

**The run (submitted — job `5108070`, commit `731df4`).** ReLU-Dale, γ=0, N=1000. Grid = 48 jobs: 2 equations
(h, s) × **4 silenced fractions** `master_inhib_frac ∈ {0.25, 0.5, 0.75, 1.0}` × 2 penalties (`none`, `frm=0.2`) ×
3 seeds. `frac = 1.0` = every unit except the master silenced — Pavel's extreme thought experiment (predict:
no gradient anywhere, R²≈0, nothing revives). Config `configs/model/rnn_relu_Dale_masterinhib.yaml`; launcher
`slurm/SilentReLU_masterinhib_gamma0_N1000.slurm`. Output → `data/trained_RNNs/CDDM_731df4_g0_masterinhib/`.

**Question / predictions.** Does `frm` revive the master-clamped units? Prediction: **no** (or far less than the
boost case) — with the clamp frozen and no firing to seed a gradient, the targets should stay silent even under
`frm`, and the effect should worsen with fraction; at `frac = 1.0` the network cannot learn the task at all
(R²≈0). Contrast with the `inhibitory_boost` result (100% rescued) would confirm that rescue there depended on
the transient/drift footholds, and that genuinely gradient-isolated units are unrescuable — pinning down the
exact boundary of `frm`'s reach.

**Read-out (planned).** Reconstruct init, regenerate the target set from the seed, and (as in
`plot_silentinit_rescue.py`) compare target-active% under `none` vs `frm` across the four fractions, plus task R².

### Result (2026-07-03) — the prediction was WRONG: frm rescues by killing the inhibitor, not the targets

Analysed by [`plot_masterinhib_rescue.py`](../trainRNNbrain/experiments_and_analysis/plot_masterinhib_rescue.py)
(35 valid nets; ~13 of 48 diverged to NaN and were excluded — see caveat). Init check passes: targets 100% silent,
master active. Trained target-active% (peak ≥ 0.01) and the master's own peak, mean over valid nets:

| eq | frac | `none`: master peak / T active | `frm`: master peak / T active | R² (both) |
|----|----|----|----|----|
| h | 0.25 | 0.94 / 12.7% | 0.62 / **100%** | ~0.85 |
| h | 0.50 | 1.43 / 10.2% | 0.12 / **100%** | ~0.85 |
| h | 0.75 | 2.05 /  9.6% | 0.43 / **100%** | ~0.85 |
| h | 1.00 | 4.53 / 29.3% | 0.44 / **100%** | ~0.85 |
| s | 0.25–0.75 | 1.5–1.8 / ~18% | 0.38–0.45 / **100%** | ~0.85 |

![Master-inhibitor rescue](../img/internal_figures/masterinhib_rescue.png)

**Contrary to the prediction, `frm` fully rescues the master-clamped units (100% at every fraction, including
frac=1.0), and the task is solved throughout (R²≈0.85).** The gradient argument was *locally* correct — the dead
targets get ~no direct gradient — but it missed the escape route:

- **`frm` rescues indirectly, by suppressing the inhibitor itself.** The master is *active* and, at ~1–4.5, sits
  far **above** the `frm` target `cap` (~0.3). `frm` penalizes over-cap units, and because the master fires
  (`ReLU′ > 0`) that gradient is real — it reduces the master's context drive, collapsing its peak to ~0.1–0.6.
  With the clamp's source suppressed, the targets' input rises, they start firing, and *then* their own gradients
  flow and they climb into the active mode. The dead targets are never lifted directly; `frm` removes the cause.
- **Under `none` the clamp holds:** the master stays hyperactive (peak 1–4.5, nothing pressures it down) and only
  ~10–30% of targets recover (via incidental/task-gradient footholds), roughly flat across fraction.
- **Even frac=1.0 recovers under `frm`:** the single always-active master is `frm`-attackable, so taming it
  releases everyone and the net solves the task — the "no gradient anywhere" prediction fails precisely because
  the inhibitor is an active, penalizable unit.

**What this means.** `frm`'s reach is broader than "revive units that occasionally fire": it will also dismantle a
*structured* silencing mechanism if that mechanism runs through an active, penalizable unit. The construction did
**not** produce truly-unrescuable units, because the master's weights are trainable and its over-cap activity
makes it a `frm` target. The genuinely-unrescuable case (Pavel's thought experiment) requires the inhibition
source to be **un-tamable** — e.g. freeze the master's weights (non-trainable `W_inp`/`W_rec` for that unit) so
`frm` cannot reduce it. That is the clean follow-up.

**Caveat — divergence.** ~13/48 nets trained to NaN (folder score `nan`), disproportionately under `frm`; the
master's hyperactivity (peak up to 4.5) plus the `frm` pressure likely destabilizes training. Excluded from the
means above; the surviving nets are internally consistent (every condition retained ≥1, most ≥2–3). A rerun with
gradient clipping or a smaller `master_ctx_drive`/`master_inhib_strength` would firm up the counts, but the
qualitative result (frm 100% vs none ~15%) is unambiguous.

## 2026-07-06 — frozen master inhibitor: closing the escape route (submitted)

The unfrozen result showed `frm` rescues the clamped units *indirectly*, by suppressing the over-cap, **active**
master (a legitimate gradient, since the master fires). To realise the genuinely gradient-proof clamp — Pavel's
true thought experiment — we now **freeze the master's weights** so `frm` cannot touch it.

**Implementation (`RNN_torch(freeze_master=True)`, no Trainer changes).** Two mechanisms together: (1) a gradient
hook zeros the gradient on the master's input row `W_inp[m,:]` and its recurrent input row + output column
`W_rec[m,:]`, `W_rec[:,m]`; (2) a `forward_pre_hook` restores those entries to their init values before *every*
forward pass. (1) alone is **insufficient** — Adam + `weight_decay=1e-6` reintroduce a tiny effective gradient
inside `step()` that Adam's normalization amplifies into a full ~`lr` update (empirically the master drifted ~1e-2
over 25 iters with the hook alone); (2) guarantees the master is exactly at init during every training/eval
forward. Verified: [`verify_master_freeze.py`](../trainRNNbrain/experiments_and_analysis/verify_master_freeze.py)
shows master gradient `0.374` (unfrozen) → `0.000` (frozen) on the Trainer's `autograd.grad` path; an end-to-end
25-iter run leaves the master weights unchanged (4.6e-4 save-time artifact) while the rest of the net trains
normally (~1e-2), targets 100% silent, master peak 1.0.

**The run (submitted — job `5115568`, commit `931680`).** Same grid as the unfrozen run: 48 jobs = 2 eq (h, s) ×
4 fractions `master_inhib_frac ∈ {0.25,0.5,0.75,1.0}` × 2 penalties (`none`, `frm=0.2`) × 3 seeds. Config
`configs/model/rnn_relu_Dale_masterinhib_frozen.yaml`; launcher `slurm/SilentReLU_masterinhib_frozen_gamma0_N1000.slurm`.
Output → a **distinct** folder `data/trained_RNNs/CDDM_931680_g0_masterinhib_frozen/`.

**Prediction.** With the clamp frozen, `frm` can no longer tame the inhibitor, so the target units should **stay
silent even under `frm`** (a sharp contrast with the unfrozen 100% rescue) — and at `frac=1.0` the network should
fail the task (R²≈0), since the always-on inhibition is now immovable. Read out with the
`plot_masterinhib_rescue.py` machinery (target-active% under none vs frm across fractions, plus R²).

### Result (2026-07-06) — freeze verified, `none` arm clean, but the `frm` arm is wrecked by divergence

`plot_masterinhib_rescue.py CDDM_931680_g0_masterinhib_frozen`. Freeze confirmed in the data: **master peak is
pinned at exactly 1.0 in every net** (vs 1–4.5 in the unfrozen run), and targets are 100% silent at init. But
**22/48 nets diverged to NaN**, and the divergence is almost entirely in the `frm` arm — every `frm` condition
lost a net (leaving **only 1 valid `frm` net per condition**), while every `none` condition kept all 3.

**`none` arm (clean, 3 nets/cond):**

| eq | frac=0.25 | 0.5 | 0.75 | 1.0 (R²) |
|----|----|----|----|----|
| h — targets active | (diverged) | 0.2% | 0.1% | 0.0%  (R²=**−0.38**) |
| s — targets active | 15.2% | 16.5% | 14.8% | 0.0%  (R²=**−0.38**) |

- **The frozen clamp holds under `none` for h** — targets stay ~0% active (vs ~10–29% in the *unfrozen* run):
  freezing the master makes the clamp materially harder to escape without rescue pressure. For **s** it still
  leaks ~15% (equation-dependent — the `s` dynamics let some targets escape via built-up excitation).
- **`frac=1.0` breaks the task** (R² ≈ **−0.38**, worse than predicting the mean) under `none` — an immovable
  all-unit clamp prevents learning, close to the extreme-case prediction.

**`frm` arm — NOT interpretable (n=1/cond, survivorship-biased).** The surviving `frm` nets show 100% target
active with genuine participation (~0.09–0.13) and, at frac 0.25–0.75, a solved task (R²≈0.83–0.86); at frac=1.0
even the survivor fails (R²=−0.38). Taken at face value this would say `frm` *still* rescues the frozen-clamp
targets at frac<1.0 — plausibly by a **different** route than the unfrozen case: since the master can no longer
be tamed, `frm` would have to drive enough compensating **excitation** onto the targets to overcome the fixed −5
inhibition. But with only one non-diverged net per `frm` condition (the seed that happened to train stably), this
is a hint, not a result.

**Why divergence is worse when frozen.** In the unfrozen run `frm` could suppress the destabilizing, over-cap
master (that was both the rescue route *and* a stabilizer). Freezing removes that outlet, so the hyperactive
master (peak 1.0 ≫ cap 0.3) and its deep −5 currents persist and the `frm`-driven dynamics blow up more often
(22 NaN vs 12 unfrozen), concentrated in `frm`.

**Verdict / next step.** The freeze mechanism works and the `none` results are informative (clamp holds for h;
frac=1.0 kills the task). But the **central question — can `frm` rescue a genuinely gradient-proof clamp — is not
yet answered**, because the `frm` arm mostly diverged. Needs a **stabilized rerun** to get ≥3 valid `frm` nets/cond:
lower `master_ctx_drive` toward the cap (~0.3 so the master isn't over-cap) and/or `master_inhib_strength`,
tighten `max_grad_norm` (50→~5), and add seeds. (See `TODO.md`.)

**Stabilized rerun submitted (job `5116848`, `CDDM_f4b706_masterinhib_frozen_gamma`).** First stabilizer tried
(Pavel's suggestion): turn the **cubic term back on** (`gamma=0.1`) — the `-gamma*x^3` soft-saturation bounds the
runaway activity that produced the NaNs. It only bites at large positive `x`, so it does not lift the dead (`x<0`)
targets and therefore leaves the rescue question intact (only removes the blow-ups). Caveat: `gamma≠0` departs from
the gamma=0 experiment line, acceptable for a mechanism probe. Smoke test confirmed targets 100% silent and the
master saturating slightly (peak 1.0→0.92). If gamma alone doesn't cut the divergence enough, next levers are a
lower `master_ctx_drive` and a tighter `max_grad_norm`.

### Result (2026-07-07) — stabilized run answers it: frm overcomes even a frozen clamp, *unless the whole net is clamped*

`plot_masterinhib_rescue.py CDDM_f4b706_masterinhib_frozen_gamma`. gamma helped modestly (**18/48 NaN vs 22**) but
did not fix the `frm` arm — every `frm` cell still lost ≥1 net (mostly 1 valid; `s`/frac=1.0 kept all 3). Master
peak pinned at **0.921** (frozen + cubic-saturated). Reporting both peak-based active% and the more honest
**median participation** of the target set (`part`, active-mode ≈ 0.1):

| eq | penalty | frac=0.25 | 0.5 | 0.75 | 1.0 |
|----|----|----|----|----|----|
| h | none | 0.8% | 0.3% | 0.0% | 0.0% (R²=**−0.38**) |
| h | frm  | (div) | (div) | 100%, part 0.056, R²=0.84 | 100%, **part 0.004**, R²=**−0.38** |
| s | none | 9.5% | 8.7% | 9.7% | 0.0% (R²=**−0.38**) |
| s | frm  | 100%, part 0.10, R²=0.83 | (div) | 100%, part 0.12, R²=0.82 | 100%, part 0.11, R²=**−0.38** (n=3) |

**The answer (with the frac=1.0 nuance):**

1. **`frm` overcomes even the frozen, gradient-proof clamp at frac<1.0 — genuinely.** Targets reach real active-mode
   participation (`s`: 0.10–0.12; `h`/0.75: 0.056) and the task is solved (R²≈0.82–0.86). Since the master can no
   longer be tamed (frozen, verified), `frm` must be routing around it: driving the *non-clamped* units to build
   **compensating excitation** onto the dead targets, strong enough to overcome the fixed −5 inhibition and pull
   them active. So freezing the inhibitor does **not** make the targets unrescuable — as long as a functional
   surrounding network remains, `frm` recruits it to rescue them. (The stark `none` contrast — h ~0%, s ~9% — is
   clean at 3 nets/cond and confirms the clamp genuinely holds without `frm` pressure.)
2. **frac=1.0 is the true failure regime — this confirms Pavel's thought experiment.** With *every* unit clamped
   there is no scaffold left to build compensating drive, and the task cannot be solved under either penalty
   (R²≈**−0.38**, a degenerate constant-output solution, consistent across the 3 `s`/1.0/frm nets). `frm` still
   forces *nominal* threshold-crossing — genuinely for `s` (part 0.11) but negligibly for `h` (part 0.004, i.e.
   not really rescued) — yet the network is functionally dead either way.

**Synthesis of the whole rescue arc.** Per unit, a truly-dead ReLU gets no direct gradient (Pavel's argument holds
at the single-unit level). But rescue is a *network-level* phenomenon: `frm` makes "all units active" the trained
solution and reaches it from any init, and it can even dismantle a structured, gradient-proof silencing mechanism
by recruiting the rest of the network to compensate. The **only** way silence survives `frm` is to remove the
network's capacity to compute entirely (clamp everything, frac=1.0) — which is not a per-unit property, and which
also destroys task performance. In short: **you cannot keep an individual unit silent under `frm` if the rest of
the network still works.**

**Caveats.** The frac<1.0 `frm` cells rest on 1 surviving net each (gamma cut but didn't eliminate divergence),
though they are mutually consistent and agree with the unfrozen (100% rescue) and gamma=0-frozen survivors; the
frac=1.0 failure is solid (`s` n=3). The peak≥0.01 "active%" overstates rescue at frac=1.0 (see `h` part=0.004) —
participation is the honest readout. `gamma=0.1` regime. Firming up the frac<1.0 `frm` counts would need a further
stability pass, but the qualitative conclusion is already supported.

### Root cause of the NaN divergence (2026-07-07) — a forward-dynamics instability frm builds; the cubic is not the cause

Diagnosed from diverged nets' saved `LossBreakdown.json` / `GradsRaw.json` (no Spock needed), in **both** the
gamma=0 and gamma=0.1 runs. (An earlier note pinned this on the cubic term — corrected here after checking: the
gamma=0 runs, which have *no* cubic term, diverged too, with the identical signature.)

**The signature is the same everywhere and is sudden, not gradual.** Losses are *normal* until the last finite
step and then go NaN in a *single* step (γ=0.1 net: iter 1595 task 0.090/frm 0.109 → iter 1596 NaN; γ=0 net: iter
801 → 802 NaN), and the gradient norms just before are **modest** — `g_task ≈ 0.1–0.3`, `g_fr_magnitude ≲ 1.4`,
nowhere near `max_grad_norm=50`. So it is **not** a gradient explosion and **not** gradual weight growth, and
gradient clipping cannot catch it — it is a **forward-pass dynamical instability of the learned weights**, which
crosses a stability boundary in one training step and then overflows within the 300-step trial.

**The common cause: `frm` builds a self-exciting recurrent loop with gain > 1.** To activate the deeply-inhibited
targets against the fixed −5 clamp, `frm` drives the network to build strong compensating excitation onto them.
That creates recurrent loops whose effective gain can exceed 1; the continuous system `dx/dt = −x + W·ReLU(x)+…`
is then unstable (activity grows), and explicit forward Euler (`x_{t+1}=x_t+α(−x_t+W_rec·ReLU(x_t)+input−γx_t³)`,
`α=dt/τ=0.1`) integrates that growth over 300 steps. What (if anything) bounds it depends on γ:

- **γ=0 (no saturation):** nothing stops it — a loop with gain > 1 grows unbounded → overflow → NaN. Verified
  numerically: the self-exciting map `x ← x + α(−x + g·ReLU(x))` stays bounded for g ≤ 1 but blows up for g > 1
  (g=1.5 → ~10⁶ in 300 steps).
- **γ=0.1 (cubic on):** the cubic *does* bound moderate growth (helping — NaN rate 22→18), **but** the cubic
  itself is unstable under explicit Euler for large x: `x ← x + α(−x − γx³)` decays for x₀ ≤ 13 and diverges for
  x₀ ≥ 14 (threshold `√(2/(αγ)) ≈ 14`). So γ trades unbounded linear growth for a bounded-then-overshoot regime —
  a partial fix, not a cure.

**This explains every feature:** **sudden** (a stability-boundary crossing, not growth); **disproportionately
under `frm`** (`frm` is what builds the destabilizing excitation; `none` doesn't, so `none` almost never diverges);
**why γ only helped 22→18** (it removes the unbounded route but adds the cubic-overshoot route); and **why
`max_grad_norm` is irrelevant** (the blow-up is in the forward dynamics, gradients are ~1). The divergence and the
rescue are two faces of the same thing — `frm` overcoming the clamp by building excitation, pushed to the point of
instability.

**Correct fixes (for any firm-up rerun), most-direct first:** (1) **milder clamp** — `master_inhib_strength`
−5→−2 (still silences the targets) so far less compensating excitation is needed and loop gains stay < 1 — attacks
the source; (2) **hard-bound the state** in `forward` (clip `x` / saturating map) so no loop can overflow; (3)
**smaller integration step** (dt↓ / τ↑, or sub-step Euler) — raises the explicit-Euler stability threshold; (4)
lower `lambda_frm` so the excitation is built less aggressively. Gradient clipping and γ tweaks act on the wrong
layer.

**Fix search (2026-07-07), by cheap 16–32-job frm-only tests before any full run:**
- **Milder clamp `−2`:** did **not** help — same ~50% NaN (6/12), only *delayed* onset (iter ~4430 vs 800–1600).
  Lowering the clamp just changes how long `frm` takes to build a supercritical loop; it still gets there.
- **`dt=0.5` (finer integration; α 0.1→0.05, raising the loop-gain boundary g<11→g<21), two γ variants:**
  - `dt=0.5 + γ=0.1` → **3/16 NaN (~19%)** — a big drop from ~50%.
  - `dt=0.5 + γ=0` (dt alone) → **16/16 NaN (100%)** — *worse*.
  So you need **both**: γ provides the bounded fixed point (γ=0 has none → any gain>1 grows unbounded, and the
  larger per-step noise at small α, ∝√(2/α), makes γ=0 blow up every time), and the smaller step keeps the
  explicit-Euler integration of that cubic stable. This also corrects the earlier "γ value doesn't matter" note:
  γ=0 vs γ>0 matters enormously (bounded vs unbounded); it's only the *magnitude* of γ>0 that doesn't move the
  boundary. **Chosen fix: `dt=0.5 + γ=0.1`.**

**Full stabilized rescue run submitted (job `5127540`, `CDDM_b5fafb_masterinhib_frozen_dt05`).** `dt=0.5 + γ=0.1`,
frozen master, clamp −5; 64 jobs = 2 eq × 4 frac × (3 `none` + 5 `frm`) seeds (5 `frm` seeds → ~4 valid/cond after
the ~19% divergence; `none` never diverges so 3 suffice). Config
`rnn_relu_Dale_masterinhib_frozen_dt05_g01.yaml`, launcher `SilentReLU_masterinhib_frozen_dt05_N1000.slurm`,
wall 12 h (dt=0.5 ≈ 6 h/job measured). This should finally give a clean, well-powered read on whether `frm`
rescues a genuinely gradient-proof clamp (the earlier answer — yes at frac<1.0, no at frac=1.0 — but on n≈1).

### Result (2026-07-08) — well-powered, and it CONFIRMS the thin answer

Stabilization worked: only **6/61 nets diverged (~10%)**, and every `frm` condition retained **3–5 valid nets**
(vs n≈1 before). Master frozen (peak 0.921). `plot_masterinhib_rescue.py CDDM_b5fafb_masterinhib_frozen_dt05`.
Target-active% (peak≥0.01), target median participation, and R² (mean over valid nets):

| eq | penalty | frac=0.25 | 0.5 | 0.75 | 1.0 |
|----|----|----|----|----|----|
| h | none | 0.8% | 0.2% | 0% | 0% (R²=**−0.38**) |
| h | frm  | 100%, part 0.073, R²=0.84 | 100%, 0.042, 0.84 | 100%, 0.019, 0.85 | 100%, 0.048, **R²=−0.38** |
| s | none | 0% | 0% | 0% | 0% (R²=**−0.38**) |
| s | frm  | 100%, part 0.105, R²=0.83 | 100%, 0.081, 0.85 | 100%, 0.089, 0.83 | 100%, 0.094, **R²=0.08** |

The conclusion from 2026-07-07 holds, now robustly:

1. **`frm` overcomes even the frozen, gradient-proof clamp at frac<1.0** — 100% of targets active with genuine
   participation (`s`: 0.08–0.10; `h`: 0.02–0.07), task solved (R²≈0.83–0.86), consistent across 3–5 nets/cond.
   Since the master is immovable (frozen, verified: peak fixed at 0.921), `frm` rescues by recruiting the
   *non-clamped* units to build compensating excitation onto the dead targets.
2. **frac=1.0 (whole net clamped) fails the task** — with no scaffold left, the network cannot solve CDDM under
   `frm` (h R²=**−0.38**, s R²=**0.08**; both essentially failed) even though `frm` still forces the targets to
   nominal activity. This is the genuinely-unrescuable regime — Pavel's thought experiment confirmed.
3. **Cleaner `none` arm at dt=0.5:** the clamp now holds `s` targets at exactly 0% (they leaked ~9–15% in the
   dt=1 runs) — the finer integration removed the spurious leak, so the `none` control is unambiguous.

**Bottom line for the whole rescue arc:** a single dead ReLU has no direct gradient (per-unit), but silence is a
*network-level* property — `frm` makes "all units active" the trained solution and will dismantle even a
structured, gradient-proof silencing mechanism by recruiting the rest of the network, **unless** you remove the
network's capacity to compute entirely (frac=1.0), which also destroys the task. You cannot keep an individual
unit silent under `frm` while the rest of the network still works. (Method note: getting here required fixing an
explicit-Euler forward-integration instability — `dt=0.5 + γ=0.1` — not the clamp magnitude or gradient clipping.)

## 2026-07-25 — participation trajectories during training: *when* does the silent mode appear? (submitted)

### Why

Every result above is an **endpoint**. We know the two ends of the story — at initialisation all units sit in one
narrow participation band (~0.05–0.08, 0% silent; 2026-07-01) and after 30k iterations the population is bimodal
with ~45–55% in a near-zero mode under `none`/`rws` and a single active mode under `frm` (2026-06-26). The path
between them was never observed, so four questions stayed open:

1. **When** does the bifurcation happen — early (while task loss is still dropping) or as a slow drift after the
   task is already solved?
2. **Is it reversible?** Does a unit that falls below the silent line ever climb back under `none`?
3. **What does `frm` do in time** — hold the population up from iteration 0 (prevention), or let it start
   splitting and then pull it back (resurrection)?
4. **Why does `rws` fail where `frm` succeeds?** Does it accelerate the descent or leave the `none` trajectory
   unchanged? (Answering this needs the `rws`-only *and* `frm`-only cells, which is why the grid is 4 conditions,
   not 2.)

This is the "logged training" item — the last HIGH entry in [`TODO.md`](../TODO.md).

### Implementation (commits `5b74972`, `d29ed5a`)

Per-unit participation is logged **during** training, on a **noise-free** forward pass, every `track_every`
iterations:

- `Trainer.participation_from_states_` — `std(fr) + 0.9-quantile(|fr|)` pooled over (time, trials), with
  `fr = activation(states)` for `equation_type="h"` and `states` for `"s"`. Deliberately identical to
  `PerformanceAnalyzer.plot_participation`, so the trace is comparable to every participation figure in this
  document. (Not to be confused with the pre-existing `Trainer.get_participation_`, which serves the dropout
  controller and uses `|x|` in both terms without the activation.)
- `Trainer.track_participation_` — one extra `RNN(input, w_noise=False)` forward, taken in `run_training`
  **before** the train step, so snapshot 0 is the untrained network. Overhead ≈ 3% wall clock at
  `track_every=10`. The noise-free pass is the point: the training pass runs at σ_rec = 0.05 and would not be
  comparable to the offline figures.
- Config: `trainer.track_participation` (default `False`) and `trainer.track_every` (10) in
  `configs/trainer/trainer.yaml`; this experiment selects `configs/trainer/trainer_ptrack.yaml`, which is
  `trainer.yaml` with tracking on.
- Output per net: `{score}_ParticipationTrace.pkl` = `{"iters": [0, 10, ...], "participation": [array(N,) float32,
  ...]}` (~12 MB at 3000 snapshots × 1000 units; the same data is ~100 MB as indented JSON, hence pickle), plus
  `participation_trace.png` via the new `PerformanceAnalyzer.plot_participation_trace` (log₁₀ heatmap, units
  sorted by final participation). The pre-existing `participation.png` is unchanged.

**Validation (thresholds set before running).** Smoke test at N=50 / 500 iters produced exactly 50 snapshots, and
the last online (torch) snapshot matched the offline (`RNN_numpy` + `PerformanceAnalyzer`) readout **per unit** at
**r = 0.991** (threshold r > 0.99). These are two independently-written integrators, so this validates the metric,
not just the plumbing; the residual 0.009 is the 10 training steps between the last snapshot and the saved weights.

### Cluster note — the sweep moved to Della

Spock's `scotty-l40s` **lost its SLURM client** in the 2026-07-24 Rocky Linux 9.8 update: `rpm -qa | grep slurm`
returns only `slurm-example-configs-24.05.4-1.sdl9.2`, and no `sbatch`/`squeue`/`sinfo` exists anywhere on the
filesystem, while `/etc/slurm/slurm.conf` (ClusterName `Spockmk2`) and a running `munge` show the host is still
meant to be a submit client. Needs a sysadmin package reinstall. **This sweep therefore runs on Della**, which is
also why the launcher differs from every previous one in this document (`--gpus-per-node`, `--mem-per-gpu`, and
`--time=2:30:00` — under 2 h silently lands on the `gputest` QOS with a 3-concurrent-job cap).

### Cluster test run (Spock, direct on scotty's GPU — no scheduler)

One net before committing to the sweep: eq `h`, N=1000, γ=0, no penalties, **3000** iterations. 7 min 43 s
(**0.154 s/iter** on an L40S), r² = 0.79, trace exactly 300 × 1000 float32 = 1.2 MB. Output in
`data/trained_RNNs/CDDM_ptrack_TEST/`. It already shows three things:

1. **The bifurcation is early and then frozen.** The split completes by iteration **~400–600** and the silent
   fraction is flat at ~55% from there to 3000 — consistent with the 44–47% known for h/`none` at 30000, i.e.
   essentially nothing changes over the remaining 27000 iterations. Supports the "fate decided early" prediction.
2. **A global collapse comes first.** Every quantile crashes to ~3e-4 within the first ~20 iterations — the
   *whole* network goes quiet — and then the eventual-active subset climbs back out by iteration ~300 while the
   rest stays down. This is **not** the predicted picture of half the units drifting downward; it is
   collapse-then-partial-recovery, and it was not visible in any endpoint measurement.
3. **Silencing deepens in discrete steps.** A second sharp event at iteration ~2150 drops q5/q25/q50 by another
   half-decade (a clean vertical boundary in the heatmap).

**Threshold caveat for the analysis:** the init band straddles the 0.05 inter-mode dip (init median 0.0504, range
0.044–0.060), so "fraction < 0.05" reads a meaningless 43% at iteration 0. Use `< 0.01` (final: 53%) or a per-net
dip when scoring the trace.

### The sweep (submitted — Della array `11609846`, commit `d29ed5a`)

40 jobs = **2 equations {h, s} × 4 penalties {none, rws, frm, both} × 5 seeds**, N=1000.

| Axis | Config key | Values |
|---|---|---|
| Equation type | `model.equation_type` | `h`, `s` |
| Sparsity penalty | `trainer.lambda_rws` | `0`, `0.05` |
| FR-magnitude penalty | `trainer.lambda_frm` | `0`, `0.2` |
| Seeds | `seed="random"`, 5 array reps | 5 |

Fixed: `configs/model/rnn_relu_Dale.yaml` (**γ=0**, ReLU slope 1.0, `dt=1`, `tau=10`, sticky Dale boundary,
`bias_range=[0,0]`, `spectral_rad=1.2`, σ_rec=σ_inp=0.05), `+experiment=silent_units_N1000` (N=1000,
`max_iter=30000`, empty trainer tag), `trainer=trainer_ptrack` (`track_every=10` → 3000 snapshots/net). All
resolved values were verified on Della with `--cfg job` before submitting. These are **exactly** the settings of
the `CDDM_4a031e_g0` sweep, whose endpoints are known (h/`none` 44%, s/`none` 55%, `fr`/`both` 0%) — so the final
snapshot of every net must reproduce that distribution, which is the built-in correctness check on the run.

Launcher: [`slurm/SilentReLU_ptrack_gamma0_N1000_della.slurm`](../slurm/SilentReLU_ptrack_gamma0_N1000_della.slurm)
(`--array=1-40`, `--gpus-per-node=1`, `--mem-per-gpu=32G`, `--time=2:30:00`; task 1 copies
[`docs/experiments/participation_trace.md`](experiments/participation_trace.md) into the sweep folder as
`EXPERIMENT.md`, stamped with the array ID and commit). Output →
`/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs/CDDM_ptrack_g0/EqType=<eq>_N=1000_LmbdRWS=<rws>_LmbdFR=<frm>/`.
Submitted **2026-07-25 15:53 EDT**; task 1 started 2 minutes later on `della-l03g14`. ~80 min/job expected.

### Predictions (recorded before the results)

- **`none`:** most of the eventual silent set crosses the dip within the first ~2–4k of 30k iterations,
  coincident with the steepest part of the task-loss curve, and does not return (<5% of units that stay below the
  dip for ≥500 iterations climb back). *The 3000-iteration test already supports the timing, with the collapse
  even earlier than predicted, and adds the unpredicted global-collapse-first phase.*
- **`frm` / `both`:** participation rises from the init band toward the cap and no unit descends below the dip.
- **`rws` only:** nearly indistinguishable from `none`, slightly faster/deeper descent.

**Falsifiers.** A descent spread gradually over all 30k iterations kills "fate is decided early". Units churning
in and out of the silent set kills the "fate" framing altogether — silence would be a dynamic state, not an
outcome. `frm` nets that dip and then recover would contradict the prevention result of 2026-07-01.

### Read-out (planned)

Analysis script `trainRNNbrain/experiments_and_analysis/plot_participation_trace.py` (to be written):
(1) trace heatmap per condition; (2) 5/25/50/75/95th percentile bands vs iteration; (3) silent fraction vs
iteration with the task loss from `*_LossBreakdown.json` on a twin axis; (4) per-unit crossing statistics — first
iteration below the dip, total time below, number of upward re-crossings (the reversibility answer);
(5) endpoint validation against the offline participation, per unit, r > 0.99.

## 2026-07-25 — trainable-bias control: does the silent mode survive a learnable offset? (CANCELLED — replaced by the unconstrained version, see the 17:00 entry)

> **Status: cancelled and superseded.** Array `11610299` was cancelled at ~16:50 EDT after 4 of 40 tasks
> had run 9–13 minutes; no data was written (all outputs are saved at the end of training), so nothing
> was lost. The reasoning below — why a trainable bias is the right control, how the ±1 range was
> measured, the `bias_init` implementation and the DC-offset caveat — is unchanged and carries over to
> the replacement sweep, which runs the same bias manipulation on **unconstrained** networks
> (`CDDM_ptrack_g0_nodale_trainablebias`). The Dale+bias cell is simply not being collected.

### Why

Every silent-unit result in this document was obtained with `bias_range=[0,0]` — **no bias at all**. A ReLU unit
whose total input is negative at every timestep then has nothing to lift it into the active range, so the standing
objection is that the ~45–55% silent population is an artifact of an unusually constrained model rather than a
property of trained RNNs. A trainable bias is arguably the more standard architecture. This is the MED-priority
"trainable bias" item in [`TODO.md`](../TODO.md), and the most obvious reviewer question left open.

- If silence collapses → every claim narrows honestly to *bias-free* ReLU-Dale networks.
- If it persists → the result is much stronger and the objection is closed.

### Choosing the bias range — measured, not guessed

The `h` dynamics are `dx/dt = −x + W_rec·r + W_inp·I + b`, so at steady state `x* = drive + b`: the bias is
commensurate with the state and directly comparable to how negative a silent unit sits. Measured on a trained
`h`/`none` N=1000 net from `CDDM_4a031e_g0` (43.3% silent):

| | time-averaged state `x` |
|---|---|
| silent units (n=433) | median **−0.187**, p5–p95 −0.238 … −0.156, most negative **−0.296** |
| active units (n=567) | median −0.154, p95 +0.087 |

So **`b = +0.30` lifts 100% of the silent units to threshold** (+0.24 lifts 95%), against an active-unit median
peak rate of 0.23 and `cap_fr = 0.3`. The range is set to **±1** — a rail, not a prior: the optimum sits at
~0.2–0.3, well inside, so training is unconstrained in practice and "the bias could not reach far enough" is not
an available objection. (±0.3 would have been the measured-sufficient alternative; ±1 was chosen for robustness
to that critique.)

### `bias_init` — new option, so this stays a one-variable change

`RNN_torch` previously tied the bias *initialisation* to the range: any non-degenerate `bias_range` seeded the
biases **uniformly over that range** (`RNN_torch.__init__`). With ±1 that would start the network with offsets 5×
the drive scale — a different initial condition, breaking both the one-variable comparison and the established
"0% silent at init" fact. Added `bias_init` (default `"uniform"` = legacy behaviour, unchanged for every prior
config; `"zeros"` used here). Verified: with `bias_init="zeros"` the initial weights are **bit-identical** to the
bias-free baseline at the same seed, the bias is a trainable Parameter starting at exactly 0, an invalid value
raises, and in a 30-iteration run every bias moves off zero and stays inside the clamp
(`Trainer.enforce_bias_range_`).

### Pre-registered readout — a bias can fake participation

A unit with `b = 0.3` and no task input fires at a **constant** 0.3: peak rate 0.3, `q0.9(|fr|)` 0.3 — healthy
participation — while `std(fr) ≈ 0` and it carries no information. That is a DC offset, not participation, and
with `frm` on it is the *cheapest* way to satisfy the penalty, so the `frm` cells are expected to do exactly this.
Recorded before results: **report `std(fr)` over (time, conditions) separately from participation, and count a
unit as rescued only if its `std` rises into the active population's range.** A "0% silent" number based on peak
rate alone is uninterpretable in this sweep.

### The sweep (submitted — Della array `11610299`, commit `1492041`)

40 jobs = 2 equations {h, s} × 4 penalties {none, rws, frm, both} × 5 seeds, N=1000, γ=0, 30000 iterations,
participation tracked every 10 iterations. Identical to the `CDDM_ptrack_g0` grid submitted earlier today, with
`model=rnn_relu_Dale_trainablebias` (`bias_range: [-1, 1]`, `bias_init: zeros`) as the **only** difference — so
that sweep is this one's control.

Config `configs/model/rnn_relu_Dale_trainablebias.yaml`; launcher
[`slurm/SilentReLU_ptrack_bias_gamma0_N1000_della.slurm`](../slurm/SilentReLU_ptrack_bias_gamma0_N1000_della.slurm);
descriptor [`docs/experiments/participation_trace_bias.md`](experiments/participation_trace_bias.md).

**Folder separation (the two sweeps must not be mixed):**

| Folder | Model |
|---|---|
| `CDDM_ptrack_g0` | bias-free (`bias_range: [0,0]`) — the control |
| `CDDM_ptrack_g0_trainablebias` | trainable bias (`bias_range: [-1,1]`, zero init) |

both under `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs/`, each with its own
`EXPERIMENT.md` written by array task 1.

### Prediction

Silence **persists** at a broadly similar level (~40–55% under `none`/`rws`), because the 2026-07-01 tracking
showed the silencing is created during training and only weakly predicted by init activity (`corr ≈ 0.3–0.5`) —
a learnable offset gives units a way up, but nothing pushes them to take it. `frm`/`both` stay at 0%, partly for
the trivial DC-offset reason above.

**Falsifier:** if the silent fraction under `none` drops substantially (below ~15%), the bias-free constraint was
doing the work, and every claim in this document narrows to bias-free networks.

> **Repro caveat (same failure mode as the 2026-06-29 note; fixed for the next sweep — see the
> worktree note in the 16:45 entry).** Della's single working copy was advanced
> `d29ed5a` → `1492041` at 16:20 EDT to submit this sweep, while array `11609846` (the bias-free sibling) still had
> 6 tasks pending — those tasks therefore imported `1492041`. The diff is a **no-op for them**: the only code change
> is the new `bias_init` branch, which a degenerate `bias_range=[0,0]` never reaches, and `rnn_relu_Dale.yaml` is
> untouched. Already-running tasks imported their source at start and are unaffected either way. The durable fix is
> one `git worktree` per sweep on the cluster rather than a single shared checkout.

## 2026-07-25 (16:45) — removing Dale's law and I/O positivity: is the silent mode an artifact of the constraints? (submitted)

### Why this was missing

Every network in this document is **Dale-constrained**, and "Dale" here bundles three separate
restrictions: (i) `W_rec` is sign-split (excitatory units excite, inhibitory units inhibit, ratio
`exc2inhR = 4`) with the sign re-projected after every optimizer step; (ii) the readout is restricted to
the excitatory subpopulation (the inhibitory columns of `W_out` are structurally masked to zero); and
(iii) `W_inp` and `W_out` are clamped non-negative. The 2026-06-29 boundary control (`sticky` vs
`reflective`) tested only two *implementations* of (i)'s projection — it never removed the constraint.
This matters twice over: most of the RNN literature uses unconstrained networks, so as it stands the
result describes a special case; and a sign constraint could plausibly *cause* silence, since a weight
cannot change role and a unit needing mixed-sign output is stuck.

### The finding that motivated it — silence is an *excitatory* phenomenon

Free diagnostic on existing trained nets (`CDDM_4a031e_g0`, `h`, N=1000, γ=0, 3 nets/condition),
splitting the silent population by unit type:

| Condition | overall silent | E units (800) | I units (200) |
|---|---|---|---|
| `none` | 43–45% | **53–55%** | **3.5–5.0%** |
| `rws` | 52–54% | 58–60% | 27–29% |

Under `none`, inhibitory units are almost **never** silent while more than half the excitatory units
are. This **falsifies the natural Dale hypothesis**: the excitatory-only readout gives inhibitory units
the weaker gradient path, so they should have been the ones to go quiet — the opposite happens. The
likely explanation is load-bearing redundancy — 200 inhibitory units supply the whole network's
inhibition at 4× weight, so each is individually indispensable, while 800 excitatory units are mutually
redundant and half can be dropped. This is a new descriptive result and belongs in the paper regardless
of the sweep; it is also a Dale-specific structure, which is precisely why the unconstrained control is
needed.

### Implementation — two independent switches (commit `a5bde03`)

- **`model.dale`** (default `true`). `false` → new `get_connectivity_unconstrained`: signed zero-mean
  weights, no E/I split, `dale_mask = None`, **every** unit reads out; identical 1/√N scale, zero
  diagonal and spectral-radius rescaling to the Dale version, so the two are comparable.
  `Trainer.enforce_dale_` is skipped.
- **`model.io_nonnegativity`** (default `true`). `false` → the `W_inp ≥ 0`, `W_out ≥ 0` clamps are
  skipped. Independent of `dale`, so "Dale recurrence with a signed readout" is one override away.
- `_constrained_weights` (the `reflective` path) and `get_params`/`set_params` honour both flags;
  networks saved before this commit have no flags and default to both-on.
- `run_experiment.py` grouped units by the sign of their outgoing recurrent weights for the sorted /
  clustered figures — meaningless without Dale, so it now falls back to a single group when
  `model.dale` is false.

**Verification.** (1) The Dale path is **bit-identical** to a pre-change snapshot at the same seed
(`W_rec`, `W_inp`, `W_out`, `dale_mask`) — no existing result is disturbed. (2) The unconstrained init
has `dale_mask = None`, 50% negative weights, **no zeroed readout columns**, spectral radius exactly
1.200, zero diagonal. (3) Over 30 training iterations the unconstrained net shows **648 `W_rec` sign
crossings** (Dale net: 0), 167 negative `W_inp` and 54 negative `W_out` entries — the projections really
are skipped. (4) The switches compose. (5) `get_params`→`set_params` round-trips the flags, and legacy
params default to both-on. (Two apparent anomalies were chased to ground and are pre-existing, not new:
the Dale net's "negative" `W_out` entries are `enforce_dale_` writing `eps·dale_mask = −1e-12` on
inhibitory columns, and the "entries at eps" count is the zeroed diagonal.)

### The sweep (submitted — Della array `11610813`)

40 jobs = 2 equations {h, s} × 4 penalties {none, rws, frm, both} × 5 seeds, N=1000, γ=0, 30000
iterations, participation tracked every 10 iterations. Identical to `CDDM_ptrack_g0` except
`model=rnn_relu_noDale` (`dale: false`, `io_nonnegativity: false`).

Config `configs/model/rnn_relu_noDale.yaml`; launcher
[`slurm/SilentReLU_ptrack_nodale_gamma0_N1000_della.slurm`](../slurm/SilentReLU_ptrack_nodale_gamma0_N1000_della.slurm);
descriptor [`docs/experiments/participation_trace_nodale.md`](experiments/participation_trace_nodale.md).
Output → `.../CDDM_ptrack_g0_nodale/`, kept separate from the two sibling sweeps
(`CDDM_ptrack_g0`, `CDDM_ptrack_g0_trainablebias`).

**Isolation from in-flight jobs (the fix for this morning's repro caveat).** This sweep runs from a
**separate git worktree**, `$HOME/trainRNNbrain_nodale`, so `$HOME/trainRNNbrain` — the checkout that
the pending tasks of arrays `11609846` and `11610299` will import when they start — is never touched.
Because the editable install resolves `trainRNNbrain` to `$HOME/trainRNNbrain`, the launcher prepends
the worktree to `PYTHONPATH` **and refuses to run** if `trainRNNbrain.__file__` does not resolve inside
it: training with the wrong code is exactly the failure this guard exists to prevent.

### Prediction

Silence **persists** at a broadly similar level (~40–55% under `none`/`rws`): the apparent driver is
that CDDM needs far fewer units than the network has (unpenalized nets solve it with an effective
~60–150 units at any N), which sign constraints do not touch. The E/I asymmetry disappears by
construction.

**Falsifier:** a silent fraction under `none` below ~15% would mean Dale was doing the work, and every
claim in this document narrows to Dale-constrained networks.

## 2026-07-25 (17:00) — the reference condition becomes the *unconstrained* RNN (submitted)

### Decision

Pavel's call, and it changes the paper's frame rather than just adding a control: **Dale's law, the
excitatory-only readout and the I/O sign clamps make these networks a biologically-motivated special
case, and a result stated only for that case is easy to dismiss as niche.** Most RNN work trains
unconstrained networks with biases. So the reference architecture becomes the unconstrained one, and
the Dale sweep becomes the constrained comparison rather than the main result.

Concretely: array `11610299` (Dale + trainable bias) was **cancelled** at ~16:50 EDT — 4 of 40 tasks
had run 9–13 min and written no data, since outputs are saved only at the end — and replaced by the
same bias manipulation without the constraints.

### The three-point constraint ladder now in flight

| Array | Folder | Architecture | Code (worktree) |
|---|---|---|---|
| `11609846` | `CDDM_ptrack_g0` | Dale + non-negative I/O, no bias — links to every earlier result | `1492041` (`~/trainRNNbrain`) |
| `11610813` | `CDDM_ptrack_g0_nodale` | unconstrained, no bias | `a5bde03` (`~/trainRNNbrain_nodale`) |
| `11610886` | `CDDM_ptrack_g0_nodale_trainablebias` | **unconstrained + trainable bias** — the field-standard architecture | `733c021` (`~/trainRNNbrain_nodalebias`) |

Each is 40 jobs (2 eq × 4 penalties × 5 seeds), N=1000, γ=0, 30000 iterations, participation tracked
every 10 iterations. If the low-activity population appears at all three rungs, no architectural
constraint explains it and the claim is about trained RNNs in general; if it collapses at the
unconstrained rungs, the phenomenon belongs to constrained networks and the paper narrows honestly.

Config `configs/model/rnn_relu_noDale_trainablebias.yaml`; launcher
[`slurm/SilentReLU_ptrack_nodalebias_gamma0_N1000_della.slurm`](../slurm/SilentReLU_ptrack_nodalebias_gamma0_N1000_della.slurm);
descriptor [`docs/experiments/participation_trace_nodale_bias.md`](experiments/participation_trace_nodale_bias.md).
The pre-registered readout is unchanged from the cancelled sweep: a constant bias fakes participation,
so **rescue is scored on `std(fr)`, not peak rate**.

**One worktree per sweep is now the standing practice.** Three concurrent arrays run three different
commits with no interference: `git worktree add`, plus a `PYTHONPATH` guard in each launcher that
**aborts the job** if `trainRNNbrain` does not resolve inside its own worktree. That guard is not
theoretical — the editable install resolves the package to `~/trainRNNbrain`, so without it a job
launched from a worktree silently trains with the other sweep's code.

### Caveat on the design

Cancelling the Dale+bias cell leaves the 2×2 of {Dale, unconstrained} × {no bias, bias} incomplete: we
will be able to say what a bias does *without* Dale, but not whether it interacts *with* Dale. That is
the right trade given the framing decision — the unconstrained+bias cell is the one a general reader
cares about — but if a reviewer asks specifically whether biases rescue Dale networks, that cell has to
be run.

## 2026-07-26 — first submission of the three sweeps: 55/120 nets, and two failure modes worth recording

### Outcome

| Array | Sweep | Usable nets | Lost | SLURM state |
|---|---|---|---|---|
| `11609846` | `CDDM_ptrack_g0` (Dale) | **18** | 22 | 18 COMPLETED, 22 TIMEOUT |
| `11610813` | `CDDM_ptrack_g0_nodale` | **18** | 22 | 18 FAILED, 22 TIMEOUT |
| `11610886` | `CDDM_ptrack_g0_nodale_trainablebias` | **19** | 21 | 19 FAILED, 21 TIMEOUT |

### Failure 1 — "FAILED" is misleading: the non-Dale nets are complete

`cluster_neurons` splits units by `dale_mask` into an E and an I group. The `dale=false` fallback added
that morning put **every** unit in the "positive" group, so the recursion into the empty group died on
`cannot reshape array of size 0`. The crash sits in post-processing **after** the config, params,
`LossBreakdown`, `GradsRaw/Scaled`, `ParticipationTrace.pkl`, `participation.png` and
`participation_trace.png` are all written — verified on disk: 18/18 and 19/19 of those nets have every
essential file. Only `avg_responses.png` and `intercluster_connectivity_matrices.png` are missing.
**No retraining was needed for those 37 nets.** Fixed by passing `dale_mask=None` when `model.dale` is
false, which `cluster_neurons` already handles as a single group.

### Failure 2 — the timeouts were not random: they were exactly the `frm` cells

Every `LmbdFR=0.2` condition came back empty in all three sweeps (20 jobs each), plus 5 stragglers.
Measured from the logs at N=1000 / 30000 iterations:

| Job type | s/iter | wall time for 30000 |
|---|---|---|
| no `frm` | 0.203 | 1 h 42 m ✓ |
| `frm`, unconstrained | 0.343 | ~2 h 51 m ✗ |
| `frm`, Dale, **on a MIG slice** | 0.632 | ~5 h 16 m ✗ |

Two compounding causes. (i) `frm` adds a backward pass per iteration — its gradient-norm monitoring
plus the task-safe projection — so `frm` cells are ~1.7× slower than `none`/`rws` cells. The 2:30
request was calibrated on a **non-`frm`** test run (0.154 s/iter on a Spock L40S) and never covered
them. (ii) Della's `gpu` partition mixes full A100s with `3g.40gb` **MIG slices** (`della-l01g3-12`, 10
of 89 nodes); the slowest job had landed on `della-l01g8`, a slice, and ran at half speed.

**Fixes:** `--time=7:30:00` (~2.6× the expected `frm` runtime) and `--constraint=nomig` to exclude the
slices — 79 of 89 nodes remain available, so queue access is barely affected. Both are now in all three
launchers, together with the measured numbers, so the calibration is not lost again.

### Resubmission (commit `5360791`)

Only the missing cells, 65 jobs, one array per sweep — the `frm` tasks map exactly onto array indices
`6-10, 16-20, 26-30, 36-40`, plus per-sweep top-ups for the non-`frm` stragglers (`seed="random"`, so
re-running an index simply adds another net to that condition):

```bash
cd ~/trainRNNbrain            && sbatch --array=1,6-11,16-20,26-30,36-40 slurm/SilentReLU_ptrack_gamma0_N1000_della.slurm         # 11635039, 22 jobs
cd ~/trainRNNbrain_nodale     && sbatch --array=1,6-10,16-21,26-30,36-40 slurm/SilentReLU_ptrack_nodale_gamma0_N1000_della.slurm   # 11635040, 22 jobs
cd ~/trainRNNbrain_nodalebias && sbatch --array=6-11,16-20,26-30,36-40   slurm/SilentReLU_ptrack_nodalebias_gamma0_N1000_della.slurm # 11635041, 21 jobs
```

All three worktrees were advanced to `5360791` first — safe here because the queue was empty, unlike
the mid-flight edit recorded in the 16:00 entry.

### Lesson

Calibrate wall time on the **slowest** cell of a grid, not a representative one, and check whether the
partition is heterogeneous before trusting a single timing measurement. A penalty that adds a backward
pass is a wall-time change, not just a science change.

## 2026-07-27 — RNN standardisation: re-basing the whole project on unconstrained networks

### Why

Pavel's call, and it is a framing decision rather than a control: **every constraint in the model
makes the result easier to dismiss as niche.** The silent-unit effect had been established only in
Dale-constrained networks with non-negative I/O weights, no bias, no self-connections, and a
custom gradient rule. A reader can wave all of that away as "a biologically-motivated special case".
Most of the RNN literature trains plain unconstrained networks, so from now on **the reference
architecture is a vanilla RNN** and the constrained variants become a supplementary comparison.

The 2026-07-26/27 results made this affordable: silence persists at 42–58% without Dale's law and
without I/O positivity, and a trainable bias changes nothing, so nothing is lost by dropping them.

### The audit — what was still non-standard

Auditing the *saved config of an actually-running net* (not the intent) turned up exactly four
deviations from a textbook continuous-time RNN, plus one deliberate choice:

| Deviation | Status |
|---|---|
| Dale's law on `W_rec` + excitatory-only readout | already switchable (`dale`), now **off** |
| `W_inp`, `W_out` clamped non-negative | already switchable (`io_nonnegativity`), now **off** |
| `W_rec` diagonal zeroed every step — **no self-connections** | new switch, now **on** |
| Penalty gradients projected to never oppose the task gradient | new switch, now **off** |
| Training noise σ_rec = σ_inp = 0.05, σ_out = 0.03 | **kept** — standard in neuroscience RNNs, stated in Methods |

Confirmed already standard: `dx/dt = −x + W_rec·ReLU(x) + W_inp·u + b` with α = dt/τ = 0.1, **γ = 0
so no cubic term**, spectral radius 1.2 init, `y_init = 0`, Adam 1e-3, weight decay 1e-6, grad clip
50, and no active penalties in the `none` cells.

### Parameters introduced (all default to the legacy value, so nothing already run is affected)

| Parameter | Default | Standard-RNN value | Effect |
|---|---|---|---|
| `model.dale` | `true` | **`false`** | signed `W_rec`, no E/I split, every unit reads out, `dale_mask=None` |
| `model.io_nonnegativity` | `true` | **`false`** | no `W_inp ≥ 0`, `W_out ≥ 0` clamps |
| `model.self_connections` | `false` | **`true`** | `W_rec` diagonal free and trainable instead of re-zeroed each step |
| `model.bias_init` | `"uniform"` | **`"zeros"`** | bias starts at 0, so widening `bias_range` adds a degree of freedom without changing the init |
| `trainer.task_safe_gradients` | `true` | **`false`** | plain descent on `task + Σ λ_k·penalty_k` |
| `trainer.monitor` | `true` | **`false`** | drops the per-penalty loss/gradient breakdown, saving one backward pass per active term |

New configs: `configs/model/rnn_relu_standard.yaml`, `configs/trainer/trainer_ptrack_plain.yaml`,
`configs/experiment/silent_units_std.yaml`. Because `monitor=false` was the only thing recording the
loss trajectory, `run_experiment` now **always** writes `{score}_TrainLosses.json` (~300 KB) so the
participation trace can still be aligned against task learning.

**Verification (each a test that could have failed).** Legacy defaults are **bit-identical** to a
pre-change snapshot at the same seed. With `self_connections=true` the diagonal trains to 0.397
instead of returning to zero. `task_safe_gradients` on vs off is **identical** when no penalty is
active — the projection is a no-op there — and differs once `frm` is on (max ΔW_rec 0.0045 over 15
iterations), proving the switch touches only penalty combination. Both paths smoke-tested end to end
through `run_experiment` with `monitor=false`.

### Re-running the experiments (submitted 2026-07-27 ~13:20 EDT, commit `1325549`)

| Array | Folder | Grid | Jobs |
|---|---|---|---|
| `11672037` | `CDDM_std_g0` | 2 eq × 2 λ_rws × 2 λ_frm × 5 seeds, N=1000 | 40 |
| `11672038` | `CDDM_std_g0_Nsweep` | 3 sizes {100, 250, 500} × 2 eq × {none, frm} × 5 seeds | 60 |

Both from their own git worktrees (`~/trainRNNbrain_std`, `~/trainRNNbrain_stdN`) with the
PYTHONPATH guard. `lr` keeps its runtime `(100/N)^0.333` rescaling so the new size curve is directly
comparable to the original 120-net sweep — N and lr co-vary by design, to be stated in Methods.
N=1000 for the size curve comes from the reference sweep's `λ_rws=0` cells.

**Predictions.** Silence persists at ~40–55% under `none`/`rws` with `frm` at 0% and R² ≈ 0.85;
self-connections rescue little, since a positive diagonal only helps a unit that is *already* above
threshold. The size scaling holds (near-0% at N=100 → ~50% at N=1000). **Falsifiers:** under `none`,
below ~15% silent would mean one of the just-removed constraints (most plausibly the zeroed
diagonal) was producing the effect; a flat curve across N would break the "recruits what it needs"
reading entirely.

**Design caveat, recorded now.** `CDDM_std_g0` differs from `CDDM_ptrack_g0_nodale_trainablebias` in
**two** ways at once — self-connections on *and* the gradient projection off. If the two sweeps
disagree, that comparison alone cannot say which change is responsible; isolating it would need one
extra cell (self-connections on, projection still on).

### Planned Round 2 — the remaining rescue pathways

To run on the standard architecture once Round 0 confirms the baseline. Each is a route by which
silent units could plausibly be revived; all use `none` unless stated, 5 seeds, N=1000.

| Axis | Grid | Jobs | Question |
|---|---|---|---|
| **Connectivity scale** | `spectral_rad` ∈ {0.6, 0.9, 1.6, 2.0} × 2 eq | 40 | Pavel's suggestion — is silence simply weak recurrent drive? Only ever checked *at init* (0% silent at every radius), never in trained networks |
| **Connectivity density** | `connectivity_density_rec` ∈ {0.25, 0.5} × 2 eq | 20 | sparser connectivity → fewer, more variable inputs per unit |
| **Activation** | softplus(β=25), leaky-ReLU × 2 eq | 20 | sharper than the Dale version: without I/O positivity, hard zeros are impossible *by construction*, so this isolates whether the low-activity population survives when exact zeros cannot occur |
| **Recurrent noise** | σ_rec ∈ {0, 0.01, 0.1} × 2 eq | 30 | σ=0 was a distinct ~80%-silent regime in Dale nets |

~110 jobs, ~150 GPU-hours. Trimming to 3 seeds on the exploratory axes roughly halves that; the
sensible order is to run whichever axis Round 0 makes most urgent rather than all four at once.

Still open and untouched by any of this: the **spare-capacity** question (§5.1 of `paper.md`) —
scale the *task* rather than the network, and show whether the distributed solution buys anything
(lesion/noise robustness, generalisation) using networks we already have.

## 2026-07-27 (15:30) — large-N benchmark: does the ACTIVE unit count saturate? (benchmark submitted)

### The objection this addresses

If a trained RNN leaves half its units silent, the obvious rebuttal to any rescue method is:
**"why bother making every unit compute — just train a bigger network and prune the silent ones.
Compute is cheap, so what are you actually solving?"** Answering it requires knowing how the number
of *active* units grows with N. Two regimes, with completely different consequences:

- **Growth.** Active units keep increasing with N. Then "train big and prune" genuinely works, and
  the contribution is a cheaper route to the same thing, plus whatever population-level differences
  the rescued networks show (`paper.md` §5).
- **Saturation.** Active units plateau at a task-determined ceiling. Then **pruning cannot get you
  there at all** — no network size yields a large active population, and activity regularization is
  the only route. That would be a much stronger result.

### What the existing data says — and why it cannot settle it

Active-unit counts from the original Dale size sweep (`CDDM_4a031e`, h/none, 5 nets/cell):

| N | 100 | 500 | 1000 |
|---|---|---|---|
| silent | 0.0% | 23.7% | 47.1% |
| **active** | **100** | **382** | **529** |

Two models fit these three points about equally well and diverge wildly beyond them:

| Model | Fit | N=2000 | N=5000 | N=10000 |
|---|---|---|---|---|
| **A: power law** | `active = 3.57·N^0.723` | 873 | 1695 | **2798** |
| **B: saturating** | `active = 1011·N/(N+911)` | 695 | 855 | **926** |

At N=10000 they differ by 3×, so the experiment is decisive. **The local exponent is already
falling** — 0.833 between N=100 and 500, then 0.470 between 500 and 1000 — which leans toward B, a
ceiling near ~1000 active units for CDDM. That is precisely why the global "N^0.72" summary should
not be trusted: it averages over a slope that is visibly decaying.

### Explicit prediction (recorded before the run)

**I expect the power-law summary `active ≈ 3.6·N^0.72` to OVERSTATE growth, and the truth to lie
closer to the saturating fit — a ceiling of order 1000 active units.** The alternative hypothesis is
sustained power-law growth. Concretely, at N=10000 the two predict **2798 vs 926** active units
(72% vs 91% silent). Anything below ~1200 active at N=10000 confirms saturation; anything above
~2000 confirms growth.

If saturation holds, the paper's answer to the objection becomes sharp: *pruning cannot deliver a
large active population at any size, because the ceiling is set by the task, not the budget.*

### Why this needs careful engineering — three separate failure modes

1. **Wall time, not memory, binds.** Cost scales as N² per timestep, 300 sequential steps, 450
   conditions. From 0.2 s/iter at N=1000: N=5000 ≈ 41 h and N=10000 ≈ 165 h for 30000 iterations —
   beyond `gpu-short` (24 h) and even `gpu-long` (6 days). **Mitigation:** the silent fraction is
   decided in the first ~400–600 iterations and frozen thereafter (measured in the traces; our N=1000
   test gave 53% silent at iteration 3000 vs ~55% at 30000), so large-N cells can run ~5000
   iterations — *with an N=1000 control at both 5000 and 30000 to prove the truncation is harmless*.
2. **Parameter saving breaks before the GPU does.** At N=10000, `W_rec` is 100 M floats and
   `run_experiment` writes `LastParams` *and* `BestParams` as indented JSON — ~2 GB each, and
   `jsonify` first materializes Python lists at ~24 bytes/float (~2.4 GB RAM per file).
   **Mitigation:** save large parameter sets as `.npz` (400 MB binary, seconds) above a size
   threshold.
3. **A hidden per-iteration cost.** `run_training` does `deepcopy(RNN.get_params())` on every
   training-loss improvement — nearly every iteration early on. At N=10000 that is a 400 MB
   tensor→numpy conversion plus deep copy per iteration; invisible at N=1000, potentially dominant
   at N=10000.

### The benchmark (submitted — Della job `11677325`, commit `6a1119a`)

`trainRNNbrain/experiments_and_analysis/benchmark_large_N.py` runs real `Trainer.train_step`
iterations on the production task and model at N ∈ {1000, 2000, 5000, 10000} and reports peak
allocated/reserved GPU memory, measured s/iter extrapolated to full runs, and the size the saved
JSON parameter files would reach. Nothing is submitted at scale until these numbers exist.

Queued behind our own jobs: the two standard-RNN arrays hold all 44 slots of the `gpu-short`
per-user cap.

> **Process note (a near miss).** The benchmark commit was checked out in `~/trainRNNbrain_std` —
> the worktree that array `11672037` was still using. It happened to be harmless (0 tasks pending;
> the only diff under `trainRNNbrain/` was the new benchmark file), but it is exactly the mistake the
> one-worktree-per-sweep rule exists to prevent. Benchmarks and analysis get their own worktree.

## 2026-07-27 (16:10) — standard-RNN results: size sweep and penalty sweep, all 100 jobs complete

Arrays `11672037` (reference, 40 jobs) and `11672038` (size sweep, 60 jobs) finished **100/100
COMPLETED** — clean exits, confirming the non-Dale clustering fix. Every cell holds exactly 5 nets
and 5 participation traces. Statistics computed on Della from the final trace snapshot
(`collect_stats.py` → `silent_stats_all.csv`, 220 networks including the earlier sweeps); figures by
[`plot_silent_summary.py`](../trainRNNbrain/experiments_and_analysis/plot_silent_summary.py).

**Two metrics throughout**, because a single threshold misleads (participation
`p = std(fr) + q_0.9(|fr|)`):
- **truly silent** — `p < 1e-6`; a ReLU unit that never fires has exactly `p = 0`.
- **scale-free** — `p < 5%` of the network's 95th-percentile participation; comparable across
  conditions whose overall activity scale differs.

### Figure 1 — silent units vs network size (standard RNNs)

![Silent units vs N, standard RNNs](../img/internal_figures/silent_vs_N_std.png)

| unpenalized | N=100 | N=250 | N=500 | N=1000 |
|---|---|---|---|---|
| h — truly silent / scale-free | 3% / 12% | 9% / 27% | 22% / 41% | **42% / 59%** |
| s — truly silent / scale-free | 3% / 7% | 7% / 22% | 19% / 40% | **40% / 59%** |

**Observation.** The size scaling survives de-constraining: silence rises monotonically with N in
both equation types, from a nearly clean N=100 (3% truly silent) to ~40% truly dead and ~59%
functionally silent at N=1000. **`frm` gives exactly 0 on both metrics at every size**, so the
rescue is not a large-N-only effect. The gap between the two metrics (~17 points at N=1000) is the
population that fires, but so weakly it is irrelevant beside the active units — visible only to the
scale-free criterion. The consequence for the "just train big and prune" objection is that the
*active* count (97 → 221 → 340 → 448 for h) grows with a **falling local exponent**
(0.899 → 0.621 → 0.398), pointing at a ceiling near ~750 active units rather than unbounded growth.

### Figure 2 — penalty configurations at N=1000 (standard RNNs)

![Silent units by penalty, N=1000](../img/internal_figures/silent_by_penalty_N1000_std.png)

| N=1000, truly silent / scale-free | none | rws | frm | both |
|---|---|---|---|---|
| h | 42% / 59% | 40% / **64%** | **0% / 0%** | **0% / 0%** |
| s | 40% / 59% | **22%** / 53% | **0% / 0%** | **0% / 0%** |

**Observation.** Only the firing-rate penalty rescues, and it does so completely — zero silent units
on both metrics, in both equation types. `rws` does not: in `h` it leaves the truly-silent count
unchanged while making the scale-free count *worse* (59% → 64%). The `s` case is the more
interesting one and is only visible because two metrics are reported: `rws` nearly halves the
truly-silent count (40% → 22%) while the scale-free count barely moves (59% → 53%). **It lifts units
off exact zero without making them participate** — cosmetic rescue, which a single hard threshold
would have scored as a genuine improvement.

### Figure 3 — constrained vs standard architecture (h equation, N=1000)

![Constrained vs standard](../img/internal_figures/silent_constrained_vs_unconstrained_h.png)

| h, N=1000 | none | rws | frm | both |
|---|---|---|---|---|
| truly silent — constrained vs standard | **6% vs 42%** | 46% vs 40% | 0 vs 0 | 0 vs 0 |
| scale-free — constrained vs standard | **56% vs 59%** | 65% vs 64% | 0 vs 0 | 0 vs 0 |

**Observation, and the reason both metrics are needed.** On the **scale-free** criterion the two
architectures are **indistinguishable** (56% vs 59% unpenalized; 65% vs 64% under `rws`) — the
phenomenon is not an artifact of Dale's law or I/O positivity, which is the point this figure exists
to make. On the **strict** criterion they look wildly different (6% vs 42%), but that gap is entirely
the I/O positivity floor: non-negative input weights with non-negative task inputs guarantee every
unit a small positive push, so no unit in a constrained network can be exactly zero. Same silent
population, different floor. Reporting only the strict metric would have suggested the constrained
architecture is nine times healthier, when it is merely nine times better at hiding it.

### Note on scope

**No networks have been trained at N=2000 or N=5000.** The only runs at those sizes were the
benchmark's 8 timed iterations for memory/speed sizing (job `11677325`). Trained networks exist for
N ∈ {100, 250, 500, 1000} only; the saturation-vs-growth question stays open until the large-N runs.

## 2026-07-27 (21:45) — strategy: how the saturation question gets decided (array `11686960` submitted)

Written down explicitly because the answer determines whether the paper has a reply to *"why bother —
just train a bigger network and prune the silent units"*, and because the decision rules must be
fixed before the data lands.

### The measurement

**Active units = N × (1 − silent fraction)**, with the silent fraction read from the **final
participation snapshot** of each network's trace, on **both** metrics:

- **truly silent**, `p < 1e-6` — unambiguous (a ReLU unit that never fires has `p = 0` exactly);
- **scale-free**, `p < 5%` of that network's 95th-percentile participation — the metric that is
  comparable *across network sizes*, which matters here because the activity scale differs with N.

Both curves are reported. If they disagree about saturation, that disagreement is itself the result
and gets reported rather than resolved by picking the more convenient one.

### The two hypotheses, fitted to N ≤ 1000 (standard RNNs, h, no penalty: 97, 221, 340, 448)

| Model | Fit | N=2000 | N=5000 | N=10000 |
|---|---|---|---|---|
| **A — power law** | `active = 4.55·N^0.665` | 710 | 1305 | **2069** |
| **B — saturating** | `active = 749·N/(N+672)` | 561 | 660 | **702** |

They differ by 3× at N=10000, so the experiment discriminates cleanly rather than requiring a
statistical argument.

### Decision rules, fixed in advance

- **< ~1200 active units at N=10000 → saturation (B).** Pruning cannot deliver a large active
  population at *any* size; the ceiling is set by the task, not the budget. `frm` at N=1000 already
  yields 1000 active units, which no amount of scaling could reach.
- **> ~2000 active units at N=10000 → growth (A).** Pruning works; the paper's argument must then
  rest entirely on the population-level differences (`paper.md` §5) rather than on reachability.
- **Between 1200 and 2000 → ambiguous.** Report as such and extend the curve rather than argue.

Secondary evidence, decided the same way: **the local exponent**. It has fallen monotonically so far
(0.899 → 0.621 → 0.398 across N = 100→250→500→1000). Continued decline toward 0 supports B; a
stabilisation near ~0.6–0.7 supports A.

### Two controls without which the answer is not interpretable

1. **Truncation control.** N=5000 and N=10000 train for 5000 iterations rather than 30000, because
   48 h exceeds the queue limit. The justification is that the silent fraction is decided by
   iteration ~400–600 and frozen — but that is a claim, so cells 0–2 of the array measure it: N=1000
   and N=2000 at 5000 iterations, plus N=2000 at 30000. **If 5000 iterations does not reproduce the
   30000-iteration silent fraction at both sizes, the large-N numbers become an upper bound and are
   reported as such.**
2. **Task-performance control.** A large network that simply failed to learn would trivially show
   many silent units, and the active-unit curve would be measuring training failure rather than
   capacity. So R² must be checked at every N before the curve is interpreted: if it degrades
   materially at N=5000 or N=10000, the corresponding points are not comparable to the smaller ones.

### Scope and known limits

3 seeds per cell (not 5) and h equation only — enough to separate a 3× difference, not enough for a
precise exponent. The fit is over four points spanning one decade; extrapolating two decades beyond
it is exactly why the N=10000 point is being measured rather than predicted. `s` equation and 5-seed
confirmation are cheap follow-ups if the answer lands near the ambiguous band.

## 2026-07-28 — metabolic-cost sweep: the standard regularizer never rescues, and at strength deepens the problem

Array `11687340`, 36/36 COMPLETED (4 λ × 3 sizes × 3 seeds, h equation, standard RNNs, 30000
iterations). λ=0 baselines come from the unpenalized cells of `CDDM_std_g0` / `CDDM_std_g0_Nsweep`.

![Silent units vs metabolic-cost strength](../img/internal_figures/silent_vs_metabolic.png)

| N | metric | λ=0 | 0.01 | 0.1 | 1.0 | 10.0 |
|---|---|---|---|---|---|---|
| 100 | truly silent | 3% | 0.3% | 0.3% | 2% | **20%** |
| 100 | scale-free | 12% | 12% | 17% | 33% | **59%** |
| 500 | truly silent | 22% | 23% | 21% | 22% | **32%** |
| 500 | scale-free | 41% | 41% | 41% | 50% | **69%** |
| 1000 | truly silent | 42% | 44% | 40% | 39% | 43% |
| 1000 | scale-free | 59% | 61% | 57% | 57% | 62% |

R² is **0.81–0.87 across every cell**, so nothing here is a penalty destroying the task.

**Observations.**

1. **No λ rescues anything.** Across four decades the silent fraction never falls meaningfully below
   baseline at any size. Compare `frm`, which drives it to exactly 0 at every size — the contrast is
   total, not a matter of degree.
2. **At λ=10 the penalty makes it substantially worse**, and most visibly where there was headroom:
   at N=100 the scale-free silent fraction goes **12% → 59%**, a five-fold increase, while R² only
   drops from 0.858 to 0.807. At N=500, 41% → 69%. The mechanism is the obvious one — `mean(fr²)`
   penalizes rate *magnitude*, so it pushes the whole population down and pushes the weakest units
   through the floor.
3. **At N=1000 it is flat** (39–44% truly silent across all λ). Not a contradiction: the network is
   already ~42% silent without any penalty, so there is little room left for the penalty to add. The
   effect appears where the baseline is low, which is why sweeping *sizes* as well as λ mattered.
4. **The two metrics separate at intermediate λ.** At N=100, λ=1 shows 2% truly silent but 33%
   scale-free silent: the penalty has pushed a third of the population into near-irrelevance without
   driving it to exact zero. A strict threshold alone would have reported this cell as healthy.

**Why this matters for the paper.** §2 previously rested on a prediction. It is now measured: the
activity regularizer this literature routinely uses does not solve the silent-unit problem, and
applied at strength it deepens it while the task is still solved. That is a stronger claim than
"here is another penalty", and it is the answer to the first objection any referee raises.

**Falsifier check.** The pre-registered falsifier was "any λ that drives silence toward 0 while
keeping R² ≈ 0.85 collapses §2." No λ at any size did that. §2 stands.

## 2026-07-28 (10:40) — 30000 iterations is not convergence: the horizon experiment (submitted)

### The problem

Every silent-unit number in this project was measured at a **fixed 30000 iterations**, on the
assumption that training had settled. Checking that assumption directly shows it is false, and in a
way that matters for a conclusion we already drew.

**The training loss has converged. The network has not.** Across the standard-RNN reference sweep,
the loss over the last 10% of training differs from the preceding 10% by only −0.4 to −0.6% at every
size — flat. But the hard-silent fraction is still climbing when training stops:

| N | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| loss change (last 10% vs previous 10%) | −0.46% | −0.38% | −0.47% | −0.56% | −0.61% |
| **hard-silent gained in the final 5000 iterations** | +0.20 pp | +0.92 pp | +2.86 pp | **+3.67 pp** | **+3.08 pp** |

Tracing one condition (standard, N=1000, no penalty) across the whole run makes the scale of it
plain — the silent fraction roughly **quintuples** after iteration 5000:

| iteration | 0 | 1000 | 5000 | 10000 | 20000 | 30000 |
|---|---|---|---|---|---|---|
| hard silent (`p < 1e-6`) | 4.4% | 5.4% | 7.4% | 13.9% | 30.9% | **41.5%** |
| `p < 0.01` | 21.2% | 23.6% | 28.9% | 37.3% | 48.0% | **54.7%** |
| scale-free | 21.3% | 29.0% | 39.8% | 47.4% | 52.2% | **58.1%** |

The network solves the task early and then goes on quietly switching units off — drift along a flat
loss manifold.

### Two corrections this forces

1. **"The bifurcation completes by iteration ~400–600 and is frozen thereafter" was wrong.** That
   claim came from the 3000-iteration test run, where the curve does look flat after ~600 *within
   that window*. Extrapolating a 3000-iteration window to 30000 was the error, and it propagated
   into the design of the large-N runs (which is why their truncation control failed). Any statement
   of the form "silencing is decided early" should be struck.
2. **"0% of units are silent at initialisation" is Dale-specific.** In standard unconstrained
   networks, **4.4% are hard-silent and 21% sit below the 0.01 line before a single training step**.
   With signed input weights a unit can start net-negative and never fire; I/O positivity is what
   guaranteed the old result. Training still creates most of the silence (4.4% → 41.5%), but not all
   of it.

### Why it changes the saturation conclusion, not just the bookkeeping

Residual drift **grows with N** (+0.2 pp at N=100 → +3.7 pp at N=1000). Larger networks therefore
sit further from their asymptote at any fixed iteration count, which **undercounts silence at large
N**, **overcounts active units at large N**, and biases the size curve toward "growth" — exactly the
result the large-N runs produced. The learning-rate scaling `lr·(100/N)^0.333` is the obvious
mechanism: a 10000-unit network needs ~2.2× as many steps to move its weights as far as a 1000-unit
one. So the evidence against saturation may be an artifact of undertrained large networks, and
spending ~185 GPU-hours on N=5000/10000 at 30000 iterations would buy a differently-wrong answer.

### The experiment (submitted — Della array `11706899`, commit `f5aa558`)

**12 jobs = 2 sizes × 2 weight-decay values × 3 seeds**, h equation, no penalties, standard RNNs,
**100000 iterations** (3.3× the previous horizon), participation tracked every 10:

| Tasks | N | `weight_decay` | seeds |
|---|---|---|---|
| 1–3 | 1000 | 1e-6 | 3 |
| 4–6 | 1000 | **0** | 3 |
| 7–9 | 2000 | 1e-6 | 3 |
| 10–12 | 2000 | **0** | 3 |

Runtime 4.0 h (N=1000) and 8.8 h (N=2000) at measured rates; ~76 GPU-hours. All 12 started
immediately.

**`weight_decay` is the second axis** because Adam's 1e-6 decay keeps pulling weights down after the
loss plateaus, making it a plausible driver of the late-phase silencing. If `wd=0` removes the drift,
the slow silencing is a regularisation artifact rather than something the task demands.

### What is measured

1. **New silent units per 1000 iterations** — the derivative of the participation trace. Needs no new
   logging: the trace has recorded every unit every 10 iterations since the first sweep.
2. **Relative parameter drift** `‖W(t) − W(t−Δ)‖_F / ‖W(t)‖_F` for `W_rec`, `W_inp`, `W_out`, `bias`
   — **new**, added to `Trainer.track_participation_` and stored under the `"drift"` key of the
   trace pickle (additive, so existing analysis scripts are unaffected). This is the direct test:
   the loss can be flat while the weights still move, and only this separates the two. Verified
   working — `W_rec` drift declines 0.054 → 0.040 over the first 50 iterations. (`bias` drift reads
   1.0 at the first snapshot because the bias starts at exactly zero; it settles immediately.)
3. **R² and the loss curve**, so "converged" is never confused with "stopped learning".

### What each outcome forces

- **Both curves flatten well inside 100000** → read off the horizon, scale it by N, and size the
  large-N runs from measurement instead of assumption.
- **Still moving at 100000** → the fixed-iteration protocol is unusable for cross-N comparison, and
  every silent-fraction number in this project carries the caveat *"at 30000 iterations"* rather than
  *"at convergence"*. The size sweep would have to be re-run to a convergence **criterion** (stop
  when new silent units per 1000 iterations falls below a threshold).
- **`wd=0` removes the late drift** → the late-phase silencing is a weight-decay artifact and the
  headline phenomenon needs restating in those terms. It would not erase the result — the silence is
  present at 30000 iterations either way, and `frm` still removes it — but it would change what the
  phenomenon *is*.

## 2026-07-28 (11:10) — do self-connections rescue silent units? No — the network trains them into self-*inhibition*

### Why it was worth asking

Allowing a unit to connect to itself is the **cheapest conceivable escape from silence**: a positive
self-weight lets a unit sustain its own activity with no help from the rest of the network. It was
also the last non-standard constraint in the architecture — the `W_rec` diagonal had been zeroed at
init *and* re-zeroed after every optimizer step — so if it mattered, a reviewer could fairly say the
whole phenomenon was an artifact of forbidding self-recurrence. `self_connections: true` has been
part of the standard architecture since 2026-07-27; this entry reports what it did.

### Test 1 — cross-sweep, confounded but bounding

`CDDM_ptrack_g0_nodale_trainablebias` (self-connections **off**) vs `CDDM_std_g0` (**on**), both
unconstrained with a trainable bias, N=1000, 5 nets/cell:

| eq / penalty | hard: off → on | Δ | scale-free: off → on | Δ |
|---|---|---|---|---|
| h / none | 41.6% → 42.4% | +0.9 | 59.6% → 58.6% | −1.1 |
| h / rws | 41.3% → 39.9% | −1.3 | 64.5% → 63.9% | −0.6 |
| s / none | 39.8% → 40.0% | +0.2 | 59.7% → 59.1% | −0.7 |
| s / rws | 22.5% → 22.5% | 0.0 | 53.1% → 53.1% | 0.0 |
| h,s / frm, both | 0% → 0% | 0.0 | 0% → 0% | 0.0 |

Every delta is within ±1.3 pp. **Caveat, recorded when these sweeps were submitted:** the two differ
in *two* ways — `self_connections` and `task_safe_gradients` — so a difference could not have been
attributed. But two simultaneous changes producing nothing means neither mattered, barring an
implausible exact cancellation.

### Test 2 — within-network, confound-free, and the more informative one

Everything below is measured *inside* networks that all had `self_connections: true`, so no
cross-sweep comparison is involved. Standard RNNs, h, N=1000, 5 nets:

| unpenalized (`none`) | |
|---|---|
| mean self-weight `W_rec[i,i]`, **silent** units | **−0.0073** |
| mean self-weight, **active** units | **−0.0601** |
| corr(self-weight, log participation) | **−0.51** |
| mean \|self-weight\| vs mean \|off-diagonal\| | **4.7×** larger |

**The correlation is negative, and strong.** Active units have *more negative* self-weights than
silent ones. The network does not use self-connections to keep units alive — it trains them into
**self-inhibition**, and the most active units are the ones inhibiting themselves hardest.

That is mechanistically sensible. A positive self-weight is a gain>1 loop on a single unit — the same
instability that produced the NaN divergences in the frozen-clamp arc (2026-07-07). Gradient descent
finds self-*inhibition* useful as a per-unit gain control and self-excitation dangerous. **The
cheapest escape from silence is one the network actively declines to take.**

The diagonal is not being ignored: self-weights are **4.7× the typical off-diagonal magnitude**, so
they are genuinely used, just not in the direction that would keep units alive. Under `frm` there are
no silent units left to compare and the correlation flips to a weak **+0.12** — once every unit is
forced active, the self-weight stops predicting participation.

### Conclusion

Self-connections join the list of interventions that do not rescue (intervention #7 in `paper.md`
§3). Two things follow. The residual worry that the zeroed diagonal was a hidden non-standard
constraint doing real work is settled — it was not, and we now know *why* rather than merely *that*.
And the "a unit could just excite itself" objection has an empirical answer: it could, it is allowed
to, and it chooses the opposite.

## 2026-07-28 (12:30) — convergence curves: how far from settled is each network size?

Direct answer to "does the participation stop changing?", computed from the participation traces of
the standard-RNN sweeps (h equation, **no penalties**, 5 nets per size, 30000 iterations). Two
measures over a trailing **1000-iteration window** (100 snapshots at `track_every=10`), bands are
95% t-intervals across the 5 networks.
[`plot_convergence_curves.py`](../trainRNNbrain/experiments_and_analysis/plot_convergence_curves.py).

![Convergence curves](../img/internal_figures/convergence_curves.png)

| N | new silent units / 1000 iters at the end | `‖Δp‖/‖p‖` at the end | decay exponent | extrapolated iterations to reach 1% |
|---|---|---|---|---|
| 100 | **0.05 ± 0.26** | 0.045 | −0.29 | ~5,600,000 |
| 500 | **3.30 ± 3.52** | 0.054 | −0.49 | ~871,000 |
| 1000 | **9.57 ± 7.90** | 0.055 | −0.57 | ~562,000 |

### What the two panels say

**Left — the silent population.** N=100 has genuinely stopped (0.05 ± 0.26 new silent units per 1000
iterations; the interval contains zero). N=500 is marginal (3.3 ± 3.5, interval contains zero).
**N=1000 is still gaining ~10 units per 1000 iterations at the end, with the interval clear of
zero.** The curve is *decelerating* — it peaks near ~27 per 1000 iterations around iteration 11000
and falls to ~10 by 30000 — but it has not reached zero, and larger networks are further from it.
Note the early negative excursion: in the first ~1500 iterations the silent count *drops* sharply,
the tail of the global collapse-and-recovery, before the steady silencing begins.

**Right — the participation vector.** The stricter test, and it is unambiguous: **no size has
converged.** At iteration 30000 the participation vector still changes by **4.5–5.5% per 1000
iterations**, at every N. The silent *count* can look flat while units continue to rearrange, which
is exactly what N=100 is doing — 0.05 new silent units per 1000 iterations, yet a 4.5% relative
change in participation.

### The uncomfortable implication

The relative change decays as a power law in iteration (exponent −0.29 to −0.57), not exponentially.
Extrapolating to a 1% criterion gives **0.5–5.6 million iterations** — 20× to 190× the current
horizon, i.e. weeks to months of GPU time per network. **On any practical horizon these networks
never fully stop changing.**

So "train to convergence" is not an available option, and the running 100000-iteration horizon
experiment will not deliver a converged state either — it will show how much further things move
with 3.3× the training. The honest resolutions are:

1. **Define convergence by a stated threshold** (e.g. "< 1 new silent unit per 1000 iterations", or
   "‖Δp‖/‖p‖ < 2% per 1000") and report the horizon required to reach it at each N — accepting that
   the horizon itself grows with N.
2. **Report every number as "at iteration X"**, and demonstrate that the *comparison* between
   conditions is stable across X even though the absolute values are not. This is the cheaper and
   probably more honest route: `frm` gives 0% silent from early on, so the `frm`-vs-`none` contrast
   is not in question — only the precise baseline value is.
3. For the size sweep specifically, **match training by drift rather than by iteration count**, since
   `lr ∝ N^(−1/3)` means a fixed iteration count buys less progress at larger N.

The saturation question (`paper.md` §6.2) is the one genuinely at risk, because it compares
*absolute* active-unit counts across N, and the residual drift is precisely the term that grows with
N. Option 2 does not rescue it; that comparison needs option 1 or 3.

## 2026-07-30 — STATED GOAL: is there an M*, a ceiling on active units for a given task?

Written as a goal statement rather than a result, because the next block of compute is aimed at
exactly one question and it is worth fixing what that question is, why it matters, and what would
answer it — before spending the hours.

### The question

For a given task, is there an **M\*** — an upper limit on the number of *active* units a trained
network converges to, **regardless of how large N is**? Or does the active count keep growing with N
without bound?

### Why it decides the value of the penalties

This is the argument that makes the whole project matter, so it should be stated plainly:

- **If M\* exists (saturation):** you cannot obtain a network with more than M\* active units by
  standard training at any size. Training bigger and pruning the silent units does not help — the
  ceiling is set by the task, not the budget. In that case the firing-rate + sparsity penalties are
  **the only route** to a network with M > M\* participating units, and they are therefore genuinely
  useful rather than merely tidy. `frm` at N=1000 already delivers 1000 active units; if M\* is ~750,
  no unpenalized network of any size can match that.
- **If there is no M\* (unbounded growth):** pruning works, the penalties are a convenience rather
  than a necessity, and the paper's weight must shift entirely onto the population-level
  consequences (`paper.md` §5) — that silent units distort dimensionality, selectivity distributions
  and cluster structure, and so corrupt the comparison between model and data.

Either answer is publishable. They are different papers.

### Why we cannot answer it yet

The current active-unit curve is measured at a fixed 30000 iterations, and **the networks are not
converged at that point — with the residual drift growing with N**:

| N | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| active units (30k, wd=1e-6) | 97 | 227 | 390 | 580 | 862 |
| local exponent | — | 0.93 | 0.78 | 0.57 | 0.57 |
| silencing rate still running at 30k | ~0 | — | 3.3 ± 1.5 | 9.6 ± 3.5 | ~12 |

Larger networks sit further from their asymptote, which **inflates their active count** and biases
the curve toward "growth" — which is exactly the answer we currently get (a power-law fit
`4.55·N^0.665` beats the saturating `749·N/(N+672)` at the one new point, N=2000). That bias has the
same sign as the effect we are trying to detect, so the current evidence cannot be trusted either
way.

### What has to be measured

**All of the drift channels, per model — not just the silent fraction.** The silent count is one
projection of a network that is still moving, and it can be flat while the network reorganises
underneath (measured: at N=100 the silent count is stationary while the participation vector still
changes 4.5% per 1000 iterations). So each run records:

| channel | quantity |
|---|---|
| weights | `‖W(t) − W(t−L)‖_F / ‖W(t)‖_F` separately for `W_inp`, `W_rec`, `W_out`, at lags 100 / 1000 / 10000 |
| direction | cosine between consecutive displacements, per matrix — ≈0 = jitter, >0 = still marching |
| participation | `‖p(t) − p(t−L)‖ / ‖p(t)‖` at the same lags |
| outcome | silent-unit count (participation < 1e-6) at every probe |

All reduced to scalars during training; no weights are written to disk.

### The design

1. **Characterise the drift curves per model** and extract **T(N)** — the iteration at which motion
   stops being directional. Criterion: the consecutive-displacement **cosine**, not the silencing
   rate, because the rate is noisy and near zero by construction at the end while the cosine is
   well-conditioned. The other channels corroborate.
2. **Report the active-unit count at T(N)** rather than at an arbitrary fixed iteration. This is the
   converged curve, and it is the only one the M\* question can legitimately be asked of.
3. **Fit the converged curve** with the two competing models and decide by the pre-registered rule
   already recorded (2026-07-27): saturating vs power law, discriminated most sharply at large N.

Sizes and lengths, set from measured decay rates (N=2000 decays ~2.5× slower than N=1000 and needs
the longest run):

| N | iterations | h/job | 3 seeds |
|---|---|---|---|
| 100 | 50000 | 1.8 | 5 h |
| 500 | 200000 | 7.2 | 22 h |
| 1000 | 200000 | 8.0 | 24 h |
| 2000 | 300000 | 26.4 | 79 h |

~130 GPU-hours. `weight_decay = 1e-6` throughout — the field default, kept deliberately (2026-07-30)
rather than switching to 0, even though 0 gives a more stationary measurement.

### What would settle it

- **M\* exists** if the converged active count flattens with N — successive local exponents
  continuing to fall toward 0.
- **No M\*** if the exponent stabilises (it currently sits at 0.57 between N=500→1000 and
  N=1000→2000, having fallen from 0.93; a genuine plateau in the exponent is the signature of
  unbounded power-law growth).
- The discrimination is sharpest at large N, which is why the ceiling estimate (~750 active units
  from the saturating fit) has to be tested against a converged N=2000 point rather than an
  extrapolation from N ≤ 1000.

**Caveat to carry:** even a converged curve over N ∈ {100 … 2000} is one decade. A ceiling at ~750
would be visible in that range; unbounded growth with a slowly-falling exponent would not be
distinguishable from a very high ceiling. State the range of validity rather than claiming the
asymptote.

### Submitted 2026-08-17 16:00 EDT — Della arrays `12545109` (9 jobs) and `12545110` (3 jobs)

The drift-characterisation sweep defined by the goal statement above is now running, from its own
git worktree at commit `2f9bcc7`.

| Array | Cells | Jobs | Iterations | Wall request | QOS |
|---|---|---|---|---|---|
| `12545109` | N = 100, 500, 1000 | 9 | 50000 / 200000 / 200000 | 12:00:00 | `gpu-short` |
| `12545110` | N = 2000 | 3 | **300000** | 40:00:00 | `gpu-medium` |

Split into two submissions because the N=2000 cells need ~26.4 h, past the 24 h `gpu-short` ceiling.
**Iteration budgets are per size, set from measured decay rates** rather than uniform: the silencing
rate decays ~2.5× more slowly at N=2000 than at N=1000 (a factor of 4.2 vs 10.8 between iterations
30k and 100k), so a flat budget would leave the largest and most decisive cell still drifting while
over-running N=100, which is already settled by iteration ~3000.

Configuration verified on Della before submitting: `dale: false`, `self_connections: true`,
`weight_decay: 1.0e-06` (the field default, kept deliberately — see 2026-07-30), `track_drift: true`,
`track_every: 10`, `store_participation_every: 100`, `light_outputs: true`, no penalties, h equation.

**Recorded per run** (all reduced to scalars during training; no weights written to disk):
`drift_{W_inp,W_rec,W_out}_lag{100,1000,10000}`, `cos_{W_inp,W_rec,W_out}`,
`dp_lag{100,1000,10000}`, and `silent_1em6` at every probe.

**Planned analysis:** drift curves per model → **T(N)** from the directional-cosine criterion →
active-unit counts *at* T(N) → fit the converged curve against the saturating and power-law models
using the thresholds pre-registered on 2026-07-27, to test for **M\***.

> **Near-miss worth recording.** The task-index decode was verified under `zsh`, which indexes arrays
> from 1, while the launcher runs under `bash`, which indexes from 0 — the check appeared to show
> every cell shifted by one size. Re-checked under `bash` and the mapping is correct (tasks 1–3 →
> N=100, 4–6 → 500, 7–9 → 1000, 10–12 → 2000). A real off-by-one here would have silently mislabelled
> every output folder with the wrong N.

## 2026-08-17 (16:45) — population-level consequences: the same task, very different circuits

The analysis that decides whether this project is about tidiness or about model validity. **No new
training** — computed on the existing standard-RNN reference sweep (N=1000, 5 nets per cell) with
[`population_distortion.py`](../trainRNNbrain/experiments_and_analysis/population_distortion.py).

**The premise, verified first:** all four penalty conditions solve CDDM equally well. R² per net:
`none` 0.840–0.875, `rws` 0.850–0.873, `frm` 0.848–0.872, `both` 0.837–0.853. No condition is
trading task performance for anything below.

![Population statistics](../img/internal_figures/population_distortion.png)

| h equation, N=1000 | none | rws | **frm** | **both** |
|---|---|---|---|---|
| silent units | 42.6% | 40.0% | **0%** | **0%** |
| **effective dimensionality (PR)** | 2.22 ± 0.07 | 2.39 ± 0.06 | **7.74 ± 0.21** | 6.14 ± 0.18 |
| **units selective to choice** | 19.2 ± 1.7% | 23.9 ± 1.5% | **58.3 ± 1.5%** | **82.1 ± 2.3%** |
| units selective to context | 24.3 ± 1.8% | 24.0 ± 1.3% | 31.1 ± 3.4% | **61.7 ± 3.2%** |
| **total metabolic cost** | 31.1 | 31.6 | **14.6** | 16.0 |
| **concentration of that cost (HHI)** | 0.123 | 0.140 | **0.0012** | 0.0011 |

The `s` equation gives the same picture (PR 3.03 → 7.06; choice selectivity 27.9% → 73.2%).

### What this establishes

1. **Effective dimensionality differs 3.5×** — 2.22 vs 7.74, with tight non-overlapping intervals.
   "How many dimensions does the circuit use" is *the* standard quantity for comparing model
   population activity against neural recordings, and it depends almost entirely on whether activity
   was regularised, not on how well the network does the task.
2. **Choice selectivity differs 3–4×** — 19% of units carry choice information without penalties,
   58% with `frm`, 82% with `frm`+`rws`. Any claim of the form "X% of units in the model are
   choice-selective, comparable to Y% in the data" is a statement about the penalty, not the circuit.
3. **The concentrated solution costs MORE energy, not less** — total metabolic cost is 31.1 without
   penalties and 14.6 with `frm`, less than half. This inverts the intuitive efficiency argument: a
   network that concentrates its computation into a few high-rate units pays more, because cost goes
   as the square of the rate. Spreading the same computation thinly is cheaper.
4. **Energy concentration differs 100×.** HHI 0.123 vs 0.0012, i.e. the unpenalized network's
   metabolic cost is carried by an effective **8 units** (1/HHI), the penalized one's by **~850**.

### The sampling distortion, separately

Within unpenalized networks, the same statistic computed over all units versus over active units only
(h equation):

| penalty | context: all units → active only | choice: all → active |
|---|---|---|
| none | 24.3% → **42.3%** | 19.2% → **33.5%** |
| rws | 24.0% → 40.0% | 23.9% → 39.8% |
| frm | 31.1% → 31.1% | 58.3% → 58.3% (no silent units) |

A recording experiment does not see silent neurons — it sees the active ones. So the fraction a
modeller reports from the full model population is ~1.7× lower than what the equivalent experiment
would measure from the same circuit. Under `frm` the two numbers are identical by construction.

### A prediction of mine that was wrong, and why the correction matters

Before running this I argued participation ratio would be **insensitive** to silent units, since
appending all-zero units adds zero eigenvalues and changes neither `(ΣΛ)²` nor `ΣΛ²`. That algebra is
correct but irrelevant: `frm` and `none` are not the same network with zeros removed, they are
**different solutions**. The penalized network genuinely spreads its computation over ~3.5× more
dimensions. The distortion is not an artifact of counting dead units — it is a real difference in the
circuit that the training choice produces.

### Consequence for the paper

`paper.md` §5 moves from planned to measured, and its claim strengthens: it is no longer only "silent
units dilute your statistics" but **"two networks that perform identically on the task differ by 3.5×
in dimensionality, 3× in selectivity and 100× in energy concentration, and nothing in a typical
methods section tells the reader which one you trained."** That is a validity argument about
model-based inference, not an aesthetic preference — and it is the strongest case for the penalties
independent of whether M* exists.

### CORRECTION (2026-08-17 17:00) — the selectivity fractions above were diluted, and one reverses

Pavel's objection: a selectivity **percentage** computed over all units is trivially depressed when
42% of them are silent, since silent units are non-selective by construction. Correct — and the
active-units-only comparison, which the analysis already computed but which I failed to headline,
changes the conclusion.

| h, N=1000, **active units only** | none | rws | frm | both |
|---|---|---|---|---|
| context-selective | **42.3 ± 2.9%** | 40.0 ± 2.6% | **31.1 ± 3.4%** | 61.7 ± 3.2% |
| choice-selective | 33.5 ± 2.7% | 39.8 ± 2.9% | **58.3 ± 1.5%** | 82.1 ± 2.3% |

| s, N=1000, **active units only** | none | rws | frm | both |
|---|---|---|---|---|
| context-selective | **65.3 ± 1.9%** | 51.6 ± 2.0% | **40.4 ± 2.3%** | 63.7 ± 0.8% |
| choice-selective | 46.5 ± 1.5% | 47.1 ± 3.1% | **73.2 ± 1.3%** | 85.6 ± 0.6% |

**What changes:**

- **Context selectivity REVERSES.** Among active units `frm` is *lower*, not higher: 42.3% → 31.1%
  (h, ratio 0.74) and 65.3% → 40.4% (s, ratio 0.62). The apparent increase reported above
  (24.3% → 31.1%) was **entirely a dilution artifact**. That claim is withdrawn.
- **Choice selectivity survives but shrinks by half.** 33.5% → 58.3% (h) is 1.74×, not the 3.0×
  claimed from the all-units figures; 46.5% → 73.2% (s) is 1.57×. Still a large, CI-separated effect.

**What is untouched, and why:**

- **Participation ratio.** `pr_active` equals `pr` to the printed digit in every cell — appending
  zero-variance units adds zero eigenvalues, which change neither `(ΣΛ)²` nor `ΣΛ²`. The **3.5×
  dimensionality difference is not dilution**; it is a real difference between the two solutions.
  (This is the same algebra I raised as a caveat before running the analysis, then wrongly set aside
  when the numbers came in.)
- **Total metabolic cost and its HHI concentration.** Zero-energy units contribute nothing to either
  sum, so neither statistic is diluted. The 2× cost difference and 100× concentration difference
  stand.

**A nuance the fractions hide.** Fractions fall while counts rise. Under `none`, 42.3% of 574 active
units = **243** context-selective units; under `frm`, 31.1% of 1000 = **311**. So `frm` produces ~28%
*more* context-selective units in absolute terms while lowering the fraction — the revived units are
disproportionately choice-tuned rather than context-tuned. That is a change in the **composition** of
the population, not a rescaling of it, and it is a more interesting statement than either fraction
alone.

**Methodological rule for the write-up:** every population fraction gets reported over active units,
with the all-units version shown only to make the dilution point explicitly. Scale-free statistics
(PR, HHI) are exempt, and it is worth saying why in the text rather than leaving the reader to check.

**Figure regenerated** with the corrected metric — the version above showed all-unit selectivity and
is superseded:

![Population statistics, corrected](../img/internal_figures/population_distortion.png)

Eight panels, each axis carrying its formula, and the header defining the symbols. Participation
ratio, context- and choice-selectivity **over active units**, total metabolic cost, cost
concentration, **σ_log with the cortical reference line at 1**, within-trial modulation, and the
p90/median rate tail.

**Exact definitions.** `r_i(t,c)` is the noise-free rate of unit *i* at timestep *t* in condition
*c*, over the full 450-condition CDDM batch; `r̄_i` its mean over time and conditions.

| statistic | definition | notes |
|---|---|---|
| **PR** | `(Σᵢλᵢ)² / Σᵢλᵢ²`, where λ are eigenvalues of the units × units covariance of `r` with time and conditions flattened into one sample axis and each unit centred | 1 = all variance in one dimension, N = variance spread equally. **Invariant to appending silent units** (they add zero eigenvalues, changing neither sum), which is why it needs no active-only correction |
| **η²** selectivity | each unit reduced to its mean rate over the decision epoch (t ≥ 200) per condition, giving one value per condition; conditions grouped by factor (context: motion vs colour; choice: ±1); `η² = σ²_between / σ²_total`; a unit is selective if **η² > 0.10**; reported as a fraction of **active** units | ⚠️ the 0.10 threshold is arbitrary and its sensitivity is untested, and there is **no null distribution** — this means "explains >10% of across-condition variance", not "statistically significant". A permutation null would make it a proper test, and should be added before publication |
| **E** | `Σᵢ ⟨r_i²⟩` over time and conditions | zero-rate units contribute nothing, so no dilution |
| **HHI** | `Σᵢ pᵢ²` with `pᵢ = ⟨r_i²⟩ / E` | 1/N = perfectly even, 1 = one unit carries everything; `1/HHI` is the effective number of units bearing the cost |
| **σ_log** | `std_i[log₁₀ r̄_i]` over active units | the lognormal shape parameter; cortex ≈ 1 |
| **within-trial CV** | `med_i [ ⟨σ_t(r_i)⟩_c / r̄_i ]` over active units | temporal std within a trial, averaged over conditions, normalised by the unit's own mean rate |
| **rate tail** | `r̄_(90) / r̄_(50)` across active units | scale-free, assumption-free heterogeneity check | Reading across, `frm` raises dimensionality 3.5×, *lowers*
context selectivity among active units, raises choice selectivity 1.7×, halves total metabolic cost,
spreads that cost over ~100× more units, keeps units genuinely modulated within the trial — and
flattens the across-unit rate distribution far below the biological range, which is the one result
that goes against it.

### The result that cuts against the penalty: rate heterogeneity, measured as sigma_log

**How it is measured.** Each unit is collapsed to one number — its mean firing rate over all 300
timesteps and all 450 conditions of the noise-free batch. Silent units are excluded, then the spread
of that distribution *across units* is summarised three ways:

| statistic | definition | why |
|---|---|---|
| **σ_log** | std of `log10(mean rate)` across active units | cortical rate distributions are close to **lognormal**, and σ_log is the shape parameter that literature reports — about **1 in log10 units**, i.e. roughly a decade of spread between typical slow and fast cells. Scale-free by construction, and stable on heavy tails. Requires positive rates, hence active units only. |
| CV | `std / mean` of the rate across active units | intuitive but **not** scale-free — `frm` raises the mean rate, so part of a CV drop is a mean shift rather than a narrowing |
| p90 / median | tail ratio across active units | scale-free, robust, no distributional assumption |
| **within-trial CV** | per unit, the temporal std of its rate averaged over conditions and divided by its own mean rate; median over active units | asks whether a unit is genuinely **modulated by the task** or merely sitting at a constant rate — the direct test of whether an activity penalty is satisfied by tonic firing |

| N=1000, active units | σ_log | CV | p90/median | within-trial CV |
|---|---|---|---|---|
| h/none | **1.20 ± 0.05** | 3.62 | 6.22 | **1.43 ± 0.05** |
| h/rws | 1.27 ± 0.08 | 3.64 | 7.84 | 1.36 ± 0.08 |
| **h/frm** | **0.26 ± 0.01** | 0.49 | 1.73 | 1.29 ± 0.04 |
| h/both | 0.19 ± 0.01 | 0.43 | 1.66 | **0.96 ± 0.01** |
| s/none | **1.01 ± 0.08** | 2.41 | 6.46 | 1.31 ± 0.06 |
| s/rws | 1.39 ± 0.07 | 2.77 | 7.55 | 1.14 ± 0.06 |
| **s/frm** | **0.15 ± 0.01** | 0.31 | 1.38 | 1.16 ± 0.02 |
| s/both | 0.16 ± 0.00 | 0.35 | 1.52 | **0.89 ± 0.01** |

**Two findings, pointing in opposite directions.**

1. **Unpenalized networks match cortex on rate heterogeneity; penalized ones do not.** σ_log is
   **1.20 (h) and 1.01 (s)** without penalties — essentially the cortical value of ~1, a full decade
   of spread across the active population. Under `frm` it collapses to **0.26 and 0.15**, a fifth of
   a decade. `rws` does not restore it, and `both` is flatter still (0.19, 0.16). All three
   heterogeneity measures agree, which is what makes this believable rather than an artifact of one
   statistic. **This is a real limitation of the penalty and belongs in the paper as one:** `frm`
   trades one unrealism (half the population silent) for another (a population too uniform).
2. **But the revived units are genuinely modulated, not tonic.** Within-trial CV is 1.29 under `frm`
   against 1.43 without — only ~10% lower. So `frm` units are *not* sitting at the cap doing nothing;
   they vary over the trial almost as much as the units of an unpenalized network. **This closes the
   "the penalty is satisfied by tonic firing" worry** that has been open since the earliest sweeps.
   It is `both` that materially flattens modulation (0.96, a 33% drop) — quantifying, at last, the
   old qualitative claim that `rws` makes responses "more sustained".

**Concrete follow-up** (no new machinery): `frm` has a `cap_fr` target and an already-implemented,
never-used `aggregation: logsumexp` option with temperature `tau_n`. Mean aggregation pulls *every*
unit toward the cap, which is exactly what would flatten the rate distribution; logsumexp penalises
the worst offenders instead. A one-axis sweep over those two knobs could plausibly keep every unit
active while preserving σ_log near 1 — turning a stated limitation into a tuning result.

**Figure regenerated** with the corrected metric — the version above showed all-unit selectivity and
is superseded:

![Population statistics, corrected](../img/internal_figures/population_distortion.png)

Eight panels, each axis carrying its formula, and the header defining the symbols. Participation
ratio, context- and choice-selectivity **over active units**, total metabolic cost, cost
concentration, **σ_log with the cortical reference line at 1**, within-trial modulation, and the
p90/median rate tail.

**Exact definitions.** `r_i(t,c)` is the noise-free rate of unit *i* at timestep *t* in condition
*c*, over the full 450-condition CDDM batch; `r̄_i` its mean over time and conditions.

| statistic | definition | notes |
|---|---|---|
| **PR** | `(Σᵢλᵢ)² / Σᵢλᵢ²`, where λ are eigenvalues of the units × units covariance of `r` with time and conditions flattened into one sample axis and each unit centred | 1 = all variance in one dimension, N = variance spread equally. **Invariant to appending silent units** (they add zero eigenvalues, changing neither sum), which is why it needs no active-only correction |
| **η²** selectivity | each unit reduced to its mean rate over the decision epoch (t ≥ 200) per condition, giving one value per condition; conditions grouped by factor (context: motion vs colour; choice: ±1); `η² = σ²_between / σ²_total`; a unit is selective if **η² > 0.10**; reported as a fraction of **active** units | ⚠️ the 0.10 threshold is arbitrary and its sensitivity is untested, and there is **no null distribution** — this means "explains >10% of across-condition variance", not "statistically significant". A permutation null would make it a proper test, and should be added before publication |
| **E** | `Σᵢ ⟨r_i²⟩` over time and conditions | zero-rate units contribute nothing, so no dilution |
| **HHI** | `Σᵢ pᵢ²` with `pᵢ = ⟨r_i²⟩ / E` | 1/N = perfectly even, 1 = one unit carries everything; `1/HHI` is the effective number of units bearing the cost |
| **σ_log** | `std_i[log₁₀ r̄_i]` over active units | the lognormal shape parameter; cortex ≈ 1 |
| **within-trial CV** | `med_i [ ⟨σ_t(r_i)⟩_c / r̄_i ]` over active units | temporal std within a trial, averaged over conditions, normalised by the unit's own mean rate |
| **rate tail** | `r̄_(90) / r̄_(50)` across active units | scale-free, assumption-free heterogeneity check | Reading across, `frm` raises dimensionality 3.5×, *lowers*
context selectivity among active units, raises choice selectivity 1.7×, halves total metabolic cost,
spreads that cost over ~100× more units, keeps units genuinely modulated within the trial — and
flattens the across-unit rate distribution far below the biological range, which is the one result
that goes against it.

### The result that cuts against the penalty: firing-rate heterogeneity

| N=1000 | active units | rate CV (active) | p90 / median rate |
|---|---|---|---|
| h/none | 574 | **3.62 ± 0.51** | 6.22 |
| h/rws | 600 | 3.64 ± 0.28 | 7.84 |
| **h/frm** | 1000 | **0.49 ± 0.01** | **1.73** |
| h/both | 1000 | 0.43 ± 0.02 | 1.66 |
| s/none | 600 | 2.41 ± 0.17 | 6.46 |
| **s/frm** | 1000 | **0.31 ± 0.02** | **1.38** |

Unpenalized networks have **strongly heterogeneous** rates among their active units — CV ≈ 2.4–3.6,
with the 90th percentile 6–8× the median, i.e. a long right tail. `frm` networks are nearly
**uniform**: CV ≈ 0.3–0.5, p90 only 1.4–1.7× the median. `rws` does not restore the spread; `both`
is, if anything, flatter still.

**This is a genuine limitation of the penalty, and it should be reported as one.** Cortical firing
rates are strongly heterogeneous — approximately lognormal with a long tail — so a model population
in which every active unit fires at nearly the same rate is *less* data-like in this respect, not
more. `frm` trades one unrealism (half the population silent) for another (a population that is too
uniform). The honest statement for the paper is that it fixes the silent-unit problem and introduces
a homogeneity problem, and that both matter for the same reason: they change what the model
population looks like relative to a recorded one.

**Concrete follow-up this suggests** (no new machinery needed): `frm` has a `cap_fr` target and an
`aggregation` option (`mean` vs `logsumexp` with temperature `tau_n`, already implemented and
unused). A weaker cap, or logsumexp aggregation that penalises only the worst offenders rather than
pulling every unit toward the target, might keep every unit active *without* collapsing the rate
distribution. That is a one-axis sweep and would turn a limitation into a tuning result.

---

## 2026-08-17 17:59 — Drift sweep, N=100 cell complete: motion stops being *directed* by ~10⁴, but does not stop

All three N=100 seeds finished on Spock (array 5660818, 50k iterations each, r² 0.87–0.90).
Figure: `img/internal_figures/drift_N100.png`, produced by
`trainRNNbrain/experiments_and_analysis/plot_drift_curves.py`.

### The criterion that works, and the one that does not

Two readouts were meant to identify T(N), the iteration at which motion stops being systematic.

**The directional cosine fails as a criterion.** `cos(ΔW_t, ΔW_{t-1})` falls from 0.7–0.85 to a
floor of **0.32–0.35** by iteration ~850–1300 and then stays flat for the remaining 48k iterations.
The floor is positive for the reason anticipated when the metric was designed — consecutive
displacements are separated by only `track_every=10` iterations and Adam's momentum correlates them —
so once the curve is at its floor it can no longer distinguish "still marching" from "jittering".
It saturates an order of magnitude earlier than the real transition and must not be used to set T(N).

**The lag-scaling exponent works and needs no threshold.** With
`α = Δlog d / Δlog L` for `d(L) = ‖W(t)−W(t−L)‖_F / ‖W(t)‖_F`, the value 0.5 is the theoretical
random-walk exponent and 1.0 is straight-line motion — both are predictions, not tuning knobs.
Measured on W_rec between lags 100 and 1000:

| seed | α at iter 1000 | T(α = 0.5) | α at 50k |
|---|---|---|---|
| 0 | 1.10 | **12000** | 0.41 |
| 1 | 1.06 | **12000** | 0.40 |
| 2 | 1.05 | **9000** | 0.39 |

So **T(N=100) ≈ 1×10⁴ iterations**, ten times later than the cosine floor would have suggested.

### The finding: α is timescale-dependent — pinned at short lags, still diffusing at long ones

**Correction to the first reading of this figure.** The initial entry reported only the short-lag
exponent (0.39–0.41), read it as sub-diffusive, and concluded the weights sit in a confined
Ornstein–Uhlenbeck basin. Comparing both lag pairs **at the same iteration** (t = 40000, so the two
are not being read off different points of a decaying curve) does not support that:

| seed | α(100→1000) | α(1000→10000) |
|---|---|---|
| 0 | 0.37 | 0.52 |
| 1 | 0.42 | 0.55 |
| 2 | 0.37 | 0.53 |

α *increases* with lag. A confined OU process predicts the opposite — beyond its relaxation time
displacement saturates and α falls toward 0. So the weights are **not** in a closed basin. The
consistent reading is two regimes:

- **10²–10³ iterations: sub-diffusive (α ≈ 0.4).** Displacement over 1000 steps is only ~2.4× that
  over 100, less than the √10 = 3.16 a free walk would give. Successive steps partly cancel. Two
  candidate causes, neither tested here: local confinement at that timescale, or overshoot
  oscillation along sharp curvature directions (the "edge of stability" behaviour reported for
  Adam/SGD at practical learning rates).
- **10³–10⁴ iterations: diffusive (α ≈ 0.53).** Ratio 3.3–3.6 versus √10 = 3.16. At the longest lag
  measured the weights are still executing an unbounded random walk.

**What survives, and what does not.** T(N=100) ≈ 10⁴ stands: that is where the strongly *systematic*
phase (α ≈ 1, straight-line motion) ends, and it is the quantity the sweep was designed to extract.
What does not survive is the stronger claim that training has arrived somewhere. It has not — it has
stopped going anywhere *in particular*. "Undirected" is the defensible word; "converged" is not.

The functional readout is better behaved than the weights. For the participation vector at t = 40000,
**α_p(100→1000) = 0.04–0.08** — essentially no growth with lag, i.e. waiting ten times longer buys
almost no additional change in *p*. But α_p(1000→10000) = 0.58–0.69, mildly *super*-diffusive. So
even at N=100 after 50k iterations there is still a slow systematic component in how participation
evolves at the 10³–10⁴ timescale. The weights wander; *p* is pinned minute-to-minute but not
hour-to-hour.

Caveat: measured at N=100, the size that silences least. It does not license any claim at N=1000 or
N=2000, where the silencing rate is still visibly non-zero at 100k. The larger cells are the test.

### Silent units at N=100: transient silencing, then recruitment back

`silent_1em6` peaks **early** (15, 11, 3 units at iteration ~30–60) and then *declines* for the rest
of training, ending at **4, 3, 1** of 100 units — i.e. 96, 97 and 99 active. The per-unit
participation vectors say the same thing: active counts go 92 → 87 → **96**, 92 → 92 → **97**,
98 → 97 → **99**. Small networks do not exhibit the phenomenon: they transiently mute a few units in
the first ~50 iterations and then recruit them back.

This is the opposite of the N=1000 behaviour (65–76% silent at 100k) and is the first data point of
the M* curve: **M(100) ≈ 97**. One point cannot distinguish saturation from growth — that needs the
N=500/1000/2000 cells, which are still running (arrays 5660819/20/21, ~1:45 elapsed at the time of
writing).

### Secondary observation, not yet explained

In panel (b) the **output** weights drift most, and their relative drift at lag 1000 *rises* after
iteration ~2×10³ while W_inp and W_rec keep falling. The likely cause is the denominator: with
`weight_decay=1e-6` and no penalty holding W_out up, ‖W_out‖ shrinks, so a constant absolute step
becomes a growing relative one. Checking this needs the raw norms, which are not stored — worth a
scalar `norm_W_*` addition to the tracker if it matters later.

### The stopping criterion, stated properly: α_p at the longest lag crossing 0.5

The exponent gives a threshold-free stopping rule — "train until the motion is no longer directed",
i.e. until α falls stably below the random-walk value 0.5. Three things have to be pinned down before
that rule is usable, and the N=100 cell settles all three.

**1. It must name the lag.** α is lag-dependent, so "α < 0.5" is meaningless without saying at what
timescale. At N=100 the W_rec exponent crosses 0.5 at ~10⁴ for the 100→1000 pair but only at
~4–5×10⁴ for the 1000→10⁴ pair. Using the short lag declares victory 4× too early. **Use the longest
resolvable lag.**

**2. It should be measured on the participation vector, not the weights.** The weights are free to
move along functionally degenerate directions — W_out is doing exactly that (see the W_out section
below), sliding along the `W_out·r` rescaling invariance for the whole run without changing what any
unit computes. A weight-based rule therefore reports motion that has no functional consequence.
`p` is the quantity the silent-unit question is about, so `α_p` is the right criterion.

| N=100, α at the end of the run | 100→1000 | 1000→10000 |
|---|---|---|
| W_rec | 0.37–0.42 | 0.52–0.55 |
| **p** | **0.04–0.08** | **0.38–0.65** |

`α_p` at the short lag is ~0 from very early on: over 10²–10³ iterations the participation vector
does not move at all in the sense that matters, which is why it looked "converged" in every earlier
short-window analysis. At the long lag it starts **ballistic** (1.12–1.35 at iteration 10⁴ — still
straight-line motion) and only reaches 0.5 at **30–40k**.

**3. "Stably" is doing real work.** The short-lag α bounces between 0.3 and 0.5 after 2×10⁴, so the
rule needs "stays below for the remainder", not "first touch". At the long lag there are only four
measurements in a 50k run (one per 10⁴), so stability is not even assessable at N=100.

**Consequence: T(N=100) ≈ 3–4×10⁴, not 10⁴.** The earlier figure quoted 10⁴ from the short-lag
weight curve; on the criterion above it is 3–4× larger, i.e. the 50k budget for N=100 was only just
sufficient and the crossing sits at the edge of the run. The larger cells have 200k (N=500, N=1000)
and 300k (N=2000), giving 20–30 points on the criterion curve — enough to assess stability properly,
provided T does not grow faster than ~4× with N. That is now a thing to watch, not an assumption.

### Why the output weights appear to "speed up": a shrinking denominator

`d = ‖ΔW‖_F/‖W‖_F`. Under Adam the per-parameter step is roughly uniform within a matrix, so
`‖ΔW‖_F ≈ step·√P·f(L)` and `‖W‖_F ≈ w·√P` for P parameters and typical weight magnitude w — the
√P cancels and **d ≈ step·f(L)/w**. Parameter count is irrelevant; a rising d at fixed learning rate
means w is falling.

Measured directly, after adding `norm_W_*` to the tracker (commit a11e179): over the first 300
iterations at N=100, **‖W_out‖ 1.35 → 0.66** (halves) while ‖W_inp‖ 2.44 → 2.64 and
‖W_rec‖ 12.03 → 12.12 barely move. Two consistency checks from the existing traces: ‖p‖ grows
monotonically **8.8 → 18.4 (×2.1)** between iteration 1k and 50k, and d_out at L=1000 rises from
~0.20 at its minimum to ~0.40 — the same factor of 2.

Mechanism: the output is `W_out·r`, invariant under `W_out → W_out/c`, `r → c·r`. Nothing in the loss
pins the split, so the network slides along that degenerate direction throughout training. Note
weight decay is *not* the driver — wd=1e-6 cannot halve a norm in 300 steps; the task gradient is.
So W_out is not being updated faster in absolute terms, and panel (b) is a normalisation artifact —
but the artifact reveals a real, never-terminating slide.

**Hypothesis, untested:** median participation *falls* (0.57 → 0.37) while ‖p‖ *rises* (×2.1), so the
rate distribution is stretching — a few units get much louder, the typical unit gets quieter. This is
the same heterogeneity the population-distortion analysis measures, appearing here as a dynamic
process. Whether this slide is what terminates in silencing at large N is testable against the
N=1000/2000 cells and has not been tested.

### α resolved at every lag: it is not a number, it is a curve — and the network never settles

The two-point exponent uses the only lags the online tracker records. But the stored participation
vectors give `p` every 100 iterations, so the displacement can be time-averaged over all pairs in a
window at **every** multiple of 100 and the local slope read off directly. No re-run needed. (The
weight matrices are not stored, only scalar drifts, so the same trick is unavailable for W — see the
design note at the end.) Figure: `img/internal_figures/drift_msd_N100.png`.

**α_p(L) rises smoothly with lag.** Late in training (window 24k–49k) it is ~0.10 at L=100, crosses
0.5 at L ≈ 2500–3000, and reaches **0.85–0.95** by L=10⁴. So the two-point estimate over 1000→10⁴
(0.38–0.65, reported above) is **biased, not merely noisy**: it averages a curve that rises from
~0.15 to ~0.9 across that decade, and the average of a rising function sits far below its endpoint.
Any conclusion drawn from a two-point α should be re-derived from the resolved curve.

**The corrected reading: at long lags the participation vector is still nearly ballistic.** At
N=100, after 50k iterations, over 10⁴-iteration windows the network is still marching in a
consistent direction (α_p ≈ 0.9), not diffusing. It has not converged in any sense.

**And it is not rate inflation.** Repeating everything on `p̂ = p/‖p‖` — which removes the ×2.1
growth of ‖p‖ entirely — gives the same picture (α_p̂ ≈ 0.85–0.9 at L=10⁴). The *pattern* of who
participates is reorganising directionally, not merely scaling up.

### The reframing this forces: L*(t), and aging

If α depends on lag, "the iteration at which α crosses 0.5" is ill-posed. The well-posed quantity is
its inverse: **L\*(t) = the lag at which α_p crosses 0.5, measured at training age t.** Below L* the
participation vector is caged (changes cancel); above it, motion is directed. L* is "how long you
must wait before training takes the network somewhere new".

Measured on sliding 20k windows, 3 seeds, both definitions of the displacement:

| | scaling | L*/t |
|---|---|---|
| full `p` | L\* ∝ t^0.95 | 0.081 ± 0.009 |
| direction only `p̂` | L\* ∝ t^0.89 | 0.079 ± 0.008 |

**L\* grows in proportion to training age, at ≈ 8% of it.** The network is always still directed on
timescales longer than about a twelfth of its own age, however long it has trained. Train 10× longer
and the caging timescale stretches 10× with it. This is the **aging** signature of glassy relaxation:
there is no fixed relaxation time to wait out, because the relaxation time is set by how long you
have already waited.

This supersedes and mechanistically explains the earlier power-law extrapolation ("a 1% criterion
needs 0.5–5.6 M iterations"). The extrapolation is not a long wait — it is unreachable in principle,
because the criterion recedes as fast as you approach it. **"Trained to convergence" is not
available for these networks and should not be claimed anywhere in the paper.** What can be claimed
is a stated training budget plus the measured L*/t, which says exactly how much residual directed
change that budget leaves.

**Caveats, stated before the numbers are used.** The lever arm is short: t spans only 15k–40k, a
factor 2.7, so the exponent 0.89–0.95 is consistent with 1.0 but also with somewhat less; the
windows overlap, so points are not independent; and this is N=100 only. The N=500/1000/2000 cells
(200k–300k iterations) give a 4–6× longer lever arm and are the real test of whether L*/t is
constant, and of whether the constant depends on N.

**Design note for the next sweep.** To resolve α for the weight matrices as well, the cheap fix is
not more entries in `drift_lags` (each costs a full CPU-resident snapshot of every matrix) but
storing a fixed random projection of each matrix at every probe — ~200 floats instead of ~4 M at
N=2000. Johnson–Lindenstrauss keeps pairwise distances, so any lag becomes computable post hoc,
exactly as it already is for `p`.

### Correction: this is not "aging", it is a flat valley — and the loss HAS converged

The previous section framed L*/t ≈ const as glassy aging with no stationary regime. That framing is
wrong and should not be used. Two measurements settle it.

**The loss has essentially converged.** Over the entire second half of training (25k → 50k) the
training loss falls by **1.8–2.1%** (0.0231 → 0.0227, all three seeds). The optimiser is not still
descending in any meaningful sense.

**The participation vector has not.** Fitting the distance from the final state,
`D(t) = ‖p(T) − p(t)‖/‖p(T)‖ = A·t^(−γ)`, over the window [T/12, T/3]:

| seed | γ | remaining change after T=50k | iterations to halve it |
|---|---|---|---|
| 0 | 0.33 | 30.0% of ‖p‖ | ×8.2 |
| 1 | 0.35 | 29.7% | ×7.3 |
| 2 | 0.35 | 31.0% | ×7.3 |

`p` moved ~37% of its norm between iteration 25k and 50k while the loss improved 1.8%.

**Those two facts together are the explanation.** Large configuration change at negligible loss
change means the motion is **along a flat direction of the loss**, not down it. The minimum is a
manifold, not a point: the `W_out·r` rescaling degeneracy is one exactly-flat direction, and it is
measurably being traversed (‖W_out‖ halves in 300 iterations while the other matrices hold). Motion
along a flat valley has no restoring force to stop it, and because it is driven by a systematic bias
(weight decay, plus the small consistent component of the gradient) rather than by noise, it is
**directed** — which is exactly why α ≈ 1 at long lags.

So the network does reach the stationary regime one expects. It just is not a *point*: it is a
valley floor, and it keeps sliding along it.

**Why α made this look alarming: α is blind to magnitude.** α ≈ 1 says "there is a consistent
direction", not "a lot is happening". The amount is small and shrinking — displacement over a 10⁴
window is 16% of ‖p‖ late in training, down from 40% early. Every future report should pair α with
the *amount* of remaining change; α alone is misleading. The number to quote for a training budget
is the ~30% remaining change with γ ≈ 1/3, not L*/t.

**The Adam question, and a testable prediction.** Adam's update is `lr·m̂/(√v̂+ε)`. Along a direction
where the gradient is tiny but *consistent*, `m̂/√v̂ ≈ ±1`, so Adam takes a full `lr`-sized step where
SGD would take one proportional to the (tiny) gradient. Adam therefore traverses flat directions at
roughly constant speed while SGD would nearly stop. This is a substantial part of why directed
motion persists here. **Prediction: re-run one cell with SGD+momentum and the persistent directed
component should largely disappear — γ should rise and the remaining-change estimate should fall.**
That is a cheap, decisive experiment (one N=100 cell, ~2 h) and it has not been run.

### How to choose the budget: T(N) is defined on the LOSS; participation supplies the error bar

Figure: `img/internal_figures/drift_budget_N100.png`. The question is made budget-relevant rather
than metaphysical by asking, at each training age t, **what one more doubling of the budget buys** —
comparing t/2 with t. No fit, no extrapolation to infinity, no estimate of a limit.

| t | loss gain / doubling | Δp / doubling | of which reorganisation (Δp̂) |
|---|---|---|---|
| 2000 | 7.4% | 15.0% | 10.5% |
| 10000 | 5.0% | 19.4% | 17.3% |
| 20000 | 4.1% | 27.6% | 25.1% |
| 30000 | 3.1% | 28.7% | 26.7% |
| 50000 | 2.1% | 28.4% | 26.7% |

**The loss converges and gives a usable criterion.** Gain per doubling falls as t^−0.41, cleanly and
monotonically. Declaring training done when one more doubling buys **< 2%** gives

> **T(N=100) ≈ 9.7 × 10⁴ iterations** — i.e. the 50k budget actually used was about half of it.

**Participation does not converge and cannot supply a criterion.** Its change per doubling *rises*
to ~28% and then sits there. It is not decaying, so no threshold on it will ever be met, at any
budget. This is the flat-valley motion: ~26 of those 28 points are genuine reorganisation of `p̂`,
only ~16% is rate inflation. (This also retires the earlier "30% remaining, halve it with 8× more
training" estimate — that came from fitting a power law over a narrow window to a quantity that is
actually flat in log-time. Do not use it.)

**So the recipe is:**

1. **Set T(N) from the loss**: train until the loss gain over the last doubling drops below a stated
   threshold. Scale-free, interpretable ("doubling my compute buys < 2%"), measurable during the run
   without extrapolation, and comparable across N — which is what the T(N) curve needs.
2. **Do not wait for participation to stabilise.** It does not. Looking for that point is what made
   the last three analyses confusing.
3. **Quote the participation sensitivity as an error bar on every reported statistic.** Not on `p`
   itself, but on the number actually reported: silent fraction, PR, selectivity. For each, report
   its value at T *and* how much it moves between T/2 and T. That converts an unanswerable
   convergence question into a stated, honest sensitivity.

**Robustness check to run before quoting T(N):** compute T at 3%, 2% and 1% and confirm the *shape*
of T(N) versus N is the same. The threshold is a choice; the scaling with N must not depend on it.

**Caveats.** T(N=100) is extrapolated ~2× beyond the data (the fit reaches 2% at 97k, the run ended
at 50k), so it is a projection, not a measurement — the last few points sit slightly below the fit,
which would make the true T somewhat smaller. The N=500/1000 (200k) and N=2000 (300k) cells should
reach the criterion inside their budgets and will give measured rather than extrapolated values.

### Correction to the recipe above: the loss criterion is a denominator artifact

Fitting `L(t) = L_∞ + A·t^(−γ)` on all three N=100 seeds (t > 2000):

| seed | L_∞ | γ | loss at T | irreducible | still reducible |
|---|---|---|---|---|---|
| 0 | 0.02149 | 0.56 | 0.02261 | 95% | 0.00112 (5.0%) |
| 1 | 0.02113 | 0.48 | 0.02273 | 93% | 0.00160 (7.0%) |
| 2 | 0.02102 | 0.47 | 0.02265 | 93% | 0.00163 (7.2%) |

**93–95% of the loss at T is irreducible** — it is the noise floor of the task, not something more
training can remove. The "2% gain per doubling" figure divides by that floor, so it shrinks as the
floor comes to dominate, and the loss *looks* convergent.

Divide instead by what is still on the table and the picture inverts. For any power law, the
fraction of the *reducible* loss removed per doubling is `1 − 2^(−γ)` — with γ ≈ 0.5 that is a
**constant 29% per doubling, forever**. Exactly the behaviour that was called "does not converge"
for participation.

**The general point: no power-law-relaxing quantity ever converges in the fractional sense.** Asking
"when has it converged" has no answer for the loss, for `p`, or for the weights. It is the wrong
question, and chasing it is what generated three rounds of confusing analysis.

### The criterion that actually works: movement of the reported statistic, in its own units

T is not a property of the network. It is a property of **(network, statistic, precision required)**.
So define it that way:

> Train until **the number that goes in the paper** moves by less than a stated amount over the last
> doubling of the budget.

For this project that number is the silent fraction. If it moves less than ~1 percentage point from
T/2 to T, then "X% of units are silent" is budget-independent to the precision claimed, and no
statement about convergence is needed at all. The same test is then run for every other reported
statistic (PR, selectivity, σ_log), each in its own units, and each may give a different T — which is
honest, because a coarse claim needs less training than a fine one.

Advantages over both earlier proposals: it is in interpretable units (percentage points of silent
units, not fractions of a norm), it needs no fit and no extrapolation, it is directly the
sensitivity the reader wants, and it cannot be gamed by the choice of denominator.

At N=100 the silent counts are too small to test this (1–3 units, drifting up ~0.7 pp/doubling late
in training). The N=1000 and N=2000 cells, where the fraction is 65–76%, are where it can actually
be measured. **Supersedes the loss-threshold recipe in the section above; that section is kept for
the record but should not be used.**

---

## 2026-08-17 19:37 — Applying the budget criterion to the M* sweep, and closing a fairness gap

The point of the Spock sweep is M*: does the number of ACTIVE units saturate as N grows? That makes
the reported statistic the active count itself, so the per-doubling test from the previous section
applies to it directly — no separate convergence machinery is needed.

**Nothing has to be re-run.** The traces already hold the whole M(N, t) surface: `silent_1em6` is
recorded at every probe (every 10 iterations) and the full per-unit `p` vector every 100 iterations,
so the active count at any threshold can be reconstructed at any t up to each run's length. The
sweep was designed as "train, then count at the end"; it can be analysed instead as a family of
curves, which is strictly more informative and costs nothing.

**Two fairness confounds, both now identified.**

1. *Unequal budgets.* 50k / 200k / 200k / 300k for N = 100 / 500 / 1000 / 2000. These were set from
   extrapolated silencing rates — a reasonable guess at the time, but not a principled matching.
2. *Unequal learning rates.* `run_experiment.py:65` sets `lr = 1e-3 · (100/N)^(1/3)`, so
   lr(N=2000) = 3.68e-4 is **2.7× smaller** than lr(N=100) = 1e-3. Equal iterations therefore are
   not equal optimisation, independently of the budget issue. This is a further argument against
   matching on iteration count and for matching on a measured statistic.

| N | lr | × vs N=100 | budget |
|---|---|---|---|
| 100 | 1.000e-3 | 1.00 | 50k → **now also 200k** |
| 500 | 5.848e-4 | 0.58 | 200k |
| 1000 | 4.642e-4 | 0.46 | 200k |
| 2000 | 3.684e-4 | 0.37 | 300k |

**The analysis that avoids needing convergence at all.** Rather than picking one T and hoping every
size has settled there, evaluate the M(N) curve at several **common** budgets and ask whether the
*verdict* — saturating versus still growing — is the same at each. If M(N) saturates at t = 25k, 50k,
100k and 200k alike, the conclusion is budget-independent, which is stronger than any single matched
comparison and does not require M to have stopped moving. If the verdict flips with t, then M* is
budget-dependent and that is the honest finding.

Each point on the M(N) curve then carries **two** error bars: seed spread, and its own per-doubling
movement (the budget sensitivity). The saturating-vs-power-law fit must be judged against both.

**Gap closed.** N=100 had only 50k, so the largest budget common to all four sizes was 50k — the
smallest network would have set the resolution of the whole comparison. Submitted **array 5660933**
(3 seeds, N=100, 200k iterations, ~7 h each, 12 h wall). Verified running and writing to
`EqType=h_N=100_iters=200000`, a separate folder from the original 50k run, so nothing can mix.
Implemented via a one-line `ITERS_OVERRIDE` env hook in the launcher (the launcher is the only place
allowed to vary swept parameters); the run script still overrides nothing.

All four sizes now share a common 200k budget, with N=2000 additionally reaching 300k.

---

# 2026-08-17 21:02 — SELF-CONTAINED SUMMARY: there is no simple convergence metric, and why

*This section is written to be read on its own. It assumes no other part of this document. All
measurements are from the standard ReLU RNN, no penalties, N=100, 3 seeds, 50,000 iterations
(Spock array 5660818). Figures: `img/internal_figures/drift_N100.png`,
`drift_msd_N100.png`, `drift_budget_N100.png`, `silent_stopping_criterion.png`.*

## 1. Why we needed a convergence criterion

The scientific question is **M\***: does the number of *active* units in a trained RNN saturate as
the network gets bigger? If it saturates, then adding units past M\* buys nothing, and activity
penalties become the only route to a network where every unit does something. To answer it we train
networks at N = 100, 500, 1000, 2000 and count active units at each size.

That comparison is only meaningful if every size is trained "equally far". So we needed a rule for
when a network is trained enough — and, ideally, a rule that transfers across sizes. Note the sizes
do *not* even share a learning rate: `run_experiment.py` sets `lr = 1e-3·(100/N)^(1/3)`, so N=2000
trains at 3.68e-4 versus N=100 at 1e-3, a factor of 2.7. Equal iteration counts were never equal
optimisation.

The naive expectation — the one we started with, and the one most people have — is that a gradient
optimiser near a minimum reaches a stationary regime: the loss flattens, the weights stop going
anywhere, and everything afterwards is small noise-driven jitter. Finding that point should be easy.

**It is not. Six different metrics were tried. Every one of them failed, and each failed for a
different and instructive reason.**

## 2. Attempt 1 — absolute drift magnitude

**Metric.** `d(L) = ‖W(t) − W(t−L)‖_F / ‖W(t)‖_F`: how far the weight matrix moved over a window of
L iterations, as a fraction of its size. Declare convergence when d is small.

**Result.** d falls steeply, then flattens onto a **noise floor that is not zero and whose height
depends on L** — about 0.03 at L=100, 0.07 at L=1000, 0.23 at L=10,000. The L=100 curve is flat and
pure noise from iteration ~1000 onward.

**Why it fails.** Three reasons.
- The magnitude of d conflates *how big the steps are* with *how they combine*. Tiny steps all in
  the same direction (slow but going somewhere) and large steps that cancel (fast, going nowhere)
  give the same d. That distinction is exactly what we care about.
- "d stopped decreasing" is not "motion stopped". The L=100 curve flattened at iteration 10³ while
  directed motion demonstrably continued to 10⁴.
- Any threshold ("converged when d < 0.05") is arbitrary *and does not transfer across N*, because
  the natural scale of d turns out to be `step size / typical weight magnitude`, which depends on N.
  Since the whole point is to compare across N, a non-transferable threshold assumes the answer.

## 3. Attempt 2 — directional persistence (cosine)

**Metric.** The cosine between consecutive weight displacements. For an uncorrelated random walk the
expected cosine is exactly 0; while the trajectory keeps a consistent heading it is positive. So:
declare convergence when the cosine reaches 0. Appealingly, 0 is a *prediction*, not a tuned value.

**Result.** It never reaches 0. It falls from 0.7–0.85 to a floor of **0.32–0.35** by iteration
~850–1300 and sits there for the remaining 48,000 iterations.

**Why it fails.** The floor is an artifact of how it was recorded. Consecutive displacements are
separated by only `track_every = 10` iterations, and Adam's momentum (β₁ = 0.9, roughly a 10-step
memory) *explicitly* correlates successive updates. So even a completely settled network reports
cos ≈ 0.33 here. The null value is not 0; it is an unknown positive number set by the optimiser.

The evidence that the floor is an optimiser property and not a network property: **all three weight
matrices floor at the same value** (W_inp 0.30–0.33, W_rec 0.32–0.35, W_out 0.32–0.33) despite
starting at very different values (0.5 for W_rec, 0.85 for W_out). A floor reflecting real residual
structure would not be identical across three matrices of different shape and role.

Worse than an offset, this is a **dynamic-range** failure: the cosine hits its floor at iteration
~10³, while the exponent below shows directed motion continuing to ~10⁴. Using it would have put T
an order of magnitude too early.

*Fixable in principle* — measure the cosine between displacements separated by much more than the
momentum timescale (e.g. successive 1000-iteration displacements) and the artifact disappears. Only
the lag-10 version was stored, so this would need a re-run.

## 4. Attempt 3 — the lag-scaling exponent α

**Metric.** Measure `d` at several lags *at the same t*, and take the log-log slope
`α = Δlog d / Δlog L`. The physics is unambiguous:

| motion | steps combine | d(L) ∝ | α |
|---|---|---|---|
| every step the same direction | linearly | L | **1.0** |
| independent random steps | in quadrature | √L | **0.5** |
| steps partly cancel / confined | saturates | L⁰ | **→ 0** |

This is mean-squared-displacement analysis, standard in diffusion physics and single-particle
tracking. Its appeal: taking a ratio of d at two lags **cancels both the step size and ‖W‖**, so α
is dimensionless, threshold-free (0.5 and 1.0 are predictions), and comparable across N.

**Result, and the first real finding.** α is **not a number — it is a curve in L**. Estimated from
two lags on W_rec at a matched iteration (t = 40,000):

| seed | α(100→1000) | α(1000→10000) |
|---|---|---|
| 0 | 0.37 | 0.52 |
| 1 | 0.42 | 0.55 |
| 2 | 0.37 | 0.53 |

α *increases* with lag. That rules out the tidy interpretation (a confined basin, which predicts α
falling toward 0 at long lags).

**Two errors we made here, both worth recording.**

*Error 1 — reading one lag pair.* The first write-up reported only α(100→1000) ≈ 0.4, called it
sub-diffusive, and concluded the weights sit in a confined Ornstein–Uhlenbeck basin. Comparing both
pairs at the same iteration killed that.

*Error 2 — using a two-point slope at all.* The participation vector `p` is stored in full every 100
iterations, so its displacement can be time-averaged over **every** lag that is a multiple of 100,
and the local slope read off directly — no re-run needed. Doing that shows α_p rises smoothly from
~0.10 at L=100, through 0.5 at L ≈ 2500–3000, to **0.85–0.95 at L=10⁴**, late in training.

So the two-point estimate over 1000→10⁴ (which gave 0.38–0.65) was **biased, not merely noisy**: it
averages a curve rising from ~0.15 to ~0.9 across that decade, and the mean of a rising function
sits far below its endpoint. *Any conclusion drawn from a two-point α must be re-derived from the
resolved curve.*

**Why α fails as a criterion.** Two reasons.
- It is lag-dependent, so "α < 0.5" is meaningless without naming a timescale. At N=100 the crossing
  is at ~10⁴ for the short lag pair and ~4×10⁴ for the long one — a factor of 4.
- **α is blind to magnitude.** α ≈ 1 says "there is a consistent direction", not "a lot is
  happening". A tiny but consistent motion has α = 1. This is what made the results look alarming:
  the amount of motion is small and shrinking (displacement over a 10⁴ window is 16% of ‖p‖ late in
  training, down from 40% early) while α screamed "still directed".

*(One check worth keeping: repeating everything on `p̂ = p/‖p‖`, which removes the ×2.1 growth of
‖p‖ entirely, gives the same picture — α ≈ 0.85–0.9 at L=10⁴. So the persistent directed motion is
genuine reorganisation of who participates, not merely overall rate inflation.)*

## 5. Attempt 4 — the caging timescale L\*(t), and a wrong story about "aging"

**Metric.** If α depends on lag, invert the question: define **L\*(t) = the lag at which α_p crosses
0.5, measured at training age t**. Below L\* the participation vector is caged (changes cancel);
above it, motion is directed. L\* is "how long must I wait before training takes the network
somewhere new".

**Result.** L\* grows in proportion to training age: L\* ∝ t^0.95 (full `p`) and t^0.89 (`p̂`), with
**L\*/t = 0.081 ± 0.009** and 0.079 ± 0.008 respectively.

**What we concluded, and why it was wrong.** We read this as glassy *aging*: no fixed relaxation
time to wait out, because the relaxation time is set by how long you have already waited; therefore
the network never converges. That framing is wrong and should not be repeated. Two measurements
killed it (next section). It is recorded here because it was written into this document and into
three rounds of analysis, and because L\*/t ≈ 0.08 is still a correct *measurement* — it is only the
interpretation that was overreached.

## 6. Attempt 5 — the loss

**Metric.** The obvious one: stop when the loss stops improving. Made scale-free by asking what one
more **doubling** of the budget buys — compare the loss at t/2 with the loss at t. ("Doubling" is
the natural unit because the loss decays as a power law: improvement from 1k→2k is comparable to
10k→20k, not to 10k→11k. It is also literally the compute question.)

**Result.** Clean and monotone. Loss gain per doubling falls as t^−0.41:

| t | 2,000 | 10,000 | 20,000 | 30,000 | 50,000 |
|---|---|---|---|---|---|
| loss gain / doubling | 7.4% | 5.0% | 4.1% | 3.1% | 2.1% |

A 2%-per-doubling threshold gives T(N=100) ≈ 9.7×10⁴ — i.e. the 50k actually run was about half.
This looked like the answer.

**Why it fails: the denominator.** Fitting `L(t) = L_∞ + A·t^(−γ)`:

| seed | L_∞ | γ | loss at T | irreducible | still reducible |
|---|---|---|---|---|---|
| 0 | 0.02149 | 0.56 | 0.02261 | 95% | 5.0% |
| 1 | 0.02113 | 0.48 | 0.02273 | 93% | 7.0% |
| 2 | 0.02102 | 0.47 | 0.02265 | 93% | 7.2% |

**93–95% of the loss is an irreducible noise floor** — the task's own stochasticity, which no amount
of training removes. Dividing by it makes the gain shrink simply because the floor comes to
dominate. The loss only *looks* convergent.

Divide instead by what is actually still on the table and it inverts: for any power law the fraction
of the **reducible** loss removed per doubling is `1 − 2^(−γ)`, which with γ ≈ 0.5 is a **constant
29% per doubling, forever**. Exactly the behaviour we had called "non-convergent" for participation.

The criterion was not measuring convergence. It was measuring our choice of denominator.

## 7. Attempt 6 — the participation vector per doubling

**Metric.** Same doubling logic, applied to `p` directly: `‖p(t) − p(t/2)‖ / ‖p(t)‖`.

**Result.** It **rises** from 15% and then sits flat at ~28%, from t ≈ 20,000 to the end of the run.
Decomposed: ~26 points of that is genuine reorganisation of the pattern `p̂`, only ~16% is growth of
‖p‖ (which itself goes 8.8 → 18.4, ×2.1, while the *median* participation falls 0.57 → 0.37 — the
rate distribution is stretching, not uniformly inflating).

**Why it fails.** It is not decaying at all, so **no threshold on it is ever met, at any budget**.

## 8. The underlying reason all six failed

Two facts, and together they explain everything above.

**(a) Everything here relaxes as a power law, and no power-law quantity converges in the fractional
sense.** Loss, participation, weight drift — all of them. For a power law, the fraction of the
remaining distance covered per doubling of time is a *constant*. So "has it converged?" has no
answer for any of them; it is the wrong question. Chasing it is what produced attempts 1–6.

**(b) The loss has converged; the configuration has not, and need not, because the minimum is a
manifold rather than a point.** Over the entire second half of training the loss falls 1.8–2.1%
while `p` moves ~37% of its norm. Large configuration change at negligible loss change means the
motion is **along a flat direction of the loss, not down it**.

We measured one such direction explicitly. The network output is `W_out · r`, which is invariant
under `W_out → W_out/c` together with `r → c·r`. Nothing in the loss pins the split. Adding raw
norm tracking to the trainer shows ‖W_out‖ falling **1.35 → 0.66 in the first 300 iterations** while
‖W_inp‖ (2.44 → 2.64) and ‖W_rec‖ (12.03 → 12.12) barely move. Weight decay is *not* the driver —
wd = 1e-6 cannot halve a norm in 300 steps; the task gradient is. This also explains a puzzle in the
drift figure: the apparent late "speed-up" of W_out is a shrinking denominator, not faster updates.

Motion along a flat valley has no restoring force to stop it, and because it is driven by a
systematic bias rather than by noise it is **directed** — which is exactly why α ≈ 1 at long lags.

**The role of Adam.** Adam's update is `lr·m̂/(√v̂+ε)`. Along a direction where the gradient is tiny
but *consistent*, `m̂/√v̂ ≈ ±1`, so Adam takes a **full lr-sized step** where SGD would take one
proportional to the (tiny) gradient. Adam traverses flat directions at roughly constant speed while
SGD would nearly stop. This is likely a substantial part of why directed motion persists.
**Untested, and a cheap decisive experiment:** re-run one N=100 cell with SGD+momentum; if this is
right, the persistent directed component should largely disappear.

## 9. Where we landed

Stop asking whether the network converged. Ask whether **the number that goes in the paper** has
stopped moving, in its own units:

> **T = the first training age after which the reported statistic moves by less than a stated
> precision over a doubling of the budget, and stays there.**

For the M\* question that statistic is the silent fraction (equivalently the active count), and the
precision is **1 percentage point of N**. This is superior to all six attempts above because it is
in interpretable units, needs no fit and no extrapolation, is directly the sensitivity a reader
wants, and cannot be gamed by the choice of denominator.

**And where convergence cannot be reached, the fallback does not need it.** Evaluate the M(N) curve
at several *common* budgets and ask whether the **verdict** — saturating versus still growing — is
the same at each. Budget-independence of the conclusion substitutes for convergence of the quantity,
and is a stronger claim than any single matched comparison.

## 10. The result of applying it — and an unresolved problem

Applied at N=100 (`silent_stopping_criterion.png`), the two silent-unit definitions **disagree
sharply**:

| criterion | silent fraction at 50k | movement per doubling |
|---|---|---|
| hard, `p < 1e-6` | 2.6% | 0.93 pp, still rising |
| scale-free, `p < 0.05·q₉₅(p)` | **29%** | 14.7 pp, accelerating |

Neither reaches T. The mechanism for the gap is the distribution stretch noted above: ‖p‖ grows
while the median falls, so `q₉₅` rises while the bulk sinks, and a *relative* threshold sweeps up
ever more units. The scale-free criterion is partly tracking rate heterogeneity, not only silencing.

Two guards had to be added to the detector, both after it initially gave a wrong answer (T = 3013
for the hard criterion): a curve can sit below threshold merely because **the run ended** while it is
trending back up, and a threshold finer than one unit is **below the measurement resolution**
(at N=100, 1 pp = 1 unit). Both now report "NOT REACHED" with the reason.

**Open items.**
- The scale-free criterion may not support a budget-matched comparison at all if it never settles at
  any N. The multi-budget shape test decides this.
- The M\* verdict must be reported under **both** criteria. If they disagree about saturation the
  way they disagree about magnitude here, that disagreement is the finding.
- N=100 is permanently the weak cell for this test (1 pp = 1 unit). N=1000 and N=2000, where 1 pp is
  10 and 20 units, are where the criterion has real power.
- The Adam-versus-SGD experiment in section 8 has not been run.

## 11. One-paragraph version

We wanted a rule for "trained enough" so that networks of different sizes could be compared fairly.
Six candidate metrics all failed: absolute drift has a lag-dependent noise floor and an untransferable
threshold; the directional cosine has a positive floor set by Adam's momentum, not by the network;
the lag-scaling exponent is a curve rather than a number and is blind to *how much* is moving; the
caging timescale grows in proportion to training age; the loss appears to converge only because 93–95%
of it is an irreducible noise floor sitting in the denominator; and the participation vector's change
per doubling is simply flat at ~28%. The reason none of them work is that every quantity here relaxes
as a power law — for which the fraction of remaining distance covered per doubling is constant, so
"converged" has no meaning — and, more fundamentally, that the loss minimum is a flat manifold rather
than a point: the loss falls 2% over the second half of training while the configuration moves 37%,
because the network slides along directions the loss does not penalise (we measured one, the
`W_out·r` rescaling degeneracy, with ‖W_out‖ halving in 300 iterations). The resolution is to stop
asking about convergence and instead report how much **the specific number being published** moves
when the training budget is doubled — and, where even that does not settle, to show that the
scientific conclusion is the same at every budget.

### Loss fits for N=100, and how well L_inf is actually determined

Figure: `img/internal_figures/loss_fit_N100.png`, from
`trainRNNbrain/experiments_and_analysis/plot_loss_fit.py`. Fit is `L(t) = L_inf + A·t^(−γ)` on
log-binned medians (so late iterations do not dominate), started at t = 2000.

| seed | L_∞ | γ | L(T) at 50k | irreducible | reducible left | L_∞ spread across fit windows |
|---|---|---|---|---|---|---|
| 0 | 0.02135 | 0.55 | 0.02261 | 94.4% | 5.3% | 3.2% |
| 1 | 0.02041 | 0.40 | 0.02273 | 89.8% | 9.2% | 7.1% |
| 2 | 0.02083 | 0.45 | 0.02265 | 91.9% | 7.8% | 6.3% |

**The power law holds cleanly.** Panel (b) plots `L(t) − L_∞` on log-log; all three seeds are
straight lines over the whole fitted range, which is the goodness check for the functional form.

**But L_∞ is not tightly determined, and this must be stated wherever the decomposition is used.**
Re-fitting from t = 2000, 5000 and 10000 moves L_∞ by 3.2–7.1%. Since the reducible part is only
5–9% of the total loss, that uncertainty is **comparable in size to the quantity being estimated**
(e.g. seed 0: L_∞ uncertainty ≈ 0.00068 versus reducible ≈ 0.0012, so ~57% of it). An earlier fit in
this document, run on raw rather than log-binned data, gave L_∞ = 0.02149/0.02113/0.02102,
γ = 0.56/0.48/0.47 and "93–95% irreducible"; the difference between the two procedures is of the
same order as the window spread, which is itself a consistency check.

**Corrected statement:** roughly **90–94%** of the loss at 50k is irreducible, with real uncertainty
on the exact split. The qualitative conclusion — most of the remaining loss cannot be trained away,
so raw loss improvement flatters the apparent convergence — is robust. The precise percentage is not,
from a 50k run. The 200k and 300k cells will constrain L_∞ far better, since the lever arm on the
asymptote grows with run length.

**Seeds genuinely differ**: γ ranges 0.40–0.55 and the reducible remainder 5.3–9.2%. This is not
fit noise; it is real seed-to-seed variation in how fast each network approaches its own floor.

**What a longer budget buys (panel c), reducible loss remaining as a % of total:**

| budget | 50k | 100k | 200k | 300k | 1M |
|---|---|---|---|---|---|
| seed 0 | 5.3% | 3.7% | 2.6% | 2.1% | 0.9% |
| seed 1 | 9.2% | 7.2% | 5.5% | 4.7% | 3.0% |
| seed 2 | 7.8% | 5.8% | 4.3% | 3.6% | 2.1% |

**This confirms the recommendation not to re-run at 300k.** Going 200k → 300k moves the remaining
reducible loss from ~4.1% to ~3.5% on average — a 0.6 percentage point gain for 9 jobs of ~11 h each.
Even 1M iterations, a 5× extension, only reaches ~2%.

---

## 2026-08-17 21:5x — N=500 complete: five times the units, identical loss

All 3 N=500 seeds finished (200k iterations, r² 0.867–0.873). Figures:
`img/internal_figures/loss_fit_N500.png` and `loss_fit_compare.png`.

### The loss curves are indistinguishable from N=100

| budget | N=100 | N=500 |
|---|---|---|
| 5,000 | 0.02564 ± 0.00032 | 0.02629 ± 0.00026 |
| 10,000 | 0.02460 ± 0.00008 | 0.02470 ± 0.00017 |
| 20,000 | 0.02350 ± 0.00008 | 0.02354 ± 0.00028 |
| 50,000 | 0.02266 ± 0.00005 | 0.02273 ± 0.00006 |
| 100,000 | — | 0.02237 ± 0.00003 |
| 200,000 | — | 0.02222 ± 0.00004 |

At every matched budget the two sizes agree to within about one standard deviation, and where they
differ at all it is **N=100 that is marginally lower**. r² agrees too (N=100: 0.870/0.873/0.898;
N=500: 0.867/0.871/0.873). Panel (a) of `loss_fit_compare.png` shows the curves lying on top of each
other over the whole overlapping range.

**Five times the units buys exactly nothing in task performance.**

### The fits, and why the N=500 ones are the trustworthy ones

| | L_∞ | γ | seed spread in L_∞ | L_∞ spread across fit windows |
|---|---|---|---|---|
| N=100 (50k) | 0.02086 ± 0.00039 | 0.47 ± 0.06 | 4.5% | 3.2–7.1% |
| N=500 (200k) | **0.02135 ± 0.00003** | 0.52 ± 0.01 | **0.3%** | 1.5–2.4% |

The prediction made when only N=100 was available — that a longer run would pin L_∞ far better —
held: the seed spread is **15× tighter** at N=500 and the fit-window spread roughly halved. The two
L_∞ estimates are consistent with a single common floor of **≈ 0.0213**; the lower N=100 value is
within its own (large) uncertainty and is most likely biased by extrapolating from a 4× shorter run.

At 200k, N=500 has **3.0–3.2%** of its loss still reducible.

### Why this matters for M*

If both sizes sit at the same floor, then **every network in this sweep is over-parameterised** —
the task's capacity requirement is already met at N=100, and probably well below. That reframes the
silent-unit phenomenon: silencing is what training does with capacity it does not need.

But it does *not* imply the network collapses down to the needed capacity. The earlier
population-distortion analysis measured **574 active units at N=1000** (h, no penalty), versus ~97 at
N=100 — so the active count grows several-fold with N while task performance does not improve at all.
If that holds up when the N=1000 drift cell lands (which is the matched-protocol version of the same
measurement, so it must be re-checked rather than assumed), the sharpened statement is:

> the number of active units is not set by what the task requires — it is set by the network's own
> size and its optimisation dynamics, and the surplus units contribute nothing measurable.

That is a stronger and more interesting claim than "some units go silent", and it is testable
against exactly the M(N) curve this sweep produces.

**Caveat to carry:** the loss compared here is the *training* loss, which includes the injected noise,
so L_∞ is a genuine stochastic floor rather than an optimisation failure. That is the correct
interpretation of "both sizes are at the floor", but it also means the loss cannot distinguish
networks once they are both at it — which is precisely why the M(N) comparison has to be made on
active-unit counts and not on performance.

### Attempt 7 — normalise by the trainable amplitude: right instinct, but it inverts a power law

**Proposal.** Estimate the total trainable amplitude `A = L(0) − L_∞` and train until the remaining
reducible loss is below a fraction of it: `L(t) − L_∞ < f·A`, e.g. f = 1%.

**What is right about it.** It fixes the exact flaw identified in Attempt 5: dividing by the *total*
loss flatters convergence because 90–96% of that loss is an irreducible floor. Dividing by the
trainable amplitude instead is the honest normalisation, and as a *descriptive* statement — "this run
covered 99% of the loss the optimiser could ever remove" — it is scale-free, interpretable, and worth
reporting.

**Why it fails as a stopping rule.** Inverting `L(t) − L_∞ = A_fit·t^(−γ)` for t gives
`T ∝ (1/f)^(1/γ)` and `T ∝ A^(1/γ)`. With γ ≈ 0.5 both exponents are ≈ **2**, so every arbitrary
choice is squared.

*Sensitivity to the reference point defining L(0)* — the same seeds, same fit, only the choice of
which early iteration counts as "the start":

| N | seed | L(0) = iter 1 | iter 100 | iter 1000 |
|---|---|---|---|---|
| 100 | 0 | 11,276 | 108,598 | 1,144,691 |
| 100 | 1 | 30,517 | 587,913 | 13,834,023 |
| 100 | 2 | 19,452 | 369,835 | 5,841,236 |
| 500 | 0 | 18,245 | 248,718 | 1,927,711 |
| 500 | 1 | 14,362 | 233,784 | 1,894,168 |
| 500 | 2 | 12,533 | 217,127 | 1,915,397 |

A factor of **100** in the recommended budget, from a choice nothing in the problem determines.
(The loss falls steeply early, so L(0) is not a well-defined quantity: 0.294 at iteration 1, 0.100 by
iteration 100, ~0.05 by iteration 1000.)

*Sensitivity to the threshold f* (N=500, seed 0):

| f | 5% | 2% | 1% | 0.5% | 0.1% |
|---|---|---|---|---|---|
| T | 11,715 | 66,714 | 248,718 | 927,251 | 19,686,006 |

A factor of 20 in f becomes a factor of **1700** in T.

*And it needs a long run to be usable at all.* T(f=1%) is 355,449 ± 195,944 at N=100 (55% seed
spread, because L_∞ and γ are poorly determined from 50k) versus 233,209 ± 12,903 at N=500 (5.5%).
Circular: a reliable estimate of how long to train requires having already trained long.

**The general lesson, which applies to every criterion of this family.** Any rule of the form "train
until quantity X falls below a threshold", where X decays as a power law, must invert that power law
and therefore inherits an exponent of 1/γ ≈ 2 on both the threshold and any scale factor. It
*amplifies* arbitrariness. The per-doubling rule does not invert anything — it reads a local rate —
which is why its output moves roughly linearly with the threshold rather than quadratically. That is
the structural reason to prefer it.

**Verdict.** Keep amplitude-normalised remaining loss as a *reported* number (it is the honest way to
say "the optimisation is essentially complete"). Do not use it to choose the budget. And note that
even a perfectly well-defined loss criterion would still answer the wrong question here, since the
loss finishes long before the statistic being published does.

---

## 2026-08-17 22:2x — N=1000 complete, a retraction, and the comparability problem SOLVED

### Retraction: the N=1000 seed spread was a reading error

An earlier note in this conversation reported N=1000 losses of 0.0178 / 0.0222 / 0.0251, called the
seed spread "~700× wider than N=500", and inferred that the loss floor is not universal across sizes.
**All of that is wrong.** Those numbers were read from the tail of the SLURM log, which prints the
*instantaneous single-batch* loss — pure noise around the true value. The smoothed losses are
0.02225 / 0.02235 / 0.02232, a spread of **0.4%**, and the fitted floors are
0.02135 / 0.02138 / 0.02147.

### The three sizes share one floor

| N | L_∞ | γ | seed spread in L_∞ | L(T) | reducible left |
|---|---|---|---|---|---|
| 100 (50k) | 0.02086 ± 0.00039 | 0.47 ± 0.06 | 4.5% | 0.02266 | 5.3–9.2% |
| 500 (200k) | 0.02135 ± 0.00003 | 0.52 ± 0.01 | 0.3% | 0.02222 | 3.0–3.2% |
| 1000 (200k) | **0.02140 ± 0.00005** | 0.54 ± 0.01 | 0.6% | 0.02225 | 2.9–3.3% |

L_∞ agrees to **0.2%** between N=500 and N=1000. N=100's slightly lower estimate is inside its own
much larger uncertainty and is a short-run artefact. A single task-imposed floor at ≈ **0.0214**
governs every size: 10× the units buys no better performance.

### Loss at matched budgets — the curves nearly coincide

| budget | N=100 | N=500 | N=1000 |
|---|---|---|---|
| 10,000 | 0.02460 ± 0.00008 | 0.02470 ± 0.00017 | 0.02484 ± 0.00005 |
| 20,000 | 0.02350 ± 0.00008 | 0.02354 ± 0.00028 | 0.02369 ± 0.00011 |
| 50,000 | 0.02266 ± 0.00005 | 0.02273 ± 0.00006 | 0.02284 ± 0.00006 |
| 100,000 | — | 0.02237 ± 0.00003 | 0.02250 ± 0.00003 |
| 200,000 | — | 0.02222 ± 0.00004 | 0.02225 ± 0.00002 |

Bigger networks are *very slightly* behind at equal iterations — consistent with their smaller
learning rate — but the gap is under 1%.

### THE ANSWER: match on performance, not on iterations

Comparability does **not** require convergence, and the previous seven attempts failed because they
all tried to answer the harder question. The tractable question is: *when is each size equally good
at the task?* Read the iteration at which each size first reaches a chosen loss:

| target loss | N=100 | N=500 | N=1000 | T(1000)/T(100) |
|---|---|---|---|---|
| 0.0250 | 6,500 ± 707 | 8,333 ± 236 | 9,667 ± 236 | 1.49 |
| 0.0240 | 12,167 ± 471 | 13,000 ± 408 | 16,167 ± 236 | 1.33 |
| 0.0235 | 18,000 ± 1,780 | 19,500 ± 408 | 23,167 ± 943 | 1.29 |
| 0.0230 | 30,000 ± 3,189 | 32,500 ± 1,080 | 37,333 ± 2,321 | 1.24 |
| 0.0228 | 37,167 ± 2,248 | 40,500 ± 408 | 49,000 ± 2,041 | 1.32 |
| 0.0226 | not reached | 59,500 ± 3,342 | 64,333 ± 2,656 | — |
| 0.0224 | not reached | 78,667 ± 4,989 | 106,833 ± 7,663 | — |

**Why this works where everything else failed.** It needs no fit, no L_∞, no asymptote, no threshold
inverted through a power law, and no claim that anything has converged. It is simply: *compare
networks that are equally good at the task.* It also absorbs the learning-rate difference
automatically — a network trained at lower lr just needs more iterations to reach the same loss,
which is exactly what a fair matching should charge it.

**The ratios are stable**, which is the property that matters: T(500)/T(100) = 1.07–1.28 and
T(1000)/T(100) = 1.24–1.49 across the whole usable range of target losses. So the *relative* budgets
are well determined even though the absolute budget depends on which loss level is chosen.

**Recipe for the M\* comparison.**
1. Choose a target loss every size reaches with margin — **0.0230** is a good default (all sizes,
   ~30–37k iterations, comfortably inside every run).
2. Read M(N) — the active-unit count — at each size's own T_N for that target.
3. Repeat at several target losses (0.0240, 0.0235, 0.0230, 0.0228) and confirm the
   saturating-versus-growing **verdict** is unchanged. That is the robustness check that makes the
   arbitrary choice of level harmless.
4. Report alongside each M(N) its movement per budget doubling, as the sensitivity.

Implemented as `matched_performance_budget()` in
`trainRNNbrain/experiments_and_analysis/plot_loss_fit.py`. Every number above comes from traces
already on disk — no re-run, and the 200k budgets are ample, since matching happens at 30–50k.

### Evidence that the loss floor is shared across sizes (and one caveat about its value)

Figure: `img/internal_figures/shared_floor.png`, from
`trainRNNbrain/experiments_and_analysis/plot_shared_floor.py`.

**Why a naive comparison would not have settled it.** L_∞ comes from extrapolating
`L(t) = L_∞ + A·t^(−γ)`, and that extrapolation is biased by how much data it sees — fitting a curve
that has not yet flattened systematically *underestimates* the asymptote. N=100 had 50k iterations
and the others 200k, so their raw L_∞ estimates are not comparable, and the apparent "N=100 has a
lower floor" was entirely this artefact.

**The controlled test.** Refit every size using only its first `t_max` iterations, so the same bias
applies to all of them, and ask whether the estimates agree:

| t_max | N=100 | N=500 | N=1000 | max pairwise diff |
|---|---|---|---|---|
| 20,000 | 0.01954 ± 0.00163 | 0.01905 ± 0.00143 | 0.01952 ± 0.00111 | 0.00048 (0.2 sd) |
| 30,000 | 0.02025 ± 0.00069 | 0.02044 ± 0.00062 | 0.02036 ± 0.00056 | 0.00019 (0.2 sd) |
| 50,000 | 0.02086 ± 0.00039 | 0.02080 ± 0.00013 | 0.02086 ± 0.00016 | 0.00007 (0.2 sd) |
| 75,000 | — | 0.02102 ± 0.00010 | 0.02101 ± 0.00013 | 0.00001 (0.1 sd) |
| 100,000 | — | 0.02111 ± 0.00007 | 0.02118 ± 0.00012 | 0.00007 (0.5 sd) |
| 150,000 | — | 0.02127 ± 0.00003 | 0.02131 ± 0.00008 | 0.00004 (0.5 sd) |
| 200,000 | — | 0.02135 ± 0.00003 | 0.02140 ± 0.00005 | 0.00005 (0.9 sd) |

**At every matched data length the sizes agree to within 0.1–0.9 standard deviations.** A tenfold
change in N moves the floor by at most 0.00005 out of 0.0214 — **under 0.3%**. The falsification test
was explicit: if bigger networks reached a lower floor, the curves in panel (a) would separate as N
increases. They lie on top of each other.

**Honest note on the residual.** At the three largest t_max the N=1000 estimate is above N=500 every
time (+0.00007, +0.00004, +0.00005). Individually all are ≤0.9 sd, and three same-signed differences
from 3 seeds is not significant (p ≈ 0.25 two-tailed), but it is a consistent sign and should not be
hidden. If real it would mean bigger networks are *marginally worse*, not better — the opposite of the
confound we were guarding against. Either way the magnitude (<0.3%) is far too small to affect an
M(N) comparison where the active count changes several-fold.

**The caveat that does matter: the VALUE of the floor is not determined.** Panel (a) shows the
estimate still climbing with t_max — 0.0195 → 0.0202 → 0.0209 → 0.0210 → 0.0211 → 0.0213 → 0.0214 —
with no sign of levelling. So **L_∞ ≳ 0.0214 and rising with fit range**; quoting 0.0214 as a
converged asymptote would be wrong, and any statement of the form "X% of the loss is irreducible"
inherits that uncertainty (the irreducible share is a *lower* bound; the reducible share an upper
bound).

Crucially, **the shared-floor conclusion does not depend on knowing the value**, because the
comparison is controlled: whatever bias remains applies equally to every size. That is the entire
point of matching on t_max.

**Pre-registered prediction for N=2000** (in progress, ~11 h out): refitted at t_max = 200,000 it
should give L_∞ within about 0.0001 of the 0.02135–0.02140 that N=500 and N=1000 give. If it comes in
materially lower, the shared-floor claim fails and the M(N) comparison becomes confounded by
performance. Stated now, before the data exists.

### The loss in power-law coordinates, without assuming the asymptote

Figure: `img/internal_figures/powerlaw_coords.png`, from
`trainRNNbrain/experiments_and_analysis/plot_powerlaw_coords.py`.

**The circularity to avoid.** The conventional linearisation of `L = L_∞ + A·t^(−γ)` plots
`log(L − L_∞)` against `log t`. It does give a straight line — but L_∞ is a fitted parameter chosen
to make the power law fit, so its straightness is partly guaranteed by construction. Panel (c) is
that plot, labelled as such.

**The L_∞-free coordinate.** Differentiating with respect to log t removes the constant entirely:

`−dL/d(log t) = A·γ·t^(−γ)`  ⟹  `log(−dL/dlog t) = log(Aγ) − γ·log t`

so the log-log slope is exactly **−γ** with no asymptote anywhere in it. In practice the pointwise
derivative of a noisy loss is unusable (the first version of this figure scattered over an order of
magnitude), so the same quantity is measured across a whole factor of two instead:

`L(t/2) − L(t) = A·t^(−γ)·(2^γ − 1) ∝ t^(−γ)`

— identical scaling, still no L_∞, far less noise. Panel (b) is straight over ~1.7 decades for every
seed at every size. **The power-law form is supported without assuming the asymptote.**

**γ does not depend on N.** Applying the lesson from the L_∞ analysis, the exponent is compared at
matched fit ranges:

| fit range | N=100 | N=500 | N=1000 |
|---|---|---|---|
| ≤ 50,000 | 0.468 ± 0.060 | 0.507 ± 0.016 | 0.500 ± 0.059 |
| ≤ 100,000 | — | 0.563 ± 0.019 | 0.561 ± 0.025 |
| ≤ 200,000 | — | 0.592 ± 0.025 | 0.609 ± 0.024 |

At matched range the sizes agree within error. The apparent "γ increases with N" in the unmatched
table was the same run-length artefact that produced the apparent L_∞ difference.

**But γ drifts upward with fit range — so it is not a single global power law.** 0.47–0.51 at 50k,
0.56 at 100k, 0.59–0.61 at 200k. A true fixed-exponent power law would give the same slope at every
range. The loss is decaying *faster* than a pure power law at late times, i.e. the approach to the
floor accelerates slightly.

Two consequences.
- It **explains** why L_∞ keeps rising with fit window: a fixed-γ fit forced onto a steepening decay
  compensates by pushing the asymptote down. The two biases are the same phenomenon.
- The three-parameter fit's γ (0.524, 0.541 at N=500, 1000) is systematically **below** the
  L_∞-free estimate (0.592, 0.609) — by ~0.07 in all six long-run seeds. The L_∞-free value is the
  one to trust, since it has no asymptote to trade against.

**Caveat.** Part of the late steepening could be an edge effect: the last points of panel (b) sit
below their fitted line, and they are the ones nearest the end of the trace. The N=2000 run (300k)
gives a longer lever arm and will show whether the drift continues or flattens.

### The local exponent: the loss is NOT a single power law, and earlier extrapolations were pessimistic

Figure: `img/internal_figures/local_exponent.png`, from
`trainRNNbrain/experiments_and_analysis/plot_local_exponent.py`. This measures
`gamma_eff(t) = −d log D / d log t` for `D(t) = L(t/2) − L(t)`, in a sliding factor-4 window — a
local slope, using no L_∞ anywhere.

| N | t=8k | t=15k | t=30k | t=60k | t=120k | t=180k |
|---|---|---|---|---|---|---|
| 100 | 0.411 ± 0.069 | 0.459 ± 0.097 | 0.727 ± 0.041 | — | — | — |
| 500 | 0.315 ± 0.095 | 0.484 ± 0.073 | 0.636 ± 0.083 | 0.672 ± 0.142 | 0.860 ± 0.203 | 0.960 ± 0.265 |
| 1000 | 0.324 ± 0.095 | 0.608 ± 0.032 | 0.601 ± 0.075 | 0.685 ± 0.068 | 0.799 ± 0.116 | 0.668 ± 0.212 |

**It is not an edge effect.** The figure was built to make that distinction: an artefact would show
gamma_eff flat until the last few points, whereas a real steepening shows it rising smoothly through
the middle. It rises monotonically across the whole span, roughly **0.3 → 0.8**, from t = 8k to
t = 120k. Nor can noise produce it: noise in log D adds variance to the slope estimate but no
systematic steepening.

**The sizes agree** (panel c, bands overlap throughout), so the *shape* of the decay is
size-independent — consistent with everything else about the shared trajectory.

**Consequence 1 — this explains two earlier puzzles at once.** A fixed-γ fit forced onto a steepening
decay must compensate by pushing L_∞ down, which is exactly why the estimated floor kept rising with
fit range, and why the three-parameter γ sat ~0.07 below the L_∞-free estimate. Both are one
misspecification: the model `L_∞ + A·t^(−γ)` has a constant exponent and the data do not.

**Consequence 2 — the pessimistic extrapolations in this document should be softened.** Every "you
would need N million iterations" estimate assumed a fixed γ ≈ 0.5, and a decay that steepens beats a
fixed power law by a growing margin. Concretely, halving the remaining reducible loss costs
`2^(1/γ)` in extra training: **4× at γ = 0.5, but only 2.2× at γ = 0.9**. The following claims are
therefore upper bounds, probably substantial ones, and should not be quoted as-is:
- "a 1% criterion extrapolates to 0.5–5.6 M iterations"
- "T(f = 1% of amplitude) = 233k–355k"
- the panel-(c) budget projections in `loss_fit_N*.png`

**What does NOT change.** The shared-floor conclusion, because it was established by a *matched*
comparison in which any misspecification applies equally to every size. The performance-matching
recipe for T(N), because it uses no fit at all. And the fact that participation keeps moving after
the loss has effectively settled, since that rests on measured per-doubling movement, not on
extrapolation.

**Open.** The functional form is unidentified — a steadily rising log-log slope is consistent with a
stretched exponential or similar, but this run does not have the range to pin it down, and it is not
needed for any current conclusion. The N=2000 300k trace extends the lever arm by another 0.2 decades
and will show whether gamma_eff keeps climbing or levels near 1.

### Where did "power law" come from? A correction to the framing used throughout this document

**Direct answer: the loss decay is NOT a power law, and this project's own data rejects it.** The
form `L = L_∞ + A·t^(−γ)` was an assumption introduced during this analysis, not a result. It was
never established here, and the local-exponent measurement falsifies the constant-exponent version:
a power law requires gamma_eff(t) to be flat, and it climbs 0.3 → 0.8.

**Why it was a reasonable first guess.** Near a minimum, gradient descent on a quadratic decays each
Hessian eigendirection as `exp(−η·λ_i·t)`. A network has a *broad spectrum* of curvatures, and a sum
of exponentials with a power-law-distributed set of rates looks like a power law over any window in
which many modes are still decaying. This is the standard spectral account of learning curves
(Bordelon, Canatar & Pehlevan 2020, on spectrum-dependent learning curves in kernel regression and
wide networks; Canatar et al. 2021; Bahri et al., "Explaining Neural Scaling Laws"). *Cited from
memory — verify before use.*

**And that same account predicts the steepening we observe.** The power-law phase is transient: it
lasts while a broad range of modes is still relaxing, and must steepen once only the slowest modes
remain. Our gamma_eff rising at late times is what the spectral picture predicts, not an anomaly.

**What the literature actually established, and where.** The well-known power laws — Hestness et al.
2017 (Baidu); Kaplan et al. 2020 (OpenAI); Rosenfeld et al. 2019; Hoffmann et al. 2022 (Chinchilla) —
are scaling of loss with **model size, dataset size and compute**, in large language models, image
classifiers, machine translation and speech. They are *not* a general law for how loss decays with
training iteration in a small recurrent network. Note also that theory does not uniformly predict
power laws at all: in the NTK/linearised regime (Jacot et al. 2018; Du et al. 2019) the training loss
converges **exponentially**.

**Applicability to this project is weak.** These runs use `same_batch=True` on a fixed 450-condition
batch, so there is no dataset-size scaling whatsoever — it is pure optimisation on a fixed objective
plus injected noise, in a 100–2000 unit RNN. The LLM scaling-law literature is at best a loose
analogy.

**What our data can and cannot distinguish.** A stretched exponential,
`L − L_∞ = A·exp(−(t/τ)^β)`, predicts `gamma_eff = β[(t/τ)^β − 1]`, which rises — matching the
observation. Fitting it to gamma_eff(t) gives R² = 0.25–0.67 against 0.00 for a constant exponent, so
"rising" beats "flat". But β = 0.10–0.25 and τ = 1–696 across sizes: the parameters are wildly
unstable and the form is **not identified**. Supported: *faster than any fixed power law*.
Not supported: any specific alternative.

### Consequence for the "six metrics failed" summary

That summary rests partly on "every quantity here relaxes as a power law, and for a power law the
fraction of remaining distance covered per doubling is constant, so 'converged' has no meaning".
**That argument is weakened for the loss**: with gamma_eff rising from 0.3 to 0.8, the fraction of
remaining reducible loss removed per doubling goes from ~19% to ~43% — increasing, not constant. The
loss converges *faster* than the argument assumed.

**But the participation conclusion is unaffected**, because it never depended on the power-law gloss.
The 28%-per-doubling figure for `p` is a *direct measurement*, flat across the whole late run, with no
fit and no assumed functional form. The metrics failed for the reasons individually documented —
lag-dependent floors, momentum artefacts, magnitude-blindness, denominator choice — and those stand.
What should be dropped is the tidy theoretical story that they all failed *because* of power laws.

---

## THE PROTOCOL: how the sizes are matched (definitive; supersedes attempts 1-7)

**Criterion: train each size until its training loss reaches the same value L\*.**

That is the whole rule. No convergence claim, no fit, no asymptote, no threshold inverted through a
decay law.

### Why equal loss is the right matching variable here

It is right *because the floor is shared*, which was measured rather than assumed (see the
shared-floor section: L_∞ agrees across N=100/500/1000 to within 0.3% at matched fit length, 0.1–0.9
sd). The chain is:

> shared L_∞  ⟹  equal L means equal L − L_∞  ⟹  equal distance from the achievable floor
> ⟹  equal fraction of the trainable improvement completed.

Had the floor differed with N, raw loss would have been the *wrong* variable — a network with a lower
floor sitting at the same raw loss is further from its own optimum. The shared-floor measurement is
what licenses the simple rule, which is why that test was worth doing.

Note what is NOT needed: the *value* of L_∞ (still poorly determined, and rising with fit range), the
functional form of the decay (power law vs stretched — unresolved and irrelevant here), and any
statement about convergence. The rule uses only the raw loss trace.

It also absorbs the learning-rate difference automatically. lr = 1e-3·(100/N)^(1/3), so N=2000 trains
at 0.37× the rate of N=100; a slower network simply needs more iterations to reach L*, which is
exactly what a fair matching should charge it.

### The numbers

| L* | N=100 | N=500 | N=1000 |
|---|---|---|---|
| 0.0240 | 12,167 ± 471 | 13,000 ± 408 | 16,167 ± 236 |
| 0.0235 | 18,000 ± 1,780 | 19,500 ± 408 | 23,167 ± 943 |
| **0.0230** | **30,000 ± 3,189** | **32,500 ± 1,080** | **37,333 ± 2,321** |
| 0.0228 | 37,167 ± 2,248 | 40,500 ± 408 | 49,000 ± 2,041 |

`matched_performance_budget()` in `plot_loss_fit.py`.

### Robustness: the choice of matching method barely matters

Because the loss curves nearly coincide across sizes (within 1% at equal iterations), matching on
loss and matching on iterations give almost the same answer: T(1000)/T(100) = 1.24–1.49 and
T(500)/T(100) = 1.07–1.28. So the M(N) comparison is not sensitive to which is used. Loss-matching is
the principled choice; iteration-matching is a close approximation that happens to be defensible here
but would not be in general.

### Full procedure for the M* result

1. **Primary comparison at L\* = 0.0230**, i.e. ~30k / 32.5k / 37.3k iterations. Read the
   active-unit count M(N) at each size's own T_N from the stored participation traces.
2. **Repeat at L\* = 0.0240, 0.0235, 0.0228** and confirm the saturating-versus-growing *verdict* is
   unchanged. This is what makes the arbitrary choice of level harmless.
3. **Also report the matched-ITERATION comparison at the common 200k budget**, as a complementary
   view at a much later stage of training. The verdict must agree with (1)–(2); if it does not, M* is
   budget-dependent and that is the finding.
4. **Report both silent-unit criteria** (hard `p<1e-6` and scale-free `p<0.05·q95`), since at N=100
   they disagree by an order of magnitude and may not agree about saturation either.
5. **Attach two error bars to every M(N) point**: seed spread, and movement of M over the last budget
   doubling (the sensitivity, since M itself does not converge).

### What is still open

- Whether the deepest matched level can be pushed below 0.0226 for all four sizes. The N=100 top-up
  to 200k (running, ~1.5 h out) will let N=100 reach ≈0.0222 and unlock the deeper levels.
- N=2000 (~11 h out) must satisfy the pre-registered floor prediction (L_∞ within ~0.0001 of
  0.02135–0.02140 at t_max=200k). If it does not, the shared-floor justification fails for that size
  and its point on the M(N) curve becomes confounded by performance.

### How the matching levels L* are chosen (replaces the hand-picked set)

The levels used in the first pass (0.0240 / 0.0235 / 0.0230 / 0.0228) were picked by eye from the
range every size reaches. That is not a principle, and it has a concrete flaw: those four levels span
T ≈ 12k–50k, a factor of 4 — only two doublings — so they cluster at the deep end and leave the early
range unsampled. Four points in two doublings is over-sampling a narrow window.

Two constraints fix the range, one choice fixes the spacing, and the NUMBER of levels then follows
from the data rather than being chosen.

**Constraint 1 — the deepest level is the worst final loss across all seeds.** Every seed of every
size must actually reach L*, otherwise the comparison silently drops the seeds that did not, which is
survivorship. Currently `L_deep = 0.02273`, set by the slowest N=100 seed — i.e. by the shortest run
in the sweep. *(The N=100 top-up to 200k, running, removes exactly this bottleneck: it should reach
≈0.0222, moving L_deep to ≈0.0223 and unlocking one to two deeper rungs.)*

**Constraint 2 — the shallowest level must be past the transient.** Before the solution has formed,
active-unit counts reflect initialisation rather than training. Operationalised as T ≥ 5000 for the
slowest size. This one is a judgement call; the alternative (a fixed fraction of the total loss drop)
was rejected because it reintroduces the arbitrary L(0) reference that sank Attempt 7.

**Spacing — halve the reference size's budget at each rung.** log t is the natural axis, because
every quantity here decays as a power of t, and because it is the same axis as the per-doubling
sensitivity measure already in use. The reference is the largest size present.

| rung | T_ref | L* | T(N=100) | T(N=500) | T(N=1000) |
|---|---|---|---|---|---|
| 0 | 60,400 | 0.02273 | 40,267 ± 1,543 | 46,067 ± 3,866 | 54,867 ± 3,934 |
| 1 | 30,200 | 0.02334 | 18,733 ± 1,636 | 21,467 ± 2,217 | 26,600 ± 1,840 |
| 2 | 15,100 | 0.02417 | 11,400 ± 816 | 12,867 ± 340 | 14,600 ± 993 |
| 3 | 7,550 | 0.02574 | 5,067 ± 94 | 5,667 ± 189 | 6,467 ± 94 |

Four rungs spanning T ≈ 5k–55k, a factor of 11 (3.5 doublings) — considerably better coverage than
the hand-picked set, and the count is determined by the constraints rather than assumed.

**Use:** compute M(N) at every rung. If the saturating-versus-growing verdict is the same across a
factor of 11 in training time, it is not an artefact of where we chose to look. If it changes, that
change *is* the result and must be reported as such — the first pass already showed the ratio
M(1000)/M(500) moving from 1.64 to 1.30 across a narrower range, so this is a live possibility rather
than a formality.

**Finalise after the N=100 top-up lands**, since L_deep is currently pinned by the one short run and
will move once it does.

---

## 2026-08-18 00:07 — N=5000 added to the sweep (array 5663153)

**Why.** The M(N) curve currently has three points and effectively only *two* informative ones:
N=100 sits at 98% active, pinned against the M ≤ N ceiling, so the steep first segment is partly
forced and k(100→500) is inflated. The only ceiling-free evidence for bending is the single pair
500→1000. N=5000 gives three ceiling-free points (1000, 2000, 5000) spanning a factor of 5 — enough
to separate saturation from a power law, which the present data cannot do.

**Feasibility was measured, not assumed** (`benchmark_large_N.py`, real training steps):

| N | params | peak allocated | peak reserved | s/iter | GPU |
|---|---|---|---|---|---|
| 2000 | 4.0 M | 10.90 G | 13.22 G | 0.245 | L40S-46G |
| 5000 | 25.0 M | 27.50 G | **32.98 G** | 1.007 | L40S-46G |
| 5000 | 25.0 M | 27.50 G | **30.58 G** | 1.689 | A100-40G + expandable_segments |

N=5000 is **memory-constrained, not merely slow**: 2.5× the memory and 4× the time of N=2000. At
33 GB reserved it is 72% of an L40S but 82% of an A100-40G, and every Spock A100 is the 40 GB
variant. Pinning to L40S was rejected because 11 of its 12 GPUs were busy — two of three seeds would
have queued indefinitely. Instead `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is now set in
the launcher: it drops reserved memory to 30.6 GB (72% of the A100) so both card types fit. It is an
allocator setting only — it changes how CUDA memory is carved up, not the numerics.

Unexpected: the **A100-PCIE is 1.7× slower than the L40S** for this workload (1.689 vs 1.007 s/iter),
so the wall-time request is sized for the slow case.

**Budget: 100k iterations, not 300k.** Matched-performance reading means each size is read at the
iteration where its loss reaches L*, and T scales only as ~N^0.13 (40k / 46k / 55k at N=100/500/1000
for L*=0.02273), giving T(5000) ≈ 68k. 100k leaves margin for a deeper rung once the N=100 top-up
extends the common range. Runtime 28 h (L40S) to 47 h (A100); 72 h requested.

**Submitted and verified running:** array 5663153, tasks 13–15, one seed on an L40S (spockmk2-22-04)
and two on an A100 node (spockmk2-13), all started immediately with no queueing.

Launcher now carries `NS=(100 500 1000 2000 5000)` and `ITS=(50000 200000 200000 300000 100000)`;
the size arrays remain the only place a swept parameter is varied.

---

## 2026-08-18 11:45 — M(N) with four sizes: NO SATURATION. Both pre-registered tests resolved.

N=100 top-up (200k, 3 seeds) and the first N=2000 seed (300k) are in, so the ladder now reaches
L_deep = 0.02226 (was 0.02273) and all four sizes are on it. Figures: `img/internal_figures/M_vs_N.png`
(four levels x two criteria) and `M_vs_N_deepest.png`.

### Pre-registered test 1 — the shared floor: PASSES, and the N=100 anomaly is explained

Predicted (before the data existed): L_∞ for N=2000 within ~0.0001 of the 0.02135–0.02140 given by
N=500 and N=1000, fitted on the first 200k iterations.

| N | L_∞ (t_max = 200k) | n |
|---|---|---|
| 100 | 0.02141 ± 0.00000 | 3 |
| 500 | 0.02135 ± 0.00003 | 3 |
| 1000 | 0.02140 ± 0.00005 | 3 |
| **2000** | **0.02145** | 1 |

N=2000 lands at 0.02145 — 0.00005 above the range, inside the stated tolerance. Total spread across a
**20-fold** range of N is 0.00010, i.e. **0.5%**.

Independently, the N=100 top-up settles the one loose end in that analysis. Its 50k run gave
L_∞ = 0.02086, which was attributed to run-length bias rather than to a genuinely lower floor. Refitted
on 200k it gives **0.02141**, in line with every other size. The artefact explanation was correct.

### Pre-registered test 2 — M(2000): the "no saturation" branch wins

Predicted at L* = 0.02273: **583** if k holds at 0.38, **~515** if bending continues, **448** if
genuinely saturated.

| criterion | N=100 | N=500 | N=1000 | N=2000 | k(100→500) | k(500→1000) | k(1000→2000) |
|---|---|---|---|---|---|---|---|
| hard | 99.3 ± 0.5 | 344.0 ± 12.0 | 448.0 ± 9.0 | **566** | 0.77 | 0.38 | **0.34** |
| scale-free | 80.0 ± 1.4 | 280.7 ± 7.6 | 360.7 ± 21.5 | **470** | 0.78 | 0.36 | **0.38** |

**Measured 566, against 583 predicted for "k holds" and 448 for "saturated".** Saturation is refuted.
The local exponent stops falling: 0.38 → 0.34 (hard) and 0.36 → 0.38 (scale-free), i.e. flat within
the scatter rather than heading to zero.

### The result, across all four levels (hard criterion)

| L* | N=100 | N=500 | N=1000 | N=2000 | k(500→1000) | k(1000→2000) |
|---|---|---|---|---|---|---|
| 0.02226 | 94.0 | 259.0 | 310.3 | 371 | 0.26 | **0.26** |
| 0.02253 | 99.0 | 306.3 | 400.3 | 469 | 0.39 | 0.23 |
| 0.02301 | 99.3 | 387.3 | 528.0 | 686 | 0.45 | 0.38 |
| 0.02369 | 99.3 | 445.7 | 692.7 | 917 | 0.64 | 0.40 |

**M grows without saturating, as a sub-linear power law M ∝ N^k with k ≈ 0.25–0.40.** At the deepest
level the exponent is *identical* for the last two pairs (0.26, 0.26) — a clean power law, not a
curve bending to zero. The bending seen earlier with three sizes was the N=100 ceiling effect
(N=100 sits at 94–99% active, pinned against M ≤ N), exactly as flagged: k(100→500) is inflated at
every level, and the two ceiling-free pairs agree with each other.

**What this means for the project's central question.** There is no finite M* in the range tested.
Bigger networks do recruit more active units — but so inefficiently that at the deepest matching
level, **doubling the active count requires a 14-fold increase in N** (2^(1/0.26)). The active
*fraction* M/N falls steadily with no plateau (panel b). So the motivating argument for penalties
survives in a modified form: penalties are not the *only* route to more active units, but they are
overwhelmingly the cheaper one.

**Caveat: N=2000 is n=1.** Error bars on that point are zero because there is one seed; the other two
land ~19:10 today. Every k involving N=2000 is provisional until then. N=5000 (~00:30 tonight for the
first seed, ~22:30 Aug 19 for the other two) extends the ceiling-free range to a factor of 5 and is
the real test of whether k ≈ 0.26 holds or drifts further.

---

## 2026-08-18 12:05 — Saturation vs power law: a formal test. VERDICT: cannot yet distinguish.

Figure: `img/internal_figures/saturation_test.png`, from
`trainRNNbrain/experiments_and_analysis/test_saturation.py`. Criteria fixed before running:
curvature significant at p < 0.05; model comparison by AICc with dAICc > 4 substantial, > 10 decisive.

**N=100 is excluded from every fit.** At N=100 the networks sit at 94–99% active, hard against the
M ≤ N ceiling — a 100-unit network cannot show that it "wants" 120 active units. That censoring
flattens M(100), inflates the apparent exponent from 100 to 500, and manufactures exactly the
downward curvature that saturation predicts. Only N ≥ 500 (all below 78% active) is fitted. This is
why the earlier three-size figure looked like it was bending.

### Test 1 — pooled curvature: NOT significant

`log M = a_level + k·log N (+ c·(log N)²)`, one intercept per matching level, curvature shared across
levels (which is what buys the degrees of freedom a single level cannot).

| criterion | k (straight) | c (curvature) | F(1,22) | p | verdict |
|---|---|---|---|---|---|
| hard | 0.389 | −0.081 ± 0.063 | 1.65 | 0.212 | not significant |
| scale-free | 0.384 | −0.030 ± 0.043 | 0.48 | 0.496 | not significant |

Both point estimates of c are **negative**, the direction saturation predicts, but neither is
distinguishable from zero. The data over N ∈ [500, 2000] are consistent with a straight line in
log-log, i.e. a pure power law. **This is a failure to detect curvature, not evidence of its absence**
— the range spans only a factor of 4 and N=2000 has a single seed.

### Test 2 — explicit saturating fits: mixed, no consistent winner

dAICc = AICc(saturating) − AICc(power law); positive favours the power law.

| L* | hard: hyperbolic | hard: exponential | scale-free: hyperbolic | scale-free: exponential |
|---|---|---|---|---|
| 0.02226 | +2.1 | +6.0 | −1.2 | +2.9 |
| 0.02253 | −2.4 | −1.2 | +0.3 | +2.6 |
| 0.02301 | +0.4 | +2.6 | +3.8 | +7.3 |
| 0.02369 | **−11.3** | **−13.6** | +4.0 | +9.0 |

Eight of sixteen comparisons favour each model; one (hard, shallowest level) is decisive *for*
saturation while the same level under the other criterion is substantial *against* it. No consistent
winner.

**The apparent M\* upper limits are an artefact and must not be quoted.** The profile-likelihood
95% upper bounds look reassuringly tight (e.g. 418, upper 480 at the deepest level) but that
tightness is manufactured by the short N range: with three sizes spanning a factor of 4, a saturating
curve can always be fitted with the ceiling placed just above the last data point. A ceiling
extrapolated from a factor-of-4 range is not credible.

### Test 3 — identification of k: clean

Bootstrap over seeds, N ≥ 500:

| L* | k, hard (95% CI) | k, scale-free (95% CI) |
|---|---|---|
| 0.02226 | **0.260** [0.184, 0.328] | **0.315** [0.264, 0.409] |
| 0.02253 | 0.327 [0.265, 0.460] | 0.387 [0.311, 0.511] |
| 0.02301 | 0.420 [0.370, 0.527] | 0.384 [0.297, 0.451] |
| 0.02369 | 0.549 [0.488, 0.669] | 0.450 [0.417, 0.516] |

k is well identified (CI width ≈ 0.15) and **depends systematically on the matching level**: deeper
training gives a smaller exponent, 0.55 → 0.26 under the hard criterion. So "the" exponent is not one
number — it is a function of how far the networks are trained, which is itself a result and matches
the earlier finding that saturation is something training produces rather than an architectural fact.
Pooled across levels, k ≈ 0.39 for both criteria.

### The sharp discriminator: what N=5000 will decide

Both models fitted on N = 500, 1000, 2000 only, then extrapolated. N=5000 is not yet on disk.

| L* | criterion | power law predicts | saturating predicts | gap |
|---|---|---|---|---|
| 0.02226 | hard | **471** | **393** | 78 |
| 0.02253 | hard | 642 | 526 | 116 |
| 0.02301 | hard | 1012 | 797 | 214 |
| 0.02369 | hard | 1513 | 1171 | 342 |
| 0.02226 | scale-free | 463 | 380 | 84 |
| 0.02369 | scale-free | 914 | 715 | 199 |

Seed-to-seed scatter at N=1000–2000 is 9–36 units, so a 78-unit gap at the deepest level is roughly
2–8 sigma — comfortably resolvable, and larger at every shallower level. **N=5000 extends the
ceiling-free range from a factor of 4 to a factor of 10 and will settle this.** First seed ~00:30
tonight; the remaining two ~22:30 on Aug 19. The two outstanding N=2000 seeds (~19:10 today) will
also tighten every k above and give that point real error bars.

---

## 2026-08-18 12:30 — ESTABLISHED: the loss floor does not depend on network size

Figure: `img/internal_figures/floor_vs_N.png`, from
`trainRNNbrain/experiments_and_analysis/test_floor_vs_N.py`. Criteria fixed before running: a size
dependence is accepted only if the slope differs from zero at p < 0.05 AND the best model beats the
constant by dAICc > 4 AND both hold at every fit length tested.

**Why this had to be settled first.** Matching sizes at equal training loss is only a valid notion of
"equally trained" if every size is heading for the same floor. If bigger networks could reach a lower
loss, two networks at equal loss would not be equally far from their own optima and every M(N) number
would inherit the confound. There is also a concrete mechanism that could produce a size-dependent
floor, so this is not a formality: lr = 1e-3·(100/N)^(1/3), and a noisy optimiser equilibrates in a
noise ball whose width grows with lr, so **larger N (smaller lr) should give a lower floor**.

**Result — the constant model wins at every fit length.** Four candidate shapes,
`L_inf = c` / `c + bN` / `c + b·log10 N` / `a·N^(-b)`, fitted on per-seed estimates, all sizes fitted
on the same first `t_max` iterations so the extrapolation bias is common:

| t_max | best model | dAICc: linear | log-linear | power law | size-term p |
|---|---|---|---|---|---|
| 50k | **constant** | +3.2 | +2.2 | +2.2 | 0.36–0.90 |
| 100k | **constant** | +3.2 | +2.1 | +2.1 | 0.368 |
| 150k | **constant** | +3.2 | +2.2 | +2.2 | 0.384 |
| 200k | **constant** | +2.4 | +3.2 | +3.2 | 0.429–0.976 |

Every size-dependent model is *worse* than the constant at every fit length, and no size term comes
close to significance. **Size dependence accepted at 0 of 4 fit lengths.**

**Stated as an equivalence bound, not as "no effect".** "Not significant" is not "equal", so the
right statement is how large an effect the data can still hide:

| t_max | slope per decade of N | bound on total change over N=100→2000 |
|---|---|---|
| 100k | −0.000073 ± 0.000149 | < 0.00019 = **0.92%** of L_inf |
| 150k | −0.000043 ± 0.000091 | < 0.00012 = **0.55%** |
| 200k | +0.000001 ± 0.000069 | < 0.00009 = **0.42%** |

So across a 20-fold range of N the floor is constant to better than half a percent, and the bound
tightens as more data enters the fit — the signature of a real null rather than of low power.

**Per-size values at t_max = 200k** (the tightest): 0.02141, 0.02135, 0.02140, 0.02145 for
N = 100, 500, 1000, 2000. The residual scatter is ~2–4× the within-size seed sd but is
**non-monotone** in N, which is why no trend model fits it — it is scatter in the extrapolation, not
a size effect.

**Model-free cross-check (panel c).** Raw measured loss at matched iteration count, no extrapolation
at all: it *increases* slightly with N at every checkpoint (e.g. at 200k: 0.02211, 0.02222, 0.02225,
0.02236). That is the expected consequence of the smaller learning rate at larger N, and it is the
opposite sign from "bigger networks reach a lower floor". Both the extrapolated and the measured
quantity therefore agree that nothing supports a decreasing floor.

**The lr/noise-ball mechanism finds no support.** Despite lr differing by 2.7× across the range, no
corresponding floor difference appears. Either the floor is set by task stochasticity rather than by
optimiser noise, or the effect is below 0.42%.

**Consequence.** Equal training loss = equal distance from a common optimum, so the matched-performance
protocol is licensed. This is now a stated fact of the project rather than an assumption.

**Caveats.** N=2000 rests on one seed (two more ~19:10 today), and the range is 1.3 decades; N=5000
extends it to 1.7 decades. The test re-runs unchanged on both, and the equivalence bound should
tighten further.

### Floor test, extended: in-progress runs recovered from logs, and the one marginal result explained

**Unfinished runs are usable for anything that needs only the loss.** `TrainLosses.json` is written
only on completion, but the per-iteration loss is printed to SLURM stdout as training goes, so an
in-progress curve is fully recoverable. Validated before use: parsing the log of a COMPLETED run and
comparing against its saved JSON gives a maximum difference of **5e-7**, i.e. exactly the 6-decimal
rounding of the printout. Runs that reached their declared max_iter are skipped so the JSON copy is
not double-counted. Implemented as `losses_from_logs()` in `test_floor_vs_N.py` (`--logs=DIR`).

This adds **2 more N=2000 seeds** (222k iterations each) and brings **N=5000 into the test** at 26–49k
iterations, extending the size range from 1.3 to 1.7 decades — today, rather than after Aug 19.

| t_max | N=100 | N=500 | N=1000 | N=2000 | N=5000 | slope p | best model |
|---|---|---|---|---|---|---|---|
| 25k | 0.02093 | 0.02018 | 0.01992 | 0.02003 | **0.01937** | **0.051** | power law |
| 50k | — | — | — | — | — | 0.570 | constant |
| 100k | — | — | — | — | — | 0.633 | constant |
| 150k | — | — | — | — | — | 0.623 | constant |
| 200k | 0.02141 | 0.02135 | 0.02140 | **0.02142** | — | 0.922 | constant |

N=2000 now rests on 3 seeds: **0.02142 ± 0.00016**, which meets the pre-registered prediction
(within ~0.0001 of 0.02135–0.02140) more comfortably than the single-seed 0.02145 did.

**The t_max = 25k anomaly is an artefact, and its mechanism is identified.** Matching the fit *length*
does not match the fit *content*: at a fixed iteration count the sizes are not equally far along,
because lr falls with N.

| loss actually reached at iteration 25,000 | N=100 | N=500 | N=1000 | N=2000 | N=5000 |
|---|---|---|---|---|---|
| | 0.02322 | 0.02339 | 0.02351 | 0.02375 | 0.02406 |

A bigger network is further from its floor at 25k, so its curve has flattened less, so its
extrapolated L_∞ is biased further **downward** — which mimics precisely a floor that decreases with
N. This corrects a claim made earlier in this document: fitting every size on the same t_max makes
the bias common only when all sizes are near their floor. At long t_max that holds; at 25k it does not.

**Control — fit each size on its own window ending at a COMMON LOSS**, so every fit sees the same
amount of curvature rather than the same number of steps:

| L* | N=100 | N=500 | N=1000 | N=2000 | N=5000 | slope p | best |
|---|---|---|---|---|---|---|---|
| 0.02450 | 0.01945 | 0.01767 | 0.01582 | 0.01882 | 0.01858 | 0.765 | constant |
| 0.02420 | 0.02035 | 0.01927 | 0.01820 | 0.01953 | 0.01842 | 0.223 | constant |
| 0.02400 | 0.02095 | 0.01941 | 0.01821 | 0.01959 | 0.01925 | 0.208 | constant |

Constant wins at every level and no slope approaches significance. The estimates are noisy (sd
0.0006–0.0034, since the windows are short), so this control has low power on its own — but it
removes the specific artefact and does not reproduce the trend.

**Verdict unchanged and now better supported: the loss floor is independent of network size.** Size
dependence accepted at 0 of 5 matched-length fits and 0 of 3 matched-progress fits; at t_max=200k any
change over N=100→2000 is bounded below **0.70%**.

### Correction: L_inf is model-dependent and the model is wrong — restate the floor result model-free

Fitting one N=1000 seed at four cut-offs gives L_inf = 0.01887 / 0.02070 / 0.02106 / 0.02135 and
gamma = 0.35 / 0.47 / 0.50 / 0.53 for t_max = 25k / 50k / 100k / 200k. Both parameters drift
**monotonically** with the fit window. A correctly specified model gives estimates that scatter about
the truth; ones that march in one direction mean the functional form is wrong. This is the same
misspecification found independently from the local exponent (gamma_eff rising 0.3 -> 0.8), reached
here from the opposite direction. Figure: `img/internal_figures/tmax_explainer.png`.

**Consequences.**

*Drop:* the absolute value of L_inf, and everything derived from it — including the repeated claim
that "93–96% of the loss is irreducible". That figure is model-dependent and should not appear
anywhere.

*Keep, restated without any fit:* the lowest loss each size actually reached —

| N | 100 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| best loss reached | **0.02211** | 0.02222 | 0.02225 | 0.02227 |

No larger network achieves a lower loss than the smallest. No extrapolation, no functional form.

*Why the fitted comparison also survives:* the bias direction is known — less progress gives a lower
fitted L_inf. At t_max = 200k, N=100 is ahead (loss 0.02211) while every larger size is behind
(0.02222–0.02234), so the larger networks' L_inf estimates are biased **downward relative to N=100**.
They nonetheless come out equal. Correcting the misspecification would raise the large-N floors,
i.e. move further from "bigger reaches lower". The misspecification cannot be manufacturing the null.

*Unresolved:* whether a much longer run would let a large network overtake N=100. Extrapolation
cannot answer it; only training longer can. The residual gaps are 0.0001–0.0002, far too small to
move any M(N) conclusion.

*Unaffected:* the matched-performance protocol, which never used L_inf — it reads raw loss only, and
the shared-floor fact it relies on is now established without the fit.

---

## 2026-08-18 13:00 — Audit of what "loss" actually means here, and a confound it exposes

Prompted by the question of whether the "lowest achievable loss" statistic was measuring anything
real. It was not, and checking the code turned up three facts plus one genuine confound.

### What the recorded loss is

1. **There is no validation set.** `Trainer.run_training` creates `val_losses`, never appends to it,
   and returns it empty; `run_experiment.py` saves that empty list. **Every loss number in this
   project is a training loss.**
2. **`same_batch=True`**: one fixed 450-condition batch is drawn once and reused for all 200k
   iterations. No batch resampling.
3. **The recorded loss comes from a NOISY forward pass** (`train_step` uses `w_noise=True`), so the
   only iteration-to-iteration variation is injected noise plus the weights moving.

**Consequence: the earlier "median of the 101 lowest losses" statistic is retracted.** Both it
(~0.0155) and the smoothed loss (~0.0221) are noisy training losses — the gap between them is the
lower tail of the noise distribution versus its mean, not noise-free versus noisy. Taking the minimum
of a noisy series is a noise lottery whose winner depends on how many draws the run had, so it is not
a property of the trained network at all.

### The well-defined replacement: noise-free loss of the final parameters

Deterministic, one number per seed (`eval_noisefree_loss.py`). Decomposed against the noisy loss in
the SAME implementation so the split is internally consistent:

| N | clean MSE | noisy MSE | noise share | n |
|---|---|---|---|---|
| 100 | 0.00851 | 0.03036 | 72.0% | 3 |
| 500 | 0.00851 | 0.03062 | 72.2% | 3 |
| 1000 | 0.00907 | 0.02952 | 69.3% | 3 |
| 2000 | 0.00991 | 0.03046 | 67.5% | 1 |

### The confound this exposes

**The noisy loss is flat across sizes (0.0295–0.0306, no trend) while the clean loss rises
monotonically, +16% from N=100 to N=2000.** Two effects cancel: bigger networks average injected
noise better across the readout (noise share 72.0% → 67.5%) but fit the deterministic task worse.

So **matching on training loss is not the same as matching on deterministic task performance.** At
equal noisy loss a larger network has systematically higher clean error. If that reflects being less
well fit, it biases the M(N) comparison in exactly the dangerous direction: a less-trained large
network is less silenced, M is inflated, and the result leans toward "no saturation" — the conclusion
we drew.

**This is not yet resolvable from existing data.** Clean loss can only be evaluated where weights
were saved, i.e. at the final parameters, so there is one point per run and no clean-loss trajectory
to match on.

**Cheap fix for future runs:** `track_participation_` already performs a noise-free forward pass every
`track_every` iterations. Computing the masked MSE from that same pass costs essentially nothing and
would give a noise-free loss trajectory, making the whole matching protocol re-runnable on the clean
quantity.

### Two further audit findings

- **The folder score is a single noise draw.** `run_experiment.py:119` calls
  `eval_step(..., noise=True)` once. Across 30 noise realisations that quantity has sd 0.0085 and
  range 0.818–0.854, so the saved r2 carries about +-0.02 of pure noise and should not be used as a
  performance measure.
- **Torch and numpy noisy evaluations disagree by ~4 sd.** One N=100 net: numpy gives
  r2 = 0.8395 +- 0.0085 over 30 draws, the torch-recorded score is 0.8737. The noise scaling formula
  is identical in both (`sqrt(2/alpha)*sigma`), so the cause is elsewhere and is not yet identified.
  The M(N) pipeline is unaffected (participation traces and loss curves are torch-native), but the
  numpy-based offline analyses — `population_distortion.py` in particular — carry an unquantified
  offset.
