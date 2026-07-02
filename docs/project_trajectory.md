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
