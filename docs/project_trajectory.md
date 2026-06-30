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
Computed for all 120 nets by [`count_silent_units.py`](../count_silent_units.py) →
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
