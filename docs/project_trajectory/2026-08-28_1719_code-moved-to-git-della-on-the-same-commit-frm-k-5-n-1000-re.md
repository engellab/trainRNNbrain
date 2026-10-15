## 2026-08-28 — code moved to git; Della on the same commit; frm k=5 N=1000 reseeded

### Everything now flows through git, no rsynced worktrees

Three commits pushed, and both clusters plus the local checkout are at **`6a35cff`**:

| commit | contents |
|---|---|
| `816f6bd` | the three data-integrity fixes, the analysis consolidation, the new scripts, the launchers (65 files) |
| `a22451a` | untrack four generated outputs under `data/` |
| `6a35cff` | launchers use `~/trainRNNbrain` and report the real git commit, warning if the checkout is dirty |

The `~/trainRNNbrain_ff` worktrees were populated by rsync, so `CODE_VERSION.txt` had to be written by
hand and could silently disagree with what was running. Both clusters already had `~/trainRNNbrain` as
a checkout of this remote, so the launchers now point there and print `git rev-parse HEAD`.

### ⚠️ A near-miss worth recording: a merge would have orphaned 127 GB

The Della pull ABORTED, and the reason mattered. **`~/trainRNNbrain/data` on Della is a SYMLINK to
`/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data`, holding 127 GB of results.** Four generated files
were tracked under `data/`, so git wanted to replace that symlink with a real directory — which would
have made every result on that cluster invisible. Git's refusal to overwrite untracked paths is the
only reason it was caught.

Those four files had been force-added against the repo's own `.gitignore`, which already declares
`/data/` and `data/trained_RNNs/`. Untracking them was both the correct fix and what unblocked the
pull. Verified afterwards: symlink intact, 127 GB still present. Spock's `data` is a real directory
(33 GB) and was never at risk, but the same commit would have hit any future symlinked checkout.

⚠️ **Spock keeps `~/trainRNNbrain_ff` until its 67 in-flight jobs drain.** SLURM copies the batch
script at submission, so those jobs still reference that path and deleting it would break them. It
carries both fixes, so those runs are consistent with the git version. They were NOT cancelled and
resubmitted: that would discard days of compute to change a folder name, and the code they run is
already the fixed code.

### frm k=5 N=1000 reseeded

Original seeds: tasks 40 and 41 both diverged (and collided, losing one); task 42 is healthy at
330k/400k with r2 = 0.939. So the cell had **1 usable seed of 3**.

Submitted 2 replacements on Della (job 13100032, `--array=40,41`). `seed="random"` resolves from
`time.time_ns()`, so these get fresh seeds automatically. Put on Della rather than Spock because
Spock has 67 jobs in flight and Della had 24.

**With the naming fix in place, a repeat divergence will now be visible rather than silently
overwriting its sibling** — which is the whole reason this cell was worth re-running rather than just
accepting n=1.

---

# ▶ SESSION HANDOFF — 2026-08-28 12:05  (read this first next session)

## What we are trying to establish, in priority order

1. **Is there a CEILING on active units, or is M ~ N^0.44 unbounded?** Over N = 500..2000 the
   exponent b is constant, which is a pure power law with no ceiling — but CDDM's fit put a ceiling
   near 880 active units. Extrapolating `M = 18.9 N^0.44` predicts M = 673 at N=4000, so a ceiling
   near 800 only starts bending the curve above N ≈ 4000. **This is the one open question the
   existing data cannot answer**, and the N=4000 run is the discriminating measurement.

2. **Does `both` (rws+frm) behave like rws, like frm, or in between?** Answered on two axes already
   (rws does NOT rescue frm: `both` recovers 16% of the frm→none gap on W_inp settling and 0% on
   silence). Still needs the PR law, which requires k≥3 at **two sizes** — hence the N=1000 cells.

3. **Does either penalty change the interference term b in `floor(k) = a + b√k`?** rws: no (CIs
   overlap at all three sizes). **frm: b +50%, a unchanged** — the pre-registered structure with the
   sign REVERSED. Needs N=1000/2000 frm to confirm beyond N=500.

4. **Complete the k=1 baseline.** `none` at k=1 was only ever submitted at N=500 (tasks 64-66 of the
   ksweep); k=1 anchors the floor law's intercept and the low end of every "k=1→8" statement.

## Jobs in flight

### SPOCK — 67 tasks (`~/trainRNNbrain`, commit 6a35cff; ⚠️ in-flight jobs still reference `~/trainRNNbrain_ff`)

| array | job | state | what | why |
|---|---|---|---|---|
| 5904416 | **FFbigN** | 12 R, 36 PD | N=4000, `none`+`rws`, k=1..8, 3 seeds, 100k iter | **the ceiling test (Q1)** |
| 5904341/2/3 | **FFpenL** | 36 R, 73 PD | frm+both @400k | Q2, Q3 |
| 5924467/8 | **FFk** | 2 PD | `none` k=1 at N=1000 (500k) and N=2000 (400k) | Q4 |

FFpenL queued breakdown: frm N=1000 ×10, frm N=2000 ×24, both N=500 ×15, both N=2000 ×24.
FFbigN progress: k=1 at ~58k/100k, k=2 ~57k, k=3 ~14–50k, k=4 ~19–26k. **Runs at 2.20 s/iter in
production (not the 1.24 the short calibration suggested) → 61 h/job**, inside the 96 h request.
Only `none` (tasks 1–24) matters for Q1; `rws` (25–48) is secondary.

### DELLA — 26 tasks (`~/trainRNNbrain`, commit 6a35cff)

| array | state | what | why |
|---|---|---|---|
| 13099674 | 4 R, 20 PD | `both` k=1..8 × N=1000 × 3 seeds, 400k | **unlocks the PR law for `both` (Q2)** |
| 13100032 | 2 PD | frm k=5 N=1000, replacement seeds | that cell had 2 of 3 seeds diverge |

## Expected timing

- **~29–30 Aug** frm N=1000 completes; `both` N=1000 completes on Della (~29 h/job, one gpu-medium wave)
- **~1 Sep** N=4000 `none` k=1..8 — **the ceiling answer**
- **~1–2 Sep** `both` becomes fittable (k≥3 at two sizes)
- **~2–3 Sep** everything else; N=4000 `rws` ~6 Sep

## Standing rules that keep being violated

- **Read-out criterion is `excess`** — 1.10× each run's OWN floor, floor fitted over that condition's
  OWN budget. Do NOT force a common range: that is what invalidated frm's floor and forced a retraction.
- **Report M and PR together.** They dissociate: M is k-independent, PR rises with k (c ≈ 0.05–0.12
  under all six criteria). M saturates under frm/both (M/N → 1.00) and cannot be used there at all.
- **frm has no diffusion read-out** — its W_inp never settles (α plateaus at 0.90). Use matched loss.
- **Della's `data` is a SYMLINK to /scratch (127 GB).** Never let git materialise `data/` as a real
  directory there.
- **Sync Della results to the local mirror before analysing** — every analysis script reads the local
  copy, not a cluster. Della writes to its own tree.

## Analysis that is ready to re-run the moment data lands

`pr_matrix.py` (PR/N over the grid, per penalty, ± δ sweep), `flipflop_penalties.py`,
`drift_matrix.py`, `excess_time_matrix.py`, `flipflop_decisive.py`, `criterion_search.py`.
The `both` panels are already wired and will populate automatically.

---

## ▶ frm DIVERGES SYSTEMATICALLY AT LARGE N AND k — 2026-08-28 14:54

Health check of the in-flight grid. **18 of 50 running Spock jobs were dead** — NaN loss, still
burning wall-clock. Every one of them was `frm`. 0 `both`, 0 `none`, 0 `rws`, 0 bigN. Della clean
(20/20 healthy, but only 10-44k iterations in, so not yet informative).

### The failure is one-step overflow, not slow drift

```
iteration 6335/400000, train: 0.083706, r2: 0.909225     <- healthy
iteration 6336/400000, train: inf,      r2: -1.75e+26    <- one step
iteration 6337/400000, train: nan,      r2: nan          <- dead for the next 355k iterations
```

**Mechanism.** `Trainer.py:803` clips with `torch.nn.utils.clip_grad_norm_(params, max_norm=50)`.
That call does NOT guard non-finite gradients (`error_if_nonfinite` defaults to False). With an
`inf` anywhere in the gradient the total norm is `inf`, so `clip_coef = 50/inf = 0`, and the clip
multiplies: `inf * 0 = nan`. Clipping therefore CONVERTS a single overflow into permanently NaN
weights. The run then trains on NaN forever without ever exiting non-zero.

### Divergence probability rises with BOTH N and k

Fraction of seeds lost (3 seeds/cell):

| frm     | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|
| N=500   |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
| N=1000  |  0  |  0  |  0  | 1/3 | 2/3 | 1/3 | 2/3 |  -  |
| N=2000  | 1/3 | 1/3 | 1/3 | 1/3 | 2/3 |**3/3**|**3/3**|**3/3**|

N=500 is untouched — which is exactly why this was invisible until the large-N cells landed.
`both` (which CONTAINS frm) has 0/15 losses so far; rws stabilising frm is a live hypothesis, not
yet a claim, since `both` at N=2000 k>=6 has not run.

### Impact on the analysis: none silently, but real gaps

`pr_matrix.py:70` and `drift_matrix.load()` both drop any run with NaN in its loss, so **no
contaminated cell can reach a figure**. The damage is missing data, not wrong data:

* **frm N=2000 k=6,7,8 = ZERO surviving seeds**; k=5 = 1 seed. Q3 (does frm change the
  interference coefficient b?) is fittable at N=2000 only over k=1..5 with 2,2,2,2,1 seeds.
* frm N=500 complete 24/24; frm N=1000 thin at k>=5.

### Action taken

Cancelled the 18 dead jobs (they were blocking ~18 GPUs from the 32 pending `both` tasks, which
are the ones Q2 needs). Remaining 33 Spock jobs verified healthy, 0 NaN. Surviving frm N=2000
k=1..4 runs are at 290-350k/400k and will finish normally.

Only **3** pending tasks are frm (N=1000 k=8); everything else queued is `both`, bigN or FFk.

### The fix, not yet deployed

Skip the update instead of letting the clip poison it:

```python
if not all(torch.isfinite(g).all() for g in g_tot):
    self.optimizer.zero_grad(set_to_none=True)
    return                      # drop this batch, keep the weights
torch.nn.utils.clip_grad_norm_(params, max_norm=self.max_grad_norm)
```

This is inert for any run that never overflows, so it does not break comparability with the
completed N=500 frm cells. NOT deployed — deploying means re-running frm N=2000 k=6,7,8 (9 jobs
x ~61 h) and that is a compute decision, not a code decision.

### Fix deployed and frm re-submitted — 2026-08-28 15:10

Guard committed as `ab98f89` and pushed. Deployed to: Della `~/trainRNNbrain` (git pull, symlink
intact), Spock `~/trainRNNbrain` (git pull), and Spock `~/trainRNNbrain_ff` (NOT a git repo — a
plain rsync copy — so `Trainer.py` was copied in; it differed from the fixed file by the patch
alone). Updating `_ff` matters because the 66 pending tasks in the OLD arrays launch from it, and
20 of them are `both` at N=2000 k>=3 — `both` contains frm and carries the same risk. Running
jobs already imported their module, so they are unaffected either way.

Validated before trusting: `tests/test_clip_nan.py` (mechanism + fix) and a 60-iteration local
frm run on the patched trainer (loss 0.42 -> 0.39, r2 0.47 -> 0.51, saved normally) — the guard
is inert when nothing overflows.

⚠️ The first version of the diagnosis was WRONG in one detail and the test caught it. I claimed
clip spreads NaN to every parameter in one step. It does not: clean grads are scaled to a
harmless 0.0, and only the parameter carrying the overflow goes NaN. The whole-network
contamination happens on the NEXT forward pass. Outcome identical, mechanism different.

**Seeds.** The launcher passes `seed="random"`, so re-runs draw fresh seeds automatically.

**Re-submitted (all frm; task ids verified against 4 known jobs before submitting):**

| where | array | tasks | cells | time / QOS |
|-------|-------|-------|-------|-----------|
| Della | 13115552 | 43,44,52,53,54,61,62,63,70,71,72 | N=2000: k=5 x2, k=6/7/8 x3 | 96 h, `gpu-long` (10 concurrent) |
| Spock | 5928695 | 31,49,58,59 | N=1000: k=4 x1, k=6 x1, k=7 x2 | 48 h |

N=2000 went to Della `gpu-long` deliberately: at ~0.55 s/iter x 400k it needs ~61 h (83 h on a
slow node), which does not fit `gpu-medium`'s 72 h cap, and `gpu-long`'s 10-concurrent limit
exactly fits 11 jobs while being a far less contended queue than Spock's 66-deep backlog.

**Expected inventory once everything lands** (in-flight survivors + these re-runs):
frm N=500 3/3 all k (already complete); frm N=1000 3/3 all k except k=8 (3 in flight, unfixed
code, at risk); frm N=2000 3/3 for k=5..8, but only 2/3 for k=1..4 — those four cells each lost
one seed to the bug and were NOT re-submitted, since 2 seeds still fit the floor law. Re-run them
if the k=1..4 error bars turn out to matter.

After cancelling the 18 dead jobs Spock went 33 -> 47 running as freed GPUs took pending `both`
tasks, which is the Q2 critical path.

---

## ▶ THE GUARD WORKS BUT DOES NOT RESCUE frm AT N=2000 — 2026-08-29 11:30

⚠️ **Correction to yesterday's note.** I reported "the fix is working" after inspecting ONE
re-run job. That was premature — a single sample. Across the full re-run array, **8 of 10 fixed-code
jobs failed anyway**, in a NEW failure mode.

### What the guard did and did not do

It did exactly what it was designed to do: weights are never NaN-poisoned any more. What it does
NOT do is keep the run useful. Observed on `13115552` (all at commit `ab98f89`):

| task | cell | finite loss, last 5k | iteration | verdict |
|------|------|---------------------|-----------|---------|
| _70 | k=8 N=2000 | 99.9% | 39,989 | training normally, loss 1.43 -> 0.107 |
| _62 | k=7 N=2000 | 100.0% | 40,871 | training normally |
| _43,_44,_52,_53,_54,_61,_63,_71 | k=5..8 N=2000 | **0.0%** | 82k-91k | **frozen** |

**The new failure mode is a frozen run, not a poisoned one.** The forward pass overflows to inf,
so the gradient is non-finite, so the guard skips the update — which means the weights never move,
so the next forward pass overflows identically. A deterministic trap. The frozen runs last saw a
finite loss 30k-55k iterations ago and will never recover.

**Tell for spotting it:** a frozen job runs about TWICE AS FAST, because skipping the optimizer
step is cheaper than taking it. The dead jobs were at iteration ~88k while the healthy ones were
at ~40k after identical wall-clock. Iteration count running ahead of its siblings is a red flag,
not a sign of progress.

⚠️ **"Contains nan" is no longer a valid health test.** Under the guard a perfectly healthy run
contains occasional inf/nan lines (_70 spikes ~1% of the time and recovers). Health must be
measured as the FRACTION of finite losses in a recent tail; <50% means frozen.

### It is not a learning-rate artefact

`run_experiment.py:92` sets `lr = 1e-3 * (100/N)^0.333`, so lr already FALLS with N
(5.9e-4 / 4.7e-4 / 3.7e-4 at N=500/1000/2000). Instability rises with N despite smaller steps.
Mechanism is intrinsic to frm at large N and high k, not step size.

### Surviving frm inventory (seeds saved on disk, Spock)

| frm | k=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|-----|-----|---|---|---|---|---|---|---|
| N=500 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| N=1000 | 3 | 3 | 3 | 2 | 1 | 2 | 1 | 0 |
| N=2000 | 2 | 2 | 1 | 1 | 1 | **0** | **0** | **0** |

In flight and expected to land: Spock `5928695` (fixed code) adds N=1000 k=4 +1, k=6 +1, k=7 +2
-> those three cells reach 3. Della `13115552_62/_70` add N=2000 k=7 +1 and k=8 +1 (~60 h out).
Still missing after that: N=1000 k=5 (+2) and k=8 (+3); N=2000 k=6 (all 3) and thin everywhere.

### ⚠️ SELECTION BIAS — the most important consequence

The surviving frm N=2000 runs are **conditioned on not having diverged**. If instability
correlates with anything the paper measures (recurrent gain, participation, floor), the survivors
are a biased sample and "frm at N=2000" statistics inherit that bias. Re-running until 3 seeds
survive does NOT fix this — it selects harder. Any frm N=2000 claim must state the yield
(currently ~20% at k>=5) alongside the number.

### Cancelled this session

8 frozen fixed-code jobs (Della `13115552`), 1 old-code dead (Della `13100032_41`), 2 old-code
dead (Spock `5904342_67/_68`, k=8 N=1000). Kept `13115552_62/_70` — both healthy.

### Options, not yet chosen

1. **Drop N=2000 from the frm arm** and fit the interference law on N=500 (complete) and N=1000
   (complete after `5928695` lands). Cheapest, no bias, loses the third size.
2. **Escape hatch**: checkpoint last-good weights, and on prolonged skipping restore them and back
   off lr. Real fix for the trap, but it is new training machinery mid-experiment.
3. **Warm up lambda_frm** from 0 so the network never gets pushed into the unstable region
   abruptly. Keeps the final penalty identical; changes the trajectory.
4. Over-submit and take survivors — REJECTED as primary strategy, it worsens the selection bias.

---

## ▶ frm SPIKE DEFENCES IMPLEMENTED (options 1 + 3a) — 2026-08-29

### Root cause, quantified

The penalty is not misdesigned — it faithfully reports a genuine dynamical blow-up, and T=300
timesteps is the amplifier. If the effective recurrent gain drifts to G, a perturbation grows
as G^300 within one trial:

| gain G | peak rate | over/cap | frm loss | fp32 |
|--------|-----------|----------|----------|------|
| 1.00 | 1 | 5.5 | 16.5 | ok |
| 1.04 | 1.3e5 | 7.1e5 | **3.5e16** | ok |
| 1.10 | 2.6e12 | 1.4e13 | 3.0e38 | **INF** |

Inverting the observed max loss (5e16, run `13115552_70`) gives **gain = 1.0404**; fp32 overflow
needs **gain = 1.0974**. The whole catastrophe lives between gain 1.00 and 1.11. Higher k needs
stronger recurrence (2^k attractors) and larger N shrinks `cap` (0.223 / 0.200 / 0.182 at
N=500/1000/2000), so divergence tracks both — as observed.

**The cubic gives no restoring force until it is far too late.** d(over/cap)^3/d(rate) at N=2000:

| peak rate | over/cap | p_over | gradient |
|-----------|----------|--------|----------|
| 1.0 | 0.006 | 2.5e-7 | **7.6e-6** (nothing) |
| 1.5 | 0.55 | 0.17 | 3.1 |
| 2.0 | 3.05 | 28.3 | **152** (cliff) |

Two structural notes: the logsumexp is NOT the problem — its softness scale is tau*log(M) = 1.264
with M = 300*1024 = 307,200, i.e. ~7 caps, so it behaves as a smooth average, not a hair-trigger
max. And the penalty is wildly asymmetric: `p_under <= 1` always (activity >= 0), while `p_over`
is unbounded — a silent unit costs 0.10, a unit at 10x cap costs 72.9.

### What was implemented

**(1) Drop anomalous updates; roll back when stuck.** `clip_grad_norm_` already returns the
PRE-clip norm, so the test is free. An update whose norm exceeds `spike_factor` x a running EMA is
skipped. ⚠️ **Clipping harder is NOT a substitute** — Adam divides by sqrt(v) and is scale
invariant, so one spike drags the weights 287 normal-sized steps at max_grad_norm=50 and still
233 at 0.1 (a 500x tighter clip buys 19%). Skipping costs exactly 0. Because a skip freezes the
weights, sustained skipping is itself the trap that spun 18 jobs to their wall-clock limit, so
after `restore_after` consecutive skips the weights roll back to the last snapshot and Adam's
state is cleared. The first accepted step snapshots, so a run diverging before `snapshot_every`
still has somewhere to return to.

**(3a) float64 powers.** `over`/`under` are shape (N,) — the reduction over the T*B axis already
happened — so this costs N doubles, NOT the (N, T*B) tensor. Same function, evaluated without
overflow; the wall moves from gain 1.10 to ~1.6. This also closes a hole in the previous guard,
which tested GRADIENT finiteness: between gain 1.10 and 1.60 the VALUE overflows while the
gradient stays finite, so that guard never fired there.

New config keys in `trainer_ptrack_freshbatch.yaml`: `spike_factor: 100.0`, `snapshot_every: 1000`,
`restore_after: 500`. `run_experiment.py` prints the skip/rollback count when non-zero.

### Old results are NOT compromised — verified, not assumed

* The objective is unchanged. fp64 vs fp32 differs by ~1e-7 relative, far under seed noise.
* At the default `spike_factor=100` the trigger fired **0 times** in 100- and 300-iteration local
  runs, matching the 0-1 spikes per 400,000 iterations measured in the completed N=500/1000 frm
  logs. The guard is inert in the regime the existing runs occupied.
* Forced firing (`spike_factor=1.2`, `restore_after=3`) gave 61 skips and 14 rollbacks with
  training completing normally — the machinery works end to end, through production code.

`tests/test_gradient_spikes.py` pins all three load-bearing claims and supersedes
`test_clip_nan.py`. Deployed at `5bb405d` to Spock `~/trainRNNbrain`, Della `~/trainRNNbrain`
(symlink intact) and the Spock `~/trainRNNbrain_ff` rsync copy that the old arrays launch from.

### frm N=2000 resubmitted under the defences — 2026-08-29 12:40

Cancelled Della `13115552_62` / `_70` (frm N=2000 k=7/k=8, ~40k/400k, still on the pre-fix commit
`ab98f89`). They were healthy at the time but under the old code the N=2000 k>=5 failure rate was
8/10, and at 10% through they had ~50 h each still to burn before probably dying.

Resubmitted as Della array **`13152868`**, tasks 43,44,52,53,54,61,62,63,70,71,72 = 11 jobs,
`gpu-long`, 96 h, running commit `5bb405d`:

| cell | new seeds | already saved |
|------|-----------|---------------|
| frm N=2000 k=5 | 2 | 1 |
| frm N=2000 k=6 | 3 | 0 |
| frm N=2000 k=7 | 3 | 0 |
| frm N=2000 k=8 | 3 | 0 |

⚠️ **k=1..4 were NOT topped up and remain thin**: k=1 = 2 seeds, k=2 = 2, k=3 = **1**, k=4 = 1 + one
run finishing (Spock `5904343_34` at 392k/400k) -> 2. **k=3 at a single seed has no error bar** and
is the weakest cell in the frm N=2000 row; topping k=1..4 back to 3 would be 5 more jobs.

Watch for on the first completions: the `gradient spikes: N updates skipped (...), M rollbacks`
line that `run_experiment.py` now prints. Zero on N=500/1000 is expected and confirms the guard is
inert there; a non-zero count at N=2000 with the run still converging is the fix doing its job.
A large count WITH rollbacks means the network is repeatedly re-entering the unstable region,
which would argue for escalating to option 2 (tail-only Huberisation of p_over).

⚠️ The selection-bias caveat still stands and is NOT solved by this fix: cells that needed skips
or rollbacks to survive are not drawn from the same distribution as cells that never spiked. The
printed skip count is what makes that auditable — report it alongside any frm N=2000 number.

### Every remaining gap submitted — 2026-08-29 13:05

Coverage audit across both clusters (206 networks saved). `rws` is COMPLETE (72/72, all three
sizes). `none` is complete at N=500 and needs only k=1 (FFk arrays, queued) plus one k=8 N=2000
seed. `both` was already fully queued (71 jobs). `frm` was the only arm with unqueued holes.

Submitted to close every remaining deficit:

| cluster | job | tasks | cells | time |
|---------|-----|-------|-------|------|
| Della | `13156475` | 9,18,26,27,36 | frm N=2000 k=1,2,3 x2,4 | 96 h `gpu-long` |
| Spock | `5942790` | 42,68,69 | frm N=1000 k=5, k=8 x2 | 48 h |
| Spock | `5942797` | 63 | none N=2000 k=8 | 96 h |

⚠️ **The ksweep launcher defaults to IT=500000 but every existing `none` N=2000 cell is at
iters=400000.** Submitting it plain would have written a SEPARATE `iters=500000` directory - a new
orphan cell, not a third seed - and the budget mismatch inside a cell is the error that has
manufactured a spurious trend three times in this project. Resubmitted with
`--export=ALL,ITERS_OVERRIDE=400000`; the first attempt (`5942791`) relied on ambient environment
propagation and was cancelled rather than trusted.

Also cancelled `13115552_72` (frm N=2000 k=8, pre-fix commit `ab98f89`, only 12.5k iterations in).
It was PENDING during the earlier health sweep so it escaped that cancel, then started and became
a duplicate of `13152868_72`.

⚠️ `bigN` covers **both `none` AND `rws`** at N=4000 (PEN_NAME=(none rws), 48 tasks, k=1..8);
41 still queued. No extra N=4000 jobs are needed - k=5..8 are in the array, just not yet run.

---

## ▶ `both` IS HEALTHY — IT WAS BEING KILLED BY THE WALL CLOCK — 2026-08-30 12:30

`both` at N=1000 shows **no numerical trouble whatsoever**. Every saved network and every running
job is converged and stable:

* 9 saved, r2 = 0.9378 … 0.9538, declining monotonically with k (0.952 at k=1 -> 0.938 at k=6)
* 12 running, median r2 0.931-0.946, median loss 0.040-0.051, last-probe r2 == median r2
* zero non-finite probes, zero negative r2, no spikes

**The losses in this condition were entirely operational.** The Della `both` N=1000 array was
submitted with `--time=48:00:00`, but 400k iterations at N=1000 needs **~57 h** (measured: 336k in
48 h = 0.514 s/iter — nearly 2x the 0.26 s/iter the launcher's sizing assumed for this size).
Healthy runs were being SIGKILLed at ~84% completion, and because results are written only at the
end, each timeout loses the entire run.

`sacct -j 13099674` : 9 COMPLETED, 4 TIMEOUT (tasks 77, 85, 87, 94), 11 RUNNING — of which 6 were
already past or within an hour of the wall.

⚠️ **A user cannot extend a running job's wall clock.** `scontrol update JobId=... TimeLimit=...`
returns "Access/permission denied" for both RUNNING and PENDING jobs. Once a job is submitted with
too short a `--time`, the only remedy is resubmission. Beware: `scontrol update ... && echo ok`
prints the success message anyway, because scontrol exits 0 while printing the denial to stderr —
always verify with `squeue -o "%l"` rather than trusting the exit status.

Resubmitted as Della **`13200159`**, tasks 77,85,87,94,96,104,114,122,123,140, `--time=72:00:00`
(`gpu-medium`, 24 concurrent, 57 h need + margin). Cancelled the 6 doomed runs rather than let
them burn another hour to no purpose.

### The same trap, caught before it fired, at N=2000

Measured projections for `both` N=2000 against Spock's limit:

| cell | projected total |
|------|-----------------|
| k=5 | 83.5 - 86.0 h |
| k=6 | 85.3 - 86.7 h |
| k=7, k=8 (pending) | ~90-95 h extrapolated |

⚠️ **Spock's hard wall-clock cap is 96 h** — `--time=120:00:00` and `144:00:00` are both rejected
("Requested time limit is invalid"), and `--qos=long` / `--qos=very-long` do NOT lift it. So k=7/k=8
had roughly a 1-5 h margin, and the trajectory's own 1.35x slow-node factor would blow straight
through it, losing ~96 GPU-hours per job.

Those 6 tasks (133,134,135,142,143,144) were still PENDING, so moving them cost nothing: cancelled
on Spock, resubmitted as Della **`13200242`** at `--time=144:00:00` (`gpu-long`, 6-day limit).

**Rule going forward: size `--time` from a MEASURED s/iter for that exact (penalty, N, k), not from
the launcher's header comment.** `both` is ~2x slower than the sizing table claims, because it
evaluates two penalties per step.

---

## ▶ MIG SLICES WERE THE REAL KILLER — 2026-08-31 10:55

251 networks saved. `none` N=1000 and `rws` are now COMPLETE (24/24 each at all their sizes);
`none` N=4000 is at 19/24.

### The frm "breakage" was largely starvation, not divergence

Split the frm N=2000 array by the GPU it landed on and the picture is unambiguous:

| GPU | tasks | median r2 | iterations reached | outcome |
|-----|-------|-----------|--------------------|---------|
| full `a100` | _43,_44,_52,_61,_70,_71 | **0.933 - 0.941** | 215k - 239k | all finish in 31-38 h |
| `3g.40gb` MIG | _53,_54,_62,_63 | 0.276 … -516 | 106k - 109k | ALL time out |

⚠️ **`--gpus-per-node=1` lets SLURM satisfy the request with a MIG SLICE** - a partitioned A100 that
the node advertises as gres `3g.40gb`, not `a100`. Sliced tasks ran 1.5-2.2x slower, so they both
overran their wall clock AND had far fewer iterations at any given moment. Fixed in the Della
launcher by naming the gres TYPE, which slices cannot match:

```
#SBATCH --gres=gpu:a100:1        # was: --gpus-per-node=1
```

**Recovery takes ~100k iterations.** `13152868_44` read r2 = **-1.43 at 120k** and r2 = **+0.941 at
238k** - it escaped the unstable region entirely, given time. `_54` likewise went 0.38 -> 0.933.
So a low r2 at ~100k does NOT mean a broken run; it may simply be pre-recovery. This retracts the
"5/10 broken" read from 2026-08-30, which sampled the MIG tasks mid-recovery and mistook slow
hardware for numerical failure. The spike defences are working: every task given a full GPU and
enough iterations converged.

### Wall-clock losses continue

A 5th timeout: `13100032_40` (frm k=5 N=1000) died at 303k/400k, healthy, on a 48 h limit.
The 48 h default is simply too short for N=1000 at 400k (~57 h needed).

### Actions

* Cancelled `13152868_53,_54,_62,_63` (needed 122-127 h, had 50 h - certain loss).
* Cancelled the PENDING `13156475` and `13200242`, both submitted with the MIG-permitting script.
* Resubmitted on full a100s, all verified `TresPerNode=gres/gpu:a100:1`:
  `13251599`+`13251624` (frm N=2000 k=6,7, 96 h), `13251625` (frm N=2000 k=1-4, 96 h),
  `13251626` (both N=2000 k=7,8, 144 h).
* LEFT `13200159` (both N=1000) running: 9 of its 10 tasks are on MIG slices, but they need
  42-45 h with ~50 h left, so they are projected to finish. Tight, not doomed - `_140` has the
  thinnest margin at 4.8 h. Cancelling would have cost 22 h of progress each for a worse position.

⚠️ Check `squeue -o "%N"` + `scontrol show node` for the gres type when a job looks anomalously
slow, BEFORE concluding the science is broken.

---

## ▶ SPOCK QUOTA AUDIT — 2026-09-01

Soft-limit warning on `/ifs/people` (95 G soft, 100 G hard). `df` showed the filesystem at
**100%, 0 bytes available** — running jobs would have failed on save, and results are written only
at the end, so each failure loses the whole run.

⚠️ `/usr/people/pt1290` and `/mnt/cup/people/pt1290` are the SAME filesystem
(`cup-private:/ifs/people`), so every repo copy counts once against the same quota.

### Deleted (all regenerable or already excluded from analysis) — 8 G recovered

| item | size | why safe |
|------|------|----------|
| `~/.cache/pip` | 3.5 G | download cache |
| `conda clean -a` | ~3 G | package cache; envs are hardlinked and unaffected |
| `std_bigN/*iters=400,600`, `*N=3000*`, `*N=5000*` | 451 M | timing-calibration + off-grid probes, already filtered out of every count |
| `~/trainRNNbrain_pen` | 280 M | code-only copy, NO data dir, referenced by 0 running jobs |
| logs >7 d in `~/trainRNNbrain_logs` | 2.7→2.3 G | gzipped, not deleted — forensics preserved |
| `~/trainRNNbrain/log` (1229 old slurm logs) | 460→366 M | gzipped; superseded by `~/trainRNNbrain_logs` |

Result: **97 G → 89 G, 6.3 G free.**

⚠️ The earlier coverage tables were INFLATED by those calibration folders — `none` N=4000 k=1 read
4 seeds when the truth is 3. Any inventory script must exclude `iters=` values below 50000 and
N outside {500,1000,2000,4000}. Note `std_pen` folders carry NO `iters=` field at all, so a naive
"iters < 50000" filter silently drops the entire rws arm — filter only when the field is present.

### NOT deleted — needs a decision

| item | size | note |
|------|------|------|
| `.conda/envs/RNN_training_pipeline` | **9.7 G** | last touched **2023-01-27**; all 105 `conda activate` lines in the slurm scripts use `torch-env`; unreferenced in the repo. Biggest remaining win. |
| `CDDM_*` trained nets | 23 G | the CDDM comparison arm of the paper |
| `~/OvercomingBiases` | 22 G | a different project |
| `RETRACTED_samebatch_NBitFlipFlop_ksweep` | 1.8 G | deliberately QUARANTINED as a record of the retracted same_batch sweep, not dead weight |

⚠️ `~/trainRNNbrain_ff` (1.2 G) must NOT be deleted yet — despite being a legacy rsync copy, the
still-running arrays `5904343`, `5904416`, `5924468` use it as BOTH their sbatch script path and
their live code (`code: /mnt/cup/people/pt1290/trainRNNbrain_ff`). Remove only once they drain.

### Quota resolved: 97 G -> 77 G (19 G free) — 2026-09-01

Deleted with approval, after recording both to `~/deleted_records/`:
`RNN_training_pipeline.packages.txt` (200 exact package specs, so the env is reconstructible) and
`RETRACTED_samebatch.manifest.txt` (60 entries naming every quarantined cell and its r2).

* `.conda/envs/RNN_training_pipeline` — unused since 2023-01-27, 0 references anywhere including
  a full grep of the 22 G `~/OvercomingBiases` tree; all 105 `conda activate` lines use `torch-env`.
* `RETRACTED_samebatch_NBitFlipFlop_ksweep` — the quarantined same_batch=True sweep.

⚠️ **`du` on a conda env is an UPPER BOUND, not a reclaim estimate.** Deleting the 9.7 G env freed
only ~2 G, because its files are HARDLINKS into `.conda/pkgs` — removing the env just decremented
the link count while the package cache kept the inodes alive. The space only came back on a SECOND
`conda clean -a`, which could not have run earlier: while the env existed those packages counted as
"in use" and were skipped. Order matters — delete the env FIRST, then clean. That second pass took
`.conda` from 17 G to 7.0 G.

⚠️ **A `ps | grep <name>` safety check matches ITSELF** (and the ssh command line carrying it). The
first scan reported "5 processes using RNN_training_pipeline" and would have blocked a correct
deletion; scanning `/proc/*/exe` found zero. Verify process usage by executable path, not by
command-line text.

`torch-env` verified intact after the clean: torch 2.7.0+cu126, numpy 2.4.6, `trainRNNbrain`
imports and the non-finite-gradient guard is present. No running or queued job is affected.

### Grid nearly closed; straggler moved off the gpu-long queue — 2026-09-02

309 networks saved, 9 of 14 rows COMPLETE (none N=500/1000/4000, rws N=500/1000/2000, frm N=500,
both N=500/1000). 45 completions in 24 h with ZERO failures — the MIG fix and the wall-clock
resizing are both holding.

**The N=4000 ceiling test is complete** (none N=4000 = 24/24), so Q1 from the original handoff
— does M keep following N^0.44 or bend toward a CDDM-like saturation near ~880 — can now be
answered. Nothing blocks that analysis.

Every remaining cell has enough jobs in flight; nothing new needed. Expected: rws N=4000, both
N=2000 k=5/6 and none N=2000 k=8 today; frm N=1000 k=5 on Sep 3; frm N=2000 k=6/7 Sep 3 pm;
frm N=2000 k=1-4 Sep 4; both N=2000 k=7/8 Sep 5.

⚠️ The last job to finish was NOT on the critical path anyone was watching. `13252006_72`
(frm N=2000 k=8, third seed) sat `PENDING (QOSMaxJobsPerUserLimit)` behind Della's 10-slot
gpu-long cap: 31 h until a slot freed, then ~77 h = ~108 h, i.e. Sep 6-7, one to two days behind
everything else. Moved to Spock, which had **0 pending** and measures 59-64 h for frm N=2000
(`5979262_9` projects 59 h, `_36` 64 h) inside a 96 h cap — about 38 h faster. Cancelled on Della
before it ever started, resubmitted as Spock `6009087_72`.

**Lesson: a QOS concurrency cap can make a single queued job the longest pole while every running
job looks healthy.** Check pending reasons, not just running-job ETAs, when forecasting completion.

Disk after the quota cleanup: 83 G used, 13 G free — climbing as results land. ~40 completions
still to come; watch it.

---

## ▶ ⚠️ COVERAGE COUNTS WERE INFLATED BY FAILED RUNS — 2026-09-02

Every coverage table reported in this project counted a saved run as a seed if its directory name
did not begin with `nan`. **That misses the failures that DID save.** A run that finishes training
without diverging numerically, but never learns the task, is written out with its (negative) r2 as
the directory prefix and was counted as a valid seed.

**15 of 314 saved flip-flop runs are failures.** The split is unambiguous — there is a clean gap in
the score distribution with nothing between them:

```
failures:  -12.41 -9.98 -9.62 -5.00 -3.90 -1.54 -1.51 -1.11 -1.10 -0.73 -0.32   + 4 nan
successes:                                              0.857 0.891 0.910 ...
```

Any threshold in (-0.32, 0.857) gives the same answer; `r2 >= 0.5` is used.

### Corrected coverage — three rows previously called COMPLETE are not

| cell | reported | actually usable |
|------|----------|-----------------|
| none N=1000 k=1 | 3 | **2** |
| none N=2000 k=1 | 3 | **1** |
| none N=4000 k=8 | 3 | **2** |
| rws N=4000 k=2 | 3 | **2** |
| frm N=1000 k=5 | 2 | **1** |
| frm N=1000 k=7 | 3 | **1** |
| frm N=1000 k=8 | 3 | **2** |
| both N=2000 k=4 | 3 | **2** |

⚠️ **The N=4000 ceiling test is NOT complete** — k=8 has 2 usable seeds, not 3. The claim logged
yesterday that Q1 "can now be answered" is RETRACTED until that third seed lands.

Totals: 299 usable (not 309), 90/112 full cells (not 97).

### `nan` audit

Zero `nan`-prefixed directories in the flip-flop data on Spock. The only ones anywhere are 12 in
old CDDM experiments on Della (1.28 G), unrelated to this paper. But 4 runs carry a literal `nan`
r2 INSIDE the flip-flop set — those are among the 15 above, so `nan` runs are caught by the r2
filter regardless of how the directory happens to be named.

### Every analysis script must apply the r2 filter

`pr_matrix.py` and `drift_matrix.py` drop runs whose LOSS trace contains NaN, which catches the 4
nan runs but **not** the 11 negative-r2 ones — those have perfectly finite losses, just large. Any
script that selects runs by directory listing needs `float(basename.split("_")[0]) >= 0.5`.

### Submitted to close the real gaps (11 jobs, all Spock)

`6009131` none N=1000 k=1 (iters 500000, launcher default) · `6009132` none N=2000 k=1 x2
(`ITERS_OVERRIDE=400000`) · `6009133` none N=4000 k=8 + rws N=4000 k=2 · `6009134` frm N=1000
k=5,6,7x2,8 · `6009135` both N=2000 k=4. Task ids were chosen to avoid the 16 already in flight.

---

## ▶ SESSION HANDOFF — 2026-09-02

### Read this first

⚠️ **Count usable seeds with an r2 filter, not by excluding `nan` directory names.**
`float(basename.split("_")[0]) >= 0.5`. Every coverage figure reported before today was inflated:
11 runs with NEGATIVE r2 were being counted as valid seeds. The score distribution is cleanly
bimodal — failures top out at -0.317, the lowest success is 0.857, nothing in between — so the
threshold is not a judgement call.

⚠️ **Those 11 failures were caused by the spike guard, not by seeds or task difficulty.** They
skipped 90-99.8% of all updates (798 rollbacks x 500 consecutive skips out of 400,000). The
earlier "failure rate by condition" table, including frm N=1000 at 28%, measured that bug. It is
retracted, and so is the selection-bias framing built on it.

### Guard fixed at `79a1063`, deployed to Spock, Della and the Spock `_ff` rsync copy

* `spike_factor` **100 -> 1e6**. PROVEN defect: at 100 it fired on 6.0% / 12.2% / 17.9% of ordinary
  gradients at lognormal sigma 3 / 4 / 5, which is what an frm run near instability produces. A real
  catastrophe is ~1e30x normal, so 1e6 still catches 50/50 injected catastrophes at a mild 1e12
  ratio with a 0.00-0.30% false-positive rate.
* Reference is now a **rolling median over EVERY step**, accepted or skipped. The old EMA updated
  only on acceptance and could never refresh while the guard was firing.
* `_last_good` snapshots are **gated on the loss being at or below its running median**, so a
  rollback cannot restore an already-broken state.

⚠️ NOT fully characterised: a synthetic i.i.d. model does NOT reproduce the exact 99.8% figure
(the old EMA rule skips <1% there). The false-positive rate is proven; the precise trigger for
99.8% is not. Do not write it up as a solved root cause.

### Running now (36 tasks, every cell provisioned to 3 usable seeds, none short, none over)

| where | arrays | covering |
|-------|--------|----------|
| Spock | `6009324`-`6009330` (17) | frm N=2000 k=1,3,4; frm N=1000 k=5x2,6,7x2,8; both N=2000 k=4,6; frm N=2000 k=8; none N=1000 k=1; none N=2000 k=1 x2 (`ITERS_OVERRIDE=400000`); bigN none k=8 + rws k=2 |
| Della | `13344547` (2) | frm N=2000 k=6, k=7 |
| both | pre-existing | `5904343`, `5904416`, `5942797`, `5979262_18/_26`, `13251599`, `13251626` |

`5979262_26` (frm k=3 N=2000) was deliberately LEFT running on the old guard — median r2 0.9462 at
104k and its guard never fired, so restarting would have cost 16 h for nothing.

### State

* 299 usable networks; complete rows: none N=500, rws N=500/1000/2000, frm N=500, both N=500/1000.
* Local mirror sync from both clusters was still running at sign-off (ksweep done, 741 MB; `pen`
  already current; penlong in progress). Re-run the per-directory rsync to finish it —
  ⚠️ one rsync call per directory; multiple remote sources in a single call silently prints usage
  and exits 0.
* Spock disk 83 G / 95 G after the 20 G cleanup. `~/trainRNNbrain_ff` (1.2 G) still cannot be
  deleted — `5904343`, `5904416` run their live code from it.
* ⚠️ The N=4000 ceiling test is NOT complete: `none` N=4000 k=8 has 2 usable seeds, not 3.
  Yesterday's "Q1 can now be answered" is retracted until the third lands.

---

## ▶ THE MOTIVATING QUESTION IS ANSWERED — 2026-09-08

Full grid complete (336/336 usable seeds, 14/14 rows, N = 500/1000/2000 for all four penalties,
plus N=4000 for `none` and `rws`). Everything below reads each network at the `excess` criterion -
the iteration where its loss first reaches 1.10x its OWN fitted floor - which is budget-independent
and therefore valid across the 100k / 150k / 400k / 500k budget spread.

### 1. Active units grow far too slowly to be a route to a large active population

`M = A N^b k^c` fitted per penalty on the absolute active count (Otsu threshold 4e-2):

| pen | A | b | c | M at N=4000, k=3 | N needed for M=1000 | for M=2000 |
|-----|------|-------|--------|------|---------|---------|
| none | 33.0 | **0.373** | -0.053 | 688 | **1.1e4** | **7.0e4** |
| rws | 9.0 | 0.566 | +0.043 | 1032 | 3.8e3 | 1.3e4 |
| frm | 1.37 | 0.952 | -0.012 | 3646 | 1.0e3 | 2.1e3 |
| both | 1.02 | 0.998 | -0.001 | 3981 | 1.0e3 | 2.0e3 |

**Unpenalised, reaching 1000 active units needs N ~ 11,000, and 2000 active units needs N ~ 70,000.**
That is the quantitative form of "prohibitively large": the largest network trained here is 4,000
and yields 688.

### 2. Task complexity matters, but weakly

`c` is small everywhere: -0.053 (none), +0.043 (rws), -0.012 (frm), -0.001 (both). Over the full
k = 1..8 range that is a factor of 8^0.05 ~ 1.11 - about 11%, against the factor of ~8 that N
contributes over the same span of sizes. Complexity is a real but second-order effect, and its
SIGN is not even consistent across penalties.

### 3. ⚠️ b ~ 1.0 for frm and both is NOT a scaling law - it is a 100% ceiling

Active FRACTION M/N:

| pen | N=500 | N=1000 | N=2000 | N=4000 |
|-----|-------|--------|--------|--------|
| none | 0.610 | 0.414 | 0.269 | **0.170** |
| rws | 0.639 | 0.469 | 0.356 | 0.262 |
| frm | 0.998 | 0.996 | 0.964 | - |
| both | **1.000** | **1.000** | **0.998** | - |

`both` has essentially EVERY unit active, so M = N identically and b = 0.998 [0.997, 0.999] is
arithmetic, not a discovered exponent. Do not report it as a scaling result. The penalties work by
ABOLISHING SILENCE, not by improving how recruitment scales.

Note the unpenalised trend: the active fraction FALLS with N (0.61 -> 0.17). Bigger unpenalised
networks are progressively emptier.

### 4. Power law vs ceiling - N=4000 does separate them, for rws

`flipflop_ceiling.py`. Two tests: a model-free local slope d log M/d log N (constant = power law,
falling = saturation), and AICc between `A N^b k^c` and `M_max (N/(N+N0)) k^c` (both 3 parameters).

| pen | 500->1000 | 1000->2000 | 2000->4000 | dAICc | verdict |
|-----|-----------|------------|------------|-------|---------|
| none | 0.441 | 0.379 | **0.341** | +9.9 | power favoured, but slope falls monotonically |
| rws | 0.552 | 0.606 | 0.554 | **+58.0** | power, unambiguous - flat slope |
| frm | 0.996 | 0.953 | - | -5.0 | ⚠️ only 3 sizes, cannot separate |
| both | 0.999 | 0.997 | - | -2.3 | ⚠️ only 3 sizes, cannot separate |

⚠️ **`none` is genuinely undecided and the two tests disagree.** AICc says power law; the local
slope declines at every step, which is the saturation signature. It is also the only condition
whose verdict is CRITERION-DEPENDENT: under PR the slope falls much harder (0.439/0.279/0.226) and
AICc FLIPS to saturation (-2.7). Under the scale-free count it stays power (+14.9). rws is power
under all three. The discriminating experiment is N=8000: slope ~0.34 under a power law versus
~0.25 under the ceiling implied by the fitted M_max ~ 830.

⚠️ The M_max values fitted for frm and both (2.1e4 and 4.5e5) lie far outside the data and are
meaningless; their "saturation favoured" labels are 3-point artefacts and the script flags them.

### Conclusion this supports

Unpenalised networks cannot deliver a large active population at any feasible size, and task
complexity will not rescue that. rws and frm each address it directly - frm/both by driving the
active fraction to ~1, rws by lifting the exponent from 0.37 to 0.57 while keeping a real
distribution. That is the motivation for the penalties, now with numbers behind it.

### Bugs found while producing this

* ⚠️ `pr_matrix.ROOTS` was MISSING `bigN`, the only source of N=4000. Every PR figure ever produced
  by it - including the six delta-sweep figures generated earlier today - fitted N over three sizes
  only, omitting the size that most constrains b. `drift_matrix` had bigN; `pr_matrix` never did.
  Fixed. Regenerate any PR figure produced before this.
* ⚠️ Neither `pr_matrix` nor `drift_matrix` filtered on r2. The 11 negative-r2 runs have FINITE
  losses, so the existing isnan() checks missed them entirely. Both now gate on r2 >= 0.5.
* `SILENT_FLIPFLOP = 4e-2` is now a named constant in common.py; it was previously only prose in a
  docstring, so callers kept reaching for the CDDM-calibrated `SILENT_HARD = 1e-6`, which sits
  below BOTH flip-flop modes and reports ~0% silence.

## ▶ PENALTIES RECRUIT UNITS WITHOUT RECRUITING DIMENSIONS — 2026-09-09 12:21

`flipflop_dimensionality.py`, all 336 usable runs, end-of-training nets. Effective dimensionality of
the population activity, two measures off the same spectrum:

    D_PR = (sum lambda)^2 / sum lambda^2      lambda = eigenvalues of cov(rates), centred per neuron
    D_95 = PCs carrying 95% of the variance

Rates are the noise-free forward pass on a fresh batch, relu applied (equation_type "h" stores
pre-activations), pooled over time and trials. Silent units contribute zero variance, so neither
measure can be inflated by them. Cached in `data/dimensionality_cache.npz`; delete to recompute.

### The headline

D = A N^b k^c fitted per penalty:

| pen | b (size) | c (complexity) | n |
|-----|----------|----------------|---|
| none | **-0.02** [-0.03, -0.01] | +0.89 [+0.87, +0.91] | 96 |
| rws | **+0.01** [-0.00, +0.02] | +0.91 [+0.89, +0.93] | 96 |
| frm | **+0.23** [+0.19, +0.27] | +0.77 [+0.73, +0.81] | 72 |
| both | **+0.09** [+0.07, +0.11] | +0.86 [+0.84, +0.88] | 72 |

**Dimensionality is set by the task, not by the network.** For none and rws, b is indistinguishable
from zero: D_PR is ~2.1 at k=1 and ~14 at k=8 whether the network has 500 units or 4000. An 8-fold
increase in N buys no extra dimensions. c ~ 0.9 everywhere, so D grows very nearly linearly in k,
about 1.8k.

**This runs opposite to the participation story and that is the point.** `pr_matrix` says frm and
both drive PR/N towards 1 - every unit engaged, activity spread evenly. This says frm and both put
the population in the SAME ~2k dimensions as everyone else. `both` at N=2000, k=8 has M/N = 1.00,
i.e. 2000 active units, occupying 12.7 effective dimensions. The penalties recruit units without
recruiting dimensions: the extra units are redundant within a task-determined subspace, not new
computation. M and D_PR are answering different questions and must never be quoted as if
interchangeable.

### Applying BOTH penalties lowers dimensionality; neither alone does

Per-cell D_PR ratio against `none` at matched (N, k), paired Wilcoxon over shared cells:

| pen | N=500 | N=1000 | N=2000 | median ratio, all cells | p |
|-----|-------|--------|--------|-------------------------|---|
| rws | 0.957 | 0.962 | 0.949 | 0.990 | 0.054 |
| frm | 0.886 | 1.087 | **1.245** | 1.018 | 0.26 |
| both | **0.812** | **0.863** | 0.937 | **0.868** | **8.3e-07** |

`both` is 13% BELOW unpenalised, consistently and significantly. Neither penalty alone does this:
rws is null (p = 0.054, ratio 0.99), and frm is not a reduction at all - it is 0.89 at N=500 but
1.25 at N=2000, i.e. its effect REVERSES with size, which is what b = +0.23 encodes. So the
dimensionality reduction is a property of the COMBINATION, not of either penalty summed.

⚠️ The `both` effect SHRINKS with N (0.81 -> 0.86 -> 0.94) and the N=2000 range already crosses 1.0
(max 1.13). Do not extrapolate it to larger networks without N=4000 for frm and both, which the grid
does not have.

### frm is the unstable condition

frm has the only non-zero b AND much the noisiest spectra: D_95 seed-sd reaches +-14.5 at
(N=2000, k=5) and +-9.7 at k=8, against +-0.5 in every other condition. That is consistent with the
transient behaviour seen in `flipflop_active_vs_loss.py`, where frm's loss spikes to 1e4 and its
active fraction swings between 1.0 and 0.4 at the loss floor. frm's dimensionality numbers should
carry that caveat wherever they are quoted.

### Sampling was validated before use, not assumed

D_PR is a ratio of spectrum moments and is biased when the sample count approaches the true
dimensionality. Pre-registered bar: accept if D_PR moves less than 5% when the sample count doubles.
Measured on N=2000 runs, 960 -> 9600 samples gives 6.62 -> 6.71 (+1.4%) and 5.64 -> 5.75 (+2.0%).
4800 samples used (32 trials, every 2nd timepoint), far into the converged regime for D ~ 2-15.

### Bug found while producing this

⚠️ `std_bigN` contains CALIBRATION folders trained for 400-600 iterations (`_iters=600` in the
folder name). `pr_matrix.load` drops them via `MIN_ITERS` on the trace length, but any script that
globs `*LastParams*.npz` instead of the trace picks them up, and an untrained net's spectrum is not
a result. In the first pass they added a spurious N=3000 row and a 25th none/N=4000 run. Now gated
on `trainer.max_iter >= MIN_ITERS`, giving exactly 336 = the known grid count. **Audit any other
folder-globbing script for the same hole.**

## ▶ WHAT rws ADDS TO frm: IT MOLDS TRANSIENT UNITS INTO SUSTAINED ONES — 2026-09-09 13:30

`flipflop_temporal_pr.py`, plus the peakiness comparison in the session scratchpad. N=2000, k in
(2,5,8), 3 seeds per condition, noise-free rates over 64 trials.

### The measure: temporal PR, the dual of the PR used everywhere else

    temporal PR_i = (sum_s r_is)^2 / sum_s r_is^2      s = pooled (time, trial) samples

`pr_matrix` reports PR over UNITS - the effective number of units carrying activity. This is PR over
TIME - the effective number of samples each unit is active for. Divided by the sample count it is
exactly the Treves-Rolls lifetime sparseness, bounded in (0, 1], threshold-free.

### frm hits its own target; what it does not constrain is occupancy

frm's penalised quantity is `tau * (logsumexp(r/tau) - log M)` with tau = 0.1 - a SOFT-MAX over
(time, trials), i.e. a peak-ish statistic - driven two-sided toward
`cap = 0.3 * log1p(100)/log1p(N)`. It succeeds at that:

| quantity, CV across LIVE units | frm | both |
|--------------------------------|-----|------|
| frm's own soft-max activity | 0.125 | 0.112 |
| temporal PR | **0.800** | **0.361** |
| duty cycle (frac of samples > 0.5 x own peak) | 0.766 | 0.355 |

Units under frm reach the right PEAK. They differ in how much of the time they sit there, and frm
has no term that sees this.

### Eleven peakiness measures, only the occupancy family separates the conditions

CV across live units, ratio frm/both, identical liveness mask throughout:

| measure | CV frm | CV both | ratio |
|---------|--------|---------|-------|
| sparseness (Treves-Rolls) | 0.800 | 0.361 | **2.21** |
| temporal PR | 0.800 | 0.361 | **2.21** |
| duty@0.5 | 0.768 | 0.354 | 2.17 |
| duty@0.25 | 0.805 | 0.381 | 2.11 |
| duty@0.75 | 0.789 | 0.410 | 1.93 |
| gini | 0.250 | 0.207 | 1.21 |
| peak/median | 7.392 | 6.796 | 1.09 |
| peak/mean | 0.969 | 0.921 | 1.05 |
| q90/median | 7.529 | 7.165 | 1.05 |
| kurtosis | 1.267 | 1.944 | 0.65 |
| drive_z = mean(drive)/std(drive) | 1.534 | 2.838 | **0.54** |

⚠️ `peak/mean` FAILS despite carrying the same information (`peak/mean ~ 1/duty`): the reciprocal
has a heavy right tail dominated by the few lowest-duty units, which exist in BOTH conditions and
swamp the CV. Use the bounded form.

⚠️ `peak/median` and `q90/median` are DEGENERATE HERE. The median rate is exactly zero for 75% of
frm units and 59% of both units - the typical unit is silent more than half the time. Never
substitute a median-based peakiness measure on this task.

### The distribution is BIMODAL, and rws empties one mode

Summary CVs hid the structure. Fractions of ALL N units, N=2000, k=3, 3 seeds:

| | dead | live but tPR/n < 0.1 | 0.1-0.25 | 0.25-0.45 | > 0.45 |
|--|------|----------------------|----------|-----------|--------|
| frm | 0.136 | **0.318** | 0.128 | 0.284 | 0.134 |
| both | 0.000 | **0.062** | 0.040 | 0.619 | 0.279 |

Under frm, **45% of units are low-occupancy** (13.6% fully dead + 31.8% alive but on under 10% of
the time). Under frm+rws that is **6.2%**, with 62% landing in the 0.25-0.45 band against 28%.

⚠️ THE QUASI-SILENT GROUP IS INVISIBLE TO EVERY SILENCE CRITERION IN THIS PROJECT. Those units pass
`participation >= SILENT_FLIPFLOP = 4e-2` comfortably. They have the correct peak activity - exactly
what frm asks - and are simply rarely on. The dead/alive boundary cuts through the middle of a
single continuous low-occupancy population; "silent unit" counts and temporal PR are not the same
measurement and must not be conflated.

### Reading: rws molds transient units into sustained ones

frm alone yields a population split between properly engaged units and technically-active-but-barely-on
units. Adding rws collapses the second group into the first. Silence is then the limiting case of
low occupancy rather than a separate phenomenon, which supersedes the earlier framing of rws as
doing two separate things (dead-unit rescue plus duty-cycle compression) - it is one effect along a
continuum.

⚠️ THIS IS A CROSS-SECTIONAL INFERENCE AND CANNOT SUPPORT THE WORD "MOLDS" ON ITS OWN. Comparing
independently trained frm and frm+rws networks cannot distinguish (a) the same units being pushed to
higher occupancy from (b) frm+rws finding a different solution in which different units occupy those
roles. Only a within-network intervention that tracks the SAME units across a penalty switch can
separate these. See the intervention design logged below.

### Two mechanisms proposed and BOTH REFUTED

1. *Tonic-vs-transient loophole, measured as peak/mean* - predicted CV ratio > 1.5, measured 1.06.
   The occupancy story survived only when measured with a bounded statistic.
2. *rws makes the operating point more uniform*, predicted CV of `mean(drive)/std(drive)` larger
   under frm. Measured the OPPOSITE: 1.534 frm vs 2.838 both, ratio 0.54. Capping effective
   in-degree makes drive statistics MORE heterogeneous while making occupancy MORE homogeneous.

So the phenomenon is solid across five occupancy measures, and the mechanism linking rws's
`S = (sum|w|)^2/sum w^2 <= 20` to occupancy is STILL OPEN. A nonlinear-compression account (duty
~ Phi(drive_z) saturating) is untested speculation.

## ▶ PLANNED: PENALTY-SWITCH INTERVENTION TO TEST "MOLDING" — designed 2026-09-09 13:30

The cross-sectional frm-vs-both comparison above establishes a DIFFERENCE but cannot establish
MOLDING. Two accounts fit it equally: (a) rws pushes the same low-occupancy units up, (b) frm+rws
converges on a different solution in which different units occupy those roles. Only tracking unit
IDENTITY across a penalty switch separates them.

### Design: four arms, N=2000, k=3, 3 seeds each = 12 runs

| arm | warm-start from | continue training with | role |
|-----|-----------------|------------------------|------|
| A1 | frm net | frm + rws | TREATMENT: do transient units become sustained? |
| A2 | frm net | frm | control: extra training + optimiser restart |
| A3 | both net | frm only | REVERSE: does bimodality return? |
| A4 | both net | frm + rws | control |

⚠️ A2 AND A4 ARE NOT OPTIONAL. Warm-starting resets Adam state and adds iterations; without the
same-penalty arms every change is attributable to those rather than to the penalty switch.

### Metrics, all tracked PER UNIT across the switch

* primary endpoint: fraction low-occupancy = dead + (live and tPR/n < 0.1)
* identity: Spearman(tPR_before, tPR_after). This is what makes the experiment worth running.
* molding signature: regression of `delta tPR_i` on `tPR_before_i`
* mechanism: within-network regression of `delta tPR_i` on `delta S_i`, S = (sum|w|)^2/sum w^2 per
  row. Cross-sectionally S barely varies inside a condition (all frm rows ~743), so it cannot be
  tested there; DURING the transition rows change at different rates, which finally provides the
  within-network variation needed to test the mechanism directly.
* confound guard: task r2 every probe. If adding rws breaks the task, occupancy changes are about a
  broken network, not about rws.

### Pre-registered predictions

A1  low-occupancy fraction falls 0.45 -> below 0.15; `delta tPR` vs `tPR_before` correlation
    rho < -0.4; identity preserved, Spearman(before, after) > 0.5.
    FALSIFIED IF the fraction is unchanged (rws does not act on existing units), OR identity
    correlation ~ 0 - that would mean RESHUFFLING, not molding, and the word must be dropped.
A2  no change beyond drift. If A2 moves, the effect is extra training, not rws.
A3  low-occupancy fraction rises 0.06 -> above 0.20.
A4  no change.

⚠️ A3 MAY SHOW HYSTERESIS, AND THAT IS A RESULT, NOT A FAILURE. Once rws has sparsified W_rec to
S ~ 20, nothing pushes S back up: weight decay is 1e-6 and frm has no term that sees S. If A3 does
NOT revert, rws's contribution is a STRUCTURAL change to connectivity that frm alone cannot undo -
a stronger and more interesting claim than reversibility. Predict the direction, accept either
answer.

### Cost and the implementation blocker

50k iterations at N=2000 is ~7.6 h (from the measured 400k ~ 61 h), so 12 runs fit one Della
gpu-short wave. Cheap.

⚠️ WARM START DOES NOT EXIST IN THE CODEBASE. `run_experiment.py:225` SAVES `*_AdamState_*.pt` but
nothing ever loads it, and there is no path to initialise from `*LastParams*.npz`. This must be
added first: a config key naming the source folder, with the launcher passing the per-task folder
(the one place a launcher may override config, since it is the swept parameter). Fresh Adam state is
acceptable and arguably preferable - frm-specific momentum should not carry across the switch - but
the weights must be loaded exactly.

### Cheap immediate probe: weight surgery, and what it can and cannot show

Sparsify each row of an frm net's W_rec to S ~ 20 (keep the largest weights) and measure temporal PR
with NO retraining. This asks whether low in-degree is SUFFICIENT for high occupancy.
⚠️ It destroys task performance, so a change in occupancy is confounded by the network being broken.
Its value is as a NEGATIVE result: if surgery alone does not raise occupancy, rws acts through the
LEARNING DYNAMICS rather than through the connectivity statistic per se. Do not report a positive
surgery result as evidence for the mechanism.

### IMPLEMENTED 2026-09-09 13:50 — warm start, temporal-PR tracking, launcher

Concrete parameters: N=2000, k=3. Base phase 50k iterations, warm phase 10k. 6 base runs
(frm x3 seeds, both x3 seeds) + 12 warm runs (4 arms x 3 seeds) = 18 jobs.

⚠️ 6 BASE RUNS, NOT 12, AND THIS IS DELIBERATE. A1 and A2 branch from the SAME frm parent, A3 and A4
from the same both parent. That makes A1-vs-A2 a PAIRED contrast with between-seed variance removed;
giving each arm its own parent would cost twice the base compute and lose the pairing.

⚠️ WHY A FRESH 50k BASE RATHER THAN THE EXISTING 400k penlong NETS. A 10k switch is a 2.5%
perturbation to a 400k-converged net and would probably show nothing. At 50k the net is converged in
loss (frm and both both reach 1.03x floor by ~30-100k at N=2000) but still plastic.

Three pieces of code:

1. `run_experiment.py` — the warm-start LOAD path. The AdamState commit added the SAVE half and its
   own comment recorded that "there is still no load path in this script"; this is that half.
   `paths.init_from` names a source run folder, `paths.init_adam` (default false) optionally
   restores the moments. Strict: every `named_parameters()` entry must be present with matching
   shape or it raises. `y_init` is restored explicitly because it is NOT a Parameter yet is saved
   and does change the dynamics. Any other array-valued key that goes unrestored prints a warning
   rather than being dropped silently.
   ⚠️ THE PENALTY IS NOT INHERITED — only weights. lambda_frm/lambda_rws come from the new run's
   config, which is exactly what makes the switch possible.
   ⚠️ ADAM MOMENTS OFF BY DEFAULT: frm-shaped momentum carried across the switch would import the
   old objective's search direction, the confound the experiment exists to isolate.

2. `Trainer.track_participation_` — per-unit temporal PR, stored on the
   `store_participation_every` cadence beside `participation` and sharing `participation_iters`.
   Computed from the probe's rates, which are then discarded, so it CANNOT be added retrospectively
   and cannot be recovered from the stored participation vector (std + q90). Base phase stores every
   100 iterations, warm phase every 10 (1000 samples over the switch, ~8 MB per run at N=2000).

3. `slurm/SilentReLU_flipflop_switch_della.slurm` — both phases behind `PHASE=base|warm`.
   The warm phase resolves its parent by globbing the base tag and FAILS if the match is not exactly
   one folder: zero means the base did not finish, more than one means a rerun left a duplicate and
   the arm would silently branch from whichever sorts first.

⚠️ HYDRA CANNOT PARSE AN UNQUOTED RUN FOLDER. Run folders contain ';' and '=' (e.g.
`...relu;N=2000;seed=...;Lfrm=0.1`), both hydra override-grammar characters, so `paths.init_from`
must be single-quoted INSIDE the override: `"paths.init_from='$SRC'"`. Unquoted it dies with
`LexerNoViableAltException`.

Validation before any cluster submission (local, N=40, k=2, 60 iterations):
* temporal PR recorded with the right shape and range (values < n_samples).
* warm start with `trainer.lr=0` reproduces the source weights BIT-IDENTICALLY — max|diff| = 0.000e+00
  for W_rec, W_inp, W_out, bias and y_init. Threshold set before running: exact equality, no tolerance.
* the switch takes effect: base folder tag carries `Lrws=0`, the warm run's carries `Lrws=0.05`, and
  the saved config records `init_from` for provenance.

### REVISED 2026-09-09 14:05 — reuse the existing 400k nets; no base phase needed

The base phase is now OPTIONAL and off by default. The four arms branch from the existing penlong
frm and both nets at N=2000, k=3 (3 seeds each), so only the 12 warm runs are needed.

⚠️ THE OBJECTION TO REUSE WAS WRONG AND THE DATA SAYS SO. The worry was that a short switch is a
~2% perturbation to a 400k-converged net. Measured on the existing `both` runs' participation
traces, PR/N reaches **0.90 within 1000 iterations** and 0.96 by 5k - rws acts almost immediately,
because its gradient does not care how converged the TASK loss is. At S ~ 743 the rws term is
(743-20)^2/400 ~ 1307 per row, times lambda 0.05 -> ~65, against a task loss of ~0.03. It dominates
from the first step. Reusing the nets is also strictly BETTER than fresh parents: the intervention
then speaks to the very networks the 45%-vs-6.2% result was measured on.

| | PR/N at 1k | 5k | 10k | 20k | final |
|--|-----------|-----|-----|-----|-------|
| both, 3 seeds | 0.90-0.92 | 0.71-0.96 | 0.78-0.96 | 0.92-0.94 | 0.942-0.946 |
| frm, 3 seeds | 0.55-0.88 | 0.53-0.69 | 0.64-0.68 | 0.54-0.76 | 0.468-0.708 |

⚠️ WARM PHASE RAISED FROM 10k TO 50k BECAUSE THE ARMS ARE NOT SYMMETRIC. rws acts in ~1k, so A1 is
covered many times over; but frm's heterogeneity develops SLOWLY (PR/N drifts 0.57 -> 0.47 over
400k) and nothing pushes S back up once rws has pulled it to ~20. A null in A3 at 50k is SUGGESTIVE
of hysteresis but cannot be distinguished from "not yet" - report it that way, do not call it
irreversibility.

⚠️ RUN ON SPOCK. The k=3 N=2000 penlong parents live on Spock; Della's penlong copy holds other
cells only. `SilentReLU_flipflop_switch_spock.slurm` is byte-identical below the scheduler block.
    PHASE=warm sbatch --array=1-12 --time=16:00:00 slurm/SilentReLU_flipflop_switch_spock.slurm

Parent resolution dry-run against the local mirror confirms the pairing: tasks 1 and 4 (A1/A2,
rep 0) resolve to the SAME frm parent, tasks 7 and 10 (A3/A4, rep 0) to the same both parent.

Two portability fixes found while dry-running:
* `mapfile` is a bash-4 builtin, absent on macOS bash 3.2, so the resolution logic could not be
  dry-run locally at all. Replaced with a portable `while read` loop.
* ⚠️ `SilentReLU_flipflop_frmboth_bigN_spock.slurm` had inherited `#SBATCH --array=1-144` from the
  N<=2000 script while having only 48 tasks. Submitted without an explicit --array it would have
  launched 96 tasks whose index decodes out of range. Default removed.

## ▶ FUNCTIONAL DIVERSITY AND CLUSTERING: frm+rws MAKES ~7-9 CELL TYPES — 2026-09-09 14:56

Three new measures, all at N=2000 unless stated, all on live units only (silent units otherwise
dominate every one of them). These answer a question none of the earlier heterogeneity work touched:
`pr_matrix` (PR over units), `flipflop_temporal_pr` (PR over time) and `flipflop_heterogeneity`
(A1-A6) all measure HOW MUCH and WHEN a unit is active, never WHAT it does.

### 1. Functional diversity — `flipflop_diversity.py`

    D_shape   PR of the CORRELATION spectrum: each unit's time course centred and scaled to unit
              norm, so trace = n_live and loud units cannot dominate. Counts distinct response
              SHAPES. D_PR (covariance) mixes shape with amplitude; the ratio separates them.
    redundancy  mean |correlation| over unit pairs. -> 1 clones, -> 0 unrelated.
    sel_PR    each unit regressed on the k target bit time courses; loadings normalised; PR of their
              second-moment matrix = effective number of SELECTIVITY directions, bounded by k.
    R2_med    median variance explained by those k regressors.

k=8, N=2000, 3 seeds:

| pen | live | D_PR | D_shape | ratio | redundancy | sel_PR (max 8) | R2_med |
|-----|------|------|---------|-------|------------|----------------|--------|
| none | 0.176 | 14.3 | 22.4 | 1.6 | 0.125 | 7.45 | 0.402 |
| rws | 0.215 | 14.1 | 22.0 | 1.6 | 0.113 | 7.70 | 0.413 |
| frm | 0.925 | 15.1 | **10.7** | 0.7 | **0.189** | **4.84** | 0.496 |
| both | 1.000 | 12.9 | 13.7 | 1.1 | 0.138 | **7.99** | **0.736** |

**The units are NOT redundant copies** - mean |correlation| is 0.11-0.19 everywhere, so the low
dimensionality is a low-dimensional latent structure with diverse loadings, which is what cortical
populations look like. That is the positive claim the "model of cortical computation" framing needs.

**frm collapses functional diversity while inflating the unit count**: 93% of units active but
selectivity spanning only 4.84 of 8 task dimensions, the lowest D_shape of any condition, and the
highest redundancy. **rws restores it to the ceiling**: 7.99/8, with task-explained variance nearly
doubled (0.736 vs 0.496). A second axis, independent of occupancy, on which rws converts frm's units
from nominal to meaningful.

⚠️ THE R2 REGRESSORS ARE THE TARGET BITS ONLY. A unit encoding input pulses, transitions or internal
memory would score low without being meaningless. Add the input trains as regressors before putting
"frm units are less task-explained" in the paper.

### 2. Variance explained — `flipflop_variance_explained.py`

Fixed cut through the same spectra as `dimensionality_matrix.png` (same cache), so the two are
like-for-like. Grid means: VE@5 = 0.712 none / 0.714 rws / 0.694 frm / **0.756 both**.

VE@5 is set by TASK COMPLEXITY, not by N or penalty - every condition falls ~0.95 at k=1 to ~0.50 at
k=8 with the N curves lying on top of each other. Same conclusion D_PR gave, now from a fixed cut
with no moment ratio involved; two independent readings of the spectrum agreeing is worth more than
either alone.

⚠️ frm has the LOWEST VE@5 (0.694) AND the HIGHEST D_PR (8.88) and D_95 (20.4 vs 15-17). Not a
contradiction - VE@5 sees only the first five eigenvalues, D_PR is a moment over all of them. frm
has a lighter head AND a heavier tail. Never quote VE@5 alone.

### 3. Clustering — `flipflop_clustering.py`

Units as points in sample space, centred over units, projected onto the PCs carrying 99% of the
across-unit variance, k-means for 2..30 clusters.

⚠️⚠️ THE NAIVE COMPARISON IS NOT MERELY NOISY, IT IS BACKWARDS. Raw silhouette peaks at 0.912 for
`none` - apparently near-perfect clustering. THE PERMUTATION NULL SCORES 0.912 TOO. With an
amplitude-dominated embedding, k=2 splits off one extreme unit and shuffled data does that just as
well. `none` and `rws` have NO real cluster structure; reporting raw silhouette would have inverted
the entire result.

Two controls are mandatory and both are in the script: every condition subsampled to a COMMON live
unit count (285 vs 2000 otherwise, and m(99%) ranges 13 to 54), and every curve scored against a
per-net null in which each unit's samples are permuted independently.

| pen | raw: peak sil | at k | null | GAP | | norm: peak gap | at k |
|-----|---------------|------|------|-----|-|----------------|------|
| none | 0.912 | 2 | 0.912 | **-0.000** | | 0.543 | 27 |
| rws | 0.855 | 2 | 0.857 | **-0.001** | | 0.470 | 27 |
| frm | 0.403 | 30 | 0.100 | 0.304 | | 0.426 | 21 |
| both | 0.516 | 7 | -0.023 | **0.539** | | **0.564** | **9** |

**frm+rws produces ~7-9 tight, well-separated response types.** It is the only condition with a
sharp, well-localised optimum; the others are still rising at k=30, the signature of no natural
cluster count (a continuum being chopped finer). Holds in both the raw and the unit-normalised
variant, so it is not an amplitude effect. The inertia elbow agrees independently.

⚠️ THE none/rws-vs-both CONTRAST IS NOT YET SETTLED. none and rws have only 282/302 live units, so
after subsampling to 275 the embedding is near-saturated (275 points, m=45-50 normalised) where
silhouette behaves badly. The null corrects the BIAS, not the VARIANCE. Repeat at a larger common n
before relying on it. The frm-vs-both contrast is solid - both have thousands of live units.

## ▶ THE PAPER'S STRUCTURE RESULT: SINGLE-UNIT SELECTIVITY — 2026-09-09 17:20

`flipflop_arms.py` (grid sweep, `data/arms_cache.npz`), `flipflop_selectivity_matrix.py` (the
figure, laid out like pr_matrix), `flipflop_unitcloud.py` (the picture that reframed everything).

⚠️ THIS SECTION REPLACES AN EARLIER VERSION (commit 13e23d9) WHOSE NUMBERS WERE COMPUTED ON A
MISSPECIFIED BASIS. Those regressed each unit on the k SIGNED bit traces. ReLU units are
non-negative and many drop to EXACTLY zero for one sign of a bit, which a line through three levels
cannot fit. The correct basis is HALF-WAVE RECTIFIED - relu(+b_j) and relu(-b_j), so 2k regressors
plus an intercept. It spans the linear one (relu(b) - relu(-b) = b) plus the absolute values, so it
is strictly richer, and it raised median R^2 by ~0.14 in EVERY condition. Do not quote the old
numbers.

### What the metric is, in plain terms

The task asks the network to REMEMBER k bits, each currently +1 or -1. Take one unit and write down
its firing rate at every moment of every trial; alongside it write down what the bits were at that
same moment. Then ask: **can this unit's activity be predicted from what the bits currently are?**
That is a regression, and R^2 is the fraction of the unit's activity the bits account for. A unit
near 1 is doing the task - watch it and you can read off the remembered bits. A unit near 0 is
active but its activity has nothing to do with what is being remembered.

The regression weights also say WHICH of the 2k channels a unit cares about. **Hoyer sparsity** of
that weight vector, `(sqrt(d) - ||b||_1/||b||_2)/(sqrt(d) - 1)` with d = 2k, is 1 when all the
tuning sits on one channel and 0 when it is spread evenly. It needs no threshold, no clustering and
no null, and it is the per-unit dual of the participation ratios used elsewhere here, since
`||b||_1/||b||_2 = sqrt(PR of the loading vector)`.

### The picture that reframed everything: it is a STAR

`unitcloud_sel_N2000_k3.gif`. Units do not scatter randomly in selectivity space: at k=3 they form a
**6-ARMED STAR**, dense arms along +-bit1, +-bit2, +-bit3 with a core at the origin. Most units care
about ONE bit. There are 2k arms, one per signed bit channel - which is also why the rectified basis
is the natural one.

⚠️ LOOK AT THE DATA BEFORE CHOOSING A STATISTIC. A star is neither clusters nor a manifold nor a
Gaussian, and silhouette, intrinsic dimension and multivariate kurtosis were each fitting one of
those wrong models to it. That is why five earlier claims reversed; see the retraction record below.

### The result, full grid, rectified basis

Median R^2 over TUNED units (R^2 >= 0.15), mean over N:

| pen | k=1 | k=2 | k=4 | k=6 | k=8 | mean |
|-----|-----|-----|-----|-----|-----|------|
| none | 0.658 | 0.681 | 0.635 | 0.574 | **0.540** | 0.615 |
| rws | 0.672 | 0.668 | 0.655 | 0.619 | 0.595 | 0.639 |
| frm | 0.869 | 0.770 | 0.808 | 0.883 | 0.870 | 0.843 |
| both | 0.871 | 0.875 | 0.900 | 0.904 | **0.913** | **0.897** |

**R^2 moves in OPPOSITE DIRECTIONS with task complexity**: `none` declines 0.658 -> 0.540 from k=1
to k=8 while `both` rises 0.871 -> 0.913. Penalised networks get BETTER at dedicating units to the
task as it hardens; unpenalised ones get worse. All four N curves lie on top of each other inside
each panel, so this is task complexity, not size.

Median Hoyer sparsity (mean over the grid): none 0.736, **rws 0.808**, frm 0.769, **both 0.857**.
⚠️ NOTE rws ALONE BEATS frm ALONE on sparsity, and frm REDUCES it relative to rws. Sparsity is rws's
contribution; the same division of labour the temporal-PR analysis found, now in the tuning domain.

Both halves together, per N=2000 network at k=3:

| pen | tuned/live | med R^2 given tuned | med Hoyer | live/N | tuned units per 2000 |
|-----|------------|---------------------|-----------|--------|----------------------|
| none | 0.857 | 0.609 | 0.766 | 0.141 | **241** |
| rws | 0.847 | 0.634 | 0.807 | 0.151 | 255 |
| frm | 0.811 | 0.839 | 0.748 | 0.864 | **1401** |
| both | **0.959** | **0.901** | **0.841** | **1.000** | **1918** |

`both` wins on every axis at once: most units alive, highest fraction of those tuned, best explained,
most sparsely tuned. **~8x more task-tuned units than unpenalised, each explained to 0.90 rather
than 0.61.**

⚠️ ALWAYS REPORT BOTH HALVES. Median R^2 alone is conditional on being tuned and flatters conditions
with few live units; tuned fraction alone ignores how good the tuning is. The product - tuned units
per network - is the honest summary.

⚠️ THE RECTIFIED BASIS REHABILITATES frm. On the old misspecified basis frm showed 1012 tuned units
and a "zero-R^2 quarter"; correctly fitted it has **1401** tuned units at median R^2 0.84. That
quarter was largely an artefact of fitting a line to rectified responses. frm's real deficit is
SPARSITY (0.748, lowest of the four), not tuning quality. An earlier framing here was unfair to frm.

### Method notes that must survive into the paper

* 2k+1 parameters per unit (7 at k=3), ordinary least squares, **no regularisation** - and that is
  checked, not assumed: design matrix (9600, 7), condition number 6.7, in-sample vs 5-fold-CV R^2
  gap 0.0005, n/p ~ 1370. The pairs relu(+b)+relu(-b) = 1[b != 0] do not collapse onto the intercept
  because each bit sits at zero for a different 23-33% of samples.
* One JOINT regression per unit, not k separate ones and no maximum taken. At k=3 the joint R^2 and
  the max of three single-bit R^2 differ by 0.003 - a second, free confirmation of the star, since
  mixed selectivity would make the joint far exceed the max.
* Arm counts at N=2000, k=3 are 335/328/355/342/312/328 across the six arms: near-perfectly even.
* Arm PURITY is identical in all four conditions (0.244-0.248 excess over random, curves
  superimposed at every k). How single-channel-selective a TUNED unit is, is set by the task, not by
  the penalty. Purity and Hoyer are not redundant: Hoyer also sees units whose tuning is weak.
* k=1 is degenerate for every arm statistic (one bit, two arms).

### The defensible sentence

frm makes every unit active and genuinely task-tuned, but its tuning is the least sparse of any
condition; rws sharpens tuning onto single channels. Together they give **~8x more task-tuned units
than unpenalised, each explained to R^2 = 0.90 and tuned to essentially one channel** - and unlike
the unpenalised network, they get BETTER as the task gets harder.

### ⚠️ RETRACTION RECORD FOR THIS THREAD — five claims that did not survive

Kept because each failed for a DIFFERENT and reusable reason. Every one of them looked solid when
first measured.

1. **"frm+rws produces ~7-9 discrete cell types."** From k-means silhouette at k=3. RETRACTED: the
   peak cluster count was 7 at k=3 but 22-30 at k=1, 5 and 8 - i.e. against the search boundary,
   which was the pre-registered refutation criterion. A single grid cell is not a result.

2. **"none and rws show real cluster structure."** RETRACTED and INVERTED: raw silhouette peaked at
   0.912 for `none`, apparently near-perfect - and the permutation null scored 0.912 too. With an
   amplitude-dominated embedding, k=2 splits off one extreme unit and shuffled data does the same.
   ⚠️ SILHOUETTE HAS NO MEANINGFUL ZERO. Never report it without a null.

3. **"Penalisation raises intrinsic dimensionality (2.1 -> 4.4)."** RETRACTED TWICE. First the
   values were wrong: `d = n/sum(log mu)` is the MLE for the FULL sample and the code trimmed the
   top 10% of mu before applying it, inflating every ID by ~30% (true d=3 read as 4.04). Trimming
   belongs with the CDF-slope estimator. Then, with corrected values, the ordering did not replicate
   across k at all. ⚠️ CALIBRATE AN ESTIMATOR AGAINST KNOWN GROUND TRUTH AT THE ACTUAL SAMPLE SIZE
   BEFORE TRUSTING IT - that is what caught the trim bug.

4. **"frm+rws has the LIGHTEST tails (Mardia kurtosis 696 vs 912)."** RETRACTED AND INVERTED: Mardia
   was computed on each net's own 99%-variance embedding, whose dimension m ranges 14-88, and its
   Gaussian expectation is m(m+2). m correlates with the statistic at r = +0.755. At matched m=10 the
   ordering flips: `both` is the HEAVIEST on all five independent tail measures.

5. **"frm+rws has the heaviest tails"** (the corrected version of 4). Technically right but the
   INTERPRETATION was wrong: it is not outlier contamination. The cloud is a 6-armed star and the
   arms ARE the tails, so "heavier-tailed" means "more units recruited into the tuning arms". The
   number was fine; the model behind it was not.

⚠️ THE COMMON THREAD: every failure came from applying a statistic whose implicit model (clusters,
manifold, Gaussian) did not match the data, or from a confound (dimension, sample size, amplitude)
that the statistic silently absorbed. The fix that finally worked was to PLOT THE POINT CLOUD AND
LOOK AT IT, then choose statistics matched to the shape actually there. `flipflop_unitcloud.py`
exists for that purpose; run it first next time.

Scripts retained rather than deleted, since each carries its own defect documented in its docstring:
`flipflop_clustering.py` (null design), `flipflop_epairs.py` (saturating statistic; ePAIRS is pinned
near +1 in every condition because all ratios are < 0.25), `flipflop_manifold.py` (ID with a null
that saturates), `flipflop_structure_matrix.py` (⚠️ ITS MARDIA COLUMN IS CONFOUNDED - superseded by
`flipflop_heavytail.py` at fixed m), `flipflop_heavytail.py` (five tail measures, correct but see
point 5 for how to read them).

## ▶ THE PENALTY-SWITCH INTERVENTION: rws STABILISES, IT DOES NOT MOLD — 2026-09-10 08:33

Spock array `6112117`, 12 runs of 50k iterations, warm-started from the 400k penlong nets at
N=2000, k=3. `flipflop_switch_trajectories.py` for the figures; traces carry per-unit temporal PR
every 10 iterations (5000 snapshots x 2000 units per run).

| arm | switch | median tPR (all) | median tPR (live) | dead frac | final IQR | rho(start,end) | corr(delta,start) | churn |
|-----|--------|------------------|-------------------|-----------|-----------|----------------|-------------------|-------|
| A1 | frm -> frm+rws | 0.157 -> **0.353** | 0.233 -> 0.352 | 0.106 -> **0.000** | 0.179 | +0.26 | **-0.67** | **1.7** |
| A2 | frm -> frm | 0.159 -> 0.209 | 0.231 -> 0.222 | 0.106 -> 0.094 | 0.330 | +0.48 | -0.40 | **3.2** |
| A3 | both -> frm | 0.353 -> **0.228** | 0.353 -> 0.228 | 0.000 -> **0.088** | 0.325 | +0.21 | -0.46 | 1.9 |
| A4 | both -> frm+rws | 0.354 -> 0.359 | 0.354 -> 0.359 | 0.000 -> 0.000 | 0.145 | **+0.78** | -0.23 | 1.4 |

`churn` = dead<->alive transitions per unit over the 50k iterations. It is the statistic that
turned out to matter and it was not in the pre-registered plan.

⚠️ CORRECTED 2026-09-10 09:05, TWO STATISTICS IN THIS TABLE WERE MIS-USED. See "Statistics that had
to be corrected" at the end of this entry. The churn column understates the effect (it is
sampling-rate dependent) and the corr(delta,start) column supports nothing at all.

### 1. A4 validates the design

The `both -> both` control changes NOTHING: median 0.354 -> 0.359, dead 0.000 -> 0.000, rho = 0.78,
the highest identity of any arm. Warm-starting, resetting Adam and 50k extra iterations are inert on
their own, so every effect in A1 and A3 is the penalty switch. ⚠️ A2 IS NOT INERT (median +0.050),
which is exactly why both same-penalty controls were needed rather than reading A1 against zero.

### 2. Symmetric and fully reversible

| | forward (A1-A2) | reverse (A3-A4) |
|--|-----------------|-----------------|
| median tPR | **+0.146** | **-0.130** |
| dead fraction | **-0.094** | **+0.088** |
| final IQR | -0.151 | +0.180 |

⚠️ THE HYSTERESIS PREDICTION WAS WRONG. The design doc predicted A3 might not revert, since once rws
has pulled S to ~20 nothing pushes it back (weight decay 1e-6, frm has no term seeing S). It reverts
almost exactly. **rws's effect is dynamically maintained, not a one-time structural change to
connectivity.** Do not describe it as reshaping the network into a new stable regime.

### 3. The mechanism is STABILISATION, not elevation

Under frm alone units cycle in and out of silence continuously. Sampled every 250 iterations, which
is the rate at which the comparison is meaningful (see the correction note below): **2.94
transitions per unit under frm alone against 0.30 with rws, nearly 10x**. Under frm 43% of units
never cross the boundary and **38.7% cross it four or more times**; with rws that is 83% and 1.7%.
The ~10% dead fraction under frm is a DYNAMIC EQUILIBRIUM, not a fixed set of casualties - at any
moment ~10% are dead, but not the same 10%.

That explains the identity numbers, which otherwise look like reshuffling. A2 has rho = 0.48 WITH NO
PENALTY CHANGE AT ALL; A4, where nothing changes, holds 0.78. So the low rho in A1 (0.26) is not rws
scrambling the population - it is rws STOPPING a scrambling frm was doing anyway, and then
compressing the distribution (IQR 0.330 -> 0.179) so ranks lose resolution.

### 4. No task cost

Final loss 0.027-0.030 in all four arms, differences of +-0.002. r2 held 0.934-0.948 at every
checkpoint in every arm.

### Verdict against what was pre-registered

* "Molding" as defined in the plan (identity rho > 0.5): **NOT SUPPORTED**, A1 gives 0.26.
* The second criterion, corr(delta tPR, starting tPR), **ALSO FAILS** once its null is computed -
  see the correction note. BOTH pre-registered molding criteria therefore fail.
* Correct description is neither molding nor reshuffling: **rws levels the distribution and pins it
  in place.** That rests on the dead-fraction change, the IQR compression and the churn collapse,
  none of which depend on the two broken statistics.

⚠️ SUPERSEDES THE CROSS-SECTIONAL FRAMING. "frm makes units active but transient; rws molds them
into sustained ones" becomes **"frm cannot hold units alive; rws stabilises them there"** - causal,
reversible, and what the intervention actually shows.

### Two artefacts found while reading the traces, both ruled out

⚠️ ABRUPT POPULATION-WIDE COLLAPSES appear in the A1 rep 0 trajectory figure. They are transient loss
excursions - at every large jump the noise-free loss spikes (0.030 -> 0.51, 0.031 -> 0.34) with
corr(|d median tPR|, |d loss|) = +0.48, and 1.12% of probes exceed 10x the median loss against the
guard's 498 skipped updates = 0.996% of iterations. **Only task 1 of the twelve is affected**; every
other run reports zero skips. Ruled out first:
* NOT the spike guard - `0 rollbacks to last-good weights` in every run.
* NOT batch noise - `same_batch=False` draws a fresh 1024-trial batch per probe, but on a FIXED net
  across 12 independent batches the population median moves by sd 0.0011 and NO unit moves by more
  than 0.1, against swings of 0.3-0.5 in the figure. Two orders of magnitude too small.
Plot a clean seed (rep 1) and say which; `switch_trajectories_rep0.png` is kept only as the
documented example of the frm spike problem.

### ⚠️ Statistics that had to be corrected — 2026-09-10 09:05

**corr(delta, start) SUPPORTS NOTHING AND THE CLAIM BUILT ON IT IS WITHDRAWN.** The statistic has a
large built-in negative bias: if the endpoint were unrelated to the start, `corr(a, b-a)` is already
about -0.7 to -0.8 purely from regression to the mean. Against each arm's OWN shuffled-endpoint null
(endpoints permuted across units, both marginals preserved):

| arm | observed | own null | excess |
|-----|----------|----------|--------|
| A1 | -0.667 | -0.773 | **+0.106** |
| A2 | -0.394 | -0.685 | **+0.291** |

Both observed values are LESS NEGATIVE than chance, so the statistic is measuring rank preservation,
not "low starters gain most" - and the excess is LARGER IN THE CONTROL, the opposite direction from
the claim. The raw -0.67 vs -0.40 comparison is invalid because the arms have different final
spreads, and the spread sets the null. ⚠️ NEVER COMPARE corr(x, y-x) ACROSS GROUPS WITH DIFFERENT
VARIANCES WITHOUT A PER-GROUP NULL.

**CHURN IS SAMPLING-RATE DEPENDENT AND THE REPORTED NUMBERS UNDERSTATE THE EFFECT ~5x.**

| arm | every 10 iters (as first reported) | every 250 iters | never flip | flip 4+ times |
|-----|------------------------------------|-----------------|------------|---------------|
| A1 | 1.7 | **0.30** | 83% | 1.7% |
| A2 | 3.2 | **2.94** | 43% | **38.7%** |

A1's count collapses 5.7x under coarser sampling while A2's barely moves (1.09x), so the two are
different phenomena: **A1's residual churn is fast flicker at the 10-iteration scale; A2's is slow,
persistent switching that survives any sampling rate.** Counting them as equivalent turned a ~10x
separation into 1.9x. Quote the coarse-sampled figure and state the rate.

**ALL FOUR ARMS, RECOMPUTED AT THE 250-ITERATION RATE** (2026-09-10, `flipflop_switch_summary.py`,
3-seed means):

| arm | switch | rws after? | churn | never flip | flip 4+ |
|-----|--------|-----------|-------|------------|---------|
| A4 | both -> frm+rws | ON | **0.13** | 0.95 | 0.012 |
| A1 | frm -> frm+rws | ON | **0.35** | 0.83 | 0.022 |
| A3 | both -> frm | OFF | 1.54 | 0.70 | 0.194 |
| A2 | frm -> frm | OFF | **2.70** | 0.47 | **0.359** |

**The arms sort by whether rws is active after the switch, not by which parent they came from**, with
no overlap between the two groups and a 20x spread from A4 to A2. A3 starts from a perfectly stable
`both` network and climbs to 1.54 once rws is removed, so churn is not a property the network keeps -
it is suppressed only while rws is on. The most legible form: **95% of units under frm+rws never
cross the silence boundary once in 50k iterations, against 47% under frm alone.**

⚠️ There is a SECONDARY parent effect within each group (A4 < A1 and A3 < A2): arms descended from a
`both` net churn less than arms descended from an frm net, whichever penalty they end under. Do not
state the ordering as purely rws-determined.

**rho is affected by ties at zero.** Excluding units dead at the switch: A1 0.225 -> 0.344,
A2 0.444 -> 0.507. The gap narrows but survives; quote the live-only version.

### Endpoint stability differs between arms, and it matters for how endpoints are quoted

Comparing the full-resolution endpoint (iteration 49,990) against the subsampled one (49,750):

| arm | median tPR @49,990 | @49,750 | difference |
|-----|--------------------|---------|------------|
| A1 | 0.353 | 0.355 | 0.002 |
| A4 | 0.359 | 0.357 | 0.002 |
| A2 | 0.209 | 0.239 | **0.030** |
| A3 | 0.228 | 0.200 | **0.028** |

The two rws-ON arms have endpoints reproducible to 0.002; the two rws-OFF arms move by ~0.03
depending on which iteration you happen to sample, **because they are still oscillating at the end of
the run**. This is the same instability the churn statistic measures, showing up as endpoint
irreproducibility. ⚠️ Quote A2/A3 endpoints with that uncertainty attached, or average over a window
rather than taking the last snapshot.

⚠️ A3 HAS NOT CONVERGED at 50k - its dead fraction is still rising and its occupancy still falling.
The reverse direction is therefore a LOWER BOUND on how far removing rws would eventually take it.

## ▶ WHY THE TWO PENALTIES COMPOSE: A SCALING / MEAN-FIELD ACCOUNT, TESTED — 2026-09-10 14:31

> ⚠️ **Superseded on the mechanism (2026-09-11).** This entry argues from the ReLU scale symmetry. That explanation was retracted after the bounded-sigmoid result (entry of 2026-09-11 08:50 and the retraction note below it): the cause is the objective's missing term, not the activation's homogeneity. The measurements here (in-degree S, assembly share, switch arms, occupancy) stand; the symmetry reasoning does not. For the paper's mechanism section use the 2026-09-11 entry only.


Started from Pavel's first-moment scaling argument and ended somewhere different. Every step below
was tested on the trained weights; two of the intermediate theories were refuted and are recorded.

### 1. The scaling argument, and the tension the second moment adds

Drive to unit i: `D_i = sum_j W_ij r_j`. With presynaptic mean `m`, temporal variance `v`, and
row cancellation `c_i = |sum_j W_ij| / sum_j |W_ij|`:

    E_t[D_i]   = m * c_i * ||W_i||_1
    Var_t[D_i] = v * ||W_i||_2^2          (uncorrelated-fluctuation mean field)

The first-moment requirement `E[D] = O(1)` with in-degree K and weight scale w gives `w ~ 1/K`
(Pavel's `1/p`) — but then `Var[D] ~ v/K -> 0`. The variance requirement gives `w ~ 1/sqrt(K)` —
but then `E[D] ~ sqrt(K) m` diverges unless signs cancel. **Both moments cannot be O(1) at large K
without a balanced row whose mean is a small residual of large cancelling terms. At K = O(1) the
tension vanishes**: `w = O(1)` satisfies both. `S = ||W||_1^2/||W||_2^2` is exactly the effective
in-degree and rws pins it at 20 (measured 743 -> 21).

### 2. frm: breaks the ReLU scale symmetry — this part is clean

ReLU is positively homogeneous, so `(W_rec, W_inp, b, W_out) -> (W_rec, aW_inp, ab, W_out/a)` leaves
the input-output map invariant with `h -> ah`. Per-unit activity scale is a FLAT direction of the
task loss; nothing sets it, and per unit it can drift to zero where the ReLU gradient vanishes and it
stays. frm pins that scale. This is why every other intervention in paper.md §3 fails: none of them
touch the symmetry.

### 3. ⚠️ REFUTED: the uncorrelated mean-field predictions for rws

Predicted (pre-registered): sparse rows need LESS cancellation (c larger under both); the operating
point `z_i = (m/sqrt v) c_i sqrt(S_i)` is LESS variable under both. Measured, N=2000 k=3:

| pen | S | \|w\|_2 | c | c/random-sign baseline | c*sqrt(S) |
|-----|---|---------|---|------------------------|-----------|
| frm | 835 | 1.06 | 0.155 | 4.4x ABOVE | 4.2 |
| both | 20 | 1.40 | 0.130 | 0.6x BELOW | 0.59 |

Both predictions backwards. `both` rows are MORE balanced than chance, frm rows LESS; the
mean-to-fluctuation ratio is 7x larger under frm. The uncorrelated mean field is the wrong tool
because the activity is ~6-dimensional (D_PR); presynaptic rates are anything but independent. The
right variables are the 2k bit-state subpopulations. (Weights DO grow as rows sparsify, 1.06 ->
1.40, consistent with the norm rising 60 -> 90 during the switch.)

### 4. What the weights actually show: bit-state ASSEMBLIES

Assign each tuned live unit to a bit-state by argmax over its 2k rectified loadings. Measure the
fraction of each row's total |W| mass that comes from units in the SAME state ("assembly share"),
against the chance level (fraction of units in that state). Full row, not top-20 — the top-20 view
flattered frm, whose rows are dense.

| k | none | rws | frm | both | chance |
|---|------|-----|-----|------|--------|
| 3 | 0.532 (3.1x) | 0.627 (3.7x) | **0.284 (1.7x)** | 0.665 (4.0x) | 0.17 |
| 5 | 0.494 (4.8x) | 0.619 (6.1x) | **0.266 (2.5x)** | 0.650 (6.5x) | 0.10 |
| 8 | 0.377 (5.2x) | 0.593 (8.5x) | **0.240 (3.3x)** | 0.728 (11.6x) | 0.07 |

⚠️ THE UNPENALISED NETWORK ALREADY BUILDS ASSEMBLIES. Among its ~230-330 surviving units, 38-53% of
weight mass is within-assembly, 3-5x chance, at every k. **Assemblies are the task's natural
solution. frm is the ONLY condition that departs from it** — dissolving them to near chance while
recruiting every unit. rws restores them, to above the unpenalised level, and the structure
STRENGTHENS relative to chance as k grows (11.6x at k=8, sixteen assemblies of ~125 units).

Within an assembly the connectivity is near E/I balanced: `(E-I)/(E+I)` = -0.05 (both), -0.05
(none), -0.16 (frm). ⚠️ NOT push-pull between opposite states: 81% of a `both` unit's INHIBITION
also comes from its own assembly (neg->same 0.813; neg->opposite only 0.144). Each assembly is a
self-contained E/I module — the natural motif for holding one bit.

### 5. CAUSAL: assembly share follows the active penalty, reversibly

Switch nets (N=2000, k=3, 3 seeds), cross-sectional parents frm 0.284 / both 0.665:

| arm | switch | end (3 seeds) | mean | predicted |
|-----|--------|---------------|------|-----------|
| A1 | frm -> frm+rws | 0.636 0.642 0.635 | **0.638** | rises -> hit |
| A2 | frm -> frm | 0.284 0.300 0.287 | 0.290 | stays -> hit |
| A3 | both -> frm | 0.297 0.270 0.285 | **0.284** | falls -> hit |
| A4 | both -> frm+rws | 0.635 0.685 0.688 | 0.669 | stays -> hit |

Four for four, seed spread < 0.03, endpoints on the cross-sectional values. Assembly structure is a
state variable set by whichever penalty is active — the same causal signature occupancy showed.

### 6. Occupancy is inherited from the task, and this is why it is uniform

A unit inside a bit-state assembly is ON exactly when that state holds. The task's per-state duty
(fraction of samples with a given bit at +1) is 0.36 at every k. Modal temporal PR/n:

| k | task duty | both mode | frm mode |
|---|-----------|-----------|----------|
| 1 | 0.355 | **0.377** | 0.163 |
| 3 | 0.362 | **0.350** | 0.030 |
| 5 | 0.360 | **0.363** | 0.023 |
| 8 | 0.361 | **0.357** | 0.017 |

Under rws the modal occupancy equals the task duty to within 0.02 at every k; every unit that
gained rws in the switch converged to it (A1 mode 0.350 from a parent mode near 0). The modal frm
unit at k >= 3 is near-dead. **The temporal-PR uniformity, the dead-unit rescue, and the R^2/Hoyer
results are all the same fact: units in assemblies inherit the task's statistics; units outside
them have parameter-set, heterogeneous occupancy.**

### 7. The composition argument — the answer to "why do the penalties make sense together"

frm demands every unit reach the activity cap. With 2000 units and 2k = 6 states to encode, the
cheapest way to keep everyone active is diffuse input from the whole population: it satisfies frm
and destroys selectivity (frm's assembly share near chance, R^2 0.84, Hoyer 0.75, modal occupancy
near 0). The in-degree cap forbids that route — a unit with ~20 effective inputs cannot average
over the population — so the only way to satisfy BOTH penalties is for every unit to join an
assembly. **frm forces activity; rws forbids the cheap way to get it; the network takes the
expensive way, which is the modular one the task wanted anyway.** Neither alone: frm alone gives
active-but-diffuse, rws alone gives modular-but-mostly-dead.

### 8. ⚠️ REFUTED: the training-stability mechanism

Argued: a gradient step shifts unit i's drive by `eta * delta_i * sum_j r_j ~ eta N m`, growing with
N under frm, while rws pins the small weights so only K respond (`~ eta K`, N-independent). Tested
as the RMS change in per-unit participation between consecutive 100-iteration stores, last 20% of
training, median over runs: **frm ~ N^+0.46, both ~ N^+0.59 — indistinguishable.** The mechanism
is wrong.

What survives, weakly: the DIRECT loss-spike fraction (probes > 10x own median) at N=2000 is 0.0036
under frm vs 0.0007 under both on the median run (5x), and frm's grows 12x from N=500->2000 against
both's 3.5x. `both` has its own outlier runs (N=1000 mean 0.0094 on a median of 0.0003). Real,
modest, and **no mechanism for it is established.** Do not claim one.

### Method notes

* `c_i` and `S_i` from `W_rec` rows; assembly state from argmax over the 2k rectified regression
  loadings (R^2 >= 0.15, live); assembly share = same-state fraction of the row's |W| mass; chance =
  fraction of tuned units in that state, computed per net.
* ⚠️ Use the FULL row. Top-20 |W| is the whole row under both (S~20) but a sliver under frm
  (S~835); the top-20 version reads frm at 0.350 where the full row gives 0.284.
* Intermediate weights are not saved, so the assembly share is known at switch endpoints only, not
  along the trajectory.

## ▶ THE MECHANISM IN SEVEN IDEAS — pedagogical summary, 2026-09-10 15:17

> ⚠️ **Superseded on the mechanism (2026-09-11).** This entry argues from the ReLU scale symmetry. That explanation was retracted after the bounded-sigmoid result (entry of 2026-09-11 08:50 and the retraction note below it): the cause is the objective's missing term, not the activation's homogeneity. The measurements here (in-degree S, assembly share, switch arms, occupancy) stand; the symmetry reasoning does not. For the paper's mechanism section use the 2026-09-11 entry only.


The theory entry above (14:31) records the analysis in the order it was done, refutations included.
This entry records it in the order it should be UNDERSTOOD. Each idea was tested; each rests on the
one before.

**1. A ReLU network has a hidden freedom.** relu(a·x) = a·relu(x), so scaling a unit's incoming
weights and bias by a and its outgoing weights by 1/a leaves the input-output map unchanged. The
task loss is identical for every a. Gradient descent gets no signal about how loud a unit should be;
loudness drifts, and a unit that drifts to zero has zero gradient and never returns. Weight decay
formally selects an a but at 1e-6 it barely acts, and where it acts it selects by SHRINKING - it
kills weak units, it does not save them. **frm pins the loudness above zero.** It is the only
intervention in paper.md §3 that touches the symmetry, which is why it is the only one that works.

**2. rws measures how many units a unit listens to.** S = (Σ|W_j|)²/ΣW_j² per row is the effective
in-degree: 20 equal inputs → 20; 2000 equal inputs → 2000; one dominant input → 1. rws penalises
(S-20)². Measured: frm alone S ≈ 740-835 (every unit listens to nearly everyone); with rws S ≈ 20.

**3. The task has a natural modular solution.** k bits in 2 states = 2k things to remember; the
natural recurrent memory is an assembly per state. Assign each unit to a state (argmax of its 2k
rectified loadings, or the soft tuning-overlap version - both give the same ordering) and measure
the fraction of its incoming |W| mass from same-state units, against chance.

**4. The unpenalised network already builds assemblies; frm is the one condition that destroys
them.** Assembly share, N=2000, hard/soft as multiples of chance:
    k=3:  none 3.1/2.1   rws 3.7/2.4   frm 1.7/1.3   both 4.0/2.8
    k=8:  none 5.1/2.0   rws 8.8/3.7   frm 3.2/1.8   both 11.5/6.3
The unpenalised network organises its ~230-320 survivors into assemblies unprompted. frm recruits
5x more units but their wiring drops to near chance. rws restores it above the unpenalised level.

**5. Causally, from the switch nets** (N=2000, k=3, 3 seeds, spread < 0.03):
    A1 frm → frm+rws   0.28 → 0.64   rws added: assemblies built
    A2 frm → frm       0.28 → 0.29   control
    A3 both → frm      0.67 → 0.28   rws removed: assemblies dissolved
    A4 both → frm+rws  0.67 → 0.67   control
Reversible, seed-independent, endpoints on the cross-sectional values: a state variable set by the
active penalty.

**6. A unit in an assembly inherits its activity pattern from the task.** It is on when its state
holds. A bit is at +1 for 36% of samples regardless of k, so the modal occupancy (histogram peak of
temporal PR/n over live units, bin 0.02, mean of 3 seeds) should be 0.36:
    k=1  task 0.355  both 0.377  frm 0.163
    k=3  task 0.362  both 0.350  frm 0.030
    k=5  task 0.360  both 0.363  frm 0.023
    k=8  task 0.361  both 0.357  frm 0.017
Under rws the peak sits at the task duty to within 0.02 at every k; under frm the peak sits at
zero (the modal live frm unit is quasi-silent; its MEDIAN 0.32 is the middle of an IQR of 0.40).
⚠️ Mode, not median, is the right statistic: the distributions are bimodal and the median is pulled
by the upper population. The temporal-PR uniformity, the dead-unit rescue and the R²/Hoyer results
are all this one fact.

**7. Why the two penalties need each other.** frm demands every unit be active but says nothing
about how. With 2000 units and 6 states, the CHEAP way to keep everyone alive is to listen to
everyone - some drive always arrives from somewhere - and that is exactly what frm alone produces:
active but diffuse, S ≈ 800, share near chance, modal occupancy zero. The EXPENSIVE way is to join
an assembly. Gradient descent takes the cheap way when it is available. **rws forbids it**: with S
capped at 20 a unit cannot listen to everyone, and the only remaining way to satisfy frm is the
modular one - which is what the task wanted anyway, since the unpenalised network builds the same
assemblies among the few units it keeps.
    frm alone:  active but diffuse        S≈800, share 0.28, mode ≈ 0
    rws alone:  modular but mostly dead   share 0.63, ~15% alive
    frm + rws:  active AND modular        share 0.67, all units, mode = task duty

⚠️ Two things this does NOT establish. The mean-field moment equations that started the analysis
are scaffolding: they motivated measuring S and cancellation, and their failed prediction pointed at
assemblies, but they do not derive the result and should not be presented as doing so. And the
training-stability half (frm alone spikes more at large N) has a modest effect and NO established
mechanism - the one proposed was refuted (per-step kick ~ N^0.46 frm vs N^0.59 both).

## ▶ READOUT TEST OF THE ASSEMBLY PICTURE: W_out targets clean units, but is not sparser — 2026-09-10 15:19

Pavel's prediction: under frm the readout should target the clean, selective minority and disregard
the mixed units, so W_out should be SPARSER under frm than under both. Tested on N=2000, 3 seeds,
k=3 and k=8. The intuition is right; the predicted consequence is not.

**(B) Readout weight tracks unit cleanliness under frm, not under both.** Per live unit, Spearman
between total |W_out| and cleanliness (top loading / second loading), and the median-weight ratio
clean (>=2x) vs mixed:

    k=3   frm  rho 0.70   clean/mixed 4.1x        both  rho 0.38   clean/mixed 1.09x
    k=8   frm  rho 0.69   clean/mixed 3.6x        both  rho 0.49   clean/mixed 1.28x

Under frm the readout gives clean units ~4x the weight of mixed ones. Under both it does not
discriminate - there is nothing to discriminate between.

**(A) But W_out is NOT sparser.** Effective units read per output channel, S_out = (Σ|w|)²/Σw²,
as a fraction of live units, and Hoyer sparsity of the readout rows:

    k=3   frm  0.262  Hoyer 0.500       both  0.265  Hoyer 0.496
    k=8   frm  0.126  Hoyer 0.661       both  0.145  Hoyer 0.633

Identical. frm has ~1700 live units of which ~60% are clean - ~1000 - which is plenty to spread a
readout over. **The readout is selective about WHICH units it reads, not HOW MANY.** Sparsity was
the wrong consequence to look for; the targeting ratio is where the effect lives.

**(C) The readout is aimed more precisely under both.** Fraction of |W_out[c]| on units whose
assembly is bit c (chance 1/k):
    k=3   frm 0.799 (2.4x)     both 0.909 (2.7x)
    k=8   frm 0.769 (6.2x)     both 0.875 (7.0x)

⚠️ NOTED, NOT PURSUED: within the on-bit weight, the sign-correct share (positive on '+' units,
negative on '-' units) is LOWER under both (0.708 / 0.646) than frm (0.776 / 0.724) - the wrong
direction for the picture. Small, secondary, and possibly an argmax-assignment artefact (both's
assemblies are E/I-balanced internally, so a unit's sign category may be ambiguous). Revisit with
the soft assignment if it ever matters.

## ▶ TASK-FREE CONFIRMATION OF THE ASSEMBLY PICTURE — 2026-09-10 15:36

Pavel's objection: the assembly result assigns units to states by regressing on the bit signals, so a
referee will say the modules are an artefact of the labels. Answered with three methods that never
see a bit signal. N=2000, k=3, 3 seeds, live units only.

### What works: modularity and wiring–activity agreement, with the cluster count matched

Spectral clustering with n = 2k = 6 fixed for every condition (the NUMBER of task states is a
design fact; no label is used to form any partition), on |W_rec| and separately on the activity
correlation matrix:

| pen | Q of wiring partition | ARI wiring~activity | ARI wiring~task (check) | ARI activity~task (check) |
|-----|----------------------|---------------------|-------------------------|---------------------------|
| none | 0.39 | 0.57 | 0.55 | 0.52 |
| rws | 0.54 | 0.65 | 0.55 | 0.50 |
| **frm** | **0.19** | **0.30** | **0.24** | 0.41 |
| **both** | 0.49 | 0.56 | **0.66** | **0.70** |

Two statements now hold with no labels anywhere: **frm's wiring is half as modular as any other
condition**, and **frm is the only condition where the communities found in the wiring disagree
with the communities found in the activity** (ARI 0.30 against 0.56–0.65). The check column then
vindicates the label-based analysis: the label-free wiring partition recovers the task-defined
assemblies BEST under both (0.66) and WORST under frm (0.24). Louvain at default resolution gives
the same ordering (Q = 0.53 / 0.65 / 0.26 / 0.57).

⚠️ MATCH THE CLUSTER COUNT. Unmatched Louvain (4–5 communities, differing by condition) depressed
every ARI unequally and made none-vs-both unreadable. Fixing n = 2k was what made the table honest.

### What does not work, honestly

**Louvain does not discover 6 on its own.** Resolution sweep, communities found:
    none  3 / 4 / 4.7 / 5.3 / 17      rws  4 / 4 / 4 / 5.3 / 13
    frm   1 / 4 / 28 / 98 / 296       both 3 / 5 / 11 / 16 / 35      (res 0.5 / 1 / 1.5 / 2 / 3)
none/rws/both plateau at 3–5; frm's count explodes with resolution, the signature of NO intrinsic
scale. That is a qualitative statement about frm, not a recovery of the number 6. Do not claim
"the assemblies emerge unprompted" from Louvain.

**Eigenvalue outliers do NOT count assemblies.** Outliers beyond a bulk edge set by shuffling W
(within-row shuffle: preserves each unit's in-degree and norm, destroys block alignment):
    none 19    rws 20    frm 21    both 94
Nowhere near 2k = 6, and both has the MOST. The "one outlier per block" intuition assumes each
block is net self-exciting; the assemblies here are E/I-BALANCED internally ((E-I)/(E+I) ≈ -0.05,
14:31 entry), and a balanced block has no dominant positive eigenvalue. both's 94 outliers are
genuine structure — S ≈ 20 rows with specific targets are far from any shuffle — but not structure
organised as six modes. ⚠️ RMT outlier counting is the wrong instrument for balanced assemblies.
Observed and NOT pursued: the largest real eigenvalue is 0.99 under both against 2.34 under frm.

### Net

The assembly picture survives with the labels removed: modularity, wiring–activity concordance,
and the recovery of task assemblies from wiring alone all single out frm as unstructured and both
as the most task-aligned. The two spectral/graph routes that were expected to COUNT the assemblies
do not, for reasons that are consistent with the picture rather than against it.

## ▶ WHAT ARI(W, A) MEANS, WHAT TO CALL IT, AND WHAT IT NEEDS BEFORE IT IS A RESULT — 2026-09-10 16:10

Follow-up to the task-free confirmation above, prompted by "what exactly does this tell us?"

### The meaning

ARI(W, A) high means the groups drawn from the ANATOMY (|W_rec|) are the groups drawn from the
PHYSIOLOGY (activity correlations): strongly wired units co-fluctuate, weakly wired ones don't.

Under frm it is low (0.30) for an asymmetric reason. The units ARE functionally organised - the
activity partition matches the task assemblies at 0.41, well above chance - but the recurrent
wiring does not reflect that organisation (wiring vs task 0.24; spectral clustering on a
near-uniform |W_rec| is cutting mostly noise). **Under both, the functional organisation is
implemented in the recurrent weights as blocks you can read off the matrix. Under frm the same
function exists but is implemented some other way that leaves no block structure in W_rec** -
presumably through the input weights plus a diffuse recurrent layer acting as a mixing stage rather
than as a memory. ⚠️ That last clause is a hypothesis: whether frm's selectivity is carried by W_inp
rather than W_rec is TESTABLE and has not been tested.

Why it matters: (1) cortex has like-to-like connectivity, so a network where wiring and co-activity
are unrelated is a poor model of it regardless of task performance; (2) reverse-engineering a
network from its weights requires the weights to reflect function - under frm they do not, under
both they do, which is the sharpest form of "useful as a tool" produced so far; (3) it says where
the memory lives.

### ⚠️ The word: this is STRUCTURE-FUNCTION CORRESPONDENCE, not heterogeneity

Pavel proposed reporting it as a heterogeneity measure. It is not one. frm's wiring is the OPPOSITE
of heterogeneous - it is homogeneous, diffuse, everyone connected to everyone. What is heterogeneous
under frm is the ACTIVITY (broad occupancy distribution). Modularity and heterogeneity are
different axes: a network can be homogeneous-and-non-modular (frm's wiring) or
heterogeneous-and-modular (both's activity across assemblies). Report it as "anatomical modularity
matches functional modularity under frm+rws and not under frm", and keep "heterogeneity" for the
occupancy and R² results where it is the right word.

### ⚠️ Three things it needs before the table is quotable

1. **MATCHED UNIT COUNT - NOT DONE.** Every earlier cross-condition comparison in this project had
   to be subsampled to equal n because the statistics move with n. This one was not: none ~280 live
   units, frm ~1600, both 2000. Spectral clustering and ARI on 280 vs 2000 points are not
   automatically comparable. Run it subsampled before quoting anything.
2. **Seed spread.** 3-seed means, no error bars. Bootstrap over units or add seeds. The frm-vs-rest
   gap (0.30 vs 0.56+) will survive; the ordering AMONG none/rws/both (0.56-0.65) is probably noise
   and must be reported as a tie.
3. **Cluster count.** n = 2k is principled here (the task's state count is known) but a reader will
   ask what other n give. A sweep or a stability-based choice closes it.

### Prior work — ⚠️ RECALLED, NOT VERIFIED; check every citation before use

* Neuroscience target - like-to-like connectivity: Ko et al. 2011 (Nature), similarly tuned V1
  neurons preferentially connected; Cossell et al. 2015 (Nature), synaptic strength tracks response
  correlation. That is "wiring groups = activity groups" measured in cortex - the standard frm+rws
  meets and frm does not.
* Closest ANN precedent: Hod, Casper, Filan et al. (~2021, "Detecting modularity in deep neural
  networks"; Filan et al., "clusterability") graph-cluster weights, cluster activations, and ask
  whether they agree. Recollection: standard trained nets often do NOT show aligned weight and
  activation clusters - which makes a training condition that PRODUCES alignment the novel part.
* Trained RNNs: Dubreuil, Valente, Mastrogiuseppe & Ostojic 2022 (Nat Neurosci), population
  structure in low-rank RNNs found from connectivity; Yang et al. 2019 (Nat Neurosci), functional
  clusters in multitask RNNs from task-variance profiles. Neither, as recalled, tests whether the
  wiring partition and the activity partition coincide - that is the specific addition here.

The pitch, once the matched-n table exists: cortex has this property; standard trained networks
generally don't; here is a training condition under which they do, and here is the mechanism.

## ▶ MODULARITY OF EACH PARTITION ON ITS OWN, MATCHED n — REVISES "both IS MOST MODULAR" — 2026-09-10 16:17

Pavel: "I never saw the modularity results alone." Right - the ARI table showed agreement between
two partitions without showing whether either was any good, and the matched-n gap was still open.
Both fixed here. n = 275 live units in every condition, n_clusters = 2k, 4 subsamples x 3 seeds.
Each partition scored against its OWN null (wiring: within-row permutation, which preserves every
unit's in-degree; activity: each unit's time series permuted, which destroys cross-correlation).

| pen | WIRING Q | Q_null | excess | contrast | ACTIVITY Q | Q_null | excess | contrast | silh | ARI(W,A) |
|-----|---------|--------|--------|----------|-----------|--------|--------|----------|------|----------|
| none | 0.395 | 0.069 | 0.327 | 6.5x | 0.368 | 0.030 | 0.338 | 5.5x | 0.41 | 0.61 ± 0.09 |
| rws | 0.542 | 0.146 | **0.396** | 13.5x | 0.361 | 0.030 | 0.331 | 5.5x | 0.41 | 0.62 ± 0.07 |
| frm | 0.196 | 0.058 | **0.138** | **2.1x** | 0.315 | 0.030 | 0.285 | 4.6x | 0.42 | **0.30 ± 0.08** |
| both | 0.524 | **0.321** | 0.203 | 10.8x | 0.360 | 0.030 | 0.330 | 5.4x | **0.55** | 0.48 ± 0.15 |

contrast = mean within-cluster edge / mean between-cluster edge (1 = no block structure; NOT
null-corrected, the Q-excess column is). silh = silhouette of the activity clusters at distance
1-|corr|.

### Finding 1: activity modularity is the SAME in every condition

Q excess 0.29-0.34 and contrast 4.6-5.5x across all four, frm included. **Every condition has
clearly modular activity.** The one difference is that both's activity clusters are tighter
(silhouette 0.55 vs 0.41). So "frm units are functionally disorganised" is WRONG as a statement
about activity - their activity groups as cleanly as anyone's.

### Finding 2: wiring modularity is where they differ, and frm is the outlier

Q excess 0.33 / 0.40 / **0.14** / 0.20; contrast 6.5 / 13.5 / **2.1** / 10.8. frm's wiring is
barely above its null while its activity is as modular as the others'. That is the precise form of
"function present but not implemented in the recurrent wiring": both partitions measured, each
against its own null, and only one of them collapses under frm.

### ⚠️ Correction 1: both's raw wiring Q is inflated by sparsity

both's row-shuffle null is 0.321 against 0.06-0.15 for the others. A sparse random graph (S ≈ 20
targets per row) looks modular to spectral clustering BY ITSELF - the classic modularity-of-random-
graphs effect. Null-corrected, both's wiring excess is 0.20, BETWEEN frm and none. ⚠️ Never quote
raw Q across conditions with different sparsity; the row-shuffle null is what makes it comparable.

### ⚠️ Correction 2: "both is most task-aligned" (13:xx entry) was an unmatched-n artefact — WITHDRAWN

Matched, ARI(W,A) is 0.61 / 0.62 / 0.30 / 0.48 ± 0.15. both DROPS and its spread balloons, because
subsampling 275 of 2000 units cuts each of its six assemblies to ~45 members and spectral
clustering on that is noisy. This is the known asymmetry of matched-n: fair to the statistic,
unfair to the condition with more structure per unit. Report it as: frm is the outlier (0.30);
none, rws and both are comparable within their spreads.

### What survives, stated at the honest magnitude

**Modularity is the default. frm destroys it in the wiring. rws preserves it under frm.** both is
NOT more modular than none or rws - it has the SAME modularity extended to all 2000 units instead
of 275. That was always the magnitude in the assembly-share table (none 3.1x chance, rws 3.7x, both
4.0x) and it is the right way to say it. The gain from adding rws to frm is not more structure per
unit; it is structure at all, across every unit, where frm alone has none in the wiring.

## ▶ TASK-FREE SELECTIVITY: HOW MANY POPULATION FACTORS DOES A UNIT COMBINE — 2026-09-10 17:53

**Why.** §3.1's selectivity is measured against guessed task variables (the rectified bits). Pavel
asked for the same property measured from activity alone, "with respect to factors recovered with a
more general method". Script: `flipflop_mixedsel.py [k] [N]` → `img/internal_figures/mixedsel_N{N}_k{k}.png`.

**Measure.** Rate matrix X (live units × 9,600 samples; 32 noise-free trials, no inputs, no targets).
NMF (non-negative matrix factorisation) X ≈ H F with d non-negative factor time courses F and
non-negative loadings H. Factors are scaled to unit norm, so h_ij² is the share of unit i's trace
carried by factor j, and

    n_eff_i = (Σ_j h_ij²)² / Σ_j h_ij⁴

is the effective number of factors carrying the unit: 1 = pure, 2 = two equal factors. "Pure" =
n_eff < 1.2, i.e. more than ~91% of the unit's variance on one factor. Best of six NMF starts by
reconstruction error (a bad local optimum that reads every pure unit as ~2 factors has ~6× the
error). d is swept over k, 2k, 3k, 4k; nothing in the fit uses the bits.

**Why NMF and not PCA/ICA.** PCA's factors rotate freely, so pure units read as mixtures. NMF's
non-negativity pins the factors when each factor has a pure unit (separability): a unit that is
exactly zero over a stretch can only be built from factors that are zero there. Verified: on a
synthetic pure population the 2k factors correlate 1.00 with the 2k rectified bit traces, none given.

**Calibration (fixed before looking; gates each method per k).** Synthetic populations from the real
bit traces. `pure` (relu(±b_j), random gain) must read < 1.15; `sum2` (relu(s1 b_i) + relu(s2 b_j),
equal gains, truth exactly 2, mixed in with pure anchors) must read > 1.7 on the sum2 half.
NMF: pure 1.000 / sum2 1.997 (k=3), 1.000 / 1.994 (k=8). PASS.

Three things the calibration caught before any real number was read:
- The L1 form (Σh)²/Σh² was the first statistic. It reads 2.1 for one loading of 1 plus five of 0.1
  (95% one factor) and failed the nnICA pure check at 1.55. Replaced by the variance-share form.
- The first mixed synthetic, relu(w1 b_i + w2 b_j) with random unit-norm w, read ~1.35 and "failed".
  That is its correct value (median split of a random angle is 85/15), so it is not a calibration
  target. Replaced by sum2, whose truth is 2.
- A mixed population WITHOUT pure anchors is not identifiable for NMF (it makes the mixed shapes the
  factors). Real nets always have anchors; the calibration now does too.

**Cross-check: non-negative ICA (Plumbley).** Whiten to d components keeping the mean, rotate on the
orthogonal group until every component is non-negative, best of 6 starts. No reconstruction
objective, no multiplicative updates. Passes calibration at k=3 (pure 1.045, sum2 2.08).
At k=8 it is NOT usable: on a pure synthetic in d=16 a rotation with LOWER negativity than the true
arms exists (J 0.069 vs 0.100), so no optimiser returns the arms; it reads 1.5–2.3 on pure units.
(It scraped the k=8 gate at 1.122 on one batch; ignore its k=8 columns.)

**Result, N=2000, NMF at d=2k, all live units, three seeds each.**

| | k=3 median n_eff | k=3 pure fraction | k=8 median n_eff | k=8 pure fraction |
|---|---|---|---|---|
| none | 1.11 / 1.16 / 1.14 | 0.66 / 0.52 / 0.52 | 1.26 / 1.58 / 1.34 | 0.46 / 0.32 / 0.44 |
| rws  | 1.16 / 1.13 / 1.13 | 0.57 / 0.58 / 0.55 | 1.14 / 1.11 / 1.13 | 0.56 / 0.56 / 0.54 |
| frm  | 1.54 / 1.36 / 1.09 | 0.31 / 0.41 / 0.56 | 1.52 / 1.77 / 1.30 | 0.32 / 0.28 / 0.42 |
| both | **1.005 / 1.001 / 1.001** | **0.77 / 0.88 / 0.79** | **1.03 / 1.06 / 1.09** | 0.55 / 0.64 / 0.52 |

nnICA at k=3 (median / pure fraction): none 1.03/0.71, 1.03/0.75, 1.04/0.68; rws 1.03/0.80,
1.02/0.79, 1.01/0.80; frm 1.26/0.45, 1.24/0.47, 1.21/0.49; both 1.05/0.82, 1.09/0.64, 1.07/0.63.

**What holds across methods, seeds and k.** `frm` is the mixed-selective condition: lowest pure
fraction and highest n_eff in every seed at both k under NMF, and the only condition nnICA separates
from the rest at k=3. This is the task-free version of its lowest Hoyer sparsity in §3.1.

**What holds under NMF only.** `both` is the purest: lowest median in every seed at both k, highest
pure fraction in every seed at k=3 (0.77–0.88 vs ≤ 0.66 for anyone else). At k=8 its pure fraction
(0.52–0.64) overlaps `rws` (0.54–0.56); only the median still separates them. nnICA at k=3 does not
put `both` above `none`/`rws` (its `both` reading moved 0.63–0.92 across batches in two runs, so it
is unstable on the 2000-unit nets). So "both is purest" is an NMF result with a modest margin at
k=8, not a two-method result. "frm is most mixed" is.

**Two observations from the figure.** (i) At k=3, `frm`'s n_eff histogram has a second mode at
exactly 2, and at d=k `frm` reads pure (1.00–1.02) while no one else does: a large class of frm
units loads equally on both signs of one bit, i.e. responds to a bit regardless of its sign, which
fits their switch-locked transient activity. Not seen at k=8 (d=k reads 1.7–2.3). (ii) The
sorted correlation matrices show the blocks; `both` has the cleanest 2k blocks with the negative
antipodal off-blocks. The entropy of the correlation histogram was tried as a block-crispness number
and does NOT transfer (calibrates pure 2.95 < mixed 5.22 bits, but real `both` reads highest,
4.6–4.9, because its pure units differ in temporal profile rather than being copies). No calibrated
block-crispness number is claimed; n_eff is the measure and the block matrix is its picture.

**Transferability (Pavel's XOR question).** NMF never sees the inputs, so factors need not be linear
in them. Synthetic conjunction units relu(s1 b_i + s2 b_j − 1) (active only when two bits are in a
given joint state): NMF recovers the 12 conjunction patterns (median corr 1.00 with truth) and the
conjunction units read n_eff 1.000. Caveat that must travel with the measure: "pure" is relative
to the finest patterns that have anchor units. In a population holding both arms and conjunctions,
the arm units read as mixtures of conjunctions (median 1.5) because an arm IS a non-negative sum of
conjunctions. n_eff therefore measures "how many of the population's elementary patterns does the
unit combine", with the elementary patterns set by the population, not the task. That is a
different question from Rigotti-style mixed selectivity (nonlinearity in task variables), and in
the flip-flop they coincide only because the elementary patterns are single bits. d must be chosen
by a task-agnostic criterion in general; here the sweep k..4k shows the ordering is stable.

## ▶ STORYLINE RESTRUCTURED; FOUR-AXIS CHARACTERISATION ON BOTH TASKS — 2026-09-10 19:02

**Trigger.** A peer-review pass on paper.md found it was two papers on two datasets (CDDM for
silence/saturation/distortion, flip-flop for mechanism/composition), with `rws` a villain in §2 and a
hero in §3, §1.2 and §6.4 contradicting each other on whether 1000 active units is reachable,
§3.1 contradicting itself on mixed selectivity, and the leaky-ReLU row of the "what we tried"
table undermining the dead-gradient half of the mechanism. Pavel's instructions: restructure;
run the missing analyses on CDDM from local data (no cluster); de-emphasise saturation ("saturating
or not, a lot of units are needed" — show on both tasks); define mixed selectivity via regression;
characterise none/rws/frm/both on both tasks with ONE bounded statistic (Hoyer sparsity) for
dimensionality, participation, temporal sparseness and selectivity; separate main from
supplementary; propose (not run) the missing experiments in `research_directions.md`.

**New files.** `characterize.py` (→ `characterize_matrix.png`, `data/characterize_cache.pkl`),
`docs/research_directions.md`. `docs/paper.md` rewritten from scratch (old structure in git).

### The four Hoyer axes, both tasks, N = 2000, mean of 3 seeds (full N tables in the script output)

hoyer(v) = (√d − ‖v‖₁/‖v‖₂)/(√d − 1); ‖v‖₁/‖v‖₂ = √PR, so it is PR normalised to [0,1] and flipped.

| axis (d) | CDDM none / rws / frm / both | flip-flop k=3 none / rws / frm / both |
|---|---|---|
| active fraction | 0.15 / 0.14 / 1.00 / 1.00 | 0.14 / 0.15 / 0.86 / 1.00 |
| participation (N) | 0.93 / 0.92 / 0.22 / 0.23 | 0.77 / 0.78 / 0.21 / 0.03 |
| selectivity (7 or 2k coefs) | 0.48 / 0.41 / 0.32 / 0.45 | 0.76 / 0.80 / 0.63 / 0.90 |
| temporal (S samples) | 0.73 / 0.69 / 0.71 / 0.74 | 0.41 / 0.43 / 0.53 / 0.39 |
| D_PR (raw) | 2.3 / 2.2 / 8.2 / 8.5 | 6.2 / 6.1 / 8.9 / 5.5 |
| dim Hoyer (N) | 0.988 / 0.989 / 0.958 / 0.956 | 0.966 / 0.966 / 0.955 / 0.969 |

CDDM: 150 of 450 conditions (every third) so N=10000 fits in memory; live = scale-free rule;
regression on decision-epoch mean per condition with [ctx, relu(±mot), relu(±col), relu(±choice)] +
intercept. Flip-flop: as flipflop_arms.py (time-resolved, 2k rectified regressors), live = 4e-2.

**What replicates across tasks (every seed):**
- Active count without a floor grows as ~√N: CDDM b = 0.53 (N = 100…10000), flip-flop b = 0.58
  (500…4000). N for 1000 active units: 1.3e4 and 1.4e4. rws: 2.6e4 / 1.7e4. frm/both: b ≈ 1.
  End-of-training nets (CDDM budgets differ by N) → conservative. Saturation vs power law is now
  a supplement sentence.
- Participation concentration: none/rws 0.8–0.9, frm/both ≤ 0.3. On CDDM frm ≈ both; on the
  flip-flop both (0.03) is flatter than frm (0.21).
- Selectivity: both > frm on both tasks (CDDM 0.45 vs 0.32; flip-flop 0.90 vs 0.63), and frm's
  selectivity FALLS with N on both tasks (CDDM 0.37 → 0.36 → 0.32; flip-flop 0.86 → 0.82 → 0.63 over
  N = 500 → 2000) while both's holds or rises (0.40 → 0.45; 0.84 → 0.90). At CDDM N = 5000 frm and
  both are equal (0.38 vs 0.39, sd 0.05) — the CDDM margin is modest.
- frm raises D_PR with N on both tasks (CDDM 5.0 → 10.3 over 500 → 5000; flip-flop 5.6 → 8.9 over
  500 → 2000); none/rws are flat (task-set).

**What is flip-flop-specific (now marked so in the paper):**
- Temporal sparseness: frm 0.53 vs both 0.39 on the flip-flop; on CDDM 0.71 vs 0.74 vs none 0.73 —
  no penalty effect at all. The "rws makes units sustained" result does not transfer.
- The participation floor (both 0.03 vs frm 0.21) and the dimensionality reduction under both (5.5
  < none 6.2; on CDDM both raises D_PR like frm).
- frm's live fraction falling with N (0.99 → 0.86); on CDDM it is 1.00 at every N.

**Dimensionality Hoyer with d = N is uninformative** (0.94–0.99 everywhere) because D_PR ≪ N in every
network; D_PR is the number to read. Kept in the table for completeness, said so in Methods.

### Joint vs best-single-bit R² — resolves the §3.1 contradiction

The old text said "joint and best-single-bit R² differ by 0.003, so no mixed selectivity" AND "frm is
the mixed condition (Hoyer 0.748)". Measured per condition, N = 2000, k = 3, tuned units:

| | median joint R² | median best-bit R² | median gap | q90 gap | frac gap > 0.1 |
|---|---|---|---|---|---|
| none | 0.66–0.81 | 0.64–0.81 | 0.001–0.002 | 0.006–0.051 | 0.005–0.064 |
| rws | 0.59–0.71 | 0.59–0.71 | 0.002 | 0.010–0.043 | 0.020–0.037 |
| both | 0.87–0.88 | 0.87–0.88 | 0.001 | 0.010–0.107 | 0.023–0.105 |
| **frm** | 0.60–0.79 | 0.45–0.75 | 0.006–0.044 | **0.22** | **0.24–0.38** |

Both statements were true of different things: the MEDIAN tuned unit is single-bit in every
condition, and frm has a distinct SUBPOPULATION (a quarter to a third of tuned units) that is
genuinely multi-bit. "Mixed selectivity under frm" is a subpopulation claim, and is now written that
way. Mixed selectivity is DEFINED in the paper as 1 − Hoyer of the rectified-regression coefficients.

### The restructured storyline (paper.md)

1. Most units silent; active count ~√N on both tasks → 10⁴ units for 10³ active; not spare capacity.
2. Why: the ReLU scale symmetry. Stated as the best-supported interpretation with the two direct
   tests it lacks (T1 activation without dead zone, T2 gain-normalisation control) named in the text.
   rws introduced here as the regularizer that looks like it helps and does not (threshold artifact)
   because it does not touch the symmetry.
3. The fix: frm; zero silent; no task cost; prevents rather than resurrects; rescued units tuned.
4. What frm alone produces: alive, diffuse, degrading with N (the four axes, both tasks).
5. What rws adds: the modular solution extended to every unit — four-axis table (both tasks),
   assemblies (flip-flop; CDDM = A3 in research_directions), causal switch (flip-flop).
6. Consequences for RNN-as-model; two limitations in the main text (σ_log; purity ≠ realism, and the
   dark-neuron literature).
7. Methods (Hoyer as the one statistic; rectified basis; criteria; matched performance; loss term).
8. Venue: Nat Comms if T1–T2 are run; PLOS CB with §2 as interpretation otherwise.
S1–S7 supplements; retracted-claims list extended with the four retractions of the last two days.

Main vs supplementary is now enforced by a rule written into research_directions.md: a result
enters the main text only if removing it breaks a link in the storyline.

## ▶ IS SILENCE A PROPERTY OF THE ReLU FAMILY? — softplus/leaky re-read, and a sigmoid baseline SUBMITTED — 2026-09-10 23:08

**Trigger.** Pavel: "as soon as we swap ReLU by some nonlinear positive-only function, the whole
phenomenon may disappear — didn't we test that? shifted sigmoid?" Two answers, one from the disk and
one from a new run.

### 1. What the disk already says: smooth positive activations with gradient everywhere do NOT escape

⚠️ **A bug in my first quick read, recorded so it is not repeated.** For `h`-equation networks
`RNN_numpy.get_history()` returns the STATE y, and the rate is `activation(y)`. The flip-flop
loaders apply `np.maximum(·, 0)` to the history, which is correct for ReLU only. My first pass on
the softplus/leaky Dale nets used the raw history as the rate: strongly negative (silent) units then
have large |y| and read as the MOST active, giving "0% silent" for every `h` net. Corrected by
`rate = rnn.activation(y)` for `h`, `y` for `s`; the simulation was then verified against the task
(R² 0.90–0.93 on the decision epoch vs 0.84–0.86 saved with noise).

Four axes on the June Dale N=1000 CDDM sweeps (`CDDM_ptrack_g0` = ReLU, `CDDM_fb2792_g0_softplus25`,
`CDDM_fb2792_g0_leakyrelu`; 4–5 nets per cell; live = scale-free participation rule):

| act | eq | pen | live | 1/HHI | part Hoyer | sel Hoyer | temp Hoyer | D_PR |
|---|---|---|---|---|---|---|---|---|
| relu | h | none | 0.44 | 80 | 0.741 | 0.370 | 0.578 | 2.2 |
| softplus25 | h | none | 0.45 | 86 | 0.729 | 0.384 | 0.562 | 2.4 |
| leakyrelu | h | none | 0.45 | 83 | 0.741 | 0.365 | 0.579 | 2.2 |
| relu | s | none | 0.40 | 131 | 0.659 | 0.319 | 0.530 | 2.8 |
| softplus25 | s | none | 0.42 | 137 | 0.650 | 0.334 | 0.527 | 3.0 |
| leakyrelu | s | none | 0.42 | 138 | 0.652 | 0.327 | 0.527 | 2.9 |
| relu | h | both | 1.00 | 889 | 0.059 | 0.314 | 0.598 | 5.2 |
| softplus25 | h | both | 1.00 | 881 | 0.063 | 0.340 | 0.586 | 4.9 |
| leakyrelu | h | both | 1.00 | 884 | 0.062 | 0.338 | 0.602 | 5.1 |

**Indistinguishable across activations on every axis.** Softplus (β=25) has no exact zero and
leaky-ReLU (leak 0.01) has gradient everywhere; both concentrate activity onto the same ~45% of units
with the same participation sparsity (0.73–0.74) as ReLU, and `frm+rws` redistributes it identically
(0.06). This confirms the 2026-06 record (peak-based criterion, "40–64% persists") and kills the
dead-gradient-trap half of the mechanism as the *cause*: units are not stuck at zero because the
gradient vanishes there; they are driven there. What all three share is a LINEAR POSITIVE PART with
a floor at zero — the scale symmetry is exact for ReLU and leaky-ReLU (both positively homogeneous)
and asymptotically exact for softplus. Paper §2.1 must be rewritten around the symmetry alone; the
"absorbing zero" is where the drift ends, not why it starts.

### 2. What the disk cannot say: a bounded, nonlinear positive part

Every sigmoid/tanh run on disk (`CDDM_SparsityAndFRMagnitude-sigmoid_shifted*`, 26 `CDDM_tanh_slope=*`
roots, GELU) was trained with `frm+rws` ON under Dale. No unpenalized bounded-activation network
exists. A sigmoid has no scale symmetry (not homogeneous), so if the symmetry is the cause the
concentration should weaken or vanish; if "nothing keeps a unit alive" is the cause independent of
the symmetry, it should persist as units parked at a constant output.

**Submitted (Spock job `6147881`, 6 tasks, all RUNNING at 23:08; code pinned at `a831629`):**
`slurm/SilentReLU_flipflop_sigmoid_spock.slurm` → `NBitFlipFlop_std_sigmoid/EqType=h_k=3_N={500,1000}_iters=150000/`.
3-bit flip-flop, standard RNN (no Dale, no I/O positivity, trainable bias), `h` equation,
activation `sigmoid(7.5·(x − 0.3))` (`configs/model/rnn_sigmoid_shifted_standard.yaml`; slope and
shift from the earlier sigmoid_shifted sweep so f(0) ≈ 0.10 — a resting unit is nearly off, as with
ReLU). No penalties. 3 seeds × N ∈ {500, 1000}. 150k iterations, batch 1024 fresh: ReLU costs
0.215–0.24 s/iter here, slow tail ~0.3 → ≤ 12.5 h, inside the 14 h request. Smoke-tested locally
first (N=100, 30 iters: trains, loss falls, config records the activation, npz saved).

**Read-out, fixed now.** Comparison set: the ReLU `none` ksweep cells at k=3, N=500/1000.
- Silence: (a) participation scale-free `p < 0.05·q95(p)`; (b) MODULATION criterion
  `std_t(r) < 0.05·q95(std_t(r))` — required for sigmoid, where a unit saturated at a constant output
  has q90 > 0 and passes (a) while being functionally dead; (c) the flip-flop absolute threshold
  4e-2 for reference only (meaningless for a bounded activation).
- Threshold-free: participation Hoyer sparsity and 1/HHI (effective participating units).
- The four axes (`characterize.measure`), and the participation trajectory from
  `ParticipationTrace.pkl` (fraction silent under (b) and participation Hoyer vs iteration), against
  the ReLU trajectory.
- ⚠️ `flipflop_fixedpoints.load_net` FORCES `activation_name="relu"`; the sigmoid analysis must build
  RNN_numpy from the saved `activation_args` (script `flipflop_sigmoid_silence.py`, tested on the
  smoke run).

**Decision rule, fixed before the data.** ReLU baseline at these cells: live ≈ 0.32 (N=500) and 0.21
(N=1000), participation Hoyer 0.66 / 0.73. If the sigmoid nets show participation Hoyer < 0.3 and
unmodulated fraction < 10% at both N with task R² ≥ 0.8: **the concentration is a property of the
homogeneous (ReLU-family) activation**, the paper's scope narrows to that family (which is the
neuroscience default and says so up front), and "make the positive part nonlinear and bounded" is
reported as a second, penalty-free fix. If participation Hoyer ≥ 0.5 or unmodulated ≥ 30%: the
phenomenon is general to positive activations, §2 broadens to "nothing keeps a unit alive", and the
symmetry is demoted to the ReLU-specific *form* it takes. In between: reported as graded, with the
Hoyer numbers, no headline.

### Autonomous read-out arranged — 2026-09-10 23:25

All six training tasks confirmed TRAINING at 23:10 (log check: code `a831629`, R² 0.57–0.73 within
the first 450 iterations, no errors; ETA ~6.5 h at N=500, ~9 h at N=1000, inside the 14 h limit).

- **Analysis script** `flipflop_sigmoid_silence.py` (commit `5d2f962`): sigmoid runs vs the ReLU
  ksweep baseline (k=3, N=500/1000); silence under the scale-free, modulation and absolute criteria;
  participation Hoyer and 1/HHI; the four axes with live = modulated units; trajectories from
  `ParticipationTrace.pkl`. Activation is read from each run's config (older npz files store only
  the dict keys). Tested end to end on the local smoke run + baselines; imports verified on Spock.
- **Dependent Spock job** `6147887` (`SilentReLU_flipflop_sigmoid_analysis_spock.slurm`,
  `--dependency=afterany:6147881`, CPU): runs the script once every training task has ended and
  writes the table into `~/trainRNNbrain_logs/FFsigA.6147887.out` and the figure into the Spock
  repo's `img/internal_figures/sigmoid_silence.png`. Readable in the morning without any laptop.
- **Local monitor** polls Spock's queue every 15 min; when both jobs leave the queue the results are
  rsynced to `data/trained_RNNs/NBitFlipFlop_std_sigmoid`, the analysis is rerun locally, and the
  result is logged here against the pre-registered decision rule above.
- ReLU baseline at these cells for reference (end of 500k iterations, this script): N=500 scale-free
  silent 0.73, unmodulated 0.69, participation Hoyer 0.66, 1/HHI 68, R² 0.97; N=1000: 0.81, 0.79,
  0.73, 87, 0.96. Note the ReLU nets ran 500k iterations against the sigmoid 150k; the trajectory
  panels read both at matched iteration.

⚠️ `slurm/*.slurm` is gitignored (`.gitignore:31`); every launcher in the repo was force-added.
Use `git add -f` for new launchers or the commit silently omits them (it did here, once).

### 2026-09-11 01:58 — connection to Spock lost from the laptop; jobs unaffected

The local monitor lost SSH to Spock at 01:58 (`Could not resolve hostname` — the laptop's network
or VPN dropped, and the ControlMaster socket in `~/.ssh/sockets/` is gone, so reconnecting needs
Pavel's interactive `ssh spock` with Duo). Nothing on Spock depends on the laptop: the six training
tasks and the dependent read-out job `6147887` run to completion on their own. **Morning step:** open
`ssh spock`, then (a) read `~/trainRNNbrain_logs/FFsigA.6147887.out` for the table, and (b) rsync
`~/trainRNNbrain/data/trained_RNNs/NBitFlipFlop_std_sigmoid/` to the local tree and rerun
`flipflop_sigmoid_silence.py` for the figure and the trajectory entry against the decision rule.
The local monitor now stays silent while the connection is down and reports once it is back.

## ▶ RESULT: SILENCE IS NOT A ReLU PROPERTY — bounded sigmoid networks silence the same units — 2026-09-11 08:50

Spock `6147881` finished (6/6; N=500 6.3 h, N=1000 8.8 h). Read-out job `6147887` ran; runs synced
to `data/trained_RNNs/NBitFlipFlop_std_sigmoid` and linked into `dead_ReLU_data/flipflop/`.
Figure `img/internal_figures/sigmoid_silence.png`; script `flipflop_sigmoid_silence.py`.

**Unpenalised sigmoid(7.5(x − 0.3)) vs ReLU, 3-bit flip-flop, standard RNN, 3 seeds each:**

| act | N | iters | R² | scale-free silent | unmodulated | participation Hoyer | 1/HHI | sel | temp | D_PR |
|---|---|---|---|---|---|---|---|---|---|---|
| relu | 500 | 500k | 0.97 | 0.73 ± 0.00 | 0.69 ± 0.01 | 0.66 ± 0.01 | 67 | 0.74 | 0.41 | 6.4 |
| sigmoid | 500 | 150k | 0.93* | 0.64 ± 0.04 | 0.62 ± 0.06 | 0.45 ± 0.04 | 163 | 0.67 | 0.18 | 5.5 |
| relu | 1000 | 500k | 0.96 | 0.81 ± 0.00 | 0.79 ± 0.00 | 0.73 ± 0.01 | 86 | 0.82 | 0.42 | 6.4 |
| sigmoid | 1000 | 150k | 0.97 | 0.76 ± 0.01 | 0.76 ± 0.01 | 0.54 ± 0.02 | 227 | 0.85 | 0.28 | 6.0 |

\* one N=500 seed stalled at R² 0.84 (loss 0.116; its sel 0.37 / temp 0.01 pull the N=500 means
down); the other two are at 0.97 like every other run.

**At matched iteration (150k) the silent fractions coincide**: ReLU N=1000 read from its own trace
at 150k is 0.72–0.75 vs sigmoid 0.75–0.77; N=500 0.60–0.65 vs 0.62–0.69. The trajectories in the
figure lie on top of each other from ~10³ iterations on, and the sigmoid nets are STILL silencing at
150k (N=1000: 0.63–0.73 at 37k → 0.68–0.75 at 75k → 0.75–0.77 at 150k), exactly as ReLU does.

**The silent sigmoid units are parked at the lower asymptote, not saturated high**: unmodulated
units have mean rate 0.0015–0.0026 (5th–95th percentile) and their max over all samples is < 0.1;
none has mean > 0.5. The modulated units span 0.02–0.99. So "silent" means the same thing as under
ReLU — held far below threshold — with the difference that the sigmoid unit still has a (small,
0.02) gradient there. The concentration is somewhat weaker in the threshold-free measures
(Hoyer 0.54 vs 0.73 at N=1000; 1/HHI 227 vs 86) because the floor is soft: silent units sit at
0.002 rather than at exactly 0, which raises their participation share.

**Decision (pre-registered 23:08).** Unmodulated 62–76% ≫ 30% and participation Hoyer ≥ 0.5 at
N=1000 → **the phenomenon is general to positive activations.** Together with softplus and
leaky-ReLU (entry of 23:08) that is four activations — hard floor, soft floor, leaky floor, and a
bounded sigmoid with no homogeneity at all — with the same silent fraction at the same size and
iteration. The ReLU scale symmetry is therefore NOT the cause; it is the exact form the flat
direction takes for a homogeneous activation. The general statement, which is what §2 of the paper
now says: **the task loss contains no term that keeps any particular unit's gain up. Gradient descent
with weight decay walks the units the solution does not need down to the activation's floor, and
they stay there because the loss does not care and the gradient at the floor is small (zero for
ReLU, 0.01–0.02 for the others). A bounded nonlinear positive part does not change this.**

What DOES remain open in the mechanism is only T2 (research_directions): whether weight decay is
the walker (sweep WD ∈ {0, …, 1e-3}) and whether fixing each unit's gain directly (a
gain-normalisation step, no penalty) prevents it. T1 is closed.

**Consequence for the paper's scope.** Not "silent units in ReLU RNNs" but "silent units in trained
RNNs with positive activations", which is the standard neuroscience setting. Sigmoid units are
also the one case where the standard participation criterion needs a modulation criterion beside
it — here they agreed to 0.00 because the floor is 0.002, but that is a property of this shift and
slope, not a guarantee.

### RETRACTION — the ReLU scale symmetry as the cause of silence — 2026-09-11, between 08:50 and 08:56 (exact time not recorded)

Pavel: the sigmoid result negates the symmetry explanation (`relu(a·x) = a·relu(x)`, a unit's gain as
a flat direction of the loss); strip it everywhere. Done in `paper.md` (five-sentence summary, §2.1,
§2.3, §3 title and lead, retracted-claims list) and `research_directions.md` (T1 wording; T2's
gain-normalisation control withdrawn — it was only function-preserving for homogeneous activations;
T2 is now the weight-decay sweep alone). Entries above that argued from the symmetry (2026-09-10
14:31 "scaling / mean-field account", the seven ideas, and the 23:08 entry's reading) are left as
the record of what was believed and when; this note supersedes them. What stands: the task
objective has no term that keeps any particular unit active, and the walk to the floor is
activation-general (ReLU, softplus, leaky-ReLU, bounded sigmoid). What is still asserted rather
than measured: that weight decay is the walker (T2).

## ▶ THE ACTIVATION-FUNCTION EXPERIMENT, EXPLAINED — the phenomenon is general to positive activations — 2026-09-11 08:56

Self-contained account of the experiment whose pieces are logged at 2026-09-10 23:08 and 2026-09-11
08:50–08:56, written so it can be read on its own.

**The question.** Is the silent-unit phenomenon a property of the ReLU nonlinearity, or of training
itself? Every result in the paper had been obtained with ReLU units, whose positive part is linear
and whose floor is a hard zero with zero gradient. A natural objection — Pavel's — is that swapping
in a positive-only activation with a *nonlinear, bounded* positive part could make the whole thing
disappear, and that the project had never checked. Two activation-specific explanations were in
circulation and both would predict that: the dead-gradient trap (a unit at exactly zero receives no
gradient and cannot return) and the scale symmetry (`relu(a·x) = a·relu(x)` makes a unit's gain a
flat direction of the loss, so nothing holds it up).

**What already existed on disk, once read correctly.** June 2026 CDDM sweeps at N = 1000 under
Dale with softplus (β = 25) and leaky-ReLU (leak 0.01): smooth or leaky floors, gradient everywhere,
no dead-gradient trap. A first quick read gave 0% silence for the `h` equation — a bug: for `h`
networks the stored history is the pre-activation state and the rate is `activation(state)`; with
the raw state used as the rate, strongly negative (silent) units read as the most active. Corrected
(and the simulation verified against the task, R² 0.90–0.93), softplus and leaky-ReLU are
indistinguishable from ReLU on every axis: live fraction 0.45 / 0.45 / 0.44, effective participating
units 86 / 83 / 80, participation Hoyer 0.73 / 0.74 / 0.74. So the dead-gradient trap was already
excluded. The symmetry was not: softplus is asymptotically homogeneous and leaky-ReLU exactly so.

**What had never been run.** An *unpenalized* network with a bounded, non-homogeneous positive
activation. Every sigmoid and tanh network on disk (`CDDM_SparsityAndFRMagnitude-sigmoid_shifted*`,
26 `CDDM_tanh_slope=*` roots, GELU) had been trained with `frm+rws` on, under Dale — usable for
nothing here.

**Design.** 3-bit flip-flop, standard RNN (no Dale, no I/O positivity, self-connections on,
trainable bias — `rnn_relu_standard` in every respect except the activation), `h` equation, no
penalties, activation `sigmoid(7.5·(x − 0.3))` (`configs/model/rnn_sigmoid_shifted_standard.yaml`;
slope and shift from the earlier sigmoid_shifted sweep so a unit at rest outputs ≈ 0.10, nearly off,
as with ReLU). N ∈ {500, 1000} × 3 seeds, 150k iterations, batch 1024 fresh trials. Spock array
`6147881` (launcher `slurm/SilentReLU_flipflop_sigmoid_spock.slurm`, code `a831629`), smoke-tested
locally first; a dependent CPU job `6147887` ran the read-out automatically. Comparison set: the
unpenalized ReLU ksweep networks at the same k and N (500k iterations), read both at their end and,
from their participation traces, at 150k.

**Read-out, fixed before the data.** Three silence criteria: the project's scale-free rule
`p < 0.05·q95(p)` on participation `p = std + q90`; a *modulation* criterion `std_t(r) < 0.05·q95` —
necessary for a bounded activation, because a unit saturated at a constant output has q90 > 0 and
passes the standard rule while doing nothing; and the flip-flop absolute threshold 4e-2 for
reference only. Two threshold-free measures: participation Hoyer sparsity and 1/HHI. The four Hoyer
axes with live = modulated units. Trajectories of the scale-free silent fraction and participation
Hoyer along training from `ParticipationTrace.pkl`. Decision rule: participation Hoyer < 0.3 and
unmodulated < 10% at both N → the concentration is a ReLU-family property; Hoyer ≥ 0.5 or
unmodulated ≥ 30% → general to positive activations; between → graded, no headline.
Script: `flipflop_sigmoid_silence.py` (reads the activation from each run's config, because older
npz files store only the dict's keys).

**Result.**

| act | N | iters | R² | scale-free silent | unmodulated | participation Hoyer | 1/HHI |
|---|---|---|---|---|---|---|---|
| relu | 500 | 500k | 0.97 | 0.73 | 0.69 | 0.66 | 67 |
| sigmoid | 500 | 150k | 0.93* | 0.64 ± 0.04 | 0.62 ± 0.06 | 0.45 ± 0.04 | 163 |
| relu | 1000 | 500k | 0.96 | 0.81 | 0.79 | 0.73 | 86 |
| sigmoid | 1000 | 150k | 0.97 | 0.76 ± 0.01 | 0.76 ± 0.01 | 0.54 ± 0.02 | 227 |

\* one N=500 seed stalled at R² 0.84; the other five runs reached 0.96–0.97.

At matched iteration (150k) the silent fractions coincide: ReLU 0.72–0.75 vs sigmoid 0.75–0.77 at
N = 1000, 0.60–0.65 vs 0.62–0.69 at N = 500. The trajectories overlap from ~10³ iterations on, and
the sigmoid networks are still silencing at 150k (N = 1000: 0.63–0.73 at 37k → 0.75–0.77 at 150k),
as ReLU networks do. The silent sigmoid units sit at the lower asymptote — mean rate 0.0015–0.0026,
max over all samples < 0.1, none saturated high — while modulated units span 0.02–0.99. The
threshold-free concentration is milder (Hoyer 0.54 vs 0.73 at N = 1000) only because the floor is
soft: a silent unit contributes 0.002 rather than 0 to the participation mass.

**Conclusion.** Unmodulated 62–76% ≫ 30% → by the pre-registered rule the phenomenon is **general
to positive activations**: four activations — hard floor (ReLU), soft floor (softplus), leaky floor,
and a bounded sigmoid with no homogeneity at all — silence the same fraction of units at the same
size and iteration. Neither activation-specific explanation survives. The dead-gradient trap was
excluded by softplus/leaky; the scale symmetry by the sigmoid, and it has been removed from the
paper (retraction entry above). What stands is a statement about the objective: **the task loss has
no term that keeps any particular unit active; a unit the solution does not need is walked down
to the activation's floor and left there.** `frm` works because it adds that term.

**What this experiment does not say.** tanh was not run (a signed activation, where "silent" would
mean unmodulated around zero rather than parked at a floor; not needed for the claim about positive
activations). Sigmoid was trained for 150k against ReLU's 500k, so end-of-training numbers are not
matched — the matched-iteration comparison from the traces is the one to quote. Whether weight decay
is what walks the units down is still asserted, not measured (T2: the sweep with weight decay 0).

**What changed in the paper.** §2 title and mechanism rewritten around the objective's missing
term; the five-sentence summary, §2.3, §3's lead and the retracted-claims list updated; the scope
is "trained RNNs with positive activations", not "ReLU RNNs".

**A process note, recorded because it cost two days.** The symmetry explanation was introduced as
an interpretation (2026-09-10 14:31), was then built on as if established — the seven-ideas summary,
§3.3's "frm breaks a symmetry", the composition story, T2's gain-normalization control — and was
dislodged only by Pavel asking whether a sigmoid network had ever been trained. Nothing in the
intervening work tested it; every step inherited it. The rule that should have applied: a mechanism
claim gets a falsifying test named and run before anything is built on it, and until then it is
labelled as interpretation in every place it is used, not only where it was introduced.

## ▶ W_inp: AMPLITUDE, LATE DRIFT, AND THE INITIALISATION-SCALE QUESTION — 2026-09-11 09:02

**Pavel's questions.** Under `both`, W_inp keeps moving directionally after 400k iterations (lag
exponent α > 0.5). Why? Is its amplitude growing or shrinking? Was W_inp initialised at a wrong
scale — and could that be why unpenalised units go silent?

**Measured (flip-flop, k=3, N=2000, from the drift metrics in `ParticipationTrace.pkl` and the
Best/Last parameter snapshots).**

1. **Amplitude: up 50× early, then flat.** ‖W_inp‖_F, init → 1k → 10k → 100k → 400k:
   none 1.7 → 7 → 27 → 94 → 94; both 1.7 → 8 → 23 → 68 → 73; frm 1.7 → 7 → 22 → 75 → 78; rws (150k)
   1.7 → 8 → 30 → 93. Growth over the last 100k: −0.4% to +1.1% (rws +2–3%, it ended earlier).
   ‖W_out‖ falls 2× (1.7 → 0.84–1.26); ‖W_rec‖ falls under none (53 → 40) and rises under frm/both
   (53 → 66/83). So the late W_inp motion is a ROTATION at fixed norm, not growth.
2. **Persistence of the late motion.** Over the last 100k: α_W_inp (from the lag-100 → lag-10000
   drift ratio) none 0.51–0.58, rws 0.75–0.80, frm 0.83–0.85, both 0.65–0.67; cosine between
   consecutive lag-1000 displacements none +0.66–0.69, rws +0.74–0.79, frm +0.70–0.72, both
   +0.36–0.37. α_W_rec: none 0.20–0.26 (caged), penalised 0.52–0.64. α_W_out: none 0.01, both 0.2,
   frm 0.3. The penalties keep every matrix moving directionally; unpenalised W_rec and W_out are
   essentially frozen and only W_inp still moves.
3. **How much moves, in absolute terms (Best → Last snapshots).** `both`, 343k → 400k: W_inp moves
   6% of its norm, median row norm change 4.5%, median row angle change 0.9° (W_rec 17%, W_out 22%).
   Small but persistent. `none`, 247k → 400k: W_inp 18%, W_rec 29%, W_out 29%, but the top-10% rows
   by norm carry only 11% of the W_inp displacement — the motion is in the small rows. `frm`, 100k →
   400k: W_rec moves 101% of its norm, W_out 95%, W_inp 56%, median row angle 79°: the frm network
   reorganises its entire wiring between 100k and 400k while its loss is flat. This is the
   weight-level face of frm's churn (§3.2 of the paper) and a stronger statement of it.
4. **Per-unit input rows at the end.** Row-norm at init is 0.039 (std 1/√N per entry, 3 inputs).
   none: live rows median 5.6 (q10–q90 0.03–8.3), silent rows 0.003 (0.002–0.006) — the silent
   units' input rows decayed 10× BELOW init while live rows grew 150×. rws: same (3.4 vs 0.003).
   frm: live 0.28, silent 0.15; both: 0.95 (0.06–2.5). Spearman(row norm, log participation) 0.43
   under none.

**Interpretation, labelled as such.** (i) Why the late drift persists under the penalties: the frm
and rws gradients do not vanish at the task optimum, so the balance point of task gradient + penalty
gradients + weight decay is a slow slide along near-flat directions of the combined objective, with
W_inp following W_rec. Not tested; the test is to evaluate the four gradient components on the
final weights and check that they nearly cancel while each is large (A7 below). (ii) The
initialisation scale IS far from the task's operating scale: a 10-step ±1 pulse through a 0.022
weight into a τ=10 unit produces a drive of ~0.02, below the recurrent noise (σ_rec 0.05) and the
spectral-radius-1.2 recurrence. The network spends its first 100k iterations growing W_inp 50×.
The global collapse (all units quiet within ~20 iterations, §S5) happens before W_inp has grown
(1.7 → 7 by 1k), and the survivors are exactly the units whose input rows are then amplified; the
rest never receive drive and their rows decay. Whether a correctly scaled W_inp at init would
change the silent fraction is therefore a live question and NOT answered by anything on disk: the
sigmoid, softplus, leaky and ReLU runs all share this init. Proposed as T5, and it is cheap.
(iii) One argument against init scale as the whole story: on CDDM the input is a sustained O(1)
signal integrated over 100+ steps, so the init mismatch is much smaller, yet the silent fraction is
the same (~45% at N=1000). Not decisive — a test, not an argument, settles it.

### Final W_inp scale vs N, unpenalised, k=3 (Pavel's follow-up) — 2026-09-11 09:29

| N | live | mean\|W\| at init | mean\|W\| all | mean\|W\| live rows | mean\|W\| silent rows | median live row norm | ‖W_inp‖_F |
|---|---|---|---|---|---|---|---|
| 500 | 158 | 0.036 | 0.587 | 1.84 | 0.007 | 6.4 | **93.0** |
| 1000 | 206 | 0.025 | 0.355 | 1.72 | 0.002 | 6.3 | **93.4** |
| 2000 | 282 | 0.018 | 0.213 | 1.50 | 0.003 | 5.5 | **93.7** |
| 4000 | 543 | 0.013 | 0.132 | 0.95 | 0.003 | 1.8 | **93.7** |

(3 seeds per N; N=4000 from the 100k-iteration bigN runs, the others 500k.)

**The total input weight is N-independent: ‖W_inp‖_F = 93.0–93.7 at every N (∝ N^0.00), against an
initial 1.7 at every N.** What changes with N is how that fixed total is distributed: it is spread
over more live units (158 → 543), so the per-live-unit scale falls, mean|W| over live rows
∝ N^−0.31, and over all entries ∝ N^−0.72. Silent rows sit at 0.002–0.007, an order of magnitude
below their init, at every N. The N=4000 live-row median (1.8 vs 5.5–6.4) is partly the shorter
budget and partly the larger live count; the total is already at 93.7 there.

Recorded as a measurement only. It says the task fixes the total input weight the solution needs
(55× the init at every N), and the number of units the network chooses to carry it is the quantity
the paper is about; it does not say why that number is what it is. Relevant to T5: a scaled init
should target the total, i.e. ~93/√(N·k) per entry if spread over all units, and a
spread-over-all-units init at that total is the natural control.

### Final W_inp scale vs k, unpenalised, N=2000 — 2026-09-11 09:36

| k | live | ‖W_inp‖_F | per-channel column norm | median live-row norm | max-channel share of a live row | mean\|W\| live rows | mean\|W\| silent rows |
|---|---|---|---|---|---|---|---|
| 1 | 253 | 82.6 ± 0.3 | 82.6 | 4.1 | 1.00 | 4.09 | 0.034 |
| 2 | 240 | 89.5 ± 0.1 | 63.3 | 5.4 | 1.00 | 2.30 | 0.004 |
| 3 | 282 | 93.7 ± 0.0 | 54.1 | 5.5 | 1.00 | 1.50 | 0.003 |
| 4 | 299 | 96.6 ± 0.1 | 48.3 | 5.2 | 1.00 | 1.08 | 0.001 |
| 5 | 303 | 98.9 ± 0.2 | 44.2 | 5.4 | 1.00 | 0.88 | 0.002 |
| 6 | 307 | 100.7 ± 0.3 | 41.1 | 5.2 | 1.00 | 0.73 | 0.001 |
| 7 | 319 | 102.3 ± 0.0 | 38.7 | 5.4 | 1.00 | 0.63 | 0.001 |
| 8 | 351 | 103.3 ± 0.1 | 36.5 | 3.6 | 0.97 | 0.52 | 0.001 |

Fits: ‖W_inp‖_F ∝ k^0.11; per-channel column norm ∝ k^−0.39; median live-row norm ∝ k^0.01;
mean|W| over live rows ∝ k^−0.99; live units ∝ k^0.16.

**Reading.** (i) A live unit's input row has norm ≈ 5.2–5.5 at every k from 2 to 7 and is
single-channel (max-channel share 1.00): each live unit listens to exactly one bit with weight
~5.4. That per-unit weight is the unit scale the dynamics require to flip a unit with a 10-step
pulse, and it does not depend on k. (ii) The total ‖W_inp‖_F is therefore ≈ 5.4·√(live) and moves
only as the live count does, k^0.11 (83 → 103), with seed spread ≤ 0.3 — the tightest invariant in
the project. (iii) The mean over live rows falls as 1/k only because a row has k entries and one is
non-zero; per channel, the network dedicates live/k units per bit, 253 at k=1 down to 44 at k=8:
the same "active count does not track task demand" result (§1.3) in wiring terms. (iv) At k=8 the
live-row median drops to 3.6 and the max-channel share to 0.97 — the first sign of rows carrying a
second channel. Measurement only; no mechanism claimed. For T5 this fixes the target: the natural
scaled init is per-row norm ≈ 5.4 on every unit (total 5.4·√N, i.e. ≈ 240 at N=2000), or the
task-total ≈ 93–103 spread evenly, and both are worth a point in the sweep.

## ▶ T5 SUBMITTED: IS THE SILENCE PARTLY AN INITIALISATION ARTEFACT? — W_inp initial row-norm sweep — 2026-09-11 09:45

**Question.** Every network in the project initialises W_inp at std 1/√N per entry — row norm
0.039 at N=2000 — which drives a τ=10 unit by ~0.02 per 10-step pulse, below the recurrent noise
(σ_rec = 0.05). Trained unpenalised networks grow ‖W_inp‖_F 50× in their first 100k iterations;
the units that survive are those whose rows are amplified to norm ≈ 5.4 (single-channel, at every
k), and silent units' rows decay to ≈ 0.003, ten times below init (entries of 09:02, 09:29, 09:36).
Does the silence depend on starting far below the operating scale? Nothing on disk answers this: all
four activations tested share the init.

**Design.** Unpenalised standard ReLU RNN, 3-bit flip-flop, `h` equation, N ∈ {500, 1000} × 3
seeds, 150k iterations, batch 1024 fresh. New knob `model.input_row_norm` (RNN_torch, default null =
the existing draw): every row of the drawn W_inp is rescaled to exactly the requested L2 norm s, so
EVERY unit starts with the same input drive (the alternative — one global rescale — would keep the
chi-distributed row norms of the random draw, a ~40% spread at k=3, and leave a seed for a race).
Sweep s ∈ {0.5, 2, 5, 20}; the default 0.039 is the existing ksweep and is not rerun.

| s | per-entry std (N=1000) | drive per pulse | what it tests |
|---|---|---|---|
| 0.039 (existing) | 0.022 | ~0.02 | below the noise floor |
| 0.5 | 0.29 | ~0.2 | above noise, far below the unit scale |
| 2 | 1.15 | ~0.7 | the task's total (~93) spread evenly over all units at N≈1000 |
| 5 | 2.9 | ~1.8 | the unit scale every live unit ends at (5.4), on every unit |
| 20 | 11.5 | ~7 | 4× oversupply: the network must shed input weight; weight decay 1e-6 is slow |

Parameterised by per-row norm rather than total or per-entry amplitude because the per-row norm is
what the trained networks hold invariant (5.4 at every k) while the total follows the live count.
Launcher `slurm/SilentReLU_flipflop_winp_spock.slurm` (24 tasks; decode REP=TID%3,
N_IDX=(TID/3)%2, S_IDX=TID/6). Output `NBitFlipFlop_std_winp/EqType=h_k=3_N=<N>_s=<s>_iters=150000/`.
Code `06082b1`. Smoke-tested locally (N=100, s=5, 30 iterations: config records the knob, every row
starts at norm 5, ‖W_inp‖_F = 50 = 5·√100; analysis script runs on the output). Spock array
`6154727`, all 24 tasks RUNNING at 09:44; dependent read-out job `6154728`
(`SilentReLU_flipflop_winp_analysis_spock.slurm`, CPU, afterany) writes the table to
`~/trainRNNbrain_logs/FFwinpA.6154728.out`.

**Read-out, fixed now** (`flipflop_winp_silence.py`, reusing `flipflop_sigmoid_silence.analyse`):
silence under the scale-free, modulation and 4e-2 criteria; participation Hoyer and 1/HHI; the four
axes; the silent fraction and Hoyer along training from the participation trace — the first thing
to look at is whether the early global collapse (all units quiet within ~20 iterations) still
happens when every unit is driven from the start; and the FINAL row norms of live and silent units
plus ‖W_inp‖_F per s (does the network keep, grow or shed what it was given?). Comparison: the
ksweep ReLU networks at the same k and N, read at matched iteration (150k) from their traces:
scale-free silent 0.60–0.65 (N=500), 0.72–0.75 (N=1000); participation Hoyer 0.55–0.57, 0.64–0.66.

**Decision rule.** Silent fraction < 20% at any s with task R² ≥ 0.9 → the initialisation scale is
a cause of the silence: the paper must say the phenomenon depends on starting far below the
operating scale, and a scaled input init becomes the first, penalty-free recommendation; §2
rewritten. Silent fraction within the baseline's seed spread at matched iteration at every s →
excluded, one sentence in §2.2 and the objective-level explanation stands. Intermediate (a monotone
dependence on s that does not reach 20%) → reported as a dependence, and s = 20 tells whether an
oversupplied input concentrates anyway.

### W_inp: none vs both, k=3, N ∈ {500, 1000, 2000} (Pavel's follow-up) — 2026-09-11 09:48

| metric (3 seeds) | none 500 | both 500 | none 1000 | both 1000 | none 2000 | both 2000 |
|---|---|---|---|---|---|---|
| live units | 158 | 500 | 206 | 998 | 282 | 2000 |
| ‖W_inp‖_F | 93.0 | 55.5 | 93.4 | 66.1 | 93.7 | 72.8 |
| per-channel column norm | 53.7 | 32.0 | 53.9 | 38.2 | 54.1 | 42.0 |
| mean\|W\| all entries | 0.587 | 0.729 | 0.355 | 0.586 | 0.213 | 0.389 |
| mean\|W\| live rows | 1.84 | 0.73 | 1.72 | 0.59 | 1.50 | 0.39 |
| live row norm: mean | 5.51 | 2.17 | 5.14 | 1.74 | 4.48 | 1.15 |
| live row norm: median | 6.34 | 2.17 | 6.28 | 1.65 | 5.45 | 0.93 |
| live row norm: q10 / q90 | 0.02 / 11.8 | 1.01 / 3.25 | 0.03 / 9.8 | 0.36 / 3.12 | 0.03 / 8.2 | 0.05 / 2.53 |
| Hoyer of row norms over ALL units (1 = one unit carries the input) | 0.61 | 0.13 | 0.66 | 0.17 | 0.71 | 0.30 |
| max-channel share of a live row (median) | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 0.99 |
| max \|W\| entry | 16.2 | 8.8 | 14.1 | 8.4 | 11.9 | 8.1 |

(median \|W\| over live-row entries is ~0.005–0.01 in both conditions because two of a row's three
entries are ~0: rows are single-channel everywhere. `both` has 0–2 silent units per net, so its
silent-row statistics are not meaningful and are omitted.)

**Reading (measurement only).** (i) `none` holds the total at 93 regardless of N and puts it on
158–282 units at row norm ≈ 5.4–6.3 each. (ii) `both` builds LESS total input weight (55 → 73,
rising with N) and spreads it over every unit: per-unit row norm 2.2 → 1.65 → 0.93, falling roughly
as N^−0.6, and the max entry is 8–9 against 12–16. (iii) Under `both` at N=2000 the 10th-percentile
live row is 0.05 — a tenth of the units are alive with almost no direct input, i.e. driven through
the recurrent assembly rather than by W_inp; at N=500 every unit still has direct input (q10 1.0).
(iv) Rows are single-channel in both conditions (share ≈ 1.00). So the penalties do not change WHAT a
unit listens to (one bit) but HOW the task's input weight is distributed: concentrated on a few
units at the unit scale under `none`, thinned over everyone under `both`, with the recurrent
assembly supplying the drive that W_inp no longer does per unit. Relevant to T5: `both` reaches a
per-unit input scale of ~1–2 at N ≤ 1000, which sits between the s = 0.5 and s = 2 points of the
sweep; if s = 2 alone spreads activity without penalties, the comparison to `both` at matched
per-unit input scale is the one to make.

### One logged T5 run, to watch silencing as it happens — 2026-09-11 09:57

The sweep's runs write their participation trace only on completion, so nothing about silence is
visible mid-run. Added `trainer.log_silent_every` (null = off): on that cadence the participation
probe prints the silent count under three criteria — hard `p < 1e-6`, scale-free `p < 0.05·q95(p)`,
and the flip-flop absolute `p < 4e-2` — plus q95(p). Launched one run identical to the sweep's
N=1000, k=3, s=5 cell (the per-unit input scale unpenalised networks converge to) with logging every
500 iterations: Spock `6156830`, code `9c5f72d`,
`slurm/SilentReLU_flipflop_winp_logged_spock.slurm`, output
`NBitFlipFlop_std_winp_logged/EqType=h_k=3_N=1000_s=5_iters=150000/`; watch with
`grep '[silence]' ~/trainRNNbrain_logs/FFwinpL.6156830_1.out`. Smoke-tested locally (N=100, 30
iterations, logging every 10: lines print, counts move). Reference for reading it: the default-init
ReLU N=1000 trace has the global collapse within ~20 iterations and is at scale-free silent ≈ 0.3
by 100 iterations, 0.5 by 10k, 0.72–0.75 at 150k.

### Paired logged run at s=1, same seed — 2026-09-11 10:00

Spock `6157301`, code `92837f8`: identical to `6156830` (N=1000, k=3, unpenalised, silence logged
every 500 iterations) but W_inp rows at norm **1** instead of 5, with the SAME seed (3637570379), so
the initial draw of every weight is identical up to the row rescaling of W_inp and the training
batches follow the same sequence. The two logs give a direct pair of silencing trajectories at two
input scales: `grep '[silence]'` on `FFwinpL.6156830_1.out` (s=5) and `FFwinpL.6157301_1.out` (s=1).
Launcher now takes `S_OVERRIDE` / `SEED_OVERRIDE` for further pairs.

### Third logged run: the DEFAULT init, same seed — 2026-09-11 11:03

Spock `6163424`: identical to `6156830` (s=5) and `6157301` (s=1) — N=1000, k=3, unpenalised, seed
3637570379, silence logged every 500 iterations — but with W_inp left as drawn
(`model.input_row_norm=null`, i.e. entries at std 1/√N, row norm √(3/1000) ≈ 0.055, randomly spread
across rows). This is the control the pair was missing: s=1 is 18× the default row norm and s=5 is
90×, so neither is "the default". Read the three together with `grep '[silence]'` on
`FFwinpL.{6163424,6157301,6156830}_1.out`.

Interim from the pair at 14k iterations (scale-free / absolute silent of 1000): s=5 131 / 62, s=1
139 / 72, both flat or falling since ~6k; the default-init ksweep trace at this size is ≈ 500 by
10k. s=5's q95(p) has fallen 5.0 → 1.9 as the network sheds its oversized input drive, so its
absolute count rises for that reason alone; its scale-free count is flat.

### The s=1 run lost 42% of its units in ONE loss spike — 2026-09-11 11:13

`6157301` (s=1, seed 3637570379) was at 139–145 scale-free silent units of 1000 from 10k to 12.5k
iterations, task R² 0.947. At iteration ~12,640 the training loss jumped from 0.038 to **1.63** (43×;
R² 0.11 for one probe), recovered to R² 0.93 within ~100 iterations and to 0.947 by 16.5k — and in
that event **41 units went to exact zero and the scale-free silent count went 141 → 559**. Since
then it has held at 509–535 (hard 53–55) for 4,000 iterations: the units did not come back. The
spike guard did not fire (spike_factor 1e6 catches only ~1e30× events; this was 43×), and no
rollback happened. The default-init ksweep trace at this size is ≈ 500–550 at 13–17k: **after the
spike the s=1 network sits exactly where the default-init network is, having got there by a single
avalanche instead of an early collapse plus a slow walk.**

Status of the three logged runs at this reading (largest single-iteration loss after iteration
2000, and iteration ranges where the loss exceeded 0.2 = ~5× the floor):
6156830: at iteration 17553; largest loss after iteration 2000 = 0.063312; loss>0.2 episodes (iteration ranges): none
6157301: at iteration 17167; largest loss after iteration 2000 = 1.626027; loss>0.2 episodes (iteration ranges): 12647-12666
6163424: at iteration 2966; largest loss after iteration 2000 = 0.371093; loss>0.2 episodes (iteration ranges): 2205-2216

What this adds, as measurement: silencing does not have to be gradual. A properly scaled input
removes the early collapse and holds silence at ~14% for 12k iterations, and then one transient
instability of the recurrent dynamics kills ~400 units irreversibly while the task is re-solved by
the rest. Whether the s=5 and default twins show the same event type is what their logs will say;
the 24-run sweep's end-of-training silent fractions will be a mixture of "never spiked" and
"spiked" networks unless the traces are read for the jump. **The read-out must therefore report
silence along training, not only at the end, and count the jumps.** Interpretation is deferred to
the full sweep.

## ▶ WORKING ABSTRACT AGREED; CLAIM-BY-CLAIM ELABORATION WRITTEN — 2026-09-11 11:41

The story was built with Pavel in five rounds. Framing: RNNs as model organisms for systems
neuroscience (digital circuits with known ground truth on which analysis methods are honed), not
as fits to data. Three takeaways, fixed: (1) two penalties solve the silent-unit problem;
(2) the sparsity penalty makes the networks compute with highly selective units; (3) the penalties
shape the solution along axes that really matter. Corrections made on the way: §1.3 of `paper.md`
rewritten — task complexity recruits marginally more units, criterion-dependent (c = +0.18 at
fixed compute, ≈ 0 at matched state, +0.08 under rws; PR rises with k under every criterion), and
"cube root to square root" replaces any single exponent. The two 2026-09-10 entries that argue from
the ReLU scale symmetry now carry a superseded note pointing at the 2026-09-11 result.

The abstract lives at the top of `paper.md`; copied here as the record of what was agreed:

*Motivation and problem.* Recurrent neural networks trained on cognitive tasks serve as model
organisms for systems neuroscience: digital circuits with known ground truth, on which analysis
methods are developed, tested and honed. Larger networks are trained to bring this testbed closer
to the scale of real circuits. However, a trained RNN with a non-negative activation leaves most of
its units silent: the number of active units scales between the cube root and the square root of
the total number of units, and task complexity changes it only marginally. As a result, a thousand
active units costs training a network of ten thousand. Most of the digital organism is dead weight,
and the population that remains is the small subset that training happened to keep.

*Characterization of the problem.* We show that silencing afflicts RNNs trained with any
non-negative activation function, and that it worsens with longer training and larger networks. We
tried many interventions to eliminate it, from architecture and noise to training length and the
standard regularizers. Most failed, and the majority of units persistently remained silent. The
reason they stay silent is that the task loss has no term that rewards keeping a unit active: early
in training the whole population is suppressed, only the units the solution needs recover, and
nothing in the objective ever pulls the rest back up.

*Solution.* The intervention that works is a convex penalty designed to keep each unit's
participation (a measure of how much the unit contributes to the network's activity) at a specific
level, so that every unit takes part in the computation. On both tasks (context-dependent decision
making and the k-bit flip-flop) it keeps every unit active at no cost in performance. We later found
that this penalty is easily taken advantage of: the cheapest way for a unit to stay active is to
receive weak input from the whole population, which satisfies the participation demand
superficially. Such units mix task variables and become less selective as the network grows, and on
the flip-flop task they respond only transiently, flickering in and out of silence.

*Characterization of the solution.* We characterize the trained networks along a continuum from
pure to mixed selectivity, together with their dimensionality and temporal structure. The
unpenalized network organizes the few units it keeps into highly selective ones, each following a
single task variable and wired to other units performing a similar computational role; the
participation penalty alone produces a population of mixed, diffusely connected units. A second
intervention, a penalty that encourages sparse connectivity, does nothing on its own to prevent
silence. On top of the participation penalty, however, it closes the loophole and moves the network
to the pure end of the continuum: every unit becomes selective for a single task variable, wired to
units performing the same computational role, and stays persistently active. Turning the sparsity
penalty on or off in a trained network makes the population more or less sharply selective,
reversibly. Networks with identical performance therefore differ several-fold in dimensionality,
selectivity, and whether units fire persistently or transiently.

*Takeaways.* First, two penalties solve the silent-unit problem afflicting large RNNs: the
participation penalty keeps every unit active at no cost in performance, on two tasks, and the
sparsity penalty keeps the rescued units stably active. Second, adding the sparsity penalty on the
connectivity makes the networks compute with highly selective units, each following one task
variable and connected to units that follow the same one. Third, the penalties shape the network's
solution in ways that really matter: dimensionality, selectivity and temporal structure are decided
by the training objective, and it should be chosen with that in mind.

**Elaboration (layer two of `paper.md`, "Elaboration of the abstract").** Every abstract claim
tied to script, figure and trajectory entry. What the audit found loose: the wiring statistics
(in-degree S, assembly share, wiring/activity modularity, ARI, W_out targeting — 2026-09-10
14:31 onward) and the prevention-vs-resurrection statistic (96 per 1,000 recover unpenalized,
0.6 under frm) have **no script in the repo**; the code was in a session scratchpad and the
surviving scratchpad (`84f6b90b`) does not contain it. Softplus/leaky-ReLU were run only on the
Dale-constrained CDDM network and the sigmoid only on the flip-flop; no non-ReLU activation on the
standard CDDM net. The switch experiment never measured selectivity, so "more or less sharply
selective" in the abstract is inferred from cross-sectional data. "Every unit active" is exact on
CDDM and 86–100% on the flip-flop under frm alone. "Persistently vs transiently" is flip-flop only.
Action list (scripts S1–S4, experiments E1–E3, A3/A5/A6, T5 pending) is at the end of the
elaboration section.

### Figure for abstract claim P1 — 2026-09-11 11:52

`fig_P1_silence.py` → `img/internal_figures/fig_P1_silence.png`. Three panels from cached
outputs only (characterize cache; the three CDDM `silent_units_per_condition.csv`; flip-flop
`ParticipationTrace.pkl`), no simulation. (a) Unpenalised ReLU silent fraction vs N at end of
training: CDDM 0.26 / 0.61 / 0.73 / 0.85 / 0.88 / 0.91 at N = 100 / 500 / 1000 / 2000 / 5000 /
10,000; flip-flop 0.68 / 0.79 / 0.86 / 0.86 at 500 / 1000 / 2000 / 4000 (n = 3 each). (b) CDDM
N = 1000, Dale h, no penalty: scale-free 49.0 ± 0.5 (ReLU, n=5) / 40.6 ± 1.1 (softplus, n=4) /
45.2 ± 1.2% (leaky, n=5); hard 45.8 / 0.0 / 32.5%. (c) Flip-flop ReLU vs sigmoid scale-free
silent fraction along training; at 150k: ReLU N=500 0.60–0.65, N=1000 0.72–0.75; sigmoid N=500
0.62–0.69, N=1000 0.75–0.77 — reproduces the 2026-09-11 08:50 table from the traces alone.
⚠️ Panel (a) is END-OF-TRAINING; these fractions are higher than the matched-performance numbers
in §1.1 (41–46% at N=1000 CDDM) because the drift-sweep networks trained far longer. The panel
caption says so. Linked from the elaboration's P1 entry in `paper.md`.

### Three logged twins at 30k iterations (default at 15k) — 2026-09-11 12:11

Scale-free / absolute silent of 1000 (hard zeros in brackets), seed 3637570379, N=1000, k=3:

| iteration | default (row norm 0.055) | s=1 | s=5 |
|---|---|---|---|
| 5k | 505 / 503 [59] | 155 / 106 [0] | 125 / 18 [0] |
| 10k | 623 / 606 [56] | 150 / 84 [0] | 142 / 37 [0] |
| 15k | 596 / 565 [41] | 520 / 426 [52] ← after the 12.6k spike | 133 / 65 [0] |
| 20k | — | 478 / 383 [44] | 132 / 72 [2] |
| 25k | — | 466 / 356 [18] | 147 / 79 [4] |
| 30k | — | 521 / 423 [34] | 176 / 104 [3] |

Loss > 0.2 episodes after 2k: default 2205–2216, 4056–4086, 8669–8714 (three); s=1 12647–12666
(one, the avalanche); s=5 none. Task R² 0.953 in both scaled runs at 30k.

s=5 without any spike: a slow walk, scale-free 125 → 176 over 25k iterations (~2 units per 1000),
with q95(p) still falling (2.8 → 1.6) as the oversupplied input is shed. s=1 after its spike sits at
the default's level and flickers (hard zeros 18–52). Default: early collapse + spikes → 60%.

## ▶ ONE STATEMENT, ONE FIGURE: elaboration figures rebuilt from cached outputs — 2026-09-11 12:21

Pavel: the elaboration cited too many figures per claim and the existing figures (pr_matrix,
M_vs_N, characterize_matrix) carry far more than any one claim needs. Rule adopted: one statement →
one figure, two panels (CDDM | flip-flop) where both tasks have data, no unneeded panel. New
scripts, all reading cached outputs only (no simulation):

| claim | script | figure | key numbers |
|---|---|---|---|
| P1 silent fraction vs N | `fig_P1_silence.py` | `fig_P1_silence.png` | CDDM 0.26→0.91 (N=100→10k); FF 0.68→0.86 (500→4000); end of training |
| P2 M vs N, matched read-out | `fig_P2_scaling.py` | `fig_P2_scaling.png` | CDDM b=0.36 (L*=0.0230, N≥500); FF b=0.45 at k=3, 0.40 [0.37,0.44] all-k, excess criterion |
| C1 activations | `fig_C1_activations.py` | `fig_C1_activations.png` | CDDM scale-free 0.49/0.41/0.45 (hard 0.46/0.00/0.33); FF ReLU 0.72–0.75 vs sigmoid 0.75–0.77 at 150k |
| C2 silence vs iteration | `fig_C2_training.py` | `fig_C2_training.png` | N=1000: CDDM 0.46–0.51 at 10k → 0.72–0.74 at 200k; FF 0.60 → 0.81 at 500k |
| S1 active fraction vs N | `fig_cache_axis.py active` | `fig_S1_active.png` | frm CDDM 1.00 all N; FF 0.99/0.98/0.86; both 1.00 |
| S4/CS3 selectivity vs N | `fig_cache_axis.py sel` | `fig_S4_selectivity.png` | FF frm 0.86→0.63, both 0.84→0.90; CDDM frm 0.37/0.36/0.32/0.38, both 0.40–0.45, none 0.50→0.37 |
| S5 temporal sparsity vs N | `fig_cache_axis.py temp` | `fig_S5_temporal.png` | FF frm 0.36→0.53 vs 0.37–0.45; CDDM equal to N=2000, frm 0.79 vs 0.70–0.72 at N=5000 |
| CS5 dimensionality vs N | `fig_cache_axis.py d_pr` | `fig_CS5_dimensionality.png` | CDDM none 2.1–2.4, frm 5.0→10.3, both 4.7→8.5→6.8; FF none 6.1–6.3, frm 5.6→8.9, both 5.0–5.5 |

Three things the clean figures show that the prose had glossed over, now stated in the
elaboration: (i) on CDDM frm's selectivity loss with N is small and non-monotone (0.37→0.32→0.38)
and the UNPENALISED network's selectivity falls more (0.50→0.37) — "less selective as N grows" is a
flip-flop result; on CDDM it is "lowest of the four at every N"; (ii) on CDDM `both` only matches
the unpenalised selectivity, it does not exceed it; (iii) on CDDM at N=5000 frm's temporal
sparsity is 0.79 vs 0.70–0.72 for the others — the "no transients on CDDM" null may be
size-limited. Existing single-purpose figures kept for P3 (`flipflop_decisive.png`, to be reduced),
C5 (`silent_at_threshold.png`), S2 (`noisefree_loss.png`), CS4 (`switch_summary.png`), CS5
(`population_distortion.png`). Still without any figure: C3 (interventions bar chart), C4 (early
collapse; needs the recovery script), S3/CS1/CS3 wiring (needs `wiring_structure.py`). Build list
F1–F4 / S1–S4 / E1–E3 at the end of the elaboration in `paper.md`.

### P1–P4 collapsed into one figure — 2026-09-11 12:31

Pavel: show active units (not silent), both criteria, CDDM and flip-flop separately with all k,
fit the laws with CIs on b and c, extrapolate to 1000 active units. `fig_P_active_units.py` →
`fig_P_active_units.png` (replaces `fig_P1_silence` and `fig_P2_scaling`, both deleted).
CDDM matched performance L*=0.0230, fit N≥500, seed-bootstrap CI: scale-free b=0.360 [0.342,
0.375], N(1000)=1.4e4; hard b=0.308 [0.286, 0.328], N(1000)=9.8e3. Flip-flop excess criterion,
92 runs, k=1..8, N=500..4000: scale-free b=0.404 [0.372, 0.435], c=−0.063 [−0.099, −0.026];
absolute 4e-2 b=0.373 [0.343, 0.404], c=−0.053 [−0.083, −0.021]; N(1000, k=3)=1.08e4 / 1.09e4.
⚠️ With N=4000 in the grid, c is slightly NEGATIVE and its CI excludes zero under both criteria
(8× the bits → ~12% fewer units); §1.3's "CI straddles 0" row updated to −0.06 … +0.02. The
elaboration's P1–P4 are now one block "P" with subclaims (a)–(d), all read off this figure.

### fig_P_active_units: second row at matched compute (iteration 100k) — 2026-09-11 12:37

Pavel asked for a fixed-iteration row under the matched-state row. 100k is the largest budget every
cell reaches (CDDM N=5000 and flip-flop N=4000 both trained 100k), so that is the read-out, not
150k. Results at 100k: CDDM scale-free b=0.436 [0.423, 0.448], N(1000)=1.4e4; hard b=0.424
[0.399, 0.459], N(1000)=1.1e4 (N=5000 included after tolerating one probe step past the last
trace sample, 99,900). Flip-flop, 96 runs incl. N=4000: scale-free b=0.411 [0.382, 0.444],
c=+0.076 [−0.023, +0.158], N(1000, k=3)=1.8e4; absolute b=0.354 [0.325, 0.387], c=+0.086
[−0.014, +0.169], N(1000, k=3)=2.2e4. So b is 0.31–0.44 under every read-out and criterion; the
k exponent flips sign between read-outs (−0.06 matched, +0.08 fixed compute) and at 100k with
N=4000 the fixed-compute effect is weaker than the +0.18 recorded at 150k on the three-size grid
(2026-08-25). Elaboration P block now carries both rows in one table.

### The s=5 twin avalanched too — all three twins at the default level by 50k — 2026-09-11 13:32

`6156830` (s=5, 91× the default row norm), spike-free and on a slow walk until 30k (176 scale-free
silent), jumped between 32k and 34k: **201 → 448 scale-free, hard zeros 13 → 119**, coinciding with a
MILD loss excursion at iteration 32,339 (max 0.24, ~6× the floor, versus the 43× spike that did the
same to s=1). It has held at 450–520 since (hard zeros decaying 119 → 72 as some units flicker
back). At 49.5k the three twins read (scale-free / absolute / hard): default 624 / 592 / 2;
s=1 556 / 463 / 0; s=5 515 / 454 / 62. Task R² 0.953 in all three.

Measurement summary for this seed: the input scale sets WHEN the collapse happens — at once (0.055),
at 12.6k after a large spike (18×), at 32k after a small one (91×) — not WHETHER. Every trajectory
ends near 50–60% silent by 50k. Whether any seed at any s escapes to 150k is what the sweep says
tonight; the single-seed answer to "does the init scale alleviate the problem" is: it delays it.

Sweep status at 13:31: N=500 tasks at 75–89k iterations, N=1000 at ~53k; ETA ~16:30 and ~20:30.

## ▶ ELABORATION FIGURES ROUND 2; E1 SUBMITTED; REPORTING RULE — 2026-09-11 13:39

**Rule (Pavel):** never report silent fractions; report the number of active units. Saved to
memory; fig_C1, fig_C2, fig_S1 and the S3 bar converted to counts.

**New figures (all from cached outputs or a few simulated nets, scripts committed in `0f9688d`):**
- `fig_S2_cost.py` → `fig_S2_cost.png`: task loss (noise off, task term only) of none/rws/frm/both
  vs N — CDDM via `penalty_matched.clean_loss`, flip-flop k=3 via `pr_matrix.fit_floor`.
- `fig_S3_transients.py` → `fig_S3_transients.png`: temporal PR / n per live unit, four conditions,
  N=2000 k=3, 3 seeds, 64 trials. Mode 0.514±0.111 (none, bimodal) / 0.242±0.155 (rws) / **0.025±0.000
  (frm)** / 0.369±0.016 (both; task duty 0.36); median 0.358 / 0.331 / 0.232 / 0.373; burst units
  (tPR/n < 0.05) 16% / 12% / **34%** / 4.6%; live units 282 / 302 / 1634–1728 / 2000 of 2000.
- `fig_S4_cloud.py` → `fig_S4_cloud.png`: units in the 3-bit selectivity space, one seed per
  condition: star (275 live) / star (297) / filled blob (1634) / star with all 2000.
- `fig_C1_activations.py`, `fig_C2_training.py`, `fig_S1_active.png` now in active-unit counts.

**Weight decay (Pavel remembered a sweep):** yes — 2026-07-28 horizon experiment, wd ∈ {0, 1e-6} ×
N ∈ {1000, 2000} × 3 seeds, 100k iterations (Della 11706899). Its active-unit read-out was never
recorded, and the laptop copy (`CDDM_std_g0_horizon/`) holds only TrainLosses.json + config, no
participation trace or parameters. R² 0.847–0.881 (wd=0) vs 0.860–0.883 (wd=1e-6). To close T2:
sync the traces from Della and read them; no new training needed.

**E1 submitted:** `slurm/SilentReLU_std_activations_spock.slurm`, Spock array **6163782**, 9 jobs
(softplus25 / leakyrelu / sigmoid × 3 seeds), N=1000, CDDM, standard network, drift-sweep recipe,
200k iterations, code `0f9688d`. New configs `rnn_softplus_beta25_standard.yaml`,
`rnn_leaky_relu_standard.yaml` (rnn_relu_standard with only the activation changed); all three
smoke-tested locally at N=100 / 30 iterations (config records the activation; R² 0.31 / 0.54 / 0.47
after 30 iterations, i.e. training). Output `CDDM_std_g0_activations/EqType=h_N=1000_act=<name>_iters=200000/`.
Read-out: scale-free criterion (+ modulation criterion for the sigmoid), ReLU reference =
`CDDM_std_g0_drift` N=1000 at matched iteration; then redraw fig_C1's CDDM panel.

**Elaboration edits:** C3 is now a bullet list of interventions on unpenalized networks (no
architecture-constraint language anywhere, per Pavel); C5 cites `penalty_matched.png` panels
(a)–(b) — the previously cited `silent_at_threshold.png` was the matched-performance size figure,
not the rws artifact; S1 says "nearly every" and labels the flip-flop panel k=3; S2/S3/S4 point at
the new figures; the in-degree gap moved from S3 to CS1.

### ⚠️ S2 MEASURED: the penalties DO cost task loss on the flip-flop — 2026-09-11 (after 13:39)

`fig_S2_cost.py` → `fig_S2_cost.png` (CDDM half cached in `data/fig_S2_cost_cache.pkl`).
CDDM, noise-free masked MSE of the final weights (penalty_matched.clean_loss), 3 seeds:
none 0.0085 / 0.0085 / 0.0091 / 0.0089 / 0.0085 at N=100/500/1000/2000/5000; rws 0.0084 / 0.0079 /
0.0071 / 0.0068 (N≥500); frm 0.0098 / 0.0088 / 0.0076 / 0.0053; both 0.0084 / 0.0083 / 0.0058 /
0.0058. No cost, a gain at N≥2000 — as recorded 2026-08-20.
Flip-flop k=3, fitted floor of loss_clean_train over each run's OWN budget (pr_matrix.fit_floor):
none 0.0258 / 0.0256 / 0.0257 / 0.0260 at N=500/1000/2000/4000; rws 0.0259 / 0.0260 / 0.0259 /
0.0256; **frm 0.0285 / 0.0275 / 0.0291 (n=2 at 2000); both 0.0293 / 0.0305±0.0024 / 0.0287.**
The participation penalty raises the flip-flop floor by 7–19% at every N, alone or with rws. This
is on the 400k penlong runs whose loss was shown converged on 2026-08-26, so it is NOT the 150k
artefact retracted on 2026-08-25 (night 4). In R²: ~0.953 (none) vs 0.941–0.948 (penalized), about
one point. The switch experiment's "no task cost" (0.027–0.030 in every arm) compared penalized
arms only and could not see it. **Consequence:** the abstract's "at no cost in performance" and
takeaway 1's "at no cost" are true for CDDM only; flagged in the elaboration's wording list, not
yet changed in the abstract (Pavel's call). Possible follow-up: the frm cap `0.3·log1p(100)/log1p(N)`
is a hyper-parameter (T4); a lower cap may remove the flip-flop cost — untested.

### fig_C3_interventions built; the three legacy silent-percentage figures dropped from C3 — 2026-09-11

Pavel: C3 still showed % silent (via the embedded legacy figures). `fig_C3_interventions.py` →
`fig_C3_interventions.png`, active-unit COUNTS at N=1000, CDDM, no penalty. ⚠️ First draft mixed
two criteria (participation-based tables vs count_silent_units' peak-rate tables) and read
"self-connections off" as 520 vs a 414 baseline where the record says the two differ by ~1 pp;
fixed by sourcing self-conn/bias from `silent_stats_all.csv` (participation criterion: 404 / 409 vs
414) and putting the traceless 2026-07-01 noise and activation sweeps in a SEPARATE panel on their
own peak-rate criterion with their own reference. Left: baseline 414±18; 5k/30k/200k iterations
587/414/272; self-conn off 404; bias fixed 409; metabolic λ=0.01/0.1/1/10 393/431/432/380. Right:
noise σ=0/0.01/0.05/0.1 184/521/524/540; ReLU/softplus/leaky 510/594/548. Build item F2 closed.

## ▶ ★ THE COMPOSITION RESULT IS FLIP-FLOP ONLY: on CDDM the sparsity penalty does not remove mixed selectivity or bursts — 2026-09-11 14:25

Pavel asked for count-based S4/S5 figures (the Hoyer curves "look almost the same"), mirror plots
on CDDM, N=100 excluded everywhere, example units, C3 read-out iterations, S3 axes, S4 cloud 2x2.
New script `unit_stats.py` (one simulation pass per network, cache `data/unit_stats_cache.pkl`,
93 nets: flip-flop k=3 N=500-4000 and CDDM N=500-10000, four conditions, all seeds): per network
live / burst (tPR/n < 0.05) / tuned (joint rectified R² ≥ 0.15) / mixed (joint beats best
single-variable model by > 0.1 R²) counts, signed loadings, example traces.

| N=2000, mean±sd (3 seeds) | none | rws | frm | both |
|---|---|---|---|---|
| flip-flop mixed units | 7±6 | 3±3 | **322±68** | 102±70 |
| flip-flop burst units | 46±2 | 35±4 | **591±78** | 94±4 |
| CDDM mixed units | 225±6 (of 308 live) | 188±5 | 393±6 | **829±104** |
| CDDM burst units | 80±9 | 68±8 | 811±64 | 787±61 |
| CDDM N=5000 mixed / burst | 430 / 182 | 261 / 142 | 1087 / 2618 | **2288** / 1103 |

Flip-flop: exactly the abstract's story — the participation penalty alone makes mixed and burst
units, the sparsity penalty removes both. **CDDM: every condition builds mixed-selective units
(most unpenalised survivors are mixed — the task is context-dependent), the sparsity penalty
DOUBLES the mixed count, and leaves the burst count unchanged.** CS3 ("every unit becomes
selective for a single task variable, persistently active") and takeaway 2 are flip-flop results.
Figures: `fig_S4_mixed.png`, `fig_S5_bursts.png`, `fig_S4_cloud_cddm.png` (CDDM units in
motion/colour/choice loading space: no star, a mixed core), `fig_S5_examples.png` (burst vs
sustained units, both tasks). `fig_S4_selectivity.png` / `fig_S5_temporal.png` (Hoyer) superseded
and un-embedded. fig_S3 now three panels with the burst COUNT (588±80 frm vs 93 both, N=2000).
fig_C3 carries the read-out iteration on every bar (30k except the 5k/200k bars) and the bias pair
(self-conn off, bias trainable 404 vs bias fixed 409). N=100 excluded from fig_P and the cache
figures. Abstract wording list extended (S4/CS3/S5 task scope). Not changed in the abstract itself.
Interpretation, not measured: on CDDM "single-variable selectivity" is not the task's natural
solution, so a wiring constraint has nothing to restore; the flip-flop's 2k-state assembly
structure is what rws recovers. The consequence for the paper's framing is Pavel's call.

### S4: per-unit R² and coefficient Hoyer as DISTRIBUTIONS (Pavel: "the medians look indistinguishable") — 2026-09-11 14:33

`unit_stats.py tuning` → `fig_S4_tuning.png`, N=2000, seeds pooled. Fit = each live unit on the
rectified task factors + constant (flip-flop 6 factors time-resolved; CDDM 7 factors on the
decision-epoch means). Flip-flop: median R² none 0.60 / rws 0.58 / frm 0.47 / both 0.87; Hoyer of
the 6 coefficients over tuned units 0.83 / 0.81 / 0.62 / 0.89, with `both` spiking at 1.0 and frm
spread over the whole range plus an R² spike at 0.1 (the burst units). CDDM: Hoyer 0.49 / 0.40 /
0.32 / 0.44 — frm unimodal at ~0.3, `both` BIMODAL (pure units at 0.88, very mixed at 0.15), which
reconciles "mixed count doubles" with "median barely moves". CDDM R² needs a NaN-safe median (units
with zero variance in the decision-epoch means). The Hoyer-vs-N median curves (fig_S4_selectivity)
were the medians of these distributions — flat because the distributions move in shape, not centre.
Dimensionality as actual D_PR is fig_CS5 (already embedded); pointed to from S4.
CDDM frm: only 834/2000 live units are tuned although the median R² is 0.68 — ~1000 units have zero
variance across conditions in the decision-epoch mean (the onset-burst units), so their R² is undefined.

### Twins at 60–70k: parallel slow walks, within 8% of each other — 2026-09-11 14:58

Scale-free silent of 1000: default 633 (60k); s=1 580 (60k); s=5 557 (70k). All three now climb at
the same slow rate (~2–3 units per 1000 iterations), hard zeros have decayed to 0–16 (the units sit
just above exact zero), q95(p) 1.8–2.2 in all three. No further spikes. The ordering default > s=1
> s=5 is the residue of when each collapsed; the trajectories are otherwise the same curve.

## ▶ SIDE QUEST SUBMITTED: combinatorial task demand — the flip-flop with a 2^k−1 channel read-out (`NBitFlipFlopHyper`) — 2026-09-11 15:02

**Why.** The plain k-bit flip-flop's demand is linear in k (k output channels), and the network's
recruitment barely notices it: live units ∝ k^0.16, and at matched loss M ≈ 21·N^0.42 independent
of k (the ksweep's pre-registered pathology outcome). The deflationary reply is that k bits are
cheap — 2^k states but a k-dimensional read-out, and 44 units per bit at k=8 is still plenty. Pavel's
steelman: make the demand itself combinatorial and see whether recruitment follows it, or whether
the networks fail to train at all.

**Task.** `trainRNNbrain/tasks/TaskNBitFlipFlopHyper.py`, config `configs/task/NBitFlipFlopHyper.yaml`.
Same k pulse-train inputs and bit dynamics as `TaskNBitFlipFlop` (subclass; `get_batch` and
`generate_input_target_stream` post-process the parent's target, inputs and conditions unchanged).
Target = product of the bit states over EVERY non-empty subset of bits: the k singles, the C(k,2)
pairs, …, the k-way parity — 2^k − 1 channels (3 / 15 / 63 / 255 at k = 2 / 4 / 6 / 8). Channel
order is by binary mask (channel m−1 = ∏_{j∈m} bit_j; `task.subsets[c]` gives the bit tuple), so
the singles are NOT channels 0..k−1. A bit is 0 before its first pulse, so every product containing
it is 0 there. Built by the mask recursion P[m] = P[m∖top]·bit[top], one in-place multiply per
channel: batch cost 0.16 s at k=8, B=1024 (vs 0.04 plain). Self-check (`python …Hyper.py`) verifies
every channel against an independent itertools/np.prod construction at k = 2, 3, 4, 6, 8, and that
inputs and conditions equal the parent's for the same seed. `n_outputs` must equal 2^k − 1; the
class raises otherwise and the launcher passes both (Hydra cannot compute 2^k).

⚠️ **Target variance is no longer channel-independent.** The plain task's cross-k comparability
rested on every channel having variance ≈ 0.735. Here it falls with subset size, because a product
is 0 until ALL its bits have pulsed once: at k=8, singles 0.70, pairs 0.56, 4-way 0.42, 8-way 0.29
(k=4: 0.71 / 0.58 / 0.50 / 0.44). Consequences fixed now: (i) R² must be computed PER CHANNEL against
that channel's own variance, not as 1 − MSE/0.735; (ii) the loss is the mean over all 2^k − 1
channels, so at k=8 the singles are 8/255 of it — the network is judged almost entirely on
high-order products, which is the intended demand but means the loss is not comparable in level to
the plain task's floor law.

**Grid (72 jobs, `slurm/SilentReLU_flipflop_hyper_spock.slurm`).** k ∈ {2, 4, 6, 8} × N ∈
{500, 1000, 2000} × 3 seeds × pen ∈ {none, both}; tasks 1–36 unpenalised, 37–72 `both` (rws 0.05 +
frm 0.1, the project's standard pair — Pavel's "frm + rws" read as the `both` arm, not two arms).
Standard ReLU, h equation, `trainer_ptrack_freshbatch` (B=1024, fresh trials every iteration),
150k iterations, participation stored every 100, 48 h requested (worst cell N=2000, k=8: ~0.71 s/iter
→ 30 h). Output: `NBitFlipFlopHyper_std_hyper/EqType=h_k=<k>_N=<N>_pen=<none|both>/`. Smoke-tested
locally end to end (N=50, k=3, 20 iterations, both penalties): saved W_out is 7×50, config records
`n_outputs: 7`, participation trace written.

**Pre-registered read-out (fixed before any cell finishes).** Active counts at 150k on the
scale-free criterion, read exactly as the ksweep/penalty grids were (never a 150k penalised number
against a 500k unpenalised one). Three outcomes, each a result:
1. **M(k) rises steeply** with 2^k − 1 (exponent on k well above 0.16, or clearly multiplicative in
   the channel count) → recruitment DOES track demand once demand is real; the plain flip-flop's
   flat M(k) was cheap demand, and the paper's "active count does not track task demand" must be
   narrowed to "…does not track the number of independent bits".
2. **M(k) flat again (~k^0.2) AND the task solved** (per-channel R² at every subset size comparable
   to the plain task's) → the pathology reading survives its strongest test.
3. **High-k cells fail to train** (R² collapses at k=6/8 while M stays flat) → the silence is a
   capacity ceiling the task cannot lift. Pavel's prior. Recorded as a result, not a failed run.
The `both` arm asks whether the penalty rescues whichever of 2/3 happens (more live units AND
higher R² at k=6/8 than `none` = rescue). Numbers to record per cell: live count at 150k, R² by
subset size, clean task loss trace. Analysis scripts that assume n_inputs == n_outputs
(`flipflop_*.py`, `unit_stats.py`) do NOT apply to this task unmodified — the read-out needs its
own script keyed on `task.subsets`.

Submitted 15:04: Spock jobs 6163876 (tasks 1-36, none) and 6163877 (tasks 37-72, both), code 884e4a3, all pending behind ~37 queued jobs.

## ▶ AUTONOMOUS ROUND: build list F1/F3/F4/F5 + S1/S2/S3 done; E3 and A6 launchers — 2026-09-11 15:07

Pavel: "work autonomously to address gaps in the story; address things from What to build."

**New scripts / figures (all from data on disk):**
- `fig_P3_complexity.py` → `fig_P3_complexity.png`: M/(A N^b) vs k, flip-flop unpenalised, 100k
  (c=+0.08 [−0.02,+0.16]) vs matched state (c=−0.06 [−0.10,−0.03]).
- `fig_C5_criteria.py` → `fig_C5_criteria.png`: none vs rws under each task's criteria. CDDM
  N=2000: hard 790 (rws) vs 423 (none), scale-free 279 vs 383 — the artefact; N=5000 2086/711 vs
  440/629. Flip-flop: 1e-6 counts 1995/2000 unpenalised units active (why 4e-2 exists); under
  4e-2 and scale-free rws is modestly ABOVE none (708 vs 583; 638 vs 551 at N=2000).
- `participation_recovery.py` → `fig_C4_recovery.png`: N=1000, traces every 100 iterations.
  Units silent ≥500 iterations then active at end / silent at end: CDDM none 99±4 / 728±10, frm
  2±1 / 0; flip-flop none 82±6 / 793±4, frm 784±18 / 23±16 (the flicker). The ~20-iteration
  collapse is below this resolution (every-10 traces of the 2026-07-25 sweeps are not local).
- `wiring_structure.py` → `fig_CS1_wiring.png` (cache `data/wiring_cache.pkl`), N=2000, 3 seeds,
  both tasks, roles = argmax rectified loading of tuned units, matched n=275, 3 subsamples:

  | | none | rws | frm | both |
  |---|---|---|---|---|
  | FF in-degree S median | 80 | 20 | **726** | 20 |
  | FF same-role share / chance | 3.09 | 3.62 | **1.60** | 3.77 |
  | FF wiring Q excess / activity Q excess / ARI | 0.33/0.33/0.61 | 0.39/0.33/0.67 | **0.13**/0.28/**0.35** | 0.13/0.31/0.57 |
  | CDDM in-degree S median | 236 | 20 | **577** | 20 |
  | CDDM same-role share / chance | **0.93** | 1.14 | 1.21 | **2.11** |
  | CDDM wiring Q excess / activity Q excess / ARI | 0.01/0.38/0.57 | 0.10/0.29/0.52 | 0.09/0.36/0.63 | 0.19/0.32/0.47 |

  Reproduces the 2026-09-10 scratchpad numbers on the flip-flop (3.1/3.7/1.7/4.0 shares; 0.33/0.40/
  0.14/0.20 Q excess; 0.61/0.62/0.30/0.48 ARI). NEW on CDDM: unpenalised survivors do NOT wire by
  role (0.93×, wiring Q at null); `both` creates same-role wiring (2.11×) — consistent with the
  bimodal Hoyer (a pure subpopulation). `both`'s wiring Q excess is 0.13 here vs 0.20 recorded.
- `flipflop_switch_selectivity.py` → `fig_CS4_selectivity.png` (cache
  `data/switch_selectivity_cache.pkl`): parent vs endpoint per arm, 3 paired seeds. Mixed units
  A1 319→90, A2 319→276, A3 101→385, A4 101→95; burst units 589→108, 589→643, 91→390, 91→92;
  median Hoyer 0.64→0.84, 0.64→0.71, 0.90→0.69, 0.90→0.90. **Selectivity is reversible across
  the switch** — the CS4 gap ("more or less sharply selective" was inferred) is closed.

**Launchers written (smoke-tested locally, N=100 / 30 iterations for the wd override; CDDM warm
start from a local N=2000 `both` net):** `slurm/SilentReLU_std_weightdecay_spock.slurm` (E3: wd
0/1e-5/1e-4 × 3 seeds, N=1000, drift recipe) and `slurm/SilentReLU_cddm_switch_spock.slurm` (A6:
four arms × 3 seeds from CDDM_std_g0_penalties N=2000, 50k, participation every 10). Submission
recorded below once done. Elaboration: P(c), C4, C5, S3, CS1, CS3, CS4 rewritten with the new
figures; `penalty_matched.png` un-embedded; build list replaced by a status list.
**Submitted 2026-09-11 15:08, code `aebe8d2`:** E3 weight-decay array **Spock 6163899** (9 jobs);
A6 CDDM switch array **Spock 6163900** (12 jobs; parents verified present on Spock: the three
CDDM_std_g0_penalties N=2000 `both`/`frm` folders). E1 6163782 still running.

### Two retracted-claims lines made specific: what rws stabilizes, and modularity by five instruments — 2026-09-11 16:03

Pavel: "what does it stabilize?" → membership of the active set (2.94 → 0.30 dead↔alive transitions
per unit per 250 iterations; ≥4 crossings 38.7% → 1.7%), each unit's temporal occupancy (median
tPR/n 0.157 → 0.353, IQR 0.330 → 0.179, mode at the duty cycle), and the weights (W_inp settles at
~55k with rws, never under frm); all reversible (A3). Written into the retracted-claims list.
Pavel: "can we measure modularity by multiple uncorrelated metrics?" → `wiring_structure.py` now has
five: spectral Q excess (clustering), task-partition Q excess (labels), like-to-like Spearman
(neither), same-role share (labels, per unit), ARI (two partitions). Flip-flop none/rws/frm/both:
0.33/0.40/0.15/0.16; 0.33/0.44/0.12/0.49; 0.53/0.45/0.17/0.33; 3.1/3.7/1.5/3.8; 0.63/0.64/0.32/0.55.
CDDM: 0.01/0.10/0.10/0.19; 0.00/0.08/0.04/0.14; 0.09/0.16/**0.61/0.58**; 0.9/1.1/1.2/2.1;
0.61/0.51/0.67/0.48. Robust across all five: frm is the outlier; rws ≥ none. For `both` the
label-based instruments read above none and the label-free ones below (sparsity-sensitive), so
"extends role-organized wiring to all units" is the defensible sentence, "more modular" is not.
NEW on CDDM: penalized nets wire strongly by activity correlation (like-to-like 0.6 vs 0.09
unpenalized) with almost no task-role structure. Cache recomputed (`data/wiring_cache.pkl`).

## ▶ Hyper flip-flop at ~22 h: 25/72 cells done; six k=8 N=2000 cells rescued from the 48 h limit; FIRST READ-OUT — recruitment tracks combinatorial demand — 2026-09-12 12:59

**Status.** Finished: k=2 at N=500/1000 (both arms), k=4 at N=500 (both arms) and N=1000 (none ×3,
both ×1), k=6 N=500 (none). 41 running. The k=8 cells run at 0.7–0.8 s/iter at N≤1000 but
**1.2–1.3 s/iter at N=2000** (the 0.71 estimate in the launcher was wrong: the 255-channel read-out
and its gradient are not free at N=2000), projecting 49–54 h against a 48 h request, plus one k=8
N=500 job on a slow node at 1.38 s/iter (57 h). Nothing is written before completion, so those six
would have burned 48 h each for nothing. `scontrol update TimeLimit` is denied; 120 h is rejected
even with `--qos=long`; **96 h is the largest accepted**. Cancelled tasks 30, 34, 36, 70, 71, 72
(2–9 h lost each) and resubmitted them at 96 h: jobs 6173767 (30, 34, 36) and 6173768 (70–72),
pending. Two more k=8 cells (29, 35) project at 42–43 h and were left running — under the limit
but with little margin; if killed, resubmit the same way.

**Preliminary read-out** (`flipflop_hyper_readout.py`, run on Spock): live units at the 150k
snapshot, scale-free criterion (absolute 4e-2 criterion in the script output agrees to within ~30
units everywhere), mean ± sd over seeds, against the plain flip-flop at the SAME k, N and iteration
(ksweep read backward at 150k; plain `both` from the 400k penlong grid).

| k | N | plain none | **hyper none** | plain both | hyper both | r² plain / hyper (none) |
|---|---|---|---|---|---|---|
| 2 | 500 | 178 ± 7 | **216 ± 14** (+21%) | 500 | 500 | 0.956 / 0.941 |
| 2 | 1000 | 237 ± 11 | **287 ± 11** (+21%) | — | 1000 | 0.956 / 0.941 |
| 4 | 500 | 198 ± 3 | **342 ± 5** (+73%) | 500 | 500 | 0.952 / 0.887 |
| 4 | 1000 | 281 ± 10 | **491 ± 21** (+75%) | — | 1000 (n=1) | 0.952 / 0.886 |
| 6 | 500 | 225 ± 8 | **464 ± 2** (+106%) | 500 | pending | 0.950 / 0.798 |

Plain task, N=500, k=2→6: 178 → 225 (×1.26, the k^0.16 law). Hyper task: 216 → 464 (×2.15), and
at k=6 the network is at 93% of its 500 units — the count is hitting N, not a task-set ceiling.
The gap opens with k exactly as the read-out demand does (3 → 15 → 63 channels). **This is
pre-registered outcome 1: recruitment DOES track task demand once the demand is combinatorial.** The
plain flip-flop's flat M(k) was cheap demand. Not yet a finding on the paper's claim — see caveats.

Performance: the hyper task is NOT solved to the plain task's level. r² 0.94 / 0.89 / 0.80 at
k=2/4/6 (aggregate over channels of unequal variance, so per-subset-size R² is still to be done),
falling with k; the k=8 running cells sit at r² ≈ 0.45–0.57 at 20–45k iterations (none) and
0.25–0.32 (both). So outcome 3 (capacity ceiling at high k) is live for k=8 and the `both` arm
is NOT rescuing it — at every finished cell `both` has every unit live (500/500, 1000/1000) yet a
LOWER r² than `none` (0.926 vs 0.941 at k=2; 0.841 vs 0.887 at k=4). More live units, worse task.

⚠️ Caveats fixed before more cells land: (i) matched ITERATION, not matched performance — the hyper
cells are further from their floor at 150k, and silence in this project keeps growing with
training depth, so part of the excess may be convergence depth. The k=2 cells are the control: r²
within 0.015 of the plain task, still +21% units. The k-trend (+21 → +73 → +106%) is much larger
than any depth effect seen (the plain ksweep moved +27–33% over its whole 150k→500k range).
(ii) N=500 at k=6 is ceiling-limited (464/500); the N=1000/2000 cells decide whether it is 2× or
more. (iii) k=8 is the case Pavel expected to fail; not in yet. (iv) Per-channel R² by subset size
and a proper M(k) fit wait for the full grid.

## ▶ T5 RESULT: THE INPUT INITIALISATION SCALE DELAYS THE COLLAPSE AND LOWERS THE ENDPOINT BY 5–9 pp; IT IS NOT THE CAUSE — 2026-09-12 13:09

Spock `6154727` (24/24 finished), read-out `6154728`; runs synced to
`data/trained_RNNs/NBitFlipFlop_std_winp` (+ the three logged twins in `_winp_logged`), linked into
`dead_ReLU_data/flipflop/`. Figure `img/internal_figures/winp_silence.png`; script
`flipflop_winp_silence.py` (now also reports the silent fraction at 150k for the 500k-iteration
baseline and the largest 1000-iteration jump per run).

**End of training (150k), unpenalised, 3 seeds per cell; baseline = default draw (row norm 0.039–0.055) read at 150k from its trace:**

| s (row norm) | N | scale-free silent | unmodulated | participation Hoyer | 1/HHI | live row norm | silent row norm | ‖W_inp‖_F | R² | max 1k-jump |
|---|---|---|---|---|---|---|---|---|---|---|
| default | 500 | **0.62** (0.73 at 500k) | 0.69 | 0.66 | 67 | 6.4 | 0.004 | 93 | 0.96 | 0.38 ± 0.04 |
| 0.5 | 500 | 0.56 ± 0.04 | 0.48 | 0.54 | 119 | 5.3 | 0.005 | 93 | 0.96 | 0.25 ± 0.10 |
| 2 | 500 | 0.53 ± 0.01 | 0.45 | 0.53 | 123 | 5.0 | 0.005 | 94 | 0.97 | 0.23 ± 0.05 |
| 5 | 500 | 0.55 ± 0.01 | 0.47 | 0.54 | 119 | 5.8 | 0.005 | 94 | 0.96 | 0.21 ± 0.06 |
| 20 | 500 | 0.57 ± 0.02 | 0.51 | 0.57 | 104 | 4.2 | 0.005 | 94 | 0.96 | 0.32 ± 0.05 |
| default | 1000 | **0.74** (0.81 at 500k) | 0.79 | 0.73 | 87 | 6.3 | 0.004 | 93 | 0.96 | 0.32 ± 0.08 |
| 0.5 | 1000 | 0.70 ± 0.01 | 0.64 | 0.64 | 146 | 4.7 | 0.004 | 94 | 0.97 | 0.34 ± 0.03 |
| 2 | 1000 | 0.66 ± 0.00 | 0.61 | 0.62 | 157 | 4.5 | 0.003 | 94 | 0.97 | 0.34 ± 0.06 |
| 5 | 1000 | 0.68 ± 0.01 | 0.63 | 0.63 | 152 | 4.7 | 0.004 | 94 | 0.96 | 0.39 ± 0.04 |
| 20 | 1000 | 0.70 ± 0.01 | 0.66 | 0.66 | 130 | 3.9 | 0.004 | 95 | 0.96 | 0.39 ± 0.13 |

(The default's unmodulated / Hoyer / 1/HHI columns are end-of-500k values; only its silent
fraction was re-read at 150k.)

**Iteration of each run's largest 1000-iteration jump in silent fraction** (3 seeds): default
1.8k–3.3k (N=500), 2.2k–2.6k (N=1000); s=0.5 1k–7.6k, 2.2k–8.3k; s=2 11k–45k, 11k–13k; s=5
12.5k–75k, 17k–21k; s=20 35k–41k, 5k–69k. Jump sizes 0.11–0.53 of the population, median ≈ 0.3;
**every one of the 30 runs (24 sweep + 6 baseline) has one.** The three same-seed twins ended at
(scale-free / absolute): default 0.73 / 0.70, s=1 0.70 / 0.66, s=5 0.64 / 0.60.

**Against the decision rule.** Not "< 20% at any s" — the lowest cell is 53%. Not "within the
baseline's seed spread at matched iteration" either: the scaled inits end 5–9 pp below the default
(0.53–0.57 vs 0.62 at N=500; 0.66–0.70 vs 0.74 at N=1000; seed sd 0.00–0.04). So the intermediate
outcome: **a real but small dependence, and it is flat in s from 0.5 to 20** — once the input is
above the noise floor its scale does not matter. The initialisation scale is excluded as the cause
of the silence.

**What the sweep adds beyond the decision rule (measurements):**
1. **Silencing is avalanche-dominated.** In every run the silent fraction rises by 0.1–0.5 within
   a 1000-iteration window, coincident with a transient loss excursion (43× the floor for the s=1
   twin, 6× for the s=5 twin, the guard never fires). The input scale sets WHEN: at 2–3k for the
   default (on top of its collapse in the first ~20 iterations), later and more variably for the
   scaled inits (up to 75k). It does not set whether.
2. **The network converges to the same input weights from any start.** ‖W_inp‖_F ends at 93–95 for
   every s — from 632 at s=20, N=1000, it sheds 85% — live rows at 3.9–5.8, silent rows at 0.004
   (decayed from 20 to 0.004 in the s=20 case). The silent units' input is removed by training
   regardless of what they were given.
3. The scaled inits keep the threshold-free concentration modestly lower (participation Hoyer
   0.53–0.66 vs 0.66–0.73 at 500k; 1/HHI 104–157 vs 67–87) and the selectivity slightly higher
   (0.81–0.88 vs 0.76–0.80 at N=1000), with identical task performance.

**Paper.** One paragraph in §2.2 (init scale tested and excluded; the avalanche characterisation)
and the "what does not work" list gains a row; §2.1's "walked down" becomes "walked or dropped": the
walk exists, but most of the silencing in every run happens in one to three discrete events.

### Logged as intervention #16 in the "what we tried" table: input initialisation scale — FAILS — 2026-09-12 13:19

W_inp rows equalised at norm 0.5 / 2 / 5 / 20 at init (9–360× the default), unpenalised, N=500 and
1000, k=3, 3 seeds each: 53–70% of units silent at 150k against 62–74% for the default draw at the
same iteration; flat in the scale; task R² unchanged. Removes the collapse of the first iterations,
delays the avalanche, does not eliminate silence. Added to paper §2.2 / §S3.

### α (lag-scaling drift exponent) of W_inp: do the networks reach the diffusive regime? — 13:19

α = log10(drift_lag10000 / drift_lag100) / 2 averaged over a 50k-iteration window (0.5 = diffusive,
1 = ballistic); cos = mean cosine of consecutive lag-1000 displacements. 3 seeds per cell.

| s | N | window | α_W_inp | α_inp (100→1k) | α_inp (1k→10k) | cos_inp | α_W_rec | α_W_out |
|---|---|---|---|---|---|---|---|---|
| default | 500 | 100k–150k | 0.56 | 0.28 | 0.83 | 0.64 | 0.34 | 0.02 |
| default | 500 | 450k–500k | **0.49** | 0.30 | 0.68 | 0.62 | 0.27 | 0.01 |
| default | 1000 | 100k–150k | 0.61 | 0.39 | 0.84 | 0.72 | 0.33 | 0.02 |
| default | 1000 | 450k–500k | **0.43** | 0.18 | 0.67 | 0.64 | 0.16 | 0.00 |
| 0.5 | 500 / 1000 | 100k–150k | 0.66 / 0.68 | 0.40 / 0.50 | 0.91 / 0.87 | 0.59 / 0.63 | 0.33 / 0.31 | 0.02 |
| 2 | 500 / 1000 | 100k–150k | 0.77 / 0.78 | 0.61 / 0.64 | 0.93 / 0.92 | 0.58 / 0.60 | 0.43 / 0.35 | 0.09 / 0.04 |
| 5 | 500 / 1000 | 100k–150k | 0.76 / 0.78 | 0.62 / 0.61 | 0.89 / 0.95 | 0.57 / 0.64 | 0.39 / 0.37 | 0.05 / 0.04 |
| 20 | 500 / 1000 | 100k–150k | 0.72 / 0.72 | 0.51 / 0.51 | 0.94 / 0.93 | 0.63 / 0.68 | 0.41 / 0.37 | 0.02 |

**Reading.** (i) The default networks DO reach the diffusive regime for W_inp, but only late: α =
0.56–0.61 over 100k–150k, 0.43–0.49 over 450k–500k (at N=2000 the 2026-09-11 09:02 check gave
0.51–0.58 at 300k–400k). Pavel's recollection that they do not was right for the budgets he was
looking at; by 500k they do. (ii) At 150k the scaled-init networks are MORE persistent than the
default at the same iteration (α 0.66–0.78 vs 0.56–0.61): they collapsed later, so they are younger
in their post-collapse reorganisation. The residue is a slow directed drift at long lags (α over
1k→10k ≈ 0.9 in every condition, including the default) under short-lag jitter (0.2–0.6 over
100→1k); the displacement cosine stays +0.6–0.7 everywhere. (iii) W_rec is caged (α 0.16–0.43) and
W_out frozen (α ≈ 0) in every unpenalised network; only W_inp keeps moving, and its motion is
mostly in the small rows (Best→Last decomposition, 2026-09-11). Interpretation deferred; the
measured statement is that unpenalised W_inp motion decays toward diffusive on a ~10⁵-iteration
scale regardless of the initial scale.

### Della: the depth control for the hyper recruitment read-out — 15 unpenalised cells at 500k — 2026-09-12 14:07

All 46 remaining Spock hyper jobs are running (nothing pending since ~14:40), so Della adds no
speed there. It is used instead for the one control the recruitment result lacks: the 150k read-out
is matched in ITERATION while the hyper cells are short of their floor, so the excess could be
convergence depth. `slurm/SilentReLU_flipflop_hyper_long_della.slurm`: unpenalised hyper task,
500k iterations (the plain ksweep's budget), output `NBitFlipFlopHyper_std_hyper_long/`.
Della's H100s are closed to this account (`all` has a 10-min limit; `pli`/`pli-lc` reject the
QOS; h100 gres in `gpu` is "node configuration not available"), so it runs on A100-80GB
(`--constraint=gpu80`, gpu-long, 144 h cap). At A100 speed only the fast cells fit 500k with
margin: **job 13791591, tasks 1-6, 10-15, 19-21 = k=2,4 at N=500/1000 and k=6 at N=500**, three
seeds each, estimated start 2026-09-13 ~15:00. Those include the two most informative cells:
k=4 N=1000 (+75% units at 150k, not ceiling-limited) and k=6 N=500 (+106%, 464/500). Read-out fixed
now: at 500k, does r² reach the plain task's ~0.95, and does the live count fall back toward the
plain task's (depth artefact) or hold (demand)? k=6 at N≥1000, k=4 at N=2000 and all k=8 cells are
NOT submitted; they need a warm start from the Spock 150k checkpoints or a faster GPU.

### Hyper flip-flop at ~25 h: k=6 lands — the penalty's cost is a higher FLOOR, not slower convergence — 2026-09-12 16:08

New cells (150k, scale-free live count, mean over seeds): hyper `none` k=2 N=2000 374 ± 20 vs plain
309 (+21%, 2 seeds); **hyper `none` k=6 N=1000 722 ± 10 vs plain 301 (+140%, r² 0.807, 2 seeds)** —
72% of the network, so NOT ceiling-limited, the strongest demand-tracking number yet; hyper `both`
k=6 N=500 500/500 at r² 0.665 vs `none` 0.798 (2 seeds). Running `both` cells at k=6 N=1000/2000
sit at r² 0.70–0.72 against `none`'s 0.80–0.81 at the same iteration, so extra capacity does not
close the penalty's gap.

Noise-free task loss along training (mean over seeds), `none` / `both`:

| k, N | @10k | @50k | @150k | both/none at 150k | change 120k→150k |
|---|---|---|---|---|---|
| 2, 500 | 0.048 / 0.052 | 0.032 / 0.039 | 0.032 / 0.037 | +16% | +0.4% / −1.0% |
| 2, 1000 | 0.047 / 0.060 | 0.032 / 0.039 | 0.032 / 0.037 | +14% | +1.4% / −0.9% |
| 4, 500 | 0.092 / 0.102 | 0.059 / 0.069 | 0.056 / 0.065 | +17% | −0.1% / −0.5% |
| 4, 1000 | 0.083 / 0.106 | 0.058 / 0.068 | 0.056 / 0.064 | +15% | −0.8% / −1.7% |
| 6, 500 | 0.166 / 0.201 | 0.101 / 0.126 | 0.093 / 0.116 | +25% | +1.8% / +0.1% |

Both arms have plateaued by 100–150k (last-30k change within ±2%), and `both` is behind at EVERY
checkpoint from 10k on. So the penalty's cost on this task is a higher loss floor, 14–25% and
growing mildly with k, not slower convergence — the same sign and size as the 7–19% frm floor
cost on the plain flip-flop (2026-09-11), now visible in r² because the losses are larger.
**Nothing so far suggests the penalty helps on the complex task**: on every finished cell it
recruits every unit and solves the task worse. Unmeasured: what the penalty-recruited units DO
(subset selectivity, bursts — the S4/S5 analysis needs simulation), and a weaker λ.

### Hyper grid at ~43 h: 48/72 cells done; Della depth control already at 300–440k — r² does NOT move with depth — 2026-09-13 09:30

Spock: every k≤6 cell has landed except two seeds of k=4 N=2000 `both` (finishing now); k=8 N≤1000
and k=8 N=2000 seed 35 finish within the hour; the five resubmitted k=8 N=2000 cells and the slow
k=8 N=500 seed have 15–49 h left (Monday). The three "error" logs are the cancelled originals.
New 150k cells: hyper `none` k=4 N=2000 **672 ± 28 vs plain 368 (+83%)**, k=2 N=2000 389 ± 27 vs
309 (+26%); hyper `both` k=6 N=1000 1000/1000 at r² 0.712 vs `none` 0.807. Running k=6 N=2000:
`none` r² 0.81, `both` 0.74; k=8 N=1000 at 130k: `none` 0.61–0.68, `both` 0.40; k=8 N=500 at 145k:
`none` 0.59, `both` 0.31.

Della (job 13791591) started 2026-09-12 14:11, a day earlier than the estimate, at 0.16–0.28 s/iter
on A100-80 (FASTER than Spock's L40S at these sizes; the 1.85× A100 penalty was an N=4000 number).
k=2 N=500 at 440k: r² 0.941; k=4 N=500 at 330k: 0.887; k=4 N=1000 at 245k: 0.888 — identical to
the 150k values (0.941 / 0.887 / 0.886). **The hyper cells were AT their floor at 150k**; the
matched-iteration read-out was a matched-performance read-out after all, and the depth confound
is confined to whatever the live COUNT does between 150k and 500k (lands 2026-09-13 evening for
k=2, 2026-09-14 for k=4/6). k=6 N=500 and the last k=4 N=1000 seeds are still queued there.

## ▶ NBitFlipFlopWalsh: combinatorial demand with a ONE-channel read-out — task built, launcher ready, not submitted — 2026-09-13 14:47

**Why.** The hyper task's demand (2^k−1 products) and its read-out width grew together, so "the
network fills output weights" could not be excluded; and its high-order channels switched faster
than τ (8-way product: hold 18 steps, 39% of transitions within a pulse width, alive only 31% of the
trial), which is the likely origin of the k-set r² floor. Pavel: fix the read-out, keep the
combinatorial demand, make the expansion analytically computable, and keep it out of the weeds.

**Task** (`trainRNNbrain/tasks/TaskNBitFlipFlopWalsh.py`, `configs/task/NBitFlipFlopWalsh.yaml`).
Same k pulse-train inputs and set/reset bits as the plain flip-flop; one output
y = Σ_{|S|≤degree} c_S Π_{i∈S} b_i / √Σc_S², c_S ~ N(0,1) drawn once from `coef_seed` (shared by
every network). On ±1 states the products are the Walsh basis, so y is one random combination of
Σ_{m≤degree} C(k,m) linearly independent functions of the state: at k=8, degree 1/2/4/8 → 8/36/162/255
latent products, at k=6 full → 63. `degree` is the demand knob at fixed k with EVERYTHING else
fixed (inputs, one output, trial statistics). No shortcut: a degree-d representation captures
exactly Σ_{m≤d}C(k,m)/(2^k−1) of the variance (k=8: 0.03/0.14/0.36/0.64 for d=1..4); the random
coefficient tensor has no low-rank structure; a random function has no small circuit. Generator
changes vs the plain task: random initial state (every bit pulsed in steps 0–9), TOTAL pulse rate
`mu_total`=8 (each bit gets 8/k, so the temporal demand is matched across k), and `always_flip`
(each pulse carries the sign opposite to the bit, so every event moves y; the network still
implements a set/reset flip-flop). Measured at k=8: 7.4 changes of y per trial, hold 36 steps,
24% of changes within 10 steps of the previous, 0.1% static trials, y std 1.00, |y|>2 in 4.7%.
Figure: `img/internal_figures/walsh_trials.png`.
Self-check (`python …Walsh.py`): target against an independent itertools construction at
k=3/6/8 and k=8 degree 2; every bit pulsed at start; every pulse flips its bit; pulse timing equals
the parent's; `spectrum()` recovers the coefficients to 1e-15 when all states are visited.

**Reproducibility (Pavel's requirement).** The coefficient vector, subset list and generator
settings are saved per net as `*_TaskSpec_NBitFlipFlopWalsh.json` (run_experiment.py saves `spec()`
for any task that has it). The task RNG is seeded from the run seed (`task.seed: ${seed}`), so the
trial sequence is reproducible. The RNN's initial hidden state is a fixed zero vector saved with the
parameters. The launcher uses FIXED seeds 101–103 per rep, so the `none` and `both` arms of one rep
share initialisation and trials and differ by the penalty alone.

**Read-out** (`flipflop_walsh_eval.py`): r², live count (both criteria), and the recovered Walsh
spectrum by order — the projection of the network's output onto every product, exact via per-state
means — i.e. which latent products each network built, with no unit-level criterion; plus the
spurious coefficient energy (what the output contains that is not in the target).

**Launcher** `slurm/SilentReLU_flipflop_walsh_spock.slurm`: (k,degree) ∈ {(6,6),(8,1),(8,2),(8,4),(8,8)}
× N ∈ {500,1000,2000} × 3 seeds × {none, both} = 90 jobs, 150k, tasks 1–45 none, 46–90 both.
Pre-registered: `none` recruits up the degree ladder with one output (the hyper result was not
read-out width); `both` lights every unit — does it carry any of the missing high-order terms where
`none` is short of units (k=8 d=8 N=500), or is it noise? Smoke-tested end to end (N=50, 20 iters):
W_out 1×50, TaskSpec saved with 15 coefficients at k=4, config records seed 1 and n_outputs 1, the
eval script runs (r² 0.008 after 20 iterations, as expected). A 3000-iteration local run was
started and cancelled by Pavel as unnecessary. NOT submitted — awaiting the go-ahead.

## ▶ ★ WALSH TEST (k=6, N=1000, 30k, one seed): recruitment is DEMAND, not read-out width — the one-output task recruits like the 63-output task — 2026-09-13 18:10

Della jobs 13839248_4 (`none`) and 13839249_49 (`both`), paired seed 101, 30k iterations at 0.36 s/iter
(3 h). Folders `NBitFlipFlopWalsh_std_walsh/EqType=h_k=6_deg=6_N=1000_pen=<arm>_iters=30000/` on Della.

| k=6, N=1000, read at 30k | plain (k outputs) | hyper (63 outputs) | **Walsh `none` (1 output)** | Walsh `both` |
|---|---|---|---|---|
| live, scale-free / abs 4e-2 (training trace) | 412 / 440 | 737 / 767 | **778 / 826** | 974 / 841 |
| live, noise-free evaluation | — | — | 736 / 804 | 983 / 856 |
| r² (validation, noise on) | 0.95 | 0.81 | **0.831** | 0.796 |
| r² noise-free, 512 fresh trials | — | — | 0.873 | 0.856 |
| clean loss change 20k→30k | | | −2.9% (descending) | +1.1% (flat) |

**The one-output network recruits 778 units, the 63-output network 737, the plain task 412.** So the
hyper recruitment was the combinatorial demand, not the width of the read-out: with a single output
channel and the same 63 latent products the network lights up the same ~75% of itself. The
pre-registered read-out closes: recruitment tracks the number of latent functions the target needs.
Also new: the `none` live count RISES along training here (717 → 726 → 778 at 10k/20k/30k), the
opposite of every earlier grid, i.e. the network is still recruiting as it builds higher orders.

Recovered Walsh spectrum (fraction of the target's coefficient energy per order, `flipflop_walsh_eval.py`):

| order | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| target energy | 0.03 | 0.21 | 0.43 | 0.13 | 0.15 | 0.06 |
| `none` recovered | 0.97 | 0.96 | 0.92 | 0.90 | 0.86 | 0.88 |
| `both` recovered | 0.96 | 0.93 | 0.89 | 0.86 | 0.83 | 0.83 |

Both arms built every order (per-order R² of the recovered coefficients 0.97–1.00; spurious energy
0.01–0.015), with recovery falling gently with order; `none` is ahead at every order. So the penalty's
extra ~200 live units carry nothing the unpenalised network lacks — the same 63 products, slightly
worse. Pavel's default picture, with the penalty behind by 0.035 (noise on) / 0.017 (noise-free),
and the `both` loss already flat while `none` is still descending, so the gap should widen at 150k.
No r² flip. Under the absolute criterion `both` lost units over training (904 → 841): frm holds
units above the scale-free floor but not at working participation.

Caveats: one seed, 30k iterations, neither arm at its floor (r² per batch still rising in both).
Pavel's read before the data: "the test jobs will be enough to see whether this direction is worth
the time". The full 90-job grid (`SilentReLU_flipflop_walsh_spock.slurm`) remains unsubmitted.

### Depth control, first cells (Della, hyper k=2 N=500 at 500k): the excess survives matched depth — 2026-09-13 18:23

Live count (scale-free) along the SAME trace, mean of 3 seeds, hyper `none` (Della 500k) vs plain
`none` (ksweep 500k), k=2 N=500:

| iteration | 30k | 150k | 300k | 500k |
|---|---|---|---|---|
| hyper | 309 | 207 | 180 | 164 |
| plain | 268 | 178 | 148 | 129 |
| ratio | 1.15 | 1.16 | 1.22 | 1.27 |

Both counts keep falling with depth (the documented slow silencing), r² unchanged at 0.940 from 150k
to 500k on the hyper task. The hyper/plain excess does NOT shrink with depth — it grows slightly.
So the matched-iteration read-out at 150k was not inflated by convergence depth, at least at k=2.
k=4 N=500 and N=1000 cells finish overnight (plain references at 500k: 150 and 196), k=6 N=500 on
2026-09-15 (plain 164).

Spock status 18:21: 10 hyper jobs left, all k=8 (and one k=6 N=2000 `both`), every one within
1.2 h of finishing; the full 150k grid lands tonight.

### Hyper grid: k=8 at N≤1000 lands — the unpenalised network saturates its units at k=8 — 2026-09-13 19:59

62/72 cells. New: hyper `none` k=8 N=500 **494/500** (1 seed) at r² 0.651; k=8 N=1000 **977/1000**
(3 seeds) at r² 0.681; k=6 N=2000 1016 ± 57 (51%) at r² 0.811; `both` k=8: 500/500 at 0.308 and
1000/1000 at 0.407 (Δr² −0.34 / −0.27, twice the k=6 gap); `both` k=4/6 N=2000 landed (Δr² −0.028 /
−0.073). At k=8 the unpenalised network lights essentially every unit it has, so the recruitment
ladder reaches N itself at 255 latent products; whether N=2000 also saturates (or lands near the
1016 of k=6) is the last open cell. r² still depends on k alone, not N (0.65 / 0.68 at N=500/1000).
⚠️ Timeline correction: my "hours left" for jobs older than 24 h mis-parsed the day field; the
five k=8 N=2000 cells run at 1.1–1.6 s/iter and finish 2026-09-14 12:30 → 2026-09-15 06:30, not
tonight. k=8 N≤1000 completes by 2026-09-14 ~06:00.

## ▶ Walsh k=6 N=1000 at 100k (warm-started +70k, Della): the penalty does not catch up; the spectra converge but r² does not — 2026-09-14 10:28

Continuations 13847182_4 / 13847183_49 (seed 201 trials, Adam moments carried), folders
`…_pen=<arm>_iters=70000_warm/`. Read at 100k total:

| k=6, N=1000 | none 30k | none 100k | both 30k | both 100k |
|---|---|---|---|---|
| r² validation (noise on) | 0.831 | **0.880** | 0.796 | **0.846** |
| r² noise-free, 512 fresh trials | 0.873 | 0.907 | 0.856 | 0.884 |
| live scale-free / abs 4e-2 (noise-free eval) | 736 / 804 | 774 / 819 | 983 / 856 | 991 / 897 |
| recovered spectrum, orders 1–6 | .97 .96 .92 .90 .86 .88 | .98 .96 .94 .93 .91 .92 | .96 .93 .89 .86 .83 .83 | .97 .96 .95 .93 .92 .92 |

Both arms improved by ~0.05 and the gap is unchanged (0.034 with noise, 0.023 noise-free). The
penalised network did NOT catch up. **But its recovered spectrum did**: at 100k the two arms recover
the same fraction of every order (spurious energy 0.004 in both). The spectrum is the per-state
MEAN of the output, so the penalised network's average answer in each joint state is as accurate as
the unpenalised one's; its lower r² is within-state variability — slower settling after a flip, or
activity fluctuations — the temporal signature of frm networks (bursts, S5). Interpretation, not yet
measured: the penalty's cost on this task is temporal precision, not missing latent variables. The
unpenalised live count is stable at ~775 from 30k to 100k (not falling as on the plain task).

Also landed: hyper k=8 N=2000 `none` first seed **1609/2000 at r² 0.697** — 80%, so N=2000 is NOT
saturated at k=8 (N=500: 494/500, N=1000: 977/1000). k=8 N=500 `none` now 3 seeds: 494 ± 2, r² 0.625.
Walsh k=8 d=8 N=1000 pair (13851493/4) at 90k of 150k: `none` r² 0.86, `both` 0.81, both still
rising slowly; 0.56 s/iter on this node, done ~20:00.

### Where the penalty's r² deficit lives on the Walsh task (k=6 N=1000, 100k nets): the transitions — 2026-09-14 10:41

Pavel's hypothesis: some units' trigger conditions are absent from a batch by chance, so frm pushes
them up "when it is not their turn". Checked on the two 100k networks (noise-free, 1024 fresh trials):
- Literal version false here: every one of the 64 states appears ≥ 3492 times in a 1024-trial batch
  (k=8: 256 states, ≥ ~800). No unit lacks its condition within a batch.
- The penalty did not densify the code: median fraction of states a live unit is on 0.39 (`both`)
  vs 0.42 (`none`); units on in ≤ 25% of states 267 vs 258; on in ≥ 75% 103 vs 159.
- Error decomposition (1 − r², share of target variance): per-state bias 0.004 / 0.005 (negligible,
  = the identical spectra); within-state 0.115 (`both`) vs 0.090 (`none`). Within-state splits into
  steady 0.017 vs 0.011 and transient (≤ 15 steps after a flip, 31% of steps) 0.334 vs 0.267 per
  step. **85% of the penalty's deficit is in the 15 steps after a flip**: slower or rougher
  transitions, not wrong levels and not mis-timed units. With noise on the gap is larger (0.034 vs
  0.023), so noise-driven jitter adds the rest. Mechanism (why frm slows transitions) not measured.

## ▶ FIGURE: hyper flip-flop grid at 150k — live units and r² vs k, none vs both (67/72 cells) — 2026-09-14 10:45

`fig_hyper_readout.py` (reads `data/hyper_readout_150k.txt`, the table printed by
`flipflop_hyper_readout.py` on Spock) → `img/internal_figures/hyper_readout_150k.png`.

![Hyper grid at 150k: live units and r² vs k](../img/internal_figures/hyper_readout_150k.png)

| k | N | plain none | hyper none | hyper both | r² plain / none / both |
|---|---|---|---|---|---|
| 2 | 500 | 178 ± 7 | 216 ± 14 | 500 | 0.956 / 0.941 / 0.926 |
| 2 | 1000 | 237 ± 11 | 287 ± 11 | 1000 | 0.956 / 0.941 / 0.929 |
| 2 | 2000 | 309 ± 33 | 389 ± 27 | 1997 ± 3 | 0.956 / 0.940 / 0.929 |
| 4 | 500 | 198 ± 3 | 342 ± 5 | 500 | 0.952 / 0.887 / 0.841 |
| 4 | 1000 | 281 ± 10 | 491 ± 21 | 1000 | 0.952 / 0.886 / 0.855 |
| 4 | 2000 | 368 ± 9 | 672 ± 28 | 1998 ± 2 | 0.952 / 0.887 / 0.859 |
| 6 | 500 | 225 ± 8 | 464 ± 2 | 500 | 0.950 / 0.798 / 0.657 |
| 6 | 1000 | 301 ± 2 | 715 ± 13 | 1000 | 0.950 / 0.807 / 0.712 |
| 6 | 2000 | 367 ± 28 | 1016 ± 57 | 2000 | 0.950 / 0.811 / 0.738 |
| 8 | 500 | 238 ± 11 | 494 ± 2 | 500 | 0.948 / 0.625 / 0.308 |
| 8 | 1000 | 313 ± 8 | 977 ± 2 | 1000 | 0.948 / 0.681 / 0.408 |
| 8 | 2000 | 618 ± 318 | 1609 (n=1) | pending | 0.948 / 0.697 / — |

What the figure shows, top row: the unpenalised hyper network recruits along the demand ladder at
every N — from the plain task's level at k=2 (+21%) to the whole network at k=8 for N ≤ 1000 and 80%
at N=2000 — while the plain task's count is nearly flat in k (k^0.16) and the penalised network is
at N everywhere. Bottom row: hyper r² depends on k only (the three columns are the same curve),
the penalty is below `none` at every cell and the gap grows with k (0.01 → 0.03–0.05 → 0.07–0.14
→ 0.27–0.34), and N does not close it. The plain k=8 N=2000 reference carries one aberrant seed
(618 ± 318); the other two seeds sit near 400.
Pending: k=8 N=2000 seeds 2–3 (`none`) and all three `both` (Spock, 2026-09-14 evening → 09-15 morning).

10:48 — `fig_hyper_readout.py` now plots every seed (jittered in k) with the mean as the line, from `flipflop_hyper_readout.py --seeds` (`data/hyper_readout_150k_seeds.txt`); the sd error bars were invisible at this scale (2–30 units). Seed spread: ≤ 30 units in every hyper cell; the plain k=8 N=2000 reference has one seed at 1070 against two near 390.

10:50 — Walsh k=6 N=1000 networks (100k) added to `hyper_readout_150k.png` as stars on the N=1000 panels (`data/walsh_points.txt`: none 781 live / r² 0.880; both 992 / 0.846; trace live count at the last snapshot, same criterion as the hyper table). Read against the hyper k=6 N=1000 cells (none 715 ± 13 / 0.807; both 1000 / 0.712): with ONE output instead of 63 the unpenalised network recruits the same ~75% of itself, and both arms score higher because the one-channel target has no fast-switching high-order channels — the k-set r² floor of the hyper task was the temporal demand, as suspected. The k=8 pair slots in when it finishes.

### Is hyper k=8 N=500 `none` still improving at 150k? No — plateaued — 2026-09-14 10:52

Noise-free clean loss (mean over seeds) at 30k / 60k / 90k / 120k / 149k, and the relative slope per
10k over 90–120k then 120–150k:

| cell | @30k | @60k | @90k | @120k | @149k | slope 90–120k | slope 120–150k |
|---|---|---|---|---|---|---|---|
| k=8 N=500 none | 0.211 | 0.187 | 0.166 | 0.161 | 0.163 | −0.96 %/10k | **+0.40 %/10k** |
| k=8 N=1000 none | 0.205 | 0.163 | 0.145 | 0.138 | 0.140 | −1.71 | **+0.53** |
| k=8 N=2000 none (1 seed) | 0.177 | 0.145 | 0.136 | 0.133 | 0.127 | −0.80 | −1.70 |
| k=8 N=500 both | 0.315 | 0.291 | 0.283 | 0.280 | 0.277 | −0.39 | −0.38 |
| k=8 N=1000 both | 0.267 | 0.234 | 0.225 | 0.220 | 0.218 | −0.74 | −0.27 |
| k=6 N=500 none | 0.111 | 0.100 | 0.093 | 0.091 | 0.093 | −0.61 | +0.61 |

At N=500 and N=1000 the unpenalised k=8 loss stopped falling between 120k and 150k (slightly UP),
after −1 to −1.7 %/10k in the previous window: r² ≈ 0.62–0.68 is a plateau at this budget, not a
slow descent. Same at k=6 N=500. Only N=2000 (one seed, and the slowest cell) is still descending.
The penalised arms are still creeping down at −0.3 to −0.4 %/10k, i.e. they converge more slowly
here, but from far below. Whether the plateau is the true floor or a long plateau before another
descent is what the Della 500k cells (k=4 N=500/1000 landing tonight, k=6 N=500 on 09-16) will say;
at k=2 and k=4 r² did not move between 150k and 440k.

## ▶ ★★ DECISION POINT (Pavel, 2026-09-14): the recruited-unit count reflects task complexity; the penalty story is closed; pivot or re-report — 10:57

**Pavel's assessment, recorded verbatim in substance.** The number of recruited units almost
certainly reflects task complexity. The one remaining confusion is the read-out: the active-unit
count could in principle be a property of the number of read-out channels rather than of complexity
per se. The Walsh task (one output, combinatorial demand) is disproving that, but its results are
still preliminary (one seed, k=6; k=8 pending). Either the project pivots, or it reports this
finding instead — well characterised — without the frm + rws penalty line.

**The evidence this rests on, all in this document above:**
1. *Recruitment tracks demand.* Hyper flip-flop, 150k, unpenalised: live units rise with k at every
   N (N=500: 216 → 342 → 464 → 494 for k = 2/4/6/8; N=1000: 287 → 491 → 715 → 977; N=2000: 389 →
   672 → 1016 → 1609) while the plain flip-flop is nearly flat in k (178 → 238 at N=500). At k=8 the
   network uses all of N=500 and N=1000 and 80% of N=2000. Seed spread ≤ 30 units per cell.
2. *Not a depth artefact.* Della 500k: hyper k=2 N=500 excess over plain is 1.16× at 150k and 1.27×
   at 500k; r² unchanged 150k → 440k at k=2 and k=4.
3. *Not read-out width (preliminary).* Walsh k=6 N=1000, one output: 778 live at 30k, 781 at 100k,
   vs hyper (63 outputs) 715 and plain (6 outputs) 412 at the same iteration. One seed.
4. *The penalty (frm 0.1 + rws 0.05) recruits every unit and costs performance at every cell*, with
   the cost growing with demand (Δr² −0.01 at k=2 → −0.27/−0.34 at k=8), not closed by N, not a
   convergence-speed effect (unpenalised plateaus by 120k; penalised creeps at −0.4 %/10k from a
   1.7× higher loss). On the Walsh task the deficit is in the 15 steps after a flip, not in the
   levels or the latent variables built. Both arms converge to the same spectrum.
5. *The unpenalised r² plateau is set by k, not N* (hyper: 0.94 / 0.89 / 0.80 / 0.65 at
   k = 2/4/6/8 at every N; flat from 120k). On the Walsh task, which has no fast-switching channels,
   k=6 reaches 0.88 at 100k, so the hyper plateau is at least partly the temporal demand of the
   high-order product channels.

**What is still open before the report can be written:** the Walsh k=8 N=1000 pair (tonight;
r² 0.86 vs 0.81 at 90k), Walsh seeds 2–3 and the degree ladder (the 90-job launcher is ready, not
submitted), the last hyper k=8 N=2000 cells (2026-09-15 morning), and the Della 500k depth
control at k=4 and k=6. The mechanism of the penalty's transition cost is unmeasured and, under
this decision, not needed.

**Consequence for the paper, Pavel's call:** the storyline "silence is a pathology, the penalty
fixes it" is retired. The candidate replacement is the capacity law — silence measures how much
of the network the task requires, ~10 units per latent function on the flip-flop family, with the
mechanism results (spikes, init, input-row norm) as how a network arrives at that size — or a
pivot away from silence altogether. The penalty sections become at most a characterisation.

## ▶ THIRD TASK: DMTS with a 16-tau delay, four penalty arms x three sizes x three seeds (Spock) — 2026-09-14 20:29

**Why a third task.** CDDM and the flip-flop family can both be carried partly by transients: CDDM's
stimulus stays on until the decision, and a flip-flop bit is re-pulsed every ~7 tau. Pavel asked for
a task with a long silent gap between sample and match, so the memory must be SUSTAINED activity,
and for the same four penalty arms as the other two tasks to complete the picture. Second question
(Pavel): does the penalty make such a task train FASTER, since the task itself needs sustained
activity and frm pushes every unit's mean rate up?

**Task: `configs/task/DMTS_long.yaml`** (the class is the existing `TaskDMTS`; the default
`DMTS.yaml` with its 6-tau delay is untouched). T=300 like CDDM and the flip-flop. Sample 2–4 tau,
silent delay 4–20 tau (16 tau, no input), match 20–22 tau, decision cue + target 24–30 tau; sample
and match onsets jittered independently by ±1 tau. 4 stimuli (n_inputs = 4 + cue) → 16 sample/match
pairs (4 matches), 2 bits to hold; 64 repeats per batch → 1024 trials, as on the flip-flop.
n_outputs=2 (match / non-match channels, like CDDM's two-choice read-out). Scored: 0–24 tau (outputs
must stay at 0) and 25–30 tau. Batch generation 0.026 s per 1024 trials (fresh batches every
iteration, `trainer_ptrack_freshbatch`).

**Launcher: `slurm/SilentReLU_dmts_penalties_spock.slurm`**, 36 jobs = N ∈ {500, 1000, 2000} ×
pen ∈ {none, rws 0.05, frm 0.1, both} × 3 seeds, h equation, standard ReLU, 150k iterations, the
same lambdas and iteration budget as `NBitFlipFlop_std_pen` and the hyper grid. Output
`DMTS_std_pen/EqType=h_N=<N>_pen=<name>/`. Smoke-tested locally (N=100, 30 iterations, CPU:
pipeline runs end to end, trace / params / Adam state / config written, mask = 290 steps).

**Pre-registered read-outs at 150k** (nothing new logged; all from the participation trace + r²):
1. Live units vs N per arm.
2. r² per arm; solved := r² ≥ 0.9 in all 3 seeds.
3. Convergence speed: t90 := first probe (clean loss, every 10 iterations) within 10% of the arm's
   own 150k value, plus the matched-loss time (first probe at which each arm reaches `none`'s 150k
   loss). **Falsifier for "the penalty speeds training"**: supported only if t90(frm) < t90(none) AND
   t90(both) < t90(none) in every seed at every N. One seed or one N the other way → "no, or not
   reliably". A fast descent to a worse floor does not count as speed (hence the matched-loss time).

20:35 — Calibration (600 iterations, N=2000 `none`, job 6201204): 0.456 s/iter → 19 h for 150k;
r² already 0.56 at 600 iterations, so the task is learnable at this horizon. Full grid submitted at
commit 800ee30: array **6201205** (tasks 1–24, N=500/1000, 24 h) and **6201206** (tasks 25–36,
N=2000, 48 h); all 36 started at once (17 idle A100 nodes). Expected done: N≤1000 by ~07:00 on
2026-09-15, N=2000 by ~16:00.

## ▶ FIGURE + READ-OUT: DMTS 16-tau delay, N=500/1000 cells at 150k (24 nets; N=2000 lands tonight) — 2026-09-15 09:03

`dmts_readout.py data/trained_RNNs [--seeds] [--dump]` (run on Spock) and `fig_dmts_pen_curves.py`
(from the dumped `data/dmts_curves_150k.npz`) → `img/internal_figures/dmts_pen_curves.png`.

![DMTS penalty arms: clean loss and live units vs iteration](../img/internal_figures/dmts_pen_curves.png)

Clean r² = 1 − L/0.0788 (target variance over the scored steps); r² = 0.9 ↔ L = 0.0079. Live
units at the last snapshot ≤ 150k under three criteria (scale-free / absolute 1e-6 / Otsu on the
pooled log participation of all 24 nets, threshold 0.011). Times are iterations, three seeds each,
sorted. `unstable` = fraction of probes after 75k at which the clean loss is above the r² = 0.9 level.

| N | pen | live sf / 1e-6 / Otsu | L at 150k | r² (noisy) | t(r² ≥ 0.9) | t90 (own floor) | t(match none's floor) | unstable |
|---|---|---|---|---|---|---|---|---|
| 500 | none | 249 ± 22 / 388 ± 5 / 350 ± 7 | 3.1e-4 | 0.999 | 2300 / 3180 / 3750 | 3580 / 3810 / 6590 | 2720 / 3600 / 4070 | 0.8% |
| 500 | rws | 184 ± 27 / 337 ± 42 / 263 ± 32 | 3.9e-4 | 0.999 | 6120 / 6430 / 8050 | 6300 / 6740 / 11540 | 6950 / 7490 / 8900 | 2.0% |
| 500 | frm | 500 / 500 / 500 | 2.3e-4 | 0.992 | **420 / 970 / 1210** | 2260 / 3360 / 10590 | 2690 / 3120 / 3660 | 3.3% |
| 500 | both | 500 / 500 / 500 | 3.7e-3 (one seed mid-episode) | 0.948 | **460 / 550 / 630** | 520 / 11550 / 23770 | 6780 / 7170 / 7210 | 4.4% |
| 1000 | none | 424 ± 39 / 655 ± 86 / 502 ± 43 | 1.5e-4 | 0.999 | 1810 / 2300 / 4730 | 2570 / 3880 / 9660 | 2950 / 4210 / 4980 | 1.6% |
| 1000 | rws | 329 ± 22 / 636 ± 43 / 493 ± 18 | 5.2e-4 | 0.999 | 6870 / 7980 / 8000 | 7650 / 8150 / 8270 | 13130 / 13530 / 13890 | 2.0% |
| 1000 | frm | 1000 / 1000 / 1000 | 3.4e-4 | 0.994 | **760 / 1520 / 2640** | 3340 / 4060 / 8440 | 8380 / 9610 / 10430 | 2.2% |
| 1000 | both | 1000 / 1000 / 1000 | 5.2e-5 | 0.995 | **1110 / 1550 / 2360** | 26680 / 26740 / 29300 | 13970 / 14410 / 14480 | 0.2% |

**1. The rate penalty gets the network off the no-memory plateau earlier; the sparsity penalty
delays it.** Every network starts on a plateau at L ≈ 0.033 (r² ≈ 0.58: it reads the decision cue
but carries nothing across the delay). Time to leave it (clean r² ≥ 0.9): frm 0.4–1.2k and both
0.5–0.6k against none 2.3–3.8k at N=500 — a 3× lead, every frm/both seed ahead of every `none`
seed. At N=1000 the lead is 2× in the mean (frm 0.8–2.6k, both 1.1–2.4k vs none 1.8–4.7k) but the
seeds overlap (frm's slowest 2640 > none's fastest 1810). rws alone is last everywhere (6–8k). From
the raw logs, the same holds at N=2000 at 60k: frm 2.1–2.9k, both 4.7–6.2k, none 2.0k / 11k / 30k,
rws 9.7–13.9k — two `none` seeds sat on the plateau for 11k and 30k iterations before escaping.
So: the effect is on the ESCAPE, it is the rate term, and it grows with N.

**2. By the pre-registered criterion the claim "the penalty speeds training" is NOT supported.**
t90 (first probe within 10% of the arm's own 150k loss) is not consistently smaller for frm or
both: at N=500 frm 2.3k / 3.4k / 10.6k vs none 3.6k / 3.8k / 6.6k; at N=1000 both takes 27–29k
against none's 2.6–9.7k. The penalised arms leave the plateau early and then descend to their
floor SLOWLY (the s-shaped orange/yellow curves in the figure), while `none` drops from the plateau
to near its floor within ~1k iterations once it escapes. Which one is "faster" depends on where the
bar is set: at r² = 0.9 the penalty wins by 2–3×; at the floor it does not. The matched-loss time
(reaching `none`'s 150k loss) is 2.7–3.7k for frm at N=500 (same as none itself) but 8–10k at
N=1000, and 7k / 14k for both. Recorded as the honest answer to Pavel's question: the penalty
makes the memory APPEAR earlier, it does not make the network CONVERGE earlier.

**3. Unlike CDDM and the flip-flop, the penalty costs no clean loss here.** Floors at 150k are
1–5e-4 for every arm, and `both` at N=1000 has the lowest of all (5e-5). The noisy r² (0.992–0.995
vs 0.999) is the only place the penalised nets lag: they are less robust to the sigma 0.05 noise,
not worse noise-free. Interpretation, not measured: with every unit driven to cap_fr the noise
enters through more units.

**4. Every arm keeps losing the memory in brief episodes.** After 75k, 0.2–4.4% of probes find
the clean loss back at the plateau level (the vertical streaks in the figure). No consistent
ordering: both is worst at N=500 (4.4%, and one seed finished INSIDE an episode, r² 0.858 as the
last net) and best at N=1000 (0.2%). One `none` seed at N=1000 blew up to L = 1e7 at ~3.5k and was
rolled back by the trainer. The 150k "last network" is therefore a lottery ticket on this task;
the best-checkpoint network or an averaged late loss is the right thing to report.

**5. Recruitment: DMTS keeps far more units live than CDDM at the same N, and the count is still
falling at 150k.** Unpenalised: 388 / 655 live at N=500 / 1000 under the 1e-6 criterion (CDDM:
215 / 290, i.e. 57% / 71% silent) — 1.8–2.3× more. rws alone silences MORE (337 / 636 by 1e-6;
184 / 329 scale-free vs none's 249 / 424). The bottom panels show the live count declining from
~10k on with no plateau, the "silencing never stops" pattern of the other tasks; the numbers above
are a 150k snapshot, not a converged count. The three criteria disagree by up to 1.6× (249 vs 388
at N=500 none) — the participation distribution is not cleanly bimodal on this task.

Open until tonight: the N=2000 cells (12 jobs, ~68k/150k at 09:00, done ~00:00), which carry the
strongest version of point 1.

09:46 — `dmts_pen_curves.png` re-drawn de-noised (Pavel: too much jumping): every seed is now the
median of its probes in 60 log-spaced iteration bins (`common.logbin`), raw probes not drawn; the
memory-loss episodes they contained are counted in the read-out's `unstable` column instead.

### CHECK (Pavel, 2026-09-15): 36-tau delay — does the unpenalised network find the memory at all? — 09:48

Pavel's hypothesis: the frm advantage at 16 tau (leaving the no-memory plateau 2–3× earlier, more
at N=2000) should grow with the delay, to the point where the unpenalised network never leaves the
plateau. Check: `configs/task/DMTS_delay36.yaml` = DMTS_long with T 300 → 500 and everything after
the sample shifted by 200 steps (sample 2–4 tau, silent delay 4–40 tau = **36 tau**, match 40–42,
decision 44–50; same 4 stimuli, 2 outputs, ±1 tau jitter, scoring 0–44 and 45–50 tau).
`slurm/SilentReLU_dmts_delay36_spock.slurm`: **2 jobs, N=1000, one seed each, `none` vs `frm 0.1`**,
150k iterations, same trainer as the grid; output `DMTS_std_delay36/EqType=h_N=1000_pen=<name>/`.
Smoke-tested locally (N=100, 20 iterations; mask 490 steps, delay 360 steps). Submitted as array
**6201936** at commit 98b57c1, both running at 09:48; ~0.43 s/iter → ~18 h, done ~04:00 on 09-16.

Read-out (fixed now): t(clean r² ≥ 0.9) per arm from `dmts_readout.py` pointed at the new folder.
  - `none` never leaves the plateau within 150k while `frm` does → Pavel's prediction, but ONE
    seed: the 16-tau `none` seeds at N=2000 spanned 2k–30k, so "never in one seed" needs the
    other two seeds (and ideally N=2000) before it is quoted as a result.
  - both escape → the effect at 36 tau is again a speed ratio; report it next to the 16-tau one.
  - neither escapes → the task is too hard at N=1000 in 150k; the check says nothing about the
    penalty and is re-run at N=2000 or with a longer budget.
N=1000 rather than 2000 was chosen for cost (18 h vs ~31 h at 100k) and because both arms escaped
within 5k there at 16 tau, so a `none` failure at 36 tau would be a clean reversal.

## ▶ DROPOUT: inventory, a no-op bug since 2025-12, the fix, and the first real dropout grid (3-bit flip-flop, N=1000) — 2026-09-15 09:59

**Pavel's question:** could dropout keep units alive (force redundancy) on its own or on top of the
penalties? First, how many dropout variants exist?

**Inventory (RNN_torch.forward / rhs / get_dropout_mask, Trainer.train_step).** TWO kinds and
THREE sampling methods, 6 combinations, all switched by `trainer.dropout_args`:
- kind `mute`: the unit runs in the recurrent dynamics; only its read-out weight is masked to 0.
- kind `dead`: the unit is ablated for the batch — its rate is zeroed for every other unit, its
  drive and noise are zeroed (x decays to 0), and its read-out weight is masked.
- There is NO third "silenced but not killed" kind; `mute` is the output-only one.
- sampling `uniform` (every unit with probability drop_rate), `participation` (probability
  drop_rate·N·softmax(beta·participation_EMA): the most active units dropped more; the EMA has
  rate eta on the per-unit std + 0.9-quantile of |x|), `output_weights` (same with Σ|W_out|).
  One Bernoulli mask per unit per training batch, shared across its trials.
- Config defaults (all trainer yamls, `dropout: False`): mute / participation / drop_rate 0.05 /
  beta 1.0 / eta 0.5 / activity_q 0.9. `docs/hyperparameter_walkthrough.md` lists it as UNUSED.

**Bug: dropout has never affected training.** `Trainer.train_step` scored the task loss on the
NON-dropout forward pass (`output_full`) and handed the dropped output (`output_do`) only to the
penalties, none of which use the output (frm/rws read states and weights). Since commit 76e74b9
(2025-12-03). Verified, not inferred: 15-iteration runs with `seed=1 task.seed=1`, dropout off vs
`dead` vs `mute` at drop_rate 0.5 (uniform) — `dead` and `mute` were BIT-IDENTICAL over all 15
iterations (1.028713, 1.019228, 1.0066699, …), and both differed from `off` only from iteration 2,
i.e. only through the extra forward pass consuming the noise generator. Two different ablations
giving the same training is only possible if neither reaches the gradient. **Every earlier dropout
run (slurm/Dropout.slurm, 2025, 14 jobs) is void.**

**Fix (this commit):** task loss on `output_do`, penalties on the full pass (their meaning is
unchanged; the full pass is still needed for `states`). Same test after the fix: off / dead / mute
differ from iteration 1 (1.028713 / 1.0219723 / 1.024176) and dead ≠ mute throughout. Cost: two
forward passes per iteration whenever dropout is on.

**Grid submitted: `slurm/SilentReLU_flipflop_dropout_spock.slurm`**, 8 jobs = {none, rws, frm,
both} × {mute, dead}, 3-bit flip-flop, N=1000, one seed each, 150k iterations, sampling and rate
from the config (participation, 0.05); output `NBitFlipFlop_std_dropout/EqType=h_k=3_N=1000_
pen=<pen>_do=<kind>/`. References at the same cell and iteration: `NBitFlipFlop_std_ksweep` k=3
(none) and `NBitFlipFlop_std_pen` k=3 (rws / frm / both), 3 seeds each.
Pre-registered read-out at 150k: live units under BOTH flip-flop criteria (scale-free and absolute
4e-2) and the clean loss. "Dropout keeps units alive" := live(none+dropout) > reference `none`
mean + 3 sd under both criteria; "adds to rws" := the same for rws. frm and both are already at N,
so for them the read-out is the loss (does dropout cost or buy performance). One seed each — a
positive result gets 3 seeds before it is quoted. Smoke-tested locally (N=100, 15 iterations, the
saved config carries dropout: true / dead / participation / 0.05).

10:01 — submitted as array **6201944** at commit 1674662; all 8 running at once. ~0.4 s/iter → done ~03:00 on 09-16.

## ▶ PREPARED (not submitted): fifteen tasks in one network — does CDDM recruit more units when the network also serves 14 others? — 2026-09-15 11:30

**Pavel's question**, from the framing discussion (the subsampling goal needs a large REDUNDANT
population on the task the data were recorded on, and a cortical circuit is never doing only one
task): train the same network on 15 tasks and read the recruitment on CDDM alone. Prediction: it
rises, for two reasons that must be kept apart — units other tasks need still respond during CDDM
trials whenever they share inputs (realistic nuisance activity), and CDDM's own computation may or
may not become more redundant. The read-out splits the two.

**Built and tested (this commit):**
- `tasks/TaskMultiRule.py`: Yang-style composite. Input = [one-hot rule, on for the whole trial |
  the subtask's inputs zero-padded to the widest]; output zero-padded to the widest; shorter
  trials zero-padded to T and the padding scored against 0. Every subtask is built from its own
  `configs/task/<name>.yaml` through prepare_task_arguments / get_training_mask, so CDDM inside the
  composite has exactly the single-task timing, channels and scoring window. Subtask batch sizes are
  capped at the 68 trials drawn (the Walsh expansion alone was 115 ms per batch); composite batch
  1020 trials in ~70 ms. `task_batch(i)` gives one subtask's full batch for per-task read-outs.
- `configs/task/MultiRule15.yaml`: CDDM, NBitFlipFlop, NBitFlipFlopHyper (k=3), NBitFlipFlopWalsh,
  DMTS, DMTS_long, MemoryNumber, MemoryAntiAngle, MemoryDM, DelayDM, GoNoGo, XOR, HalfAdder,
  SquareNumber, AngleIntegration; n_inputs 21, n_outputs 7. New configs: DelayDM, MemoryDM,
  SquareNumber; AngleIntegration.yaml fixed (task_params, list amp_range).
  **AngleAddition is excluded**: it is the ReferenceFrame task and, as configured, its target lies
  inside its unscored stimulus window — the scored target is identically zero (r² = −6·10⁸ in the
  read-out). Pre-existing; anything trained on that config learned nothing.
- **Trainer: per-trial scoring masks.** A batch mixing tasks of different length needs a (T, B)
  mask, not one time index. `scored_(x, mask)` selects the scored entries for either form;
  task_penalty, r2_score, the clean-loss probe and eval_step use it; `run_training` takes the mask
  from `Task.batch_mask(conditions)` when the task provides one. **Regression: the single-task path
  is unchanged bit for bit** (seeded 15-iteration flip-flop run, losses identical to the pre-patch
  run: 1.028713, 1.019451, 1.0109345, 1.0005492 … 0.9136622), and the (T, B) form gives the same
  loss and r² as the index form on random data.
- Two pre-existing task bugs fixed on the way: `TaskSquareNumber.get_batch` never appended its
  conditions (returned an empty list); `TaskDMTS` put the decision cue on channel 2 regardless of
  n_inputs, so DMTS_long (4 stimuli + cue) had the cue sharing a line with stimulus 2. The running
  DMTS grids are still valid DMTS (the cue is time-locked at 24 tau, unambiguous, and r² reached
  0.999), but they carry that wart; the cue is now on the last channel. Not re-run.
- `experiments_and_analysis/multirule_readout.py` (pre-registered read-out, tested on a 15-iteration
  N=100 smoke net): per subtask, noise-free full single-task batch with the rule on → participation
  → live units under scale-free / 1e-6 / 4e-2, and r² on the task's own window; the mixed batch
  (live_all); and the CDDM-active set split into shared-with-another-task vs CDDM-private.
- `slurm/SilentReLU_multirule_spock.slurm`: 18 jobs = N ∈ {500, 1000, 2000} × {none, both} × 3
  seeds, 150k iterations, fresh batches; output `MultiRule_std_multi/EqType=h_N=<N>_pen=<name>/`.

**Pre-registered read-out at 150k**, against the single-task CDDM references at the same N and
iteration (CDDM_std_g0_drift for none, CDDM_std_g0_penalties for both, local traces):
"multi-task raises recruitment on CDDM" := live_CDDM(multi) > single-task mean + 3 sd in every
seed under BOTH the scale-free and an absolute criterion. Then the split: if the extra units are
shared with other tasks, it is nuisance activity from shared inputs; if CDDM-private units rise,
CDDM's own code became more redundant. Per-task r² flags any subtask the network failed to learn
(a failed subtask means the demand was not delivered and its units do not count).
Caveat fixed now: 150k may be short for 15 tasks; the clean-loss trace decides, and a still-falling
loss at 150k means a warm-started continuation before the read-out, not a read-out at 150k.

11:52 — Calibration on Spock (job 6202041, N=2000 `none`, 600 iterations, commit 51237d6): 0.81 s/iter
→ 34 h for 150k (fits the 48 h request); r² 0.49 on the mixed batch after 600 iterations, pipeline
clean on the GPU. Calibration net removed from the cell. **Grid not submitted — awaiting Pavel.**

## ▶ PREPARED (not submitted): Yang et al. (2019) 20-task family ported; CDDM alone vs 20 tasks, read on CDDM — 2026-09-15 12:45

Pavel rejected the 15-task set of 11:30 (the repo's tasks do not map onto one another: the
flip-flop family shares nothing with the rest, the hyper task is not a cognitive task) and asked
for Yang's family instead, as inspiration rather than exact replication, with the goal fixed:
**compare a CDDM-trained network with a 20-task-trained network and read how units are recruited
on CDDM in both.** MultiRule15.yaml and its launcher are deleted; TaskMultiRule stays (generic).

**`tasks/TaskYang.py`** — the 20 rules of the paper in its shared space: fixation input, two
32-unit direction rings, 20 rule channels (85 inputs); fixation output + 32-unit response ring
(33 outputs). Go / RT Go / Delay Go and Anti versions; DM1, DM2, Ctx DM1, Ctx DM2, MultSen DM and
the five delayed versions; DMS, DNMS, DMC, DNMC. Stimulus = 0.8·exp(−d²/2σ²) bump, σ = π/8, DM
strengths 1 ± c with |c| ∈ {0.04, 0.08, 0.16, 0.32}; fixation output 0.85 → 0.05 at the go signal;
response ring 0.05 + 0.8 bump; 1-tau grace after the go unscored. **T = 300 steps, tau = 10 steps,
the units of every other grid**; epoch lengths drawn per trial (fix 3–5 tau, stimulus 3–10, delay
3–10, response 4), all trials end by step 285. Unused steps are zero input and unscored.
Deviations from the paper, deliberate: uniform loss weight (the paper ×5 on the response epoch),
per-trial rather than per-batch epoch lengths, this repo's Adam / lr scaling / noise, mixed batches.
`configs/task/Yang20.yaml` (all 20 rules, batch 20 × 50) and **`Yang_ctxdm1.yaml` (contextdm1
alone, batch 1000, the SAME 85/33 layout — the single-task CDDM reference)**. contextdm1 = Mante's
context-dependent decision in ring coding (both modalities show the same two directions with
independent evidence; respond on modality 1).
Checks: every rule's scored target has variance (0.022–0.036), match tasks respond on 46–55% of
trials, the example-trial figure `img/internal_figures/yang_tasks_examples.png` reads correctly,
15-iteration smoke trainings of both configs run and `multitask_readout.py` (the generalised
read-out: `--focus contextdm1`) produces the per-rule table on both, with the single-rule net's
active units all "private" as they must be. Trial generation is a Python loop, 0.17 s per
1000-trial batch — an overhead of ~0.17 s/iter to be vectorised if it matters.

**Launcher `slurm/SilentReLU_yang_spock.slurm`, 36 jobs** = {multi20, ctxdm1} × N ∈ {500, 1000,
2000} × {none, both} × 3 seeds, 150k iterations; output `Yang_std_multi/EqType=h_set=<set>_N=<N>_
pen=<name>/`. **Pre-registered read-out** (same script on both sets): live units on contextdm1
trials under the scale-free AND an absolute criterion; "multi-task raises recruitment on CDDM" :=
live(multi20) > live(ctxdm1) mean + 3 sd in every seed at that N under both; the focus-active set
split into shared-with-another-rule vs private (nuisance from common inputs vs the task's own code
becoming redundant); per-rule r² so an unlearned rule does not count as delivered demand; and the
clean-loss trace read first — still falling at 150k means a warm-started continuation before the
read-out.

12:33 — Pavel: vectorise, batch 1024, T=300 confirmed, and one figure per task with 3 example
trials and a description. Done: `TaskYang.batch_of` generates a rule's trials in one vectorised
pass (float32; 1024-trial batch in ~90 ms, was 170 ms per 1000 in the loop version); `batch_size:
1024` replaces n_per_task in both configs (rules assigned near-evenly at random per batch, 51–52
each); `fig_yang_tasks.py` → `img/internal_figures/yang_tasks/<rule>.png` (20 figures: all 85 input
and 33 output channels of 3 trials, go signal and grace marked, description in the title) plus a
`README.md` index of the descriptions there. Every rule re-checked after vectorisation: scored
target variance 0.022–0.036, fixation input and response ring consistent with the trial's go time
and respond flag in 200/200 trials per rule, match tasks respond on 47–54%.

12:40 — Pavel's questions on the Yang port, and two changes. (1) Trials now use ALL 300 steps: the
response epoch runs from the go signal to T (5–27 tau depending on the trial, must be sustained);
everything is scored except the 1-tau grace, so the scored fraction is 0.967 instead of 0.52.
(2) Ring view added: `<rule>_rings.png` draws ring 1, ring 2 and the response target as circles
with a bar per unit (32 per ring, preferred directions evenly spaced) at three moments — mid-
stimulus, just before go, mid-response — so the two-bump DM stimuli (strengths 1 ± c at two
directions) and the response bump read directly. For the record: 85 inputs = 1 fixation + 32 +
32 + 20 rules; 33 outputs = 1 fixation + 32; the smear is Yang's Gaussian population code (σ = π/8);
two bumps in one ring are the two choice options of a DM trial. Re-checked all 20 rules after the
change (fixation off and response bump consistent with each trial's go time and respond flag).

12:59 — Pavel on the ring view: uninformative snapshots (empty rings), replaced by a TRIAL-STRUCTURE
figure (`fig_yang_tasks.py --structure <rule>` → `<rule>_structure.png`): per trial the epochs as
step functions (fixation input, stimulus on each ring, rule, go, target fixation, target response)
with three arrowed snapshot times chosen per task, and the rings at exactly those times as polar
bars with the two strengths printed; horizontal bar under each snapshot's rings. Done for
contextdm1 (A fixation, B stimulus before go, C response), other rules once the format is agreed.
Pavel's second point, "in CDDM the second modality is independent": it is here too — the two
directions are the two CHOICE OPTIONS shared by both rings (motion and colour both speak about
left vs right), and each ring's evidence (sign and size) is independent, so half the trials are
conflicts; the first figure had drawn three congruent trials by chance. The figure now picks a
congruent trial, a conflict with a strong distractor, and a conflict with weak attended evidence.
**Reference changed to the pair:** Mante's task has both contexts in one task with a context cue;
in Yang's family the context is the rule, so the single-task CDDM reference is `Yang_ctxdm.yaml`
= contextdm1 + contextdm2 trained together (rule channel = context cue), not contextdm1 alone.
Launcher set `ctxdm`; `multitask_readout.py --focus contextdm1,contextdm2` takes the union as the
focus-active set. Smoke-tested (10 iterations, N=100; read-out reports both rules, all active
units private as they must be in the single-task net).

13:06 — Structure figures for all 20 rules (`fig_yang_tasks.py`, `<rule>_structure.png`), with
per-family snapshot times (fixed-delay Go / DM: fixation, stimulus before go, response; reaction-
time: fixation, stimulus onset = go, response; delayed Go / DM: stimulus on, delay with stimulus
off, response; match family: sample on, delay, test on = go with response) and per-family trial
selection (context DM: congruent / conflict / weak-attended conflict; other DM: strong, weakest,
strong for the other option; multisensory: agree-strong, disagree, weak sum; match family: match,
non-match, match). Checked reactanti, delaydm1, dnmc, multidm, contextdm1 by eye: epochs, go,
targets and the ring snapshots agree with each trial's parameters and response.

13:29 — Pavel could not read the match/category tasks from the figures (the category rule was never
stated). Every task description rewritten in full on the figure (inputs, what to do, correct
output, what each snapshot shows), plus the common trial rules; trial titles now say for DMS/DNMS
whether the test equals the sample or differs by how many degrees, and for DMC/DNMC the category
of sample and test (A = 0–180 deg, B = 180–360 deg; the A half is shaded on every ring). The
categories are about the half-circle, not the direction: in DMC two different directions in the
same half are a match.

13:44 — Yang calibration on Spock (job 6202165, multi20 N=2000 `none`, 600 iterations, commit
d8d5e84 = vectorised task, response to T): 0.90 s/iter → 37 h for 150k (fits 48 h); r² 0.88 on the
mixed batch after 600 iterations (the family learns its fixation/response structure fast).
Calibration net removed from the cell. **36-job grid ready, not submitted — awaiting Pavel.**

### Is r² 0.88 after 600 iterations on 20 tasks real? Yes — falsification checks pass — 2026-09-15 15:16

Pavel: "looks miraculous". Pooled r² is a poor yardstick here, so the paper's own criterion was
added to `multitask_readout.py`: per-rule ACCURACY = decoded response direction (population vector
of the response ring over the second half of the response epoch) within 36° of the target with
the fixation output down, or fixation kept and ring flat on no-response trials; chance 0.10 for a
responding trial. A trivial input-copying predictor scores pooled r² 0.07–0.11, so target structure
does not explain 0.88. A fresh 600-iteration N=500 `none` network (job 6203736, r² 0.88):
accuracy 0.95–1.00 on 19 rules, dnms 0.84. Three checks that a shortcut would fail
(`yang_checks.py`, 256 trials per rule, noise-free, numpy forward pass):
1. Context tasks on CONFLICT trials only (rings favour different options): contextdm1 0.99,
   contextdm2 0.98, contextdelaydm1 0.94, contextdelaydm2 0.95 — a strongest-bump heuristic
   would be ~0.5. The rule is used.
2. Memory tasks by delay length, shortest vs longest third: delaygo 1.00/1.00, delaydm1 1.00/1.00,
   dms 1.00/1.00, dmc 0.99/0.94, dnms 0.81/0.80 — no decay with delay; the memory is real.
3. Rule channels zeroed: anti tasks collapse (fdanti 0.26, reactanti 0.09), match family
   collapses to ~0.42, context tasks drop to 0.66–0.84, while rule-independent DM tasks stay at
   0.88–0.95 — exactly the pattern of a network that uses the rule where it matters.
Conclusion: the family is genuinely learned in ~600 iterations (600 × 1024 ≈ 6·10⁵ trials, of
the order of the paper's training) — in ring coding with this optimizer these tasks are easy.
150k iterations is far beyond what performance needs; it is kept so the read-out is at the same
budget as every other grid (silencing evolves over training regardless of the loss). "Solved" in
the pre-registration now means accuracy ≥ 0.95 per rule (dnms may sit lower). Test net removed.

16:25 — Pavel: before the grid, one pilot pair at N=1000, 10k iterations, unpenalised — the 20-task
network and the CDDM-pair network — to see how many units each uses on CDDM and what performance
each reaches. Batch 1024 for both (same optimizer noise as every grid; the single-task net sees
~10× more CDDM trials per iteration, irrelevant at 10k when 600 already suffices). Submitted as
job **6206472** (launcher tasks 7 and 25, `SAVE_TO=…/pilot10k` so the nets stay out of the grid
cells), ~1 h each once running. Read-out on completion: `multitask_readout.py` (live units on
CDDM trials under three criteria, per-rule accuracy, shared/private split) and `yang_checks.py`.

## ▶ PILOT: CDDM alone vs 20 tasks, N=1000, 10k iterations, unpenalised (job 6206472) — 2026-09-15 17:48

Both networks solve everything: accuracy 1.00 on both context rules in the CDDM-pair net; 0.97–1.00
on all 20 rules in the multi-task net (dnmc 0.97, dmc 0.98); `yang_checks.py` passes on both
(conflict trials 1.00; no dependence on delay length; rule ablation collapses anti to 0.00–0.04,
match family to 0.29–0.71, context to 0.73–0.80 while rule-independent DM stays ≥ 0.95).
Both at 0.44 s/iter (76 min). Live units, noise-free, 256 trials per rule, three criteria
(scale-free / absolute 1e-6 / absolute 4e-2), `multitask_readout.py`:

| network | on CDDM trials (union of both contexts) | on the mixed training batch | CDDM-active shared / private |
|---|---|---|---|
| CDDM pair alone | 574 / ~945 / ~560 | 517 / 978 / 616 | — (all private by construction) |
| 20 tasks | 512 / ~980 / ~630 | 652 / 1000 / 826 | 509 / 3 |

Per rule in the 20-task net (scale-free): Go family 406–618, DM 419–487 (contextdm1 419,
contextdm2 455), delayed DM 598–625, match family 657–703.

Live units over training (`img/internal_figures/pilot10k_live_vs_iter.png`, from the traces, mixed
batch): scale-free CDDM-alone 766 → 707 → 508 at 1k / 5k / 10k, 20-task 830 → 745 → 649; absolute
4e-2 769 → 780 → 639 vs 855 → 850 → 818; absolute 1e-6 ~985 vs 1000 throughout.

![Pilot: live units vs iteration](../img/internal_figures/pilot10k_live_vs_iter.png)

**Reading (one seed, 10k iterations, counts still falling — a snapshot, not a result):**
1. On CDDM trials the 20-task network does NOT use more units than the CDDM-only network under the
   scale-free criterion (512 vs 574) and only somewhat more under the absolute ones (~630 vs ~560 at
   4e-2). The extra recruitment is on the OTHER tasks: the multi-task net's whole-batch count is
   652 vs 517 (+26%), and the tasks with the highest counts are the delay and match families
   (600–700), not CDDM. CDDM's own demand is what it is, wherever it lives.
2. The CDDM-active units of the multi-task net are almost all shared (509 of 512): nothing is
   CDDM-private. So multi-task training gives CDDM a population that is busy with other things —
   the nuisance-activity picture — but not a more redundant CDDM code.
3. This ring-coded CDDM already recruits far more than the old 6-input CDDM: ~50–57% scale-free,
   ~95% by 1e-6 at N=1000 and 10k iterations, against ~30% / ~30% for the old task at the same N
   (the 85-input ring drive alone keeps most units above 1e-6). The old CDDM references are not
   comparable to this family; the CDDM-pair net is the reference.
4. Silencing is still in progress at 10k under the scale-free criterion in both nets (−25% from 5k
   to 10k in the CDDM-only net), as in every other grid; the 150k budget (or at least a matched
   longer one) is needed before any of these numbers is quoted. The criteria disagree by 2× (508
   vs 982): the participation distribution is not bimodal on this family either.

## ▶ CONFIRMATION RUN: CDDM alone vs 20 tasks, N=1000, 30k iterations, 3 seeds (job 6209433) — 2026-09-15 19:05

Pavel's call after the pilot: the result ("training on multiple tasks does not recruit more units
for CDDM") is worth one confirmation at this single cell with error bars, then stop. **The 36-job
grid is dropped** (penalty arms and N=500/2000 add nothing to this question).

`slurm/SilentReLU_yang_30k_spock.slurm`, 6 jobs = {multi20, ctxdm} × 3 seeds, N=1000, unpenalised,
batch 1024, output `Yang_std_30k/EqType=h_set=<set>_N=1000_pen=none/`:
- tasks 1–4: FRESH, 30k iterations (seeds 2 and 3 of each set);
- tasks 5–6: **WARM START** from the 10k pilot nets, 20k more iterations → 30k total (seed 1), at
  Pavel's suggestion — it reuses the pilot instead of retraining it (saves ~2.5 h of GPU).

Warm start restores weights AND Adam moments (`paths.init_adam=true`), which is sanctioned here
because the penalty is unchanged (none → none); the config's warning is about carrying momentum
across a penalty SWITCH. **Verified before submitting** (the load path had never been exercised):
a warm-started run resumes at the source's final loss (first iteration 0.0497 vs the source's
final 0.0505) while a fresh control at the same seed starts at 0.2584. Two gotchas found and
recorded in the launcher: the checkpoint path must be QUOTED in the Hydra override (folder names
contain ';' and '=', which the override parser rejects), and a warm-started run's iteration
counter restarts at 0, so its folder says MI=20000 and its trace covers TRUE iterations 10k–30k —
every net is read at the END of training (30k true iterations for all six) so the comparison is
matched, but any live-units-vs-iteration figure must offset those traces by +10000.

Read-out on completion: live units on CDDM trials (scale-free / 1e-6 / 4e-2) per set as mean ± sd
over 3 seeds, the shared-vs-CDDM-private split in the 20-task nets, and per-rule accuracy
(solved := ≥ 0.95). ~3.7 h fresh, ~2.4 h warm at the measured 0.44 s/iter.

## ▶ READ-OUT DAY: four finished experiments synced from Spock and Della — 2026-09-16 16:14

Both queues empty, every job COMPLETED. Synced to the local mirror and read out below. The Della
500k hyper depth control (15 of the 36 submitted cells) and the Walsh sweep are mirrored too but
not re-read here; nothing about them changed.

### 1. DROPOUT keeps units alive, and it is nearly free (job 6201944, 8 nets) — POSITIVE

`flipflop_dropout_readout.py` (new), read at 150k, 3-bit flip-flop, N=1000, one dropout seed per
cell against the no-dropout reference cells at the same k, N and iteration (3 seeds each:
`NBitFlipFlop_std_ksweep` for none, `NBitFlipFlop_std_pen` for rws, `NBitFlipFlop_std_penlong` for
frm and both). Configs verified identical to the references except `dropout` (same trainer, fresh
batch, lr, sigma_rec, N, k); `loss_clean_train` is probed with dropout OFF, so the loss column is
each net's no-dropout performance and is directly comparable.

| pen | dropout | live scale-free | live abs 4e-2 | clean loss | r² | verdict |
|---|---|---|---|---|---|---|
| none | — | 263 ± 14 | 288 ± 16 | 0.02592 | 0.954 | reference |
| none | mute | 371 | 407 | 0.02658 | 0.953 | **KEEPS ALIVE** |
| none | dead | 372 | 400 | 0.02597 | 0.952 | **KEEPS ALIVE** |
| rws | — | 218 ± 12 | 236 ± 11 | 0.02596 | 0.954 | reference |
| rws | mute | 387 | 403 | 0.02827 | 0.949 | **KEEPS ALIVE** |
| rws | dead | 324 | 373 | 0.06764 | 0.914 | **KEEPS ALIVE** |
| frm | — | 979 ± 7 | 979 ± 7 | 0.02771 | 0.946 | reference (at ceiling) |
| frm | mute / dead | 996 / 1000 | 996 / 998 | 0.02826 / 0.03166 | 0.946 / 0.940 | ceiling, read the loss |
| both | — | 1000 ± 0 | 1000 ± 0 | 0.02906 | 0.942 | reference (at ceiling) |
| both | mute / dead | 1000 / 998 | 1000 / 994 | 0.03045 / 0.03052 | 0.944 / 0.939 | ceiling, read the loss |

The pre-registered bar was reference mean + 3 reference sd under BOTH criteria. `none` clears it by
~8 sd (263 ± 14 → 372) and `rws` clears it too (218 ± 12 → 387 mute). So dropout keeps units alive
on its own AND adds to the sparsity penalty — both halves of Pavel's question, positive.

**The cost is essentially zero in the `none` arm.** `none`+`dead` holds 372 units instead of 263,
a 41% increase, at clean loss 0.02597 against the reference 0.02592. Same performance, 109 more
live units: redundancy, which is exactly what the subsampling goal needs. The counterexample is
`rws`+`dead`, which buys units but pays (loss 0.0676, r² 0.914). frm and both are already at N and
can only be read for cost, which is small.

![Dropout vs no dropout](../img/internal_figures/dropout_live_vs_iter.png)

**Caveat, from the figure, not the table:** dropout shifts the whole live-unit curve UP; it does not
flatten it. Both arms are still silencing at 150k at the same rate as their references. This is a
matched-iteration offset, not a converged difference — the same caveat as every other recruitment
number in this project. All 8 folder r² values reproduce from the saved weights (checked).

**One seed each.** Per the launcher's own pre-registration a positive result gets 3 seeds before it
is quoted. NOT submitted — awaiting Pavel.

### 2. TWENTY TASKS AT ONCE does not raise CDDM recruitment (job 6209433, 6 nets, 3 seeds) — CONFIRMED

`yang_30k_summary.py` (new, wraps `multitask_readout.readout` so both sides use one code path).
N=1000, 30k true iterations for all six nets (three were warm-started from the 10k pilots; they are
read at LastParams, so the comparison is matched).

| set | CDDM live (scale-free) | CDDM live (1e-6) | CDDM live (4e-2) | whole batch (scale-free) |
|---|---|---|---|---|
| ctxdm (CDDM alone) | 411.7 ± 3.9 | 959.7 ± 14.7 | 474.0 ± 11.0 | 379.7 ± 1.7 |
| multi20 (20 tasks) | 356.7 ± 3.3 | 950.7 ± 9.0 | 522.7 ± 11.4 | 418.7 ± 13.0 |

Shared / CDDM-private in the 20-task nets (scale-free): **355.3 ± 2.4 shared, 1.3 ± 0.9 private.**
Accuracy: ctxdm 1.000 on both rules; multi20 ≥ 0.979 on all 20 rules (lowest dnmc 0.979), so every
rule is solved by the pre-registered ≥ 0.95 bar.

**The pilot result holds with error bars, and the seed spread is tiny (± 4 units).** Training on 20
tasks does NOT give CDDM more units. Under the scale-free criterion the 20-task net uses FEWER
(357 vs 412, a 55-unit gap against ~4-unit seed spread).

**The criteria disagree in DIRECTION here, and that has to be reported.** Scale-free says multi20
uses fewer; absolute 4e-2 says multi20 uses MORE (523 vs 474); absolute 1e-6 says they are equal
(951 vs 960, both ~95% of N). The scale-free threshold is 5% of the net's own q95, so the 20-task
net's longer right tail (it has more very-active units on CDDM trials) raises its own bar. Honest
statement: on CDDM trials the 20-task network has more units above a FIXED activity level but fewer
relative to its own top percentile. No reading of the three supports "multi-task training makes
CDDM recruit substantially more".

Almost nothing is CDDM-private (1.3 of 357). The extra whole-batch recruitment (419 vs 380) is
spent on the other tasks. This is the nuisance-activity picture, not a more redundant CDDM code.

### 3. HYPER FLIP-FLOP grid complete at 72/72 — figure re-plotted

The last k=8 N=2000 cells landed (Spock 6173767/6173768). `flipflop_hyper_readout.py` re-run
locally; `data/hyper_readout_150k.txt` and `_seeds.txt` regenerated (the previous copies were diffed cell
by cell against the new ones, then discarded).
**Every previously present value is byte-identical**; the table only gained the missing cells
(hyper none k=8 N=2000 went 1 seed → 3; hyper both k=8 N=2000 0 → 3; plus plain `both` N=1000 cells
that were absent when the old table was made on Spock). `fig_hyper_readout.py` re-run.

![Hyper grid at 150k, complete](../img/internal_figures/hyper_readout_150k.png)

New cells: hyper `none` k=8 N=2000 **1616 ± 20 of 2000** (81%) at r² 0.695, against plain `none`
618 ± 318 at the same k, N and iteration — a 2.6× excess, and the largest absolute recruitment in
the project. hyper `both` k=8 N=2000 is 2000/2000 at r² 0.495. The demand ladder now runs
unbroken at N=2000: 389 / 672 / 1016 / 1616 live at k = 2 / 4 / 6 / 8. The plain k=8 N=2000
reference keeps its one outlier seed (1068 against 389 and 397), so its sd is large; that was
already in the old table and is unchanged.

### 4. DMTS 16-tau grid: the N=2000 cells land, and they are the strongest version of the escape effect

`dmts_readout.py` over the full grid (jobs 6201205/6201206, 36 nets, 3 seeds/cell). Re-run locally
after the sync and byte-identical to the Spock run. `fig_dmts_pen_curves.py` re-drawn from the
re-dumped `data/dmts_curves_150k.npz`, now three columns instead of two (N=2000 added).

![DMTS penalty arms: clean loss and live units vs iteration, all three sizes](../img/internal_figures/dmts_pen_curves.png)

| N | pen | live sf / 1e-6 / Otsu | L at 150k | t(clean r² ≥ 0.9), 3 seeds | unstable |
|---|---|---|---|---|---|
| 2000 | none | 580 ± 68 / 848 ± 2 / 705 ± 58 | 1.90e-04 | 6350 / 12720 / 29880 | 1.4% |
| 2000 | rws | 461 ± 29 / 1178 ± 38 / 786 ± 26 | 1.08e-03 | 9740 / 12380 / 13850 | 2.4% |
| 2000 | frm | 1998 ± 2 / 1998 ± 1 / 1998 ± 2 | 5.05e-04 | **2140 / 2140 / 2900** | 12.3% |
| 2000 | both | 1987 ± 18 / 1995 ± 7 / 1990 ± 14 | 3.12e-02 | 4700 / 5060 / 6160 | 13.7% |

Every frm seed leaves the no-memory plateau before every `none` seed, by 3–10×, and the margin is
larger than at N=500 (3×) or N=1000 (2×, overlapping seeds). **The rate penalty's escape advantage
grows with N**, as the 60k-iteration raw logs already suggested. The cost appears at N=2000 and is
new: frm and both lose the memory in 12–14% of late probes against 1–2% for none, and `both` ends
at clean r² 0.60 (L = 3.1e-2). Recruitment: unpenalised DMTS keeps 580 ± 68 live at N=2000
(scale-free), still falling at 150k.

### 5. DMTS 36-tau check (job 6201936, 2 nets): Pavel's prediction holds, in one seed each

`dmts_readout.py --sub DMTS_std_delay36` (a `--sub` flag was added; the read-out is otherwise the
same code, and each net's target variance and scoring window come from its own saved config).

- **`none` NEVER leaves the plateau.** Clean r² pinned at 0.6045 for all 150k iterations; best value
  ever reached 0.682; 100% of late probes above the r² = 0.9 level. It never finds the memory.
- **`frm` does.** Clean r² ≥ 0.9 first reached at 11 670; median clean r² 0.999 over the middle of
  training; 69% of the second half above 0.9; best 0.9998.

`fig_dmts_delay36.py` (new) draws it as clean r² rather than loss, because the prediction is about
r², with the raw probes behind the rolling median so the memory-loss episodes are visible and the
plateau level read off the `none` arm itself (r² = 0.605) rather than carried over from the 16-tau
grid, whose plateau loss is a different number.

![DMTS 36-tau delay: none never escapes, frm does](../img/internal_figures/dmts_delay36.png)

The right panel adds something the table does not show: the two arms differ in recruitment from the
very start. `frm` holds ~1000 of 1000 units live throughout, while `none` drops to ~550 by 3k and
then oscillates violently between ~250 and ~800 for the rest of training — it is thrashing on the
plateau, not settling.

This is the clean reversal Pavel predicted: at 16 tau both arms escape and the penalty only wins on
speed; at 36 tau the unpenalised network does not escape at all within 150k while frm does.
**One seed per arm** — the 16-tau `none` seeds at N=2000 spanned 6k–30k, so "never in one seed"
needs the other two seeds before it is quoted as a result.

Caveat on the frm net specifically: it collapses over its last ~2850 iterations (the final 285
probes sit at clean r² ≈ 0), so the SAVED 150k network is not the trained network. That is the
grid's own documented "the 150k last network is a lottery ticket on this task" (unstable = 30.9%
here). Report the trace, not the final net.

### ⚠ TaskDMTS CHANGED AFTER THE DMTS GRIDS WERE TRAINED — offline re-analysis of those nets is invalid

Found while trying to verify the 36-tau folder scores, and worth recording because it will bite any
future DMTS analysis. Rebuilding the task from the CURRENT code and re-running a saved DMTS net
gives a silent network: reconstruction MSE 8.0e-2 against the trainer's own final
`loss_clean_train` of 2.7e-5 for the same net, a factor of **2984**, with output magnitude 0.006
against a target magnitude of 0.086. The same reconstruction is exact for a flip-flop net (ratio
1.0×), and all 8 dropout nets' folder r² reproduce, so the reconstruction code is sound.

Cause: commit **51237d6** (2026-09-15 11:25) fixed the DMTS cue channel. The 16-tau grid
(commit 800ee30) and the 36-tau check (commit 98b57c1, 09:48) were both submitted BEFORE it, so
those networks were trained against a different input layout than today's `TaskDMTS` produces.

**Nothing here is corrupted and no DMTS result above is affected**, because `dmts_readout.py` reads
only the ParticipationTrace — live counts and `loss_clean_train`, both computed inside the trainer
during training against the then-current task. The target variance it rebuilds is also unaffected
(the fix moved an INPUT channel; targets depend only on the sample/match channels). But any future
analysis that loads a DMTS net and instantiates the task must pin the task version, or it will
silently score a well-trained network at r² ≈ 0.

## ▶ SUBMITTED: DMTS 36-tau `both` x3, and the dropout sweep to 7 seeds per condition — 2026-09-16 22:50

Pavel's two calls after the read-out. Both arrays on Spock.

**CODE ACTUALLY RUN: `6e43d53`**, which is what every job log prints (`code: ... @ 6e43d53`) and
what Spock's repo is checked out at. The launchers themselves were copied to Spock by `scp` and
md5-verified, not committed there (`slurm/` is gitignored on both sides); on the Mac they are
commit `882b980`. The training path is identical between `6e43d53` and the Mac's HEAD — the only
later changes are in `experiments_and_analysis/` and `docs/` — so the Mac commit describes the
launchers and `6e43d53` describes the networks.

Submitted with:

    sbatch --array=1-3     slurm/SilentReLU_dmts_delay36_both_spock.slurm
    sbatch --array=1-64%16 slurm/SilentReLU_flipflop_dropout_seeds_spock.slurm

Confirmed running at 22:52: all 3 DMTS tasks, 16 of 64 dropout tasks (the rest held by the %16
array task limit), no errors in any `.err`, and the `Task N:` line of every running job checked
against the intended (penalty, dropout kind, seed) decode.

**Job 6218720, `slurm/SilentReLU_dmts_delay36_both_spock.slurm`, array 1-3.** frm 0.1 + rws 0.05 at
a 36-tau delay, N=1000, 150k, three seeds, into the SAME `DMTS_std_delay36/` folder as the existing
`none` and `frm` seeds so `dmts_readout.py --sub DMTS_std_delay36` reads all three arms together.
Pre-registered: t(clean r² ≥ 0.9) per seed and the `unstable` fraction. All three escape → the rate
term is what finds the 36-tau memory. None escape → adding rws BREAKS the escape frm alone achieves,
a stronger statement than anything at 16 tau. Mixed → report as seed-dependent, claim no direction.
Read the TRACE, never the final net: the 36-tau frm seed finished inside a collapse.

**Job 6218721, `slurm/SilentReLU_flipflop_dropout_seeds_spock.slurm`, array 1-64%16.** Takes every
condition to SEVEN seeds: tasks 1-48 are 6 more seeds of each of the 8 dropout cells, tasks 49-64
are 4 more no-dropout controls per penalty arm, written to new `..._do=none` cells inside the
dropout sweep so they are same-launcher, same-commit controls. Throttled to 16 concurrent.
~1100 GPU-hours total; 16 running immediately, the rest pending on the array task limit.

### What was checked before submitting

1. **Index decode replayed for all 64 tasks in Python**: every one of the 12 conditions reaches
   exactly 7 seeds (1 existing + 6 for the dropout cells, 3 existing + 4 for the controls).
2. **Both launchers smoke-tested locally** at N=64 with `srun` shimmed and the module/conda lines
   stripped (tasks 1/5/49/64 and 1/3). Saved configs carry the right `dropout` flag, kind,
   sampling method, rate and lambdas; the `do=none` cells carry `dropout: false`.
3. **The flip-flop training path is BIT-IDENTICAL between the first dropout run's commit (1674662)
   and HEAD** — no dropout, `mute` and `dead` all reproduce to every printed digit over 15
   iterations. So the 7-seed pools mix commits legitimately. The `scored_` refactor (51237d6) does
   reduce to the old indexing when the task has no per-trial `batch_mask`, as its commit message
   claimed, and this now confirms it empirically rather than by reading the diff.
4. **Sanity: `mute` and `dead` still differ at HEAD**, i.e. dropout still reaches the gradient — the
   exact check that caught the original no-op bug.
5. Launchers copied to Spock and **md5-verified** against the local copies before `sbatch`.

**⚠ A test with no power, caught and fixed.** The first version of check 3 used `seed=1` alone.
Two runs at the SAME commit differed as much as the cross-commit pair, so it could not have detected
a real difference — it would have "passed" whatever the answer was. `task.seed=1` is required as
well to make a run deterministic (the task draws a fresh batch every iteration and its own seed
defaults to None). The project's own verification of the dropout bug used `seed=1 task.seed=1` for
this reason; re-deriving it the hard way is why it is recorded here. With `task.seed=1` two runs at
HEAD are bit-identical, so the test has power, and it then passed on all three paths.

### Read-out when they land

`flipflop_dropout_readout.py` now pools the new `do=none` control cells with the historical
reference seeds (`--controls-only` drops the historical side if the two ever disagree), compares
cell MEANS under the unchanged 3-sd rule, and prints a descriptive Welch test. The secondary
question, which one seed could not answer, is whether dropout changes the RATE of silencing or only
its offset: with 7 seeds, fit the late slope of the live-unit curve per condition.

### Access note

Spock and Della both became unreachable mid-session (empty `~/.ssh/sockets/`, then
`Permission denied (publickey)` on Spock and `(keyboard-interactive)` on Della) and came back when
Pavel restored the connection. Spock access still depends on his open ControlMaster session; Della
answered non-interactively earlier the same day and then did not, so that route is a cached-2FA
session, not a standing capability.

### PRELIMINARY (jobs still running, 10.5 h in) — 2026-09-17 09:19

Status: 19 running, 0 failed, 0 finished. DMTS 3/3 at ~85k/150k; dropout 16/64 at 65k–98k/150k with
48 held by the `%16` limit. Nothing below is final — the participation traces, and therefore every
live-unit number, are written only when a job ENDS, so the only mid-run observable is the noisy
training-batch r² the log prints each iteration.

**DMTS 36-tau `both` (6218720): all three seeds are still on the no-memory plateau at 85k.**

| seed | r² at 5k | 20k | 40k | 60k | 80k | max ever | first r² ≥ 0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 0.596 | 0.600 | 0.601 | 0.602 | 0.602 | 0.6036 @ 74316 | none yet |
| 2 | 0.595 | 0.601 | 0.602 | 0.603 | 0.602 | 0.6037 @ 84450 | none yet |
| 3 | 0.595 | 0.600 | 0.601 | 0.560 | 0.601 | 0.6037 @ 83128 | none yet |

Train loss is pinned at 0.0193, the plateau value. For comparison the `frm`-alone seed crossed
r² = 0.9 at **11 670** iterations, so `both` is already more than 7x past that and flat. If this
holds to 150k it is the "adding rws BREAKS the escape that frm alone achieves" branch of the
pre-registration, which would be a stronger statement than anything at 16 tau: at 16 tau `both`
escaped as fast as frm (t(r²≥0.9) 460–630 at N=500, 1110–2360 at N=1000). **Not yet quotable** —
65k iterations remain, and the 16-tau `none` seeds at N=2000 escaped as late as 29 880.

**Flip-flop dropout (6218721): all 16 running cells are training normally**, noisy train r² 0.920
to 0.952 at 65k–98k, ordered as expected (none ≈ 0.951 > rws ≈ 0.947 > frm ≈ 0.94 > both ≈ 0.93).
No live-unit numbers until the traces land.

One thing to watch: the ORIGINAL single `rws`+`dead` seed was the performance outlier of the first
grid (clean loss 0.0676, r² 0.914, against ~0.026 and ~0.95 everywhere else). Its two new siblings
are at noisy r² 0.949 and 0.949, i.e. normal. If that holds, the "rws + dead buys units but pays for
them" reading from the 1-seed table was a one-seed artefact, and the 7-seed table will say so.

**ETA.** Measured cost per 150k job: none+mute 16.4 h, rws+mute 16.4 h, none+dead 17.6 h, rws+dead
17.9 h, frm+mute 21.8 h, both+mute 22.0 h, frm+dead 23.4 h, both+dead 24.2 h; DMTS 36-tau 18.6 h.
- DMTS 6218720: all three finish **today, ~17:30**.
- Dropout 6218721: ~965 GPU-hours left over 64 jobs at 16 concurrent → **~2.5 days, i.e. late
  2026-09-19, realistically 2026-09-20** with imperfect packing. The 16 currently running finish
  between 5.6 h and 13.8 h from now.

**Decision (Pavel, 2026-09-17 09:40): keep all 64 dropout jobs.** The frm/both arms cannot move on
the primary outcome (they sit at 979–1000 of 1000 live units with and without dropout) and are 57%
of the remaining compute, but 7 seeds there sharpens the secondary question — whether dropout COSTS
performance under a rate penalty — and the first grid's frm+dead cell showed the largest loss rise.
Not cancelled.

**ETA** (list-scheduling simulation on 16 slots, per-job costs measured from the 16 running cells;
total array 1133 GPU-h):

| milestone | ETA |
|---|---|
| DMTS 36-tau `both`, all 3 seeds | Thu 17 Sep ~17:30 |
| first 16 dropout jobs | Thu 17 Sep ~23:10 |
| **none/rws complete — the primary outcome** | **Sat 19 Sep ~23:00** |
| whole array (frm/both included) | Sun 20 Sep ~04:00, ~17:00 with 20% slack |

Per-job cost (150k iterations, measured): both+dead 24.2 h, frm+dead 23.4, both+mute 22.0,
frm+mute 21.8, rws+dead 17.9, none+dead 17.6, none+mute 16.4, rws+mute 16.4; controls 9.4 h
(none/rws) and 12.4–12.7 h (frm/both). Dropout costs ~40% per iteration (two forward passes), and
the frm/both arms are a further ~35% slower than none/rws.

**The primary read-out does not need the whole array.** none/rws reach 7 seeds on Sat 19 Sep, ~5 h
before the ceiling arms; `flipflop_dropout_readout.py` can be run then, and the frm/both rows filled
in afterwards.

## ▶ RESULT: DMTS 36-tau, frm + rws — adding rws DESTROYS the escape that frm alone achieves (job 6218720) — 2026-09-18 14:35

All 3 seeds COMPLETED. This is the pre-registered branch "none escape → adding rws BREAKS the escape
frm alone achieves", and it is unanimous.

| arm | n | live sf / 1e-6 / Otsu | L at 150k | clean r² | t(clean r² ≥ 0.9) | t_match | unstable |
|---|---|---|---|---|---|---|---|
| none | 1 | 176 / 760 / 226 | 1.91e-02 | 0.603 | **never** | 11 890 | 100% |
| frm 0.1 | 1 | 1000 / 1000 / 1000 | 4.70e-02 | (see note) | **11 700** | 10 490 | 30.9% |
| frm + rws | 3 | 1000 / 1000 / 1000 | 1.92e-02 | 0.602 | **never / never / never** | 47 450 / 48 220 / 48 660 | 100% |

![DMTS 36-tau: frm alone finds the memory, frm + rws does not](../img/internal_figures/dmts_delay36.png)

**1. `both` is indistinguishable from `none`.** Final clean loss 1.92e-2 against none's 1.91e-2,
clean r² 0.602 against 0.603, and 100% of late probes on the plateau in both. Three seeds, spread
of 0.0014 in r². The rws term does not slow the escape down, it removes it.

**2. This reverses the 16-tau ordering.** At 16 tau `both` escaped as fast as frm — t(r² ≥ 0.9) of
460/550/630 at N=500 and 1110/1550/2360 at N=1000, both comfortably ahead of `none`. At 36 tau frm
alone still escapes (11 670) and `both` never does. So the two penalties are not additive on this
task, and whatever rws does to the solution is fatal once the delay is long enough. The 16-tau grid
could not have shown this: there the task is easy enough that every arm gets there.

**3. Recruitment and capability come apart, sharply.** The `both` nets hold **1000 of 1000 units
live under every criterion, for the whole of training**, and still never learn the task. The `none`
net, which does learn nothing either, ends with 176 live by the scale-free criterion. So on this
cell a network with 5.7x more live units performs identically. Keeping units alive is not
sufficient for capability — the frm term pins every unit above threshold whether or not those units
are doing anything. This is the cleanest counterexample in the project so far to reading a live-unit
count as a capacity measure, and it matters for how the dropout result (which raises live counts) is
phrased: "dropout keeps more units alive" must not be slid into "dropout gives the network more
usable capacity" without separate evidence.

**4. `both` is SLOWER even to reach the plateau-level loss.** t_match (first probe at or below
`none`'s final loss) is 47 450–48 660 for `both` against 11 890 for `none` itself — four times
longer to arrive at the same non-solution.

Note on the frm row: its folder r² of 0.954 is the end-of-training eval, but that net finished
inside one of its collapse episodes and its own trace has it at clean r² ≈ 0.07 over the last 2850
iterations. Read the trace; the final net is not the trained net on this task.

**Still one seed each for `none` and `frm`.** The `both` conclusion has 3 seeds; the frm comparison
it is measured against does not. Before this is quoted, frm and none need their other two seeds at
36 tau — 2 jobs, ~18 h each.

## ▶ INTERIM (dropout array 32/64 done): the result holds at 3 seeds, and one earlier claim is RETRACTED — 2026-09-18 14:40

All 16 controls finished; every dropout cell has 3 seeds (1 original + 2 new), which is exactly the
power the first launcher pre-registered. No job failed. Read at 150k, 3 dropout seeds against 7
no-dropout (4 new same-commit controls + 3 historical), `flipflop_dropout_readout.py`.

| pen | dropout | n | live scale-free | live abs 4e-2 | clean loss | verdict |
|---|---|---|---|---|---|---|
| none | — | 7 | 263 ± 13 | 291 ± 18 | 0.02589 ± 0.00005 | reference |
| none | mute | 3 | **377 ± 22** | 413 ± 20 | 0.02665 ± 0.00008 | KEEPS ALIVE |
| none | dead | 3 | **357 ± 11** | 390 ± 8 | 0.02589 ± 0.00021 | KEEPS ALIVE |
| rws | — | 7 | 220 ± 9 | 242 ± 10 | 0.02597 ± 0.00007 | reference |
| rws | mute | 3 | **393 ± 13** | 413 ± 18 | 0.02850 ± 0.00016 | KEEPS ALIVE |
| rws | dead | 3 | **327 ± 2** | 379 ± 5 | 0.0404 ± 0.0193 | KEEPS ALIVE |
| frm | — | 7 | 971 ± 17 | 971 ± 17 | 0.02770 ± 0.00013 | ceiling |
| frm | mute / dead | 3 | 984 ± 19 / 994 ± 7 | same | 0.02825 / 0.02997 | ceiling |
| both | — | 7 | 1000 ± 0 | 1000 ± 0 | 0.02905 ± 0.00006 | ceiling |
| both | mute / dead | 3 | 1000 ± 0 / 998 ± 0 | 1000 / 995 | 0.03041 / 0.03295 | ceiling |

Welch p on live_sf: none+mute 0.012, none+dead 3.9e-4, rws+mute 6.2e-4, rws+dead 1.2e-8.

![Dropout vs no dropout, 3 seeds per dropout cell](../img/internal_figures/dropout_live_vs_iter.png)

**1. `none` + `dead` is the clean result: 36% more live units at IDENTICAL clean loss.**
357 ± 11 against 263 ± 13, and clean loss 0.02589 ± 0.00021 against the reference's
0.02589 ± 0.00005 — Welch p = 0.97, i.e. no detectable cost at all. That is redundancy for free,
which is what the subsampling goal needs.

**2. RETRACTION: "rws + dead buys units but pays for them" was a ONE-SEED ARTEFACT.** The original
seed had clean loss 0.06764 (r² 0.914). Its two siblings are 0.02648 and 0.02711, i.e. at the
no-dropout level (0.02597). Per seed: 0.06764 / 0.02648 / 0.02711. The cell mean of 0.0404 ± 0.0193
is carried entirely by that one net and the Welch p is 0.4. The recruitment effect in the same cell
is meanwhile the tightest in the table (324 / 328 / 329 live, sd = 2). So rws + dead buys 107 units
and, in 2 of 3 seeds, pays nothing. The 2026-09-16 read-out said the opposite and was wrong — that
is what one seed buys you.

**3. Pooling the historical controls is VALID, checked rather than assumed.** The 4 new
same-commit controls against the 3 historical seeds, live scale-free: none 262 ± 12 vs 263 ± 14
(0.1 pooled sd), rws 221 ± 7 vs 218 ± 12 (0.3), frm 966 ± 21 vs 979 ± 7 (0.7), both 1000 vs 1000.
All within 2 pooled sd, so the bit-identity argument in the launcher is confirmed empirically and
`--controls-only` is not needed.

**4. mute > dead for recruitment, dead > mute for cost.** `mute` reaches higher live counts
(377/393 vs 357/327) but costs a small, statistically clear amount of clean loss (p = 0.012 and
0.0013); `dead` reaches fewer units and costs nothing measurable. Reasonable reading, not yet
tested: `mute` leaves the dropped unit driving the recurrent dynamics while removing it from the
output, so the network must keep more units usable; `dead` removes it entirely, which is a milder
constraint on the surviving population. A falsifying test would be `mute` at a lower drop rate
matching `dead`'s live count and asking whether its loss cost disappears.

**5. Unchanged: the ceiling arms cannot answer the primary question** (971–1000 live with and
without dropout) and their only movement is a small loss increase, largest for both + dead (0.03295
vs 0.02905).

**Still to come: seeds 4–7 of each dropout cell** (17 pending + 15 running at 14:35). Nothing above
is expected to move much — the 3-seed spreads are 2–22 units — but the rws + dead loss needs the
extra seeds to confirm that the 0.068 net is the outlier and not the other two.

### Equal seed count at 36 tau: 2 more `none` and 2 more `frm` (jobs 6257032, 6257033) — 2026-09-18 15:00

The 36-tau inventory was lopsided: 3 seeds on the arm that does NOTHING (frm + rws) and 1 seed each
on the two arms the conclusion rests on. Pavel: even it up. Submitted by running the ORIGINAL
2-arm launcher twice (`sbatch --array=1-2 slurm/SilentReLU_dmts_delay36_spock.slurm`, tasks 1 and 2
are `none` and `frm 0.1`), so the new seeds come from literally the same script as the first two —
no new launcher, nothing to re-verify. md5 checked identical on Mac and Spock first. Both arrays
PENDING at 15:00; Spock's nodes are all `mixed` and the dropout array holds its own 16 slots.

**`rws` alone at 36 tau deliberately NOT run.** Pavel: it will be worse than `none`. That is the
right call and the 16-tau grid already says so — rws alone was last to escape at every size
(6120–8050 at N=500, 6870–8000 at N=1000, 9740–13850 at N=2000, against `none`'s 2300–3750 /
1810–4730 / 6350–29880). A 36-tau rws-only arm would cost 18 h to confirm a prediction nothing
turns on. Recorded so nobody adds it later thinking it was an oversight.

### DISCUSSION (Pavel): the developmental reading — exuberant activity, then pruning

Pavel's framing of the 36-tau result: it looks like the developmental sequence — dense networks of
neurons, each cell's activity encouraged, pruned afterwards.

**⚠️ THIS IS INTERPRETATION, NOT A MECHANISM THE DATA ESTABLISHES.** Logged so it does not get
propagated as a finding (cf. [[no-propagating-untested-mechanisms]]). What is and is not supported:

*Supported, at 1 seed per arm until 6257032/33 land.* At a 36-tau delay the arm that holds every
unit above threshold (frm) finds a memory solution that the freely-pruning arm (`none`) never finds
in 150k. That is the core developmental claim in miniature: a regime of enforced widespread activity
reaches a solution that a network pruning from the start does not.

*NOT supported — three specific gaps.*
1. **We have never run the two-phase structure.** Development is dense THEN pruned. Our arms are
   dense THROUGHOUT (frm, live count pinned at 1000 for all 150k) versus pruning throughout (`none`,
   falling to 176). Nothing here tests whether a dense phase FOLLOWED BY release is better than
   either. The result is consistent with the story and does not test it.
2. **"Keep every cell active" is NOT sufficient, by our own data.** frm + rws also holds 1000/1000
   live under every criterion for the whole of training, in 3 seeds, and never learns the task at
   all. So a high live count does not deliver the benefit; whatever frm does that helps is not
   captured by "all units alive". Any developmental account has to say which property of the frm
   regime matters, and live-unit count is demonstrably not it.
3. **The dense solution is not a stable good solution.** The frm net spends 30.9% of its late
   probes back on the plateau and finished inside a collapse. If anything that is an argument for
   why a dense phase would need to be followed by consolidation — but that is a story, not a result.

**The experiment that would test it, and its falsifier.** Warm-start a frm net at iteration T into
an UNPENALISED continuation (frm → none), with the paired controls frm → frm and none → none, the
design already used by `slurm/SilentReLU_flipflop_switch_spock.slurm` (arms A1–A4: warm-start from
a parent, continue under a different penalty, same-penalty arms mandatory because warm-starting
resets Adam and adds iterations). The machinery exists — `paths.init_from`, quoted inside the Hydra
override because folder names contain ';' and '='.
- Developmental story holds → frm → none KEEPS the memory, its live count falls toward the `none`
  level (i.e. it prunes), and it ends better than pure `none` ever does.
- **Falsified if** frm → none loses the memory as soon as the penalty is released — then the dense
  phase is a crutch the solution cannot survive without, not a scaffold it is built on.
- Also informative: sweep T. If the benefit survives release only for large T, the "critical
  period" framing gains content; if it survives for any T, it is just initialisation.

Not submitted — this is a proposal, and the 36-tau seeds should land first so the comparison it
builds on has more than one seed per arm.

## ▶ RESULT (BALANCED, 3 seeds per arm): DMTS 36-tau — frm alone finds the memory 3/3, frm + rws 0/3 — 2026-09-20 19:20

Jobs 6257032 and 6257033 landed (2 more `none`, 2 more `frm`), so all three arms now have 3 seeds.
This supersedes the 2026-09-18 entry, whose `none` and `frm` arms were one seed each. All 7 jobs
across the three arms COMPLETED; none failed.

| arm | n | escaped (clean r² ≥ 0.9)? | t(escape), per seed | live sf | L at 150k | unstable |
|---|---|---|---|---|---|---|
| none | 3 | **0 / 3** | never / never / never | 128 ± 42 | 1.91e-02 | 100% |
| frm 0.1 | 3 | **3 / 3** | 11 700 / 12 720 / 18 800 | 997 ± 4 | 3.41e-02 | 46% |
| frm + rws | 3 | **0 / 3** | never / never / never | 1000 ± 0 | 1.92e-02 | 100% |

![DMTS 36-tau, 3 seeds per arm](../img/internal_figures/dmts_delay36.png)

**1. The result is unanimous in all three arms, and the single-seed version held.** frm escapes in
every seed (11.7k–18.8k), `none` in none of them, frm + rws in none of them. The worry recorded on
2026-09-18 — that the whole story rested on one frm net and one `none` net — is resolved: three for
three, and the spread of escape times (7k) is small against the 150k budget.

**2. Adding rws to frm converts a 3/3 success into a 0/3 failure.** The frm + rws nets are
indistinguishable from unpenalised ones: L 1.92e-2 vs 1.91e-2, clean r² 0.602 vs 0.603, 100% of
late probes on the plateau in both. This is not a slowdown, it is abolition. At 16 tau the same arm
escaped as fast as frm (460–630 at N=500, 1110–2360 at N=1000), so the ordering reverses entirely
once the delay is long enough — the penalties are not additive, and the 16-tau grid could not have
revealed it because there every arm succeeds.

**3. Live-unit count does not predict success — it anti-predicts it here.** frm + rws holds
1000/1000 units live and fails 3/3. `none` holds 128 ± 42 and fails 3/3. frm holds 997 ± 4 and
succeeds 3/3. So among the two arms that keep essentially every unit alive, one always solves the
task and the other never does. Whatever frm supplies is not "units above threshold", and a live
count cannot be read as capacity. This is now a 9-network result, not an anecdote.

**4. frm's solution is real but unstable.** 46% of late probes sit back on the plateau (per seed
0.309 / 0.466 / 0.605), and 2 of 3 seeds finished inside a collapse (L 4.70e-2 and 4.72e-2) while
the third ended in a good state (L 7.97e-3, clean r² 0.835). The best clean r² reached is 0.9998 in
all three. So the memory is found reliably and held unreliably. **The final network is a lottery
ticket on this task: read the trace, and if a net is needed, take the best checkpoint.**

**5. `none` is not merely slow, it is churning.** Its live count falls to ~550 by 3k and then
oscillates between ~130 and ~900 for the rest of training (right panel), with 100% of late probes
on the plateau. It is not converging on a poor solution; it is failing to settle at all.

**Relevance to the developmental framing** (logged 2026-09-18 as interpretation): gap 1 is
unchanged — these are still "dense throughout" vs "pruned throughout", never dense-then-pruned —
but gap 2 is now sharper, not weaker. "Encourage every cell's activity" is demonstrably NOT the
operative variable, since frm + rws does exactly that in 3/3 seeds and never learns. The
frm → none switch experiment proposed on 2026-09-18 is what would test the two-phase claim, and it
now has a 3-seed parent population to branch from.

## ▶ RESULT (FINAL, 7 seeds per condition): dropout keeps units alive, and it is free in the unpenalised arm — 2026-09-20 19:37

Array 6218721 complete: 64/64 COMPLETED, none failed, last job ended 2026-09-20 05:29 (the
2026-09-17 ETA said ~07:00). Every one of the 12 conditions is at 7 seeds — 8 dropout cells at 7,
4 control cells at 4 new plus the 3 historical reference seeds whose poolability was verified on
2026-09-18. 72 networks read at 150k.

| pen | dropout | n | live scale-free | live abs 4e-2 | clean loss | Welch p (live_sf / loss) | verdict |
|---|---|---|---|---|---|---|---|
| none | — | 7 | 263 ± 13 | 291 ± 18 | 0.02589 ± 0.00005 | — | reference |
| none | mute | 7 | **373 ± 18** | 406 ± 16 | 0.02670 ± 0.00010 | 1.4e-07 / 3.5e-08 | KEEPS ALIVE |
| none | dead | 7 | **362 ± 8** | 394 ± 7 | 0.02590 ± 0.00029 | 2.2e-08 / **0.95** | KEEPS ALIVE |
| rws | — | 7 | 220 ± 9 | 242 ± 10 | 0.02597 ± 0.00007 | — | reference |
| rws | mute | 7 | **391 ± 12** | 412 ± 14 | 0.02942 ± 0.00149 | 5.4e-12 / 0.0013 | KEEPS ALIVE |
| rws | dead | 7 | **324 ± 14** | 380 ± 6 | 0.03336 ± 0.01413 | 1.4e-08 / 0.25 | KEEPS ALIVE |
| frm | — | 7 | 971 ± 17 | 971 ± 17 | 0.02770 ± 0.00013 | — | ceiling |
| frm | mute | 7 | 991 ± 14 | 991 ± 14 | 0.02828 ± 0.00017 | 0.055 / 2.7e-05 | ceiling |
| frm | dead | 7 | 996 ± 5 | 996 ± 5 | 0.02983 ± 0.00102 | 0.012 / 0.0021 | ceiling |
| both | — | 7 | 1000 ± 0 | 1000 ± 0 | 0.02905 ± 0.00006 | — | ceiling |
| both | mute | 7 | 1000 ± 0 | 1000 ± 0 | 0.03043 ± 0.00009 | — / — | ceiling |
| both | dead | 7 | 998 ± 2 | 996 ± 3 | 0.03184 ± 0.00228 | 0.019 / 0.024 | ceiling |

![Dropout vs no dropout, 7 seeds per condition](../img/internal_figures/dropout_live_vs_iter.png)

**1. The headline survives 7 seeds and gets stronger.** All four `none`/`rws` dropout cells clear
the pre-registered bar (cell mean above the no-dropout mean by more than 3 no-dropout sd under BOTH
criteria), with Welch p from 5.4e-12 to 1.4e-7. `none` + `dead` is the clean one: **362 ± 8 live
against 263 ± 13, at clean loss 0.02590 ± 0.00029 against 0.02589 ± 0.00005, p = 0.95.** Ninety-nine
extra live units, no detectable cost, 7 seeds a side. That is redundancy for free.

**2. THE SECONDARY QUESTION IS ANSWERED, AND THE ANSWER IS ARM-DEPENDENT** (`dropout_silencing_slope.py`,
new). Live units lost per decade of iteration, fitted over [30k, 150k]:

| pen | no dropout | mute | dead |
|---|---|---|---|
| none | −210 ± 37 | −229 ± 47 (p 0.44) | −223 ± 23 (p 0.48) |
| rws | −268 ± 14 | **−97 ± 30** (p 9.4e-07) | **−150 ± 10** (p 2.6e-09) |

In the unpenalised arm dropout does **not** change the rate of silencing at all — it is a pure
OFFSET, exactly what the 1-seed figure suggested and could not establish. Under rws it **does**
slow silencing, by 2.7x for mute and 1.8x for dead. So dropout and the sparsity penalty interact:
against rws's pressure to shed units, dropout's pressure to keep them usable actually bends the
trajectory, while on its own it just starts the curve higher. The figure shows this directly — the
`none` column's curves are parallel, the `rws` column's visibly diverge.

**3. mute and dead dissociate: mute recruits more, dead costs less.** mute reaches the higher live
count in both arms (373/391 vs 362/324) and costs a small but utterly consistent amount of clean
loss (none: every one of 7 seeds between 0.02658 and 0.02691, against a reference of 0.02589,
p = 3.5e-08). dead reaches fewer units and, in the `none` arm, costs nothing (median 0.02575,
below the reference median). Plausible reading, NOT tested: `mute` leaves the dropped unit driving
the recurrent dynamics while removing it from the read-out, so the surviving population must stay
usable; `dead` removes it entirely, a milder constraint. Falsifying test: run `mute` at a lower
drop rate matched to `dead`'s live count and ask whether its loss cost disappears.

**4. The 2026-09-18 retraction is confirmed at 7 seeds.** rws + dead clean loss per seed:
0.02645 / 0.02648 / 0.02666 / 0.02682 / 0.02711 / 0.03237 / **0.06764**. Five of seven sit at the
no-dropout level, median 0.02682, Welch p = 0.25. The original single seed (0.06764) was the worst
of seven. The 2026-09-16 one-seed claim that "rws + dead buys units but pays for them" remains
wrong; the cell mean of 0.0334 ± 0.0141 is an artefact of that one net and should never be quoted
without the per-seed list.

**5. The ceiling arms behave as predicted and add only a loss cost.** frm and both move 971 → 991/996
and 1000 → 1000/998, i.e. nothing the primary question can use, while their clean loss rises
significantly (frm + mute p = 2.7e-05, frm + dead p = 0.0021, both + dead p = 0.024). Spending 57%
of the array's ~1130 GPU-hours here bought a clean "dropout costs a little performance when the
network is already fully recruited" and nothing else — as forecast on 2026-09-17 before the run.

**6. Still unresolved, and it is the same caveat as everywhere in this project:** every arm is STILL
silencing at 150k. The `none` + dropout curves are 110 units higher but falling at the same rate,
so these are matched-iteration counts, not converged ones. What the 36-tau DMTS result adds
(same day) is a reason not to over-read them: live-unit count is not capacity, since there a network
holding 1000/1000 live failed 3/3 while one holding 997 succeeded 3/3. **"Dropout keeps more units
alive" is established. "Dropout gives the network more usable capacity" is not, and needs a task
where the extra units can be shown to do something.**

### Process note: a sync-completion check that was wrong

The waiter used to decide "all data is local" was given a target of 68 traces when the correct total
is 72 (8 dropout cells x 7 + 4 control cells x 4). It fired early, and the first full read-out was
run with `rws`+`mute` at 3 of 7 seeds while every other cell had 7 — a silent, plausible-looking
under-count. It was caught by listing the per-cell counts against the cluster's own before
interpreting anything, and the table above is from all 72. **Check counts against the source, not
against a number typed from memory.**

## ▶ SUBMITTED: does the dropout recruitment effect hold at N=500 and N=2000? (job 6306913) — 2026-09-20 20:09

Pavel: repeat the unpenalised dropout cell at two more sizes. **42 jobs = 2 sizes x 3 dropout
conditions (none / mute / dead) x 7 seeds**, `slurm/SilentReLU_flipflop_dropout_sizes_spock.slurm`,
array 1-42%14. Self-contained: each size gets its own 7 no-dropout controls from the same launcher,
so unlike the N=1000 cell there is no historical reference and no cross-commit pooling argument.
Decode replayed for all 42 tasks (6 cells, 7 seeds each) and smoke-tested locally at N=48/64.

**What this adds over N=1000.** At N=1000 the effect is +99 to +110 live units (~+10% of N) and the
silencing SLOPE is unchanged (pure offset). Two things are only answerable with other sizes:
whether the gap is a constant COUNT (~+100 at both sizes) or a constant FRACTION of N (~+50 at 500,
~+200 at 2000); and whether "offset, not rate" is size-invariant or an N=1000 special case.

### ⚠️ CALIBRATION CAUGHT A BUG THAT WOULD HAVE KILLED 14 OF THE 42 JOBS

Before submitting, a 4-job 300-iteration calibration (6306867) measured the per-iteration cost at
both sizes. **The N=2000 dropout job FAILED in 28 s with CUDA OOM.** Root cause, from the traceback:
`Trainer.get_participation_`, the per-unit statistic driving `participation` dropout sampling, was
UNCHUNKED. At N=2000 its input is (2000, 307200) float32 = 2.5 GB, `x + eps` copies it, and
`torch.quantile` sorts — a further 6.9 GB request that failed on a 44 GB L40S with 41.8 GB already
in use. Every N=2000 mute/dead job would have died immediately.

The sibling `participation_from_states_` (the TRACKING path) was chunked over units years ago for
exactly this reason, with a docstring explaining it; the dropout path never was. Fixed in local
commit **74ec3dd**: same chunking, same 512 default, semantics untouched (this one reads raw
states, the tracking one applies the activation for equation_type h — they are deliberately
different quantities and the docstring now says so).

**Exactness verified two ways, not assumed.** `get_participation_` is bit-identical to the
unchunked form at chunk 512 / 128 / 37 on random states (max abs diff 0.0), and 15-iteration
same-seed runs (`seed=1 task.seed=1`, N=1000, k=3) are bit-identical before and after for BOTH
dropout kinds. So the existing 7-seed N=1000 dropout results remain directly comparable.

### Measured cost, and why the array is pinned to L40S

| cell | s/iter | 150k |
|---|---|---|
| N=500 no dropout | 0.165 | 6.9 h |
| N=500 dropout | 0.304 | 12.7 h |
| N=2000 no dropout | 0.471 | 19.6 h |
| N=2000 dropout | 0.886 | 36.9 h |

Whole array ~880 GPU-h, of which the N=2000 cells are 654. Wall time 60 h against a 36.9 h worst
case. **`--gres=gpu:L40S-46G:1`**: every rate above was measured on an L40S; A100 has run ~1.85x
slower on this task family (which would put N=2000+dropout near 68 h) and its 40 GB is below the
44 GB card the pre-fix job died on, so its post-fix headroom is unverified. The %14 throttle sits
under the 16 available L40S GPUs, so pinning costs no throughput — it only queues behind other
users' L40S jobs. Tasks 1 and 2 started immediately; the rest are held by the throttle.

### ⚠️ PROVENANCE: SPOCK IS RUNNING 6e43d53 PLUS ONE UNCOMMITTED FILE

The fix is in tracked code (`trainRNNbrain/trainer/Trainer.py`). It was copied to Spock by `scp` and
md5-verified; it was NOT pushed to the shared GitHub repo, which is Pavel's call, not something to
do unasked. So Spock's working copy is `6e43d53` with `trainRNNbrain/trainer/Trainer.py` modified
(27 insertions, 6 deletions), the logs print `code: ... @ 6e43d53`, and the launcher's own
dirty-repo check fires: "WARNING: ... has uncommitted changes; the commit above does not fully
describe this run." **That warning is accurate and should not be ignored when these runs are read.**
The Mac-side commit carrying the same file is 74ec3dd. Reconciling the two (push + fetch) would
make the recorded hash self-describing again.

Read-out when they land: `flipflop_dropout_readout.py --N 500` / `--N 2000` and
`dropout_silencing_slope.py --N ...` (both now take `--N`; they reproduce the N=1000 table
unchanged after that edit).

---

