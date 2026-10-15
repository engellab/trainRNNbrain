## 2026-08-18 — Decay-fit parameters vs size: the floor RISES slightly with N

Figure: `img/internal_figures/fit_params_vs_N.png`, from
`trainRNNbrain/experiments_and_analysis/fit_params_vs_N.py`. Every size fitted on the SAME window
t ∈ [2000, 200000] (N=2000 truncated from its 300k), because the fitted parameters drift with fit
range and mismatched ranges have already produced three spurious "trends" in this project.

### The parameterisation to use: stretched exponential with τ fixed

`L = L_∞ + A·exp(−(t/τ)^β)` fits better than the power law, but **τ is unidentifiable**: profiling it
over four orders of magnitude (0.1 → 1000) changes the residual sum of squares by under 10% while A
slides by 14× to compensate. Freeing it buys no fit quality and destroys interpretability — free-τ
fits gave β = 0.14–0.27 with τ spanning 0.24 to 258 across seeds of the same condition.

**Fixing τ = 1** leaves `L = L_∞ + A·exp(−t^β)`, three parameters — the same count as the power law,
so the comparison is fair — and pins β to ±0.001. On that footing the stretched form wins
**decisively**: ΔAICc = +6.0 / +7.3 / +22.5 / +9.5 for N = 100 / 500 / 1000 / 2000, and it also wins
out-of-sample (fit on t ≤ 50k, score on 50k–200k) with **3× lower** RMS log-error at every size. So it
is the better shape, not merely the more flexible one — which was the obvious objection and is now
answered.

### Parameters

| | N=100 | N=500 | N=1000 | N=2000 | spread over 20× in N |
|---|---|---|---|---|---|
| L_∞ | 0.02188 ± 0.00001 | 0.02189 ± 0.00000 | 0.02198 ± 0.00004 | 0.02207 | **0.9%** |
| A | 0.177 ± 0.010 | 0.183 ± 0.006 | 0.212 ± 0.005 | 0.216 | 19.8% |
| β | 0.1574 ± 0.0013 | 0.1562 ± 0.0008 | 0.1582 ± 0.0012 | 0.1570 | **1.3%** |

**β is flat** — the decay *shape* is size-independent, a stronger statement than the floor alone.

**L_∞ rises monotonically with N**, 0.021879 → 0.022068. Against per-seed sd of 0.00001–0.00004 that
is roughly **5–10 σ**, so it is a real within-family effect, not scatter. Under the power law the same
quantity looks flat and non-monotone only because its error bars are 3–10× larger.

### Why the direction matters, and why it is still "roughly independent"

**Larger networks have a slightly HIGHER floor.** That is the safe direction. The hazard for the
matched-loss protocol was always the opposite — if big networks secretly reached a *lower* floor they
would be under-trained at equal loss, M would be inflated, and the no-saturation conclusion would be
an artefact. This rules that out with the sign, not just with a bound.

It also agrees with two independent observations already on record: no larger network ever reaches a
lower loss (0.02211 / 0.02222 / 0.02225 / 0.02227), and the deterministic error rises with N.

**But the effect is small and must not be over-read.** 0.9% across a 20-fold size range, while the
*family* disagreement is larger: power law gives L_∞ ≈ 0.0214, stretched ≈ 0.0219, a gap of 0.0005
that is ~3× the size effect. Which functional form you assume moves the floor further than network
size does. Treating L_∞ as common across sizes therefore remains justified.

### Consequence: a common threshold at L* = 0.025

With a shared floor, one loss level can be used for every size. **Iterations needed to reach
L = 0.025 stably** — "stably" meaning the last time the smoothed loss is above it, using a centred
2001-iteration mean over valid windows only (zero-padded edges otherwise fake a crossing at
iteration 1):

| N | first crossing | stable crossing | raw first dip | fraction of raw iterations below |
|---|---|---|---|---|
| 100 | 7,227 ± 263 | **7,234 ± 259** | 1,015 ± 74 | 0.87 |
| 500 | 8,516 ± 137 | **8,529 ± 126** | 974 ± 170 | 0.87 |
| 1000 | 9,553 ± 288 | **9,579 ± 306** | 1,509 ± 196 | 0.86 |
| 2000 | 11,417 | **11,416** | 1,159 | 0.86 |

First and stable crossings coincide to within a few iterations — once the smoothed loss goes below
0.025 it stays there — so the threshold is unambiguous.

**The raw trace is not.** A single noisy iteration dips below 0.025 at ~1,000 iterations, **7× earlier**
than the network is actually there, and even after the stable crossing only ~86% of raw iterations sit
below the line. Any criterion applied to the unsmoothed loss would fire 7× too early; this is the same
noise-lottery failure mode that invalidated the earlier "lowest observed loss" statistic.

**Budget: ~7k–11k iterations**, rising gently with N (58% from N=100 to N=2000), which is the expected
consequence of the smaller learning rate at larger sizes. Cheap compared with the 200k budgets already
spent — but note this reads the networks very early, before most silencing has developed (see the
matching-level ladder: at shallow levels the M(N) exponent is k ≈ 0.9, at deep levels k ≈ 0.26).

---

## RESULT: silent-unit fraction at matched performance (L* = 0.023)

Figure: `img/internal_figures/silent_at_threshold.png`, from
`trainRNNbrain/experiments_and_analysis/silent_at_threshold.py`.

Every size is read at its own T_N, the iteration at which its loss **stably** reaches L* = 0.023.
"Stably" = the last time the smoothed loss is above the threshold, using a centred 2001-iteration
mean over valid windows only. This matters: the per-iteration loss comes from a noisy forward pass,
and a single favourable draw touches any threshold ~7× earlier than the network actually gets there,
so a criterion applied to the raw trace fires far too early.

The comparison is licensed by the floor result: L_∞ is common across a 20-fold size range to within
0.9%, and what difference exists runs the SAFE way — larger networks sit slightly higher, so they are
not secretly under-trained at equal loss.

### The headline

| N | T_N | silent, hard | silent, scale-free | ACTIVE (hard) |
|---|---|---|---|---|
| 100 | 32,034 ± 2,208 | **0.7 ± 0.5%** | 18.3 ± 1.7% | 99.3 ± 0.5 |
| 500 | 37,556 ± 1,090 | **26.1 ± 1.5%** | 41.5 ± 1.3% | 369.3 ± 7.7 |
| 1000 | 44,917 ± 541 | **50.8 ± 1.9%** | 62.6 ± 2.2% | 492.3 ± 19.0 |
| 2000 | 56,859 | **69.0%** | 75.3% | 620.0 |

**At equal task performance, a 2000-unit network has 69% of its units silent while a 100-unit network
has 0.7%.** Same task, same loss, a hundredfold difference in the dead fraction. Budgets are
32k–57k iterations — well inside the 200k runs, and only 1.8× spread across 20× in N.

### Level dependence: the effect is real at every threshold, but its size is not

Silent fraction, hard criterion:

| N | L*=0.025 | L*=0.024 | L*=0.023 |
|---|---|---|---|
| 100 | 0.7% | 0.7% | 0.7% |
| 500 | 5.2% | 9.9% | 26.1% |
| 1000 | 14.6% | 27.7% | 50.8% |
| 2000 | 34.3% | 49.9% | 69.0% |

Monotone in N at every level, so the ORDERING is robust; the magnitude roughly triples from 0.025 to
0.023. L* = 0.025 is too early to characterise the phenomenon. N=100 is pinned at 0.7% under the hard
criterion at all three levels because its steep silencing phase does not begin until ~5×10⁴.

### The M* question, and the practical argument

Active-unit counts give local exponents `M ∝ N^k`:

| pair | 100→500 | 500→1000 | 1000→2000 |
|---|---|---|---|
| k | 0.816 | 0.415 | **0.333** |

Still falling, so **not yet decisively saturating** — but already so shallow that the distinction
barely matters practically. At k = 0.333:

> **Doubling the active-unit count requires an 8-fold larger network.**

That is the argument for penalties in its strongest form, and it does not depend on settling the
saturation question. Whether M(N) truly plateaus or merely crawls, buying active units by growing the
network is prohibitively expensive — and it comes with the second cost already established: an
unpenalised network never settles, so there is no principled iteration at which to read the count
(silencing still moves 4–10 percentage points per budget doubling at every size).

### What the pending runs decide

Prediction for N=5000 at this threshold, extrapolating the last measured k: **M ≈ 841** (620 × 2.5^0.333).
Saturation would put it near 620–700. The two remaining N=2000 seeds (~19:10 today) give that point
real error bars; N=5000 (first seed ~00:30, remainder ~22:30 Aug 19) extends the ceiling-free range
from 4× to 10× and settles whether k keeps falling. This section is to be updated when both land.

---

# 2026-08-18 16:09 — RUNNING JOBS AND WHERE THEIR RESULTS WILL LAND

Written as a hand-off: what is in flight, on which cluster, what each is meant to establish, and
which script reads it. Three separate sweeps are running across two clusters.

## The question all of this serves

Silent units are not a niche artefact — at matched task performance a 2000-unit network has **69%**
of its units silent while a 100-unit network has **0.7%** (L* = 0.023). Two things follow, and the
running jobs are meant to nail both down:

1. **Does the active-unit count M saturate with N?** Local exponents `M ∝ N^k` are k = 0.82 → 0.42 →
   0.33 across the measured pairs — still falling, not yet decisively flat. Even if it never
   saturates, k ≈ 1/3 means **doubling the active count costs an 8-fold larger network**, which is
   the practical argument for penalties regardless of how the limit behaves.
2. **Do frm/rws penalties fix it?** Both the original claim (more units stay active) and a sharper
   one discovered along the way: an unpenalised network **never settles** — silencing still moves
   4–10 percentage points per budget doubling at every size, so there is no principled iteration at
   which to read the count. If penalties make that rate decay to zero, "you cannot say when to stop"
   becomes a measured contrast rather than a complaint.

## In flight

### A. SPOCK — drift sweep, no penalties (arrays 5660821, 5663153)

| cell | job | state at 16:09 | lands |
|---|---|---|---|
| N=2000 seeds 2–3 | 5660821_11,12 | running, 23.7 h | **~19:10 today** |
| N=5000 × 3 | 5663153_13,14,15 | running, 15.8 h | seed 1 ~00:30 tonight; seeds 2–3 ~22:30 Aug 19 |

Output: `~/trainRNNbrain/data/trained_RNNs/CDDM_std_g0_drift/EqType=h_N=<N>_iters=<I>/`
Purpose: complete the M(N) curve. N=2000 currently rests on ONE seed, which carries the high end of
every regression in the saturation analysis; N=5000 extends the ceiling-free range from 4× to 10×.
**Pre-registered:** at L* = 0.023, extrapolating the last k predicts M(5000) ≈ 841; saturation would
give 620–700.

### B. SPOCK — penalty sweep (array 5664681, 18 jobs)

2 sizes (500, 1000) × 3 conditions (rws / frm / both) × 3 seeds, 200k iterations.
`lambda_rws = 0.05`, `lambda_frm = 0.1` — **half** the configured 0.2, because a smoke test at 0.2
drove silent units 5 → 0 within 150 iterations, i.e. it saturates and would show the penalty works
while revealing nothing about the dynamics.
Output: `CDDM_std_g0_penalties/EqType=h_N=<N>_pen=<rws|frm|both>/`  Lands **~00:30 tonight**.
Code: `~/trainRNNbrain_pen` (rsync, provenance in `CODE_VERSION.txt`, commit 4c68331).

**These are the first runs carrying the new monitoring**: `loss_clean_train` (masked MSE from the
noise-free probe, free — that forward pass already happened) and `loss_clean_valid` (same, on a
held-out batch of interleaved coherences, cadence 50). Everything before this recorded ONLY the noisy
training loss on a single fixed batch, with no validation anywhere in the pipeline.

### C. DELLA — N=10000 (array 12599054, 3 seeds) — NEW CLUSTER FOR THIS PROJECT

Pending at submission; 80,000 iterations, ~5.2 days each once started.
Output: `~/trainRNNbrain/data/trained_RNNs/CDDM_std_g0_N10k/EqType=h_N=10000_iters=80000/`
Code: `~/trainRNNbrain_N10k` (rsync, commit 24a0847).
Purpose: the largest point on the M(N) curve, spanning **100 → 10,000, a factor of 100**.

## Reading the results

| script | what it produces |
|---|---|
| `silent_at_threshold.py` | THE headline: silent % at each size's stable crossing of L* |
| `plot_M_vs_N.py` | M(N) at every rung of the matching ladder, both criteria |
| `test_saturation.py` | formal saturating-vs-power-law test, curvature + AICc + bootstrap k |
| `plot_silencing_trajectory.py N` | silent count vs iteration, and rate per budget doubling |
| `plot_loss_curves.py N` | loss with power-law and stretched fits, plus residuals |
| `fit_params_vs_N.py` | fit parameters vs size at matched fit range |
| `test_floor_vs_N.py --logs=DIR` | floor-vs-size test; `--logs` recovers UNFINISHED runs |
| `eval_heldout_loss.py`, `eval_noisefree_loss.py` | final-parameter evaluation, clean and held-out |

**Unfinished runs are usable for anything needing only the loss**: the per-iteration loss is printed
to SLURM stdout and parsing it reproduces the saved JSON to 5e-7. `losses_from_logs()` in
`test_floor_vs_N.py` does this and skips runs that already completed.

## Three traps that have each bitten this project more than once

1. **Mismatched fit ranges manufacture trends.** Fitted L_∞ and γ both drift with the length of the
   window they are fitted on. Comparing N=2000 (300k) against others (200k) produced a spurious
   monotone γ, and earlier a spurious "N=100 has a lower floor". Always truncate to a common t_max.
2. **The recorded loss is NOISY and single-batch.** Its minimum is a noise lottery — a single
   favourable draw touches any threshold ~7× earlier than the smoothed loss. Every threshold must be
   applied to a smoothed trace (centred 2001-iteration mean, VALID windows only; zero-padded edges
   fake a crossing at iteration 1).
3. **A timeout loses everything.** `ParticipationTrace.pkl` is written only when a run completes. This
   is why N=10000 is 80k iterations rather than the requested 100k: 100k needs 6.5 days against
   Della's 6-day `gpu-long` cap, and losing a 5-day job to a timeout costs more than the extra
   iterations buy.

## Cluster notes (differ between the two, and cost a failed submission each)

- **Spock**: `--partition=all` REQUIRED. GRES names are `gpu:A100-40G:1` / `gpu:L40S-46G:1`
  (bare `gpu:L40S:1` is rejected). No wall-time limit. `module load anacondapy/2024.02`.
- **Della**: `--partition` is FORBIDDEN ("You specified a partition of gpu. This is not allowed").
  Use `--constraint=gpu80` for A100-80GB and `--qos=gpu-long` for up to 6 days (gpu-short 1 d,
  gpu-medium 3 d). `module load anaconda3/2024.6`. `$HOME=/home/pt1290`.
- **N=10000 fits ONLY on Della**: 62.7 GB reserved, versus 47.7 GB on Spock's largest card.
- Both worktrees are populated by **rsync, not git checkout**, so their git HEAD is stale — the
  launchers read `CODE_VERSION.txt` instead and abort if it is missing.

---

