## 2026-08-25 (night, 5) — frm/both re-submitted at 400k; large-N point sized by measurement

### Actions taken

1. **Code synced to Spock with the `storable_` saver fix** (version 95320a6). The frm/both launcher
   refuses to start on code lacking `storable_`, so no new run can write a .npz with dict-valued
   parameters silently discarded.
2. **Cancelled every PENDING frm and both task in the 150k grid** (44 frm + 47 both; the first count
   of "25 both" came from a truncated `squeue %K` field, and SLURM re-expanded bracketed ranges as
   neighbours were cancelled, so it took several passes). No compute was lost — none had started.
   **Running frm jobs (11) were left to finish**: the compute is already spent and the cells document
   the non-convergence. All rws cells were left untouched; rws converges inside 150k.
3. **Submitted `frm` + `both` at 400k**, k=1..8, N ∈ {500,1000,2000}, 3 seeds = 144 jobs
   (`SilentReLU_flipflop_frmboth_long_spock.slurm`, arrays 5904341/2/3), one array per size so the
   wall-time request matches the cell: 48h at N=500/1000, 96h at N=2000.

**Why 400k and not 500k.** 500k at N=2000 is 76h median but 103h on a 1.35x slow node, over Spock's
4-day cap — the failure mode the k-sweep header warns about. 400k also MATCHES the unpenalised N=2000
budget exactly, so floors can be fitted over an identical range instead of a mismatched one.

### The large-N point: what is affordable, and what is not

Cost model from the three measured points: `t = 0.163 + 9.667e-8 N^2` s/iter (batch 1024, fresh).

| N | lever | none+rws @100k | frm+both @ N-scaled converged budget |
|---|---|---|---|
| 3000 | 6x | 39h slow — Spock | 116h slow — Della, fits |
| 3500 | 7x | 51h slow — Spock | 147h slow — **just over Della's 144h cap** |
| 4000 | 8x | 64h slow — Spock | 186h — impossible |
| 5000 | 10x | 97h slow — Della | 271h — impossible |

**The full request (all k, all four penalties, N=5000) is 21 225 GPU-h ≈ 68 days at 13 concurrent
GPUs, and its frm/both cell cannot run at any converged budget on any available QOS.**

⚠️ **Large-N penalty cells are DEFERRED, and not only on cost.** We do not yet know what budget frm
actually needs — 400k is an extrapolation from "it had not settled by 150k". Submitting N≥3000 frm
before the 400k re-run reports back would risk repeating exactly the mistake that produced the
retracted 150k floors. The correct order is: let 400k establish frm's real settling time, then size
the large-N penalty cells from a measurement.

**What the large-N point is FOR** is the ceiling question, and that needs only `none` and `rws`. Over
N=500..2000, M ~ N^0.44 with b constant — a power law with NO ceiling — whereas CDDM's fit put a
ceiling near 880. Extrapolating M = 18.9 N^0.44 predicts M = 594 at N=3000 and 741 at N=5000, so a
ceiling near 800 only starts bending the curve above N ~ 4000. **Larger N is the discriminating
measurement, not a precision gain**, which argues for 4000-5000 over 3000.

`SilentReLU_flipflop_bigN_spock.slurm` written (48 jobs = none+rws x k=1..8 x 3 seeds at one size,
100k iterations, `BIGN` selects the size). ⚠️ 100k is sized for the diffusion and matched-performance
read-outs (which land at 29k-48k at this size) and is NOT deep enough for a floor comparison against
the 400-500k unpenalised cells.

**Timing calibration submitted at N = 3000, 4000, 5000** (600 iterations each) rather than trusting
the model, which mispredicts the measured N=500 point by 25%. Submission of the real grid waits on it.
⚠️ SUPERSEDED — the calibration found a CUDA OOM at N>=4000, not a timing number. See "night, 6".

### N=4000 submitted (array 5904416)

`BIGN=4000 sbatch --array=1-48 --time=96:00:00 --mem=48G SilentReLU_flipflop_bigN_spock.slurm`
= none + rws × k=1..8 × 3 seeds, 100k iterations. Index decode verified by hand before submitting
(task 1 → k=1/none/rep0; task 25 → k=1/rws/rep0; task 48 → k=8/rws/rep2).

**Wall time requested at Spock's full 96h cap deliberately.** The model predicts 47h median / 64h on
a 1.35x slow node, so 96h survives a **2x cost-model error** — and the model is only anchored by
measurements up to N=2000. Under-running costs a full re-run from scratch (no checkpoint/resume);
over-requesting costs nothing but queue priority. Memory raised 32G → 48G: activations at N=4000,
batch 1024, T=300 are ~4.9 GB and the older request had no headroom.

The three timing calibrations (N=3000/4000/5000, 600 iterations) are still queued behind the 144
frm/both jobs. They are now a confirmation rather than a blocker, and will be checked against the
first real N=4000 job's reported s/iter.

**Queue after all submissions:** 36 FFpen running (rws + 11 frm at 150k, left to finish), 144 FFpenL
(frm/both @ 400k), 48 FFbigN (N=4000), 3 calibrations, 1 FFk.

---

# STATE OF PLAY — 2026-08-25, 17:45

A standalone summary of what the flip-flop line has established, each claim with the figure that
supports it, plus what is currently in the queue. Everything below is the fresh-batch sweep; the
`same_batch=True` data remains retracted and quarantined.

---

## 1. The headline: active units scale with SIZE, not with task COMPLEXITY

Matched at the point where the input weights stop moving directionally:

| read-out | b (size) | c (complexity) | M(k=8)/M(k=1) |
|---|---|---|---|
| fixed compute, t = 150 000 | 0.405 [0.377, 0.434] | 0.179 [0.150, 0.208] | 1.45× |
| **matched W_inp diffusion** | **0.441 [0.403, 0.477]** | **−0.026 [−0.078, 0.021]** | **0.95×** |

![Decisive figure](../img/internal_figures/flipflop_decisive.png)

`b` = exponent on N, `c` = exponent on k in `M = A·N^b·k^c`. b ≈ 0.44 means quadrupling N buys only
1.8× the active units — sublinear, so added units are increasingly not recruited. c ≈ 0 means 8× the
task complexity changes the count by −5% [−15%, +4%].

**This is the pre-registered signature of pathology.** The k-sweep launcher fixed in advance: *"if M\*
tracks task demand it must move with k... Flat M\*(k) would mean a genuine pathology rather than
spare capacity."* M\*(k) is flat. The spare-capacity reading predicted the opposite and fails.

---

## 2. The result does not depend on WHEN the networks are read

Five comparison criteria, and the k-effect appears only in the two compute-based ones:

![Law under five criteria](../img/internal_figures/flipflop_fig5_law.png)

| criterion | b | c | verdict |
|---|---|---|---|
| endpoint (budgets differ ⚠️) | 0.39 | 0.137 [0.075, 0.222] | law REJECTED — mixed depths |
| fixed compute | 0.41 | **0.179 [0.152, 0.208]** | law OK |
| matched loss L\* | 0.43 | −0.023 [−0.064, 0.019] | law OK |
| matched excess over own floor | 0.41 | −0.038 [−0.074, 0.002] | law OK |
| matched dynamics | 0.46 | 0.001 [−0.045, 0.050] | law OK |

Read-out times, and how each criterion moves the answer:

![Criterion comparison](../img/internal_figures/flipflop_fig4_criteria.png)
![Active units under four criteria](../img/internal_figures/flipflop_fig3_active.png)

⚠️ `endpoint` is disqualified: it reads N=2000 at 400k and the other sizes at 500k, mixing depths and
manufacturing per-cell structure no smooth law can absorb. With budgets equalised, the identical law
on the same data fits at p = 0.77 instead of being rejected at p = 1.7e-08.

---

## 3. Nor on WHICH dynamical variable, or WHICH threshold

Thirteen trusted (variable, threshold) combinations spanning read-out times from 16k to 73k:

![Diffusion robustness](../img/internal_figures/flipflop_diffusion.png)

**b = 0.465 ± 0.025, c = −0.009 ± 0.030.** The variables settle in a clear order — W_out freezes by
~5k, the participation vector by ~25k, W_rec at ~19–41k, and **W_inp last at ~35–122k**, which is why
W_inp is the criterion of record.

⚠️ `W_out` and `M(t)` are unusable (already below every threshold at the first probe — no directed
phase to transition from). `W_inp` and the participation vector fail the single-power-law check
(per-decade α disagreement 0.302 and 0.204 vs W_rec's 0.091), so W_rec is the best-*founded* variable
even though W_inp is the longest-lived.

**The mechanism, measured:** the settling time itself obeys `T ~ N^(−0.16)·k^(0.458)`. Reading at a
k-compensated iteration `t ∝ k^0.458` — with γ from the WEIGHT dynamics, never from M — removes
**100%** of the fixed-compute k-effect (c: 0.179 → 0.001) while leaving b untouched. γ is bracketed:
too little compensation leaves c > 0, too much drives it negative.

---

## 4. The loss floor: exact, N-independent, and √k in complexity

![Floor vs complexity and size](../img/internal_figures/flipflop_fig2_floor.png)

`floor_per_channel(k) = a + b√k` with **a = 0.0205, b = 0.0030**, residuals at the seed-noise level,
and a and b constant across a 4× size range — so the floor is a **task property, not a capacity
limit**. Fits come from a stretched exponential:

![Stretched-exponential fits](../img/internal_figures/flipflop_fig1_curves.png)

⚠️ `A` and `τ` in `L∞ + A·exp(−(t/τ)^β)` are **not individually identified** once τ falls below the
fit's start — compare through `L∞` and `excess_time`, never A or τ alone.

---

## 5. "Silent" needs a task-calibrated threshold; `p < 1e-6` does not travel

![Task-calibrated threshold](../img/internal_figures/flipflop_hard_threshold.png)

The flip-flop's participation is bimodal with its silent mode at ~1e-3 and active mode at ~1e0.
`1e-6` — calibrated on CDDM, whose silent mode is exactly 0 — sits below **both**, reports ~0%
silence, and makes the whole law degenerate (`M = N`, b = 1.00, nothing measured):

![Degenerate under 1e-6](../img/internal_figures/flipflop_fig5_law_hard.png)

Derived from the data by Otsu's method on log participation, the flip-flop threshold is **4e-2**
(three adoption checks passed: spread 0.50 decades, agreement with scale-free 0.030, 41 000× larger
than 1e-6). Under it every conclusion is unchanged:

![Law under the calibrated threshold](../img/internal_figures/flipflop_fig5_law_abs.png)

⚠️ Otsu (4.1e-2) and the scale-free rule's `0.05·q95` (1.13e-1) differ by 2.7× and are **uncorrelated
across nets (r = 0.13)**, yet classify the same units (|ΔM|/N = 0.030) — because the valley between
the modes is nearly empty. That insensitivity is the strongest evidence the bimodal split is
structural rather than an artifact of either rule.

---

## 6. CDDM's exact zeros are real, and CDDM saturates under the same battery

![Held-out coherence test](../img/internal_figures/silence_is_taskset.png)

Hypothesis that CDDM's 72–83% exactly-zero units were an artifact of measuring on the same 450 frozen
conditions: **FALSIFIED.** Re-measured on 392 unseen coherence midpoints, the hard-silent fraction
moved 0.621 → 0.624 (−0.5%, inside the pre-set 5% band). The units are dead for any input.

![CDDM under the same criteria](../img/internal_figures/cddm_criteria.png)

CDDM saturates under every usable criterion (**b = 0.52–0.60**) against the flip-flop's 0.41–0.46.
⚠️ The `drift` criterion does **not** port to CDDM — α is already ~0.5 by the second probe, so
`common.diffusive_onset` now requires a prior directed phase and correctly returns nan there.

---

## 7. Open questions, stated as such

- **Is there a ceiling?** b = 0.44 is constant over 500–2000, i.e. an *unbounded* power law, whereas
  CDDM's fit put a ceiling near 880. Extrapolating `M = 18.9·N^0.44` gives 673 at N=4000 — a ceiling
  near 800 only bends the curve above N ≈ 4000. **The N=4000 run now in flight is the discriminating
  measurement.**
- **The N × k interaction is unresolved.** Under matched loss the k-trend is +7% at N=500 but −15% at
  N=2000; a single `c` averages across that sign change. Needs more seeds, not more cells.
- **An unexplained offset:** diffusion read-outs give b ≈ 0.465, loss-based ones 0.41–0.43. The gap
  exceeds the spread within either family and does not track read-out depth.
- **What budget does frm actually need?** 400k is extrapolated from "had not settled by 150k".

---

## 8. Jobs in flight (2026-08-25 17:45)

| array | job | count | configuration | purpose |
|---|---|---|---|---|
| 5904416 | **FFbigN** | 48 | N=4000, none+rws, k=1..8, 3 seeds, 100k, 96h/48G | **the ceiling test** |
| 5904341/2/3 | **FFpenL** | 144 | frm+both, N ∈ {500,1000,2000}, k=1..8, 3 seeds, **400k** | replaces the unconverged 150k frm |
| 5749772/3/4 | FFpen | 36 running | rws @150k (converged, kept) + 11 frm @150k (left to finish) | rws cells stand |
| 5904289/90/91 | FFcal | 3 | N=3000/4000/5000, 600 iters | timing confirmation |
| 5688852 | FFk | 1 | k=7 N=2000, 3rd seed | 2 seeds already in hand |

**Cancelled:** 44 pending frm + 47 pending both at 150k (no compute lost — none had started).

### What each will settle

- **FFbigN (~1–2 days):** whether b falls below 0.44 at N=4000 (real ceiling) or holds (unbounded
  power law). This is the one open question the current data cannot answer.
- **FFpenL (~2–4 days):** first check is whether 400k actually converges frm — by BOTH measures
  (α reaching 0.6, loss flat over the final third). Large-N penalty cells stay deferred until that
  reports, so they are sized from a measurement rather than a second extrapolation.

⚠️ **Deferred and why:** frm/both at N ≥ 3000 needs ~290k iterations at 1.35 s/iter = 147 h on a slow
node, against Della's 144 h cap — and the full request (all k, all four penalties, N=5000) came to
**21 225 GPU-h ≈ 68 days** at 13 concurrent GPUs. The ceiling question needs only none+rws, so that
is what was submitted.

---

