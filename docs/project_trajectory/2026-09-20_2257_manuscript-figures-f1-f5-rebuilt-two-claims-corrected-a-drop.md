## 2026-09-20 22:57 — Manuscript figures F1–F5 rebuilt; two claims corrected; a drop-rate ladder submitted

Pavel asked for the paper to be written properly: a motivation figure, a dropout figure, then
figures for `frm` and `frm+rws`, task schematics to Supplementary, and detail rather than an
outline. He also said new experiments were fair game, naming the dropout rate specifically. This
entry records what was measured, what was corrected, and what is running.

### The five figures

`trainRNNbrain/experiments_and_analysis/paperstyle.py` is new: the manuscript house style, kept
separate from `plotstyle.py` (which remains the style for internal diagnostics). Millimetre widths,
7 pt base, grid off, and schematic primitives — unit-grid pictograms, circuit glyphs, annotated
connectors — because the previous round of figures plotted data and explained nothing. Its five
categorical slots pass all five checks of the dataviz validator at the light surface; an earlier
version failed the contrast check on its green and gold and those two were darkened until it passed.

- `fig_paper_F1` the problem. Active fraction falls with N on both tasks (flip-flop 41.4 → 13.3%
  over N = 500 → 4000; CDDM 76 → 9.3% over N = 100 → 10,000). Both fits extrapolate to **N ≈ 1.4 ×
  10⁴ for 1,000 active units**, independently. Seven task cells at N = 1000, none above half.
  Every intervention as a change from its own matched reference: the best knob is worth ~120 units,
  and weight decay is a monotone dose–response poison (395 → 272 → 171 → 84 active as λ goes
  0 → 1e-6 → 1e-5 → 1e-4), which matters because it is on by default in most training code.
- `fig_paper_F2` dropout. See the correction below.
- `fig_paper_F3` the rate penalty. The mechanism panel is the one worth keeping: plotted from the
  implementations actually optimised, the metabolic cost `mean(r^2)` is minimised at r = 0 — it
  *pays* a unit to fall silent — while `frm` is two-sided about a target cap (0.2004 at N = 1000).
  Then the five-task recovery matrix: `frm` restores 98–100% of units on every task tested, `rws`
  alone makes it worse on every task where it was run, at ≤ 0.008 in r².
- `fig_paper_F4` what `rws` adds and breaks. Effective in-degree median **835 → 20.1** (q10–q90
  19.9–20.6, i.e. pinned exactly on the target of 20). The penalty switch is bidirectional with
  matched controls and **replicates on CDDM**: adding rws −30.0 points (flip-flop) and −15.3
  (CDDM); removing rws +18.2 and +14.8, treatment minus control.
- `fig_paper_F5` why it matters, recomputed from networks on disk at the manuscript's own
  λ_frm = 0.1 rather than from `population_distortion.csv`, whose source sweep `CDDM_std_g0` is
  deleted and which was run at λ_frm = 0.2. Effective dimensionality is *provably* unchanged by the
  active-units-only rule (a silent unit contributes no variance, hence no eigenvalue) but differs
  threefold between arms (2.2 → 6.6). σ_log is *not* robust to the rule (0.60 → 2.03 under rws),
  and measured correctly every arm sits below the cortical decade. Median per-unit task R² over
  active units: **0.33 unpenalised vs 0.95 under frm** — the first direct evidence in this project
  that the rescued units carry task information, not merely non-zero activity.

### Correction 1 — the 36-tau memory result was framed wrongly

The claim carried in `docs/paper.md` and in the old `fig_remedies` caption was that `frm+rws` holds
1000/1000 units live and fails the long-delay memory task, i.e. that the rescue costs capacity.
Read per seed from `data/dmts_curves_delay36.npz` (clean r² = 1 − loss_clean/Var(target)):

| arm | best clean r² | final r² | verdict |
|---|---|---|---|
| none | 0.605, 0.682, 0.605 | all 0.605 | never escapes, 3/3 |
| frm | 0.9998 ×3 | 0.635, −0.031, −0.016 | finds it 3/3, then loses it 3/3 |
| both | 0.605 ×3 | all 0.605 | never escapes, 3/3 |

**The unpenalised baseline fails it too.** So this is not a cost of the rescue relative to normal
training. What is actually true is more interesting: `frm` is the only arm that ever finds the
memory solution, it finds it in every seed, and it does not hold it — all three collapse before
150k, and only two of three keep a usable best checkpoint (0.954, 0.965). Adding `rws` abolishes
the escape entirely. There is no `rws`-alone arm at 36 tau, so we cannot say what `rws` alone does.

### Correction 2 — panel F2(b) was computing the wrong participation

The dropout sampler calls `Trainer.get_participation_`, which reads the **raw states** — for
`equation_type = "h"` those are pre-activations — while every active-unit count in this project uses
`participation_from_states_`, the same formula on the ReLU'd rate. Trainer's own docstring says the
two are "deliberately kept distinct". The first version of this panel read p_drop off the logged
vector, which is the wrong quantity.

Recomputed correctly (4 trained `none`/`dead` nets, noise-free probe of the saved weights):
Spearman(sampler score, rate participation) = **0.24**; median sampler score 1.39 for active units
against **1.09 for silent ones**; mean normalised rank 0.59 against 0.45. Consequently
**51.3 ± 4.5% of the drop mass lands on units that were already silent — about 25 of the 49 dropped
per iteration.** A unit held below threshold has a large *negative* pre-activation, hence a large
|x|, hence a high score. This is a quantitative explanation for why the method recovers ~100 units
and not ~700, and it predicts the fix (sample on the rate). The fix is untested.

### Correction 3 — an automated audit of this repo was wrong about the CDDM switch

A scouting agent reported the CDDM penalty-switch replication as null at −3.4 points. Re-measured
with a single pipeline that reproduces the flip-flop numbers exactly (−30.0 / +18.2 against the
known −28.5 / +17.8), CDDM gives **−15.3 / +14.8**. The replication holds. Recorded so it is not
re-litigated. The CDDM controls do drift, which is why the treatment-minus-control contrast rather
than the raw before/after is the statistic.

### Submitted: a drop-rate and targeting ladder

`slurm/SilentReLU_flipflop_droprate_{spock,della}.slurm`. 3-bit flip-flop, N = 1000, pen = none,
`dead`, 40,000 iterations, 3 seeds per arm: drop_rate ∈ {0.10, 0.20, 0.40} at β = 1, plus
drop_rate = 0.05 at β = 4 (sharper targeting). β = 16 was evaluated and **rejected before
submission**: the softmax collapses onto ~6 units, so it would test "permanently ablate the busiest
six", not sharper targeting.

Submitted to Spock as **6307716** (12 tasks, pending — every GPU on the cluster was allocated) and
to Della as **14214329** (12 tasks, all running by 22:35, due ~05:00). Della's repo was at 9aa3f9a,
which is **before the dropout fix**, so its `train_step` still scored the task loss on the
non-dropout pass; it was checked out to 881f28c and the fix verified present before submitting.
Had that not been checked the whole array would have been void.

40,000 iterations is a valid read-out and needs no fresh controls: the only max_iter-dependent
quantity in `Trainer` is the noise-annealing schedule, and `trainer_ptrack_freshbatch.yaml` sets
`anneal_noise: False`, so a 40k run is step-for-step identical to the first 40k steps of a 150k run.
Verified, not assumed. At 40k the standard setting already shows its full effect (487.0 ± 28.0
against no-dropout 387.0 ± 37.3).

**Pre-registered before submission.** An arm counts as an improvement only if its mean exceeds
**571 active units (scale-free) and 567 (absolute 4e-2)** — reference + 3 reference SD under both
criteria. Cost guard: an arm clearing the bar whose clean loss exceeds the no-dropout reference by
>10% is a trade, not a win. If no arm clears the bar, that is the result and it goes in the figure:
the drop rate is not the limiting factor and dropout's ceiling is structural.

The N=500/2000 size sweep (6306913) had produced **zero** output after 2 h (its trace is written
only at completion, and N=2000 + dropout is ~37 h/job). Its array throttle was lowered to 2 so it
would stop taking GPUs from the ladder; no running job was killed.

### Where the paper stands

`/Users/pt1290/Documents/Projects/Dead_ReLU_paper` builds to 33 pages with zero LaTeX errors, zero
undefined citations and zero undefined references. `refs.bib` now holds 114 scouted entries (109 of
them metadata-verified against real records), 63 of which are cited. `docs/measured_facts.md` is
new and is the single source of numerical truth — every number in the manuscript must appear there.
The seven stale pandoc sections were moved to `docs/superseded_sections/`.

---

