## 2026-10-01 15:05 — standardising the architecture, and the re-runs it forced

**What started it.** Three CDDM "control" numbers that should have agreed did not: 272 on slides
8/10/11, 414 on 12/13, 443 on 14. Two causes, only one a fault.

272 against 414 is training budget. Reading the slides-8/10/11 reference at 30,000 instead of 200,000
gives 402/400/430, mean 411, beside the metabolic reference's 414. The control's own trajectory is
411 (30k) → 368 (50k) → 325 (100k) → 272 (200k), which is the paper's claim showing up in the
reference rather than an inconsistency between panels. New deck slide 7b draws it.

443 was architecture. The recurrent-noise sweep behind slide 14 was built from `rnn_relu_Dale`, whose
`self_connections: false` zeroes the W_rec diagonal, and its config predates `dale` and
`io_nonnegativity` so both took the model default. Confirmed in the trained weights: all 1000
diagonal entries exactly 0 and the bias 0 in every unit, where the metabolic, drift and weight-decay
sweeps all carry a trained bias (|b| up to 0.5) and nonzero self-connections.

**A measurement error found on the way, and corrected.** The offline re-score first used a strided
128 of the 450 CDDM trials. Checked against the Trainer's own trace on five networks that have both,
the full batch reproduces the trace to 1.0 unit while the 128-trial subsample undercounts by 22 —
participation is std + q_0.9 over pooled (time, trial) samples and both terms shrink when trials are
dropped, so q_95 falls and the 0.05·q_95 bar falls with it. The re-score now uses the whole batch.
Predicting the noise sweep's full-batch value from that offset (443 + 22) was wrong: measured, it is
still 443, because the bias depends on the distribution's shape at the bar.

**The standard, now explicit:** self_connections=True, bias fixed at 0, dale=False,
io_nonnegativity=False, gamma=0, at least 50,000 iterations. `standard_audit.py` checks every sweep
against it from the sweeps' own saved configs, distinguishing a key set to the default from a key
ABSENT. `dale`, `io_nonnegativity` and `gamma` conform everywhere.

**A config bug this exposed.** Commit `ae52c58` moved `rnn_relu_standard.yaml` to a fixed bias but
missed `rnn_leaky_relu_standard`, `rnn_softplus_beta25_standard` and `rnn_sigmoid_shifted_standard`,
which still shipped `bias_range [-1, 1]`. Every activation arm therefore carried a trainable bias
while its ReLU reference did not, so slides 8 and 9 would have confounded activation with bias. All
three are now `[0, 0]`.

**Submitted** from `~/trainRNNbrain_stdfix`, a clean worktree at `77fbd97`, every output tagged
`_bias0` so the old data survives for comparison. CDDM only, by instruction, plus slide 11's
input-scale cells which are flip-flop but 27 runs rather than the 127-run ksweep/bigN set that was
ruled out.

| job | sweep | array | budget |
|---|---|---|---|
| 14839185 | `std_g0_drift_bias0` | 1–12 | 50k/200k/200k/300k |
| 14839186 | `std_g0_weightdecay_bias0` | 1–9 | 200k |
| 14839187 | `std_g0_activations_bias0` | 1–9 | 200k |
| 14839188 | `std_g0_metabolic_bias0` | 1–36 | 50k (was 30k) |
| 14839703 | `std_g0_bias0` | 1–40 | 50k (was 30k) |
| 14839704 | `*_g0_noise_std` | 1–40 | 50k (was 30k), now `rnn_relu_standard` |
| 14839955 | `std_winp_bias0` | 1–24 | 150k |
| 14839956 | `std_ksweep_bias0` k=3 N=1000 | 13–15 | 150k, the matched default draw for slide 11 |

The first started task confirms `gamma 0, dale False, io_nonnegativity False, self_connections True,
bias_range [0, 0], max_iter 50000`.

**Not re-run, by instruction:** `NBitFlipFlop_std_ksweep` (76 runs at 150k–500k) and `std_bigN`
(51 runs), about 30M of the 54M iteration-runs a full standardisation would cost. Slide 9 and
Figure 1 panel (c)'s flip-flop series therefore still rest on trainable-bias networks, and the deck
says so.

**Also still deviating:** the Figure 2 flip-flop grid trains 40,000 iterations, under the 50,000
floor, and its own read-out lands at 35,000–43,000 — the budget ends where the measurement begins.
Re-running it means redoing Figure 2, whose cache rebuild is still in flight.

### What "dimensionality" means on slide 19, and two other counts that disagree — 2026-10-01 16:03

**Slide 19 now says what it plots.** The measure is the participation ratio of the active units'
noise-free firing rates: collect their rates across time and trials, take the covariance across
units, and with λ₁…λₙ its eigenvalues, PR = (Σλᵢ)² ∕ Σλᵢ². A soft count — exactly n when n
directions carry equal variance and the rest none, tending to 1 when one direction dominates, never
an integer, bounded by the number of active units. The three properties are asserted in the deck, so
they were checked against the project's own `participation_ratio`: n = 1/3/7/20 equal directions
return 1.0000/2.9988/6.9919/19.9125, a dominant direction returns 1.0000 at amplitudes 10/100/1000,
and 12 isotropic units return 11.967.

**Slide 19b: PCs for 99% of the variance against stable rank.** `f2_dimensionality_extra.py` →
`data/f2_dimensionality_extra.npz` → `fig_slides.dims_extra_slide` → `slide_f2_srank_vs_pc99`. All
three measures come from one eigendecomposition per network, of the same covariance slide 19 uses.
Log x, because every claim here is a ratio and on a linear axis the penalty arm sits alone on the
right and squeezes the other five into the left quarter.

| arm | active | PR | PCs for 99% | stable rank |
|---|---|---|---|---|
| no intervention | 297 | 5.41 | 33 | 3.05 |
| dropout: mute | 522 | 7.11 | 64 | 3.63 |
| prune + duplicate | 741 | 6.65 | 65 | 3.06 |
| rescale | 454 | 7.32 | 57 | 4.07 |
| synaptic noise | 547 | 6.69 | 49 | 4.27 |
| penalty: frm + rws | 938 | 12.05 | 219 | 6.13 |

**Prune + duplicate doubles the hard count and leaves the soft one untouched** — 33 → 65 PCs for 99%
of the variance at a stable rank of 3.06 against the control's 3.05. The directions it adds carry
almost no variance. The penalty pair is the only arm that moves all three together (6.6× the PC
count, 2.0× the stable rank, 2.2× the participation ratio). Synaptic noise is duplication's
opposite: the second-highest stable rank on a modest PC count, so fewer directions more evenly
weighted. Reading any one measure alone hides this.

**The ambiguity in "stable rank when the number of PCs captures 99% variance" is moot.** It can be
read as the stable rank of the 99%-variance subspace rather than of the whole spectrum; both are
computed, and dropping the last 1% of Σλ changes Σλ/λ₁ by **at most 1.00%** across all 22 networks,
which is below the between-arm differences by two orders of magnitude.

**Two wrong turns, recorded because each cost a re-run.**

⚠️ *A per-arm tolerance is not a per-project tolerance.* The first validation required the recomputed
PR within 5% of the cache's, calibrated on a control network whose PR moves 2.6% between batches.
Duplication networks move **9–11%**, because duplication makes pairs of near-identical units whose
covariance is near-degenerate, so the check failed that arm for no reason. Replaced with a
per-network interval: the cache's single-batch value must fall inside the range that network's own
six draws span.

⚠️ *The r2 gate compares `r2_common`, not the clean score.* Finding a duplication net rebuilding at
clean r² 0.654 against a folder score of 0.9428 looked like the version-pin failure this project has
hit before, and the first conclusion drawn was that dev-branch code could not reconstruct those nets.
It was the wrong comparison. The cache's OWN `r2_clean` for that arm is 0.693 / 0.774 / 0.798 and the
rebuild gives 0.662 / 0.764 / 0.780 — agreeing within 0.03. These networks simply score lower with
the noise off than on, which the cache records; the folder name is the noisy condition, and
`R2_KEY_GATE` is `r2_common` (0.9435 against the folder's 0.9434). Nothing was broken. The
simulation path is also unchanged since the cache's build commit `99c877a`, so no version gap exists
to blame: `git diff 99c877a HEAD -- trainRNNbrain/rnns/ trainRNNbrain/tasks/ Trainer.py` is empty.

**What the validation does and does not establish.** Arm-mean PR reproduces the cache within ~2% for
five of six arms and 11% for duplication, the arm whose own batch spread is 9–11%. Active counts
agree to ≤3 units for the control and ≤1 for the penalty pair, but run 25–43 units (≈5%) below the
cache for duplication and 14–25 below for synaptic noise, with rank order preserved in every arm.
That residual is **not root-caused**: it survives the statefulness test (nine warm-up noisy passes
change the clean count by 0), and 52 of duplication's units sit within 0.8–1.25× of the active/silent
cutoff against 2 for the penalty arm, so threshold sensitivity is the suspect but is not demonstrated.
A 5% uncertainty in which units are counted does not touch any claim on 19b, where the arms differ by
factors of 2 to 7 — but it is a discrepancy, not a pass, and anyone quoting these numbers to more
than two significant figures should settle it first.

### Slide 20 answers its own heading now: the weights err toward too many small ones — 2026-10-01 16:11

Pavel pointed out that slide 20 is titled "lognormal, and which way it errs" and never says which
way. The statistic existed — `fig_paper_F2.lognormal_verdict` and the table `fig_paper_F2.main`
prints — but it reached stdout and the panel's one-line title, not the deck. The panel title names
only the WIDTH of the worst arm ("rescale is 2.3x wider"); the direction of the shape error was
nowhere.

**The answer: toward an excess of very small weights, in every arm including the control.** The
skewness of ln|W| is negative everywhere, never positive, so the departure from lognormal is always a
left tail — more near-silent synapses than a lognormal predicts, never more strong ones. Checked per
network rather than per arm mean: all 22 networks are negative on the full `W_rec` (range −2.83 to
−0.15) and all 22 on the rows of active units only (−1.98 to −0.41).

| arm | sd ln\|W\| | vs control | skew ln\|W\| | KS ÷ control's | q99/q01 |
|---|---|---|---|---|---|
| no intervention | 1.53 | 1.00× | −0.42 | 1.0 | 1,700× |
| dropout: mute | 1.59 | 1.03× | −0.21 | 1.2 | 2,000× |
| prune + duplicate | 1.43 | 0.93× | −0.53 | 1.4 | 1,300× |
| rescale | 3.48 | 2.27× wider | −2.31 | 7.8 | 15,800,000× |
| synaptic noise | 1.40 | 0.91× | −0.51 | 1.8 | 1,100× |
| penalty: frm + rws | 3.13 | 2.04× wider | −0.80 | 4.7 | 713,000× |

Dropout, prune + duplicate and synaptic noise leave the control's width alone (0.91–1.03×) and sit
within 1.8× of its distance from lognormal. Rescale and the penalty pair are the two that move:
about twice as wide in log units, 4.7–7.8× the control's departure, and a magnitude range running
from 1,700-fold in the control to 713,000-fold under the penalty pair and 16-million-fold under
rescale. Rescale is the extreme on every column.

**The comparison has to be relative to the control, not to lognormality in the abstract.** The
control's own KS distance is 0.026 against 0.0006 for a same-size sample drawn from its fitted
normal — about 40× the sampling scale — so no arm here is "lognormal" in an absolute sense and a
test against that null would reject all six. Every statement above is a ratio to the control.

Restricting to the rows of ACTIVE units tells the same story and does not soften the two movers:
rescale widens to 3.10× the control there and the penalty pair holds at 2.06×.

The deck now carries the table and the direction under slide 20. `panel_e`'s title still names only
the width, because it is a manuscript panel and this change belongs to the talk; extending
`lognormal_verdict` to name the skew direction would improve both, and is not done here.

### Slide 20, corrected: kurtosis is the column that separates the two wide arms — 2026-10-01 16:16

Pavel asked to drop the q99/q01 column and said the table did not show what was good and what was
bad. Both were my fault, and the second was a real loss of information: the 13:05 entry's own finding
is that **width and shape are separate failures and only rescale commits both**, and the column that
carries it — excess kurtosis — was the one I left out. With it back:

| arm | sd ln\|W\| | vs control | skew ln\|W\| | excess kurtosis | verdict |
|---|---|---|---|---|---|
| no intervention | 1.53 | 1.00× | −0.42 | 1.00 | the reference |
| dropout: mute | 1.59 | 1.03× | −0.21 | 0.71 | unchanged |
| prune + duplicate | 1.43 | 0.93× | −0.53 | 1.45 | unchanged |
| synaptic noise | 1.40 | 0.91× | −0.51 | 1.60 | unchanged |
| penalty: frm + rws | 3.13 | 2.04× wider | −0.80 | **0.11** | wider, shape near-lognormal |
| rescale | 3.48 | 2.27× wider | **−2.31** | **6.41** | wider **and** misshapen |

Three arms sit within 9% of the control's width and 1.8× of its distance from lognormal — nothing
happens. The penalty pair is twice as wide but its excess kurtosis, 0.11, is NEARER an exact
lognormal than the control's own 1.00; its skew is 1.9× the control's, so wider and somewhat more
left-tailed but not misshapen. Rescale is twice as wide AND worse on both shape statistics. So wide
is not automatically bad: the arm that works is wide and keeps its shape, and what distinguishes
rescale is that it widens by driving a subpopulation toward zero rather than stretching the whole
distribution. The slide now carries a verdict column saying this per arm, which no amount of raw
statistics was going to convey on its own.

**⚠️ There is no brain number for this, and the slide now says so.** Asked what the skewness should
be measured from cortex, the answer is that we do not have one. The experimental claim behind
"lognormal weights" (`song2005highly`, `buzsaki2014log`, both already in
`docs/refs_population_statistics.bib`) is that a lognormal FITS measured EPSP amplitudes; an
acceptable fit implies skew near 0 but pins no value, and no published skewness of ln(EPSP) was
found. The methodological point matters more than the missing number: paired recordings are
detection-limited, so a connection too weak to resolve is scored as no connection and the measured
distribution is censored at the low end — precisely where these networks put their excess. The one
direction the weights err in is the direction the experiment is least able to see.

Worth stating because the project is inconsistent about this: for firing RATES there IS a cortical
anchor in use here (σ_log ≈ 1 in log₁₀ units, recorded at trajectory line ~2340), so a model rate
distribution is scored against cortex directly. For WEIGHTS there is no equivalent, so slide 20 is
arm-against-control and must not be read as a claim about biological realism. The statistics chosen
are at least the right family for such a comparison if a number ever appears: sd, skew and kurtosis
of ln|W| are all invariant under multiplying every weight by a constant, so they do not depend on the
arbitrary scale of a model coupling against millivolts of EPSP.

### 19b did not show a disagreement; slides 19–21 cut to essentials — 2026-10-01 16:30

**⚠️ Correction: the two measures agree, and the panel's own heading was wrong.** 19b went up as "two
other ways to count dimensions, and they disagree", on the strength of duplication doubling the 99%
PC count (33 → 65) while its stable rank sat on the control's (3.06 against 3.05). Pavel said they
looked like they mostly agree. He was right, twice over:

- The claimed dissociation is not a result. Duplication's stable rank against the control's is
  **Welch p = 0.99**, n = 3, ranges [2.22, 3.53] against [2.61, 3.49] — fully overlapping. That is
  "cannot tell", presented as "does not move". One of its three networks sits at 2.22 and drags the
  mean; it is the same network whose participation ratio is the most batch-unstable in the set.
- Measured rather than asserted, the three summaries **agree**: pairwise Pearson r over the 22
  networks is +0.93 (PR vs 99% count), +0.93 (PR vs stable rank) and +0.82 (99% count vs stable
  rank). Dropping the penalty arm, which is extreme on every axis and could be manufacturing the
  correlation on its own, weakens it but does not reverse it: +0.71, +0.80 and +0.33 (Spearman
  +0.77, +0.81, +0.48).

All three measures rank the control lowest and frm + rws highest, and all three separate only those
two: of the six pairings among the middle four arms, one reaches p < 0.05 on the 99% count and none
on the other two. So the panel stays, reframed as a **robustness check** — slide 19's result does not
depend on the participation ratio's particular weighting of the spectrum — which is a smaller claim
than the one it carried and the one the data supports.

**Slides 19, 19b, 20 and 21 cut from 1,203 words to 420**, and slide 20's six-row, four-column table
replaced by a figure. `fig_paper_F2.panel_weight_shape` → `slide_f2_weight_shape` plots the width of
ln|W| against its excess kurtosis, one colour per arm, with a dotted line at kurtosis 0 for an exact
lognormal. The finding the table was carrying — width and shape are independent failures and only
rescale commits both — is the shape of the cloud: four arms clustered at low width and modest
kurtosis, rescale alone at high width AND high kurtosis, frm + rws equally wide but sitting on the
lognormal line. `panel_e` is untouched; the new panel is additive, as `panel_r2_vs_active` was.

Kept in the cut text, because removing them would leave a claim unsupported: the participation-ratio
formula itself, the direction of the weight error (a left tail, in all 22 networks), and the warning
that the reference is the control and not cortex.

### Slide 20's weight panel was cutting every curve off at the right — 2026-10-01 16:34

Pavel: "Figure 20 breaks off on the right." It did. `panel_e` windowed x as (q01 − 0.3, **q99** + 0.3),
and every arm's largest weights run **0.9 to 1.6 decades past its own q99** — so each curve stopped
against the right frame mid-descent instead of falling to the floor, and the panel read as though the
data had been truncated there. Measured per arm, last nonzero bin minus q99: control +1.20, dropout
+0.90, duplication +1.10, rescale +1.55, synaptic noise +1.00, frm + rws +1.60.

The right edge now follows the END OF THE SUPPORT (last nonzero histogram bin + 0.3), which takes the
window from (−8.33, −0.37) to (−8.33, 1.13). **The two edges are deliberately asymmetric now**, which
is worth stating because it looks like an inconsistency: |W| has a hard upper end about a decade above
q99, so a support-based right edge costs ~1.2 decades of width and nothing else; the lower tail runs
all the way to the 1e-12 zero tolerance, so a support-based LEFT edge would stretch the axis to −12
and squeeze every arm but rescale into a sliver. The left edge stays on q01.

Checked rather than eyeballed, with the pass condition fixed first — each curve's density at the
right edge must be under 1% of that curve's own peak, i.e. visibly fallen rather than cut. Observed:
0.001% to 0.004% across all six arms.

This is a defect, not a style preference, and it was in the manuscript figure too, so `panel_e` itself
is fixed rather than a deck-only copy. `fig_paper_F2` rebuilt. The companion scatter
`slide_f2_weight_shape` was checked at the same time and has no clipping: its rightmost markers sit
at x = 3.48 against an axis limit of 3.76.

### Deck slides 23–24c: dropout, as seven one-panel slides — 2026-10-01 21:10

**Seven slides, one panel and one claim each**, after a first version that packed three panels onto
slide 23 and two onto slide 24 and broke the deck's own rule at the top of `presentation.md`.
Written by `fig_slides.py`, each at the single-panel width (110 mm):

| slide | claim | figure |
|---|---|---|
| 23 | each unit gets its own drop probability, then its own coin flip | `slide_23_dropout_selection` |
| 23b | beta decides how hard dropout aims at the busiest units | `slide_23b_dropout_targeting` |
| 23c | the drop rate is a share of the units still alive | `slide_23c_dropout_dose` |
| 23d | a dropped unit can lose its output, or everything | `slide_23d_dropout_kinds` |
| 24 | a higher rate, and a sharper aim, keep more units alive | `slide_24_dropout_rate_units` |
| 24b | mute recruits units for free; dead pays for them | `slide_24b_dropout_rate_cost` |
| 24c | dropout slows the silencing; it does not stop it | `slide_24c_dropout_along_training` |

**Slide 23c was rebuilt twice, and the second rebuild found something.** Pavel asked what was going
on in it. Two things were. First, it drew the dose `rho * M` as a second curve beside `M`, which on
a log axis is the same curve shifted down by a constant — one measurement drawn twice, reading as a
comparison. Second, it drew a single control seed, picked by sort order, and that seed passes
through a transient blow-up.

**The scale-free active count is blind to a 10^6-fold rate excursion.** Measured on the three
`NBitFlipFlop_std_dropfix` control seeds: seed 0 at iterations 3,600–4,000 and seed 2 at
9,900–10,300 reach q95(participation) of 2.0e5 and 4.7e7 against a healthy value under 5, with the
clean training loss hitting 1.9e9 and 1.3e14. Both recover and finish at 0.049 and 0.120. Through
seed 0's excursion the scale-free count reads 391, 462, … — entirely ordinary — because the bar is
0.05 * q95 and rises with the rates. The fixed 4e-2 threshold does register it.

Also checked while there, because the trace visibly wobbles: the step-to-step wobble in the count is
NOT a q95 artefact. The correlation between step changes in the count and step changes in q95 is
-0.05, and the fixed-threshold count wobbles just as much (sd of step changes 19.9 against 18.8).
Units really do cross the bar in both directions between snapshots.

The panel now draws all three seeds, keeps only the `rho * N` = 250 reference, and shades the two
excursion windows with one line saying what they are. The blindness is a property of the criterion,
not of this panel, and it applies to every active count in the deck and the manuscript.

**Slide 23 exists because Pavel read 23b and 23c and said they do not stand on their own.** They
are each a zoom into one step of a selection chain the deck never showed, so the curve in 23b was a
probability nothing had introduced. Slide 23 now walks the chain on a real control network at
iteration 40,000: fourteen of its thousand units with their firing rate, whether the live-pool rule
admits them, their rank inside the pool, the probability the production `drop_probabilities`
assigns them, and one Bernoulli draw. The four silent rows are marked "never drawn", which is what
23c then explains. The whole selection chain's formulas sit on slide 23; 23b and 23c carry prose
only, so nothing is written twice.

Internal check: the busiest unit in slide 23's table (β = 2, ρ = 0.25, M = 279) has p = 0.58, and
23b's β = 2 curve ends at 0.578 with M = 400 — the rank normalisation is what makes the two agree
across pool sizes, which is the reason it is there.

The old slide 24c, which explained why the first sweep was null, is gone — it was project history
rather than a result, and it is now one sentence under slide 24. Its figure
`dropout_sampler_blindness.png` is no longer referenced by the deck.

**Slide 23b is drawn by calling the production sampler.** `drop_probabilities` on the production
softmax-over-ranks weight vector, so the curve cannot drift from what training uses: the busiest
living unit is e^β times likelier than the quietest (2.7, 7.4, 54.6), and 48 rather than 54.6 at
ρ = 0.25, β = 4, the one cell of the grid where `p_max` binds.

**Slide 23c** uses the control's own live count: ρ = 0.25 of N would ablate 250 of the 279 units
that seed still has alive at iteration 40,000, which is why ρ multiplies the live pool.

**Slides 24 and 24b, the sweep.** The post-fix grid in
`NBitFlipFlop_std_bernoulli` — 4 drop rates × 3 targeting exponents × 2 kinds × 3 seeds, 3-bit
flip-flop, N = 1000, no penalty, 40,000 iterations — against the `do=none` cell of the matched
`NBitFlipFlop_std_dropfix` launcher. Active units and clean loss are read from the participation
traces, one pipeline for both kinds, because the Figure 2 cache holds the `mute` cells only.

| | ρ = 0.05 | 0.10 | 0.175 | 0.25 |
|---|---|---|---|---|
| control | 316.3 ± 33.3 | | | |
| mute β = 1 | 388.3 ± 58.9 | 410.0 ± 12.0 | 445.0 ± 35.9 | 542.0 ± 23.9 |
| mute β = 2 | 395.0 ± 19.1 | 425.0 ± 14.8 | 488.7 ± 22.9 | 562.0 ± 37.2 |
| mute β = 4 | 413.0 ± 26.1 | 456.0 ± 16.5 | 525.3 ± 37.1 | 636.0 ± 20.3 |
| dead β = 1 | 344.0 ± 2.6 | 330.0 ± 6.2 | 343.0 ± 15.0 | 416.3 ± 16.2 |
| dead β = 2 | 357.7 ± 11.9 | 363.0 ± 4.6 | 420.3 ± 8.7 | 654.7 ± 26.7 |
| dead β = 4 | 377.3 ± 11.1 | 410.7 ± 11.4 | 809.7 ± 10.2 | 963.3 ± 3.2 |

Sharper targeting adds units at all eight (kind, ρ) settings. A higher rate adds units in every
series but `dead` at β = 1, which is flat (344, 330, 343) to ρ = 0.175 and then climbs to 416. The
caption that states this is COMPUTED from the cells, not typed, and it took two corrections. The
first draft said "monotone in both knobs", which is false for `dead` at β = 1. The second said
"sharper targeting adds units at every setting", which passed a monotonicity test on the traces and
failed a cross-check: `data/fig_paper_F2_cache.npz`, which rebuilds each network from its saved
parameters rather than reading the trace, has `mute` at ρ = 0.05 going 386 → 384 → 406 as β rises,
so its first step is DOWN where the trace has it up. The two counts never differ by more than 17
units of 1000 over the twelve `mute` cells and agree on every other ordering, so the finding is that
a 25-unit rise against a 59-unit seed spread was never resolvable. A series now counts only if it
rises at every step AND rises by more than its seeds scatter, which drops that one setting, and the
exception list would name a different one if the cells changed.

**The cost depends on the read-out, and the two disagree in sign.** On the trace's noise-free,
dropout-off probe (a fresh batch — `same_batch: False`) `mute` sits below the control's 0.101 at
every setting, 0.052 to 0.086, and `dead` reaches 0.207. The Figure 2 cache's `r2_common`, which
keeps the recurrent and input noise the nets trained with, has `mute` giving up 0.945 → 0.922; its
`r2_clean` agrees with the trace probe (control 0.8651, every mute cell 0.887–0.925). Both are
quoted on the slide. `dead` is absent from the cache, so its cost is quoted on the noise-free probe
only — rebuilding the cache over the 12 `dead` cells would settle it.

**Renumbering.** The old slide 23 (dropout delays the silencing) is now 24c, with its deviation
stated: those are the only dropout nets trained past 40,000 iterations and they predate the sampler
rewrite, so the "delays or halts" question has not been re-asked of the corrected rule. The old
slide 24's claim, "dropout is capped", is withdrawn — the post-fix grid contradicts it.

**⚠️ Noted, not fixed: `rescale`'s clamp in `RNN_torch.get_dropout_mask` is inert.**
`(1.0 - p).clamp_max(1.0 / (1.0 - p_max))` clamps 1 − p (always ≤ 1) to a maximum of 10, which never
binds. The factor is bounded anyway because water-filling caps p_i at p_max, so the behaviour is
correct; the line reads as a guard that does nothing. Left alone — not my change to make.

**⚠️ My work was swept into someone else's commit.** `724365f` ("Closing section: penalty slides
moved to the end, plus temporal PR and the four-measure comparison") carries my
`dropout_variants_slide`, `dropout_rate_sweep_slide` and `_right_labels` in
`trainRNNbrain/experiments_and_analysis/fig_slides.py`, plus the first version of the slide 23–24c
text in `docs/presentation.md`. I did not make that commit and history is not being rewritten. The
later corrections to `docs/presentation.md` are committed separately below.

