## 2026-10-01 20:17 — the budget is 75,000, and short runs are topped up rather than retrained

**50,000 iterations is not enough, and what it misses is systematic.** Reading all 493 runs on disk
through the project's own criterion (`fit_floor_at` / `excess_time_at`, margin 0.07, each floor
fitted over that run's own budget), 365 of them — 74% — reach 1.07× their own fitted floor by 50k.
75k covers 465 of 493, which is 94%; 100k adds only 8 more. Per task family the median crossing is
13k on CDDM, 35k on DMTS and 40k on the flip-flop, but the upper quarter is what decides a cap:
CDDM's 95th percentile is 45k against DMTS's 105k and the flip-flop's 65k.

The failures cluster rather than scatter, which is why a single number had to be chosen on the tail
and not the median. Unpenalised flip-flop at k=8: zero of nine runs clear 50k. Flip-flop N=4000 at
k≥4: one of fifteen. frm+rws at k≥7: zero of fifteen. DMTS under rws: zero of nine, with medians of
65k, 98k and 132k at N = 500, 1000, 2000. CDDM unpenalised is comfortable everywhere, worst run 32k.

**A 50k run also cannot certify itself.** Fitting each floor on the first 50k only and asking whether
the run reaches 1.07× *that*, 355 of 493 say yes — and the disagreements run both ways. DMTS under
rws at N=500 is the dangerous direction: no run there reaches its floor by 50k, yet all three would
declare themselves converged from a 50k trace, because a floor fitted on 50k sits above the floor the
run eventually reaches. Same mechanism as the frm retraction recorded in `pr_matrix.py`'s header.

**75,000 now lives in the experiment configs**, not in launchers: `silent_units_std`,
`silent_units_N1000` (both were 30000) and `silent_units_largeN` (which had none). The three Della
launchers that carried `trainer.max_iter=50000` — `b0_met`, `b0_stdref`, `b0_noise` — drop the
override and read it from there.

**⚠️ The near-miss that cost nothing only because it was checked.** Removing those overrides made the
budget come from the cluster worktree's OWN config copy, and `trainRNNbrain_stdfix`'s copy still said
`max_iter: 30000`. A submission in that window would have trained 30k — below even the 50k it
replaced — with nothing in the launcher to show it. The worktree configs were updated to 75000 before
anything was submitted. Any launcher that stops overriding a value has to be followed to the config
the cluster actually reads, not the one in the repo on the Mac.

**Warm start: the clock, and one trace instead of two.** `paths.init_from` already restored weights
and Adam moments, but a continuation's loop counts from 0, so every trace it wrote claimed to start
at iteration 0 — and the read-out machinery, which fits over [T_START, budget] and reports crossings
in absolute iterations, read a 50k→75k continuation as a fresh 0→25k run. `Trainer.iter_offset` now
dates a continuation's samples from where its parent stopped, resolved from the parent's saved config
and written back the way the seed is, and `join_parent_trace` prepends the parent's samples so the
file is one complete trace. No analysis script changed. The loader also reads json parameters, not
only npz: the three lines being topped up save json, so it could not previously read its own sweeps.

Verified against bars fixed before the runs finished — joined length equals parent plus own samples,
every iteration step equals `track_every` with no gap or overlap at the boundary, the window is
absolute, the saved config records the offset, and the loss step across the boundary is *smaller*
than the typical step either side (0.0039 against 0.0072 locally; 0.0011 against 0.0119 on the json
path), so the continuation continues rather than restarts. On a real N=1000 production parent
(30,000 iterations, json, no Adam state) r² at iteration 50 of the continuation is 0.858 against the
parent's 0.840, where a fresh network scores near zero.

**Top-ups are queued contingent on their parents, so nothing waits on a person.**
`--dependency=afterany:<parent array>`, not `aftercorr`: nothing in a run folder records which rep it
was, so a task takes the rep-th finished run in its cell by sort order, and that assignment is only
stable once every rep has landed. Della: MetB0c 14850601 on 14839188, StdRefB0c 14850602 on
14839703, NoiseB0c 14850603 on 14849248 (+25,000 each). Spock: F2fillSc 6562854 on 6560906 and
6562855 on 6560907, +35,000, since those cells train 40,000. Each continuation writes to its own
root (`..._c75000`) rather than beside its parent, so no analysis globbing cells counts both the
truncated original and the continuation.

**⚠️ Two things the top-ups cannot fix.** Spock's `ff_both40k` N=4000 cell has no parent to continue:
tasks 8 and 9 of 6560907 died in ~20 s with CUDA out-of-memory, the frm penalty's `logsumexp` asking
4.58 GiB on a 44.4 GiB card, so that cell needs Della's 80 GB nodes rather than a top-up. And a
continuation starts with empty drift buffers, so a warm-started run has no lag-10,000 drift value for
the 10,000 iterations after the boundary; the loss and participation traces are continuous, the drift
series is not. None of the three Della sweeps being topped up track drift, so only the Spock cells
carry that hole.

**Noted, not fixed:** the non-light analysis path crashes on NBitFlipFlop with
`unsupported operand type(s) for *: 'dict' and 'float'`, after training and before saving. CDDM on
the same path is fine (checked end to end), and every flip-flop sweep uses `light_outputs: true`, so
nothing in flight is affected. Not my change to make.

### Slides 26–26d: synaptic noise on both tasks — 2026-10-01 22:54

`fig_slides.synnoise_ladder_slide` → `slide_26_synnoise_ladder`, `synnoise_size_slide` →
`slide_26b_synnoise_size`, `synnoise_scatter_slide` → `slide_26c_synnoise_scatter` (flip-flop) and
`slide_26d_synnoise_cddm`. Slide 27 is deleted: it was a placeholder for the penalty pair, which
slides 28–32 carry.

⚠️ **THE FIRST VERSION OF THESE SLIDES WAS DRAWN FROM A STALE CACHE AND TWO OF ITS CLAIMS WERE
WRONG.** `data/fig_paper_F2_cache.npz` as of 2026-10-01 12:48 counted active units on the NOISY
forward pass. The current builder counts them noise-free, and Della job 14832898 rebuilt the cache
with that fix at 22:48 (311 networks, 29 of them synaptic noise). What changed:

| | stale cache | rebuilt |
|---|---|---|
| flip-flop N = 1000, σ_w = 2 | 722 active | **529** |
| flip-flop N = 1000, σ_w = 3 | 762 active | **484** |
| CDDM N = 1000, control | 210 active | **307** |
| CDDM N = 1000, σ_w = 1 | 301 active, 1 net | **453, 3 nets** |

The claim "every rung buys units, all the way up" came from counting units under the very noise that
lights them up. Counted noise-free the ladder PEAKS at σ_w = 1 and falls back: 297 → 370 → 371 →
**548** → 529 → 484 at σ_w = 0 → 0.1 → 0.3 → 1 → 2 → 3. R² at σ_w = 0 falls 0.946 → 0.948 →
0.945 → 0.928 → 0.810 → 0.679 while each net's score in its own noise holds 0.923–0.947, so past
σ_w = 1 the network works only while its synapses jitter.

**How it was caught.** Rebuilding the three CDDM synaptic-noise networks locally to find out why two
of them missed the cache gave 445–467 active units against the cache's 301, and 303–309 for the
control against 210 — a 45–55% gap where the trajectory's recorded residual is 5%. The Della rebuild
landed an hour later and matched the local numbers to the unit, so the local rebuild was right and
the cache rows were stale. Nothing was drawn for CDDM before that resolved.

**N = 4000 landed**, as `NBitFlipFlop_size_series/EqType=h_k=3_N=4000_arm=synnoise_sw1.0`, 5 nets.
It spells the rung `sw1.0`, with no equals sign, so `SW_RE` is `sw=?([0-9.]+)` and the rung is parsed
rather than substring-matched — a `"sw=1" in cell` test silently dropped the whole size.

| N | active | R² at σ_w = 0 | dimensionality | sd ln\|W_rec\| |
|---|---|---|---|---|
| 500 | 210 → 374 (1.78x) | 0.946 → 0.920 | 5.09 → 7.23 | 1.435 → 1.369 |
| 1000 | 297 → 548 (1.85x) | 0.946 → 0.928 | 5.53 → 6.67 | 1.535 → 1.402 |
| 2000 | 454 → 612 (1.35x) | 0.947 → 0.930 | 4.25 → 6.37 | 1.514 → 1.360 |
| 4000 | 621 → 844 (1.36x) | 0.946 → 0.923 | 4.38 → 6.62 | 1.379 → 1.314 |

Every measure moves the same way at all four sizes. The multiplier does not: 1.85x at N = 1000 and
about 1.35x at both larger sizes.

**Slide 26d, CDDM at N = 1000, 100,000 iterations:** 304 → 453 active (1.49x), R² 0.874 → 0.866,
dimensionality 2.03 → 2.31, sd ln|W| 1.742 → 1.734. Three networks per arm.

⚠️ **The CDDM control has to be pinned to the matched cells.** The cache files anything without an
intervention as `control`, and CDDM's controls include a weight-decay horizon sweep: at N = 1000 its
wd = 1e-3 and wd = 1e-2 cells hold 63 and 89 active units against 307 for the matched wd = 1e-6.
Drawn unfiltered the panel showed 14 "control" networks spanning 50–420 units, which is three
experiments in one cloud. `control_cells` pins it to `CDDM_paper_grid` plus the wd = 1e-6 horizon
cell, both verified against the synaptic-noise cell's own config on max_iter, weight decay, gamma,
sigma_rec and sigma_inp.

**R² fully clean is not drawn.** It is below the noisy score in the control too (0.879 against
0.946) and non-monotone across the low rungs. The comparison read-out is `r2_common`.

**Slide 26e: live units along training, the question slide 24c asks of dropout.**
`fig_slides.synnoise_along_training_slide` → `slide_26e_synnoise_along_training`, read from the
trainer's own `*ParticipationTrace.pkl` through `fig_dropout_live.cell`, so the deck does not grow a
second parser for the same files. The probe is the trainer's `w_noise=False` pass, which ignores
sigma_w, so a synaptic-noise network is counted in the same noise-free condition as the control.

| σ_w | floor | at iteration | end, 40k | recovery | final clean loss |
|---|---|---|---|---|---|
| 0 | 226 | 21,600 | 296 | +69 | 0.088 |
| 0.1 | 261 | 13,700 | 371 | +110 | 0.132 |
| 0.3 | 285 | 16,500 | 372 | +87 | 0.109 |
| 1 | **377** | 9,100 | **549** | +173 | **0.055** |
| 2 | 451 | 15,500 | 537 | +86 | 0.152 |
| 3 | 414 | 16,700 | 480 | +66 | 0.253 |

⚠️ **THE PER-DECADE SLOPE SLIDE 24c USES DOES NOT TRANSFER, AND QUOTING IT WOULD HAVE BEEN WRONG.**
These curves are not straight. Every arm, control included, falls to a minimum between iteration
9,100 and 21,600 and then partly recovers, so a line fitted over the last decade crosses the minimum
and reports the recovery as growth: the fit returns **+47 units per decade at σ_w = 1**, which reads
as "silencing reversed" and is an artefact of fitting a line to a U. The floor and the endpoint are
reported instead, both read off the trace rather than fitted. It was computed, seen to disagree with
the curve, and dropped - not published and corrected later.

**Slide 26e draws the control against σ_w = 1 alone.** All six rungs on one axes is the ladder,
which slide 26 already carries, and twelve traces bury the one comparison this panel exists for. The
other rungs' numbers stay in the table above.

⚠️ **A CLAIM THAT ARM MEANS SUPPORTED AND THE SEEDS DID NOT.** "σ_w = 1 ends at the lowest clean
loss of any condition, 0.055 against the control's 0.088" went on the slide and was wrong. Per seed
the control spans **0.047–0.134** and two of its six nets beat every σ_w = 1 net (0.054–0.056). The
arm mean is lower only because the control has a heavy upper tail. `r2_clean` says the same thing:
σ_w = 1 gives 0.923 / 0.928 / 0.931 against a control spanning 0.820–0.931, so it does not beat the
control's best seed either. What IS true is that synaptic noise removes the bad tail - every σ_w = 1
net performs like the control's best and none like its worst - which is a statement about spread,
not level, and the slide now says that.

**The unit counts ARE separated per seed**: 254–343 with no noise against 530–580 at σ_w = 1, with
no overlap. σ_w = 2 holds a higher floor than σ_w = 1 but ends lower and at three times the loss.

**What this does NOT say.** The budget is 40,000 iterations against the 150,000 slide 24c reads, and
the control is falling again at the end of its own recovery, so nothing here establishes that
synaptic noise halts the silencing. It raises the floor and the endpoint within the budget measured.

**Open.** No DMTS synaptic-noise cell exists, on either cluster, and no σ_w ladder outside the
flip-flop. Checked three ways - a broad folder listing on both clusters, a grep of all 180 DMTS
configs for a nonzero sigma_w, and the rebuilt cache's own arm counts - after a first check with the
glob `*/*sw=*` proved unsound: the N = 4000 cell is named `arm=synnoise_sw1.0`, with no equals sign,
and that glob could not match it.


### Slide 33, the bottom line: every knob and every mechanism on one ratio axis — 2026-10-01 23:25

`fig_slides.bottom_line_slide` → `slide_33_bottom_line`. Two blocks on one log axis of active units
relative to each row's OWN control: the one-knob families from `fig_slides.collect()` (which is
Figure 1d's own loader) and the five mechanisms from the rebuilt Figure 2 cache at the operating
point `F2.pick_cell` chooses, so the slide cannot drift from slides 15, 17 and 18.

| one knob at a time | span | | a mechanism added | units | R² |
|---|---|---|---|---|---|
| equation form, trainable bias | 0.97–0.99x | | rescale | 1.53x | −1.4% |
| metabolic penalty | 0.92–1.04x | | dropout: mute | 1.76x | −1.9% |
| activation function | 0.76–1.02x | | synaptic noise | 1.85x | −1.8% |
| input scale | 1.15–1.29x | | prune + duplicate | 2.47x | −0.2% |
| recurrent noise | 0.35–1.02x | | penalty: frm + rws | 3.16x | −2.8% |
| weight decay | 0.31–1.45x | | | | |

⚠️ **TWO HEADLINES WERE WRITTEN AND BOTH WERE CONTRADICTED BY THE PANEL UNDER THEM.** The first
read "no standard knob raises the count" — weight decay at zero reaches 1.45x. The second read "both
knobs that move it furthest move it down" — the furthest reach in BOTH directions is the same knob,
weight decay, at two doses. The headline now splits the knobs by a threshold rather than a sentence:
a knob counts as moving the count if any of its arms halves it, which puts weight decay and
recurrent noise on one side and the other four on the other, and the span of each side is computed.
Writing a summary line before reading the rows it summarises is how both errors happened.

**The axis is ratios, not counts, and the panel says so.** The one-knob rows are CDDM and the
flip-flop at four different budgets; the mechanism rows are the flip-flop at N = 1000 and 40,000
iterations. Absolute counts cannot share an axis across those; each row against its own control can,
which is the convention slide 15 already uses.

**The mechanism numbers come from the rebuilt cache**, so they supersede slide 18's text: duplication
is 2.47x at −0.2%, not the −0.1% recorded there, and the arm means for mute, rescale and synaptic
noise all moved. Slides 17–22 still carry figures drawn from the pre-rebuild cache and prose quoting
its numbers; they were left alone rather than silently re-rendered under someone else's text.

**Input scale moved from the knob block to the mechanism block** on Pavel's call: 1.29x of its own
ladder's default rung at −0.1% of R², so it is small but it is an effect, and it is the cheapest
row on the panel. ⚠️ Its two numbers come from `winp_cell` at 150,000 iterations against the
ladder's own default, where the five rows below it are `r2_common` at 40,000 against the Figure 2
control. The RATIO is comparable across rows, the budget and the probe are not, and the panel's
footnote says which row is which. The headline recomputes itself from the split, so moving the row
changed "5 knobs / five mechanisms" to "3 knobs / 6 mechanisms" without anything being retyped.

**Two open questions added to the deck** (both Pavel's):
- Re-run every arm on a standardised budget and read each network where its own loss first reaches
  1.07x its own fitted floor, which is slide 6's rule, so that no comparison rests on a budget set
  per sweep. Tighten to 1.03x once the floor fits support it; at 1.07x some runs are still moving.
- Test whether the interventions stack: prune + duplicate + synaptic noise + dropout + a small
  frm + rws penalty in one network. Nothing in this deck combines them. If the effects are additive
  the count should clear what any arm reaches alone; if they are not, which pair cancels is the
  result.


### WHAT DOES WORK, gone over: schematics, titles, and the size panels — 2026-10-02 00:15

Pavel took the second half of the deck; this is that pass.

**Slide 16's two sloppy schematics, rewritten in `fig_paper_F2.panel_a`.**
- The penalty box said "loss + penalty", which says nothing — every penalised loss is a loss plus a
  penalty. It now names the two terms as the Trainer computes them: frm is `(r_i − r*)²` per unit,
  the squared miss from a per-size rate target, and rws charges a row of `W_rec` whose effective
  in-degree, |w|₁² / |w|₂², runs over 20. Operation line: "silence and dense rows both cost".
- Synaptic noise drew an arc onto a dotted SILENT unit labelled "not targeted", which invites the
  reader to look for a target the rule does not have. It is now one live unit to another live unit,
  with a Gaussian drawn on the connection and the label `w → w(1 + σ_w ε)`, `ε ~ N(0,1)`, every
  timestep. ⚠️ The noise is MULTIPLICATIVE and relative; a bare bell labelled N(0, σ_w) would have
  said additive, which is not what `RNN_torch.forward` does.

⚠️ **MY FIRST TITLE FOR SLIDE 19 WAS WRONG TWICE OVER, AND PAVEL CAUGHT IT.** It read
"Dimensionality rises with the count, but only frm + rws clears the control". Dimensionality does
NOT rise with the count: prune + duplicate adds the second-most units of any arm, 438, and the least
dimensionality, 5.98 against the control's 5.53, with its seeds overlapping the control's. And
"clears the control" was jargon standing in for a comparison I had not made. Checked per seed:
control 4.74–6.35, mute 6.64–7.49 (disjoint), duplicate 4.40–6.89, rescale 6.09–8.71, synnoise
6.29–7.03 (all three overlapping), frm + rws 11.55–12.80 (disjoint, 2.2x). The title is now "Only
frm + rws adds directions: 5.5 to 12. The other four sit at 6–7, whatever they add in units", and
the body carries the duplication contrast. Slide 21's title had the matching fault — "none reaches
the diagonal" contradicted its own body, which says frm + rws nearly does.

**Titles that named a measure now make a statement**: 17 "Every rule raises the count: 297 of 1000
with no intervention, 453–939 with one", 19 "Dimensionality rises with the count, but only frm + rws
clears the control", 20 "Weights are near-lognormal everywhere, and err toward too many small ones",
21 "Every arm beats the control at every size it has, and none reaches the diagonal", 23c "Rho sets
the fraction of the *active* units dropped, not of all N", 23d "`mute` cuts a unit out of the
read-out; `dead` cuts it out of the network". Slides 25b–25d and 27–27d were left alone: another
agent is editing them.

**23d also answers a question the slide never addressed.** `get_dropout_mask` is called once inside
`forward`, OUTSIDE the time loop, and the mask carries no time or trial axis, so a dropped unit is
dropped for every timestep of every trial in that batch and the coin is flipped again at the next
iteration. Checked in the code, not inferred from the pseudocode.

**Slides 17–22 re-rendered on the rebuilt cache**, which moved their numbers:

| | slide text before | now |
|---|---|---|
| duplication at N = 1000 | +474 units, −0.1% | **+438, −0.16%** |
| duplication seeds | "inside the control's range" | **overlap it** (0.9424–0.9451 against 0.9433–0.9475) |
| dropout / rescale / synnoise | 1.3–1.9% | **1.5–1.9%** |
| duplication at N = 4000 | costs 0.027 | **0.028** |
| frm + rws | 939 units for 0.025 | 939 for **0.027** |

Slide 22's ⚠ about an eight-draw cache scoring CDDM's training grid is deleted: that rebuild landed.

**⚠️ A table I computed and had to throw away.** Pooling every cell of an arm at each N gave
synaptic noise an R² of 0.862 at N = 1000, which would have been a real finding if the panel worked
that way. It does not: `F2._size_series` already picks ONE cell per size with `pick_cell`, and the
pooled number was my own crude aggregation mixing σ_w = 0.1 through 3 into one point. Read the
panel's own selection rule before quoting a number as the panel's.

**Slides 21 and 22: what is actually coming.** Checked on Della rather than assumed.
- **frm + rws stays a lone point at N = 1000.** `F2fill` (14841830/14841831) is the job that would
  fill N = 500, 2000 and 4000. Tasks 1–6 (both at N = 500 and 2000) were CANCELLED at 16:45 on
  2026-10-01 with zero elapsed; task 9 FAILED after 17 s; tasks 8, 10–12 cancelled. **Only task 7
  survives** — one seed of `both` at N = 4000, 7.5 h in with about 32 h left. So two of the three
  sizes need resubmitting before this axis can be drawn, and N = 4000 will have one seed, not three.
- **Synaptic noise reached N = 4000** and is on the panels now.
- **rescale's N = 4000 cell exists on disk** (3 nets, written 15:40–16:02) but the cache pass began
  at 13:49, so it was not globbed. One more cache pass puts it on.
- The deleted sentence about k = 7 / k = 8 and the 400,000-iteration sweep was Pavel's call; the
  F2fill launcher records the same reasoning anyway (the `std_penlong` series is a 10x budget).


### Slide 27: the rescaling rule as a diagram, not a 14-line code block — 2026-10-02 00:30

Pavel: the pseudocode "doesn't make sense presenting it this way". `fig_slides.rescale_rule_slide`
→ `slide_27_rescale_rule`, two panels.

**(a) the life of one dormant unit**, as a state diagram: dormant → GROWING → either "fires again"
or "given up", with the conditions on the arrows and the per-iteration operation under the GROWING
box. The reader follows a path instead of matching four labelled blocks to four formulas. The
numbers on it are the launcher's own swept settings (alpha 1.002, gain cap 64x, 20% of N growing at
once), not round ones chosen to look tidy.

**(b) one boost on one incoming row**, computed BY THE RULE: positives times alpha, negatives
divided, then the row rescaled to the length it started with. Drawn at an exaggerated alpha so one
step is visible, and labelled as a schematic.

⚠️ **THE FIRST VERSION OF (b) DREW A RULE THE CODE DOES NOT IMPLEMENT.** It boosted every bar.
`Trainer.rescale_rows_` takes a `live` mask over PREsynaptic units and the swept cells pass it
(`rescale_active_only`), so the recurrent boost reaches only columns of units that are firing. The
panel now leaves the shaded columns untilted — they move only because the whole row is renormalised,
which is faithful — and that makes the rule's point visible instead of stating it underneath.

**Three lines of the pseudocode survive as prose**: the silence rule, frm's activity statistic, and
the target A* = frac x frm's cap (0.200 at N = 1000, so frac = 10 means an activity of 2.00). The
state transitions are the diagram.

⚠️ **TWO DIFFERENT MEASUREMENTS OF THE SAME QUANTITY ARE IN THE REPOSITORY, AND NEITHER IS ON THE
FIGURE.** How a dormant unit's excitatory incoming weight splits between firing and silent sources:
slide 27's bullet says **1.55 onto firing, 1.88 onto silent, 55%**; the docstring of
`Trainer.rescale_rows_` says **1.49 onto active, 2.16 onto silent, 59%**. Both claim unpenalised
controls at N = 1000. They cannot both be right and I did not resolve which is, so the figure says
only that the skipped columns carry no current and the per-cent number stays where it already was.
**Someone should settle this before either number is quoted in the manuscript.**


### Slide 20 on firing rates, slide 31b, and three figures de-fluffed — 2026-10-02 00:46

**Slide 20 was measuring the wrong thing.** Pavel meant the lognormality of FIRING RATES, not of the
recurrent weights. `f2_rate_lognormal.py` → `data/f2_rate_lognormal.npz` (177 networks) measures the
rate version with the Figure 2 cache's own loader, gate and arm classification, so the rows describe
the same networks slides 17–19 draw. `fig_slides.rate_dist_slide` and `rate_shape_slide` draw it.

Three choices, none forced by the data, all recorded on the slide: ACROSS UNITS rather than across
time; the per-unit number is the MEAN RATE on the noise-free pass, with frm's soft maximum and the
participation statistic carried beside it as cross-checks; and OVER ACTIVE UNITS ONLY, because
log(0) does not exist. The last makes the comparison conditional - the arms are described over
different numbers of units - so every statistic is also computed over each network's 200 loudest.

| flip-flop, N = 1000 | sd ln r | skew | KS ÷ sampling scale | q99/q01 |
|---|---|---|---|---|
| no intervention | 1.17 | −0.03 | **1.1** | 127× |
| rescale | 1.08 | −0.11 | 1.4 | 82× |
| synaptic noise | 1.07 | +0.02 | 1.7 | 172× |
| prune + duplicate | 0.98 | −0.19 | 2.9 | 47× |
| penalty: frm + rws | 0.72 | −0.60 | 3.0 | 16× |
| dropout: mute | 1.02 | −1.17 | **5.2** | 79× |

The flip-flop control's rates ARE lognormal - 1.1x what a same-size sample from its own fitted
lognormal gives, so there is nothing to reject. That is the opposite of the weights, where the
control sat 40x its sampling scale. The verdict survives changing the statistic: the control is
nearest lognormal on all three (1.1, 0.9, 1.5x) and `mute` furthest on all three (5.2, 4.1, 6.9x).

⚠️ **AND IT IS A STATEMENT ABOUT ONE TASK, WHICH I ONLY FOUND BY MEASURING THE OTHER.** The title
first read "The control's firing rates are lognormal". `cddm_penalty_rates.py` then put CDDM's
unpenalised control at 200,000 iterations at **3.3x** its sampling scale with skew **+1.50** -
right-tailed, a few loud units over a long quiet tail. The title now names the task and slide 20
carries the CDDM number as a caveat.

**Slide 31b: the rate distribution of the four CDDM penalty arms.** Slide 31's own figure CANNOT be
rebuilt - `frm_vs_both_slide` is not in this repository - so this is a new panel beside it rather
than a change to it. frm turns the skew from +1.50 to −0.69 and the width from 1.32 to 0.83: it
does not lift quiet units into the existing distribution, it builds a different one. rws alone is
the closest to lognormal of the four (1.3x) while holding the fewest active units (175), so shape
and count are not the same axis.

⚠️ **THE OBVIOUS CDDM CONTROL IS THE WRONG ONE.** `CDDM_std_g0/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0`
trains 30,000 iterations against the penalties' 200,000. `CDDM_std_g0_drift/EqType=h_N=1000_iters=200000`
is matched on budget, size, gamma and weight decay, and is what the cache uses.

⚠️ **`main()` CRASHED BEFORE ANY OF THIS.** It named `frm_vs_both_slide`, which does not exist in
the repository, so a full run died with a NameError before writing any slide after it. The name is
now looked up rather than referenced, and the run prints what is missing.

**Three figures de-fluffed, on Pavel's reading of them.**
- Slide 20's distribution panel is a vertical stack, one row per arm sharing the rate axis, each
  with its own fitted lognormal dashed over it. It was jagged because the cache bins log10 r at
  0.05 and 300 units over 20 bins is ~15 counts a bin; bins are merged to 0.25 and counts pooled
  within an arm, which puts 50–175 in a bin. Pooling weights a network by its active count; the
  per-network widths in the shape panel are unweighted and agree.
- Slide 27c: four lines of prose glossing circles, squares, triangles and a grey band were a legend
  typed out as a sentence. They are a legend now, the panels carry (a)/(b), and the title carries
  the claim.
- Slide 27d: suptitle cut from four lines to two. The held-out score is defined once at the top of
  the deck and did not need redefining here; the five runs below the clipped axis are annotated
  where they are drawn; the headline is the computed exchange rate, 0.008 of r2 per 100 extra units.


### Slide 27's suptitles, after Pavel asked what 27c even shows — 2026-10-02 00:52

**27c did not say what its x axis was.** "Activity target" means nothing on its own, and the panel
swept it without naming it: the target is the activity a boosted unit must reach before the rule
releases it, which is slide 27's A*. The suptitle now leads with that, then names the two panels,
then makes the claim; the axis label is shortened so it does not repeat the suptitle; and the "none"
tick reads "no target", because it is a different form of the rule and not a target of zero. The
deck text lost the paragraph that described the panels in prose - the figure says it now - and the
heading became the finding: "Having a target is what matters; its value does not".

**27's own suptitle restated the slide heading** ("Synaptic rescaling: the unit keeps every synapse
it has...") and is cut to the one thing the heading does not carry: the swept settings.

**27d's heading named two things instead of saying one** and is now the finding, 0.008 of r2 per 100
extra units.


### Dropout section trimmed; slide 20's banners cut — 2026-10-02 00:52

**23b and 23c are commented out and 23d is now 23a**, on Pavel's call. The two hidden slides swept
the sampler's two knobs (beta, the targeting exponent; rho, the share of the ACTIVE pool dropped),
and slide 24 already draws the whole rho x beta grid, so the deck reaches the same point without
them. Both are left in the file inside an HTML comment rather than deleted.

**Slide 23 ended by pointing at them** - "The next two slides are the two knobs in that chain" - so
that sentence now names beta and rho in place and sends the reader to slide 24. Checked that no
other live line references 23b or 23c by stripping the comments and grepping what was left.

**Noted, not changed:** the dropout block now runs 23, 23a, 24, 24c. There is no 24a or 24b - 24b
was folded into 24 in an earlier pass - so the lettering is not consecutive. Not mine to renumber
without being asked.

**Slide 20's banners are gone.** The stacked rate panel had a three-line suptitle repeating what the
deck heading, the row labels and the x axis already said; the shape scatter had a two-line suptitle
and a grey footnote that shrank the panel. Both are cut, with one inline "lognormal fit" label on
the top row where the dashed curve needed naming. The same banner was removed from slide 31b, which
was a copy of it.

**Finer bins on slide 20**, 0.10 in log10 (1.26-fold) instead of 0.25. Counts per occupied bin were
checked at three binnings BEFORE the default moved - at 0.10 every arm still carries 68-188 units
per bin, about a 10% Poisson wobble - rather than widening or narrowing until the curve looked
right.


### The deck rebuilt around pictures: a motivation section, a why-it-is-hard section, and five mechanism diagrams — 2026-10-02 02:34

Pavel's brief: "as graphical as possible, where possible every claim explained graphically, minimize
text to only what is necessary", understandable to someone who does not know the topic. The talk body
went from 6,432 words to about 1,500 across 33 slides, every caveat moved into one appendix, and the
discussion section cut entirely.

**The provenance paragraph was wrong, and the mistake was a false negative from a composed stem.**
The deck opened by listing six panels whose builders were "not in the repository — grepping the tree
for their output stems finds nothing". Five of the six do have builders: slides 25b/25c/25d come from
`fig_slides.prune_duplicate_all_slide`, which assembles its stem as
`"slide_25b_prune_duplicate_" + task.split(",")[0]...`, and slides 28/29 come from
`penalty_size_slide`, whose stem is `f"slide_22b_penalty_{task.lower()}"`. A grep for the full stem
cannot find a stem that is built at runtime. Verified by running the builder and watching the file's
mtime move, not by grepping again. Only slide 31 was genuinely missing.

**`frm_vs_both_slide` is now written**, so slide 31 is auditable and reads the current measure. It
takes active units, dimensionality and the recurrent weight histogram from
`data/cddm_penalty_cache.npz` and calls `heldout_r2` for the score, because the cache predates the
move to held-out r2 and still carries the noise-free number. The four arms score 0.8358 (untouched),
0.8356 (rate), 0.8381 (sparsity) and 0.8267 (both) — the pair is 0.009 below the control, where the
figure used to draw the retracted ordering that made it the best arm.

**One colour meant two conditions in one deck.** `PEN_KINDS` and `PEN_ARMS` assigned frm = SLOTS[0],
rws = SLOTS[2], both = SLOTS[1], which put "frm + rws" in orange on slides 28-31 and in green on
slides 17-19, and gave green to rws here and to frm + rws there. Both now read from `ps.COND_COL`.

**The metabolic slide was making the wrong claim, and a new panel shows why.** The old heading said
the penalty "moves nothing beyond seed scatter". Recomputed from the participation traces: on the
project's own relative bar the ladder is flat (414 / 393 / 431 / 432 / 380 active across λ = 0 to 10)
and on the unpenalised control's own bar, held fixed, the same networks go 413 / 392 / 392 / 286 /
119. The 95th percentile of participation falls from 0.328 to 0.054 over the ladder — sixfold — so
the bar falls with it and divides the effect out. `metabolic_ruler_slide` draws the two counts
diverging above the ruler shrinking. The deck's old figure for this read an archived CSV; the two
paths agree on the relative count element for element, which is the check that the recompute is
right.

**A new section, "why it is hard", which the deck did not have.** Three figures, all measured:

- `slide_wh_zero_gradient` — a silent ReLU unit's incoming and outgoing weight gradients are both
  exactly 0.0, against 0.00111 for a firing unit, produced by autograd rather than asserted.
  `tests/test_prune_and_reinit.py::test_dead_units_have_exactly_zero_gradient` passes.
- `slide_wh_treadmill` — redrawing dormant units fired 26,012 times over 591 distinct units on CDDM
  at N = 1000 (per seed 25,448 / 25,957 / 26,632 and 595 / 587 / 591) and moved the active count by
  4 (414.3 ± 9.9 against 410.3 ± 10.6). The units that never fire at all DID fall, 348 to 219. So the
  rule reaches the units and the network does not keep them.
- `slide_wh_decomposition` — what a donor carries, recomputed from four matched cells: the outgoing
  wires alone buy +142 of the +410 (35%), the incoming weight magnitudes in scrambled positions
  another +178 (43%), the right positions the last +90 (22%). The trajectory's earlier split was
  34 / 47 / 19; this is 35 / 43 / 22 from a fresh recompute.

**The bias kick's "keeps 70, an 85% re-suppression rate" could not be reproduced and is not drawn.**
The trainer logs how many distinct units the rule ever lifted, not which ones, so how many are still
firing at the end is not recoverable. What is recoverable: the arm ends at 374 ± 6 against its
control's 277 ± 22, so 97 of the 475 lifted survive — 80% put back, not 85%. Neither number is on a
figure.

**A measurement trap found while verifying, and it bites CDDM only.** Active counts must be read on
the full 450-trial batch: CDDM's batch enumerates its coherence-by-context grid in a fixed order, so
a 256-trial stride collapses to 1 and samples one corner, giving 309 / 319 / 337 against the full
batch's 399 / 412 / 414. Checked that `motivation_cache.py`'s own 64-trial stride does NOT have the
problem — stride 7 over 450 trials spans the grid, and it reproduces the full batch to within 2
units on both tasks.

**Four motivation figures, measured, in `fig_motivation.py` and `motivation_cache.py`.** The opening
used to begin at the measurement, where "248 live units of 1000" has nothing to land on.

- `slide_m1_model_organism` — what a trained RNN is read FOR. Schematic, no numbers.
- `slide_m2_training_empties` — untrained twins of the same architecture against the trained nets.
  Untrained: 518-569 active of 1000, with about 343 at exactly zero. Trained: 248-279. The silent
  MODE — units firing about a thousandth of a working unit's rate — exists only after training.
- `slide_m3_two_criteria` — this project's rule against Sokar et al.'s (ICML 2023) dormancy score,
  which shares no term with it: a mean-rate statistic normalised by the population mean and cut at a
  fixed τ, against a spread-plus-level statistic cut at 5% of the network's own 95th percentile.
  262 against 316 on the flip-flop and 271 against 276 on CDDM.
- `slide_m4_the_question` — the two axes every later result is a point on, with only the control
  drawn. Deliberately the axes of the payoff slide.

**τ was picked after seeing the answer, and that is fixed.** The first version drew Sokar's rule at
τ = 0.1 because it put the two counts within five units. 0.025 is the value their paper uses and the
figure now draws it. Across every threshold they report the deep-RL count stays between 267 and 322
on the flip-flop and 268 and 294 on CDDM. τ = 0 is an outlier on the flip-flop only (997 "working"),
because that task's quiet units sit near 1e-3 rather than at 0 — which is the argument for a relative
rule, arriving as a bonus rather than as a caveat.

**Five full-slide mechanism diagrams, in `fig_mechanisms.py`,** replacing a five-column strip in
which each rule got a fifth of a slide and three stacked text blocks. Each is capped at about 25
words and the cap is enforced by a word counter that fails the build. The rescaling one is the slide
Pavel named: its predecessor was a state machine carrying roughly 80 words that never drew the
mechanism. It now draws the drive crossing the rectifier's threshold, the excitatory and inhibitory
masses trading at a constant total, the columns from silent sources being skipped, and the stop.

**What the mechanism diagrams deliberately leave out** is recorded in the deck's appendix: the
inverted-dropout rescaling of survivors, donor sampling in proportion to activity, the donor
self-weight block, and that the rescaling steps are drawn far larger than the real 1.002.

**A critique pass named one defect that turned out to be wrong.** It said the dropout diagram's
mute-against-dead panel wrongly leaves a muted unit's recurrent arrows intact. `RNN_torch.DropoutMask`'s own
docstring settles it: "`dead` applies it to the unit's outgoing rate (which feeds W_rec) and to its
read-out column; `mute` only to the read-out column." A muted unit does go on driving the network.
The panel stands.

**Panels cut from the talk, each with its reason, all listed in the appendix** rather than deleted:
the five companion r2 scatters (five panels for one null), slide 13's equation form and trainable
bias (inside seed scatter), slide 15's Figure 1 (twenty rows of code shorthand on a projected
slide), slides 5b and 7b (bookkeeping that reconciles control numbers across sweeps), 20 and 20b
(weight lognormality, with no cortical number to compare against), 24c (dropout networks that
predate the sampler fix), 26b-26e (four more panels of one arm), 25d (DMTS runs that never solved),
32 (three 3-D clouds at 30,000 iterations in a configuration gone by 200,000).

**Open, and named as open in the appendix:** whether recruiting the units restores plasticity. The
motivation section borrows the loss-of-plasticity framing from Dohare et al. (Nature 2024) and this
project has not tested it. The test is to train a control network and a recruited one on a task
neither has seen and compare how fast each learns. Until that is run the plasticity claim is
motivation and the deck says so.

### Three knobs nothing on disk has ever turned: recurrent gain, a horizon without decay, tanh — 2026-10-03 02:57

All three sweeps run from `~/trainRNNbrain_recinit` on Della, a worktree pinned at e00924b
(branch `recinit-sweeps` = the dimensionality-analysis tip plus `configs/model/rnn_tanh_standard.yaml`
and the three launchers). CDDM, N = 1000, h equation, standard RNN, `trainer_ptrack_plain`, no rws
or frm penalty.

**The recurrent weight scale had never been varied.** All 3,529 saved configs under `data/` and
`dead_ReLU_data/` carry `spectral_rad: 1.2` and `connectivity_density_rec: 1.0` — every result in
this project sits at one initial recurrent gain. The recurrent weight MAGNITUDE penalty
(`lambda_rwm`, a soft cap on |W_rec| entries) was last given a nonzero weight in July 2025, in
`CDDM_RecurrentWeightMagnitude-{77398e,c21be0}` on the old Dale architecture at 30,000 iterations
with no participation read-out, so on the standard network it is unmeasured too.
`SilentReLU_std_recinit_della` (job 14919348, 10 jobs, 75,000 iterations) covers both: spectral
radius 0.5 / 1.2 / 2.0 and lambda_rwm 0.05 / 0.2, two seeds each, with spectral radius 1.2 as the
in-sweep control. Prediction, before the data: the count rises with the gain, since a unit that
fires early has a gradient. Falsifier: all three radii inside each other's two-seed range.

**The no-decay horizon triples.** At 100,000 iterations wd = 0 ends at 394 / 428 / 438 active units
against 321 / 314 / 301 at wd = 1e-6, and 100,000 is the longest horizon that line has.
`SilentReLU_std_wd0_long_della` (job 14919349, 3 seeds, 300,000 iterations, wd = 0, everything else
default) answers whether the count has stopped falling or is only falling more slowly. Saturation
criterion, fixed before the run: the mean count over 250k–300k inside the three-seed spread of
200k–250k, and a fitted slope under 5 units per 100,000 over the last third. These are current-code
runs, not an extension of the f5aa558 horizon series — that series has a trainable bias, fixed at
zero since 4f7d73f — so the saturation read is within-run and the 1e-6 line only bounds it.

**Tanh is the first signed activation on the standard network.** ReLU, leaky ReLU, softplus and the
shifted sigmoid are all non-negative. A tanh unit can fail in a second way with no ReLU analogue:
pinned at ±1, where the gradient is as dead as a silent unit's but the unit is loud. Participation
(std(r) + q90(|r|)) scores a pinned unit as ACTIVE, so the saturated fraction has to be read
separately from the saved parameters — |r| above 0.99 on at least 95% of samples of a noise-free
probe. `SilentReLU_std_tanh_della` (job 14919350, 150,000 iterations) runs tanh and a ReLU control,
three seeds each, in one array so the activation is the only thing that differs.

`rnn_tanh_standard.yaml` is `rnn_relu_standard.yaml` with `activation_args` changed and nothing
else. Each configuration was smoke-tested locally at N = 60 for 20 iterations before submission; the
check caught that `+trainer.lambda_rwm` alone raises TypeError ("argument after ** must be a
mapping, not NoneType") because `rwm_args` defaults to None, so the launcher passes
`+trainer.rwm_args={}` alongside it. The saved configs of the three smoke runs record
spectral_rad 2.0, lambda_rwm 0.2 and activation tanh respectively.

### Two of the three knobs read out: the gain falsifies its prediction, the weight cap adds 100 units, wd = 0 has not saturated — 2026-10-05 15:06

All three arrays from 2026-10-03 finished with every task COMPLETED and exit 0 (`sacct` on 14919348,
14919349 and 14919350). Tanh (14919350) was already in the deck, on the "tanh shows no silent units"
slide. The other two are read here with the rule fixed before submission: active if
p ≥ 0.05 × q95(p), at the last probe (74,900 / 299,900).

**Recurrent gain and weight cap (14919348, CDDM, N = 1000, 75,000 iterations, two seeds per cell).**
No cell trips the r² < 0.7 gate: every run lands between 0.861 and 0.891.

| cell | at 1.07× own floor | at end | per seed (end) |
|---|---|---|---|
| spectral radius 0.5 | 576 ± 8 | 361 ± 6 | 355, 367 |
| spectral radius 1.2 (control) | 522 ± 4 | 340 ± 7 | 333, 347 |
| spectral radius 2.0 | 474 ± 2 | 312 ± 4 | 308, 317 |
| lambda_rwm 0.05 | 590 ± 36 | 438 ± 4 | 441, 434 |
| lambda_rwm 0.2 | 566 ± 8 | 448 ± 2 | 445, 450 |

- **The prediction is wrong in sign.** We predicted that the count rises with gain; it falls, 361 → 340 → 312,
  with no overlap between the two-seed ranges. So the falsifier as written ("all three inside each
  other's range") does NOT fire: the starting gain does set the count, by about 50 units across a
  fourfold range, reversed. The effect is larger at the loss-floor crossing (576 → 474) and shrinks
  by the end, as on every other knob ladder. At the first participation probe, spectral radius 2.0 has
  already lost most units (456 and 472 active, against 760–816 elsewhere). The reading for the talk is
  that silence is not simply weak recurrent drive.
- **The weight-magnitude penalty is the bigger result.** It adds about 100 units at the end
  (340 → 438 / 448) with r² unchanged, and the two strengths sit within 10 units of each other. It
  sits beside the metabolic penalty (266 → 356–417, a different budget and baseline). This is the
  penalty's first measurement on the standard network. n = 2 per cell, so the plateau between 0.05
  and 0.2 is not yet a dose-response claim.

**wd = 0 out to 300,000 (14919349, three seeds).** Final counts are 384, 388 and 397, from 436, 434 and
430 at 100,000. Criterion 1 holds: the 250k–300k means (389.8, 387.0, 399.0) are each inside their
seed's 200k–250k range (377–422). Criterion 2 fails: the fitted slope from 200k on is −9.8, −11.6 and
−14.3 units per 100,000, against a bar of 5. So the count is decelerating and has not saturated;
the deck says so in one sentence on the weight-decay slide.

**Deck changes (`docs/presentation.md`):** two slides in ONE KNOB AT A TIME, after the metabolic one:
`slide_k_rwm` ("A cap on recurrent weights raises the count from 340 to 440, at no cost in score")
and `slide_k_recgain` ("Starting with weaker or stronger recurrence moves the count by 50 units, the
opposite way to the guess"). Plus the 300,000 sentence on the weight-decay slide. Both figures come
from two new `LADDERS` entries in `fig_knobs_at_floor.py`, which share the sr = 1.2 cell as reference.
`check_presentation.py` passes (38 slides, 2,461 words).

**How the figures were built, because the local tree could not build them.** The local tree has no
`.git`, no `data/`, and no `common.py`, `plotstyle.py` or several other helpers that the figure scripts
import. The build ran on Della in `/scratch/gpfs/TENGEL/pt1290/figbuild_recinit`: the local
`trainRNNbrain/`, `style/` and `configs/` were copied there, the missing helpers were filled from
`~/trainRNNbrain` (branch dimensionality-analysis, 99c877a) with `cp -n` so no local file was
overwritten, and `data` was symlinked to the scratch data. `pdftoppm` exists on neither machine, so
the PNGs were rasterised with macOS `sips -Z 1462`, which matches the width of the existing knob
PNGs. The same was done for `slide_k_tanh_saturation.png`: the deck already linked it, but it was
missing, and the checker failed on it. `docs/presentation.html` was NOT rebuilt, because
`deck_html.py` needs `gh`, which is not installed here. Nothing was committed, because this tree is
not a git checkout.

### Gradient read-out: ReLU silence IS gradient loss; tanh loses gradient across the whole population, not in a split — 2026-10-05

Pavel asked for a silence measure that does not depend on the activation: run the trained network
noise-free on its own training batch, backpropagate the masked MSE, and count units whose gradient
clears 0.05 × q95, the same scale-free bar. Script `trainRNNbrain/experiments_and_analysis/gradient_participation.py`,
job 15064275 on Della (run from `~/trainRNNbrain_recinit` @ e00924b), on `CDDM_std_g0_tanh`, with 9 ReLU
and 9 tanh seeds, CDDM, N = 1000, 150,000 iterations. Per-unit JSONs are in
`/scratch/gpfs/TENGEL/pt1290/figbuild_recinit/gradpart/`. Each network is also read at its untrained twin,
which is rebuilt from the saved seed the way `run_experiment.py` builds it. Its participation matches the
trace's iteration-0 probe at r = 1.000 in all 18 runs.

The measures:
- `g_state`: the RMS over time and trials of dL/dh_i(t).
- `g_in`: the incoming-weight gradient norm.
- `g_out`: the outgoing-weight gradient norm.
- `slope`: the mean of f'(h_i), the fraction of the gradient the unit passes.

**ReLU: the gradient count reproduces the participation count.** Trained g_state counts are 269–292,
against participation counts of 281–301. Per seed, the two rules agree on 264–282 units, 12–25 units
are counted by participation only and 1–11 by gradient only. The g_in and g_out counts give the same
number (258–281). At the untrained twin g_state counts 802–863. The working units carry MORE
gradient per unit of error than the untrained median unit: per √loss they are 4–5× larger. The
quiet ~720 carry none, with a median g/q95 of 0.00.

**tanh: no split, a population-wide fall.** The g_state count stays at 660–1000 (seven seeds of nine
at 986 or more), because the distribution is unimodal and the relative bar sits below all of it. What
falls is the level:
- The median slope falls from 0.93 at the untrained twin to 0.05–0.64 trained (median across seeds
  0.20).
- The median g_state per √loss is 6–250× smaller than at the untrained twin (median 27×).
- The seeds differ hugely. Units with slope < 0.1 number 950, 318 and 233 in three seeds and 3–24 in
  the other six. Three seeds keep 688–921 units with slope ≥ 0.5.

The incoming-weight count (227–898) is NOT used. At a trained stationary point the batch-summed weight
gradient cancels by construction, so it measures leftover non-stationarity, not whether gradient
reaches a unit. The per-sample state gradient has no such cancellation. The per-√loss normalisation
assumes the gradient scales linearly with the residual, which is approximate.

Deck: the "because its units die at ±1 instead of at 0" clause was removed from the tanh slide title
at Pavel's request. The slide's figure is still the old two-panel one and is to be replaced.

**Two candidate tanh panels, drawn like the participation histogram (2026-10-05).**
`fig_gradient_by_unit.py` reads the gradient_participation JSONs and writes `slide_k_unit_gradient` and
`slide_k_unit_slope`. Each has two panels, ReLU and tanh, one seed each (the seed with the median count of
nine). The trained histogram is split at the criterion and the same network untrained is drawn as an
outline. The criterion is 5% of the UNTRAINED network's 95th percentile. The trained network's own q95
moves down with a tanh population that shifts as a whole, so a bar set by it counts 660–1000 in every
seed. The gradient is divided by √loss so that the untrained and trained networks compare per unit of
error.
- Units above the bar, gradient: ReLU 274–319 (drawn seed 285), tanh 19–964 (drawn seed 97). In seven
  tanh seeds of nine it is 19–195, and two seeds keep 944 and 964.
- Units above the bar, slope: ReLU 282–315, tanh 535–1000. The tanh slope falls about 9× (median 0.93
  to about 0.1 in the drawn seed) but stays above a 5% bar.
- The gradient falls more than the slope, because it also carries how little the loss depends on
  each unit's rate downstream.
Built in the Della scratch build directory; the PNGs are not yet in `img/internal_figures`, pending a
choice.

### Three loss-of-gradient measures, per unit, ReLU vs tanh — 2026-10-05

Ran `gradient_sensitivity.py` as Della job 15066631: the same 18 networks as above (9 ReLU, 9 tanh), each
measured untrained and trained. Bar = 5% of the untrained network's q95. Units above the bar, 9 seeds:

| measure | ReLU | tanh |
|---|---|---|
| A: loss gradient on the state, /√L | 274–319 | 19–964 (two seeds at 944 and 964) |
| B: per-trial loss gradient on incoming weights, /√L | 277–305 | 65–238 |
| C: output sensitivity to incoming weights (no loss) | 277–293 | 25–147 |
| C′: output sensitivity to the state | 275–289 | 21–82 |

- **B is computed per trial** because the batch weight gradient cancels at convergence. Each trial's
  row norms come from a Gram-matrix identity, which was checked against plain autograd on two trials
  (relative error 5.6e-7 and 3.8e-7).
- **C is a Hutchinson estimate** with 16 random cotangents over the scored outputs.
- **The two tanh outliers of A fall to 51 and 82 under C**, so they come from the √L normalisation,
  not the networks.
- **On ReLU all four measures agree with the participation count.**

The deck now carries A (`slide_k_unit_gradient`) and C (`slide_k_unit_sensitivity_in`). The slope
panel moved to the appendix. B, C′ and the seed strip (`slide_k_unit_measures_seeds`) are built in the
Della scratch build directory and are not in the deck.

Removed from the talk at Pavel's request: `slide_wh_treadmill`, listed under cut panels in the
appendix.


