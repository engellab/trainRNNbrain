# Paper audit: claims, status, and what is left

The status board for the silent-units paper. Update it whenever a result changes a claim; record the
result itself as a new entry in `docs/project_trajectory/`. How networks are read (the 1.05×
noisy-loss floor, active units, held-out R²) is in `docs/AGENT_GUIDE.md`; every number here is read
that way unless it says otherwise.

## The paper in one paragraph

Neuroscientists train RNNs on tasks animals do, then read the units like a recorded population.
After training, most units take no part in the computation, and that is not brain-like. The task
loss reads only the output, so nothing in it asks a unit to stay involved, and training sheds the
surplus. A shed ReLU unit goes silent; a shed tanh unit saturates. Bigger networks keep a smaller
share of their units active, and longer training keeps shedding them after the loss has stopped
improving. The standard knobs change how many units are lost, not the trend. Rules that give
participation a drive of its own recruit units: bio-inspired ones (a two-sided firing-rate target,
"the urge to spike, but not to overspike"; noisy synapses; rescaling a silent unit's incoming
weights) and engineering ones (prune-and-duplicate, read-out dropout). As Dohare et al. (2024)
concluded for continual learning, gradient descent on the task loss alone does not keep units
participating; participation needs a drive of its own (decision dead-relu-paper#17).

**Terms.** *Standard network*: see the guide. *Pre-standard*: older sweeps with a trainable bias.
*Multiplier*: an arm's active-unit count ÷ its own control's count, at the same N and read-out.
*Rate penalty* (`frm`): a two-sided penalty around a target firing rate. *Sparsity penalty* (`rws`):
charges a unit for spreading its input over many partners. *Tasks*: 3-bit flip-flop, CDDM
(context-dependent decision making), DMTS (delayed match-to-sample). The paper itself uses the
round-2 glossary (blueprint section 2, decision dead-relu-paper#36): default network, fold change,
per-unit rule, setting sweep, floor crossing, 75,000 iterations, rate-penalty size test (B2),
combination test (B3), transfer test (B4), optimizer comparison (B5), removing the added units (A3).
The board keeps its internal labels.

**Figures**, as the paper has them (dead-relu-paper `main` 46622b2, every round-2 package merged).
F1, inactive units: (a) one 3BFF network, (b) each unit's participation before and after
training, (c) active units at N = 1000 on the three tasks, (d) held-out R² as units are removed.
F2, network size and training time: (a) active units against N, (b) the active share, (c) the
fitted size exponent, (d) N = 1000 networks over training, (e) 3BFF networks trained past 150k.
F3, standard training settings: (a) every value of every setting sweep at 75k, (b) weight decay on
CDDM over training, (c) the initial input-weight scale on 3BFF, (d) the optimizer comparison, (e)
the two revival methods (input-weight cap; re-initialising inactive units). F4, routes back to
activity: (a) net input current of a silent, a low-active and an active unit on CDDM and 3BFF,
(b) incoming-weight gradient against the share of steps with positive input (gradient reach),
(c) share of each unit class active 500 and 5,000 iterations after the floor crossing, (d) ReLU
against tanh, gradient on the net input per unit of error, (e) transfer test: share of the
inactive units active on the new task, (f) transfer test: active units against untrained
networks. F5, per-unit rules: (a) active units over training at N = 1000, (b) active units
against R² at N = 1000 and 75k, (c) fold change and R² cost against N, (d) the rate penalty
alone, (e) removing the added units, (f) the combination test. Extended Data and Supplementary
items are cited here by content, because their printed numbers follow first citation (blueprint
section 7).

## Claims

| # | Claim | Status | What remains |
|---|---|---|---|
| C1 | Training leaves most units inactive, and a network retrained without them scores within 0.004 R² of the intact network retrained the same way | **Done**: Figure 1 rebuilt (a-d) and panel d re-run at the current floor, rule passed (31 Oct); the fourth 3BFF network at N = 1000 is counted (decision dead-relu-paper#20, option 2). Round 2 (fig1-r2, dead-relu-paper#52, PR #65; entry 2026-11-01_0716): Fig. 1c, Results 1 and the caption with four 3BFF networks; the Extended Data figure of removing inactive units at every size drawn on 35 networks | — |
| C2 | Bigger networks keep a smaller share of units active | **Done**: Figure 2 (31 Oct, entry 2026-10-31_1957). Round 2: threshold sensitivity recomputed on the networks Fig. 2a fits (methods-r2, dead-relu-paper#51, PR #64; entries 2026-10-31_2209 and _2211); Figure 2 and the Extended Data figure of floor crossings against snapshots redrawn (fig2-r2, dead-relu-paper#53, PR #67; entry 2026-11-01_0716) | — |
| C3 | Longer training keeps shedding units, and no run has stopped | **Done**; B5 complete, read at 75k (31 Oct); Fig. 2d,e and the optimizer comparison in Fig. 3d and the Extended Data figure of every standard-setting sweep (fig2-r2, fig3-r2) | — |
| C4 | On CDDM 260-313 of 1,000 units are silent (never above zero on the noise-free batch, zero gradient there, 60-290× below the reach threshold under training noise); on 3BFF none is, and low-active 3BFF units get gradient yet stay below threshold | **Done**: re-read at the current floor, rule passed (31 Oct, entry 2026-10-31_2207); Figure 4a-d, the Extended Data figures of gradient reach and of tanh (fig4-no-return, dead-relu-paper#8, PR #60) | Results 4's tanh sentence ("most tanh units ... are left with little gradient") fails in 2 of the 9 networks on the state-gradient measure, and its 25-147 against 277-293 counts cite panels b,c of the tanh figure where only b has them (dead-relu-paper#70; owner layout-r2, #58) |
| C5 | No standard setting stops the fall of the active count; at 75k CDDM settings move it by +116/−177 units, 3BFF input scale raises it 1.7× and the gain shrinks | **Done at 75k** (31 Oct, entry 2026-10-31_2010): Figure 3 rebuilt; both measurement points in the Extended Data figure of every standard-setting sweep; round 2 (entry 2026-11-01_0735): Figure 3 redrawn at print width with panel a cut to 9 rows, the optimizer comparison merged into that Extended Data figure, every network with a floor crossing checked to have fewer active units at the end of training (116 of 116 distinct networks, rate penalty aside; entry 2026-11-01_1000 corrects the earlier 124, which counted 8 default networks shared between sweeps more than once); round 3 (dead-relu-paper#83, #87, #88): filled markers at the floor crossing, open at 75,000 iterations, Fig. 3c median traces, Fig. 3d legend outside the data; merged by fig3-r2 (dead-relu-paper#54, PR #69), with the wording of decision dead-relu-paper#24 (option 1) and the CDDM rate-penalty values in Fig. 3a as a labelled reference (decision dead-relu-paper#35 item 4) | — |
| C6 | Rules that give participation its own drive recruit units at a small cost | **Done**: remedies read at the floor and A3 done (31 Oct, entry 2026-10-31_1732); B2 failed its rule; Figure 5, Results 5, the Extended Data figures of the rate penalty alone on 3BFF and of per-unit rules across sizes at three margins, the table of per-unit rules at every size and the table of pre-registered tests built (entry 2026-10-31_1953, dead-relu-paper#9); rescaling and read-out dropout above N = 1000 reported at 75k (decision dead-relu-paper#16, 31 Oct). Round 2 (fig5-r2, dead-relu-paper#55, PR #72; entries 2026-11-01_0745, _0801, _0813, _0833): Figure 5 redrawn with panel a the active count over training at N = 1000; the rate penalty told once, in Results 5, with both tasks (decision dead-relu-paper#35 item 4); every R² checked against loss bursts and non-finite stretches, and the affected values ringed in Fig. 5 and counted in the per-unit-rule table (decision dead-relu-paper#18); the Discussion states the limitation (frame-discussion, PR #73) | — |
| C7 | Participation needs a drive of its own; inactive units do not become active again | **Done**: B4 supports "do not become active again" at N = 1000 and 4000 (31 Oct); B3 inconclusive (decision dead-relu-paper#17, 31 Oct); the conclusion is "a drive of its own". Results 4 and Fig. 4e,f (transfer test) merged (fig4-no-return, PR #60); B3 in Results 5, Fig. 5f and the Extended Data table of pre-registered tests (fig5-r2) | The combination test's R² of 0.889 (rate penalty with noisy synapses, at the floor crossing) comes from one of the two networks with a crossing, the other's weights lying inside a loss burst; neither Results 5 nor the table of pre-registered tests says so (dead-relu-paper#27, round 3) |
| C8 | Counting the inactive units changes some population statistics computed over all N units | **Done** (1 Nov, entries 2026-10-31_2212 rule, 2026-10-31_2220 results; trainRNNbrain wp/pop-stats): participation-ratio dimensionality of the rates *unchanged* on all three tasks (at most 1.1% apart); normalised-rate dimensionality and the selective and mixed-selective shares *change* on all three; k-means classes *change* on 3BFF and CDDM (k* 4 vs 3), *unchanged* on DMTS. On 3BFF the selective count is unchanged (0-1 inactive units selective) and only the share's denominator moves; on CDDM 119-150 inactive units are selective. Post hoc (entry 2026-11-01_0741, PR #66 review): the S2 gap depends on the normaliser. With 5% of the active units' median sd added to each divisor it is +21-24% on CDDM (13.3-13.9 vs 10.9-11.2) and +24-33% on 3BFF, and on DMTS it reverses sign (-21 to -48% becomes +9 to +13%). S4 on 3BFF comes from a flat-profile class of 146-252 inactive units without selectivity (selective units alone give equal k*); on CDDM 4 vs 3 rests on silhouette margins of 0.01-0.07 (at fixed k = 3, shares within 5.6-7.3 points). The CDDM inactive selective units respond a tenth as strongly as active ones. No N = 4000 CDDM weights within 5% of the crossing exist. In the paper: the Results 1 population paragraph, the Extended Data figure of population statistics and Methods "Population statistics" (pop-stats, dead-relu-paper#56, PR #66); the Introduction and Discussion name S1 as unchanged and the equally weighted statistics as changed, with no 4,000-unit CDDM sentence (frame-discussion, dead-relu-paper#57, PR #73) | — |

**C1.** Figure 1 (round 1, entries 2026-10-31_1702 and 1755). Active units at N = 1000, each run of
the size series read at 1.05× its floor (read-out 79932ef): 3BFF 277.5 ± 26.6 (n = 4: 296, 277,
297, 240; s3374117298 counted per decision dead-relu-paper#20, option 2; as in the Supplementary
table of every network of Figs. 1 and 2), CDDM 431 ± 10, DMTS 218 and 422 (a third
run has no read-out), at held-out R² 0.95, 0.83 and 0.999 (estimated from the end of training).
Ablation, re-run on 31 Oct at each network's current crossing on
three snapshot networks per task: removing every inactive unit leaves CDDM at 0.83 (changes of
−0.003 to 0.000) and lowers 3BFF from 0.95 to 0.26-0.52 and DMTS from 0.999 to 0.75-0.99; as many
random units take every network to 0.09 or below. A 3,000-iteration retrain without the inactive
units ends within 0.004 of the intact network retrained the same way on every task (3BFF 0.945-0.948
against 0.949-0.950); random pruning plus retrain reaches 0.920-0.930 on 3BFF and ends within 0.01 of
the inactive-pruned retrain on CDDM (0.821-0.828 against 0.825-0.829); on DMTS it reaches 0.998 in two networks and 0.419 in s3322309204. The pre-registered rule (entry 1702) passes in all nine networks. Across sizes (the Extended
Data figure of removing inactive units at every size; entry 2026-11-01_0716): 35 default networks
of 500-4000 units, 3BFF (15) at 75k and CDDM (12) and DMTS (8) at the end of training; the 36th,
DMTS s597040509 at N = 2000 (R² 0.417 in the Supplementary table of every network), never learned
the task and is left out. Removing the inactive units changes R² by −0.010 to +0.001 on CDDM and
−0.046 to −0.001 on DMTS, and in all 35 networks it costs less than removing as many units at random.

**C2.** Figure 2 (entry 2026-10-31_1957; Della job 16558986, read-out 79932ef). Read at 1.05× each
run's floor and fitted on the size means from N = 500, the active count grows as N^0.48 on the 3-bit
flip-flop (41% of 500 units active, 14% of 4000), N^0.44 on CDDM (298 of 500, 767 of 4000) and N^0.55 on
DMTS (four sizes now; 191 of 500, 620 of 4000 from the one N = 4000 network that reads out). At
N = 100, 66-90% of units are active (drawn, not fitted). At 1.05×, 1.10× and 1.20× the exponents lie
between 0.44 and 0.61 and every bootstrap 95% upper bound is below 0.79, so the rule of entry
2026-10-31_1900 says C2 holds on all three tasks. 1,000 active units would need N ≈ 7,300 on CDDM,
9,200 on DMTS and 14,200 on the flip-flop. The learning-rate confound (lr ∝ N^−1/3) is left in and
handled by reading each network at its own floor; mean crossings move from 59.0k to 65.0k (flip-flop) and
26.7k to 32.5k (CDDM) between N = 500 and 4000, "by about a fifth or less" in Results 2. The fourth flip-flop network at N = 1000
(s3374117298) is counted (decision dead-relu-paper#20, option 2), as in C1: 277.5 ± 26.6, n = 4.

**C3.** CDDM, N = 1000: every weight-decay setting reaches its floor early and keeps losing units to
200-300k with no gain in loss. Flip-flop (Figure 2e, entry 2026-10-31_1957): from 150k, 13 of 15 runs
with a flat loss (< 3% change per 100k) lose 8-121 units per 100k: N = 500 4 of 5, N = 1000 5 of 6
(the two 300k runs 23 and 37 per 100k, slowing to 11-22 in their last 100k, entry 2026-10-15_0955),
N = 4000 3 of 3 (31-121); N = 2000 has one flat run (57 per 100k), so Results 2's "13 lost units, at
each of the four sizes" rests on that one network at N = 2000. **Not the optimizer (B5), read at 75k** (entry 2026-10-31_2010; all 13 arms, SGD lr 3 wd 0 copied
from Spock): Adam (wd 1e-6) has 233-270 active (mean 246) at R² 0.951; Adam with wd 0 (mean 305) and
AdamW (306-358, mean 325) have more at a higher R² (0.956-0.959), the direction weight decay 0 takes on
CDDM; SGD has more units only at lower R² (721-759 at lr 0.03, R² 0.873-0.875) and 212-285 at lr 0.3-1
(R² ≤ 0.915); lr 3 does not learn (R² 0.21-0.44 at 200k). Every Adam/AdamW arm keeps losing units to
200k (169-226 at 200k; means from entry 2026-10-31_2035). Under the current read-out most SGD runs have no floor
read, so the 14 Oct floor numbers are withdrawn. Figure: Fig. 3d at 75k; the floor crossing,
200k and SGD learning rate 3 are a panel of the Extended Data figure of every standard-setting
sweep (fig3-r2 merged `fig_ed_optimizer` into it, PR #69). Results 3 states the AdamW and
no-weight-decay gain as "fewer than 80 more" active units (+59 and +79 over Adam's mean of 246) "at an
R² higher by less than 0.01".

**C4.** Re-read at the current floor (entry 2026-10-31_2207, rule in 2026-10-31_2137, Della job
16566988), on the six Figure 1d networks. On CDDM 260-313 of a network's 1,000 units are silent:
their net input current is at or below zero on every step of every trial of the noise-free
reference batch, so there their rate and the loss gradient on their incoming weights are exactly
zero (numpy float64 and autograd). Under the training noise they are above zero on about 1.6% of
steps and their gradient is 60-290 times (median per network) below the reach threshold; 8 of 866
cross it in some batch (entry 2026-11-01_0716, which withdraws "only through weight decay"). At
most 1 silent unit per network is active 5,000 iterations later. On 3BFF no unit is silent; the gradient
reaches 143-165 of the 656-732 low-active units (5% of the network's q95 of g_in), whose current is
positive on a median 3-4% of steps, almost always within 2τ of an input pulse. 5,000 iterations
later 0.7-5.5% of them are active (0-2.1% after 500). On CDDM 20-39% of the 39-56 reached
low-active units are active by then (turnover near the criterion while the count falls). The reach
does not shrink with training (C4b, entry 2026-10-14_1218: 199/155/175 reached at 50k, 174/152/156
at the end, no fewer after lr/10). Tanh (CDDM, N = 1000, 9 networks, end of 150k, not a floor read):
0 inactive units; median per-unit state gradient per unit of error 27× lower than untrained (6-247×);
output sensitive to the incoming weights of 25-147 units against 277-293 for ReLU. Figure 4a-d,
the Extended Data figures of gradient reach and of tanh. The paper words the silent criterion as
"negative on every step after the initial state (x = 0) of every trial" (Methods, Results 4 and
the Introduction; decision dead-relu-paper#59), because x = 0 at step 0 of every trial.

**C5.** Read at 75k, the same-size rule (entry 2026-10-31_2010; Figure 3, and the Extended Data figure of every
standard-setting sweep at both measurement points). On CDDM the default has 325-345 active at R² 0.833-0.835; 18 non-default values of weight
decay, noise, metabolic penalty, spectral radius, recurrent-weight magnitude penalty, activation and sparsity give means of
160-453 (largest gain +116, weight decay 0; largest loss −177, weight decay 1e-4); metabolic λ = 10
costs R² (0.787-0.789) without adding units. The count falls under every setting (weight decay 0:
466-494 at the floor, 442-469 at 75k, 400-428 at 200k). On 3BFF the input-weight norm 20 (400× the default draw's mean norm of 0.050; entry 2026-10-31_2051),
the largest gain at equal R², keeps 455-500 (mean 477) against 245-296 (mean 275) at estimated R² 0.953 vs 0.951 (1.7×), down from 2.5× at the floor (means 718 vs 283) and 1.6× at 150k (312-340
vs 189-235); wording "no standard setting stops the loss of active units" (decision
dead-relu-paper#24, option 1). On CDDM (default mean norm 0.074), input-weight norms 0.5-20 lower the count at 50k (166-336 vs 355-376).
Interventions designed to raise the count add at most 60 units: the CDDM input-weight cap 0.3 gives
390-400 at R² 0.827-0.832 (vs 332-340 at 0.833-0.835); re-initialising inactive units (CDDM 330-346)
and redrawing them on 3BFF (205-274 vs 236-288) do not raise it. Only with the rate penalty, alone or
with the sparsity penalty, are all 1000 units active: held-out R² 0.828-0.830 with the rate penalty
alone (λ_frm = 0.2) against 0.831-0.834 without it, 0.004 lower on the mean ("about 0.004" in
Results 5 and the abstract), and 0.008 lower with both penalties (entries 2026-11-01_0735 and _0813).

**C6.** The remedy top-ups (RMtop200, 267 runs in 83 cells) are read at 1.05× each run's own
noisy-loss floor (entry 2026-10-31_1732; `read_remedies.py` 2d32b3b). 146 runs are read and 121
excluded. 92 of the exclusions never stay under the bar: their loss is still falling at 200k
(`rm_exclusion_diag.py`). Noisy synapses (σ_w = 1) are read in every run at every size and give
1.81×, 2.03×, 1.54× and 1.60× the control's active count at N = 500, 1000, 2000 and 4000, at an R²
cost of 0.021-0.025 for N ≥ 1000 (0.043 at 500). Prune-and-duplicate gives 2.06-3.59× at a cost of
at most 0.018. Rescaling and read-out dropout have no floor read at N = 2000 or 4000 (0 of 16 runs);
at 75k they give 1.7-2.0× there, and 1.55-2.05× across N = 500-4000 (Results 5: "1.5- to
2.1-fold"; the per-unit-rule table). B2 (rate penalty alone, entry 2026-10-31_1532) fails its pre-registered
rule: no run at N = 500 or 1000 reaches a floor read, and at N = 2000 it costs 0.024 in held-out R²
(0.926 against 0.950). B2's third test, against synaptic noise, does not falsify: 4.81× against
1.54× at N = 2000, and 5.64× against 1.60× at 4000. A3 (75k): removing the recruited units (active
units beyond the control's count, least participating first) lowers R² from 0.93-0.95 to 0.18-0.62
in every remedy at N = 1000 and 4000. Removing as many random active units costs more for four rules
at N = 1000 (median R² 0.06-0.16 against 0.22-0.58) and for all five at 4000; read-out dropout at
N = 1000 is the exception, where the two cost the same (0.18 against 0.19; entry 2026-10-31_1953).
Figure 5 (entry 2026-10-31_1953) draws rescaling and read-out dropout at 75k at every size (decision
#16); at 1.20x their floor margin has reads in 15 of 16 runs at
N ≥ 2000, giving 1.60-1.90x (dropout) and 1.23-1.81x (rescaling) across N = 500-4000 (the
Extended Data figure of per-unit rules across sizes at three margins). Burst check (round 2,
entries 2026-11-01_0745 rule and _0801 result): no R² at a floor crossing of any per-unit-rule cell
lies inside a loss burst or a non-finite stretch; 21 networks of the per-unit-rule table have
their 75k weights inside one (rescaling 14, rate penalty 3, rate + sparsity 3, prune-and-duplicate
1), and Fig. 5 rings every affected value it draws. The 3BFF rate-penalty networks at N = 2000 and 4000 have
500-3,686 non-finite loss values each (entries _0813 and _0833).

**C7.** B4 at N = 1000 (entry 2026-10-31_1532): when a trained flip-flop network learns the 3-bit
hyper flip-flop, 5-9% of its inactive units become active, and it ends with 232-259 active against
421-457 for an untrained network trained on the new task; R² is the same (0.91). Verdict, by the
pre-registered rule: inactive units do not come back. B4 at N = 4000 (entry 2026-10-31_1750) gives
the same verdict: old networks revive 2.3-7.8% of their inactive units and end with 446-613 active
against 806-956 for untrained networks; R² 0.909-0.914 for both. Figure 4e,f draws both sizes from
`read_b2_b4.json` of Della job 16553903 (dead-relu-paper#8); `fig_paper_F4_noreturn.py` recomputes
the gate and verdict from the per-run rows and matches the read's own at both sizes. B3 (N = 1000, rate × noise ×
rescaling): the rule binds at the floor, where its literal verdict is "supports several sources",
reached only through its stability clause. Rate+noise reads at the floor and rate alone does not,
because its loss is still falling at 200k. Rate+noise has 970 active units against 556 for noise
alone, at a cost of 0.04 in R². At 75k the verdict is between the bars. Decided on 31 Oct
(dead-relu-paper#17, manager on Pavel's delegation): B3 is treated as between the bars. Figure 5f and
the Extended Data table of pre-registered tests state both measurement points; Results 5 does not take B3 as evidence that raising the active count
requires combining rules. The burst check (entry 2026-11-01_0801) finds six B3 R² values from weights
inside a loss burst or a non-finite stretch, all with the rate penalty; Fig. 5f rings them and the
Discussion gives the count. Rate+noise's floor R² of 0.889 rests on one network (the other's
snapshot scores −27.5 inside a burst; dead-relu-paper#27, open).

## The remedies catalogue

*Bio-inspired*: uses only the unit's own activity and synapses and could run online. *Engineering*:
copies units, ranks the whole population, or acts outside the dynamics. Flip-flop multipliers are
read at 1.05× each run's own floor (RMtop200, entry 2026-10-31_1732), at N = 1000 and then 4000;
'75k' marks a cell with no floor read at that size.

| Remedy | Class | What it does | Active units | Status |
|---|---|---|---|---|
| Rate penalty | Bio-inspired | Two-sided penalty around a target rate | CDDM: all 1000; flip-flop 4.8× at N = 2000, 5.6× at 4000 | Flip-flop alone (B2): unstable at N ≤ 1000, −0.024 R² at N = 2000; its multiplier is above synaptic noise's |
| Noisy synapses | Bio-inspired | Each weight × (1 + σ_w·noise), every step | 2.03×; 1.60× at N = 4000 (read in 14 of 14 runs, N = 500-4000) | −0.021 to −0.025 R² at N ≥ 1000 |
| Rescaling incoming weights | Bio-inspired | Tilts a silent unit's input row toward its positive weights, at fixed length, until a release target | 1.51× (1 of 4 runs read); 1.95× at N = 4000 at 75k, no floor read | No floor read at N ≥ 2000: loss still falling at 200k (dead-relu-paper#16); 1.23-1.81× at 1.20× the floor (Extended Data, per-unit rules across sizes) |
| Threshold lift | Bio-inspired | Raises a silent unit's bias once | 1.36× | Screen only |
| Prune-and-duplicate | Engineering | Overwrites a silent unit with a jittered copy of a working one | 2.77×; 3.59× at N = 4000 | −0.005 R² at N = 1000, −0.018 at 4000; fails on DMTS above N = 500 |
| Read-out dropout | Engineering | Randomly hides busy units from the read-out | 1.81× (1 of 3 runs read); 2.01× at N = 4000 at 75k, no floor read | No floor read at N ≥ 2000 (dead-relu-paper#16); 1.60-1.90× at 1.20× the floor (Extended Data, per-unit rules across sizes); fails on DMTS |
| Full dropout | Engineering | Randomly removes busy units from the dynamics | 904/1000 | Parked: costs 12-17 R² points |
| Redraw silent units | Engineering (≈ continual backprop / ReDo) | Re-initialises a silent unit | 0.84-0.96× | Negative (Fig. 3e) |
| Rate + sparsity | Bio-inspired pair | Both penalties | 3.51×; 5.87× at N = 4000 (1 run) | Not in Fig. 5; reported in the per-unit-rule table, the Extended Data figure of per-unit rules at three margins, and on CDDM in Fig. 3a (all 1000 active, −0.008 R²) |

## Parked (do not restart without a `decision` issue on dead-relu-paper; PROTOCOL.md, 31 Oct)

- The selective/modular solution under rate + sparsity: burst units, assemblies, penalty switching.
- Single-unit selectivity, clustering, cell types, NMF, wiring-versus-activity. Exception: the four
  population statistics of decision dead-relu-paper#34 (C8), measured by pop-stats.
- The task-demand ceiling (M*, M ∝ N^b k^c), the Walsh and hyper flip-flops as a demand series,
  MultiRule15, Yang 20-task (except the A5 supplementary panel).
- Long-delay DMTS.
- Master-inhibitor, frozen-clamp and silent-at-init rescue tests.
- Dale-constrained and sign-constrained networks (one supplementary panel at most).
- Loss-floor laws, convergence criteria, drift exponents, lognormality of rates and weights.

## Open work

**Analyses (no new training)**
- **A3.** Done 31 Oct (entry 2026-10-31_1732). Recruited units carry task signal in every remedy.
  Remedy networks lose less to random 10% removal (median R² 0.58-0.86 against 0.48 for the control
  at N = 1000).
- **A5.** Supplementary task-demand panel (`fig_supp_task_demand`) is drafted on pre-standard data:
  harder tasks recruit more units, no solved task fills the network. Not in the paper (round 2
  has no task-demand item, and the blueprint plans none); a re-run on the standard network would be
  needed before it could go in.

**Training**
- **B2.** Done, read 31 Oct: fails its rule (entry 2026-10-31_1532). Third test (against synaptic
  noise) does not falsify.
- **B3.** Read 31 Oct (entry 2026-10-31_1750, Della job 16553903). At the floor: literal "supports",
  through the stability clause only. At 75k: between the bars. Rate+rescale skips 44-67% of updates
  and the triple 8-11%. Under today's read-out the λ = 0.03 pilot rung reads at 0 of 3 runs (2 of 3
  when it was chosen). Stage 2 not run (decision dead-relu-paper#17: inconclusive, no stage 2).
- **B4.** Done. N = 1000 and N = 4000 both: inactive units do not come back (entries
  2026-10-31_1532 and 2026-10-31_1750).
- **RMtop200.** Read 31 Oct (entry 2026-10-31_1732).
- **Redo with snapshots (not scheduled; after the initialisation question):** retrain the size-series
  controls with snapshots every 1000 iterations (~800 GPU-h, so it needs the editor's written
  go-ahead under PROTOCOL.md). CDDM regenerates exactly; flip-flop and DMTS give new networks.

**Writing**
- **W1.** Manuscript rewrite to this framing: merged into the paper's `main` (15 Oct); the
  Introduction now states the C4 result with the glossary terms silent and low-active
  (frame-discussion, PR #73).
- **W2, W3.** "Not brain-like" pinned to measured properties (Discussion, second paragraph), and
  the criteria for per-unit rules: merged. Round 2 replaced the board's bio-inspired/engineering
  split with three criteria, local, continuous and a biological counterpart (Discussion and the
  Supplementary table of training rules sorted by three biological criteria); the board's
  catalogue keeps its old classes.

**Should**
- Published continual backprop and ReDo as baselines on the flip-flop (N = 1000, 4000).
- A third seed on the weight-cap and spectral-radius ladders.
- One Dale panel from existing runs.

## Open questions

Decisions are the team's (PROTOCOL.md, 31 Oct): an open question is settled on a `decision` issue on
dead-relu-paper and recorded in a notebook entry.

- **Does the old correlated initialisation change silencing?** 3 seeds at N = 1000, CDDM and the
  flip-flop, independent draws against the current controls (~45 GPU-h); entry 2026-10-15_2206.
  Not scheduled in round 2; the Discussion states the shared initial draw as a limitation ("the
  counts hold for this initialisation").
- **C4 wording** settled: the glossary terms silent and low-active (decision dead-relu-paper#36),
  defined in Methods (activity) and Results 4 and used in the Introduction.
- **Redo with snapshots** (above). It should start after the initialisation question, so the redo
  is not run twice.

## Decisions taken

Round 2 (31 Oct, editor; rationale on each issue and in notebook entry
2026-10-31_2210_round-2-decisions):

- **Population statistics: measure** (dead-relu-paper#34, option 1). Four statistics over all units
  and over active units on the nine N = 1000 networks of Fig. 1d, under a rule fixed before any value
  was computed; the paper keeps a general claim only for statistics that pass (C8, pop-stats,
  dead-relu-paper#56).
- **The rate penalty is told once, in Results 5** (dead-relu-paper#35, item 4). Results 3 keeps the
  CDDM values in Fig. 3a as a labelled reference; Results 5 gives the CDDM result (λ_frm = 0.2, 75k)
  and then the 3BFF result (λ_frm = 0.1, fails the rate-penalty size test), and attributes the
  difference to neither task nor strength. This settles the earlier open question of which biological
  penalty goes in the main text: the rate penalty stays, with both results.
- **One name per thing** (dead-relu-paper#36): the glossary of blueprint section 2.
- **No change to the floor-crossing code this round** (dead-relu-paper#18, option 2). fig5-r2 marks
  every R² affected by loss bursts or non-finite stretches; the Discussion states the limitation.
- **Threshold sensitivity recomputed on the networks Fig. 2a fits** (dead-relu-paper#21, comment 3
  item 2), DMTS N = 500-4000 included, so that Methods gives Fig. 2a's exponents at the factor 0.05.
- **Extended Data held to 10 items** (blueprint section 7; editor's round-2 plan on
  dead-relu-paper#1): the task schematics, the list of every network, the per-unit-rule table at
  every size and the criteria table move to Supplementary Information; the optimizer comparison
  merges into the Extended Data figure of every standard-setting sweep.

Round 1 (31 Oct, manager on Pavel's delegation):

- **Rescaling and read-out dropout above N = 1000 are reported at 75k**, labelled, with their 1.20×
  floor crossing in Extended Data; no continuation past 200k (dead-relu-paper#16).
- **B3 is inconclusive** ("between the bars"). The conclusion is "participation needs a drive of its
  own", without "several"; no stage 2 (dead-relu-paper#17). B3's literal floor verdict, "supports
  several sources", came only through its stability clause and is reported as history.
- **The fourth 3BFF network at N = 1000 (s3374117298) is counted**: 277.5 ± 26.6, n = 4
  (dead-relu-paper#20, option 2). It supersedes the 15 Oct entry below.
- **Results 3 and Figure 3 are about the fall of the active count** ("no standard training setting
  stops the loss of active units"; dead-relu-paper#24, option 1).

Earlier:

- Read-out: the noisy training loss, 1.05× its fitted floor, with the smoothed-minimum fallback for
  plateaued fits; never the noise-free loss.
- Standard network only; the 3-bit flip-flop trains to 200k.
- The biological penalty in the main text is the rate penalty alone; rate + sparsity goes to the
  supplement.
- Two main tasks: CDDM carries the knobs, the gradient analysis and tanh; the flip-flop carries the
  remedies. DMTS appears as a third series on the size curve and in claim 1.
- B4 is in the paper, in a minimal form (N = 1000 and 4000, one new task, 3 seeds).
- The rate penalty's R² cost on CDDM (< 1%) is treated as negligible.
- Held-out scoring is seeded project-wide; new runs keep snapshots every 1000 iterations.
- Initial weight matrices are drawn independently from 15 Oct (`init_draws: independent`).
- The paper branches numbers-seeded, figure2-methods and w1-draft are merged (15 Oct).
- (15 Oct, superseded by dead-relu-paper#20) The N = 1000 flip-flop straggler stays excluded; a new
  seed replaces it (Della 15657728_4).
