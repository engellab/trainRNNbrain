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
(c) share of each unit class active 500 and 5,500 iterations after the Fig. 1d weights, (d) ReLU
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
| C1 | Training leaves most units inactive, and a network retrained without them scores within 0.004 R² of the intact network retrained the same way | **Done**: Figure 1 rebuilt (a-d) and panel d re-run at the current floor, rule passed (31 Oct); the fourth 3BFF network at N = 1000 is counted (decision dead-relu-paper#20, option 2). Round 2 (fig1-r2, dead-relu-paper#52, PR #65; entry 2026-11-01_0716): Fig. 1c, Results 1 and the caption with four 3BFF networks; the Extended Data figure of removing inactive units across sizes drawn on 35 networks of 500-4000 units, which are not the size series (fig1-r3, dead-relu-paper#89) | — |
| C2 | Bigger networks keep a smaller share of units active | **Done**: Figure 2 (31 Oct, entry 2026-10-31_1957). Round 2: threshold sensitivity recomputed on the networks Fig. 2a fits (methods-r2, dead-relu-paper#51, PR #64; entries 2026-10-31_2209 and _2211); Figure 2 and the Extended Data figure of floor crossings against snapshots redrawn (fig2-r2, dead-relu-paper#53, PR #67; entry 2026-11-01_0716). 10 Nov (Pavel's request of 9 Nov): N = 250 networks on 3BFF and CDDM and N = 100 networks on Maze trained and read (rules entry 2026-11-09_1925, results 2026-11-10_0806); the primary fits (N ≥ 500) are unchanged | Paper package r22-sizes (dead-relu-paper #226; entry 2026-11-10_0826): new sizes drawn in Fig. 2a,b and Extended Data Figs. 2 and 4b, ten rows in Supplementary Table 1, every-size fits from N = 100 in Methods: Statistics and Supplementary Table 7 with the bend at small N; review and merge pending |
| C3 | Longer training keeps shedding units, and no run has stopped | **Done**; B5 complete, read at 75k (31 Oct); Fig. 2d,e and the optimizer comparison in Fig. 3d and the Extended Data figure of every standard-setting sweep (fig2-r2, fig3-r2) | — |
| C4 | On CDDM 253-310 of 1,000 units are silent (never above zero on the noise-free batch, zero gradient there, 94-155× below the reach threshold under training noise); on 3BFF none is, and low-active 3BFF units get gradient yet stay below threshold | **Done**: re-read on the Fig. 1d weights, rule passed (1 Nov, entries 2026-11-01_0948 and _1021, dead-relu-paper#91; first read 2026-10-31_2207 on weights 281-480 iterations later); Figure 4a-c and the Extended Data figures of gradient reach and of tanh (round 7, r7-figures: the tanh gradient panel, Fig. 4d until then, is panel d of the tanh Extended Data figure) (fig4-no-return, dead-relu-paper#8, PR #60); the tanh paragraph of Results 4 now reports the median unit's gradient instead of "most tanh units" and cites only panel b for its counts (dead-relu-paper#70, fixed in round 3; checked by methods-r4, #113 item 5); the 3BFF reach numbers depend on the trial batch: over twelve batches of 1,024 trials (task seed 12345, five more from the production code, six from independent code) 16-30% of the low-active units are reached and 0.7-6.3% of those are active 5,500 iterations later (entries 2026-11-01_1150 on wp/results-r4 and 2026-11-01_1200 on wp/methods-r4; dead-relu-paper#112); Results 4 (merged, PR #117) and Methods M07 (PR #119) state "fewer than a tenth" of the reached units active later in every batch tested, with no maximum, because the maximum moves with the batch: "at most 6%" fails in 2 of the 12 batches (6.2% and 6.3%), and the PR #117 reviewer's independent batch 1004 gives 6.9%; the Fig. 4 caption says panel c draws one 3BFF batch (sweep-r4, dead-relu-paper#112) | — |
| C5 | The number of active units falls after the floor crossing in 126 of the 127 networks checked (entry 2026-11-06_2219: CDDM 91 of 91, eight of ten settings, sparsity alone no crossing, optimizer and tanh not read at a crossing; 3BFF 36: input-weight norm 15, re-initialisation 10, optimizer comparison 11; the exception is SGD lr 3, 124 -> 198, which did not learn; 125 of 127 by the median over the last 2%); at 75k CDDM settings move it by +116/−177 units, 3BFF input scale raises it 1.7× and the gain shrinks | **Done at 75k** (31 Oct, entry 2026-10-31_2010): Figure 3 rebuilt; both measurement points in the Extended Data figure of every standard-setting sweep; round 2 (entry 2026-11-01_0735): Figure 3 redrawn at print width with panel a cut to 9 rows, the optimizer comparison merged into that Extended Data figure, every network with a floor crossing checked to have fewer active units at the end of training (116 of 116 distinct networks, rate penalty aside; entries 2026-11-01_1000 and 1028 correct the earlier 124, which counted cache entries: 5 default networks shared between sweeps fill 13 entries; the 116 are 91 CDDM and 25 3BFF networks, 19 of them from the revival methods); round 3 (dead-relu-paper#83, #87, #88): filled markers at the floor crossing, open at 75,000 iterations, Fig. 3c median traces, Fig. 3d legend outside the data; merged by fig3-r2 (dead-relu-paper#54, PR #69), with the wording of decision dead-relu-paper#24 (option 1) and the CDDM rate-penalty values in Fig. 3a as a labelled reference (decision dead-relu-paper#35 item 4) | — |
| C6 | At the setting shown, each of five per-unit rules raises the active count at a cost; some settings of rescaling lower it | **Done**: remedies read at the floor and A3 done (31 Oct, entry 2026-10-31_1732); B2 failed its rule; Figure 5, Results 5, the Extended Data figures of the rate penalty alone on 3BFF and of per-unit rules across sizes at three margins, the table of per-unit rules at every size and the table of pre-registered tests built (entry 2026-10-31_1953, dead-relu-paper#9); rescaling and read-out dropout above N = 1000 reported at 75k (decision dead-relu-paper#16, 31 Oct). Round 2 (fig5-r2, dead-relu-paper#55, PR #72; entries 2026-11-01_0745, _0801, _0813, _0833): Figure 5 redrawn with panel a the active count over training at N = 1000; the rate penalty told once, in Results 5, with both tasks (decision dead-relu-paper#35 item 4); every R² checked against loss bursts and non-finite stretches, and the affected values ringed in Fig. 5 and counted in the per-unit-rule table (decision dead-relu-paper#18); the Discussion states the limitation (frame-discussion, PR #73). Round 3 (fig5-r3, dead-relu-paper#90, PR #99; entries 2026-11-01_1007 and its corrections _1046 and _1100): the claim is scoped to the setting shown in Fig. 5; Supplementary Table 3a lists every setting trained to 200,000 iterations (58, one not read) and 3b the 64 cells not continued, among them six rules screened at N = 1,000 for 40,000 iterations (synaptic scaling, synaptogenesis, disinhibition, orthogonal-row and donor-mix replacement, bias offset; eleven per-unit rules tried in all); seven rescaling settings lower the count to 0.67-0.93 of the control at 75k; Methods states how each shown setting was chosen (four of five post hoc, from N = 1,000 results). Round 4 (methods-r4, dead-relu-paper#114): 14 rows of the rule-settings table's block b (the std_pen, std_penlong and std_dropout sweeps) come from networks with a trainable bias (bias_range [-1, 1], started at 0); every network of Fig. 5, block a and the per-unit-rule table, and every network of Fig. 3 and its Extended Data figure, has a fixed zero bias (saved configs, checked on Della and the Mac, 1 Nov); Methods (Per-unit rules) says so | — |
| C7 | Participation needs a drive of its own; inactive units do not become active again | **Done**: B4 supports "do not become active again" at N = 1000 and 4000 (31 Oct); B3 inconclusive (decision dead-relu-paper#17, 31 Oct); the conclusion is "a drive of its own". Results 4 and Fig. 4d,e (transfer test; 4e,f until round 7) merged (fig4-no-return, PR #60); B3 in Results 5, the Extended Data figure of the combination test (Fig. 5f until round 7, r7-figures) and the Extended Data table of pre-registered tests (fig5-r2); Results 5 and the table of pre-registered tests state that the combination test's R² of 0.889 (rate penalty with noisy synapses, at the floor crossing) comes from one of the two networks with a crossing (dead-relu-paper#27, fixed in round 3; checked by methods-r4, dead-relu-paper#113 item 5) | — |
| C8 | Counting the inactive units changes some population statistics computed over all N units | **Done** (1 Nov, entries 2026-10-31_2212 rule, 2026-10-31_2220 results; trainRNNbrain wp/pop-stats): participation-ratio dimensionality of the rates *unchanged* on all three tasks (at most 1.1% apart); normalised-rate dimensionality and the selective and mixed-selective shares *change* on all three; k-means classes *change* on 3BFF and CDDM (k* 4 vs 3), *unchanged* on DMTS. On 3BFF the selective count is unchanged (0-1 inactive units selective) and only the share's denominator moves; on CDDM 119-150 inactive units are selective. Post hoc (entry 2026-11-01_0741, PR #66 review): the S2 gap depends on the normaliser. With 5% of the active units' median sd added to each divisor it is +21-24% on CDDM (13.3-13.9 vs 10.9-11.2) and +24-33% on 3BFF, and on DMTS it reverses sign (-21 to -48% becomes +9 to +13%). S4 on 3BFF comes from a flat-profile class of 146-252 inactive units without selectivity (selective units alone give equal k*); on CDDM 4 vs 3 rests on silhouette margins of 0.01-0.07 (at fixed k = 3, shares within 5.6-7.3 points). The CDDM inactive selective units respond a tenth as strongly as active ones. No N = 4000 CDDM weights within 5% of the crossing exist. In the paper: the Results 1 population paragraph, the Extended Data figure of population statistics and Methods "Population statistics" (pop-stats, dead-relu-paper#56, PR #66); the Introduction and Discussion name S1 as unchanged and the equally weighted statistics as changed, with no 4,000-unit CDDM sentence (frame-discussion, dead-relu-paper#57, PR #73); round 7 (r7-figures, plan risk R1): Fig. 1e draws the selective shares over all units and over the active units in the nine Fig. 1d networks, from the same `data/population_stats.json`. Round 12 (r12-weights, Pavel's item 52): the s.d. of the recurrent weights is 52-83% lower over all units than over the active units in 39 unseen default networks of N = 1,000 (rule fixed before computing, entries 2026-11-02_2218/_2223, result _2232; a first rule on the nine Fig. 1e networks failed on its control design, _2203/_2210). In CDDM the cause is weight decay: weights between inactive units keep 0% of their initial values with weight decay 1e-6 and 98-99% without it (#172). In the paper: Extended Data Fig. 2a (replacing the dimensionality of rates), two Results 1 sentences, the Introduction list, Methods and Supplementary Methods "Spread of the recurrent weights" | — |

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
Data figure of removing inactive units across sizes; entries 2026-11-01_0716 and the fig1-r3 entry
of 1 Nov, dead-relu-paper#89): 35 default networks of 500-4000 units, 3BFF (15: six at N = 500,
three at each larger size) at 75k and CDDM (12) and DMTS (8, none at N = 4000) at the end of
training. **This set is not the size series.** 34 of the 35 are among the 37 networks Fig. 2a fits
at N ≥ 500; the 35th, DMTS s2946835052, learned the task but has no floor crossing (loss ends below
its floor). Not tested: 3BFF s4186074845 (trained in one 200k run, after the ablation), DMTS
s632794669 (N = 2000) and s2627187151 (N = 4000), and every N = 100 network. A 36th ablation file,
DMTS s597040509 at N = 2000 (R² 0.417), never learned the task and is left out. Removing the inactive units changes R² by −0.010 to +0.001 on CDDM and
−0.046 to −0.001 on DMTS, and in all 35 networks it costs less than removing as many units at random.

**C1 on the maze (5 Nov; trainRNNbrain issue #3, dead-relu-paper#182).** The maze replaces DMTS
in Figure 1c-e. Of the three N = 1,000 ReLU agents of the uniform series (2,000 PPO updates),
seeds 1 and 2 learned the task (probe success at or above 0.9 for 5 probes and at the end) and
seed 3 ended at 0.73 and enters no participation statistic. Active units at the learned point: 451
and 517 (panel c, success 0.945 and 0.914); at the end, 421 and 451. Ablation at the end weights
(entry 2026-11-05 maze-ablations-uniform-series; data/rl_maze_ablation): removing every inactive
unit changes the success under the training noise by -0.01 to +0.03 in all six networks ablated
(N = 250 to 2,000), as many units at random lower it to 0.07-0.31, the active units removed to
0.01-0.03. Prune-and-retrain (rl_maze_retrain.py; pre-registered rule in entry 2026-11-05_0832:
within 0.05 of the intact agent retrained the same way, 200 further updates) passes (entry
2026-11-05 maze-retrain-sigmaw-rescale-n2000-results): the agents retrained without their inactive
units end at 0.925 against 0.913 and 0.903 for the intact agents retrained the same way; as many
random units removed recover from 0.23-0.37 to 0.81-0.83. Panel d draws the strip (open markers). Selectivity (entry
2026-11-05_0845; data/population_stats_maze.json): 154 and 200 of 1,000 units selective over all
units, 154 of 421 and 199 of 451 over the active units, 0 and 1 inactive units counted selective
(as on 3BFF); the raw-rate dimensionality is the same over all and over active units (10.7 against
10.6, 10.3 against 10.2) and the standardised-rate one is more than twice as large over all units
(46 against 20, 43 against 20), as on the supervised tasks (C8).

**C2.** Figure 2 (entry 2026-10-31_1957; Della job 16558986, read-out 79932ef). Read at 1.05× each
run's floor and fitted on the size means from N = 500, the active count grows as N^0.48 on the 3-bit
flip-flop (41% of 500 units active, 14% of 4000), N^0.44 on CDDM (298 of 500, 767 of 4000) and N^0.55 on
DMTS (four sizes now; 191 of 500, 620 of 4000 from the one N = 4000 network that reads out). At
N = 100, 66-90% of units are active (drawn, not fitted). At 1.05×, 1.10× and 1.20× the exponents lie
between 0.44 and 0.61 and every bootstrap 95% upper bound is below 0.79, so the rule of entry
2026-10-31_1900 says C2 holds on all three tasks. The 0.79 (DMTS at 1.20×) rests on three networks:
leaving out any one of them gives 0.86-0.87, still below 1, so C2 holds with any one network left out (at sizes with two or more)
(entry 2026-11-01_0957). 1,000 active units would need N ≈ 7,300 on CDDM,
9,200 on DMTS and 14,200 on the flip-flop. The learning-rate confound (lr ∝ N^−1/3) is left in and
handled by reading each network at its own floor; mean crossings move from 59.0k to 65.0k (flip-flop) and
26.7k to 32.5k (CDDM) between N = 500 and 4000, "by about a fifth or less" in Results 2. The fourth flip-flop network at N = 1000
(s3374117298) is counted (decision dead-relu-paper#20, option 2), as in C1: 277.5 ± 26.6, n = 4.

**C2, new sizes (10 Nov; rules entry 2026-11-09_1925, results entry 2026-11-10_0806; Spock 7412712, 7412726,
7412727; 81.5 GPU-h).** Trained at the code of the N = 100 cells (63c6c97) and read with `read_one`
(`size250_n100maze_read.py`). At N = 250, each network's 1.05× crossing: 3BFF 143, 133 and 125 of 250 units active
(mean 134), CDDM 187, 174 and 188 (mean 183), held-out R² 0.950 and 0.826-0.828 as at the other sizes; every
network has a crossing. Maze at N = 100 (current code, `freeze_after_end` true; bitwise equal to false for these
networks over 3 updates): all four learned by update 1,780; 61, 71, 64 and 75 of 100 active at the learned point
(mean 68), 56-65 after update 1. The primary exponents are unchanged (3BFF 0.478, CDDM 0.443, Maze 0.343 / 0.495 /
0.441). Every-size fits, old → new: 3BFF 0.514 → 0.516 at 1.05× (inside the old interval at all three margins);
CDDM 0.578 → 0.571, 0.623 → 0.610, 0.702 → 0.693 at 1.05×, 1.10×, 1.20× (the last two outside the old interval);
Maze from N = 250 → from N = 100: 0.471 → 0.584 (0.551-0.618) at the learned point, 0.554 → 0.619 at update
4,000, 0.518 → 0.595 at the end (all outside the old interval). The every-size intervals cover only the spread
between networks, not the departure of the size means from one straight line on log axes. Paper package r22-sizes
(entry 2026-11-10_0826, dead-relu-paper #226): Results 2 reads "100 to 4,000 units on each of the three tasks";
Methods: Statistics gives the every-size fits from N = 100 and the bend (CDDM slope 0.78 between N = 100 and 250, 0.44
over N ≥ 500; Maze N = 100 mean 68 against 117 from the N ≥ 250 law); Methods: Participation gives the Maze
first-probe counts (56-68% of N active after update 1 at every size).

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

**C4.** Re-read on the weights Fig. 1d uses (entry 2026-11-01_1021, rule in 2026-11-01_0948, Mac,
dead-relu-paper#91: the snapshot nearest each crossing, 20-219 iterations before it; the first read,
entry 2026-10-31_2207, Della job 16566988, used the snapshot 281-480 iterations after it). On CDDM
253-310 of a network's 1,000 units are silent:
their net input current is at or below zero on every step of every trial of the noise-free
reference batch, so there their rate and the loss gradient on their incoming weights are exactly
zero (numpy float64 and autograd). Under the training noise they are above zero on about 1.6% of
steps and their gradient is 94-155 times (median per network) below the reach threshold; 6 of 851
cross it in some batch (entry 2026-11-01_0716 withdrew "only through weight decay"). At
most 1 silent unit per network is active 5,500 iterations later. In the three 3BFF networks of Fig. 1d no unit is silent (silent units were counted only there); the gradient
reaches 136-161 of the 658-731 low-active units (5% of the network's q95 of g_in), whose current is
positive on a median 2-4% of steps, almost always within 2τ of an input pulse. 5,500 iterations
later 0.7-4.3% of them are active (0-1.3% after 500). These 3BFF numbers depend on the trial batch
(dead-relu-paper#112): over twelve batches of 1,024 trials 108-222 low-active units are reached
(16-30%) and 0.7-6.3% of them are active 5,500 iterations later (entries 2026-11-01_1150 and its
corrections _1204 and _1215 on wp/results-r4, 2026-11-01_1200 on wp/methods-r4); the review of PR #117
found 6.9% in a 17th batch. The paper says "a sixth to three-tenths" and, with no batch maximum, the
pre-registered "fewer than a tenth", which holds in all 17 batches. On CDDM 23-39% of the 40-60 reached
low-active units are active by then (turnover near the criterion while the count falls). The reach
does not shrink with training (C4b, entry 2026-10-14_1218: 199/155/175 reached at 50k, 174/152/156
at the end, no fewer after lr/10). Tanh (CDDM, N = 1000, 9 networks, end of 150k, not a floor read):
0 inactive units; median per-unit state gradient per unit of error 27× lower than untrained (6-247×);
output sensitive to the incoming weights of 25-147 units against 277-293 for ReLU. Figure 4a-d,
the Extended Data figures of gradient reach and of tanh. The paper words the silent criterion as
"negative on every step after the initial state (x = 0) of every trial" (Methods, Results 4 and
the Introduction; decision dead-relu-paper#59), because x = 0 at step 0 of every trial.

**C5.** Scope of the falling count since round 17: the first column of the row above; the older wording "no standard setting stops" below is superseded. Read at 75k, the same-size rule (entry 2026-10-31_2010; Figure 3, and the Extended Data figure of every
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

**C6.** Scope (round 3, dead-relu-paper#90): the claim holds at the setting of each rule shown in
Figure 5. Of eleven per-unit rules tried on 3BFF, five were trained at four sizes to 200,000
iterations; six were screened at N = 1,000 for 40,000 iterations and stopped (synaptic scaling, 1.26-1.69×
where its networks did not diverge, 7 of 12 diverged; synaptogenesis 0.80-1.88×; disinhibition 0.98×;
orthogonal-row 1.07× and donor-mix 1.21× replacement; bias offset 1.36×). Random redraw and zeroing act on
units too and are reported as revival methods (Fig. 3e). Seven rescaling settings lower the count, to
0.67-0.93 of the control at 75k (Supplementary Table 3a; `table_rule_settings.py`, `table_rule_screen.py`,
entries 2026-11-01_1046 and _1100). Four of the five shown settings were chosen after N = 1,000 results (Methods,
Per-unit rules). The remedy top-ups (RMtop200, 267 runs in 83 cells) are read at 1.05× each run's own
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

**C2 on the maze (5 Nov).** Learned runs of the uniform series (2,000 updates; N = 2,000 on a
2,000 + 2,000 chain with a fresh Adam at the join): active units at the end 147, 158 (N = 250),
265, 241, 276 (500), 426, 449 (1,000), 400, 506 (2,000; learned at updates 3,000 and 2,480); fit
of the size means N^0.55 (bootstrap 95% 0.48-0.61). N = 2,000 seed 3 learned at 2,840 (414 -> 388). N = 4,000 (entry 2026-11-06 maze-package3-readout-and-n4000):
seed 1 (a 1,000 + 3,000 chain after a stop at update 1,050) learned at 2,860, 619 -> 676 active,
success 0.96; seed 2 ended its 4,000 updates at 0.84 (516 active), not learned; seed 3 (continued
from its update-1,700 checkpoint) runs to 4,000; seeds 4 and 5 submitted (Della 17012367). Fit over
N = 250 to 4,000, learned runs: N^0.52 (bootstrap 0.49-0.55 with the one seed both scripts use since round 17, entry 2026-11-06_2135; 0.48-0.55 with fig_ed_maze's old seed); size means 146 (n = 3), 261 (3), 438
(2), 431 (3), 676 (1). N = 250 seed 2 and N = 1,000 seed 3 are not learned and enter no statistic.

**When the rules were written (entry 2026-11-06_2259).** "Only stronger weight decay on all three tasks" holds for networks that learned, a restriction chosen after the data: 18 values, 2 the same; counting every finished network, among values all of whose networks finished, 25 values, 4 the same (SGD 0.03 and 0.1 too). **Corrected (entry 2026-11-09_1509):** the 4 used the two learned Maze controls as the range; with all three controls, as the every-network rule counts them, it is 3. Since G12 the tallies are 17 and 1 (learned networks) and 24 and 3 (every finished network). Prune-and-duplicate alone passes on all three tasks at R^2 tolerances from 0.0046 to about 0.0213, SE of the upper limit 0.0015 (entry 2026-11-06_2317); the abstract says no "only". The Maze technique conditions predate every Maze technique network; the R^2 tolerance postdates every 3BFF and most CDDM networks; the at-one-value rule and the 75,000-iteration 3BFF read postdate the data.

**C6 across the three tasks (5 Nov; Pavel's instruction, trainRNNbrain issue #6, dead-relu-paper#183:
report the rules that work on every task first).** Criterion per cell at N = 1,000: more active
units than every control in all seeds at a performance within the tolerance (held-out R^2 within
0.01 of the controls' mean at 75k; maze success within 0.05 at update 2,000). CDDM, read on 5 Nov
(entry 2026-11-05_0926; read_uniform_cells.py): read-out dropout works (517 against 336 active,
1.54x, R^2 0.826 against 0.834), prune-and-duplicate works (709, 2.11x, R^2 0.830), noisy synapses
sigma_w 1.0 fail the tolerance (495, 1.47x, R^2 0.819, 0.015 below), as on 3BFF (2.03x at 0.021-0.025
below). Maze (issue #5's criterion as written, dead-relu-paper#194, entry 2026-11-06_2238, correcting entry 2026-11-05_0812: all three controls, 430 active, success 0.898; a seed below 0.9 fails the value; the two learned controls, 437.5 and 0.984, give the same verdict for every technique): the rate penalty
at 0.03 and 0.3 works (2.0x and 2.1x, success 0.086 and 0.089 above), at 0.1 and with
the sparsity penalty one seed ends below 0.9 (0.63, 0.43; means 0.034 and 0.107 below); read-out dropout fails (1.0x, success 0.104 below); noisy synapses fail marginally (1.07x, mean 0.023 above, seed 3 at 115 of 128 episodes, 0.898); prune-and-duplicate works (1.7x, 0.078 above);
rescaling fails at every setting: the paper's values blew up at the end of the warm-up ramp, and
the two gentler arms (alpha 1.0005 with cap 8; protect 0.05) end with FEWER active units than the
controls (231-321 against 426-449) at success 0.63-0.81 (one run collapsed to 0 active units and
stopped at update 800); prune-and-duplicate's active counts 712-747 are from entry 2026-11-06
maze-arms-readout, and it passes on all three tasks.
3BFF input-weight cap 0.1, 0.5, 2.0 fails (R^2 0.019-0.105 below).
Della batch 1 (6 Nov, entry 2026-11-06_0900, read_uniform_cells_2.py): CDDM rescaling at the
paper's values FAILS (196, 183, 169 active against 333-342, R^2 0.819 against 0.834), so rescaling
works on 3BFF only; the CDDM rate penalty 0.1 works (1,000 of 1,000 active, R^2 0.827); CDDM's
optimiser comparison does not reproduce 3BFF's pattern (Adam wd 0 and AdamW keep more units at the
same R^2 on both tasks, but SGD at 0.1 and 0.3 keeps 845-895 units on CDDM within the 0.01
tolerance, where on 3BFF every SGD arm is 0.05-0.08 below); 3BFF weight decay 1e-5 and 1e-4 lower
the count (203-229 and 144-163 against 233-270), the direction of CDDM's ladder. Across the three
tasks on 6 Nov, judged at the SAME-SIZE read-out (75,000 iterations; maze update 2,000) with the
fixed tolerances (decision dead-relu-paper#185; fig_paper_F5_tasks.py, entry 2026-11-06_0934):
prune-and-duplicate works on all three (3.07x, 2.11x, 1.68x at 0.005, 0.003 and 0.008 below the
controls); the rate penalty on CDDM (0.1, 0.2) and the maze (0.03, 0.3), not on 3BFF (0.021
below); rate + sparsity on CDDM only; read-out dropout on CDDM only (on 3BFF 2.05x at 0.017 below:
its earlier "works" was a floor read of 1 of 3 runs); rescaling on none (3BFF 1.65x at 0.013
below, its earlier "works" a floor read of 1 of 4; CDDM 0.54x; the maze stopped or lowered);
noisy synapses raise the count on all three (2.19x, 1.47x, 1.05x) and miss the tolerance on all
three. The floor reads stay in the Extended Data table. Maze package 3 (entry 2026-11-06
maze-package3-readout-and-n4000): the cap 0.3, the recurrent-weight penalty, the metabolic penalty
at 0.01 and 0.1 and AdamW change nothing; the metabolic penalty at 1 and 10 lowers the count; SGD
at 0.03 to 1 never learns the maze (success 0.00-0.02), so the maze cannot test 3BFF's SGD pattern;
redraw and zero-out fail the rule (1.06x and 0.99x). Della batch 2 (entry 2026-11-06_1830;
every supervised cell of the programme complete, 86 of 86): CDDM rate 0.1 + sparsity 0.05 works
(1,000 of 1,000 active, R^2 0.826); 3BFF at 75k: lambda_rwm 0.05 and 0.2 raise the count (516-567
against 233-270, R^2 0.006 below) as on CDDM; the metabolic penalty 0.01-1 raises it on 3BFF, 10
lowers it (188-206, R^2 0.048 below); the cap 0.3 raises it at R^2 0.046 below (CDDM 0.004 below);
activity noise, spectral radius, sparsity alone and the activations change nothing on 3BFF by the
per-seed rule; tanh has every unit above the participation bar with 41-536 saturated. Of 19
setting values with cells on both supervised tasks, 8 move the count the same way (weight decay
1e-5 and 1e-4, spectral radius 2.0, lambda_rwm 0.05 and 0.2, metabolic 0.1 and 1, cap 0.3) and
11 do not; weight decay 1e-5 and 1e-4 lower the count on all three tasks. **Cross-task figures (entry 2026-11-06_0934;
`fig_paper_F5_tasks.py`, `fig_paper_F3_tasks.py`): at the same-size read-out point, 75,000 iterations,
and by the one rule, read-out dropout (R² 0.017 below the control's mean) and rescaling (0.013 below)
FAIL on 3BFF; the "works" of the previous sentence is the floor read of 1 of 3 and 1 of 4 runs. By the
drawn read-out prune-and-duplicate works on all three tasks (3.07x, 2.11x, 1.68x at 75k / update
2,000), the rate penalty on two (CDDM at λ 0.1 and 0.2; the maze at 0.03 and 0.3 and not at 0.1; 3BFF at
0.1 fails, R² 0.021 below), rate + sparsity on CDDM only (0.2 + 0.05; 3BFF 0.1 + 0.05 is 0.028 below),
read-out dropout on CDDM only; noisy synapses and rescaling on none.** **In the paper (6 Nov evening, dead-relu-paper PR #188, wp/cross-task 45be63d; code main 3f0bb49):**
Figure 3 is `fig_paper_F3_tasks` (tasks x settings) and Figure 5 is `fig_paper_F5_tasks` (tasks x
techniques); Results 3 and 5, the captions, Methods and the front matter carry these verdicts and the
settings table (28 values finished on both supervised tasks, 11 in the same direction; 25 on all
three, 4 in the same direction: weight decay 1e-5 and 1e-4, SGD 0.03 and 0.1); the previous Fig. 5's
a, d, e are Extended Data `fig_ed_techniques_3bff`; the maze settings and interventions figures are
Extended Data figures.
Removing the inactive units and then the recruited units (570 units in all; 3BFF's A3 removes only the recruited units, entry 2026-11-07 fig5d-two-removals) of the maze rate-penalty networks lowers success from 0.84-0.96 to
0.55-0.77, as many at random from all units to 0.07-0.35 (k from the learned controls' 437.5; from all three controls' 430 since round 17, interpolated on the same grids: 0.53-0.76 and, as the median, 0.08-0.33; entry 2026-11-06_2238) (entry 2026-11-05 maze-rate-penalty-ablations): the maze
form of A3. Maze settings (entry 2026-11-06 maze-arms-readout; read_maze_arms.py,
fig_ed_maze_settings.py; a setting changes the count when all three seeds lie outside the learned
controls' 426-449 on one side): input row norm 0.5 increases it (549-593, 1.31x, every seed learned,
success 0.016 above); weight decay 1e-5 (370-417) and 1e-4 (335-378, one seed learned), activity
noise 0.1 (311-346, no seed learned), spectral radius 0.5 (338-414) and the sparsity penalty alone
(378-399, one seed learned) decrease it; weight decay 0, input norm 2, activity noise 0 and 0.01,
spectral radius 1.5 and the input-weight cap 0.3 leave it inside the range; input norms 5 and 20
diverge (five of six runs stopped by the rollback guard at 0 active units, the sixth at success
0.00). Still training on Spock (7389398): the recurrent-weight and metabolic penalties, AdamW, SGD,
redraw and zero-out. Queued or pending elsewhere: CDDM rescaling and the 3BFF settings ladders
(Della, 16957556-16957562).

**C6, round 18 (7 Nov; rules 2026-11-07_0100_round18-rules, results entry 2026-11-07 round18-results;
dead-relu-paper#201; data/round18/g1g2.json).** G1: the 3BFF rate penalty at 75k with 36-draw R²
against the four default networks (236-288 active, R² 0.9513): λ 0.01 has 611-869 active (2.75x) at
R² 0.0161 below, λ 0.03 606-844 (2.77x) at 0.0230 below, λ 0.1 935-995 (3.72x) at 0.0211 below; none
passes at 0.01, λ 0.01 passes at 0.02. G2: the per-network criterion on all 57 N = 1,000 values of
Supplementary Table 3a. Two ways of reporting, side by side (coordinator, 7 Nov):
(a) the paper's one-value rule: at 0.01, 3BFF passes for prune-and-duplicate (9 of 9 values),
noisy synapses (σ_w 0.1 and 0.3, R² 0.0008 and 0.0039 below), read-out dropout (rate 0.05 with β 1, 2,
4 and rate 0.10 with β 1; 0.006-0.010 below) and rescaling (1 of 26 values, revive_ops2 target 4.0
protect 0.20, 0.0097 below: 0.0003 inside the tolerance, 0.7 standard errors of the network-to-network
difference, so it does not survive that error); techniques on all three tasks: prune-and-duplicate at
0.005 and 0.01, prune-and-duplicate and the rate penalty (via λ 0.01) at 0.02. (b) matched values,
the one value run on all three tasks (prune-and-duplicate copy noise 1 rate 0.025; rate penalty 0.1;
rate 0.1 + sparsity 0.05; noisy synapses σ_w 1.0; read-out dropout rate 0.2 β 4; rescaling target
10x): prune-and-duplicate alone at all three tolerances. The asymmetry: 3BFF has 13 to 26 values per
technique (rate penalty 3), CDDM and Maze one each except the rate penalty (CDDM 0.1, 0.2; Maze 0.03,
0.1, 0.3). **Provisional** until the training package runs noisy synapses σ_w 0.1 and 0.3, read-out
dropout rate 0.05 and rescaling target 4.0 on CDDM and Maze: the (a) verdicts for those techniques and
the "in CDDM networks only ... in none" sentence of Results 5. **Also provisional (corrected 7 Nov,
entry 2026-11-07_0331_correction-r18-headline-provisional; dead-relu-paper#205):** the headline
"prune-and-duplicate is the one technique on all three tasks" at 0.01 and the one-value range 0.0032-0.0161.
Both hold on present data under (a) and (b), but a parity value that meets the criterion on CDDM and Maze
(for read-out dropout, on Maze alone) would make its technique pass on all three tasks; under (b) the parity
values become matched values once run, and the matched-values rule was chosen after the (a) verdicts were
computed (PR #202 review). The paper marks both as provisional (PR #202); recompute both when the parity
runs are read. M09's
"only technique at all three tolerances" is false under (a) (the rate penalty at 0.02). M09's range
"from 0.0046 to about 0.0213" changes under both: under (a) prune-and-duplicate alone passes on all three
tasks from 0.0032 (its CDDM difference; on 3BFF its best value is 0.0002 above the controls) to 0.0161
(the rate penalty at λ 0.01); under (b) from 0.0046 with no upper limit among the networks that exist,
because every other technique's matched Maze value fails the 0.9 condition or did not finish.

**C6, round 18, removals and sizes (entry 2026-11-07_0235_round18-results).** G3, CDDM removal of the k added
units (A3 procedure): read-out dropout (R² drops 0.19-0.29) and the rate penalty 0.1 and 0.2 (0.27-0.35)
depend on them; prune-and-duplicate is mixed (drops 0.07, 0.16, 0.07; 0.83 to 0.67-0.76); k random active
units lower R² further in every network. G4, Maze, 3BFF removal on 12 networks: prune-and-duplicate (drops
0.27-0.39) and rate 0.3 depend, rate 0.03 and 0.1 mixed (one network each below the 0.10 bar; rate 0.1 s1 did
not learn). The 570-unit values of Fig. 5d and Results 5 were linear interpolations; exact values are 0.36-0.78
(paper 0.53-0.76) and 0.06-0.37 random (paper 0.08-0.33). G5: the Maze form holds on 3BFF for every technique
(random lower in 19 of 19). G6: CDDM prune-and-duplicate raises the count at N = 500-4,000 in every network,
1.4-2.1x at the floor crossing (R² within 0.004), 1.7-2.7x at 75k; read-out dropout falls to 1.02x at the floor
at N = 4,000 (count condition fails) and 1.14x at 75k. The Discussion's "techniques across sizes in 3BFF only"
changes.

**Round 18, other claims (same entry).** C2: the Maze exponent at N >= 500, the supervised fit range, is 0.35
at the learned point (bootstrap 0.31-0.38) and 0.41 at the end (0.38-0.44), against 0.47 and 0.52 over every
size; intervals narrow because N = 4,000 has one learned network; budgets 2,000 updates at N <= 1,000 and 4,000
above. C3: the flat-loss test on CDDM, 9 of 9 flat networks lose 34-64 active units per 100k; on Maze no
learned network has a flat training return (still rising by 10-55% of its rise per 1,000 updates), so Maze has
no flat window. C5: 40 of 48 learned Maze networks of the control, setting and re-initialisation arms have fewer
active units at update 2,000 than at the learned point; pooled 166 of 175. C4: 208-283 of the 1,000 units of
the N = 1,000 Maze networks are silent on the frozen rollout (with CDDM). C7: from the logs, 3.6-9.7% of the
inactive units at the learned point are active 700 updates later in 8 Maze networks and 17% and 19% in the two
N = 1,000 networks: mixed, so C7 does not gain Maze. C8: the recurrent-weight s.d. is 24-28% lower over all
units than over the active units in the three N = 1,000 Maze networks.

**Round 18 training package, interim (entry 2026-11-07_0707_round18-training-interim-results; rules
2026-11-07_0110 and _0136).** Settings (Fig. 3): G12's three new N = 1,000 Maze controls all learned, with 394,
398 and 417 active units at update 2,000, so weight decay 1e-5 on Maze (370-417) no longer lies below every
learned control. Its Maze verdict becomes "no change", and "only weight decay changes the number on all three
tasks" loses its Maze direction. The other Maze setting verdicts are to be recomputed against the five learned
controls (394-449). C7: G18, the CDDM transfer test (CDDMHyper, B4's rules), passes the gate. At the floor
crossing the verdict is "between the bars": the old networks revive 12%, 12% and 25% of their inactive units,
read at 2,000-4,890 iterations against 24,906-28,578 for the fresh networks. The 1.10x and 1.20x margins give
the same verdict. At 75k the old networks keep 4.8-6.0% of those units active and end 101 units below the fresh
networks ("do not come back", not binding). G17 (Maze transfer) not run: no second maze passed the difficulty
check (entry 2026-11-07_0421; decision on #201). C2: G11 brings N = 250-1,000 to 4,000 updates; all ten runs
learned, and every run has fewer or equal active units at 4,000 than at 2,000. C6 parity on CDDM, partial:
rescaling target 4.0 fails (224-231 active against 333-342); σ_w 0.1 fails (316 and 333 in two of three
networks); σ_w 0.3 passes in two of three networks so far; read-out dropout 0.05 and every Maze parity arm
are still running.

**Round 18 training package, final (entry 2026-11-08_0859_round18-training-final-results; settles the
provisional notes of the C6 round-18 paragraph).**

*C6 parity (the 3BFF values that pass, run on CDDM and Maze at N = 1,000).*
- CDDM: σ_w 0.3 passes, with 356-380 active units against a control maximum of 342. Its mean R² is 0.0008
  above the controls (SE 0.0012; 8.7 SE inside the 0.01 tolerance), and it also passes against the second
  control. σ_w 0.1 (316-333), read-out dropout at rate 0.05 with β 4 (326-350) and rescaling at target 4.0
  (224-231) fail on the count.
- Maze: all four values fail on the count: σ_w 0.3 (396-432 against 449), σ_w 0.1 (342-418), dropout 0.05
  (373-455), and rescaling target 4 (one network stopped, one at success 0.01).

*Technique verdicts at tolerance 0.01.*
- (a) Under the one-value rule, prune-and-duplicate is the only technique on all three tasks.
- Noisy synapses now pass on 3BFF (σ_w 0.1, 0.3) and CDDM (σ_w 0.3), and fail on Maze.
- Read-out dropout passes on 3BFF and CDDM; rescaling on 3BFF only; the rate penalty on CDDM and Maze (and
  on 3BFF at 0.02, through λ 0.01).
- (b) Under matched values, prune-and-duplicate is the only value passing on all three tasks. σ_w 0.3 passes
  on 3BFF and CDDM.
- **Headline unchanged: noisy synapses do not pass on all three tasks at σ_w 0.3.** "Prune-and-duplicate is
  the one technique that meets the criterion on all three tasks" holds at 0.005 and 0.01 under both rules.
- The Results 5 sentence "noisy synapses and rescaling in none" is false for noisy synapses under (a):
  CDDM at σ_w 0.3 passes.

*Settings (Fig. 3).* Against the five learned Maze controls (394-449 at update 2,000), weight decay 1e-5,
spectral radius 0.5, the sparsity penalty 0.05 and the metabolic penalty 0.01 change from "decreases" to
"no change" on Maze. Weight decay 1e-4 and the metabolic penalty at 1 still decrease the number of active
units, and input norm 0.5 still increases it.

*C2.* The Maze exponent at N ≥ 500 on equal 4,000-update budgets is 0.35 (0.32-0.38) at the learned point
and 0.49 (0.46-0.52) at the end. It replaces the mixed-budget 0.35 and 0.41; N = 4,000 has one learned
network.

*C7.*
- G17, Maze transfer on the first maze mirrored left to right: the gate fails (fresh 437 ± 47 against old
  420 at transfer), so there is no B4 verdict. Old networks have 13.7-18.2% of their inactive units active
  again at the learned point, between B4's 10% and 30% bars in all five.
- G18, CDDM transfer: gate passes; between the bars at the floor; "do not come back" at 75k (not binding).

**Round 19, Maze networks trained until they learn (entry 2026-11-08_0915_round19-results; rules 2026-11-07_1008,
_1111, _1723).**
- *Protocol.* Every Maze network trains in 2,000-update stages until learned, cap 8,000. Every N ≤ 2,000
  network is learned at 4,000 updates, softplus seed 3 included.
- *Activations at N = 1,000, 4,000 updates.* ReLU 314-404 active, softplus 345-368, leaky ReLU 548-620.
- *N = 4,000, final (entry 2026-11-09_1341).* Budget 8,000 updates. Seeds 1, 2, 3 and 5 are learned (learned
  points 2,860, 3,900, 4,120, 3,970; 557, 595, 702 and 533 active at 8,000). Seed 4 stopped at 5,940 with
  non-finite weights and is left out.
- *C2, Maze exponent refit (same entry; replaces round 18's 0.35 and 0.49 above).* At the learned point,
  N ≥ 500, with four N = 4,000 networks: **0.34 (0.29-0.40)**. On equal 4,000-update budgets (N = 4,000: seed 1
  alone, the one network learned by then): 0.50 (0.47-0.52). At each size's final budget (unequal: 8,000 at
  N = 4,000, 4,000 below): 0.44 (0.39-0.50), labelled unequal.
- *Settings (Fig. 3), Maze at 4,000 beside the stated 2,000 read* (controls at 4,000: 314-404):
  - weight decay 1e-4 decreases at both points (3 of 3 learned at 4,000, 301-311);
  - activity noise 0.1 goes from "no direction" to "no change";
  - SGD does not learn within 4,000 updates at any rate.
- *Techniques on Maze at 4,000.* σ_w 1.0 works (409-455 active against a bar of 404); it fails at 2,000. The
  bar falls from 449 to 404 because the controls lose units between 2,000 and 4,000.
- **Withdrawn (entries 2026-11-08_0940 and _0943): every Maze verdict of the rate penalty (frm0.03, 0.1, 0.3,
  rate + sparsity) and the metabolic penalty (0.01-10), at 2,000 and 4,000.** Until 8 Nov the Maze trainer kept
  stepping the RNN after an episode ended and averaged both penalties over those steps too (20-56% of RNN steps
  late in training). That froze met0.01 seed 1 (every optimiser step skipped from update 1,501: "frozen by a
  non-finite loss"). The 24 networks are rerun with `trainer.freeze_after_end=true` (Della 17107385); "the rate
  penalty works on Maze at 0.03 and 0.3" and the cross-task technique count wait for them. Active-unit
  read-outs never include post-episode steps.
- *Reruns at 2,000 with the fix (entries 2026-11-08_2346 and 2026-11-09 duplicate-rerun-headline):* the rate
  penalty works on Maze at 0.03 and 0.1 and fails at 0.3 (one seed at 0.87); rate + sparsity fails (success
  0.34-0.68). Metabolic: 0.01 no change, 0.1 decreases on one learned seed, 1 and 10 no direction.
  **Prune-and-duplicate, rerun with the fix, still works on Maze (749-763 active, success 0.98-1.00): the
  headline stands.**
- *Reruns at 4,000, beside the stated 2,000 read (entry 2026-11-09_1221; controls 314-404 active, success
  1.00):* the rate penalty 0.3 works at 4,000 (all three seeds learned, 1,000 of 1,000 active, success 1.00),
  so on Maze the rate penalty works at 0.03 and 0.1 at 2,000 and at 0.3 only at 4,000. Rate + sparsity fails
  at both points (success 0.84, 0.99, 0.66 at 4,000); before the fix it worked at 4,000. The cross-task count
  keeps the Maze rate penalty at 0.03 and 0.1 (update 2,000). Metabolic at 4,000: 0.01 and 0.1 no change, 10 no
  direction (no seed learned), 1 incomplete (seed 2 stopped at 1,900, seed 3 at 3,902, both within running
  episodes). No metabolic value has a Maze direction at both points.
- *Read-out dropout and rescaling rerun with the fix, at 2,000 (entry 2026-11-10_2014; rules 2026-11-09_1513;
  Della 17207738, 17207739):* read-out dropout fails on Maze at rate 0.05 (444, 416, 498 active against the bar of
  449; success 1.00, 0.98, 0.97) and at rate 0.2 (428, 456, 436; seed 2 ends at 0.84). Rescaling fails or is
  incomplete at all four values (233-375 active, success 0.19-0.81; three networks stopped within running
  episodes). **The headline stands under both rules, the tolerance ranges are unchanged, and the post-episode
  caveat (entry 2026-11-08_0953) is removed for read-out dropout and rescaling.** It remains for
  re-initialisation, which was not rerun. *At 4,000* (entry 2026-11-11_0306; Della 17289218; the criterion
  used for noisy synapses at 4,000, bar 404): read-out dropout 0.2 is incomplete (seed 3 stopped at 2,100,
  within running episodes), and its two complete seeds are below the bar (397, 388; success 0.98, 0.96).
  Rescaling has no verdict (rescale), is incomplete (rescale_mild) or fails (rescale_few, rescale_tgt4; 190-333
  active, success 0.41-0.84). **The paper's 4,000 statement (noisy synapses at σ_w 1.0 meet the criterion on Maze,
  so on all three tasks under the one-value rule) stands; read-out dropout does not join them.**
- *tanh* learns under no tested setting within 4,000 updates. *The shifted sigmoid* learns at initial spectral
  radius 0.45, a per-activation change decided after the pilot was read. Full runs (entry 2026-11-09_0213):
  three sigmoid networks learned at 1,410-3,330 with 227-250 active at 4,000; the same-radius ReLU control has
  two networks (344 and 273; the third stopped). Numbers only, no verdict.

**Round 19 text package (r19-text, dead-relu-paper#206; entry 2026-11-09_1442_r19-text-package).** The paper
carries rounds 18 and 19. Recomputed for it on trainRNNbrain wp/r19-text:
- *C5 settings tallies (Fig. 3; fig_paper_F3_tasks.py with the reader maze_arms_r19.py).* With the five
  learned Maze controls (394-449) and the metabolic reruns: 17 values have a direction on all three tasks and
  1 agrees (weight decay 1e-4); counting every finished network, 24 and 3 (wd 1e-4, SGD 0.03, 0.1). The paper's
  "only stronger weight decay" becomes "only the strongest weight decay tested" (1e-4).
- *C5 Maze check after the learned point.* G9 without the six withdrawn metabolic networks and with the three
  fz metabolic networks learned by update 2,000, plus default seeds 4-6 and the five leaky ReLU and softplus
  networks learned by then: 44 of 53 (pooled 170 of 180; entry 2026-11-09_1509). Prune-and-duplicate rerun: 64 to
  109 more active units at 2,000 than at the learned point.
- *C6 matched-values range.* With the Maze rate penalty at 0.1 working after the fix, prune-and-duplicate is the
  only technique on all three tasks under matched values from 0.0046 to 0.021 (the rate penalty at 0.1 joins at
  its 3BFF difference, 0.0211 at 36 draws); round 18's "no upper limit" is false. One-value range unchanged
  (0.0032-0.0161).
- *C6 Maze removal of the added units, redone on the 12 fz networks at update 2,000*
  (data/round19/fz_maze_added_summary.json): every arm depends (drops 0.33-0.49 prune-and-duplicate, 0.13-0.55
  rate penalty); k random active units lower success further in 11 of 12 (not prune-and-duplicate s1, 0.47
  against 0.45). Pooled with 3BFF and CDDM: 43 of 45 networks drop by at least 0.10.
- *C2 size series.* 24 of 27 followed networks have fewer active units at the end than at the floor crossing
  or the learned point: 7 of 7 supervised, 17 of 20 Maze (the end at 4,000 for N <= 2,000, 8,000 at N = 4,000;
  decreases of 6-113 units, 4-26%; increases: N = 2,000 s1 393 -> 400, N = 4,000 s2 532 -> 595, s3 681 -> 702).
  10 Nov (entry 2026-11-10_0826): with the four N = 100 Maze networks, 27 of 31 (20 of 24 Maze; decreases 1-113
  units, 2-26%; N = 100 s2 unchanged at 71).
- *Figures redrawn:* Fig. 1 (Maze c: seeds 1-3; #195 fixed), Fig. 2, Fig. 3, Fig. 5 (parity values, reruns),
  Extended Data Maze (#203 fixed) and Maze settings and techniques (2,000 and 4,000; the old removal panel
  dropped), Extended Data techniques (#200 fixed).

**C7.** B4 at N = 1000 (entry 2026-10-31_1532): when a trained flip-flop network learns the 3-bit
hyper flip-flop, 5-9% of its inactive units become active, and it ends with 232-259 active against
421-457 for an untrained network trained on the new task; R² is the same (0.91). Verdict, by the
pre-registered rule: inactive units do not come back. B4 at N = 4000 (entry 2026-10-31_1750) gives
the same verdict: old networks revive 2.3-7.8% of their inactive units and end with 446-613 active
against 806-956 for untrained networks; R² 0.909-0.914 for both. Figure 4e,f (4d,e since round 7) draws both sizes from
`read_b2_b4.json` of Della job 16553903 (dead-relu-paper#8); `fig_paper_F4_noreturn.py` recomputes
the gate and verdict from the per-run rows and matches the read's own at both sizes. B3 (N = 1000, rate × noise ×
rescaling): the rule binds at the floor, where its literal verdict is "supports several sources",
reached only through its stability clause. Rate+noise reads at the floor and rate alone does not,
because its loss is still falling at 200k. Rate+noise has 970 active units against 556 for noise
alone, at a cost of 0.04 in R². At 75k the verdict is between the bars. Decided on 31 Oct
(dead-relu-paper#17, manager on Pavel's delegation): B3 is treated as between the bars. Figure 5f and
the Extended Data table of pre-registered tests state both measurement points; Results 5 does not take B3 as evidence that raising the active count
requires combining rules. The burst check (entry 2026-11-01_0801) finds six B3 R² values from weights
inside a loss burst or a non-finite stretch, all with the rate penalty; Fig. 5f (since round 7 the Extended Data figure of the combination test) rings them and the
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
| Threshold lift | Bio-inspired | Raises a silent unit's bias once | 1.36× | Screen only; DROPPED on Pavel's instruction (5 Nov): not run on CDDM or the maze, not in the cross-task comparison |
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
- **Absorbing boundary (Pavel's proposed mechanism).** Not supported (2 Nov, entries
  2026-11-01_2119 plan and 2026-11-02_0058 result; branch wp/absorb, 19.4 GPU-h). Silent CDDM and
  DMTS units return above zero (d > 0) but not to activity; recurrent noise, and removing all
  training noise, show that noise slows the loss of units instead of driving it (noise-free CDDM:
  146-159 active at 75k, against 329-358 with the default noise, same seeds); the drift-diffusion
  curve loses to a stretched exponential. No claim changes; the mechanism does not go into the
  Discussion.
- **DMTS setting sweeps (C5 on DMTS).** Fails its rule (2 Nov, entries 2026-11-01_2124 plan and 2026-11-02_1822 result; Della jobs 16628171, 16697623; branch wp/dmts-readout): of 9 networks that learned and have a floor crossing, 8 end with fewer active units, 1 (σ_rec 0.1, s4098854564) ends with more (288 → 308); weight decay 0, metabolic penalty 1 and recurrent-weight penalty 0.2 are untested. C5 stays scoped to CDDM and 3BFF.

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
