## 2026-10-01 13:23 — the read-out margin is 7%, and Figure 2 was pooling three tasks

**The margin.** Every figure now reads a network where its loss first comes within **7%** of the
reference, replacing 10%. The 10% margin let a run be read while its loss was still a tenth above
what it would reach, which on CDDM and DMTS is thousands of iterations of real improvement; 5% was
tight enough that runs whose floor fit sits slightly low are never read at all. One constant,
`pr_matrix.EXCESS_DELTA`, and `fig_paper_F1.CONV_MARGIN` and `PLATEAU_TOL` set to match it.
`cddm_criteria.py` and `flipflop_figures.py` had their own copies at 0.10 and now import the shared
one, so the project reads at one margin instead of four.

Two axis labels spelled the margin out as the literal "1.10×". Both are now computed from the
constant: a label that disagrees with the number it describes is worse than no label.

**What moved.** Read-out times come later, as they should: on the 3-bit flip-flop the per-size means
go from 31k/34k/31k/38k to 35k/39k/37k/43k over N = 500 to 4000. The pooled size exponent is
essentially unchanged, T ∝ N^0.143 [0.065, 0.244] against N^0.158 [0.086, 0.251] — 1.35× over an 8×
size range, so "size barely matters within a task" survives the change and is if anything slightly
weaker. Re-measuring the old margin reproduced the previously recorded β = +0.159 and the 31k–38k
range exactly, which is the check that the re-measurement is the same arithmetic and not a new one.

The scaling exponents of Figure 1 panel (c) move by at most 0.003: 3-bit flip-flop 0.445, 6-bit
0.323, CDDM 0.427, DMTS 0.372. The deck had been calling that range "N^0.3–0.4" and the panel's own
docstring "0.33 to 0.42"; both now say 0.32 to 0.45.

**The cost, stated rather than hidden.** Six of the 96 unpenalised runs never come within 7% of their
own fitted floor and so are absent from the read-out panels; at 10% it was four. A run goes missing
when its fitted floor sits a little below what it actually reaches, putting the threshold under its
whole loss curve. The deck says so under the slide.

**The bug this uncovered.** Rebuilding the deck aborted on Figure 2's operating-point assertion: the
pattern `paper_grid/EqType=h_N=` matched two cells, the flip-flop's duplication grid and CDDM's. The
assertion was right, and the cause was worse than an ambiguous string. Figure 2's banner says "all
interventions trained on the 3-bit flip-flop", and that was true while the cache held only flip-flop
runs. It now holds 270 networks across three tasks, and `restrict()` had no task filter — so at
N = 1000 the flip-flop control was being pooled with 13 CDDM and DMTS control networks (288.3 mean
active units against the flip-flop's own 298.2), and the `both` arm with 3 more (969.3 against
938.7). Every contrast in panels b–e was measured against a contaminated reference. `restrict()` now
takes `task=TASK_MAIN` and the duplication pattern is task-qualified. The label guard passes on the
rebuilt figure: 47 labels, none touching data.

**Not caused by the margin change, but visible in the rebuild:** CDDM crosses its absolute
convergence bar at roughly 1,000 of 100,000 iterations, and DMTS at 518 / 31,627 / 78,640 iterations
for N = 500 / 1000 / 2000. The bar is 1.07× the worst final clean loss among that task's runs, so one
bad run lifts it until everything clears it almost immediately. The numbers are the same at 1.10×, so
this is the absolute-bar rule's own weakness and not something the new margin introduced. The matched
read-out offset past the crossing is unchanged for both tasks, which is why panel (c) still holds.

Figures rebuilt: all 25 deck figures in `img/internal_figures/`, plus `fig_paper_F1` and
`fig_paper_F2`.

### The input-scale panel's x axis was mislabelled, and that was most of the "non-monotone" — 2026-10-01 13:25

Pavel asked why slide 11's active-unit count is not monotone in the input scaling factor. Two answers,
one of them a labelling bug.

**The axis was wrong.** `model.input_row_norm = s` rescales every row of the drawn `W_inp` to the
**absolute** L2 norm `s` (`RNN_torch.__init__`, ~line 359). The labels read "input w. ×0.5 … ×20" and
the reference read "input w. ×1 (default)", which says multiples. The default draw's rows sit at
√(n_inputs/N); measured over three seeds at N = 1000, k = 3 that is 0.050, so the four rungs are
**10×, 40×, 100× and 400×** the default and the reference is the **bottom** of the ladder, not a middle
rung. The panel drew it second (`ref_at=1`), which put a 0.050 point between 0.5 and 2 and manufactured
a dip. With every rung at its true scale the curve is single-peaked: 263 → 302 → 339 → 324 → 306.
Labels are now "row norm 0.5 … 20" with the reference "row norm 0.05 (default draw)" at `ref_at=0`, in
`fig_paper_F1.TRACE_FAMILIES` and `fig_slides.INTERVENTIONS`.

**The peak replicates.** The N = 500 cells of the same sweep, independent seeds and never plotted,
give the same ordering: 191 → 219 → 237 → 226 → 213. So the turnover at row norm 2 is not n = 3
scatter. Under the flip-flop's absolute criterion (4e-2) at N = 1000 the shape is also the same
(288 → 353 → 377 → 363 → 342), and q95(p) is 1.95–2.05 in every arm, so the relative criterion's bar is
not moving between arms either.

**Why it turns over.** (i) The knob is an initialisation and nothing holds it: `W_inp` is trainable and
every arm up to row norm 5 ends at the same total, ‖W_inp‖_F ≈ 94 (93.4 / 93.8 / 94.5 / 94.4), from
starting totals of 1.7 / 16 / 63 / 158. Row norm 20 starts at 632 and is still at 377 at 150k, so that
arm is not at a comparable state. (ii) With the total fixed, what the init buys is spread: trained mean
row norm is 1.09 (default) / 1.46 / **1.67** / 1.56 / 11.7, peaking where the active count peaks.
(iii) Every arm is still falling at the read-out, and the bigger the init the later and steeper the
collapse — at 40k the five arms read 378 / 446 / 517 / 506 / **758**, and rung 20 crosses below rung 2
between 100k and 130k. A fixed 150,000-iteration read-out catches five curves that are still crossing,
which is slide 5's own argument applied to this panel.

**Scale of the whole effect.** The reference cell has a 500k budget; training it past 150k with nothing
else changed takes it 263 → 239 → 216 → 198 → 190 at 150k/200k/300k/400k/500k. Every input scale tried
buys less (+38 to +75, each with a CI excluding zero) than the next 350,000 iterations take away.

Slide 11's heading was "Scaling the input weights does not help". Every rung is significantly **above**
the reference, so the heading is now "A bigger input scale adds 40–75 units of 1000, and not
monotonically", and the figure's own note carries the read-out iteration (150,000), which it did not
before. `slide_x_inputscale` and `fig_paper_F1` rebuilt.

### Slide 8 and 9 audited, and the two activation arms the flip-flop never had — 2026-10-01 13:29

**What the two panels actually rest on.** Slide 8 ("A different activation does not help", CDDM) draws
three non-ReLU activations; slide 9 ("Nor on the other task", 3-bit flip-flop) draws **one**. Softplus
and leaky-ReLU were never run on the flip-flop at all, so the two slides were not making the same
claim with the same evidence, and the figure labels named no activation's shape parameter — a reader
could not tell a leak of 0.01 from one of 0.3, or softplus at β = 25 (whose floor is log 2/25 = 0.028,
within a hair of ReLU's zero) from β = 1 (floor 0.69, nothing like a ReLU at all).

**Re-scored independently** (`scratchpad/verify_activations.py`, not importing the analysis package).
The production scorer counts a unit active when `p ≥ 0.05·np.quantile(p, 0.95)`, and `np.quantile`
linearly interpolates between order statistics, which the rule as written never asked for. The check
reimplements q95 as a nearest-rank order statistic, `sorted(p)[⌈0.95N⌉−1]`, and compares the two
**per unit**, not per mean. Thresholds fixed before the run: ≥ 99.5% per-unit agreement in every
network and a per-network gap of at most 2 units. Observed: **100.00% agreement, 0 units** in all 18
networks. The counts behind both slides are not an artefact of the quantile convention.

| slide | arm | active units of 1000, per seed | mean | vs its own ReLU reference |
|---|---|---|---|---|
| 8 | ReLU (reference), CDDM N=1000 @199.9k | 259, 272, 284 | 272 | — |
| 8 | leaky ReLU, leak 0.01 | 279, 278, 276 | 278 | **+6** |
| 8 | softplus, β = 25 | 249, 252, 247 | 249 | **−22** |
| 8 | sigmoid, 7.5(x−0.3) | 191, 191, 234 | 205 | **−66** |
| 9 | ReLU (reference), flip-flop N=1000 @150k | 282, 261, 247 | 263 | — |
| 9 | sigmoid, 7.5(x−0.3) | 226, 242, 252 | 240 | **−23** |

**Reading the table.** "Does not help" survives: no activation raises the active count. But the panel
is not a flat null. Leaky-ReLU's +6 sits well inside the reference's own 259–284 seed spread, so it is
indistinguishable from ReLU. Softplus is not: its three seeds (247–252) fall entirely below the
reference's lowest seed (259), a separation of about 8% of the count. Sigmoid loses a quarter of the
active units. On the flip-flop the sigmoid decrease (−23) has seed ranges that overlap (226–252
against 247–282), so there it is inside scatter. The heading stays as it is; the direction of every
arm is down or flat, never up.

**Two confounds checked rather than assumed.** (i) Slide 8's four arms are identical in every trainer
and model field except `activation_args` — same lr, weight decay, `same_batch`, noise, spectral radius,
bias range — read from the trained networks' own saved configs. (ii) Slide 9 compares a ksweep control
with a 500k budget, read *backward* at 150k from its trace, against a sigmoid run that stopped at 150k.
That is only fair if nothing in training depends on `max_iter`. One thing does: `anneal_noise_levels_`
centres its schedule at `max_iter/3`. Both configs carry `anneal_noise: False`, so it never ran, and
every other field matches (batch 1024 fresh, T = 300, μ = 4, 10 flip steps). The read-back is sound.

**The two missing arms, submitted.** `slurm/SilentReLU_flipflop_activations_della.slurm`, Della array
`14832040` (calibration) → `NBitFlipFlop_std_activations/EqType=h_k=3_N={500,1000}_act={softplus25,leakyrelu}_iters=150000/`.
2 activations × 2 sizes × 3 seeds = 12 jobs at 150k, the sigmoid arm's own budget, so all three
activations are read at one iteration and no cross-budget matching is needed. Both activations keep a
nonzero gradient everywhere, which is the mechanism the dead-unit account is most exposed to: a unit
whose input never crosses zero still receives `dL/dw ≠ 0`, so the absorbing state that traps a ReLU
unit's weights does not exist. If the concentration survives both on this task as it did on CDDM,
"it is the hard zero" is ruled out on two tasks rather than one.

**Pinned to `a831629`**, the commit that trained both the sigmoid arm and the ReLU ksweep control.
`Trainer.py` has since gained ~1200 lines, `RNN_torch.py` ~315, and all four trainer configs ~161 each,
so the development branch would have confounded the activation with a code version. Branch
`ffact-leaky-softplus` is `a831629` plus exactly two model configs and the launcher (`git diff --stat`:
3 files, 201 insertions, 0 deletions). The configs are `0f9688d`'s minus its `input_row_norm` field,
which `RNN_torch` does not accept at this commit; both carry `bias_range [-1, 1]`, the **trainable**
bias that the control and the sigmoid arm were trained with (the default flipped to a fixed zero bias
later, at `ae52c58`, so scoring these against today's `rnn_relu_standard` would import a bias change
into an activation comparison). The launcher enforces all of this at runtime: it aborts unless
`a831629` is an ancestor of HEAD *and* the diff against it is exactly those three files.

⚠️ The first version of that guard compared `rev-parse HEAD` to `a831629` directly, which is the branch
tip and never equals the pin, so every array task would have aborted — caught by checking out the
branch on Della and reading the guard against the real HEAD, not by reasoning about it.

**Validated before the full submission, not after.** Locally: 120 iterations of each activation at
N = 100, both reaching r² ≈ 0.56, both writing a `ParticipationTrace.pkl` and an npz whose
`activation_args` survives as a dict with its values (`{'name': 'softplus', 'slope': 1.0, 'beta': 25.0}`
and `{'name': 'leaky_relu', 'slope': 1.0, 'leak_slope': 0.01}`) rather than as bare keys — the
`storable_` regression that would make these networks unreconstructable offline. On Della: a
600-iteration run of each arm through the real launcher, which exercised every guard, at 0.26–0.27
s/iter → 150k ≈ 11.4 h, inside the 24 h request with room for a 0.35 s/iter slow-node tail.

**Figure labels.** `fig_paper_F1.TRACE_FAMILIES` now names each activation's own parameter — "leaky
ReLU, leak 0.01", "softplus, β = 25", "sigmoid, 7.5(x−0.3)" — on both the CDDM and the flip-flop
family, with the values read off the trained networks' saved configs rather than off a launcher
comment, and the flip-flop family carries the two new arms so `slide_x_activation_ff` picks them up
when the runs land. Results must be synced from Della's own `data/trained_RNNs` first.

### Slide 11b: the input-scale ladder costs nothing in performance — 2026-10-01 13:41

Pavel asked for R² against active units under panel 11, one colour per rung. `fig_slides.inputscale_r2_slide`
→ `slide_x_inputscale_r2`, built on the pattern `metabolic_r2_slide` already set: per-seed dots, a pale
cell mean, a grey line through the means so the dose order survives categorical colour, BASE for the
reference rather than a palette slot.

**Result.** 15 networks spanning 247–342 active units sit within 0.002 of R² = 0.964. The ladder moves
95 units of 1000 and not 0.2% of performance.

**Where R² comes from, and why not the folder name.** The input-scale reference is the ksweep cell with
a 500,000-iteration budget while the four rungs stop at 150,000, so its score prefix is the best over a
run 3.3× longer than theirs while the count beside it is read at 150,000. R² is computed instead from
`loss_clean_train` — the noise-free probe the Trainer records beside the participation vector — at the
same iteration as the count, over `VAR_TARGET_FF = 0.721`, the masked target variance `Trainer.r2_score`
divides by (measured over six independent 1024-trial batches: 0.7158 / 0.7206 / 0.7232 / 0.7184 /
0.7265 / 0.7221).

**End-to-end validation, threshold set before looking.** Load each cell's LastParams into `RNN_numpy`,
run a freshly drawn noise-free batch, take the masked MSE, compare to the trace's own clean loss at its
last probe. Pass threshold 0.005 absolute. Result: 0.00044 / 0.00033 / 0.00064 / 0.00026 / 0.00048 —
pass with an order of magnitude to spare, on independent batch draws.

**A prediction that failed, recorded because it was wrong.** Before running it, the guess was that the
folder-name score is penalised by the input scale itself, since `sigma_inp` is injected through `W_inp`
and row norm 20 carries a `‖W_inp‖_F` of 377 against the others' 94. Re-scoring every cell offline with
noise on and off says no: the clean-minus-noisy gap is **0.0047 at all five rungs**, identical to four
decimals, and the folder scores are 0.9537–0.9543 across the ladder. The offset between the folder score
and the clean probe is uniform, so either source would have ordered the rungs the same way. What
actually differs for row norm 20 is its *minimum* clean loss over training, not its final one — which is
a separate observation and is not in any figure.

**⚠️ A bug found on the way: `penalty_matched.clean_loss` runs every net under the wrong equation.**
`equation_type` is NOT among the keys saved in `*LastParams*.npz`, and `RNN_numpy` defaults it to `"s"`.
`clean_loss` builds its net straight from the npz, so networks trained with `equation_type: h` — which
is every net in `CDDM_std_g0_penalties` and `CDDM_std_g0_drift`, the two sweeps it reads — are scored
under the `s` dynamics. Measured on one `CDDM_std_g0_drift` net: masked MSE 0.00947 under the default
against 0.00926 passed `h` explicitly, so **~2% on CDDM**, small but systematic, and it propagates to
`fig_S2_cost`. On the 3-bit flip-flop the same mistake is not small: 2.08 against 0.026, an 80× error
that scores a solved network as broken. Not fixed here — flagged.

### A commit rule, because the tree is shared — 2026-10-01 13:46

Two commits today, `7695f65` and `b8e8329`, each swept up another agent's in-flight edits to
`fig_paper_F1.py` and `fig_slides.py` and carried them under a message that did not mention them.
Nothing was lost, but it is only luck that the edits were compatible.

`CLAUDE.md` now exists at the repo root, which every agent working here loads: never `git commit -a`
or `git add -A`, name every path, read `git diff <path>` before staging it, and leave any file whose
diff contains hunks you do not recognise — saying in your reply which file you left and why.
Interactive staging does not work in this environment, so there is no partial-stage escape hatch and
the only safe move on a mixed file is to not commit it.

