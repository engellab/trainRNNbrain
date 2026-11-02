## 2026-11-01 22:40 — Paper push, round 6: the whole paper rewritten so the reader does no work

Pavel read paper main after round 5 on Overleaf and found the writing still too compressed
("simplify the sentences as much as possible … not more than one-two important bits per sentence";
"reduce the reader's work"). His comments are items 15-24 of
`~/Documents/Projects/paper_push/PAVEL_FEEDBACK_2026-11-01.md`. They are now rule 8 of the
plain-english skill: one or two important pieces per sentence, read left to right, every reference
in the same or the previous sentence, concrete names, clarity over brevity.

**Cause of the cramming.** Word caps that the manager set on every section. Each review fix had to
fit the cap, so writers compressed. All caps were lifted; `paper_push/tools/sentence_load.py` flags
long or overloaded sentences for a human to judge.

**Framing changes (Pavel items 15, 19).** No "no study has measured" gap; the problem is shown and
interventions that increase the number of active units are named with their effect on R²; the
paper closes on a cautionary tale. The claim "standard training leaves most units inactive" is
scoped to the Adam optimizer with the default settings: 3BFF networks trained with SGD have 721-759
of 1,000 units active (R² 0.873-0.875; 23 of 24 have no floor crossing; entries 2026-10-31_2010,
_2035). New title: "Most units of large ReLU recurrent neural networks are inactive after training
with default settings". Pavel's own plain rewrite of the removal result was false for 3BFF and DMTS
without further training; the text states both results.

**Errors of fact found and fixed by the reviews of this round** (each checked against code, configs
or notebook by the reviewer):
- Results 5 combination test: the criteria fixed before training (entry 2026-10-15_1342) do not
  define "best single rule"; the outcome depends on that choice and the test is inconclusive under
  both readings. The full account moved to Methods (Combination test); Extended Data Table 2 now
  agrees (fail clause on the logit of the fraction of active units).
- Noisy synapses act on the recurrent and input weights, not every weight.
- The rate penalty is a per-unit rule, not a standard setting, everywhere.
- CDDM rate-penalty networks of Fig. 3a used λ_frm = 0.2 (saved configs); 0.1 was the August sweep.
- Bias is zero except in two groups of Supplementary Table 2b networks (trainable bias; the
  bias-offset rule, `Trainer.py` bias_kick).
- "No silent unit in any 3BFF network" is established only for the three networks of Fig. 1d
  (Fig. 4a redrawn; board C4 scoped).
- R² drop for noisy synapses is at most 0.043 (Supplementary Table 3), not 0.04.
- Methods procedures re-checked against the code: removal-count grid, retraining, burst removal
  order, prune-and-duplicate (four weights set to half the donor's self-weight, d12065d predates
  every run), Adam-moment reset, read-out dropout pool.
- Issue #137 (activation configs and the initial draw): no network in the paper is affected; all
  activation comparisons were trained 2-4 Oct with the shared draw in both arms.

**Code.** trainRNNbrain main: docstring of `Trainer.zero_protected_grads_` corrected (a8bdb68: Adam
adds weight decay after the zeroing); figure panel text for Figs. 1-5 and Extended Data Figs. 1, 2,
4-6 and Supplementary Fig. 1 (ada60b5, 22ee263, ad9b646); `sigma_w` in three activation configs
(cd2e0db, future runs only).

**State.** Paper main 55741fd, 59 pages, clean build. PRs #131-135 and #140-142 merged after
independent reviews; three fresh final readers approved the fixes of their parts.

**Pending.**
- DMTS setting sweeps (entry 2026-11-01_2124; Della 16628171, 27 networks, ~176 GPU-h), asked for by
  Pavel ("why not three?"). If they pass the rule fixed there, the "two tasks" scope of Results 3,
  the abstract and the Discussion widens; their networks use independent initial draws, so Methods
  M01 and the Discussion's initial-draw limitation must then change.
- Pavel's absorbing-boundary hypothesis (item 23): pre-registered analysis running (package absorb);
  enters the Discussion only if supported, as labelled interpretation.
