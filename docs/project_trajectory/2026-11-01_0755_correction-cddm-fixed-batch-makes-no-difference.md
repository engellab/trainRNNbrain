## 2026-11-01 07:55 — Correction to 2026-11-01_0735: the CDDM "fixed batch" is no difference; the per-network fall excludes the rate penalty

Work package fig3-r2 (dead-relu-paper#54), after the review of PR #69. Corrects the bridge paragraph
and one sentence of entry `2026-11-01_0735_figure-3-round-2-redraw-and-results-3-numbers.md`.

**The fixed batch.** Entry 0735 said that the CDDM sweep networks differ from the size series in
`trainer.same_batch` (true against false) and treated that as a training difference. It is not one.
`TaskCDDM.get_batch` builds the same 2 × 15 × 15 = 450 trials on every call: it draws no random
number unless `shuffle=True`, and `Trainer.run_training` is called with `shuffle=False`
(`run_experiment.py`, the `trainer.run_training(...)` call). With `same_batch: true` the batch is built
once; with `false` it is rebuilt each iteration, identical. So the two groups of default CDDM networks
differ only in seed and budget, and 420 (416-424, the sweep defaults at their own crossing) against
431 ± 10 (Fig. 1c) is seed variation, within one standard deviation. The entry's source comment also
over-generalised: `same_batch: true` holds for the weight-decay, recurrent-noise, metabolic,
spectral-radius, recurrent-weight-penalty, activation and sparsity sweeps; the CDDM input-weight-scale
sweep (`CDDM_c1_snapshots`) and the CDDM revival screen (`CDDM_screen3_c75000`) have `false`
(checked by the reviewer on Della, 15 of 15 and 9 of 9 configs). Consequences: the fixed-batch
sentence is removed from Methods M08; Results 3 now gives the sweep defaults' own count at their
crossing (420) next to 337; issue dead-relu-paper#68, which rested on the difference, is closed as
based on a false premise.

**The per-network fall.** Entry 0735 states the check correctly (124 of 124 networks without the
rate penalty; the 5 rate-penalty networks reach a crossing and stay at 1,000). Results 3 as first
written in PR #69 dropped the exclusion; the text now says "every network without the rate penalty
that reaches a floor crossing", and the range of setting means (160 to 453) is stated as excluding the
rate penalty in the same clause.

**Figure changes in the same review** (no numbers change): the sparsity penalty alone is drawn in the
teal setting palette in Extended Data Fig. 4f, as in Fig. 3a; revival methods have one marker per
method (cap: square; re-initialising: diamond, shade for outgoing weights kept or zeroed; the CDDM
screen re-initialises with outgoing weights kept, Trainer's default `random` mode, no `reinit_mode`
in its configs); the means row of Fig. 3a uses open diamonds; SGD at learning rate 3 takes the
darkest teal shade instead of black.
