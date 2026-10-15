## 2026-10-13 — Claim 1 closed: the flip-flop at its floor, the retrain bug fixed, Figure 1 and the manuscript updated

**The fine-tune bug.** The intact-CDDM drop under fine-tuning (yesterday's open item) was a bug in
`retrain_curve`, found by a delegated investigation: scoring reseeded the generator the training
noise is drawn from, so training replayed the same 250 noise samples after every score and, on
CDDM's fixed grid, memorised them by waking silent units. Fixed (7608099: generator state saved and
restored around scoring; non-finite updates skipped). Every retrain test re-run; the intact controls
are flat and the conclusion stands (numbers in the entry above, now corrected). The project's own
warm start is unaffected (< 0.001 over 3,000 iterations, with or without the saved Adam moments).

**The flip-flop reaches its floor within 200k.** The claim-1 runs continued to 200k cross 1.05x their
noisy-loss floor at 27k, 39k and 69k (266-354 active there). At the floor, removing every silent unit
takes held-out r2 0.94-0.95 → 0.25-0.48; pruning and retraining restores 0.937-0.947, random pruning
0.920-0.924. Size series: N = 500 6 of 6 runs reach the floor at 200k (38-111k), N = 1000 2 of 3
(64-66k; the third ends 7% above its fitted floor - continuing it to 300k is offered, not done);
N = 2000 and 4000 still training (~3 h and ~2 days on 2026-10-13 morning).

**Panel c under the noisy floor** (`_panel_c_numbers.py` run on Della; per size, mean ± SD):
CDDM 299.0 ± 3.0 / 413.7 ± 13.3 / 561.7 ± 12.1 / 788.7 ± 11.8 at N = 500 / 1000 / 2000 / 4000
(59.8 → 19.7%, N^0.46, crossings 26-30k); DMTS 5 tau 226.3 ± 35.0 / 439.7 ± 201.1 / 462.0 ± 11.3 at
N = 500 / 1000 / 2000 (45.3 → 23.1%, N^0.51); the flip-flop at the end of 75k 197.7 / 269.3 / 361.3
/ 541.7 (39.5 → 13.5%, N^0.48) until every size has a floor.

**Figure 1, final for claim 1** (e68a591): panel b is the example network untrained (its initial
weights rebuilt from the chain's first seed, 945 of 1000 active, one mode) above the same network
after 200,000 iterations (183 active, two modes), on the same trials; the example is the flip-flop
control continued to 200k. Panel d reads all three tasks at their floor with the retrain stars;
panel e reads the flip-flop's 200k continuations (116 more units silenced after its floor).

**Pushed.** `engellab/trainRNNbrain` branch `restore-oct5` (first pushed at 31ff939). The paper repo
`ptolmachev/dead-relu-paper` (cloned to ~/Documents/GitHub/dead-relu-paper), `main`: Figure 1 and its
caption (fa083ee), then claim 1 in the text (41c5def) - results rewritten as "most units of a trained
network fall silent, and the network does not need them", Methods sections for the ablation,
prune-and-retrain and the noisy-floor read-out, the supplementary ablation figure, and every stale
Fig. 1d / N^0.443 / N ≈ 14,000 / "3.5 decades of empty valley" removed. The knob paragraphs keep
their own subsection (`sec:knobs`) until Figure 2 (what fails) is rebuilt; their numbers are from the
trainable-bias sweeps and flagged. Two NUMBER NEEDED notes added: the Otsu-criterion agreement on
standard networks.

