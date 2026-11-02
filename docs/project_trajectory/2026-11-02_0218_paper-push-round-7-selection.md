## 2026-11-02 02:18 — Paper push, round 7: ruthless selection

Pavel, after round 6 (feedback item 25): the paper became clear but "huge"; "whenever you choose to
explain something you should elaborate, but you should ruthlessly prioritize". Now rule 9 of the
plain-english skill: rank a section's points, keep the few it exists to make and explain each fully,
move or cut the rest; length budgets test the selection and are met by removing points, never by
compressing sentences.

**Process.** One planning agent read the whole paper and wrote a selection plan (issue #127, "Round 7
selection plan"): seven story claims S1-S7, a KEEP/MOVE/CUT verdict per paragraph, a budget per
section, thirteen cross-section risks. Five packages then worked in parallel on disjoint files; each
was reviewed by an independent agent. Reviewers caught two places where a budget had been met by
compression (abstract, Methods) and one move that removed an answer (Results 4); all were reverted.

**Result (paper main 9c1f764, 57 pages, clean build).**

| Section | Before | After |
|---|---:|---:|
| Abstract (text) | 385 | 315 |
| Introduction | 1,084 | 677 |
| Results | 4,461 | 2,850 |
| Discussion | 3,025 | 1,425 |
| Methods | 7,714 | 5,439 |
| Main captions | 374-692 | 296-364, one page each |

New: Supplementary Methods (2,075 words: burst detection, no-floor-crossing cases, grids, per-rule
optimizer details, post hoc checks, history of parameter choices) and four Supplementary Notes
(combination test; tanh networks; biological counterparts of the per-unit rules; positive rates of
inactive units).

**Figures (trainRNNbrain acd22c0).** New Fig. 1e: fraction of units selective for a task variable
over all units and over the active units, nine Fig. 1d networks; the reviewer recomputed all 18
values with independent code (own forward pass, partial R² from t statistics) and they match to four
decimals. Fig. 4d (tanh) moved to Extended Data Fig. 7d; old 4e,f are now 4d,e. Fig. 5f (combination
test) is its own Extended Data figure. Correction to the merge-label counts: entry 2026-11-02_0207.

**Facts fixed on the way.** Results 3 closing scoped to the setting sweeps and the two
machine-learning methods (116 networks: 91 CDDM, 25 3BFF), not the optimizer or tanh networks; the
abstract's rule claim scoped to N = 1,000 (rate-penalty 3BFF networks of 500 units lose 0.15 in R²);
under rescaling the Adam moments of every inactive unit's outgoing weights are zeroed every
iteration (now stated, Supplementary Methods).

**Pending.** DMTS setting sweeps (Della 16628171; entry 2026-11-01_2124).
