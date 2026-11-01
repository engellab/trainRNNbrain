## 2026-11-01 12:04 — Correction to 2026-11-01_1150: "at most 6%" fails on twelve batches; Results 4 says "fewer than a tenth (at most 6.3%)"

Corrects `2026-11-01_1150_c4-3bff-reach-over-seven-batches.md`. That entry's wording "at most 6% of
the reached units are active 5,500 iterations later" held in its seven batches (largest 6.0%).
methods-r4 then ran five more batches with the production code (task seeds 101-105; trainRNNbrain
`wp/methods-r4`, rule `2026-11-01_1143`, result `2026-11-01_1200`, which also reproduces the paper's
seed-12345 numbers exactly). In s4113307577, task seeds 102 and 105 give 6.2% (10 of 160) and 6.3%
(10 of 159), so "at most 6%" fails. I read the per-batch table of entry 1200 before writing this
entry.

Over all twelve batches (the paper's, five production-code, six independent-code), 16-30% of the
low-active units are reached (108-222 of 658-735) and 0.7-6.3% of those are active 5,500 iterations
later. "About a sixth to a third" still holds. The C4 rule's 10% bound (`2026-11-01_0948`) holds in
every batch.

**Paper change** (results-r4, PR #117): Results 4 now reads "In each of twelve batches of 1,024
trials, about a sixth to a third of the low-active units are reached, and fewer than a tenth of these
(at most 6.3%) are active later", matching M07's "twelve batches" (methods-r4). Board row C4's
detail paragraph is updated on this branch.
