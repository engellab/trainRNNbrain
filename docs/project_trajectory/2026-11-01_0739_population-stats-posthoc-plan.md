## 2026-11-01 07:39 — Population statistics: post-hoc checks requested by the PR #66 review (plan, before running)

**Post hoc.** These checks were defined after the results of `2026-10-31_2220_population-stats-results.md`
were seen and after the independent review of dead-relu-paper PR #66 (comment 6560093282). They do
not change the pre-registered labels. They say what produces the labels. Code:
`trainRNNbrain/experiments_and_analysis/population_stats_posthoc.py`, using the same networks, batch,
active units and responses as `population_stats.py`. The results go in a separate entry.

**P1, soft normalisation of S2.** Divide each unit's rates by (sd + c) instead of sd, with c = f x the
median rate sd of the network's active units, for f = 0.01, 0.05 and 0.10 (Churchland et al. 2012,
Nature 487:51-56, use a soft normaliser of this form). Compute the PR over the units with nonzero
variance and over the active units, and the relative difference (all - active) / active. For each f,
apply the rule's 10% and 2% limits per task, for information only. Expected from the review: CDDM
and 3BFF stay above 10% at f = 0.05; the sign on DMTS reverses.

**P2, what the extra S4 class holds.** For each class of the all-unit run: its size, its inactive
units, its selective units and its centroid. Also the median summed partial R^2 of the clustered
inactive units and of the clustered active units.

**P3, S4 on selective units only.** Repeat S4 (same k-means, k = 2-8, silhouette) on the units that
are selective (S3), over all units and over the active units. Compare k*.

**P4, S4 at fixed k.** k = 3 and k = 4 in both runs. Report the largest matched-share difference.

**P5, size of the CDDM inactive units' selective responses.** The median sd of the S3 epoch response
of inactive selective units, against that of active selective units. Then add Gaussian noise
(seed 20261031) with sd = 1% and 5% of the median epoch-response sd of the active units, and
recount the inactive selective units. All three tasks are reported.
