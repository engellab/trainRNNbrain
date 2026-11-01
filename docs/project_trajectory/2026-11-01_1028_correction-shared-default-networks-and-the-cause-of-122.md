## 2026-11-01 10:28 — Correction to 2026-11-01_1000: 5 shared default networks, not 8; the cause of the committee's 122

Work package fig3-r3 (dead-relu-paper PR #96, review comment 6562010265). Corrects entry
`2026-11-01_1000_figure-3-round-3-markers-and-the-116-network-recount.md`. The count of 116 stands.

**Shared networks.** Entry 1000 says "Eight default networks appear in more than one sweep". The 8
is the surplus of cache entries over networks (140 − 132), and it comes from 5 networks. The 3
default networks of `CDDM_std_g0_drift_bias0` each appear 3 times (default of the weight-decay,
metabolic-penalty and activation-function sweeps), and the 2 default networks of `CDDM_std_g0_recinit`
each appear twice (default of the spectral-radius and recurrent-weight penalty sweeps). So 5
networks fill 13 cache entries, 8 more than they would counted once. All 5 reach a floor crossing,
which is why the gap between 124 entries and 116 networks is also 8.

**Where the 116 come from.** The 116 cover both tasks and the revival methods: 91 CDDM and 25 3BFF
networks; 97 from the setting sweeps and 19 from the revival methods (the networks of Extended Data
Fig. 4a-k without the rate penalty that reach a floor crossing).

**Why the round-2 rigor note got 122.** The rigor note's script selected runs with
`if run.get('t_star'):`. A run with no floor crossing stores `t_star = NaN`, and `bool(NaN)` is
`True`, so all 11 runs without a crossing passed. The script then dropped the 10 runs labelled rate
penalty or "both", which left 132 − 10 = 122 networks. Six of the 122 have no floor crossing: the 5
sparsity-penalty networks (λ_rws = 0.05) and 1 CDDM network at initial input-weight scale 0.5. For
those six, `argmin(|it_act − NaN|)` returns index 0, so their count "at the crossing" was read at
iteration 0, 745 to 790 active units, and each of them appeared to fall. So 122 = 116 + 6. Found by
the PR #96 reviewer; I reproduced it from the cache: the same selection gives 122 distinct
networks, 6 of them without a crossing, with those labels.

The sentence of entry 1000 that said 122 "is not reproduced by any of the variants below" is
withdrawn.
