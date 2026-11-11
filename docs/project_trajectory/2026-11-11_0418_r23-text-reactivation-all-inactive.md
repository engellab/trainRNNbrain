## 2026-11-11 04:18: Results 4 gets one denominator for reactivation: 3 to 5% of all inactive CDDM units are active 5,500 iterations later (dead-relu-paper #233)

The round-22 committee (#231, Reviewer W item W6) found a mismatch in Results 4. The sentence beginning "By contrast" gives the
Maze fraction over all inactive units (17% and 19% of the units inactive at the learned point are active
700 updates later; entry `2026-11-07_0235`, G15). The 3BFF and CDDM fractions it is compared with are fractions of
subclasses: silent units, and low-active units above the gradient threshold (entry `2026-11-01_1021`). For CDDM
the subclass fraction (a fifth to two-fifths) is higher than the Maze one. The paper package r23-text (#233) needs the
3BFF/CDDM fraction over all inactive units.

**How it was computed.** `trainRNNbrain/experiments_and_analysis/r23_reactivation_all_inactive.py` (branch
`wp/r23-text`), run from the main checkout on the Mac (seconds, CPU). It reads the participation vectors that
`fig_paper_F4_noreturn.py fig1d` stored for the Fig. 4a-c networks (`data/fig4_noreturn/activity_fig1d/`, `p_0` on
the Fig. 1d weights and `p_5500` 5,500 iterations later) and the class inputs in `data/fig4_noreturn/c4_fig1d/`
(`ref_n_on` for silent units, `strict_g_in` for the gradient threshold). Active means p >= 5% of the network's
95th-percentile participation, the paper's rule.

**Oracle.** The class counts recomputed by the script equal the ones in entry `2026-11-01_1021` for every network:
active 430, 416, 438 (CDDM) and 269, 342, 287 (3BFF); silent 310, 288, 253; above-threshold low-active 40, 44, 60
and 136, 155, 161; of these, active at +5,500: 9, 17, 19 (23%, 39%, 32% in that entry) and 1, 4, 7 (0.7%, 2.6%,
4.3%); silent units active at +5,500: 0, 0, 1. The per-unit vectors therefore line up with the classes behind Fig. 4c.

**Result** (inactive units on the Fig. 1d weights that are active 5,500 iterations later):

| Network | Inactive | Active at +5,500 | Fraction |
|---|---|---|---|
| CDDM s1423687489 | 570 | 17 | 3.0% |
| CDDM s2657899762 | 584 | 31 | 5.3% |
| CDDM s2670295995 | 562 | 25 | 4.4% |
| 3BFF s1258717025 | 731 | 1 | 0.1% |
| 3BFF s2461724787 | 658 | 4 | 0.6% |
| 3BFF s4113307577 | 713 | 11 | 1.5% |

The 3BFF values use the one reference batch (task seed 12345). The paper's 3BFF numbers are ranges over twelve
batches (Methods: Incoming-weight gradients), so the 3BFF values are recorded here only and not printed.

**What changes in the paper.** Results 4 now reads: "In each CDDM network, 3 to 5% of all inactive units are active
5,500 iterations later. By contrast, in the two Maze networks of 1,000 units, 17% and 19% of all units inactive at
the learned point are active 700 updates later." Methods: Incoming-weight gradients gives the three CDDM counts
(17 of 570, 31 of 584, 25 of 562). With the same denominator, the Maze fraction is 3 to 6 times the CDDM one, so
the words "By contrast" now match the numbers. The intervals differ (5,500 iterations compared with 700 PPO updates),
and the paper states both.
