## 2026-11-01 08:13 — fig5-r2: the paper side, and one more fact for the rate-penalty size test

Package fig5-r2 (dead-relu-paper#55). Follows `2026-11-01_0801`.

**Paper changes** (dead-relu-paper branch `wp/fig5-r2`, b23d850): Results 5 rewritten (heading
"Training rules that act on individual units"; 5.2 numerals per 100 words by `tools/textstats.py`),
Figure 5 and Extended Data Figs. 7-8 from code commit 0e03245, captions within the blueprint limits,
Extended Data Table 2 with the glossary names of the three pre-registered tests, Supplementary
Table 2 regrouped (one block per rule; new columns: networks whose 75,000-iteration weights lie
inside a loss burst, and the removal analysis's medians), Methods "Per-unit rules" (448 words by
the same tool) and "Combination test" (112).

**The rate penalty on CDDM and on 3BFF, as Results 5 now tells it.** On CDDM (Fig. 3a; λ_frm = 0.2,
75,000 iterations) all 1,000 units of each of five networks are active, all five reach a floor
crossing (18,762-21,150 iterations) and held-out R² at 75,000 iterations is 0.828-0.830 against
0.831-0.834 without the penalty (`data/fig_paper_F2_fails_cache.pkl`, re-read here: rungs "none"
and "rate" of `slide_k_frm_rws`). On 3BFF (λ_frm = 0.1, 200,000 iterations) no network at
N ≤ 1,000 reaches a floor crossing and the size test fails. Results 5 names both strengths and both
budgets and attributes the difference to none of them (blueprint section 3).

**A fact the burst check adds to the size test.** All three rate-penalty networks at N = 2,000 have
non-finite loss values: 2,797, 3,686 and 2,004 of their 200,000 iterations (`burst_check.json`,
field `n_nonfinite`); at N = 4,000, 1,009, 605 and 500; at N = 500 and 1,000, none. The size test's
stability clause asks for "a finite loss", so at N = 2,000 it is not met either. The outcome (fail)
does not change; Extended Data Table 2 now states the fact in its result column.
