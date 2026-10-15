### 2026-10-14 morning — Pavel's decisions

1. Runs whose floor fit has beta on its bound are read at 1.05x the minimum of their smoothed noisy
   loss (they are converged; the fit fails). Implemented as a fallback inside `common.noisy_floor`
   with `floor_kind` recorded (agent-fix, running).
2. The held-out batch is seeded project-wide (the flip-flop's task generator was unseeded: ±0.03 SD
   on one ablation score); every held-out R² is recomputed under a versioned cache key (agent-fix).
3. C4: test whether the flip-flop has really converged - "if there is a difference there should be
   some other difference in trainability" (agent-c4b, running).
4. The rate penalty's R² cost on standard CDDM: Pavel "negligible - is it even significant?".
   Across networks (5 per arm, each network's own held-out R², 36 draws): at 50k 0.8331 ± 0.0019
   vs 0.8282 ± 0.0029, difference 0.0050, Welch p = 0.016, Mann-Whitney p = 0.03; at 75k 0.8331 ±
   0.0039 vs 0.8265 ± 0.0031, difference 0.0066, p = 0.018 / 0.016 (the same networks, so not two
   replications). Significant at n = 5 but under 1% of R²; wording: "keeps every unit at a cost of
   under 1% of R²".
5. B3 rate cells after the pilot and B4 at N = 4000 on Della's 80 GB cards: approved.

