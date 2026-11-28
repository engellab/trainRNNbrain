## 2026-11-25 13:30 — Part C on older networks: the estimates replicate, the decay-time test is refuted as written, and a post hoc check says why

Rules: `2026-11-25_1311_rules-mechanism-noise-decay-and-switch.md`, part C. Code: `mech_optimizer.py` (ba5cd89 for the runs, 2e5ab14 for
the verdicts, committed before the results were read; 78a157d for the post hoc mode `bins`). Della CPU jobs 18376947 (tasks 30-42) and 18377426;
3BFF Adam states on the Mac. Outputs: Della `trained_RNNs/CDDM_mech_reads/` (and `posthoc/`), copies in firstmate `data/mechanism-tests/partC_old/`.
The same reads on part A's networks follow in the result entry. No GPU was used.

**Terms.** A weight "IA" goes from an active unit onto an inactive one. *Predicted decay time*: (√v + ε)/(lr · 1e-6), v Adam's running mean of
the squared gradient. *Measured decay time*: −Δ/ln b, b the slope of w(t + Δ) on w(t).

### Verdicts by the pre-set bars

| Test | Verdict | Numbers |
|---|---|---|
| C1, the reviewer's estimates from the Adam states (3 CDDM controls, N = 1,000) | **replicated** | gradient ratio 28.6, 52.9, 47.2; predicted decay time IA 489, 340, 416, active-to-active 13,421, 16,847, 18,616, II 53, 32, 34; step ratio 0.15-0.34 over nine controls; row ratio 0.199, 0.166, 0.187 |
| C2, measured IA decay time within a factor 3 of 345-496 | **refuted** | median of 18 values (3 networks, 3 times, Δ = 500 and 1,000) 5,734 iterations; the bar for refuted was 5,000. The three c1 controls give 5,483 |
| C3, noise-free weights decay at less than half the time | **refuted** | predicted ratio 0.05, but measured ratio 1.02 (5,851 against 5,734) |
| C4 (N2), weight change against gradient | **supported** | 3 CDDM controls: loss-gradient norm on a unit's incoming weights 263 to 397 times larger for active units; weight change 1.4 to 1.6 times larger over 500 and 5,500 iterations |

3BFF, no bar: gradient ratio 6.3, 6.9, 13.3; weight-change ratio 1.2 over 500 and 2.1-2.5 over 5,500 iterations. Predicted IA decay time
20,000-31,000 (default, 200,000 iterations), 50 to 75 times the CDDM value; measured 38,000-114,000, 7 to 20 times the CDDM value.

### Why C2 and C3 failed as written, and what that does not change (post hoc)

The slope in C2 weights every weight by its square. The largest 5% of the IA weights hold 71-80% of the summed squared IA weight in the noisy
CDDM networks, so the slope is mostly theirs. A per-weight check at the end of training (no bar in the rules) groups the IA weights by their own
predicted decay time and measures the decay within each group, over 500 and 1,000 iterations, in six noisy networks:

| Predicted decay time (iterations) | Share of IA weights | Measured, Δ = 500 | Measured, Δ = 1,000 |
|---|---|---|---|
| 100-300 | 27-33% | 229-287 | 328-418 |
| 300-1,000 | 32-37% | 645-716 | 797-893 |
| 1,000-3,000 | 15-19% | 2,068-2,233 | 2,439-2,656 |
| 3,000-10,000 | 7-9% | 7,045-8,419 | 8,379-10,537 |
| above 10,000 | 3-4% | 37,409-67,932 | 51,695-70,960 |

(The bins below 100 hold 6-11% of the weights; their measured times are 204-558 over 500 iterations, longer than predicted, because v at the end of
training is not the v of the interval.) Between 100 and 10,000 predicted iterations the measured time lies inside its bin or at most 1.4 times above its top,
so the formula describes individual weights there. The weights of middle size (40th to 60th percentile of |w|) decay in 449-620 iterations over 500 and 659-857 over 1,000. The
largest 5% decay in 7,700-12,000. So:
- The estimate 345-496 is right for the *typical* weight onto an inactive unit and was replicated.
- It is not the time in which an inactive unit's *input* shrinks, because most of the squared weight is in a few weights that decay about 20 times slower.

Noise-free networks: 90-92% of the IA weights are exactly 0 at the end (float16 snapshots; 0.0% in the noisy networks), so their predicted 22
iterations is the floor ε/(lr·1e-6) of weights that have already vanished, and the measured slope is that of the surviving 8-10%. The C3 ratio
compares survivors and is not evidence against faster decay without noise; the number of vanished weights is evidence for it. The
verdict "refuted" stands for the test as written.

### What this means for the account

- Adam equalises the steps: weight change differs 1.4- to 1.6-fold where gradients differ 263- to 397-fold (C4). The reviewer's statement
  "a gradient 28-52 times smaller gives a step only 3-5 times smaller" is about Adam's own running averages; the loss-gradient norm on a unit's
  rows under the training noise differs more.
- Decay through Adam is a real and fast process on typical CDDM IA weights and a slow one on 3BFF.
- None of this is causal. Part A decides whether the decay changes M.
