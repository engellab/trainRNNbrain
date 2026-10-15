## 2026-07-28 — metabolic-cost sweep: the standard regularizer never rescues, and at strength deepens the problem

Array `11687340`, 36/36 COMPLETED (4 λ × 3 sizes × 3 seeds, h equation, standard RNNs, 30000
iterations). λ=0 baselines come from the unpenalized cells of `CDDM_std_g0` / `CDDM_std_g0_Nsweep`.

![Silent units vs metabolic-cost strength](../img/internal_figures/silent_vs_metabolic.png)

| N | metric | λ=0 | 0.01 | 0.1 | 1.0 | 10.0 |
|---|---|---|---|---|---|---|
| 100 | truly silent | 3% | 0.3% | 0.3% | 2% | **20%** |
| 100 | scale-free | 12% | 12% | 17% | 33% | **59%** |
| 500 | truly silent | 22% | 23% | 21% | 22% | **32%** |
| 500 | scale-free | 41% | 41% | 41% | 50% | **69%** |
| 1000 | truly silent | 42% | 44% | 40% | 39% | 43% |
| 1000 | scale-free | 59% | 61% | 57% | 57% | 62% |

R² is **0.81–0.87 across every cell**, so nothing here is a penalty destroying the task.

**Observations.**

1. **No λ rescues anything.** Across four decades the silent fraction never falls meaningfully below
   baseline at any size. Compare `frm`, which drives it to exactly 0 at every size — the contrast is
   total, not a matter of degree.
2. **At λ=10 the penalty makes it substantially worse**, and most visibly where there was headroom:
   at N=100 the scale-free silent fraction goes **12% → 59%**, a five-fold increase, while R² only
   drops from 0.858 to 0.807. At N=500, 41% → 69%. The mechanism is the obvious one — `mean(fr²)`
   penalizes rate *magnitude*, so it pushes the whole population down and pushes the weakest units
   through the floor.
3. **At N=1000 it is flat** (39–44% truly silent across all λ). Not a contradiction: the network is
   already ~42% silent without any penalty, so there is little room left for the penalty to add. The
   effect appears where the baseline is low, which is why sweeping *sizes* as well as λ mattered.
4. **The two metrics separate at intermediate λ.** At N=100, λ=1 shows 2% truly silent but 33%
   scale-free silent: the penalty has pushed a third of the population into near-irrelevance without
   driving it to exact zero. A strict threshold alone would have reported this cell as healthy.

**Why this matters for the paper.** §2 previously rested on a prediction. It is now measured: the
activity regularizer this literature routinely uses does not solve the silent-unit problem, and
applied at strength it deepens it while the task is still solved. That is a stronger claim than
"here is another penalty", and it is the answer to the first objection any referee raises.

**Falsifier check.** The pre-registered falsifier was "any λ that drives silence toward 0 while
keeping R² ≈ 0.85 collapses §2." No λ at any size did that. §2 stands.

