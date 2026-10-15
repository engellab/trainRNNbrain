## 2026-09-26 10:20 — weight decay sets the active-unit count

The four-point series is complete (CDDM, 100000 iterations, code pinned at f5aa558 so the new points
match the 2026-07-28 originals). Each count is paired with its OWN network's r2, because the cell
mean hides what matters here.

| N | weight decay | active units, each with its own r2 |
|---|---|---|
| 1000 | 0 | 394 @ 0.847, 428 @ 0.866, 438 @ 0.881 |
| 1000 | 1e-6 | 301 @ 0.882, 314 @ 0.866, 321 @ 0.861 |
| 1000 | 1e-3 | **61 @ 0.835, 64 @ 0.848, 65 @ 0.864** |
| 1000 | 1e-2 | 107 @ 0.750, 113 @ 0.752, *1000 @ 0.750* |
| 2000 | 0 | 580, 583, 594 @ 0.85–0.88 |
| 2000 | 1e-6 | 420, 433, 449 @ 0.86–0.88 |
| 2000 | 1e-3 | **100 @ 0.848, 110 @ 0.836**, *2000 @ 0.843* |
| 2000 | 1e-2 | 145 @ 0.781, *2000 @ 0.770*, *2000 @ 0.792* |

The prediction written before submission holds: decay is a DOSE, not a threshold. At N=1000 the
count falls 420 -> 312 -> 63 across 0, 1e-6 and 1e-3, a 6.7-fold collapse, for 0.016 of r2. A
hyperparameter normally set without thought controls this project's central measurement more
strongly than any intervention built for the purpose.

⚠️ THE ITALICISED ROWS ARE THE CRITERION FAILING, NOT NETWORKS USING EVERY UNIT. Those seeds have
the same r2 as their siblings (0.750 against 0.750 and 0.752), so they have not diverged. Heavy
decay shrinks every weight, activity goes uniform, and `p >= 0.05 * q95` then passes everybody. The
scale-free criterion assumes a BIMODAL population; under heavy decay the population is unimodal and
the criterion returns N. This is the third distinct way it has now broken: 996 of 1000 under
shrink-and-perturb at r2 0.009, 0 of 1000 under diverging synaptic scaling, and N of N here. Any
cell whose count equals N needs its participation histogram read before the number is used.

