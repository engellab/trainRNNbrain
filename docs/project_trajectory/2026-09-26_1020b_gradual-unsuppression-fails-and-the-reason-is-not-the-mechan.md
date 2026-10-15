## 2026-09-26 10:20 — gradual unsuppression fails, and the reason is not the mechanism

`reinit_mode="rescale"`, all 12 jobs (2 normalisation x 2 alpha x 3 seeds). The rule keeps a silent
unit's own trained incoming weights and nudges them every step -- excitation up by alpha, inhibition
down by alpha -- until it fires, redrawing the outgoing column at 1/sqrt(N) each step, stopping on
revival, capped at a cumulative 8x.

| cell | active per seed | mean | r2 |
|---|---|---|---|
| control | 254, 269, 301 | 275 | 0.945 |
| normalised, alpha 1.0005 | 220, 262, 275 | 252 | 0.939 |
| normalised, alpha 1.002 | 232, 265, 274 | 257 | 0.944 |
| unnormalised, alpha 1.0005 | 246, 272, 301 | 273 | 0.940 |
| unnormalised, alpha 1.002 | 233, 259, 259 | 250 | 0.942 |

Every cell sits inside the control's range, which is the falsifier written before submission.
Normalisation changes nothing, as the algebra says it cannot: with excitatory drive P and inhibitory
drive M the post-event drive is lambda*(alpha*P − M/alpha), and lambda is a positive common factor,
so the revival condition alpha_total^2 > M/P is identical either way.

**WHY IT FAILS, from the weights themselves.** The rule can only RAISE a unit's excitation relative
to its inhibition. The units that end up silent have the opposite:

| cell | E/I ratio, silent units | E/I ratio, active units | row norm, silent |
|---|---|---|---|
| control | 0.79, 0.98, 1.14 | 0.91–1.06 | 0.35–0.39 |
| rescale, alpha 1.0005 | 0.41, 0.47, 0.78 | 1.33–1.45 | 0.30–0.35 |
| rescale, alpha 1.002 | 0.17, 0.25, 0.52 | 1.09–1.22 | 0.18–0.41 |

Their E/I ratio ends far BELOW the control's, and further below at the larger alpha, so the rule
caused it while running opposite to its own arithmetic. The only route: the unit revives, and a
revived unit has gradient again, and the gradient's preferred direction is to put it back. Each
cycle overshoots, the unit lands with less excitation than it started with, the rule boosts it
again, and the cap eventually abandons it there. The units that stuck ended at E/I 1.09–1.45.

Row norms rule out any runaway: 0.18–0.41 against the control's 0.35–0.39, so the cap and the
normalisation did their jobs. This is a clean negative -- the mechanism ran as designed and the
network undid it.

**Two independent rules now say the same thing.** `bias_kick` lifts about 475 units per network and
keeps 70 of them, an 85% re-suppression rate; `rescale` drives the E/I ratio of its failures below
where it found them. Neither the frozen gradient nor the weight magnitudes are what keeps these
units off. The network rejects units it has no use for, and the thing duplication supplies that
these rules do not is a REASON for the gradient to keep the unit -- which the decomposition above
prices at 34% outgoing column, 47% incoming weight distribution, 19% placement.

