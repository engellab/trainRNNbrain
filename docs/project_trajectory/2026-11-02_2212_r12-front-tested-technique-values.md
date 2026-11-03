## 2026-11-02 22:12 — r12-front: what the tested parameter values of the five training techniques show

Package r12-front (dead-relu-paper #127, Pavel's item 50). Pavel objected to the abstract's
"At parameter values we chose, each technique increases the number of active units": it reads as if
the values were picked because the techniques work there. Before rewording, I tallied every set of
parameter values in Supplementary Table 3 block a (`floats/ST_rule_settings.tex`, paper main
99de503). No new computation on networks: the numbers are the table's own columns.

**Scope.** 3BFF networks of N = 1,000 units, block a (sets trained to 200,000 iterations), read at
75,000 iterations (the same-size read-out rule). The rate-penalty set lambda_frm = 0.01 is in block
a but was not measured at 75,000 iterations, so 55 sets remain. The two prune-and-duplicate rows at
N = 500 and 2,000 are outside the scope.

**Rule.** A set "increases the number of active units" if its Fold column (mean number of active
units over the control networks' mean, same N and measurement point) is above 1. "Slightly lower
R^2" means -0.03 <= Delta R^2 < 0 (Delta R^2: mean held-out R^2, 36 noise draws, minus the control
mean of about 0.95).

| Technique | Sets | Fold > 1 | Fold range | Delta R^2 range | -0.03 <= Delta R^2 < 0 |
|---|---|---|---|---|---|
| Rate penalty | 2 | 2 | 2.77-3.72 | -0.023 to -0.021 | 2 |
| Noisy synapses | 5 | 5 | 1.31-2.19 | -0.311 to -0.001 | 3 |
| Rescaling | 26 | 19 | 0.67-3.82 | -0.387 to -0.004 | 24 |
| Prune-and-duplicate | 9 | 9 | 1.48-3.07 | -0.008 to +0.000 | 7 |
| Read-out dropout | 13 | 13 | 1.52-2.62 | -0.021 to -0.006 | 13 |
| All | 55 | 48 | | | 49 |

- The seven sets with fold below 1 are all rescaling sets (folds 0.67-0.93): the four sets of the
  first form (no target), m = 0.2 and 0.5 with all sources, and m = 2.5 with length to the active
  median and the input row rescaled.
- Delta R^2 outside [-0.03, 0): noisy synapses sigma_w = 2 and 3 (-0.17, -0.31); rescaling
  m = 2.5 length not kept (-0.39, one network with R^2 = -45 excluded) and hold 4,000 (-0.06);
  prune-and-duplicate two sets at -0.000 and +0.000. Both conditions hold together with 42 of 55 sets.
- Does R^2 fall as the strength grows? For noisy synapses, yes (-0.001, -0.004, -0.022, -0.17,
  -0.31 for sigma_w = 0.1, 0.3, 1, 2, 3). For read-out dropout, roughly, with q at each beta. For
  rescaling (m = 6-30: -0.012 to -0.015), prune-and-duplicate (s = 0-3: -0.005 to -0.006) and the
  rate penalty (-0.023 at 0.03, -0.021 at 0.1), no. So the abstract states no trade-off with strength.
- Block b (sets not continued to 200,000 iterations; no held-out R^2) adds rate-penalty,
  prune-and-duplicate and read-out-dropout sets, every one with fold above 1 (1.20-4.92), and no
  rescaling set.
- Rate penalty on 3BFF: every network at 75,000 iterations has more active units than any control
  (Results 5), although the number oscillates during training between about the control number and
  N (Fig. 5d, Extended Data Fig. 8). "Each technique increases the number" is true for it at the
  75,000-iteration read-out.

**Wording adopted (abstract):** "With most of the parameter values we tested, each technique
increases the number of active units, and the networks' task performance is slightly lower than
that of networks trained without a technique." The Introduction gives the count (48 of 55; all seven
exceptions are rescaling sets) and states that the rate-penalty value was fixed before measurement
and the other four values after (Methods, How the parameter values in Fig. 5 were chosen).

Tally script (stdlib only, parses block a of the table): kept with the PR at
`~/Documents/Projects/paper_push/scratchpad/r12-front/tested_values.py`; the table above is its output.
