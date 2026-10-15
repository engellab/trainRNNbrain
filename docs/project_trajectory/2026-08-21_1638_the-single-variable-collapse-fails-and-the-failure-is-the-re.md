## 2026-08-21 16:38 — the single-variable collapse FAILS, and the failure is the result

New script: `experiments_and_analysis/flipflop_collapse.py` → `img/internal_figures/flipflop_collapse.png`.

**Hypothesis under test.** The (N, k) contour map showed iso-M contours running diagonally, which
would mean silencing depends not on N and k separately but on one combination `x = N / D(k)` —
capacity per unit of task demand. Two candidate demands from the task's own structure: `D = k^alpha`
(readout dimension) and `D = 2^(beta k)` (attractor count). Tested by whether a 4-parameter model in
`log x` fits as well as a 15-parameter SATURATED model with a free mean per cell.

**It fails, at every matched performance level.** The fitted exponent runs to the search boundary
and stays there when the grid is widened from 5 to 20 (power) and 2 to 8 (exponential); the bootstrap
CIs span nearly the whole grid. The diagnosis is in the variance column:

| reading (scale-free) | N only | k only | COLLAPSE | saturated |
|---|---|---|---|---|
| endpoint (300k) | 0.260 | 0.452 | 0.746 | 0.840 |
| R² = 0.949 | 0.040 | **0.820** | 0.810 | 0.916 |
| R² = 0.970 | 0.009 | **0.704** | 0.721 | 0.853 |
| R² = 0.980 | 0.142 | **0.856** | 0.899 | 0.950 |

At R² = 0.949 the "collapse" explains LESS than k alone. The exponent diverges because that makes
`N/D(k)` a pure function of k — the two-variable model silently degenerates into the one-variable
k-only model. A degeneracy guard was added so this reports as DEGENERATE rather than passing the
F-test, which it otherwise does: "k alone matches 15 cell means" is true but is a different and much
weaker claim than a capacity-vs-demand collapse.

**What this actually establishes, which is more interesting than the collapse would have been.**
At matched performance, **task complexity dominates and network size carries almost no independent
weight**: N alone explains 0.9–14% of the variance in silent fraction, k alone explains 65–86%.

The apparent capacity-vs-demand trade-off in the contour map was an ENDPOINT artifact. Only at the
endpoint does N carry real weight (0.260 scale-free, 0.530 hard) — and the endpoint is the reading
confounded by training depth, since reaching a fixed 300k means very different depths for different
cells. Read at matched performance, the diagonal structure disappears.

⚠️ **Revises the framing offered two entries ago.** "Silencing is a property of both the network and
the task, via too many units and too simple a task" is not symmetric. Both directions are real, but
at matched performance the task term dominates and the size term is small. And training duration
remains the third factor, unbounded and omitted from that summary entirely.

⚠️ Does not contradict the earlier finding that silencing rises monotonically with N at every k: a
monotone but SMALL effect is exactly what "N explains 1–14% of variance" describes. The effects
differ by an order of magnitude in size, not in sign.

**Consequence for N=5000.** The collapse was the stated gate: a collapse holding over three N values
would have made a fourth an out-of-sample test worth ~710 GPU-h. It does not hold, and N is the axis
that turns out to matter least at matched performance. So N=5000 on the flip-flop is NOT justified —
it would buy better resolution on the weaker of the two factors.

---

