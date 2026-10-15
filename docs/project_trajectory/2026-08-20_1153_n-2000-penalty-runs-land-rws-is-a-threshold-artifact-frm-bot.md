## 2026-08-20 11:53 — N=2000 penalty runs land: `rws` is a threshold artifact, `frm`/`both` improve task loss

All nine N=2000 penalty nets (rws/frm/both × 3 seeds, 150k iterations) completed on Spock
(`5670492`/`5670493`) and are synced to `data/trained_RNNs/CDDM_std_g0_penalties`. N=5000 still
running, 9 tasks, ~11.5 h elapsed.

New script: `experiments_and_analysis/penalty_matched.py` → `img/internal_figures/penalty_matched.png`.
It exists because the naive endpoint table has **two confounds**, both of which reverse conclusions.

### Confound 1: the recorded loss column is not the task loss

`TrainLosses.json` holds what the optimizer descends — task + λ·penalty, evaluated with **noise on**.
The noise floor is ~65% of that number. At N=2000 it reads:

| | `none` | `rws` | `frm` | `both` |
|---|---|---|---|---|
| recorded `train_losses` | 0.02223 | **0.02432** (worst) | 0.02348 | 0.02368 |
| noise-free task loss | 0.00888 | 0.00713 | 0.00760 | **0.00578** (best) |

The ordering **completely inverts**. `rws` looks worst and is second best; `both` looks worst-but-one
and is best. This is the same total-vs-task trap that invalidated the first penalty table on
2026-08-19, and it propagated into `paper.md` §2 as a bogus "`frm` costs +5% in loss" claim, written
this morning and now retracted. **Rule: never compare `TrainLosses.json` across penalty conditions.**
`penalty_matched.py` evaluates the noise-free masked MSE of the final weights with `RNN_numpy` on a
shared batch, plus a held-out batch of interleaved coherence midpoints.

### Confound 2: the budgets differ

The `none` baseline comes from the drift sweep (200k at N=500/1000, **300k** at N=2000); the penalty
sweep ran 200k/200k/**150k**. Silencing never settles, so a baseline given 2× the budget is
guaranteed to look worse. Fixed by reading `none`'s silent count from its participation trace at the
iteration where the matching penalty run stopped. At N=2000 this moves `none` from 83.0% (its own
endpoint) to 78.9% (matched) — not enough to change any conclusion, but it had to be checked.

### RESULT 1: `rws`'s apparent rescue is a hard-threshold artifact

At matched budget, N=2000:

| | `none` | `rws` | `frm` | `both` |
|---|---|---|---|---|
| silent, hard (`p<1e-6`) | 78.9% | **60.5%** | 0.0% | 0.0% |
| silent, scale-free (`p<0.05·q95`) | 80.9% | **86.0%** | 0.0% | 0.0% |

`rws` lifts units just across 1e-6 while leaving them far below any functional threshold — it turns
hard zeros into a long tail of tiny-but-nonzero rates. The hard/scale-free gap grows with N:
**11.9 / 20.7 / 25.5 pp** at N = 500 / 1000 / 2000. `none`'s two criteria agree to within 2 pp
everywhere.

This also explains the late-training "resurrection" visible in `penalty_comparison.png` panel (a):
under `rws` the hard-silent count peaks and falls back — 75.5% → 60.5% at N=2000, peaking at
iteration 76k — with the drop scaling as **3.2 / 10.2 / 15.0 pp** at N = 500 / 1000 / 2000 and the
peak arriving *earlier* at larger N (183k → 135k → 76k). It is absent in `none` (≤1.0 pp drop,
monotone). I flagged this as a genuine size-dependent resurrection before checking the second
criterion; it is not. No units are revived. **Anyone reporting a single hard threshold would publish
this as a partial rescue.** Strongest argument yet for the two-criterion policy.

### RESULT 2: `frm`/`both` cost nothing, and at N=2000 they beat the baseline

Noise-free masked MSE, final weights, shared batch (train / held-out):

| N | `none` | `rws` | `frm` | `both` |
|---|---|---|---|---|
| 500 | 0.00851 / 0.01598 | 0.00835 / 0.01569 | 0.00980 / 0.01757 | 0.00844 / 0.01571 |
| 1000 | 0.00907 / 0.01664 | 0.00792 / 0.01532 | 0.00876 / 0.01628 | 0.00828 / 0.01509 |
| 2000 | 0.00888 / 0.01628 | 0.00713 / 0.01442 | 0.00760 / 0.01463 | **0.00578 / 0.01226** |

`frm` goes from 15% worse than `none` at N=500, to level at N=1000, to **14% better at N=2000**;
`both` from level to **35% better**. The benefit grows exactly where unpenalized silence is worst.
Held-out tracks train at a constant ~1.85× in every condition — no penalty trades train for
generalization.

⚠️ Caveat to carry: this is 3 seeds per cell and the N=500 `frm` cell is the only one that is worse,
so the trend rests on three points. N=5000 (running) is the test — if `both` keeps pulling ahead,
"activity regularization improves large-network training" becomes a claim in its own right rather
than a footnote to the silence result.

### Files touched

- `experiments_and_analysis/penalty_matched.py` — new
- `docs/paper.md` — §2 rewritten as §2.1 (threshold artifact) + §2.2 (task cost); `frm` bullet in §3
  corrected; two entries added to the retracted-claims appendix

---

