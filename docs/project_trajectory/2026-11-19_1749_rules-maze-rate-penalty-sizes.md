## 2026-11-19 17:49 — Rules, gap C: B2, the rate-penalty size test, on Maze (N = 500, 1,000, 2,000)

Written and committed before any run of this gap is submitted. Ranking and cluster choice:
`2026-11-19_1749_claim-gaps-ranking.md`. The CDDM form of the same test: `2026-11-19_1749_rules-cddm-rate-penalty-sizes.md`.

**Claim at stake.** The rate penalty alone fails the stable-training test across sizes (B2), shown on 3BFF only.

**Terms.** *Rate penalty*: λ_frm = 0.1 (`trainer.lambda_frm=0.1`; B2's value and the value run on all three
tasks). *Learned point*, *active unit*, *probe*: as in the ranking entry and the gap A rules. *Skipped* and
*restored* are fields of each update's record in `*Progress.json`: `skipped` counts the minibatch steps whose
loss was not finite (the step is skipped), `restored` counts rollouts discarded and weights rolled back by the
rollback guard. A run *stops* when the trainer raises "the rollout was not finite ... the run is stopped".

### Networks

| N | Rate penalty networks | Controls |
|---|---|---|
| 500 | new: arm `frm0.1` (interventions indices 7-9), seeds 1-3, 2,000 updates + 2,000 with the Adam state restored | as gap A: `Maze_suite/N=500_act=relu` s1-3 with `_cont` |
| 1,000 | the three runs of `trained_RNNs_rl_maze_fz/Maze_interventions/frm0.1/s1-3` (Della 17107385, `freeze_after_end` true), continued 2,000 → 4,000 with their Adam state restored | `Maze_suite/N=1000_act=relu` s1-3 with `_cont` |
| 2,000 | new: as at N = 500 | `Maze_suite/N=2000_act=relu` s1-3 with `_cont` |

Jobs, output trees and the known differences from the controls are those of the gap A rules, with indices 7-9
in place of 52-54 (`SAVE_TO=..._cg_N<N>`; both arms share each size's tree, in their own arm folders).

### B2's quantities in their Maze form

B2 reads 3BFF at the floor crossing; Maze's counterpart is the learned point, and B2's R² tolerance of 0.02 has
the paper's Maze tolerance, 0.05 in success, as its counterpart (decision dead-relu-paper#185).
- **Stable**: the run reaches update 4,000 without stopping, has a learned point by update 4,000, and fewer
  than 5% of its 4,000 updates have `skipped` > 0 or `restored` > 0 (B2: a floor read, fewer than 5% of updates
  skipped, a finite loss). A stopped run counts as not stable and is not resumed for this test.
- **Multiplier**: mean active count at the learned point over the same-size controls' mean at their learned
  points.
- **Performance cost**: mean probe success at update 4,000 minus the same-size controls' mean at update 4,000.

### Rule (B2's)

- **Supports** (the rate penalty trains stably across sizes on Maze) if at every N (500, 1,000, 2,000): at least
  2 of 3 runs are stable; the multiplier is at least 1.5; and the mean success at update 4,000 is no more than
  0.05 below the controls'.
- **Falsifies** (the rate penalty fails B2 on Maze) if at any N ≥ 1,000: fewer than 2 of 3 runs are stable, or the
  success cost exceeds 0.05, or the multiplier is below synaptic noise's at the same N. Synaptic noise
  (σ_w = 1.0) exists on Maze only at N = 1,000 (`trained_RNNs_rl_maze/Maze_interventions/sigmaw1.0` s1-3, read at
  its learned point by the same script against the same controls); at 500 and 2,000 the clause cannot be
  evaluated and is reported as such.
- **Neither:** numbers, no verdict. A failure at N = 500 alone keeps "supports" from holding but does not falsify.
- Reported without a threshold: the active count at update 4,000, the learned update, and Σ skipped and
  Σ restored per run.

### What each outcome does to the paper

- **Falsifies:** the stable-training failure holds on Maze as well as 3BFF; with gap B, the results entry states
  the claim on every task where it holds.
- **Supports:** the failure is specific to 3BFF on this evidence (with gap B, on two of three tasks it passes).
- **Neither:** the claim stays scoped; numbers reported.

N = 4,000 is not run (see the ranking entry).
