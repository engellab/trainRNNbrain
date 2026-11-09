## 2026-11-08 17:58 — Round 19 penalty reruns: the same "one stage, report both points" policy (amends 2026-11-08_0940_round19-penalty-reruns-rules.md)

**The policy.** Pavel's choice for the Maze settings networks (`2026-11-07_1111_round19-settings-stage2-rules.md`)
now applies to the `freeze_after_end=true` reruns as well:
- any rerun network not learned at update 2,000 continues one stage, to 4,000;
- the other seeds of its arm continue too, so the arm keeps one budget;
- each arm is reported at 2,000 and at 4,000, with the read-out and labels of the part 2 rules;
- the 2,000 read stays the stated one.
This covers the rate penalty with the sparsity penalty (frm0.1_rws0.05) now. It will cover frm0.3 and the
metabolic arms if they need it when they finish; each such submission is listed in a later entry with its job
id.

**The state at 2,000.** Read from the copied fz tree with `common.maze_learned_index` and issue #5's criterion
(`fig_ed_maze_interventions.rule_verdict`): every seed's count above 449, mean success within 0.05 of 0.898,
no seed below 0.9.

| Arm (freeze_after_end=true) | Per seed: success, active units (learned update) | Verdict at 2,000 | Before the fix |
|---|---|---|---|
| frm0.03 | 1.00, 998 (770); 0.97, 997 (1,110); 1.00, 952 (460) | works | works |
| frm0.1 | 1.00, 998 (560); 0.99, 1,000 (760); 1.00, 997 (1,390) | works | fails (seed 1 at 0.63) |
| frm0.1_rws0.05 | 0.39, 1,000; 0.68, 1,000; 0.34, 1,000 (none learned) | fails: mean success 0.47, 0.43 below the controls' 0.898; all three seeds below 0.9 | fails (seeds at 0.43, 0.95) |

frm0.3 is at updates 1,700-1,800 and the metabolic arms are still running; they are read when they finish.

**Submitted under this entry.** frm0.1_rws0.05 seeds 1-3, updates 2,001-4,000. None learned at 2,000, so the
whole arm continues. The settings are interventions indices 13-15, `PARENT_ROOT=<fz tree>/Maze_interventions`,
`SAVE_TO=<fz tree>`, `FREEZE=true` (the parents' setting; the launcher checks it) and `INIT_ADAM=false`. The run
uses the reruns' pinned worktree, `~/trainRNNbrain_r19fz` (fd564f2), on Della, about 3 × 5.3 = 16 GPU-h.

**The sparsity penalty alone (rws0.05) needs no rerun.** `TrainerPPO.rec_weights_sparsity_penalty` is computed
from `self.RNN.W_rec` alone: the effective number of synapses per row, (Σ|W|)² / ΣW². No state enters it, so
post-episode steps never did, and `freeze_after_end` does not change it. The same holds for the recurrent-weight
magnitude penalty (`rec_weights_magnitude_penalty`, on W_rec). In a run with either penalty and no rate or
metabolic penalty, the switch changes only whether post-episode states can become non-finite. That is how
rwm0.2's stops arose (`2026-11-08_0943`), and it is a guard matter, not a penalty definition. Only the rate and
metabolic penalties averaged over states, and those arms are the ones rerun.
