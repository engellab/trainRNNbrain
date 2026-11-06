# Maze package 3 submitted: cap, two penalties, three optimisers, two redraw modes (Spock 7389398, 42 jobs)

**Date:** 2026-11-05 23:10
**Issue:** #6
**Code:** 64c3293 (TrainerPPO: inp_weight_cap, lambda_rwm, lambda_met, the optimiser switch in
run_rl.py, reinit_mode random and zero_out; 33 of 33 maze tests), entry 2026-11-05_2306.

Submitted from the Spock worktree ~/trainRNNbrain_rlmaze8 at 19ba724: interventions launcher
indices 73-114, N = 1,000, 3 seeds per arm: cap 0.3; lambda_rwm 0.05, 0.2; lambda_met 0.01, 0.1,
1, 10; AdamW (wd 1e-6); SGD at 0.03, 0.1, 0.3, 1; redraw (random) and zero_out at the supervised
replace-variants cadence. About 230 GPU-h; the 12 h limit holds at 5.5 h per run.

With this the whole programme of docs/interventions_matrix.md is in the queue or done, bar the
threshold lift (dropped by Pavel). Queue at submission: 33 running, 16 + 42 pending on Spock;
Della holds package 1 (16955456, 16955457), the CDDM cells (16957418, 16957419), the 3BFF cells
(16957556 to 16957562) and the N = 4,000 chains (16907630, 16955490, 16955491).

Read-out, pre-registered as in the earlier maze entries: at update 2,000 against the two learned
N = 1,000 controls (437.5 active, success 0.984); a setting "changes the count" when all three
seeds lie outside the controls' range on the same side; a rule "works" when every seed has more
active units than the largest control (449) and the mean success is within 0.05 of the controls'.
The optimiser question is whether the maze reproduces 3BFF's pattern (AdamW and no weight decay
keep more units at no loss; SGD keeps more only at lower performance).
