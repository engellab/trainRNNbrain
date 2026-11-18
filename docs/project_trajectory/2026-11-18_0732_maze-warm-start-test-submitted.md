# Maze warm-start test: submitted on Della (17851661-17851663)

Rules: `2026-11-17_2249_rules-maze-warm-start-test.md`. Preparation: `2026-11-17_2302_maze-warm-start-test-prepared.md`.

Submitted at 10:27 Eastern, 18 Nov, with `ssh della 'bash -s' < slurm/rl_maze_warmtest_submit_della.sh`. The run came from
the terminal, not from this session's heartbeat, which found the jobs already queued when Della became reachable from
the Mac again. The script stops at its first failed check, and it reached submission.

| Arm | Stage, updates | Array job | Dependency |
|---|---|---|---|
| warm (`INIT_ADAM=true`) | 11, 20,001-22,000 | 17851661 | none |
| warm | 12, 22,001-24,000 | 17851662 | aftercorr:17851661 |
| cold (`INIT_ADAM=false`) | 11, 20,001-22,000 | 17851663 | none |

Indices 4-9 and 26-28 in each array. Checkout `~/trainRNNbrain_mazefinite` at 8c6b36e; output tree
`~/trainRNNbrain/data/trained_RNNs_rl_maze_warmtest/Maze_suite/`.

**Login-node checks** (logs in `/tmp/pt1290_mazewarm_smoke_XEeIVl` on the login node):
- index 7, warm 11: Adam restored from `N=1000_act=relu_cont10/s1`; clock 20,001-20,002;
- index 7, warm 12: Adam restored from the warm stage 11 just written; clock 20,003-20,004;
- index 7, cold 11: no restore; clock 20,001-20,002;
- index 4, warm 11: Adam restored from `N=500_act=relu_resume_cont10/s1`; clock 20,001-20,002.

**Parents, read again afterwards (read only).** All nine end at update 20,000, and every Adam moment is finite. Adam's
step count is 16,000 (8 per update over 2,000 updates) in five networks, 15,991 in N = 1,000 seed 3 (9 minibatches skipped
for a non-finite loss), and 15,632, 15,088 and 11,272 in the resumed N = 500 seeds 1, 2 and 3. Each fallback restore of the
fixed trainer reloads the optimizer state saved with the snapshot, so the counter goes back to the snapshot's count.
This was not examined further. It affects only Adam's bias correction, which is near 1 at these step counts.
