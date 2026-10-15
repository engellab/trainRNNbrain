## 2026-10-01 13:05 — DMTS re-run at a 5 tau delay

**Why.** At a 7 tau delay the task is not reliably solvable, so the DMTS column of Figure 2 compares
arms that mostly failed rather than arms that differ: control solved 19 of 28 seeds, the frm+rws pair
7 of 9, prune-and-duplicate 2 of 9, mute dropout 0 of 9. An intervention cannot be shown to help or
hurt on a task the control itself only half-solves. 5 tau keeps the delay long enough to require
holding the sample (50 integration steps at tau = 10) while landing inside what the architecture
trains to.

**The task.** `configs/task/DMTS_d5.yaml`: T = 190, sample on 20 / off 40, match on 90 / off 110,
decision from 130, 10-step random jitter of stimulus onset, 4 inputs (2 sensory, 1 tonic, 1 decision
cue), 2 outputs. Everything except the delay is the 7 tau config unchanged. Smoke-tested locally
before submitting: inputs (4, 190, 1024), targets (2, 190, 1024), mask 180 steps, tonic channel mean
1.00, cue on from step 130, stimulus channels max 1.0.

**The version trap, and how it was avoided.** `~/trainRNNbrain_papergrid` is pinned at `852a17f`,
and the development branch has since changed `RNN_torch.py` (105 lines) and `Trainer.py` (589 lines).
Checking out the development branch to pick up the new config would have made the 5 tau runs differ
from the 7 tau runs in the trainer as well as in the delay — two changes, one comparison. Instead:
branch `papergrid-dmts5` = `852a17f` plus the config file alone, commit `c35d71b`,
`git diff --stat 852a17f HEAD` = 1 file changed, 47 insertions. The worktree reports clean, so the
provenance check passes.

**Submitted.** Job 14830790, 27 tasks: 3 arms (control, prune-and-duplicate, mute dropout) x
N = 500/1000/2000 x 3 seeds, 150,000 iterations, 24 h limit — the four 7 tau controls hit an 8 h wall
previously, and the control is the arm that struggles. `SWEEP=paper_grid_d5` sends output to
`DMTS_paper_grid_d5/`, so the 7 tau cells survive for comparison. First task's echo confirms
`dmts N=500 arm=control`, T = 190, sample off 40, match on 90.

**Still open.** The CDDM reproduction gap: the stored r2 of the first `N=1000 arm=control` network is
0.8653 and the rebuild reads about 0.06 lower, systematically, on 43 of 45 gate failures — all CDDM.
The first diagnostic died on a wrong assumption of mine, that the CDDM task config carries a
`batch_size`; it does not, because `TaskCDDM.get_batch` enumerates the whole 15 x 15 x 2 coherence
grid deterministically. That rules out batch sampling as the cause and leaves the three noise
channels, the synaptic noise and the scoring mask. Job 14831010 scores two networks under each.

