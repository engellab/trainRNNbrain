# Maze warm-start test: continuity check exact, launcher ready, not submitted (Della unreachable)

Follows `2026-11-17_2249_rules-maze-warm-start-test.md` (rules; nothing in them is changed here). Della timed out
from the Mac (`ssh: connect to host della9.princeton.edu port 22: Operation timed out`, at 23:01 and 23:02), so
nothing was submitted. All local work is done and committed (cd4e13a, pushed to origin/main). One command remains:

```
ssh della 'bash -s' < slurm/rl_maze_warmtest_submit_della.sh
```

## The continuity check ("Before submission" in the rules): exact

Script `trainRNNbrain/experiments_and_analysis/maze_warm_start_continuity.py`, run on the Mac CPU against an export
of the trainer commit 8c6b36e (`git archive 8c6b36e | tar -x`), through the production entry point
`run_experiment.py` with no change to the trainer. The script wraps `TrainerPPO.run_training` from outside to dump
the optimizer state the trainer starts with (after `run_rl` has loaded the AdamState and set the rate) and to save
or load every random-number state. Network N = 50, 4 episodes of 6 decisions per update, 4 probe episodes,
warm-up 2 updates (so that the unsplit run has finished its ramp before the split). Four runs: A, unsplit, 6
updates; B1, the first half, 3 updates; B2, the second half from B1 with `paths.init_adam=true`; B2r, B2 with every
random-number state set to B1's at the end of its training. Pass conditions were written into the script before
the first run.

| Condition | Result |
|---|---|
| C1: optimizer state at the start of B2 equals B1's saved AdamState (every tensor, `torch.equal`) | equal; step count 24 for each of the 3 parameters in both; B2 starts at the base rate 0.00126 |
| C2: every update of B2 uses the base rate (no ramp) | B2: 0.00126 at updates 4, 5, 6. B1: 0.00063, then 0.00126 (its ramp). A repeated ramp would give 0.00063 at update 4 |
| C3: the update clock continues | B2 records updates 4-6 and probes at 4-6; its saved config has `iter_offset: 3` |
| C4: Adam's step count continues | 24 at the end of B1, 48 at the end of B2 (24 + 3 updates x 8 optimizer steps, none skipped), 48 at the end of A. A lost step count would give 24 |
| C5: with the random states restored, the split run equals the unsplit run | B2r is bitwise identical to A's updates 4-6: every array of the parameter file, every AdamState tensor and step count, every progress and probe record |
| C6 (descriptive): what differs without the restore | the random states at the start of B2 equal those at the start of B1, for all five sources; B2's final W_rec and W_out differ from A's |

So the warm start continues the optimizer exactly (moments, step count, rate, clock), and the only difference from
an unsplit run is in the random numbers drawn after the split.

**The random streams after a split (from the code, confirmed by C6).** `run_rl` seeds python, numpy and torch's
global stream (the action noise) from `seed`, the RNN builds its generator (the RNN noise) from `seed`, and the task
builds its drop generator from `seed + 1`, at the start of every run. A continuation therefore draws the same
random numbers as the first updates of the chain's first run. Three consequences:

- every stage of the Maze long training (10 stages of 2,000 updates for the never-stopped networks) drew the same
  drops, RNN noise and action noise at its k-th update as every other stage, as long as no rollout was replayed
  (a replay draws extra numbers and shifts the stream);
- the warm and cold stage 11 of one network draw the same random numbers at the same update (the drops are
  identical while no rollout is replayed; the network weights differ after the first update, so the episodes
  differ from then on);
- the probe is unaffected (its own generator, fixed seed `probe_seed`).

Also not carried over by a warm start (from the code, inert in this test): the non-finite-recovery snapshots
(the list starts with the state at the start of the stage), the rescale rule's state and the dropout EMA (both off).

## Launcher and submit script

- `slurm/rl_maze_warmtest_della.slurm` (gitignored, added with `-f`): `slurm/rl_maze_resume_della.slurm` with job
  name MazeWarm, array 4-9 and 26-28, outputs in the new tree `~/trainRNNbrain/data/trained_RNNs_rl_maze_warmtest`,
  cells `N=<N>_act=relu_<arm>_cont<stage>`, `ARM=warm|cold` setting `paths.init_adam` (true for warm), explicit
  parents per index (stage 11 from the update-20,000 stage; warm stage 12 from warm stage 11), and a refusal if the
  parent's last recorded update is not 20,000 + (STAGE - 11) x EXTRA_UPDATES. Pinned to 8c6b36e; refuses a dirty
  checkout and an existing run folder. md5 12d0a903ad27507977a8f29c214b3f90.
- `slurm/rl_maze_warmtest_submit_della.sh`: one command, stops at the first failure. (1) The existing checkout
  `~/trainRNNbrain_mazefinite` must be at 8c6b36e; it writes the launcher from origin/main into it and checks the
  checkout is clean. (2) It reads the nine parents and prints each one's last update and Adam step counts. It
  refuses to submit if a file is missing, if the last update is not 20,000, or if any Adam moment is not finite;
  three earlier stops had NaN moments. (3) It refuses if the new tree already holds a cell. (4) It runs four
  login-node launcher checks with srun stubbed (2 updates, tiny episodes, output in a new temporary folder):
  index 7 warm 11, then warm 12, then cold 11, and index 4 warm 11. It checks the update clocks and whether Adam
  was restored. It also checks that the warm step counts continue the real parent's (parent + 16, then + 16
  again), that the warm rate is constant, and that the cold run restarts at step 16 with the ramp. (5) It submits
  three arrays: warm stage 11, warm stage 12 (`--dependency=aftercorr:<warm 11>`), and cold stage 11, all with
  `--time=12:00:00`.

Local checks of the launcher (the Mac, a scratch worktree at 8c6b36e, module/conda/srun stubbed, copies of the
real parents of index 7 and index 4; their AdamState files exist only on Della, so each parent copy received a
**stand-in** AdamState, the 2-update state of the local cold run, step 16, labelled in the folder):

| Check | Result |
|---|---|
| cold 11, index 7 and index 4 | parent update 20,000; clock 20,001-20,002; no Adam restore; rate 1/50 and 2/50 of base (ramp) |
| warm 11, index 7 and index 4 | Adam restored; clock 20,001-20,002; rate equal to base (0.000464 at N = 1,000) at both updates |
| warm 12, index 7 and index 4 | parent is warm 11 at update 20,002; Adam restored; clock 20,003-20,004 |
| refusals | existing run folder; cold stage 12; index 10; INIT_ADAM=true with ARM=cold; warm 12 without warm 11; parent at 20,002 when 20,003 is required |
| saved configs, warm 11 vs cold 11 | differ only in `paths.init_adam` and `experiment_tag` |

The whole submit script was also run on the Mac with a fake HOME holding copies of all nine parents (stand-in
AdamStates) and sbatch stubbed: every step passed (warm steps 16 -> 32 -> 48, cold 0 -> 16), and the three sbatch
calls carried ARM/STAGE warm 11, warm 12 with `aftercorr`, cold 11, `EXTRA_UPDATES=2000`, INIT_ADAM unset.

## Cost and expected timing

27 stage tasks: 9 warm 11, 9 warm 12, 9 cold 11. The rules estimate about 160 GPU-h, at 5-7 GPU-h per stage. The
18 stage-11 tasks fit in gpu-short at once (44 GPUs per user). Each warm stage 12 starts when its warm stage 11
ends, so the test finishes about 12-14 h after the stage-11 tasks start, plus queue time.

## Reading the two arms

`common.maze_load_runs` copes with a parent that has two children (tested on a synthetic root laid out as below,
with the Della paths in `paths.init_from`). It returns two records per network with the same N, activation and
seed: one ending in `_warm_cont12` (chain: parent, warm 11, warm 12; updates 18,001-24,000) and one ending in
`_cold_cont11` (chain: parent, cold 11; updates 18,001-22,000). The reading script tells the arms apart by
`"_warm_" in r["folder"]` or `"_cold_" in r["folder"]`. The loader prints a warning that the parent's own parent
is not on disk; the records still start at update 18,001, which covers the noise window 18,250-19,750.

**Do not copy these cells into `data/trained_RNNs_rl_maze/Maze_suite`.** There, the cont10 parents would become
"continued" folders, and every reading on that root, `maze_long_read_20000.py` among them, would change. Use a
new local root that holds the new cells and links to the nine parents:

```
W=~/Documents/GitHub/trainRNNbrain/data/trained_RNNs_rl_maze_warmtest/Maze_suite
M=~/Documents/GitHub/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite
mkdir -p "$W"
rsync -a --exclude='*AdamState_Maze.pt' --exclude='*.png' \
  della:trainRNNbrain/data/trained_RNNs_rl_maze_warmtest/Maze_suite/ "$W/"
cd "$W"
for c in N=1000_act=relu_cont10/s1 N=1000_act=relu_cont10/s3 N=1000_act=relu_cont10/s4 N=1000_act=relu_cont10/s5 \
         N=1000_act=relu_resume_cont10/s2 N=1000_act=relu_resume_cont10/s6 \
         N=500_act=relu_resume_cont10/s1 N=500_act=relu_resume_cont10/s2 N=500_act=relu_resume_cont10/s3; do
  mkdir -p "$(dirname "$c")"; [ -e "$c" ] || ln -s "$M/$c" "$c"
done
```

For the reading script: each stage probes at its first update as well as every 10th. The first five probes after a
boundary U are therefore at U+1, U+10, U+20, U+30 and U+40, and the last five up to U are at U-40 to U.

## Not tested

- Nothing ran on Della. The real parents' AdamState files have not been opened: their step counts (expected
  about 16,000 = 8 x 2,000, since each cont10 stage restarted Adam) and their finiteness are checked by the
  submit script before it submits.
- The loading of a CUDA-saved AdamState onto a GPU, and torch 2.7 (Della) as opposed to torch 2.10 (Mac). The
  login-node checks in the submit script load the real files on CPU with Della's torch.
- The continuity check ran at N = 50 on CPU, not at N = 500 or 1,000 on GPU. The mechanism does not depend on N:
  the same `load_state_dict` call and the same `adam_restored` switch.
