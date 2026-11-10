## 2026-11-09 19:42 — Correction to `2026-11-09_1937_n250-n100maze-submitted.md`: the N = 100 Maze jobs are Spock 7412726 (stage 1) and 7412727 (stage 2)

**What went wrong.** The Maze arrays named in the submission entry, 7412713 (stage 1) and 7412714 (stage 2), did no
training. All four stage-1 tasks exited after 2-3 s with code 1. The suite launcher's `REPO` defaults to
`~/trainRNNbrain_rlmaze`. I submitted with `--chdir=$HOME/trainRNNbrain_rlmaze9` but did not set `REPO`, so the
launcher's import check found the package in `..._rlmaze9` while `REPO` named `..._rlmaze` and stopped with
"FATAL: trainRNNbrain resolves to /mnt/cup/people/pt1290/trainRNNbrain_rlmaze9, expected
/mnt/cup/people/pt1290/trainRNNbrain_rlmaze". Slurm then cancelled the stage-2 array, whose dependency could no
longer be met. No output folder was created (`Maze_suite/N=100_*` does not exist on Spock). My local smoke test
had set `REPO` explicitly, so it did not test the default.

**Resubmitted** from the same worktree (`~/trainRNNbrain_rlmaze9`, detached at 6c689b4, clean), with `REPO`
exported for both submissions:

```
export REPO=$HOME/trainRNNbrain_rlmaze9
sbatch --chdir=$REPO --array=32-35 --time=12:00:00 slurm/rl_maze_suite_spock.slurm                    # 7412726
STAGE=2 PARENT_ROOT=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite EXTRA_UPDATES=2000 INIT_ADAM=false \
  sbatch --chdir=$REPO --array=32-35 --time=12:00:00 --dependency=aftercorr:7412726 slurm/rl_maze_suite_spock.slurm   # 7412727
```

All four stage-1 tasks started at once; task 32's log reads `code: /mnt/cup/people/pt1290/trainRNNbrain_rlmaze9 @
6c689b4b` and "Using cuda for RNN!". The 3BFF and CDDM array 7412712 is not affected: its launcher's default
`REPO` is the worktree it runs from, and its logs print `code: ... @ 63c6c976 (git)`.
