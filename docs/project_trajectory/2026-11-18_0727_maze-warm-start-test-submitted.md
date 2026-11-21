# Maze warm-start test submitted on Della (Phase A)

Follows `2026-11-17_2249_rules-maze-warm-start-test.md` (rules, unchanged) and
`2026-11-17_2302_maze-warm-start-test-prepared.md` (launcher and submit script). Della was reachable on 18 Nov;
the prepared one-command submission ran as written, from the Mac at 07:27 Pacific (10:27 Eastern):

```
ssh della 'bash -s' < slurm/rl_maze_warmtest_submit_della.sh
```

Every check in the script passed:

- Code: `~/trainRNNbrain_mazefinite` at 8c6b36e, clean; launcher taken from origin/main fa472ea, md5
  12d0a903ad27507977a8f29c214b3f90, the same as the committed file.
- Output tree `~/trainRNNbrain/data/trained_RNNs_rl_maze_warmtest` did not exist before submission.
- The nine parents: each has a parameter file, an AdamState and a progress record ending at update 20,000, and
  every Adam moment is finite. Adam step counts per parent (the same for the three parameters):

| Parent (tree) | Adam steps |
|---|---|
| N=500 s1 (resumed) | 15,632 |
| N=500 s2 (resumed) | 15,088 |
| N=500 s3 (resumed) | 11,272 |
| N=1000 s1 (long) | 16,000 |
| N=1000 s2 (resumed) | 16,000 |
| N=1000 s3 (long) | 15,991 |
| N=1000 s4 (long) | 16,000 |
| N=1000 s5 (long) | 16,000 |
| N=1000 s6 (resumed) | 16,000 |

  16,000 is 2,000 updates x 8 Adam steps, the full stage since Adam last restarted at update 18,000. The counts
  below 16,000 are steps the trainer skipped (non-finite losses or gradients) or replayed after a fallback
  restore; N=500 s3 is missing about 4,700 steps. Not investigated further: the warm arm continues whatever count
  the parent has.
- Login-node launcher checks (srun stubbed, 2 updates per stage): warm 11 from the real parent of index 7 restores
  Adam and records updates 20,001-20,002, with steps 16,000 -> 16,016 at the constant rate 0.000464; warm 12
  records 20,003-20,004, steps 16,016 -> 16,032; cold 11 records 20,001-20,002, steps 0 -> 16, with the ramp
  (rates 1/50 and 2/50 of the base rate); warm 11 from index 4 (resumed tree) restores Adam.

Jobs (all `--time=12:00:00`, array 4-9, 26-28, job name MazeWarm, logs
`~/trainRNNbrain_logs/MazeWarm.<job>_<task>.out`):

| Arm, stage | Job | Updates |
|---|---|---|
| warm 11 | 17851661 | 20,001-22,000 |
| warm 12 | 17851662 (`aftercorr:17851661`) | 22,001-24,000 |
| cold 11 | 17851663 | 20,001-22,000 |

27 stage tasks, about 160 GPU-hours. The readings R1-R5 are as written in the rules entry; the result goes in a
new entry.
