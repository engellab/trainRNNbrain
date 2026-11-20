# Maze long warm continuation submitted on Della (Phase B)

Follows `2026-11-18_0745_rules-maze-long-warm-continuation.md` (rules, unchanged) and the Phase A result
`2026-11-18_2018_maze-warm-start-test-result.md` (R1 and R4 pass, so Phase B runs as its rules say). Submitted after
all nine warm stage 12 tasks (17851662) had ended, from the Mac at 20:17 Pacific (23:17 Eastern), 18 Nov:

```
ssh della 'bash -s' < slurm/rl_maze_warmlong_submit_della.sh
```

**Cluster: Della** (gpu-short, `--constraint=nomig`). Spock was checked first: it has no checkout of the pinned
trainer, the nine update-24,000 parents and their Adam states are on Della, and its GPUs (A100 and L40S) differ from
those of the earlier Maze stages; Della had started every Phase A task within 30 minutes of submission. Nothing was
written on Spock.

Every check in the script passed:

- Code: `~/trainRNNbrain_mazefinite` at 8c6b36e, clean; launcher taken from origin/wp/maze-plateau 7ebb7f7, md5
  b857fb77e058310d47316dd709ba4ddb. Warm stage 12 (17851662) no longer in the queue.
- The nine parents (`trained_RNNs_rl_maze_warmtest/Maze_suite/N=<N>_act=relu_warm_cont12/s<seed>`): parameter file,
  AdamState and a progress record ending at update 24,000, every Adam moment finite. Adam step counts (the same for
  the three parameters):

| Parent | Adam steps at update 24,000 | at update 20,000 (Phase A entry) | added over 4,000 updates |
|---|---|---|---|
| N=500 s1 | 47,632 | 15,632 | 32,000 |
| N=500 s2 | 46,880 | 15,088 | 31,792 |
| N=500 s3 | 42,480 | 11,272 | 31,208 |
| N=1000 s1 | 47,997 | 16,000 | 31,997 |
| N=1000 s2 | 47,996 | 16,000 | 31,996 |
| N=1000 s3 | 47,991 | 15,991 | 32,000 |
| N=1000 s4 | 46,214 | 16,000 | 30,214 |
| N=1000 s5 | 47,488 | 16,000 | 31,488 |
| N=1000 s6 | 48,000 | 16,000 | 32,000 |

  4,000 updates x 8 = 32,000 steps; the shortfalls are steps the trainer skipped (non-finite loss or gradient) or
  replayed after a fallback restore during the warm arm, largest in N=1000 s4 (1,786 steps, about 220 updates).
  The step count was carried over across the warm boundary at 22,000 in all nine (no reset to 0).
- The output tree `trained_RNNs_rl_maze_warmlong` held no cell.
- Login-node launcher checks (srun stubbed, 2 updates per stage, under a new /tmp folder): stage 13 from the real
  parent of index 7 (N=1000 s1) restores Adam and records updates 24,001-24,002, steps 47,997 -> 48,013 at the
  constant rate 0.000464; stage 14 from that check records 24,003-24,004, steps 48,013 -> 48,029; stage 13 from
  index 4 (N=500 s1) restores Adam and records 24,001-24,002.

Jobs (`--time=23:00:00`, array 4-9, 26-28, job name MazeWarmLong, `EXTRA_UPDATES=5000`, logs
`~/trainRNNbrain_logs/MazeWarmLong.<job>_<task>.out`):

| Stage | Job | Updates |
|---|---|---|
| 13 (warm from warm stage 12) | 17897469 | 24,001-29,000 |
| 14 (warm from stage 13) | 17897470 (`aftercorr:17897469`) | 29,001-34,000 |

18 stage tasks of 5,000 updates. At the 5.0-7.1 h per 2,000 updates of Phase A, a stage takes 12.5-17.7 h, so
225-320 GPU-h; with Phase A's 147 GPU-h, 372-467 GPU-h in all (budget about 600). The scheduler's start estimates
for stage 13 were all before 19 Nov 00:15 Eastern, so stage 14 should end between Thursday afternoon and Friday
midday Eastern, before the reading deadline (Friday 22:00 Eastern). The readings P1-P5 are as written in the rules
entry; the result goes in a new entry, and each recovery, if any, in its own entry.
