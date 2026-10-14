# B2 and B3 on Spock: what was submitted, and the decisions made overnight

*Written 14 October 2026, 00:15 EDT, on branch `agent-b23`. The plan is `docs/notes/agent-bplan.md`
on `restore-oct5`. Pavel approved B2 full and B3 minimal for Spock; he was asleep when these jobs
went in, so every choice that departs from the plan is listed in §3 with its reason.*

## 1. What is running

**Code.** Worktree `~/trainRNNbrain_b23` on Spock, detached at `618fc9e` = `7739a60` plus one file,
`slurm/SilentReLU_remedy_grid_spock.slurm` (`git diff --stat 7739a60 618fc9e` shows only it). No
training code changed. The worktree was clean at submission, and the launcher refuses to run from a
dirty one.

**Paths.**
- Runs: `~/trainRNNbrain/data/trained_RNNs/NBitFlipFlop_remedy_grid/EqType=h_N=<N>_arm=<arm>/<run>/`
- Kept snapshots: `/mnt/cup/labs/tengel/home/tolmach/trainRNNbrain_b23_snapshots/NBitFlipFlop_remedy_grid/EqType=h_N=<N>_arm=<arm>/s<seed>/it<iteration>/`
  (LastParams and config in each)
- Logs: `~/trainRNNbrain_logs/<job name>.<job id>_<task>.out`

**Jobs**, submitted 14 Oct 00:11, every one 3 seeds (`seed=random`; each resolved seed is in the
run's saved config and in its snapshot folder name), 200,000 iterations, from scratch, in one job:

| Job | Name | Item | N | Arm (knobs) | Snapshots every | --time | --mem |
|---|---|---|---|---|---|---|---|
| 6929384 | B3pilot | B3 stage 0 | 1000 | `frm0.01` (λ_frm 0.01) | 500 | 36 h | 32G |
| 6929385 | B3pilot | B3 stage 0 | 1000 | `frm0.03` (λ_frm 0.03) | 500 | 36 h | 32G |
| 6929386 | B2frm | B2 | 2000 | `frm0.1` | 1000 | 60 h | 64G |
| 6929387 | B3fact | B3 stage 1 | 1000 | `sw1.0` (σ_w 1.0) | 500 | 40 h | 32G |
| 6929388 | B3fact | B3 stage 1 | 1000 | `rsc10` (rescale, target 10) | 500 | 40 h | 32G |
| 6929389 | B3fact | B3 stage 1 | 1000 | `sw1.0_rsc10` | 500 | 40 h | 32G |
| 6929390 | B2frm | B2 | 1000 | `frm0.1` | 500 | 36 h | 32G |
| 6929391 | B2frm | B2 | 500 | `frm0.1` | 500 | 24 h | 32G |

Any GPU type (`--gres=gpu:1`, partition `all`): A100-40G or L40S-46G. All 24 runs were pending at
00:12, behind 19 queued `B5opt` tasks and other users' jobs.

**Speed and cost.** Measured on Spock, not taken from Della. The current `B5opt` runs (N = 1000,
no remedy) take 0.21 s/it on L40S and 0.25 s/it on A100. The pre-standard rate-penalty runs
(`FFpenL`, 400k) took, median / slowest: 0.22 / 0.29 s/it at N = 500, 0.30 / 0.37 at N = 1000,
0.71 / 0.74 at N = 2000 on A100, with L40S 10-20% faster. Each time limit covers the slowest
measured rate plus at least 50%.

| Group | Runs | Hours per run (expected) | GPU-h |
|---|---|---|---|
| B2, N = 500 | 3 | ~12 | ~37 |
| B2, N = 1000 | 3 | ~17 | ~50 |
| B2, N = 2000 | 3 | ~40 | ~120 |
| B3 pilot | 6 | ~17 | ~100 |
| B3 noise, rescale, noise + rescale | 9 | ~17-18 | ~160 |
| **Submitted** | 24 | | **~465** |
| B3 rate cells, not yet submitted (§2) | 9 | ~18 | ~165 |

So B2 comes to ~205 GPU-h (the plan's budget was ~570, which included N = 4000) and B3 minimal to
~425 when complete (the plan's budget was ~390; Spock at N = 1000 runs about 15% slower than
Della's 0.26 s/it). At B5's speed, which is the faster end of the Spock range, B3 comes to ~370.

**Expected finish**, if each job starts between 03:00 and 12:00 on 14 Oct, which depends on when B5
releases GPUs: N = 500 on the evening of 14 Oct; N = 1000 (B2, pilot, factorial cells) from the
evening of 14 Oct to the morning of 15 Oct; N = 2000 from the evening of 15 Oct to the morning of
16 Oct.

**Disk.** A kept snapshot is ~2 MB at N = 500, 7.7 MB at 1000 and ~31 MB at 2000, which gives
~77 GB for the submitted runs and ~28 GB more for the rate cells. The lab share has 23 TB free.
The run folders in home are small (tens of MB each).

## 2. B3: the pilot's rule, and what still has to be submitted

**The rule, fixed before any result exists** (the plan's stage 0, with the operational details
added):
1. Read each pilot run where its noisy training loss first reaches 1.05× its own fitted floor
   (`common.noisy_floor` on `common.noisy_loss_series`, with the usual exclusions). Active units
   come from the participation trace at that crossing (noise-free, 5%-of-q95 rule). The noisy loss
   of a penalised run includes the penalty (plan §0); read it that way, as `RMtop200` will be.
2. A rung's multiplier = its mean active units at the floor ÷ the mean active units at the floor of
   the N = 1000 control, `NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control` (Della),
   read the same way. Its active fraction = its mean active units ÷ 1000.
3. Among the rungs whose active fraction is at or below 0.80, take the one whose multiplier is
   closest to 2. If both are equally close, take 0.01, which leaves more headroom.
4. If both rungs exceed 0.80, the plan adds a rung at 0.003 (3 more runs, ~50 GPU-h). That rung
   would take B3 past its budget, so it waits for Pavel.
5. If both multipliers fall below 1.3, use 0.03 anyway and say so.
6. A rung where fewer than 2 of 3 runs reach a read-out cannot be chosen on its numbers. If neither
   rung reaches a read-out, the choice goes to Pavel.

**Still to submit once the rule has picked λ** (from `~/trainRNNbrain_b23` on Spock, same commit):

```
C=--chdir=$HOME/trainRNNbrain_b23; L=slurm/SilentReLU_remedy_grid_spock.slurm; LAM=<chosen>
LAMBDA_FRM=$LAM SIGMA_W=1.0           N=1000 sbatch $C --export=ALL --job-name=B3fact --time=40:00:00 --array=1-3 $L
LAMBDA_FRM=$LAM RESCALE=1             N=1000 sbatch $C --export=ALL --job-name=B3fact --time=40:00:00 --array=1-3 $L
LAMBDA_FRM=$LAM SIGMA_W=1.0 RESCALE=1 N=1000 sbatch $C --export=ALL --job-name=B3fact --time=40:00:00 --array=1-3 $L
```

The factorial's "none" cell is the claim-1 control above (Della, same launcher family, code
7739a60), and its "rate" cell is the chosen pilot rung.

## 3. Decisions made in Pavel's absence

1. **B2 at N = 4000 was not submitted: it cannot run on any Spock GPU at the pinned code.** One
   (N, T, B) float32 tensor at N = 4000, T = 300, B = 1024 is 4.6 GiB. The forward pass holds about
   five of them (noise, states, the saved ReLU outputs, the stacked trajectory, the rate), and the
   rate penalty's log-sum-exp adds three or four more. Spock job 6111685_22 (9 Sep, commit 79a1063,
   whose penalty code is unchanged in 7739a60) ran the rate penalty at N = 4000 on an L40S and ran
   out of memory with 41.8 GB of 44.4 GiB in use, while allocating one more 4.6 GiB tensor inside
   the log-sum-exp. A100-40G is smaller still, and Della's 40 GB card failed the same way on 1 Oct
   (`F2fill.14841831_9`). Running it would need a smaller batch or a chunked penalty, and either
   would change the network or the code against the controls. The plan's own commands ask for
   Della's 80 GB cards (`--constraint=gpu80`); the N = 4000 rate-only cell belongs there. Unpenalised
   N = 4000 fits on Spock: the August runs took ~33 h per 100k on L40S and ~62 h on A100.
2. **B3 was split as the plan stages it, not run at both candidate strengths.** The pilot needs
   ~17 h, so it cannot finish tonight. Running the three rate-containing cells at both 0.01 and
   0.03 would cost ~165 GPU-h more and take B3 to ~590, against its ~390 budget. The cells that do
   not involve the rate penalty (noise, rescale, noise + rescale) were submitted now; the rate cells
   wait for the rule in §2. The cost is about one day of wall time.
3. **Snapshots go to the lab share, not home.** Home had 70 GB free of a 95 GB quota at submission,
   and B5 is still writing to it; the snapshots need ~105 GB. `/mnt/cup/labs/tengel/home/tolmach` is
   Pavel's own folder on the Engel lab share. Spock's `/scratch` is not writable for this account.
4. **The launcher is `_spock`, not the plan's `_della` file.** Same knobs, same arm naming and the
   same output root `NBitFlipFlop_remedy_grid` as the plan specifies, so a later Della N = 4000 cell
   from a `_della` copy lands in the same tree. The Spock header has no `nomig`/`gpu80` features
   (Spock has neither), 96 h is its longest accepted limit, and every run here fits one job.

## 4. Checks done before submission

- **Local smoke test (Mac, CPU), through the launcher file itself** with `module`/`conda`/`srun`
  stubbed: N = 50, 30 iterations, for the arms `frm0.1`, `frm0.03`, `sw1.0`, `rsc10`, `sw1.0_rsc10`
  and `frm0.01_sw1.0_rsc10`, plus a reference control from
  `SilentReLU_c1_snapshots_della.slurm TASK=ff` at the same commit. All seven trained and wrote kept
  snapshots (LastParams + config).
- **Pass condition, set before the diff:** each arm's flattened saved config may differ from the
  reference control only in its remedy keys, the snapshot cadence and bookkeeping (seed, tag,
  paths). Result: `frm*` arms differ only in `trainer.lambda_frm`; `sw1.0` only in `model.sigma_w`;
  `rsc10` only in `prune_reinit` and the seven rescale `prune_args` values; the combinations in the
  union of those; `checkpoint_every` where it was set to 10. Nothing else.
- **Against the controls on disk:** the reference control equals the first-segment configs of
  `NBitFlipFlop_ff_revive` (N = 500, 1000, 2000, arm none) and `NBitFlipFlop_paper_grid` (N = 4000,
  control) on every shared key except N and the iteration count. The keys newer than those runs are
  inert in a control: `sigma_w` 0, `iter_offset` 0, the checkpoint keys, and rescale keys that act
  only with `prune_reinit` on. The standard network holds: ReLU, h equation, bias range [0, 0],
  self-connections on, no Dale, γ 0, weight decay 1e-6, lr 1e-3 with exponent 1/3.
- **Spock environment:** the all-three arm ran 30 iterations on the Spock login node through the
  launcher (only `srun` stubbed): it trained, wrote snapshots and saved its run at `618fc9e`, and
  the worktree stayed clean.

## 5. What is left for Pavel or the next agent

- Read the pilot by the rule in §2 when 6929384 and 6929385 land, then submit the three rate cells.
- Decide whether B2 at N = 4000 runs on Della (80 GB cards) and when. On Della at 2.25 s/it it is
  two 100k segments per seed, ~375 GPU-h (plan §B2).
- The controls are on Della and these runs are on Spock; no direct route exists. The relay through
  the Mac in `slurm/sync_b5_spock_to_della.sh` (branch `b5-optimizer`) is the template. The snapshot
  tree is on the lab share, not under `~/trainRNNbrain/data`, so a sync needs both paths.
