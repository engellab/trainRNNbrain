# B2, B3 and B4 moved from Spock to Della

*Agent move, night of 14-15 October 2026, branch `agent-move`. Pavel approved the move and the runs
(B2 full, B3 minimal, B4 minimal). The records of what these runs are and how to read them stay in
`docs/notes/agent-b23.md` (B2, B3) and `docs/notes/agent-b4.md` (B4); this note records only where
they now run.*

**Why.** Every one of the 33 runs (job names `B2frm`, `B3pilot`, `B3fact`, `B4xfer`) had been
pending on Spock for 24 hours without starting, behind `B5opt` and other users' jobs. Della's
gpu-medium and gpu-long queues had per-user slots free.

## 1. Code

- **Della worktree `~/trainRNNbrain_redo`, detached at `709c17d`** = `restore-oct5` at `fcb3749`
  plus one file, `slurm/SilentReLU_b4_transfer_della.slurm`. `git diff --stat fcb3749 709c17d`
  shows only that file. The worktree was clean at submission; both launchers refuse a dirty tree.
- **Same training code as the Spock runs would have used.** `git diff 7739a60 fcb3749` touches no
  file under `trainRNNbrain/trainer`, `rnns` or `tasks`. It changes `run_experiment.py` (the B4
  transfer loader `load_warm_start` with `paths.transfer_from`; kept snapshots in float16 with a
  `resume/` copy), the `paths` configs (`transfer_from: null`) and the trainer configs
  (`checkpoint_every: 1000`, `checkpoint_keep: true`). The B4 Spock pin `c1396ae` already held the
  transfer code; the B2/B3 Spock pin `618fc9e` did not, and needs none, since `transfer_from` is
  null there.
- **Snapshots differ from the Spock submission in two ways, both on purpose.** Kept snapshots come
  every 1,000 iterations (Spock: 500 at N ≤ 1000) and store W_rec in float16 (Pavel's decisions of
  14 Oct, trajectory "Snapshot storage"). They go to
  `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots/<prefix>_<tag>/s<seed>/it<iter>/`,
  outside the data tree, with the exact resumable copy in `.../s<seed>/resume/`.
- **The Della B4 launcher** is the Spock one with a Della header (modules, log path,
  `--constraint=nomig`, `--mem-per-gpu=32G`, `--time=1-12:00:00`) and Della defaults for `REPO`
  and `SNAP_TO`. Its `srun` command is the Spock file's, line for line (`diff` of the two from
  `srun` to the end is empty). On Della the young sources sit among every other kept snapshot, so
  `SOURCES` names the three snapshots one by one; the launcher word-splits and sorts it, which
  gives the same task-to-source order as on Spock.

## 2. Queue placement

Della's `job_submit.lua` sets the QOS from `--time` alone: up to 1,441 minutes is gpu-short, up to
4,321 minutes is gpu-medium, anything longer (to 8,641) is gpu-long. Per-user caps
(`sacctmgr show qos`, 15 Oct): gpu-medium 20 GPUs and 24 jobs, gpu-long 16 GPUs and 10 jobs.
At submission gpu-long held 6 of our jobs (3 `FFtop4000`, due ~15 Oct afternoon; 3 `B2frm4k`),
leaving 4 job slots. Cluster-wide pending at 00:40: gpu-short 1,847, gpu-medium 133, gpu-long 11.
Our 77 pending `RMtop200` tasks carry `nice=10000`, so the moved jobs start ahead of them.

Every job asks for `--constraint=nomig`, which keeps it off the 3g.40gb MIG slices (partial A100s
on which N = 1000 runs took up to 0.74 s/it, three times a full card). All other GPU nodes are full
A100s, 40 or 80 GB. The rate penalty at N = 2000 fits on 40 GB: the pre-standard `FFpenL` runs
trained it on Spock's A100-40G cards (estimate: nine (N, T, B) float32 tensors of 2.3 GiB, ~21 GB).

Time limits, from the plan's measured Della speeds on full cards (`agent-bplan.md` §0):

| Cells | s/it | Hours for 200k | --time | QOS |
|---|---|---|---|---|
| B2 N = 500 | 0.18 | ~10 | 36 h | gpu-medium |
| B2 N = 1000, B3 pilot, B3 noise | 0.26 | ~14 | 36 h | gpu-medium |
| B3 rescale, noise + rescale | ~0.30 (assumed) | ~17 | 36 h | gpu-medium |
| B4 old, young | 0.24 | ~13 | 36 h | gpu-medium |
| B4 fresh | 0.24 | ~13 | 76 h | gpu-long (the 4th free slot) |
| B2 N = 2000 | 0.71 | ~40 | 84 h | gpu-long |

Submission order sets the start order among our own jobs: pilot, B2, B4, then the B3 factorial,
whose cells are compared with the pilot's rate cells anyway and so lose nothing by starting a few
hours later.

## 3. Checks before submission

- **Local smoke test (Mac, CPU), through the launcher files themselves** from a clean `agent-move`
  worktree, `module`/`conda`/`srun` stubbed: remedy grid at N = 50, 30 iterations, arms `control`,
  `frm0.1`, `frm0.01`, `frm0.03`, `sw1.0`, `rsc10`, `sw1.0_rsc10`; B4 at N = 1000 (the real
  sources), 30 iterations, old (task 2), young (task 3, sources given as the explicit list), fresh
  (task 3). All ten exited 0.
- **Config diff, pass condition fixed before the diff:** each remedy arm's flattened saved config
  may differ from the control's only in its own remedy keys plus seed, tag and paths. Result: the
  `frm` arms differ only in `trainer.lambda_frm`; `sw1.0` only in `model.sigma_w`; `rsc10` only in
  `trainer.prune_reinit` and seven `trainer.prune_args` values; `sw1.0_rsc10` in the union. PASS.
  The new control differs from the claim-1 control (`NBitFlipFlop_c1_snapshots`, N = 1000) only in
  N, iteration count, snapshot spacing (500 → 1,000) and the new key `paths.transfer_from: null`.
- **B4 configs:** each transfer differs from its source only in the task swap (taskname, target,
  7 outputs), `max_iter`, `iter_offset` (old: 75,000 → 0), the snapshot spacing, seed, tag and
  paths (`init_from` null, `transfer_from` = the source). Fresh differs from the young transfer of
  the same source only in `transfer_from` (null), seed and tag. PASS. Logs show W_rec, W_inp and
  y_init loaded and W_out kept at the run's own draw, shape (7, 1000).
- **Snapshots:** a 2,100-iteration run of the `frm0.1` arm at N = 50 wrote `it0001000/` and `it0002000/`
  (W_rec float16, other arrays float32, compressed) and `resume/` (float32 weights plus
  `AdamState`), with no launcher flag; the launcher passes no override that turns snapshots off.
  The jobs were submitted after `it0001000/` and `resume/` appeared; `it0002000/` and the run's
  normal end (exit 0) came minutes later.
- **Della login node,** through both launchers at `709c17d` with only `srun` stubbed: the all-three
  arm at N = 50 for 20 iterations, and the young transfer task 2 for 3 iterations on the real
  Della snapshot (it picked `s2670297180/it0067500`, as Spock's task 2 did). Both exited 0, the
  import check resolved to `~/trainRNNbrain_redo @ 709c17d`, and the worktree stayed clean.

## 4. Submitted on Della

Submitted 15 Oct 00:55 EDT from `~/trainRNNbrain_redo` (`709c17d`, clean), each array 3 seeds
(`seed=random`), 200,000 iterations from scratch in one job, `--constraint=nomig`, snapshots every
1,000. Launchers: `slurm/SilentReLU_remedy_grid_della.slurm` (B2, B3) and
`slurm/SilentReLU_b4_transfer_della.slurm` (B4); `REPO=~/trainRNNbrain_redo` exported.

| Job | Name | Item | Cell | --time | --mem-per-gpu | QOS |
|---|---|---|---|---|---|---|
| 15598507 | B3pilot | B3 stage 0 | N = 1000, `frm0.01` | 36 h | 32G | gpu-medium |
| 15598508 | B3pilot | B3 stage 0 | N = 1000, `frm0.03` | 36 h | 32G | gpu-medium |
| 15598509 | B2frm | B2 | N = 2000, `frm0.1` | 84 h | 64G | gpu-long |
| 15598510 | B2frm | B2 | N = 1000, `frm0.1` | 36 h | 32G | gpu-medium |
| 15598511 | B2frm | B2 | N = 500, `frm0.1` | 36 h | 32G | gpu-medium |
| 15598512 | B4xfer | B4 | `src=control_age=old` (the three 200k runs) | 36 h | 32G | gpu-medium |
| 15598513 | B4xfer | B4 | `src=control_age=young` (snapshots 52.5k, 67.5k, 60k) | 36 h | 32G | gpu-medium |
| 15598514 | B4xfer | B4 | `src=fresh` (MODE=fresh, the young configs) | 76 h | 32G | gpu-long |
| 15598515 | B3fact | B3 stage 1 | N = 1000, `sw1.0` | 36 h | 32G | gpu-medium |
| 15598516 | B3fact | B3 stage 1 | N = 1000, `rsc10` | 36 h | 32G | gpu-medium |
| 15598517 | B3fact | B3 stage 1 | N = 1000, `sw1.0_rsc10` | 36 h | 32G | gpu-medium |

B4 sources on Della, task i = i-th in sorted order, the same pairing as on Spock (`agent-b4.md` §4):
old = `trained_RNNs/NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control/*/` (seeds
2461724787, 4113307577, 1258717025); young =
`c1_snapshots/NBitFlipFlop_c1_snapshots/EqType=h_N=1000_arm=control/` `s1542248578/it0052500`,
`s2670297180/it0067500`, `s2851654641/it0060000`.

**Cost.** ~540 GPU-h expected: B2 ~190 (N = 500 ~30, 1000 ~43, 2000 ~118), B3 pilot ~87, B3
noise / rescale / both ~141, B4 ~120. Snapshot disk ~20 GB on TENGEL scratch (1.4 TB free); run
folders ~60 MB each in home (21 of 49 GB used).

**Queue at 01:05.** All 33 tasks pending on priority. Every full (non-MIG) GPU on the `gpu`
partition was allocated. Our jobs head the gpu-medium queue: the only pending jobs with higher
priority competing for the same nodes were 485 gpu-short jobs and 4 della-shakemov jobs. Slurm
estimated 02:22 for the first pilot array and gave no estimate for the rest.

**Expected start and finish** (not a Slurm estimate). The per-user caps let 20 gpu-medium tasks
and 4 gpu-long tasks run at once; Slurm starts them in submission order as cards free up.
- Running at once, if cards free up: both pilot rungs, B2 N = 1000 and 500, B4 old and young,
  and 2 of the 3 `sw1.0` tasks (20, gpu-medium); B2 N = 2000 and B4 fresh task 1 (4, gpu-long).
  If they start by ~03:00, they finish: B2 N = 500 ~13:00, the N = 1000 cells ~17:00-19:00 on
  15 Oct, B2 N = 2000 ~19:00 on 16 Oct.
- Waiting for our own slots: 1 `sw1.0` task and the 6 rescale tasks (gpu-medium, start when
  B2 N = 500 ends, ~13:00, or the N = 1000 cells, ~17:00); B4 fresh tasks 2 and 3 (gpu-long,
  start when `FFtop4000` releases its 3 slots, due ~15 Oct afternoon). These finish ~16 Oct
  03:00-10:00.
- **The B3 pilot answers in under a day** if it starts tonight: ~14 h of training, read at the end
  of training by `agent-b23.md` §2, so ~15 Oct evening. The rate cells then need ~17 h more, so
  B3 completes ~17 Oct.
- If the jobs are still pending by morning, dropping `--constraint=nomig` would add the 10 MIG
  nodes (80 slices). A slice runs N = 1000 at about a third of a full card's speed, so 200k would
  take ~41 h, past the 36 h limit. Dropping the constraint therefore means resubmitting with a
  longer limit (gpu-medium allows up to 72 h), not editing the pending jobs.

## 5. Cancelled on Spock

15 Oct 00:55 EDT, after the Della jobs were submitted and pending: `scancel -u pt1290` on
6929384, 6929385 (B3pilot), 6929386, 6929390, 6929391 (B2frm), 6929387, 6929388, 6929389
(B3fact), 6929407, 6929408, 6929409 (B4xfer). Before cancelling, every one of the 33 tasks was
checked to be PENDING and named B2frm, B3pilot, B3fact or B4xfer. None had started, so Spock holds
no partial output from them. Spock afterwards held only `B5opt` (another session; 5 running,
6 pending), untouched. B2 at N = 4000 (Della 15527679) was not touched.

## 6. What is left

- **B3 rate cells** wait for the pilot's rule (`agent-b23.md` §2). The pilot needs ~14 h once it
  starts, so its answer is due within a day of submission. Submit from the same worktree, after
  setting `LAM` to the chosen rung:

  ```
  R=$HOME/trainRNNbrain_redo; cd $R; export REPO=$R; L=slurm/SilentReLU_remedy_grid_della.slurm
  M="--chdir=$R --export=ALL --job-name=B3fact --constraint=nomig --mem-per-gpu=32G --time=1-12:00:00 --array=1-3"
  LAMBDA_FRM=$LAM SIGMA_W=1.0           N=1000 sbatch $M $L
  LAMBDA_FRM=$LAM RESCALE=1             N=1000 sbatch $M $L
  LAMBDA_FRM=$LAM SIGMA_W=1.0 RESCALE=1 N=1000 sbatch $M $L
  ```
- **Everything is now on Della**, next to the controls, so no Spock-to-Della sync is needed.
  Outputs: `~/trainRNNbrain/data/trained_RNNs/NBitFlipFlop_remedy_grid/EqType=h_N=<N>_arm=<arm>/`
  and `~/trainRNNbrain/data/trained_RNNs/NBitFlipFlopHyper_b4_transfer/EqType=h_N=1000_<cell>/`.
  Logs: `~/trainRNNbrain_logs/<job name>.<job id>_<task>.out`.
- A run that times out can be continued from `.../s<seed>/resume/` (float32 weights and Adam
  state) with `paths.init_from`; not tested here.
