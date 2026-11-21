## 2026-11-20 23:22 — Entropy and correlation set points: pilot cost, and the 3BFF screen submitted

Follows the rules `2026-11-20_2307_rules-entropy-correlation-setpoint.md` (committed at 0ca7929 before
any run). Branch `wp/entropy-setpoint`.

**Pinned code.** `~/trainRNNbrain_entsp` on Della, a clean clone detached at 0ca7929. Reads run from a
second clone at the same commit, `~/trainRNNbrain_entsp_ana`, whose `data/trained_RNNs` links to
`/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs` (the tree the launcher writes to).

**Pilot** (Della job 18047037, array tasks 6 and 12, 2,000 iterations each, the `gputest` partition,
nodes della-l02g3 and della-l03g3; output in `NBitFlipFlop_entropy_setpoint_pilot/`, which enters no result):

| arm | s per iteration, iterations 1,000 to 2,000 | skipped updates | job wall time | training-batch R² printed at 500 / 1,500 / 2,000 |
|---|---|---|---|---|
| `ent0.1` | 0.251 | 0 | 9 min 17 s | 0.788 / 0.868 / 0.864 |
| `cor0.1` | 0.289 | 0 | 10 min 14 s | 0.781 / 0.858 / 0.478 |

Neither pilot ran out of memory, ended with a non-finite loss or skipped an update, so the screen starts.
The `cor0.1` value at iteration 2,000 is a single training iteration's R², not a read-out; it is recorded
here because it is the only sign so far of what the correlation term does to training.

**Budget rule applied.** c = 0.289 s × 75,000 / 3,600 s + 0.2 h = 6.2 GPU-hours per network (the larger
of the two terms). The pilots used P = 0.33 GPU-hours. 12c + P = 75 GPU-hours, within the 100 allowed, so
all 12 networks run.

**Screen submitted.** Della job 18047724, array 1-12 (`ent0.01` 1-3, `ent0.1` 4-6, `cor0.01` 7-9,
`cor0.1` 10-12), gpu-short, `--time=23:00:00`, `--constraint=nomig`:

    ssh della 'cd ~/trainRNNbrain_entsp && sbatch --chdir=$HOME/trainRNNbrain_entsp --export=ALL --array=1-12 slurm/entropy_setpoint_della.slurm'

Output: `$HOME/trainRNNbrain/data/trained_RNNs/NBitFlipFlop_entropy_setpoint/EqType=h_N=1000_arm=<arm>/`;
logs `~/trainRNNbrain_logs/EntSP.18047724_<task>.{out,err}`.

**Read, when the jobs finish:**

    sbatch ... --chdir=$HOME/trainRNNbrain_entsp_ana --wrap "... PYTHONPATH=$HOME/trainRNNbrain_entsp_ana CUDA_VISIBLE_DEVICES= \
      python trainRNNbrain/experiments_and_analysis/entropy_setpoint_read.py read /scratch/gpfs/TENGEL/pt1290/entropy_setpoint/read"
    python trainRNNbrain/experiments_and_analysis/entropy_setpoint_read.py verdict <reads.json> --fig img/internal_figures/entropy_setpoint_active

A dry run of the reader on the controls and the rate-penalty networks alone (job 18047763) checks that it
reproduces their recorded values (279, 288, 242, 236 active units; R² 0.9514, 0.9517, 0.9509).
