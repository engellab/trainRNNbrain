#!/bin/bash
# THE UNATTENDED STEPS OF THE INDEPENDENT-INIT TEST (DELLA): the 3BFF continuation behind the GPU-hour rule, and the read.
# Rules: docs/project_trajectory/2026-11-25_1303_rules-independent-init.md and the unattended-chain amendment.
#
# MODE=topup  (GPU array 1-3, after every stage 1 task has ended). Sums the elapsed hours of the stage 1 tasks (sacct, job
#             name IndInit, since 25 Nov 2026; the nine tasks must all have ended). If used + 27 > 80 it exits without
#             training. Otherwise it continues one 3BFF network to 200,000 iterations for at most (80 - used) / 3 hours, so
#             the three continuations cannot pass 80 GPU-hours in total. If sacct cannot be read, it exits without training.
# MODE=read   (CPU, after the continuation). The read, the Maze read, the verdict, the figure and the independent
#             recounts, into $OUT, next to the networks.
#
#   C=--chdir=$HOME/trainRNNbrain_indinit; A=$HOME/trainRNNbrain_indinit_ana
#   T=$(MODE=topup sbatch --parsable $C --export=ALL,MODE=topup --dependency=afterany:<3BFF>:<CDDM>:<Maze> --array=1-3 \
#         --job-name=IndInitTop --gres=gpu:1 --constraint=gpu80 --mem-per-gpu=32G --cpus-per-task=2 --time=17:00:00 $A/slurm/independent_init_chain_della.slurm)
#   sbatch --chdir=$A --export=ALL,MODE=read --dependency=afterany:$T --job-name=IndInitRead --cpus-per-task=8 --mem=48G \
#         --time=06:00:00 $A/slurm/independent_init_chain_della.slurm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.err
set -o pipefail
DATA=/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data
STAGE1_JOBS="${STAGE1_JOBS:?set STAGE1_JOBS to the three stage 1 array ids, comma-separated}"
CAP=80; NEED=27; NTASKS=9

if [ "$MODE" = topup ]; then
  for try in $(seq 1 20); do
    # -X: one row per array task; the name and the start date keep out old jobs whose ids sacct reuses
    ROWS=$(sacct -u pt1290 -X -n -P -S 2026-11-25 --name=IndInit --format=jobid,state,elapsedraw | grep -E "^($(echo "$STAGE1_JOBS" | tr , '|'))_[0-9]+\|") || ROWS=""
    N=$(echo "$ROWS" | grep -c '|'); LIVE=$(echo "$ROWS" | grep -cE '\|(RUNNING|PENDING|REQUEUED|COMPLETING|SUSPENDED)')
    [ "$N" -eq "$NTASKS" ] && [ "$LIVE" -eq 0 ] && break
    sleep 30
  done
  if [ "$N" -ne "$NTASKS" ] || [ "$LIVE" -ne 0 ]; then echo "GATE: sacct shows $N of $NTASKS tasks, $LIVE not ended; not training." >&2; exit 1; fi
  USED=$(echo "$ROWS" | awk -F'|' '{s+=$3} END {print s}')     # GPU-seconds, one GPU per task
  echo "$ROWS"; echo "GATE: stage 1 used $USED GPU-seconds"
  if [ $((USED + NEED * 3600)) -gt $((CAP * 3600)) ]; then echo "GATE: used + ${NEED} h is above ${CAP} h; not continuing."; exit 0; fi
  BUDGET=$(( (CAP * 3600 - USED) / 3 - 120 ))
  # the run of one task: its own limit in seconds from now; the launcher's own checks follow
  echo "GATE: this continuation may run $BUDGET s"
  export REPO=$HOME/trainRNNbrain_indinit KEEP=1 TARGET=200000 SAVE_TO=$DATA/trained_RNNs
  export PARENTS="$DATA/trained_RNNs/NBitFlipFlop_independent_init/EqType=h_N=1000_draws=independent/*/"
  timeout --kill-after=60 "$BUDGET" bash "$REPO/slurm/SilentReLU_ff_topup200k_della.slurm"
  exit $?
fi

module purge; module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch-env
export CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 PYTHONPATH=$HOME/trainRNNbrain_indinit_ana
OUT=${OUT:-$DATA/trained_RNNs/independent_init_reads}; mkdir -p "$OUT"
E=trainRNNbrain/experiments_and_analysis/independent_init_read.py; cd "$PYTHONPATH" || exit 1
git log --oneline -1
python -u $E read "$OUT/reads.json" --logs "$HOME/trainRNNbrain_logs"
python -u $E maze-read "$OUT/maze_reads.json" --new "$DATA/trained_RNNs_rl_maze/Maze_independent_init" --suite "$HOME/indinit_out/Maze_suite_ref"
python -u $E verdict "$OUT/reads.json" "$OUT/maze_reads.json" --fig "$OUT/independent_init_active" | tee "$OUT/verdict.txt"
python -u $HOME/indinit_out/scripts/recount_indinit.py "$OUT/reads.json" | tee "$OUT/recount_supervised.txt"
python -u $HOME/indinit_out/scripts/recount_maze.py "$OUT/maze_reads.json" "$DATA/trained_RNNs_rl_maze_checkpoints/Maze_independent_init" | tee "$OUT/recount_maze.txt"
