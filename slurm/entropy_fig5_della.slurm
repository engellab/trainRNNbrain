#!/bin/bash
# Entropy set point in Figure 5: removal and pre-activation reads (no training).
# Rules: docs/project_trajectory/2026-11-25_1235_rules-entropy-removal-and-preactivation.md
#   MODE=gpu   supervised removal (with its gates), then the pre-activations of all three tasks; one GPU job
#   MODE=maze  array 0-3, CPU: the Maze removal, 0 = gate network (prune-and-duplicate s1), 1-3 = ent0.1 s1-s3
# Submit (from the pinned clone):
#   sbatch --export=ALL,MODE=gpu --gres=gpu:1 --constraint=nomig --mem-per-gpu=32G --time=02:30:00 slurm/entropy_fig5_della.slurm
#   sbatch --export=ALL,MODE=maze --array=0-3 --mem=32G --time=04:00:00 slurm/entropy_fig5_della.slurm
#SBATCH --job-name=EntFig5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.out
set -euo pipefail
REPO=${REPO:-$HOME/trainRNNbrain_entfig5}
OUT=${OUT:-/scratch/gpfs/TENGEL/pt1290/entfig5}
DATA=/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data
A3=${A3:-$OUT/a3_N1000.json}
set +u
module purge; module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch-env
set -u
cd "$REPO"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 1; }
git rev-parse HEAD
export PYTHONPATH=$REPO OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=trainRNNbrain/experiments_and_analysis
if [ "$MODE" = gpu ]; then
  python -u $PY/entropy_fig5.py removal "$OUT" --a3 "$A3"
  python -u $PY/entropy_fig5.py preact "$OUT" --maze-root "$DATA"
else
  export CUDA_VISIBLE_DEVICES=
  if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
    python -u $PY/rl_maze_ablate.py "$DATA/trained_RNNs_rl_maze_fz/Maze_interventions/duplicate/s1" --added 430 --out "$OUT/maze_gate"
  else
    python -u $PY/rl_maze_ablate.py "$DATA/trained_RNNs_rl_maze_entsp_N1000/Maze_interventions/ent0.1/s$SLURM_ARRAY_TASK_ID" --added 430 --out "$OUT/maze"
  fi
fi
