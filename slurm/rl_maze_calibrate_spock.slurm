#!/bin/bash
# MAZE DIFFICULTY CALIBRATION (SPOCK, CPU only): the check that chose the first maze (configs/task/Maze.yaml:
# N = 100, 300 updates, probe success) run on candidate second mazes for the Maze transfer test (G17,
# round 18). Rules: docs/project_trajectory/2026-11-07_*_round18-second-maze-calibration.md.
#
# Each array task trains one N = 100 network for 300 updates on one maze from
# `make_maze.py --cells 7 --seed <MAZE_SEED> --loops 4` (the first maze is seed 20), with network seed
# 1, 2 or 3. Everything else is the uniform series' configuration (task=Maze, trainer_ppo,
# rnn_relu_standard). The calibration protocol's size and length (model.N=100, trainer.n_updates=300)
# and the maze are the keys this sweep sets; the maze is passed as task.maze, built by make_maze.py.
# Index decode: i = SLURM_ARRAY_TASK_ID - 1; maze = MAZES[i / 3]; network seed = i % 3 + 1.
#   MAZES="20 1446 1610 ..." REPO=$R sbatch --chdir=$R --export=ALL --array=1-<3 x n> $R/slurm/<this>
# Output: $SAVE_TO/Maze_calib/maze=<seed>/s<seed>/ (the standard run layout; probe success in
# s<seed>_Participation.npz). CPU: about 30-45 min per run with 4 threads.

#SBATCH --job-name=MazeCalib
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/usr/people/pt1290/trainRNNbrain_logs/%x.%A_%a.out
#SBATCH --error=/usr/people/pt1290/trainRNNbrain_logs/%x.%A_%a.err

module purge
module load anacondapy/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env

set -eo pipefail
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

REPO="${REPO:-$HOME/trainRNNbrain_r18calib}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_calib}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
[ -n "$GITSHA" ] || { echo "FATAL: $REPO is not a git checkout." >&2; exit 1; }
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $REPO @ $GITSHA"

read -r -a MZ <<< "${MAZES:?set MAZES to a space-separated list of make_maze seeds}"
i=$((SLURM_ARRAY_TASK_ID - 1))
MSEED="${MZ[$((i / 3))]}"; SEED=$((i % 3 + 1))
[ -n "$MSEED" ] || { echo "FATAL: no maze for task $SLURM_ARRAY_TASK_ID" >&2; exit 1; }
MAZE=$(python -c "
import sys, numpy as np
from trainRNNbrain.tasks.make_maze import dfs_maze, add_loops, rows_of
rng = np.random.default_rng(int(sys.argv[1])); w, g = dfs_maze(7, rng); add_loops(w, 4, rng)
print('[' + ','.join(repr(r) for r in rows_of(w, g)) + ']')" "$MSEED")
TAG="calib/maze=${MSEED}"
echo "Task ${SLURM_ARRAY_TASK_ID}: maze seed ${MSEED}, network seed ${SEED} -> ${SAVE_TO}/Maze_${TAG}/s${SEED}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  task=Maze trainer=trainer_ppo seed="$SEED" model=rnn_relu_standard model.N=100 trainer.n_updates=300 \
  "task.maze=${MAZE}" paths=paths_local paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\""
