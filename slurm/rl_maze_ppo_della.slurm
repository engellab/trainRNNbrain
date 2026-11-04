#!/bin/bash
# THE MAZE TASK, FIRST LOOK (DELLA). Does the standard RNN learn to run the maze under PPO, and do
# its units go dormant there too? Task and read-out: configs/rl/maze_ppo.yaml and
# trainRNNbrain/rl/run_maze_ppo.py. Tracking issue: engellab/trainRNNbrain #2.
#
# GRID (8 jobs), one per array index:
#   1-2   N= 500  blind (walls only)   seeds 1, 2
#   3-4   N=1000  blind                seeds 1, 2
#   5-6   N= 500  cue (bearing input)  seeds 1, 2
#   7-8   N=1000  cue                  seeds 1, 2
# Everything else comes from the config (1500 updates of 256 episodes x 300 steps).
#
# PASS/FAIL, set before submission (issue #2): the cue arm reaches a probe success rate of 0.8 by the
# end of training at both sizes. The blind arm's result is a measurement, not a pass condition.
#
#   sbatch slurm/rl_maze_ppo_della.slurm
#
#SBATCH --job-name=MazePPO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-8%8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.err

module purge
module load cudatoolkit/12.8
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env

set -eo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="${REPO:-$HOME/trainRNNbrain_rlmaze}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"   # runs land in ${SAVE_TO}_rl_maze

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO); aborting." >&2; exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
if [ -z "$GITSHA" ]; then
  echo "FATAL: $REPO is not a git checkout; the provenance of this run cannot be recorded." >&2; exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA"

# index 0 unused so the decode reads SLURM_ARRAY_TASK_ID directly
TASK_N=(0 500 500 1000 1000 500 500 1000 1000)
TASK_CUE=(x false false false false true true true true)
TASK_SEED=(0 1 2 1 2 1 2 1 2)

i=$SLURM_ARRAY_TASK_ID
N=${TASK_N[$i]}; CUE=${TASK_CUE[$i]}; SEED=${TASK_SEED[$i]}
TAG="first/N=${N}_cue=${CUE}"
echo "Task ${i}: N=${N} goal_cue=${CUE} seed=${SEED} -> ${SAVE_TO}_rl_maze/Maze_${TAG}/s${SEED}"

srun python "$REPO/trainRNNbrain/rl/run_maze_ppo.py" \
  seed="$SEED" model.N="$N" env.goal_cue="$CUE" \
  paths=paths_DELLA paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\""
