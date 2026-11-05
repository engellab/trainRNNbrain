#!/bin/bash
# THE MAZE SUITE (SPOCK), sizes up to 2000 (40 GB cards): does the loss of participation seen on the 3-bit flip-flop and CDDM
# appear under PPO on the maze? 8 egocentric range sensors plus the efference copy of the previous
# action, nothing about the cheese; the 15 x 15 maze of
# configs/rl/maze_ppo.yaml, 2,000 updates of 256 episodes x 500 decisions (5 RNN steps each).
# Tracking issue: engellab/trainRNNbrain #3, which holds the pre-registration (read-out points,
# what would falsify "the problem persists"); the notebook entry written at submission names it.
#
# GRID (23 jobs), one per array index:
#   1-3     N=  250  relu              seeds 1-3
#   4-6     N=  500  relu              seeds 1-3
#   7-9     N= 1000  relu              seeds 1-3
#   10-12   N= 2000  relu              seeds 1-3        (about 3x slower per update)
#   13-14   N= 1000  tanh              seeds 1-2
#   15-16   N= 1000  sigmoid (shifted) seeds 1-2
#   17-18   N= 1000  softplus (beta 25) seeds 1-2
#   19-20   N= 1000  leaky relu        seeds 1-2
#   21-23   N= 4000  relu              seeds 1-3        (80 GB cards: --constraint=gpu80)
# Model configs: configs/model/rnn_<act>_standard.yaml, the same files the supervised series use.
# ONE PROTOCOL FOR EVERY SIZE: the paper's learning-rate rule with the warm-up of
# configs/rl/maze_ppo.yaml (ppo.warmup_updates). Before the warm-up existed, every N = 2000 run
# at the paper's rate diverged within its first updates (arrays 16856487 and 16859389, 4 Nov
# 2026) and three seeds ran at a quarter of the rate instead (array 16866951); those are the
# exception data, not part of the uniform series.
#
# SUBMIT IN TWO GROUPS (N = 4000, indices 21-23, does not fit Spock's 40 GB cards: Della only):
#   sbatch --array=1-9,13-20 --time=12:00:00 slurm/rl_maze_suite_spock.slurm
#   sbatch --array=10-12     --time=24:00:00 slurm/rl_maze_suite_spock.slurm
#
#SBATCH --job-name=MazeSuite
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=1-20%20
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/usr/people/pt1290/trainRNNbrain_logs/%x.%A_%a.out
#SBATCH --error=/usr/people/pt1290/trainRNNbrain_logs/%x.%A_%a.err

module purge
module load anacondapy/2024.02
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
TASK_N=(0 250 250 250 500 500 500 1000 1000 1000 2000 2000 2000 1000 1000 1000 1000 1000 1000 1000 1000 4000 4000 4000)
TASK_ACT=(x relu relu relu relu relu relu relu relu relu relu relu relu tanh tanh sigmoid_shifted sigmoid_shifted softplus_beta25 softplus_beta25 leaky_relu leaky_relu relu relu relu)
TASK_SEED=(0 1 2 3 1 2 3 1 2 3 1 2 3 1 2 1 2 1 2 1 2 1 2 3)

i=$SLURM_ARRAY_TASK_ID
N=${TASK_N[$i]}; ACT=${TASK_ACT[$i]}; SEED=${TASK_SEED[$i]}
TAG="suite/N=${N}_act=${ACT}"
echo "Task ${i}: N=${N} model=rnn_${ACT}_standard seed=${SEED} -> ${SAVE_TO}_rl_maze/Maze_${TAG}/s${SEED}"

srun python "$REPO/trainRNNbrain/rl/run_maze_ppo.py" \
  seed="$SEED" model="rnn_${ACT}_standard" model.N="$N" \
  paths=paths_local paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\""
