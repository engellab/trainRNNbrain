#!/bin/bash
# PRUNE-AND-RETRAIN ON THE MAZE (SPOCK), N = 1,000: the retrain strip of Figure 1d for the maze.
# For each of the three N = 1,000 ReLU agents of the uniform series, three arms, each trained 200
# further PPO updates by trainRNNbrain/experiments_and_analysis/rl_maze_retrain.py (its docstring
# has the definitions): intact (the control), silent_removed (every inactive unit's outgoing
# weights zeroed and held at zero), random_removed (as many units at random). The score is the
# success rate under the training noise over 256 fixed drops x 4 draws, at update 0 and every 25
# updates. One JSON per (run, arm) under OUT. Tracking issue: engellab/trainRNNbrain#3 (Figure 1).
#
# GRID (9 jobs), one per array index; seed = (i - 1) / 3 + 1, arm = (i - 1) % 3:
#   1-3   s1   intact, silent_removed, random_removed
#   4-6   s2   intact, silent_removed, random_removed
#   7-9   s3   intact, silent_removed, random_removed
# Expected: about 200 updates x 10 s + 9 scorings x 4 rollouts, under 1.5 h per job on a 40 GB card.
#
# SUBMIT:  sbatch slurm/rl_maze_retrain_spock.slurm
#
#SBATCH --job-name=MazeRetrain
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=1-9%9
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

REPO="${REPO:-$HOME/trainRNNbrain_rlmaze5}"
RUNS="${RUNS:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite/N=1000_act=relu}"
OUT="${OUT:-$HOME/trainRNNbrain/data/rl_maze_retrain}"

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

ARMS=(intact silent_removed random_removed)
i=$SLURM_ARRAY_TASK_ID
SEED=$(( (i - 1) / 3 + 1 )); ARM=${ARMS[$(( (i - 1) % 3 ))]}
RUN="${RUNS}/s${SEED}"
if ! ls "${RUN}"/*Params_Maze.npz >/dev/null 2>&1; then
  echo "FATAL: ${RUN} has no *Params_Maze.npz." >&2; exit 1
fi
echo "Task ${i}: run=${RUN} arm=${ARM} -> ${OUT}"

srun python "$REPO/trainRNNbrain/experiments_and_analysis/rl_maze_retrain.py" "$RUN" --arm "$ARM" --out "$OUT"
