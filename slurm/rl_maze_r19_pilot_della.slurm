#!/bin/bash
# ROUND 19 PILOT (DELLA): can a tanh or a shifted-sigmoid network learn the Maze under a changed learning
# rate or a smaller initial recurrent gain? At the standard settings neither activation learned in
# 2,000 updates (3 seeds each, success 0.00-0.38 and 0.15-0.21 at update 2,000). The rules, the reasons
# for each condition, what counts as a pass and how a passing setting is reported are in
# docs/project_trajectory/*_round19-train-all-rules.md, written and pushed before this launcher was
# submitted. Tracking: ptolmachev/dead-relu-paper#201.
#
# GRID (16 jobs) = 2 activations x 4 conditions x 2 seeds, N = 1,000, 2,000 updates, the Maze suite's
# configuration (task=Maze trainer=trainer_ppo model=rnn_<act>_standard) with only the keys below changed.
#   COND lr0.3     trainer.lr=0.0003            the size rule's rate x 0.3
#   COND lr0.1     trainer.lr=0.0001            the size rule's rate x 0.1
#   COND gain      model.spectral_rad=G_ACT     initial W_rec spectral radius such that radius x max f' = 0.85,
#                                               the ReLU network's RMS gain at the standard radius
#                                               (1.2 x sqrt(1/2)): tanh 0.85 (max f' 1), shifted sigmoid 0.45
#                                               (max f' = 7.5 / 4 = 1.875)
#   COND gain_lr0.3  both keys
#   Index decode (TID = SLURM_ARRAY_TASK_ID - 1): SEED = TID % 2 + 1; COND = (TID / 2) % 4; ACT = TID / 8
#     1-8   tanh             lr0.3 s1 s2, lr0.1 s1 s2, gain s1 s2, gain_lr0.3 s1 s2
#     9-16  sigmoid_shifted  the same
# Seeds 1 and 2 are the seeds of the standard-setting runs (Maze_suite/N=1000_act=<act>/s1, s2), so each
# pilot run has a paired run at the standard setting from the same initial draw (the gain conditions
# rescale that draw: RNN_torch rescales W_rec to the requested radius).
#
#   sbatch --array=1-16 --time=12:00:00 slurm/rl_maze_r19_pilot_della.slurm
# Output: $SAVE_TO/Maze_r19pilot/N=1000_act=<act>_<cond>/s<seed>
#
#SBATCH --job-name=R19pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=48G
#SBATCH --time=12:00:00
#SBATCH --array=1-16
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

REPO="${REPO:-$HOME/trainRNNbrain_r19}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze}"   # the maze runs' own tree

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

ACTS=(tanh sigmoid_shifted)
GAIN=(0.85 0.45)                    # radius x max f' = 0.85 for tanh (max f' 1) and the shifted sigmoid (1.875)
CONDS=(lr0.3 lr0.1 gain gain_lr0.3)
TID=$((SLURM_ARRAY_TASK_ID - 1))
SEED=$((TID % 2 + 1)); C=$(((TID / 2) % 4)); A=$((TID / 8))
ACT=${ACTS[$A]}; COND=${CONDS[$C]}
case "$COND" in
  lr0.3)      KEYS=(trainer.lr=0.0003) ;;
  lr0.1)      KEYS=(trainer.lr=0.0001) ;;
  gain)       KEYS=(model.spectral_rad="${GAIN[$A]}") ;;
  gain_lr0.3) KEYS=(model.spectral_rad="${GAIN[$A]}" trainer.lr=0.0003) ;;
esac
TAG="r19pilot/N=1000_act=${ACT}_${COND}"
OUT="${SAVE_TO}/Maze_${TAG}/s${SEED}"
if ls "$OUT"/*config.yaml >/dev/null 2>&1; then
  echo "FATAL: $OUT already holds a run; refusing to overwrite it." >&2; exit 1
fi
echo "Task ${SLURM_ARRAY_TASK_ID}: act=${ACT} cond=${COND} seed=${SEED} keys=${KEYS[*]} -> ${OUT}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  task=Maze trainer=trainer_ppo seed="$SEED" model="rnn_${ACT}_standard" model.N=1000 \
  paths=paths_DELLA paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\"" "${KEYS[@]}"
