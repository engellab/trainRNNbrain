#!/bin/bash
# B5, THE OPTIMISER COMPARISON: is the number of active units a property of Adam? (DELLA)
#
# The Della copy of SilentReLU_b5_optimizer_spock.slurm (5 Nov 2026; the uniform-interventions
# programme, engellab/trainRNNbrain issue #6, docs/interventions_matrix.md). The Spock launcher is
# unchanged and 3BFF-only; this one has the Della module loads, the git provenance check of the
# other Della launchers and a TASK switch. The fixed settings live in
# configs/experiment/b5_optimizer_<task>.yaml; the launcher varies only the swept triple
# (optimizer, weight_decay, base lr). Base lr is scaled by (100/N)^(1/3) = 0.464 at N = 1000.
#
#   TASK=ff (default)  +experiment=b5_optimizer_ff:   NBitFlipFlop, 3 bits, 200,000 iterations
#                      (what the Spock launcher runs), folder NBitFlipFlop_b5_optimizer
#   TASK=cddm          +experiment=b5_optimizer_cddm: CDDM, 75,000 iterations (the same-size
#                      budget of silent_units_largeN), folder CDDM_b5_optimizer
#
#   arm  optimizer  wd     base lr   why
#   1    adam       1e-6   1e-3      the line's default (coupled L2), the in-sweep control
#   2    adam       0      1e-3      Adam's normalisation without any decay
#   3    adamw      1e-6   1e-3      decoupled decay: w -= lr*wd*w, ~5e-10 per step
#   4-9  sgd        1e-6   0.03..10  same coupled L2 as the control, no normalisation
#   10-15 sgd       0      0.03..10  SGD with no decay at all
#
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..44): ARM = TID/3, REP = TID%3.
# The CDDM cells of the matrix are arms 3-7 (AdamW; SGD at wd 1e-6 and base lr 0.03, 0.1, 0.3
# and 1; SGD at lr 3 and 10 did not learn the flip-flop), indices 7-21, 3 seeds each. Adam at
# wd 0 exists in the CDDM weight-decay ladder, and the control is
# CDDM_paper_grid/EqType=h_N=1000_arm=control (Adam, wd 1e-6, base lr 1e-3, 100,000 iterations,
# the same model and trainer configs), so arms 1 and 2 are not run on CDDM.
#   TASK=cddm sbatch --chdir=$REPO --time=08:00:00 --constraint=nomig --array=7-21 slurm/SilentReLU_b5_optimizer_della.slurm
# Output: $SAVE_TO/<Task>_b5_optimizer/EqType=h_N=1000_opt=<o>_wd=<wd>_lr=<lr>/
# Cost: 75,000 iterations at N = 1000 take about 5.5 h on Della (0.24-0.30 s per iteration).

#SBATCH --job-name=B5opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=32G
#SBATCH --time=08:00:00
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

REPO="${REPO:-$HOME/trainRNNbrain_b5}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2
  echo "       submit with --chdir=\$REPO; python -c puts the working directory ahead of PYTHONPATH." >&2
  exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
if [ -z "$GITSHA" ]; then
  echo "FATAL: $REPO is not a git checkout - provenance of this run cannot be recorded." >&2; exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

# TASK selects the experiment config (task, budget) and the folder prefix (see the header).
case "${TASK:-ff}" in
  ff)   EXP=b5_optimizer_ff;   PREFIX=NBitFlipFlop ;;
  cddm) EXP=b5_optimizer_cddm; PREFIX=CDDM ;;
  *) echo "FATAL: TASK must be ff or cddm; got '${TASK}'" >&2; exit 1 ;;
esac

ARMS=("adam 1e-6 1e-3" "adam 0 1e-3" "adamw 1e-6 1e-3"
      "sgd 1e-6 0.03" "sgd 1e-6 0.1" "sgd 1e-6 0.3" "sgd 1e-6 1" "sgd 1e-6 3" "sgd 1e-6 10"
      "sgd 0 0.03" "sgd 0 0.1" "sgd 0 0.3" "sgd 0 1" "sgd 0 3" "sgd 0 10")
TID=$((SLURM_ARRAY_TASK_ID - 1))
read -r OPT WD LR <<< "${ARMS[$((TID / 3))]:?array index out of range}"
TAG="b5_optimizer/EqType=h_N=1000_opt=${OPT}_wd=${WD}_lr=${LR}"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${TASK:-ff} optimizer=$OPT wd=$WD base lr=$LR rep=$((TID % 3))"
echo "  -> $SAVE_TO/${PREFIX}_$TAG"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  "+experiment=$EXP" \
  trainer.optimizer="$OPT" trainer.weight_decay="$WD" trainer.lr="$LR" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"$TAG\""
