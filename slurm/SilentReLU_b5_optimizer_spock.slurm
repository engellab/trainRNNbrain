#!/bin/bash
# B5, THE OPTIMISER CONTROL: is losing units after the loss floor specific to Adam? (SPOCK)
#
# Standard 3-bit flip-flop, N = 1000, 200k iterations, 3 seeds per arm; fixed settings live in
# configs/experiment/b5_optimizer_ff.yaml. This launcher varies only the swept triple
# (optimizer, weight_decay, base lr). Base lr is scaled by (100/N)^(1/3) = 0.464 at N = 1000.
#
#   arm  optimizer  wd     base lr   why
#   1    adam       1e-6   1e-3      the line's default (coupled L2), the in-sweep control
#   2    adam       0      1e-3      Adam's normalisation without any decay
#   3    adamw      1e-6   1e-3      decoupled decay: w -= lr*wd*w, ~5e-10 per step
#   4-9  sgd        1e-6   0.03..10  same coupled L2 as the control, no normalisation; the ladder
#                                    spans 300x because SGD's working rate here is unmeasured
#   10-15 sgd       0      0.03..10  SGD with no decay at all (added 13 Oct, after 1-27 were queued)
#
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..44): ARM = TID/3, REP = TID%3.
#   sbatch --array=1-45 SilentReLU_b5_optimizer_spock.slurm      (from the repo, --chdir=$REPO)
# Output: $SAVE_TO/NBitFlipFlop_b5_optimizer/EqType=h_N=1000_opt=<o>_wd=<wd>_lr=<lr>/
# Cost: 200k at N = 1000 is ~15 GPU-h (Della paper grid: 2.9 h per 40k); 72 h allows a slow node.

#SBATCH --job-name=B5opt
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=1-45
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

REPO="${REPO:-$HOME/trainRNNbrain_b5}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
if [ "$PKG" != "$(readlink -f "$REPO")" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO - submit with --chdir=\$REPO." >&2; exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
[ -n "$GITSHA" ] || { echo "FATAL: $REPO is not a git checkout." >&2; exit 1; }
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

ARMS=("adam 1e-6 1e-3" "adam 0 1e-3" "adamw 1e-6 1e-3"
      "sgd 1e-6 0.03" "sgd 1e-6 0.1" "sgd 1e-6 0.3" "sgd 1e-6 1" "sgd 1e-6 3" "sgd 1e-6 10"
      "sgd 0 0.03" "sgd 0 0.1" "sgd 0 0.3" "sgd 0 1" "sgd 0 3" "sgd 0 10")
TID=$((SLURM_ARRAY_TASK_ID - 1))
read -r OPT WD LR <<< "${ARMS[$((TID / 3))]:?array index out of range}"
TAG="b5_optimizer/EqType=h_N=1000_opt=${OPT}_wd=${WD}_lr=${LR}"
echo "Task ${SLURM_ARRAY_TASK_ID}: optimizer=$OPT wd=$WD base lr=$LR rep=$((TID % 3))"
echo "  -> $SAVE_TO/NBitFlipFlop_$TAG"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=b5_optimizer_ff \
  trainer.optimizer="$OPT" trainer.weight_decay="$WD" trainer.lr="$LR" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"$TAG\""
