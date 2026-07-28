#!/bin/bash
# HORIZON: how long until training actually stops changing the network? (DELLA)
#
# 30000 iterations is NOT convergence. Measured on the standard-RNN reference sweep: the training
# loss is flat over the last 10% (-0.4 to -0.6%) at every size, yet the hard-silent fraction is
# still climbing when training stops — +0.2 pp (N=100), +2.9 (N=500), +3.7 (N=1000), +3.1 (N=2000)
# over the final 5000 iterations. The loss has converged; the network has not. It keeps switching
# units off while task performance stays constant.
#
# This matters beyond bookkeeping: residual drift GROWS with N, so larger networks sit further from
# their asymptote at a fixed iteration count. That undercounts silence at large N, overcounts active
# units at large N, and therefore biases the saturation curve toward "growth" — which is exactly the
# result we got. The learning rate scaling lr*(100/N)^0.333 is the obvious mechanism: a 10000-unit
# net needs ~2.2x more steps to move its weights as far as a 1000-unit one.
#
# So: find the horizon at two sizes BEFORE spending ~185 GPU-hours on N=5000/10000 at a length we
# already know is too short.
#
# Grid (12 jobs) = 2 sizes x 2 weight_decay x 3 seeds, h equation, no penalties, 100000 iterations:
#   N in {1000, 2000}      weight_decay in {1e-6, 0}      3 reps, seed="random"
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..11): REP=TID%3; WD_IDX=(TID/3)%2; N_IDX=(TID/6)%2
#
# weight_decay is the second variable because Adam's 1e-6 decay keeps acting after the loss
# plateaus, and is a plausible driver of the late-phase silencing. If wd=0 stops the drift, the slow
# silencing is a regularisation artifact rather than something the task demands.
#
# Two measurements come out of every run:
#   - new silent units per 1000 iterations, derived from the participation trace (no extra logging)
#   - relative parameter drift ||W(t)-W(t-delta)||_F / ||W(t)||_F per tracked step, for W_rec, W_inp,
#     W_out and bias — new, stored under the "drift" key of the participation trace pkl
#
# Runtime at measured rates: N=1000 0.144 s/iter -> 4.0 h; N=2000 0.317 s/iter -> 8.8 h.

#SBATCH --job-name=StdHorizon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=14:00:00
#SBATCH --constraint=nomig
#SBATCH --array=1-12
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

REPO="${REPO:-$HOME/trainRNNbrain_horizon}"
SAVE_TO="${SAVE_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.dirname(os.path.dirname(trainRNNbrain.__file__)))")
if [ "$PKG" != "$REPO" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO — aborting." >&2; exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short=7 HEAD)"

NS=(1000 2000); WDS=(1e-6 0)
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); WD_IDX=$(((TID / 3) % 2)); N_IDX=$(((TID / 6) % 2))
N=${NS[$N_IDX]}; WD=${WDS[$WD_IDX]}

TAG="std_g0_horizon/EqType=h_N=${N}_wd=${WD}"
mkdir -p "${SAVE_TO}/CDDM_std_g0_horizon"
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  cp "$REPO/docs/experiments/standard_rnn_horizon.md" "${SAVE_TO}/CDDM_std_g0_horizon/EXPERIMENT.md"
  echo -e "\n---\nRun: SLURM array ${SLURM_ARRAY_JOB_ID} on Della, code commit $(git -C "$REPO" rev-parse HEAD)" \
    >> "${SAVE_TO}/CDDM_std_g0_horizon/EXPERIMENT.md"
fi
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} weight_decay=${WD} rep=${REP} iters=100000"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  model=rnn_relu_standard trainer=trainer_ptrack_plain \
  model.equation_type=h model.N="$N" \
  trainer.max_iter=100000 trainer.weight_decay="$WD" \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
