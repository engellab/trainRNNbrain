#!/bin/bash
# METABOLIC-COST sweep: does the field-standard regularizer rescue silent units, or deepen them?
#
# Activity regularization is routine in this literature — metabolic-cost terms of the form
# mean(fr^2) are standard practice. But that penalizes rate MAGNITUDE, pushing the whole population
# down, so mechanistically it should make silence WORSE rather than better — the same way the
# sparsity penalty did in Dale networks. If standard practice actively worsens the pathology it is
# assumed to guard against, that is a far stronger finding than "here is another penalty".
#
# The sweep is over lambda_met rather than a single value ON PURPOSE: reporting the silent fraction
# as a FUNCTION of lambda removes the "you chose lambda badly" objection. Scale check: with rates
# ~0.3-0.5, mean(fr^2) ~ 0.05-0.25 against a task loss of ~0.02-0.09, so lambda=1 is roughly equal
# weight and the four decades bracket it symmetrically.
#
# Grid (36 jobs) = 4 lambda_met x 3 sizes x 3 seeds, h equation, no other penalties:
#   LAM in {0.01, 0.1, 1.0, 10.0}      N in {100, 500, 1000}      3 reps, seed="random"
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..35):
#   REP=TID%3; N_IDX=(TID/3)%3; LAM_IDX=(TID/9)%4
#
# Comparisons already in hand at the same settings: lambda=0 (CDDM_std_g0 / CDDM_std_g0_Nsweep,
# 'none') and the firing-rate penalty (same sweeps, 'frm', 0% silent at every size).
#
# Output: $SAVE_TO/CDDM_std_g0_metabolic/EqType=h_N=<N>_LmbdMet=<LAM>/<per-net subfolder>
# Descriptor: docs/experiments/standard_rnn_metabolic.md
#
# Wall time: with monitor=false and task_safe_gradients=false the penalties are summed into ONE
# backward pass, so an extra penalty costs almost nothing — these run at 'none' speed (0.144 s/iter
# at N=1000, i.e. ~1.2 h for 30000 iterations). 5 h is generous.

#SBATCH --job-name=StdMetabolic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=5:00:00
#SBATCH --constraint=nomig
#SBATCH --array=1-36
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

REPO="${REPO:-$HOME/trainRNNbrain_met}"
SAVE_TO="${SAVE_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.dirname(os.path.dirname(trainRNNbrain.__file__)))")
if [ "$PKG" != "$REPO" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO — aborting." >&2; exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short=7 HEAD)"

LAMS=(0.01 0.1 1.0 10.0); NS=(100 500 1000)
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); N_IDX=$(((TID / 3) % 3)); LAM_IDX=$(((TID / 9) % 4))
LAM=${LAMS[$LAM_IDX]}; N=${NS[$N_IDX]}

TAG="std_g0_metabolic/EqType=h_N=${N}_LmbdMet=${LAM}"
mkdir -p "${SAVE_TO}/CDDM_std_g0_metabolic"
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  cp "$REPO/docs/experiments/standard_rnn_metabolic.md" "${SAVE_TO}/CDDM_std_g0_metabolic/EXPERIMENT.md"
  echo -e "\n---\nRun: SLURM array ${SLURM_ARRAY_JOB_ID} on Della, code commit $(git -C "$REPO" rev-parse HEAD)" \
    >> "${SAVE_TO}/CDDM_std_g0_metabolic/EXPERIMENT.md"
fi
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} lambda_met=${LAM} rep=${REP}"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_std \
  model=rnn_relu_standard trainer=trainer_ptrack_plain \
  model.equation_type=h model.N="$N" \
  trainer.lambda_met="$LAM" trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
