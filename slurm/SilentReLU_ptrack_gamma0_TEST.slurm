#!/bin/bash
# Participation-trace TEST run — ONE network, no penalties, N=1000, 3000 iterations.
#
# Purpose: validate the participation tracker on the cluster at the real network size before
# committing to the full 40-job sweep. With track_every=10 this yields 300 snapshots x 1000 units
# (~1.2 MB pkl). No penalties (lambda_rws=0, lambda_frm=0) = the condition that is expected to
# develop the silent mode, so 3000 iterations should already show the start of the bifurcation.
#
# Fixed params (N=1000, gamma=0 in the model config, empty trainer tag) come from
# +experiment=silent_units_N1000; trainer=trainer_ptrack turns tracking on. max_iter is the ONE
# CLI value override, explicitly sanctioned for smoke tests by that experiment config.
#
# Output: $SAVE_TO/CDDM_ptrack_TEST/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0/<per-net subfolder>
#   -> ..._ParticipationTrace.pkl  {"iters": [0,10,...], "participation": [array(N,), ...]}
# Overridable env vars: MAXITER (default 3000), SAVE_TO (default $HOME/trainRNNbrain/data/trained_RNNs).

#SBATCH --job-name=PtrackTEST
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/usr/people/pt1290/trainRNNbrain/log/%x.%j.out
#SBATCH --error=/usr/people/pt1290/trainRNNbrain/log/%x.%j.err

module purge
module load anacondapy/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env

set -eo pipefail
export PYTHONUNBUFFERED=1   # otherwise iteration lines sit in an 8 KB buffer and the log looks stalled

REPO="$HOME/trainRNNbrain"
SAVE_TO="${SAVE_TO:-$REPO/data/trained_RNNs}"
MAXITER="${MAXITER:-3000}"
TAG="ptrack_TEST/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0"

# descriptor of what this data is, next to the data itself
mkdir -p "${SAVE_TO}/CDDM_ptrack_TEST"
cp "$REPO/docs/experiments/participation_trace.md" "${SAVE_TO}/CDDM_ptrack_TEST/EXPERIMENT.md"

echo "ptrack TEST: eq=h N=1000 Lrws=0 Lfrm=0 max_iter=${MAXITER} hash=$(git -C "$REPO" rev-parse --short=6 HEAD)"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_N1000 \
  trainer=trainer_ptrack \
  model=rnn_relu_Dale model.equation_type=h \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  trainer.max_iter="$MAXITER" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
