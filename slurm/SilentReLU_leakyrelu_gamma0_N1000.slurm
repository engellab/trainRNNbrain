#!/bin/bash
# Activation control: LEAKY-RELU (leak 0.01), gamma=0, N=1000. Same grid/regime as the relu gamma=0
# sweep; only the activation changes. Leaky-ReLU has nonzero gradient for x<0 (no dead-gradient
# trap), the other control alongside softplus(beta=25).
#
# Grid (40 jobs) = 2 eq x 2 lambda_rws x 2 lambda_frm x 5 nets, N=1000.
#   TID = SLURM_ARRAY_TASK_ID-1 (0..39) = 20*EQ + 10*RWS + 5*FRM + REP
# Output: $SAVE_TO/CDDM_<hash>_g0_leakyrelu/EqType=<EQ>_N=1000_LmbdRWS=<RWS>_LmbdFR=<FRM>/<net>
# Overridable: MAXITER (default 30000), SAVE_TO.

#SBATCH --job-name=Silent_lrelu
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --array=1-40
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/usr/people/pt1290/trainRNNbrain/log/%x.%A_%a.out
#SBATCH --error=/usr/people/pt1290/trainRNNbrain/log/%x.%A_%a.err

module purge
module load anacondapy/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env
set -eo pipefail

REPO="$HOME/trainRNNbrain"
SAVE_TO="${SAVE_TO:-$REPO/data/trained_RNNs}"
HASH=$(git -C "$REPO" rev-parse --short=6 HEAD)
EQS=(h s); RWSS=(0 0.05); FRMS=(0 0.2)
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 5)); FRM_IDX=$(((TID / 5) % 2)); RWS_IDX=$(((TID / 10) % 2)); EQ_IDX=$(((TID / 20) % 2))
EQ=${EQS[$EQ_IDX]}; RWS=${RWSS[$RWS_IDX]}; FRM=${FRMS[$FRM_IDX]}

TAG="${HASH}_g0_leakyrelu/EqType=${EQ}_N=1000_LmbdRWS=${RWS}_LmbdFR=${FRM}"
echo "Task ${SLURM_ARRAY_TASK_ID}: leakyrelu EQ=${EQ} N=1000 Lrws=${RWS} Lfrm=${FRM} rep=${REP} gamma=0 -> CDDM_${TAG}"
srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_N1000 \
  model=rnn_leaky_relu_Dale model.equation_type="$EQ" \
  trainer.lambda_rws="$RWS" trainer.lambda_frm="$FRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
