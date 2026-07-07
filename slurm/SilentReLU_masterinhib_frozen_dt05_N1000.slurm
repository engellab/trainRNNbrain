#!/bin/bash
# Silent-ReLU — frozen master-inhibitor, STABILIZED with dt=0.5 + gamma=0.1 (the combination that cut the
# NaN divergence from ~50% to ~12%; dt=0.5 alone was 100%, gamma=0.1 alone was ~50%). Full rescue analysis.
# Same task as the other frozen runs (n_steps auto-doubles to 600 via T/dt). Model config carries dt=0.5,
# gamma=0.1, frozen master, clamp=-5 (rnn_relu_Dale_masterinhib_frozen_dt05_g01.yaml).
#
# Grid (64 jobs) = 2 eq x 4 fracs x (3 none + 5 frm) seeds, N=1000:
#   EQ   in {h, s}                 (model.equation_type)
#   FRAC in {0.25,0.5,0.75,1.0}    (model.master_inhib_frac)
#   penalty: 3 seeds for none (never diverges) + 5 seeds for frm (~19% NaN -> ~4 valid/cond) = 8 per cell
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..63): BLOCK=TID/8 (0..7), W=TID%8
#   FRAC_IDX=BLOCK%4; EQ_IDX=BLOCK/4;  W<3 -> FRM=0,REP=W ;  W>=3 -> FRM=0.2,REP=W-3
#
# Output: $SAVE_TO/CDDM_<hash>_masterinhib_frozen_dt05/EqType=<EQ>_N=1000_MIF=<FRAC>_LmbdFR=<FRM>/...
# Overridable env vars: SAVE_TO (default $HOME/trainRNNbrain/data/trained_RNNs).

#SBATCH --job-name=SilentReLUdtf
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-64
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

EQS=(h s)
FRACS=(0.25 0.5 0.75 1.0)

TID=$((SLURM_ARRAY_TASK_ID - 1))
BLOCK=$((TID / 8))
W=$((TID % 8))
FRAC_IDX=$((BLOCK % 4))
EQ_IDX=$((BLOCK / 4))
EQ=${EQS[$EQ_IDX]}
FRAC=${FRACS[$FRAC_IDX]}
if [ "$W" -lt 3 ]; then FRM=0; REP=$W; else FRM=0.2; REP=$((W - 3)); fi

TAG="${HASH}_masterinhib_frozen_dt05/EqType=${EQ}_N=1000_MIF=${FRAC}_LmbdFR=${FRM}"
echo "Task ${SLURM_ARRAY_TASK_ID}: EQ=${EQ} N=1000 MIF=${FRAC} Lfrm=${FRM} rep=${REP} hash=${HASH} (dt=0.5, gamma=0.1, frozen)"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

# Fixed params (dt=0.5, gamma=0.1, N=1000, master_*, freeze_master) live in config; this only sets swept IVs.
srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_N1000 \
  model=rnn_relu_Dale_masterinhib_frozen_dt05_g01 model.equation_type="$EQ" \
  model.master_inhib_frac="$FRAC" \
  trainer.lambda_rws=0 trainer.lambda_frm="$FRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
