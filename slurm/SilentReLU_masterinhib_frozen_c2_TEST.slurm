#!/bin/bash
# DIVERGENCE TEST (not the full analysis) — milder clamp master_inhib_strength=2.0, N=1000.
#
# Cheap check of whether lowering the clamp -5 -> -2 removes the NaN divergence before committing the full
# 48-job run. Runs ONLY the frm arm (the divergent one; `none` never diverged) with a reduced max_iter=6000
# (real divergences hit at iter 800-1600, so this covers the window ~4x) to a distinct TEST folder.
# Success = ~0/16 nets end with folder score "nan".
#
# Grid (16 jobs) = 2 eq x 4 fracs x 2 seeds, frm=0.2 fixed, N=1000:
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..15): TID = 8*EQ + 2*FRAC + SEED
#   SEED=TID%2; FRAC_IDX=(TID/2)%4; EQ_IDX=(TID/8)%2
#
# Output: $SAVE_TO/CDDM_<hash>_masterinhib_frozen_c2_test/EqType=<EQ>_N=1000_MIF=<FRAC>_LmbdFR=0.2/...

#SBATCH --job-name=SilentReLUc2t
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=1-16
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
SEED_I=$((TID % 2))
FRAC_IDX=$(((TID / 2) % 4))
EQ_IDX=$(((TID / 8) % 2))
EQ=${EQS[$EQ_IDX]}
FRAC=${FRACS[$FRAC_IDX]}

TAG="${HASH}_masterinhib_frozen_c2_test/EqType=${EQ}_N=1000_MIF=${FRAC}_LmbdFR=0.2"
echo "TEST task ${SLURM_ARRAY_TASK_ID}: EQ=${EQ} MIF=${FRAC} frm=0.2 seed_i=${SEED_I} hash=${HASH} (clamp=-2, max_iter=6000)"

# clamp=2.0 lives in the config; this is a divergence test so it also shortens max_iter (test-only override).
srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_N1000 \
  model=rnn_relu_Dale_masterinhib_frozen_gamma_c2 model.equation_type="$EQ" \
  model.master_inhib_frac="$FRAC" \
  trainer.lambda_rws=0 trainer.lambda_frm=0.2 \
  trainer.max_iter=6000 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
