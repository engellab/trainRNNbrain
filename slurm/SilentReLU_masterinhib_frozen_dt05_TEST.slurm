#!/bin/bash
# DIVERGENCE TEST — smaller integration step dt=0.5 (raises explicit-Euler stability boundary g<11 -> g<21),
# two variants: gamma=0.1 and gamma=0.0. Same task (n_steps auto-doubles to 600). Original diverging setup
# otherwise (frozen master, clamp=-5). Runs ONLY the frm arm (the divergent one) with max_iter=6000 (real
# divergences hit by iter ~4400) to distinct TEST folders. Success = ~0 nan for a variant.
#
# Grid (32 jobs) = 2 configs x 2 eq x 4 fracs x 2 seeds, frm=0.2, N=1000:
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..31):
#   CFG_IDX=TID/16 (0=dt05_g01, 1=dt05_g0);  t=TID%16;  SEED=t%2; FRAC_IDX=(t/2)%4; EQ_IDX=(t/8)%2
#
# Output: $SAVE_TO/CDDM_<hash>_masterinhib_frozen_<gtag>_test/EqType=<EQ>_N=1000_MIF=<FRAC>_LmbdFR=0.2/...

#SBATCH --job-name=SilentReLUdt
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=1-32
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

CFGS=(rnn_relu_Dale_masterinhib_frozen_dt05_g01 rnn_relu_Dale_masterinhib_frozen_dt05_g0)
GTAGS=(dt05_g01 dt05_g0)
EQS=(h s)
FRACS=(0.25 0.5 0.75 1.0)

TID=$((SLURM_ARRAY_TASK_ID - 1))
CFG_IDX=$((TID / 16))
T=$((TID % 16))
SEED_I=$((T % 2))
FRAC_IDX=$(((T / 2) % 4))
EQ_IDX=$(((T / 8) % 2))
CFG=${CFGS[$CFG_IDX]}
GTAG=${GTAGS[$CFG_IDX]}
EQ=${EQS[$EQ_IDX]}
FRAC=${FRACS[$FRAC_IDX]}

TAG="${HASH}_masterinhib_frozen_${GTAG}_test/EqType=${EQ}_N=1000_MIF=${FRAC}_LmbdFR=0.2"
echo "TEST task ${SLURM_ARRAY_TASK_ID}: cfg=${GTAG} EQ=${EQ} MIF=${FRAC} frm=0.2 seed_i=${SEED_I} hash=${HASH} (dt=0.5, max_iter=6000)"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_N1000 \
  model="$CFG" model.equation_type="$EQ" \
  model.master_inhib_frac="$FRAC" \
  trainer.lambda_rws=0 trainer.lambda_frm=0.2 \
  trainer.max_iter=6000 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
