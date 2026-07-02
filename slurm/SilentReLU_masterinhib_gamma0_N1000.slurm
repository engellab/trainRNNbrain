#!/bin/bash
# Silent-ReLU — master-inhibitor silent-at-init test (clean "no gradient -> no rescue"), gamma=0, N=1000.
#
# One inhibitory unit, driven only by the two CDDM context cues, holds a fixed random fraction of the
# other units silent by deep context-locked inhibition (model=rnn_relu_Dale_masterinhib; see
# calibrate_master_inhibitor.py + docs/project_trajectory.md). The clamp is unreachable by gradients, so
# unlike the inhibitory_boost run these units are truly unrescuable. We sweep the silenced fraction and
# train with/without frm: does frm revive the master-clamped units, or (as predicted) not?
#
# Grid (48 jobs) = 2 eq x 4 fracs x 2 penalty x 3 nets, N=1000:
#   EQ   in {h, s}                 (model.equation_type)
#   FRAC in {0.25,0.5,0.75,1.0}    (model.master_inhib_frac)  -> {quarter ... all-but-master} silenced
#   FRM  in {0, 0.2}               (trainer.lambda_frm)       -> none (control) vs frm (rescuer)
#   3 reps, seed="random"
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..47): TID = 24*EQ + 6*FRAC + 3*FRM + REP
#   REP=TID%3; FRM_IDX=(TID/3)%2; FRAC_IDX=(TID/6)%4; EQ_IDX=(TID/24)%2
#
# Output: $SAVE_TO/CDDM_<hash>_g0_masterinhib/EqType=<EQ>_N=1000_MIF=<FRAC>_LmbdFR=<FRM>/<per-net subfolder>
# Overridable env vars: SAVE_TO (default $HOME/trainRNNbrain/data/trained_RNNs).

#SBATCH --job-name=SilentReLUmi
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --array=1-48
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
FRMS=(0 0.2)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3))
FRM_IDX=$(((TID / 3)  % 2))
FRAC_IDX=$(((TID / 6) % 4))
EQ_IDX=$(((TID / 24) % 2))
EQ=${EQS[$EQ_IDX]}
FRAC=${FRACS[$FRAC_IDX]}
FRM=${FRMS[$FRM_IDX]}

TAG="${HASH}_g0_masterinhib/EqType=${EQ}_N=1000_MIF=${FRAC}_LmbdFR=${FRM}"
echo "Task ${SLURM_ARRAY_TASK_ID}: EQ=${EQ} N=1000 MIF=${FRAC} Lfrm=${FRM} rep=${REP} hash=${HASH}"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

# Fixed params (gamma=0, N=1000, master_inhib_strength, master_ctx_drive) live in config; this only sets
# the swept IVs (equation type, silenced fraction, frm penalty).
srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_N1000 \
  model=rnn_relu_Dale_masterinhib model.equation_type="$EQ" \
  model.master_inhib_frac="$FRAC" \
  trainer.lambda_rws=0 trainer.lambda_frm="$FRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
