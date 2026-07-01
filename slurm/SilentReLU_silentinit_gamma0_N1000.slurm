#!/bin/bash
# Silent-ReLU — deliberate silent-at-init test (prevention vs resurrection), gamma=0, N=1000.
#
# Init is perturbed so a fixed random 25% of units (set S, reconstructable from the net's seed) are
# SILENT at initialisation, via model=rnn_relu_Dale_silentinit (inhibitory_boost=2.0, silent_init_frac=0.25;
# see calibrate_inhibitory_boost.py + docs/project_trajectory.md). We then train with/without frm and
# track set S: if S stays silent -> frm PREVENTS; if S climbs into the active mode -> frm RESURRECTS.
#
# Grid (20 jobs) = 2 eq x 2 penalty x 5 nets, N=1000, lambda_rws fixed 0:
#   EQ  in {h, s}            (model.equation_type)
#   FRM in {0, 0.2}          (trainer.lambda_frm)  -> penalty: none (control) vs frm (rescuer)
#   5 reps, seed="random"
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..19): TID = 10*EQ + 5*FRM + REP
#   REP=TID%5; FRM_IDX=(TID/5)%2; EQ_IDX=(TID/10)%2
#
# Output: $SAVE_TO/CDDM_<hash>_g0_silentinit/EqType=<EQ>_N=1000_LmbdRWS=0_LmbdFR=<FRM>/<per-net subfolder>
# Overridable env vars: SAVE_TO (default $HOME/trainRNNbrain/data/trained_RNNs).

#SBATCH --job-name=SilentReLUsi
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --array=1-20
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
FRMS=(0 0.2)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 5))
FRM_IDX=$(((TID / 5)  % 2))
EQ_IDX=$(((TID / 10) % 2))
EQ=${EQS[$EQ_IDX]}
FRM=${FRMS[$FRM_IDX]}

TAG="${HASH}_g0_silentinit/EqType=${EQ}_N=1000_LmbdRWS=0_LmbdFR=${FRM}"
echo "Task ${SLURM_ARRAY_TASK_ID}: EQ=${EQ} N=1000 Lrws=0 Lfrm=${FRM} rep=${REP} hash=${HASH}"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

# Fixed params (gamma=0, N=1000, inhibitory_boost=2.0, silent_init_frac=0.25) live in config; this
# only sets the swept IVs (equation type, frm penalty).
srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_N1000 \
  model=rnn_relu_Dale_silentinit model.equation_type="$EQ" \
  trainer.lambda_rws=0 trainer.lambda_frm="$FRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
