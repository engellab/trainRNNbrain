#!/bin/bash
# Noise control: ReLU, gamma=0, NO penalties (Lrws=0, Lfr=0), N=1000, sweeping the recurrent noise
# sigma_rec in {0, 0.01, 0.05, 0.1} (0.05 is the default used everywhere else). Tests whether the
# low-activity / silent population is driven by the injected training noise.
#
# Grid (40 jobs) = 2 eq x 4 sigma_rec x 5 nets, N=1000, none penalty.
#   TID = SLURM_ARRAY_TASK_ID-1 (0..39):  REP=TID%5; SIG_IDX=(TID/5)%4; EQ_IDX=(TID/20)%2
# Output: $SAVE_TO/CDDM_<hash>_g0_noise/EqType=<EQ>_N=1000_sigrec=<SIG>/<net>
# Overridable: MAXITER (default 30000), SAVE_TO.

#SBATCH --job-name=Silent_noise
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
EQS=(h s); SIGS=(0 0.01 0.05 0.1)
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 5)); SIG_IDX=$(((TID / 5) % 4)); EQ_IDX=$(((TID / 20) % 2))
EQ=${EQS[$EQ_IDX]}; SIG=${SIGS[$SIG_IDX]}

TAG="${HASH}_g0_noise/EqType=${EQ}_N=1000_sigrec=${SIG}"
echo "Task ${SLURM_ARRAY_TASK_ID}: noise EQ=${EQ} N=1000 sigma_rec=${SIG} rep=${REP} gamma=0 none -> CDDM_${TAG}"
srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_N1000 \
  model=rnn_relu_Dale model.equation_type="$EQ" model.sigma_rec="$SIG" \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
