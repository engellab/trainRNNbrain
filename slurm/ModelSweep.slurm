#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=6:01:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=../log/%x.%A_%a.out
#SBATCH --error=../log/%x.%A_%a.err
#SBATCH --exclude=della-l01g[8-9]

set -e -o pipefail

# ---- sweep config (edit these) ----
SWEEP_KEY="model"  # just a placeholder since we're not sweeping any hyperparameters here
VALS=("rnn_relu_Dale" "rnn_leaky_relu_Dale" "rnn_sigmoid_Dale" "rnn_softplus_Dale")        # values for SWEEP_KEY
NS=(1000)                      # network sizes
REPEATS_PER_SIZE=7                # 7 networks per size
MAX_CONCURRENT=150

# ---- derived indexing (no need to edit) ----
V=${#VALS[@]}; S=${#NS[@]}; P=$((S*REPEATS_PER_SIZE)); TOTAL=$((V*P))
if [[ -z "${SLURM_ARRAY_TASK_ID+x}" ]]; then
  CONCUR=$((TOTAL<MAX_CONCURRENT?TOTAL:MAX_CONCURRENT))
  echo "Submitting array 1-${TOTAL}%${CONCUR}  | values=${V}  sizes=${S}  repeats/size=${REPEATS_PER_SIZE}  => TOTAL=${TOTAL}"
  exec sbatch --job-name="${SWEEP_KEY}" --array=1-${TOTAL}%${CONCUR} "$0"
fi

module purge
module load cudatoolkit/12.9
module load anaconda3/2024.10
module load openmpi/cuda-12.6/gcc/4.1.6
conda activate torch-env

TID=$SLURM_ARRAY_TASK_ID; Z=$((TID-1))
VAL_IDX=$((Z/P)); REM=$((Z%P))
N_IDX=$((REM/REPEATS_PER_SIZE)); REP_IDX=$((REM%REPEATS_PER_SIZE))
VAL=${VALS[$VAL_IDX]}; N=${NS[$N_IDX]}

REPO="$HOME/trainRNNbrain"
GITHASH="$(git -C "$REPO" rev-parse --short=6 HEAD 2>/dev/null || echo nogit)"

echo "param: ${SWEEP_KEY}=${VAL}  | N=${N}  | repeat=${REP_IDX}  | task_id=${TID}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" paths=paths_DELLA model.N=${N} model=${VAL} model.equation_type='h' \
  trainer.same_batch=True trainer.max_iter=30000 trainer.lambda_orth=0.0 trainer.lambda_frm=0.0 trainer.lambda_rws=0.0 \
  experiment_tag="${VAL}-${GITHASH}"
