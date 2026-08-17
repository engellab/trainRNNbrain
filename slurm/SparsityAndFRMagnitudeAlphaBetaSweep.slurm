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
ALPHAS=(0 1)
BETAS=(0 1)
NS=(1000)
REPEATS_PER_SIZE=7
MAX_CONCURRENT=150

# ---- derived indexing (no need to edit) ----
A=${#ALPHAS[@]}; B=${#BETAS[@]}; S=${#NS[@]}
P=$((S * REPEATS_PER_SIZE))
AB=$((A * B))
TOTAL=$((AB * P))

if [[ -z "${SLURM_ARRAY_TASK_ID+x}" ]]; then
  CONCUR=$((TOTAL<MAX_CONCURRENT?TOTAL:MAX_CONCURRENT))
  echo "Submitting array 1-${TOTAL}%${CONCUR}  | alphas=${A} betas=${B} sizes=${S} repeats/size=${REPEATS_PER_SIZE} => TOTAL=${TOTAL}"
  exec sbatch --job-name="frm_ab" --array=1-${TOTAL}%${CONCUR} "$0"
fi

module purge
module load cudatoolkit/12.9
module load anaconda3/2024.10
module load openmpi/cuda-12.6/gcc/4.1.6
conda activate torch-env

TID=$SLURM_ARRAY_TASK_ID; Z=$((TID - 1))

AB_IDX=$((Z / P)); REM=$((Z % P))
N_IDX=$((REM / REPEATS_PER_SIZE)); REP_IDX=$((REM % REPEATS_PER_SIZE))

A_IDX=$((AB_IDX / B))
B_IDX=$((AB_IDX % B))

ALPHA=${ALPHAS[$A_IDX]}
BETA=${BETAS[$B_IDX]}
N=${NS[$N_IDX]}

REPO="$HOME/trainRNNbrain"
GITHASH="$(git -C "$REPO" rev-parse --short=6 HEAD 2>/dev/null || echo nogit)"

echo "frm_args.alpha=${ALPHA} | frm_args.beta=${BETA} | N=${N} | repeat=${REP_IDX} | task_id=${TID}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" paths=paths_DELLA model.N=${N} model="rnn_relu_Dale" model.equation_type='h' \
  trainer.same_batch=True trainer.max_iter=30000 trainer.lambda_orth=0.0 trainer.lambda_frm=0.2 \
  trainer.frm_args.alpha=${ALPHA} trainer.frm_args.beta=${BETA} trainer.lambda_rws=0.05 \
  experiment_tag="SparsityAndFRMag-ab-${GITHASH}"
