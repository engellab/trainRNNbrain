#!/bin/bash
# Train 15 GT-RNNs: sigmoid (5), softplus β=5 (5), softplus β=25 (5).
# All at N=1000, same hyperparameters as existing relu/tanh GT-RNNs.
#
# Task layout (TASK_IDX = SLURM_ARRAY_TASK_ID - 1):
#   MODEL_IDX = TASK_IDX / 5   → 0=sigmoid, 1=softplus_beta05, 2=softplus_beta25
#   REP_IDX   = TASK_IDX % 5   → repeat index within model type (0..4)
#
# Tasks  1- 5: sigmoid              (rnn_sigmoid_Dale)
# Tasks  6-10: softplus β=5         (rnn_softplus_Dale)
# Tasks 11-15: softplus β=25        (rnn_softplus_beta25_Dale)

#SBATCH --job-name=TrainGTRNN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=6:01:00
#SBATCH --array=1-15
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/home/pt1290/trainRNNbrain/log/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain/log/%x.%A_%a.err

module purge
module load cudatoolkit/12.9
module load anaconda3/2024.10
module load openmpi/cuda-12.6/gcc/4.1.6
conda activate torch-env

MODEL_LIST=(rnn_sigmoid_Dale rnn_softplus_Dale rnn_softplus_beta25_Dale)

TASK_IDX=$((SLURM_ARRAY_TASK_ID - 1))
MODEL_IDX=$((TASK_IDX / 5))
REP_IDX=$((TASK_IDX % 5))
MODEL=${MODEL_LIST[$MODEL_IDX]}

REPO="$HOME/trainRNNbrain"
GITHASH="$(git -C "$REPO" rev-parse --short=6 HEAD 2>/dev/null || echo nogit)"

echo "Task $SLURM_ARRAY_TASK_ID: model=$MODEL  repeat=$REP_IDX"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" paths=paths_DELLA model.N=1000 model="$MODEL" model.equation_type='h' \
  trainer.same_batch=True trainer.max_iter=30000 trainer.lambda_orth=0.0 trainer.lambda_frm=0.2 trainer.lambda_rws=0.05 \
  experiment_tag="SparsityAndFRMagnitude-${GITHASH}"
