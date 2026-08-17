#!/bin/bash
# Train 5 GT-RNNs each for tanh slopes: 3.5, 3.0, 2.5, 2.0, 1.5
# 5 slopes x 5 networks = 25 array tasks, max_iter=50000.
#
# Task layout (TASK_IDX = SLURM_ARRAY_TASK_ID - 1):
#   SLOPE_IDX = TASK_IDX / 5   → 0=3.5, 1=3.0, 2=2.5, 3=2.0, 4=1.5
#   REP_IDX   = TASK_IDX % 5   → repeat index (0..4)
#
# Tasks  1- 5: tanh slope=3.5  (rnn_tanh_slope35_Dale)
# Tasks  6-10: tanh slope=3.0  (rnn_tanh_slope30_Dale)
# Tasks 11-15: tanh slope=2.5  (rnn_tanh_slope25_Dale)
# Tasks 16-20: tanh slope=2.0  (rnn_tanh_slope20_Dale)
# Tasks 21-25: tanh slope=1.5  (rnn_tanh_slope15_Dale)

#SBATCH --job-name=TrainGT_tanh_slopes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=10:00:00
#SBATCH --array=1-25
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/home/pt1290/trainRNNbrain/log/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain/log/%x.%A_%a.err

module purge
module load cudatoolkit/12.9
module load anaconda3/2024.10
module load openmpi/cuda-12.6/gcc/4.1.6
conda activate torch-env

MODEL_LIST=(rnn_tanh_slope35_Dale rnn_tanh_slope30_Dale rnn_tanh_slope25_Dale rnn_tanh_slope20_Dale rnn_tanh_slope15_Dale)
TAG_LIST=('tanh_slope=3.5' 'tanh_slope=3.0' 'tanh_slope=2.5' 'tanh_slope=2.0' 'tanh_slope=1.5')

TASK_IDX=$((SLURM_ARRAY_TASK_ID - 1))
SLOPE_IDX=$((TASK_IDX / 5))
REP_IDX=$((TASK_IDX % 5))

MODEL=${MODEL_LIST[$SLOPE_IDX]}
TAG=${TAG_LIST[$SLOPE_IDX]}

REPO="$HOME/trainRNNbrain"

echo "Task $SLURM_ARRAY_TASK_ID: model=$MODEL  repeat=$REP_IDX  tag=$TAG"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" paths=paths_DELLA model.N=1000 model="$MODEL" model.equation_type='h' \
  trainer.same_batch=True trainer.max_iter=50000 trainer.max_grad_norm=1.0 trainer.lambda_orth=0.0 trainer.lambda_frm=0.2 trainer.lambda_rws=0.05 \
  "experiment_tag=\"$TAG\""
