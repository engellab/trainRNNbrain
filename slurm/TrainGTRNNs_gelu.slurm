#!/bin/bash
# Train 10 GT-RNNs: GELU alpha=2 (5 nets) + GELU alpha=5 (5 nets), N=1000.
# Same regime as the existing relu/softplus/sigmoid GT-RNNs (Lfrm=0.2, Lrws=0.05, eq=h,
# spectral_rad=1.2). For the planned 5x5 activation comparison; pick the best of each class.
#   MODEL_IDX = TASK_IDX/5  (0=gelu_alpha2, 1=gelu_alpha5),  REP = TASK_IDX%5

#SBATCH --job-name=TrainGTRNN_gelu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=6:01:00
#SBATCH --array=1-10
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/home/pt1290/trainRNNbrain/log/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain/log/%x.%A_%a.err

module purge
module load cudatoolkit/12.9
module load anaconda3/2024.10
module load openmpi/cuda-12.6/gcc/4.1.6
conda activate torch-env

MODEL_LIST=(rnn_gelu_alpha2_Dale rnn_gelu_alpha5_Dale)

TASK_IDX=$((SLURM_ARRAY_TASK_ID - 1))
MODEL_IDX=$((TASK_IDX / 5))
REP_IDX=$((TASK_IDX % 5))
MODEL=${MODEL_LIST[$MODEL_IDX]}

REPO="$HOME/trainRNNbrain"

echo "Task $SLURM_ARRAY_TASK_ID: model=$MODEL  repeat=$REP_IDX"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" paths=paths_DELLA model.N=1000 model="$MODEL" model.equation_type='h' \
  trainer.same_batch=True trainer.max_iter=30000 trainer.lambda_orth=0.0 trainer.lambda_frm=0.2 trainer.lambda_rws=0.05 \
  experiment_tag="SparsityAndFRMagnitude-gelu"
