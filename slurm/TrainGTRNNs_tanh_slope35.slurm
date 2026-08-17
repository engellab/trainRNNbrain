#!/bin/bash
# Train 5 GT-RNNs with tanh activation, slope=3.5.
# Saves to experiment_tag "tanh_slope=3.5" (separate from SparsityAndFRMagnitude).
# Same hyperparameters as existing relu/tanh GT-RNNs.

#SBATCH --job-name=TrainGTRNN_tanh35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=6:01:00
#SBATCH --array=1-5
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/home/pt1290/trainRNNbrain/log/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain/log/%x.%A_%a.err

module purge
module load cudatoolkit/12.9
module load anaconda3/2024.10
module load openmpi/cuda-12.6/gcc/4.1.6
conda activate torch-env

REPO="$HOME/trainRNNbrain"

echo "Task $SLURM_ARRAY_TASK_ID: model=rnn_tanh_slope35_Dale  N=1000  repeat=$((SLURM_ARRAY_TASK_ID - 1))"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" paths=paths_DELLA model.N=1000 model="rnn_tanh_slope35_Dale" model.equation_type='h' \
  trainer.same_batch=True trainer.max_iter=30000 trainer.lambda_orth=0.0 trainer.lambda_frm=0.2 trainer.lambda_rws=0.05 \
  'experiment_tag="tanh_slope=3.5"'
