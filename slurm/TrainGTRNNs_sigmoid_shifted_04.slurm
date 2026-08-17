#!/bin/bash
# Train 5 GT-RNNs: sigmoid with shift=0.4, N=1000.
# Same hyperparameters as the existing shift=0.3 networks.

#SBATCH --job-name=TrainSigShift04
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

echo "Task $SLURM_ARRAY_TASK_ID: model=rnn_sigmoid_shifted_04_Dale  N=1000"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" paths=paths_DELLA \
  model=rnn_sigmoid_shifted_04_Dale \
  model.N=1000 \
  model.equation_type='h' \
  trainer.same_batch=True \
  trainer.max_iter=30000 \
  trainer.lambda_orth=0.0 \
  trainer.lambda_frm=0.2 \
  trainer.lambda_rws=0.05 \
  experiment_tag="SparsityAndFRMagnitude-sigmoid_shifted_04"
