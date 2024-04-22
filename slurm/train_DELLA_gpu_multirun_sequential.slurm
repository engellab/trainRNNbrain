#!/bin/bash
#SBATCH --job-name="trainDMTS"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-10%10
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=15G
#SBATCH --time=12:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=../log/%x.%A_%a.out
#SBATCH --error=../log/%x.%A_%a.err

# Load modules for CUDA, CUDNN, and conda
module purge
module load cudatoolkit/12.0
module load cudnn/cuda-11.x/8.2.0
module load anaconda3/2022.5

# Activate conda environment
conda activate RNN_training_pipeline

# to run a script with specific parameters run the following command from src/slurm (as an example):
# sbatch --export=task="DMTS" train_DELLA_gpu_multirun.slurm

python ~/rnn_coach/src/training/run_training_sequential.py --multirun task=$task trainer=trainer_sequential model=rnn_sigmoid_Dale,rnn_relu_Dale,rnn_tanh_Dale,rnn_sigmoid_noDale,rnn_relu_noDale,rnn_tanh_noDale