#!/bin/bash
#SBATCH --job-name=hprprm_srch_DMTS_RNNs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=15G
#SBATCH --time=14:00:00
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

python ~/rnn_coach/src/train_configs/set_train_config_DMTS.py
python ~/rnn_coach/src/experiments_and_analysis/hyperparam_optimization_DMTS.py

