#!/bin/bash
#SBATCH --job-name=training_CDDM_RNNs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-100%25
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=../log/O-%x.%A_%a.out
#SBATCH --error=../log/E-%x.%A_%a.err

# Load modules for CUDA, CUDNN, and conda
module purge
module load anacondapy/2022.05

# Activate conda environment
conda activate RNN_training_pipeline

python /mnt/cup/people/pt1290/RNN_training_pipeline/src/train_configs/set_train_config_CDDM.py
python /mnt/cup/people/pt1290/RNN_training_pipeline/src/training_RNNs/training_CDDM.py