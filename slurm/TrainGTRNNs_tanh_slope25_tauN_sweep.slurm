#!/bin/bash
# Sweep over tau_n (log-sum-exp neuron aggregation) for tanh slope=2.5 GT-RNNs.
# Fixed: cap_fr=0.3, g_top=3.0, g_bot=3.0, lambda_frm=0.2 (same as baseline).
# 5 tau_n values x 3 seeds = 15 array tasks, same 3 seeds for every tau_n.
#
# tau_n controls neuron aggregation in the FRM penalty:
#   large tau_n (1.0)  -> mean (current behaviour)
#   small tau_n (0.01) -> max (fully outlier-sensitive)
#
# Task layout (TASK_IDX = SLURM_ARRAY_TASK_ID - 1):
#   TAU_IDX = TASK_IDX / 3   -> tau_n index (0..4)
#   SEED    = TASK_IDX % 3 + 1 -> seed (1..3)
#
# Tasks  1- 3: tau_n=1.0
# Tasks  4- 6: tau_n=0.3
# Tasks  7- 9: tau_n=0.1
# Tasks 10-12: tau_n=0.03
# Tasks 13-15: tau_n=0.01

#SBATCH --job-name=TanhSlope25_tauN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=10:00:00
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

TAU_N_LIST=(1.0 0.3 0.1 0.03 0.01)

TASK_IDX=$((SLURM_ARRAY_TASK_ID - 1))
TAU_IDX=$((TASK_IDX / 3))
SEED=$((TASK_IDX % 3 + 1))
TAU_N=${TAU_N_LIST[$TAU_IDX]}

REPO="$HOME/trainRNNbrain"

echo "Task $SLURM_ARRAY_TASK_ID: tau_n=$TAU_N  seed=$SEED"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed=$SEED paths=paths_DELLA model.N=1000 model=rnn_tanh_slope25_Dale model.equation_type=h \
  trainer.same_batch=True trainer.max_iter=50000 trainer.max_grad_norm=1.0 \
  trainer.lambda_orth=0.0 trainer.lambda_frm=0.2 trainer.lambda_rws=0.05 \
  "++trainer.frm_args.aggregation=logsumexp" \
  "++trainer.frm_args.tau_n=$TAU_N" \
  "experiment_tag=\"tanh_slope=2.5_tauN=$TAU_N\""
