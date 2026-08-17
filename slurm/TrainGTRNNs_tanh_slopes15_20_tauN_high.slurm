#!/bin/bash
# tau_n sweep (high values: 2, 5, 10) for tanh GT-RNNs with slopes 1.5 and 2.0.
# Explores the Pareto front between R² and participation in the region where
# log-sum-exp aggregation approaches mean (tau_n → ∞ = baseline).
# 2 slopes x 3 tau_n values x 3 seeds = 18 array tasks.
#
# Task layout (TASK_IDX = SLURM_ARRAY_TASK_ID - 1):
#   SLOPE_IDX = TASK_IDX / 9        → 0=slope1.5, 1=slope2.0
#   TAU_IDX   = (TASK_IDX % 9) / 3  → 0=tau_n=2, 1=tau_n=5, 2=tau_n=10
#   SEED      = (TASK_IDX % 3) + 1  → seed (1..3)

#SBATCH --job-name=TanhHighTauN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=10:00:00
#SBATCH --array=1-18
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/home/pt1290/trainRNNbrain/log/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain/log/%x.%A_%a.err

module purge
module load cudatoolkit/12.9
module load anaconda3/2024.10
module load openmpi/cuda-12.6/gcc/4.1.6
conda activate torch-env

MODEL_LIST=(rnn_tanh_slope15_Dale rnn_tanh_slope20_Dale)
TAU_N_LIST=(2.0 5.0 10.0)
SLOPE_TAG_LIST=(1.5 2.0)

TASK_IDX=$((SLURM_ARRAY_TASK_ID - 1))
SLOPE_IDX=$((TASK_IDX / 9))
LOCAL_IDX=$((TASK_IDX % 9))
TAU_IDX=$((LOCAL_IDX / 3))
SEED=$((LOCAL_IDX % 3 + 1))

MODEL=${MODEL_LIST[$SLOPE_IDX]}
TAU_N=${TAU_N_LIST[$TAU_IDX]}
SLOPE_TAG=${SLOPE_TAG_LIST[$SLOPE_IDX]}

REPO="$HOME/trainRNNbrain"

echo "Task $SLURM_ARRAY_TASK_ID: slope=$SLOPE_TAG  tau_n=$TAU_N  seed=$SEED"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed=$SEED paths=paths_DELLA model.N=1000 model="$MODEL" model.equation_type=h \
  trainer.same_batch=True trainer.max_iter=50000 trainer.max_grad_norm=1.0 \
  trainer.lambda_orth=0.0 trainer.lambda_frm=0.2 trainer.lambda_rws=0.05 \
  "++trainer.frm_args.aggregation=logsumexp" \
  "++trainer.frm_args.tau_n=$TAU_N" \
  "experiment_tag=\"tanh_slope=${SLOPE_TAG}_tauN=$TAU_N\""
