#!/bin/bash
# tau_n sweep for tanh GT-RNNs with slopes 1.5 and 2.0.
# These slopes train successfully (R²~0.80), so we can meaningfully test
# whether logsumexp neuron aggregation reduces outlier/silent units.
# Fixed: cap_fr=0.3, g_top=3.0, g_bot=3.0, lambda_frm=0.2.
# 2 slopes x 5 tau_n values x 3 seeds = 30 array tasks.
#
# Task layout (TASK_IDX = SLURM_ARRAY_TASK_ID - 1):
#   SLOPE_IDX = TASK_IDX / 15       → 0=slope1.5, 1=slope2.0
#   TAU_IDX   = (TASK_IDX % 15) / 3 → tau_n index (0..4)
#   SEED      = (TASK_IDX % 3) + 1  → seed (1..3)
#
# Tasks  1-15: tanh slope=1.5  (rnn_tanh_slope15_Dale)
# Tasks 16-30: tanh slope=2.0  (rnn_tanh_slope20_Dale)
# Within each slope group, tau_n cycles: 1.0(x3), 0.3(x3), 0.1(x3), 0.03(x3), 0.01(x3)

#SBATCH --job-name=TanhSlopes_tauN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=10:00:00
#SBATCH --array=1-30
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
TAU_N_LIST=(1.0 0.3 0.1 0.03 0.01)
SLOPE_TAG_LIST=(1.5 2.0)

TASK_IDX=$((SLURM_ARRAY_TASK_ID - 1))
SLOPE_IDX=$((TASK_IDX / 15))
LOCAL_IDX=$((TASK_IDX % 15))
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
