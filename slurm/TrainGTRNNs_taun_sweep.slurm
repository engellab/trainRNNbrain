#!/bin/bash
# tau_n aggregation sweep for the FR-magnitude penalty (with x.abs() REMOVED in Trainer.py:105, so
# the penalty now drives the SIGNED soft-max-over-time firing rate toward +cap, not |fr|).
#
# Goal: test how the neuron-level aggregation of the per-unit FR-deviation penalty affects the
# resulting GT-RNNs — does logsumexp (small tau_n ≈ max → "fight the worst outlier/hub units")
# yield more ablation-robust GTs than mean, and at what task-R² cost? Also yields a pool of fresh
# GELU(alpha=2) and tanh(slope=1) GT candidates to pick from.
#
# Grid: 2 activations × 4 aggregation conditions × 5 seeds = 40 jobs.
#   ACT_IDX  = TASK_IDX / 20      -> 0 = GELU alpha=2 (rnn_gelu_alpha2_Dale), 1 = tanh slope=1 (rnn_tanh_Dale)
#   COND_IDX = (TASK_IDX % 20)/5  -> 0 = mean ; 1,2,3 = logsumexp tau_n in {0.1, 1.0, 10.0}
#   REP      = TASK_IDX % 5       -> seed="random" (5 distinct nets per condition)
# Same regime as the other GTs: N=1000, eq=h, Lfrm=0.2, Lrws=0.05, cap_fr=0.3, max_iter=30000.

#SBATCH --job-name=TauNSweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=6:01:00
#SBATCH --array=1-40
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/home/pt1290/trainRNNbrain/log/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain/log/%x.%A_%a.err

module purge
module load cudatoolkit/12.9
module load anaconda3/2024.10
module load openmpi/cuda-12.6/gcc/4.1.6
conda activate torch-env

MODELS=(rnn_gelu_alpha2_Dale rnn_tanh_Dale)
ACT_SHORT=(gelu2 tanh1)
COND_NAMES=(mean taun0.1 taun1 taun5)
TAUN=(NA 0.1 1.0 5.0)             # index 0 unused (mean); 1..3 are the logsumexp tau_n values

TASK_IDX=$((SLURM_ARRAY_TASK_ID - 1))
ACT_IDX=$((TASK_IDX / 20))
COND_IDX=$(((TASK_IDX % 20) / 5))
REP_IDX=$((TASK_IDX % 5))
MODEL=${MODELS[$ACT_IDX]}
ASHORT=${ACT_SHORT[$ACT_IDX]}
COND=${COND_NAMES[$COND_IDX]}

# aggregation/tau_n are declared fields in configs/trainer/trainer.yaml (no '+' needed — standard override)
if [ "$COND_IDX" -eq 0 ]; then
  AGG_OVERRIDE="trainer.frm_args.aggregation=mean"
else
  AGG_OVERRIDE="trainer.frm_args.aggregation=logsumexp trainer.frm_args.tau_n=${TAUN[$COND_IDX]}"
fi

REPO="$HOME/trainRNNbrain"
echo "Task $SLURM_ARRAY_TASK_ID: model=$MODEL ($ASHORT)  cond=$COND  rep=$REP_IDX  override='$AGG_OVERRIDE'"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" paths=paths_DELLA model.N=1000 model="$MODEL" model.equation_type='h' \
  trainer.same_batch=True trainer.max_iter=${MAXITER:-30000} \
  trainer.lambda_orth=0.0 trainer.lambda_frm=0.2 trainer.lambda_rws=0.05 \
  $AGG_OVERRIDE \
  experiment_tag="TauNSweep-${ASHORT}-${COND}"
