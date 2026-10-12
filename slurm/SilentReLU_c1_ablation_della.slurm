#!/bin/bash
# CLAIM 1 ABLATION: remove units from trained standard controls and score them on held-out data. (DELLA)
#
# One array task per line of $LIST. Each line is
#   end   <run_dir>               ablate the run's end state (LastParams)
#   floor <run_dir> <snap_dir>    fit the held-out loss floor over its kept snapshots and ablate the
#                                 snapshot where the loss first reaches 1.05x that floor
# See trainRNNbrain/experiments_and_analysis/ablate_silent_units.py for the method.
#
# Per network at N = 1000, ~2,500 forward passes (9 noise draws x ~40 removal counts x 3 orders
# + 5 random orders); N = 4000 costs ~16x the FLOPs. No clean-tree check: this reads networks and
# writes JSON, it trains nothing.
#
#   LIST=/scratch/gpfs/TENGEL/pt1290/c1_ablation/lists/end.txt \
#     sbatch --array=1-$(wc -l < $LIST) slurm/SilentReLU_c1_ablation_della.slurm
#
#SBATCH --job-name=C1abl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/scratch/gpfs/TENGEL/pt1290/c1_ablation_logs/%x.%A_%a.out
#SBATCH --error=/scratch/gpfs/TENGEL/pt1290/c1_ablation_logs/%x.%A_%a.err

module purge
module load cudatoolkit/12.8
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env
set -eo pipefail
export PYTHONUNBUFFERED=1

REPO="${REPO:-$HOME/trainRNNbrain_c1ana}"
OUT="${OUT:-/scratch/gpfs/TENGEL/pt1290/c1_ablation}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
echo "code: $REPO @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
read -r READOUT RUN_DIR SNAP_DIR <<< "$LINE"
echo "Task ${SLURM_ARRAY_TASK_ID}: $READOUT $RUN_DIR ${SNAP_DIR:-}"
cd /scratch/gpfs/TENGEL/pt1290
python "$REPO/trainRNNbrain/experiments_and_analysis/ablate_silent_units.py" \
  "$READOUT" "$RUN_DIR" ${SNAP_DIR:+"$SNAP_DIR"} --out "$OUT"
