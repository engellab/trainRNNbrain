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

cd /scratch/gpfs/TENGEL/pt1290
ABL="$REPO/trainRNNbrain/experiments_and_analysis/ablate_silent_units.py"

# SNAPSHOT=<Task> instead of LIST: the runs of slurm/SilentReLU_c1_snapshots_della.slurm for that task
# (CDDM, NBitFlipFlop or DMTS), found on disk when the task starts, so this array can be queued
# behind the training array with --dependency. Each run is read at its floor (both loss series)
# and at its end, so the same network is compared across the two read-outs.
#   SNAPSHOT=CDDM sbatch --dependency=afterany:<training job> --array=1-3 slurm/SilentReLU_c1_ablation_della.slurm
if [ -n "${SNAPSHOT:-}" ]; then
  DATA="${DATA:-$HOME/trainRNNbrain/data/trained_RNNs}"
  SNAP_TO="${SNAP_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots}"
  mapfile -t RUNS < <(ls -d "$DATA/${SNAPSHOT}_c1_snapshots"/EqType=*/*/ 2>/dev/null | sort)
  RUN_DIR="${RUNS[$((SLURM_ARRAY_TASK_ID - 1))]%/}"
  [ -n "$RUN_DIR" ] || { echo "FATAL: no run $SLURM_ARRAY_TASK_ID for $SNAPSHOT" >&2; exit 1; }
  SEED=$(basename "$RUN_DIR" | sed -n 's/.*seed=\([0-9]*\).*/\1/p')
  SNAP_DIR="$SNAP_TO/${SNAPSHOT}_c1_snapshots/$(basename "$(dirname "$RUN_DIR")")/s$SEED"
  echo "Task ${SLURM_ARRAY_TASK_ID}: $RUN_DIR"
  echo "  snapshots: $SNAP_DIR ($(ls -d "$SNAP_DIR"/it* 2>/dev/null | wc -l) kept)"
  python "$ABL" floor "$RUN_DIR" "$SNAP_DIR" --out "$OUT"
  python "$ABL" end "$RUN_DIR" --out "$OUT"
  exit 0
fi

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
read -r READOUT RUN_DIR SNAP_DIR <<< "$LINE"
echo "Task ${SLURM_ARRAY_TASK_ID}: $READOUT $RUN_DIR ${SNAP_DIR:-}"
python "$ABL" "$READOUT" "$RUN_DIR" ${SNAP_DIR:+"$SNAP_DIR"} --out "$OUT"
