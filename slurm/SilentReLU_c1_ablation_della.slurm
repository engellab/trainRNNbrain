#!/bin/bash
# CLAIM 1 ABLATION: remove units from trained standard controls and score them on held-out data. (DELLA)
#
# One array task per line of $LIST. Each line is
#   end   <run_dir>               ablate the run's end state (LastParams)
#   floor <run_dir>               ablate the kept snapshot nearest where the run's noisy loss first
#                                 reaches 1.05x its fitted floor
# See trainRNNbrain/experiments_and_analysis/ablate_silent_units.py for the method.
#
# Per network at N = 1000, ~2,500 forward passes (9 noise draws x ~40 removal counts x 3 orders
# + 5 random orders); N = 4000 costs ~16x the FLOPs. No clean-tree check: this reads networks and
# writes JSON, it trains nothing.
#
#   LIST=/scratch/gpfs/TENGEL/pt1290/c1_ablation/lists/end.txt \
#     sbatch --array=1-$(wc -l < $LIST) slurm/SilentReLU_c1_ablation_della.slurm
# ABL_ARGS is appended to every list-mode call, e.g. ABL_ARGS="--at floor" for `retrain` lines.
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

# SNAPSHOT=<Task> instead of LIST: the runs of the claim-1 snapshot sweep for that task (CDDM,
# NBitFlipFlop or DMTS), found on disk when the task starts, so this array can be queued behind the
# training array with --dependency. SWEEP overrides the folder (the flip-flop is read from its 200k
# continuations, NBitFlipFlop_c1_snapshots_c200000). Each run is read at its noisy-loss floor -
# the kept snapshots are collected along its warm-start chain - and at its end.
#   SNAPSHOT=CDDM sbatch --dependency=afterany:<training job> --array=1-3 slurm/SilentReLU_c1_ablation_della.slurm
if [ -n "${SNAPSHOT:-}" ]; then
  DATA="${DATA:-$HOME/trainRNNbrain/data/trained_RNNs}"
  SWEEP="${SWEEP:-${SNAPSHOT}_c1_snapshots}"
  mapfile -t RUNS < <(ls -d "$DATA/$SWEEP"/EqType=*/*/ 2>/dev/null | sort)
  RUN_DIR="${RUNS[$((SLURM_ARRAY_TASK_ID - 1))]%/}"
  [ -n "$RUN_DIR" ] || { echo "FATAL: no run $SLURM_ARRAY_TASK_ID in $SWEEP" >&2; exit 1; }
  echo "Task ${SLURM_ARRAY_TASK_ID}: $RUN_DIR"
  python "$ABL" floor "$RUN_DIR" --out "$OUT"
  python "$ABL" end "$RUN_DIR" --out "$OUT"
  exit 0
fi

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
read -r READOUT RUN_DIR <<< "$LINE"
echo "Task ${SLURM_ARRAY_TASK_ID}: $READOUT $RUN_DIR"
python "$ABL" "$READOUT" "$RUN_DIR" --out "$OUT" ${ABL_ARGS:-}
