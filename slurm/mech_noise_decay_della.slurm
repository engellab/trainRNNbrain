#!/bin/bash
# MECHANISM TESTS, PART A: training noise x weight decay on CDDM, N = 1,000 (DELLA)
#
# Rules: docs/project_trajectory/2026-11-25_*_rules-mechanism-noise-decay-and-switch.md, committed before any run.
# Two factors, four cells, the same three seeds in every cell (so the four networks of a seed start from the same
# weights): training noise at the config values or all three levels at 0; Adam's coupled weight decay 1e-6 or 0.
# Every fixed value is in configs/experiment/mech_noise_decay.yaml; this script sets the swept values only.
#
# GRID: 4 cells x 3 seeds = 12 jobs. CELL = (TID-1)/3, SEED = (TID-1)%3.
#   1-3   noise=default wd=1e-6   (the default network; part B continues these three)
#   4-6   noise=default wd=0
#   7-9   noise=zero    wd=1e-6
#   10-12 noise=zero    wd=0
#
# Usage (from a clean clone pinned to the submission commit; --chdir is required by the import check):
#   sbatch --chdir=$HOME/trainRNNbrain_mech --export=ALL --array=1-12 slurm/mech_noise_decay_della.slurm
#
# Submitted in two halves, the second after the first (slurm/mech_submit_della.sh): tasks 1-6, then 7-12.
# --time=08:30:00 is twice the slowest CDDM run of 75,000 iterations at N = 1,000 (4 h 04 min; usual 3 h 15 min).
# slurm/mech_budget.sh refuses a task that could take the experiment past its GPU-hour cap.
#
# Output:    $SAVE_TO/CDDM_<SWEEP>/EqType=h_N=1000_noise=<noise>_wd=<wd>/<run>/
# Snapshots: $SNAP_TO/CDDM_<SWEEP>/EqType=h_N=1000_noise=<noise>_wd=<wd>/s<seed>/it<iters>/
#
#SBATCH --job-name=MechA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=32G
#SBATCH --time=08:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.err

module purge
module load cudatoolkit/12.8
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env
set -eo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="${REPO:-$HOME/trainRNNbrain_mech}"
SAVE_TO="${SAVE_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs}"
SNAP_TO="${SNAP_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots}"
SWEEP="${SWEEP:-mech_noise_decay}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
if [ "$PKG" != "$(readlink -f "$REPO")" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO - submit with --chdir=\$REPO." >&2; exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
[ -n "$GITSHA" ] || { echo "FATAL: $REPO is not a git checkout." >&2; exit 1; }
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

SEEDS=(1153179695 2873231781 2425511753)
NOISES=(default default zero zero)
DECAYS=(1e-6 0 1e-6 0)
TID=$((SLURM_ARRAY_TASK_ID - 1))
CELL=$((TID / 3))
NOISE=${NOISES[$CELL]}
WD=${DECAYS[$CELL]}
SEED=${SEEDS[$((TID % 3))]}
[ -n "$NOISE" ] || { echo "FATAL: array index ${SLURM_ARRAY_TASK_ID} is outside 1-12." >&2; exit 1; }
if [ "$NOISE" = zero ]; then KNOB=(model.sigma_rec=0 model.sigma_inp=0 model.sigma_out=0); else KNOB=(); fi
# N and the budget come from the config; these two exist for the smoke run only
SMOKE=()
[ -z "${N_OVERRIDE:-}" ] || SMOKE+=("model.N=$N_OVERRIDE")
[ -z "${ITERS_OVERRIDE:-}" ] || SMOKE+=("trainer.max_iter=$ITERS_OVERRIDE")

TAG="${SWEEP}/EqType=h_N=1000_noise=${NOISE}_wd=${WD}"
mkdir -p "${SAVE_TO}/CDDM_${SWEEP}" "$SNAP_TO"
source "$REPO/slurm/mech_budget.sh"
budget_start $((8 * 3600 + 1800))
echo "Task ${SLURM_ARRAY_TASK_ID}: noise=${NOISE} wd=${WD} seed=${SEED}"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="$SEED" \
  +experiment=mech_noise_decay \
  trainer.weight_decay="$WD" "${KNOB[@]}" "${SMOKE[@]}" \
  paths.checkpoint_to="$SNAP_TO" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
