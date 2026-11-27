#!/bin/bash
# MECHANISM TESTS, PART B: resurrect or prevent? A penalty switched on in a trained default CDDM network (DELLA)
#
# Rules: docs/project_trajectory/2026-11-25_*_rules-mechanism-noise-decay-and-switch.md, committed before any run.
# Parents: the three default networks of part A (noise=default, wd=1e-6) at 75,000 iterations, weights and Adam moments.
# Each is continued for 25,000 iterations (to 100,000) three times, with the parent's seed:
#   1-3  none     no penalty (the same run trained on)
#   4-6  ent0.1   entropy set point, trainer.lambda_ent=0.1
#   7-9  rate0.1  rate penalty, trainer.lambda_frm=0.1
# ARM = (TID-1)/3, network = (TID-1)%3. Every fixed value is in configs/experiment/mech_switch.yaml.
#
# Usage (same pinned clone as part A, after part A's tasks 1-3 have finished):
#   sbatch --chdir=$HOME/trainRNNbrain_mech --export=ALL --array=1-9 slurm/mech_switch_della.slurm
#
# --time=03:00:00 is twice the usual 1 h 05 min to 1 h 20 min of 25,000 iterations, rounded up.
# slurm/mech_budget.sh refuses a task that could take the experiment past its GPU-hour cap.
#
#SBATCH --job-name=MechB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=32G
#SBATCH --time=03:00:00
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
SWEEP="${SWEEP:-mech_switch}"
PARENT_SWEEP="${PARENT_SWEEP:-mech_noise_decay}"
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
ARMS=(none ent0.1 rate0.1)
TID=$((SLURM_ARRAY_TASK_ID - 1))
ARM=${ARMS[$((TID / 3))]}
SEED=${SEEDS[$((TID % 3))]}
case "$ARM" in
  none)    PEN=() ;;
  ent0.1)  PEN=(trainer.lambda_ent=0.1) ;;
  rate0.1) PEN=(trainer.lambda_frm=0.1) ;;
  *) echo "FATAL: array index ${SLURM_ARRAY_TASK_ID} is outside 1-9." >&2; exit 1 ;;
esac
CELL="$SAVE_TO/CDDM_${PARENT_SWEEP}/EqType=h_N=1000_noise=default_wd=1e-6"
PARENTS=("$CELL"/*"seed=${SEED};"*)
if [ "${#PARENTS[@]}" -ne 1 ] || ! ls "${PARENTS[0]}"/*AdamState*.pt "${PARENTS[0]}"/*LastParams* >/dev/null 2>&1; then
  echo "FATAL: need exactly one finished parent with an Adam state for seed ${SEED} in $CELL (found ${#PARENTS[@]})" >&2; exit 1
fi
PARENT="${PARENTS[0]}"
# N and the budget come from the config; these two exist for the smoke run only
SMOKE=()
[ -z "${N_OVERRIDE:-}" ] || SMOKE+=("model.N=$N_OVERRIDE")
[ -z "${ITERS_OVERRIDE:-}" ] || SMOKE+=("trainer.max_iter=$ITERS_OVERRIDE")

TAG="${SWEEP}/EqType=h_N=1000_arm=${ARM}"
mkdir -p "${SAVE_TO}/CDDM_${SWEEP}" "$SNAP_TO"
source "$REPO/slurm/mech_budget.sh"
budget_start $((3 * 3600))
echo "Task ${SLURM_ARRAY_TASK_ID}: arm=${ARM} seed=${SEED}"
echo "  parent: $PARENT"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="$SEED" \
  +experiment=mech_switch \
  "${PEN[@]}" "${SMOKE[@]}" \
  "paths.init_from='$PARENT'" \
  paths.checkpoint_to="$SNAP_TO" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
