#!/bin/bash
# ENTROPY SET POINT ON CDDM, N = 1,000 (DELLA)
#
# Rules: docs/project_trajectory/2026-11-22_*_rules-entropy-setpoint-cddm.md, committed before any run.
# The command is the CDDM command of SilentReLU_paper_grid_della.slurm (TASK=cddm, which trained the
# control networks CDDM_paper_grid/EqType=h_N=1000_arm=control), with +experiment=entropy_setpoint in place
# of silent_units_largeN (the same block plus the two set-point terms, off, and model.init_draws: shared),
# the entropy term on at one lambda per arm, 75,000 iterations and snapshots kept every 1,000 iterations.
# No control arm: the controls are the existing three networks of that cell, not retrained.
#
# GRID: 2 arms x 3 networks = 6 jobs. ARM = (TID-1)/3.
#   1-3  ent0.01   trainer.lambda_ent=0.01
#   4-6  ent0.1    trainer.lambda_ent=0.1
#
# Usage (from a clean clone pinned to the submission commit; --chdir is required by the import check):
#   sbatch --chdir=$HOME/trainRNNbrain_entcddm --export=ALL --array=1-6 slurm/entropy_setpoint_cddm_della.slurm
#
# --time=11:00:00 caps the screen at 6 x 11 = 66 GPU-hours, under the 70 allowed, whatever the speed.
#
# Output:    $SAVE_TO/CDDM_<SWEEP>/EqType=h_N=1000_arm=<arm>/<run>/
# Snapshots: $SNAP_TO/CDDM_<SWEEP>/EqType=h_N=1000_arm=<arm>/s<seed>/it<iters>/
#
#SBATCH --job-name=EntCDDM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=32G
#SBATCH --time=11:00:00
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

REPO="${REPO:-$HOME/trainRNNbrain_entcddm}"
SAVE_TO="${SAVE_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs}"
SNAP_TO="${SNAP_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots}"
SWEEP="${SWEEP:-entropy_setpoint}"
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

ARMS=(ent0.01 ent0.1)
TID=$((SLURM_ARRAY_TASK_ID - 1))
ARM=${ARMS[$((TID / 3))]}
[ -n "$ARM" ] || { echo "FATAL: array index ${SLURM_ARRAY_TASK_ID} is outside 1-6." >&2; exit 1; }
N=${N:-1000}
IT=${ITERS_OVERRIDE:-75000}

TAG="${SWEEP}/EqType=h_N=${N}_arm=${ARM}"
mkdir -p "${SAVE_TO}/CDDM_${SWEEP}" "$SNAP_TO"
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} arm=${ARM} rep=$((TID % 3)) iters=${IT}"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"
echo "  snapshots -> ${SNAP_TO}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=entropy_setpoint \
  task=CDDM \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.prune_reinit=false trainer.synaptic_scaling=false trainer.dropout=false \
  trainer.lambda_ent="${ARM#ent}" trainer.lambda_cor=0 \
  trainer.checkpoint_every=1000 ++trainer.checkpoint_keep=true \
  paths.checkpoint_to="$SNAP_TO" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
