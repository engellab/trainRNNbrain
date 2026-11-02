#!/bin/bash
# ABSORBING-BOUNDARY TEST P3b: CDDM with ALL training noise removed, paired with default noise. (DELLA)
#
# WHY THIS EXISTS. Pavel's hypothesis (notebook 2026-11-01_2119_absorbing-boundary-plan.md) is that
# training noise drives units across the zero-gradient boundary. On CDDM the only randomness in
# training is the three noise terms (sigma_rec, sigma_inp, sigma_out): the task builds the same 450
# trials on every call and dropout is off. With all three at 0 the gradient is a deterministic
# function of the weights, so a noise-free network that still silences units at the paired rate
# falsifies "noise drives the crossing".
#
# ARMS (the one swept knob is the noise): tasks 1-3 noise=zero, tasks 4-6 noise=default, each with
# the three CDDM control seeds of CDDM_c1_snapshots, so each pair starts from the same initial
# weights. Everything else is the CDDM control's launcher (SilentReLU_c1_snapshots_della.slurm):
# standard network, N = 1000, snapshots kept every 500 iterations, participation trace every 100.
#
#   sbatch --time=05:00:00 --array=1-6 slurm/absorb_noise_della.slurm
# ~2.9 h per 75k on an A100 (CDDM control: 3.9 h per 100k); ~18 GPU-h for the array.
#
# Output: $SAVE_TO/CDDM_absorb_noise/EqType=h_N=1000_noise=<arm>/<run>/
#         $SNAP_TO/CDDM_absorb_noise/EqType=h_N=1000_noise=<arm>/s<seed>/it<iters>/
#
#SBATCH --job-name=absorbN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=64G
#SBATCH --time=05:00:00
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

REPO="${REPO:-$HOME/trainRNNbrain_absorb}"
SAVE_TO="${SAVE_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs}"
SNAP_TO="${SNAP_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) - aborting." >&2
  exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
if [ -z "$GITSHA" ]; then
  echo "FATAL: $REPO is not a git checkout - provenance of this run cannot be recorded." >&2; exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

N=${N:-1000}
IT=${ITERS_OVERRIDE:-75000}
SEEDS=(1423687489 2657899762 2670295995)
ID=${SLURM_ARRAY_TASK_ID:?array task id required}
SEED=${SEEDS[$(( (ID - 1) % 3 ))]}
if [ "$ID" -le 3 ]; then
  ARM=zero; KNOB=(model.sigma_rec=0 model.sigma_inp=0 model.sigma_out=0)
else
  ARM=default; KNOB=()
fi
TAG="absorb_noise/EqType=h_N=${N}_noise=${ARM}"
mkdir -p "${SAVE_TO}/CDDM_absorb_noise" "$SNAP_TO"
echo "Task ${ID}: CDDM N=${N} seed=${SEED} noise=${ARM} iters=${IT}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="$SEED" \
  +experiment=silent_units_largeN \
  task=CDDM \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" "${KNOB[@]}" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.prune_reinit=false trainer.synaptic_scaling=false trainer.dropout=false \
  trainer.checkpoint_every=500 ++trainer.checkpoint_keep=true \
  paths.checkpoint_to="$SNAP_TO" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
