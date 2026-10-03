#!/bin/bash
# DOES THE RECURRENT WEIGHT SCALE MOVE THE ACTIVE-UNIT COUNT? (DELLA)
#
# WHY. Every one of the 3,529 saved configs on disk carries spectral_rad: 1.2 and
# connectivity_density_rec: 1.0. The initial recurrent gain has never been varied in this project,
# so "the network silences units" has only ever been observed at one initialisation scale. The
# recurrent weight MAGNITUDE penalty (lambda_rwm, a soft cap on |W_rec| entries) was last run in
# July 2025 on the old Dale architecture at 30k iterations with no participation read-out, so on
# the standard network it is also unmeasured.
#
# GRID (10 jobs) = 5 cells x 2 seeds, CDDM, N = 1000, h equation, standard RNN, no rws/frm,
# weight_decay at its default 1e-6, 75000 iterations (the standard budget, configs/experiment):
#   sr=0.5    spectral_rad 0.5   (sub-critical init, weak recurrence)
#   sr=1.2    spectral_rad 1.2   (the project default — the in-sweep control, same code, same budget)
#   sr=2.0    spectral_rad 2.0   (super-critical init; trains to finite loss, checked locally)
#   rwm=0.05  lambda_rwm 0.05 at spectral_rad 1.2
#   rwm=0.2   lambda_rwm 0.2  at spectral_rad 1.2   (the value the 2025 sweep used)
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..9): REP = TID%2; CELL = TID/2
#
# READ-OUT, fixed before submission: active units at 75000 iterations under the scale-free rule
# (p_i >= 0.05 x q95(p)) from the participation trace, and mean r2 per cell, both against the
# sr=1.2 cell in this same array.
# PREDICTION: the count rises with the initial gain — a larger spectral radius drives more units
# above threshold early, and a unit that fires has a gradient — so sr=0.5 < sr=1.2 < sr=2.0.
# FALSIFIER: if all three spectral-radius cells land inside each other's 2-seed range, the
# initialisation scale does not set the count and the silencing is a property of the trained
# solution rather than of where training started.
# GATE: any cell with mean r2 < 0.7 has its active count REPORTED BUT NOT INTERPRETED — a network
# that cannot do the task has a meaningless participation distribution under a relative criterion.
#
# Runtime 0.144 s/iter at N=1000 (measured, drift sweep) -> ~3.0 h per job; 8 h requested.
# Output: $SAVE_TO/CDDM_std_g0_recinit/EqType=h_N=1000_<cell>_iters=75000/<per-net subfolder>

#SBATCH --job-name=StdRecInit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=08:00:00
#SBATCH --constraint=nomig
#SBATCH --array=1-10
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

REPO="${REPO:-$HOME/trainRNNbrain_recinit}"
SAVE_TO="${SAVE_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2; exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short=7 HEAD 2>/dev/null || true)
if [ -z "$GITSHA" ]; then
  echo "FATAL: $REPO is not a git checkout - provenance of this run cannot be recorded." >&2; exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

# One flat cell list: "<tag>|<hydra overrides>". Seeds are the inner index.
CELLS=(
  "sr=0.5|model.spectral_rad=0.5"
  "sr=1.2|model.spectral_rad=1.2"
  "sr=2.0|model.spectral_rad=2.0"
  "rwm=0.05|+trainer.lambda_rwm=0.05 +trainer.rwm_args={}"
  "rwm=0.2|+trainer.lambda_rwm=0.2 +trainer.rwm_args={}"
)
NSEED=2
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % NSEED)); CELL=${CELLS[$((TID / NSEED))]}
CELL_TAG=${CELL%%|*}; CELL_ARGS=${CELL#*|}

TAG="std_g0_recinit/EqType=h_N=1000_${CELL_TAG}_iters=75000"
mkdir -p "${SAVE_TO}/CDDM_std_g0_recinit"
echo "Task ${SLURM_ARRAY_TASK_ID}: cell=${CELL_TAG} overrides='${CELL_ARGS}' rep=${REP} iters=75000"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  model=rnn_relu_standard trainer=trainer_ptrack_plain \
  model.equation_type=h model.N=1000 \
  trainer.max_iter=75000 \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  ${CELL_ARGS} \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
