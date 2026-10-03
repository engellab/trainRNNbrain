#!/bin/bash
# WITH NO WEIGHT DECAY, DOES THE ACTIVE-UNIT COUNT SATURATE? (DELLA)
#
# WHY. Weight decay is the largest single lever on the active count found outside the explicit
# interventions: at 100000 iterations the standard network at N=1000 ends with 394 / 428 / 438
# active units at weight_decay 0 against 321 / 314 / 301 at 1e-6, with no cost in r2. What those
# runs cannot say is whether the wd=0 count has STOPPED falling or is only falling more slowly:
# 100000 iterations is the longest horizon that line has, and the loss was flat over its last 10%
# while the silent fraction was still climbing. This run removes the decay and triples the horizon.
#
# GRID (3 jobs) = 3 seeds, CDDM, N = 1000, h equation, standard RNN, defaults everywhere except
# weight_decay = 0, 300000 iterations.
#
# READ-OUT, fixed before submission: the active-unit count (scale-free rule, p_i >= 0.05 x q95(p))
# as a function of iteration, from the participation trace, over the last third of training.
# SATURATION CRITERION: the mean count over iterations 250k-300k is within the 3-seed spread of the
# mean over 200k-250k, AND the fitted slope over the last 100k is under 5 units per 100k iterations.
# FALSIFIER: a count still falling by more than 5 units per 100k at 300k means the wd=0 network has
# not reached an asymptote either, and no run on disk has — the decay only sets the RATE of the walk.
# GATE: mean r2 < 0.7 in a seed -> that seed's count is reported but not interpreted.
#
# The companion reference is the default-decay line at matched iterations (weight_decay 1e-6), read
# from CDDM_std_g0_drift at N=1000 (200000 iterations) — a different code commit, so it bounds the
# comparison rather than settling it. The saturation question above is answered WITHIN these runs.
#
# Runtime 0.144 s/iter at N=1000 (measured, drift sweep) -> ~12 h per job. 30 h requested on
# --qos=gpu-medium: nothing is written to disk before the run completes, so a job that hits the
# wall loses its whole allocation, and the 20 h that 12 h "should" need leaves no margin for a
# slower node.
# Output: $SAVE_TO/CDDM_std_g0_wd0_long/EqType=h_N=1000_wd=0_iters=300000/<per-net subfolder>

#SBATCH --job-name=StdWD0Long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=30:00:00
#SBATCH --qos=gpu-medium
#SBATCH --constraint=nomig
#SBATCH --array=1-3
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

TAG="std_g0_wd0_long/EqType=h_N=1000_wd=0_iters=300000"
mkdir -p "${SAVE_TO}/CDDM_std_g0_wd0_long"
echo "Task ${SLURM_ARRAY_TASK_ID}: N=1000 weight_decay=0 iters=300000"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  model=rnn_relu_standard trainer=trainer_ptrack_plain \
  model.equation_type=h model.N=1000 \
  trainer.max_iter=300000 trainer.weight_decay=0 \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
