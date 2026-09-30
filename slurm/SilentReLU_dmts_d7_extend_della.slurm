#!/bin/bash
# DMTS AT 7 TAU: A FOURTH SIZE, AND REPLACEMENTS FOR THE SEEDS THAT NEVER LEARNED (DELLA)
#
# Extends DMTS_d7_pen/EqType=h_N=*_pen=none, the unpenalised series Figure 1c plots. Same task
# (DMTS_d7), same model, same trainer, same 150,000 iterations as SilentReLU_dmts_d7_penalties_della
# -- only the sizes and the seed count differ, so the new runs land in the existing cells and are
# poolable with what is already there.
#
# TWO THINGS AT ONCE.
#
# 1. A FOURTH SIZE, N = 4000. The DMTS series has three sizes against four for every other task in
#    Figure 1c, so its exponent -- 0.87, against 0.32 to 0.46 for the flip-flops and CDDM -- is
#    fitted but not extrapolated, and the caption says so. A fourth point is what lets it be read
#    like the others. 4000 rather than 3000 because that is the size the other three series use and
#    a shared grid is worth more than a cheaper point.
#
# 2. REPLACEMENTS FOR TWO SEEDS THAT NEVER LEARNED THE TASK. Unpenalised DMTS at this delay is
#    bimodal per seed: it either solves the task or sits at the constant-output solution. Of the
#    nine seeds run, two landed at r2 = 0.4278 (N = 500) and 0.4275 (N = 2000) -- the same number to
#    three decimals, which is what a constant output scores -- against 0.9837 to 0.9995 for the
#    other seven. Figure 1c excludes them, leaving n = 2 in those two cells.
#
#    TWO REPLACEMENTS PER CELL, NOT ONE. The measured failure rate is 2 of 9. With one replacement
#    per cell the chance that at least one fails again is about 39%, and a repeat costs another
#    20 h at N = 2000 plus the round trip. Two per cell takes that to about 5% for roughly 25 extra
#    GPU-hours, which is the cheaper of the two mistakes.
#
# GRID (7 jobs), one entry per array index rather than a decode, because the cells differ in size
# and in why they exist:
#   1,2   N =  500  pen=none   replacements for the failed N = 500 seed
#   3,4   N = 2000  pen=none   replacements for the failed N = 2000 seed
#   5,6,7 N = 4000  pen=none   the new fourth size
#
# WALL TIME, measured on SilentReLU_dmts_d7_penalties_della rather than guessed: the medians were
# 5.5 h at N = 500, 7.5 h at N = 1000 and 19.5 h at N = 2000. From 1000 to 2000 that is an exponent
# of 1.38, which puts N = 4000 at about 51 h; if it turns quadratic above 2000 it is 78 h. Submit
# the two groups separately so the short jobs are not queued behind a 5-day request:
#
#   sbatch --array=1-4   --time=30:00:00                  slurm/SilentReLU_dmts_d7_extend_della.slurm
#   sbatch --array=5-7   --time=120:00:00 --qos=gpu-long  slurm/SilentReLU_dmts_d7_extend_della.slurm
#
# gpu-medium allows 3 days and gpu-long 6, so N = 4000 needs no checkpoint-restart; an earlier note
# in this project claiming it did was wrong about the QOS limits.
#
# READ-OUT. Active units under the scale-free criterion at 150,000 iterations, joining the existing
# cells. A replacement seed is only usable if it LEARNED THE TASK: the cut is r2 >= 0.8, which sits
# in the empty valley between 0.43 and 0.98 that every run so far falls on one side of, and it is
# the same cut fig_paper_F1.TASK_MIN_R2 applies. A replacement that lands at 0.43 is reported and
# excluded, exactly as the seeds it replaces were.
#
# Output: $SAVE_TO/DMTS_d7_pen/EqType=h_N=<N>_pen=none/   (existing cells for 500 and 2000)

#SBATCH --job-name=DMTSext
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=30:00:00
#SBATCH --array=1-7%7
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

REPO="${REPO:-$HOME/trainRNNbrain}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2; exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
if [ -n "$GITSHA" ]; then
  if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
    echo "FATAL: $REPO is a git checkout with uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
  fi
  echo "code: $PKG @ $GITSHA (git)"
elif [ -f "$REPO/CODE_VERSION.txt" ]; then
  echo "code: $PKG @ $(cat "$REPO/CODE_VERSION.txt") (rsync)"
else
  echo "FATAL: $REPO is neither a clean git checkout nor carries CODE_VERSION.txt — provenance of this run cannot be recorded." >&2; exit 1
fi

IT=${ITERS_OVERRIDE:-150000}

# one entry per array index; index 0 is unused so the decode reads off SLURM_ARRAY_TASK_ID directly
TASK_N=(0 500 500 2000 2000 4000 4000 4000)
TASK_WHY=(- reseed reseed reseed reseed newsize newsize newsize)

N=${TASK_N[$SLURM_ARRAY_TASK_ID]}
WHY=${TASK_WHY[$SLURM_ARRAY_TASK_ID]}

TAG="${SWEEP:-d7_pen}/EqType=h_N=${N}_pen=none"
mkdir -p "${SAVE_TO}/DMTS_${SWEEP:-d7_pen}"
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} pen=none purpose=${WHY} iters=${IT}"
echo "  -> ${SAVE_TO}/DMTS_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=DMTS_d7 \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
