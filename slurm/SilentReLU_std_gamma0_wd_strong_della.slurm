#!/bin/bash
# DOES THE ACTIVE-UNIT COUNT KEEP FALLING AS WEIGHT DECAY GETS STRONGER? (DELLA)
#
# WHY. The 2026-07-28 horizon runs covered weight decay 0 and 1e-6 only, and their active-unit
# read-out was never taken. Reading it now (participation traces on /scratch, 100000 iterations):
#
#     N=1000  wd=0     r2 0.865   active 394, 428, 438  -> 420
#     N=1000  wd=1e-6  r2 0.870   active 321, 314, 301  -> 312
#     N=2000  wd=0     r2 0.863   active 594, 580, 583  -> 586
#     N=2000  wd=1e-6  r2 0.873   active 449, 420, 433  -> 434
#
# A decay of 1e-6 -- small enough to look like a rounding choice -- already removes a QUARTER of the
# active units at both sizes, with seed ranges that do not overlap and no cost in r2. That makes
# weight decay the largest single lever on the active count found so far outside the explicit
# interventions, and the series stops one step after it starts.
#
# WHAT THIS ADDS. Two more decay strengths, 1e-3 and 1e-2, on the same task, the same sizes, the
# same 100000 iterations and the same code commit as the original four points, so the five-point
# series 0 / 1e-6 / 1e-3 / 1e-2 is one curve rather than a comparison across runs.
#
# GRID (12 jobs) = 2 weight_decay x 2 sizes x 3 seeds, CDDM, h equation, no penalties, 100000 iters:
#   weight_decay in {1e-3, 1e-2}      N in {1000, 2000}      3 reps, seed="random"
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..11): REP=TID%3; WD_IDX=(TID/3)%2; N_IDX=(TID/6)%2
#
# READ-OUT, fixed before submission. Active-unit count at 100000 iterations under the scale-free
# criterion (p_i >= 0.05 x the 95th percentile of p), the same criterion the table above uses, and
# mean r2 per cell.
# PREDICTION: if the late-phase silencing is decay-driven, both new cells sit BELOW the wd=1e-6
# count -- under 312 at N=1000 and under 434 at N=2000.
# FALSIFIER: if wd=1e-3 and wd=1e-2 land inside the wd=1e-6 seed range (301-321 at N=1000,
# 420-449 at N=2000), then decay strength beyond 1e-6 does nothing, the 0 -> 1e-6 step is a
# threshold rather than a dose, and weight decay cannot be described as setting the count.
# GATE: any cell whose mean r2 falls below 0.7 has its active count REPORTED BUT NOT INTERPRETED.
# A network that cannot do the task has a meaningless participation distribution under a relative
# criterion -- measured directly while probing shrink-and-perturb, where a network at r2 0.009 had
# 996 of its 1000 units clear the 5% bar. Adam here takes weight_decay as coupled L2 added to the
# gradient (torch.optim.Adam, not AdamW), and lr is scaled by (100/N)^lr_scale_exp, so 1e-2 against
# a base lr of 1e-3 is a real candidate for exactly that collapse.
#
# Cost from the ORIGINAL runs' measured elapsed times, not an estimate: N=1000 3:48-3:52,
# N=2000 8:21-8:51. Six jobs at each size is about 74 GPU-hours. 14 h requested, as before.
#   REPO=$HOME/trainRNNbrain_horizon sbatch --array=1-12 slurm/SilentReLU_std_gamma0_wd_strong_della.slurm
#
# Output: $SAVE_TO/CDDM_std_g0_horizon/EqType=h_N=<N>_wd=<wd>/  -- beside the original four cells.

#SBATCH --job-name=StdWDstrong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=14:00:00
#SBATCH --constraint=nomig
#SBATCH --array=1-12
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

# PINNED to the worktree the original four points were trained from. The two new decay strengths are
# only comparable to them if the code is the same, so this is not interchangeable with $HOME/trainRNNbrain.
REPO="${REPO:-$HOME/trainRNNbrain_horizon}"
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
if [ "$GITSHA" != "f5aa558" ]; then
  echo "FATAL: $REPO is at $GITSHA, not f5aa558 — the new decay points would not match the original four." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

NS=(1000 2000); WDS=(1e-3 1e-2)
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); WD_IDX=$(((TID / 3) % 2)); N_IDX=$(((TID / 6) % 2))
N=${NS[$N_IDX]}; WD=${WDS[$WD_IDX]}

TAG="std_g0_horizon/EqType=h_N=${N}_wd=${WD}"
mkdir -p "${SAVE_TO}/CDDM_std_g0_horizon"
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} weight_decay=${WD} rep=${REP} iters=100000"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  model=rnn_relu_standard trainer=trainer_ptrack_plain \
  model.equation_type=h model.N="$N" \
  trainer.max_iter=100000 trainer.weight_decay="$WD" \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
