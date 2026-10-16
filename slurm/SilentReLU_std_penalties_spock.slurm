#!/bin/bash
# PENALTIES vs NO PENALTY: do frm/rws keep units active, and do they make training SETTLE? (SPOCK)
#
# The no-penalty runs established a problem that is itself an argument for penalties. Silencing never
# stops: at the end of every run the silent fraction is still moving by 4-10 percentage points per
# DOUBLING of the budget (N=100 8.3, N=500 10.4, N=1000 6.7, N=2000 4.4), the participation vector is
# still changing directionally at long lags, and no stopping criterion exists - three separate ones
# were tried and all failed (directional cosine: floors on Adam's momentum; lag-scaling exponent: the
# diffusive transition never arrives; caging timescale L*: size-dependent, 10.2% -> 6.4% of training
# age from N=100 to N=2000). So an unpenalised network gives you neither a defensible unit count nor
# a defensible time to read it at.
#
# This sweep tests whether penalties fix BOTH: more active units (the original claim) AND a silent
# fraction that actually settles (the new one). The second is the sharper result if it holds, because
# it converts "we could not find a stopping rule" from a limitation of our analysis into a measured
# property of the unpenalised condition.
#
# Grid = 4 sizes x 3 penalty conditions x 3 seeds, h equation. Sizes 500/1000 ran first (array
# 5664681); 2000/5000 were added afterwards because "the penalty works regardless of size" cannot be
# supported from the two sizes where silencing is MILDEST - unpenalised nets are 57%/71% silent at
# N=500/1000 but 70%/85% at N=2000/5000.
#   Index decode (TID = SLURM_ARRAY_TASK_ID-1): REP = TID%3; PEN = (TID/3)%3; N_IDX = TID/9
#   tasks  1-9  N=500    10-18 N=1000   19-27 N=2000   28-36 N=5000
#   Submit one array per size, so wall time and memory match the cell:
#     sbatch --array=1-18                              # N=500,1000 (done, array 5664681)
#     sbatch --array=19-27 --time=30:00:00             # N=2000, ~0.25 s/iter -> ~11 h
#     sbatch --array=28-36 --time=96:00:00 --mem=64G   # N=5000, ~1.7 s/iter -> ~57 h
#   N=5000 needs 30.6 GB reserved GPU memory, which fits on both Spock card types ONLY because
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments is set below (33.0 GB without it, vs 40 GB on the
#   A100). Host --mem must also rise to 64G at that size.
#   PEN 0 = rws only   (lambda_rws=0.05, lambda_frm=0)
#   PEN 1 = frm only   (lambda_rws=0,    lambda_frm=0.1)
#   PEN 2 = both       (lambda_rws=0.05, lambda_frm=0.1)
# The no-penalty arm already exists in CDDM_std_g0_drift and is NOT repeated here.
#
# Two sizes, not one: the interesting claim is that penalties remove the SIZE DEPENDENCE of silencing,
# which cannot be shown at a single N.
#
# lambda_frm is 0.1, HALF the 0.2 in trainer_ptrack.yaml. A smoke test at 0.2 drove the silent count
# from 5 to 0 within 150 iterations, i.e. the penalty was saturating: it would demonstrate that frm
# removes silent units while telling us nothing about how it changes the DYNAMICS, which is the point
# of this sweep. lambda_rws stays at its configured 0.05.
#
# Monitoring (all new-since-the-drift-sweep items marked *):
#   loss_clean_train *  masked MSE from the noise-free probe, every track_every. FREE - that forward
#                       pass already happens for participation. The per-iteration training loss is
#                       NOISY, so its minimum is a noise lottery and it cannot tell "learned" from
#                       "got a good draw"; this can.
#   loss_clean_valid *  same, on a HELD-OUT batch (coherences interleaved with the training grid), on
#                       its own coarser cadence. With same_batch=True the pipeline otherwise cannot
#                       separate learning from memorising the 450 fixed trials.
#   silent_1em6         silent-unit count at every probe
#   participation       full per-unit vector every store_participation_every
#   drift_* / cos_*     weight change and directional persistence at three lags
#
# Runtime ~8 h per job at N<=1000 (measured on the no-penalty sweep, plus ~4% for held-out probes).
# Output: $SAVE_TO/CDDM_std_g0_penalties/EqType=h_N=<N>_pen=<name>/<per-net subfolder>
#   -> a SEPARATE top-level folder from CDDM_std_g0_drift, so nothing can mix with the running sweep.

#SBATCH --job-name=StdPen
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --array=1-18
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/usr/people/pt1290/trainRNNbrain_logs/%x.%A_%a.out
#SBATCH --error=/usr/people/pt1290/trainRNNbrain_logs/%x.%A_%a.err

module purge
module load anacondapy/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env

set -eo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="${REPO:-$HOME/trainRNNbrain_pen}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# Compare RESOLVED paths: /usr/people is a symlink to /mnt/cup/people on Spock, so Python reports the
# physical path while $REPO holds the symlinked one.
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2; exit 1
fi
# The worktree is populated by rsync from the Mac, not by a git checkout, so its own HEAD is stale
# and must NOT be reported as the provenance. CODE_VERSION.txt carries the actual source commit.
if [ -f "$REPO/CODE_VERSION.txt" ]; then
  echo "code: $PKG @ $(cat "$REPO/CODE_VERSION.txt")"
else
  echo "FATAL: $REPO/CODE_VERSION.txt missing — cannot verify which code this is." >&2; exit 1
fi

NS=(500 1000 2000 5000)
# Per size, comfortably above the iteration at which the UNPENALISED net stably reaches L*=0.023
# (58.5k at N=2000, 91.1k at N=5000), so the matched-performance read-out is inside every run.
ITS=(200000 200000 150000 120000)
PEN_NAME=(rws frm both)
PEN_RWS=(0.05 0 0.05)
PEN_FRM=(0 0.1 0.1)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); PEN=$(((TID / 3) % 3)); N_IDX=$((TID / 9))
N=${NS[$N_IDX]}; IT=${ITS[$N_IDX]}
PNAME=${PEN_NAME[$PEN]}; LRWS=${PEN_RWS[$PEN]}; LFRM=${PEN_FRM[$PEN]}

TAG="std_g0_penalties/EqType=h_N=${N}_pen=${PNAME}"
mkdir -p "${SAVE_TO}/CDDM_std_g0_penalties"
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} pen=${PNAME} (rws=${LRWS} frm=${LFRM}) rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  model=rnn_relu_standard trainer=trainer_ptrack_plain \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=true trainer.track_valid_every=50 \
  trainer.lambda_rws="$LRWS" trainer.lambda_frm="$LFRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
