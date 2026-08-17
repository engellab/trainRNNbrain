#!/bin/bash
# DRIFT CHARACTERISATION: when does systematic drift become mere jitter? (DELLA)
#
# Convergence in the strict sense is unreachable: the relative change of the participation vector
# decays as a POWER LAW in iteration (exponent -0.29 to -0.57), so a 1% criterion extrapolates to
# 0.5-5.6 MILLION iterations. What IS reachable is the point where the motion stops being
# systematic and becomes noise-driven jitter — and that is the defensible definition of "trained
# enough", plus the basis for matching training across N by drift rather than by iteration count.
#
# Why several lags and a cosine, rather than one distance: noise is injected every step, so weights
# random-walk forever and ||W(t)-W(t-L)|| plateaus at a NOISE FLOOR rather than reaching zero.
#   diffusion  -> distance grows as sqrt(L);  systematic drift -> grows as L
#   cosine between consecutive displacements: ~0 = jitter, >0 = still marching one way
#
# Grid (12 jobs) = 4 sizes x 3 seeds, h equation, no penalties. Iterations are set PER SIZE from the
# measured decay rates, not uniformly: the silencing rate decays ~2.5x more slowly at N=2000 than at
# N=1000 (factor 4.2 vs 10.8 between iteration 30k and 100k), so a uniform budget would leave the
# largest and most important cell still drifting while over-running the smallest.
#   N=100   ->  50000   (rate is already ~0 by iteration 3000)
#   N=500   -> 200000
#   N=1000  -> 200000   (rate 0.89 units/1000 at 100k, nearly settled)
#   N=2000  -> 300000   (rate 2.90 units/1000 at 100k, extrapolates to settle near 250-300k)
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..11): REP=TID%3; N_IDX=TID/3
# Submit in two parts: tasks 1-9 (N<=1000) fit gpu-short; tasks 10-12 (N=2000, 26.4 h) need
# --time>24h, i.e. the gpu-medium QOS.
#
# Metrics, all reduced to scalars during training (nothing about the weights is written to disk;
# reference snapshots live in CPU RAM, ~80 MB at N=2000, independent of training length):
#   drift_{W_inp,W_rec,W_out}_lag{100,1000,10000}   relative Frobenius change at three lags
#   cos_{W_inp,W_rec,W_out}                          directional persistence
#   dp_lag{100,1000,10000}                           ||p(t)-p(t-L)|| / ||p(t)||
#   silent_1em6                                      silent-unit count at every probe
# The per-unit participation matrix is the only bulky item and gets a coarser cadence
# (store_participation_every=100 -> 2000 vectors, 16 MB at N=2000).
#
# Runtime at measured rates: N=100 ~2.8 h, N=500 ~4.4 h, N=1000 8.0 h, N=2000 17.6 h.
#
# Output: $SAVE_TO/CDDM_std_g0_drift/EqType=h_N=<N>/<per-net subfolder>

#SBATCH --job-name=StdDrift
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=12:00:00   # overridden to 40:00:00 for the N=2000 tasks
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

REPO="${REPO:-$HOME/trainRNNbrain_drift}"
SAVE_TO="${SAVE_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# Compare RESOLVED paths: on Spock /usr/people is a symlink to /mnt/cup/people, so Python reports the
# physical path while $REPO holds the symlinked one — same directory, different spelling.
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2; exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short=7 HEAD)"

NS=(100 500 1000 2000)
ITS=(50000 200000 200000 300000)
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); N_IDX=$((TID / 3))
N=${NS[$N_IDX]}; IT=${ITS[$N_IDX]}

TAG="std_g0_drift/EqType=h_N=${N}_iters=${IT}"
mkdir -p "${SAVE_TO}/CDDM_std_g0_drift"
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  cp "$REPO/docs/experiments/standard_rnn_drift.md" "${SAVE_TO}/CDDM_std_g0_drift/EXPERIMENT.md"
  echo -e "\n---\nRun: SLURM array ${SLURM_ARRAY_JOB_ID} on Della, code commit $(git -C "$REPO" rev-parse HEAD)" \
    >> "${SAVE_TO}/CDDM_std_g0_drift/EXPERIMENT.md"
fi
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} iters=${IT} rep=${REP} track_drift=true"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  model=rnn_relu_standard trainer=trainer_ptrack_plain \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
