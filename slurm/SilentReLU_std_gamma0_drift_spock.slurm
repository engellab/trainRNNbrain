#!/bin/bash
# DRIFT CHARACTERISATION: when does systematic drift become mere jitter? (SPOCK)
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
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..14): REP=TID%3; N_IDX=TID/3
# Submit ONE ARRAY PER SIZE, so the wall-time request matches the cell (--time is per array, not per
# task, and an oversized request backfills poorly):
#   sbatch --array=1-3   --time=4:00:00   # N=100,  runtime ~1.8 h
#   sbatch --array=4-6   --time=15:00:00  # N=500,  runtime ~7.2 h
#   sbatch --array=7-9   --time=16:00:00  # N=1000, runtime ~8.0 h
#   sbatch --array=10-12 --time=40:00:00  # N=2000, runtime ~26.4 h
#   sbatch --array=13-15 --time=72:00:00 --mem=64G  # N=5000, runtime 28 h (L40S) - 47 h (A100-40G)
# N=5000 is memory-constrained, not just slow. Measured on an L40S: 27.5 GB allocated, 33.0 GB
# reserved, 1.007 s/iter (vs 10.9/13.2 GB and 0.245 s/iter at N=2000). 33 GB reserved is 72% of a
# 46 GB L40S but 82% of an A100-40G, which is why PYTORCH_CUDA_ALLOC_CONF is set below - with it the
# reserved figure drops to 30.6 GB and BOTH card types fit, so the job need not be pinned to the
# scarce L40S pool (11 of its 12 GPUs were busy at submission). The A100-PCIE is 1.7x SLOWER than the
# L40S here (1.689 vs 1.007 s/iter), hence the 72 h request sized for the slow case. Its budget is 100k rather than 300k because the matched-performance comparison reads
# each size at the iteration where its loss reaches L*, and T scales only as ~N^0.13 (40k/46k/55k
# at N=100/500/1000 for L*=0.02273), so ~70k suffices for the current deepest rung; 100k leaves
# margin for a deeper one once the N=100 top-up extends the common range.
# Spock's partitions have no wall-time limit, so the long cell needs no special QOS as it does on
# Della, where 26.4 h exceeds the gpu-short ceiling.
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
# Runtime at measured rates (~0.13-0.32 s/iter): N=100 1.8 h, N=500 7.2 h, N=1000 8.0 h,
# N=2000 26.4 h. Spock A100-40G and L40S-46G are comparable to Della's A100 for this workload.
#
# Output: $SAVE_TO/CDDM_std_g0_drift/EqType=h_N=<N>/<per-net subfolder>

#SBATCH --job-name=StdDrift
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=16:00:00   # per-size override on the sbatch command line; see the header
#SBATCH --array=1-12
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
# Keep reserved memory close to allocated. This decides whether N=5000 fits on an A100-40G at all:
# measured peak is 27.5 GB allocated, but 33.0 GB RESERVED with the default allocator (82% of a 40 GB
# card) versus 30.6 GB with expandable segments (72%). Purely an allocator setting - it changes how
# CUDA memory is carved up, not the numerics - so it is set for every size rather than conditionally.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="${REPO:-$HOME/trainRNNbrain_drift}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# Compare RESOLVED paths: on Spock /usr/people is a symlink to /mnt/cup/people, so Python reports the
# physical path while $REPO holds the symlinked one — same directory, different spelling.
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2; exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short=7 HEAD)"

NS=(100 500 1000 2000 5000)
ITS=(50000 200000 200000 300000 100000)
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); N_IDX=$((TID / 3))
N=${NS[$N_IDX]}; IT=${ITS[$N_IDX]}
# ITERS override, for budget-matching a size that was originally given a shorter run. The M(N)
# comparison needs every size evaluated at a COMMON budget; N=100 was originally given 50k while the
# others got 200k-300k, so a top-up run is needed to put all four sizes on the same axis. The output
# tag carries $IT, so an override lands in its own folder and cannot mix with the original run.
IT=${ITERS_OVERRIDE:-$IT}

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
