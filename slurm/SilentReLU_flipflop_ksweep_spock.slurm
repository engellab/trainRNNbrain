#!/bin/bash
# TASK-COMPLEXITY SWEEP: does the active-unit ceiling M* track task demand? (SPOCK)
#
# The CDDM result is that active units saturate: a 50x larger network buys 7.4x the active units,
# and the hyperbolic fit puts the ceiling near 880. The objection that result cannot yet answer is
# the deflationary one - CDDM is a single, low-dimensional task, so a ceiling of ~880 is exactly what
# "the task is easy and the rest is spare capacity" predicts. Headline result and dismissal are
# observationally identical on one task.
#
# The n-bit flip-flop breaks that tie because COMPLEXITY IS A DIAL. k bits demand 2^k stable states
# and a readout of dimension k, so there is an a-priori prediction fixed before running: if M* tracks
# task demand it must move with k, and the shape of M*(k) - additive, multiplicative, or flat -
# is the result. Flat M*(k) would mean a genuine pathology rather than spare capacity.
#
# WHY THIS TASK RATHER THAN A SINGLE HARDER ONE. Each bit is generated i.i.d. and the loss averages
# over channels, so the target variance is k-INDEPENDENT (measured: 0.727-0.738 over k=2..6, a 1.5%
# spread). MSE is therefore already per-channel-normalised and R^2 = 1 - MSE/0.735 is directly
# comparable across k. Comparing CDDM to DMTS would carry no such guarantee, and any difference in M*
# would confound complexity with input dimensionality, trial length and loss-floor structure.
#
# Grid (63 jobs) = 7 complexities x 3 sizes x 3 seeds, h equation, no penalties.
#   k in {2,3,4,5,6,7,8}  swept via task.n_inputs = task.n_outputs (one channel per bit)
#   N in {500,1000,2000}
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..62): REP=TID%3; N_IDX=(TID/3)%3; K_IDX=TID/9
# Target variance stays k-independent out to k=8 (0.720-0.738 measured), so R^2 remains directly
# comparable across the whole extended range - the property the entire cross-k comparison rests on.
#
# WHY 1000000 ITERATIONS (raised from 300000 after the first sweep).
# The 300k sweep could not read any cell closer than ~1.1x excess over its own floor, while CDDM was
# read at 0.08x. Since the M-vs-N exponent b falls monotonically with reading depth (1.11 / 1.04 /
# 0.95 / 0.87 / 0.58 at 10.4x / 5.7x / 3.5x / 2.0x / 1.1x excess), the 300k sweep sits entirely in a
# shallow regime where M looks linear in N and CANNOT test saturation at all. Extrapolation puts
# 0.1x excess at 0.4M iterations for k=2 rising to 0.9M for k=6, so 1M is the smallest budget with a
# chance of reaching CDDM-comparable depth across the whole k range - and that extrapolation divides
# by a floor the halved-range check says is not identified, so it is a lower bound, not a promise.
# k is extended to 8 because it is the axis that dominates at matched performance (k explains 65-86%
# of the variance in silent fraction, N explains 1-14%), and 2-6 is a short range to fit a law over.
#
# HOW ANY BUDGET IS JUDGED SUFFICIENT, unchanged from the first sweep. Not from theory. It has to satisfy
# one thing: the FLOOR ESTIMATE MUST BE STABLE UNDER HALVING THE FIT RANGE. That is the direct lesson
# of the CDDM analysis, where mismatched fit ranges manufactured a spurious trend three separate
# times - fitting N=2000 over 300k while the other sizes were fitted over 200k produced a monotone
# decay exponent that evaporated once the ranges were matched. So the test is: fit L_inf on [0,T] and
# on [0,T/2]; if they agree the budget was sufficient, and if not it was not.
# That test can only be run AFTER the fact, which is why the budget is set generously rather than
# tightly. Over-running costs GPU-hours; under-running costs a re-run from scratch, because there is
# no checkpoint/resume and the participation trace is written only on completion. Spock's partitions
# have no wall-time limit, so a long request carries no timeout risk. For calibration, CDDM read its
# matched-performance points at 32k-91k inside runs of 200k-300k; 300k here leaves the same margin.
#
# The READ-OUT iteration T(k,N) is an OUTPUT of the analysis, never an input: every cell is run to
# the same generous budget and then read BACKWARD at the iteration where its smoothed noise-free loss
# stably crosses the matched level. Budgets do not need to be matched, only sufficient.
#
# What this sweep decides, in order:
#   1. the floor per k, from loss_clean_train (the NOISE-FREE task loss, recorded unconditionally at
#      every probe - not TrainLosses.json, which is task + lambda*penalty evaluated with noise on and
#      which has now inverted a conclusion twice in this project)
#   2. whether the floor is N-INDEPENDENT within each k. This is the check that licenses matching on
#      loss at all, and it does NOT transfer from CDDM - it must be redone per k.
#   3. if the floors are also close ACROSS k, a single universal L* serves the whole grid and the
#      cross-k comparison needs nothing further. If they are not, M*(k) is read on a per-k ladder of
#      depths and the k-ORDERING must be shown stable across that ladder.
#
# track_valid is OFF: the held-out construction in run_experiment.py interpolates CDDM coherence
# midpoints and has no meaning here. For this task a fresh random batch IS a genuine held-out set
# (trials are drawn i.i.d. every call), so generalisation is evaluated post-hoc instead.
#
# Cost is N-INDEPENDENT in this range (0.13-0.21 s/iter at every size; the T=300 sequential loop
# dominates, not the N^2 matmul), so all three sizes get the same wall-time request. The 0.21 s/iter
# tail is real node-to-node variation, and 1M x 0.21 = 58 h, hence 72 h - the first sweep's N=500
# array was nearly killed by a request sized from the median rather than the slow tail.
#   sbatch --array=1-63 --time=72:00:00
# Timing override for a calibration run: ITERS_OVERRIDE=600 sbatch --array=45 ...
#
# Output: $SAVE_TO/FlipFlop_std_ksweep/EqType=h_k=<k>_N=<N>_iters=<IT>/<per-net subfolder>

#SBATCH --job-name=FFk
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=1-63
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

REPO="${REPO:-$HOME/trainRNNbrain_ff}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# /usr/people is a symlink to /mnt/cup/people on Spock, so Python reports the physical path while
# $REPO holds the symlinked one - compare RESOLVED paths.
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2; exit 1
fi
# The worktree is populated by rsync from the Mac, not a git checkout, so its own HEAD is stale.
if [ -f "$REPO/CODE_VERSION.txt" ]; then
  echo "code: $PKG @ $(cat "$REPO/CODE_VERSION.txt")"
else
  echo "FATAL: $REPO/CODE_VERSION.txt missing — cannot verify which code this is." >&2; exit 1
fi

KS=(2 3 4 5 6 7 8)
NS=(500 1000 2000)
IT=1000000

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); N_IDX=$(((TID / 3) % 3)); K_IDX=$((TID / 9))
K=${KS[$K_IDX]}; N=${NS[$N_IDX]}
IT=${ITERS_OVERRIDE:-$IT}

TAG="std_ksweep/EqType=h_k=${K}_N=${N}_iters=${IT}"
mkdir -p "${SAVE_TO}/FlipFlop_std_ksweep"
echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K} N=${N} rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/FlipFlop_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=NBitFlipFlop \
  task.n_inputs="$K" task.n_outputs="$K" \
  model=rnn_relu_standard trainer=trainer_ptrack_plain \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
