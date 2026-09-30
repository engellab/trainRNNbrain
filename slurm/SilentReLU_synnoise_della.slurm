#!/bin/bash
# DOES A CONNECTIVITY THAT NEVER REPEATS KEEP UNITS ALIVE? (DELLA)
#
# Every intervention this project has tried acts on a unit AFTER it has gone silent: replace it,
# duplicate a donor onto it, tilt its incoming balance. This one acts continuously and on nothing
# in particular. model.sigma_w multiplies W_rec and W_inp by (1 + sigma_w * eps) with a fresh eps
# at EVERY timestep, so the network never runs the same connectivity twice.
#
# THE MECHANISM IT IS BETTING ON. A ReLU unit held below threshold on every step has exactly zero
# gradient on every incoming and outgoing weight -- that absorbing state is the whole subject of
# the paper. A unit sitting just under threshold under synaptic jitter crosses it on some steps and
# not others, and on those steps its weights have a gradient that is not zero.
#
# ⚠️ IT IS NOT sigma_rec BY ANOTHER NAME, and the difference is the reason to run it. State noise
# adds a current every unit receives whatever its weights are, so a deeply disconnected unit gets
# the full kick and still cannot use it; sigma_rec = 0 is already known to be much WORSE than the
# default (-340 units, panel d of Fig. 1). Synaptic noise scales with the weights, so a
# disconnected unit feels nothing: measured state sd 0.00e+00 under sigma_w = 0.5 against 3.71e-01
# under sigma_rec = 0.5 for a unit with its incoming rows zeroed.
#
# GRID (9 jobs) = 3 noise levels x 3 seeds, N=1000 on the 3-bit flip-flop, 40k iterations, no
# penalties, gamma = 0.
#   sigma_w  0.1   each synapse varies by about 10% per step; no sign ever flips
#   sigma_w  0.3   moderate; the regime where release variability in cortex sits
#   sigma_w  1.0   extreme: the perturbation is the size of the weight, so a synapse changes sign
#                  on about 16% of steps. An upper bound, included so a null at 0.1 and 0.3 cannot
#                  be blamed on the knob never having been turned far enough.
#   Index decode (TID = SLURM_ARRAY_TASK_ID - 1, 0..8): REP = TID % 3; CELL = TID / 3
#
# NO CONTROL CELL IS RUN. sigma_w = 0 is bit-for-bit the historical forward pass -- pinned by
# tests/test_synaptic_weight_noise.py, which checks two seeded runs are identical entry for entry --
# so NBitFlipFlop_ff_revive/EqType=h_k=3_N=1000_pen=none_arm=none IS the matched control at
# 275 +- 24 active units, r2 0.945. Running it again would only re-measure seed noise.
#
# PRE-REGISTERED, WRITTEN BEFORE THESE JOBS EXISTED. Read-out at 40,000 iterations, scale-free
# criterion (a unit is silent below 5% of its own network's 95th-percentile participation).
#   PRIMARY    active units against the control's 275 +- 24.
#   CO-PRIMARY held-out r2 against the control's 0.945. Jitter this large is a regulariser and may
#              cost performance; a cell more than 0.10 below the control is BROKEN whatever its
#              unit count, as is one discarding more than 5% of its gradient updates.
#   DIMENSIONS activity dimensionality, because "recruits units that do nothing new" has been the
#              failure mode of half the interventions here.
#   ⚠️ THE CRITERION CAN BREAK AT BOTH ENDS AND HAS DONE. Heavy weight decay drove it to N/N by
#              compressing the whole population toward zero, and a diverging rescale run scored
#              988/1000 on a dead network. At sigma_w = 1.0 in particular, check the participation
#              SPREAD (q99/q01) before reading the count: a jitter that stops any unit from
#              settling would raise the floor and pass everything.
#   WHAT WOULD CHANGE MY MIND: if the active count rises with sigma_w at no r2 cost, then keeping
#              a unit near threshold is enough to keep it trainable and the whole replace-it-after
#              -the-fact family was treating a symptom. If it does nothing at any level, the
#              absorbing state is not about marginal units -- the silent population is far enough
#              below threshold that no plausible jitter reaches it, which the drive measurements
#              already suggest (input weights 21x smaller at N = 1000).
#
# COST. Measured at N = 200: 12 iterations took 21 s at sigma_w = 0 and 23 s at 0.2. The extra work
# is an (N, N) normal draw per timestep, so it grows as N^2 and will be a good deal worse at
# N = 1000; the autograd graph also holds one N x N tensor per step. 8 h and 64 GB requested for
# that reason rather than the 4 h this cell takes without it.
# Output: $SAVE_TO/NBitFlipFlop_synnoise/EqType=h_k=3_N=1000_sw=<level>/

#SBATCH --job-name=SynNoise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=08:00:00
#SBATCH --array=1-9%9
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

K=3
N=1000
IT=40000

CELL_SW=(0.1 0.3 1.0)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); CELL=$((TID / 3))
SW=${CELL_SW[$CELL]}

TAG="synnoise/EqType=h_k=${K}_N=${N}_sw=${SW}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_synnoise"
echo "Task ${SLURM_ARRAY_TASK_ID}: cell=${CELL} sigma_w=${SW} rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/NBitFlipFlop_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=NBitFlipFlop \
  task.n_inputs="$K" task.n_outputs="$K" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  "model.sigma_w=${SW}" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  trainer.prune_reinit=false trainer.synaptic_scaling=false \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
