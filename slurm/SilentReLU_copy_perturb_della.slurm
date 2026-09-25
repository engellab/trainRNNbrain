#!/bin/bash
# HOW FAR FROM ITS DONOR MUST A COPY START? (DELLA)
#
# WHY. Duplication is the only revival rule that recruits units AND adds activity dimensions, but
# the units it adds are cheap: at N=1000 it reaches 702 active units against a control's 275 while
# dimensions per active unit fall from 0.0202 to 0.0130. Copying a working unit is also, as Pavel
# put it, a bit of cheating. The obvious dial is the jitter on the copy, and the two ends of that
# dial are ALREADY MEASURED at this size, iteration count and replacement rate:
#
#   control                 275 active   5.6 dims   0.0202 dims/active   r2 0.945
#   copy, jitter 0.05       702 active   9.3 dims   0.0130 dims/active   r2 0.943
#   random (no donor)       262 active   5.0 dims   0.0193 dims/active   r2 0.946
#
# So this sweep fills in a curve whose endpoints are known, rather than exploring blind.
#
# WHAT THE KNOB DOES. Multiplicative jitter grows the copied row's norm by sqrt(1 + jitter^2), 3.2x
# at jitter 3, so sweeping it raw would vary the copy's weight MAGNITUDE and DIRECTION together --
# and magnitude is the axis that has already destabilised this network twice (uncapped duplication,
# synaptic scaling in every configuration). `copy_norm_preserve=true` rescales the jittered row to
# the donor row's length, so the noise level sets ONE quantity: the angle between the copy's weight
# vector and the donor's, at expected cosine 1/sqrt(1 + jitter^2). Verified in
# tests/test_prune_copy_norm_preserve.py: measured cosines 0.954 / 0.699 / 0.295 against theory
# 0.958 / 0.707 / 0.316, with the row norm held to within 2% of the donor's.
#
# GRID (15 jobs) = 5 jitter levels x 3 seeds, at N=1000, mode=copy, rate 0.025, maturity 1000,
# 40000 iterations -- every other setting matching the measured `copy` cell, so jitter is the only
# difference.
#   jitter     0      0.05    0.3     1.0     3.0
#   cosine     1.000  0.999   0.958   0.707   0.316
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..14): REP = TID%3; CELL = TID/3
#
# ⚠️ THIS RUN USES A CORRECTED DUPLICATION. Until 2026-09-24 the construction zeroed all three
# entries of the 2x2 block spanning donor and copy. That was right for self_connections=false,
# where the diagonal is masked, and wrong for the self_connections=TRUE every experiment here uses:
# the donor's self-weight lives in its row AND its outgoing column, so halving the column halved
# the donor's own self-loop, and zeroing the copy's diagonal left the copy with none. BOTH twins
# were detuned. The donor's self-weight is now split in half across all four entries of the block,
# which each twin then draws (w/2)*r from itself and (w/2)*r from the other, summing to the w*r the
# donor drew alone. Measured: one duplication now changes the network's output by 4.5e-08, float32
# rounding, against 5.9e-04 to 8.9e-03 before -- on an output scale of 0.27. Duplication was never
# function-preserving in production, and the test that said it was built its network with a masked
# diagonal. The jitter=0.05 cell is therefore ALSO the direct comparison against the old measured
# 702 / 9.3 / 0.0130, isolating the effect of that fix.
#
# PRE-REGISTERED, WRITTEN BEFORE THESE JOBS EXISTED. Read-out at 40000 iterations: active units
# under the scale-free criterion (0.05 * q95) from the participation trace, and activity dimensions
# per active unit (participation ratio of the firing-rate covariance), each gated on recomputed r2.
#   PREDICTION  the cells interpolate monotonically between copy and random: as the cosine to the
#               donor falls, the active count falls and dimensions per active unit rise.
#   A WIN       any cell with at least 500 active units AND at least 0.016 dimensions per active
#               unit -- most of duplication's recruitment, most of the way back to the control's
#               0.0202. Stated now so it cannot be relaxed later.
#   FALSIFIER   if every cell sits on a monotone line between copy and random and none clears both
#               bars, there is no sweet spot: duplication's redundancy is the price of its
#               recruitment, and the jitter level is not the knob worth having.
#   JITTER 0    tests the claim in this code's own comments, that without injected noise the twins
#               "take identical drive and identical gradients and never separate". These networks
#               inject INDEPENDENT recurrent noise per unit (sigma_rec = 0.05), so identical twins
#               see different noise, fire differently and receive different gradients from the
#               first step. If jitter 0 recruits and adds dimensions like jitter 0.05 does, the
#               injected jitter is doing nothing and the network's own noise separates the twins.
#   STABILITY   any cell discarding more than 5% of its gradient updates is reported as BROKEN
#               whatever its unit count.
#   TREADMILL   reinit_events against reinit_units_ever: many replacements with no gain in the
#               active count is a failure, not partial success.
#
# Cost: ~1h25m at N=1000 on this task, so about 21 GPU-hours for the grid. 4 h requested.
#   sbatch --array=1-15%8 slurm/SilentReLU_copy_perturb_della.slurm
#
# THE PERMUTATION CONTROL. PERMUTE=true scrambles the donor's incoming row across source units
# instead of copying it in place: the same multiset of weights, signs and zeros, different sources.
# Jitter keeps the donor's wiring POSITIONS and scrambles the VALUES; a permutation does the
# opposite. The jitter sweep found recruitment RISING as the copy is rotated away from its donor
# (685 active units at jitter 0, 752 at jitter 3.0, against 262 for a fresh random draw), so what a
# copy carries is not its exact weights. This control asks whether it is the weights' STATISTICS or
# their PLACEMENT, and it is the single-variable comparison against the jitter-0 cell.
#   PERMUTE=true SWEEP=copy_permute sbatch --array=1-3 slurm/SilentReLU_copy_perturb_della.slurm
#
# Output: $SAVE_TO/NBitFlipFlop_copy_perturb/EqType=h_k=3_N=1000_cn=<jitter>/

#SBATCH --job-name=CopyPerturb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=04:00:00
#SBATCH --array=1-15%8
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
if [ -z "$GITSHA" ]; then
  echo "FATAL: $REPO is not a git checkout - provenance of this run cannot be recorded." >&2; exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

K=3
N=${N_OVERRIDE:-1000}
IT=${ITERS_OVERRIDE:-40000}

CELL_NOISE=(0 0.05 0.3 1.0 3.0)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); CELL=$((TID / 3))
CN=${CELL_NOISE[$CELL]}

TAG="${SWEEP:-copy_perturb}/EqType=h_k=${K}_N=${N}_cn=${CN}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_${SWEEP:-copy_perturb}"
echo "Task ${SLURM_ARRAY_TASK_ID}: cell=${CELL} copy_noise=${CN} rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/NBitFlipFlop_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=NBitFlipFlop \
  task.n_inputs="$K" task.n_outputs="$K" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  trainer.prune_reinit=true trainer.synaptic_scaling=false \
  trainer.prune_args.reinit_mode=copy \
  "trainer.prune_args.copy_noise=${CN}" \
  "trainer.prune_args.copy_permute=${PERMUTE:-false}" \
  trainer.prune_args.copy_norm_preserve=true \
  trainer.prune_args.max_replace_frac=0.025 \
  trainer.prune_args.maturity=1000 \
  trainer.prune_args.utility=contribution \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
