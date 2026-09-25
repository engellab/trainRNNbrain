#!/bin/bash
# CAN A SILENT UNIT BE TALKED BACK INTO FIRING, WITHOUT REPLACING WHAT IT KNOWS? (DELLA)
#
# WHY. Across the revival rules measured at N=1000 (rate 0.025, maturity 1000, 3 seeds each), every
# rule that discarded a unit's trained incoming weights landed between 262 and 333 active units --
# random 262, zero_out 289, orth 294, mix 333, against a control's 275. The two that KEPT trained
# weights reached 373 (bias_kick, which touches no weight at all) and 702 (copy, which imports a
# working unit's). Keeping what the unit already knows is the property that separates them.
#
# This rule keeps the unit's own incoming weights and only rescales them: excitation up by alpha,
# inhibition down by alpha, every step, until the unit fires. Outgoing weights are redrawn at
# 1/sqrt(N) each step rather than zeroed, so a revived unit is visible to the loss immediately --
# zeroing makes a firing unit invisible, which leaves its incoming weights with no gradient until
# the outgoing column grows back, and is the likeliest reason zero_out did nothing here.
#
# WHAT MAKES THIS DIFFERENT FROM synaptic_scaling_, WHICH BLEW UP IN SIX CONFIGURATIONS. That rule
# scaled every unit toward a population set-point at every event and never stopped, so a unit it
# could not rescue stayed below the set-point and was scaled at all 400 events. Measured on the one
# configuration that survived: the rows of units it FAILED to revive ended at twice the control's,
# while the rows of units it rescued were only 15% above. Three things differ here -- it stops the
# moment the unit passes the activity criterion, it never scales an active unit down, and the
# cumulative boost per unit is capped at 8x, so "until it revives" is bounded when it never does.
#
# WHY ALPHA IS SO SMALL. Applied every step, the rule pushes the same direction every step, unlike
# gradient steps which partly cancel, so matching a gradient step's size (about 0.8% relative at
# lr 4.6e-4 on a weight of 0.06) would boost a unit 148-fold inside 1000 steps. The step comes from
# the timescale instead. Measured on the control networks, no silent unit has zero excitatory input
# -- all ~680 are reachable in principle -- the median needs a total boost of about 1.25 and the
# 90th percentile 3-5. (Approximate: participation stands in for presynaptic firing rate, which
# ranks correctly but is not the rate.) At 1.0005 per step the median revives in ~450 steps and the
# tail in ~3200; at 1.002, ~110 and ~800.
#
# GRID (12 jobs) = 2 normalisation x 2 alpha x 3 seeds, N=1000, 3-bit flip-flop, 40000 iterations,
# check every step, patience 1, maturity 0, every silent unit boosted each event.
#   normalize in {true, false}   alpha in {1.0005, 1.002}
# normalize=true holds each incoming row's total synaptic weight fixed, making the event a pure
# redistribution from inhibition to excitation. It is the only setting synaptic scaling was ever
# stable under (3/3 seeds, 0.02-2.5% of updates discarded) where turning it off diverged 3/3 at the
# same step size. It is swept anyway because this rule differs from that one in three ways and the
# stability may no longer depend on it -- and because normalisation caps how far a unit can be
# pushed, which is the tail the rule is meant to reach.
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..11): REP = TID%3; CELL = TID/3
#   CELL 0: norm=true  alpha=1.0005      CELL 1: norm=true  alpha=1.002
#   CELL 2: norm=false alpha=1.0005      CELL 3: norm=false alpha=1.002
#
# PRE-REGISTERED, WRITTEN BEFORE THESE JOBS EXISTED. Read-out at 40000 iterations: active units
# under the scale-free criterion (0.05 * q95) from the participation trace, AND activity dimensions
# per active unit, each gated on a recomputed r2 that reproduces the stored value.
#   BASELINES, already measured at this size and iteration count: control 275 active units
#   (254-301 across seeds), 5.6 clean dimensions, 0.0202 dimensions per active unit, r2 0.9450.
#   A WIN, fixed now so it cannot be relaxed later: a cell whose active-unit seed range clears the
#   control's 254-301 AND whose dimensions per active unit stay at or above 0.016. bias_kick
#   already reaches 373 units at 0.0161, so beating THAT means more than 380 units at 0.016 or
#   better.
#   FALSIFIER: if the active count sits inside 254-301 in every cell, gradual unsuppression does
#   not work here, and the earlier finding that only rules keeping trained weights recruit anything
#   was about bias_kick and copy specifically, not about trained weights as such.
#   TREADMILL, the measurement that matters most for this rule: of the units it boosts, what
#   fraction is still active at the end. bias_kick lifts about 475 units per network and keeps 70
#   of them (15%). Anything at or below that is a treadmill whatever the headline count says.
#   STABILITY: any cell discarding more than 5% of its gradient updates is reported as BROKEN
#   whatever its unit count. Six synaptic-scaling configurations failed exactly here.
#
# Cost: MEASURED, not estimated, because checking every step runs participation_from_states_ 100x
# more often than the usual check_every=100 and that function sorts 307,200 values per unit.
# Job 14401617, 300 iterations at N=1000 on this task, same code and settings as below:
#   check_every=100  72.06 s  ->  0.2402 s/iteration
#   check_every=1    87.55 s  ->  0.2918 s/iteration     (+21.5%)
# The ratio is what transfers; the absolute pace depends on which GPU the job lands on. Against the
# production pace of this task at N=1000 (1h25m for 40000 iterations) that is about 1h43m per job;
# taken at the absolute measured rate it is 3h15m. 6 h requested, covering both.
#   sbatch --array=1-12%6 slurm/SilentReLU_rescale_revive_della.slurm
#
# Output: $SAVE_TO/NBitFlipFlop_rescale_revive/EqType=h_k=3_N=1000_norm=<t|f>_a=<alpha>/

#SBATCH --job-name=RescaleRev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=06:00:00
#SBATCH --array=1-12%6
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

REPO="${REPO:-$HOME/trainRNNbrain_rescale}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2
  echo "       submit with --chdir=\$REPO; python -c puts the working directory ahead of PYTHONPATH." >&2
  exit 1
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

CELL_NORM=(true true false false)
CELL_ALPHA=(1.0005 1.002 1.0005 1.002)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); CELL=$((TID / 3))
NORM=${CELL_NORM[$CELL]}
ALPHA=${CELL_ALPHA[$CELL]}
NTAG=$([ "$NORM" = "true" ] && echo t || echo f)

TAG="${SWEEP:-rescale_revive}/EqType=h_k=${K}_N=${N}_norm=${NTAG}_a=${ALPHA}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_${SWEEP:-rescale_revive}"
echo "Task ${SLURM_ARRAY_TASK_ID}: cell=${CELL} normalize=${NORM} alpha=${ALPHA} rep=${REP} iters=${IT}"
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
  trainer.prune_args.reinit_mode=rescale \
  trainer.prune_args.check_every=1 \
  trainer.prune_args.patience=1 \
  trainer.prune_args.maturity=0 \
  trainer.prune_args.max_replace_frac=1.0 \
  "trainer.prune_args.rescale_alpha=${ALPHA}" \
  "trainer.prune_args.rescale_normalize=${NORM}" \
  trainer.prune_args.rescale_cap=8.0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
