#!/bin/bash
# WHICH REPLACEMENT RULE RECRUITS UNITS: OURS, THEIRS, OR SOMETHING BETWEEN? (DELLA)
#
# Dohare et al., Nature 632:768-774 (2024), "Loss of plasticity in deep continual learning",
# replace low-utility units during training to keep a network plastic across thousands of tasks.
# Four of their choices differ from ours, and one of the differences invalidates a conclusion we
# had already drawn.
#
#   1. WHAT REPLACES A UNIT. They resample the incoming weights and ZERO the outgoing ones, so a
#      new unit cannot disturb anything already learned and has to earn its output back through
#      the gradient. Our `random` mode resamples incoming weights and LEAVES the outgoing ones as
#      training left them. Those are different interventions, so our measured failure of `random`
#      (26,012 redraws over 591 units, 44 deaths each, ~1% survival, active count unmoved) says
#      NOTHING about their published method. `zero_out` is their rule; this sweep tests it.
#   2. WHO GETS REPLACED. Their utility is |r| * sum|outgoing weight|, run through a decaying
#      average. Ours was the firing rate alone, which scores a unit driving severed weights as
#      healthy. Utility is held at `contribution` in EVERY arm here so the mode comparison is not
#      confounded with it.
#   3. MATURITY. They protect a new unit for a fixed number of iterations WHATEVER IT DOES, because
#      a zeroed-output unit has zero utility and would otherwise be the next one chosen. Our strike
#      counter only protects a unit that starts firing, and at N=500 that produced 54.6
#      replacements per unit with the active count unmoved.
#   4. RATE. Theirs is ~1e-5 of units per step. Ours was uncapped: 4.7e-4 per step on CDDM at
#      N=1000, which was stable, and 1.4e-3 on this task at N=500, which discarded 29-44% of its
#      gradient updates and reached a clean loss of 273,489. Rate is the most likely cause of that
#      failure and it is the main axis here.
#
# GRID (24 jobs) = 8 cells x 3 seeds, at N=1000 on the 3-bit flip-flop, no penalties, 40k iters.
#   cells 0-5, the mode x rate factorial at maturity 1000:
#     0 zero_out rate 0.001   1 zero_out rate 0.005   2 zero_out rate 0.025
#     3 copy     rate 0.001   4 copy     rate 0.005   5 copy     rate 0.025
#   cells 6-7, maturity OFF at the middle rate, to isolate maturity against cells 1 and 4:
#     6 zero_out rate 0.005 maturity 0    7 copy rate 0.005 maturity 0
#   max_replace_frac is per CHECK and checks run every 100 iterations, so at N=1000 the three rates
#   are 1, 5 and 25 units per check = 1e-5, 5e-5 and 2.5e-4 of units per step. The lowest is
#   Dohare et al.'s rate; the highest is still below our uncapped CDDM run.
#   Index decode (TID = SLURM_ARRAY_TASK_ID - 1, 0..23): REP = TID % 3; CELL = TID / 3
#   sbatch --array=1-24%12 slurm/SilentReLU_replace_variants_della.slurm
#
# CELLS 8-11, THE NON-COPY RULES (--array=25-36), added 2026-09-24 after measuring what duplication
# actually buys. At N=1000 it raises the active count 2.5x (277 -> 705) while activity
# dimensionality rises only 1.4x (6.3 -> 9.1), so dimensions per active unit FALL from 0.020 to
# 0.013: the copies largely ride directions the network already used. `dead` dropout does the
# reverse -- 32.2 dimensions on 913 units, 0.034 each -- at a cost of 12 points of r2. The bar is
# therefore no longer "recruit units cheaply", which duplication already does; it is RECRUIT UNITS
# THAT ADD DIMENSIONS, which an exact copy cannot do by construction.
#   8  orth       a row orthogonal to every live unit's, so the new unit reads an input combination
#                 nobody reads. Verified to 1.9e-07. Falls back to a plain draw when the live rows
#                 already span R^N, since no complement exists then.
#   9  mix        a uniformly random blend of 4 live units: a twin of nobody (max cosine against
#                 any single donor 0.87) but placed where the network is active, so it fires.
#   10 bias_kick  no donor and NO WEIGHT CHANGED: a bias offset of -median(h) unfreezes the unit
#                 and leaves it to find its own function. PREDICTION recorded before the run
#                 (Pavel's): this fails, because the recurrent input that silenced the unit is
#                 still there and still training. If it treadmills where copy does not, the
#                 donor's FUNCTION is what makes duplication stick, not escaping the frozen state.
#   11 random     the plain redraw at the same rate and maturity, as the floor. Its earlier failure
#                 was at an UNCAPPED rate, so it has never been run under these settings.
# All four at rate 0.025 and maturity 1000, matching cell 5 (copy), so the rule is the only
# difference. READ-OUT: active units AND activity dimensions per active unit, not the count alone.
#
# NO CONTROL IS RUN HERE. The `none` arm of NBitFlipFlop_ff_revive at N=1000 is the matched
# control -- same task, size, iteration count and model config -- and has already landed at
# 274.7 +- 24.0 active units, clean loss 0.0775. The control path is untouched by the changes in
# this commit: prune_reinit=false returns before any of the new code. The `prune_copy` arm of that
# same sweep is the reference for OUR original rule (uncapped, participation utility, no maturity)
# and is still running.
#
# PRE-REGISTERED, WRITTEN BEFORE ANY OF THESE JOBS EXISTED. Read-out at 40,000 iterations, active
# units under the scale-free criterion (0.05 * q95), from the participation trace.
#   PRIMARY    active units against the matched control's 274.7 +- 24.0.
#   CO-PRIMARY clean dropout-off loss (loss_clean_train), NOT the training-pass loss.
#   STABILITY  any cell discarding more than 5% of its gradient updates is reported as BROKEN
#              whatever its unit count. This is the rule that failed our N=500 duplication run and
#              both synaptic-scaling attempts, and it is the reason rate is the main axis.
#   TREADMILL  reinit_events vs reinit_units_ever. Many replacements with no gain in the active
#              count is a failure, not partial success.
#   WHAT WOULD CHANGE MY MIND, stated so it can: if `zero_out` matches or beats `copy` at equal
#              rate, then duplication was never the important part and the donor machinery should
#              be dropped for the simpler rule. If low rates are stable and high rates are not,
#              the N=500 blow-up was a rate problem and gamma was treating a symptom. If maturity
#              changes nothing, the treadmill was a rate artifact rather than a protection failure.
#
# Cost: no extra forward pass; checks run every 100 iterations. ~1h25m at N=1000 on this task.
# 4 h requested.
# Output: $SAVE_TO/NBitFlipFlop_replace_variants/EqType=h_k=3_N=1000_mode=<m>_rate=<r>_mat=<m>/

#SBATCH --job-name=Repvar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=04:00:00
#SBATCH --array=1-24%12
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
N=${N_OVERRIDE:-1000}
# N_OVERRIDE runs this same grid at another size. Use a distinct SWEEP when doing so, or
# cells at different sizes land in folders that differ only by a field the analysis does
# not group on.
IT=${ITERS_OVERRIDE:-40000}

CELL_MODE=(zero_out zero_out zero_out copy copy copy zero_out copy orth mix bias_kick random)
CELL_RATE=(0.001    0.005    0.025    0.001 0.005 0.025 0.005    0.005 0.025 0.025 0.025 0.025)
CELL_MAT=(1000      1000     1000     1000  1000  1000  0        0     1000 1000 1000     1000)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); CELL=$((TID / 3))
MODE=${CELL_MODE[$CELL]}
RATE=${CELL_RATE[$CELL]}
MAT=${CELL_MAT[$CELL]}

TAG="${SWEEP:-replace_variants}/EqType=h_k=${K}_N=${N}_mode=${MODE}_rate=${RATE}_mat=${MAT}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_${SWEEP:-replace_variants}"
echo "Task ${SLURM_ARRAY_TASK_ID}: cell=${CELL} mode=${MODE} rate=${RATE} maturity=${MAT} rep=${REP} iters=${IT}"
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
  "trainer.prune_args.reinit_mode=${MODE}" \
  "trainer.prune_args.max_replace_frac=${RATE}" \
  "trainer.prune_args.maturity=${MAT}" \
  trainer.prune_args.mix_k=4 \
  trainer.prune_args.utility=contribution \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
