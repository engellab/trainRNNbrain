#!/bin/bash
# IS 5% THE LIMITING FACTOR? A DROPOUT-RATE AND TARGETING-SHARPNESS LADDER (SPOCK)
#
# WHY. Participation dropout at the project's standard setting (drop_rate = 0.05, dropout_beta = 1)
# keeps ~100 more units alive on the 3-bit flip-flop at N = 1000 and costs nothing measurable, but
# that is only a tenth of what the firing-rate-magnitude penalty recovers. Before the paper says
# "dropout is a weak remedy", it has to say whether the weakness is in the METHOD or only in the
# SETTING we happened to use. Measured from a real trained network's participation vector
# (2026-09-20), the standard setting is mild on both axes:
#
#   p_drop_i = clamp(drop_rate * N * softmax(beta * participation)_i, 0, 0.999),  Bernoulli, (N,1),
#   resampled EVERY iteration and SHARED across the batch.
#
#   at drop_rate = 0.05, beta = 1, N = 1000:  E[#dropped] = 50.0 units per iteration (exactly
#   drop_rate * N, because the softmax weights sum to one and nothing clips), and the targeting is
#   weak - the 50 highest-participation units absorb only 14.5% of the drop mass against 5% under
#   uniform sampling, i.e. about 3x enrichment. So the busiest units are barely preferred.
#
# This array walks both knobs, with the predicted effective drop count printed for each arm (again
# measured from a real participation vector at 40k, not assumed):
#
#   arm              drop_rate  beta   E[#dropped]/iter   top-50 share of drops
#   A  rate x2          0.10      1        100.0                 14.5%
#   B  rate x4          0.20      1        200.0                 14.5%
#   C  rate x8          0.40      1        391.5  (41 clipped)   12.7%
#   D  sharp targeting  0.05      4         42.8  (7 clipped)    58.9%
#
#   beta = 16 was evaluated and REJECTED before submission: the softmax collapses onto a handful of
#   units (E[#dropped] = 5.7 of 1000, 99.8% of the mass on the top 50), so it does not test
#   "sharper targeting", it tests "permanently ablate the six busiest units". Not run.
#
# PRE-REGISTERED, WRITTEN BEFORE ANY OF THESE JOBS EXISTED. Read-out at 40,000 iterations, live
# units under BOTH criteria (scale-free 0.05 q95, and the flip-flop absolute 4e-2), plus the clean
# task loss. Reference cells already on disk, read at the same 40k iteration:
#
#   no dropout           scale-free 387.0 +- 37.3   absolute 416.2 +- 38.9   (n = 4)
#   dropout dead, 0.05   scale-free 487.0 +- 28.0   absolute 508.1 +- 19.6   (n = 7)
#
#   H1 "a higher drop rate recovers more units" is ACCEPTED for an arm only if its mean exceeds the
#      0.05 reference by more than 3 reference sd under BOTH criteria, i.e. > 571 (scale-free) AND
#      > 567 (absolute). H2 is the same bar for arm D against the same reference.
#   COST GUARD: an arm that clears the bar but whose clean loss exceeds the no-dropout reference by
#      more than 10% is reported as a trade, not as a win.
#   FALSIFICATION, and the outcome I actually expect: if no arm clears the bar, the paper says the
#      drop rate is NOT the limiting factor and dropout's ceiling is structural. That is a result,
#      it goes in the figure, and it is the honest answer to "did you just not try hard enough".
#
# WHY 40,000 ITERATIONS IS A VALID READ-OUT, AND WHY NO CONTROLS ARE RE-RUN HERE. Nothing in
# training depends on max_iter in this configuration: the only max_iter-dependent quantity in
# Trainer is the noise-annealing schedule (`anneal_noise_levels_`, centred at max_iter/3), and
# `trainer_ptrack_freshbatch.yaml` sets `anneal_noise: False`, so it is never called. There is no
# learning-rate scheduler. A 40k run is therefore step-for-step identical to the first 40k steps of
# a 150k run, and these arms are directly comparable to the existing 150k cells read at 40k - which
# is where the reference numbers above come from. THIS WAS CHECKED, NOT ASSUMED; had
# `anneal_noise` been true, every arm here would have needed its own matched control.
# The effect is fully resolvable this early: at 40k the 0.05 dead cell already sits 100 units above
# no-dropout (487 vs 387), the same gap it has at 150k (362 vs 263).
#
# WHY `dead` ONLY. `dead` (the unit is removed from the recurrent dynamics AND the read-out) is the
# cleaner ablation and is the arm whose task cost is positively equivalent to no dropout
# (TOST p = 1.2e-05); `mute` (read-out weights masked, dynamics untouched) costs +3.1%. If the rate
# ladder moves anything, extending it to `mute` is a cheap follow-up.
#
# Index decode (TID = SLURM_ARRAY_TASK_ID - 1):  ARM = TID / 3 ; rep = TID % 3
#   ARM 0 -> rate 0.10 beta 1 | 1 -> rate 0.20 beta 1 | 2 -> rate 0.40 beta 1 | 3 -> rate 0.05 beta 4
#
# Cost: dropout runs do two forward passes per iteration, ~0.5-0.6 s/iter at N = 1000, so 40k is
# ~6-7 h. 12 h requested so a slow node cannot time out - a timeout loses the ParticipationTrace
# entirely, since it is written only when the run completes.
#   sbatch --array=1-12%12 slurm/SilentReLU_flipflop_droprate_spock.slurm
#
# Output: $SAVE_TO/NBitFlipFlop_std_droprate/EqType=h_k=3_N=1000_pen=none_do=dead_rate=<r>_beta=<b>/

#SBATCH --job-name=FFdrate
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-12%12
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

REPO="${REPO:-$HOME/trainRNNbrain}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2; exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "WARNING: $REPO has uncommitted changes; the commit above does not fully describe this run." >&2
fi

K=3
N=1000
KIND=dead
IT=${ITERS_OVERRIDE:-40000}

ARM_RATE=(0.10 0.20 0.40 0.05)
ARM_BETA=(1    1    1    4)

TID=$((SLURM_ARRAY_TASK_ID - 1))
ARM=$((TID / 3))
REP=$((TID % 3))
RATE=${ARM_RATE[$ARM]}
BETA=${ARM_BETA[$ARM]}

TAG="std_droprate/EqType=h_k=${K}_N=${N}_pen=none_do=${KIND}_rate=${RATE}_beta=${BETA}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_std_droprate"
echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K} N=${N} pen=none dropout=${KIND} rate=${RATE} beta=${BETA} rep=${REP} iters=${IT}"
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
  trainer.dropout=true \
  "trainer.dropout_args.dropout_kind=${KIND}" \
  "trainer.dropout_args.drop_rate=${RATE}" \
  "trainer.dropout_args.dropout_beta=${BETA}" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
