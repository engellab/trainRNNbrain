#!/bin/bash
# IS THE MATURITY TARGET A THRESHOLD OR A KNIFE EDGE? (DELLA)
#
# The rescale rule releases a dormant unit once its activity reaches rescale_target_frac times the
# firing-rate penalty's own cap (0.200 at N = 1000, so a target of 6.0 is an activity of 1.202).
# Everything else in the family is settled: adding magnitude collapses activity dimensionality
# whichever weight matrix carries it (2.33 growing W_rec alone, 1.90 growing it onto every column,
# against 5.55 for an untreated control), so the rule redistributes at fixed row length and only
# the target is left open.
#
# WHAT IS KNOWN, AND WHY IT IS NOT ENOUGH. Target 6.0 recruits at dimensionality parity and has
# done so in three independent runs: 428 units at 5.80 dimensions, 381 at 4.90, 474 at 5.66,
# against a control's 278 at 5.55. Targets 2.5 and 4.0 recruit too -- 384 and 438 units -- but
# collapse dimensionality to 3.36 and 3.33. So the working configuration sits directly above two
# that fail, and nothing yet says whether 6.0 is the bottom of a plateau or a point.
#
#   THRESHOLD: 8.0 and 10.0 also hold near 5.5 dimensions. Releasing a unit late enough is what
#              makes the rule work, the claim is mechanistic, and 6.0 is simply the cheapest
#              target that clears the bar.
#   KNIFE EDGE: 8.0 or 10.0 collapse as 2.5 and 4.0 did. Then 6.0 was found by sweeping, the arm
#              is too fragile to build a paper claim on, and that is the finding.
#
# A SPECIFIC MECHANISM COULD PRODUCE THE KNIFE EDGE, which is why the upper end is worth running
# rather than assumed safe. The cumulative boost budget is rescale_cap = 64 at
# rescale_alpha = 1.002, so a unit has about 2,082 boost steps before it is cut loose whether or
# not it matured. A high enough target means more units exhaust that budget and lose gradient
# protection while still weak -- the exact failure mode measured at target 2.5, where units revived,
# protection lifted, and the gradient put them back with an excitation/inhibition ratio of
# 0.17-0.52 against a control's 0.79-1.14. If that is what limits the target, the collapse returns
# at the top of this ladder.
#
# GRID (12 jobs) = 3 targets x 4 seeds, N=1000 on the 3-bit flip-flop, 40k iterations, no penalties.
#   cell 0  target  6.0   IN-SWEEP CONTROL
#   cell 1  target  8.0
#   cell 2  target 10.0
#   Index decode (TID = SLURM_ARRAY_TASK_ID - 1, 0..11): REP = TID % 4; CELL = TID / 4
#
# ⚠️ WHY TARGET 6.0 IS RE-RUN RATHER THAN TAKEN FROM NBitFlipFlop_recgrow. Two commits have landed
# since that sweep: the synaptic-noise forward pass (inert at sigma_w = 0, pinned bit-for-bit by
# tests/test_synaptic_weight_noise.py) and the deletion of eight never-used penalties (all of whose
# lambdas were 0). Both SHOULD be inert for this configuration, and neither is verified to be, and
# this arm's mean has moved 428 -> 381 -> 474 across three runs of nominally identical settings. A
# 20% swing between sweeps is too large to also absorb an unverified code difference, so the
# reference is measured here.
#
# PRE-REGISTERED, WRITTEN BEFORE THESE JOBS EXISTED. Read-out at 40,000 iterations.
#   PRIMARY    activity dimensionality (participation ratio over active units) against cell 0 of
#              THIS sweep. A target holds if its mean sits within the control's per-seed range;
#              it fails if it lands near the 3.3 that targets 2.5 and 4.0 gave.
#   CO-PRIMARY active units, which must not fall back to the untreated 278.
#   STABILITY  more than 5% of gradient updates discarded, OR mean r2 more than 0.10 below the
#              untreated control's 0.945, is BROKEN whatever else the cell shows. Cell 1 of the
#              recgrow sweep failed exactly this way at r2 0.816.
#   MECHANISM  the fraction of episodes that exhaust the 64x budget without reaching the target.
#              If dimensionality collapses at the top of the ladder, this number should rise with
#              it; if it collapses while this stays flat, the budget is not the cause and the
#              explanation is elsewhere.
#   WHAT WOULD CHANGE MY MIND: if 8.0 and 10.0 both hold, the target is a threshold and the two
#              failing low targets are explained by units being released before they can hold
#              themselves up. If either collapses, 6.0 is a point rather than a plateau and the
#              honest write-up says the rule works only in a narrow band that we located by sweep.
#
# ~3h20m per job at N=1000 measured on this cell. 6 h requested.
# Output: $SAVE_TO/NBitFlipFlop_target_ladder/EqType=h_k=3_N=1000_tgt=<t>/

#SBATCH --job-name=TgtLadder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=06:00:00
#SBATCH --array=1-12%12
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

CELL_TGT=(6.0 8.0 10.0)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 4)); CELL=$((TID / 4))
TGT=${CELL_TGT[$CELL]}

TAG="target_ladder/EqType=h_k=${K}_N=${N}_tgt=${TGT}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_target_ladder"
echo "Task ${SLURM_ARRAY_TASK_ID}: cell=${CELL} target=${TGT} rep=${REP} iters=${IT}"
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
  trainer.prune_args.rescale_norm_mode=self \
  "trainer.prune_args.rescale_target_frac=${TGT}" \
  trainer.prune_args.rescale_active_only=true \
  trainer.prune_args.rescale_normalize=true \
  trainer.prune_args.rescale_alpha=1.002 \
  trainer.prune_args.rescale_cap=64.0 \
  trainer.prune_args.rescale_protect_frac=0.20 \
  trainer.prune_args.check_every=1 trainer.prune_args.patience=1 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
