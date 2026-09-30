#!/bin/bash
# HOW FAR UP DOES THE MATURITY TARGET GO? (DELLA)
#
# The first ladder answered the question it was built for: the target is a threshold, not a knife
# edge. Measured at N = 1000 on the 3-bit flip-flop, 40k iterations, against an untreated control
# of 278 active units at 5.56 dimensions and r2 0.945:
#
#   target  2.5   384 active   3.36 dims        (collapses)
#   target  4.0   438 active   3.33 dims        (collapses)
#   target  6.0   465 active   5.89 dims   r2 0.9321
#   target  8.0   455 active   6.35 dims   r2 0.9349
#   target 10.0   453 active   7.20 dims   r2 0.9320
#
# So there is a sharp transition between 4.0 and 6.0, and ABOVE it dimensionality keeps rising
# while the active count and r2 stay flat. Nothing yet bounds the top: 10.0 is simply the highest
# target run, and it is the best of them.
#
# This ladder continues to 14, 20 and 30. The target is a multiple of the firing-rate penalty's
# cap (0.200 at N = 1000), so 30.0 asks a revived unit to reach an activity of 6.0 -- well inside
# the range active units occupy (participation 1-15) but far above what the penalty would ask of
# any unit.
#
# ⚠️ A CEILING IS EXPECTED SOMEWHERE AND THE MECHANISM IS KNOWN. The cumulative boost budget is
# rescale_cap = 64 at rescale_alpha = 1.002, about 2,082 boost steps, after which a unit is cut
# loose whether or not it matured. The higher the target, the more units spend the whole budget and
# lose gradient protection while still weak -- which is the failure measured at target 2.5, where
# units revived, protection lifted, and the gradient put them back at an excitation/inhibition
# ratio of 0.17-0.52 against a control's 0.79-1.14. If dimensionality falls at the top of THIS
# ladder, the budget is the binding constraint and raising rescale_cap is the next move, not
# raising the target further.
#
# GRID (12 jobs) = 3 targets x 4 seeds, N=1000 on the 3-bit flip-flop, 40k iterations, no penalties.
#   cell 0  target 14.0      cell 1  target 20.0      cell 2  target 30.0
#   Index decode (TID = SLURM_ARRAY_TASK_ID - 1, 0..11): REP = TID % 4; CELL = TID / 4
#
# NO CONTROL CELL. Target 6.0 was measured on this same commit in NBitFlipFlop_target_ladder
# (465 active, 5.89 dims, r2 0.9321, n=4) and is the reference; nothing in the code has changed
# since, so re-running it would only re-measure seed noise.
#
# PRE-REGISTERED. Read-out at 40,000 iterations, scale-free criterion.
#   PRIMARY    activity dimensionality against target 10.0's 7.20. Rising = the threshold reading
#              holds and the useful range is wider than measured. Falling = a ceiling, and the
#              budget check below says whether the budget is why.
#   CO-PRIMARY active units, which must not fall back toward the untreated 278.
#   STABILITY  r2 more than 0.10 below the untreated 0.945, or more than 5% of gradient updates
#              discarded, is BROKEN whatever else the cell shows.
#   SPREAD     the rescale family compresses the participation distribution -- spread 59k-85k at
#              targets 6-10 against the control's 265k -- and synaptic noise does not. If the
#              compression worsens up this ladder, that is a cost to weigh against the dimensions.
#
# ~3h20m per job at N=1000. 6 h requested.
# Output: $SAVE_TO/NBitFlipFlop_target_ladder/EqType=h_k=3_N=1000_tgt=<t>/

#SBATCH --job-name=TgtHigh
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

CELL_TGT=(14.0 20.0 30.0)

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
