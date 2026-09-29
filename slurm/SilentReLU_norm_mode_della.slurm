#!/bin/bash
# CAN A REVIVED UNIT BE GIVEN ITS INPUT BUDGET BACK, INSTEAD OF ONLY REDISTRIBUTING IT? (DELLA)
#
# The rescale rule tilts a dormant unit's incoming weights from inhibition toward excitation,
# holding the row's L2 norm fixed so the event is a pure redistribution. At maturity target 6.0
# that recruits 400/432/453 units against a control's 257/272/305, at dimensionality parity
# (5.80 against 5.56) and 1 point of r2 (0.934 against 0.945).
#
# THE PROBLEM THIS SWEEP TESTS. Holding the row at its OWN length cannot add magnitude, and
# magnitude is a large part of what a silent unit lacks. Measured on unpenalised controls at
# N=1000: an active unit's incoming recurrent row is 4.9x longer than a silent unit's (1.818
# against 0.373) and its input row 22.8x longer (0.769 against 0.034). So a revived unit is
# redistributing a budget several-fold too small, and the magnitude it ends up with arrives from
# the gradient AFTER maturity, when its rows leave protection -- by draining the units that had
# it. Under the target-6.0 arm, active-unit input rows fell 4.4x (0.769 -> 0.173) while
# silent-unit rows rose 2.9x (0.034 -> 0.097); total input weight mass fell from 211 to 73. The
# 22.8x spread closed to 1.78x mostly because the ceiling came down.
#
# rescale_norm_mode=median_active grows a short row toward the median row norm of the FIRING
# population, at most rescale_alpha per step and never past the median, and never shrinks a row
# already longer. Turning normalisation off entirely is the other way to add magnitude and it
# diverged (r2 -44.8 at target 2.5, 988/1000 units scoring "active" on a dead network), so this
# is the bounded version of the same idea.
#
# GRID (12 jobs) = 4 cells x 3 seeds, N=1000 on the 3-bit flip-flop, 40k iterations, no penalties.
#   cell 0  self          tgt 6.0   the published arm, re-run on this commit as the matched control
#   cell 1  median_active tgt 6.0   the same arm with the budget allowed to grow
#   cell 2  median_active tgt 2.5   does a growing budget rescue the target that failed before?
#   cell 3  median_active tgt 6.0, rescale_active_only=false
#                                   the boost currently only touches columns of FIRING units, which
#                                   is right when redistributing (a synapse from a silent source
#                                   delivers nothing) but may be wrong when GROWING: a row that is
#                                   allowed to lengthen could take mass on quiet columns that fire
#                                   later. This is the one cell where the two flags interact.
#   Index decode (TID = SLURM_ARRAY_TASK_ID - 1, 0..11): REP = TID % 3; CELL = TID / 3
#
# PRE-REGISTERED, WRITTEN BEFORE THESE JOBS EXISTED. Read-out at 40,000 iterations.
#   PRIMARY   active units against cell 0 on the same commit, NOT against the numbers above.
#   CO-PRIMARY activity dimensionality (participation ratio over active units). The whole point of
#             target 6.0 was that it recruited WITHOUT collapsing dimensions, which targets 2.5 and
#             4.0 did (3.33). A cell that recruits more and drops below ~4.5 has not improved on it.
#   MAGNITUDE the median input row norm of ACTIVE units. Cell 0 sits at 0.173 against an untreated
#             control's 0.769. If median_active recruits by lifting the floor, that number RISES.
#             If it recruits the same way cell 0 did, it stays near 0.173 and this sweep has bought
#             nothing but a different route to the same homogenisation.
#   STABILITY more than 5% of gradient updates discarded, OR mean r2 more than 0.10 below the
#             control's 0.945, is BROKEN whatever the unit count. The spike gate alone has passed
#             two diverged runs on this rule, which is why the r2 clause is here.
#   WHAT WOULD CHANGE MY MIND: if cell 1 matches cell 0 on every read-out, then row LENGTH was
#             never the constraint and the deficit measured on the controls is a consequence of
#             silence rather than a cause of it. If cell 2 works where target 2.5 failed, the
#             earlier failure was a magnitude problem misdiagnosed as a maturity problem, and the
#             target can come back down.
#
# MEASURED 3h20m at N=1000 on this task, at ~3.3 iterations/second for 40,000 iterations,
# with several array tasks sharing a GPU. 4 h requested, so the margin is ~40 min. The 1h25m
# this header first claimed was carried over from an earlier sweep without being checked
# against this one; size the limit from a rate measured on the sweep being submitted.
# Output: $SAVE_TO/NBitFlipFlop_norm_mode/EqType=h_k=3_N=1000_nm=<mode>_tgt=<t>_act=<bool>/

#SBATCH --job-name=NormMode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=04:00:00
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

CELL_NM=(self          median_active median_active median_active)
CELL_TGT=(6.0          6.0           2.5           6.0)
CELL_ACT=(true         true          true          false)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); CELL=$((TID / 3))
NM=${CELL_NM[$CELL]}
TGT=${CELL_TGT[$CELL]}
ACT=${CELL_ACT[$CELL]}

TAG="norm_mode/EqType=h_k=${K}_N=${N}_nm=${NM}_tgt=${TGT}_act=${ACT}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_norm_mode"
echo "Task ${SLURM_ARRAY_TASK_ID}: cell=${CELL} norm_mode=${NM} target=${TGT} active_only=${ACT} rep=${REP}"
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
  "trainer.prune_args.rescale_norm_mode=${NM}" \
  "trainer.prune_args.rescale_target_frac=${TGT}" \
  "trainer.prune_args.rescale_active_only=${ACT}" \
  trainer.prune_args.rescale_normalize=true \
  trainer.prune_args.rescale_alpha=1.002 \
  trainer.prune_args.rescale_cap=64.0 \
  trainer.prune_args.rescale_protect_frac=0.20 \
  trainer.prune_args.check_every=1 trainer.prune_args.patience=1 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
