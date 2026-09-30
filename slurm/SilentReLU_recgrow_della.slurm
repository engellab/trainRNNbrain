#!/bin/bash
# DOES GROWING ONLY THE RECURRENT BUDGET RECRUIT WITHOUT BUILDING SENSORY UNITS? (DELLA)
#
# The rescale rule tilts a dormant unit's incoming weights from inhibition toward excitation.
# With rescale_normalize=true and rescale_norm_mode=self, each row is pinned at the length it
# already had, so the rule can only redistribute a budget it never changes -- and that budget is
# several-fold too small to begin with (an active unit's incoming recurrent row is 4.9x longer
# than a silent unit's at N=1000, its input row 22.8x longer).
#
# The previous sweep (norm_mode, 14690314) let the row GROW toward the firing population's median.
# It grew BOTH incoming rows, which was wrong. Measured there, at N=1000, 3 seeds per cell:
#
#     cell                         active  sensory frac  |W_inp| act  |W_rec| act   dims    r2
#     untreated control               278         0.122        0.769        1.802   5.56  0.945
#     self          tgt 6.0           382         0.040        0.177        1.252   4.90  0.925
#     median_active tgt 6.0 allcols   464         0.154        0.978        2.068   4.46  0.914
#     median_active tgt 6.0           491         0.168        1.052        2.356   3.46  0.881
#     median_active tgt 2.5           456         0.209        1.180        1.685   1.33  0.916
#
# Sensory fraction is the share of an active unit's input current arriving through W_inp rather
# than W_rec. Across the four TREATED cells it runs in strict opposite order to dimensionality.
# The reason is structural: the input row is n_inputs wide -- 3 on this task -- against a recurrent
# row of N, so a unit recruited by enlarging its input row can only become a function of three
# signals. On a task whose computation runs through intermediate variables such a unit adds
# nothing, because the variables it would need to read live in the recurrent population.
# rescale_rows_ made this worse than it looks: it rescaled W_inp in FULL on every boost while
# restricting the recurrent part to firing columns.
#
# As of commit 4287801, W_inp is pinned at its own length however rescale_norm_mode is set, so
# growth reaches the recurrent row only. THIS SWEEP TESTS WHETHER THAT IS ENOUGH.
#
# ⚠️ WHAT IS NOT YET SEPARATED, stated because it is the reason this sweep exists rather than a
# conclusion it assumes: the median_active cells above grew their recurrent rows a long way too
# (1.252 -> 2.356, past the untreated control's 1.802) and dimensionality still fell. Recurrent
# growth has never been observed WITHOUT input growth alongside it, so it is not established that
# recurrent growth is the harmless half. If it is not, this sweep says so.
#
# GRID (12 jobs) = 3 cells x 4 seeds, N=1000 on the 3-bit flip-flop, 40k iterations, no penalties.
# Four seeds rather than three: the `self` arm gave 422 active units in one sweep and 381 in the
# next, so seed noise, not cell design, has been the limiting factor on every comparison here.
#   cell 0  self          tgt 6.0  firing columns   in-sweep control; unaffected by commit 4287801
#   cell 1  median_active tgt 6.0  firing columns   recurrent growth, onto units that already fire
#   cell 2  median_active tgt 6.0  ALL columns      recurrent growth onto quiet columns too, which
#                                                   matters more now that growth is the only way
#                                                   magnitude enters: a lengthening row may want
#                                                   mass on columns that fire later.
#   Index decode (TID = SLURM_ARRAY_TASK_ID - 1, 0..11): REP = TID % 4; CELL = TID / 4
#
# PRE-REGISTERED, WRITTEN BEFORE THESE JOBS EXISTED. Read-out at 40,000 iterations.
#   PRIMARY    active units against cell 0 of THIS sweep, not against any earlier number.
#   CO-PRIMARY activity dimensionality. Cell 0 sits at 4.90 and the untreated control at 5.56. A
#              cell that recruits more and falls below 4.5 has not improved on cell 0, whatever
#              its unit count -- that bar is the same one the previous sweep failed.
#   MECHANISM  sensory fraction. The untreated control is 0.122. Any cell above that has built
#              sensory units and is rejected on this read-out alone, however it scores elsewhere.
#   CODE CHECK median |W_inp| row norm of active units must NOT differ between cells 0, 1 and 2.
#              W_inp is pinned as of 4287801; if these differ the pin does not work and every
#              other number in the sweep is uninterpretable.
#   STABILITY  more than 5% of gradient updates discarded, OR mean r2 more than 0.10 below the
#              untreated control's 0.945, is BROKEN whatever the unit count.
#   WHAT WOULD CHANGE MY MIND: if cell 1 recruits past cell 0 at dimensionality >= 4.5 and sensory
#              fraction <= 0.122, then restoring the recurrent budget was the right fix and the
#              previous sweep failed only because it also inflated the input row. If cell 1
#              collapses dimensionality again with the input row pinned, then magnitude
#              restoration per se is the problem, not which channel carries it, and `self` at
#              target 6.0 is the ceiling of this whole family -- which is the result that should
#              then be written up.
#
# MEASURED 3h20m at N=1000 on this task, ~3.3 iterations/second, several array tasks per GPU.
# 5 h requested: the previous sweep finished with only 40 min of headroom against a 4 h limit.
# Output: $SAVE_TO/NBitFlipFlop_recgrow/EqType=h_k=3_N=1000_nm=<mode>_tgt=6.0_act=<bool>/

#SBATCH --job-name=RecGrow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=05:00:00
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
TGT=6.0

CELL_NM=(self          median_active median_active)
CELL_ACT=(true         true          false)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 4)); CELL=$((TID / 4))
NM=${CELL_NM[$CELL]}
ACT=${CELL_ACT[$CELL]}

TAG="recgrow/EqType=h_k=${K}_N=${N}_nm=${NM}_tgt=${TGT}_act=${ACT}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_recgrow"
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
