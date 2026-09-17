#!/bin/bash
# DROPOUT x PENALTY on the 3-bit flip-flop, N=1000: TAKE EVERY CONDITION TO SEVEN SEEDS (SPOCK)
#
# The first grid (array 6201944, 8 nets, ONE seed per cell, read out 2026-09-16) was positive:
# under the pre-registered rule (reference mean + 3 reference sd under BOTH silence criteria)
# dropout kept units alive in the `none` arm (263 +/- 14 -> 371 mute / 372 dead, ~8 sd) and in the
# `rws` arm (218 +/- 12 -> 387 mute / 324 dead), and in the `none` arm it cost nothing measurable
# (clean loss 0.02597 dead vs the reference 0.02592). The launcher's own pre-registration said a
# positive result is re-run with three seeds before it is quoted. Pavel (2026-09-16) asked for
# SEVEN per condition instead, which also gives the no-dropout references a matched n.
#
# This array adds:
#   tasks  1-48  SIX more seeds of each of the 8 dropout cells   (4 penalties x 2 kinds x 6)
#   tasks 49-64  FOUR more seeds of each no-dropout control      (4 penalties x 4)
# so that, pooled with what already exists (1 dropout seed per cell; 3 no-dropout seeds per arm),
# every one of the 12 conditions has SEVEN seeds.
#
# Index decode (TID = SLURM_ARRAY_TASK_ID - 1):
#   TID < 48 : P_IDX = TID % 4 ; D_IDX = (TID / 4) % 2 ; rep = TID / 8      (dropout)
#   TID >= 48: J = TID - 48 ; P_IDX = J % 4 ; rep = J / 4 ; dropout OFF     (control)
#   pen  in {none, rws, frm, both}   kind in {mute, dead}
#
# WHERE THE NEW NETS LAND. The dropout nets go into the SAME cells as the first grid
# (NBitFlipFlop_std_dropout/EqType=h_k=3_N=1000_pen=<pen>_do=<kind>/), so seeds simply accumulate.
# The no-dropout controls go into NEW cells in the same sweep, `..._pen=<pen>_do=none/`, rather than
# into the historical reference folders - they are then same-launcher, same-commit controls for the
# dropout nets, and flipflop_dropout_readout.py pools them with the 3 historical seeds it already
# reads from NBitFlipFlop_std_ksweep / _std_pen / _std_penlong.
#
# WHY POOLING THE HISTORICAL SEEDS IS LEGITIMATE. Those 3 seeds per arm were trained before commit
# 1674662 (the dropout fix). That fix only changes the graph when dropout is ON (`output_do` is
# `output_full` otherwise), so no-dropout runs are unaffected. The later `scored_` refactor
# (51237d6) reduces to the identical indexing whenever the task has no per-trial `batch_mask`,
# which TaskNBitFlipFlop does not - VERIFIED, not assumed, by a same-seed 15-iteration run at both
# commits before this array was submitted (losses identical to every printed digit).
#
# PRE-REGISTERED READ-OUT AT 150k (unchanged from the first grid, now with n = 7 both sides):
#   live units under BOTH criteria (scale-free and the flip-flop absolute 4e-2) and the clean loss.
#   "Dropout keeps units alive" := the dropout cell's MEAN exceeds the no-dropout mean at the same
#   penalty by more than 3 no-dropout sd under BOTH criteria. With n = 7 a Welch t-test on the two
#   samples is also reported; the 3-sd rule remains the decision rule, the test is descriptive.
#   frm and both sit at the N ceiling with and without dropout, so for them the read-out is the
#   clean loss only (does dropout cost performance).
#   SECONDARY, and the reason the whole trace is kept: the first grid's figure showed dropout
#   shifts the live-unit curve UP without flattening it - both arms were still silencing at 150k at
#   the same rate. With 7 seeds, fit the late slope per condition and say whether dropout changes
#   the RATE of silencing or only its offset. This is the question the single seed could not answer.
#
# Cost: dropout runs do two forward passes per iteration (~0.4-0.6 s/iter at N=1000; the first grid
# ran 16-24 h), controls do one (~8-10 h). 36 h requested for all. Array throttled to 16 concurrent
# jobs (%16) so 64 tasks do not monopolise the shared GPUs; raise or drop the throttle freely.
#   sbatch --array=1-64%16 slurm/SilentReLU_flipflop_dropout_seeds_spock.slurm
#
# Output: $SAVE_TO/NBitFlipFlop_std_dropout/EqType=h_k=3_N=1000_pen=<pen>_do=<mute|dead|none>/

#SBATCH --job-name=FFdropS
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --array=1-64%16
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
PEN_NAME=(none rws frm both)
PEN_RWS=(0 0.05 0 0.05)
PEN_FRM=(0 0 0.1 0.1)
DO_KIND=(mute dead)
IT=${ITERS_OVERRIDE:-150000}

TID=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$TID" -lt 48 ]; then
  P_IDX=$((TID % 4)); D_IDX=$(((TID / 4) % 2)); REP=$((TID / 8))
  KIND=${DO_KIND[$D_IDX]}
  DO_ARGS=(trainer.dropout=true "trainer.dropout_args.dropout_kind=${KIND}")
else
  J=$((TID - 48)); P_IDX=$((J % 4)); REP=$((J / 4))
  KIND=none
  DO_ARGS=(trainer.dropout=false)
fi
PNAME=${PEN_NAME[$P_IDX]}; LRWS=${PEN_RWS[$P_IDX]}; LFRM=${PEN_FRM[$P_IDX]}

TAG="std_dropout/EqType=h_k=${K}_N=${N}_pen=${PNAME}_do=${KIND}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_std_dropout"
echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K} N=${N} pen=${PNAME} (rws=${LRWS} frm=${LFRM}) dropout=${KIND} rep=${REP} iters=${IT}"
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
  trainer.lambda_rws="$LRWS" trainer.lambda_frm="$LFRM" \
  "${DO_ARGS[@]}" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
