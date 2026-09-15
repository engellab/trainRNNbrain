#!/bin/bash
# DROPOUT x PENALTY on the 3-bit flip-flop, N=1000: does dropout keep units alive? (SPOCK)
#
# Pavel's question (2026-09-15): silent units are the network's way of NOT using capacity; a
# dropout that removes units during training forces the rest to cover for them, which could keep
# more of them alive without a rate penalty - or add to it. Test cell: the 3-bit flip-flop at
# N=1000, where the unpenalised reference (NBitFlipFlop_std_ksweep k=3) and the rws/frm/both
# reference (NBitFlipFlop_std_pen k=3) already exist at 150k with three seeds each.
#
# ⚠️ DROPOUT WAS A NO-OP UNTIL 2026-09-15 (Trainer.train_step scored the task loss on the
# non-dropout forward pass; fixed in the commit this launcher ships with, verified by same-seed
# runs: 'dead' and 'mute' were bit-identical before the fix and differ from iteration 1 after).
# Every dropout run before that commit (slurm/Dropout.slurm, 2025) is void. This is the first
# real one.
#
# The two dropout KINDS in RNN_torch.forward / rhs:
#   mute  the unit runs in the recurrent dynamics but its read-out weight is masked to 0
#   dead  the unit is ablated: its rate is zeroed for every other unit, its drive and noise are
#         zeroed, and its read-out weight is masked (x decays to 0 for the trial batch)
# Mask: one Bernoulli draw per unit per training batch (shared across the trials of the batch).
# Sampling method and rate come from the trainer config (trainer_ptrack_freshbatch.dropout_args:
# sampling_method=participation, drop_rate=0.05, dropout_beta=1.0, eta=0.5, activity_q=0.9):
# a unit's drop probability is 0.05 * N * softmax(beta * participation_EMA), i.e. ~0.05 with the
# most active units dropped somewhat more often. Only the kind is swept here.
#
# Grid (8 jobs) = 4 penalty arms x 2 dropout kinds, one seed each, 150k iterations.
#   pen  in {none, rws, frm, both}     P_IDX = TID % 4
#   kind in {mute, dead}               D_IDX = TID / 4
#   tasks 1-4 mute, 5-8 dead.
# Pre-registered read-out at 150k, against the no-dropout references at the same iteration:
#   live units (scale-free AND the flip-flop absolute 4e-2 criterion, both reported) and the
#   clean loss. "Dropout keeps units alive" := live(none + dropout) exceeds the reference `none`
#   mean by more than 3 reference sd under BOTH criteria; "adds to the penalty" := the same for
#   rws (frm and both are already at N). One seed each: a positive result is re-run with 3 seeds.
# Cost: two forward passes per iteration with dropout on (~0.4 s/iter at N=1000), 150k -> ~17 h;
# 30 h requested.
#   sbatch --array=1-8 slurm/SilentReLU_flipflop_dropout_spock.slurm
#
# Output: $SAVE_TO/NBitFlipFlop_std_dropout/EqType=h_k=3_N=1000_pen=<pen>_do=<kind>/<per-net subfolder>

#SBATCH --job-name=FFdrop
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --array=1-8
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
IT=150000

TID=$((SLURM_ARRAY_TASK_ID - 1))
P_IDX=$((TID % 4)); D_IDX=$((TID / 4))
PNAME=${PEN_NAME[$P_IDX]}; LRWS=${PEN_RWS[$P_IDX]}; LFRM=${PEN_FRM[$P_IDX]}
KIND=${DO_KIND[$D_IDX]}
IT=${ITERS_OVERRIDE:-$IT}

TAG="std_dropout/EqType=h_k=${K}_N=${N}_pen=${PNAME}_do=${KIND}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_std_dropout"
echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K} N=${N} pen=${PNAME} (rws=${LRWS} frm=${LFRM}) dropout=${KIND} iters=${IT}"
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
  trainer.dropout=true trainer.dropout_args.dropout_kind="$KIND" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
