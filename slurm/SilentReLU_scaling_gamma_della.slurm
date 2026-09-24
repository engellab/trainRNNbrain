#!/bin/bash
# DOES CUBIC SATURATION RESCUE SYNAPTIC SCALING, AND WHICH RULE IS BEST? (DELLA)
#
# WHY THIS IS WORTH ONE MORE ATTEMPT. Synaptic scaling has failed four times, every one at gamma=0:
#   median set-point, eta 0.05          19-27% of gradient updates discarded, all 3 seeds
#   rank-matched lognormal target       31-55% discarded, final r2 NaN or -inf on all 3 seeds
#   row-norm preserved, eta 0.5         99.75% discarded in the first 400 iterations
#   row-norm preserved, eta 0.05        49.9% discarded, final r2 -2.5e19
# Each fix was validated on a static test with no training in the loop and each failed on contact
# with a real run, so the repairs were never the problem to begin with.
#
# What changed is the duplication result. Prune-and-copy had the SAME signature -- 29-44% of
# updates discarded at N=500, 49-52% at N=1000, clean losses of 273,489 and 696,300 -- and adding
# -gamma*x^3 to the dynamics fixed it completely: 0.0% of updates discarded at N=500, 1000 and
# 2000, with the loss BEATING its own control. If both failures came from unbounded ReLU dynamics
# rather than from anything specific to either rule, scaling should recover the same way. That is
# the hypothesis, and if it is wrong the discarded-update rate will say so within the first
# few hundred iterations, as it has four times already.
#
# GRID (24 jobs) = 8 cells x 3 seeds, all at gamma=0.1, N=1000, 3-bit flip-flop, 40k iterations.
#   target in {median, lognormal}     x  preserve_row_norm in {true, false}  x  eta in {0.05, 0.2}
#   `median` pulls every unit toward one set-point and homogenises: it is what took frm's rate
#   spread down to 6x where cortex has ~100x. `lognormal` pulls the unit of rank k toward the
#   value a lognormal population would give it, so it cannot homogenise, but it demands the busiest
#   unit reach ~51x the median, which is the runaway direction. Both are worth a bounded retry.
#   preserve_row_norm is tested rather than assumed: the sign split multiplies a row's magnitude by
#   about (alpha + 1/alpha) > 2, which is why it was added, but gamma may already bound the
#   consequence and the renormalisation weakens the per-event effect.
#   Index decode (TID = SLURM_ARRAY_TASK_ID - 1, 0..23): REP = TID%3; CELL = TID/3
#   sbatch --array=1-24%12 slurm/SilentReLU_scaling_gamma_della.slurm
#
# CONTROL comes from NBitFlipFlop_ff_revive_g01 at N=1000, gamma=0.1, arm=none, already landed at
# 323.3 +- 30.4 active units with clean loss 0.09208 and 0.0% of updates discarded. Same task,
# size, iteration count, model config and gamma.
#
# PRE-REGISTERED, WRITTEN BEFORE ANY OF THESE JOBS EXISTED. Read-out at 40,000 iterations.
#   GATE FIRST, AND IT IS A GATE NOT A CAVEAT: any cell discarding more than 5% of its gradient
#             updates is reported as BROKEN and its unit count is not quoted at all. Four scaling
#             arms have already produced unit counts while throwing away a fifth to a half of
#             training, and two of them would have looked like wins on the count alone.
#   PRIMARY   among the cells that pass the gate, active units against the control's 323.3 +- 30.4.
#   CO-PRIMARY clean dropout-off loss (loss_clean_train) against the control's 0.09208. The
#             duplication arm at this gamma beat its control on loss, so "recruits units AND
#             trains better" is the bar this can be held to, not merely "recruits units".
#   SHAPE     the ratio of the busiest to the quietest firing unit. This is the one measure on
#             which scaling is supposed to beat duplication: `median` should compress it and
#             `lognormal` should hold it near the cortical ~100x. A cell that recruits units while
#             leaving the range at the control's value has not done what scaling is for.
#   EXPECTED, stated so it can be wrong: if gamma is the whole story, all eight cells pass the
#             gate and the differences between rules are small. If any cell still blows up, the
#             rule matters independently of boundedness and `lognormal` at eta 0.2 is the most
#             likely to fail, since it pushes the largest units hardest.
#
# Cost: no extra forward pass; events fire every 100 iterations. ~2.9 h at N=1000. 8 h requested.
# Output: $SAVE_TO/NBitFlipFlop_scaling_gamma/EqType=h_k=3_N=1000_tgt=<t>_norm=<b>_eta=<e>/

#SBATCH --job-name=ScaleG
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=08:00:00
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
IT=${ITERS_OVERRIDE:-40000}
GAMMA=${GAMMA:-0.1}

CELL_TGT=(median median median median lognormal lognormal lognormal lognormal)
CELL_NRM=(true   true   false  false  true      true      false     false)
CELL_ETA=(0.05   0.2    0.05   0.2    0.05      0.2       0.05      0.2)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); CELL=$((TID / 3))
TGT=${CELL_TGT[$CELL]}
NRM=${CELL_NRM[$CELL]}
ETA=${CELL_ETA[$CELL]}

TAG="${SWEEP:-scaling_gamma}/EqType=h_k=${K}_N=${N}_tgt=${TGT}_norm=${NRM}_eta=${ETA}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_${SWEEP:-scaling_gamma}"
echo "Task ${SLURM_ARRAY_TASK_ID}: cell=${CELL} target=${TGT} preserve_row_norm=${NRM} eta=${ETA} gamma=${GAMMA} N=${N} rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/NBitFlipFlop_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=NBitFlipFlop \
  task.n_inputs="$K" task.n_outputs="$K" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" model.gamma="$GAMMA" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  trainer.prune_reinit=false trainer.synaptic_scaling=true \
  "trainer.scaling_args.scale_target=${TGT}" \
  "trainer.scaling_args.preserve_row_norm=${NRM}" \
  "trainer.scaling_args.eta=${ETA}" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
