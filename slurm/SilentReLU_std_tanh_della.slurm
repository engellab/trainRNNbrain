#!/bin/bash
# DOES A TANH NETWORK EMPTY OUT TOO, AND DOES IT SATURATE INSTEAD? (DELLA)
#
# WHY. Every activation tested on the standard network so far is non-negative: ReLU, leaky ReLU,
# softplus, shifted sigmoid. Tanh is the first signed one, and it has a failure mode the others do
# not: a unit can be pinned at +/-1, where the gradient is as dead as a silent ReLU's but the unit
# is loud rather than quiet. The project's criterion counts a pinned unit as ACTIVE — participation
# is std(r) + q90(|r|), and a unit at -1 scores ~1 — so the two have to be read separately.
#
# GRID (6 jobs) = 2 activations x 3 seeds, CDDM, N = 1000, h equation, standard RNN, no penalties,
# default weight decay 1e-6, 150000 iterations:
#   act=tanh   model=rnn_tanh_standard   (new config: rnn_relu_standard with tanh, nothing else)
#   act=relu   model=rnn_relu_standard   — the in-sweep control at the same code, budget and seeds
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..5): REP = TID%3; ACT = TID/3
#
# READ-OUT, fixed before submission, both at 150000 iterations:
#   (1) active units under the scale-free rule from the participation trace — the SILENT measure;
#   (2) the saturated fraction, computed offline from the saved parameters: units whose |r| exceeds
#       0.99 on at least 95% of (time, trial) samples of a noise-free probe. There is no run on
#       disk to compare (2) against, so the ReLU control supplies (1) and (2) is descriptive.
# PREDICTION: tanh ends with fewer silent units than ReLU — a tanh unit has gradient on both sides
# of zero, so the absorbing state that traps a sub-threshold ReLU unit does not exist — and the
# units that stop being useful end up pinned instead, giving a saturated fraction well above 0.
# FALSIFIER: a tanh active count inside the ReLU control's 3-seed range with a saturated fraction
# near zero means the signed activation changes neither failure mode, and the emptying is a
# property of the task-and-training setup rather than of the rectifier.
# GATE: mean r2 < 0.7 in a cell -> counts reported, not interpreted.
#
# Runtime 0.144 s/iter at N=1000 (measured, drift sweep) -> ~6 h per job; 14 h requested.
# Output: $SAVE_TO/CDDM_std_g0_tanh/EqType=h_N=1000_act=<name>_iters=150000/<per-net subfolder>

#SBATCH --job-name=StdTanh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=14:00:00
#SBATCH --constraint=nomig
#SBATCH --array=1-6
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

REPO="${REPO:-$HOME/trainRNNbrain_recinit}"
SAVE_TO="${SAVE_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2; exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short=7 HEAD 2>/dev/null || true)
if [ -z "$GITSHA" ]; then
  echo "FATAL: $REPO is not a git checkout - provenance of this run cannot be recorded." >&2; exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

ACTS=(tanh relu); MODELS=(rnn_tanh_standard rnn_relu_standard)
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); ACT_IDX=$((TID / 3))
ACT=${ACTS[$ACT_IDX]}; MODEL=${MODELS[$ACT_IDX]}

TAG="std_g0_tanh/EqType=h_N=1000_act=${ACT}_iters=150000"
mkdir -p "${SAVE_TO}/CDDM_std_g0_tanh"
echo "Task ${SLURM_ARRAY_TASK_ID}: act=${ACT} model=${MODEL} rep=${REP} iters=150000"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  model="$MODEL" trainer=trainer_ptrack_plain \
  model.equation_type=h model.N=1000 \
  trainer.max_iter=150000 \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
