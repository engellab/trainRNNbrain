#!/bin/bash
# frm / both on the n-bit flip-flop at N=4000 — SPOCK.
#
# Byte-identical index decode and science to SilentReLU_flipflop_frmboth_bigN_della.slurm, so a task
# number means the same cell on both clusters and work can be split by --array without either
# cluster disagreeing about what it is computing. Only scheduler directives, modules and log paths
# differ.
#
# ⚠️ 400k DOES NOT FIT SPOCK. Measured on the 48-job bigN array at N=4000, 100k iterations:
#     L40S-46G     33.9 h / 100k  [32.2, 35.5]   n=26     -> 400k = 136 h
#     A100-40G     62.7 h / 100k  [60.4, 65.7]   n=22     -> 400k = 251 h
# Spock's hard cap is 96 h and the node type cannot be chosen, so a 400k job must be sized for the
# A100 case and cannot exceed ~150k. Submit the long budget on Della; use this script only for a
# reduced IT, passed explicitly:
#   ITERS_OVERRIDE=150000 sbatch --array=1-24 --time=96:00:00 <this>
#
# ⚠️ CALIBRATE AT k=8, NOT k=1. Array task 22 is frm/k=8 and task 46 is both/k=8 - the most
# expensive cells. Task 1 is k=1 and will flatter the estimate.
#
# ⚠️ NO --array DEFAULT. Pass it explicitly, and cancel the matching Della tasks once these are
# running, or the same cell is computed twice.
#
# Grid = 2 penalties x 8 complexities x 3 seeds = 48 tasks, at N=4000.
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..47): REP=TID%3; K_IDX=(TID/3)%8; P_IDX=TID/24
#
# Output: $SAVE_TO/NBitFlipFlop_std_bigN/EqType=h_k=<k>_N=4000_pen=<name>_iters=<IT>/

#SBATCH --job-name=FFpenBigNS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
# NO --array DEFAULT ON PURPOSE: this grid has 48 tasks, and the 1-144 inherited
# from the N<=2000 script would launch 96 tasks whose index decodes out of range.
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
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL — aborting." >&2; exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "WARNING: $REPO has uncommitted changes; the commit above does not fully describe this run." >&2
fi
# The .npz saver lost dict-valued parameters (activation_args stored as its KEYS) until the
# storable_ fix. Refuse to run on code that predates it, or these nets cannot be reconstructed.
python - <<'PYCHK' || exit 1
import sys
from trainRNNbrain.training import run_experiment as m
sys.exit(0 if hasattr(m, "storable_") else
         print("FATAL: run_experiment lacks storable_(); re-sync the repo before submitting.") or 1)
PYCHK

KS=(1 2 3 4 5 6 7 8)
N=${BIGN:-4000}
PEN_NAME=(frm both)
PEN_RWS=(0 0.05)
PEN_FRM=(0.1 0.1)
IT=400000

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); K_IDX=$(((TID / 3) % 8)); P_IDX=$((TID / 24))
K=${KS[$K_IDX]}
PNAME=${PEN_NAME[$P_IDX]}; LRWS=${PEN_RWS[$P_IDX]}; LFRM=${PEN_FRM[$P_IDX]}
IT=${ITERS_OVERRIDE:-$IT}

TAG="std_bigN/EqType=h_k=${K}_N=${N}_pen=${PNAME}_iters=${IT}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_std_bigN"
echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K} N=${N} pen=${PNAME} (rws=${LRWS} frm=${LFRM}) rep=${REP} iters=${IT}"

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
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
