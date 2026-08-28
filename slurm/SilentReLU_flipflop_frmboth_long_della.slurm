#!/bin/bash
# frm / both on the n-bit flip-flop at a converged budget — DELLA.
#
# Identical science to SilentReLU_flipflop_frmboth_long_spock.slurm, and a BYTE-IDENTICAL index
# decode, so task N means the same cell on both clusters and work can be split between them by
# --array without any risk of the two disagreeing about what a task is. Only the scheduler
# directives, module loads and log path differ.
#
# ⚠️ DELLA ASSIGNS QOS BY WALL TIME, AND LONGER REQUESTS REDUCE CONCURRENCY:
#     < 2 h    gputest      3 concurrent   <- never land here by accident
#     2-24 h   gpu-short   44 concurrent
#     24-72 h  gpu-medium  24 concurrent   <- 48 h lands here
#     72-144 h gpu-long    10 concurrent
# The 48 h request is deliberate: 400k x 0.26 s/iter = 29 h at N=1000, so it carries a 1.65x margin
# AND keeps a 24-task array inside a single gpu-medium wave. Asking for 72 h+ would more than halve
# the concurrency for no benefit. Raise --time only for N=2000 (61 h), and accept 10 concurrent then.
#
# ⚠️ NO --array DEFAULT. Always pass it explicitly, and cancel the matching Spock tasks once these
# are confirmed running, or the same cell is computed twice.
#
# ⚠️ RESULTS LAND IN DELLA'S OWN data/trained_RNNs and must be synced to the local mirror before any
# analysis — every analysis script reads the local copy, not a cluster.
#
# Grid = 2 penalties x 8 complexities x 3 sizes x 3 seeds = 144 tasks.
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..143):
#   REP=TID%3;  N_IDX=(TID/3)%3;  K_IDX=(TID/9)%8;  P_IDX=TID/72   (0=frm, 1=both)
#   both @ N=1000  ->  --array=76-78,85-87,94-96,103-105,112-114,121-123,130-132,139-141
#
# Output: $SAVE_TO/NBitFlipFlop_std_penlong/EqType=h_k=<k>_N=<N>_pen=<name>_iters=<IT>/

#SBATCH --job-name=FFpenD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=48:00:00
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
NS=(500 1000 2000)
PEN_NAME=(frm both)
PEN_RWS=(0 0.05)
PEN_FRM=(0.1 0.1)
IT=400000

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); N_IDX=$(((TID / 3) % 3)); K_IDX=$(((TID / 9) % 8)); P_IDX=$((TID / 72))
K=${KS[$K_IDX]}; N=${NS[$N_IDX]}
PNAME=${PEN_NAME[$P_IDX]}; LRWS=${PEN_RWS[$P_IDX]}; LFRM=${PEN_FRM[$P_IDX]}
IT=${ITERS_OVERRIDE:-$IT}

TAG="std_penlong/EqType=h_k=${K}_N=${N}_pen=${PNAME}_iters=${IT}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_std_penlong"
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
