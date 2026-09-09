#!/bin/bash
# frm / both on the n-bit flip-flop at N=4000 — DELLA.
#
# Extends the penlong grid (N = 500/1000/2000) to the size that most constrains the exponents. The
# unpenalised bigN array already covers none and rws at N=4000; this is the deferred penalty half of
# that measurement, which SilentReLU_flipflop_bigN_spock.slurm explicitly postponed until a
# converged frm budget was known.
#
# ⚠️ WALL CLOCK IS THE BINDING CONSTRAINT, AND IT IS MEASURED, NOT MODELLED. The 48-job bigN array
# ran 100k iterations at N=4000 and split cleanly by GPU:
#     L40S-46G     33.9 h / 100k  [32.2, 35.5]   n=26     -> 400k = 136 h
#     A100-40G     62.7 h / 100k  [60.4, 65.7]   n=22     -> 400k = 251 h
# So 400k does NOT fit Spock's 96 h cap on either GPU, and fits Della's 144 h gpu-long cap only if
# the node is materially faster than an A100-40G. Della has h100 and h200, which the sibling script
# never used because it pins a100 to dodge MIG slices. CALIBRATE BEFORE SUBMITTING THE ARRAY:
#   ITERS_OVERRIDE=1200 sbatch --array=22 --time=02:30:00 --gres=gpu:h100:1 <this>
#   ITERS_OVERRIDE=1200 sbatch --array=22 --time=02:30:00 --gres=gpu:h200:1 <this>
# then read the per-iteration rate out of the log and size --time and IT from it. The cost model in
# the bigN header mispredicted the measured N=500 point by 25%; do not trust it here either.
#
# ⚠️ CALIBRATE AT k=8, NOT k=1. Array task 22 is frm/k=8 and task 46 is both/k=8 - the most
# expensive cells. Task 1 is k=1 and will flatter the estimate.
#
# ⚠️ MIG SLICES. --gpus-per-node=1 lets SLURM satisfy the request with a partitioned A100 advertised
# as `3g.40gb`, measured 2.2x slower on the frm N=2000 array. Always name the gres TYPE.
#
# ⚠️ NO --array DEFAULT. Pass it explicitly, and cancel the matching Spock tasks once these are
# running, or the same cell is computed twice.
#
# ⚠️ RESULTS LAND IN DELLA'S OWN data/trained_RNNs and must be synced to the local mirror before any
# analysis — every analysis script reads the local copy, not a cluster.
#
# Grid = 2 penalties x 8 complexities x 3 seeds = 48 tasks, at N=4000.
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..47): REP=TID%3; K_IDX=(TID/3)%8; P_IDX=TID/24
#   frm  -> --array=1-24        both -> --array=25-48
#
# Output: $SAVE_TO/NBitFlipFlop_std_bigN/EqType=h_k=<k>_N=4000_pen=<name>_iters=<IT>/

#SBATCH --job-name=FFpenBigND
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
# Named gres TYPE, never --gpus-per-node, so a MIG slice (`3g.40gb`) can never satisfy this.
# h100 by default: the bigN timings show this workload is bandwidth-bound (L40S beats A100-40G by
# 1.85x), so the A100 pin used by the N<=2000 sibling script is the wrong default at N=4000.
# Override on the command line to compare: --gres=gpu:h200:1 / --gres=gpu:a100:1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=140:00:00
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
