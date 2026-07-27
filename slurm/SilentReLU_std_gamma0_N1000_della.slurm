#!/bin/bash
# STANDARD RNN reference sweep + participation trace (DELLA)
#
# The reference condition for the whole project from 2026-07-27 onwards: a vanilla continuous-time
# ReLU RNN with NO project-specific machinery —
#   dale=false, io_nonnegativity=false  (no sign constraints anywhere)
#   self_connections=true               (W_rec diagonal free and trainable)
#   gamma=0                             (no cubic saturation term)
#   trainable bias in [-1,1], zero init
#   task_safe_gradients=false           (penalties combine with the task loss the ordinary way)
#   monitor=false                       (no per-penalty breakdown; saves a backward pass per term)
# configs/model/rnn_relu_standard.yaml + configs/trainer/trainer_ptrack_plain.yaml.
# The only deliberate non-vanilla choice left is training noise (sigma_rec/inp/out), standard in
# neuroscience RNNs and stated in Methods.
#
# Doubles as rescue pathway #1: a unit that can excite ITSELF is the cheapest possible escape from
# silence, so if self-connections rescue anything it shows up here.
#
# Grid (40 jobs) = 2 eq x 2 lambda_rws x 2 lambda_frm x 5 nets, N=1000:
#   EQ  in {h, s}      RWS in {0, 0.05}      FRM in {0, 0.2}      5 reps, seed="random"
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..39): TID = 20*EQ + 10*RWS + 5*FRM + REP
#
# Output: $SAVE_TO/CDDM_std_g0/EqType=<EQ>_N=1000_LmbdRWS=<RWS>_LmbdFR=<FRM>/<per-net subfolder>
# Descriptor: docs/experiments/standard_rnn_reference.md
# Runs from its own git worktree ($HOME/trainRNNbrain_std) with a PYTHONPATH guard.
#
# --time: >=2h avoids the gputest QOS (3 concurrent jobs). With monitor+projection off these are
# faster than the 07-26 runs (which needed 1h42 without frm, ~2h51 with); 7:30 is ample.
# --constraint=nomig: the gpu partition mixes full A100s with 3g.40gb MIG slices (~half speed).

#SBATCH --job-name=StdRef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=7:30:00
#SBATCH --constraint=nomig
#SBATCH --array=1-40
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
# Logs go to a checkout-independent directory: SBATCH directives are parsed at submit time and
# cannot expand $REPO, and per-worktree log dirs would vanish with the worktree. Job name (%x) and
# array id (%A) keep the sweeps apart.
#SBATCH --output=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.err

module purge
module load cudatoolkit/12.8
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env

set -eo pipefail
export PYTHONUNBUFFERED=1

REPO="${REPO:-$HOME/trainRNNbrain_std}"
SAVE_TO="${SAVE_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.dirname(os.path.dirname(trainRNNbrain.__file__)))")
if [ "$PKG" != "$REPO" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO — aborting." >&2; exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short=7 HEAD)"

EQS=(h s); RWSS=(0 0.05); FRMS=(0 0.2)
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 5)); FRM_IDX=$(((TID / 5) % 2)); RWS_IDX=$(((TID / 10) % 2)); EQ_IDX=$(((TID / 20) % 2))
EQ=${EQS[$EQ_IDX]}; RWS=${RWSS[$RWS_IDX]}; FRM=${FRMS[$FRM_IDX]}

TAG="std_g0/EqType=${EQ}_N=1000_LmbdRWS=${RWS}_LmbdFR=${FRM}"
mkdir -p "${SAVE_TO}/CDDM_std_g0"
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  cp "$REPO/docs/experiments/standard_rnn_reference.md" "${SAVE_TO}/CDDM_std_g0/EXPERIMENT.md"
  echo -e "\n---\nRun: SLURM array ${SLURM_ARRAY_JOB_ID} on Della, code commit $(git -C "$REPO" rev-parse HEAD)" \
    >> "${SAVE_TO}/CDDM_std_g0/EXPERIMENT.md"
fi
echo "Task ${SLURM_ARRAY_TASK_ID}: EQ=${EQ} N=1000 Lrws=${RWS} Lfrm=${FRM} rep=${REP}"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_std \
  model=rnn_relu_standard trainer=trainer_ptrack_plain \
  model.equation_type="$EQ" \
  trainer.lambda_rws="$RWS" trainer.lambda_frm="$FRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
