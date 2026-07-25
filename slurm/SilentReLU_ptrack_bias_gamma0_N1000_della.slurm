#!/bin/bash
# Trainable-bias control + participation trace (DELLA)
#
# Does the silent population survive when every unit can learn its own offset? All previous
# silent-unit results used bias_range=[0,0] (no bias at all), so a ReLU unit with always-negative
# input had nothing to lift it. This is the same grid as CDDM_ptrack_g0 (the bias-free sibling
# sweep) with model=rnn_relu_Dale_trainablebias: bias_range=[-1,1], bias_init=zeros (so the initial
# weights are bit-identical to the bias-free baseline and the ONLY change is the learnable offset).
#
# Participation is logged every 10 iterations as *_ParticipationTrace.pkl + participation_trace.png.
# CAUTION on the readout: a constant bias fakes participation (b=0.3 -> constant rate 0.3 -> healthy
# peak and q0.9, but std ~ 0 and no information). Score rescue on std(fr), not peak. Full rationale,
# predictions and pre-registered criteria: docs/experiments/participation_trace_bias.md (copied into
# the sweep folder as EXPERIMENT.md by array task 1).
#
# Grid (40 jobs) = 2 eq x 2 lambda_rws x 2 lambda_frm x 5 nets, N=1000, gamma=0:
#   EQ  in {h, s}            (model.equation_type)
#   RWS in {0, 0.05}         (trainer.lambda_rws)
#   FRM in {0, 0.2}          (trainer.lambda_frm)
#   5 reps, seed="random"
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..39): TID = 20*EQ + 10*RWS + 5*FRM + REP
#   REP=TID%5; FRM_IDX=(TID/5)%2; RWS_IDX=(TID/10)%2; EQ_IDX=(TID/20)%2
#
# Same regime as CDDM_4a031e_g0 / CDDM_ptrack_g0 (gamma=0, N=1000, 30000 iters, dt=1) EXCEPT for the
# trainable bias — that is the single manipulated variable, and CDDM_ptrack_g0 is its control.
#
# Output: $SAVE_TO/CDDM_ptrack_g0_trainablebias/EqType=<EQ>_N=1000_LmbdRWS=<RWS>_LmbdFR=<FRM>/<per-net subfolder>
# Overridable env vars: MAXITER (default: config, 30000), SAVE_TO (default: TENGEL scratch).
#
# NOTE --time: >= 2 h is required on Della, otherwise the job lands on the `gputest` QOS with a
# hard cap of 3 concurrent jobs. Measured 0.154 s/iter at N=1000 on an L40S -> ~80 min for 30000.

#SBATCH --job-name=PtrackBias
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=2:30:00
#SBATCH --array=1-40
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/home/pt1290/trainRNNbrain/log/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain/log/%x.%A_%a.err

module purge
module load cudatoolkit/12.8
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env

set -eo pipefail
export PYTHONUNBUFFERED=1   # otherwise iteration lines sit in an 8 KB buffer and the log looks stalled

REPO="$HOME/trainRNNbrain"
SAVE_TO="${SAVE_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs}"

EQS=(h s)
RWSS=(0 0.05)
FRMS=(0 0.2)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 5))
FRM_IDX=$(((TID / 5)  % 2))
RWS_IDX=$(((TID / 10) % 2))
EQ_IDX=$(((TID / 20) % 2))
EQ=${EQS[$EQ_IDX]}
RWS=${RWSS[$RWS_IDX]}
FRM=${FRMS[$FRM_IDX]}

TAG="ptrack_g0_trainablebias/EqType=${EQ}_N=1000_LmbdRWS=${RWS}_LmbdFR=${FRM}"
mkdir -p "${SAVE_TO}/CDDM_ptrack_g0_trainablebias"
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  # descriptor of what this data is, next to the data itself, stamped with the code commit
  cp "$REPO/docs/experiments/participation_trace_bias.md" "${SAVE_TO}/CDDM_ptrack_g0_trainablebias/EXPERIMENT.md"
  echo -e "\n---\nRun: SLURM array ${SLURM_ARRAY_JOB_ID} on Della, code commit $(git -C "$REPO" rev-parse HEAD)" \
    >> "${SAVE_TO}/CDDM_ptrack_g0_trainablebias/EXPERIMENT.md"
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: EQ=${EQ} N=1000 Lrws=${RWS} Lfrm=${FRM} rep=${REP} hash=$(git -C "$REPO" rev-parse --short=6 HEAD)"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

# Fixed params (gamma=0, N=1000, max_iter, trainer_tag, track_every) live in config; this only
# selects the tracking trainer and sets the swept IVs.
srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_N1000 \
  trainer=trainer_ptrack \
  model=rnn_relu_Dale_trainablebias model.equation_type="$EQ" \
  trainer.lambda_rws="$RWS" trainer.lambda_frm="$FRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
