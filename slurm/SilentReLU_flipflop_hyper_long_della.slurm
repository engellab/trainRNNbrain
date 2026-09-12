#!/bin/bash
# HYPER FLIP-FLOP AT THE PLAIN K-SWEEP'S BUDGET (500k) — the depth control, DELLA, H100.
#
# The 150k hyper grid on Spock (SilentReLU_flipflop_hyper_spock.slurm) read recruitment at a MATCHED
# ITERATION while the hyper cells were still short of their floor (r^2 0.94 / 0.89 / 0.80 at
# k = 2 / 4 / 6 vs 0.95 on the plain task). Silence in this project keeps growing with training
# depth, so part of the excess recruitment (+21 / +73 / +106% at N=500) could be convergence depth,
# not demand. This array trains the unpenalised hyper cells to 500k, the plain ksweep's budget, so
# the two tasks can be compared at the plain sweep's own endpoint and along the whole trace.
# It also answers whether k=8 is a capacity ceiling or merely slow (r^2 ~0.5 at 45k on Spock).
#
# Grid (36 jobs) = 4 complexities x 3 sizes x 3 seeds, unpenalised only (the penalty's cost is a
# matched-iteration question and penalised runs converge faster; not repeated here).
#   k in {2,4,6,8}   task.n_inputs=k, task.n_outputs=2^k-1
#   N in {500,1000,2000}
# Index decode (TID = SLURM_ARRAY_TASK_ID-1): REP=TID%3; N_IDX=(TID/3)%3; K_IDX=TID/9
#   k=8 cells are tasks 28-36.
#
# WALL CLOCK. Spock rates (L40S / A100-40G) were 0.35-0.82 s/iter for k<=6 and 0.7-1.7 s/iter at
# k=8, i.e. up to 114 h at k<=6 and up to 236 h at k=8 N=2000 for 500k. 140 h covers every k<=6
# cell even at Spock speed; k=8 N=2000 is NOT safe without a measured H100 rate, so:
#   sbatch --array=1-27 <this>                                        # k<=6 now
#   ITERS_OVERRIDE=600 sbatch --array=36 --time=00:40:00 <this>        # calibrate k=8 N=2000
#   then size --time for --array=28-36 from the measured rate (gpu-long caps at 144 h).
# Always name the gres TYPE (h100): --gpus-per-node=1 can land on a 3g.40gb MIG slice, 2.2x slower.
#
# Results land in DELLA's data/trained_RNNs and must be synced to the local mirror before analysis.
# Output: $SAVE_TO/NBitFlipFlopHyper_std_hyper_long/EqType=h_k=<k>_N=<N>_pen=none_iters=<IT>/

#SBATCH --job-name=FFhypD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-gpu=64G
#SBATCH --qos=gpu-long
#SBATCH --time=140:00:00
#SBATCH --array=1-36
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
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "WARNING: $REPO has uncommitted changes; the commit above does not fully describe this run." >&2
fi

KS=(2 4 6 8)
NS=(500 1000 2000)
PNAME=none; LRWS=0; LFRM=0
IT=500000

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); N_IDX=$(((TID / 3) % 3)); K_IDX=$((TID / 9))
K=${KS[$K_IDX]}; N=${NS[$N_IDX]}; NOUT=$((2 ** K - 1))
IT=${ITERS_OVERRIDE:-$IT}

TAG="std_hyper_long/EqType=h_k=${K}_N=${N}_pen=${PNAME}_iters=${IT}"
mkdir -p "${SAVE_TO}/NBitFlipFlopHyper_std_hyper_long"
echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K} (${NOUT} outputs) N=${N} pen=${PNAME} (rws=${LRWS} frm=${LFRM}) rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/NBitFlipFlopHyper_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=NBitFlipFlopHyper \
  task.n_inputs="$K" task.n_outputs="$NOUT" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws="$LRWS" trainer.lambda_frm="$LFRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
