#!/bin/bash
# FIFTEEN TASKS IN ONE NETWORK: does CDDM recruit more units when the network also serves 14 others? (SPOCK)
#
# The capacity law says a single-task network recruits ~10 units per latent function, so plain CDDM
# leaves most of a large network silent - and that is what makes it a poor stand-in for a recorded
# population, which is a redundant sample of a circuit that is never doing only one task. The
# honest model of that is multi-task training (Yang et al. 2019): ONE network, a one-hot rule input,
# fifteen of the repo's tasks (configs/task/MultiRule15.yaml, tasks/TaskMultiRule.py), CDDM among
# them with exactly its single-task timing, channels and scoring.
#
# Pre-registered read-out (multirule_readout.py, fixed before any run finishes):
#   1. live_CDDM: units active on CDDM-ONLY trials (rule on, noise-free, every criterion the
#      project reports) in the multi-task net vs the single-task CDDM references at the same N
#      (CDDM_std_g0_drift for `none`; CDDM_std_g0_penalties for `both`), read at the same
#      iteration. "Multi-task raises recruitment on CDDM" := live_CDDM exceeds the single-task mean
#      by > 3 sd in every seed, under the scale-free AND the absolute criterion.
#   2. live_all: units active over the mixed batch (the total the network uses), and per task.
#   3. Which of the CDDM-active units are ALSO active on other tasks (shared vs CDDM-private):
#      the split says whether the extra recruitment is CDDM's own computation becoming more
#      redundant or other tasks' units responding to shared inputs. Either is a result; they
#      mean different things for subsampling and must not be conflated.
#   4. Per-task r2 on the task's own scoring window, so an unlearned subtask is visible.
#
# Grid (18 jobs) = 3 sizes x 2 penalty arms x 3 seeds, h equation, standard ReLU, fresh batches:
#   N   in {500, 1000, 2000}
#   pen in {none, both}   both = rws 0.05 + frm 0.1
# Index decode (TID = SLURM_ARRAY_TASK_ID-1): REP=TID%3; P_IDX=(TID/3)%2; N_IDX=TID/6
#   tasks 1-6 N=500, 7-12 N=1000, 13-18 N=2000 (within a size: 3 none, then 3 both).
# 150k iterations, as every other grid; multi-task training may need more - the loss trace says.
# Cost: batch 1020 x T=300 x 27 inputs is the flip-flop's per-iteration cost (0.24-0.26 s/iter at
# N<=1000, 0.55 at N=2000) plus batch generation, ~0.075 s (subtask batches capped at the 68
# trials drawn; AngleIntegration's per-trial Python loop is the remaining cost):
#   sbatch --array=1-12  slurm/SilentReLU_multirule_spock.slurm                    # 24 h
#   sbatch --array=13-18 --time=48:00:00 slurm/SilentReLU_multirule_spock.slurm    # N=2000
# Timing calibration: ITERS_OVERRIDE=600 sbatch --array=13 ...
#
# Output: $SAVE_TO/MultiRule_std_multi/EqType=h_N=<N>_pen=<name>/<per-net subfolder>

#SBATCH --job-name=Multi15
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=1-18
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

NS=(500 1000 2000)
PEN_NAME=(none both)
PEN_RWS=(0 0.05)
PEN_FRM=(0 0.1)
IT=150000

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); P_IDX=$(((TID / 3) % 2)); N_IDX=$((TID / 6))
N=${NS[$N_IDX]}
PNAME=${PEN_NAME[$P_IDX]}; LRWS=${PEN_RWS[$P_IDX]}; LFRM=${PEN_FRM[$P_IDX]}
IT=${ITERS_OVERRIDE:-$IT}

TAG="std_multi/EqType=h_N=${N}_pen=${PNAME}"
mkdir -p "${SAVE_TO}/MultiRule_std_multi"
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} pen=${PNAME} (rws=${LRWS} frm=${LFRM}) rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/MultiRule_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=MultiRule15 \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws="$LRWS" trainer.lambda_frm="$LFRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
