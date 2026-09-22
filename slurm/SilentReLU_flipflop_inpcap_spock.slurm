#!/bin/bash
# DOES CAPPING THE INPUT WEIGHTS RECRUIT UNITS? (SPOCK)
#
# THE HYPOTHESIS (Pavel, 2026-09-22): input drive in the unpenalised network is concentrated on a
# few units, and that concentration is what leaves the rest silent. Capping |W_inp| should force
# the drive to spread and recruit units.
#
# WHAT MOTIVATES IT, measured on trained flip-flop networks (N=1000, no dropout, 4 seeds per arm):
#   arm      input row norm CV   active units   corr(row norm, log participation)
#   none            2.28              262.5              +0.65
#   rws             2.54              221.0              +0.51
#   frm             1.08              965.8              +0.07
#   frm+rws         0.74             1000.0              -0.61
# Input drive becomes DRAMATICALLY more equal exactly as the active count rises, and the coupling
# between drive and activity inverts: unpenalised, a unit's input weights essentially decide whether
# it fires; under frm+rws they do not. At the weight level the unpenalised network puts its drive
# into ~6% of entries - 99th percentile |W_inp| = 8.57 against a median of 0.002, a 4000x ratio.
#
# THIS IS CORRELATIONAL AND THAT IS THE POINT OF THE EXPERIMENT. frm both equalises input weights
# AND recruits units; the direction is unknown. Recruiting a unit may FORCE its input weights up
# (a unit that fires needs drive), in which case equalisation is a consequence and capping will do
# nothing. Imposing the cap directly is what separates the two.
#
# WHY A CLAMP, NOT THE EXISTING PENALTY. inp_weights_magnitude_penalty is wired in as lambda_iwm but
# its gamma=5 hinge is unusable here: at init every |W_inp| ~ 0.03 sits under any sensible cap so
# the penalty is exactly 0 with no gradient, while on a trained net the largest weight is 41x a cap
# of 0.36 and the term reaches ~1e8. Any lambda is inert at one end or explosive at the other, and
# this project already carries spike_factor/restore_after because frm misbehaved far more mildly.
# The clamp has one parameter, cannot spike, and mirrors the existing post-step bias clamp.
#
# GRID (9 jobs) = 3 caps x 3 seeds, N = 1000, k = 3, pen = none, bias fixed at 0, 150k iterations.
#   cap in {0.1, 0.5, 2.0}, clamping 8.0% / 5.9% / 5.6% of the trained weight distribution.
#   All three sit above the initialisation scale 1/sqrt(N) = 0.032, so training can start.
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..8): REP = TID%3; C_IDX = TID/3
#
# READ-OUT, fixed before submission. CO-PRIMARY, and both must be reported for every arm:
#   1. active units at 150,000, scale-free criterion, against the pen=none reference 262.5 +- 14.3.
#   2. clean task r2. A cap that recruits units by destroying the task is not a result. An arm
#      counts as SUPPORTING the hypothesis only if active units exceed 262.5 + 3*14.3 = 305 AND
#      the clean loss is within 10% of the reference.
# Also report the input row-norm CV per arm: if the cap recruits units, CV should fall toward the
# 0.74 that frm+rws reaches on its own. If CV falls but the active count does not, the causal
# direction is settled the OTHER way - equalisation does not cause recruitment - which is just as
# informative and should be written up as such.
#
# Cost: no extra forward pass, so ~150k at N=1000 is ~11 h as for any flip-flop cell. 24 h requested.
#   sbatch --array=1-9 slurm/SilentReLU_flipflop_inpcap_spock.slurm
#
# Output: $SAVE_TO/NBitFlipFlop_std_inpcap/EqType=h_k=3_N=1000_cap=<cap>_pen=none/

#SBATCH --job-name=FFinpcap
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=1-9
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
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
if [ -z "$GITSHA" ]; then
  echo "FATAL: $REPO is not a git checkout - provenance cannot be recorded." >&2; exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

K=3
N=1000
IT=${ITERS_OVERRIDE:-150000}
CAPS=(0.1 0.5 2.0)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); C_IDX=$((TID / 3))
CAP=${CAPS[$C_IDX]}

TAG="std_inpcap/EqType=h_k=${K}_N=${N}_cap=${CAP}_pen=none"
mkdir -p "${SAVE_TO}/NBitFlipFlop_std_inpcap"
echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K} N=${N} inp_weight_cap=${CAP} rep=${REP} iters=${IT}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=NBitFlipFlop \
  task.n_inputs="$K" task.n_outputs="$K" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  model.inp_weight_cap="$CAP" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
