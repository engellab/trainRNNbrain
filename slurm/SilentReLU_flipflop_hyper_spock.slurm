#!/bin/bash
# COMBINATORIAL TASK DEMAND: the k-bit flip-flop with a 2^k-1 channel read-out (SPOCK)
#
# The plain k-bit flip-flop's demand is LINEAR in k (k output channels), and the active-unit count
# barely moves with it: live units ~ k^0.16, M ~ 21 N^0.42 independent of k at matched loss - the
# pre-registered pathology outcome of the ksweep. The deflationary reply is that k bits are simply
# cheap. This sweep is the steelman: same inputs and bit dynamics, but the target is the product of
# the bit states over EVERY non-empty subset of bits, 2^k-1 channels (3 / 15 / 63 / 255 at
# k = 2 / 4 / 6 / 8). If active units track task demand at all, they must track THIS.
#
# Pre-registered read-out (fixed before any run finishes; entry of 2026-09-11 15:02 in
# docs/project_trajectory.md):
#   1. M(k) rises steeply with 2^k-1 (exponent on k well above the plain task's 0.16, or clearly
#      multiplicative in the channel count) -> recruitment DOES track demand once demand is real;
#      the plain flip-flop's flat M(k) was cheap demand, not pathology.
#   2. M(k) stays flat (~k^0.2) AND the task is solved (per-channel R^2 comparable to the plain
#      task at every subset size) -> the pathology reading survives its strongest test.
#   3. High-k cells FAIL to train (R^2 collapses at k=6/8 while M stays flat) -> the silence is a
#      capacity ceiling the task cannot lift; equally a result, recorded as such.
#   The `both` arm (rws 0.05 + frm 0.1) asks whether the penalty rescues whichever of 2/3 happens.
#
# Read-out iteration: 150k for every cell, on the scale-free criterion, exactly as the ksweep and
# penalty grids were read (a 150k penalised number against a 500k unpenalised one is invalid).
# Per-channel R^2 against each channel's OWN variance: unlike the plain task the target variance is
# NOT channel-independent here - it falls with subset size (0.70 singles -> 0.29 for the 8-way
# product at k=8), so the raw MSE mixes channels of different variance and R^2 = 1 - MSE/0.735 no
# longer applies.
#
# Grid (72 jobs) = 4 complexities x 3 sizes x 3 seeds x 2 penalty arms, h equation, standard ReLU.
#   k   in {2,4,6,8}      task.n_inputs=k, task.n_outputs=2^k-1 (the class asserts this)
#   N   in {500,1000,2000}
#   pen in {none, both}   both = rws 0.05 + frm 0.1, the project's standard pair
# Index decode (TID = SLURM_ARRAY_TASK_ID-1): REP=TID%3; N_IDX=(TID/3)%3; K_IDX=(TID/9)%4; P_IDX=TID/36
#   tasks 1-36 unpenalised, 37-72 both.
#
# Cost: the batch expansion adds 0.16 s/iter at k=8 (255 in-place multiplies over 300x1024) on top
# of 0.24-0.26 s/iter (N=500/1000) and 0.55 (N=2000): worst cell ~0.71 s/iter x 150k = 30 h.
# 48 h requested for all (Spock partitions have no wall-time limit). Memory: targets are
# 255 x 300 x 1024 float32 = 313 MB at k=8, well inside the 32G that N=2000 already needs.
#   sbatch --array=1-36 slurm/SilentReLU_flipflop_hyper_spock.slurm    # none
#   sbatch --array=37-72 slurm/SilentReLU_flipflop_hyper_spock.slurm   # both
# Timing override for a calibration run: ITERS_OVERRIDE=600 sbatch --array=36 ...
#
# Output: $SAVE_TO/NBitFlipFlopHyper_std_hyper/EqType=h_k=<k>_N=<N>_pen=<name>/<per-net subfolder>

#SBATCH --job-name=FFhyp
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=1-72
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

KS=(2 4 6 8)
NS=(500 1000 2000)
PEN_NAME=(none both)
PEN_RWS=(0 0.05)
PEN_FRM=(0 0.1)
IT=150000

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); N_IDX=$(((TID / 3) % 3)); K_IDX=$(((TID / 9) % 4)); P_IDX=$((TID / 36))
K=${KS[$K_IDX]}; N=${NS[$N_IDX]}; NOUT=$((2 ** K - 1))
PNAME=${PEN_NAME[$P_IDX]}; LRWS=${PEN_RWS[$P_IDX]}; LFRM=${PEN_FRM[$P_IDX]}
IT=${ITERS_OVERRIDE:-$IT}

TAG="std_hyper/EqType=h_k=${K}_N=${N}_pen=${PNAME}"
mkdir -p "${SAVE_TO}/NBitFlipFlopHyper_std_hyper"
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
