#!/bin/bash
# WALSH FLIP-FLOP: combinatorial demand with a ONE-CHANNEL read-out, none vs both (SPOCK)
#
# The hyper grid (2^k-1 product channels) showed recruitment tracking demand, but its demand and its
# read-out width grew together, so "the network fills output weights" could not be excluded, and its
# high-order channels switched faster than tau, capping r^2 by k. NBitFlipFlopWalsh removes both:
# the target is ONE channel, y = sum_{|S|<=degree} c_S prod_S b_i (c_S ~ N(0,1), one fixed draw,
# unit variance), the k bits start in a random state, every pulse flips its bit, and the TOTAL pulse
# rate is fixed (mu_total=8, ~7 changes of y per trial, hold ~36 steps at every k).
# `degree` is the demand knob at fixed k: the number of latent products the target contains is
#   k=8: degree 1 -> 8, 2 -> 36, 4 -> 162, 8 -> 255;  k=6 full -> 63.
# Everything else (inputs, output, trial statistics, k) is identical across degrees.
#
# Pre-registered read-out (docs/project_trajectory.md, 2026-09-13): live count at 150k under both
# criteria, r^2, and the RECOVERED WALSH SPECTRUM by order (flipflop_walsh_eval.py) - which latent
# products each network built, with no unit-level criterion. Predictions: `none` recruits up the
# degree ladder (8 -> 255 terms) with the same one output; `both` lights every unit as always -
# the question is whether its extra units carry any of the missing high-order terms at the cell
# where `none` is short of units (k=8 degree 8, N=500), or are noise.
#
# Grid (90 jobs) = 5 (k, degree) cells x 3 sizes x 3 seeds x 2 arms.
#   KD in {6:6, 8:1, 8:2, 8:4, 8:8}   (k, degree); task.n_outputs=1 always
#   N  in {500, 1000, 2000}
#   pen in {none, both}   both = rws 0.05 + frm 0.1
# Index decode (TID = SLURM_ARRAY_TASK_ID-1): REP=TID%3; N_IDX=(TID/3)%3; KD_IDX=(TID/9)%5; P_IDX=TID/45
#   tasks 1-45 none, 46-90 both.
#
# PAIRED SEEDS. seed = 101 + REP, NOT "random": the none and both arms of one rep share the network
# initialisation AND (task.seed = ${seed}) the same trial sequence, so their difference is the
# penalty alone. The coefficient draw (coef_seed=0) is the same for every job; the coefficient
# vector and generator settings are saved per net as *_TaskSpec_*.json (run_experiment.py).
#
# Cost: one output channel, so per-iteration cost is the plain task's (0.24 / 0.26 / 0.55 s/iter at
# N = 500 / 1000 / 2000 on L40S; slow nodes up to 1.4x). 150k iterations -> <= 32 h; 48 h requested.
#   sbatch --array=1-45 slurm/SilentReLU_flipflop_walsh_spock.slurm    # none
#   sbatch --array=46-90 slurm/SilentReLU_flipflop_walsh_spock.slurm   # both
# A test with ITERS_OVERRIDE lands in a separate folder (suffix _iters=<IT>), never in a real cell.
# Task index of a cell: 1 + REP + 3*N_IDX + 9*KD_IDX + 45*P_IDX  (k=6 N=1000 rep0: 4 none / 49 both).
#
# Output: $SAVE_TO/NBitFlipFlopWalsh_std_walsh/EqType=h_k=<k>_deg=<d>_N=<N>_pen=<name>/<per-net subfolder>

#SBATCH --job-name=FFwal
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=1-90
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

KK=(6 8 8 8 8)
DD=(6 1 2 4 8)
NS=(500 1000 2000)
PEN_NAME=(none both)
PEN_RWS=(0 0.05)
PEN_FRM=(0 0.1)
IT=150000

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); N_IDX=$(((TID / 3) % 3)); KD_IDX=$(((TID / 9) % 5)); P_IDX=$((TID / 45))
K=${KK[$KD_IDX]}; D=${DD[$KD_IDX]}; N=${NS[$N_IDX]}
PNAME=${PEN_NAME[$P_IDX]}; LRWS=${PEN_RWS[$P_IDX]}; LFRM=${PEN_FRM[$P_IDX]}
SEED=${SEED_OVERRIDE:-$((101 + REP))}   # SEED_OVERRIDE for warm-start continuations (fresh trials)
IT=${ITERS_OVERRIDE:-$IT}

# WARM START: INIT_FROM=<finished net folder> continues that network (weights, initial state AND Adam
# moments - same penalty, so the moments remove a restart transient rather than adding a bias) for
# $IT more iterations under THIS task's penalty arm. Pass a fresh seed so the continuation sees new
# trials (the trial RNG is seeded from the run seed); both arms of a pair should get the same one.
# The trace of the continuation starts at 0; add the source run's budget when reading it.
TAG="std_walsh/EqType=h_k=${K}_deg=${D}_N=${N}_pen=${PNAME}${ITERS_OVERRIDE:+_iters=${IT}}${INIT_FROM:+_warm}"  # overridden budgets / warm starts get their own folder
WARM=(); [ -n "${INIT_FROM:-}" ] && WARM=(paths.init_from="\"${INIT_FROM}\"" paths.init_adam=true)
mkdir -p "${SAVE_TO}/NBitFlipFlopWalsh_std_walsh"
echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K} degree=${D} N=${N} pen=${PNAME} (rws=${LRWS} frm=${LFRM}) rep=${REP} seed=${SEED} iters=${IT}"
echo "  -> ${SAVE_TO}/NBitFlipFlopWalsh_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="$SEED" \
  +experiment=silent_units_largeN \
  task=NBitFlipFlopWalsh \
  task.n_inputs="$K" task.n_outputs=1 task.degree="$D" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws="$LRWS" trainer.lambda_frm="$LFRM" \
  paths.save_to="$SAVE_TO" \
  "${WARM[@]}" \
  experiment_tag="\"${TAG}\""
