#!/bin/bash
# DMTS SETTING SWEEPS (DELLA): does any standard training setting stop the decrease of the number of
# active units on DMTS, as Results 3 tests on CDDM and 3BFF? Package dmts-sweeps (dead-relu-paper
# #127). Plan, read-out and pass/fail rule: docs/project_trajectory/*_dmts-sweeps-plan.md.
#
# The CDDM setting sweeps behind Results 3 and Extended Data Fig. 4 vary eight settings at N = 1000;
# 3BFF varies the initial input-weight scale. This array runs, on DMTS (5-tau delay, task DMTS_d5),
# the default network and, for each of those eight settings, the one value that gave the most active
# units at 75,000 iterations on CDDM (3BFF for the input-weight scale), i.e. the value most likely to
# stop the decrease (entry dmts-sweeps-plan; means from data/fig_paper_F2_fails_cache.pkl):
#
#   arm          override                                         CDDM (3BFF) mean at 75k vs default
#   default      (none)                                           337 (275)
#   wd0          trainer.weight_decay=0                           453
#   sigrec0.1    model.sigma_rec=0.1                              397 vs 327 (in-sweep default)
#   met1         trainer.lambda_met=1.0                           375
#   sr0.5        model.spectral_rad=0.5                           361 vs 340 (in-sweep default)
#   rwm0.2       +trainer.lambda_rwm=0.2 +trainer.rwm_args={}     448 vs 340 (in-sweep default)
#   softplus25   model.activation_args.name=softplus              375
#                +model.activation_args.beta=25.0
#   rws0.05      trainer.lambda_rws=0.05                          265 vs 325 (its only value)
#   winp20       model.input_row_norm=20                          3BFF 477 vs 275
#
# softplus is set on rnn_relu_standard, NOT with model=rnn_softplus_beta25_standard: that file has
# no init_draws key, so it falls back to the code default "shared" (the correlated initial draw)
# while rnn_relu_standard draws "independent"; the arm would vary two things at once.
#
# Every run: standard network (rnn_relu_standard, trainer_ptrack_freshbatch), N = 1000, 150,000
# iterations (the DMTS default budget: DMTS stalls before it escapes, and its floor crossings at
# N <= 1000 fall at 18-118k), a kept weight snapshot every 1,000 iterations, seed drawn at random
# and recorded in the saved config.
#
# GRID: 9 arms x 3 seeds = 27 tasks. TID = SLURM_ARRAY_TASK_ID - 1; REP = TID % 3; ARM = TID / 3.
#   sbatch --chdir=$REPO --array=1-27 slurm/SilentReLU_dmts_settings_della.slurm
# Runtime: DMTS at N = 1000 took 6.4-6.6 h per 150,000 iterations on the A100 80 GB nodes
# (jobs 14830790_4-6, 15422708_1-3), hence --constraint=gpu80 and a 14 h limit (gpu-short QOS).
# Cost: 27 x 6.5 = about 176 GPU-h.
#
# Output: $SAVE_TO/DMTS_std_settings/EqType=h_N=1000_arm=<arm>_iters=150000/<run>/
#         $SNAP_TO/DMTS_std_settings/EqType=h_N=1000_arm=<arm>_iters=150000/s<seed>/it<iter>/
#
#SBATCH --job-name=DMTSset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem-per-gpu=64G
#SBATCH --time=14:00:00
#SBATCH --array=1-27
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

REPO="${REPO:-$HOME/trainRNNbrain_dmtsset}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"
SNAP_TO="${SNAP_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) - aborting." >&2
  echo "       submit with --chdir=\$REPO; python -c puts the working directory ahead of PYTHONPATH." >&2
  exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
if [ -z "$GITSHA" ]; then
  echo "FATAL: $REPO is not a git checkout - provenance of this run cannot be recorded." >&2; exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

N=${N:-1000}
IT=${ITERS_OVERRIDE:-150000}
NSEED=3
# One flat cell list: "<arm>|<hydra overrides, space-separated>". Seeds are the inner index.
CELLS=(
  "default|"
  "wd0|trainer.weight_decay=0"
  "sigrec0.1|model.sigma_rec=0.1"
  "met1|trainer.lambda_met=1.0"
  "sr0.5|model.spectral_rad=0.5"
  "rwm0.2|+trainer.lambda_rwm=0.2 +trainer.rwm_args={}"
  "softplus25|model.activation_args.name=softplus +model.activation_args.beta=25.0"
  "rws0.05|trainer.lambda_rws=0.05"
  "winp20|model.input_row_norm=20"
)
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % NSEED)); CELL=${CELLS[$((TID / NSEED))]}
ARM=${CELL%%|*}
read -r -a ARM_ARGS <<< "${CELL#*|}"

TAG="std_settings/EqType=h_N=${N}_arm=${ARM}_iters=${IT}"
mkdir -p "${SAVE_TO}/DMTS_std_settings" "$SNAP_TO"
echo "Task ${SLURM_ARRAY_TASK_ID}: dmts N=${N} arm=${ARM} overrides='${CELL#*|}' rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/DMTS_${TAG}"
echo "  snapshots -> ${SNAP_TO}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=DMTS_d5 \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" "${ARM_ARGS[@]}" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.prune_reinit=false trainer.synaptic_scaling=false trainer.dropout=false \
  trainer.checkpoint_every=1000 ++trainer.checkpoint_keep=true \
  paths.checkpoint_to="$SNAP_TO" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
