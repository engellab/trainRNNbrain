#!/bin/bash
# SYNAPTIC NOISE: HOW FAR DOES IT GO, AND DOES IT TRANSFER? (DELLA)
#
# model.sigma_w multiplies W_rec and W_inp by (1 + sigma_w * eps) with a fresh eps at every
# timestep. On the 3-bit flip-flop at N = 1000, 40k iterations, against a control of 277 active
# units at 5.46 dimensions and r2 0.945:
#
#   sigma_w 0.1   369 active   5.69 dims   r2 0.9466   spread 218,698
#   sigma_w 0.3   370 active   5.66 dims   r2 0.9406   spread 413,093
#   sigma_w 1.0   545 active   6.69 dims   r2 0.9277   spread 258,481
#   control       277 active   5.46 dims   r2 0.9450   spread 265,470
#
# Monotone on all three, and the participation spread is untouched -- this is not the failure where
# a jitter violent enough to stop any unit settling lifts the floor and passes all 1000, which is
# how heavy weight decay once produced N/N. It also costs nothing: these runs took 2h52m-3h04m
# against 3h23m for the same cell without weight noise.
#
# TWO THINGS ARE UNKNOWN AND THIS SWEEP ASKS BOTH.
#
# 1. WHERE IT STOPS. Every measured level improves on the one below it, so the top of the range is
#    untested. At sigma_w 2.0 a synapse changes sign on about 31% of steps and at 3.0 on about 37%,
#    which is a different regime from the 16% at 1.0 -- the perturbation stops being a jitter on the
#    trained connectivity and starts being a resampling of it. If the trend continues there, the
#    mechanism is not what I think it is and the interpretation has to change; if training breaks,
#    that bounds the useful range.
#
# 2. WHETHER IT IS ONE TASK AND ONE SIZE. Everything above is N = 1000 on the 3-bit flip-flop.
#    Silencing gets worse with size (the active FRACTION falls from 41% to 13% over N = 500 to
#    4000), so a rule that recruits at N = 1000 and not at N = 2000 is worth much less.
#
# GRID (15 jobs), one entry per index because the cells differ in task, size and iteration budget:
#   1-3    3-bit flip-flop  N=1000  sigma_w 2.0   40k
#   4-6    3-bit flip-flop  N=1000  sigma_w 3.0   40k
#   7-9    3-bit flip-flop  N= 500  sigma_w 1.0   40k
#   10-12  3-bit flip-flop  N=2000  sigma_w 1.0   40k
#   13-15  CDDM             N=1000  sigma_w 1.0  100k
#
# NO CONTROL CELLS. sigma_w = 0 is bit-for-bit the historical forward pass, pinned by
# tests/test_synaptic_weight_noise.py, so the existing arms ARE the matched controls:
# NBitFlipFlop_ff_revive arm=none at N = 500/1000/2000 (211, 275, 451 active) and
# CDDM_paper_grid arm=control at N = 1000 (316 active, r2 0.875, 1.96 dims).
#
# ⚠️ CDDM's CONTROL IS AT 100,000 ITERATIONS, which is why cells 13-15 run to 100k and not 40k.
# Reading a 40k run against a 100k control would compare a network that has barely begun silencing
# with one that has finished, which is the confound Fig. 1c was just rebuilt to avoid.
#
# SUBMIT IN TWO GROUPS so the short cells are not queued behind a 40 h request:
#   sbatch --array=1-12  --time=24:00:00  slurm/SilentReLU_synnoise_ext_della.slurm
#   sbatch --array=13-15 --time=40:00:00  slurm/SilentReLU_synnoise_ext_della.slurm
# Measured: 3 h at N = 1000 on the flip-flop; N = 2000 draws a 4x larger matrix per step; CDDM at
# 100k took 25 h in the paper grid without weight noise.
#
# PRE-REGISTERED. Read-out at the cell's own iteration count, scale-free criterion.
#   PRIMARY    active units against that size and task's own control.
#   CO-PRIMARY activity dimensionality, and held-out r2: more than 0.10 below the control is BROKEN
#              whatever the count, as is more than 5% of gradient updates discarded.
#   ⚠️ SPREAD BEFORE COUNT at sigma_w 2.0 and 3.0. The criterion is relative to each network's own
#              95th percentile, so a network whose units all rattle equally passes every unit. The
#              control's participation spread (q99/q01) is 265,470 and every working level so far
#              has stayed near it; a cell that recruits while its spread collapses has broken the
#              criterion, not the silence.
#   WHAT WOULD CHANGE MY MIND: if 2.0 and 3.0 keep improving, the effect is not "jitter keeps
#              marginal units across threshold" but something closer to resampling the network, and
#              the mechanism claim must be rewritten. If the flip-flop result does not survive to
#              N = 2000 or to CDDM, it is a single-cell curiosity and should be reported as one.

#SBATCH --job-name=SynExt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-15%15
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
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
if [ -n "$GITSHA" ]; then
  if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
    echo "FATAL: $REPO is a git checkout with uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
  fi
  echo "code: $PKG @ $GITSHA (git)"
elif [ -f "$REPO/CODE_VERSION.txt" ]; then
  echo "code: $PKG @ $(cat "$REPO/CODE_VERSION.txt") (rsync)"
else
  echo "FATAL: $REPO is neither a clean git checkout nor carries CODE_VERSION.txt — provenance of this run cannot be recorded." >&2; exit 1
fi

# index 0 unused, so the decode reads SLURM_ARRAY_TASK_ID directly
TASK_TASK=(x NBitFlipFlop NBitFlipFlop NBitFlipFlop NBitFlipFlop NBitFlipFlop NBitFlipFlop \
             NBitFlipFlop NBitFlipFlop NBitFlipFlop NBitFlipFlop NBitFlipFlop NBitFlipFlop \
             CDDM CDDM CDDM)
TASK_N=(0 1000 1000 1000 1000 1000 1000 500 500 500 2000 2000 2000 1000 1000 1000)
TASK_SW=(0 2.0 2.0 2.0 3.0 3.0 3.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)
TASK_IT=(0 40000 40000 40000 40000 40000 40000 40000 40000 40000 40000 40000 40000 100000 100000 100000)

i=$SLURM_ARRAY_TASK_ID
TASK=${TASK_TASK[$i]}; N=${TASK_N[$i]}; SW=${TASK_SW[$i]}; IT=${TASK_IT[$i]}

TAG="synnoise_ext/EqType=h_N=${N}_sw=${SW}_iters=${IT}"
mkdir -p "${SAVE_TO}/${TASK}_synnoise_ext"
echo "Task ${i}: task=${TASK} N=${N} sigma_w=${SW} iters=${IT}"
echo "  -> ${SAVE_TO}/${TASK}_${TAG}"

if [ "$TASK" = "NBitFlipFlop" ]; then
  srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
    seed="random" +experiment=silent_units_largeN \
    task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3 \
    model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
    model.equation_type=h model.N="$N" "model.sigma_w=${SW}" \
    trainer.max_iter="$IT" \
    trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
    trainer.track_valid=false trainer.lambda_rws=0 trainer.lambda_frm=0 \
    trainer.prune_reinit=false trainer.synaptic_scaling=false \
    paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\""
else
  srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
    seed="random" +experiment=silent_units_largeN \
    task=CDDM \
    model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
    model.equation_type=h model.N="$N" "model.sigma_w=${SW}" \
    trainer.max_iter="$IT" \
    trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
    trainer.track_valid=false trainer.lambda_rws=0 trainer.lambda_frm=0 \
    trainer.prune_reinit=false trainer.synaptic_scaling=false \
    paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\""
fi
