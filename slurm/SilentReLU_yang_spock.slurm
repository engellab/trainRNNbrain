#!/bin/bash
# YANG'S 20 COGNITIVE TASKS IN ONE NETWORK vs ONE OF THEM ALONE: does multi-task load raise the
# recruitment on context-dependent decision-making? (SPOCK)
#
# The capacity law says a single-task network recruits ~10 units per latent function, which makes
# plain CDDM networks poor stand-ins for recorded populations (redundant samples of a circuit that
# is never doing only one task). The honest model of that is multi-task training. This is the
# Yang et al. (2019) family ported to this repo (tasks/TaskYang.py): one fixation input, two
# 32-unit direction rings, a 20-channel rule vector, one response ring - a stimulus direction means
# the same thing in every task, so shared units share a computation. The focus task is CDDM =
# the pair contextdm1 + contextdm2 (Mante's context-dependent decision in ring coding: both rings
# show the same two choice options with independent evidence; the rule channel is the context cue).
#
# Two sets, same architecture (85 inputs incl. all 20 rule channels, 33 outputs), same trainer:
#   multi20  task=Yang20       all 20 rules, batch 1024
#   ctxdm    task=Yang_ctxdm   contextdm1 + contextdm2 alone, batch 1024   <- the single-task reference
#
# Pre-registered read-out (multitask_readout.py --focus contextdm1,contextdm2, both sets through
# the same code):
#   1. live units on CDDM trials (either context, rule on, noise-free), scale-free AND absolute criteria.
#      "Multi-task raises recruitment on the focus task" := live_multi > live_single mean + 3 sd
#      in every seed at that N, under both criteria.
#   2. The split of the focus-active units into shared-with-another-task vs focus-private
#      (multi20 only): shared = nuisance from common inputs, private = the task's own code got
#      more redundant. Different meanings for subsampling; not to be conflated.
#   3. Per-rule r2 on each rule's scored window, so an unlearned rule is visible and its units
#      do not count as demand delivered. live_all on the mixed batch.
#
# Grid (36 jobs) = 2 sets x 3 sizes x 2 penalty arms x 3 seeds, h equation, standard ReLU, fresh batches:
#   set in {multi20, ctxdm};  N in {500, 1000, 2000};  pen in {none, both} (both = rws 0.05 + frm 0.1)
# Index decode (TID = SLURM_ARRAY_TASK_ID-1): REP=TID%3; P_IDX=(TID/3)%2; N_IDX=(TID/6)%3; S_IDX=TID/18
#   tasks 1-18 multi20 (1-6 N=500, 7-12 N=1000, 13-18 N=2000; 3 none then 3 both per size)
#   tasks 19-36 ctxdm, same order.
# 150k iterations as every other grid; multi-task may need more - the clean-loss trace decides, and
# a still-falling loss at 150k means a warm-started continuation, not a read-out at 150k.
# Deviation from the paper: uniform loss weight over scored entries (the paper weights the response
# epoch x5); Adam with this repo's lr scaling; no L2 on activity.
# Cost: batch 1000 x T=300 x 85 inputs ~ the flip-flop's per-iteration cost (0.24-0.26 s/iter at
# N<=1000, 0.55 at N=2000) plus trial generation (vectorised, ~0.09 s per 1024-trial batch):
#   Measured (calibration job 6202165, multi20 N=2000 none, 600 iterations, commit d8d5e84): 0.90 s/iter
#   -> 37 h at 150k; r2 0.88 on the mixed batch after 600 iterations.
#   sbatch --array=1-12,19-30  slurm/SilentReLU_yang_spock.slurm                      # N<=1000, 24 h
#   sbatch --array=13-18,31-36 --time=48:00:00 slurm/SilentReLU_yang_spock.slurm      # N=2000
# Timing calibration: ITERS_OVERRIDE=600 sbatch --array=13 --time=00:40:00 ...
#
# Output: $SAVE_TO/Yang_std_multi/EqType=h_set=<set>_N=<N>_pen=<name>/<per-net subfolder>

#SBATCH --job-name=Yang20
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=1-36
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

SETS=(multi20 ctxdm)
TASKCFG=(Yang20 Yang_ctxdm)
NS=(500 1000 2000)
PEN_NAME=(none both)
PEN_RWS=(0 0.05)
PEN_FRM=(0 0.1)
IT=150000

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); P_IDX=$(((TID / 3) % 2)); N_IDX=$(((TID / 6) % 3)); S_IDX=$((TID / 18))
N=${NS[$N_IDX]}; SET=${SETS[$S_IDX]}; TCFG=${TASKCFG[$S_IDX]}
PNAME=${PEN_NAME[$P_IDX]}; LRWS=${PEN_RWS[$P_IDX]}; LFRM=${PEN_FRM[$P_IDX]}
IT=${ITERS_OVERRIDE:-$IT}

TAG="std_multi/EqType=h_set=${SET}_N=${N}_pen=${PNAME}"
mkdir -p "${SAVE_TO}/Yang_std_multi"
echo "Task ${SLURM_ARRAY_TASK_ID}: set=${SET} (task=${TCFG}) N=${N} pen=${PNAME} (rws=${LRWS} frm=${LFRM}) rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/Yang_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task="$TCFG" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws="$LRWS" trainer.lambda_frm="$LFRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
