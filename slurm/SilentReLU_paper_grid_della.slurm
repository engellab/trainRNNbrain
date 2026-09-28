#!/bin/bash
# THE PAPER GRID: the two interventions that worked, against their control, at four sizes, on
# three tasks. (DELLA)
#
# WHY THIS EXISTS. The intervention results were gathered piecemeal and the duplication arm is not
# the same rule in every folder. The size series (`ff_revive_g01_fix`) ran duplication UNCAPPED
# with no maturity protection at gamma 0.1, while the jitter sweep (`copy_perturb`) ran it capped
# at 0.025 with maturity 1000 at gamma 0. Those differ in five settings, and the difference shows:
# at N=1000 the uncapped version lands at r2 0.914 with two seeds at 0.896, the capped one at
# 0.944. Anything going in the paper has to be ONE rule.
#
# THE RULE, fixed here for every cell:
#   duplication  reinit_mode=copy, copy_noise=1.0 with copy_norm_preserve, replacement capped at
#                0.025 of units per event, maturity 1000, contribution utility, gamma 0.
#                copy_noise 1.0 puts the copy at an expected cosine of 0.707 to its donor, so it is
#                not a twin; measured at N=1000 it recruits 782 active units against 706 at
#                copy_noise 0 and costs no r2 (0.944 against 0.941).
#   mute dropout drop rate 0.20, beta 4, participation sampling -- the settings of the completed
#                flip-flop size series, which this reuses rather than re-running.
#   control      neither, otherwise identical.
#
# WHY copy_noise 1.0 AND NOT 0.05. Across the jitter sweep the active count rises monotonically
# (685, 676, 708, 751, 752 at jitter 0, 0.05, 0.3, 1.0, 3.0) and the final weight distribution is
# invariant -- sigma of log|W| sits at 1.42-1.43 across the whole range, because the 40000
# iterations after each replacement reshape whatever the jitter did. So the largest jitter that is
# still clearly a copy is the defensible choice: it recruits the most and owes the donor the least.
#
# GRID (36 jobs per task) = 3 arms x 4 sizes x 3 seeds.
#   ARM  0 control, 1 duplication, 2 mute dropout
#   SIZE 0 N=500, 1 N=1000, 2 N=2000, 3 N=4000
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..35): REP = TID%3; S_IDX = (TID/3)%4; A_IDX = TID/12
#   control      tasks  1-12   (N=500: 1-3, N=1000: 4-6, N=2000: 7-9, N=4000: 10-12)
#   duplication  tasks 13-24
#   mute dropout tasks 25-36
#
# TASK selects the task and its iteration count, which are NOT interchangeable: DMTS needs 150000
# iterations to be solved at all (at 40000 the unpenalised network does not escape), CDDM needs
# 100000 (the 2026-07-28 horizon experiment showed 30000 is not convergence -- the loss is flat
# while the active count is still falling), and the flip-flop series is established at 40000.
#   TASK=ff    NBitFlipFlop, 3 bits, 40000 iters
#   TASK=dmts  DMTS_d7, 150000 iters
#   TASK=cddm  CDDM, 100000 iters
#
# MEASURED RUNTIMES, from the completed sweeps, per job:
#   ff    0.75 h (N=500), 2.9 h (1000), 7 h (2000), 25 h (4000)
#   dmts  5-7 h,          7-9 h,        18-20 h,    ~45 h  <-- N=4000 EXCEEDS ANY SINGLE ALLOCATION
#   cddm  2 h,            3.9 h,        8.5 h,      ~20 h
# Submit each task with its own --time; DMTS at N=4000 needs checkpoint-restart via the warm-start
# path (`paths.init_from` + `trainer.init_adam`), not a longer limit, and is NOT included below.
#
# READ-OUT, fixed before submission: active units under the scale-free criterion, activity
# dimensionality (participation ratio of the firing-rate covariance over the active units), and
# held-out r2, each gated on a recomputed r2 matching the stored value.
# STABILITY: any cell discarding more than 5% of its gradient updates is reported as BROKEN.
# ⚠️ For DMTS, report the SOLVE RATE per cell alongside every count: at 7 tau the unpenalised
# network solves 2/3 seeds at N=500 and N=2000, and an active-unit count pooled over solvers and
# failures describes no network that exists.
#
#   TASK=ff   sbatch --time=04:00:00   --array=13-21%6 slurm/SilentReLU_paper_grid_della.slurm
#   TASK=dmts sbatch --time=30:00:00   --array=13-21,25-33%6 ...
#   TASK=cddm sbatch --time=16:00:00   --array=1-3,10-36%6 ...
#
# Output: $SAVE_TO/<Task>_paper_grid/EqType=h_N=<N>_arm=<arm>/

#SBATCH --job-name=PaperGrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=64G
#SBATCH --time=08:00:00
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

REPO="${REPO:-$HOME/trainRNNbrain_papergrid}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2
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

SIZES=(500 1000 2000 4000)
ARMS=(control duplication mute)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); S_IDX=$(((TID / 3) % 4)); A_IDX=$((TID / 12))
N=${SIZES[$S_IDX]}
ARM=${ARMS[$A_IDX]}

case "${TASK:-ff}" in
  ff)   TASK_ARGS=(task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3); IT=40000;  PREFIX=NBitFlipFlop ;;
  dmts) TASK_ARGS=(task=DMTS_d7);                                       IT=150000; PREFIX=DMTS ;;
  cddm) TASK_ARGS=(task=CDDM);                                          IT=100000; PREFIX=CDDM ;;
  *) echo "FATAL: TASK must be ff, dmts or cddm; got '${TASK}'" >&2; exit 1 ;;
esac
IT=${ITERS_OVERRIDE:-$IT}

ARM_ARGS=(trainer.prune_reinit=false trainer.synaptic_scaling=false trainer.dropout=false)
case "$ARM" in
  duplication) ARM_ARGS=(trainer.synaptic_scaling=false trainer.dropout=false
                         trainer.prune_reinit=true
                         trainer.prune_args.reinit_mode=copy
                         trainer.prune_args.copy_noise=1.0
                         trainer.prune_args.copy_norm_preserve=true
                         trainer.prune_args.max_replace_frac=0.025
                         trainer.prune_args.maturity=1000
                         trainer.prune_args.utility=contribution) ;;
  mute)        ARM_ARGS=(trainer.prune_reinit=false trainer.synaptic_scaling=false
                         trainer.dropout=true
                         trainer.dropout_args.dropout_kind=mute
                         trainer.dropout_args.sampling_method=participation
                         trainer.dropout_args.drop_rate=0.20
                         trainer.dropout_args.dropout_beta=4) ;;
esac

TAG="${SWEEP:-paper_grid}/EqType=h_N=${N}_arm=${ARM}"
mkdir -p "${SAVE_TO}/${PREFIX}_${SWEEP:-paper_grid}"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${TASK:-ff} N=${N} arm=${ARM} rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/${PREFIX}_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  "${TASK_ARGS[@]}" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  "${ARM_ARGS[@]}" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
