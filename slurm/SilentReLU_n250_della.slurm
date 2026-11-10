#!/bin/bash
# N = 250 CONTROLS FOR THE 3BFF AND CDDM SIZE SERIES (DELLA). Pavel, 9 Nov 2026: "can you also train 250
# unit networks: CDDM and 3BFF, to complete the picture." The size series of Figure 2 and Supplementary
# Table 1 has N = 100, 500, 1000, 2000, 4000; these cells fill the gap between 100 and 500.
#
# CODE VERSION: run from a detached worktree at 63c6c97, the commit the N = 100 cells ran from
# (SilentReLU_n100_spock.slurm, Spock job 6965913, entry 2026-10-15_2213). The training command below is
# that launcher's, unchanged except N and the task list. The N = 100 configs equal the N = 500-4000
# configs of the series in every field except N, seed, tag, save path and (3BFF only) the warm-start
# chain and budget split (checked on Della, 9 Nov 2026; rules entry 2026-11-09_*_n250-n100maze-rules.md).
# This launcher is ignored by git (/slurm/*.slurm), so copying it into the 63c6c97 worktree leaves the
# worktree clean; it is committed on main with git add -f for the record.
#
# Budgets as in the series: 3BFF 200k in one run (as at N = 100), CDDM 100k.
#
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..5): REP = TID%3; task = (ff, cddm)[TID/3].
#   sbatch --chdir=$HOME/trainRNNbrain_n250 --array=1-6 slurm/SilentReLU_n250_della.slurm
# Output: $SAVE_TO/<Task>_n250/EqType=h_N=250_arm=control/

#SBATCH --job-name=N250
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=16G
#SBATCH --time=24:00:00
#SBATCH --array=1-6
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

REPO="${REPO:-$HOME/trainRNNbrain_n250}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
if [ "$PKG" != "$(readlink -f "$REPO")" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO - submit with --chdir=\$REPO." >&2; exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
[ -n "$GITSHA" ] || { echo "FATAL: $REPO is not a git checkout." >&2; exit 1; }
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

N=250
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3))
TASKS=(ff cddm)
TASK=${TASKS[$((TID / 3))]:?array index out of range}
case "$TASK" in
  ff)   TASK_ARGS=(task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3); IT=200000; PREFIX=NBitFlipFlop ;;
  cddm) TASK_ARGS=(task=CDDM);                                          IT=100000; PREFIX=CDDM ;;
esac
IT=${ITERS_OVERRIDE:-$IT}
TAG="n250/EqType=h_N=${N}_arm=control"
mkdir -p "${SAVE_TO}/${PREFIX}_n250"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${TASK} N=${N} rep=${REP} iters=${IT}"
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
  trainer.prune_reinit=false trainer.synaptic_scaling=false trainer.dropout=false \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
