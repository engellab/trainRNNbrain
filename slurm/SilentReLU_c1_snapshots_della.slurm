#!/bin/bash
# CLAIM 1 AT THE FLOOR: unpenalised standard controls that KEEP their weights along training. (DELLA)
#
# WHY THIS EXISTS. Claim 1 of the paper is "training leaves most units silent, and the silent units
# do no work". The second half is an ablation, and the project reads every network where its own
# loss first reaches 1.05x its own fitted floor. No standard network on disk has weights there: a
# run saves only its end state (LastParams), its noisy-loss minimum (BestParams) and, for top-ups,
# its parent's end state. On CDDM the floor crossing lands at 6-17k iterations, far from all of
# them. These runs are the existing control cells re-trained with `trainer.checkpoint_keep`, so
# every 500 iterations leaves a weight snapshot and the network can be read wherever its floor is.
#
# THE CELLS THEY REPLICATE, setting for setting (paper-grid control arm, standard network: ReLU,
# bias fixed at 0, self-connections on, no Dale, gamma 0, wd 1e-6, lr 1e-3 (100/N)^(1/3)):
#   TASK=ff    NBitFlipFlop 3-bit, 75,000 iters  -> NBitFlipFlop_ff_revive_c75000 arm=none (40k + 35k)
#   TASK=cddm  CDDM,         100,000 iters       -> CDDM_paper_grid arm=control
#   TASK=dmts  DMTS_d5,      150,000 iters       -> DMTS_paper_grid_d5 arm=control
# Comparing each new run's end state with its cell's is the check that nothing but the snapshots
# changed.
#
# SNAPSHOTS go to a separate tree (paths.checkpoint_to), never inside the data tree: every cell
# inventory counts subfolders holding *LastParams* as finished runs. One snapshot is LastParams +
# config, ~8 MB at N = 1000; 150-300 per run.
#
# GRID: 3 seeds per task, N from the environment (default 1000).
#   TASK=ff   sbatch --time=12:00:00 --array=1-3 slurm/SilentReLU_c1_snapshots_della.slurm
#   TASK=cddm sbatch --time=10:00:00 --array=1-3 slurm/SilentReLU_c1_snapshots_della.slurm
#   TASK=dmts sbatch --time=18:00:00 --array=1-3 slurm/SilentReLU_c1_snapshots_della.slurm
# Runtimes measured on the paper grid at N = 1000: ff 2.9 h per 40k, cddm 3.9 h per 100k,
# dmts 7-9 h per 150k.
#
# INPUT_ROW_NORM=<s> (optional) is the one swept knob: it sets model.input_row_norm, the
# initialisation scale of each unit's input weights, and the arm becomes winp<s>. The flip-flop's
# input-scale ladder repeated on CDDM, against these same controls as its reference:
#   TASK=cddm INPUT_ROW_NORM=5 ITERS_OVERRIDE=50000 sbatch --time=04:00:00 --array=1-3 <this>
#
# Output: $SAVE_TO/<Task>_c1_snapshots/EqType=h_N=<N>_arm=control/<run>/
#         $SNAP_TO/<Task>_c1_snapshots/EqType=h_N=<N>_arm=control/s<seed>/it<iters>/
#
#SBATCH --job-name=C1snap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=64G
#SBATCH --time=12:00:00
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

REPO="${REPO:-$HOME/trainRNNbrain_c1abl}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"
SNAP_TO="${SNAP_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots}"
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

N=${N:-1000}
REP=$((SLURM_ARRAY_TASK_ID - 1))
case "${TASK:-ff}" in
  ff)   TASK_ARGS=(task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3); IT=75000;  PREFIX=NBitFlipFlop ;;
  cddm) TASK_ARGS=(task=CDDM);                                          IT=100000; PREFIX=CDDM ;;
  dmts) TASK_ARGS=(task=DMTS_d5);                                       IT=150000; PREFIX=DMTS ;;
  *) echo "FATAL: TASK must be ff, cddm or dmts; got '${TASK}'" >&2; exit 1 ;;
esac
IT=${ITERS_OVERRIDE:-$IT}
ARM=control
KNOB=()
if [ -n "${INPUT_ROW_NORM:-}" ]; then ARM="winp${INPUT_ROW_NORM}"; KNOB=(model.input_row_norm="$INPUT_ROW_NORM"); fi
TAG="c1_snapshots/EqType=h_N=${N}_arm=${ARM}"
mkdir -p "${SAVE_TO}/${PREFIX}_c1_snapshots" "$SNAP_TO"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${TASK:-ff} N=${N} rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/${PREFIX}_${TAG}"
echo "  snapshots -> ${SNAP_TO}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  "${TASK_ARGS[@]}" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" "${KNOB[@]}" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  trainer.prune_reinit=false trainer.synaptic_scaling=false trainer.dropout=false \
  trainer.checkpoint_every=500 +trainer.checkpoint_keep=true \
  paths.checkpoint_to="$SNAP_TO" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
