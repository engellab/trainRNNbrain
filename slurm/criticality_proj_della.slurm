#!/bin/bash
# CRITICALITY PROJECTION: four W_rec projections against a matched control, N = 1000, 3BFF and CDDM. (DELLA)
#
# Rules, written before any run: docs/project_trajectory/2026-11-20_1643_rules-criticality-projection.md.
# The command is SilentReLU_paper_grid_della.slurm's (the launcher of the CDDM paper-grid control and of
# prune-and-duplicate on CDDM) with two differences: +experiment=criticality_proj (silent_units_largeN plus
# the projection block, off) and trainer.wrec_projection=<arm>, the one swept key. Period, strength and the
# Lyapunov settings live in configs/experiment/criticality_proj.yaml.
#
# GRID (15 tasks per TASK) = 5 arms x 3 seeds. TID = SLURM_ARRAY_TASK_ID - 1; REP = TID % 3; A_IDX = TID / 3.
#   control 1-3, spectral (a) 4-6, singular (b) 7-9, rownorm (c) 10-12, lyapunov (d) 13-15
# TASK selects the task and its budget:
#   TASK=cddm  CDDM, 100000 iterations (the paper grid's budget)          ~4 h at N=1000
#   TASK=ff    NBitFlipFlop 3 bits, 200000 iterations (the remedy grid's) ~15 h at N=1000
#
# Usage (from a clean clone pinned to the submission commit; --chdir is required by the import check):
#   C=--chdir=$HOME/trainRNNbrain_crit; L=slurm/criticality_proj_della.slurm
#   TASK=cddm sbatch $C --export=ALL --time=08:00:00 --array=1-15 $L
#   TASK=ff   sbatch $C --export=ALL --time=23:00:00 --array=1-15 $L
#
# Output:    $SAVE_TO/<Task>_criticality_proj/EqType=h_N=<N>_arm=<arm>/<run>/
# Snapshots: $SNAP_TO/<Task>_criticality_proj/EqType=h_N=<N>_arm=<arm>/s<seed>/it<iters>/ (every 1000,
#   a multiple of the projection period, so every snapshot holds projected weights)
#
#SBATCH --job-name=CritProj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

REPO="${REPO:-$HOME/trainRNNbrain_crit}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs_crit}"
SNAP_TO="${SNAP_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) - submit with --chdir=\$REPO." >&2
  exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
[ -n "$GITSHA" ] || { echo "FATAL: $REPO is not a git checkout." >&2; exit 1; }
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

ARMS=(control spectral singular rownorm lyapunov)
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); A_IDX=$((TID / 3))
ARM=${ARMS[$A_IDX]}
[ -n "$ARM" ] || { echo "FATAL: array index ${SLURM_ARRAY_TASK_ID} is outside 1-15." >&2; exit 1; }
N=${N:-1000}
PROJ=$([ "$ARM" = control ] && echo none || echo "$ARM")

case "${TASK:-}" in
  ff)   TASK_ARGS=(task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3); IT=200000; PREFIX=NBitFlipFlop ;;
  cddm) TASK_ARGS=(task=CDDM);                                          IT=100000; PREFIX=CDDM ;;
  *) echo "FATAL: TASK must be ff or cddm; got '${TASK:-}'" >&2; exit 1 ;;
esac
IT=${ITERS_OVERRIDE:-$IT}

TAG="criticality_proj/EqType=h_N=${N}_arm=${ARM}"
mkdir -p "${SAVE_TO}/${PREFIX}_criticality_proj" "$SNAP_TO"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${TASK} N=${N} arm=${ARM} (wrec_projection=${PROJ}) rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/${PREFIX}_${TAG}"
echo "  snapshots -> ${SNAP_TO}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=criticality_proj \
  "${TASK_ARGS[@]}" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.prune_reinit=false trainer.synaptic_scaling=false trainer.dropout=false \
  trainer.wrec_projection="$PROJ" \
  paths.checkpoint_to="$SNAP_TO" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
