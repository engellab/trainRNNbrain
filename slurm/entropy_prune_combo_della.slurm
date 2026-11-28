#!/bin/bash
# ENTROPY SET POINT x PRUNE-AND-DUPLICATE, 2 x 2, ON 3BFF AND CDDM, N = 1,000 (DELLA)
#
# Rules: docs/project_trajectory/2026-11-25_*_rules-entropy-prune-combo.md, committed before any run.
# The command is that of entropy_setpoint_della.slurm / entropy_setpoint_cddm_della.slurm (which trained the
# entropy-alone networks), with the prune-and-duplicate keys of SilentReLU_paper_grid_della.slurm's
# `duplication` arm (check_every 100, patience 3, active_rel 0.05 from trainer_ptrack_freshbatch.yaml). Every cell
# uses +experiment=entropy_setpoint, so the cells differ only in lambda_ent and the prune keys.
#
# GRID: 4 cells x 3 networks; cell = (TID-1)/3, network = (TID-1)%3; the seeds are the entropy-alone networks' seeds.
#   1-3  none            4-6  ent0.1 (stage 1 refused: the existing networks are reused)
#   7-9  dup             10-12 ent0.1_dup
# STAGE=1 trains 75,000 iterations from scratch. STAGE=2 continues each cell's network to TARGET iterations from
# its 75,000-iteration weights and Adam state, with seed + 1 (a continuation with the same seed would replay the
# flip-flop trials of stage 1).
#
# Usage (from a clean clone pinned to the submission commit; --chdir is required by the import check):
#   C="--chdir=$HOME/trainRNNbrain_epc --export=ALL"; L=slurm/entropy_prune_combo_della.slurm
#   TASK=ff   STAGE=1 sbatch $C --time=14:00:00 --array=1-3,7-12 $L
#   TASK=cddm STAGE=1 sbatch $C --time=09:00:00 --array=1-3,7-12 $L
#   TASK=ff   STAGE=2 TARGET=<L> sbatch $C --time=23:00:00 --array=1-12 $L
#   TASK=ff   STAGE=1 SWEEP=entropy_prune_pilot ITERS_OVERRIDE=2000 sbatch $C --time=02:30:00 --array=1,7,10 $L
#
# Output:    $SAVE_TO/<Prefix>_<SWEEP>[_c]/EqType=h_N=1000_arm=<cell>/<run>/   (_c: stage 2)
# Snapshots: $SNAP_TO/<Prefix>_<SWEEP>[_c]/EqType=h_N=1000_arm=<cell>/s<seed>/it<iters>/
#
#SBATCH --job-name=EntPrune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=32G
#SBATCH --time=14:00:00
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

REPO="${REPO:-$HOME/trainRNNbrain_epc}"
SAVE_TO="${SAVE_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs}"
SNAP_TO="${SNAP_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots}"
SWEEP="${SWEEP:-entropy_prune}"
STAGE="${STAGE:-1}"
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

case "${TASK:-}" in
  ff)   TASK_ARGS=(task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3); PREFIX=NBitFlipFlop
        SEEDS=(4200852863 1256299004 4168822465) ;;
  cddm) TASK_ARGS=(task=CDDM); PREFIX=CDDM
        SEEDS=(3116616303 1299181146 2184072010) ;;
  *) echo "FATAL: TASK must be ff or cddm; got '${TASK:-}'" >&2; exit 1 ;;
esac

CELLS=(none ent0.1 dup ent0.1_dup)
TID=$((SLURM_ARRAY_TASK_ID - 1))
CELL=${CELLS[$((TID / 3))]}
[ -n "$CELL" ] || { echo "FATAL: array index ${SLURM_ARRAY_TASK_ID} is outside 1-12." >&2; exit 1; }
SEED=${SEEDS[$((TID % 3))]}
N=1000

CELL_ARGS=(trainer.lambda_ent=0)
case "$CELL" in ent0.1*) CELL_ARGS=(trainer.lambda_ent=0.1) ;; esac
case "$CELL" in
  *dup) CELL_ARGS+=(trainer.prune_reinit=true
                    trainer.prune_args.reinit_mode=copy
                    trainer.prune_args.copy_noise=1.0
                    trainer.prune_args.copy_norm_preserve=true
                    trainer.prune_args.max_replace_frac=0.025
                    trainer.prune_args.maturity=1000
                    trainer.prune_args.utility=contribution) ;;
  *)    CELL_ARGS+=(trainer.prune_reinit=false) ;;
esac

CONT=()
case "$STAGE" in
  1) if [ "$CELL" = ent0.1 ]; then echo "FATAL: stage 1 of ent0.1 exists and is reused." >&2; exit 1; fi
     IT=${ITERS_OVERRIDE:-75000}; ROOT="$SWEEP" ;;
  2) PSWEEP="$SWEEP"; if [ "$CELL" = ent0.1 ]; then PSWEEP=entropy_setpoint; fi
     mapfile -t PAR < <(ls -d "${PARENT_ROOT:-$SAVE_TO}/${PREFIX}_${PSWEEP}/EqType=h_N=${N}_arm=${CELL}/"*"seed=${SEED};"* 2>/dev/null)
     if [ "${#PAR[@]}" -ne 1 ]; then echo "FATAL: ${#PAR[@]} parents for cell ${CELL} seed ${SEED}, expected 1." >&2; exit 1; fi
     IT=$((${TARGET:?stage 2 needs TARGET} - ${PARENT_ITERS:-75000})); SEED=$((SEED + 1)); ROOT="${SWEEP}_c"
     # quoted for Hydra's override grammar: the run folder names hold "=" and ";"
     CONT=("++paths.init_from='${PAR[0]}'" ++paths.init_adam=true trainer.iter_offset=auto)
     echo "  parent: ${PAR[0]}" ;;
  *) echo "FATAL: STAGE must be 1 or 2; got '${STAGE}'" >&2; exit 1 ;;
esac

TAG="${ROOT}/EqType=h_N=${N}_arm=${CELL}"
mkdir -p "${SAVE_TO}/${PREFIX}_${ROOT}" "$SNAP_TO"
if ls -d "${SAVE_TO}/${PREFIX}_${TAG}/"*"seed=${SEED};"* >/dev/null 2>&1; then
  echo "FATAL: ${PREFIX}_${TAG} already holds a run of seed ${SEED}." >&2; exit 1
fi
echo "Task ${SLURM_ARRAY_TASK_ID}: ${TASK} stage ${STAGE} cell=${CELL} seed=${SEED} iters=${IT} (${CELL_ARGS[*]})"
echo "  -> ${SAVE_TO}/${PREFIX}_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="$SEED" \
  +experiment=entropy_setpoint \
  "${TASK_ARGS[@]}" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="${N_OVERRIDE:-$N}" model.sigma_w=0 \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_frm=0 trainer.lambda_cor=0 \
  trainer.synaptic_scaling=false trainer.dropout=false \
  "${CELL_ARGS[@]}" "${CONT[@]}" \
  trainer.checkpoint_every=1000 ++trainer.checkpoint_keep=true \
  paths.checkpoint_to="$SNAP_TO" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
