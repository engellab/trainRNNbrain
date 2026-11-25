#!/bin/bash
# IS ANOTHER OPTIMIZER A REMEDY? (DELLA), N = 1,000: 3BFF, CDDM and Maze
#
# Rules: docs/project_trajectory/2026-11-25_*_rules-optimizer-remedy.md, committed before any run. The
# commands are those of slurm/independent_init_della.slurm (3BFF, CDDM: the control networks' launcher, so
# a network and its control differ only in the optimizer keys and, on CDDM, in the budget) and of
# slurm/rl_maze_interventions_spock.slurm (Maze: the matched control networks' launcher).
#
# GRID: arm = (i - 1) / 3, seed = (i - 1) % 3; every arm has the control networks' seeds.
#   1-3    control     the config's Adam, weight decay 1e-6 (Maze continuation only; the supervised controls
#                      are the independent-init networks, and are trained here only if those are missing)
#   4-6    nodecay     trainer.weight_decay=0 (Adam; AdamW at decay 0 and at 1e-6 are the same run in float32)
#   7-9    adamw1e-2   trainer.optimizer=adamw trainer.weight_decay=0.01 (torch's AdamW default)
#   10-12  amsgrad     trainer.optimizer=amsgrad
#   13-15  sgdm0.03    trainer.optimizer=sgd trainer.momentum=0.9 trainer.lr=0.03
#   16-18  sgdm0.1     trainer.optimizer=sgd trainer.momentum=0.9 trainer.lr=0.1
#
# Usage (from a clean clone pinned to the submission commit; --chdir is required by the import check):
#   C=--chdir=$HOME/trainRNNbrain_optrem; L=slurm/optimizer_remedy_della.slurm
#   TASK=ff   sbatch $C --export=ALL --job-name=OptRemFF   --array=4-18 --time=10:00:00 $L
#   TASK=cddm sbatch $C --export=ALL --job-name=OptRemCDDM --array=4-12 --time=07:00:00 $L
#   TASK=maze sbatch $C --export=ALL --job-name=OptRemMaze --array=4-9 --time=14:00:00 --constraint="nomig&gpu80" \
#     --mem-per-gpu=48G $L
# Maze stage 2 (updates 2,001-4,000, Adam state kept), one task per finished stage-1 task:
#   TASK=maze EXTRA_UPDATES=2000 PARENT_ROOT=<folder of the arms> sbatch ... --dependency=aftercorr:<stage 1> $L
#
#SBATCH --job-name=OptRem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=32G
#SBATCH --time=10:00:00
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

REPO="${REPO:-$HOME/trainRNNbrain_optrem}"
DATA="${DATA:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data}"
MAZE_TO="${MAZE_TO:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_optrem}"
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

i=$SLURM_ARRAY_TASK_ID
a=$(( (i - 1) / 3 )); k=$(( (i - 1) % 3 ))
ARMS=(control nodecay adamw1e-2 amsgrad sgdm0.03 sgdm0.1)
OVERRIDES=("" "trainer.weight_decay=0" "trainer.optimizer=adamw trainer.weight_decay=0.01" "trainer.optimizer=amsgrad"
           "trainer.optimizer=sgd trainer.momentum=0.9 trainer.lr=0.03"
           "trainer.optimizer=sgd trainer.momentum=0.9 trainer.lr=0.1")
ARM=${ARMS[$a]}
[ -n "$ARM" ] || { echo "FATAL: array index $i is outside 1-18." >&2; exit 1; }
N=${N:-1000}
SWEEP="${SWEEP:-optimizer_remedy}"
COMMON=(trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100
        trainer.track_valid=false trainer.prune_reinit=false trainer.synaptic_scaling=false
        trainer.checkpoint_every=1000 ++trainer.checkpoint_keep=true
        paths.checkpoint_to="$DATA/c1_snapshots" paths.save_to="$DATA/trained_RNNs"
        trainer.max_iter="${ITERS_OVERRIDE:-75000}" model.init_draws=independent)
TAG="${SWEEP}/EqType=h_N=${N}_arm=${ARM}"
CONT=()
case "$TASK" in
  ff)   SEEDS=(1558264694 254798194 254859059)
        ARGS=(+experiment=silent_units_largeN task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3
              model=rnn_relu_standard trainer=trainer_ptrack_freshbatch model.equation_type=h model.gamma=0
              "${COMMON[@]}") ;;
  cddm) SEEDS=(3833537590 1272763220 457853)
        ARGS=(+experiment=silent_units_largeN task=CDDM
              model=rnn_relu_standard trainer=trainer_ptrack_freshbatch model.equation_type=h
              trainer.dropout=false "${COMMON[@]}") ;;
  maze) SEEDS=(1 2 3)
        TAG="${SWEEP}/${ARM}"
        ARGS=(task=Maze trainer=trainer_ppo model=rnn_relu_standard paths=paths_local paths.save_to="$MAZE_TO"
              ${UPDATES_OVERRIDE:+trainer.n_updates=$UPDATES_OVERRIDE})
        if [ -n "${PARENT_ROOT:-}" ]; then
          : "${EXTRA_UPDATES:?PARENT_ROOT needs EXTRA_UPDATES}"
          PARENT="$PARENT_ROOT/$ARM/s${SEEDS[$k]}"
          ls "$PARENT"/*Params_Maze.npz "$PARENT"/*AdamState_Maze.pt >/dev/null \
            || { echo "FATAL: $PARENT cannot be continued." >&2; exit 1; }
          TAG="${TAG}_cont"
          CONT=(paths.init_from="\"$PARENT\"" paths.init_adam=true trainer.n_updates="$EXTRA_UPDATES" trainer.iter_offset=auto)
        fi
        # a checkpoint overwrites the run folder, so a folder that already holds a run is never reused
        if ls "$MAZE_TO/Maze_$TAG/s${SEEDS[$k]}"/*config.yaml >/dev/null 2>&1; then
          echo "FATAL: $MAZE_TO/Maze_$TAG/s${SEEDS[$k]} already holds a run." >&2; exit 1
        fi ;;
  *) echo "FATAL: TASK must be ff, cddm or maze; got '${TASK}'" >&2; exit 1 ;;
esac
SEED=${SEEDS[$k]}
echo "Task ${i}: ${TASK} N=${N} arm=${ARM} seed=${SEED} -> ${TAG}"

# shellcheck disable=SC2086
srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="$SEED" "${ARGS[@]}" model.N="$N" experiment_tag="\"${TAG}\"" ${OVERRIDES[$a]} "${CONT[@]}"
