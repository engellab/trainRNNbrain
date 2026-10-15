#!/bin/bash
# B4, THE PLASTICITY TEST: when a trained network learns a new task, do its silent units come back?
# (SPOCK)  Rules: docs/project_trajectory/2026-10-15_1342_open-rules-and-commands-moved-from-agent-notes.md.
#
# The new task is NBitFlipFlopHyper at k = 3: the 3-bit flip-flop's own 3 input channels, and 7
# outputs (every product of the bit states). Only the read-out is new, so a source network's input
# and recurrent weights carry over unchanged and no silent unit is handed a new drive by fresh input
# weights.
#
# EACH TASK IS BUILT FROM ITS SOURCE'S OWN SAVED CONFIG (--config-path/--config-name), as the top-up
# launchers do, so model and trainer settings carry over by construction; the parameter file carries
# no architecture. On top of that config this launcher sets only the task swap and the run's
# bookkeeping:
#   task.taskname / task._target_ / task.n_outputs=7     the new task
#   trainer.max_iter=$ITERS, seed=random                 the budget, a new seed
#   ++trainer.iter_offset=0 ++paths.init_from=null       a 200k source is itself a continuation
#                                                        (iter_offset 75000, init_from its parent);
#                                                        the new task's clock starts at 0, alone
#   ++paths.init_adam=false                              the source's moments fit its own read-out
#   ++paths.transfer_from='<source>'   (MODE=transfer)   copy W_rec, W_inp, bias, y_init; fresh W_out
#   ++paths.transfer_from=null         (MODE=fresh)      untrained network, same config: the control
#   kept snapshots every $CKPT_EVERY into $SNAP_TO, outside the data tree
# The source goes under transfer_from, NOT init_from, so common.warm_start_chain stops at the
# transfer run and the new task's floor is fitted on its own loss (tests/test_transfer_warm_start.py).
#
# SOURCES, one per array task, is a glob of source folders (finished runs, or kept snapshots, which
# hold a config and LastParams), sorted; task i uses the i-th. In MODE=fresh a source supplies only
# its config.
#
# Submit from the pinned checkout (the guard refuses a dirty tree), e.g.
#   R=$HOME/trainRNNbrain_b4; S=$HOME/trainRNNbrain/data/b4_sources
#   SOURCES="$S/c1_snapshots/NBitFlipFlop_c1_snapshots/EqType=h_N=1000_arm=control/s*/it*/" \
#     CELL=src=control_age=young sbatch --chdir=$R --export=ALL --array=1-3 $R/slurm/<this>
#   SOURCES="$S/trained_RNNs/NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control/*/" \
#     CELL=src=control_age=old sbatch ... (same)
#   SOURCES=<young glob> MODE=fresh CELL=src=fresh sbatch ... (same)
# Output: $SAVE_TO/NBitFlipFlopHyper_b4_transfer/EqType=h_N=<N>_<CELL>/<run>/
#         $SNAP_TO/NBitFlipFlopHyper_b4_transfer/EqType=h_N=<N>_<CELL>/s<seed>/it<iters>/
# Cost at N = 1000: ~0.21 s/it on Spock (B5, 13 Oct), so 200k is ~12 h; 48 h allows a slow node.
# At N = 4000 raise --mem and --time, and split 200k into warm-started segments (96 h is Spock's cap).

#SBATCH --job-name=B4xfer
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
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

REPO="${REPO:-$HOME/trainRNNbrain_b4}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"
SNAP_TO="${SNAP_TO:-$HOME/trainRNNbrain/data/b4_snapshots}"
MODE="${MODE:-transfer}"
ITERS="${ITERS:-200000}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
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

mapfile -t ALL < <(ls -d ${SOURCES:?set SOURCES to a glob of source folders} 2>/dev/null | sort)
SRC="${ALL[$((SLURM_ARRAY_TASK_ID - 1))]%/}"
[ -n "$SRC" ] || { echo "FATAL: no source number $SLURM_ARRAY_TASK_ID under $SOURCES" >&2; exit 1; }
CFG=$(ls "$SRC"/*_config.yaml | head -1)
N=$(python -c "import sys; from omegaconf import OmegaConf; print(OmegaConf.load(sys.argv[1]).model.N)" "$CFG")
TAG="b4_transfer/EqType=h_N=${N}_${CELL:?set CELL, e.g. src=control_age=old}"
case "$MODE" in
  transfer) XFER="++paths.transfer_from='$SRC'" ;;
  fresh)    XFER="++paths.transfer_from=null" ;;
  *) echo "FATAL: MODE must be transfer or fresh, got $MODE" >&2; exit 1 ;;
esac
echo "Task ${SLURM_ARRAY_TASK_ID}: $MODE, N=$N, $ITERS iterations, snapshots every $CKPT_EVERY"
echo "  source: $SRC"
echo "  -> $SAVE_TO/NBitFlipFlopHyper_$TAG"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  --config-path "$(dirname "$CFG")" --config-name="$(basename "$CFG" .yaml)" \
  seed="random" \
  task.taskname=NBitFlipFlopHyper \
  task._target_=trainRNNbrain.tasks.TaskNBitFlipFlopHyper.TaskNBitFlipFlopHyper \
  task.n_outputs=7 \
  trainer.max_iter="$ITERS" \
  ++trainer.iter_offset=0 \
  ++paths.init_from=null ++paths.init_adam=false "$XFER" \
  ++trainer.checkpoint_every="$CKPT_EVERY" ++trainer.checkpoint_keep=true "++paths.checkpoint_to=$SNAP_TO" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"$TAG\""
