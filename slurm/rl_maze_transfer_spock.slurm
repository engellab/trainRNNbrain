#!/bin/bash
# THE MAZE TRANSFER TEST (SPOCK), G17 of the 7 Nov 2026 audit: the counterpart of B4 (the 3-bit flip-flop
# moved to the 3-bit hyper flip-flop, slurm/SilentReLU_b4_transfer_della.slurm). A network trained on
# the first maze (configs/task/Maze.yaml) moves to a second maze of the same size and statistics
# (configs/task/Maze_transfer.yaml): its recurrent and input weights and y_init are kept, the read-out
# W_out is this run's own fresh draw, Adam starts afresh with its warm-up ramp, and the update clock
# starts at 0 (run_rl.py, paths.transfer_from), as B4 does on the flip-flop. The control is an
# untrained network of the same size trained on the second maze (MODE=fresh). Rules and read-out:
# docs/project_trajectory/2026-11-07_*_round18-training-rules.md.
#
# Everything except the maze is the uniform series' configuration (slurm/rl_maze_suite_spock.slurm:
# task, trainer and model configs as they are), so the launcher sets only the swept keys:
#   task=Maze_transfer                                  the second maze
#   seed                                                transfer: the source's own seed (one run per
#                                                       source, the folder s<seed> names it); fresh:
#                                                       the i-th of FRESH_SEEDS
#   model=rnn_<act>_standard model.N                    the source's activation and size (read from its
#                                                       saved config); fresh: ACT and N
#   paths.transfer_from='"<source>"'  (MODE=transfer)   the source folder, in literal double quotes
#                                                       for Hydra's override grammar (it holds '=')
# SOURCES is a space-separated list of finished maze run folders (either layout), in order; array
# task i uses the i-th. A source may not exist yet at submission (chained with
# --dependency=aftercorr on the job that trains it); it is looked up when the task starts.
#   R=$HOME/trainRNNbrain_r18; M=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite/N=1000_act=relu
#   MODE=transfer SOURCES="$M/s1 $M/s2" REPO=$R sbatch --chdir=$R --export=ALL --array=1-2 $R/slurm/<this>
#   MODE=fresh FRESH_SEEDS="1 2 3" REPO=$R sbatch --chdir=$R --export=ALL --array=1-3 $R/slurm/<this>
# Output: $SAVE_TO/Maze_transfer/N=<N>_src=old/s<seed>/ (MODE=transfer) and
#         $SAVE_TO/Maze_transfer/N=<N>_src=fresh/s<seed>/ (MODE=fresh), the standard run layout.
# Cost at N = 1,000: 5.5-5.8 h per run on Spock (the suite's 7383494_7-9); 12 h allows a slow card.

#SBATCH --job-name=MazeXfer
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
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

REPO="${REPO:-$HOME/trainRNNbrain_r18}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze}"
MODE="${MODE:-transfer}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO); aborting." >&2; exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
if [ -z "$GITSHA" ]; then
  echo "FATAL: $REPO is not a git checkout; the provenance of this run cannot be recorded." >&2; exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA"

i=$SLURM_ARRAY_TASK_ID
case "$MODE" in
  transfer)
    read -r -a ALL <<< "${SOURCES:?set SOURCES to a space-separated list of source run folders}"
    SRC="${ALL[$((i - 1))]%/}"
    [ -n "$SRC" ] || { echo "FATAL: no source number $i in SOURCES" >&2; exit 1; }
    CFG=$(ls "$SRC"/*config.yaml 2>/dev/null | head -1 || true)   # an absent source must reach the FATAL below, not set -e
    if [ -z "$CFG" ] || ! ls "$SRC"/*Params_Maze.npz >/dev/null 2>&1; then
      echo "FATAL: $SRC has no *config.yaml and *Params_Maze.npz to transfer from." >&2; exit 1
    fi
    read -r SEED N ACT < <(python -c "import sys; from omegaconf import OmegaConf; c = OmegaConf.load(sys.argv[1]); print(c.seed, c.model.N, c.model.activation_args.name)" "$CFG")
    XFER=(paths.transfer_from="\"${SRC}\"")
    CELL="src=old"
    ;;
  fresh)
    read -r -a SEEDS <<< "${FRESH_SEEDS:-1 2 3}"
    SEED="${SEEDS[$((i - 1))]}"
    [ -n "$SEED" ] || { echo "FATAL: no seed number $i in FRESH_SEEDS" >&2; exit 1; }
    N="${N:-1000}"; ACT="${ACT:-relu}"
    XFER=()
    CELL="src=fresh"
    ;;
  *) echo "FATAL: MODE must be transfer or fresh, got $MODE" >&2; exit 1 ;;
esac
TAG="transfer/N=${N}_${CELL}"
echo "Task ${i}: $MODE, N=${N} model=rnn_${ACT}_standard seed=${SEED} -> ${SAVE_TO}/Maze_${TAG}/s${SEED}"
[ "$MODE" = transfer ] && echo "  source: $SRC"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  task=Maze_transfer trainer=trainer_ppo seed="$SEED" model="rnn_${ACT}_standard" model.N="$N" \
  paths=paths_local paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\"" "${XFER[@]}"
