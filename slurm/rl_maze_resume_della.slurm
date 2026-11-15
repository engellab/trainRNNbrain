#!/bin/bash
# MAZE RESUMED NETWORKS (DELLA), 14 Nov 2026: the three default ReLU Maze networks whose long training
# stopped with "the rollout was not finite 6 times in a row" (Della 17379992_8, 17379994_28,
# 17379995_5), resumed from the end of their last complete stage to update 20,000 in stages of 2,000
# updates, with the trainer of branch wp/maze-finite (1a80f5e plus the recovery from non-finite values
# only). Rules: docs/project_trajectory/2026-11-14_1107_rules-maze-flat-return-by-stage-and-stopped-networks.md;
# fix, tests and submission: docs/project_trajectory/2026-11-14_1605_maze-nonfinite-recovery.md.
#
# This file is slurm/rl_maze_long_della.slurm (main) with these changes and nothing else:
#   1. the job name (MazeResume) and the default array (5, 8, 28);
#   2. REPO defaults to ~/trainRNNbrain_mazefinite and the pin is the wp/maze-finite commit below;
#   3. outputs go to a new tree, SAVE_TO = .../trained_RNNs_rl_maze_long_resumed, in cells named
#      N=<N>_act=<act>_resume_cont<k>, so they collide neither with the stopped stages' partial folders
#      (e.g. N=1000_act=relu_cont3/s2, kept as they are) on Della nor with any folder on the Mac;
#   4. the parent: the network's first resumed stage (FIRST_STAGE below) reads the last complete stage
#      from the long-training tree OLD_ROOT (read only); every later stage reads the previous resumed
#      stage from SAVE_TO. PARENT_ROOT is replaced by OLD_ROOT;
#   5. only indices 5, 8 and 28, only stages FIRST_STAGE..10;
#      index 6 (N = 500 seed 3, stopped in stage 7 at update 12,400, job 17546732_6) added 15 Nov from stage 7.
# The training command line is unchanged; the resumed configs record the new trainer keys
# (trainer.rollout_tries, trainer.snapshot_keep) from configs/trainer/trainer_ppo.yaml of the pin.
#
# GRID used here:
#   5    N=  500  relu  seed 2   stopped in stage 6  -> stages 6-10 (from _cont5, update 10,000)
#   8    N= 1000  relu  seed 2   stopped in stage 3  -> stages 3-10 (from _cont,  update 4,000)
#   28   N= 1000  relu  seed 6   stopped in stage 5  -> stages 5-10 (from _cont4, update 8,000)
#
# SUBMIT one chain per network, each stage a one-index array waiting with aftercorr on the previous
# stage of the same network (the commands are in the notebook entry above):
#   export REPO=$HOME/trainRNNbrain_mazefinite SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_long_resumed
#   export OLD_ROOT=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_long/Maze_suite EXTRA_UPDATES=2000 INIT_ADAM=false
#   STAGE=3 sbatch --parsable --chdir=$REPO --array=8 slurm/rl_maze_resume_della.slurm
#   STAGE=4 sbatch --parsable --chdir=$REPO --array=8 --dependency=aftercorr:<stage 3 id> slurm/rl_maze_resume_della.slurm
#
#SBATCH --job-name=MazeResume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=48G
#SBATCH --time=12:00:00
#SBATCH --array=5,8,28
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

REPO="${REPO:-$HOME/trainRNNbrain_mazefinite}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_long_resumed}"   # the resumed stages' own tree
OLD_ROOT="${OLD_ROOT:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_long/Maze_suite}"   # the stages before the stop, read only

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
PIN=8c6b36e9ea20c449aad550ac1b544d318b8c0631
if [ "$(git -C "$REPO" rev-parse HEAD)" != "$PIN" ]; then
  echo "FATAL: $REPO is at $GITSHA, not at $PIN, the wp/maze-finite trainer of the resumed networks." >&2; exit 1
fi
if [ -z "${EXTRA_UPDATES:-}" ] || [ -z "${INIT_ADAM:-}" ] || [ -z "${STAGE:-}" ]; then
  echo "FATAL: set EXTRA_UPDATES, INIT_ADAM and STAGE." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA"

# index 0 unused so the decode reads SLURM_ARRAY_TASK_ID directly
TASK_N=(0 250 250 250 500 500 500 1000 1000 1000 2000 2000 2000 1000 1000 1000 1000 1000 1000 1000 1000 4000 4000 4000 4000 4000 1000 1000 1000)
TASK_ACT=(x relu relu relu relu relu relu relu relu relu relu relu relu tanh tanh sigmoid_shifted sigmoid_shifted softplus_beta25 softplus_beta25 leaky_relu leaky_relu relu relu relu relu relu relu relu relu)
TASK_SEED=(0 1 2 3 1 2 3 1 2 3 1 2 3 1 2 1 2 1 2 1 2 1 2 3 4 5 4 5 6)

i=$SLURM_ARRAY_TASK_ID
case "$i" in 5) FIRST_STAGE=6 ;; 6) FIRST_STAGE=7 ;; 8) FIRST_STAGE=3 ;; 28) FIRST_STAGE=5 ;;
  *) echo "FATAL: index $i is not one of the resumed networks (5, 6, 8, 28)." >&2; exit 1 ;; esac
N=${TASK_N[$i]}; ACT=${TASK_ACT[$i]}; SEED=${TASK_SEED[$i]}
if ! [[ "$STAGE" =~ ^[0-9]+$ ]] || [ "$STAGE" -lt "$FIRST_STAGE" ] || [ "$STAGE" -gt 10 ]; then
  echo "FATAL: STAGE must be ${FIRST_STAGE}-10 for index $i; got '${STAGE}'." >&2; exit 1
fi
stage_suffix() { case "$1" in 1) echo "" ;; 2) echo "_cont" ;; *) echo "_cont$1" ;; esac; }
if [ "$STAGE" -eq "$FIRST_STAGE" ]; then
  PARENT="${OLD_ROOT}/N=${N}_act=${ACT}$(stage_suffix $((STAGE - 1)))/s${SEED}"
else
  PARENT="${SAVE_TO}/Maze_suite/N=${N}_act=${ACT}_resume$(stage_suffix $((STAGE - 1)))/s${SEED}"
fi
TAG="suite/N=${N}_act=${ACT}_resume$(stage_suffix "$STAGE")"
echo "Task ${i}: N=${N} model=rnn_${ACT}_standard seed=${SEED} stage ${STAGE} -> ${SAVE_TO}/Maze_${TAG}/s${SEED}"

if ! ls "${PARENT}"/*Params_Maze.npz >/dev/null 2>&1; then
  echo "FATAL: ${PARENT} has no *Params_Maze.npz to continue from." >&2; exit 1
fi
if [ "$INIT_ADAM" = true ] && ! ls "${PARENT}"/*AdamState_Maze.pt >/dev/null 2>&1; then
  echo "FATAL: ${PARENT} has no *AdamState_Maze.pt; continue it with INIT_ADAM=false." >&2; exit 1
fi
# never overwrite a run folder: a stage resubmitted after a failure needs its partial folder moved first
if ls "${SAVE_TO}/Maze_${TAG}/s${SEED}"/*config.yaml >/dev/null 2>&1; then
  echo "FATAL: ${SAVE_TO}/Maze_${TAG}/s${SEED} already holds a run; refusing to overwrite it." >&2; exit 1
fi
# the path is quoted for Hydra's override grammar, which refuses an unquoted value holding '=' (the
# cell name N=<N>_act=<act>), as experiment_tag is below
CONT=(paths.init_from="\"${PARENT}\"" paths.init_adam="$INIT_ADAM" trainer.n_updates="$EXTRA_UPDATES" trainer.iter_offset=auto)
echo "continuation, stage ${STAGE}: from ${PARENT} (init_adam=${INIT_ADAM}), ${EXTRA_UPDATES} further updates -> ${SAVE_TO}/Maze_${TAG}/s${SEED}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  task=Maze trainer=trainer_ppo seed="$SEED" model="rnn_${ACT}_standard" model.N="$N" \
  paths=paths_DELLA paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\"" "${CONT[@]}"
