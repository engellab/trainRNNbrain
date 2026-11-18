#!/bin/bash
# MAZE WARM-START TEST (DELLA), 17-18 Nov 2026: does a warm start (Adam state reloaded) remove the disruption at
# Maze stage boundaries? Rules, written before any run:
# docs/project_trajectory/2026-11-17_2249_rules-maze-warm-start-test.md. Local continuity check, launcher
# checks and submission: the notebook entry written with this launcher (2026-11-18_*_maze-warm-start-test-submitted.md).
#
# This file is slurm/rl_maze_resume_della.slurm (main) with these changes and nothing else:
#   1. the job name (MazeWarm) and the default array (the nine networks at update 20,000: 4-9, 26-28);
#   2. outputs go to a new tree, SAVE_TO = .../trained_RNNs_rl_maze_warmtest, in cells named
#      N=<N>_act=relu_<arm>_cont<stage>, so nothing existing is written;
#   3. ARM=warm|cold replaces INIT_ADAM (warm: paths.init_adam=true, the parent's Adam state including its step
#      count is loaded and the warm-up ramp is skipped; cold: paths.init_adam=false, fresh moments and the
#      50-update ramp, the protocol used so far); INIT_ADAM, if set, must agree with ARM;
#   4. explicit parents per index instead of FIRST_STAGE: stage 11 continues the network's update-20,000 stage
#      (the long tree for the four never stopped, the resumed tree for the five resumed, both read only); warm
#      stage 12 continues warm stage 11 from SAVE_TO; the cold arm has stage 11 only;
#   5. the parent's last recorded update must be 20,000 + (STAGE - 11) x EXTRA_UPDATES, so a stage never
#      continues a parent that stopped early.
# The training command line is unchanged; the trainer is the wp/maze-finite pin of the resumed networks.
#
# GRID used here (index: N, seed, parent at update 20,000):
#   4  N=500  s1  resumed  N=500_act=relu_resume_cont10/s1
#   5  N=500  s2  resumed  N=500_act=relu_resume_cont10/s2
#   6  N=500  s3  resumed  N=500_act=relu_resume_cont10/s3
#   7  N=1000 s1  long     N=1000_act=relu_cont10/s1
#   8  N=1000 s2  resumed  N=1000_act=relu_resume_cont10/s2
#   9  N=1000 s3  long     N=1000_act=relu_cont10/s3
#   26 N=1000 s4  long     N=1000_act=relu_cont10/s4
#   27 N=1000 s5  long     N=1000_act=relu_cont10/s5
#   28 N=1000 s6  resumed  N=1000_act=relu_resume_cont10/s6
#
# SUBMIT (slurm/rl_maze_warmtest_submit_della.sh does this after its checks):
#   export EXTRA_UPDATES=2000
#   ARM=warm STAGE=11 sbatch --parsable --chdir=$REPO slurm/rl_maze_warmtest_della.slurm
#   ARM=warm STAGE=12 sbatch --parsable --chdir=$REPO --dependency=aftercorr:<warm stage 11 id> slurm/rl_maze_warmtest_della.slurm
#   ARM=cold STAGE=11 sbatch --parsable --chdir=$REPO slurm/rl_maze_warmtest_della.slurm
#
#SBATCH --job-name=MazeWarm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=48G
#SBATCH --time=12:00:00
#SBATCH --array=4-9,26-28
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
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_warmtest}"   # the warm-start test's own tree
LONG_ROOT="${LONG_ROOT:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_long/Maze_suite}"              # read only
RESUMED_ROOT="${RESUMED_ROOT:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_long_resumed/Maze_suite}"   # read only

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
if [ -z "${EXTRA_UPDATES:-}" ] || [ -z "${ARM:-}" ] || [ -z "${STAGE:-}" ]; then
  echo "FATAL: set EXTRA_UPDATES, ARM (warm or cold) and STAGE (11, or 12 for warm)." >&2; exit 1
fi
case "$ARM:$STAGE" in
  warm:11|warm:12) ARM_ADAM=true ;;
  cold:11) ARM_ADAM=false ;;
  *) echo "FATAL: ARM=$ARM STAGE=$STAGE is not one of warm 11, warm 12, cold 11." >&2; exit 1 ;;
esac
if [ -n "${INIT_ADAM:-}" ] && [ "$INIT_ADAM" != "$ARM_ADAM" ]; then
  echo "FATAL: INIT_ADAM=$INIT_ADAM contradicts ARM=$ARM (init_adam=$ARM_ADAM); unset it." >&2; exit 1
fi
INIT_ADAM=$ARM_ADAM
echo "code: $PKG @ $GITSHA"

# index 0 unused so the decode reads SLURM_ARRAY_TASK_ID directly
TASK_N=(0 250 250 250 500 500 500 1000 1000 1000 2000 2000 2000 1000 1000 1000 1000 1000 1000 1000 1000 4000 4000 4000 4000 4000 1000 1000 1000)
TASK_ACT=(x relu relu relu relu relu relu relu relu relu relu relu relu tanh tanh sigmoid_shifted sigmoid_shifted softplus_beta25 softplus_beta25 leaky_relu leaky_relu relu relu relu relu relu relu relu relu)
TASK_SEED=(0 1 2 3 1 2 3 1 2 3 1 2 3 1 2 1 2 1 2 1 2 1 2 3 4 5 4 5 6)

i=$SLURM_ARRAY_TASK_ID
case "$i" in
  4|5|6|8|28) PARENT_ROOT=$RESUMED_ROOT; PARENT_SUFFIX=_resume_cont10 ;;
  7|9|26|27)  PARENT_ROOT=$LONG_ROOT;    PARENT_SUFFIX=_cont10 ;;
  *) echo "FATAL: index $i is not one of the nine networks at update 20,000 (4-9, 26-28)." >&2; exit 1 ;;
esac
N=${TASK_N[$i]}; ACT=${TASK_ACT[$i]}; SEED=${TASK_SEED[$i]}
if [ "$STAGE" -eq 11 ]; then
  PARENT="${PARENT_ROOT}/N=${N}_act=${ACT}${PARENT_SUFFIX}/s${SEED}"
else
  PARENT="${SAVE_TO}/Maze_suite/N=${N}_act=${ACT}_${ARM}_cont$((STAGE - 1))/s${SEED}"
fi
TAG="suite/N=${N}_act=${ACT}_${ARM}_cont${STAGE}"
echo "Task ${i}: N=${N} model=rnn_${ACT}_standard seed=${SEED} arm ${ARM} stage ${STAGE} -> ${SAVE_TO}/Maze_${TAG}/s${SEED}"

if ! ls "${PARENT}"/*Params_Maze.npz >/dev/null 2>&1; then
  echo "FATAL: ${PARENT} has no *Params_Maze.npz to continue from." >&2; exit 1
fi
if [ "$INIT_ADAM" = true ] && ! ls "${PARENT}"/*AdamState_Maze.pt >/dev/null 2>&1; then
  echo "FATAL: ${PARENT} has no *AdamState_Maze.pt; the warm arm needs it." >&2; exit 1
fi
WANT_LAST=$((20000 + (STAGE - 11) * EXTRA_UPDATES))
LAST=$(python -c "import glob, json, sys; print(json.load(open(sorted(glob.glob(sys.argv[1] + '/*Progress.json'))[0]))[-1]['update'])" "$PARENT")
if [ "$LAST" != "$WANT_LAST" ]; then
  echo "FATAL: ${PARENT} ends at update ${LAST}, not ${WANT_LAST}; it stopped early or is the wrong parent." >&2; exit 1
fi
# never overwrite a run folder: a stage resubmitted after a failure needs its partial folder moved first
if ls "${SAVE_TO}/Maze_${TAG}/s${SEED}"/*config.yaml >/dev/null 2>&1; then
  echo "FATAL: ${SAVE_TO}/Maze_${TAG}/s${SEED} already holds a run; refusing to overwrite it." >&2; exit 1
fi
# the path is quoted for Hydra's override grammar, which refuses an unquoted value holding '=' (the
# cell name N=<N>_act=<act>), as experiment_tag is below
CONT=(paths.init_from="\"${PARENT}\"" paths.init_adam="$INIT_ADAM" trainer.n_updates="$EXTRA_UPDATES" trainer.iter_offset=auto)
echo "continuation, ${ARM} stage ${STAGE}: from ${PARENT} (update ${LAST}, init_adam=${INIT_ADAM}), ${EXTRA_UPDATES} further updates -> ${SAVE_TO}/Maze_${TAG}/s${SEED}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  task=Maze trainer=trainer_ppo seed="$SEED" model="rnn_${ACT}_standard" model.N="$N" \
  paths=paths_DELLA paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\"" "${CONT[@]}"
