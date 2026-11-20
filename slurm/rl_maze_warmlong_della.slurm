#!/bin/bash
# MAZE LONG WARM CONTINUATION (DELLA), 18-20 Nov 2026 (Phase B): the nine Maze networks of the warm-start test,
# continued from the end of their warm arm (update 24,000) with the Adam state reloaded at every stage boundary.
# Rules, written before any run: docs/project_trajectory/2026-11-18_0745_rules-maze-long-warm-continuation.md.
#
# This file is slurm/rl_maze_warmtest_della.slurm (main) with these changes and nothing else:
#   1. the job name (MazeWarmLong) and the time limit (23 h, gpu-short; a 5,000-update stage took 12.5-17.7 h
#      at the 5.0-7.1 h per 2,000 updates of the earlier stages);
#   2. outputs go to a new tree, SAVE_TO = .../trained_RNNs_rl_maze_warmlong, in cells named
#      N=<N>_act=relu_warm_cont<stage>; the warm-start test's tree is read only;
#   3. always warm (paths.init_adam=true); ARM is gone, INIT_ADAM if set must be true;
#   4. stages 13 and 14 of EXTRA_UPDATES (5,000) each: stage 13 continues warm stage 12 of the warm-start test
#      (update 24,000), stage 14 continues stage 13 (its recovery cell _warm_cont13r when one exists);
#   5. the parent must end at 24,000 + (STAGE - 13) x EXTRA_UPDATES;
#   6. RECOVER=1 continues a stage that stopped early (wall clock, node failure) from its own partial folder,
#      whose last checkpoint (every 100 updates) holds the parameters and the Adam state, to the stage's end,
#      into the cell N=<N>_act=relu_warm_cont<stage>r. The partial folder is kept as it is; common.maze_load_runs
#      joins parent, partial folder and recovery cell into one record.
# The training command line is unchanged; the trainer is the wp/maze-finite pin of the warm-start test.
#
# GRID (index: N, seed): 4 N=500 s1, 5 N=500 s2, 6 N=500 s3, 7 N=1000 s1, 8 N=1000 s2, 9 N=1000 s3,
#   26 N=1000 s4, 27 N=1000 s5, 28 N=1000 s6.
#
# SUBMIT (slurm/rl_maze_warmlong_submit_della.sh does this after its checks):
#   export EXTRA_UPDATES=5000
#   STAGE=13 sbatch --parsable --chdir=$REPO slurm/rl_maze_warmlong_della.slurm
#   STAGE=14 sbatch --parsable --chdir=$REPO --dependency=aftercorr:<stage 13 id> slurm/rl_maze_warmlong_della.slurm
# RECOVER one stopped task (after its job has ended; then resubmit the next stage of that index by hand):
#   STAGE=<s> RECOVER=1 sbatch --parsable --chdir=$REPO --array=<index> slurm/rl_maze_warmlong_della.slurm
#
#SBATCH --job-name=MazeWarmLong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=48G
#SBATCH --time=23:00:00
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
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_warmlong}"                  # this phase's own tree
WARMTEST_ROOT="${WARMTEST_ROOT:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_warmtest/Maze_suite}"   # read only

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
  echo "FATAL: $REPO is at $GITSHA, not at $PIN, the wp/maze-finite trainer of the warm-start test." >&2; exit 1
fi
if [ -z "${EXTRA_UPDATES:-}" ] || [ -z "${STAGE:-}" ]; then
  echo "FATAL: set EXTRA_UPDATES (5000) and STAGE (13 or 14)." >&2; exit 1
fi
case "$STAGE" in
  13|14) ;;
  *) echo "FATAL: STAGE=$STAGE is not 13 or 14." >&2; exit 1 ;;
esac
if [ -n "${INIT_ADAM:-}" ] && [ "$INIT_ADAM" != true ]; then
  echo "FATAL: INIT_ADAM=$INIT_ADAM; every stage of this phase is warm (init_adam=true); unset it." >&2; exit 1
fi
INIT_ADAM=true
RECOVER="${RECOVER:-0}"
echo "code: $PKG @ $GITSHA"

# index 0 unused so the decode reads SLURM_ARRAY_TASK_ID directly
TASK_N=(0 250 250 250 500 500 500 1000 1000 1000 2000 2000 2000 1000 1000 1000 1000 1000 1000 1000 1000 4000 4000 4000 4000 4000 1000 1000 1000)
TASK_ACT=(x relu relu relu relu relu relu relu relu relu relu relu relu tanh tanh sigmoid_shifted sigmoid_shifted softplus_beta25 softplus_beta25 leaky_relu leaky_relu relu relu relu relu relu relu relu relu)
TASK_SEED=(0 1 2 3 1 2 3 1 2 3 1 2 3 1 2 1 2 1 2 1 2 1 2 3 4 5 4 5 6)

i=$SLURM_ARRAY_TASK_ID
case "$i" in
  4|5|6|7|8|9|26|27|28) ;;
  *) echo "FATAL: index $i is not one of the nine networks (4-9, 26-28)." >&2; exit 1 ;;
esac
N=${TASK_N[$i]}; ACT=${TASK_ACT[$i]}; SEED=${TASK_SEED[$i]}
CELL="N=${N}_act=${ACT}_warm_cont"
STAGE_END=$((24000 + (STAGE - 12) * EXTRA_UPDATES))
LAST_OF() { python -c "import glob, json, sys; print(json.load(open(sorted(glob.glob(sys.argv[1] + '/*Progress.json'))[0]))[-1]['update'])" "$1"; }
if [ "$RECOVER" = 1 ]; then
  PARENT="${SAVE_TO}/Maze_suite/${CELL}${STAGE}/s${SEED}"
  TAG="suite/${CELL}${STAGE}r"
else
  if [ "$STAGE" -eq 13 ]; then
    PARENT="${WARMTEST_ROOT}/${CELL}12/s${SEED}"
  elif ls "${SAVE_TO}/Maze_suite/${CELL}$((STAGE - 1))r/s${SEED}"/*config.yaml >/dev/null 2>&1; then
    PARENT="${SAVE_TO}/Maze_suite/${CELL}$((STAGE - 1))r/s${SEED}"
  else
    PARENT="${SAVE_TO}/Maze_suite/${CELL}$((STAGE - 1))/s${SEED}"
  fi
  TAG="suite/${CELL}${STAGE}"
fi
echo "Task ${i}: N=${N} model=rnn_${ACT}_standard seed=${SEED} warm stage ${STAGE} (recover ${RECOVER}) -> ${SAVE_TO}/Maze_${TAG}/s${SEED}"

if ! ls "${PARENT}"/*Params_Maze.npz >/dev/null 2>&1 || ! ls "${PARENT}"/*AdamState_Maze.pt >/dev/null 2>&1; then
  echo "FATAL: ${PARENT} lacks *Params_Maze.npz or *AdamState_Maze.pt; a warm stage needs both." >&2; exit 1
fi
LAST=$(LAST_OF "$PARENT")
if [ "$RECOVER" = 1 ]; then
  # a partial folder of this stage: its last checkpoint lies strictly inside the stage
  if [ "$LAST" -le $((STAGE_END - EXTRA_UPDATES)) ] || [ "$LAST" -ge "$STAGE_END" ]; then
    echo "FATAL: ${PARENT} ends at update ${LAST}, not inside stage ${STAGE} ($((STAGE_END - EXTRA_UPDATES + 1))-$((STAGE_END - 1)))." >&2; exit 1
  fi
  N_UPD=$((STAGE_END - LAST))
else
  if [ "$LAST" != $((STAGE_END - EXTRA_UPDATES)) ]; then
    echo "FATAL: ${PARENT} ends at update ${LAST}, not $((STAGE_END - EXTRA_UPDATES)); it stopped early or is the wrong parent." >&2; exit 1
  fi
  N_UPD=$EXTRA_UPDATES
fi
# never overwrite a run folder
if ls "${SAVE_TO}/Maze_${TAG}/s${SEED}"/*config.yaml >/dev/null 2>&1; then
  echo "FATAL: ${SAVE_TO}/Maze_${TAG}/s${SEED} already holds a run; refusing to overwrite it." >&2; exit 1
fi
# the path is quoted for Hydra's override grammar, which refuses an unquoted value holding '=' (the
# cell name N=<N>_act=<act>), as experiment_tag is below
CONT=(paths.init_from="\"${PARENT}\"" paths.init_adam="$INIT_ADAM" trainer.n_updates="$N_UPD" trainer.iter_offset=auto)
echo "continuation, warm stage ${STAGE}: from ${PARENT} (update ${LAST}, init_adam=${INIT_ADAM}), ${N_UPD} further updates to ${STAGE_END} -> ${SAVE_TO}/Maze_${TAG}/s${SEED}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  task=Maze trainer=trainer_ppo seed="$SEED" model="rnn_${ACT}_standard" model.N="$N" \
  paths=paths_DELLA paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\"" "${CONT[@]}"
