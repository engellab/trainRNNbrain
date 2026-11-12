#!/bin/bash
# MAZE LONG TRAINING (DELLA), 12 Nov 2026: the default ReLU Maze networks of the size series continued
# from update 4,000 to 12,000 in four more stages of 2,000 updates (stages 3-6). Rule:
# docs/project_trajectory/2026-11-11_2322_rules-maze-long-training-and-retrain-at-250.md, section 1;
# stage protocol: docs/project_trajectory/2026-11-07_1008_round19-train-all-rules.md.
#
# This file is slurm/rl_maze_suite_della.slurm of commit 1a80f5e with five changes and nothing else:
#   1. indices 26-28 added to the grid: N = 1,000 ReLU seeds 4-6 (the Della grid had no index for them;
#      on Spock they are indices 29-31);
#   2. the job name (MazeLong) and the default array;
#   3. REPO defaults to ~/trainRNNbrain_mazelong, SAVE_TO to .../trained_RNNs_rl_maze_long: a tree of
#      its own, because Della's Maze_suite/N=1000_act=relu holds Della's first uniform series, which are
#      other networks than the Spock runs continued here;
#   4. the launcher refuses any code commit other than 1a80f5e, the code of the stage-2 runs of seeds 4-6
#      (Spock 7398076); stage 2 of seeds 1-3 and of N = 500 ran at 6317f30 (Spock 7397701), whose trainer,
#      run script, model, task and configs are identical to 1a80f5e's;
#   5. a continuation is required (PARENT_ROOT), so no index can start a fresh run here.
#
# GRID used here (the other indices are the suite's and are not submitted):
#   4-6     N=  500  relu  seeds 1-3
#   7-9     N= 1000  relu  seeds 1-3
#   26-28   N= 1000  relu  seeds 4-6
#
# SUBMIT (stage k waits for the same index of stage k-1 with aftercorr):
#   export REPO=$HOME/trainRNNbrain_mazelong SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_long
#   export PARENT_ROOT=$SAVE_TO/Maze_suite EXTRA_UPDATES=2000 INIT_ADAM=false
#   STAGE=3 sbatch --parsable --chdir=$REPO --array=4-9,26-28 slurm/rl_maze_long_della.slurm
#   STAGE=4 sbatch --parsable --chdir=$REPO --array=4-9,26-28 --dependency=aftercorr:<stage 3 id> slurm/rl_maze_long_della.slurm
#   (and so on to STAGE=6)
#
#SBATCH --job-name=MazeLong
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

REPO="${REPO:-$HOME/trainRNNbrain_mazelong}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_long}"   # the maze suite's own tree, a sibling
                                                                     # of the supervised runs' folder, where every
                                                                     # maze run so far is; run_rl.py writes the
                                                                     # standard layout under it

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
PIN=1a80f5ea05d6c823d19bebd17776bfc6ed9373a8
if [ "$(git -C "$REPO" rev-parse HEAD)" != "$PIN" ]; then
  echo "FATAL: $REPO is at $GITSHA, not at $PIN, the code of these networks' earlier stages." >&2; exit 1
fi
if [ -z "${PARENT_ROOT:-}" ]; then
  echo "FATAL: this launcher only continues runs; set PARENT_ROOT, EXTRA_UPDATES, INIT_ADAM and STAGE." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA"

# index 0 unused so the decode reads SLURM_ARRAY_TASK_ID directly
TASK_N=(0 250 250 250 500 500 500 1000 1000 1000 2000 2000 2000 1000 1000 1000 1000 1000 1000 1000 1000 4000 4000 4000 4000 4000 1000 1000 1000)
TASK_ACT=(x relu relu relu relu relu relu relu relu relu relu relu relu tanh tanh sigmoid_shifted sigmoid_shifted softplus_beta25 softplus_beta25 leaky_relu leaky_relu relu relu relu relu relu relu relu relu)
TASK_SEED=(0 1 2 3 1 2 3 1 2 3 1 2 3 1 2 1 2 1 2 1 2 1 2 3 4 5 4 5 6)

i=$SLURM_ARRAY_TASK_ID
N=${TASK_N[$i]}; ACT=${TASK_ACT[$i]}; SEED=${TASK_SEED[$i]}
TAG="suite/N=${N}_act=${ACT}"
echo "Task ${i}: N=${N} model=rnn_${ACT}_standard seed=${SEED} -> ${SAVE_TO}/Maze_${TAG}/s${SEED}"

# CONTINUATION (optional, since 5 Nov 2026). With PARENT_ROOT set to the folder holding the cells to
# continue (the suite's <SAVE_TO>/Maze_suite) and EXTRA_UPDATES to the number of further updates,
# each index continues its cell's run (same N, activation, seed) from the parent's weights
# (paths.init_from) and, with INIT_ADAM=true (the default), its Adam moments (paths.init_adam; the
# warm-up ramp is then not repeated), counting updates on from the parent's last
# (trainer.iter_offset=auto), into the cell <tag>_cont beside the parent, which the readers
# (common.maze_load_runs) join with the parent into one run. A parent written before the AdamState
# file existed (every run of the first uniform series) is continued with INIT_ADAM=false: Adam
# starts afresh and the ramp runs again. Without PARENT_ROOT the extra arguments are empty and the
# run is the fresh one above. The paper's interventions keep their launcher; this one is the suite.
#   PARENT_ROOT=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite EXTRA_UPDATES=2000 INIT_ADAM=false \
#     sbatch --array=10-12 --time=24:00:00 slurm/rl_maze_suite_della.slurm
CONT=()
if [ -n "${PARENT_ROOT:-}" ]; then
  if [ -z "${EXTRA_UPDATES:-}" ]; then
    echo "FATAL: PARENT_ROOT is set but EXTRA_UPDATES (the number of further updates) is not." >&2; exit 1
  fi
  INIT_ADAM="${INIT_ADAM:-true}"
  # STAGES (round 19, 7 Nov 2026; docs/project_trajectory/*_round19-train-all-rules.md). A Maze network is
  # trained in stages of 2,000 updates: stage 1 is the fresh run in the cell <tag>, stage 2 the first
  # continuation in <tag>_cont (every continuation before round 19), stage k >= 3 continues stage k-1
  # (<tag>_cont for k = 3, <tag>_cont<k-1> after) into <tag>_cont<k>. STAGE (default 2) names the stage
  # this submission trains, and PARENT_ROOT is the suite folder holding every stage's cell. The readers
  # (common.maze_load_runs) follow paths.init_from, so a chain of any length is read as one run.
  #   STAGE=3 PARENT_ROOT=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite EXTRA_UPDATES=2000 \
  #     INIT_ADAM=false sbatch --array=22-23 --time=24:00:00 --constraint=gpu80 slurm/rl_maze_suite_della.slurm
  STAGE="${STAGE:-2}"
  if ! [[ "$STAGE" =~ ^[0-9]+$ ]] || [ "$STAGE" -lt 2 ]; then
    echo "FATAL: STAGE must be an integer >= 2 (stage 1 is the fresh run); got '${STAGE}'." >&2; exit 1
  fi
  stage_suffix() { case "$1" in 1) echo "" ;; 2) echo "_cont" ;; *) echo "_cont$1" ;; esac; }
  PARENT="${PARENT_ROOT}/N=${N}_act=${ACT}$(stage_suffix $((STAGE - 1)))/s${SEED}"
  if ! ls "${PARENT}"/*Params_Maze.npz >/dev/null 2>&1; then     # s<seed>_Params_Maze.npz, or
    echo "FATAL: ${PARENT} has no *Params_Maze.npz to continue from." >&2; exit 1   # Params_Maze.npz (runs before 5 Nov)
  fi
  if [ "$INIT_ADAM" = true ] && ! ls "${PARENT}"/*AdamState_Maze.pt >/dev/null 2>&1; then
    echo "FATAL: ${PARENT} has no *AdamState_Maze.pt; continue it with INIT_ADAM=false." >&2; exit 1
  fi
  TAG="${TAG}$(stage_suffix "$STAGE")"
  # never overwrite a run folder: a stage resubmitted after a failure needs its partial folder moved first
  if ls "${SAVE_TO}/Maze_${TAG}/s${SEED}"/*config.yaml >/dev/null 2>&1; then
    echo "FATAL: ${SAVE_TO}/Maze_${TAG}/s${SEED} already holds a run; refusing to overwrite it." >&2; exit 1
  fi
  # the path is quoted for Hydra's override grammar, which refuses an unquoted value holding '=' (the
  # cell name N=<N>_act=<act>), as experiment_tag is below
  CONT=(paths.init_from="\"${PARENT}\"" paths.init_adam="$INIT_ADAM" trainer.n_updates="$EXTRA_UPDATES" trainer.iter_offset=auto)
  echo "continuation, stage ${STAGE}: from ${PARENT} (init_adam=${INIT_ADAM}), ${EXTRA_UPDATES} further updates -> ${SAVE_TO}/Maze_${TAG}/s${SEED}"
fi

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  task=Maze trainer=trainer_ppo seed="$SEED" model="rnn_${ACT}_standard" model.N="$N" \
  paths=paths_DELLA paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\"" "${CONT[@]}"
