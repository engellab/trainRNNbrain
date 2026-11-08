#!/bin/bash
# ROUND 19 PILOT (DELLA): can a tanh or a shifted-sigmoid network learn the Maze under a changed learning
# rate or a smaller initial recurrent gain? At the standard settings neither activation learned in
# 2,000 updates (3 seeds each, success 0.00-0.38 and 0.15-0.21 at update 2,000). The rules, the reasons
# for each condition, what counts as a pass and how a passing setting is reported are in
# docs/project_trajectory/*_round19-train-all-rules.md, written and pushed before this launcher was
# submitted. Tracking: ptolmachev/dead-relu-paper#201.
#
# GRID (16 jobs) = 2 activations x 4 conditions x 2 seeds, N = 1,000, 2,000 updates, the Maze suite's
# configuration (task=Maze trainer=trainer_ppo model=rnn_<act>_standard) with only the keys below changed.
#   COND lr0.3     trainer.lr=0.0003            the size rule's rate x 0.3
#   COND lr0.1     trainer.lr=0.0001            the size rule's rate x 0.1
#   COND gain      model.spectral_rad=G_ACT     initial W_rec spectral radius such that radius x max f' = 0.85,
#                                               the ReLU network's RMS gain at the standard radius
#                                               (1.2 x sqrt(1/2)): tanh 0.85 (max f' 1), shifted sigmoid 0.45
#                                               (max f' = 7.5 / 4 = 1.875)
#   COND gain_lr0.3  both keys
#   Index decode (TID = SLURM_ARRAY_TASK_ID - 1): SEED = TID % 2 + 1; COND = (TID / 2) % 4; ACT = TID / 8
#     1-8   tanh             lr0.3 s1 s2, lr0.1 s1 s2, gain s1 s2, gain_lr0.3 s1 s2
#     9-16  sigmoid_shifted  the same
# Seeds 1 and 2 are the seeds of the standard-setting runs (Maze_suite/N=1000_act=<act>/s1, s2), so each
# pilot run has a paired run at the standard setting from the same initial draw (the gain conditions
# rescale that draw: RNN_torch rescales W_rec to the requested radius).
#
#   sbatch --array=1-16 --time=12:00:00 slurm/rl_maze_r19_pilot_della.slurm
# Output: $SAVE_TO/Maze_r19pilot/N=1000_act=<act>_<cond>/s<seed>
#
# FULL RUNS (8 Nov 2026, docs/project_trajectory/*_round19-full-sigmoid-rules.md): the shifted sigmoid at
# initial spectral radius 0.45 passed the pilot at 4,000 updates, and the coordinator approved full runs under
# the stage protocol, with a ReLU control at the same radius. Indices 17-22 (seeds 3-5, new draws, so the
# pilot's seeds 1-2, chosen on their outcome, are not reused), only model.spectral_rad=0.45 changed:
#     17-19  sigmoid_shifted  rad0.45  seeds 3, 4, 5
#     20-22  relu             rad0.45  seeds 3, 4, 5
#   sbatch --array=17-22 --time=12:00:00 slurm/rl_maze_r19_pilot_della.slurm
# Output: $SAVE_TO/Maze_r19full/N=1000_act=<act>_rad0.45/s<seed>; later stages continue them with
# PARENT_ROOT=<SAVE_TO>/Maze_r19full (the continuation block below).
#
#SBATCH --job-name=R19pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=48G
#SBATCH --time=12:00:00
#SBATCH --array=1-16
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

REPO="${REPO:-$HOME/trainRNNbrain_r19}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze}"   # the maze runs' own tree

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

ACTS=(tanh sigmoid_shifted)
GAIN=(0.85 0.45)                    # radius x max f' = 0.85 for tanh (max f' 1) and the shifted sigmoid (1.875)
CONDS=(lr0.3 lr0.1 gain gain_lr0.3)
TID=$((SLURM_ARRAY_TASK_ID - 1))
SEED=$((TID % 2 + 1)); C=$(((TID / 2) % 4)); A=$((TID / 8))
ACT=${ACTS[$A]}; COND=${CONDS[$C]}
case "$COND" in
  lr0.3)      KEYS=(trainer.lr=0.0003) ;;
  lr0.1)      KEYS=(trainer.lr=0.0001) ;;
  gain)       KEYS=(model.spectral_rad="${GAIN[$A]}") ;;
  gain_lr0.3) KEYS=(model.spectral_rad="${GAIN[$A]}" trainer.lr=0.0003) ;;
esac
PREFIX=r19pilot
if [ "$SLURM_ARRAY_TASK_ID" -ge 17 ]; then
  F=$((SLURM_ARRAY_TASK_ID - 17))
  if [ "$F" -ge 6 ]; then
    echo "FATAL: index ${SLURM_ARRAY_TASK_ID} is not in the grid (1-22)." >&2; exit 1
  fi
  SEED=$((F % 3 + 3))
  if [ "$F" -lt 3 ]; then ACT=sigmoid_shifted; else ACT=relu; fi
  COND=rad0.45; KEYS=(model.spectral_rad=0.45); PREFIX=r19full
fi
TAG="${PREFIX}/N=1000_act=${ACT}_${COND}"
# CONTINUATION (7 Nov 2026, docs/project_trajectory/*_round19-pilot-stage2-rules.md): with PARENT_ROOT (the
# folder holding the pilot cells, <SAVE_TO>/Maze_r19pilot) and EXTRA_UPDATES set, the index continues its own
# pilot run from the parent's weights, as the suite launcher's continuation does: INIT_ADAM (default true)
# restores the optimizer state or, false, starts it afresh with the warm-up ramp; the update clock continues
# (trainer.iter_offset=auto); STAGE (default 2) continues <cell> into <cell>_cont, stage k >= 3 continues
# <cell>_cont (k = 3) or <cell>_cont<k-1> into <cell>_cont<k>. The condition's keys are passed again.
#   PARENT_ROOT=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_r19pilot EXTRA_UPDATES=2000 INIT_ADAM=false \
#     sbatch --array=5,6,13,14 --time=12:00:00 slurm/rl_maze_r19_pilot_della.slurm
CONT=()
if [ -n "${PARENT_ROOT:-}" ]; then
  if [ -z "${EXTRA_UPDATES:-}" ]; then
    echo "FATAL: PARENT_ROOT is set but EXTRA_UPDATES (the number of further updates) is not." >&2; exit 1
  fi
  INIT_ADAM="${INIT_ADAM:-true}"
  STAGE="${STAGE:-2}"
  if ! [[ "$STAGE" =~ ^[0-9]+$ ]] || [ "$STAGE" -lt 2 ]; then
    echo "FATAL: STAGE must be an integer >= 2 (stage 1 is the fresh run); got '${STAGE}'." >&2; exit 1
  fi
  stage_suffix() { case "$1" in 1) echo "" ;; 2) echo "_cont" ;; *) echo "_cont$1" ;; esac; }
  PARENT="${PARENT_ROOT}/N=1000_act=${ACT}_${COND}$(stage_suffix $((STAGE - 1)))/s${SEED}"
  if ! ls "${PARENT}"/*Params_Maze.npz >/dev/null 2>&1; then
    echo "FATAL: ${PARENT} has no *Params_Maze.npz to continue from." >&2; exit 1
  fi
  if [ "$INIT_ADAM" = true ] && ! ls "${PARENT}"/*AdamState_Maze.pt >/dev/null 2>&1; then
    echo "FATAL: ${PARENT} has no *AdamState_Maze.pt; continue it with INIT_ADAM=false." >&2; exit 1
  fi
  TAG="${TAG}$(stage_suffix "$STAGE")"
  CONT=(paths.init_from="\"${PARENT}\"" paths.init_adam="$INIT_ADAM" trainer.n_updates="$EXTRA_UPDATES" trainer.iter_offset=auto)
  echo "continuation, stage ${STAGE}: from ${PARENT} (init_adam=${INIT_ADAM}), ${EXTRA_UPDATES} further updates"
fi
# AFTER AN EPISODE ENDS (8 Nov 2026; docs/project_trajectory/*_round19-freeze-default-rules.md): a new run takes
# configs/trainer/trainer_ppo.yaml's freeze_after_end (true since 8 Nov); a continuation keeps its parent's
# setting, read from the parent's saved config (false for every run before 8 Nov, whose config has no key),
# so a chain never changes it.
FZ=()
if [ -n "${PARENT_ROOT:-}" ]; then
  PARENT_FZ=false
  if grep -q "freeze_after_end: true" "${PARENT}"/*config.yaml 2>/dev/null; then PARENT_FZ=true; fi
  FZ=(trainer.freeze_after_end="$PARENT_FZ")
  echo "freeze_after_end=${PARENT_FZ}, the parent's"
fi

OUT="${SAVE_TO}/Maze_${TAG}/s${SEED}"
if ls "$OUT"/*config.yaml >/dev/null 2>&1; then
  echo "FATAL: $OUT already holds a run; refusing to overwrite it." >&2; exit 1
fi
echo "Task ${SLURM_ARRAY_TASK_ID}: act=${ACT} cond=${COND} seed=${SEED} keys=${KEYS[*]} -> ${OUT}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  task=Maze trainer=trainer_ppo seed="$SEED" model="rnn_${ACT}_standard" model.N=1000 \
  paths=paths_DELLA paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\"" "${KEYS[@]}" "${CONT[@]}" "${FZ[@]}"
