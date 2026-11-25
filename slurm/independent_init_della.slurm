#!/bin/bash
# THE INITIAL DRAWS, VARIED (DELLA), N = 1,000: does the paper's active-unit count depend on every
# weight matrix starting from the same random sequence?
#
# Rules: docs/project_trajectory/2026-11-25_*_rules-independent-init.md, committed before any run.
# model.init_draws (RNN_torch.sparse): "shared" restarts the generator before each of W_rec, W_inp and
# W_out; "independent" draws the three one after another from one stream. W_rec takes the first draws in
# both modes, so with the same seed only the initial W_inp and W_out differ.
#
# GRID (9 jobs), three networks per task, each with the seed of one of the paper's default networks:
#   1-3  3BFF  independent  75,000 iterations (then 200,000, below)  the paper's 3BFF networks: shared
#   4-6  CDDM  independent  100,000 iterations                       the paper's CDDM networks: shared
#   7-9  Maze  SHARED       2,000 updates                            the paper's Maze networks: independent
# The commands are those of the launchers that trained the paper's networks, with the seed and
# model.init_draws set: SilentReLU_ff_revive_della.slurm (3BFF, arm none), SilentReLU_paper_grid_della.slurm
# (CDDM, arm control), rl_maze_suite_della.slurm (Maze, N = 1000 relu).
#
# Usage (from a clean clone pinned to the submission commit; --chdir is required by the import check):
#   C=--chdir=$HOME/trainRNNbrain_indinit; L=slurm/independent_init_della.slurm
#   sbatch $C --array=1-3 --time=10:00:00 $L
#   sbatch $C --array=4-6 --time=11:00:00 $L
#   sbatch $C --array=7-9 --time=12:00:00 --mem-per-gpu=48G $L
# 3BFF to 200,000 (the floor crossing), only if the GPU-hour rule of the rules entry allows it:
#   REPO=$HOME/trainRNNbrain_indinit KEEP=1 \
#     PARENTS="/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs/NBitFlipFlop_independent_init/EqType=h_N=1000_draws=independent/*/" \
#     sbatch $C --export=ALL --constraint=gpu80 --time=17:00:00 --array=1-3 slurm/SilentReLU_ff_topup200k_della.slurm
#
#SBATCH --job-name=IndInit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem-per-gpu=32G
#SBATCH --time=11:00:00
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

REPO="${REPO:-$HOME/trainRNNbrain_indinit}"
DATA="${DATA:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data}"
SWEEP="${SWEEP:-independent_init}"
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
TASKS=(x ff ff ff cddm cddm cddm maze maze maze)
# the paper's networks' seeds: the first runs of the 3BFF chains, the CDDM controls, the Maze suite
SEEDS=(x 1558264694 254798194 254859059 3833537590 1272763220 457853 1 2 3)
TASK=${TASKS[$i]}; SEED=${SEEDS[$i]}
[ -n "$TASK" ] && [ "$TASK" != x ] || { echo "FATAL: array index $i is outside 1-9." >&2; exit 1; }
N=${N:-1000}
COMMON=(trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100
        trainer.track_valid=false trainer.prune_reinit=false trainer.synaptic_scaling=false
        trainer.checkpoint_every=1000 ++trainer.checkpoint_keep=true
        paths.checkpoint_to="$DATA/c1_snapshots" paths.save_to="$DATA/trained_RNNs")
case "$TASK" in
  ff)   DRAWS=${DRAWS:-independent}
        ARGS=(+experiment=silent_units_largeN task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3
              model=rnn_relu_standard trainer=trainer_ptrack_freshbatch model.equation_type=h model.gamma=0
              trainer.max_iter="${ITERS_OVERRIDE:-75000}" "${COMMON[@]}") ;;
  cddm) DRAWS=${DRAWS:-independent}
        ARGS=(+experiment=silent_units_largeN task=CDDM
              model=rnn_relu_standard trainer=trainer_ptrack_freshbatch model.equation_type=h
              trainer.max_iter="${ITERS_OVERRIDE:-100000}" trainer.dropout=false "${COMMON[@]}") ;;
  maze) DRAWS=${DRAWS:-shared}
        ARGS=(task=Maze trainer=trainer_ppo model=rnn_relu_standard paths=paths_DELLA
              paths.save_to="$DATA/trained_RNNs_rl_maze" trainer.checkpoint_keep=true) ;;
esac
# the N=/act= form is the Maze suite's cell name; the supervised cells follow the paper grid's
if [ "$TASK" = maze ]; then TAG="${SWEEP}/N=${N}_act=relu_draws=${DRAWS}"; else TAG="${SWEEP}/EqType=h_N=${N}_draws=${DRAWS}"; fi
echo "Task ${i}: ${TASK} N=${N} seed=${SEED} init_draws=${DRAWS} -> ${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="$SEED" "${ARGS[@]}" model.N="$N" model.init_draws="$DRAWS" experiment_tag="\"${TAG}\""
