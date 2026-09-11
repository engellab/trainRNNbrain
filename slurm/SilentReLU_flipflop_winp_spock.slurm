#!/bin/bash
# IS THE SILENCE PARTLY AN INITIALISATION ARTEFACT? — W_inp initial per-row norm sweep (SPOCK), T5.
#
# Every network in the project initialises W_inp at std 1/sqrt(N) per entry (row norm 0.039 at
# N=2000): a 10-step +-1 pulse through such a row drives a tau=10 unit by ~0.02, below the recurrent
# noise (0.05). Trained unpenalised networks grow ||W_inp||_F 50x in the first 100k iterations; the
# units that survive are those whose input rows get amplified to norm ~5.4 (single-channel, at every
# k), and the silent units' rows decay to ~0.003. This sweep gives EVERY unit the same input row
# norm s at init and asks whether the network still concentrates onto ~20% of its units.
#
# Grid (24 jobs) = 4 row norms x 2 sizes x 3 seeds, 3-bit flip-flop, h equation, no penalties,
# standard ReLU RNN (model=rnn_relu_standard, model.input_row_norm=$S).
#   S in {0.5, 2, 5, 20}      per-row L2 norm of W_inp at init; total ||W_inp||_F = S*sqrt(N)
#      0.5: above the noise floor (drive ~0.2 per pulse) but far below the unit scale
#      2  : the task's total (~93) spread evenly over all units at N~1000
#      5  : the unit scale every live unit ends at (5.4), on every unit
#      20 : 4x oversupply; the network must shed input weight (weight decay 1e-6 is slow)
#      (the default draw, row norm 0.039, is the existing ksweep; not rerun)
#   N in {500, 1000};  k = 3;  150k iterations (6-9 h; ReLU costs 0.215-0.24 s/iter here)
# Index decode (TID = SLURM_ARRAY_TASK_ID-1): REP=TID%3; N_IDX=(TID/3)%2; S_IDX=TID/6
#
# Read-out (fixed before the data): silent fraction along training (does the early global collapse
# still happen?), participation Hoyer, 1/HHI, the four axes, against the ksweep ReLU baseline at the
# same k, N and iteration. Decision rule in docs/project_trajectory.md (entry of 2026-09-11).
#
# Output: $SAVE_TO/NBitFlipFlop_std_winp/EqType=h_k=3_N=<N>_s=<S>_iters=<IT>/<per-net subfolder>

#SBATCH --job-name=FFwinp
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --array=1-24
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

REPO="${REPO:-$HOME/trainRNNbrain}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2; exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "WARNING: $REPO has uncommitted changes; the commit above does not fully describe this run." >&2
fi

SS=(0.5 2 5 20)
NS=(500 1000)
K=3
IT=${ITERS_OVERRIDE:-150000}

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); N_IDX=$(((TID / 3) % 2)); S_IDX=$((TID / 6))
S=${SS[$S_IDX]}; N=${NS[$N_IDX]}

TAG="std_winp/EqType=h_k=${K}_N=${N}_s=${S}_iters=${IT}"
echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K} N=${N} s=${S} rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/NBitFlipFlop_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=NBitFlipFlop \
  task.n_inputs="$K" task.n_outputs="$K" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" model.input_row_norm="$S" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
