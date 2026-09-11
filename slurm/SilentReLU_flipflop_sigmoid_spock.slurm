#!/bin/bash
# IS SILENCE A PROPERTY OF THE ReLU-FAMILY ACTIVATION? (SPOCK)
#
# Every activation tested so far has a linear positive part and a floor at zero: ReLU, softplus
# (beta=25) and leaky-ReLU. All three show the same concentration of activity onto ~45% of units at
# N=1000 on CDDM (Dale, h equation) - nonzero gradient everywhere does not prevent it. What has never
# been trained WITHOUT penalties is a bounded, nonlinear positive activation. The earlier
# sigmoid_shifted sweeps were all frm+rws under Dale. This is that baseline.
#
# Grid (6 jobs) = 2 sizes x 3 seeds, 3-bit flip-flop, h equation, no penalties, standard RNN,
# activation sigmoid(7.5 (x - 0.3)) (model=rnn_sigmoid_shifted_standard).
#   N in {500, 1000};  k = 3
# Index decode (TID = SLURM_ARRAY_TASK_ID-1): REP=TID%3; N_IDX=TID/3
#
# BUDGET 150k ITERATIONS. Measured ReLU cost at batch 1024 fresh: 0.24 s/iter at N=500, 0.215 at
# N=1000 (k=3); the slow node tail is ~0.3. 150k x 0.3 = 12.5 h, inside the 14 h request Pavel asked
# for. The ReLU ksweep reached its floor region well before 150k at these sizes (matched-level
# read-outs at 30-90k), so the silent fraction is readable; the floor-stability check (fit on [0,T]
# vs [0,T/2]) decides after the fact whether it was enough.
#
# READ-OUT. For a sigmoid unit "silent" cannot mean p < threshold: a unit saturated at a constant
# output passes the participation criterion (q90 > 0) while being functionally dead. Read with the
# modulation criterion (temporal std of the rate vs the population's) alongside the standard one, and
# with the participation Hoyer sparsity, which is threshold-free.
#
# Output: $SAVE_TO/NBitFlipFlop_std_sigmoid/EqType=h_k=3_N=<N>_iters=<IT>/<per-net subfolder>

#SBATCH --job-name=FFsig
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --array=1-6
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

NS=(500 1000)
K=3
IT=${ITERS_OVERRIDE:-150000}

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); N_IDX=$((TID / 3))
N=${NS[$N_IDX]}

TAG="std_sigmoid/EqType=h_k=${K}_N=${N}_iters=${IT}"
echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K} N=${N} rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/NBitFlipFlop_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=NBitFlipFlop \
  task.n_inputs="$K" task.n_outputs="$K" \
  model=rnn_sigmoid_shifted_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
