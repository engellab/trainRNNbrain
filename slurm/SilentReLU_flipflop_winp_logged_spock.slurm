#!/bin/bash
# ONE LOGGED RUN of the T5 sweep: N=1000, k=3, W_inp rows at norm s=5 (the per-unit input scale the
# unpenalised networks converge to), unpenalised, with the silent count PRINTED to the log every
# 500 iterations under three criteria (trainer.log_silent_every). Exists because the sweep's runs
# write their participation trace only on completion, and Pavel asked to watch silencing as it
# happens. Everything else identical to SilentReLU_flipflop_winp_spock.slurm.
#
# Output: $SAVE_TO/NBitFlipFlop_std_winp_logged/EqType=h_k=3_N=1000_s=5_iters=<IT>/<per-net subfolder>
# Watch:  grep '\[silence\]' ~/trainRNNbrain_logs/FFwinpL.<jobid>_1.out

#SBATCH --job-name=FFwinpL
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --array=1
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

SS=(5)
NS=(1000)
K=3
IT=${ITERS_OVERRIDE:-150000}

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=0; N_IDX=0; S_IDX=0
S=${SS[$S_IDX]}; N=${NS[$N_IDX]}

TAG="std_winp_logged/EqType=h_k=${K}_N=${N}_s=${S}_iters=${IT}"
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
  trainer.track_valid=false trainer.log_silent_every=500 \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
