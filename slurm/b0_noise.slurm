#!/bin/bash
# Noise control: ReLU, gamma=0, NO penalties (Lrws=0, Lfr=0), N=1000, sweeping the recurrent noise
# sigma_rec in {0, 0.01, 0.05, 0.1} (0.05 is the default used everywhere else). Tests whether the
# low-activity / silent population is driven by the injected training noise.
#
# Grid (40 jobs) = 2 eq x 4 sigma_rec x 5 nets, N=1000, none penalty.
#   TID = SLURM_ARRAY_TASK_ID-1 (0..39):  REP=TID%5; SIG_IDX=(TID/5)%4; EQ_IDX=(TID/20)%2
# Output: $SAVE_TO/CDDM_<hash>_g0_noise_std/EqType=<EQ>_N=1000_sigrec=<SIG>/<net>
# Overridable: ITERS_OVERRIDE (default: the experiment config's 75,000), SAVE_TO, REPO.
# The cell name carries the budget, so runs of different lengths never pool.

#SBATCH --job-name=NoiseB0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --array=1-40
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

REPO="${REPO:-$HOME/trainRNNbrain}"
SAVE_TO="${SAVE_TO:-$REPO/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) - aborting." >&2
  exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
HASH=$(git -C "$REPO" rev-parse --short=6 HEAD)
IT=${ITERS_OVERRIDE:-75000}
EQS=(h s); SIGS=(0 0.01 0.05 0.1)
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 5)); SIG_IDX=$(((TID / 5) % 4)); EQ_IDX=$(((TID / 20) % 2))
EQ=${EQS[$EQ_IDX]}; SIG=${SIGS[$SIG_IDX]}

TAG="${HASH}_g0_noise_ptrack/EqType=${EQ}_N=1000_sigrec=${SIG}_iters=${IT}"
echo "Task ${SLURM_ARRAY_TASK_ID}: noise EQ=${EQ} N=1000 sigma_rec=${SIG} rep=${REP} iters=${IT} -> CDDM_${TAG}"
srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_N1000 \
  model=rnn_relu_standard trainer=trainer_ptrack_plain \
  model.equation_type="$EQ" model.sigma_rec="$SIG" \
  trainer.max_iter="$IT" \
  trainer.track_every=10 trainer.store_participation_every=100 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
