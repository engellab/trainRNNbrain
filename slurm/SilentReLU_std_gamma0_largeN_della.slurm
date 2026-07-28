#!/bin/bash
# LARGE-N sweep: does the ACTIVE unit count saturate? (DELLA)
#
# Settles the "why bother — just train a bigger network and prune the silent units" objection.
# Two models fit the N<=1000 data equally well and diverge sharply beyond it:
#   power law   active = 4.55*N^0.665  -> 710 / 1305 / 2069 active at N = 2000 / 5000 / 10000
#   saturating  active = 749*N/(N+672) -> 561 /  660 /  702      (ceiling ~749 active units)
# The local exponent already falls (0.899 -> 0.621 -> 0.398), which favours saturation.
#
# Standard RNNs, no penalties (the silent fraction is what is being measured), h equation, 3 seeds.
# Grid (15 jobs), TID = SLURM_ARRAY_TASK_ID-1:
#   TID  0-2 : N=1000,  5000 iters   <- truncation control vs the existing 30000-iter N=1000 data
#   TID  3-5 : N=2000,  5000 iters   <- truncation control at a second size
#   TID  6-8 : N=2000, 30000 iters   <- full-length reference for that control
#   TID  9-11: N=5000,  5000 iters
#   TID 12-14: N=10000, 5000 iters
# Truncation is legitimate because the silent fraction is decided by iteration ~400-600 and frozen
# (measured in the participation traces); the two control cells verify that rather than assume it.
#
# MEASURED on an A100 (job 11677325): s/iter 0.144 / 0.317 / 1.595 / 5.788 at N = 1k/2k/5k/10k, and
# peak GPU 6.6 / 13.1 / 33.1 / 65.9 GB. N=10000 therefore REQUIRES an 80 GB card (--constraint=gpu80,
# which also excludes the MIG slices) and 30000 iterations there would take 48 h, beyond gpu-short.
#
# +experiment=silent_units_largeN sets light_outputs=true: skips all numpy/CPU post-processing
# (~3 h single-threaded at N=10000, a 10.8 GB float64 array, PCA over 10000 x 135000) and saves
# parameters as .npz rather than ~4 GB of json. The readout is the participation trace.
#
# Output: $SAVE_TO/CDDM_std_g0_largeN/EqType=h_N=<N>_iters=<IT>/<per-net subfolder>

#SBATCH --job-name=StdLargeN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=12:00:00
#SBATCH --constraint=gpu80
#SBATCH --array=1-15
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

REPO="${REPO:-$HOME/trainRNNbrain_largeN}"
SAVE_TO="${SAVE_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.dirname(os.path.dirname(trainRNNbrain.__file__)))")
if [ "$PKG" != "$REPO" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO — aborting." >&2; exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short=7 HEAD)"

TID=$((SLURM_ARRAY_TASK_ID - 1))
CELL=$((TID / 3))
case $CELL in
  0) N=1000;  IT=5000  ;;
  1) N=2000;  IT=5000  ;;
  2) N=2000;  IT=30000 ;;
  3) N=5000;  IT=5000  ;;
  4) N=10000; IT=5000  ;;
  *) echo "bad task id"; exit 1 ;;
esac

TAG="std_g0_largeN/EqType=h_N=${N}_iters=${IT}"
mkdir -p "${SAVE_TO}/CDDM_std_g0_largeN"
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  cp "$REPO/docs/experiments/standard_rnn_largeN.md" "${SAVE_TO}/CDDM_std_g0_largeN/EXPERIMENT.md"
  echo -e "\n---\nRun: SLURM array ${SLURM_ARRAY_JOB_ID} on Della, code commit $(git -C "$REPO" rev-parse HEAD)" \
    >> "${SAVE_TO}/CDDM_std_g0_largeN/EXPERIMENT.md"
fi
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} iters=${IT} rep=$((TID % 3))"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  model=rnn_relu_standard trainer=trainer_ptrack_plain \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
