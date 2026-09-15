#!/bin/bash
# DMTS WITH A 36-TAU DELAY: does the unpenalised network still find the memory solution? (SPOCK)
#
# At 16 tau (DMTS_std_pen, 2026-09-15 read-out) frm networks left the no-memory plateau 2-3x
# earlier than unpenalised ones, and the gap grew with N (two `none` seeds at N=2000 needed 11k and
# 30k iterations). Pavel's check: lengthen the silent delay to 36 tau (T=500) and ask whether the
# unpenalised network gets off the plateau AT ALL within 150k, against frm.
#
# Two jobs, one seed each, N=1000 (the middle cell of the 16-tau grid, where both arms escaped
# within 5k): task 1 = none, task 2 = frm 0.1. Same trainer, iteration budget and read-outs as the
# 16-tau grid (dmts_readout.py works unchanged: point it at DMTS_std_delay36).
#   Read-out: t(clean r2 >= 0.9) per arm; "never" for `none` within 150k while frm escapes is the
#   result Pavel expects. One seed each: a positive answer needs the other two seeds before it is
#   quoted (the 16-tau `none` seeds at N=2000 spanned 2k-30k).
# Cost: T=500 is 1.67x the 16-tau per-iteration cost (0.26 s/iter at N=1000 -> ~0.43), 150k ->
# ~18 h; 30 h requested.
#   sbatch --array=1-2 slurm/SilentReLU_dmts_delay36_spock.slurm
#
# Output: $SAVE_TO/DMTS_std_delay36/EqType=h_N=1000_pen=<name>/<per-net subfolder>

#SBATCH --job-name=DMTSd36
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --array=1-2
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

N=1000
PEN_NAME=(none frm)
PEN_RWS=(0 0)
PEN_FRM=(0 0.1)
IT=150000

P_IDX=$((SLURM_ARRAY_TASK_ID - 1))
PNAME=${PEN_NAME[$P_IDX]}; LRWS=${PEN_RWS[$P_IDX]}; LFRM=${PEN_FRM[$P_IDX]}
IT=${ITERS_OVERRIDE:-$IT}

TAG="std_delay36/EqType=h_N=${N}_pen=${PNAME}"
mkdir -p "${SAVE_TO}/DMTS_std_delay36"
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} pen=${PNAME} (rws=${LRWS} frm=${LFRM}) iters=${IT} delay=36tau"
echo "  -> ${SAVE_TO}/DMTS_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=DMTS_delay36 \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws="$LRWS" trainer.lambda_frm="$LFRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
