#!/bin/bash
# frm AND both, RE-RUN AT A CONVERGED BUDGET (SPOCK)
#
# WHY THIS EXISTS. The original penalty grid ran 150000 iterations, a budget taken from a CDDM
# measurement quoted in SilentReLU_flipflop_penalties_spock.slurm: "the clean task loss reaches
# within 2% of its final value at 4.6k-14.6k (rws, frm)". Measured on the flip-flop:
#
#   condition   W_inp reaches alpha<0.6 in budget      loss fall over the last third
#   none              64/64, median 63k                        -0.3% .. +0.1%
#   rws               43/43, median 54k                         0.0% .. +0.4%
#   frm                0/4  - STILL BALLISTIC                   +5.4% .. +8.4%
#
# frm's lag exponent starts at ~1.0 and ENDS at 0.87-1.005; its loss is still descending ~8% over
# the final tenth of the run. It has not stopped learning. Every quantity the analysis needs -
# fitted floors, matched-performance read-outs, the diffusion criterion - assumes convergence, so
# the 150k frm data cannot support any of them. A "frm worsens the floor by 8-10%" flag drawn from
# it has already been retracted: that was a stretched-exponential fitted to a still-descending
# curve, and the excess matched the amount the loss was still falling.
#
# `both` contains frm and is re-run for the same reason. `rws` is NOT re-run - it converges inside
# 150k by both measures and its existing cells stand.
#
# WHY 400000. frm has not settled by 150k, so it is >2.4x slower than unpenalised (which settles at
# 63k median). Assuming 3-4x gives ~200-250k, and 400k is ~1.8x beyond that. It also MATCHES the
# unpenalised N=2000 budget exactly, so floors can be fitted over an identical range rather than a
# mismatched one - the error that manufactured a spurious trend three separate times in this project.
# 500k was considered and rejected for N=2000: 500k x 0.55 s/iter = 76 h median but 103 h on a
# 1.35x slow node, over Spock's 4-day cap. Sizing from the median rather than the slow tail nearly
# killed an earlier array.
#
# Grid = 2 penalties x 8 complexities x 3 sizes x 3 seeds = 144 jobs.
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..143):
#   REP=TID%3;  N_IDX=(TID/3)%3;  K_IDX=(TID/9)%8;  P_IDX=TID/72
#
# Wall time by size at 400k (median / 1.35x slow node):
#   N=500  -> 0.24 s/iter -> 27 h / 36 h   request 48 h
#   N=1000 -> 0.26 s/iter -> 29 h / 39 h   request 48 h
#   N=2000 -> 0.55 s/iter -> 61 h / 83 h   request 96 h  (Spock's cap; no margin above this)
# Submit ONE ARRAY PER SIZE so the request matches the cell:
#   N=500 : sbatch --array=$(seq -s, 1 3 144 | tr '\n' ' ')  --time=48:00:00   # N_IDX=0
# In practice use the helper at the bottom rather than hand-built index lists.
#
# Output: $SAVE_TO/NBitFlipFlop_std_penlong/EqType=h_k=<k>_N=<N>_pen=<name>_iters=<IT>/

#SBATCH --job-name=FFpenL
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=1-144
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
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL — aborting." >&2; exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "WARNING: $REPO has uncommitted changes; the commit above does not fully describe this run." >&2
fi
# The .npz saver lost dict-valued parameters (activation_args stored as its KEYS) until the
# storable_ fix. Refuse to run on code that predates it, or these nets cannot be reconstructed.
python - <<'PYCHK' || exit 1
import sys
from trainRNNbrain.training import run_experiment as m
sys.exit(0 if hasattr(m, "storable_") else
         print("FATAL: run_experiment lacks storable_(); re-sync the repo before submitting.") or 1)
PYCHK

KS=(1 2 3 4 5 6 7 8)
NS=(500 1000 2000)
PEN_NAME=(frm both)
PEN_RWS=(0 0.05)
PEN_FRM=(0.1 0.1)
IT=400000

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); N_IDX=$(((TID / 3) % 3)); K_IDX=$(((TID / 9) % 8)); P_IDX=$((TID / 72))
K=${KS[$K_IDX]}; N=${NS[$N_IDX]}
PNAME=${PEN_NAME[$P_IDX]}; LRWS=${PEN_RWS[$P_IDX]}; LFRM=${PEN_FRM[$P_IDX]}
IT=${ITERS_OVERRIDE:-$IT}

TAG="std_penlong/EqType=h_k=${K}_N=${N}_pen=${PNAME}_iters=${IT}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_std_penlong"
echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K} N=${N} pen=${PNAME} (rws=${LRWS} frm=${LFRM}) rep=${REP} iters=${IT}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=NBitFlipFlop \
  task.n_inputs="$K" task.n_outputs="$K" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws="$LRWS" trainer.lambda_frm="$LFRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
