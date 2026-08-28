#!/bin/bash
# LARGE-N POINT: does the active-unit exponent FALL with N, or is M ~ N^0.44 unbounded? (SPOCK)
#
# THE QUESTION THIS ANSWERS, which the existing grid cannot. Over N = 500..2000 the active-unit
# count follows M ~ N^0.44 with b constant - a pure power law with NO ceiling. But the CDDM result
# was a SATURATING law, with a hyperbolic fit putting a ceiling near 880 active units. Those are
# different claims and 4x of size range cannot separate them: extrapolating M = 18.9 N^0.44 from
# the current fit predicts M = 594 at N=3000 and 741 at N=5000, so a ceiling anywhere near 800 only
# starts bending the curve above N ~ 4000. A larger N is not precision, it is the discriminating
# measurement.
#
# ONLY none AND rws. Both converge well inside 100k on this task (unpenalised settles at 63k median,
# rws at 54k; 43/43 rws runs reach the W_inp diffusive point). frm does NOT - it is still ballistic
# at 150k with its loss falling 8% over the final tenth - and a converged frm budget at N >= 3500
# needs ~290k iterations, which is 147 h on a slow node against Della's 144 h cap. Large-N penalty
# cells are therefore DEFERRED until the 400k re-run (SilentReLU_flipflop_frmboth_long_spock.slurm)
# establishes what budget frm actually needs. Submitting them first would repeat the mistake that
# produced the retracted 150k frm floors.
#
# WHY 100000 ITERATIONS. The read-out is the W_inp diffusion point, which scales as N^-0.16: it lands
# at 63k median over N=500..2000, so ~48k at N=4000. 100k is ~2x that. Matched-loss and matched-excess
# read-outs land earlier still (29k-41k). This budget is NOT sufficient for a floor fit deep enough
# to compare against the 400-500k unpenalised cells, and no floor comparison should be drawn from it.
#
# WHY N=4000 BY DEFAULT. Largest size whose 100k run fits Spock's 4-day cap with slow-node margin
# (0.17 s/iter predicted -> 47 h median, 64 h slow). N=5000 needs Della (97 h slow). Override with
# BIGN=5000 and submit to Della if the queue allows.
#   sbatch --array=1-48 --time=96:00:00 SilentReLU_flipflop_bigN_spock.slurm
#   BIGN=5000 sbatch ...          # Della only
# ⚠️ RUN THE TIMING CALIBRATION FIRST - the cost model mispredicts the measured N=500 point by 25%:
#   ITERS_OVERRIDE=600 sbatch --array=1 --time=02:00:00 SilentReLU_flipflop_bigN_spock.slurm
#
# Grid = 2 penalties x 8 complexities x 3 seeds = 48 jobs, at one size.
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..47): REP=TID%3; K_IDX=(TID/3)%8; P_IDX=TID/24
#
# Output: $SAVE_TO/NBitFlipFlop_std_bigN/EqType=h_k=<k>_N=<N>_pen=<name>_iters=<IT>/

#SBATCH --job-name=FFbigN
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --array=1-48
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
PEN_NAME=(none rws)
PEN_RWS=(0 0.05)
PEN_FRM=(0 0)
IT=100000
N=${BIGN:-4000}

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); K_IDX=$(((TID / 3) % 8)); P_IDX=$((TID / 24))
K=${KS[$K_IDX]}
PNAME=${PEN_NAME[$P_IDX]}; LRWS=${PEN_RWS[$P_IDX]}; LFRM=${PEN_FRM[$P_IDX]}
IT=${ITERS_OVERRIDE:-$IT}

TAG="std_bigN/EqType=h_k=${K}_N=${N}_pen=${PNAME}_iters=${IT}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_std_bigN"
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
