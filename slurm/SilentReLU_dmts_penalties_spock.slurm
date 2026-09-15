#!/bin/bash
# THIRD TASK: delayed match-to-sample with a 16-tau delay, all four penalty arms x three sizes (SPOCK)
#
# CDDM and the flip-flop family gave the recruitment picture (live units track task demand; the
# frm/rws penalty recruits every unit at a performance cost). Both tasks can be carried partly by
# transients: CDDM's stimulus is on until the decision, and a flip-flop bit is re-pulsed every
# ~7 tau. DMTS_long removes that: a 2-tau sample, then 16 tau of silent input, then a 2-tau match
# and a decision. The sample identity has to be held by SUSTAINED activity, which is the regime in
# which a firing-rate penalty (frm keeps every unit's mean rate near cap_fr) could plausibly HELP
# rather than only cost.
#
# Pre-registered read-outs, all at 150k iterations, all from the participation trace + the saved
# r2 (nothing new to log):
#   1. Live units vs N per arm (the usual count; never a silent fraction).
#   2. r2 on the scored epochs, none vs each penalty. Solved := r2 >= 0.9 in all 3 seeds.
#   3. CONVERGENCE SPEED (Pavel's question: does the penalty make a sustained-activity task train
#      FASTER?). Read-out: t90 := first probe at which the clean loss (noise-free training-batch
#      MSE, logged every 10 iterations) is within 10% of its 150k value.
#        Falsifier, fixed now: the claim "the penalty speeds training" is SUPPORTED only if
#        t90(frm) < t90(none) and t90(both) < t90(none) in every seed at every N. One seed or one
#        N in the other direction and the answer is "no, or not reliably". If the penalised arms
#        reach a HIGHER final loss (as on every task so far) t90 is compared against each arm's OWN
#        floor and a matched-loss time (first probe at which each arm reaches none's 150k loss) is
#        reported alongside, so a fast descent to a worse floor is not mistaken for speed.
#
# Grid (36 jobs) = 3 sizes x 4 penalty arms x 3 seeds, h equation, standard ReLU, fresh batches.
#   N   in {500, 1000, 2000}
#   pen in {none, rws, frm, both}   rws 0.05, frm 0.1, the project's standard values
# Index decode (TID = SLURM_ARRAY_TASK_ID-1): REP=TID%3; P_IDX=(TID/3)%4; N_IDX=TID/12
#   tasks 1-12 N=500, 13-24 N=1000, 25-36 N=2000; within a size: 3 none, 3 rws, 3 frm, 3 both.
# Submit ONE ARRAY PER SIZE so the wall-time request matches the cell (T=300, batch 1024, same
# per-iteration cost as the plain flip-flop: 0.24-0.26 s/iter at N=500/1000, 0.55 at N=2000):
#   sbatch --array=1-24  slurm/SilentReLU_dmts_penalties_spock.slurm                    # 24 h
#   sbatch --array=25-36 --time=48:00:00 slurm/SilentReLU_dmts_penalties_spock.slurm    # N=2000
# Timing calibration: ITERS_OVERRIDE=600 sbatch --array=25 ...
#
# Output: $SAVE_TO/DMTS_std_pen/EqType=h_N=<N>_pen=<name>/<per-net subfolder>

#SBATCH --job-name=DMTSpen
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=1-36
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

NS=(500 1000 2000)
PEN_NAME=(none rws frm both)
PEN_RWS=(0 0.05 0 0.05)
PEN_FRM=(0 0 0.1 0.1)
IT=150000

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); P_IDX=$(((TID / 3) % 4)); N_IDX=$((TID / 12))
N=${NS[$N_IDX]}
PNAME=${PEN_NAME[$P_IDX]}; LRWS=${PEN_RWS[$P_IDX]}; LFRM=${PEN_FRM[$P_IDX]}
IT=${ITERS_OVERRIDE:-$IT}

TAG="std_pen/EqType=h_N=${N}_pen=${PNAME}"
mkdir -p "${SAVE_TO}/DMTS_std_pen"
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} pen=${PNAME} (rws=${LRWS} frm=${LFRM}) rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/DMTS_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=DMTS_long \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws="$LRWS" trainer.lambda_frm="$LFRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
