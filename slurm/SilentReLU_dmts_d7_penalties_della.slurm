#!/bin/bash
# DMTS AT A 7-TAU DELAY: ACTIVE UNITS vs NETWORK SIZE, ACROSS PENALTY ARMS (DELLA)
#
# This is the DMTS series for Figure 1c. The delay was chosen by measurement, not assumption: the
# ladder run on 2026-09-23 (job 14318554, 150k iterations, N=1000, no penalties) gave
#   8 tau  3/3 solved   r2 0.999, 0.999, 0.969
#   10 tau 1/3 solved   r2 0.996, 0.439, 0.438
#   12 tau 0/3 solved   r2 0.443, 0.443, 0.442
#   16 tau 0/3 solved   r2 0.452, 0.452, 0.450   (from the deleted DMTS_v2_pen sweep)
# The decision rule fixed before that ladder ran was "take the shortest delay reaching 3/3", and 8
# tau was the only rung that did. 7 tau is one rung shorter still: it keeps the 3/3 margin while
# cutting T from 220 to 210 steps, and every timing other than the delay is unchanged.
#
# ⚠️ 150,000 ITERATIONS IS REQUIRED, NOT PADDING. The 8-tau networks sit at r2 ~ 0.43 for most of
# training and then jump: one seed crossed between iteration 110,000 and 120,000, another by
# 80,000. Cutting to 100k would have scored the first as a failure and turned 3/3 into 2/3. Since
# the read-out here is a SOLVE RATE, an iteration budget that truncates a late solver does not make
# the result noisier, it makes it wrong.
#
# GRID (36 jobs) = 4 penalty arms x 3 sizes x 3 seeds.
#   arms   none, rws (lambda_rws 0.05), frm (lambda_frm 0.2), both
#   sizes  N = 500, 1000, 2000
#   Index decode (TID = SLURM_ARRAY_TASK_ID - 1, 0..35): REP = TID%3; REST = TID/3;
#                                                        SIZE = REST%3; ARM = REST/3
#   sbatch --array=1-36%12 slurm/SilentReLU_dmts_d7_penalties_della.slurm
#
# N=4000 IS DELIBERATELY ABSENT. Measured cost is 7.5 h per job at N=1000 and runtime grows about
# 2.5x per doubling, so N=2000 is ~21 h and N=4000 would be 50-60 h -- past any single-job wall
# limit, needing checkpoint-restart rather than a longer --time. The flip-flop and CDDM series
# already carry the large-N end of Figure 1c.
#
# PRE-REGISTERED, WRITTEN BEFORE ANY OF THESE JOBS EXISTED. Read-out at 150,000 iterations.
#   PRIMARY    the SOLVE RATE per (arm, size), n=3. Solved := clean r2 >= 0.9. NOT a mean r2:
#              these runs are bimodal at ~0.44 and ~0.999, and an average of the two describes no
#              network that exists.
#   SECONDARY  active units under the scale-free criterion (0.05 * q95), reported SEPARATELY for
#              solvers and failures and never pooled. The gap is real and large: at 10 tau the
#              solver had 204 active units against 150 for the two failures, 36% more, so any
#              pooled count is a selection artifact that grows with the failure rate.
#   REPORTING RULE: every cell states its solve rate next to its unit count. A cell that solves
#              0/3 contributes no unit count to Figure 1c at all -- counts from networks that
#              never learned the task say nothing about how many units a trained network uses.
#   EXPECTED, stated so it can be wrong: at 16 tau frm and both solved 3/3 at N=500 where none
#              solved 1/3 and rws 0/3, so frm is expected to hold 3/3 everywhere here and rws to
#              be the weakest arm. If none also solves 3/3 at every size, the delay is too easy to
#              discriminate the arms and the series should move back to 8 tau.
#
# Cost: ~4 h at N=500, ~7.2 h at N=1000, ~21 h at N=2000. 30 h requested to cover the largest.
# Output: $SAVE_TO/DMTS_d7_pen/EqType=h_N=<n>_pen=<arm>/

#SBATCH --job-name=DMTSd7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=30:00:00
#SBATCH --array=1-36%12
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

REPO="${REPO:-$HOME/trainRNNbrain}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2; exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
if [ -n "$GITSHA" ]; then
  if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
    echo "FATAL: $REPO is a git checkout with uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
  fi
  echo "code: $PKG @ $GITSHA (git)"
elif [ -f "$REPO/CODE_VERSION.txt" ]; then
  echo "code: $PKG @ $(cat "$REPO/CODE_VERSION.txt") (rsync)"
else
  echo "FATAL: $REPO is neither a clean git checkout nor carries CODE_VERSION.txt — provenance of this run cannot be recorded." >&2; exit 1
fi

IT=${ITERS_OVERRIDE:-150000}
ARM_NAME=(none rws frm both)
ARM_RWS=(0    0.05 0    0.05)
ARM_FRM=(0    0    0.2  0.2)
ARM_SIZE=(500 1000 2000)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); REST=$((TID / 3)); S_IDX=$((REST % 3)); A_IDX=$((REST / 3))
NAME=${ARM_NAME[$A_IDX]}
LRWS=${ARM_RWS[$A_IDX]}
LFRM=${ARM_FRM[$A_IDX]}
N=${ARM_SIZE[$S_IDX]}

TAG="${SWEEP:-d7_pen}/EqType=h_N=${N}_pen=${NAME}"
mkdir -p "${SAVE_TO}/DMTS_${SWEEP:-d7_pen}"
echo "Task ${SLURM_ARRAY_TASK_ID}: arm=${NAME} N=${N} lambda_rws=${LRWS} lambda_frm=${LFRM} rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/DMTS_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=DMTS_d7 \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws="$LRWS" trainer.lambda_frm="$LFRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
