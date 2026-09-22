#!/bin/bash
# HOW LONG A DELAY CAN THE UNPENALISED NETWORK ACTUALLY LEARN? (SPOCK)
#
# WHY. The redesigned 2-stimulus DMTS is much harder than the 4-stimulus version it replaced. At
# 16 tau, N=500, pen=none solved 1 of 3 seeds (r2 0.9991 against 0.4516 and 0.4519); rws solved
# 0 of 3; frm and frm+rws solved 3 of 3. At 36 tau nothing escapes at all (none 0.4717 +- 0.00004,
# rws 0.4714 - six runs within 0.0006 of each other, every seed finding the same degenerate
# solution: emit zero through the pre-decision epoch, which is most of the scored timesteps, then
# guess). The old task was solved reliably by everything, which is exactly why it discriminated
# nothing: at 25/75 class balance a network scored 75% by always answering non-match.
#
# That leaves Figure 1c without a usable DMTS scaling series, because active-unit counts from
# networks that never learned the task mean nothing for a claim about how many units a TRAINED
# network uses. Worse, the two are coupled: at N=500 the solver had 170 active units against 123
# and 113 for the failures, +52 units or ~45%. So conditioning on "solved" is a real selection
# effect and the solve rate has to be reported next to any count.
#
# WHAT THIS MEASURES. The solve rate against delay, so the scaling series can use the SHORTEST
# delay the baseline learns reliably. Passive decay is e^-8 = 3.4e-4 at the easiest rung, already
# far below sigma_rec = 0.05, so the sample cannot be carried by a transient anywhere on this
# ladder - and with spectral radius 1.2 the network can build modes far slower than the single-unit
# time constant anyway. What the delay sets is how slow a mode must be learned, not whether one is
# needed. Dropping 16 -> 10 tau therefore costs nothing in the "no transients" argument.
#
# GRID (9 jobs) = 3 delays x 3 seeds, at N = 1000, pen = none, h equation, bias fixed at 0.
#   delay in {8, 10, 12} tau. The 16 tau rung is ALREADY RUNNING as part of DMTSv2 and is not
#   repeated here; read it from DMTS_v2_pen/EqType=h_N=1000_pen=none.
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..8): REP = TID%3; D_IDX = TID/3
#
# READ-OUT, fixed before submission. Solved := clean r2 >= 0.9 at 150,000. The headline is the
# SOLVE RATE per delay, n = 3, not a mean r2 - averaging a bimodal 0.45/0.99 split describes no
# network that exists. Report the active-unit count for solvers and failures SEPARATELY, never
# pooled, because they differ by ~45%.
# DECISION RULE: the scaling series takes the shortest delay with 3/3. If no rung reaches 3/3,
# three seeds cannot distinguish the rungs and the answer is more seeds, not a shorter delay.
#
# Cost: T shrinks with the delay (220/240/260 against 300 at 16 tau), so these are CHEAPER than the
# 16 tau cell - roughly 8-10 h at N=1000 for 150k. 24 h requested.
#   sbatch --array=1-9 slurm/SilentReLU_dmts_delayladder_spock.slurm
#
# Output: $SAVE_TO/DMTS_v2_delay/EqType=h_N=1000_d=<delay>_pen=none/

#SBATCH --job-name=DMTSdelay
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=1-9
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
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
if [ -z "$GITSHA" ]; then
  echo "FATAL: $REPO is not a git checkout - provenance of this run cannot be recorded." >&2; exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

N=1000
IT=${ITERS_OVERRIDE:-150000}
DELAYS=(8 10 12)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); D_IDX=$((TID / 3))
D=${DELAYS[$D_IDX]}

TAG="v2_delay/EqType=h_N=${N}_d=${D}_pen=none"
mkdir -p "${SAVE_TO}/DMTS_v2_delay"
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} delay=${D}tau pen=none rep=${REP} iters=${IT}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=DMTS_d${D} \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
