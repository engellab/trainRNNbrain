#!/bin/bash
# E3 / T2 — IS WEIGHT DECAY WHAT WALKS UNNEEDED UNITS DOWN? (SPOCK)
#
# The old paper plan asserted that weight decay is the driver of the slow walk to the activation floor; the
# only data (Della 11706899, 2026-07-28, wd 0 vs 1e-6) never had its participation read out and its
# traces are not reachable from the laptop. This sweep re-runs the question on the standard network
# with the exact recipe of the unpenalised baseline (SilentReLU_std_gamma0_drift: N=1000, 200k
# iterations, trainer_ptrack_plain) and three weight-decay values around the default 1e-6, which the
# drift sweep already provides.
#
# Grid (9 jobs) = 3 weight decays x 3 seeds, CDDM, N = 1000, h equation, no penalties.
#   WD 0 = 0        (no walker at all)
#   WD 1 = 1e-5     (10x default)
#   WD 2 = 1e-4     (100x default)
#   Index decode (TID = SLURM_ARRAY_TASK_ID-1): REP = TID%3; WD = TID/3
#
# Read-out: active units (scale-free) along training from the participation trace against the
# drift-sweep N=1000 baseline (wd 1e-6) at matched iteration. Decision rule (before the data): if
# wd=0 leaves the active count within the baseline's seed spread at 200k, weight decay is NOT the
# walker and §2.1 must name the task gradient instead; if wd=0 keeps substantially more units
# active, weight decay is (part of) the driver and its strength is a knob.
#
# Runtime ~8 h per job (measured on the drift sweep).
# Output: $SAVE_TO/CDDM_std_g0_weightdecay_bias0/EqType=h_N=1000_wd=<wd>_iters=200000/<per-net subfolder>
#
# TASK SWITCH (5 Nov 2026; the uniform-interventions programme, engellab/trainRNNbrain issue #6,
# docs/interventions_matrix.md). TASK unset or cddm runs the ladder exactly as above. TASK=ff runs
# the same three weight decays on the 3-bit flip-flop (task=NBitFlipFlop, 3 bits) with
# trainer_ptrack_freshbatch, the trainer of every 3BFF control and of the B5 wd 0 cell
# (NBitFlipFlop_b5_optimizer), at the 75,000-iteration same-size budget of silent_units_largeN.
# The CDDM ladder's trainer_ptrack_plain reuses one batch of trials for every iteration
# (same_batch true; the saved configs of the three CDDM cells say so); the 3BFF controls draw a
# fresh batch each iteration, so the ff arm matches them and not the CDDM arm. wd 0 on 3BFF exists
# (B5), so the missing cells are wd 1e-5 and 1e-4, indices 4-9:
#   TASK=ff sbatch --time=08:00:00 --constraint=nomig --array=4-9 slurm/b0_wd.slurm
# Output: $SAVE_TO/NBitFlipFlop_std_g0_weightdecay_bias0/EqType=h_N=1000_wd=<wd>_iters=75000/<per-net subfolder>

#SBATCH --job-name=WdB0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --array=1-9
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
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "WARNING: $REPO has uncommitted changes; the commit above does not fully describe this run." >&2
fi

N=${N_OVERRIDE:-1000}
WDS=(0 1e-5 1e-4)

# TASK selects the task words, the trainer, the budget and the folder prefix (see the header).
case "${TASK:-cddm}" in
  cddm) TASK_ARGS=(task=CDDM);                                          TRAINER=trainer_ptrack_plain;      IT=200000; PREFIX=CDDM ;;
  ff)   TASK_ARGS=(task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3); TRAINER=trainer_ptrack_freshbatch; IT=75000;  PREFIX=NBitFlipFlop ;;
  *) echo "FATAL: TASK must be cddm or ff; got '${TASK}'" >&2; exit 1 ;;
esac
IT=${ITERS_OVERRIDE:-$IT}

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); WD=${WDS[$((TID / 3))]}

TAG="std_g0_weightdecay_bias0/EqType=h_N=${N}_wd=${WD}_iters=${IT}"
mkdir -p "${SAVE_TO}/${PREFIX}_std_g0_weightdecay_bias0"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${TASK:-cddm} N=${N} weight_decay=${WD} rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/${PREFIX}_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  "${TASK_ARGS[@]}" \
  model=rnn_relu_standard trainer="$TRAINER" \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" trainer.weight_decay="$WD" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
