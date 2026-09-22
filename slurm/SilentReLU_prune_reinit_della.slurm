#!/bin/bash
# PRUNE-AND-REINITIALISE ("recycle a dead unit") ON UNPENALISED NETWORKS — DELLA
#
# THE MECHANISM, AND WHY IT IS NOT A PENALTY. A silent ReLU unit is FROZEN, not merely quiet. With
# r_i = 0 at every timestep, dL/dW_rec[i,j] ~ relu'(h_i) * r_j = 0 and dL/dW_rec[j,i] ~ r_i = 0, so
# every weight into and out of the unit has EXACTLY zero gradient (measured, not argued: see
# tests/test_prune_and_reinit.py::test_dead_units_have_exactly_zero_gradient). No term added to the
# loss can revive it, because a penalty acts through the same vanished gradient. This is the reason
# every intervention tried so far has had to act BEFORE a unit dies rather than after. Redrawing the
# weights directly is the only escape from that absorbing state, and it is the "recycle a useless
# neuron" mechanism from biology: the unit is not punished, it is replaced.
#
# WHAT IS REDRAWN: incoming weights only, W_rec[i,:] and W_inp[i,:], from the N(0, 1/sqrt(N)) the
# network was initialised from, after the unit has been silent for `patience` consecutive checks
# (check_every = 100 iterations, silence = participation < 0.05 * q95, the project-wide criterion).
# Outgoing weights are left alone: they received no gradient while the unit was dead, so they still
# hold whatever training last left there, and redrawing them would discard learned structure for
# units that died late. Adam's moments for the redrawn entries are zeroed — without that, stale
# momentum from before the unit died pushes the new weights straight back to the dead configuration.
#
# GRID (12 jobs) = 3 patience values x 3 seeds, plus a 3-seed MATCHED CONTROL.
#   patience in {2, 5, 20}  ->  200 / 500 / 2000 iterations of silence before a redraw
#   PRUNE=none              ->  the control arm: identical commit, config and seeds, mechanism off
# The control is run rather than read off disk. The archived CDDM h/none cells at this 30k read-out
# disagree across architecture — g0 438.8 +- 6.8, g0_nodale 408.8 +- 11.4, g0_nodale_trainablebias
# 403.8 +- 13.6 (n=5 each) — and all three predate the bias-fixing change in rnn_relu_standard.yaml.
# None of them is a clean reference for a bias-free non-Dale run on today's commit.
#
# PRE-REGISTERED, WRITTEN BEFORE ANY OF THESE JOBS EXISTED.
# Read-out at 30,000 iterations, active units under the scale-free criterion (0.05 * q95), taken
# from the participation trace. Planning estimate for the control: ~409 +- 11 (g0_nodale, above).
#   WIN:   an arm's mean clears control_mean + 3 * control_sd, using the MATCHED control measured
#          in this sweep, not the archived numbers.
#   COST GUARD: clean training loss at 30k (dropout-off, noise-free — the loss_clean_train series in
#          the trace, NOT the training-pass loss) must not exceed the control's by more than 10%.
#          An arm that clears the bar but fails this is reported as a TRADE, not a win. Recycling a
#          unit at iteration 29,900 hands the network an untrained random row; if that is costly it
#          will show up here, and the fix is to stop recycling near the end — not yet implemented,
#          deliberately, because this sweep is what decides whether it is needed.
#   TREADMILL FALSIFIER: the trace logs reinit_events (redraws including repeats) and
#          reinit_units_ever (distinct units touched). If events >> ever while the active count sits
#          at the control's, the mechanism is CHURNING — redrawing units that promptly re-die — and
#          that is a failure, not a partial success. This is a live possibility, not a formality:
#          nothing in the mechanism prevents a redrawn unit from being re-silenced by the same
#          recurrent inhibition that killed it, and if the network's solution genuinely needs only
#          ~410 active units, recycling will spin forever.
#   NULL:  within control +- 3 sd -> no effect, and that is what gets written. A null here is
#          informative: it would say the silent state is the network's PREFERRED configuration
#          rather than an accident it cannot escape, which is a stronger claim than the paper
#          currently makes in either direction.
#
# Cost: no extra forward pass — the silence check reuses the training pass's states, and fires once
# every 100 iterations. Overhead is a quantile over N units per check, i.e. negligible. ~30k iters
# at N=1000 is ~2-3 h; 6 h requested.
#   sbatch --array=1-9 slurm/SilentReLU_prune_reinit_della.slurm
#   PRUNE=none sbatch --array=1-3 slurm/SilentReLU_prune_reinit_della.slurm    # matched control
#
# SWEEP=<name> puts the runs in their own folder; use it if the redraw rule changes at all.
# Output: $SAVE_TO/CDDM_prune_reinit/EqType=h_N=1000_pen=none_prune=<patience>/

#SBATCH --job-name=PruneReinit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=06:00:00
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

N=1000
IT=${ITERS_OVERRIDE:-30000}
ARM_PATIENCE=(2 5 20)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); ARM=$((TID / 3))
PATIENCE=${ARM_PATIENCE[$ARM]}

if [ "${PRUNE:-on}" = "none" ]; then
  PR_ARGS=(trainer.prune_reinit=false)
  TAG="${SWEEP:-prune_reinit}/EqType=h_N=${N}_pen=none_prune=none"
else
  PR_ARGS=(trainer.prune_reinit=true "trainer.prune_args.patience=${PATIENCE}")
  TAG="${SWEEP:-prune_reinit}/EqType=h_N=${N}_pen=none_prune=${PATIENCE}"
fi

mkdir -p "${SAVE_TO}/CDDM_${SWEEP:-prune_reinit}"
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} prune=${PRUNE:-on} patience=${PATIENCE} rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=CDDM \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  "${PR_ARGS[@]}" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
