#!/bin/bash
# REVIVING UNITS ON THE 3-BIT FLIP-FLOP: DUPLICATION AND REDISTRIBUTION, ACROSS THREE SIZES (DELLA)
#
# WHY THIS TASK. The dropout sweep runs on NBitFlipFlop k=3 at N=1000 for 40,000 iterations, so
# these arms are directly comparable to it. Its no-dropout control sits at 316.3 +- 33.3 active
# units and its best dropout arm (`mute`, 10% drop rate, beta=2) reaches 425.0 +- 14.8. Everything
# here uses the same task, size, iteration count and read-out, so the numbers can sit in one table.
#
# TWO INTERVENTIONS, both acting on the weights directly rather than through the loss. A silent
# ReLU unit has exactly zero gradient on every weight into and out of it, so nothing added to the
# loss can revive one; only overwriting weights can.
#
#   prune_copy - a unit silent for 300 consecutive iterations is replaced by a DUPLICATE of a live
#     unit, drawn in proportion to participation, with the donor's outgoing weights split between
#     the two so the network's function is unchanged at the instant of duplication. On CDDM this
#     took the active count from 410.3 +- 10.6 to 716.7 +- 21.4 with no gradient instability, at a
#     clean-loss cost of 0.0051 -> 0.0085. Random redraws in the same slot did nothing (414.3),
#     dying 44 times each at about 1% survival, against 70% for duplication.
#
#   scale - homeostatic redistribution of a unit's input weights, excitation up and inhibition
#     down, at FIXED ROW NORM. The fixed norm is the repair, not a detail: the sign split alone
#     changes a row's magnitude by about (alpha + 1/alpha), which exceeds 2 for every alpha but 1,
#     so every event pumped magnitude in whichever way it scaled. All three seeds of the first
#     attempt discarded 19-27% of their gradient updates and the rank-matched variant 31-55%, every
#     one ending at r2 of NaN or -inf. Renormalising makes an event a pure redistribution.
#     ⚠️ A FIXED ROW NORM IS NECESSARY AND NOT SUFFICIENT. Recurrent stability is governed by the
#     SPECTRAL RADIUS, which row norms do not control: shifting magnitude toward excitation aligns
#     the rows and lifts the leading eigenvalue with every row norm held exactly. Measured over 5
#     events at N=300, eta=0.5 took the radius 1.20 -> 1.85 -> 2.81 -> 3.82 -> 4.51, and a real
#     calibration run at eta=0.5 discarded 399 of its first 400 gradient updates and ended at NaN.
#     Over 100 events every eta up to 0.1 held the radius between 1.20 and 1.26, so eta is capped
#     at 0.05 here. An earlier claim that saturation made a large eta safe came from a static test
#     on 40 units with no recurrent dynamics and was wrong.
#
# GRID: 3 arms x 3 sizes x 3 seeds = 27 jobs. ARM = TID/9, SIZE = (TID/3)%3, REP = TID%3.
#   arms  none, prune_copy, scale        sizes  N = 500, 1000, 2000
#   control + duplication:  sbatch --array=1-18 slurm/SilentReLU_ff_revive_della.slurm
#   redistribution:         sbatch --array=19-27 slurm/SilentReLU_ff_revive_della.slurm
#   with cubic saturation:  GAMMA=0.1 SWEEP=ff_revive_g01 sbatch --array=1-3,10-12 ...
#
# WHY A GAMMA ARM EXISTS. Duplication at N=500 with gamma=0 destabilised on ALL THREE seeds:
# 29-44% of gradient updates discarded, clean loss 273,489, active count 285.0 +- 193.9 with a
# standard deviation almost as large as the mean, and 27,299 replacements over 500 units (54.6
# each, the treadmill signature). At N=1000 on CDDM the same mechanism ran with zero skipped
# updates. Size does not remove a failure mode, it changes how often it fires, and three seeds
# cannot tell "rarer" from "gone" -- so the N=1000 and N=2000 cells locate the instability but do
# not fix it. The cubic term bounds x directly while keeping ReLU's exact zero and its frozen
# gradient, which are the properties the whole duplication argument rests on. A bounded activation
# such as the shifted sigmoid would also stop the blow-up, and would remove the phenomenon with it:
# no unit can be exactly zero there and every unit has gradient everywhere, so duplication would be
# solving a problem the activation had already dissolved.
#
# PRE-REGISTERED, WRITTEN BEFORE ANY OF THESE JOBS EXISTED.
#   PRIMARY   active units at 40,000 iterations, scale-free criterion (0.05 * q95), from the trace.
#   CO-PRIMARY clean dropout-off loss at 40k (loss_clean_train), NOT the training-pass loss.
#   WIN       an arm beats its own matched control at the same N by more than 3 control sd, with
#             clean loss within 10% of that control.
#   THE SIZE QUESTION, which is why three sizes are here rather than one: on CDDM duplication held
#             an EQUILIBRIUM at about 70% of units (707 of 1000), not a ceiling it was still
#             approaching -- the count was flat from iteration 11,000 onward while replacements
#             continued at 450-550 per 1,000 iterations. If that 70% is a property of the METHOD it
#             should reappear at N = 500 and 2000; if it is a property of that particular network it
#             should not. A constant FRACTION across sizes and a constant COUNT across sizes are
#             different findings and the paper needs to know which it has.
#   TREADMILL the trace logs reinit_events and reinit_units_ever. Many replacements with no gain in
#             the active count is a failure, not partial success; that is exactly what the random
#             redraw did.
#   STABILITY any arm discarding more than 5% of its gradient updates is reported as broken rather
#             than as a result, whatever its unit count. Two previous scaling arms cleared their
#             unit bar while discarding a fifth to a half of training.
#   NULL      within 3 sd of the matched control -> no effect, written as such.
#
# Cost: neither mechanism adds a forward pass; both reuse the training pass and fire every 100
# iterations. ~1h25m per job at N=1000 on this task; 4 h requested to cover N=2000.
# Output: $SAVE_TO/NBitFlipFlop_ff_revive/EqType=h_k=3_N=<n>_pen=none_arm=<name>/

#SBATCH --job-name=FFrevive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=48G
# TIME LIMIT, set from measurement rather than estimate: N=2000 reached only 21,944 of 40,000 iterations in 4h05m, so 40k needs ~7.4 h.
# The first submission used the line below this one and lost every N=2000 cell to TIMEOUT.
#SBATCH --time=12:00:00
#SBATCH --array=1-27%12
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

K=3
IT=${ITERS_OVERRIDE:-40000}
# Cubic saturation in the dynamics: dx/dt gains a -GAMMA*x^3 term, which bounds x while leaving
# ReLU's exact-zero floor and its frozen-gradient behaviour intact. 0 = off, which is what
# rnn_relu_standard sets and what every run in this sweep used before 2026-09-23. Runs at a
# different gamma MUST go in their own SWEEP folder: gamma changes the dynamics for the control
# too, so a gamma=0.1 arm and a gamma=0 arm are not the same experiment.
GAMMA=${GAMMA:-0}
ARM_NAME=(none prune_copy scale)
ARM_SIZE=(500 1000 2000)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); REST=$((TID / 3)); S_IDX=$((REST % 3)); A_IDX=$((REST / 3))
NAME=${ARM_NAME[$A_IDX]}
N=${ARM_SIZE[$S_IDX]}
# N_OVERRIDE runs this same grid at a size the ARM_SIZE table does not list. Use a distinct
# SWEEP when doing so, or cells at different sizes land in folders that differ only by a
# field the analysis does not group on.
N=${N_OVERRIDE:-$N}

ARGS=(trainer.prune_reinit=false trainer.synaptic_scaling=false)
case "$NAME" in
  prune_copy) ARGS=(trainer.synaptic_scaling=false trainer.prune_reinit=true
                    trainer.prune_args.reinit_mode=copy) ;;
  scale)      ARGS=(trainer.prune_reinit=false trainer.synaptic_scaling=true
                    "trainer.scaling_args.eta=${ETA:-0.05}"
                    trainer.scaling_args.preserve_row_norm=true) ;;
esac

TAG="${SWEEP:-ff_revive}/EqType=h_k=${K}_N=${N}_pen=none_arm=${NAME}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_${SWEEP:-ff_revive}"
echo "Task ${SLURM_ARRAY_TASK_ID}: arm=${NAME} N=${N} gamma=${GAMMA} rep=${REP} iters=${IT}"
echo "  args: ${ARGS[*]}"
echo "  -> ${SAVE_TO}/NBitFlipFlop_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=NBitFlipFlop \
  task.n_inputs="$K" task.n_outputs="$K" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" model.gamma="$GAMMA" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  "${ARGS[@]}" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
