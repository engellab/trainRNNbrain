#!/bin/bash
# SCREEN: THREE NON-PENALTY INTERVENTIONS, SINGLY AND IN PAIRS — CDDM, N=1000, NO PENALTIES (DELLA)
#
# This is a SCREEN, not a confirmatory sweep: n = 3 per arm, one task, one size. It answers "does
# this do anything at all", and nothing more. No arm graduates to a claim on this evidence.
#
# THE THREE INTERVENTIONS, and why none of them is a penalty. A silent ReLU unit has EXACTLY zero
# gradient on every weight into and out of it (tests/test_prune_and_reinit.py), so a term added to
# the loss cannot revive one - it can only act before the unit dies. All three of these act outside
# the loss, on the weights directly, which is the one place the frozen state is reachable from.
#   cap   - inp_weight_cap: hard clamp on |W_inp| after each step. Tests whether CONCENTRATED input
#           drive is what leaves units silent. The unpenalised net puts its drive into ~6% of its
#           input weights (99th pct 8.57 vs median 0.002) and row norm predicts participation at
#           r = +0.65; under frm+rws that inverts to -0.61. Cap 0.3 is the measured frm+rws q90 at
#           N=1000 (free fit 41.7*N^-0.709 = 0.31), and sits ~10x above the init scale 1/sqrt(N).
#   prune - prune_and_reinit_: redraw the INCOMING weights of a unit silent for patience*100 iters.
#           The only mechanism here that can reach a unit with no excitatory input left.
#   scale - synaptic_scaling_: multiplicative, excitation UP and inhibition DOWN, toward the median
#           participation of the live pool. The sign split is not cosmetic: scaling a net-inhibited
#           unit's whole row UP makes its drive MORE negative and buries it deeper (measured in
#           tests/test_synaptic_scaling.py: h -30.25 -> -19.90 sign-split, vs -45.37 naive).
#
# WHY PAIRS ARE IN THE SAME GRID. `scale` provably cannot rescue a unit whose excitatory input is
# gone - shrinking inhibition drives h toward 0 from below but never across it (measured: h =
# -0.0000 after 30 scalings, still negative). `prune` can, but hands back an untrained random row.
# They address disjoint failure modes, so prune+scale should beat either alone. That is a real
# prediction with a real way to lose: if the pair is no better than the best single, the two are
# redundant and the "disjoint failure modes" story is wrong.
#
# ARMS (8) x 3 seeds = 24 jobs. ARM = TID/3, REP = TID%3.
#   0 none   1 cap   2 prune   3 scale   4 cap+prune   5 cap+scale   6 prune+scale   7 all three
#   Singles + control:  sbatch --array=1-12  slurm/SilentReLU_screen3_della.slurm
#   Pairs + triple:     sbatch --array=13-24 slurm/SilentReLU_screen3_della.slurm
#   Rank-matched demo:  sbatch --array=25-27 slurm/SilentReLU_screen3_della.slurm
#
# ARM 8 (`scale_ln`) IS A DEMONSTRATION, NOT A SCREEN ARM. Measured on 5 nets per arm, the trained
# networks do not match cortical statistics in the RATE distribution, and the two existing arms
# fail in opposite directions: `none` carries a 31% atom at exactly zero with a 5.8-decade spread,
# `frm+rws` has no zeros but only 0.8 decades, i.e. near-homogeneity. Cortex has no atom at zero
# and ~2 decades. The WEIGHTS need no fixing -- they are already lognormal at 2.5 decades in both
# arms, and the penalties only rescale them (~1.8x) without changing sigma_log at all.
# The single set-point is what homogenises: it asks the busiest unit in a lognormal population to
# fall from 58.0 to 1.3. A rank-matched target asks it for 68.3 instead, and asks each silent unit
# to rejoin the lower TAIL (0.025x-0.69x the median, never zero) rather than to become average.
# Read-out for this arm is the rate distribution itself -- zero atom and spread in decades, via
# weight_distribution_shape.py -- not the active count alone. Success is "a couple of networks show
# it is possible", which is all the discussion needs to claim.
# Stage the submission: run 1-12 first and only submit 13-24 for combinations whose singles did
# something. That is greedy forward selection, and it is deliberate - the point of the programme is
# the MINIMAL set of interventions that works, so an arm that adds nothing must be visibly dropped
# rather than absorbed into a stack of knobs that collectively "works".
#
# READ-OUT AT 30,000 ITERATIONS. NOT SHORTER, and this is the one design choice worth defending:
# an early read-out on this system is actively misleading, not merely noisy. In the archived ptrack
# traces (equation_type h, the config used here) the rws arm rises to 918 of 1000 active units at
# iteration 3620 - the best-looking arm on offer - and then collapses to 357.7 +- 13.2 by 30k,
# BELOW the 417.1 +- 19.0 of no penalty at all (Welch t = 9.96, p = 3.5e-10, n = 15). A screen that
# stopped at 5k would have selected the worst intervention in the set. Silencing has long latency
# and an intervention that merely DELAYS it is indistinguishable from one that PREVENTS it until
# late. Every arm here runs the full 30k for that reason.
#
# PRE-REGISTERED, WRITTEN BEFORE ANY OF THESE JOBS EXISTED.
#   PRIMARY   active units at 30k, scale-free criterion (0.05 * q95), from the participation trace.
#   CO-PRIMARY clean training loss at 30k (loss_clean_train: dropout-off, noise-free - NOT the
#             training-pass loss, which is structurally blind to this class of failure; a previous
#             result in this project was retracted for quoting the wrong one).
#   ADVANCE:  an arm advances to a full sweep if mean > control_mean + 2*control_sd AND clean loss
#             is within 10% of control. Two sd, not three, BECAUSE this is a screen - the cost of a
#             false positive here is one sweep, the cost of a false negative is abandoning a real
#             mechanism. The confirmatory bar stays at 3 sd.
#   TREADMILL (prune arms): the trace logs reinit_events and reinit_units_ever. events >> ever with
#             no gain in the active count means units are being redrawn and promptly re-dying. That
#             is a FAILURE, not partial success, and it is a live possibility: nothing stops a
#             redrawn unit being re-silenced by the same inhibition that killed it.
#   HOMOGENISATION TRADE (scale arms): scaling is two-sided, so it pulls busy units DOWN toward the
#             set-point as well as lifting quiet ones. If the active count rises while clean loss
#             degrades, the mechanism is buying units by flattening the representation, and that is
#             reported as a trade. It is the specific way this intervention is expected to fail.
#   NULL:     within control +- 2 sd -> no effect, written as such. A null on `cap` in particular is
#             informative: it would settle the causal direction left open by the input-weight
#             analysis - that spread-out input drive is a CONSEQUENCE of having many active units
#             rather than a cause of it.
# Control is run here rather than read off disk: the archived CDDM h/none cells disagree across
# architecture (g0 438.8 +- 6.8, g0_nodale 408.8 +- 11.4, g0_nodale_trainablebias 403.8 +- 13.6,
# n=5 each) and all three predate the bias-fixing change in rnn_relu_standard.yaml.
#
# Cost: no extra forward pass for any arm - both new mechanisms reuse the training pass's states and
# fire once per 100 iterations. ~30k at N=1000 is ~2-3 h; 6 h requested.
# Output: $SAVE_TO/CDDM_screen3/EqType=h_N=1000_pen=none_arm=<name>/

#SBATCH --job-name=Screen3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=06:00:00
#SBATCH --array=1-12
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
CAP=${CAP:-0.3}            # measured frm+rws q90 at N=1000; ~10x the init scale 1/sqrt(N)=0.032
PATIENCE=${PATIENCE:-5}    # 5 * check_every(100) = 500 iterations of silence before a redraw
ETA=${ETA:-0.05}           # scaling step size; alpha clipped to [1/(1+eta), 1+eta]

ARM_NAME=(none cap prune scale cap+prune cap+scale prune+scale all3 scale_ln prune_copy)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); ARM=$((TID / 3))
NAME=${ARM_NAME[$ARM]}

# Each mechanism is switched on by whether its name appears in the arm label. Defaults are OFF in
# the config, so an arm only ever enables what it names.
ARGS=(trainer.prune_reinit=false trainer.synaptic_scaling=false model.inp_weight_cap=null)
case "$NAME" in *cap*)   ARGS[2]="model.inp_weight_cap=${CAP}" ;; esac
case "$NAME" in *prune*) ARGS[0]="trainer.prune_reinit=true"; ARGS+=("trainer.prune_args.patience=${PATIENCE}") ;; esac
case "$NAME" in *scale*) ARGS[1]="trainer.synaptic_scaling=true"; ARGS+=("trainer.scaling_args.eta=${ETA}") ;; esac
case "$NAME" in all3)    ARGS=(trainer.prune_reinit=true trainer.synaptic_scaling=true
                               "model.inp_weight_cap=${CAP}"
                               "trainer.prune_args.patience=${PATIENCE}"
                               "trainer.scaling_args.eta=${ETA}") ;; esac
# Arm 8, the DEMONSTRATION arm for the discussion: scaling with a RANK-MATCHED lognormal target
# instead of a single set-point. Not part of the screen proper -- it exists to show that the
# measured departure from cortical statistics is fixable, not to compete with the other arms.
# Arm 9: revive a dead unit by DUPLICATING a live one instead of redrawing it at random. The
# random redraw failed in a specific, measured way -- 26,012 redraws over 591 units, 44 deaths
# each, active count unmoved -- and the obvious explanation is that a random unit has no function,
# so the task gradient has no reason to keep it. A copy has one by construction. The donor's
# outgoing weights are split between it and the copy, which perturbs the network output 21x less
# than a random redraw (tests/test_prune_copy_mode.py). If the copies ALSO die, the explanation is
# not "the new unit was useless" and the silent state is a property of the task, not the unit.
case "$NAME" in prune_copy) ARGS=(trainer.synaptic_scaling=false model.inp_weight_cap=null
                                  trainer.prune_reinit=true
                                  "trainer.prune_args.patience=${PATIENCE}"
                                  trainer.prune_args.reinit_mode=copy
                                  "trainer.prune_args.copy_noise=${COPY_NOISE:-0.05}") ;; esac
case "$NAME" in scale_ln) ARGS=(trainer.prune_reinit=false model.inp_weight_cap=null
                                trainer.synaptic_scaling=true
                                "trainer.scaling_args.eta=${ETA}"
                                trainer.scaling_args.scale_target=lognormal
                                "trainer.scaling_args.sigma_log=${SIGMA_LOG:-1.2}") ;; esac

TAG="${SWEEP:-screen3}/EqType=h_N=${N}_pen=none_arm=${NAME}"
mkdir -p "${SAVE_TO}/CDDM_${SWEEP:-screen3}"
echo "Task ${SLURM_ARRAY_TASK_ID}: arm=${NAME} rep=${REP} N=${N} iters=${IT}"
echo "  args: ${ARGS[*]}"
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
  "${ARGS[@]}" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
