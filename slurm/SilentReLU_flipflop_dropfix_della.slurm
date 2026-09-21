#!/bin/bash
# THE CORRECTED DROPOUT SWEEP, STAGE 1: N = 1000, 4 drop rates x 3 betas x 3 seeds (DELLA)
#
# WHY THIS SUPERSEDES THE DROP-RATE LADDER. Every dropout run before commit f151570 sampled units
# with a score computed on the RAW STATES, which for equation_type "h" are pre-activations:
#   v_i = q_0.9(|h_i|) + std(|h_i|).
# That score is EVEN in h, so a unit pinned at h = -c (permanently silent, r = 0) scored identically
# to one pinned at h = +c (maximally active). Consequences, all measured on the trained nets:
#   - Spearman rho between the sampler score and the rate-based participation: 0.36 +- 0.10 (n = 4)
#     over all units, and -0.04 +- 0.05 among the ACTIVE units - i.e. late in training the sampler's
#     ordering carried NO information about which living unit was busiest.
#   - 51 +- 4% of the drops landed on units that were already silent, where dropping is provably
#     inert (max|dh| over every other unit = 0.000e+00, both kinds).
#   - Instrumenting a real train_step for 1000 iterations: the waste is already 36 +- 8% by
#     iteration 100, and selectivity (p_drop given silent / given active) is 1.23 +- 0.15, i.e. the
#     sampler actively PREFERRED silent units early on.
# A second, independent defect made beta uninterpretable. The mask was an independent Bernoulli with
# p_i = clamp(drop_rate*N*softmax(beta*v)_i, 0, 0.999), and a saturated unit's excess above 0.999 was
# discarded rather than redistributed, so raising beta silently cut the DOSE:
#   beta = 1 -> 49.1 units dropped, beta = 2 -> 33.3, beta = 4 -> 7.9, beta = 8 -> 2.1 (nominal 50).
# The ladder's beta = 4 arm (394.7 +- 71.7, against no-dropout 387.0 +- 37.3) was therefore not
# "sharper targeting is worse"; it was dropout almost switched off. That arm is VOID as a test of
# targeting, and the targeting axis is unexplored rather than explored-and-null.
#
# WHAT THE SAMPLER DOES NOW (f151570, 9a05f27, a38eadd):
#   1. score  v_i = std(r_i) + q_{activity_q}(r_i) on the FIRING RATE r = activation(h), the same
#      quantity every active-unit count in the paper uses. activity_q is finally honoured.
#   2. pool   only the live units, {i : v_i >= active_rel * q95(v)} - the paper's scale-free
#      criterion. Wasted drops are 0 by construction rather than by tuning beta.
#   3. draw   EXACTLY k = round(drop_rate * |pool|) units, without replacement, with probability
#      proportional to softmax(beta * v) over the pool (torch.multinomial). Dose is fixed at every
#      beta, so beta now does one job: how sharply to prefer the busiest of the living.
#   drop_rate multiplies the LIVE POOL, not N. drop_rate * N is unsafe here - a 1000-unit net has
#   ~260 live units late in training, so 0.25 * N would ablate most of the working network.
# Verified on Della (A100-80GB, torch 2.7.0+cu128) that torch.multinomial with a CUDA generator
# draws exactly k at beta = 1, 4, 16 and is seed-reproducible.
#
# GRID (36 jobs) = 4 drop rates x 3 betas x 3 seeds, at N = 1000, k = 3, pen = none, do = dead.
#   drop_rate in {0.05, 0.10, 0.175, 0.25}   (fraction of the LIVE pool)
#   beta      in {1, 2, 4}
# `mute`, NOT `dead`, and this is not a preference - `dead` is unstable under accurate targeting.
# With `dead` the unit is removed from the recurrent rhs, and since the task loss is scored on the
# dropout pass, a unit drawn on a fraction p of iterations contributes gradient on only (1-p) of
# them. As p -> 1 it is unregularised and runs away. Measured p(busiest unit dropped) under rank
# scoring is ~ beta * drop_rate, so the grid's own corner is dangerous:
#       rho\beta      1        2        4
#       0.05       0.078    0.140    0.200
#       0.10       0.178    0.238    0.376
#       0.175      0.276    0.390    0.584
#       0.25       0.386    0.484    0.772
# At rho=0.25 beta=4 all three seeds blew up within 900 iterations (r2 -42521, -884, -24146 with a
# healthy train loss), so rank scoring bounds the problem but does NOT remove it.
# `mute` is immune by construction: RNN_torch.rhs masks nothing unless dropout_kind == "dead"
# (line ~489), so a muted unit still drives every other unit through W_rec. Its activity therefore
# reaches the output through its neighbours, the task loss sees it, and it cannot grow unopposed.
# Only its own read-out column is masked. A `mute` reference at n=7 already exists on disk.
# The old rationale for `dead`, retained for the record: `mute` is a
# stage-2 question: with silent units out of the pool the two kinds are no longer equivalent on the
# dropped units (they were identical whenever a silent unit was drawn, which was half the time).
# Index decode (TID = SLURM_ARRAY_TASK_ID - 1, 0..35):
#   REP = TID % 3;  ARM = TID / 3;  B_IDX = ARM % 3;  R_IDX = ARM / 3
#
# PRE-REGISTERED, WRITTEN BEFORE ANY OF THESE JOBS EXISTED. Read-out at 40,000 iterations, active
# units under the scale-free criterion (0.05 * q95), from the participation trace. Reference cells
# already on disk at the same 40k read-out, and still valid: the only training-path commits between
# them and now are the `scored_` refactor (identical for a time-index mask, which this task uses)
# and the OOM chunking (exact in float32).
#   no dropout                      387.0 +- 37.3  (n = 4)
#   dropout mute, rate 0.05, beta 1 -- read from disk at the same 40k point; the `mute` cell has
#     n = 7 and is the matched reference now that the sweep is `mute`.
#   dropout dead, rate 0.05, beta 1 487.0 +- 28.0  (n = 7)   <- the old `dead` reference
# ⚠️ BIAS. Every reference above was trained with a TRAINABLE bias in [-1, 1]; as of commit 4f7d73f+
# rnn_relu_standard.yaml fixes the bias at 0. Fixing the bias was separately measured to be worth
# +5.0 units, but under an archived peak-rate criterion, so it is not a clean zero on this axis.
# Matched bias-free controls (no dropout, and mute at rate 0.05 beta 1) MUST be run alongside this
# grid; do not compare a bias-free arm against the with-bias references above.
#   BAR: an arm WINS only if its mean clears 571 = 487.0 + 3 * 28.0. Same bar as the drop-rate
#   ladder, reused unchanged so the two are directly comparable.
#   COST GUARD: an arm that clears the bar but whose clean training loss at 40k exceeds the
#   no-dropout reference by more than 10% is reported as a trade, not a win.
#   If NO arm clears the bar, the conclusion is that dropout's ceiling is structural and survives
#   the correction of both defects - which is a stronger negative than the one the paper has now,
#   and it is what will be written.
#
# STAGE 2, ALREADY DECIDED: the same grid at N = 500 and N = 2000 (72 further jobs), to test whether
# any benefit scales with network size. Run only if stage 1 produces an arm worth scaling; if
# nothing clears the bar, stage 2 is a single confirmatory arm rather than the full grid.
#
# Cost: dropout runs two forward passes per iteration, ~0.5-0.6 s/iter at N = 1000, so 40k is
# ~6-7 h. 12 h requested so a slow node cannot time out - a timeout loses the ParticipationTrace
# entirely, since it is written only when the run completes.
#   sbatch --array=1-36%12 slurm/SilentReLU_flipflop_dropfix_della.slurm
#   DROP_KIND=dead sbatch ...        # the unstable kind; see below before using it
# Timing calibration: ITERS_OVERRIDE=600 sbatch --array=1 --time=01:00:00 ...
#
# Output: $SAVE_TO/NBitFlipFlop_std_dropfix/EqType=h_k=3_N=1000_pen=none_do=dead_rate=<r>_beta=<b>/

#SBATCH --job-name=FFdropfix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
# Named gres TYPE, never --gpus-per-node, so a MIG slice can never satisfy this.
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=12:00:00
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
# Provenance, and it is FATAL rather than a warning: a run whose code cannot be identified cannot be
# reproduced or retracted. Prefer the git checkout (the repo is synced by push/fetch, not rsync);
# fall back to CODE_VERSION.txt only for a tree that was rsynced.
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
N=1000
KIND=${DROP_KIND:-mute}
IT=${ITERS_OVERRIDE:-40000}

ARM_RATE=(0.05 0.10 0.175 0.25)
ARM_BETA=(1 2 4)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); ARM=$((TID / 3)); B_IDX=$((ARM % 3)); R_IDX=$((ARM / 3))
RATE=${ARM_RATE[$R_IDX]}
BETA=${ARM_BETA[$B_IDX]}

TAG="std_dropfix/EqType=h_k=${K}_N=${N}_pen=none_do=${KIND}_rate=${RATE}_beta=${BETA}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_std_dropfix"
echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K} N=${N} do=${KIND} rate=${RATE} beta=${BETA} rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/NBitFlipFlop_${TAG}"

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
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  trainer.dropout=true \
  "trainer.dropout_args.dropout_kind=${KIND}" \
  "trainer.dropout_args.drop_rate=${RATE}" \
  "trainer.dropout_args.dropout_beta=${BETA}" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
