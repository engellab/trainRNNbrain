#!/bin/bash
# DOES TARGETED DROPOUT RECRUIT A FRACTION OF THE NETWORK, OR A FIXED COUNT? (DELLA)
#
# THE QUESTION. At N=1000 on the 3-bit flip-flop, `dead` dropout at a 25% rate with beta=4 reached
# 963.3 +- 3.2 of 1000 active units and at 17.5% reached 809.7 +- 10.2, against a no-dropout
# control of 316.3 +- 33.3. Those are close enough to the 1000-unit ceiling that ONE SIZE CANNOT
# DISTINGUISH TWO DIFFERENT MECHANISMS: "96% of units" predicts ~1920 active at N=2000, while
# "about 960 units" predicts ~960. A scaling figure cannot use the result until that is settled,
# and a constant fraction and a constant count are different findings about what dropout does.
#
# GRID (12 jobs) = 2 kinds x 2 sizes x 3 seeds, at drop rate 0.20 and beta=4.
#   kinds  mute, dead        sizes  N = 500, 2000        rate 0.20 of the LIVE pool, beta 4
#   Index decode (TID = SLURM_ARRAY_TASK_ID - 1, 0..11):
#     REP = TID % 3;  REST = TID / 3;  SIZE = REST % 2;  KIND = REST / 2
#   sbatch --array=1-12 slurm/SilentReLU_dropout_sizes_della.slurm
#
# WHY 0.20 AND NOT A CELL THAT WAS ALREADY RUN. 0.20 sits between the 17.5% and 25% rungs, both of
# which are in `dead`'s steep region at beta=4 (810 and 963 units). Picking the midpoint keeps the
# N=500 and N=2000 runs away from the ceiling at the larger size while staying inside the regime
# where the effect is large, so a fraction and a count are still separable at N=2000. It is a NEW
# rate, so N=1000 is not available for it and is not claimed here; the N=1000 anchors for the
# fraction-vs-count reading are the 17.5% and 25% cells already on disk.
#
# CONTROLS ARE NOT RUN HERE, deliberately. The `none` arm of the flip-flop revival sweep
# (NBitFlipFlop_ff_revive, job 14310257) provides N = 500, 1000 and 2000 on this same task, same
# 40,000 iterations, same model config and the SAME COMMIT as these jobs, so it is a matched
# control rather than an archived one. That sweep also gives a free consistency check: its N=1000
# `none` cell must land near the dropout sweep's 316.3 +- 33.3. If it does not, the controls are
# not comparable across commits and these numbers need their own controls before they mean
# anything -- read that cell BEFORE reading these.
#
# PRE-REGISTERED, WRITTEN BEFORE ANY OF THESE JOBS EXISTED. Read-out at 40,000 iterations, active
# units under the scale-free criterion (0.05 * q95), from the participation trace.
#   PRIMARY   the active-unit count as a FRACTION of N at each size. Constant fraction across
#             500/2000 -> dropout recruits a proportion of the network. Constant count -> it
#             recruits a fixed number and the fraction result at N=1000 was a ceiling artifact.
#             Neither, with a trend -> report the exponent and do not claim either.
#   CO-PRIMARY clean dropout-off loss (loss_clean_train), NOT the training-pass loss, which is
#             structurally blind to this failure class.
#   STABILITY any cell discarding more than 5% of its gradient updates is reported as broken
#             whatever its unit count. An earlier generation of this sweep had `dead` seeds blow up
#             at high drop rates with a healthy-looking training loss, and `dead` is the arm at
#             risk here: it removes a dropped unit from the recurrent right-hand side, so a unit
#             drawn on a fraction p of iterations is unregularised on those iterations.
#   EXPECTED DIRECTION, stated so it can be wrong: `mute`'s N=1000 cells scale smoothly with rate
#             and never exceed the control's loss, so I expect it to hold a roughly constant
#             fraction. `dead` turns on sharply between beta=2 and beta=4 and costs loss at every
#             rate above 5%, so I expect it to be the one that fails to scale, and to cost more
#             loss at N=2000 than at N=500.
#
# Cost: dropout runs two forward passes per iteration. ~1h25m at N=1000, so ~45m at N=500 and
# ~3h at N=2000. 6 h requested.
# Output: $SAVE_TO/NBitFlipFlop_dropout_sizes/EqType=h_k=3_N=<n>_pen=none_do=<kind>_rate=0.20_beta=4/

#SBATCH --job-name=DOsizes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=48G
# TIME LIMIT, set from measurement rather than estimate: N=2000 reached only 23,124 of 40,000 iterations in 6h05m, so 40k needs ~10.5 h; dropout runs two forward passes per iteration.
# The first submission used the line below this one and lost every N=2000 cell to TIMEOUT.
#SBATCH --time=20:00:00
#SBATCH --array=1-12%12
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
RATE=${RATE:-0.20}
BETA=${BETA:-4}
ARM_KIND=(mute dead)
ARM_SIZE=(500 2000)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); REST=$((TID / 3)); S_IDX=$((REST % 2)); K_IDX=$((REST / 2))
KIND=${ARM_KIND[$K_IDX]}
N=${ARM_SIZE[$S_IDX]}
# N_OVERRIDE runs this same grid at a size the ARM_SIZE table does not list. Use a distinct
# SWEEP when doing so, or cells at different sizes land in folders that differ only by a
# field the analysis does not group on.
N=${N_OVERRIDE:-$N}

TAG="${SWEEP:-dropout_sizes}/EqType=h_k=${K}_N=${N}_pen=none_do=${KIND}_rate=${RATE}_beta=${BETA}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_${SWEEP:-dropout_sizes}"
echo "Task ${SLURM_ARRAY_TASK_ID}: kind=${KIND} N=${N} rate=${RATE} beta=${BETA} rep=${REP} iters=${IT}"
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
  trainer.dropout_args.rescale=true \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
