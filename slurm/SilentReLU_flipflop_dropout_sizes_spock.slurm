#!/bin/bash
# DOES THE DROPOUT RECRUITMENT EFFECT HOLD AT OTHER NETWORK SIZES? N=500 and N=2000 (SPOCK)
#
# The N=1000 result (arrays 6201944 + 6218721, 7 seeds per condition, read out 2026-09-20):
# on the 3-bit flip-flop, unpenalised, dropout raises the live-unit count from 263 +/- 13 to
# 373 +/- 18 (mute) and 362 +/- 8 (dead), Welch p 1.4e-07 and 2.2e-08, and `dead` costs NOTHING
# measurable (clean loss 0.02590 +/- 0.00029 vs the reference 0.02589 +/- 0.00005, p = 0.95).
# The silencing SLOPE was unchanged in this arm (-210 vs -223/-229 units per decade, p ~ 0.45), so
# at N=1000 dropout is a pure OFFSET on the recruitment curve, not a change in its rate.
#
# Pavel (2026-09-20): repeat the cell at N=500 and N=2000. Unpenalised only - the frm/both arms sit
# at the N ceiling with and without dropout and cannot move on this question, and rws is not needed
# because the question here is the SIZE dependence of the basic effect, not the penalty interaction.
#
# GRID: 42 jobs = 2 sizes x 3 dropout conditions x 7 seeds. Everything else matches the N=1000 cell
# exactly (k=3, equation h, relu, fresh batch, 150k iterations, lambda_rws=0, lambda_frm=0, dropout
# sampling/rate from the trainer config: participation, 0.05).
#   N    in {500, 2000}                 N_IDX = (TID / 3) % 2
#   kind in {none, mute, dead}          K_IDX = TID % 3
#   rep  0..6                           REP   = TID / 6
#   (TID = SLURM_ARRAY_TASK_ID - 1; tasks 1-6 are one full rep of the 6 cells, and so on.)
#
# SELF-CONTAINED: each size gets its OWN 7 no-dropout controls (kind=none) from this same launcher,
# so no historical reference cell is needed and no cross-commit pooling argument has to be made -
# unlike the N=1000 cell, where 3 of the 7 controls are historical and their poolability had to be
# verified separately (it passed: 0.1-0.7 pooled sd, 2026-09-18).
#
# PRE-REGISTERED READ-OUT at 150k, per size, identical to the N=1000 one:
#   live units under BOTH criteria (scale-free and the flip-flop absolute 4e-2) and the clean loss.
#   "Dropout keeps units alive" := the dropout cell MEAN exceeds the same-size no-dropout mean by
#   more than 3 no-dropout sd under BOTH criteria. Welch reported as descriptive.
#   SECONDARY: the late silencing slope (dropout_silencing_slope.py) per size - at N=1000 dropout
#   changed the offset and not the rate, and whether that is size-invariant is the question this
#   array actually adds. A rate change appearing at N=2000 would mean the N=1000 "pure offset"
#   reading is a small-network special case.
#   THIRD: does the EFFECT SIZE scale with N? At N=1000 it is +99 to +110 units, i.e. ~+10% of N.
#   If it is a constant FRACTION of N the gap should be ~+50 at N=500 and ~+200 at N=2000; if it is
#   a constant COUNT it should be ~+100 at both. These are distinguishable with 7 seeds a side
#   (the N=1000 sds are 8-18 units).
#
# ⚠️ NOTHING IS WRITTEN UNTIL A JOB COMPLETES, so a job that hits the wall limit wastes its whole
# allocation. WALL TIME IS MEASURED, NOT EXTRAPOLATED (calibration arrays 6306867 / 6306885, 300
# iterations each on an L40S, dropout on and off):
#     N=500  no dropout  0.165 s/iter ->  6.9 h      N=500  dropout  0.304 s/iter -> 12.7 h
#     N=2000 no dropout  0.471 s/iter -> 19.6 h      N=2000 dropout  0.886 s/iter -> 36.9 h
# Longest job 36.9 h; 60 h requested. Whole array ~880 GPU-h (N=2000 cells are 654 of it).
#
# ⚠️ PINNED TO L40S (--gres=gpu:L40S-46G:1), the GPU every rate above was measured on. Spock also
# has 18 A100-40G nodes, but (a) A100 has been ~1.85x slower than L40S on this task family, which
# would put N=2000+dropout near 68 h, and (b) its 40 GB is below the 44 GB card on which the
# pre-fix N=2000 dropout run died, so its headroom after the fix is unverified. The %14 throttle is
# below the 16 L40S GPUs, so pinning costs no throughput - it only waits on other users' L40S jobs.
#
# ⚠️ REQUIRES THE CHUNKED get_participation_ (local commit 74ec3dd). WITHOUT IT EVERY N=2000
# DROPOUT JOB DIES IN ~30 s with CUDA OOM: the sampler's per-unit quantile over (2000, 307200)
# asked for 6.9 GB on top of a 2.5 GB tensor and its copy (calibration job 6306867_4). Spock's
# working copy carries that one file; `git status` there shows Trainer.py modified, and the header
# check below will warn accordingly. Chunking is exact (bit-identical at chunk 512/128/37, and
# same-seed 15-iteration N=1000 runs are bit-identical before and after for both dropout kinds), so
# these runs remain comparable to the existing N=1000 cells.
#
#   sbatch --array=1-42%14 slurm/SilentReLU_flipflop_dropout_sizes_spock.slurm
#
# Output: $SAVE_TO/NBitFlipFlop_std_dropout/EqType=h_k=3_N=<N>_pen=none_do=<none|mute|dead>/
# (the same sweep folder as the N=1000 cells; the cell name carries N, so nothing collides)

#SBATCH --job-name=FFdropN
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:L40S-46G:1
#SBATCH --mem=48G
#SBATCH --time=60:00:00
#SBATCH --array=1-42%14
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

K=3
N_LIST=(500 2000)
DO_KIND=(none mute dead)
IT=${ITERS_OVERRIDE:-150000}

TID=$((SLURM_ARRAY_TASK_ID - 1))
K_IDX=$((TID % 3)); N_IDX=$(((TID / 3) % 2)); REP=$((TID / 6))
N=${N_LIST[$N_IDX]}; KIND=${DO_KIND[$K_IDX]}

if [ "$KIND" = "none" ]; then
  DO_ARGS=(trainer.dropout=false)
else
  DO_ARGS=(trainer.dropout=true "trainer.dropout_args.dropout_kind=${KIND}")
fi

TAG="std_dropout/EqType=h_k=${K}_N=${N}_pen=none_do=${KIND}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_std_dropout"
echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K} N=${N} pen=none dropout=${KIND} rep=${REP} iters=${IT}"
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
  "${DO_ARGS[@]}" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
