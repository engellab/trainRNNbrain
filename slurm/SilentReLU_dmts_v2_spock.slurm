#!/bin/bash
# DMTS v2: the REDESIGNED delayed match-to-sample, all four penalty arms x four sizes (SPOCK)
#
# WHY v2. The old task spent one input channel per stimulus identity on FOUR stimuli, a legacy of
# the era when inputs were constrained non-negative (io_nonnegativity is false now). Two problems
# followed, and the second is the serious one:
#   1. 4 channels carried 2 bits.
#   2. get_batch enumerated all 4x4 = 16 (sample, match) pairs, of which only the 4 DIAGONAL ones
#      are a match. The task was 25% match / 75% non-match, so a network that always answered
#      "non-match" scored 75%. Every DMTS number in the paper was measured on that imbalance.
# v2 uses TWO stimuli, giving 2x2 = 4 pairs, 50/50 balanced. The memory load drops from 2 bits to 1,
# which is acceptable: one bit held across 16 tau of silence still cannot be carried by a transient,
# and being un-carryable-by-transients is the only reason this task is in the paper.
#
# CHANNELS (n_inputs = 4):
#   ch0, ch1   the two stimulus identities; a +1 pulse at the sample (20-40) and at the match
#              (200-220), onsets jittered +/-1 tau so the task cannot be solved by counting
#   ch2        tonic drive, constant 1.0 for the whole trial
#   ch3        decision cue, 240-300
# Batch: num_rep=256 x 4 conditions = 1024 trials, matching the flip-flop and CDDM sweeps.
#
# ⚠️ THE TONIC CHANNEL IS A CONFOUND AND MUST BE REPORTED AS ONE. A constant input reaches unit i
# as W_inp[i,2] at every timestep, which is mathematically a bias - and the bias was deliberately
# fixed at 0 for every other task on 2026-09-21. It sets an operating point and cannot carry memory
# (it is identical on every trial). Since every measure in this project is a COUNT OF NON-SILENT
# UNITS, a constant drive is a confound on the main measure, and DMTS is now the only task with one.
# Bound on the effect: a trainable bias in [-1,1] was worth +5.0 active units against a bias fixed
# at 0, so this should be small - but if DMTS lands anomalously high on active units relative to the
# flip-flop and CDDM, re-run pen=none with `task.tonic=false` before believing it. 3 jobs.
#
# EVERY PRE-2026-09-21 DMTS RUN USED THE 4-STIMULUS LAYOUT AND HAS BEEN DELETED. Nothing from the
# old series may be pooled with this one; the output folder is DMTS_v2_pen, not DMTS_std_pen, so
# the two cannot mix even by accident.
#
# GRID (48 jobs) = 4 sizes x 4 penalty arms x 3 seeds, h equation, standard ReLU (bias fixed at 0),
# fresh batches.
#   N   in {500, 1000, 2000, 4000}
#   pen in {none, rws, frm, both}   rws 0.05, frm 0.1, the project's standard values
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..47): REP=TID%3; P_IDX=(TID/3)%4; N_IDX=TID/12
#   tasks 1-12 N=500, 13-24 N=1000, 25-36 N=2000, 37-48 N=4000; within a size: 3 none, 3 rws,
#   3 frm, 3 both.
#
# ITERATIONS. 150k at N <= 2000, because the frm arm converges slowly and the penalty table needs
# the headroom. 100k at N=4000: that cell exists only for the Figure 1c scaling point, which reads
# every cell at a MATCHED 100,000 anyway, and 150k there would be ~54 h median against Spock's 96 h
# cap. anneal_noise is false, so nothing in the trajectory depends on max_iter and a 100k run is
# exactly the first 100k of a 150k one.
#
# Cost (batch 1024, T=300, measured on the old series which had the same shape): ~7 h at N=500,
# ~11 h at N=1000, ~29 h at N=2000, ~36 h at N=4000/100k. 72 h requested so a slow node cannot time
# out - a timeout loses the ParticipationTrace entirely, since it is written only on completion.
#   sbatch --array=1-48%16 --time=72:00:00 slurm/SilentReLU_dmts_v2_spock.slurm
# Timing calibration: ITERS_OVERRIDE=600 sbatch --array=37 --time=02:00:00 ...   # the N=4000 cell
#
# Output: $SAVE_TO/DMTS_v2_pen/EqType=h_N=<N>_pen=<name>/<per-net subfolder>

#SBATCH --job-name=DMTSv2
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=1-48%16
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

NS=(500 1000 2000 4000)
PEN_NAME=(none rws frm both)
PEN_RWS=(0 0.05 0 0.05)
PEN_FRM=(0 0 0.1 0.1)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); P_IDX=$(((TID / 3) % 4)); N_IDX=$((TID / 12))
N=${NS[$N_IDX]}
# 150k except the N=4000 scaling cell, which is read at a matched 100k regardless (see header)
if [ "$N" -ge 4000 ]; then IT=100000; else IT=150000; fi
PNAME=${PEN_NAME[$P_IDX]}; LRWS=${PEN_RWS[$P_IDX]}; LFRM=${PEN_FRM[$P_IDX]}
IT=${ITERS_OVERRIDE:-$IT}

TAG="v2_pen/EqType=h_N=${N}_pen=${PNAME}"
mkdir -p "${SAVE_TO}/DMTS_v2_pen"
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
