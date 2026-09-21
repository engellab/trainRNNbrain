#!/bin/bash
# DMTS at a 36-tau delay, v2 task: does the rate penalty rescue a task the baseline cannot learn?
#
# THE QUESTION. Every other cell in this project asks what a penalty COSTS on a task the
# unpenalised network already solves. This one asks the opposite. At a 16-tau delay the baseline
# solves DMTS; at 36 tau (T=500) the previous generation of this experiment found that it never
# did, in 3/3 seeds, while frm found the solution and then lost it and frm+rws never escaped. That
# result was measured on the OLD 4-stimulus task whose batch was 25% match / 75% non-match, so a
# network answering "non-match" always scored 75% and "never escapes" was partly a statement about
# a degenerate baseline. It has to be redone on the balanced task before it means anything.
#
# TASK. DMTS_delay36 with the v2 channel layout, identical to DMTS_long except for the timing:
#   ch0, ch1   two stimulus identities, +1 pulses at sample (20-40) and match (400-420)
#   ch2        tonic drive, constant 1.0
#   ch3        decision cue, 440-500
# 2x2 = 4 conditions, 50/50 balanced, num_rep=256 -> 1024 trials. Delay is 360 steps = 36 tau of
# silence, against 16 tau in DMTS_long. ⚠️ The tonic channel is mathematically a bias; see
# DMTS_long.yaml. It matters more here than anywhere else in the project, because this is the cell
# where the network has to hold a bit for 36 tau, so re-run pen=none with `task.tonic=false` before
# concluding anything about what sustains the memory.
#
# GRID (12 jobs) = 4 penalty arms x 3 seeds, N = 1000, h equation, bias fixed at 0.
#   pen in {none, rws, frm, both}   rws 0.05, frm 0.1
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..11): REP = TID%3; P_IDX = TID/3
#
# READ-OUT, fixed before submission. The headline is NOT an active-unit count, it is whether the
# task is learned at all: clean r2 on the scored epochs, per seed, read at 150,000. "Solved" :=
# clean r2 >= 0.9 in all 3 seeds. Because the baseline may genuinely fail, quote BOTH the matched
# compute value (median clean r2 over the last 500 probes) AND the best checkpoint, for every arm -
# the previous generation of this result was retracted precisely because one read-out was applied
# to the treatment and another to the control. A penalty "rescues" the task only if it is solved
# under the matched-compute read-out while none is not.
# Chance level on the balanced task is r2 = 0 (always answering one class), not 0.75 as it
# effectively was on the old imbalanced version.
#
# Cost: T=500 against DMTS_long's 300, so ~1.7x the per-iteration cost, ~18 h at N=1000 for 150k.
# 48 h requested; the ParticipationTrace is written only on completion.
#   sbatch --array=1-12 --time=48:00:00 slurm/SilentReLU_dmts36_v2_spock.slurm
#
# Output: $SAVE_TO/DMTS_v2_delay36/EqType=h_N=1000_pen=<name>/<per-net subfolder>

#SBATCH --job-name=DMTS36v2
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=1-12
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
PEN_NAME=(none rws frm both)
PEN_RWS=(0 0.05 0 0.05)
PEN_FRM=(0 0 0.1 0.1)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); P_IDX=$((TID / 3))
PNAME=${PEN_NAME[$P_IDX]}; LRWS=${PEN_RWS[$P_IDX]}; LFRM=${PEN_FRM[$P_IDX]}

TAG="v2_delay36/EqType=h_N=${N}_pen=${PNAME}"
mkdir -p "${SAVE_TO}/DMTS_v2_delay36"
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} pen=${PNAME} (rws=${LRWS} frm=${LFRM}) rep=${REP} iters=${IT}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=DMTS_delay36 \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws="$LRWS" trainer.lambda_frm="$LFRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
