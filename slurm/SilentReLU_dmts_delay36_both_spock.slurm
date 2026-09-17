#!/bin/bash
# DMTS WITH A 36-TAU DELAY, frm + rws ("both") ARM, THREE SEEDS (SPOCK)
#
# Follows the 2-job check (array 6201936, one seed each of `none` and `frm 0.1`, read out
# 2026-09-16): at a 36-tau silent delay the UNPENALISED network never reaches clean r2 = 0.9 in
# 150k iterations (best probe 0.682, 100% of late probes on the no-memory plateau at r2 = 0.605)
# while frm first crosses at 11 670 and spends 69% of its second half above it. That is the clean
# reversal of the 16-tau picture, where both arms escape and the penalty only wins on speed.
#
# Pavel's follow-up (2026-09-16): run the `both` arm (frm 0.1 + rws 0.05) at 36 tau with THREE
# seeds. It is the arm the 16-tau grid found most different from the others (at N=1000 it escapes
# the plateau as early as frm but then converges to its floor slowest, t90 27-29k; at N=2000 it
# ends at clean r2 0.60 with 13.7% unstable late probes), so whether it holds the 36-tau memory at
# all, and how stably, is not predictable from frm alone.
#
# Three seeds, N=1000, 150k iterations, everything else identical to the 2-job check, so the three
# arms (none / frm / both) sit in one folder and dmts_readout.py reads them together:
#   python dmts_readout.py data/trained_RNNs --sub DMTS_std_delay36 [--seeds]
#   python fig_dmts_delay36.py data/dmts_curves_delay36.npz
#
# PRE-REGISTERED READ-OUT (fixed before submission):
#   primary   t(clean r2 >= 0.9) per seed, and the fraction of the second half of training above
#             that level. `both` escapes in all three seeds -> the rate penalty, not the sparsity
#             penalty, is what finds the 36-tau memory (frm is in both arms; rws alone was last
#             everywhere at 16 tau). `both` fails in all three -> adding rws BREAKS the escape that
#             frm alone achieves, which would be a stronger statement than anything at 16 tau.
#             Mixed -> report as seed-dependent and do not claim a direction.
#   secondary live units at 150k under all three criteria (expected at the N ceiling, as every
#             frm-containing arm has been on every task so far), and `unstable`, the fraction of
#             late probes back at the plateau - the 36-tau frm seed sat at 30.9%, so instability,
#             not failure to learn, may be the real cost at this delay.
#   NOTE the saved 150k network is NOT the trained network on this task (the frm seed finished
#   inside a collapse). Read the TRACE, never the final net or its folder r2.
#
# Cost: the 36-tau frm seed ran 18.4 h and the none seed 14.5 h (T=500, ~0.44 s/iter at N=1000);
# 30 h requested, as for the first two.
#   sbatch --array=1-3 slurm/SilentReLU_dmts_delay36_both_spock.slurm
#
# Output: $SAVE_TO/DMTS_std_delay36/EqType=h_N=1000_pen=both/<per-net subfolder>

#SBATCH --job-name=DMTSd36b
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --array=1-3
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

N=1000
PNAME=both
LRWS=0.05
LFRM=0.1
IT=${ITERS_OVERRIDE:-150000}

TAG="std_delay36/EqType=h_N=${N}_pen=${PNAME}"
mkdir -p "${SAVE_TO}/DMTS_std_delay36"
echo "Task ${SLURM_ARRAY_TASK_ID} (seed ${SLURM_ARRAY_TASK_ID} of 3): N=${N} pen=${PNAME} (rws=${LRWS} frm=${LFRM}) iters=${IT} delay=36tau"
echo "  -> ${SAVE_TO}/DMTS_${TAG}"

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
