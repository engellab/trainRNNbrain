#!/bin/bash
# THE RATE PENALTY, ALONE AND WITH THE SPARSITY PENALTY, ON THE STANDARD NETWORK (DELLA)
#
# The Della counterpart of SilentReLU_std_penalties_spock.slurm (5 Nov 2026; the
# uniform-interventions programme, engellab/trainRNNbrain issue #6, docs/interventions_matrix.md).
# The Spock launcher is unchanged. It trained CDDM_std_g0_penalties before 2026-09-21, with a
# trainable bias, trainer_ptrack_plain and 200,000 iterations, and its provenance check reads the
# CODE_VERSION.txt of a tree copied by rsync. This launcher runs the same three penalty arms on the
# standard network (configs/model/rnn_relu_standard.yaml: bias fixed at 0) with
# trainer_ptrack_freshbatch, the trainer of the CDDM paper-grid control and of every 3BFF control,
# at the 75,000-iteration same-size budget of silent_units_largeN, from a clean git worktree.
#
# The values: lambda_frm 0.1 and lambda_rws 0.05 are the 3BFF and maze values
# (NBitFlipFlop_remedy_grid arm frm0.1, NBitFlipFlop_ff_both40k arm both, Maze_interventions
# frm0.1 and frm0.1_rws0.05). The standard CDDM cells that exist (CDDM_std_g0_bias0_c75000) have
# lambda_frm 0.2, so the 0.1 cells here put the three tasks at one setting.
#
# Arms (3 seeds each, seed=random; N = 1000 unless N_OVERRIDE is set):
#   PEN 0 = rws   lambda_rws=0.05 lambda_frm=0      indices 1-3
#   PEN 1 = frm   lambda_rws=0    lambda_frm=0.1    indices 4-6
#   PEN 2 = both  lambda_rws=0.05 lambda_frm=0.1    indices 7-9
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..8): REP = TID%3; PEN = TID/3.
# TASK=cddm (default) passes task=CDDM; TASK=ff passes the 3-bit flip-flop words
# (task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3). No control is run here: the controls are
# CDDM_paper_grid/EqType=h_N=1000_arm=control and, on 3BFF, the B5 default cell
# NBitFlipFlop_b5_optimizer/EqType=h_N=1000_opt=adam_wd=1e-6_lr=1e-3.
#   TASK=cddm sbatch --chdir=$REPO --time=08:00:00 --constraint=nomig --array=4-9 slurm/SilentReLU_std_penalties_della.slurm
# Output: $SAVE_TO/<Task>_std_g0_penalties_bias0/EqType=h_N=<N>_pen=<rws0.05|frm0.1|frm0.1_rws0.05>_iters=<IT>/
# Cost: the standard trainer adds the penalties to the task loss (no extra backward pass); on
# CDDM_std_g0_bias0 (50,000 iterations, N = 1000) the frm 0.2 arms took 2:04-2:30 and the
# unpenalised arms 1:52-1:59, so 75,000 iterations take about 4 h and 8 h is the request.

#SBATCH --job-name=StdPen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=32G
#SBATCH --time=08:00:00
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

REPO="${REPO:-$HOME/trainRNNbrain_pen}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2
  echo "       submit with --chdir=\$REPO; python -c puts the working directory ahead of PYTHONPATH." >&2
  exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
if [ -z "$GITSHA" ]; then
  echo "FATAL: $REPO is not a git checkout - provenance of this run cannot be recorded." >&2; exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

N=${N_OVERRIDE:-1000}
IT=${ITERS_OVERRIDE:-75000}

# TASK selects the task words and the folder prefix (see the header).
case "${TASK:-cddm}" in
  cddm) TASK_ARGS=(task=CDDM);                                          PREFIX=CDDM ;;
  ff)   TASK_ARGS=(task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3); PREFIX=NBitFlipFlop ;;
  *) echo "FATAL: TASK must be cddm or ff; got '${TASK}'" >&2; exit 1 ;;
esac

PEN_TAG=(rws0.05 frm0.1 frm0.1_rws0.05)
PEN_RWS=(0.05    0      0.05)
PEN_FRM=(0       0.1    0.1)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); PEN=$((TID / 3))
PNAME=${PEN_TAG[$PEN]:?array index out of range}; LRWS=${PEN_RWS[$PEN]}; LFRM=${PEN_FRM[$PEN]}

TAG="std_g0_penalties_bias0/EqType=h_N=${N}_pen=${PNAME}_iters=${IT}"
mkdir -p "${SAVE_TO}/${PREFIX}_std_g0_penalties_bias0"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${TASK:-cddm} N=${N} pen=${PNAME} (rws=${LRWS} frm=${LFRM}) rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/${PREFIX}_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  "${TASK_ARGS[@]}" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws="$LRWS" trainer.lambda_frm="$LFRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
