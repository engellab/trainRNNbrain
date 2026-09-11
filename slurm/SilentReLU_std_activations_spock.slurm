#!/bin/bash
# E1 — DO SOFTPLUS, LEAKY-ReLU AND A BOUNDED SIGMOID SILENCE THE STANDARD CDDM NETWORK TOO? (SPOCK)
#
# The three non-ReLU activations were previously trained on CDDM only in the earlier architecture
# (2026-07-01 sweeps), and the bounded sigmoid only on the flip-flop (2026-09-11). The paper's claim
# C1 ("silencing afflicts any non-negative activation") is stated for the STANDARD network
# (rnn_relu_standard: signed weights, trainable bias, self-connections, gamma 0), so this puts all
# three activations on that network, on CDDM, with the exact training recipe of the unpenalised
# baseline (SilentReLU_std_gamma0_drift: trainer_ptrack_plain, 200k iterations at N=1000).
#
# Grid (9 jobs) = 3 activations x 3 seeds, N = 1000, h equation, no penalties.
#   ACT 0 = softplus(beta=25)          model=rnn_softplus_beta25_standard
#   ACT 1 = leaky-ReLU (leak 0.01)     model=rnn_leaky_relu_standard
#   ACT 2 = sigmoid(7.5 (x - 0.3))     model=rnn_sigmoid_shifted_standard
#   Index decode (TID = SLURM_ARRAY_TASK_ID-1): REP = TID%3; ACT = TID/3
#
# Read-out: the ReLU reference is CDDM_std_g0_drift/EqType=h_N=1000_iters=200000 (3 seeds), read
# from its participation trace at matched iteration. Use the scale-free criterion (softplus has no
# hard-silent unit) and, for the sigmoid, the modulation criterion alongside it
# (flipflop_sigmoid_silence.py has both). fig_C1_activations.py redraws its CDDM panel from these.
#
# Runtime ~8 h per job at N=1000 (measured on the drift sweep); softplus trained ~10% slower.
# Output: $SAVE_TO/CDDM_std_g0_activations/EqType=h_N=1000_act=<name>_iters=200000/<per-net subfolder>

#SBATCH --job-name=StdAct
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --array=1-9
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

N=${N_OVERRIDE:-1000}
IT=${ITERS_OVERRIDE:-200000}
ACT_NAME=(softplus25 leakyrelu sigmoid)
ACT_MODEL=(rnn_softplus_beta25_standard rnn_leaky_relu_standard rnn_sigmoid_shifted_standard)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); ACT=$((TID / 3))
ANAME=${ACT_NAME[$ACT]}; AMODEL=${ACT_MODEL[$ACT]}

TAG="std_g0_activations/EqType=h_N=${N}_act=${ANAME}_iters=${IT}"
mkdir -p "${SAVE_TO}/CDDM_std_g0_activations"
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} act=${ANAME} (model=${AMODEL}) rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  model="$AMODEL" trainer=trainer_ptrack_plain \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
