#!/bin/bash
# IS THE SILENCE A ReLU-FAMILY PROPERTY? THE TWO MISSING ARMS ON THE FLIP-FLOP (DELLA)
#
# The activation question is currently answered on two different networks. On CDDM the standard net
# was run with softplus(beta=25), leaky-ReLU(leak 0.01) and sigmoid(7.5(x-0.3)); on the 3-bit
# flip-flop only sigmoid was ever run (NBitFlipFlop_std_sigmoid, Spock array 6147881, code a831629).
# So slide 9 of docs/presentation.md ("Nor on the other task") rests on ONE non-ReLU activation,
# while slide 8 rests on three. This sweep adds the two missing arms so both slides carry the same
# three activations, and so the softplus/leaky result is replicated on a second task rather than
# resting on CDDM alone.
#
# WHY THESE TWO SPECIFICALLY. Both keep a nonzero gradient everywhere, which is the mechanism the
# dead-unit story would be most vulnerable to: a unit whose input never crosses zero still receives
# dL/dw != 0 under leaky-ReLU (negative slope 0.01) and under softplus (beta=25), so the absorbing
# state that traps a ReLU unit's weights does not exist. If the concentration survives both, "it is
# the hard zero / zero gradient" is ruled out on this task as well as on CDDM.
#
# PINNED TO a831629 ON PURPOSE. a831629 is the commit the sigmoid arm AND the ReLU ksweep control
# were trained at. Trainer.py has since gained ~1200 lines and RNN_torch.py ~315, and all four
# trainer configs changed by ~161 lines each, so running these arms on the development branch would
# confound the activation with a code version. This branch is a831629 plus exactly two model configs
# and this launcher - nothing else. The two configs are 0f9688d's, minus its `input_row_norm` field,
# which RNN_torch does not accept at this commit.
#
# The two configs differ from configs/model/rnn_sigmoid_shifted_standard.yaml (the sigmoid arm) in
# `activation_args` and nothing else, and both carry bias_range [-1, 1] - the TRAINABLE bias, which
# is what the ReLU ksweep control and the sigmoid arm were both trained with at this commit. (The
# default flipped to a fixed zero bias later, at ae52c58; using today's rnn_relu_standard as the
# reference instead of the ksweep networks would therefore import a bias change into an activation
# comparison.)
#
# Grid (12 jobs) = 2 activations x 2 sizes x 3 seeds, 3-bit flip-flop, h equation, no penalties,
# standard RNN, batch 1024 fresh trials:
#   ACT 0 = softplus(beta=25)        model=rnn_softplus_beta25_standard   tag act=softplus25
#   ACT 1 = leaky-ReLU (leak 0.01)   model=rnn_leaky_relu_standard        tag act=leakyrelu
#   N in {500, 1000};  k = 3
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..11): REP=TID%3; N_IDX=(TID/3)%2; ACT_IDX=TID/6
#   softplus -> --array=1-6     leaky -> --array=7-12
#
# 150k ITERATIONS, matching the sigmoid arm exactly so the three activations are read at one
# iteration and the comparison needs no cross-budget matching. Measured ReLU cost at batch 1024
# fresh: 0.24 s/iter at N=500, 0.215 at N=1000 (k=3), slow-node tail ~0.3-0.35. 150k x 0.35 = 14.6 h,
# inside the 24 h request. Whether 150k was deep enough is decided afterwards by the project's own
# floor-stability test (fit L_inf on [0,T] and on [0,T/2]; they must agree), not asserted here.
#
# READ-OUT. The scale-free rule `p < 0.05*q95(p)` on participation p = std + q90, as for every other
# arm. Both activations have an unbounded, near-linear positive part, so unlike sigmoid they need no
# separate modulation criterion - but report it anyway, since softplus(beta=25) has a floor at
# log(2)/25 = 0.028 rather than at 0, which is nonzero and could flatter the standard rule.
#
# Timing check before the real submission: ITERS_OVERRIDE=600 sbatch --array=1,7 ...
#
# Output: $SAVE_TO/NBitFlipFlop_std_activations/EqType=h_k=3_N=<N>_act=<name>_iters=<IT>/
# ⚠️ RESULTS LAND IN DELLA'S OWN data/trained_RNNs and must be synced to the local mirror before any
# analysis - every analysis script reads the local copy, not a cluster.

#SBATCH --job-name=FFact
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
# Named gres TYPE, never --gpus-per-node, so a MIG slice (`3g.40gb`) can never satisfy this.
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=48G
#SBATCH --time=24:00:00
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

REPO="${REPO:-$HOME/trainRNNbrain_ffact}"
SAVE_TO="${SAVE_TO:-$REPO/data/trained_RNNs}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) — aborting." >&2; exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; this sweep is pinned, so refusing to run." >&2; exit 1
fi
# This sweep is only meaningful against the sigmoid arm and the ReLU ksweep control, both trained at
# a831629, so the invariant is not "HEAD == a831629" (HEAD is this branch's tip) but: a831629 is an
# ANCESTOR of HEAD, and the only files that differ between them are the two activation configs and
# this launcher. Any other changed file means some trainer or model code moved, which would confound
# the activation with a code version - so refuse rather than write a non-comparable cell.
PINNED=a831629
EXPECTED="configs/model/rnn_leaky_relu_standard.yaml
configs/model/rnn_softplus_beta25_standard.yaml
slurm/SilentReLU_flipflop_activations_della.slurm"
if ! git -C "$REPO" merge-base --is-ancestor "$PINNED" HEAD; then
  echo "FATAL: $PINNED is not an ancestor of HEAD in $REPO; this is not the pinned sweep." >&2; exit 1
fi
CHANGED=$(git -C "$REPO" diff --name-only "$PINNED" HEAD | sort)
if [ "$CHANGED" != "$(echo "$EXPECTED" | sort)" ]; then
  echo "FATAL: $REPO differs from $PINNED in more than the two configs and this launcher:" >&2
  echo "$CHANGED" >&2
  echo "The activation comparison would confound a code version. Aborting." >&2; exit 1
fi
# The .npz saver lost dict-valued parameters (activation_args stored as its KEYS) until the
# storable_ fix. Refuse to run on code that predates it, or these nets cannot be reconstructed.
python - <<'PYCHK' || exit 1
import sys
from trainRNNbrain.training import run_experiment as m
sys.exit(0 if hasattr(m, "storable_") else
         print("FATAL: run_experiment lacks storable_(); re-sync the repo before submitting.") or 1)
PYCHK

ACT_NAME=(softplus25 leakyrelu)
ACT_MODEL=(rnn_softplus_beta25_standard rnn_leaky_relu_standard)
NS=(500 1000)
K=3
IT=${ITERS_OVERRIDE:-150000}

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 3)); N_IDX=$(((TID / 3) % 2)); ACT_IDX=$((TID / 6))
N=${NS[$N_IDX]}
ANAME=${ACT_NAME[$ACT_IDX]}; AMODEL=${ACT_MODEL[$ACT_IDX]}

TAG="std_activations/EqType=h_k=${K}_N=${N}_act=${ANAME}_iters=${IT}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_std_activations"
echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K} N=${N} act=${ANAME} (model=${AMODEL}) rep=${REP} iters=${IT}"
echo "  -> ${SAVE_TO}/NBitFlipFlop_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=NBitFlipFlop \
  task.n_inputs="$K" task.n_outputs="$K" \
  model="$AMODEL" trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
