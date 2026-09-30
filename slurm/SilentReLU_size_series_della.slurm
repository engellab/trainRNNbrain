#!/bin/bash
# DO THE TWO WORKING INTERVENTIONS HOLD ACROSS NETWORK SIZE? (DELLA)
#
# Both are established at N = 1000 on the 3-bit flip-flop, 40k iterations, and both are worth
# little if they only work there: the active FRACTION falls from 41% at N = 500 to 13% at N = 4000,
# so size is where the problem is.
#
#   RESCALE, maturity target 10. The target ladder is complete and has a ceiling: dimensionality
#   runs 5.93, 6.37, 7.32, 7.01, 6.52, 6.26 at targets 6, 8, 10, 14, 20, 30, so 10 is the operating
#   point. At N = 1000 it gives 453 active units at 7.32 dimensions and r2 0.932, against an
#   untreated 278 at 5.59 and 0.945.
#
#   SYNAPTIC NOISE, sigma_w = 1.0. Measured clean (w_noise=False, which disables the synaptic noise
#   too), it does not cost performance at all: r2 0.927 against the control's 0.894 -- the control
#   LOSES 5 points when its training noise is removed and the sigma_w network does not move
#   (0.9277 noisy, 0.9269 clean), so it is the more robust of the two. 543 active at 6.76
#   dimensions. sigma_w 2.0 and 3.0 are BROKEN (clean r2 0.784 and 0.655) and bound the useful
#   range from above.
#
# GRID (11 jobs), one entry per index:
#   1-4    rescale target 10   N= 500
#   5-8    rescale target 10   N=2000
#   9-11   synaptic noise 1.0  N=4000
#
# N = 1000 is NOT re-run for either: target 10 is measured at n=4 in NBitFlipFlop_target_ladder and
# sigma_w 1.0 at n=3 in NBitFlipFlop_synnoise, both on code identical to this commit for these
# settings. N = 500 and N = 2000 already exist for synaptic noise (the second is still running), so
# only N = 4000 is missing there.
#
# CONTROLS ALREADY EXIST at every size: NBitFlipFlop_ff_revive arm=none gives 211, 275, 451 and 622
# active units at N = 500, 1000, 2000 and 4000, same task, iteration count and model config.
#
# ⚠️ N = 4000 MAY NOT FIT, and is submitted knowing that. The weight-noise path builds one N x N
# tensor per timestep and autograd holds all of them: 1.2 GB at N = 1000, 4.8 GB at N = 2000 and
# 19.2 GB at N = 4000 over T = 300, before activations and the optimiser state. 80 GB is requested
# for those three jobs. If they die on memory that is a real limit of the method at scale and
# should be reported as one rather than worked around quietly -- the honest fallback is to refresh
# the perturbation every k steps instead of every step, which is a different intervention and would
# need its own arm.
#
# SUBMIT IN TWO GROUPS, since N = 500 takes about 1.5 h and N = 4000 with per-step draws is unknown
# but certainly long:
#   sbatch --array=1-8  --time=14:00:00                     slurm/SilentReLU_size_series_della.slurm
#   sbatch --array=9-11 --time=36:00:00 --mem-per-gpu=80G   slurm/SilentReLU_size_series_della.slurm
#
# PRE-REGISTERED. Read-out at 40,000 iterations, scale-free criterion, each size against its OWN
# control.
#   PRIMARY    active units as a RATIO to that size's control, so the sizes are comparable. At
#              N = 1000 the ratios are 1.63 (rescale target 10) and 1.97 (synaptic noise).
#   CO-PRIMARY activity dimensionality against the same control, and clean r2 -- NOT the stored
#              score, which is one noisy forward pass and is untrustworthy once sigma_w is large
#              (the sigma_w 2.0 cells store 0.926 while thirty draws never exceed 0.831).
#   SPREAD     participation q99/q01. The rescale family compresses it 3-4x (59k-85k against a
#              control's 265k) and synaptic noise does not at N = 1000 (258k) but does at N = 500
#              (59k against 121k). Whether that compression grows with size is open.
#   WHAT WOULD CHANGE MY MIND: if either ratio falls toward 1.0 as N grows, the intervention does
#              not address the problem the paper is about, since the problem worsens with size.

#SBATCH --job-name=SizeSeries
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=14:00:00
#SBATCH --array=1-11%11
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

IT=40000
# index 0 unused so the decode reads SLURM_ARRAY_TASK_ID directly
TASK_ARM=(x rescale rescale rescale rescale rescale rescale rescale rescale synnoise synnoise synnoise)
TASK_N=(0 500 500 500 500 2000 2000 2000 2000 4000 4000 4000)

i=$SLURM_ARRAY_TASK_ID
ARM=${TASK_ARM[$i]}; N=${TASK_N[$i]}

if [ "$ARM" = "rescale" ]; then
  TAG="size_series/EqType=h_k=3_N=${N}_arm=rescale_tgt10"
  mkdir -p "${SAVE_TO}/NBitFlipFlop_size_series"
  echo "Task ${i}: arm=rescale target=10.0 N=${N} iters=${IT}"
  echo "  -> ${SAVE_TO}/NBitFlipFlop_${TAG}"
  srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
    seed="random" +experiment=silent_units_largeN \
    task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3 \
    model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
    model.equation_type=h model.N="$N" \
    trainer.max_iter="$IT" \
    trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
    trainer.track_valid=false trainer.lambda_rws=0 trainer.lambda_frm=0 \
    trainer.prune_reinit=true trainer.synaptic_scaling=false \
    trainer.prune_args.reinit_mode=rescale \
    trainer.prune_args.rescale_norm_mode=self \
    trainer.prune_args.rescale_target_frac=10.0 \
    trainer.prune_args.rescale_active_only=true \
    trainer.prune_args.rescale_normalize=true \
    trainer.prune_args.rescale_alpha=1.002 \
    trainer.prune_args.rescale_cap=64.0 \
    trainer.prune_args.rescale_protect_frac=0.20 \
    trainer.prune_args.check_every=1 trainer.prune_args.patience=1 \
    paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\""
else
  TAG="size_series/EqType=h_k=3_N=${N}_arm=synnoise_sw1.0"
  mkdir -p "${SAVE_TO}/NBitFlipFlop_size_series"
  echo "Task ${i}: arm=synaptic noise sigma_w=1.0 N=${N} iters=${IT}"
  echo "  -> ${SAVE_TO}/NBitFlipFlop_${TAG}"
  srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
    seed="random" +experiment=silent_units_largeN \
    task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3 \
    model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
    model.equation_type=h model.N="$N" model.sigma_w=1.0 \
    trainer.max_iter="$IT" \
    trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
    trainer.track_valid=false trainer.lambda_rws=0 trainer.lambda_frm=0 \
    trainer.prune_reinit=false trainer.synaptic_scaling=false \
    paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\""
fi
