#!/bin/bash
# B2 AND B3: REMEDIES ON THE STANDARD 3-BIT FLIP-FLOP, ALONE AND COMBINED, WITH KEPT SNAPSHOTS. (SPOCK)
#
# The plan is docs/notes/agent-bplan.md (branch restore-oct5). This is that plan's remedy-grid
# launcher, written for Spock instead of Della. It is SilentReLU_c1_snapshots_della.slurm (the
# claim-1 controls: same task flags, same tracking flags, same snapshot layout, same code 7739a60)
# with four environment knobs. Each knob REPLACES the control launcher's hard-coded value, so the
# command line never carries two values for one key:
#
#   knob            sets                                                  arm name gets
#   LAMBDA_FRM=<l>  trainer.lambda_frm=<l>        (default 0; rate penalty)  frm<l>
#   SIGMA_W=<s>     model.sigma_w=<s>             (default 0; synaptic noise) sw<s>
#   RESCALE=1       trainer.prune_reinit=true + the rescale-target-10 block   rsc10
#   CKPT_EVERY=<n>  trainer.checkpoint_every=<n>  (default 1000)               -
#
# The rescale block is the operating point already on disk (NBitFlipFlop_size_series,
# NBitFlipFlop_target_ladder ..._tgt=10.0), copied from SilentReLU_size_series_della.slurm.
# Everything else is the standard network: ReLU, h equation, bias fixed at 0, self-connections on,
# no Dale, gamma 0, weight decay 1e-6, lr 1e-3 (100/N)^(1/3), batch 1024, T 300.
#
# SPOCK LIMITS. 96 h is the longest --time Spock accepts. N = 4000 with the rate penalty does NOT
# fit on any Spock GPU: one (N, T, B) float32 tensor is 4.6 GiB at N = 4000 and the rate penalty's
# log-sum-exp needs about nine of them; job 6111685_22 (9 Sep) ran out of memory on an L40S at
# 41.8 GB in use. Run N = 4000 rate-penalty cells on Della's 80 GB cards.
#
# Usage (from the pinned worktree; --chdir is required by the import check below):
#   C=--chdir=$HOME/trainRNNbrain_b23; L=slurm/SilentReLU_remedy_grid_spock.slurm
#   LAMBDA_FRM=0.1 N=1000 sbatch $C --export=ALL --time=36:00:00 --array=1-3 $L
#   SIGMA_W=1.0 RESCALE=1 N=1000 sbatch $C --export=ALL --time=40:00:00 --array=1-3 $L
#   LAMBDA_FRM=0.1 N=2000 CKPT_EVERY=1000 sbatch $C --export=ALL --mem=64G --time=60:00:00 ...
#
# Output:    $SAVE_TO/NBitFlipFlop_remedy_grid/EqType=h_N=<N>_arm=<arm>/<run>/
# Snapshots: $SNAP_TO/NBitFlipFlop_remedy_grid/EqType=h_N=<N>_arm=<arm>/s<seed>/it<iters>/
#   (outside the data tree: every inventory counts a subfolder holding *LastParams* as a run)
#
#SBATCH --job-name=RemGrid
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=36:00:00
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

REPO="${REPO:-$HOME/trainRNNbrain_b23}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"
SNAP_TO="${SNAP_TO:-/mnt/cup/labs/tengel/home/tolmach/trainRNNbrain_b23_snapshots}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
if [ "$PKG" != "$(readlink -f "$REPO")" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO - submit with --chdir=\$REPO." >&2; exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
[ -n "$GITSHA" ] || { echo "FATAL: $REPO is not a git checkout." >&2; exit 1; }
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA (git)"

N=${N:-1000}
IT=${ITERS_OVERRIDE:-200000}
LAMBDA_FRM=${LAMBDA_FRM:-0}
SIGMA_W=${SIGMA_W:-0}
CKPT_EVERY=${CKPT_EVERY:-1000}

# arm name from the knobs that are on, in a fixed order
ARM=()
[ "$LAMBDA_FRM" != 0 ] && ARM+=("frm${LAMBDA_FRM}")
[ "$SIGMA_W" != 0 ] && ARM+=("sw${SIGMA_W}")
[ "${RESCALE:-0}" = 1 ] && ARM+=("rsc10")
ARM=$(IFS=_; echo "${ARM[*]:-control}")

if [ "${RESCALE:-0}" = 1 ]; then
  RESCALE_ARGS=(trainer.prune_reinit=true
    trainer.prune_args.reinit_mode=rescale trainer.prune_args.rescale_norm_mode=self
    trainer.prune_args.rescale_target_frac=10.0 trainer.prune_args.rescale_active_only=true
    trainer.prune_args.rescale_normalize=true trainer.prune_args.rescale_alpha=1.002
    trainer.prune_args.rescale_cap=64.0 trainer.prune_args.rescale_protect_frac=0.20
    trainer.prune_args.check_every=1 trainer.prune_args.patience=1)
else
  RESCALE_ARGS=(trainer.prune_reinit=false)
fi

TAG="remedy_grid/EqType=h_N=${N}_arm=${ARM}"
mkdir -p "${SAVE_TO}/NBitFlipFlop_remedy_grid" "$SNAP_TO"
echo "Task ${SLURM_ARRAY_TASK_ID}: N=${N} arm=${ARM} (lambda_frm=${LAMBDA_FRM} sigma_w=${SIGMA_W}" \
     "rescale=${RESCALE:-0}) iters=${IT} snapshots every ${CKPT_EVERY}"
echo "  -> ${SAVE_TO}/NBitFlipFlop_${TAG}"
echo "  snapshots -> ${SNAP_TO}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3 \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  model.sigma_w="$SIGMA_W" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm="$LAMBDA_FRM" \
  "${RESCALE_ARGS[@]}" \
  trainer.synaptic_scaling=false trainer.dropout=false \
  trainer.checkpoint_every="$CKPT_EVERY" ++trainer.checkpoint_keep=true \
  paths.checkpoint_to="$SNAP_TO" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
