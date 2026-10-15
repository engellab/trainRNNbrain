#!/bin/bash
# FILL THE TWO HOLES IN FIGURE 2's SIZE PANELS (f) and (g). (DELLA)
#
# WHY THIS EXISTS. Panels (f)/(g) draw every arm against network size on the 3-bit flip-flop at
# 40,000 iterations. Two arms are short:
#   frm + rws  exists ONLY at N = 1000 (`NBitFlipFlop_ff_both40k`), so it draws as a lone point.
#              There IS a `both` series at N = 500/1000/2000 in `NBitFlipFlop_std_penlong`, but at
#              400,000 iterations - a 10x budget against every other arm's 40,000. The figure's
#              whole claim is "the intervention is the only difference", so that series cannot be
#              used here. These cells rerun it at the matched 40,000.
#   rescale    exists at N = 500/1000/2000. N = 4000 was never in the `size_series` grid.
# Synaptic noise at N = 4000 is ALREADY RUNNING (SizeSeries 14777855 tasks 9-11) and is not here.
# Duplication, mute dropout and the control are complete at all four sizes.
#
# CODE VERSION: this must run from a worktree pinned at 99c877a, the commit the rescale cells at
# N = 500/2000 were trained on, so the N = 4000 rescale cell is the same rule as its own series.
# The frm/rws penalty implementations are byte-identical between 99c877a and c35d71b (checked
# function by function), so the `both` cells are unaffected by the pin either way.
#
# SETTINGS, copied field by field from the cells these extend, NOT reconstructed:
#   both     from NBitFlipFlop_ff_both40k/EqType=h_N=1000_arm=both -
#            lambda_frm 0.1, lambda_rws 0.05, frm_args cap_fr 0.3 / tau 0.1 / g_top 3 / g_bot 3,
#            rws_args tg_deg 20. No dropout, no prune_reinit, no synaptic scaling.
#   rescale  from NBitFlipFlop_size_series/EqType=h_k=3_N=2000_arm=rescale_tgt10 -
#            reinit_mode rescale, target_frac 10, alpha 1.002, cap 64, protect_frac 0.20,
#            norm_mode self, active_only, check_every 1, patience 1.
# Both share the matched base: gamma 0, lr 1e-3, weight decay 1e-6, sigma_rec = sigma_inp = 0.05,
# equation h, 3-bit flip-flop, 40,000 iterations.
#
# GRID (12 jobs) = 4 cells x 3 seeds. Index 0 unused so the decode reads SLURM_ARRAY_TASK_ID.
#    1-3   both     N= 500
#    4-6   both     N=2000
#    7-9   both     N=4000
#   10-12  rescale  N=4000
#
# RUNTIME, from the paper-grid launcher's measured flip-flop times: 0.75 h (N=500), 7 h (2000),
# 25 h (4000). SUBMIT AS TWO ARRAYS, because one --time cannot serve both ends:
#   sbatch --time=14:00:00                     --array=1-6%6   slurm/SilentReLU_f2_fill_della.slurm
#   sbatch --time=40:00:00 --qos=gpu-medium    --array=7-12%6  slurm/SilentReLU_f2_fill_della.slurm
# gpu-medium caps at 3 days; the 25 h cell does not fit the 1-day gpu-short default.
#
# Output: $SAVE_TO/NBitFlipFlop_ff_both40k/EqType=h_N=<N>_arm=both
#         $SAVE_TO/NBitFlipFlop_size_series/EqType=h_k=3_N=4000_arm=rescale_tgt10

#SBATCH --job-name=F2fill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=64G
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

REPO="${REPO:-$HOME/trainRNNbrain_f2fill}"
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
TASK_ARM=(x both both both both both both both both both rescale rescale rescale)
TASK_N=(0 500 500 500 2000 2000 2000 4000 4000 4000 4000 4000 4000)

i=$SLURM_ARRAY_TASK_ID
ARM=${TASK_ARM[$i]}; N=${TASK_N[$i]}

if [ "$ARM" = "both" ]; then
  TAG="ff_both40k/EqType=h_N=${N}_arm=both"
  mkdir -p "${SAVE_TO}/NBitFlipFlop_ff_both40k"
  echo "Task ${i}: arm=frm+rws N=${N} iters=${IT}"
  echo "  -> ${SAVE_TO}/NBitFlipFlop_${TAG}"
  srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
    seed="random" +experiment=silent_units_largeN \
    task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3 \
    model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
    model.equation_type=h model.N="$N" \
    trainer.max_iter="$IT" \
    trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
    trainer.track_valid=false \
    trainer.lambda_frm=0.1 trainer.lambda_rws=0.05 \
    trainer.frm_args.cap_fr=0.3 trainer.frm_args.tau=0.1 \
    trainer.frm_args.g_top=3.0 trainer.frm_args.g_bot=3.0 \
    trainer.rws_args.tg_deg=20 \
    trainer.prune_reinit=false trainer.synaptic_scaling=false trainer.dropout=false \
    paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\""
else
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
fi
