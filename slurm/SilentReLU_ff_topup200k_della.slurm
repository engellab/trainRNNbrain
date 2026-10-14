#!/bin/bash
# TOP UP THE STANDARD 3-BIT FLIP-FLOP CONTROLS TO 200,000 ITERATIONS. (DELLA)
#
# WHY. On the standard (bias-0) flip-flop the noisy loss - the loss these networks are trained on,
# and since 2026-10-12 the only one the read-out uses - is still falling at 75,000 iterations in
# every control. Fitted per run, it reaches 1.05x its own floor between 21k and 368k iterations
# (median 69k); 8 of 15 runs get there by 75k, about 12 by 150k, about 14 by 250k. So the floor
# read-out does not exist for this task until the budget grows. 200k is the first step; the fit is
# an extrapolation, so each run is checked against its own trace once it lands.
#
# EACH TASK CONTINUES ITS PARENT FROM THE PARENT'S OWN SAVED CONFIG (--config-path/--config-name),
# exactly as the 40k -> 75k top-up did, so every setting carries over by construction.
# `trainer.iter_offset=auto` dates the continuation from the chain, and the participation trace is
# joined to the parent's, so a 200k continuation reads as one 0 -> 200k run. The TrainLosses of
# each segment stay in their own folders; common.noisy_loss_series joins them along paths.init_from.
#
# THE CONTINUATION LANDS IN ITS OWN ROOT, <root>_c<TARGET> (any _c<N> suffix of the parent's root is
# replaced), so no cell holds two budgets of the same network. TARGET defaults to 200000.
#
# PARENTS, one per array task, are FOUND WHEN THE TASK STARTS: PARENTS is a glob of run folders,
# sorted, and task i continues the i-th. So an array can be queued behind the job that writes its
# parents (the claim-1 snapshot runs) with --dependency.
#
# KEEP=1 keeps a weight snapshot every 500 iterations (trainer.checkpoint_keep), for the runs whose
# weights are read at their floor (claim 1). Otherwise a rolling, resumable snapshot every 5,000.
#
# Measured cost per 40k iterations on the paper grid: 0.75 h (N=500), 2.9 h (1000), 7 h (2000),
# 25 h (4000, 80 GB card). So 125k more: ~2.5 h, ~9 h, ~22 h, ~80 h.
#   D=$HOME/trainRNNbrain/data/trained_RNNs
#   PARENTS="$D/NBitFlipFlop_c1_snapshots/EqType=h_N=1000_arm=control/*/" KEEP=1 \
#     sbatch --export=ALL --dependency=afterok:<C1snap ff job> --time=16:00:00 --array=1-3 <this>
#   PARENTS="$D/NBitFlipFlop_ff_revive_c75000/EqType=h_k=3_N=500_pen=none_arm=none/*/" \
#     sbatch --export=ALL --time=06:00:00 --array=1-6 <this>
#   ... N=1000 --time=16:00:00, N=2000 --time=36:00:00,
#   PARENTS=".../NBitFlipFlop_paper_grid_c75000/EqType=h_N=4000_arm=control/*/" \
#     sbatch --export=ALL --time=5-00:00:00 --constraint=gpu80 --array=1-3 <this>
#
#SBATCH --job-name=FFtop200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
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

REPO="${REPO:-$HOME/trainRNNbrain_c1top}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs}"
SNAP_TO="${SNAP_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots}"
TARGET="${TARGET:-200000}"
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

mapfile -t ALL < <(ls -d ${PARENTS:?set PARENTS to a glob of parent run folders} 2>/dev/null | sort)
SRC="${ALL[$((SLURM_ARRAY_TASK_ID - 1))]%/}"
[ -n "$SRC" ] || { echo "FATAL: no parent number $SLURM_ARRAY_TASK_ID under $PARENTS" >&2; exit 1; }
CFG=$(ls "$SRC"/*_config.yaml | head -1)

# iterations already trained by the parent's chain, and the root/tag of the continuation
read -r TASKNAME DONE TAG < <(python - "$CFG" "$SRC" "$TARGET" <<'EOF'
import os, re, sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
off = cfg.trainer.get("iter_offset", 0)
done = (int(off) if isinstance(off, (int, float)) else 0) + int(cfg.trainer.max_iter)
cell_dir = os.path.dirname(sys.argv[2].rstrip("/"))
root, cell = os.path.basename(os.path.dirname(cell_dir)), os.path.basename(cell_dir)
task = str(cfg.task.taskname)
base = re.sub(r"_c\d+$", "", root[len(task) + 1:] if root.startswith(task + "_") else root)
print(task, done, f"{base}_c{sys.argv[3]}/{cell}")
EOF
)
TOPUP=$((TARGET - DONE))
[ "$TOPUP" -gt 0 ] || { echo "nothing to do: parent already at $DONE >= $TARGET"; exit 0; }
ADAM=false
ls "$SRC"/*AdamState*.pt >/dev/null 2>&1 && ADAM=true

if [ "${KEEP:-0}" = 1 ]; then
  CKPT=(++trainer.checkpoint_every=1000 ++trainer.checkpoint_keep=true "++paths.checkpoint_to=$SNAP_TO")
else
  CKPT=(++trainer.checkpoint_every=5000 ++trainer.checkpoint_keep=false)
fi
echo "Task ${SLURM_ARRAY_TASK_ID}: $TASKNAME, $DONE done, +$TOPUP to $TARGET, adam=$ADAM, keep=${KEEP:-0}"
echo "  parent: $SRC"
echo "  -> $SAVE_TO/${TASKNAME}_$TAG"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  --config-path "$(dirname "$CFG")" --config-name="$(basename "$CFG" .yaml)" \
  seed="random" \
  trainer.max_iter="$TOPUP" \
  ++trainer.iter_offset=auto \
  "++paths.init_from='$SRC'" ++paths.init_adam=$ADAM \
  "${CKPT[@]}" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"$TAG\""
