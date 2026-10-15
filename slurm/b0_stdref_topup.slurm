#!/bin/bash
# TOP UP THE FINISHED STANDARD REFERENCE GRID RUNS TO THE 75k STANDARD (DELLA)
#
# WHY. 50,000 iterations reaches 1.07x a run's own fitted loss floor for 74% of the 493 runs on
# disk; 75,000 reaches it for 94%. The runs that miss at 50k are concentrated rather than scattered:
# flip-flop k >= 5, N = 4000, anything under frm, and DMTS under rws, whose medians run 65k-132k.
# 75,000 is therefore the standard in configs/experiment/, and this launcher CONTINUES each finished
# run of CDDM_std_g0_bias0 for the iterations it is short, instead of retraining it.
#
# THE BUDGET IS NOT WRITTEN IN THIS FILE. Each task computes it as (the standard, from
# configs/experiment/silent_units_std.yaml) minus (what its parent actually trained, from the parent's own
# saved config), via slurm/topup_budget.py. A parent already at or past the standard is skipped, not
# shortened, and there is no iteration count here to drift out of sync with the config.
#
# SUBMIT CONTINGENT ON THE PARENT ARRAY FINISHING:
#   sbatch --dependency=afterany:<parent array id> \
#     --export=ALL,REPO=$HOME/trainRNNbrain_stdfix,SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs \
#     b0_stdref_topup.slurm
#
# afterany, NOT aftercorr. Nothing in a run folder records which rep it was - the folder name
# carries the score and the resolved seed - so a task picks the rep-th finished run in its cell by
# sort order, the pattern the penalty-switch launcher uses. That assignment is only stable once
# every rep of a cell has landed. aftercorr would release task N the moment parent task N finished,
# when the cell may hold fewer runs than reps, and two top-ups could then continue the same parent
# while another parent went uncontinued. afterany also means a failed parent task does not cancel
# every top-up: the task whose parent is missing fails loudly on its own.
#
# ⚠️ THE CONTINUATION LANDS IN ITS OWN ROOT, CDDM_std_g0_bias0_c75000. Beside the parent there
# would be two runs per cell - the 50k original and the 75k continuation - and every analysis that
# globs cells would count both, with the truncated one indistinguishable from a genuinely short run.
# The parent root stays as the 50k record.
#
# ⚠️ ADAM MOMENTS ARE RESTORED (init_adam=true), unlike the penalty-switch launcher. This continues
# under the SAME penalty, where carrying the moments removes a restart transient instead of
# importing another objective's search direction. Falls back to false, loudly, if the parent has no
# AdamState file.
#
# The continuation's trace is ONE trace: Trainer.iter_offset dates its samples from where the parent
# stopped and run_experiment prepends the parent's samples, so every analysis script reads a topped-
# up run as the single long run it is. Verified on a 300+150 iteration pair before this was written.
#SBATCH --job-name=StdRefB0c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=7:30:00
#SBATCH --array=1-40
#SBATCH --constraint=nomig
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
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO) - aborting." >&2
  exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"

# The parent sweep's grid, decoded exactly as b0_stdref.slurm does it, so task N here tops up the
# cell that task N trained there.
EQS=(h s); RWSS=(0 0.05); FRMS=(0 0.2)
TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 5)); FRM_IDX=$(((TID / 5) % 2)); RWS_IDX=$(((TID / 10) % 2)); EQ_IDX=$(((TID / 20) % 2))
EQ=${EQS[$EQ_IDX]}; RWS=${RWSS[$RWS_IDX]}; FRM=${FRMS[$FRM_IDX]}

TAG="std_g0_bias0/EqType=${EQ}_N=1000_LmbdRWS=${RWS}_LmbdFR=${FRM}"
CELL="${SAVE_TO}/CDDM_${TAG}"
EXPCFG="$REPO/configs/experiment/silent_units_std.yaml"

# Finished runs in this cell, in sort order; a run counts as finished only once it has written its
# parameters, which is the last thing a crashed run would have skipped.
mapfile -t CAND < <(for d in "$CELL"/*/; do
    compgen -G "$d*LastParams*" >/dev/null && echo "${d%/}"
  done | sort)
if [ "${#CAND[@]}" -le "$REP" ]; then
  echo "FATAL: $CELL holds ${#CAND[@]} finished run(s), so there is none at rep index $REP." >&2
  exit 1
fi
SRC="${CAND[$REP]}"

read -r PAR STD TOPUP < <(python "$REPO/slurm/topup_budget.py" "$SRC" "$EXPCFG")
echo "Task ${SLURM_ARRAY_TASK_ID}: EQ=${EQ} N=1000 Lrws=${RWS} Lfrm=${FRM} rep=${REP}"
echo "  parent:  $SRC"
echo "  trained: $PAR of the $STD standard  ->  top-up $TOPUP"
if [ "$TOPUP" -le 0 ]; then
  echo "  nothing to do: the parent is already at or past the standard."
  exit 0
fi

ADAM=false
ls "$SRC"/*AdamState*.pt >/dev/null 2>&1 && ADAM=true
[ "$ADAM" = true ] || echo "  WARNING: parent has no AdamState; continuing without its moments."

TAG_C="${TAG%%/*}_c${STD}/${TAG#*/}"
echo "  -> ${SAVE_TO}/CDDM_${TAG_C}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_std \
  model=rnn_relu_standard trainer=trainer_ptrack_plain \
  model.equation_type="$EQ" \
  trainer.lambda_rws="$RWS" trainer.lambda_frm="$FRM" \
  trainer.max_iter="$TOPUP" \
  "paths.init_from='$SRC'" paths.init_adam=$ADAM \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG_C}\""
