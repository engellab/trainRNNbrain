#!/bin/bash
# PENALTY-SWITCH INTERVENTION: does rws MOLD transient units into sustained ones? — DELLA.
#
# The cross-sectional frm-vs-both comparison shows frm leaves 45% of units low-occupancy (13.6%
# dead + 31.8% alive but active <10% of the time) against 6.2% under frm+rws. It CANNOT show that
# rws moves those same units: an equally good account is that frm+rws finds a different solution in
# which different units fill those roles. Only tracking unit IDENTITY across a penalty switch
# separates molding from reshuffling, and that is what this does.
#
# TWO PHASES. Submit PHASE=base first, wait for it, then PHASE=warm.
#   PHASE=base sbatch --array=1-6  --time=12:00:00 <this>      # 50k iters, frm x3 and both x3
#   PHASE=warm sbatch --array=1-12 --time=02:30:00 <this>      # 10k iters, 4 arms x 3 seeds
#
#   arm  warm-start from   continue under   role
#   A1   frm base          frm + rws        TREATMENT: do transient units become sustained?
#   A2   frm base          frm              control: extra training + optimiser restart
#   A3   both base         frm              REVERSE: does bimodality return?
#   A4   both base         frm + rws        control
#
# ⚠️ A2 AND A4 ARE NOT OPTIONAL. Warm-starting resets Adam and adds iterations; without the
# same-penalty arms every change is attributable to those rather than to the penalty switch.
#
# ⚠️ ARMS ARE PAIRED BY DESIGN — 6 BASE RUNS, NOT 12. A1 and A2 branch from the SAME frm parent
# (A3/A4 likewise from the same both parent), which makes A1-vs-A2 a paired comparison and removes
# between-seed variance from the contrast. Giving each arm its own parent would cost twice the base
# compute AND lose the pairing.
#
# ⚠️ 50k BASE, NOT THE EXISTING 400k penlong NETS. A 10k switch is a 2.5% perturbation to a
# 400k-converged net and would likely show nothing; a 50k net is converged in loss (frm and both
# both reach 1.03x floor by ~30-100k at N=2000) but still plastic.
#
# ⚠️ ADAM MOMENTS ARE NOT RESTORED (paths.init_adam stays false). Carrying frm-shaped momentum
# across the switch imports the old objective's search direction into the new one, which is the
# confound this experiment exists to isolate. All four arms restart Adam identically, so the restart
# is a constant across arms and is what A2/A4 measure.
#
# ⚠️ HYDRA CANNOT PARSE AN UNQUOTED RUN FOLDER. Folder names contain ';' and '=' (e.g.
# `...relu;N=2000;seed=...;Lfrm=0.1`), both of which are hydra override-grammar characters, so
# paths.init_from MUST be passed single-quoted INSIDE the override: "paths.init_from='$SRC'".
#
# ⚠️ TEMPORAL PR IS THE ENDPOINT AND IS STORED EVERY 10 ITERATIONS IN THE WARM PHASE. It cannot be
# recovered from the participation vector (std + q90) after the fact - it is computed in
# Trainer.track_participation_ from the probe's rates and stored alongside `participation`, sharing
# `participation_iters`.
#
# Output: $SAVE_TO/NBitFlipFlop_std_switch/{base,warm}_.../

#SBATCH --job-name=FFswitch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=32G
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
PHASE="${PHASE:?set PHASE=base or PHASE=warm}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL — aborting." >&2; exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "WARNING: $REPO has uncommitted changes; the commit above does not fully describe this run." >&2
fi
# Refuse to run if the warm-start LOAD path is absent: without it every warm task would silently
# train from a FRESH net and the whole experiment would be void while looking like it worked.
python - <<'PYCHK' || exit 1
import sys, inspect
from trainRNNbrain.training import run_experiment as m
src = inspect.getsource(m)
ok = hasattr(m, "storable_") and "init_from" in src
sys.exit(0 if ok else print("FATAL: run_experiment lacks storable_() or the warm-start load path; "
                            "re-sync the repo before submitting.") or 1)
PYCHK

K=3
N=2000
ROOT="std_switch"
COMMON=(seed_placeholder +experiment=silent_units_largeN task=NBitFlipFlop
        task.n_inputs=$K task.n_outputs=$K
        model=rnn_relu_standard trainer=trainer_ptrack_freshbatch
        model.equation_type=h model.N=$N
        trainer.track_drift=true trainer.track_every=10 trainer.track_valid=false)

TID=$((SLURM_ARRAY_TASK_ID - 1))
mkdir -p "${SAVE_TO}/NBitFlipFlop_${ROOT}"

if [ "$PHASE" = "base" ]; then
  # 6 tasks: TID/3 selects the penalty, TID%3 the seed replicate.
  BASE_PEN=(frm both); BASE_RWS=(0 0.05)
  P_IDX=$((TID / 3)); REP=$((TID % 3))
  PNAME=${BASE_PEN[$P_IDX]}; LRWS=${BASE_RWS[$P_IDX]}
  IT=${ITERS_OVERRIDE:-50000}
  TAG="${ROOT}/base_EqType=h_k=${K}_N=${N}_pen=${PNAME}_rep=${REP}_iters=${IT}"
  echo "BASE task ${SLURM_ARRAY_TASK_ID}: pen=${PNAME} rep=${REP} iters=${IT}"
  srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
    seed=$((1000 + REP)) "${COMMON[@]:1}" \
    trainer.max_iter="$IT" trainer.store_participation_every=100 \
    trainer.lambda_rws="$LRWS" trainer.lambda_frm=0.1 \
    paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\""

elif [ "$PHASE" = "warm" ]; then
  # 12 tasks: TID/3 selects the arm, TID%3 the seed replicate (paired to the base rep).
  ARM=(A1 A2 A3 A4)
  SRC_PEN=(frm frm both both)      # which base net each arm continues from
  TGT_RWS=(0.05 0 0 0.05)          # rws under which it continues; frm stays 0.1 throughout
  A_IDX=$((TID / 3)); REP=$((TID % 3))
  ANAME=${ARM[$A_IDX]}; SPEN=${SRC_PEN[$A_IDX]}; LRWS=${TGT_RWS[$A_IDX]}
  BASE_IT=${BASE_ITERS:-50000}
  IT=${ITERS_OVERRIDE:-10000}

  BDIR="${SAVE_TO}/NBitFlipFlop_${ROOT}/base_EqType=h_k=${K}_N=${N}_pen=${SPEN}_rep=${REP}_iters=${BASE_IT}"
  # Exactly one r2-named run folder must live under the base tag. Zero means the base phase did not
  # finish; more than one means a rerun left a duplicate and the arm would silently pick whichever
  # sorts first. Both are fatal - never guess which parent an arm branched from.
  mapfile -t CAND < <(find "$BDIR" -mindepth 1 -maxdepth 1 -type d | sort)
  if [ "${#CAND[@]}" -ne 1 ]; then
    echo "FATAL: expected exactly 1 run folder under $BDIR, found ${#CAND[@]}:" >&2
    printf '  %s\n' "${CAND[@]}" >&2; exit 1
  fi
  SRC="${CAND[0]}"
  if ! ls "$SRC"/*LastParams*.npz >/dev/null 2>&1; then
    echo "FATAL: no *LastParams*.npz in $SRC — base run did not complete." >&2; exit 1
  fi

  TAG="${ROOT}/warm_EqType=h_k=${K}_N=${N}_arm=${ANAME}_from=${SPEN}_rep=${REP}_iters=${IT}"
  echo "WARM task ${SLURM_ARRAY_TASK_ID}: arm=${ANAME} from=${SPEN} rep=${REP} rws=${LRWS} iters=${IT}"
  echo "  init_from: $SRC"
  srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
    seed=$((2000 + REP)) "${COMMON[@]:1}" \
    trainer.max_iter="$IT" trainer.store_participation_every=10 \
    trainer.lambda_rws="$LRWS" trainer.lambda_frm=0.1 \
    "paths.init_from='${SRC}'" paths.init_adam=false \
    paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\""
else
  echo "FATAL: PHASE must be 'base' or 'warm', got '$PHASE'" >&2; exit 1
fi
