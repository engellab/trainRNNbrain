#!/bin/bash
# A6 — THE PENALTY-SWITCH INTERVENTION ON CDDM (SPOCK)
#
# The flip-flop switch (SilentReLU_flipflop_switch_spock.slurm, 2026-09-10) showed that adding or
# removing the sparsity penalty in a converged participation-penalty network moves the population
# between the burst/diffuse state and the sustained/assembly state, reversibly. The elaboration's CS3
# and CS4 are flip-flop-only for that reason, and the 2026-09-11 count figures show that on CDDM the
# sparsity penalty does NOT reduce mixed selectivity or bursts cross-sectionally. This runs the same
# four arms on CDDM so the switch result is a two-task claim - or a measured task difference.
#
#   arm  warm-start from        continue under     role
#   A1   frm  (N=2000, 150k)    frm + rws          TREATMENT: what adding rws does
#   A2   frm                    frm                control (optimiser restart + iterations)
#   A3   both (N=2000, 150k)    frm                REVERSE: what removing rws does
#   A4   both                   frm + rws          control
#
# 12 jobs = 4 arms x 3 paired seeds; parents are the CDDM_std_g0_penalties N=2000 nets (150k), the
# three r2 >= 0.5 run folders of each cell sorted by name, index = rep (the flip-flop rule). Warm
# phase 50k iterations, participation stored every 10 as in the flip-flop switch. Adam moments are
# NOT restored (paths.init_adam=false) - the restart is a constant across arms and is what A2/A4
# measure. Hydra cannot parse the unquoted run folder (';' and '=' in the name): init_from is
# passed single-quoted inside the override.
#
# Read-out: unit_stats-style counts (burst, mixed) and the temporal PR at the parent and at the
# endpoint, per arm, paired by rep - `flipflop_switch_selectivity.py` adapted to CDDM.
#
# Runtime ~4 h per job (N=2000, 50k iterations, ~0.25 s/iter).
# Output: $SAVE_TO/CDDM_std_switch/warm_EqType=h_N=2000_arm=<arm>_from=<pen>_rep=<rep>_iters=50000/

#SBATCH --job-name=CDDMsw
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=1-12
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
# Refuse to run if the warm-start load path is absent (see the flip-flop switch launcher).
python - <<'PYCHK' || exit 1
import sys, inspect
from trainRNNbrain.training import run_experiment as m
ok = hasattr(m, "storable_") and "init_from" in inspect.getsource(m)
sys.exit(0 if ok else print("FATAL: run_experiment lacks the warm-start load path.") or 1)
PYCHK

N=${N_OVERRIDE:-2000}
IT=${ITERS_OVERRIDE:-50000}
ROOT="std_switch"
ARM=(A1 A2 A3 A4)
SRC_PEN=(frm frm both both)
TGT_RWS=(0.05 0 0 0.05)

TID=$((SLURM_ARRAY_TASK_ID - 1))
A_IDX=$((TID / 3)); REP=$((TID % 3))
ANAME=${ARM[$A_IDX]}; SPEN=${SRC_PEN[$A_IDX]}; LRWS=${TGT_RWS[$A_IDX]}

BDIR="${SAVE_TO}/CDDM_std_g0_penalties/EqType=h_N=${N}_pen=${SPEN}"
[ -d "$BDIR" ] || { echo "FATAL: parent dir $BDIR not found." >&2; exit 1; }
CAND=()
while IFS= read -r _d; do
  _r=$(basename "$_d"); _r=${_r%%_*}
  case "$_r" in ''|*[!0-9.eE+-]*) continue ;; esac
  awk -v r="$_r" 'BEGIN{exit !(r+0 >= 0.5)}' || continue
  CAND+=("$_d")
done < <(find "$BDIR" -mindepth 1 -maxdepth 1 -type d | sort)
if [ "${#CAND[@]}" -ne 3 ]; then
  echo "FATAL: expected exactly 3 scored run folders under $BDIR, found ${#CAND[@]}:" >&2
  printf '  %s\n' "${CAND[@]}" >&2; exit 1
fi
SRC="${CAND[$REP]}"
ls "$SRC"/*LastParams*.npz >/dev/null 2>&1 || { echo "FATAL: no *LastParams*.npz in $SRC" >&2; exit 1; }

TAG="${ROOT}/warm_EqType=h_N=${N}_arm=${ANAME}_from=${SPEN}_rep=${REP}_iters=${IT}"
mkdir -p "${SAVE_TO}/CDDM_${ROOT}"
echo "Task ${SLURM_ARRAY_TASK_ID}: arm=${ANAME} from=${SPEN} (rws=${LRWS}, frm=0.1) rep=${REP} iters=${IT}"
echo "  parent: ${SRC}"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed=$((2000 + REP)) \
  +experiment=silent_units_largeN \
  model=rnn_relu_standard trainer=trainer_ptrack_plain \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=10 \
  trainer.lambda_rws="$LRWS" trainer.lambda_frm=0.1 \
  "paths.init_from='${SRC}'" paths.init_adam=false \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
