#!/bin/bash
# CDDM ALONE vs 20 TASKS at N=1000, 30k iterations, 3 seeds each — the confirmation run (SPOCK)
#
# The 10k pilot (job 6206472, one seed per set) found that training on all 20 Yang tasks does NOT
# raise the recruitment on CDDM: 512 live units on CDDM trials (scale-free) in the 20-task network
# against 574 in the CDDM-only network, while the 20-task net's whole-batch count was higher
# (652 vs 517) — i.e. the extra units serve the OTHER tasks, and 509 of the 512 CDDM-active units
# are shared with another task (3 CDDM-private). Pavel's call: confirm at this one cell with error
# bars and a settled budget, then stop; the full 36-job grid (penalty arms, N=500/2000) is dropped.
#
# 6 jobs = 2 sets x 3 seeds, N=1000, unpenalised, batch 1024:
#   set in {multi20 (task=Yang20), ctxdm (task=Yang_ctxdm = contextdm1+contextdm2, Mante's CDDM)}
#   tasks 1-4  FRESH, 30000 iterations   (seeds 2 and 3 of each set)
#   tasks 5-6  WARM START from the 10k pilot nets, 20000 more iterations -> 30k total (seed 1)
# Index decode (TID = SLURM_ARRAY_TASK_ID-1): fresh  S_IDX = TID % 2, rep = TID / 2
#                                             warm   S_IDX = (TID - 4) % 2
#
# WARM START (tasks 5-6) continues the pilot networks rather than retraining them: weights AND Adam
# moments are restored (paths.init_adam=true), which is sanctioned here because the penalty is
# UNCHANGED (none -> none) — carrying momentum across a penalty SWITCH would be a confound, across
# a plain continuation it just removes a restart transient. Verified locally before this launcher
# was written: a warm-started run's first iteration resumes at the source's final loss (0.0497 vs
# 0.0505) while a fresh control at the same seed starts at 0.2584.
#   ⚠️ The checkpoint path MUST be quoted in the Hydra override: net folder names contain ';' and
#   '=', which Hydra's override parser rejects unquoted (OverrideParseException).
#   ⚠️ A warm-started run's own iteration counter restarts at 0, so its folder says MI=20000 and its
#   participation trace covers TRUE iterations 10000-30000. Every net is read at the END of
#   training (= 30k true iterations for all six), so the per-cell read-out is matched; any
#   live-units-vs-iteration figure must offset the MI=20000 traces by +10000.
#
# Read-out (multitask_readout.py --focus contextdm1,contextdm2, and yang_checks.py):
#   live units on CDDM trials (scale-free / 1e-6 / 4e-2) per set, 3 seeds -> mean +- sd; the
#   shared-vs-CDDM-private split in the 20-task nets; per-rule accuracy (solved := >= 0.95) so an
#   unlearned rule cannot count as delivered demand.
# Cost: 0.44 s/iter measured at this cell -> 3.7 h fresh, 2.4 h warm; 8 h requested (slow nodes).
#   sbatch --array=1-6 slurm/SilentReLU_yang_30k_spock.slurm
#
# Output: $SAVE_TO/Yang_std_30k/EqType=h_set=<set>_N=1000_pen=none/<per-net subfolder>

#SBATCH --job-name=Yang30k
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=1-6
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
PILOT="${PILOT:-$HOME/trainRNNbrain/data/trained_RNNs/pilot10k/Yang_std_multi}"

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

SETS=(multi20 ctxdm)
TASKCFG=(Yang20 Yang_ctxdm)
N=1000

TID=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$TID" -lt 4 ]; then
  S_IDX=$((TID % 2)); REP=$((TID / 2 + 2)); IT=30000; WARM=""
else
  S_IDX=$(((TID - 4) % 2)); REP=1; IT=20000
  # the pilot net of this set; there is exactly one per cell
  WARM=$(ls -d "$PILOT/EqType=h_set=${SETS[$S_IDX]}_N=1000_pen=none"/*/ 2>/dev/null | head -1)
  if [ -z "$WARM" ]; then
    echo "FATAL: no pilot net to warm-start from under $PILOT for set ${SETS[$S_IDX]} — aborting." >&2; exit 1
  fi
fi
SET=${SETS[$S_IDX]}; TCFG=${TASKCFG[$S_IDX]}
IT=${ITERS_OVERRIDE:-$IT}

TAG="std_30k/EqType=h_set=${SET}_N=${N}_pen=none"
mkdir -p "${SAVE_TO}/Yang_std_30k"
echo "Task ${SLURM_ARRAY_TASK_ID}: set=${SET} (task=${TCFG}) N=${N} rep=${REP} iters=${IT}${WARM:+ WARM START from $WARM}"
echo "  -> ${SAVE_TO}/Yang_${TAG}"

# Warm-start overrides, empty for the fresh runs. The path is quoted: folder names contain ';'/'='.
WARM_ARGS=()
if [ -n "$WARM" ]; then
  WARM_ARGS=("paths.init_from=\"${WARM%/}\"" "paths.init_adam=true")
fi

srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_largeN \
  task="$TCFG" \
  model=rnn_relu_standard trainer=trainer_ptrack_freshbatch \
  model.equation_type=h model.N="$N" \
  trainer.max_iter="$IT" \
  trainer.track_drift=true trainer.track_every=10 trainer.store_participation_every=100 \
  trainer.track_valid=false \
  trainer.lambda_rws=0 trainer.lambda_frm=0 \
  "${WARM_ARGS[@]}" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
