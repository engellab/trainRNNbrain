#!/bin/bash
# Unconstrained RNN WITH trainable bias + participation trace (DELLA)
#
# THE FIELD-STANDARD ARCHITECTURE, and the intended reference condition for a general-audience
# result: no Dale's law, no I/O sign constraints, learnable per-unit offsets
# (configs/model/rnn_relu_noDale_trainablebias.yaml -> dale=false, io_nonnegativity=false,
# bias_range=[-1,1], bias_init=zeros). If the ~50% silent population appears here too, it is a
# property of trained RNNs rather than of a heavily constrained special case.
#
# Replaces the cancelled Dale+bias sweep (array 11610299), on the grounds that the biological
# constraints make the result niche. Siblings: CDDM_ptrack_g0 (Dale, bias-free — the original
# reference) and CDDM_ptrack_g0_nodale (unconstrained, bias-free).
#
# Participation is logged every 10 iterations as *_ParticipationTrace.pkl + participation_trace.png.
# Full rationale, prediction, falsifier and analysis caveats:
# docs/experiments/participation_trace_nodale_bias.md (copied into the sweep folder as EXPERIMENT.md by
# array task 1).
#
# RUNS FROM A SEPARATE GIT WORKTREE ($HOME/trainRNNbrain_nodalebias) so that sweeps already in flight keep
# executing the code they started with. See the PYTHONPATH guard below.
#
# Grid (40 jobs) = 2 eq x 2 lambda_rws x 2 lambda_frm x 5 nets, N=1000, gamma=0:
#   EQ  in {h, s}            (model.equation_type)
#   RWS in {0, 0.05}         (trainer.lambda_rws)
#   FRM in {0, 0.2}          (trainer.lambda_frm)
#   5 reps, seed="random"
# Index decode (TID = SLURM_ARRAY_TASK_ID-1, 0..39): TID = 20*EQ + 10*RWS + 5*FRM + REP
#   REP=TID%5; FRM_IDX=(TID/5)%2; RWS_IDX=(TID/10)%2; EQ_IDX=(TID/20)%2
#
# Same regime as CDDM_4a031e_g0 / CDDM_ptrack_g0 (gamma=0, N=1000, 30000 iters, dt=1) EXCEPT that both
# architectural constraints are off — that is the manipulation, and CDDM_ptrack_g0 is its control.
#
# Output: $SAVE_TO/CDDM_ptrack_g0_nodale_trainablebias/EqType=<EQ>_N=1000_LmbdRWS=<RWS>_LmbdFR=<FRM>/<per-net subfolder>
# Overridable env vars: MAXITER (default: config, 30000), SAVE_TO (default: TENGEL scratch).
#
# NOTE --time: >= 2 h is required on Della, otherwise the job lands on the `gputest` QOS with a hard
# cap of 3 concurrent jobs. MEASURED on Della (2026-07-25, 30000 iters at N=1000): jobs WITHOUT the
# frm penalty take 1h42 (0.203 s/iter); WITH frm they take ~2h51 (0.343 s/iter) because frm adds a
# backward pass per iteration. The first submission requested 2:30 and lost all 60 frm jobs to
# TIMEOUT. 7:30 gives ~2.6x headroom.
# NOTE --constraint=nomig: the gpu partition mixes full A100s with 3g.40gb MIG slices (della-l01g3-12,
# 10 of 89 nodes); a slice runs at roughly half speed, so excluding them removes that variance at
# almost no cost in available nodes (79 of 89 remain).
# CORRECTION (2026-07-26): the 0.632 s/iter case first seen on a MIG node was NOT caused by the slice
# — Dale+frm reproduces 0.641 s/iter on full A100s. Measured rates for 30000 iters at N=1000:
#   no frm                 0.203 s/iter -> 1h42     frm, unconstrained  0.34-0.41 s/iter -> 2h51-3h25
#   frm, Dale              0.641 s/iter -> 5h21 (the binding case; 7:30 still leaves ~2h margin)

#SBATCH --job-name=PtrackNDBias
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=7:30:00
#SBATCH --constraint=nomig
#SBATCH --array=1-40
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
# Logs go to a checkout-independent directory: SBATCH directives are parsed at submit time and
# cannot expand $REPO, and per-worktree log dirs would vanish with the worktree. Job name (%x) and
# array id (%A) keep the sweeps apart.
#SBATCH --output=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.err

module purge
module load cudatoolkit/12.8
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env

set -eo pipefail
export PYTHONUNBUFFERED=1   # otherwise iteration lines sit in an 8 KB buffer and the log looks stalled

REPO="${REPO:-$HOME/trainRNNbrain_nodalebias}"   # separate git worktree: keeps in-flight sweeps on their own checkout
SAVE_TO="${SAVE_TO:-/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs}"

# The editable install points at ~/trainRNNbrain (the OTHER checkout), so `import trainRNNbrain` would
# silently resolve there. Put the worktree first on the path and REFUSE TO RUN if it did not win.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.dirname(os.path.dirname(trainRNNbrain.__file__)))")
if [ "$PKG" != "$REPO" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO — aborting so we don't train with the wrong code." >&2
  exit 1
fi
echo "code: $PKG @ $(git -C "$REPO" rev-parse --short=7 HEAD)"

EQS=(h s)
RWSS=(0 0.05)
FRMS=(0 0.2)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 5))
FRM_IDX=$(((TID / 5)  % 2))
RWS_IDX=$(((TID / 10) % 2))
EQ_IDX=$(((TID / 20) % 2))
EQ=${EQS[$EQ_IDX]}
RWS=${RWSS[$RWS_IDX]}
FRM=${FRMS[$FRM_IDX]}

TAG="ptrack_g0_nodale_trainablebias/EqType=${EQ}_N=1000_LmbdRWS=${RWS}_LmbdFR=${FRM}"
mkdir -p "${SAVE_TO}/CDDM_ptrack_g0_nodale_trainablebias"
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  # descriptor of what this data is, next to the data itself, stamped with the code commit
  cp "$REPO/docs/experiments/participation_trace_nodale_bias.md" "${SAVE_TO}/CDDM_ptrack_g0_nodale_trainablebias/EXPERIMENT.md"
  echo -e "\n---\nRun: SLURM array ${SLURM_ARRAY_JOB_ID} on Della, code commit $(git -C "$REPO" rev-parse HEAD)" \
    >> "${SAVE_TO}/CDDM_ptrack_g0_nodale_trainablebias/EXPERIMENT.md"
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: EQ=${EQ} N=1000 Lrws=${RWS} Lfrm=${FRM} rep=${REP} hash=$(git -C "$REPO" rev-parse --short=6 HEAD)"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

# Fixed params (gamma=0, N=1000, max_iter, trainer_tag, track_every) live in config; this only
# selects the tracking trainer and sets the swept IVs.
srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  +experiment=silent_units_N1000 \
  trainer=trainer_ptrack \
  model=rnn_relu_noDale_trainablebias model.equation_type="$EQ" \
  trainer.lambda_rws="$RWS" trainer.lambda_frm="$FRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
