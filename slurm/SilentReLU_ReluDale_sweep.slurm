#!/bin/bash
# Silent-ReLU baseline sweep (Spock / PNI).
#
# Question: do UNPENALIZED ReLU-Dale CDDM networks leave most units silent (dead ReLUs that
# do not contribute to the computation)? And do the recurrent-weight sparsity penalty
# (lambda_rws) and the firing-rate magnitude penalty (lambda_frm) rescue those dead units and
# prevent silent ReLUs? lambda_rws=0, lambda_frm=0 is the unmodified baseline.
#
# Pinned code commit: 4a031e7 (folder hash "4a031e"). See docs/project_trajectory.md for the
# full experimental record / reproduction recipe.
#
# Grid (120 jobs) = 2 equation types x 2 lambda_rws x 2 lambda_frm x 3 sizes x 5 nets:
#   EQ   in {h, s}            (model.equation_type)
#   RWS  in {0, 0.05}         (trainer.lambda_rws  -- recurrent-weight sparsity penalty)
#   FRM  in {0, 0.2}          (trainer.lambda_frm  -- firing-rate magnitude penalty)
#   N    in {100, 500, 1000}  (model.N)
#   5 reps, seed="random"
#
# Mixed-radix index decode (TID = SLURM_ARRAY_TASK_ID-1, range 0..119):
#   REP     = TID % 5                place value 1
#   N_IDX   = (TID / 5)  % 3         place value 5
#   FRM_IDX = (TID / 15) % 2         place value 15
#   RWS_IDX = (TID / 30) % 2         place value 30
#   EQ_IDX  = (TID / 60) % 2         place value 60
# This is a bijection onto [0,120): TID = 60*EQ_IDX + 30*RWS_IDX + 15*FRM_IDX + 5*N_IDX + REP.
#
# Output tree:
#   $SAVE_TO/CDDM_<hash>/EqType=<EQ>_N=<N>_LmbdRWS=<RWS>_LmbdFR=<FRM>/<per-net subfolder>
# The 5 reps of a condition land in 5 distinct per-net subfolders (named by r2 score).
#
# Overridable env vars (sbatch --export):
#   MAXITER  (default 30000)                      -- training iterations per net
#   SAVE_TO  (default $HOME/trainRNNbrain/data/trained_RNNs)  -- root output dir (used by smoke test)

#SBATCH --job-name=SilentReLU
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=1-120
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/usr/people/pt1290/trainRNNbrain/log/%x.%A_%a.out
#SBATCH --error=/usr/people/pt1290/trainRNNbrain/log/%x.%A_%a.err

module purge
module load anacondapy/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-env

set -eo pipefail

REPO="$HOME/trainRNNbrain"
SAVE_TO="${SAVE_TO:-$REPO/data/trained_RNNs}"
HASH=$(git -C "$REPO" rev-parse --short=6 HEAD)
MAXITER=${MAXITER:-30000}

EQS=(h s)
RWSS=(0 0.05)
FRMS=(0 0.2)
NS=(100 500 1000)

TID=$((SLURM_ARRAY_TASK_ID - 1))
REP=$((TID % 5))
N_IDX=$(((TID / 5)  % 3))
FRM_IDX=$(((TID / 15) % 2))
RWS_IDX=$(((TID / 30) % 2))
EQ_IDX=$(((TID / 60) % 2))

EQ=${EQS[$EQ_IDX]}
RWS=${RWSS[$RWS_IDX]}
FRM=${FRMS[$FRM_IDX]}
N=${NS[$N_IDX]}

TAG="${HASH}/EqType=${EQ}_N=${N}_LmbdRWS=${RWS}_LmbdFR=${FRM}"
echo "Task ${SLURM_ARRAY_TASK_ID}: EQ=${EQ} N=${N} Lrws=${RWS} Lfrm=${FRM} rep=${REP} maxiter=${MAXITER} hash=${HASH}"
echo "  -> ${SAVE_TO}/CDDM_${TAG}"

# trainer.trainer_tag="''" -> empty string, so the leaf folder is exactly the EqType=... tag (no _BaseTrainer suffix).
# experiment_tag is wrapped in literal double quotes so Hydra parses the '=' chars in the value as a quoted string.
srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  seed="random" \
  model=rnn_relu_Dale model.equation_type="$EQ" model.N="$N" \
  trainer.trainer_tag="''" trainer.max_iter="$MAXITER" \
  trainer.lambda_rws="$RWS" trainer.lambda_frm="$FRM" \
  paths.save_to="$SAVE_TO" \
  experiment_tag="\"${TAG}\""
