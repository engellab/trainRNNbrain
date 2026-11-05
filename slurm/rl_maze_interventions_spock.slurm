#!/bin/bash
# THE PAPER'S INTERVENTIONS ON THE MAZE (SPOCK), N = 1,000: do the remedies that keep units active on
# the 3-bit flip-flop (Figure 5) do so under PPO on the maze? The rate penalty at three weights, the
# rate penalty with the sparsity penalty, noisy synapses and read-out dropout, each against the
# unpenalised control, 3 seeds per arm. The task and the trainer are the suite's
# (configs/task/Maze.yaml, configs/trainer/trainer_ppo.yaml: 2,000 updates of 256 episodes x 500
# decisions, 5 RNN steps each), with one option of TrainerPPO switched on per arm; every definition
# mirrors the supervised one (trainRNNbrain/trainer/TrainerPPO.py quotes them).
# Tracking issue: engellab/trainRNNbrain #5, which holds the pre-registration: the paper's values
# first, with one lower and one higher rate-penalty weight because the PPO loss is on another scale
# than the squared error; read-out as in #3; "works" = more active units than every control at a
# probe success within 0.05 of the controls'. The notebook entry written at submission names it.
#
# GRID (21 jobs), one per array index; arm = (i - 1) / 3, seed = (i - 1) % 3 + 1:
#   1-3     control           no option
#   4-6     frm0.03           trainer.lambda_frm=0.03
#   7-9     frm0.1            trainer.lambda_frm=0.1                          (the paper's flip-flop value)
#   10-12   frm0.3            trainer.lambda_frm=0.3
#   13-15   frm0.1_rws0.05    trainer.lambda_frm=0.1 trainer.lambda_rws=0.05  (the paper's pair)
#   16-18   sigmaw1.0         model.sigma_w=1.0                               (the paper's value; the
#                             recompute regenerates every step's weight draws, forward and backward,
#                             so this arm runs slower than the others: allow about 1.5x)
#   19-21   dropout0.2_beta4  trainer.dropout=true                            (drop_rate 0.2, beta 4,
#                             kind mute, rescale on: the dropout_args block of trainer_ppo.yaml)
# The launcher passes only what an arm sweeps, plus N; every other value is the config's.
# Expected: about 5.5 h per run at N = 1,000 on a 40 GB card (the suite's rate), 21 runs.
#
# SUBMIT:  sbatch slurm/rl_maze_interventions_spock.slurm
#
#SBATCH --job-name=MazeInterv
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=1-21%21
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

REPO="${REPO:-$HOME/trainRNNbrain_rlmaze}"
SAVE_TO="${SAVE_TO:-$HOME/trainRNNbrain/data/trained_RNNs_rl_maze}"   # the maze suite's own tree, a sibling
                                                                     # of the supervised runs' folder, where every
                                                                     # maze run so far is; run_rl.py writes the
                                                                     # standard layout under it

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PKG=$(python -c "import trainRNNbrain, os; print(os.path.realpath(os.path.dirname(os.path.dirname(trainRNNbrain.__file__))))")
REPO_REAL=$(readlink -f "$REPO")
if [ "$PKG" != "$REPO_REAL" ]; then
  echo "FATAL: trainRNNbrain resolves to $PKG, expected $REPO_REAL (from $REPO); aborting." >&2; exit 1
fi
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || true)
if [ -z "$GITSHA" ]; then
  echo "FATAL: $REPO is not a git checkout; the provenance of this run cannot be recorded." >&2; exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
  echo "FATAL: $REPO has uncommitted changes; $GITSHA does not describe this run." >&2; exit 1
fi
echo "code: $PKG @ $GITSHA"

# one entry per arm; the overrides of an arm are separate words on the command line
ARM_TAG=(control frm0.03 frm0.1 frm0.3 frm0.1_rws0.05 sigmaw1.0 dropout0.2_beta4)
ARM_OVERRIDES=("" "trainer.lambda_frm=0.03" "trainer.lambda_frm=0.1" "trainer.lambda_frm=0.3"
               "trainer.lambda_frm=0.1 trainer.lambda_rws=0.05" "model.sigma_w=1.0" "trainer.dropout=true")

i=$SLURM_ARRAY_TASK_ID
a=$(( (i - 1) / 3 )); SEED=$(( (i - 1) % 3 + 1 ))
N=1000
TAG="interventions/${ARM_TAG[$a]}"
echo "Task ${i}: arm=${ARM_TAG[$a]} N=${N} seed=${SEED} overrides='${ARM_OVERRIDES[$a]}' -> ${SAVE_TO}/Maze_${TAG}/s${SEED}"

# shellcheck disable=SC2086
srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  task=Maze trainer=trainer_ppo seed="$SEED" model=rnn_relu_standard model.N="$N" \
  paths=paths_local paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\"" ${ARM_OVERRIDES[$a]}
