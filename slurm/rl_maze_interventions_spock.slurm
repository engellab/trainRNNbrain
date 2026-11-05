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
# GRID (51 jobs), one per array index; arm = (i - 1) / 3, seed = (i - 1) % 3 + 1:
#   1-3     control           no option
#   4-6     frm0.03           trainer.lambda_frm=0.03
#   7-9     frm0.1            trainer.lambda_frm=0.1                          (the paper's flip-flop value)
#   10-12   frm0.3            trainer.lambda_frm=0.3
#   13-15   frm0.1_rws0.05    trainer.lambda_frm=0.1 trainer.lambda_rws=0.05  (the paper's pair)
#   16-18   sigmaw1.0         model.sigma_w=1.0                               (the paper's value; the
#                             recompute regenerates every step's weight draws, forward and backward,
#                             so this arm runs slower than the others, about 2 to 2.5x on a
#                             GPU: submit indices 16-18 with --time=24:00:00)
#   19-21   dropout0.2_beta4  trainer.dropout=true                            (drop_rate 0.2, beta 4,
#                             kind mute, rescale on: the dropout_args block of trainer_ppo.yaml)
#   22-24   rescale           trainer.prune_reinit=true and the paper's rescaling arm as swept keys
#                             (slurm/SilentReLU_size_series_della.slurm lines 124-133: reinit_mode
#                             rescale, norm_mode self, target 10 x the rate cap, active sources
#                             only, normalised rows, alpha 1.002 per optimizer step, cap 64, protect
#                             0.2 N, a check every optimizer step, patience 1); stage 2 of issue #5,
#                             added after indices 1-21 were submitted (7383696, 7383697)
#   25-27   rescale_mild      the rescaling arm with the supervised defaults for the tilt and the
#                             cap (trainer_ptrack_freshbatch.yaml: alpha 1.0005 per optimizer step,
#                             cap 8, so an episode ends after log(8)/log(1.0005) = 4,159 steps, 520
#                             updates, instead of 2,083 steps at alpha 1.002 and cap 64); every
#                             other key the paper's. Added 5 Nov after the three paper-value runs
#                             (22-24) blew up at the end of the warm-up ramp (updates 52-60, the
#                             rollback guard stopped them): the question is whether a slower tilt
#                             keeps the recurrent dynamics finite under PPO.
#   28-30   rescale_few       the paper's arm with rescale_protect_frac 0.05 instead of 0.20: at
#                             most 50 of the 1,000 units growing at once, so the tilt moves a
#                             twentieth of the recurrent columns instead of a fifth. Same question.
#   31-39   wd0, wd1e-5, wd1e-4   trainer.weight_decay=0 / 1e-5 / 1e-4 (the config's 1e-6 is the
#                             control): the weight-decay rung of the standard-settings ladder
#                             (Figure 3 on CDDM: 0, 1e-6, 1e-5, 1e-4), the same values on every task
#                             (issue #6, Pavel's 5 Nov instruction: the same settings on all three tasks)
#   40-51   inorm0.5, inorm2, inorm5, inorm20   model.input_row_norm=0.5 / 2 / 5 / 20: every W_inp
#                             row rescaled to this norm at initialisation (the 3BFF and CDDM ladders'
#                             values; the default draw's mean row norm is about 0.05 at N = 1,000)
# The launcher passes only what an arm sweeps, plus N; every other value is the config's.
# Expected: about 5.5 h per run at N = 1,000 on a 40 GB card (the suite's rate), 24 runs.
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
#SBATCH --array=1-51%51
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
# indices 1-3 (control) duplicate the uniform series' N = 1000 ReLU runs (same seeds, the
# same code with every option off): do not submit them; the series' runs are the controls.
ARM_TAG=(control frm0.03 frm0.1 frm0.3 frm0.1_rws0.05 sigmaw1.0 dropout0.2_beta4 rescale rescale_mild rescale_few
         wd0 wd1e-5 wd1e-4 inorm0.5 inorm2 inorm5 inorm20)
ARM_OVERRIDES=("" "trainer.lambda_frm=0.03" "trainer.lambda_frm=0.1" "trainer.lambda_frm=0.3"
               "trainer.lambda_frm=0.1 trainer.lambda_rws=0.05" "model.sigma_w=1.0" "trainer.dropout=true"
               "trainer.prune_reinit=true trainer.prune_args.reinit_mode=rescale trainer.prune_args.rescale_norm_mode=self trainer.prune_args.rescale_target_frac=10.0 trainer.prune_args.rescale_active_only=true trainer.prune_args.rescale_normalize=true trainer.prune_args.rescale_alpha=1.002 trainer.prune_args.rescale_cap=64.0 trainer.prune_args.rescale_protect_frac=0.20 trainer.prune_args.check_every=1 trainer.prune_args.patience=1"
               "trainer.prune_reinit=true trainer.prune_args.reinit_mode=rescale trainer.prune_args.rescale_norm_mode=self trainer.prune_args.rescale_target_frac=10.0 trainer.prune_args.rescale_active_only=true trainer.prune_args.rescale_normalize=true trainer.prune_args.rescale_alpha=1.0005 trainer.prune_args.rescale_cap=8.0 trainer.prune_args.rescale_protect_frac=0.20 trainer.prune_args.check_every=1 trainer.prune_args.patience=1"
               "trainer.prune_reinit=true trainer.prune_args.reinit_mode=rescale trainer.prune_args.rescale_norm_mode=self trainer.prune_args.rescale_target_frac=10.0 trainer.prune_args.rescale_active_only=true trainer.prune_args.rescale_normalize=true trainer.prune_args.rescale_alpha=1.002 trainer.prune_args.rescale_cap=64.0 trainer.prune_args.rescale_protect_frac=0.05 trainer.prune_args.check_every=1 trainer.prune_args.patience=1"
               "trainer.weight_decay=0" "trainer.weight_decay=1e-5" "trainer.weight_decay=1e-4"
               "model.input_row_norm=0.5" "model.input_row_norm=2" "model.input_row_norm=5" "model.input_row_norm=20")

i=$SLURM_ARRAY_TASK_ID
a=$(( (i - 1) / 3 )); SEED=$(( (i - 1) % 3 + 1 ))
N=1000
TAG="interventions/${ARM_TAG[$a]}"
echo "Task ${i}: arm=${ARM_TAG[$a]} N=${N} seed=${SEED} overrides='${ARM_OVERRIDES[$a]}' -> ${SAVE_TO}/Maze_${TAG}/s${SEED}"

# shellcheck disable=SC2086
srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  task=Maze trainer=trainer_ppo seed="$SEED" model=rnn_relu_standard model.N="$N" \
  paths=paths_local paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\"" ${ARM_OVERRIDES[$a]}
