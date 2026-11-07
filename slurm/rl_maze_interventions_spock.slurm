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
# GRID (54 jobs), one per array index; arm = (i - 1) / 3, seed = (i - 1) % 3 + 1:
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
#   52-54   duplicate         trainer.prune_reinit=true and the paper's prune-and-duplicate arm as
#                             swept keys (slurm/SilentReLU_paper_grid_della.slurm lines 126-133, with
#                             the cadence that arm inherits from trainer_ptrack_freshbatch.yaml:
#                             reinit_mode copy, copy_noise 1.0 with the row norm preserved, at most
#                             0.025 N replaced per check by contribution utility, maturity 1,000, a
#                             check every 100 optimizer steps, patience 3). The counts are in
#                             optimizer steps (8 per update): 160 checks per run against 400 on the
#                             flip-flop at 40,000 iterations; a unit is doomed after 300 silent steps
#                             and protected for 125 updates after a replacement. The one Figure 5
#                             per-unit rule TrainerPPO did not implement before 5 Nov (issue #6)
# The launcher passes only what an arm sweeps, plus N; every other value is the config's.
# Expected: about 5.5 h per run at N = 1,000 on a 40 GB card (the suite's rate), 24 runs.
#   55-63   srec0, srec0.01, srec0.1   model.sigma_rec=0 / 0.01 / 0.1 (the config's 0.05 is the
#                             control): the activity-noise rung of the standard-settings ladder,
#                             the CDDM values (issue #6, package 2, Pavel's go-ahead of 5 Nov)
#   64-69   srad0.5, srad1.5  model.spectral_rad=0.5 / 1.5 (the config's 1.2 is the control): the
#                             initial recurrent spectral radius. CDDM's ladder has 2.0; on the maze
#                             (2,500 RNN steps per episode) an untrained network at 2.0 is not finite
#                             in its first rollout (local smoke test, N = 100), so the upper rung is
#                             1.5, which runs but starts with a huge first-update KL; the rollback
#                             guard reports what happens at N = 1,000
#   70-72   rws0.05           trainer.lambda_rws=0.05: the sparsity penalty alone (the pair with
#                             the rate penalty is arm 5)
#   73-75   cap0.3            model.inp_weight_cap=0.3 (CDDM's value): |W_inp| clamped to 0.3 after
#                             every optimizer step (TrainerPPO.enforce_inp_cap, the supervised
#                             Trainer.enforce_inp_cap_); the initial scale is 1/sqrt(1000) = 0.032.
#                             Issue #6, package 3 (Pavel's go-ahead of 5 Nov), as are 76-114
#   76-81   rwm0.05, rwm0.2   trainer.lambda_rwm=0.05 / 0.2: the recurrent-weight magnitude penalty
#                             (TrainerPPO.rec_weights_magnitude_penalty) at CDDM's two values, with
#                             rwm_args at the supervised defaults (trainer_ppo.yaml)
#   82-93   met0.01, met0.1, met1, met10   trainer.lambda_met=0.01 / 0.1 / 1 / 10: the metabolic
#                             penalty, the mean squared rate (TrainerPPO.metabolic_penalty), CDDM's
#                             ladder
#   94-96   adamw             trainer.optimizer=adamw: decoupled weight decay at the config's 1e-6
#                             (run_rl.build_optimizer, the B5 switch of run_experiment.py)
#   97-108  sgd0.03, sgd0.1, sgd0.3, sgd1   trainer.optimizer=sgd trainer.lr=<v>: SGD (momentum 0,
#                             weight decay 1e-6) at the four rates of the B5 ladder that learned
#                             the flip-flop (3 and 10 did not); the size rule and the warm-up ramp
#                             apply as to Adam, so the rate at N = 1,000 is 0.464 x <v>
#   115-126 sigmaw0.1, sigmaw0.3, dropout0.05_beta4, rescale_tgt4   (round 18 parity, 7 Nov 2026: the 3BFF
#                             values that pass the paper's criterion, run on Maze: model.sigma_w 0.1 and 0.3
#                             (submit with --time=24:00:00, about 16 h per run), read-out dropout at
#                             drop rate 0.05 with beta 4, and the rescale arm above with target 4.0 (the
#                             3BFF revive_ops2 cell); docs/project_trajectory/*_round18-parity-rules.md)
#   109-114 redraw, zeroout   trainer.prune_reinit=true and the paper's re-initialisation arms as
#                             swept keys (slurm/SilentReLU_replace_variants_della.slurm, modes random
#                             and zero_out: max_replace_frac 0.025, maturity 1000, utility
#                             contribution, with the cadence of trainer_ptrack_freshbatch.yaml,
#                             check_every 100, patience 3, as the duplicate arm): a doomed unit's
#                             incoming rows of W_rec and W_inp redrawn from N(0, 1/sqrt(N)); zeroout
#                             also zeroes its outgoing columns (Dohare et al. 2024). The counts are
#                             in optimizer steps, as for the duplicate arm
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
#SBATCH --array=1-126%126
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
         wd0 wd1e-5 wd1e-4 inorm0.5 inorm2 inorm5 inorm20 duplicate
         srec0 srec0.01 srec0.1 srad0.5 srad1.5 rws0.05
         cap0.3 rwm0.05 rwm0.2 met0.01 met0.1 met1 met10 adamw sgd0.03 sgd0.1 sgd0.3 sgd1 redraw zeroout
         sigmaw0.1 sigmaw0.3 dropout0.05_beta4 rescale_tgt4)
ARM_OVERRIDES=("" "trainer.lambda_frm=0.03" "trainer.lambda_frm=0.1" "trainer.lambda_frm=0.3"
               "trainer.lambda_frm=0.1 trainer.lambda_rws=0.05" "model.sigma_w=1.0" "trainer.dropout=true"
               "trainer.prune_reinit=true trainer.prune_args.reinit_mode=rescale trainer.prune_args.rescale_norm_mode=self trainer.prune_args.rescale_target_frac=10.0 trainer.prune_args.rescale_active_only=true trainer.prune_args.rescale_normalize=true trainer.prune_args.rescale_alpha=1.002 trainer.prune_args.rescale_cap=64.0 trainer.prune_args.rescale_protect_frac=0.20 trainer.prune_args.check_every=1 trainer.prune_args.patience=1"
               "trainer.prune_reinit=true trainer.prune_args.reinit_mode=rescale trainer.prune_args.rescale_norm_mode=self trainer.prune_args.rescale_target_frac=10.0 trainer.prune_args.rescale_active_only=true trainer.prune_args.rescale_normalize=true trainer.prune_args.rescale_alpha=1.0005 trainer.prune_args.rescale_cap=8.0 trainer.prune_args.rescale_protect_frac=0.20 trainer.prune_args.check_every=1 trainer.prune_args.patience=1"
               "trainer.prune_reinit=true trainer.prune_args.reinit_mode=rescale trainer.prune_args.rescale_norm_mode=self trainer.prune_args.rescale_target_frac=10.0 trainer.prune_args.rescale_active_only=true trainer.prune_args.rescale_normalize=true trainer.prune_args.rescale_alpha=1.002 trainer.prune_args.rescale_cap=64.0 trainer.prune_args.rescale_protect_frac=0.05 trainer.prune_args.check_every=1 trainer.prune_args.patience=1"
               "trainer.weight_decay=0" "trainer.weight_decay=1e-5" "trainer.weight_decay=1e-4"
               "model.input_row_norm=0.5" "model.input_row_norm=2" "model.input_row_norm=5" "model.input_row_norm=20"
               "trainer.prune_reinit=true trainer.prune_args.reinit_mode=copy trainer.prune_args.copy_noise=1.0 trainer.prune_args.copy_norm_preserve=true trainer.prune_args.max_replace_frac=0.025 trainer.prune_args.maturity=1000 trainer.prune_args.utility=contribution trainer.prune_args.check_every=100 trainer.prune_args.patience=3"
               "model.sigma_rec=0" "model.sigma_rec=0.01" "model.sigma_rec=0.1"
               "model.spectral_rad=0.5" "model.spectral_rad=1.5" "trainer.lambda_rws=0.05"
               "model.inp_weight_cap=0.3" "trainer.lambda_rwm=0.05" "trainer.lambda_rwm=0.2"
               "trainer.lambda_met=0.01" "trainer.lambda_met=0.1" "trainer.lambda_met=1" "trainer.lambda_met=10"
               "trainer.optimizer=adamw"
               "trainer.optimizer=sgd trainer.lr=0.03" "trainer.optimizer=sgd trainer.lr=0.1"
               "trainer.optimizer=sgd trainer.lr=0.3" "trainer.optimizer=sgd trainer.lr=1"
               "trainer.prune_reinit=true trainer.prune_args.reinit_mode=random trainer.prune_args.max_replace_frac=0.025 trainer.prune_args.maturity=1000 trainer.prune_args.utility=contribution trainer.prune_args.check_every=100 trainer.prune_args.patience=3"
               "trainer.prune_reinit=true trainer.prune_args.reinit_mode=zero_out trainer.prune_args.max_replace_frac=0.025 trainer.prune_args.maturity=1000 trainer.prune_args.utility=contribution trainer.prune_args.check_every=100 trainer.prune_args.patience=3"
               "model.sigma_w=0.1" "model.sigma_w=0.3"
               "trainer.dropout=true trainer.dropout_args.drop_rate=0.05 trainer.dropout_args.dropout_beta=4.0"
               "trainer.prune_reinit=true trainer.prune_args.reinit_mode=rescale trainer.prune_args.rescale_norm_mode=self trainer.prune_args.rescale_target_frac=4.0 trainer.prune_args.rescale_active_only=true trainer.prune_args.rescale_normalize=true trainer.prune_args.rescale_alpha=1.002 trainer.prune_args.rescale_cap=64.0 trainer.prune_args.rescale_protect_frac=0.20 trainer.prune_args.check_every=1 trainer.prune_args.patience=1")

i=$SLURM_ARRAY_TASK_ID
a=$(( (i - 1) / 3 )); SEED=$(( (i - 1) % 3 + 1 ))
N=1000
TAG="interventions/${ARM_TAG[$a]}"
echo "Task ${i}: arm=${ARM_TAG[$a]} N=${N} seed=${SEED} overrides='${ARM_OVERRIDES[$a]}' -> ${SAVE_TO}/Maze_${TAG}/s${SEED}"

# shellcheck disable=SC2086
srun python "$REPO/trainRNNbrain/training/run_experiment.py" \
  task=Maze trainer=trainer_ppo seed="$SEED" model=rnn_relu_standard model.N="$N" \
  paths=paths_local paths.save_to="$SAVE_TO" experiment_tag="\"${TAG}\"" ${ARM_OVERRIDES[$a]}
