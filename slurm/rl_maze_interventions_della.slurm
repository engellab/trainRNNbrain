#!/bin/bash
# THE MAZE INTERVENTIONS ON DELLA (round 19, 7 Nov 2026): Della's #SBATCH lines around the Spock launcher.
# The grid, the arms, the continuation mode (PARENT_ROOT, EXTRA_UPDATES, INIT_ADAM, STAGE) and every check are
# slurm/rl_maze_interventions_spock.slurm's, run from the same pinned checkout REPO with CLUSTER=della, which
# changes only the module lines. Parents of a continuation must be copied to Della first, under PARENT_ROOT.
#   REPO=$HOME/trainRNNbrain_r19 PARENT_ROOT=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_interventions \
#     EXTRA_UPDATES=2000 INIT_ADAM=false sbatch --array=91-108 --time=12:00:00 slurm/rl_maze_interventions_della.slurm
# The noisy-synapse arms (sigma_w, indices 16-18 and 115-120) run about 28 s per update: --time=24:00:00.
#
#SBATCH --job-name=MazeIntervD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --mem-per-gpu=48G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.err

set -eo pipefail
REPO="${REPO:-$HOME/trainRNNbrain_r19}"
export REPO
export CLUSTER=della
exec bash "$REPO/slurm/rl_maze_interventions_spock.slurm"
