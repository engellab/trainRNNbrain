#!/bin/bash
# THE WHOLE CHAIN OF THE ENTROPY x PRUNE-AND-DUPLICATE TEST, submitted at once (Della login node; it only calls sbatch).
#
# Rules: docs/project_trajectory/2026-11-25_1245_rules-entropy-prune-combo.md and its amendment
# *_entropy-prune-combo-length-and-chain.md (L = 175,000, gpu80 cards, time limits that cannot pass the 400 GPU-hour cap).
# Every step waits for the one it depends on; a step whose parent fails is cancelled (kill-on-invalid-dep).
#   pilots (submitted before) -> pilot check -> stage 1 (3BFF, CDDM) -> stage 1 (Maze) -> stage 2 -> reads -> verdict
# Training runs from ~/trainRNNbrain_epc (pinned at 0558a88, the pilots' commit); reads from ~/trainRNNbrain_epc_ana.
# Arguments: the pilots' three job IDs (3BFF, CDDM, Maze). Prints every job ID.
set -euo pipefail
PFF=$1 PCD=$2 PMZ=$3
export REPO=$HOME/trainRNNbrain_epc
ANA=$HOME/trainRNNbrain_epc_ana
D=$HOME/trainRNNbrain/data
OUT=/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/entropy_prune_combo_reads
LOG=$HOME/trainRNNbrain_logs
TARGET=175000
MCTRL=18374320                                    # the matched Maze controls' stage 1 (task maze-matched-controls)
mkdir -p "$OUT"
cd "$REPO"
B="sbatch --parsable --kill-on-invalid-dep=yes"
A100="--constraint=nomig&gpu80"
ENVS='module purge; module load anaconda3/2024.10; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch-env; export PYTHONPATH='$ANA' OMP_NUM_THREADS=8; cd '$ANA'; EA=trainRNNbrain/experiments_and_analysis;'
CPU="--nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --output=$LOG/%x.%j.out"

CHK=$($B $CPU --job-name=EPCCheck --time=00:20:00 --dependency=afterany:$PFF:$PCD:$PMZ --wrap "$ENVS python \$EA/entropy_prune_combo_pilot_check.py $LOG/EntPrune.${PFF}_10.out $LOG/EntPrune.${PCD}_10.out $LOG/EPCMazePilot.${PMZ}_139.out $D/trained_RNNs_rl_maze_epc_pilot/Maze_interventions/ent0.1_duplicate/s1/s1_Progress.json")
echo "pilot check $CHK"

S="$B --chdir=$REPO --export=ALL $A100"; L=slurm/entropy_prune_combo_della.slurm
FF1=$(TASK=ff   STAGE=1 $S --time=06:15:00 --array=1-3,7-12 --dependency=afterok:$CHK $L)
CD1=$(TASK=cddm STAGE=1 $S --time=04:00:00 --array=1-3,7-12 --dependency=afterok:$CHK $L)
export TARGET
FF2=$(TASK=ff   STAGE=2 $S --time=08:00:00 --array=1-3,7-12 --dependency=aftercorr:$FF1 $L)
CD2=$(TASK=cddm STAGE=2 $S --time=05:05:00 --array=1-3,7-12 --dependency=aftercorr:$CD1 $L)
FF2E=$(TASK=ff   STAGE=2 $S --time=08:00:00 --array=4-6 --dependency=afterok:$CHK $L)
CD2E=$(TASK=cddm STAGE=2 $S --time=05:05:00 --array=4-6 --dependency=afterok:$CHK $L)
echo "3BFF stage 1 $FF1, stage 2 $FF2 $FF2E; CDDM stage 1 $CD1, stage 2 $CD2 $CD2E"

M="$B --job-name=EPCMaze --export=ALL $A100 --time=08:00:00"; ML=slurm/rl_maze_interventions_della.slurm
export SAVE_TO=$D/trained_RNNs_rl_maze_epc
MZ1=$($M --array=52-54,139-141 --dependency=afterok:$CHK,afterany:$FF1:$CD1 $ML)
export EXTRA_UPDATES=2000 INIT_ADAM=true STORE_EVERY=1
MZ2=$(PARENT_ROOT=$SAVE_TO/Maze_interventions $M --array=52-54,139-141 --dependency=aftercorr:$MZ1 $ML)
MZ2E=$(PARENT_ROOT=$D/trained_RNNs_rl_maze_entsp_N1000/Maze_interventions $M --array=130-132 --dependency=afterok:$CHK,afterany:$FF1:$CD1 $ML)
MZ2D=$(PARENT_ROOT=$D/trained_RNNs_rl_maze_mctrl_N1000/Maze_interventions $M --array=1-3 --dependency=afterok:$CHK,aftercorr:$MCTRL $ML)
unset SAVE_TO EXTRA_UPDATES INIT_ADAM STORE_EVERY TARGET
echo "Maze stage 1 $MZ1, stage 2 $MZ2 $MZ2E $MZ2D"

RS=$($B $CPU --job-name=EPCRead --time=06:00:00 --dependency=afterany:$FF2:$FF2E:$CD2:$CD2E --wrap "$ENVS export CUDA_VISIBLE_DEVICES=; python \$EA/entropy_prune_combo_read.py read $OUT/reads.json --until 175000")
RM=$($B --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem-per-gpu=48G $A100 --job-name=EPCReadMaze --time=01:00:00 --output=$LOG/%x.%j.out --dependency=afterany:$RS:$MZ2:$MZ2E:$MZ2D --wrap "$ENVS python \$EA/entropy_prune_combo_read.py maze-read $OUT/reads.json --data $D")
VJ=$($B $CPU --job-name=EPCVerdict --time=01:00:00 --dependency=afterany:$RM --wrap "$ENVS python \$EA/entropy_prune_combo_read.py verdict $OUT/reads.json --fig $OUT/entropy_prune_combo_active > $OUT/verdict.txt 2>&1; python \$HOME/epc_oracle_combo.py $OUT/reads.json > $OUT/oracle.txt 2>&1")
echo "reads $RS (3BFF, CDDM), $RM (Maze), verdict and recount $VJ -> $OUT"
