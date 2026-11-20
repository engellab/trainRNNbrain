#!/bin/bash
# CLAIM GAPS (19 Nov 2026): submit gaps A, B and C on Della.
# Rules: docs/project_trajectory/2026-11-19_1749_rules-{maze-duplicate,cddm-rate-penalty,maze-rate-penalty}-sizes.md
# Run from the Mac:  ssh della 'bash -s' < slurm/claim_gaps_submit_della.sh
#
# Maze, arms frm0.1 (indices 7-9) and duplicate (52-54):
#   N = 500 and 2,000: stage 1 (2,000 updates), then stage 2 (2,000 more, Adam state restored), chained per
#   array index with aftercorr; each size in its own tree, trained_RNNs_rl_maze_cg_N<N>.
#   N = 1,000: the round-19 fz runs continued 2,000 -> 4,000 into trained_RNNs_rl_maze_cg_N1000 (parents read only).
# CDDM, rate penalty 0.1 (penalties launcher indices 4-6) at N = 500 and 2,000, 75,000 iterations, into trained_RNNs_cg.
# Every job is 23 h or less (gpu-short); no existing output tree is written.
set -euo pipefail
REPO=$HOME/trainRNNbrain_cg
PIN=07c0a83
D=$HOME/trainRNNbrain/data
FZ=$D/trained_RNNs_rl_maze_fz/Maze_interventions

[ "$(git -C "$REPO" rev-parse --short HEAD)" = "$PIN" ] || { echo "FATAL: $REPO is not at $PIN" >&2; exit 1; }
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "FATAL: $REPO is dirty" >&2; exit 1; }
for a in frm0.1 duplicate; do for s in 1 2 3; do
  ls "$FZ/$a/s$s/"*AdamState_Maze.pt >/dev/null || { echo "FATAL: no Adam state in $FZ/$a/s$s" >&2; exit 1; }
done; done
for t in trained_RNNs_rl_maze_cg_N500 trained_RNNs_rl_maze_cg_N1000 trained_RNNs_rl_maze_cg_N2000 trained_RNNs_cg; do
  [ ! -e "$D/$t" ] || { echo "FATAL: $D/$t exists; this script writes new trees only" >&2; exit 1; }
done
cd "$REPO"
export REPO

for N in 500 2000; do
  if [ "$N" = 500 ]; then T1=12:00:00; T2=12:00:00; else T1=20:00:00; T2=20:00:00; fi
  OUT=$D/trained_RNNs_rl_maze_cg_N$N
  J1=$(N_OVERRIDE=$N SAVE_TO=$OUT sbatch --parsable --chdir="$REPO" --job-name=CGmaze1_N$N --time=$T1 \
       --array=7-9,52-54 slurm/rl_maze_interventions_della.slurm)
  J2=$(N_OVERRIDE=$N SAVE_TO=$OUT PARENT_ROOT=$OUT/Maze_interventions EXTRA_UPDATES=2000 INIT_ADAM=true \
       sbatch --parsable --chdir="$REPO" --job-name=CGmaze2_N$N --time=$T2 --dependency=aftercorr:$J1 \
       --array=7-9,52-54 slurm/rl_maze_interventions_della.slurm)
  echo "maze N=$N stage1 $J1 stage2 $J2 -> $OUT"
done

J=$(SAVE_TO=$D/trained_RNNs_rl_maze_cg_N1000 PARENT_ROOT=$FZ EXTRA_UPDATES=2000 INIT_ADAM=true \
    sbatch --parsable --chdir="$REPO" --job-name=CGmaze2_N1000 --time=12:00:00 --array=7-9,52-54 \
    slurm/rl_maze_interventions_della.slurm)
echo "maze N=1000 continuation $J -> $D/trained_RNNs_rl_maze_cg_N1000"

for N in 500 2000; do
  if [ "$N" = 500 ]; then T=08:00:00; else T=23:00:00; fi
  J=$(N_OVERRIDE=$N SAVE_TO=$D/trained_RNNs_cg TASK=cddm sbatch --parsable --chdir="$REPO" --job-name=CGcddm_N$N \
      --time=$T --array=4-6 slurm/SilentReLU_std_penalties_della.slurm)
  echo "cddm N=$N $J -> $D/trained_RNNs_cg"
done
