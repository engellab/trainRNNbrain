#!/bin/bash
# MECHANISM TESTS (25 Nov 2026): submit the CPU reads on Della, after the training chain (slurm/mech_submit_della.sh).
# Run from the Mac:  ssh della 'A1=<id> A2=<id> B=<id> bash -s' < slurm/mech_submit_reads_della.sh
# The reads of the networks that existed before need no dependency. The others wait for part B to end.
set -euo pipefail
REPO=$HOME/trainRNNbrain_mechana
L=slurm/mech_read_della.slurm
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "FATAL: $REPO is dirty" >&2; exit 1; }
for d in trained_RNNs c1_snapshots; do
  [ -e "$REPO/data/$d" ] || ln -s "/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/$d" "$REPO/data/$d"
done
OLD=$(sbatch --parsable --chdir="$REPO" --array=30-42 "$REPO/$L")
C=$(sbatch --parsable --chdir="$REPO" --dependency=afterany:"$A1":"$A2" --array=10-21 "$REPO/$L")
REM=$(sbatch --parsable --chdir="$REPO" --dependency=afterany:"$B" --array=1-9 "$REPO/$L")
# the read runs after the removal jobs, so its verdict text includes them
RD=$(sbatch --parsable --chdir="$REPO" --dependency=afterany:"$REM" --array=0 "$REPO/$L")
echo "old networks: $OLD; part A weights: $C; removal: $REM; read and verdict: $RD"
