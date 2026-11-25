#!/bin/bash
# MECHANISM TESTS (25 Nov 2026): submit the training chain on Della, once.
# Rules: docs/project_trajectory/2026-11-25_1311_rules-mechanism-noise-decay-and-switch.md (committed and pushed first).
# Run from the Mac:  ssh della 'bash -s' < slurm/mech_submit_della.sh
# Part A tasks 1-6, then 7-12, then part B, each after the one before has ended (afterany: a failed task must not
# block the rest; the part B launcher refuses a network whose parent has not finished). mech_budget.sh holds the cap.
set -euo pipefail
REPO=$HOME/trainRNNbrain_mech
PIN=${PIN:?set PIN to the short commit of the rules entry}
[ "$(git -C "$REPO" rev-parse --short HEAD)" = "$PIN" ] || { echo "FATAL: $REPO is not at $PIN" >&2; exit 1; }
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "FATAL: $REPO is dirty" >&2; exit 1; }
D=/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs
for t in CDDM_mech_noise_decay CDDM_mech_switch; do
  [ ! -e "$D/$t" ] || { echo "FATAL: $D/$t exists; this script submits a new chain only" >&2; exit 1; }
done
[ ! -e "$HOME/mech_budget" ] || { echo "FATAL: ~/mech_budget exists" >&2; exit 1; }
A1=$(sbatch --parsable --chdir="$REPO" --export=ALL --array=1-6 "$REPO/slurm/mech_noise_decay_della.slurm")
A2=$(sbatch --parsable --chdir="$REPO" --export=ALL --dependency=afterany:"$A1" --array=7-12 "$REPO/slurm/mech_noise_decay_della.slurm")
B=$(sbatch --parsable --chdir="$REPO" --export=ALL --dependency=afterany:"$A2" --array=1-9 "$REPO/slurm/mech_switch_della.slurm")
echo "part A tasks 1-6: $A1; part A tasks 7-12: $A2; part B: $B"
