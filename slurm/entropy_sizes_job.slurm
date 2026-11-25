#!/bin/bash
# ONE JOB OF THE ENTROPY SIZE TEST, BEHIND THE GPU-HOUR GUARD.
# Rules: docs/project_trajectory/2026-11-25_1241_rules-entropy-setpoint-sizes.md (cap of 300 GPU-hours, pilot included).
# Submitted by entropy_sizes_submit_della.sh, which sets time, array, constraint, job name and the launcher's knobs.
# LAUNCHER=<file in $REPO/slurm> runs unchanged after the guard, so nobody has to watch the cap: at its start the job adds
# the elapsed hours of every finished EntSize{FF,CD,Mz,Pilot} job and the full time limit of every running one (its own
# included) and exits before training if the sum exceeds 300. A refused job fails, so its continuations do not start.
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.out
#SBATCH --error=/home/pt1290/trainRNNbrain_logs/%x.%A_%a.err

guard() {  # print the hours used plus reserved; returns 1 without a sacct record of this job
  sacct -u "$USER" -S 2026-11-25T15:30 -X -n -P -o JobName,State,ElapsedRaw,TimelimitRaw,JobIDRaw | awk -F'|' -v me="$SLURM_JOB_ID" '
    $1 ~ /^EntSize(FF|CD|Mz|Pilot)/ { if ($2 ~ /^(RUNNING|COMPLETING)/) h += $4 / 60; else if ($2 !~ /^PENDING/) h += $3 / 3600
                                      if ($5 == me) seen = 1 }
    END { printf "%.1f\n", h; exit !seen }'
}
for k in 1 2 3 4 5 6 7 8 9 10; do
  sleep 60                                     # the accounting record of a job that just started lags
  if H=$(guard); then break; fi
  H=""
done
[ -n "$H" ] || { echo "FATAL: guard found no accounting record of job $SLURM_JOB_ID; not starting" >&2; exit 3; }
echo "guard: $H GPU-hours used or reserved (cap 300)"
awk -v h="$H" 'BEGIN { exit !(h <= 300) }' || { echo "FATAL: $H GPU-hours exceeds the cap; not starting" >&2; exit 3; }
exec bash "$REPO/slurm/$LAUNCHER"
