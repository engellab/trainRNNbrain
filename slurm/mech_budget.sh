#!/bin/bash
# GPU-hour cap of the mechanism tests, enforced by the jobs themselves (sourced by slurm/mech_*_della.slurm).
#
# Every GPU task holds one file in $LEDGER. While the task runs the file holds its time limit in seconds; when the task
# ends the file holds the seconds it ran. A task starts only if the sum of all files plus its own time limit is at most
# CAP_S, so the cap holds whatever the node speeds are and whether or not anybody is watching. A task killed without
# running its exit trap stays counted at its full time limit, which is what it used.
LEDGER="${LEDGER:-$HOME/mech_budget}"
CAP_S=$((79 * 3600))     # 79 of the 80 GPU-hours; the last one covers job start-up, which SECONDS does not see

budget_start() {
  # $1: this task's time limit in seconds. Exits 1 if starting it could pass the cap.
  mkdir -p "$LEDGER"
  BUDGET_FILE="$LEDGER/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID}_${SLURM_RESTART_COUNT:-0}.task"
  exec 9>"$LEDGER/lock"
  flock 9
  local used=0 f
  for f in "$LEDGER"/*.task; do [ -e "$f" ] && used=$((used + $(cat "$f"))); done
  if [ $((used + $1)) -gt "$CAP_S" ]; then
    flock -u 9
    echo "FATAL: GPU budget: ${used} s used or reserved, this task could add $1 s, cap ${CAP_S} s; not started." >&2
    exit 1
  fi
  echo "$1" > "$BUDGET_FILE"
  flock -u 9
  echo "GPU budget: ${used} s used or reserved before this task, cap ${CAP_S} s"
  trap 'echo $SECONDS > "$BUDGET_FILE"' EXIT
}
