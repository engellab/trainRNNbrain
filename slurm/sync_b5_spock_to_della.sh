#!/bin/bash
# MIRROR THE B5 SWEEP FROM SPOCK TO DELLA, through this Mac (the clusters cannot reach each other:
# no host keys, and Della wants Duo). Incremental and idempotent: rsync without --delete, so a run
# caught half-written is completed on the next pass and nothing on Della is ever removed.
# Needs the multiplexed `ssh spock` / `ssh della` sessions alive (ControlPersist). Run by cron:
#   0 */2 * * * bash ~/Documents/GitHub/trainRNNbrain_b5/slurm/sync_b5_spock_to_della.sh
# Log: ~/Library/Logs/b5_sync.log.
ROOT=NBitFlipFlop_b5_optimizer
MAC=$HOME/Documents/GitHub/trainRNNbrain/data/trained_RNNs
LOG=$HOME/Library/Logs/b5_sync.log
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
{
  echo "== $(date '+%F %T')"
  rsync -a -e "$SSH" "spock:trainRNNbrain/data/trained_RNNs/$ROOT" "$MAC/" &&
    rsync -a -e "$SSH" "$MAC/$ROOT" "della:trainRNNbrain/data/trained_RNNs/" &&
    echo "ok: $(find "$MAC/$ROOT" -name '*LastParams*' | wc -l | tr -d ' ') finished runs mirrored"
} >> "$LOG" 2>&1
