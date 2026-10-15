## 2026-08-28 — Della brought online; `both` N=1000 moved there

**Della setup.** It held the repo but with PRE-FIX code (no `storable_`, no chunked
`participation_from_states_`) and no flip-flop data. Current code synced to `~/trainRNNbrain_ff`;
environment verified end to end (torch 2.7.0+cu128, repo resolves correctly, both fixes importable).
`CODE_VERSION.txt` set to `95320a6+trainer-chunked-participation-2026-08-25`, matching Spock.

**Submitted:** job 13098724, `both` k=1..8 × N=1000 × 3 seeds, 400k iterations
([`SilentReLU_flipflop_both_N1000_della.slurm`](../slurm/SilentReLU_flipflop_both_N1000_della.slurm)).

**Why this cell.** `both` is the ONLY condition whose PR law cannot be fitted — it needs k≥3 at ≥2
sizes and has k=1..4 at N=500 only. On Spock its tasks queued behind the frm backlog on ~37 shared
slots, putting the second size at 1-2 September. Della runs all 24 in one wave.

⚠️ **Della assigns QOS by WALL TIME and it caps concurrency**: <2h → gputest, 3 jobs; 2-24h →
gpu-short, 44; 24-72h → gpu-medium, 24; 72-144h → gpu-long, **10**. Requested **48h**: 1.65x margin
over the measured 29h AND keeps all 24 tasks in a single gpu-medium wave. Asking for 72h+ would have
halved concurrency to 10 for no benefit. This is a trap — the natural instinct to request generous
wall time makes the array slower.

⚠️ **A DECODE GUARD was added to the launcher.** The index arithmetic is byte-identical to the Spock
version so task 76 means the same cell on both clusters, but a mistyped `--array` would then silently
run the wrong cell and contradict Spock rather than erroring. The script aborts unless the task
decodes to `both`/N=1000; verified it rejects tasks 73 and 82 (both N=500).

**Spock duplicates cancelled** — the 24 matching tasks (5904342_76, _77, _78, _85, ...) were
identified by decoding every queued index, confirmed **all PENDING with none running** before
cancelling, so no compute was lost. Recheck confirms 0 both/N=1000 tasks remain queued on Spock, and
the freed slots go to the frm backlog.

⚠️ **Della writes to ITS OWN `~/trainRNNbrain/data/trained_RNNs`.** Those cells must be synced back to
the local mirror before any analysis — every analysis script reads the local copy, not a cluster.
This is a new failure mode: results can exist and still be invisible to the analysis.

**Queue after the move:** Spock 12 FFbigN running + 36 FFpenL running + 5 FFpenL / 1 FFbigN / 2 FFk
pending; Della 24 FFbothD.

---

