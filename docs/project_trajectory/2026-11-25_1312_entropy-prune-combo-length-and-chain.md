## 2026-11-25 13:12 — Entropy x prune-and-duplicate: L = 175,000, the cards and time limits, and the whole chain submitted at once

Amends `2026-11-25_1245_rules-entropy-prune-combo.md`, before any stage 1 job of this test has run and before any count
of it has been read. Only pilot logs (timing, score, skipped updates) have been looked at.

**Why.** Firstmate (25 Nov): the Della connection may be down for long stretches, so the whole chain is submitted now with
Slurm dependencies, and the 400 GPU-hour cap must hold with nobody watching. The rules' budget step (choose L from
stage 1 timings; stop and ask if the projection passes the cap) needs somebody watching. The step is replaced by limits
that cannot pass the cap.

**Pilot timings so far** (40 GB `i14` cards, 2,000 iterations): 3BFF D 528 s, P 524 s, EP 552 s; CDDM D 379 s. None of
these four skipped an update. The existing entropy-alone jobs give the spread between cards: on 80 GB cards 3BFF took
5.20 to 5.23 h for 75,000 iterations and CDDM 3.23 to 3.30 h; on 40 GB `i14` cards 5.77 h and 4.06 h. Maze took 6.8 to
7.1 h for 2,000 updates on 80 GB cards and 8.8 to 13.9 h on 40 GB ones.

**Changes.**
1. **L = 175,000** on 3BFF and CDDM. At 200,000 the limits below would add up to more than 400.
2. **Cards.** Every new job runs with `--constraint="nomig&gpu80"`. The slow 40 GB cards are excluded, so the rule "limits
   at twice the usual time", which covers slow cards, is replaced by limits 1.12 to 1.2 times the 80 GB time. Some reused stage 1
   networks ran on 40 GB cards (3BFF E 4200852863, Maze E seeds 1 and 2, Maze D). A card changes rounding, not the method.
3. **Time limits**, whose sum is the most this test can use:

| Step | Jobs | Limit | GPU-hours at most |
|---|---|---|---|
| 3BFF stage 1 (D, P, EP) | 9 | 6 h 15 min | 56.25 |
| CDDM stage 1 (D, P, EP) | 9 | 4 h | 36 |
| Maze stage 1 (P, EP) | 6 | 8 h | 48 |
| 3BFF stage 2, 100,000 iterations | 12 | 8 h | 96 |
| CDDM stage 2, 100,000 iterations | 12 | 5 h 5 min | 61 |
| Maze stage 2, 2,000 updates | 12 | 8 h | 96 |
| Maze read (replayed probe) and its dry run | 2 | 1 h | 2 |
| Pilots | 8 | | about 1.5 |
| Total | | | 396.75 |

4. **A job that reaches its limit** is continued from its checkpoint (as the rules say) only with the GPU-hours left under
   the cap, when the connection is up. Until then its network is unfinished.
5. **The pilot's stop rule runs as a job** (`entropy_prune_combo_pilot_check.py`): if an EP pilot did not finish, ended with
   a non-finite score or return, or skipped more than 5% of its updates, everything after it is cancelled. The rules
   name out of memory as a stop; an out-of-memory pilot did not finish.

**The chain** (`slurm/entropy_prune_combo_submit_della.sh`; a step whose parent fails is cancelled). Training runs from
`~/trainRNNbrain_epc` at 0558a88, the pilots' commit; reads run from `~/trainRNNbrain_epc_ana`. Job IDs are in the
submission entry. Pilots → pilot check → stage 1 of 3BFF and CDDM → stage 1 of Maze → stage 2 of each network after its
own stage 1 (E: after the pilot check; Maze D: after its stage 1 in job 18374320) → `read` (CPU) → `maze-read` (GPU)
→ `verdict` and the independent recount, written to
`/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/entropy_prune_combo_reads/`.

**The verdict job reads what is on disk.** Networks that are not finished are marked so, and the rules for unfinished
networks apply. The reading of the result waits for the gate, the recount and a person.

**Unchanged:** the cells, seeds, read points, criteria, reader and recount. The late window of 3BFF and CDDM is
75,000 to 175,000, and the end point is the stored vector at 174,900.
