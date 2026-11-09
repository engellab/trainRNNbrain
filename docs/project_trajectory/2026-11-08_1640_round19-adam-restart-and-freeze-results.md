## 2026-11-08 16:40 — Round 19: the Adam-restart test and the rwm0.2 s1 freeze test, read by their rules

**Rules.** `2026-11-08_0925_round19-adam-restart-test-rules.md` and `2026-11-08_0953_round19-freeze-default-rules.md`.
In both, a network "restores" if it has at least one guard restore in updates 2,001-4,000, or stops. Jobs:
Spock 7402515 (`INIT_ADAM=true`, code 7889823), Spock 7402525 (replicates with `INIT_ADAM=false`, code 3be22a0)
and Spock 7402526 (rwm0.2 s1 with `freeze_after_end=true`, code 3be22a0). All are stage 2 from the same
stage-1 weights. Read from the Progress records and logs copied on 8 Nov.

**The replicates are exact copies of the original stage-2 runs.** For srec0.1 s3 and srad1.5 s3, every one of
the 2,000 updates has the same success and return as the original run. For rwm0.2 s1, all 1,000 updates up to
its stop match, and it stopped at update 3,001 again. With the same seed, the same cluster and the switch off,
training is deterministic. So a replicate measures no chance variation, and the "limitation" of the test rules
does not apply in that form. What remains is that `INIT_ADAM=true` changes the trajectory as a whole, and its
effect cannot be split from the divergence any small change would start.

| Network | Original (INIT_ADAM=false) | Replicate | INIT_ADAM=true |
|---|---|---|---|
| srec0.1 s3 | 21 restore events from update 2,698; completes | identical | no restore; completes |
| srad1.5 s3 | 1 event at 3,920; completes | identical | 1 event at 2,550; completes |
| rwm0.2 s1 | stopped at 3,001 | identical, stopped at 3,001 | 71 restore events from 2,744; completes to 4,000 |

**Reading by the rule: the restart is not the cause.**
- The replicates restore in 3 of 3 networks.
- The `INIT_ADAM=true` runs restore in 2 of 3 (srad1.5 s3, rwm0.2 s1). The rule's clause "the INIT_ADAM=true runs
  restore in at least 2 of 3" holds.
- The protocol keeps `INIT_ADAM=false`.
- The result is not open, so the 70 GPU-h extension is not run.

Restoring the optimiser state did change two outcomes. srec0.1 s3 had no restore, and rwm0.2 s1 completed
instead of stopping, though with 71 restore events. These are single paired runs and the rule does not count
them.

**Where the replicates' restores happened** (`restored_post_episode`, recorded from e8ae647 on):
- srec0.1 s3: 3 of its 42 discarded rollouts were non-finite only after an episode's end; the other 39 were
  non-finite within a running episode.
- srad1.5 s3: 4 of 4 within a running episode.
- rwm0.2 s1's stop is not classified in the record: the trainer raises on the sixth failure before it writes
  the classification. Its last finite weights showed post-episode growth (`2026-11-08_0943`).
- So in these networks the stage-2 restores are mostly instability inside running episodes. The new default
  (`freeze_after_end=true`) does not remove that kind.

**rwm0.2 s1 with freeze_after_end=true: passes.** It completes updates 2,001-4,000 without a stop; the original
and the replicate stopped at 3,001. It still has 4 restore events (7 discarded rollouts), all within running
episodes. Its end: probe success 0.98, 469 active. The switch removed the failure that stopped this network,
the post-episode growth, and left the rare running-episode restores.

**Consequences.**
- `INIT_ADAM=false` stays.
- The stage-2 excess of restores (24 of 63 against 12 of 63) is, in the classified cases, running-episode
  instability. It is reported as a property of longer training under these settings, not as an artefact of the
  restart.
- `freeze_after_end=true` (the default for new runs) removes post-episode stops such as rwm0.2 s1's.
