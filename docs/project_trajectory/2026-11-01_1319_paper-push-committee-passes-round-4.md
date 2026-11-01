## 2026-11-01 13:19 — Paper push: the review committee passes the paper in round 4

Pavel asked on 31 Oct for the paper to be finished by a team of agents coordinating on GitHub
(ptolmachev/dead-relu-paper), stopping only when an independent committee agreed it matches the
writing and presentation standard of Tolmachev & Engel, *Nat. Mach. Intell.* 7:1742-1754 (2025).
The protocol is `~/Documents/Projects/paper_push/PROTOCOL.md`; the tracking issue is #1, where every
round's plan and every committee verdict is posted.

**Outcome.** Paper `main` at 969ccad (45 pages) passed all three committee lenses in round 4.
Scores, 8 = on par with the reference:

| Lens | Round 1 | Round 2 | Round 3 | Round 4 |
|---|---|---|---|---|
| Writing (6 dimensions) | 4-7 | 7-8 | 7-8 | all 8 |
| Presentation (6) | 5-7 | 7-9 | 7-9 | 8-9 |
| Rigor (6) | 6-9 | 7-9 | 8-9, two errors of fact | 9 on every dimension, no error of fact |

**What changed in the science along the way** (each recorded in its own entry; the board
`docs/paper_audit.md` is current):
- Silent CDDM units get exactly zero gradient only on the noise-free batch; under training noise they
  fire on ~1.6% of steps (entry 2026-11-01_0716). Figure 4 now uses the Fig. 1d snapshots; the silent
  range is 253-310 (entry 2026-11-01_1021).
- Held-out R² on DMTS is new noise on trials seen in training (Methods M05, #71).
- Population statistics (#34): including inactive units leaves dimensionality unchanged and changes
  statistics that weight units equally on at least two tasks; the normalised-rate effect depends on
  the normaliser (entries 2026-10-31_2212, _2220; 2026-11-01_0739, _0741).
- 116 distinct networks (not 124) keep losing units after their floor crossing (entry
  2026-11-01_1028, which also explains the round-2 rigor member's 122).
- Every per-unit-rule setting tried is disclosed (Supplementary Table 2a/2b, 58 + 64 rows), with how
  the shown setting was chosen; four of five were chosen post hoc (decision #90; entries
  2026-11-01_1007, _1046, _1100).
- The 3BFF reach bound is "fewer than a tenth" in every batch tested (21 batches across the reviews),
  with no maximum printed (entries 2026-11-01_1143, _1200, _1204, _1215).
- Title: "In task-trained ReLU recurrent networks, the share of active units falls with network size"
  (true at every size of Supplementary Table 1).

**Left for Pavel (cannot be filled by the team):** the `[TODO]` placeholders in the back matter —
data DOI and tagged code release (p. 20), acknowledgements (p. 25), author contributions (p. 26).
Optional polish the committee listed without blocking is in its round-4 comments on #1.
