## 2026-11-09 02:13 — Round 19: the shifted sigmoid and its same-radius ReLU control at initial spectral radius 0.45, at 2,000 and 4,000 updates

**Rules.** `2026-11-08_0914_round19-full-sigmoid-rules.md` (seeds 3-5, the stage protocol, the same budget within
each group of three, read-out without a verdict threshold). Jobs: Della 17105877 (stage 1) and 17125519 (stage 2:
sigmoid s3-s5, ReLU s3-s4). Learned status by `common.maze_learned_index` on the joined records; active units
from the training probe, over running steps only.

| Network (radius 0.45) | At 2,000 | At 4,000: success, active | Learned at update (active there) |
|---|---|---|---|
| sigmoid s3 | not learned (0.64, 236) | 0.98, 227 | 3,330 (241) |
| sigmoid s4 | not learned (0.85, 247) | 1.00, 237 | 2,810 (237) |
| sigmoid s5 | learned (1.00, 259) | 1.00, 250 | 1,410 (293) |
| ReLU s3 | learned (0.97, 408) | 1.00, 344 | 1,310 (444) |
| ReLU s4 | not learned (0.85, 325) | 1.00, 273 | 2,320 (322) |
| ReLU s5 | stopped at update ~1,700 (0 active after 28 guard restores; saved weights non-finite) | stopped | none |

- **Every network that did not stop is learned at 4,000.** Both groups therefore end at 4,000, the stop stage
  of each.
- The pilot's two sigmoid networks at the same radius, carried forward on their outcome and so labelled
  "pilot", give the same picture: learned at 1,050 and 2,510, with 237 and 216 active at 4,000.

**End weights of the sigmoid networks** (`round19_maze_inspect.saturation`, 64 clean probe episodes; a unit is
pinned when its slope is under a tenth of its maximum on at least 90% of steps):
- full runs: no unit pinned at the top; 795, 792 and 761 of 1,000 pinned at the bottom, nearly off; 226-251
  active;
- pilot: 0 at the top; 784 and 802 at the bottom;
- the standard-radius sigmoid networks (radius 1.2) had 230-265 pinned at the top and never learned.

**The comparison, by measured numbers only.**
- *Sigmoid against ReLU at the same initial radius:* at 4,000 the sigmoid networks have 227-250 active units and
  the two ReLU networks 273 and 344. The sigmoid networks learned at 1,410-3,330, the ReLU networks at 1,310 and
  2,320. At the learned point the counts are 237-293 for the sigmoid and 322 and 444 for ReLU.
- *ReLU at radius 0.45 against the standard ReLU networks (radius 1.2, seeds 1-6):*

  | ReLU | Learned by 2,000 | Active at 4,000 | Active at the learned point |
  |---|---|---|---|
  | radius 1.2 | 5 of 6 | 314-404 | 406-517 |
  | radius 0.45 | 1 of 3 | 273 and 344 | 322 and 444 |

  ReLU at radius 0.5 (srad0.5) learned all three seeds by 2,000, with 338-414 active.

**What the ReLU collapse means for the comparison.**
- One of the three same-radius controls stopped, so the control has two networks.
- At radius 0.45, ReLU learned more slowly than at the standard radius. One of three was learned by 2,000,
  against 5 of 6 at radius 1.2, and one of three stopped.
- So the same-radius comparison varies the activation at a setting that is itself less stable for ReLU on
  this task. The two remaining ReLU networks are the only same-radius reference. Any difference between them and
  the sigmoid networks rests on n = 2 against n = 3 (5 with the pilot), and it is reported as such, with no
  verdict.
- No pre-registered threshold applies. The rules entry set none for this comparison.

**Reporting.** These networks are "the shifted sigmoid trained with initial spectral radius 0.45", a
per-activation change, never the standard protocol. The sigmoid units the setting removes from the upper flat
part (the ones with no ReLU counterpart) are absent at the end. The units left inactive sit at the bottom of the
sigmoid, the counterpart of inactive ReLU units.
