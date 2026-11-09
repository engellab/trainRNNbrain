## 2026-11-09 15:09: Correction of the "25 and 4" settings tally (entry 2026-11-06_2009), and records asked for by the review of PR #207

This entry corrects `2026-11-06_2009_settings-direction-from-learned-networks.md` and the status board line that
copied it (`docs/paper_audit.md`, line 223 at 90907b9). It also completes `2026-11-09_1442_r19-text-package.md`, as
the review of dead-relu-paper PR #207 asked (comment 6753878965, items 5 and 10).

### Where "25 and 4" came from

Entry 2009 gives the tally before the learned-only scoping as "25 and 4": 25 setting values had a direction on all
three tasks, and 4 had the same direction on all three. The 4 were weight decay 1e-5, weight decay 1e-4, SGD 0.03
and SGD 0.1.

That count used the two learned default Maze networks (426 and 449 active units) as the Maze control range, as
that entry says ("the maze reference already came from the learned controls"). The rule as first written counts
every finished network, so its Maze control range includes the third default network (415). With 415 in the range,
the weight decay 1e-5 network with 417 active units lies inside it, and the count is 25 and 3. The difference is a
choice of control set, not an arithmetic error.

The paper no longer prints either version. Against the five learned Maze controls (394-449; round 18, G12), the
tallies are 17 and 1 for learned networks and 24 and 3 for every finished network, from
`fig_paper_F3_tasks.py` at trainRNNbrain 8b7e5f3. The status board line is corrected to point here.

### The Maze check after the learned point, on the full set (review item 5)

Entry 1442 counted 38 of 45. The 45 are the G9 set (`data/round18/g9_maze_learned_vs_end.json`), with the six
withdrawn metabolic networks replaced by the three fz metabolic networks that learned by update 2,000. The G9 set
holds default networks 1 and 2 only, and none of the activation-function networks.

Eight further networks learned by update 2,000. Their counts, active units at the learned point and at update
2,000 (entry 2026-11-08_0915; `data/round19/round19_results.json`):
- default networks 4-6 (G12): 427 -> 394, 412 -> 398, 406 -> 417;
- leaky ReLU: 631 -> 603, 639 -> 600, 604 -> 610;
- softplus: 452 -> 437, 510 -> 420.

Six of the eight have fewer active units at update 2,000. Over the full set:
- 44 of 53 Maze networks have fewer active units at update 2,000 than at the learned point;
- with the 127 supervised networks, 170 of 180;
- the changes still run from -206 to +50 units, and the nine increases are 4 to 50 units.

The independent reviewer of PR #207 reached the same count with separate code.

### The Maze removal of the added units on the fz networks (review item 10a)

Command, once per network, run on the Mac CPU in the worktree `paper_push/wt/r19-text-code`:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/rl_maze_ablate.py \
        data/trained_RNNs_rl_maze_fz/Maze_interventions/<arm>/s<k> --added 430 --out data/round19/fz_maze_added

- **Arms:** `duplicate`, `frm0.03`, `frm0.1` and `frm0.3`, with k = 1, 2, 3. Each stage-1 folder holds the update-2,000 weights.
- **Code:** `rl_maze_ablate.py` at its last change, ae93e0a (8 Nov), unedited. The worktree was at 9b4afd5 when the
  runs were made.
- **Run:** the 12 runs went in parallel, about 23 minutes each.
- **Outputs:** the per-network JSONs and logs are in `data/round19/fz_maze_added/`, and the summary
  (`round19_fz_removal.py`) is in `data/round19/fz_maze_added_summary.json`.
- **Reading rule:** G4's rule from `2026-11-07_0100_round18-rules.md`, applied unchanged. An arm depends on its added units
  if the drop is at least 0.10 in every network; it does not depend on them if every drop is below 0.05; anything
  else is mixed. That rule was written before any of these runs.
