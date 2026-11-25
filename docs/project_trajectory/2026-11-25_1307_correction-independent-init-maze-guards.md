## 2026-11-25 13:07 — Correction to the independent-init rules: guards in the Maze reference networks

Corrects `2026-11-25_1303_rules-independent-init.md`, before any of its networks started training. The rules say that
the N = 1,000 Maze networks of the first trainer version used neither guard (the rollback guard's restores, skipped
minibatches) in their first 2,000 updates. That is true of seeds 1-3 only: 0 restores and 0 skipped minibatches each.
Seeds 4, 5 and 6 had 6, 4 and 1 restores and 8, 7 and 3 skipped minibatches (sums over updates 1 to 2,000 of their
`Progress.json`). The new networks use seeds 1-3. No bound, read point or verdict rule changes: the reference range
(394-449 active units at update 2,000) contains all six networks, as stated.

The reader (`independent_init_read.py`) reads a suite network's learned point on its record up to update 4,000, as the
paper's run table does; on its first 2,000 updates alone seed 3 has no learned point, because its last probe (0.73) is below 0.9. A new
network's learned point is read on its 2,000 updates, so a new network that had not learned by its last probe has none.
