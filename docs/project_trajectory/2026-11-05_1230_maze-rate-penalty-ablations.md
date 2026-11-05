# Maze rate-penalty networks: the recruited units do work (removing them lowers success to 0.55-0.77)

**Date:** 2026-11-05 12:30
**Issue:** #5
**Commits:** 0361fd6 (the four JSONs), this entry's commit (the redrawn Extended Data figure)

Five rate-penalty networks at N = 1,000 ablated at their final weights (rl_maze_ablate.py on the
Mac: outgoing weights zeroed, success under the training noise over 256 fixed drops x 4 draws):
rate 0.03 seeds 1 and 2, rate 0.3 seeds 1 and 2 (this batch) and rate 0.1 seed 3 (the figure
agent's). The maze counterpart of the supervised A3 test: are the units a rule recruits doing work?
"Recruited" = the network's active count minus the learned controls' mean (437.5); they are
removed quietest first on top of the inactive units; the values are interpolated on the k grid.

| network | active | success intact (clean) | inactive removed | as many random | + recruited removed | as many random | active removed |
|---|---|---|---|---|---|---|---|
| rate 0.03 s1 | 919 | 0.94 (1.00) | 0.93 | 0.87 | 0.77 (482 units) | 0.35 | 0.05 |
| rate 0.03 s2 | 853 | 0.89 (0.98) | 0.90 | 0.81 | 0.74 (416) | 0.27 | 0.05 |
| rate 0.1 s3 | 982 | 0.94 (0.99) | 0.94 | 0.94 | 0.55 (544) | 0.07 | 0.03 |
| rate 0.3 s1 | 962 | 0.84 (0.95) | 0.83 | 0.85 | 0.68 (524) | 0.32 | 0.05 |
| rate 0.3 s2 | 930 | 0.96 (1.00) | 0.97 | 0.91 | 0.62 (492) | 0.29 | 0.06 |

- Removing the inactive units (18 to 147 of 1,000) leaves success within 0.01 of intact, as in the
  controls.
- Removing the recruited units as well (416 to 544 more, quietest first) lowers success from
  0.84-0.96 to 0.55-0.77: the recruited units take part in the computation, as on the supervised
  tasks (A3: R^2 0.93-0.95 to 0.18-0.62). Removing as many units at random lowers it further, to
  0.07-0.35, as for four of the five supervised rules.
- Removing the active units lowers success to 0.03-0.06.

The Extended Data interventions figure (fig_ed_maze_interventions.py) now draws the five
technique networks over control s1 in panel d; no overlapping text.
