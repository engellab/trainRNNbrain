## 2026-11-06 23:04 — How the technique verdicts depend on the tolerance; stale "pre-registered" labels corrected

Coordinator's follow-up on dead-relu-paper PR #193 and #194. Follows
`2026-11-06_2259_when-the-rules-were-written.md`. No network was trained and no number changed.
`fig_paper_F5_tasks.tolerance_sensitivity` prints this from the unrounded differences. The drawn
figure did not change (the PDF stays that of 6b58997).

### Verdict per technique and task (W works, - fails; a technique works on a task if it works at one value)

| technique | 3BFF at 0.005, 0.01, 0.02 | works from | CDDM at 0.005, 0.01, 0.02 | works from | Maze at 0.025, 0.05, 0.10 |
|---|---|---|---|---|---|
| prune-and-duplicate | W W W | 0.00460 | W W W | 0.00324 | W W W |
| rate penalty | - - - | 0.02134 | W W W (0.2 only at 0.005) | 0.00443 | W W W |
| read-out dropout | - - W | 0.01668 | - W W | 0.00757 | - - - |
| rate + sparsity | - - - | 0.02830 | - W W | 0.00759 | - - - |
| rescaling | - - W | 0.01267 | - - - (count) | never | - - - |
| noisy synapses | - - - | 0.02216 | - - W | 0.01504 | - - - |

Tasks passed at 0.005, 0.01, 0.02: prune-and-duplicate 3, 3, 3; rate penalty 2, 2, 2; read-out dropout
0, 1, 2; rate + sparsity 0, 1, 1; rescaling 0, 0, 1; noisy synapses 0, 0, 1.

**The headline survives at all three tolerances, with thin margins at both ends.** Prune-and-duplicate
is the only technique passing on all three tasks from 0.0046 (its 3BFF difference, 0.00460) to just
below 0.02134. Below 0.0046 no technique passes on all three. From 0.02134 the rate penalty passes too;
that value is 0.0013 above the 0.02 tested, and it rests on 9 noise draws where the others have 36.

The Maze success tolerance changes no verdict at 0.025 or 0.10, because the 0.9 condition decides them.

The abstract's headline sentence is unchanged: nothing flips at the tolerances tested, and the abstract
states no tolerance ("nearly unchanged task performance"). Its settings sentence now says "Among networks
that learned".

### Labels

The criterion across tasks is now described as chosen for this comparison. Docstrings and comments that
called it pre-registered or cited the matrix file are corrected (`fig_paper_F5_tasks`, `fig_paper_F3_tasks`,
`read_uniform_cells`, `read_uniform_cells_2`, `read_maze_arms`, `fig_ed_maze_settings`, the section
comments), and `docs/interventions_matrix.md` carries a note that it holds no criterion. Citations of
the matrix as an inventory of runs stay. Issue #5's Maze conditions are still called pre-registered where
they are named as such (`fig_ed_maze_interventions`), since they were written before those runs.
