# Result: the Maze flat-return test on the full records cannot be decided; the return never flattens

Rule: `2026-11-11_2307_maze-flat-return-late-window-rule.md` (commit 68fdf71, before the run).
Script: `trainRNNbrain/experiments_and_analysis/maze_flat_return_full.py`; output
`data/maze_flat_return_full.json`. 24 learned default networks, N = 100 to 4,000, records of 4,000 updates
(8,000 at N = 4,000).

## Verdict under the rule: cannot be tested

Reading B (primary, last quarter of each record): 24 of 24 tested, **0 with a flat return**. The return
still rises by 5.0% to 26.9% of the rise up to the learned point, per 1,000 updates, in 23 networks, and
falls by 3.2% in one (N = 1,000 seed 6); the flat bar is 3%. Fewer than 6 flat networks: no verdict.

| reading | tested | flat | decreasing among flat |
|---|---|---|---|
| B, last quarter (primary) | 24 | 0 | 0 |
| A, from the learned point (G8) | 24 | 0 | 0 |
| last half | 19 | 0 | 0 |
| B with a 2% flat bar | 24 | 0 | 0 |
| B with a 5% flat bar | 24 | 2 | 1 |

## What the numbers of active units do in the last quarter (no verdict attached)

13 of the 24 networks lose at least 5 active units per 1,000 updates in the last quarter of training; at
N = 1,000, 5 of 6 (19 to 49 per 1,000 updates; seed 4 gains 15). The four N = 100 networks lose fewer
than 5. So in the larger networks the number of active units is still falling late in training, while the
return is still rising slowly. The two cannot be separated on these records.

## Consequence

- The paper's sentence keeps its two tasks. Methods (Statistics and pre-registration) now reports this
  repeat in place of "We have not repeated the test on the longer records".
- A Maze test needs records in which the return is flat. On the present schedule the return of the
  N = 1,000 networks at update 4,000 is still rising by 5 to 27% of its early rise per 1,000 updates, so
  the networks would have to be trained several times longer, or with a decaying learning rate. That is
  new training and has not been started.
