# Rule, written before the run: the Maze flat-return test on the full training records

Pavel, 11 Nov: the abstract says the number of active units keeps decreasing after the loss has nearly
converged only for 3BFF and CDDM; the phenomenon has to be tested on the Maze too. The first Maze test
(G8, rule in `2026-11-07_0100_round18-rules.md`, result in `2026-11-07_0235_round18-results.md`) found no
network with a flat return, on records of 2,000 updates, and was labelled short-window. It has not been
repeated on the records of the stage protocol (4,000 updates for N <= 2,000, 8,000 for N = 4,000). This
entry fixes the repeat before any training return of those longer records has been read. What has been
seen: the probe success rates and the numbers of active units of the three N = 1,000 networks (Fig. 2e)
and the learned-point and end-of-training counts of all 24 networks (Extended Data Fig. 2b,c).

## Networks

Every default ReLU network of the Maze size series that learned the task
(`fig_paper_F2_size.maze_series()`, `learned` true; chains joined by `common.maze_load_runs`), N = 100 to
4,000. 24 are expected.

## Quantities (G8's, with the window as the only new choice)

- Return: training discounted return per update. R = rise up to the learned point, as in G8 (median of
  the 100 updates ending at the learned update minus median of updates 1-100).
- Window, reading B (primary): the last quarter of the network's record, from 0.75 x its last update to
  its last update. The 3BFF test of Fig. 2f starts at iteration 150,000 of 200,000, the last quarter; G8's
  window started at the learned point, where the return is still rising by construction. A network is
  tested only if its learned point lies at or before the window start.
- Window, reading A (G8 unchanged): from the learned point to the last update, at least 300 updates.
- D = (median return over the last 100 updates of the window - median over its first 100 updates), per
  1,000 updates between the block centres. Flat if |D| < 0.03 R per 1,000 updates.
- Active units lost per 1,000 updates = minus the least-squares slope of the noise-free probe's number of
  active units over every probe in the window. Decreasing if at least 5 (the same bar at every N, as in
  the 3BFF test; it is 5% of a 100-unit network and 0.1% of a 4,000-unit one).

## Verdict rule (G8's, with a minimum number of flat networks added now)

Over the networks with a flat return under reading B:
- "decreasing while the return is flat" if at least two thirds decrease;
- "not decreasing" if at least two thirds do not;
- otherwise mixed.
- Fewer than 6 flat networks (a quarter of 24): no verdict, "cannot be tested".

Reading A and three sensitivity lines are reported beside it and cannot change the verdict: the last
half of the record as window; flat bars of 2% and 5% of R.

## What each outcome does to the paper

- Decreasing: the abstract, Introduction, Results 2 and Discussion sentence covers all three tasks, with
  the counts.
- Not decreasing: the paper states the difference: in Maze networks the number of active units stops
  falling once the return is flat.
- Mixed or cannot be tested: the sentence keeps its two tasks, and Methods reports the repeat.

## What would prove the claim wrong for the Maze

At least 6 networks with a flat late return of which fewer than two thirds lose 5 or more active units
per 1,000 updates. This outcome is credible: in Fig. 2e the Maze counts fall slowly late in training
(roughly 30 to 60 units over the last 2,000 updates), close to the bar.
