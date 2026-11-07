## 2026-11-06 22:59 — When each read-out rule was written: what the paper may claim, and the settings headline as a choice made after the data

Coordinator's decisions on dead-relu-paper PR #193, recorded on #194. No network was trained.
Follows `2026-11-06_2253_pr193-third-review-rules-as-written.md`. Corrects entries that call the rules
"the pre-registered ones of `docs/interventions_matrix.md`" (`2026-11-06_0934`, and the same phrase in
the docstrings of the figure scripts and comments of the paper's sections): that file contains no
criterion and no tolerance, at its first commit bb7053c or now (grep for tolerance, criterion, "within").

### The technique criterion, condition by condition

| part | first written | what had already happened |
|---|---|---|
| Maze: count above every control's in all seeds; success within 0.05 of the controls' mean; a network below 0.9 counts against the value | issue #5, 5 Nov 05:48 UTC | no Maze technique network had been trained |
| R^2 tolerance 0.01 (3BFF, CDDM): the earliest text I found | entry `2026-11-05_0926`, which attributes it to the matrix file, which does not hold it. Issue #5 at 05:06 UTC calls it "the paper's criterion" without the number. Searched: "tolerance" and "within 0.01" in the notebook to 4 Nov, and the paper's Results and Methods | all 3BFF technique networks trained (Sep-Oct) and read at the floor crossing (31 Oct); CDDM noisy synapses, dropout, duplicate (30 Sep) and rate 0.2 trained. CDDM rate 0.1, rate 0.1 + sparsity and rescaling were trained after it |
| a technique works on a task if it works at one value | entry `2026-11-06_0934` | every Maze rate-penalty network read; it decides the rate penalty on Maze, which fails at 0.1 |
| 75,000 iterations as the 3BFF verdict's read | decision #185, 6 Nov | floor reads seen: dropout and rescaling "work" on 1 of 3 and 1 of 4 networks; they fail at 75,000 |

So the wording "before any of these networks was trained" in the decision comment holds for Maze only.
Results 5, the Introduction and M09 say what holds.

### The settings rule

First written on 5 Nov (entries `2026-11-05_0903`, `_2259`, `_2310`), before the Maze and the newer
3BFF sweeps were read, and after the CDDM sweeps, the 3BFF input-weight norm sweep (F2 cache, 31 Oct) and
the 3BFF optimizer comparison (`b5_read.py`) had been read. M08's former "before any of its networks was
read" was false for those.

Tallies recounted from the Fig. 3 cells (`F3.direction` with and without `learned`):

| count | values with a direction on all three tasks | same direction on all three |
|---|---|---|
| every finished network | 25 with every task finished (26 with the input-norm 20 cell, 1 of 3 finished on Maze) | 4: weight decay 1e-5, 1e-4; SGD 0.03, 0.1 |
| only networks that learned | 18 | 2: weight decay 1e-5, 1e-4 |

Maze SGD, none learned: 521-556 at 0.03, 540-686 at 0.1, against 415-449. The two values increase the
number on 3BFF (731-759; 578-620) and CDDM (813-842; 845-859).

The abstract's and the Fig. 3 title's "only stronger weight decay ... on all three tasks" are true when
read as networks that learned, so they stay. Results 3 says that reading next to the claim, M08 gives
both tallies and says the restriction was chosen after the data.

### Figure 5

112.5 mm (axes a 16 mm, panel d still 29 mm), so the caption's glosses of the two rescaling variants fit
again. Panel a's y label is now one line, "fold change"; two lines overran the 16 mm axis and met the
letter b. Printout unchanged, test_paperstyle ok, scan_open_markers PASS, check_overlaps clean.
