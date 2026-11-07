## 2026-11-06 23:17 — PR #193 fourth review: corrections to `2026-11-06_2259`, the tolerance range's uncertainty, the flat-loss and fit-range thresholds, the combination test's unrun stage

Fourth review of dead-relu-paper PR #193 (comment 6703794642). No network was trained.
**Corrects `2026-11-06_2259_when-the-rules-were-written.md`:** the earliest text of the R^2 tolerance is
entry `2026-11-05_0903` (commit 8adef16), not `2026-11-05_0926`. It reads "held-out R^2 is within 0.01 of
the controls' mean ... the per-unit-rule criterion of docs/interventions_matrix.md". The printed claim
(after every 3BFF network and most CDDM networks had been trained) does not change. That entry's "25
values" holds for values all of whose networks finished; 26 counts input-weight norm 20 on Maze, with
one finished network.
**Corrects `2026-11-06_2304_tolerance-sensitivity.md`:** I wrote that the 0.0213 limit "rests on 9 noise
draws", implying the draws are the weak point. They are not; see below.

### The range's uncertainty (recomputed)

Rate penalty on 3BFF: difference -0.02134, 0.00134 beyond 0.02; standard error over 3 networks and 3
controls 0.00151, so the margin is smaller than the error. 9 and 36 draws on the same weights, 19
networks of a3_N1000 matched to `rm_read_runs` by seed: largest difference 0.00035, s.d. 0.00017, mean
-0.00001. 0.02 is the R^2 limit in `docs/notes/agent-bplan.md` (8de89d7, 13 Oct 23:43 ET; B2), which
reads at the floor crossing. So the abstract no longer says "only"; Results 5 does, with the range.

### Flat-loss threshold

`docs/notes/agent-c3.md` (b692f80, 15 Oct): flat when "changes by less than 3% per 100k ... (agent-c4b's
three claim-1 runs read -2.6%, +2.8%, +0.2%)". Those three runs, the N = 1,000 claim-1 lineage, are among the 19.
Result against the threshold, from the F2 printout's 19 rows:

| threshold | flat | decreasing |
|---|---|---|
| 2% | 11 | 10 |
| 2.5% | 13 | 12 |
| 3% | 15 | 13 |
| 3.5% | 16 | 13 |
| 4% | 17 | 14 |
| 5% | 18 | 15 |

The claim does not depend on the 3%. M12 says so. The Fig. 2 caption also said the thresholds were set
before measurement (not listed by the review) and is corrected.

### M12's definition and count

The definition stands. The flat-loss test stays out of it, now for two reasons. The count of three did
not hold, for a reason unrelated to the flat-loss test: the weight-decay comparison (entry
`2026-11-02_2218`: supported if b > 0.9 and b < 0.1; falsified if b < 0.5, written before any slope was
computed) has separate pass and fail thresholds, so it also meets the definition. M12 now says four,
three of them in Extended Data Table 2. The DMTS rule and the connectivity rule have one threshold each,
with fail the complement of pass, so they do not meet it.

### Fit range

Entry `2026-08-18_1205`: "Only N >= 500 (all below 78% active) is fitted. This is why the earlier
three-size figure looked like it was bending." M12 says the range followed that figure, and why.

### Combination test

`agent-bplan.md`, B3: "Supports ... at N = 1000; and at N = 4000 the all-three cell beats every single
remedy". Stage 2 was not run. The outcome stays inconclusive; the Note and Extended Data Table 2 say so.

### SM:195 (not checked by the review)

"The thresholds (a slope above 0.9 without weight decay and below 0.1 with it) were written in the lab
notebook before the slopes were computed." Holds: entry `2026-11-02_2218` lines 158-160, and `_2223` ("no
#172 slope has been computed on any of the 48 networks"). Its numbers match `_2232` and `_2315`: 0.979-0.986;
1.90-2.14; 0.029-0.036; at most 0.008; 3-15%; 0.42-0.57; 3.5-4.2; 0.13-0.16; 0.24-0.31; 43-51%.
