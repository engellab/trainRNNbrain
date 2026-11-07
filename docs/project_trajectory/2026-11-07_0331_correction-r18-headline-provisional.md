# Correction to 2026-11-07_0235_round18-results.md: the 0.01 headline is provisional too (dead-relu-paper#205)

**What is corrected.** Under G2, "Provisional", the results entry `2026-11-07_0235_round18-results.md` says:
"Not provisional: the headline holds at 0.01 under both (a) and (b)." That sentence is wrong. The status
board (C6, round 18) repeated it and is corrected in the same commit.

**Why.** On present data the headline holds: at tolerance 0.01, prune-and-duplicate is the only technique that
meets the criterion on all three tasks, under the one-value rule (a) and under matched values (b). But the parity
runs (entry `2026-11-07_0136_round18-parity-rules.md`; Della 17031165-67, Spock 7397761-62) train on CDDM and Maze
exactly the 3BFF values that already pass at 0.01: noisy synapses σ_w 0.1 and 0.3, read-out dropout rate 0.05 with
β 4, and rescaling at target 4.0.
- Under (a), one of these values passing on CDDM and Maze makes its technique pass on all three tasks. For read-out
  dropout, which already passes on CDDM, Maze alone is enough. The headline and the upper end of the range
  0.0032-0.0161 then fall.
- Under (b), as Methods defines it ("the value of each technique that was run on all three tasks"), the parity
  values become matched values once they have run. The coordinator decided not to redefine (b) as a fixed list to
  protect the headline.

**Also disclosed (PR #202 review, item 3).** The matched-values rule (b) and the one-standard-error reading of
the rescaling and read-out-dropout passes were chosen after the one-value verdicts of G2 had been computed. The
rules entry (17b976e) names only the one-value rule; the matched values first appear in 0c2eeb9, the commit that
wrote the one-value outcome. The matched-values rule was the coordinator's decision on seeing G2.

**What follows.** The paper marks the headline and the one-value range as provisional (PR #202, each with
`% provisional: parity runs (#201)`). When the parity runs are read, the per-technique verdicts, the headline
and the tolerance range under both rules are recomputed together.

Fig. 5a,b now draws the 3BFF rate penalty at 0.01 and 0.03 beside 0.1 (G1 values; dead-relu-paper#204). The 0.01
value sets the upper end of the one-value range. The edge labels of Fig. 5d's CDDM sub-panel no longer sit on the
axis line or the arrow (PR #202 review, item 12). `fig_paper_F5_tasks.py`: scan_open_markers PASS,
test_paperstyle ok, no overlapping text.
