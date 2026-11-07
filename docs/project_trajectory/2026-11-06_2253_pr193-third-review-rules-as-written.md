## 2026-11-06 22:53 — PR #193 third review: which parts of the two Maze rules were written beforehand; what the tolerance can decide under 0.898

Third review of dead-relu-paper PR #193 (comment 6703554953). No network was trained. Corrects
`2026-11-06_2238_maze-technique-criterion-as-written.md` on one point: its 268 for s2560134940 is now
275 in the paper (the median over the 36 finite probes of the last 40), because a non-finite probe has
no number of active units.

### "As written before those networks were read" is true only in part (checked against the records)

The review proposed one sentence saying that each rule, as it stands now, was written before its
networks were read. The records do not support that for either rule.

Technique criterion, all three controls:
- Written, with all three controls, on issue #5 at 05:48 UTC on 5 Nov, before any run: true for the
  first ladder, 18 networks (rate penalty 0.03, 0.1, 0.3; rate + sparsity; noisy synapses; read-out
  dropout).
- Written with the two LEARNED controls, before those networks were read: the two gentler rescaling
  arms (issue #5 comment, 16:33 UTC: "more active units than every learned control ... the learned
  controls' mean (0.984)"), prune-and-duplicate (entry `2026-11-05_0907`, lines 59-60) and
  re-initialisation (entries `2026-11-05_2306` and `_2310`). These all follow entry `2026-11-05_0812`.
  The reference these arms now use is therefore the earlier text, not the text written just before
  they ran. No verdict depends on it: all of them get the same verdict under both references.

Setting rule, learned networks against learned controls:
- The learned controls were written before the setting networks were read (entry `2026-11-05_2310`).
- The rule written there is "all three seeds lie outside the controls' range". The restriction to
  networks that learned was decided after the read-out (entry `2026-11-06_2009`, PR #188 review, item
  1) and it changed verdicts: the Maze SGD arms lost a direction, and the metabolic penalty 0.01 went
  from "no change" to a decrease. M08 already says the scope was fixed after a review of the first
  read-out.

M09 states this. Decision #194's description of the settings rule ("stated with those controls before
the arms were read ... and is symmetric") holds for the controls, and the symmetry is from 6 Nov.

### The tolerance on Maze

With the control mean at 0.898 (345 of 384 episodes), a value whose networks all end at or above 0.9
has a mean at or above 0.9, inside 0.05 of 0.898 by construction. The four values that fail the mean
rule (rate + sparsity, read-out dropout, slow tilt, 5% grow) each have a network below 0.9. So on Maze
the 0.9 condition decides every verdict the success rate decides. Issue #5's clause says "an arm that
does not learn"; the 15:54 UTC read-out read it per network for the rate penalty at 0.1, and M09 says
that reading decides noisy synapses and the rate penalty at 0.1.

### What "above the control mean" means

The third control lowers the mean. The two default networks that learned have 0.992 and 0.977;
the rate penalty at 0.03 and 0.3 has means of 0.984 and 0.987, prune-and-duplicate 0.977. Results 5,
the Discussion and the Fig. 5 caption say a difference above zero is not an improvement.

### Figure

Fig. 5b: where crosses sit beside the marker, the number goes below the bar (`draw_group`,
`label_below`), off the tolerance line. The printout is unchanged. Checks: test_paperstyle ok,
scan_open_markers PASS, check_overlaps clean. Provenance rows of fig_paper_F2_size and fig_supp_tasks:
date corrected to their commits' 6 Nov.
