# fig2-r3: the DMTS exponent interval at 1.20 depends on three single networks; Figure 2 markers

Package fig2-r3 (dead-relu-paper #86, #81 Results 2, #83, rigor note of the round-2 verdict on #1).
Branch `wp/fig2-r3`. Read-out unchanged: every size-series network at its own floor crossing
(1.05, 1.10, 1.20 times its noisy-loss floor), collect record of Della job 16558986.

## The claim and the check

Results 2 said "every bootstrap 95% interval ends below 0.8" for the size exponent b at all three
margins. The round-2 rigor reviewer, with independent code, found that for DMTS at 1.20 the upper end
(0.784 in their run) rises to 0.86 without seed 1084904964 (N = 500), a network whose loss at the end
of training sits near the limit of the floor-crossing rule.

Check: `fig_paper_F2_size.leave_one_out` refits each (task, margin) with one network left out at a
time, at sizes with two or more networks, so every refit keeps all four sizes (N = 500 to 4,000).
Each bootstrap starts from the script's own seed (2,000 resamples). Printed by the drawing stage
under "Panel c, one network left out".

DMTS at 1.20 (9 networks; full fit b = 0.608, interval 0.491 to 0.786):

| left out | N | upper end | b |
|---|---|---|---|
| s1084904964 | 500 | 0.868 | 0.655 |
| s2106510642 | 500 | 0.868 | 0.637 |
| s86371460 | 1,000 | 0.860 | 0.689 |
| s2585017553 | 2,000 | 0.798 | 0.619 |
| s632794669 | 2,000 | 0.786 | 0.607 |
| s4124951173 | 2,000 | 0.774 | 0.595 |
| s1167683015 | 1,000 | 0.727 | 0.556 |
| s2099578504 | 500 | 0.643 | 0.540 |

DMTS at N = 4,000 has one network (s2627187151), so it cannot be left out without losing a size.

Every other (task, margin) fit: the largest upper end with one network left out is 0.649 (DMTS,
1.10). So the dependence is confined to DMTS at 1.20, and it involves three networks, not one.
Leaving out s86371460 also lifts that exponent to 0.69, above the "below two-thirds" of the text;
the text's statement is about the full fit (0.61).

s1084904964 sits near the rule's limit: `common.read_run` gives its late-loss ratio (median raw loss
over the last 10% of training over the floor) as 0.983; a network gets no floor crossing below
END_BELOW = 0.98. The reviewer's independent implementation gave 0.976, which would exclude it.

## What changed in the paper

- Results 2 states the dependence where the claim is made: the upper end, 0.79, rests on three of the
  nine networks; leaving out any one of them raises it to 0.86-0.87; one of the three ends training
  1.7% below its loss floor, near the 2% limit. (Not in the text: with any single network left out,
  every interval still ends below 1; the largest upper end is 0.868.)
- Figure 2a,b: one 2 pt point per network, full task colour, in a row beside its size's mean (3BFF
  left, CDDM and DMTS right, because 3BFF and DMTS counts overlap). Every marker filled: every value
  is at the floor crossing (#83 convention: filled = floor crossing, open = 75,000 iterations).
  N = 100 was drawn open before; it is now filled, and the fitted line starting at N = 500 marks it
  as not fitted. Error-bar caps narrowed to 1 pt under the mean marker, so a small s.d. no longer
  draws the round marker as a square.
- Figure 2d and Extended Data Fig. 3a: x tick labels at 100 and 10,000 only (ticks at every
  decade); smallest gap between two x tick labels in Fig. 2d is 11.3 mm at print size (committee
  acceptance: at least 3 mm).
- Extended Data Fig. 3b: network points placed as in Fig. 2a; 3c,d: default networks circles,
  networks with scaled input weights triangles, all filled.

Pass rule for the figure fixes, set before drawing: Fig. 2d no two x tick labels closer than 3 mm on
the PDF (measured: 11.3 mm); every network point visible on a 300 dpi render of Fig. 2a (checked by
eye against the caption's counts); `tests/test_paperstyle.py` passes on both PDFs.
