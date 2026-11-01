## 2026-11-01 10:00 — Figure 3 and Extended Data Fig. 4, round 3: marker rule, readable panels, and the count of networks that keep losing units (116, not 124)

Work package fig3-r3 (dead-relu-paper #87, #88, #83, #81, and the rigor note on #1). trainRNNbrain
branch `wp/fig3-r3` (from main 77d6d81, with `wp/style-r3` 39e5a1e merged). No new runs: every number
below comes from `data/fig_paper_F2_fails_cache.pkl` (entry 2026-10-31_2010).

**Correction to entry `2026-11-01_0735_figure-3-round-2-redraw-and-results-3-numbers.md`.** That entry
says "Of the 140 networks of the nine sweeps and the two revival screens, 129 have a floor crossing
... All 124 that are not rate-penalty networks have fewer active units at the end". The 140, 129 and
124 count entries of the cache, not networks. Eight default networks appear in more than one sweep:
the 3 default networks of `CDDM_std_g0_drift_bias0` are the default value of the weight-decay,
metabolic-penalty and activation-function sweeps (3 entries each), and 2 default networks of
`CDDM_std_g0_recinit` are the default value of both the spectral-radius and the recurrent-weight
penalty sweeps. Counted once each (by run name, which holds the seed; 132 distinct seeds):

| | entries | distinct networks |
|---|---|---|
| all | 140 | 132 |
| with a floor crossing | 129 | 121 |
| with a floor crossing, rate penalty | 5 | 5 |
| with a floor crossing, no rate penalty | 124 | **116** |
| of those, fewer active units at the end of the budget than at the crossing | 124 | **116** |

So the claim holds for every network, and the number Results 3 prints becomes 116. The committee's
reconstruction (#1, round 2: "122 that I can see") matches neither count; its note did not say how it
counted, and 122 is not reproduced by any of the variants below.

How the count was made, before looking at the result (pass rule: an exact integer match between the
variants; any network that does not fall is listed). Script in the session scratchpad, reading only
the raw per-run fields of the cache (`it_act`, `active`, `t_star`, `name`), no function from the figure
scripts: for each run with a finite `t_star`, active units at the probe nearest `t_star` against
active units at the probe nearest the sweep's `end_at` (or the run's last probe). Variants: the probe
at or before `t_star`; the run's last probe (200,000 for the shared defaults read at 75,000 in the
metabolic sweep); the median over the last 2% of the budget. All three give 124 of 124 entries and
116 of 116 distinct networks. The smallest fall is 15 units (a sigmoid network, 195 at its crossing
at 96,049 iterations, 180 at 200,000). The rate-penalty runs are those whose name has `Lfrm` other
than 0. The floor crossings `t_star` themselves come from the pipeline; the committee's round-2 check
of t* (#1) is the independent test of those.

**Floor crossings across setting values (for #81 item 6).** On CDDM the mean floor crossing of a
setting value ranges from 4,500 iterations (initial input-weight scale 2) to 76,700 (sigmoid
activation), a 17-fold range over 33 values; across network sizes it changes by about a fifth (entry
behind Results 2). Results 3 states the contrast in one clause.

**Figures** (drawn on the Mac from the cache; `tests/test_paperstyle.py` passes on both; layout checks
0 problems, now including a 2 mm minimum gap between texts on one baseline):
- Marker rule of decision #83 in both figures: filled = floor crossing, open = 75,000 iterations (or
  200,000 in Extended Data Fig. 4l's right half), drawn with `paperstyle.mark_75k`. Fill means
  nothing else: Fig. 3a and 3d, measured only at 75,000 iterations, are all open; Fig. 3a no longer
  marks measured against estimated R² by fill (the caption names the measured rows); SGD with and
  without weight decay share one marker in Fig. 3d and Extended Data Fig. 4l.
- Fig. 3c: each input-weight scale's median over its 3 networks is a thick line, the networks thin
  and faint (#87 item 1).
- Fig. 3d: legend above the axes, outside the data; R² below 0.86 is an arrowhead inside the bottom
  edge with clearance (#87 item 2).
- Extended Data Fig. 4: R² sub-panels 0.62 of the trace panel's width (was 0.42), each spanning its
  own data with two ticks one 1-2-5 step apart; iteration ticks at 10,000 and 100,000 only (the
  50,000 and 200,000 labels touched); every legend on an opaque white box, so the 75,000-iteration
  and budget lines no longer cross legend text (#88, Extended Data Fig. 4 items).
