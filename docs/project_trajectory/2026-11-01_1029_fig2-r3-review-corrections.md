# fig2-r3: corrections after the review of PR #94; Supplementary Table 1 Fig. column (#97, #100)

Corrects and extends `2026-11-01_0957_fig2-r3-bootstrap-dependence-and-markers.md`.

## Corrections

- **Fig. 2d wording.** The text said the inactive count "keeps rising" after the floor crossing in
  every network of Fig. 2d with a crossing. The reviewer found that false for DMTS s1167683015
  (N = 1,000): 782 inactive at its crossing (iteration 75,721), 886 at iteration 96,200, back to 782
  near 131,000, 813 at the end; least-squares slope after the crossing -8 ± 6 per 100,000
  iterations, against +43 to +76 in the other eight. What holds in all nine is the comparison of
  crossing and end (31 to 267 more inactive units at the end; panel-d printout). Results 2 now says
  that.
- **Fig. 2a,b point placement.** The 0957 entry put CDDM's and DMTS's network points at the same
  offsets right of the mean. At N = 1,000 DMTS s86371460 (422 active) then covered CDDM's first point
  (436). DMTS's row now starts where CDDM's ends (`network_offsets(..., n_before)`); first point
  0.04 and step 0.025 in log10 N (0.6 mm in panel a), so the longest row (CDDM 3 + DMTS 3) ends
  0.165 from its mean and the single DMTS point at N = 4,000 stays inside the axes. Checked on a
  300 dpi render: every network the caption counts shows at every size. Size-mean error bars in
  panels a and b now have no caps.
- **Leave-one-out maximum elsewhere** is 0.649 (DMTS at 1.10), reported as "0.65" in the PR body;
  the reviewer's exact enumeration gives 0.648.
- Results 2 restores "From N = 500 to 4,000" (the fit range) in place of "Above 100 units", and gives
  the borderline network's late-loss ratio as 0.983 against the 0.98 bound, as Methods states it.

## Supplementary Table 1, Fig. column

- **ED1 (#97):** the 35 size-series networks that Extended Data Fig. 1 tests, listed in
  `fig_paper_F2_size.ED1_NETWORKS` from the set fig1-r3 established (entry
  `2026-11-01_1003_fig1-r3-ablation-set-markers-and-figure-1d.md` on wp/fig1-r3): every size-series
  network at N = 500-4,000 except 3BFF s4186074845 and DMTS s632794669 and s2627187151, plus DMTS
  s2946835052. `write_table` asserts that exactly 35 rows carry ED1.
- **4a--c (#100):** the three CDDM and three 3BFF Fig. 1d networks, whose weights Fig. 4a-c now
  uses (fig4-r3); the three 3BFF rows also carry 4e,f (old networks of the transfer test).
