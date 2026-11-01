# Figure 2: revision after the second review of round 1 (dead-relu-paper PR #23)

Follows entry 2026-10-31_2010. No number of entry 2026-10-31_1957 changes and no new Della read was
made. The drawing stage of `fig_paper_F2_size.py` was re-run on the Mac on
`data/fig_paper_F2_size_collect.pkl` (Della job 16558986); only the rows of Extended Data Table 2
changed (`img/internal_figures/edt2_runs_rows.tex`). The redrawn figure files differed from the
committed ones only in matplotlib's random clip-path ids, so they were not re-committed.

## Extended Data Table 2 now marks the networks of Fig. 2e

Fig. 2e draws 19 3BFF networks (N = 500: 6, 1000: 7, 2000: 3, 4000: 3), each already a row of the
table, but no row said "2e". `tex_row` now appends "2e" to each of them. Two of the Fig. 1d 3BFF
networks are drawn in Fig. 2e as continuations to 300,000 iterations
(`NBitFlipFlop_c1_snapshots_c300000`), whose folders carry new seeds. Their parents were read from
each continuation's `paths.init_from` on Della (31 Oct):

| continuation seed (300k) | parent seed (200k, the table row) |
|---|---|
| 3204992658 | 4113307577 |
| 3194339211 | 2461724787 |

The mapping is the constant `C3_CONTINUED` in the script. `write_table` asserts that every
panel-(e) network maps to exactly one 3BFF row of the table; the assertion passes with 19 of 19. The
two parent rows now give their training length as "200k; 300k", and the table caption says that
their other columns describe the network at 200,000 iterations.

## Wording

- Results 2 no longer says flat networks lost units "at every size". It names the three sizes where
  the test applies (N = 500, 4 of 5; 1,000, 5 of 6; 4,000, 3 of 3) and says N = 2,000 had one flat
  network, too few for the test, as entries 1900 and 1957 set.
- Methods (read-out): the fit start now includes the escape clause of `noisy_floor`
  (`max(FIT_START, escape + 200)`), and the no-read-out list adds a loss that never halves from its
  first-50-iteration median and a DMTS late level at least half the starting level. The R² estimate
  check of entry 2026-10-13_1506 is now stated as it was recorded: eight of nine networks within
  0.006, the ninth (CDDM) within 0.012, with t* from an earlier read-out version.

## Correction to entry 2026-10-31_1957

Entry 1957 says that counting the 3BFF N = 1000 straggler (seed 3374117298) moves the 3BFF exponent
"by less than 0.01". Refitting with and without it on the same record gives 0.47755 and 0.47746, a
change of 0.0001. The PR's figure, "less than 0.001", is the more exact one.
