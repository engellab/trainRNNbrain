## 2026-11-09 18:58 — Round-21 figures package: Reviewer S figures items 1-7 (dead-relu-paper #217, #219)

Branch `wp/r21-figures` (code) from main 9659530; paper branch `wp/r21-figures` from paper main ad52b64. Every
figure drawn on the Mac from the existing records; no new training, no Della job.

**Fig. 1e drew each Maze network twice.** `fig_ed_population.load` already appends the learned Maze networks of
`data/population_stats_maze.json`, and `fig_paper_F1.load_population_stats_all` appended them again. Panel e now
draws each of the two learned Maze networks of panel d once (seeds 1 and 2 at update 2,000), and an assertion
checks that panel e's Maze seeds are panel d's. Printout: fraction selective over all units 0.154 and 0.200, over
active units 0.366 and 0.441 (the values of Methods: Population statistics and Supplementary Table 8). The
falsifying result, written before the redraw, was any Maze fraction other than these four values; none.

**Extended Data Fig. 5d,e had two networks per value; the data hold three.** The Della cache
(`data/fig_paper_F2_fails_cache.pkl`) was collected before the third CDDM seed of the spectral-radius and
magnitude-penalty sweeps was trained (5-6 Nov, job 16957558, entry 2026-11-05_2259). `fig_paper_F2_fails.py
--refresh-ladders` re-reads those two ladders from the local data into `data/fig_paper_F2_fails_cache_r21.pkl`
(held-out R^2 all in `data/heldout_r2_cache.json`, nothing simulated). The two older seeds read exactly as on Della
(crossings, 75,000-iteration counts and R^2 identical). Networks per value: 3, 3, 3 in both sweeps. Mean active
units at the floor crossing -> at 75,000 iterations: spectral radius 0.5 466 -> 363, 1.2 415 -> 332, 2.0 412 -> 310;
magnitude penalty 0 415 -> 332, 0.05 537 -> 445, 0.2 524 -> 444. Figure 3 already used three networks per value
(`read_uniform_cells_2.json`), so Fig. 3b prints no number there, and Methods M08 ("three networks per value and
task") is right. Figure 3's printout is identical with the old and the new cache, and Figure 5's printout is identical before and after this package's edits (both diffed; Figure 5 reads only the rate and sparsity ladder from this cache, which is unchanged).

**One code across figures (colour = task, shape = technique or setting).** `paperstyle.task_shade` and
`task_ramp` (shades 0.62, 0.85, 1, 1.35 of a task colour). Extended Data Fig. 5: every sweep in shades of its
task's colour (CDDM orange, 3BFF blue; was teal), panel f in Figure 5's shapes (rate penalty a circle, rate +
sparsity a left triangle), panel l with a key naming Adam (default), Adam without weight decay, AdamW, SGD at each
learning rate (five shades) and the downward triangle for an R^2 below the axis. Extended Data Fig. 8: tanh networks
in the CDDM colour. Extended Data Fig. 3: every task a circle (CDDM was a square, Maze a diamond, DMTS a
triangle). Figure 4c: the two measurement times by position (left +500, right +5,500, named above the first
low-active class of each task), all circles; the triangle now means only the young networks of d and e.

**Overprints.** Extended Data Fig. 2e: a stopped network on a "stopped" line below 0. Figure 5a,b: a cross
beside a mean marker sits 1.6 mm from it (was 1.3 mm), further crosses 1.2 mm apart; a slot holding crosses is as
wide as they need and the other slots are 3.1 mm (was 3.6); a number goes above its bar where the axes have room.
Figure 3a: a cross below the performance axis sits 1 mm inside its lower edge. Figure 4e: each threshold's value
is printed above it (386 at N = 1,000, 617 at N = 4,000), and the old N = 4,000 network nearest it is labelled 613;
the count criterion is applied to the old networks' mean (508; entries 2026-10-31_1532 and _1750). Extended Data
Fig. 6e: the lower row of group headers on two lines each.

**Other figure items.** Figure 5c rings the 3BFF rate-penalty means at N = 2,000 and 4,000, where every network had
non-finite loss values (Results 5; Extended Data Table 2). Figure 5d's axis labels name the quantity. Figure 2e's
bottom row plots active units (was inactive units); every network still ends with fewer active units than at its
floor crossing or learned point (3BFF -49 to -126, CDDM -150 to -162, Maze -91 to -113).

**Scans.** `tests/scan_marker_edges.py` over fig_paper_F1, F2_size, F3_tasks, F4_noreturn (draw), F5_tasks,
F2_fails, fig_ed_maze, fig_ed_population, fig_ed_maze_settings: PASS, 0 failing markers.
`tests/scan_open_markers.py` over the same scripts: 0 rule failures except two that predate this package and are
unchanged by it (Extended Data Fig. 7, two open markers; Extended Data Fig. 8b, one open marker with a centre of
178/255; the same scan of the main-branch scripts gives the same failures). The first scan of the new Figure 5c
failed (the ring was drawn as a Line2D open marker whose inside held the filled mean); the ring is now Figure 5's
scatter ring (`fig_paper_F5_drive.ring_size`) and the scan passes. Every script's text-overlap check reports none.
