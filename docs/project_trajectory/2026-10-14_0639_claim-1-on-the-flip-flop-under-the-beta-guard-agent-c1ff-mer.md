### Claim 1 on the flip-flop under the beta guard (agent-c1ff, merged; notes `docs/notes/agent-c1ff.md`)

The two claim-1 flip-flop runs continued to 300k (FFtop300) STILL have no valid floor: beta stays
pinned at 3 (2461724787: floor 0.0408, loss ends 0.83x of it; 4113307577: 0.0365, 0.93x). The extra
100k did not lower the loss (smoothed minimum 0.0339 at 200k and 300k): both have sat at ~0.034 since
52-60k, the level of the third run. The networks have converged; the stretched-exponential fit is
what fails on them. Their 1.05x-smoothed-minimum crossings are 52k (341 active) and 60k (289); not
ablated, as committed in advance. Decision for Pavel: read such plateaued runs at 1.05x their
smoothed minimum, or keep one flip-flop network in claim 1.
The one valid floor, 1258717025 (floor 0.0338, beta 0.57, crossing 69k, 266 active): all silent
units removed 0.247-0.308 over 4 repeats (paper: 0.25-0.48); pruned + retrained 0.946 (paper
0.937-0.947); as many at random + retrained 0.921 (0.920-0.924); intact fine-tuned 0.951. The
retrains repeat within 0.001 of the earlier run. Claim 1 holds on this network; the paper's
flip-flop ranges (and 266-354 active) rest on the two rejected crossings and need rewriting.
**Unseeded held-out batch on the flip-flop:** its task seed is None, so `np.random.seed(BATCH_SEED)`
never reaches the task's generator, and the held-out and ordering batches differ on every run; the
"all silent removed" score spreads 0.274 ± 0.031 (SD, 4 runs, same weights). Proposed (not applied):
seed `task.rng` in `seeded_batches` (BATCH_SEED, +1); seeding `heldout_r2.heldout_batch` itself would
move every held-out R² in the paper - Pavel's call.

**CDDM revival screen at its floor (Figure 2 panel j, all runs landed).** `CDDM_screen3_c75000`, N = 1000,
3 seeds, mean active at the floor → at 75k: no intervention 436 → 335; input-weight cap 0.3
462 → 395; re-initialising silent units (the continual-backprop move) 419 → 338. Re-initialisation
does nothing at the floor or after it, as on the pre-standard 30k screen. Figure 2 redrawn
(`img/internal_figures/fig_paper_F2_fails.png`); panel k (flip-flop zero-out / random redraw)
waits for RVtop200 (200k, ~18 h from 03:00).

