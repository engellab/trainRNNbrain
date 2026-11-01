## 2026-10-31 19:00 — Figure 2 (size and training time): what is read and the pass rule, set before any output was seen

Work package fig2-size-time (ptolmachev/dead-relu-paper#6), trainRNNbrain branch `wp/fig2-size-time`.

**Data step.** The three DMTS 5τ N = 4000 controls existed only on Spock. They were copied Spock → Mac →
Della into `DMTS_paper_grid_d5/EqType=h_N=4000_arm=control` (config, last weights, participation
trace, losses; no optimizer state). No new training.

**Job.** `fig_paper_F2_size.py --collect`, Della CPU job 16558986 (16 CPUs, no GPU), from
`~/trainRNNbrain_fig2size` detached at 8bb2939. It reads every run of the three size series (N = 100,
500, 1000, 2000, 4000) with `common.read_run` at 1.05×, 1.10× and 1.20× its own noisy-loss floor and
scores held-out R² (36 draws). This entry was written while the job ran and before any of its output
existed (its log is block-buffered and was empty at 19:00).

**Rule for C2** ("a larger network keeps a smaller share of its units active"). Fit from N ≥ 500, on
the size means; bootstrap 95% interval by resampling runs within each size (2,000 resamples).
- C2 holds for a task if the exponent's upper bound is below 1 at all three margins.
- C2 fails for a task if the exponent's interval contains 1 at the 1.05× margin.
- A task with a size that has no read-out is not fitted at the floor and is reported as such.

**Rule for C3** ("units keep becoming inactive after the floor"), the test of entry 2026-10-15_0955
reused unchanged (`c3_flat_loss_units.py`): from 150k, a run is flat if its loss changes by less than
3% per 100k; units "keep becoming inactive" at a size if at least 2 of 3 flat runs lose ≥ 5 units per
100k, and "stop" if at least 2 of 3 flat runs lose fewer. N = 4000 (paper grid continued to 200k) is
added to the sizes the test reads.
