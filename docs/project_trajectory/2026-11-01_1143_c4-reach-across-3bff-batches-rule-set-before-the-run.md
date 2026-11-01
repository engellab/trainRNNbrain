## 2026-11-01 11:43 — 3BFF gradient reach across six trial batches: rule set before the run (dead-relu-paper #112)

Work package methods-r4 (round 4). Committee round 3 (#112) found that the 3BFF gradient-reach numbers
of Results 4 come from one seeded batch of 1,024 trials (task seed 12345) and move with the batch in
its own independent code (four batches: reached low-active units 108-222, share of them active
5,500 iterations later 0.9-6.0%). Methods (Gradient reach) must state the batch size, the number of
batches and the range. This entry fixes the measurement and the rule before any number is computed.

**Measurement.** `trainRNNbrain/experiments_and_analysis/c4_reach_batches.py` (new, this branch). For
each of the three 3BFF networks of Fig. 4a-c (seeds 1258717025, 2461724787, 4113307577; the Figure 1d
weights in `data/pop_stats_nets/`, later weights `data/fig4_noreturn/snap/<seed>_d0.npz` (+500) and
`_d5000.npz` (+5,500)) and each task seed in {12345, 101, 102, 103, 104, 105}: one noise-free batch of
1,024 trials (`c4_gradient_mechanism.batch`, the run's own task config); classes on that batch
(active: p >= 0.05 q95(p); silent: x <= 0 on every step of every trial; low-active: the rest);
g_in from `c4_gradient_mechanism.backward` with the strict gate; reached: g_in >= 0.05 q95(g_in);
participation of the later weights on the same batch. These are the definitions of
`fig_paper_F4_noreturn.load_c4` and `fig1d`, applied per batch. It shares code with the production
measurement, so it tests batch dependence, not the backward pass (which the committee checked with
its own autograd code).

**Rule, fixed now.**
1. Reproduction check: task seed 12345 must give the numbers of entry 2026-11-01_1021 exactly
   (low-active 731, 658, 713; reached 136, 155, 161; share active at +5,500 0.7%, 2.6%, 4.3%). If it
   does not, stop and find why before reading the other batches.
2. Report, per network, the range over the six batches of: low-active count, reached count, reached
   share of the low-active units, and share of the reached units active at +500 and +5,500.
3. The Results 4 bound "fewer than a tenth" (the pre-registered 10% bound of entry 2026-11-01_0948)
   holds if the share active at +5,500 is under 10% in every network and every batch. Any single
   value at or above 10% fails it, and results-r4 is told on #112.
4. Sanity bound on the code: the committee's four batches gave reached shares of 16-30% and shares
   active at +5,500 of 0.9-6.0%. A value far outside that (reached share outside 10-40%, or a share
   at +5,500 above 15%) is investigated before it is reported.
