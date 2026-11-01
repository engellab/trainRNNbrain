# Extended Data Fig. 3: the snapshot estimate's batch-noise range on CDDM, rule set before output

Work package fig2-r2 (ptolmachev/dead-relu-paper#53), trainRNNbrain branch `wp/fig2-r2`. Written
before the computation below was run.

**Why.** The round-2 caption limit (120 words) leaves no room for the sentence of the round-1 caption
that gave the snapshot estimate's own uncertainty on two CDDM default networks ("90% bootstrap ranges
of 15,000-25,000 and 17,000-26,000 iterations", entry `2026-10-15_2054`). That entry does not name
the two networks, and its script was not kept. The ranges are therefore recomputed for all three CDDM
default networks with kept weights (`CDDM_c1_snapshots`, control arm) and drawn as horizontal bars in
Extended Data Fig. 3c.

**Input.** The per-batch fresh-batch losses of every kept snapshot (32 batches per snapshot, every
1,000 iterations, 99 snapshots per network) from `/scratch/gpfs/TENGEL/pt1290/oracle_readout`
(Della job 15635966), copied to the Mac. The loss floor of each network is the one in the collect
record of Della job 16558986 (`data/fig_paper_F2_size_collect.pkl`, Fig. 1d rows); these three
networks have no loss bursts, so no snapshot is masked.

**Method.** `fig_ed_readout.py --batch-noise` re-implements the snapshot estimate of
`fig_paper_F2_size.oracle_read` on arrays: per snapshot, the mean of its batch losses after dropping
points more than 3 scaled MADs from their median; at each snapshot iteration t, the mean over the
snapshots in [t/1.1, 1.1t]; the estimate is the first t after which that curve stays at or under
1.05x the floor through 1.2t. Bootstrap: 1,000 resamples, each drawing every snapshot's 32 batch
losses with replacement; the 90% range is the 5th to 95th percentile of the resampled crossings.

**Rule, fixed now.**
1. The un-resampled estimate must equal the collect record's `t_oracle` for each of the three networks
   (17,000, 16,000 and 24,000 iterations). If any differs, the re-implementation is wrong and no bar is
   drawn.
2. If rule 1 holds, the three ranges are drawn as they come out, whatever they are, and replace the
   round-1 sentence. They are not required to match 15,000-25,000 and 17,000-26,000, which came from
   an unrecorded script and an earlier floor; any difference is reported in the result entry.
