## 2026-11-02 20:48 — Active units before and after training, seven default 3BFF networks of N = 1,000

Work package r11-results1 (dead-relu-paper #127, Pavel's item 40). Results 1 quoted one network
(945 active units before training, 183 after 200,000 iterations; Fig. 1a,b). Pavel asked for
mean ± s.d. over at least three networks.

**Networks (listed on Della, not assumed).** Every default 3-bit flip-flop (3BFF) network of
N = 1,000 units in `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs` that was trained
for 200,000 iterations and whose chain starts from a seeded initialisation:

| sweep | network (final run seed) | chain (iterations per run) | root seed |
|---|---|---|---|
| `NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=1000_pen=none_arm=none` (Fig. 1c) | 3374117298 | 40k + 35k + 125k | 254798194 |
| same | 1914910909 | 40k + 35k + 125k | 254859059 |
| same | 3866906883 | 40k + 35k + 125k | 1558264694 |
| same | 4186074845 | 200k | 4186074845 |
| `NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control` (Fig. 1d) | 2461724787 | 75k + 125k | 1542248578 |
| same | 4113307577 | 75k + 125k | 2851654641 |
| same | 1258717025 | 75k + 125k | 2670297180 |

Network 3374117298 is the Fig. 1a,b example (`fig_paper_F1.EXAMPLE_NET`, first folder in sorted
order). The two `NBitFlipFlop_c1_snapshots_c300000` networks continue two of the Fig. 1d chains, so
they are not independent and are left out. The root configs differ from each other only in seed,
`max_iter`, checkpointing and unused revive/copy parameters (`diff` of the root configs on Della):
ReLU, equation `h`, bias 0, no Dale's law, Adam lr 1e-3·(100/N)^(1/3), weight decay 1e-6, input
and recurrent noise 0.05, fresh batch every iteration.

**Rule, written before the run.** Two measurements per network.

- (A) The paper's path (Methods, Participation): the participation trace logged during training,
  `*_ParticipationTrace.pkl` of the final run, which prepends its parents' samples. The probe is
  noise-free, on the training batch of that iteration (1,024 trials), and the per-unit vector is
  stored every 100 iterations. The probe at iteration 0 runs before the first update
  (`Trainer.train`, the probe precedes `train_step`), so it is the untrained network. Active units:
  `common.active_mask(p)` at iteration 0 and at the last stored iteration (199,900).
- (B) An independent simulator (numpy `RNN_numpy`, the Fig. 1a,b path of
  `fig_paper_F1.example_network_pair`): the untrained twin rebuilt from the root config with
  `gradient_participation.initial_params`, and the final weights (`LastParams`, iteration 200,000),
  both simulated noise-free on the same 24-trial batch (`np.random.seed(11)`).

Falsifiers, fixed before reading any count:
1. (A) for 3374117298 at 199,900 must equal 181, the `active_end` of the Fig. 1c collect record
   (`data/fig_paper_F1_collect.pkl`). Any other value means the trace is read wrongly.
2. (B) for 3374117298 must give 945 and 183, the Fig. 1a,b numbers. Any other value means the
   rebuild differs from the figure's.
3. Per network, before training: (A) and (B) are the same weights on different batches, so they
   must agree within 5% of the (A) count. A larger gap means the rebuilt twin is not the network
   that was trained.
4. Per network, after training: (A) at 199,900 and (B) at 200,000 differ by 100 updates and the
   batch, so they must agree within 10% of the (A) count.

The paper reports (A), the measurement Methods describes. If falsifier 3 or 4 fails on any network,
the numbers do not go into the paper until the cause is found.

**Run.** On the Mac, code worktree `wp/r11-results1` (from main 5508d2a), on copies of the files
named above (each final run's config, `LastParams` and `ParticipationTrace.pkl`, and the root
run's config; the Della path of each is in the copy's `source.txt`):

```
python trainRNNbrain/experiments_and_analysis/active_before_after.py <copy root> --out active_before_after.json
```

**Anomaly found on the first run, and its cause.** The first run seeded the 24-trial batch with
`np.random.seed(11)`, as `fig_paper_F1.example_network_pair` does. Falsifier 2 then failed:
network 3374117298 gave 942 active units before training, not 945. Repeated runs of the same
weights gave 941, 945 and 949. The weights rebuild identically on every call (maximum difference
0); the batch does not. `TaskNBitFlipFlop` has drawn its trials from its own generator `self.rng`
since 34864e3 (21 Aug), seeded from `cfg.task.seed`, which every training config leaves null. So
`np.random.seed` does not fix the batch, and the Fig. 1a,b panels (drawn on 31 Oct from the cached
record of Della job 16556248) rest on one unseeded draw of 24 trials. The figure is internally
consistent (both panels use one cached batch) but cannot be re-drawn exactly. The script now seeds
the task (`cfg.task.seed = 11`); two runs then give identical counts. Over task seeds 1-10, network
3374117298 has 939-946 active units before training and 181-187 after, so the figure's 945 and 183
lie within one network's batch-to-batch spread.

**Values** (seeded batch, task seed 11; both runs identical):

| network | (A) trace, iteration 0 | (A) trace, iteration 199,900 | (B) untrained twin | (B) final weights, 200,000 |
|---|---|---|---|---|
| 1258717025 | 921 | 183 | 917 | 193 |
| 1914910909 | 929 | 170 | 928 | 171 |
| 2461724787 | 962 | 191 | 957 | 188 |
| 3374117298 (Fig. 1a,b) | 945 | 181 | 940 | 184 |
| 3866906883 | 918 | 178 | 916 | 176 |
| 4113307577 | 931 | 195 | 923 | 197 |
| 4186074845 | 946 | 191 | 948 | 189 |
| mean ± s.d. (n = 7) | 936.0 ± 15.7 | 184.1 ± 8.8 | 932.7 ± 15.9 | 185.4 ± 9.2 |
| range | 918-962 | 170-195 | 916-957 | 171-197 |

(s.d. with n − 1 in the denominator.)

**Verdict against the rule.**
1. (A) for 3374117298 at 199,900: 181, equal to the collect record. Passes.
2. (B) for 3374117298 cannot reproduce the figure's batch exactly (cause above). With the batch
   seeded: 940 and 184; across ten seeded batches 939-946 and 181-187, which contain 945 and 183.
   The figure's numbers are consistent with the rebuild; the exact-match test is not possible.
3. Before training, |A − B| is at most 8 units (4113307577: 931 against 923, 0.9%). Passes (bar 5%).
4. After training, |A − B| is at most 10 units (1258717025: 183 against 193, 5.5%). Passes (bar 10%).

**For the paper (Results 1).** Seven default 3BFF networks of N = 1,000: 936 ± 16 active units
before training and 184 ± 9 after 200,000 training iterations (mean ± s.d.; measurement A, the
last stored probe of the 200,000-iteration training is at iteration 199,900). Fig. 1a,b stays the
illustrated example (network 3374117298, 945 and 183 on its 24-trial batch).

**Outside this package.** `fig_paper_F1.example_network_pair` draws its batch unseeded (above);
re-running it with `refresh=True` would give counts a few units different from 945 and 183.
Reported on dead-relu-paper as a figure issue; not fixed here.
