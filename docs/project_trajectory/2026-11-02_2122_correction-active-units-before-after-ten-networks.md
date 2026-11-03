## 2026-11-02 21:22 — Correction to 2026-11-02_2048: ten default 3BFF networks, not seven

Corrects `2026-11-02_2048_active-units-before-and-after-training-3bff.md` (work package
r11-results1, dead-relu-paper #127, PR #166). That entry called its seven networks "every default
3BFF network of N = 1,000 that was trained for 200,000 iterations". That claim is wrong. The review
of PR #166 (comment 6596116300) found three more networks on Della.

**What the inventory missed.** Entry 2048 listed only the sweeps `ff_revive_c200000`,
`c1_snapshots_c200000`, `c1_snapshots_c300000`, `paper_grid_c200000` and `size_series_c200000`. It
did not list every `NBitFlipFlop*` sweep. The full listing of every N = 1,000 cell of every
`NBitFlipFlop*` sweep in `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs` was made
for this entry. It contains three default cells trained for 200,000 iterations:

- `NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=1000_pen=none_arm=none` (4 networks, Fig. 1c);
- `NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control` (3 networks, Fig. 1d);
- `NBitFlipFlop_std_dropfix_c200000/EqType=h_k=3_N=1000_pen=none_do=none` (3 networks, added here).

Every other 200,000-iteration cell at N = 1,000 switches on a training technique: copy, dropout,
revive, rescale, norm, recgrow, replace, synaptic noise, target ladder, duplication or remedy grid.
`NBitFlipFlopHyper_b4_transfer` is a different task, and `std_ksweep_bias0` and `std_winp_bias0`
ran for 150,000 iterations. The `_c300000` cells continue chains that are already counted.

**The three added networks have default settings.** Each is a chain of 40k + 35k + 125k = 200,000
iterations. The continuations warm-start with `init_adam: true`.

| final seed | chain seeds (40k, 35k, 125k) |
|---|---|
| 3665415385 | 4123842379, 2075057582, 3665415385 |
| 1847344754 | 3071103934, 670908712, 1847344754 |
| 2147366219 | 1052986492, 171187417, 2147366219 |

`diff` of every chain config against the root config of the Fig. 1c network 254798194 differs only
in experiment tag, seed, `max_iter`, `iter_offset`, checkpointing and warm-start paths. It also
shows keys that the older code did not have yet: `inp_weight_cap`, the dropout `rescale` and
`p_max`, `prune_reinit`/`prune_args` and `synaptic_scaling`/`scaling_args`. All of these are absent,
not switched on. `dropout: false` holds in every link.

**Run.** The method is the same as in entry 2048: the same script, on copies of the three final
runs and their root configs.

```
python trainRNNbrain/experiments_and_analysis/active_before_after.py <copy of the three> --out active_before_after_dropfix.json
```

| network | (A) trace, iteration 0 | (A) trace, iteration 199,900 | (B) untrained twin | (B) final weights, 200,000 |
|---|---|---|---|---|
| 1847344754 | 952 | 178 | 943 | 175 |
| 2147366219 | 940 | 228 | 937 | 230 |
| 3665415385 | 925 | 182 | 931 | 177 |

The (A) counts equal the reviewer's independent counts (925/182, 952/178, 940/228). Falsifiers 3
and 4 of entry 2048 pass on all three networks. Before training, A and B differ by at most 9 units
(0.9%), against a bar of 5%. After training, they differ by at most 5 units (2.7%), against a bar
of 10%. Network 2147366219 is outside the range of the other nine (170-195). The reviewer found it
stays at 228-231 from iteration 190,000 on, with the same clean loss as its two neighbours (0.13).
Neither the configs nor the traces give a reason to exclude it. The editor's decision on PR #166 is
to include all ten.

**Corrected values, ten networks** (measurement A; s.d. with n − 1):

| | before training (iteration 0) | after 200,000 iterations (last probe, 199,900) |
|---|---|---|
| mean ± s.d., n = 10 | 936.9 ± 14.4 | 187.7 ± 16.0 |
| range | 918-962 | 170-228 |

Measurement B over the same ten networks gives 934.0 ± 13.5 before training and 188.0 ± 17.0 after.

**For the paper (Results 1).** In ten default 3BFF networks of N = 1,000, 937 ± 14 units are active
before training and 188 ± 16 after 200,000 training iterations. These numbers replace the 936 ± 16
and 184 ± 9 (n = 7) of entry 2048.
