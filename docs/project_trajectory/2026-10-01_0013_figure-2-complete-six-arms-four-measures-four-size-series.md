## 2026-10-01 — Figure 2 complete: six arms, four measures, four size series

![Figure 2](../img/internal_figures/fig_paper_F2.svg)

225 networks across 71 cells, discovered from their saved configs rather than listed, every one
rebuilt and rescored in one test condition: sigma_w = 0 with the recurrent and input noise every arm
shares, averaged over eight draws. Each arm is drawn at the operating point its size series fixed.

| arm | n | active | r2 | dimensions | PCs to 95% | sd ln\|W\| | vs control |
|---|---|---|---|---|---|---|---|
| no intervention | 6 | 298 ± 35 | 0.9458 | 5.39 | 12 | 1.53 | 1.00x |
| dropout: mute | 3 | 510 ± 42 | 0.9285 | 7.10 | 25 | 1.59 | 1.03x |
| prune + duplicate | 3 | 771 ± 25 | 0.9416 | 7.04 | 24 | 1.43 | 0.93x |
| rescale | 4 | 453 ± 56 | 0.9313 | 7.26 | 18 | **3.48** | **2.27x** |
| synaptic noise | 3 | 568 ± 14 | 0.9277 | 6.69 | 17 | 1.40 | 0.91x |
| **frm + rws** | 3 | **939 ± 19** | 0.9214 | **11.95** | **59** | 3.13 | 2.04x |

**The penalty pair is in a different class.** It wakes 939 of 1000 units (+640, Welch p = 6e-09) and
the population spans 11.95 directions against the control's 5.39 (+6.56, p = 5e-05), with 59
components needed for 95% of the variance against 12. No network-side intervention comes close: the
best of them, duplication, reaches 771 units and 7.04 dimensions. It is also the most expensive,
2.58% of r2, and it is the only arm whose cost is clearly separated from every other arm's.

**Two arms widen the weight distribution, and they fail differently.** Dropout, duplication and
synaptic noise keep the control's width to within 7% and its distance from lognormal to within 1.8x.
Rescale is 2.27x wider AND misshapen - skew -2.31 against -0.42, excess kurtosis 6.41 against 1.00,
a heavy tail of very small weights. The penalty pair is 2.04x wider but its shape stays close to
lognormal: excess kurtosis 0.11, NEARER zero than the control's own 1.00. Width and shape are
separate failures and only rescale commits both. That distinction did not exist while the summary
was a fold-range.

**The size series, four arms against each size's own control:**

| arm | N=500 | N=1000 | N=2000 | N=4000 |
|---|---|---|---|---|
| dropout | 1.81x | 1.71x | 1.55x | 1.79x |
| duplication | 2.01x | 2.59x | 2.55x | **2.96x** |
| rescale | 1.50x | 1.52x | 1.46x | — |
| synaptic noise | 1.74x | 1.90x | **1.39x** | — |

Duplication is the only arm whose advantage grows with size. Dropout and rescale hold a flat
multiplier. Synaptic noise FALLS at N = 2000, from 1.90x to 1.39x, which is the pre-registered
failure signal - an intervention whose ratio heads toward 1.0 as N grows does not address a problem
that worsens with size. One size is not a trend and N = 4000 is still training, but it is the number
to watch.

**Rescale spikes at scale but recovers; an earlier note here said otherwise and was wrong.** One
seed of four at N = 2000 and one of three at N = 4000 were seen at a training loss of 10^5 and 10^6,
sitting there across the four log lines visible at the time, and were written up as divergences the
snapshot restore had failed to catch. Both recovered. The N = 2000 cell finished with four healthy
seeds at r2 0.9283, 0.9284, 0.9326 and 0.9335, and the N = 4000 runs are at r2 0.92-0.93 with eight
hours left. A four-line window is not a trajectory, and the restore mechanism did its job. Rescale
has lost no seeds.

**Still open.** Rescale and synaptic noise at N = 4000 are training. The DMTS controls, which the
paper grid never ran, are retraining after four of nine hit an 8-hour wall I set from the wrong
reference class - the duplication and mute arms converge in under 3 hours, but the control at a 7
tau delay is precisely the one that struggles. At N = 1000 all three controls reached r2 >= 0.9987;
at N = 2000 none has escaped the 0.42 plateau yet.

