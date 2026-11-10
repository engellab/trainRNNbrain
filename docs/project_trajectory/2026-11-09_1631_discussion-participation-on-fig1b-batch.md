## 2026-11-09 16:31 — The Discussion's participation numbers recomputed on the batch that Fig. 1b draws (round 20, Reviewer R item D)

Package r20-text (dead-relu-paper #213; committee #212, Reviewer R's recommended correction D). No network
was trained or re-simulated; only stored participation vectors and rates were read.

**What was wrong.** The Discussion said that in the network of Fig. 1b, 515 of the 1,000 units have
participation below 2e-4, and that the median participation of the active units is about 17,000 times that
of the inactive units. Those numbers (entry `2026-11-06_2135_r17-claims-figures`) were computed from
`data/fig_paper_F1_collect.pkl`, the 31 Oct collection, whose example batch was unseeded (183 active units).
Fig. 1a,b has been drawn since round 13 from `data/fig_paper_F1_collect_v2.pkl` (`fig_paper_F1.COLLECT`),
whose 24-trial batch is drawn with task seed 11 (184 active units; entry
`2026-11-02_2341_r13-figfix-seeded-fig1ab-and-ed4-open-markers`).

**Recompute.** Network 3374117298 (3BFF, N = 1,000, 200,000 iterations), `['example']` of each file. Two
participation vectors per file: the stored `p`, and an independent recompute from the stored rates
(`rates`, shape 1000 x 300 x 24): p_i = std(r_i) + q0.9(|r_i|) over all time steps and trials, threshold 5%
of q95(p).

```
python3 -c "
import pickle, numpy as np
for f in ['data/fig_paper_F1_collect.pkl', 'data/fig_paper_F1_collect_v2.pkl']:
    e = pickle.load(open(f, 'rb'))['example']; R = e['rates'].reshape(1000, -1)
    for name, p in [('stored', e['p']), ('recomputed', R.std(1) + np.quantile(np.abs(R), 0.9, axis=1))]:
        act = p >= 0.05 * np.quantile(p, 0.95)
        print(f, name, act.sum(), (p < 2e-4).sum(), np.median(p[act]), np.median(p[~act]),
              np.median(p[act]) / np.median(p[~act]))
"
```

| batch | active | below 2e-4 | median p, active | median p, inactive | ratio |
|---|---|---|---|---|---|
| 31 Oct, unseeded (`fig_paper_F1_collect.pkl`) | 183 | 515 | 1.323 | 7.61e-5 | 17,386 |
| task seed 11, drawn in Fig. 1b (`fig_paper_F1_collect_v2.pkl`) | 184 | 506 | 1.282 | 7.01e-5 | 18,303 |

The stored and the recomputed vectors agree exactly on both batches.

**Paper change.** The Discussion now reads "506 of the 1,000 units have participations below 2 x 10^-4",
"the median participation of the active units of that network is 1.3, about 18,000 times the median
participation of its inactive units" (dead-relu-paper branch wp/r20-text, sections/07_discussion.tex). No
claim changes: both batches give about half the units below 2e-4 and a ratio near 2 x 10^4.
