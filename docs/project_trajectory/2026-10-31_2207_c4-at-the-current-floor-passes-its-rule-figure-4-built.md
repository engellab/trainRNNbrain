## 2026-10-31 22:07 — C4 at the current floor passes its rule; Figure 4, ED Figs 6-7 built (dead-relu-paper#8)

Follows `2026-10-31_2137_c4-re-read-at-the-current-floor-rule-set-before-the-run.md`, which set the
rule before the job ran.

**How it was run.** Della CPU job 16566988 (16 cores, 20.5 min wall, 3.9 CPU-h, 19.7 GB peak, no
GPU) ran `c4_gradient_mechanism.py --at floor`, `fig_paper_F4_noreturn.py collect` and
`c4_snapshot_drift.py`, as written in the entry above (launcher `c4floor.sh`; script md5
2ed01306793b4e19048f14ecefc5bf2a, whose `collect` is unchanged in the committed file). Output on
Della `/scratch/gpfs/TENGEL/pt1290/fig4_noreturn/`, copied to the Mac under `data/fig4_noreturn/`
(`c4_floor/`, `traces/`, `snap/`, the job log). One change after the job: `c4_snapshot_drift.py`
recomputes participation on only the first 256 trials, where a CDDM network has about 80 fewer
active units than on the full 450-trial batch that defines the classes (334 against 412 in
s1423687489). So participation at t*, t* + 500 and t* + 5,000 was recomputed on the full reference
batch on the Mac (`fig_paper_F4_noreturn.py activity`, 2 min). Its pass at t* reproduces the C4
classes in every unit of all six networks (largest relative difference in p 7e-15).

**Per network** (noise-free reference batch, task seed 12345; active / silent / low-active, and the
low-active units the gradient reaches):

| Network | Floor snapshot | Active | Silent | Low-active | Reached low-active | Reached low-active active at +500 / +5,000 |
|---|---|---|---|---|---|---|
| CDDM s1423687489 | 25,500 | 412 | 313 | 275 | 56 | 29% / 29% |
| CDDM s2657899762 | 26,500 | 421 | 293 | 286 | 39 | 33% / 38% |
| CDDM s2670295995 | 24,500 | 439 | 260 | 301 | 46 | 35% / 20% |
| 3BFF s1258717025 | 67,500 | 268 | 0 | 732 | 143 | 2.1% / 0.7% |
| 3BFF s2461724787 | 50,500 | 344 | 0 | 656 | 151 | 0% / 2.6% |
| 3BFF s4113307577 | 58,500 | 287 | 0 | 713 | 165 | 1.2% / 5.5% |

- Silent CDDM units: incoming-weight gradient exactly 0 in the float64 numpy backward and in PyTorch
  autograd (`prod_g_in`). At +5,000 iterations at most 1 per network is active (0, 0, 1).
- Oracle: per-unit g_in of the numpy backward against autograd, median relative difference
  1.3e-7 to 2.4e-6, largest 2.4e-4; reached counts identical in all six networks.
- 3BFF reached low-active units: median share of steps with x > 0 of 2.6-3.7%, on in a median
  929-941 of 1,024 trials, 96-98% of their on-steps within 2τ after an input pulse.
- Active units still active at +5,000: CDDM 89-91%, 3BFF 97-99%.
- Reached / active at the 5% bar: 3BFF 1.44-1.55, CDDM 0.99-1.04; the 3BFF ratio falls below 1
  only for bars of 19% of q95 and more.
- Against the 13 Oct read: CDDM silent 260-313 (was 271-322); 3BFF reached 143-165 (was 145, 273,
  195; the 273 came from s2461724787 read at 27,000 instead of 50,500).

**Rule: all four clauses pass.** 3BFF has 0 silent units; every CDDM network has at least 200 silent
units with g_in exactly 0; every 3BFF network has at least 100 reached low-active units; in every
3BFF network fewer than 10% of them are active 5,000 iterations later (0.7%, 2.6%, 5.5%).

**Found on the way, not a claim yet.** Silent units' incoming rows still move:
`c4_snapshot_drift.py` reports that over 5,000 iterations the incoming rows of the units the
gradient does not reach (silent units and unreached low-active units, one class in that script)
shrink by a median 39-41% on CDDM and 14-17% on 3BFF (radial part of the change); a silent unit has
no task gradient at all. The only
term acting on them is weight decay (10^-6, added to the gradient), which Adam rescales to steps of
order the learning rate. Results 4 now says the weights "change only through weight decay" instead
of "gradient descent cannot change them". Whether that decay can ever make a silent unit's input
positive is untested here; at +5,000 iterations at most 1 silent unit per network is active.

**What the paper now uses.** Figure 4 (`fig_paper_F4_noreturn`), Extended Data Fig. 6
(`fig_ed_mechanism`, panels a-b from this read, c-e from the C4b outputs of 14 Oct) and Extended
Data Fig. 7 (`fig_ed_tanh`, jobs 15064275 and 15066631 and the 18 participation traces of
`CDDM_std_g0_tanh`), all drawn by `fig_paper_F4_noreturn.py draw` on the Mac, which prints every
number. The tanh numbers reproduce entry 2026-10-01_2017 (median drop 27-fold, 6-247; output
sensitivity 25-147 tanh units against 277-293 ReLU) and 2026-10-16_1614 (0 inactive units in 9 tanh
networks; lowest unit 0.09-0.89 of q95). The B4 panels recompute the gate and verdict from the
per-run rows of `read_b2_b4.json` and match the read's own (`b4_verdict_1000`,
`b4_verdict_4000.json`) at both sizes.

**Corrections to wording elsewhere.** The old Results 4 text said an inactive 3BFF unit's current
is positive on fewer than 10% of steps; the largest share among low-active units is 21-26%, and only
the median reached unit is at 3-4%. It said the reach bar was set on the untrained network; for
ReLU (C4) it is 5% of the trained network's own q95 of g_in (the untrained bar is the tanh measure's).
It said two of three 3BFF losses change by under 1% per 100k from 150k; the C4b slopes are -0.8%
and -0.5% per 100k over 250-300k for the two 300k chains, +2.8% and +0.2% over 150-200k, and -2.6%
over 150-200k for the 200k chain.

Figures: [Figure 4](../../img/internal_figures/fig_paper_F4_noreturn.svg),
[ED Fig. 6](../../img/internal_figures/fig_ed_mechanism.svg),
[ED Fig. 7](../../img/internal_figures/fig_ed_tanh.svg).
