## 2026-11-02 22:32 — Confirmatory test on 39 unseen networks: W1, W3 and W4 differ between all units and active units; the #172 test supports weight decay

Result of the rule `2026-11-02_2218_connectivity-confirm-rule.md` (44cd160) with its amendment
`2026-11-02_2223_connectivity-confirm-rule-amendment.md` (cd92893). Both were committed before any
statistic was computed. Work package r12-weights (dead-relu-paper #127, item 52; #172).

**Run.**
- Inventory: Della CPU job 16711892, `connectivity_inventory.py`.
- Copy and trace extraction: Della CPU job 16712459, `connectivity_confirm_fetch.py`, job list from
  `connectivity_confirm_joblist.py`. Data in `data/connectivity_confirm/` and
  `data/connectivity_confirm_joblist.json` (gitignored, Mac), from `/scratch/gpfs/TENGEL/pt1290/r12w_confirm`.
- Analysis on the Mac, CPU: `PYTHONPATH=<code worktree> python trainRNNbrain/experiments_and_analysis/connectivity_confirm.py`,
  writing `data/connectivity_confirm.json`. Code commit: the commit that adds this entry.

### Checks

- **M1 (active mask against the trace): passed.** In all 39 networks, the count from `simulate` is
  within 2.4% of the trace's count at the probe nearest the read iteration (largest gap: #27, CDDM
  3090320285, 360 against 369). The active counts are 155-414.
- **Twin check: valid in 30 of 39 networks.**
  - T, on stored row norms, passes for #4 and #37-#39: maximum relative difference 3e-7 to 3e-6.
  - T', on stored participation (amendment), passes for all 26 CDDM networks: 2e-6 to 5e-5, against a
    limit of 1e-3.
  - Six 3BFF networks (#1-#3, #5-#7) and three DMTS networks (#34-#36) have first-run traces with
    neither field usable. They use the rule's fallback θ = 2 × 1.2/√N = 0.0759. The valid twins give
    θ = 0.072-0.075.
- **M3 (pure-Python hand check): passed** on #1 (3BFF 3374117298, fallback θ) and #12 (CDDM
  970815651, valid twin). θ, W1, W3 and W4, over the active units and over all units, agree to
  within 2e-16. The first rule's network still passes after the script was extended to W4.
- **The #172 slope**, recomputed with `numpy.linalg.lstsq` through the origin on #40 and #11, agrees
  with the script to six decimals (`scratchpad/r12-weights/slope_check.py`).

### Main test: all three statistics differ in all 39 networks

rel = (S_all − S_act) / S_act. Condition 1: |rel| > 0.10. Condition 2: S_act lies outside the range of
200 random subsets of |A| units.

| statistic | condition 1 | condition 2 | rel, 3BFF (7) | rel, CDDM (26) | rel, DMTS (6) | S_act / largest subset value, lowest network |
|---|---|---|---|---|---|---|
| W1, s.d. of the recurrent weights | 39 / 39 | 39 / 39 | −0.57 to −0.79 | −0.52 to −0.74 | −0.64 to −0.83 | 1.7 |
| W3, median normalised in-degree | 39 / 39 | 39 / 39 | −0.99 to −1.00 | −0.88 to −1.00 | −1.00 | 4.3 |
| W4, median normalised out-degree | 39 / 39 | 39 / 39 | −0.96 to −0.99 | −0.97 to −1.00 | −1.00 | 5.0 |

**Verdict: W1, W3 and W4 differ** between all units and active units in all 39 networks, in the same
direction: lower over all units. The prediction of the rule entry held. No network came close to the
falsifying result.

W1 values (s.d. of the recurrent weights; the initial s.d. is 0.036-0.038 in every network):

| task | networks | over all units | over the active units | largest of 200 random subsets of \|A\| units |
|---|---|---|---|---|
| 3BFF | 7 | 0.040-0.042 | 0.098-0.200 | at most 0.077 |
| CDDM | 26 | 0.016-0.021 | 0.043-0.060 | at most 0.026 |
| DMTS | 6 | 0.009-0.013 | 0.037-0.057 | at most 0.017 |

W3 and W4 are degenerate over all units. The median normalised degree there is 0 to 0.009 (exactly 0
in most CDDM and every DMTS network), because the median unit has no weight above θ. Over the active
units it is 0.03-0.25. W1 needs no threshold and has no degenerate value, so it is the statistic to
name.

Per-network values for all 39 networks (all, active, rel, subset range) are printed by the script and
stored in `data/connectivity_confirm.json`.

### The #172 test: supported

Slope b of the trained inactive–inactive weights W[I, I] on their initial values, and the s.d. of
W[I, I] over the s.d. of W0[I, I], on the final weights:

| arm | networks | b | s.d. ratio |
|---|---|---|---|
| CDDM, weight decay 1e-6 (#11-#13, 200,000 iterations) | 3 | −0.000 to 0.000 | 0.029-0.036 |
| CDDM, weight decay 0 (200,000 and 300,000 iterations) | 6 | 0.979-0.986 | 1.90-2.14 |
| DMTS, weight decay 0 (secondary, 150,000 iterations) | 3 | 0.891-0.932 | 1.87-1.99 |

b > 0.9 in all six weight-decay-0 CDDM networks and b < 0.1 in all three weight-decay-1e-6
networks, so the outcome under the rule is **supported**. With weight decay 1e-6, the weights between
inactive units keep none of their initial values, and their s.d. is 3-4% of the initial s.d. With
weight decay 0, they keep their initial values. They also gain variance uncorrelated with those values:
their s.d. is about twice the initial s.d., so the added component has an s.d. of about 1.7 times the
initial s.d. (√(2² − 1)).

The comparison establishes that weight decay is needed for the loss of the initial values. It does
not test the Adam step-size explanation in #172 (coupled decay, normalised by the gradient's root mean
square). The extra variance at weight decay 0 is consistent with that explanation: Adam would scale
the small noisy gradients of these weights to steps of order the learning rate. That reading is an
interpretation and is not tested here.

### For the paper (the manager decides; the paper PR follows the manager's reading)

- **W1** is the statistic for the Introduction list (item 52). Over all units, its value is 52-83%
  lower than over the active units in all 39 unseen networks.
- **Extended Data Fig. 2a**, the replacement for the dimensionality of rates, can show W1. The open
  choice is between the nine Fig. 1e networks, where the first rule failed (on its size control) but
  the values are known, and the 39 networks of this test. The confirmatory evidence is the 39.
- **Why the s.d. over all units is lower:** in CDDM and DMTS networks, the s.d. over all units is
  below the initial s.d. (0.009-0.021 against 0.036-0.038), because the weights between inactive units
  shrink toward zero when weight decay is on (#172 test above).
