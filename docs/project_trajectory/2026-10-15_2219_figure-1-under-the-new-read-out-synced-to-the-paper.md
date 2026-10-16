## 2026-10-15 22:19 — Figure 1 under the new read-out rule, synced to the paper

Figure 1 rebuilt on Della (`~/trainRNNbrain_c1ana` at 8a6c836) under the read-out committed in
b121e02: a Hampel-smoothed loss curve and a sustained crossing. Layout fixed:
- panel c names the 1,000-unit line and the diagonal in its legend, and its y floor is 20, so the
  legend sits under the data;
- panel d's DMTS legend sits clear of the traces.

**Panel c at 1.05× the floor** (Della job F1num; mean ± SD):

| task | N = 500 | N = 1000 | N = 2000 | N = 4000 | exponent | N for 1,000 active |
|---|---|---|---|---|---|---|
| 3-bit flip-flop | 209.5 ± 21.3 (4 of 6 runs) | 266.3 ± 29.1 (3 of 3) | 383.3 ± 24.1 | 548.3 ± 42.3 | 0.47 | ≈15,100 |
| CDDM | 298.3 ± 6.4 | 431.0 ± 9.5 | 548.3 ± 29.4 | 767.0 ± 10.1 | 0.44 | ≈7,300 |
| DMTS 5τ | 240.0 ± 36.8 (2 of 3) | 361.7 ± 133.9 | 437.0 ± 76.4 (2 of 2) | — | 0.43 | (3 sizes) |

**Changes from the committed-rule read (entry 2059):**
- Two N = 500 flip-flop runs and one N = 500 DMTS run have no read-out: their loss never stays under
  the bar.
- The N = 1000 straggler s3374117298 now reads out.

**Robustness to the margin** (exponent at 1.05× / 1.10× / 1.20×):

| task | exponent |
|---|---|
| flip-flop | 0.47 / 0.48 / 0.52 |
| CDDM | 0.44 / 0.52 / 0.57 |
| DMTS | 0.43 / 0.44 / 0.41 |

Every exponent stays between 0.41 and 0.57, so the count grows sublinearly at every margin. On CDDM
the margin moves the read-out time a long way: the crossing falls from ~26k at 1.05× to ~12k at
1.10× and ~4.5k at 1.20×.

**Panel e:** after the floor crossing, 90 more units fall silent on the flip-flop, 154 on CDDM and
194 on DMTS (N = 1000). Panel d is unchanged; its ablations were read at their earlier floor
snapshots.

**Paper** (`Dead_ReLU_paper`): the figure is synced, and every quoted number is updated, in the
caption, abstract, intro, Results (02, including the N = 1000 table) and discussion. The abstract's
"no standard training setting adds more than about 120 active units" is now scoped to the decision
task: on the flip-flop, a larger input-weight scale adds more (entry 2207).
