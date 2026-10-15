## 2026-09-26 10:20 — self-connection audit, after the 2x2 block fix

Pavel asked for a check that the self-connection cannot cause a blow-up. A ReLU unit above
threshold evolves as h <- h*(1 − alpha + alpha*w) under its own self-weight, with alpha = dt/tau =
0.1, so it self-excites once w > 1.

| cell | % self-weights negative | most negative | largest positive | units with w > 1 | rho(W_rec) |
|---|---|---|---|---|---|
| control N=500 (gamma 0.1) | 43–46 | −2.45 | 1.06–2.06 | 1, 2, 1 | 7.3–7.7 |
| duplication N=500 | 75–80 | −1.11 | 1.75–2.94 | 3, 1, 1 | 7.5–9.9 |
| control N=1000 | 33–38 | −1.73 | 0.49–0.76 | 0 | 7.6–8.3 |
| duplication N=1000 | 62–72 | −0.36 | 0.27–0.39 | 0 | 4.6–5.9 |
| control N=2000 | 26–28 | −1.41 | 0.22–0.32 | 0 | 7.9–8.2 |
| duplication N=2000 | 66–74 | −0.22 | 0.10–0.29 | 0 | 9.0–10.6 |

Self-exciting units appear only at N=500 and the CONTROL has them too, so duplication does not
create them. Duplicated networks come out substantially more self-INHIBITED (65–80% negative
diagonals against 26–46%) and at N=1000 at a lower spectral radius. 11 of the 15 jitter runs
discarded zero gradient updates and the other four 0.098–0.887%, far under the 5% gate.

⚠️ CORRECTION. An earlier entry reported "the largest |W[i,i]| is 3.27, exceeding the largest
off-diagonal weight" without the sign. The large self-weights are predominantly NEGATIVE: 26–62% of
diagonals are negative, extremes reach −2.45, and the largest positive is 0.2–0.8. The
self-connection here is mostly a damping term, which reframes the bug fixed on 2026-09-24: zeroing
it detuned units by removing inhibition they relied on, not by risking runaway excitation.

`rescale` cannot compound a self-weight at all -- the outgoing redraw overwrites the diagonal every
event. Checked at a 17292x cumulative boost with the cap disabled: largest self-weight 0.18.

