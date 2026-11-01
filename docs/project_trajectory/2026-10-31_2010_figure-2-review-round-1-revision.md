# Figure 2: revision after review round 1 (dead-relu-paper PR #23)

Revises the drawing and wording of entry 2026-10-31_1957. No number of that entry changes, and no new
Della read was made: everything below comes from `fig_paper_F2_size.py` (drawing stage) run on the
Mac on `data/fig_paper_F2_size_collect.pkl` (Della job 16558986).

## What changed in the figures and the table

- **Fig. 2c**: the line at exponent b = 1 is now labelled "constant share". With active ∝ N^b, b = 1
  means the active share is the same at every N; every unit being active would also need a prefactor
  of 1. The old label "every unit active" was wrong. Fig. 2a keeps "every unit active" on its
  identity line, where it is correct.
- **ED Fig. 3a legend**: the "fitted decay" and "crossings" swatches are drawn in the task colours
  (3BFF and CDDM for the fitted decay, since DMTS has no fit), matching the drawn lines.
- **ED Table 2**: rows grouped by task (CDDM size series, then the CDDM 200k networks of Fig. 2d),
  seeds sorted as numbers, t* rounded to the nearest 1,000 iterations.

## The estimated R² of ED Table 2

Almost every estimated R² repeats one value per task (0.950 on 3BFF, about 0.83 on CDDM). The
estimate is R²(t*) = 1 - (L(t*)/L_end)(1 - R²_end). Over the 48 networks with an estimated value,
the ratio L(t*)/L_end lies between 1.02 and 1.06 on 3BFF (19 networks), 1.04 and 1.05 on CDDM (18)
and 0.87 and 1.10 on DMTS (11). Each estimate therefore sits within 0.01 of the network's final
held-out R², and the final R² values of one task are close (3BFF 0.951-0.953, CDDM 0.832-0.839,
DMTS 0.994-1.000). The repeated values are the formula working as written, not a copy error. The
table caption now says so. (Computed from the collect record's `r2_floor` and `r2_end`:
ratio = (1 - r2_floor)/(1 - r2_end), for read-out networks whose R² is estimated.)

## Rounding

The N for 1,000 active units on 3BFF is 14,178 (script); the paper now says 14,200, as the board
and entry 1957 do.
