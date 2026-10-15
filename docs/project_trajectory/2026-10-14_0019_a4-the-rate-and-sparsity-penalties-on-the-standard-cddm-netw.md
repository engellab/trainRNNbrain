### A4: the rate and sparsity penalties on the standard CDDM network (agent-a4, merged; notes `docs/notes/agent-a4.md`)

`CDDM_std_g0_bias0_c75000` (h cells standard on every key; the s cells skipped), 5 runs per cell,
none excluded, R² at 36 draws (estimated at the floor; measured at 50k and 75k):

| cell | active at floor | crossing | R² floor (est.) | R² 50k | active 75k | R² 75k |
|---|---|---|---|---|---|---|
| no penalty | 415 (381-455) | 21-32k | 0.827 | 0.833 | 325 | 0.833 |
| rate 0.2 | 1000 | 17-21k | 0.820 | 0.828 | 1000 | 0.827 |
| sparsity 0.05 | 669 (655-692) | 12-13k | 0.820 | 0.831 | 265 | 0.835 |
| both | 1000 | 13k | 0.812 | 0.827 | 1000 | 0.828 |

**The rate penalty keeps all 1000 units, at a cost of 0.005-0.007 of held-out R²** (about 3 SE,
same sign on both measured sets) - 25-35x the 0.0002 seen on the pre-standard networks, so "no cost"
does not carry over. The sparsity arm's 669 at the floor is an early read (its loss, which includes
the penalty, crosses at ~12.5k and ends 4% below its fitted floor); at matched iterations it has MORE
units than the control only at 12.5k (659 vs 482) and fewer from 25k on (265 vs 325 at 75k). Four
pre-set checks passed (in-sweep control matches the reference; an independent numpy forward pass
within 6 units; R² at 50k vs 75k within 0.0097; re-reading the drift control 416-422). New ninth
ladder `slide_k_frm_rws` in `fig_knobs_at_floor.py`; the trade-off grid now takes as many rows as
the ladders need. Open: penalty arms' floors are fitted on task + penalty; the rate arm saturates the
count at 1000, so participation ratio would separate rate from both.

