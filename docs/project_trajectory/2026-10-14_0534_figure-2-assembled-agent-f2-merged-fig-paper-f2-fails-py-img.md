### Figure 2 assembled (agent-f2, merged; `fig_paper_F2_fails.py`, `img/internal_figures/fig_paper_F2_fails.png`)

Per ladder a pair of panels sharing the active-units axis: left, held-out R² at the floor against
active units (the scatter Pavel asked for, transposed); right, active units against iteration (log
axis), every run, its floor crossing marked with the same symbol, budget dotted. a-i the nine
ladders, j CDDM revival (re-initialising silent units tracks the control and the input-weight cap;
2 of 3 re-initialise runs landed), k flip-flop revival (placeholder until the top-ups land; the
builder picks them up with `--collect`), l every rung at its own floor (open) and at its ladder's
budget (filled). Checks: crossings equal read_rung's; an independent recount from the raw traces
matches 40 of 41 runs at every probe (one off-by-one at a unit 3e-9 from the threshold).
Contradictions with the paper's knob text, fixed there (a073755 → next commit): no-recurrent-noise's
486 is an early read (floor at 14-15k vs 28-31k; fewer units from 25k on, 274-283 vs 349-371 at
50k); flip-flop input-scale floor numbers under the beta guard are ~300 (1 run) → 764 at scale 20.

