# Agent F2: Figure 2, "what fails"

Builder: `trainRNNbrain/experiments_and_analysis/fig_paper_F2_fails.py` (new; the old
`fig_paper_F2.py` is untouched because other code imports it). Output:
`img/internal_figures/fig_paper_F2_fails.{pdf,svg}`, 179 x 197 mm. Branch `agent-f2` (from
`restore-oct5` at af727ba). Della checkout `~/trainRNNbrain_f2`; collect jobs 15538802 (data) and
15539694 / 15539716 (checks), CPU only. Started 04:56, 2026-10-14.

## The problem the layout answers

The floor read-out (each run read where its noisy loss first reaches 1.05x its own fitted floor)
reads a knob that speeds convergence earlier, while more units are still active. So every ladder is
drawn twice, on one shared active-units axis:

- left, narrow: held-out R^2 at the floor against active units at the floor, one point per run (the
  PI's trade-off scatter, transposed so the active-units axis is vertical and shared; dose ladders
  in viridis dose order, categorical ones by marker shape, the standard network ringed);
- right: active units against training iteration (log axis, 2k to the budget) for every run, each
  run's crossing marked with the same symbol. A point on the left and its crossing on the right sit
  at the same height. Dashed = no usable crossing. Dotted vertical = the ladder's common budget.

Rows: (a-c) weight decay, recurrent noise, metabolic; (d-f) spectral radius, W_rec magnitude
penalty, rate/sparsity; (g-i) activation, input scale on CDDM, input scale on the flip-flop; (j, k)
the revival screens; (l) the bottom line: every rung's mean at its own floor (open) and at its
ladder's common budget (filled). Budgets are written in every title and every row of (l); no
comparison crosses ladders.

Traces are drawn as medians in 40 log-spaced bins. Probe-to-probe noise is small (median step 1-4
units), but a flip-flop count drops by 100-350 units within a single probe 4-10 times per run on the
input-scale ladder; drawn raw, those 15 traces could not be told apart. The crossing symbol is the
raw count, so on the flip-flop it can sit off its smoothed line.

Revival screens: `collect_spec` reads `CDDM_screen3_c75000/` and
`NBitFlipFlop_replace_variants_c200000/` whenever they hold finished runs; a slot whose treated
cells have no run is a dashed placeholder. At collect time (05:03) CDDM had cap 3/3 and
re-initialise 2/3 (S3top75 still running), so panel j is drawn; the flip-flop cells were empty
(RVtop200 ~2 h into 18 h), so panel k is the placeholder. To refresh: on Della,
`python trainRNNbrain/experiments_and_analysis/fig_paper_F2_fails.py --collect` (CPU job, ~18 min),
copy `data/fig_paper_F2_fails_cache.pkl` back, rerun without flags to draw. Limitation: a cell with
2 of 3 finished runs is drawn as "2 of 2" with no hint a third is coming.

The flip-flop revival reference is `NBitFlipFlop_ff_revive_c200000` arm none: the replace_variants
configs differ from it only in the prune keys (diffed), so it is the same network without the rule.

## Checks

1. Crossings: `collect_spec` recomputes each run's crossing with the same `noisy_floor` call and
   asserts it equals `read_rung`'s set; it passed on every rung.
2. Independent recount (bar set before: every probe identical): a separate script reloads the raw
   ParticipationTrace pickles and recounts with its own 5%-of-q95 code, first run of every rung,
   41 runs x up to 2,000 probes. 40 runs identical; one differs by 1 unit at one probe (iteration
   4,900 of the no-weight-decay run). Root cause: the stored participation is float32, the recount
   used float64, and one unit sits 3e-9 from the threshold. Not a figure error.
3. Layout: `check_layout` tests every legend against every drawn vertex and symbol, every pair of
   drawn texts (tight bounding boxes, tick labels only inside view limits), and text outside the
   canvas. Final: 214 texts, 10 legends, 0 problems. The flip-flop input-scale panel has no room
   for a legend and says "rungs as in h" (same rungs and colours).
4. Matched counts reproduce the trajectory's CDDM input-scale entry (25k: control 426/430/438,
   scale 2 328/337/337; 50k: 355/355/376 and 247/257/268).

## Numbers (mean active of 1000: own floor -> common budget; runs per rung 2-5)

| ladder (budget) | rungs | floor -> budget |
|---|---|---|
| weight decay (200k) | none / **1e-6** / 1e-5 / 1e-4 | 488->412 / **418->266** / 363->166 / 377->86 |
| recurrent noise (200k) | 0 / 0.01 / **0.05** / 0.1 | 486->180 / 451->185 / **406->264** / 433->308 |
| metabolic (75k) | **0** / 0.01 / 0.1 / 1 / 10 | **418->337** / 460->337 / 452->351 / 425->375 / 396->330 |
| spectral radius (75k) | 0.5 / **1.2** / 2.0 | 465->361 / **423->340** / no floor->312 |
| W_rec penalty (75k) | **0** / 0.05 / 0.2 | **423->340** / 534->438 / 538->448 |
| rate / sparsity (75k) | **none** / rate 0.2 / sparsity 0.05 / both | **415->325** / 1000->1000 / 669->265 / 1000->1000 |
| activation (200k) | **ReLU** / leaky / softplus / sigmoid | **418->266** / 485->276 / 519->250 / 245->188 |
| input scale, CDDM (50k) | **default** / 0.5 / 2 / 5 / 20 | **430->362** / 537->335 / 706->257 / 656->209 / 434->176 |
| input scale, flip-flop (150k) | **default** / 0.5 / 2 / 5 / 20 | **298 (1 of 3 read)->206** / 394->231 / 488->298 / 611->302 / 764->329 |
| revival, CDDM (75k) | **none** / input cap 0.3 / re-initialise | **436->335** / 462->395 / 420->342 (2 runs) |

Crossings (k it): recurrent noise 0 at 13.7-15.2 against the default's 27.5-31.4; W_rec penalty
15.7-18.4 against 25.7-25.9; sparsity 12.2-12.8 against 21-32; CDDM input scale 2 at 4.4-4.8
against 24.9-25.9.

Readings, from the numbers above:
- Floor gains that are early reads (the knob crosses sooner, then has fewer units at the same
  iteration): no recurrent noise (50k: 274-283 against the default's 349-371), the sparsity
  penalty, CDDM input scale 0.5-20.
- Gains that survive to the common budget: no weight decay (+146 at 200k), recurrent noise 0.1
  (+44 at 200k), W_rec penalty (+98 to +108 at 75k), input-weight cap 0.3 (+60 at 75k),
  flip-flop input scale 2-20 (+92 to +123 at 150k; 0.5 is +25), and the rate penalty (all 1000). Leaky ReLU and
  softplus are +50 to +70 at 50k and within 16 units of ReLU at 200k.
- Re-initialising silent units on CDDM changes nothing at 75k (342 against 335; 2 runs).

## Where the data contradict the paper's knob text (`sections/02_problem.tex`, sec:knobs)

1. "removing the recurrent noise 486" is listed as a gain. It is an early read: sigma = 0 crosses at
   14-15k against 28-31k and has fewer active units than the default at matched iterations from
   25k on (25k: 359-391 against 405-428; 50k: 274-283 against 349-371; 200k: 176-188 against
   254-288).
2. "No setting moves that count by more than about 120 units": the A4 ladder now in this figure
   breaks it at the floor (sparsity 669, +250; rate penalty 1000). Sparsity is an early read (265
   against 325 at 75k); the rate penalty is a real exception and belongs with Figure 3.
3. Flip-flop input scale "about 316 to ... 717": after the beta guard the default reads 298 from 1
   of 3 runs and scale 20 reads 764 from 2 of 3 (614, 913). The end numbers (206 to 329) still hold.
