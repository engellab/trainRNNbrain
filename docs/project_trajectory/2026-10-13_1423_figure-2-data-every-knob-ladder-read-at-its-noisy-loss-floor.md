## 2026-10-13 14:23 — Figure 2 data: every knob ladder read at its noisy-loss floor

**Setup.** New laptop (restored from the Della backup; `restore-oct5` is a worktree at
`~/Documents/GitHub/trainRNNbrain_restore`, beside the other agents' `dimensionality-analysis` tree).
`FFtop2000` (15431474) finished: the three flip-flop N = 2000 controls are at 200k, held-out r2
0.953-0.954. `FFtop4000` (15431475) is at ~114k of 200k, due ~16:00 on 15 Oct.

**`fig_knobs_at_floor.py` moved to `common.noisy_floor` at 1.05x.** It was also broken: it still
called `F1.fit_floor_at` / `excess_time_at`, which now live only in `pr_matrix.py`. The noise ladder
now reads its 200k cells and the metabolic ladder its 75k continuations (`_c75000`), with the 200k
control's end count read AT 75k (`end_at`), so no ladder compares ends at different budgets. A run
whose participation trace does not cover its crossing is dropped with that reason. Every cell was
checked standard from its saved config (ReLU or the tested activation, h, gamma 0, no Dale,
self-connections on, zero bias, wd 1e-6 unless that is the knob).

Check set before reading: the reference rung (standard CDDM, N = 1000) must read 380-460 active at a
crossing of 24-38k (Figure 1c: 414 ± 13; claim-1 runs 415-438). It reads 417 / 422 / 416 at
27-29k. Pass.

Active units of 1000, mean per rung (3 seeds; 5 for noise, 2 for the recinit ladders), at the floor
→ at the end; reference rung in bold:

| knob (CDDM unless noted) | rungs | at 1.05x floor | at end |
|---|---|---|---|
| weight decay, 200k | 0 / **1e-6** / 1e-5 / 1e-4 | 488 / **418** / 363 / 377 | 412 / **266** / 166 / 86 |
| recurrent noise sigma, 200k | 0 / 0.01 / **0.05** / 0.1 | 486 / 451 / **406** / 433 | 180 / 186 / **264** / 308 |
| metabolic lambda, 75k | **0** / 0.01 / 0.1 / 1 / 10 | **418** / 460 / 452 / 425 / 397 (2 of 3) | **337** / 337 / 351 / 375 / 330 |
| spectral radius, 75k | 0.5 / **1.2** / 2.0 | 465 / **423** / none read | 361 / **340** / 313 |
| W_rec magnitude penalty, 75k | **0** / 0.05 / 0.2 | **423** / 534 (1 of 2) / 538 | **340** / 438 / 448 |
| activation, 200k | **ReLU** / leaky 0.01 / softplus 25 / sigmoid | **418** / 485 / 519 / 245 (2 of 3) | **266** / 276 / 250 / 188 |
| input scale, flip-flop, 150k | **default** / 0.5 / 2 / 5 / 20 | **316** (2 of 3) / 394 / 488 / 611 / 717 | **206** / 231 / 298 / 302 / 329 |

Reading: at the floor every CDDM knob but sigmoid (245) stays within about 120 units of the reference's 418; the end
read-out is where they spread (weight decay: 412 with none against 86 at 1e-4, a 5-fold range).
Input scale on the flip-flop is the one knob with a large effect at the floor (316 → 717), and its
crossings come early (10-12k at scale 20 against 23-73k for the default); by 150k the gain has
shrunk to 206 → 329.

Exclusions, with reasons: spectral radius 2.0, both runs (fit pinned at its bound / loss ends far
above the fit); W_rec penalty 0.05, one run (fit pinned); metabolic 10, one run (loss ends 7.8x
above its floor); sigmoid, one run (never reaches the bar); flip-flop default, one run (never
reaches the bar).

⚠️ **One read-out is suspect and not excluded by the rule:** a flip-flop default-input run crosses
at 23k but ends at 0.73x its fitted floor, i.e. its loss kept falling well past the "floor".
`noisy_floor` guards only against a late rise; a lower guard was tried and dropped on 2026-10-12
because DMTS sits below its floor between bursts. Left as is and flagged.

**The revival screens cannot be read at their floor.** `CDDM_screen3` (30k; arms none, cap, prune
= redraw silent units, prune_copy, scale, scale_ln): no arm has a single usable fit except `scale`,
whose crossings at 2-4k are before training has done anything (1 run ends with 0 active). Every
`NBitFlipFlop_replace_variants` cell (40k; bias_kick, copy, mix, orth, random, zero_out) reads at
most 2 of 3 runs, most 0-1; only copy at rate 0.025 reads all three. Both need top-ups (CDDM to
~75k, flip-flop to 200k) before they can go into Figure 2 under the floor rule. Not started:
compute to be agreed with Pavel. `CDDM_prune_reinit`, named in the audit's data table, is on
neither Della nor Spock; the re-initialisation result (26,012 redraws) is `CDDM_screen3/arm=prune`.

Figures: `img/internal_figures/slide_k_{weightdecay,recnoise,metabolic,recgain,rwm,activation,inputscale}.{pdf,svg,png}`.

