### C2: the flip-flop size series at its floor (agent-c2ff, merged)

Read at 1.05x each run's fitted noisy floor on the 200k continuations (`ff_revive_c200000`):
N = 500: 6 of 6 runs, crossings 38-111k, 201 ± 19 active; N = 1000: 2 of 3 (64-66k; the third never
reaches the bar), 291 ± 5; N = 2000: 3 of 3 (54-73k), 387 ± 24. Provisional exponent N^0.47 from
500-2000 (CDDM 0.46, DMTS 0.51); N = 4000 lands ~15 Oct. `fig_paper_F1.SCALING` now points at the
200k cells, and `panel_c` reads a series at its floor only when every LISTED size has a read-out (it
used to accept the sizes present on disk, so a missing N = 4000 would have dropped out silently).
Notes and per-run tables: `docs/notes/agent-c2ff.md`; script `c2_ff_floor.py`.

⚠️ **The floor fit is often degenerate on the flip-flop, and the guards miss it.** In 8 of 11
size-series runs the loss at 200k ends up to 8% BELOW the fitted floor, and several fits have the
stretch exponent beta pinned at its upper bound (3). `noisy_floor` rejects a floor pinned at its own
lower bound but not a fit pinned on beta or tau. Re-reading every run at 1.05x its smoothed minimum
instead (pre-registered: an exponent shift > 0.1 would mean the result depends on the floor
definition) gives N^0.45 against 0.47, so panel c is robust. Figure 1 panels d/e are less so: two of
the three claim-1 flip-flop runs have beta pinned at 3, and their crossings move 27k → 52k (352 → 341
active) and 39k → 60k (337 → 292); the third is unchanged (69k → 67k, 266). Decision for Pavel:
reject fits pinned on any bound, or read at 1.05x the smoothed minimum. The N = 1000 straggler sits
at 1.055-1.06x its floor from 125k to 175k and rises to 1.09x; continuing it to 300k ≈ 6.7 GPU-h, a
crossing not assured.

**The floor rule gains a guard (Pavel): a fit with its stretch exponent beta on the upper bound (3)
is no floor.** First Pavel chose "reject a fit with any parameter on a bound"; measured before
committing, that dropped 64 of 123 runs read at the floor, almost all healthy CDDM and DMTS fits
whose tau sits on its LOWER bound (100) only because the fast decay is over before the fit window
opens at 2,000 iterations (CDDM's loss ends at 1.00-1.01x those floors). The bad fits are the
beta-on-upper-bound ones (the loss ends at 0.73-0.92x of their "floor"). Pavel then chose the beta
guard alone. Predicted before running: the two claim-1 flip-flop runs 2461724787 and 4113307577
dropped, 1258717025 kept. Result: exactly 6 of 123 runs dropped - those two, the flip-flop control
that crossed at 23k and ended at 0.73x its floor (input-scale ladder reference), the N = 500
size-series run that crossed early at 38k, one input-scale-20 run and one DMTS N = 1000 run.

Consequences, not yet redone: Figure 1 panel d (ablation at the floor) and e read the flip-flop at
two crossings that are now invalid (27k, 39k), so the flip-flop has ONE claim-1 run at its floor
(69k, 266 active); the claim-1 flip-flop at-floor numbers in the 2026-10-12/13 entries rest partly on
them. Panel c's flip-flop N = 500 keeps 5 of 6 runs. The knob ladders lose one run each in the
flip-flop default and scale-20 rungs.

