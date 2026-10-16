## 2026-10-16 15:55 — C4 and C5 placed in the paper; Figure 2 synced under the current read-out

Pavel approved the placement on 16 Oct. Paper commits 05b2383 (text) and a8ccd16 (Figure 2).

- **C4** is now its own Results subsection, `sec:mechanism` ("Why units stay below the activity
  threshold"), after the knob section in `sections/02_problem.tex`. It defines silent units (net
  input current negative on every time step, so the rate and the gradient on incoming weights are
  exactly zero) and low-active units (below the threshold, positive at least transiently). It covers
  the CDDM and flip-flop facts, the convergence test, and tanh (moved from the intro). Wording: entry
  2026-10-16_1115, with Pavel's edits.
- **C5** is `sec:knobs`, rewritten in shorter sentences with every number from Figure 2 under read-out
  b121e02 + be17ac9 (entry 2026-10-16_1348). Most settings move the count at the floor by about 120
  units at most. The input-weight scale raises it on both tasks; on CDDM those networks reach the
  floor 4 to 6 times sooner and then lose units faster.
- **Intro, discussion and abstract** now carry short summaries that point to Results. The abstract's
  "~120 units on the decision task" was not true under the current read-out; it now says the active
  count keeps falling after the loss settles under every standard setting tried.
- **Figure 2** is synced (source 3395429). Its caption no longer marks spectral radius 2.0 as having
  no read-out.

**Still inconsistent:** outside these passages, about 170 uses of "silent" in the paper, plus the
figure labels and section titles, still mean "below the activity threshold". The new subsection
reserves "silent" for units that never fire. A paper-wide pass waits for Pavel's confirmation of the
terms.
