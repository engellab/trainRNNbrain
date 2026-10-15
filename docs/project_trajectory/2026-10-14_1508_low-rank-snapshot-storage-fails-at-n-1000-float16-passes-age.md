### Low-rank snapshot storage: fails at N = 1000; float16 passes (agent-lowrank, merged; notes `docs/notes/agent-lowrank.md`, figure `img/internal_figures/lowrank_storage.png`)

Pass bar committed first (b2c0e07): |ΔR²| ≤ 0.002 (seeded, 36 draws), active count within 1% with
≥ 99% of units classified the same, noisy loss within 1%, per-unit gradient norms r ≥ 0.99. 18
snapshots (claim-1 networks × floor/end × 3 tasks, N = 1000); the save-and-score path reproduces the
originals exactly.
- **W ≈ a·W0 + rank r fails: training erases the initial matrix.** a = 0.99 at iteration 500 but
  0.01-0.16 at the floor and 0.00-0.06 at the end, so ΔW is W itself; the W0 scheme passes at the same
  rank as a plain SVD (17 of 18; one flip-flop floor needs 700 vs 500). 99% of ‖ΔW‖² lies in 160-590
  directions at the floor and 66-254 at the end (a Gaussian matrix: 772). Smallest passing rank:
  500 on every task at N = 1000 (4.0 MB, more than the 3.8 MB compressed full matrix). CDDM at the
  end: rank 300 at N = 1000, 200 at N = 2000 (3.2 MB) and N = 4000 (6.4 MB vs 64 MB) - one network
  each, end only; no N ≥ 2000 floor snapshots exist to test.
  Interpretation, untested: Adam's normalised weight-decay step drives the erasure (test: AdamW or
  wd 0 runs).
- **Quantisation passes:** float16 (1.9 MB compressed at N = 1000) passes all 18, measures unchanged
  to 4 decimals; 12-bit with one scale per row (~1.1 MB) passes all 18; 10-bit 14; 8-bit 7.
**Decision (Pavel): kept snapshots in reduced precision.** Implemented (commit above): kept
snapshots `it<iteration>/` store W_rec as float16 (all other arrays float32), compressed, every 1,000
iterations; in keep mode an exact resume copy (float32 + Adam state) is overwritten in `resume/` at
each checkpoint. At N = 1000 a kept snapshot is ~1.9 MB (was 8.1 MB); at N = 4000 ~30 MB (was 128
MB). Purpose (Pavel): find each run's floor after training and restore the network there, so
networks can be compared at their own floors. Note: a B4-style transfer from a float16 snapshot is
no longer bit-exact (metric-equivalent per the validation above); existing snapshots stay float32.
New sweeps pick this up only if launched from code at or after this commit.
**Kept snapshots are the default (Pavel: "new runs should keep the snapshots from now on").** Every
trainer config now has `checkpoint_every: 1000` and `checkpoint_keep: true`; a launcher opts out with
`++trainer.checkpoint_keep=false`. Launchers passing `+trainer.checkpoint_keep` now pass `++`.
Smoke-tested through run_experiment (no flags, and the launcher form): it0001000, it0002000 and
resume/ written.

**Figure 2 complete with the flip-flop revival screen (panel k).** RVtop200 finished (6 runs, ~10.5 h
each). Mean active at the floor → at 200k: no intervention 290 → 176; zero-out silent units
265 → 210; random redraw 232 → 194. At the floor both revival rules have FEWER units than the
control; by 200k 18-34 more. Neither rescues participation (as on CDDM: re-initialisation 419 → 338
vs 436 → 335). Legend placement fixed (no legend over data, layout check 0 problems); the figure is
built on the laptop from the Della cache (`data/fig_paper_F2_fails_cache.pkl`), because Della's
default font makes three legends too wide to fit. `img/internal_figures/fig_paper_F2_fails.png`.

**W1 drafted (paper branch `w1-draft`, 47b8d51; not merged).** Abstract, intro and discussion rewritten
to Pavel's framing with current numbers; `04_ratepenalty` and `06_whyitmatters` brought in line with
the floor reads (old "+123 units", "~710 units", old activation deltas replaced). Compiles, no new
references. Markers: PENDING (remedies at the floor, B2, B3, B4, flip-flop N = 4000, B5/revival into
Results, C4/tanh into Results, A5, Figure 3) and PI TO CHOOSE (title, abstract's last sentence,
gradient-loss naming, C4 wording, "likely several"). Coordinator's check: the abstract's claim that
noisy synapses, rescaling, dropout and duplication "each recruit some" rests on the 75k remedy runs
(before the flip-flop converged) and should be marked pending until RMtop200 is read at the floor.
Found by the agent, not fixed: `08_methods` line 38 still says biases are trainable; main's Figure 2
caption is the old dropout figure.

