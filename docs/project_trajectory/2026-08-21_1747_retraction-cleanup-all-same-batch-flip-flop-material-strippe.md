## 2026-08-21 17:47 — retraction cleanup: all same_batch flip-flop material stripped

Acting on the `same_batch=True` finding, everything derived from the first flip-flop sweep has been
removed from circulation rather than left to be misread later.

**Data quarantined, not deleted.** The 45 runs moved to
`data/trained_RNNs/RETRACTED_samebatch_NBitFlipFlop_ksweep/` on both the Mac and Spock, with a
`RETRACTED.md` explaining why. Renaming is what enforces it: every analysis script targets
`data/trained_RNNs/NBitFlipFlop_std_ksweep` by name, so the old runs are now unglobbable. Kept for
exactly one purpose — a same-batch vs fresh-batch comparison at matched (k, N), which is a real
methodological result.

**Figures deleted** — all 8 `img/internal_figures/flipflop_*.png`. They would otherwise be
indistinguishable from current results.

**Four trajectory entries banner-marked** ⛔ RETRACTED rather than deleted: the reasoning trail
should not be falsified by removing the steps that led to the error.

**Scripts stripped of retracted numbers.** All seven `flipflop_*.py` carried results in their
docstrings as fact — the 8.4-fold k-effect, the b = 1.11 -> 0.58 depth table, the floors at 0.012 and
0.0033, the 0.0055 deepest-reachable level. All removed and replaced with a provenance banner saying
the file states METHOD only, with no results, and not to reintroduce a remembered figure. The one
number kept is the target variance (0.720-0.738), because it is measured from the data generator and
not from any trained network; it is now quoted over the extended k = 2..8 range.

**Two defaults that were silently tuned to the bad data, fixed:**
- `flipflop_ksweep.py` hardcoded `L* = {0.022, 0.015, 0.010, 0.0055}`, derived from the memorisation
  loss range. Fresh-batch runs will floor HIGHER, so those levels would have been partly unreachable.
  The ladder is now DERIVED per run: the deepest level any cell's seeds all reach, and 4x that.
- Both loaders now raise a clear SystemExit naming the retraction when the sweep folder is empty,
  instead of six different cryptic numpy errors.

**Validation is now clean, and this is the real upside.** With `same_batch=False` every iteration
draws unseen trials, so `loss_clean_train` — recorded noise-free at every probe — is already an
unbiased held-out estimate. The training-loss trace IS a validation curve. There is no train/test
distinction left to construct, no `track_valid` machinery needed for this task, and a plateau is a
genuine task-floor plateau rather than memorisation saturating. The convergence question that was
close to unanswerable on the old data is answerable on the new.

---

