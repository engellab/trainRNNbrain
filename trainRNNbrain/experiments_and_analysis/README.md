# Analysis scripts — conventions

Every script here answers one question about trained networks and writes one or more figures to
`img/internal_figures/`. Two modules are importable infrastructure; everything else is a script.

| module | holds | rule |
|---|---|---|
| [`common.py`](common.py) | data loading, fitting, participation, drift primitives | no matplotlib, no side effects at import |
| [`plotstyle.py`](plotstyle.py) | colours, rcParams, band/contour/legend/save helpers | everything about how figures LOOK |

**One variable, one visual channel, across every figure**: N is colour from a fixed map (so N=1000 is
the same red everywhere), k is viridis (dark = low k), the read-out criterion is line style or a
panel row and *never* colour, and seeds are error bands rather than a channel of their own. Mixing
these up is how a reader concludes something about N from a figure that varied k.

## The rules

1. **Import shared primitives from `common`, never from a sibling script.** Before consolidation, 14
   files did `from plot_drift_curves import IMG_DIR` — executing a 625-line analysis module to obtain
   one directory string — and the same primitives were re-typed across files (`logbin` in 3,
   `aicc` in 3, `r2_from_dir` in 3, `active_count` in 2, the budget dedupe in 4). Divergent copies of
   an analysis primitive are how a project reports two different numbers for one quantity.
2. **Keep your own domain loader** when the folder layout is specific to your sweep. Only genuinely
   shared things belong in `common`.
3. **Guard execution** behind `if __name__ == "__main__": main()`. No work at import time.
4. **Module docstring states the question, the method, and the output path.** Where a choice could
   have gone another way (fit range, read-out iteration, criterion), say why in the docstring — those
   are the decisions that get re-litigated months later.
5. **Every function gets a docstring** with inputs (types, shapes, units) and return value.
6. **Seed every bootstrap.** Use `np.random.default_rng(seed)`, never the global `np.random`. An
   unseeded bootstrap here was returning `b = [0.353, 0.430]` and `[0.349, 0.431]` on consecutive
   runs of identical code, so its published CIs could not be reproduced.

## Standing analysis rules

These are not style; each one exists because ignoring it produced a wrong result in this project.

- **Never compare `TrainLosses.json` across penalty conditions.** It is the optimiser objective,
  task loss + `λ·penalty`, evaluated with noise ON. Use the noise-free `loss_clean_train` metric
  inside the participation trace. This has inverted a conclusion twice.
- **Always report both silence criteria** (`hard` and `scale-free`, both in `common.CRITERIA`). They
  disagree, and a 60.5% vs 86.0% split at N=2000 was read as a penalty "rescue" until both were
  shown side by side.
- ⚠️ **The absolute threshold is NOT task-portable.** `SILENT_HARD = 1e-6` was calibrated on CDDM,
  whose silent mode sits at exactly 0. On the n-bit flip-flop the silent mode sits at ~1e-3, so 1e-6
  falls below BOTH modes, reports ~0% silence, and makes `M = N` exactly (b = 1.00, nothing measured).
  Derive the threshold per task with `common.otsu_threshold` — it gives **4e-2 for the flip-flop** —
  and pass it to `active_count` as a float. `flipflop_hard_threshold.py` does the derivation and its
  three adoption checks. Every conclusion (b sublinear; c ≈ 0.18 at fixed compute, ≈ 0 at matched
  performance) is unchanged under the calibrated threshold, so the earlier absolute-vs-relative
  conflict was a mis-calibrated constant, not a disagreement about the science.
- **Never pool a cell across iteration budgets.** Use `common.keep_longest_budget`. Pooling runs
  trained 3× apart as extra seeds of one condition invalidated an early CDDM penalty table.
- **Match fit ranges across cells you intend to compare**, or state and measure what the mismatch
  costs. Mismatched ranges manufactured a spurious trend three separate times.
- **A quantity that has converged may be read at any sufficient budget; one that has not must be read
  at a matched iteration.** The loss floor converges by ~25k; the active-unit count does not (still
  moving +5 pp per doubling at 490k).

## `common.py`

Run `python common.py` for its self-check.

| group | names |
|---|---|
| paths | `IMG_DIR`, `DATA_DIR`, `HERE` |
| participation | `participation`, `active_count`, `otsu_threshold`, `hhi`, `CRITERIA`, `SILENT_HARD`, `SILENT_REL` |
| loss curves | `smooth_loss`, `T_at_loss`, `stable_crossing` |
| fitting | `logbin`, `stretched`, `excess_time`, `aicc` |
| sweep loading | `keep_longest_budget`, `load_traces`, `load_losses`, `r2_from_dir` |
| weight drift | `series`, `drift_alpha`, `diffusive_onset`, `LAGS` |

⚠️ `stretched`'s `A` and `tau` are **not individually identified** once `tau` falls below the fit's
start iteration — the form degenerates toward a power law and `A` becomes an unconstrained
extrapolation to t=0. Compare fits through `L_inf` and through `excess_time`, never `A` or `tau`.

⚠️ `diffusive_onset` reports the FIRST SUSTAINED crossing, not the last. Requiring alpha to stay
below threshold until the end of the run makes one late noisy excursion reset the answer to near the
final iteration: sibling seeds of a single cell returned 40k, 44k and 458k, and high-k cells returned
~470k against a 500k budget, i.e. the criterion was reporting the BUDGET rather than the dynamics.
With `persist` consecutive probes required, within-cell seed CV drops to 0.13.

⚠️ `stable_crossing` takes `window` and `base` explicitly. Two call sites previously relied on
different local defaults (`window=201, base=0` vs `window=2001, base=1`); pass both rather than
inheriting whatever the shared default happens to be.

## Comparing networks: always more than one read-out criterion

`M(N, k)` has no value until you say WHEN the network is read, and the choice can carry the result.
[`flipflop_figures.py`](flipflop_figures.py) resolves four and reports all of them:

| criterion | what it matches | mean T | seed CV |
|---|---|---|---|
| `iter` | training budget | 150 000 | 0.00 |
| `loss` | absolute clean loss `L*` | 27 700 | 0.06 |
| `excess` | fraction of the way to each cell's OWN floor | 39 600 | 0.10 |
| `drift` | dynamical state: weight motion stops being directed | 31 000 | 0.13 |

Report the **seed CV of the read-out time** alongside any new criterion. A rule whose T scatters
across sibling seeds of one cell is measuring noise, and that is exactly how the first `drift`
definition failed its first contact with data.

⚠️ Never average a quantity over "whatever sizes this k happens to have". k=1 and k=8 currently exist
at N=500 only while k=2..7 have all three sizes, and because M/N falls with N, such a mean puts the
incomplete k values artificially high and manufactures a U shape that is about coverage, not
complexity. Use one size and say which, or restrict to complete cells.

## Legacy scripts — predate this project line, untouched

These are from the original repository (last touched 2024-09 to 2026-02), are unrelated to the
silent-ReLU work, and are the only files here with no `__main__` guard. They are left in place, not
maintained. `RE_AngleAddition.py` has a pre-existing `undefined name 'theta_retina'` bug, and
`R2&ActiveUnitsAnalysis.py` cannot be imported at all because of the `&` in its filename.

```
R2&ActiveUnitsAnalysis.py   RE_AngleAddition.py       ReferenceFrame_tuning_space.py
analysis_GoNoGo_taks.py     analyzing_unsuppressed_info.py   clustering.py
creating_RNN_dataframe.py   lmbdZ_effect_inactive_neurons.py plotting_neuron_space.py
test.py                     rerun_RNN_analysis.py
```
