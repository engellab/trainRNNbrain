## 2026-08-25 (night, 2) — npz saver bug FIXED; flip-flop absolute silence threshold recalibrated

### 1. Saver bug fixed

`run_experiment.py` built its .npz payload with `np.asarray(v)`. On an **OmegaConf DictConfig that
ITERATES to its keys**, so `activation_args = {"name": "relu", "slope": 1.0}` was written as
`array(['name','slope'])` with the values silently discarded, and any net reconstructed from a .npz
alone raised an opaque IndexError inside `RNN_numpy.configure_activation_`. New helper
`run_experiment.storable_` converts OmegaConf containers to plain Python and stores mappings as 0-d
object arrays, which round-trip through `np.load(..., allow_pickle=True)[k].item()`. Round-trip
verified for DictConfig, dict, list, ndarray and scalar.

⚠️ **Every .npz written before this fix still has the defect.** Analysis code must restore
`activation_args` from the saved config (`silence_is_taskset.load_params` does).

### 2. A task-calibrated absolute threshold for the flip-flop: p < 4e-2

`1e-6` was calibrated on CDDM and sits below BOTH of the flip-flop's modes, which is why it reported
~0% silence there. New script
[`flipflop_hard_threshold.py`](../trainRNNbrain/experiments_and_analysis/flipflop_hard_threshold.py)
derives the threshold from the data by Otsu's method on log10(p) (`common.otsu_threshold`, added with
a self-check), with the adoption criteria fixed in advance.

| pre-registered check | result |
|---|---|
| spread across cells < 1 decade | **PASS** — 0.50 decades (10–90 pct: 2.3e-2 .. 7.4e-2) |
| agrees with scale-free, median \|ΔM\|/N < 0.05 | **PASS** — 0.030 |
| must differ from 1e-6 | **PASS** — 41 000× larger |

**Adopted: `p < 4e-2` for the n-bit flip-flop** (against `1e-6` for CDDM). Silent fractions under it
track the scale-free rule closely (e.g. k=3 N=2000: 0.714 vs 0.732) where the old rule gave 0.002.

⚠️ The threshold drifts systematically with both axes — 5.5e-2 at N=500 → 3.4e-2 at N=2000, and
6.8e-2 at k=1 → 2.7e-2 at k=8, a 2–2.5× range inside the 0.5-decade spread. A single constant is a
simplification, defensible only because the valley is nearly empty (below).

**A speculation of mine was wrong and is corrected here:** I guessed Otsu would recover the
scale-free rule's own constant. It does not — Otsu gives 4.1e-2 against `0.05·q95` = 1.13e-1, a factor
2.7 apart, and the two are essentially UNCORRELATED across nets (r = 0.13). They nevertheless
classify the same units (|ΔM|/N = 0.030) because the valley between the modes is nearly empty, so any
threshold inside it returns the same answer. That insensitivity is the strongest evidence yet that
the bimodal split is a real structural feature rather than an artifact of either rule.

### 3. The law is unchanged under the recalibrated threshold

| criterion | b (p<4e-2) | b (scale-free) | c (p<4e-2) | c (scale-free) |
|---|---|---|---|---|
| endpoint | 0.34 | 0.39 | 0.118 | 0.137 |
| fixed compute | 0.36 | 0.41 | **0.178** | **0.179** |
| matched performance | 0.41 | 0.43 | −0.017 | −0.023 |
| matched perf., k-fair | 0.39 | 0.41 | −0.034 | −0.038 |
| matched dynamics | 0.44 | 0.46 | −0.003 | 0.001 |

**Every conclusion survives**: b sublinear (0.34–0.44) at every criterion, c ≈ 0.18 at fixed compute
and ≈ 0 at matched performance. So the earlier apparent conflict between the absolute and relative
rules was entirely a mis-calibrated constant, not a disagreement about the science.

### fig5 regenerated under the calibrated threshold

`flipflop_figures.py` now emits three versions of the law figure, one per silence rule:
`flipflop_fig5_law.png` (scale-free), **`flipflop_fig5_law_abs.png` (absolute, p >= 4e-2,
task-calibrated)**, and `flipflop_fig5_law_hard.png` (absolute, p >= 1e-6, kept and explicitly
labelled DEGENERATE so the failure mode stays visible rather than being quietly deleted).
`common.active_count` now accepts a float as an explicit absolute threshold.

The calibrated absolute figure reproduces the scale-free one panel for panel: `c = 0.178
[0.157, 0.201]` at fixed compute against the scale-free `0.179 [0.152, 0.208]`, and `c` straddling
zero at all three matched criteria. b runs 0.34–0.44 against the scale-free 0.39–0.46 — slightly
lower throughout, because a fixed absolute cut counts marginally fewer units active than a rule that
rescales with each network, but the ordering and every verdict are identical.

---

