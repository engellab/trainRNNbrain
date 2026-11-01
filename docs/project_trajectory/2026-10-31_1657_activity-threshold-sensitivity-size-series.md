## 2026-10-31 16:57 — Activity threshold: halving or doubling it moves the active count by 3-28%, the size exponent by at most 0.05

Work package `methods-core` (dead-relu-paper issue #10), for the Methods paragraph on the activity
criterion. Branch `wp/methods-core`.

**Why.** The Methods text said the scale-free threshold (active if p ≥ 0.05·q95(p)) "falls in a
sparsely populated valley" and backed it with an Otsu cross-check measured on pre-standard
(trainable-bias) networks. The guide's health check, q50/q95 < 0.05, was the other support. Neither
had been measured on the standard networks of the size series.

**Pass/fail set before running.**
- Health check: fails in a cell if q50/q95 ≥ 0.05 at the floor probe.
- Valley: the threshold sits in a sparse valley if the active count changes by less than 10% when
  the factor 0.05 is halved (0.025) or doubled (0.10).
- Consistency: at 0.05, the per-size-mean log-log fit must reproduce the board's C2 exponents
  (3BFF 0.47, CDDM 0.44, DMTS 0.58) within ±0.03, or the reconstruction is wrong.

**How.** The script below reads each run's participation trace at the stored probe nearest its
1.05× noisy-loss floor (`common.read_run`). Cells: the Figure 1/2 size series (`fig_paper_F1.SCALING`):
3BFF `NBitFlipFlop_ff_revive_c200000` arm=none N = 500, 1000, 2000 and
`NBitFlipFlop_paper_grid_c200000` control N = 4000 (Della job 16554497, `~/trainRNNbrain_c1ana` at
ed11dfa); CDDM `CDDM_paper_grid` control N = 500-4000 and DMTS `DMTS_paper_grid_d5` control
N = 500-2000 (Mac, main 484e0a4). Runs without a floor read are skipped (DMTS: one at N = 1000, the
unlearned one at N = 2000).

**Result.**

| | 3BFF | CDDM | DMTS |
|---|---|---|---|
| runs read | 16 | 12 | 7 |
| q50/q95 at the floor, largest | 0.038 (N = 500) | 0.099 (N = 500), 0.022 (N = 1000) | 0.015 |
| count at 0.025 vs 0.05 | +4% to +25% | +9% to +15% | +3% to +15% |
| count at 0.10 vs 0.05 | −9% to −19% | −12% to −28% | −11% to −19% |
| exponent at 0.025 / 0.05 / 0.10 | 0.43 / 0.48 / 0.49 | 0.45 / 0.44 / 0.48 | 0.57 / 0.58 / 0.62 |

Example, CDDM N = 1000, three runs: 466-492 active at 0.025, 420-437 at 0.05, 352-359 at 0.10.

**Verdicts.**
- Consistency passes (0.48, 0.44, 0.58 against 0.47, 0.44, 0.58).
- Health check fails on CDDM at N = 500 (q50/q95 0.08-0.10). The failure is by construction: more
  than half the units pass the rule exactly when q50/q95 ≥ 0.05, and 291-302 of 500 units are active
  there. The check detects a threshold that has fallen through a noise floor (entry
  2026-10-01_1350) only in networks where most units are inactive; it cannot be used at N = 500 on
  CDDM.
- Valley test fails: the count moves by 3-28% for a factor of two either way, so the valley between
  the two participation modes holds units. The active count is criterion-dependent at the
  10-25% level. The size exponent moves by at most 0.05 per task across the three factors, so the
  claim C2 (sublinear growth, exponents 0.44-0.60) does not depend on the factor.

**Consequences.**
- Methods (`M04_activity.tex`) states the sensitivity instead of "not a judgement call"; the Otsu
  and hard-threshold criteria and their pre-standard numbers are cut.
- Figure 1b's caption and Results 1 should not call the valley empty (issue opened on
  ptolmachev/dead-relu-paper for the fig1-inactive and fig2-size-time packages).

**Script** (run as `ANA_DIR=<checkout>/trainRNNbrain/experiments_and_analysis python health_q50_q95.py <cell> ...`;
exponents from the printed counts by least squares on log N against log of the per-size mean, the
fit `fig_paper_F1.py` uses):

```python
"""Health check of the scale-free activity rule on the size-series runs: q50(p) / q95(p).

More than half of a network's units count as active exactly when q50/q95 >= 0.05, so the rule is
only meaningful where that ratio sits well below 0.05. For every run under the given cell folders
this prints the ratio at the stored participation probe nearest the run's own loss-floor read-out
(common.read_run, 1.05x) when the trace covers it, the active count there under 0.025, 0.05
and 0.10 x q95 (the threshold's sensitivity), and the largest ratio over every stored probe of
the trace from that probe (or from the trace's start, when it does not cover the floor) to the end.

Usage: python health_q50_q95.py <cell folder> [<cell folder> ...]
"""
import os
import sys
import glob
import pickle
import numpy as np

sys.path.insert(0, os.environ.get("ANA_DIR", "."))
import common  # noqa: E402  (trainRNNbrain/experiments_and_analysis on the path)


def ratios(run_dir):
    """q50/q95 of participation at the floor probe and its maximum from there to the end.

    Args:
        run_dir: str, one trained-run folder with *ParticipationTrace.pkl and *TrainLosses.json.
    Returns:
        (r_floor, r_max_after, t_star, covered, counts): floats (nan where unavailable), a bool,
        and the active count at the floor probe under 0.025, 0.05 and 0.10 x q95 (None if uncovered).
    """
    tp = glob.glob(os.path.join(run_dir, "*ParticipationTrace.pkl"))
    if not tp:
        return np.nan, np.nan, np.nan, False, None
    with open(tp[0], "rb") as fh:
        tr = pickle.load(fh)
    P = np.asarray(tr["participation"], float)
    it = np.asarray(tr["participation_iters"], float)
    q95 = np.quantile(P, 0.95, axis=1)
    r = np.quantile(P, 0.5, axis=1) / q95
    counts = {f: (P >= f * q95[:, None]).sum(axis=1) for f in (0.025, 0.05, 0.10)}
    rd = common.read_run(run_dir)
    t = rd["t_star"]
    if rd["reason"] is not None or not np.isfinite(t):
        return np.nan, float(np.max(r[len(r) // 2:])), np.nan, False, None
    covered = bool(it[0] <= t <= it[-1])
    j = int(np.argmin(np.abs(it - t)))
    c = tuple(int(counts[f][j]) for f in (0.025, 0.05, 0.10)) if covered else None
    return (float(r[j]) if covered else np.nan), float(np.max(r[j:] if covered else r)), t, covered, c


def main(cells):
    """Print one row per run and the maximum ratio per cell.

    Args:
        cells: list of str, cell folders, each holding run folders.
    Returns:
        None.
    """
    for cell in cells:
        rows = []
        for run in sorted(glob.glob(os.path.join(cell, "*", ""))):
            rf, rm, t, cov, c = ratios(run)
            rows.append((rf, rm))
            print(f"{os.path.basename(cell.rstrip('/'))}  {os.path.basename(run.rstrip('/'))[:30]}  "
                  f"t*={t:9.0f}  covered={cov}  q50/q95 at floor={rf:.4f}  max after={rm:.4f}  "
                  f"active at 0.025/0.05/0.10 x q95: {c}")
        rows = np.array(rows)
        print(f"CELL {cell}: max at floor {np.nanmax(rows[:, 0]) if np.isfinite(rows[:, 0]).any() else np.nan:.4f}, "
              f"max after {np.nanmax(rows[:, 1]):.4f}, runs {len(rows)}")


if __name__ == "__main__":
    main(sys.argv[1:])
```
