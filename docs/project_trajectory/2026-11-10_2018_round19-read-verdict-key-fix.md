## 2026-11-10 20:18 — round19_read.py quoted the wrong update-2,000 verdict for four Maze technique arms; fixed, round19_results.json regenerated

**The error.** `round19_read.read_settings` quotes each technique arm's update-2,000 verdict from
`data/read_maze_arms.json`. It looked the arm up by its launcher name and, when the name was missing, fell back to
the class key from `fig_ed_maze_interventions.arm_of`. A class can hold several arms, and `read_maze_arms.json`
predates the round 18 parity arms. Four arms therefore carried another arm's verdict string in
`data/round19/round19_results.json`:

| Arm | Quoted (wrong) | From the arm |
|---|---|---|
| dropout0.05_beta4 | "fails: count not above every control's (449) in seeds 1, 3; success cost -0.104 beyond 0.05; seeds 1, 2, 3 ending below 0.9" | dropout0.2_beta4 |
| sigmaw0.1 | "fails: seed 3 ending below 0.9" | sigmaw1.0 |
| sigmaw0.3 | "fails: seed 3 ending below 0.9" | sigmaw1.0 |
| rescale_tgt4 | "no verdict (stopped)" | rescale |

The per-seed numbers in the file were right, and so are the verdicts the paper uses for these arms. Those came
from other readers (round 18 parity: entry `2026-11-08_0859`; fz reruns: `round19_read_fz.py`, which recomputes
its own 2,000 verdicts). The entry that found the error is `2026-11-10_2014_round19-dropout-rescale-fz-results-2000.md`.

**The fix** (`trainRNNbrain/experiments_and_analysis/round19_read.py`). The verdicts are keyed on the entry's
`folder` field in `read_maze_arms.json`, which is the launcher's arm name. The class-key fallback is gone. An
assert checks that no two entries share a folder. An arm without an entry gets `None`: the four parity arms above.

**Regenerated** `data/round19/round19_results.json` with `round19_read.py` (default root
`data/trained_RNNs_rl_maze`), and compared the result with the committed file:
- *settings:* the same 40 arms. The only changes are `verdict_2000` of the four arms above, now `None`. No other
  verdict string, `verdict_4000`, label or per-seed number changed, and the controls and the 4,000 reference are
  identical.
- *suite:* the file now includes the data added since it was last written: the N = 4,000 networks' later stages
  (budgets 6,000-8,000; entry `2026-11-09_1341`) and the four N = 100 networks (entry `2026-11-10_0826`). Those
  numbers are already in their entries and on the status board.
- *pilot:* unchanged.

`maze_arms_r19.py`'s cross-check against the regenerated file still reports "every read matches" (it compares
per-seed numbers, not verdict strings).
