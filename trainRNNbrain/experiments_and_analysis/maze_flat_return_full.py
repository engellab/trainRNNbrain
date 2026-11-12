"""The Maze flat-return test on the full training records (rule: notebook entry
2026-11-11_*_maze-flat-return-late-window-rule.md, written and committed before this script ran).

For every learned default network of the Maze size series it measures, inside a window of the training
record, the change of the training return (flat or not) and the slope of the number of active units
(decreasing or not), and applies the two-thirds rule of G8 (round18_maze.py) over the flat networks.

Readings:
  B (primary)  window = the last quarter of the record;
  A            window = from the learned point to the end (G8 unchanged);
  sensitivity  window = the last half of the record; flat bars of 2% and 5% of the rise (reading B).

Usage (from trainRNNbrain/experiments_and_analysis):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python maze_flat_return_full.py
Writes data/maze_flat_return_full.json and prints one table per reading.
"""
import json
import os

import numpy as np

from round18_maze import REPO, learned_default_runs, three_way

FLAT_BAR = 0.03          # flat if |D| < FLAT_BAR x rise, per 1,000 updates (G8)
DECREASE_BAR = 5.0       # decreasing if at least this many active units are lost per 1,000 updates (G8)
BLOCK = 100              # updates in the first and the last block of a window (G8)
MIN_WINDOW = 300         # reading A: shortest window tested (G8)
MIN_FLAT = 6             # fewer flat networks than this: no verdict (rule entry)
OUT = os.path.join(REPO, "data", "maze_flat_return_full.json")


def window_start(reading, learned, end):
    """First update of the test window.

    Args:
        reading: "A" (from the learned point), "B" (last quarter) or "half" (last half).
        learned: float, the learned update; end: float, the last update of the record.
    Returns:
        float, the window's first update.
    """
    return {"A": learned, "B": 0.75 * end, "half": 0.5 * end}[reading]


def measure(row, run, reading, flat_bar=FLAT_BAR):
    """The return change and the active-unit slope of one network inside one window.

    Args:
        row: dict from fig_paper_F2_size.maze_series (N, seed, t_star, it_act, active).
        run: dict from common.maze_load_runs (progress: one dict per PPO update).
        reading: "A", "B" or "half" (see window_start).
        flat_bar: float, the flat threshold as a fraction of the rise, per 1,000 updates.
    Returns:
        dict with N, seed, learned, end, start, rise, tested (bool), and when tested: D_per_1000,
        D_over_rise, flat (bool), lost_per_1000, decreasing (bool), active_first, active_last.
    """
    upd = np.array([p["update"] for p in run["progress"]], float)
    ret = np.array([p["discounted_return"] for p in run["progress"]], float)
    learned, end = float(row["t_star"]), float(upd[-1])
    start = window_start(reading, learned, end)
    rise = float(np.median(ret[(upd > learned - BLOCK) & (upd <= learned)]) - np.median(ret[upd <= BLOCK]))
    out = dict(N=int(row["N"]), seed=int(row["seed"]), learned=learned, end=end, start=start, rise=rise)
    tested = learned <= start and end - start >= MIN_WINDOW
    out["tested"] = bool(tested)
    if not tested:
        return out
    first = (upd >= start) & (upd < start + BLOCK)
    last = upd > end - BLOCK
    dist = (np.mean(upd[last]) - np.mean(upd[first])) / 1000.0
    D = float((np.median(ret[last]) - np.median(ret[first])) / dist)
    probe_upd, probe_act = np.asarray(row["it_act"], float), np.asarray(row["active"], float)
    m = probe_upd >= start
    slope = float(np.polyfit(probe_upd[m] / 1000.0, probe_act[m], 1)[0])
    out.update(D_per_1000=D, D_over_rise=D / rise, flat=bool(abs(D) < flat_bar * rise),
               lost_per_1000=-slope, decreasing=bool(-slope >= DECREASE_BAR),
               active_first=float(probe_act[m][0]), active_last=float(probe_act[m][-1]), n_probes=int(m.sum()))
    return out


def verdict(rows):
    """The rule over the flat networks of one reading.

    Args:
        rows: list of dicts from measure().
    Returns:
        (n_tested, n_flat, n_decreasing_among_flat, verdict string).
    """
    tested = [w for w in rows if w["tested"]]
    flat = [w for w in tested if w["flat"]]
    n_dec = sum(w["decreasing"] for w in flat)
    if len(flat) < MIN_FLAT:
        return len(tested), len(flat), n_dec, "cannot be tested (fewer than 6 flat networks)"
    v = three_way([w["decreasing"] for w in flat], "decreasing while the return is flat", "not decreasing")
    return len(tested), len(flat), n_dec, v


def show(tag, rows):
    """Print the per-network table and the verdict of one reading; returns the verdict tuple."""
    print(f"\n[{tag}] {'N':>5} {'seed':>4} {'learned':>7} {'start':>6} {'end':>5} {'rise':>7} {'D/rise %':>9} "
          f"{'flat':>5} {'active':>11} {'lost/1000':>9} {'decr':>5}")
    for w in rows:
        if not w["tested"]:
            print(f"  {w['N']:>9} {w['seed']:>4} {w['learned']:>7.0f} {w['start']:>6.0f} {w['end']:>5.0f} "
                  f"{w['rise']:>7.3f}   not tested (learned after the window start, or window under 300 updates)")
            continue
        print(f"  {w['N']:>9} {w['seed']:>4} {w['learned']:>7.0f} {w['start']:>6.0f} {w['end']:>5.0f} {w['rise']:>7.3f} "
              f"{100 * w['D_over_rise']:>+9.2f} {str(w['flat']):>5} {w['active_first']:>5.0f}->{w['active_last']:<5.0f} "
              f"{w['lost_per_1000']:>9.1f} {str(w['decreasing']):>5}")
    v = verdict(rows)
    print(f"  [{tag}] tested {v[0]} of {len(rows)}; flat {v[1]}; decreasing among flat {v[2]} -> {v[3]}")
    return v


def main():
    """Run every reading, print the tables and write the JSON; returns None."""
    pairs = learned_default_runs()
    res = {}
    for tag, reading, bar in (("B last quarter (primary)", "B", FLAT_BAR), ("A from the learned point", "A", FLAT_BAR),
                              ("sensitivity: last half", "half", FLAT_BAR),
                              ("sensitivity: B, flat bar 2%", "B", 0.02), ("sensitivity: B, flat bar 5%", "B", 0.05)):
        rows = [measure(row, run, reading, bar) for row, run in pairs]
        v = show(tag, rows)
        res[tag] = dict(rows=rows, n_tested=v[0], n_flat=v[1], n_decreasing=v[2], verdict=v[3])
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
