#!/usr/bin/env python3
"""Why the remedy top-ups (RMtop200) excluded as "never stays under the bar" have no floor read.

For every run that read_remedies.py excluded with that reason, the read-out is recomputed
(common.read_run, unchanged) and the smoothed noisy loss is compared with its 1.05x bar:
  late_rise        the median of the smoothed curve over the last 10% of the run, over the floor;
  t_first_under    the first iteration (after the fit start) the curve is at or below the bar;
  share_above      the share of iterations after t_first_under at which the curve is back above it;
  t_last_above     the last iteration the curve is above the bar (nan if never after t_first_under);
  t_settle         the first iteration of the final under-bar stretch (nan if the curve ends above);
  t_d10, t_d20     the sustained crossings at 1.10x and 1.20x (nan where none holds);
  n_bursts         bursts found by the read-out;
and each run gets one cause:
  "ends above the bar"   late_rise > 1.05: the curve finishes more than 5% above its own fitted floor;
  "settles too late"     the curve stays under the bar from t_settle on, but t_settle > end / 1.2,
                         so the 1.2t hold runs past the end of training;
  "crosses back"         anything else (not expected: a run that settles by end / 1.2 has a read).
Where a looser margin holds, the active count at its probe is given too (active_d10, active_d20).

Usage (repository root, CPU node):
  python trainRNNbrain/experiments_and_analysis/rm_exclusion_diag.py RUNS_DIR OUT_JSON
RUNS_DIR holds read_remedies.py's runs_*.json.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import FLOOR_DELTA, FIT_START, HOLD_FRAC, load_run, read_run

REASON = "never stays under the bar"


def diagnose(run_dir):
    """The loss-versus-bar diagnostics of one excluded run (see the module docstring).

    Args:
        run_dir: the run folder.
    Returns:
        JSON-ready dict of the diagnostics and the cause, or a dict with only `error` when the run
        cannot be loaded or the read-out stops before the crossing test.
    """
    r = load_run(run_dir)
    if r is None:
        return dict(error="no trace")
    rd = read_run(run_dir, run=r)
    if rd.get("floor") is None or rd.get("smoothed") is None:
        return dict(error=f"read_run stopped: {rd['reason']}")
    it = np.asarray(rd["it_loss"], float)
    it = it[np.isfinite(np.asarray(rd["loss"], float))]   # noisy_floor smooths the finite samples only
    sm = np.asarray(rd["smoothed"], float)
    fit_from = rd["burst_from"] if np.isfinite(rd["burst_from"]) else FIT_START
    bar = (1.0 + FLOOR_DELTA) * rd["floor"]
    after = it >= fit_from
    under = after & (sm <= bar)
    out = dict(floor=float(rd["floor"]), late_rise=float(rd["late_rise"]), it_end=float(it[-1]),
               n_bursts=len(rd["bursts"]), reason_now=rd["reason"],
               t_d10=float(rd["t_stars"].get(0.1, np.nan)), t_d20=float(rd["t_stars"].get(0.2, np.nan)))
    if not under.any():
        out.update(t_first_under=float("nan"), share_above=float("nan"), t_last_above=float("nan"))
    else:
        t1 = float(it[np.flatnonzero(under)[0]])
        post = it >= t1
        above_post = post & (sm > bar)
        last_above = float(it[np.flatnonzero(above_post)[-1]]) if above_post.any() else float("nan")
        out.update(t_first_under=t1, share_above=float(above_post.sum() / post.sum()),
                   t_last_above=last_above,
                   t_settle=t1 if np.isnan(last_above) else float(it[np.flatnonzero(it > last_above)[0]])
                   if last_above < it[-1] else float("nan"))
    if out["late_rise"] > 1.0 + FLOOR_DELTA or sm[-1] > bar:
        out["cause"] = "ends above the bar"
    elif out.get("t_settle", np.nan) > it[-1] / (1.0 + HOLD_FRAC):
        out["cause"] = "settles too late"
    else:
        out["cause"] = "crosses back"
    it_a, act = np.asarray(r["it_act"], float), np.asarray(r["active"], float)
    for key in ("t_d10", "t_d20"):
        t = out[key]
        out["active" + key[1:]] = (int(act[int(np.argmin(np.abs(it_a - t)))])
                                   if np.isfinite(t) and it_a[0] <= t <= it_a[-1] else None)
    return out


def main():
    """Diagnose every run excluded as REASON in RUNS_DIR/runs_*.json and write OUT_JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs_dir")
    ap.add_argument("out_json")
    a = ap.parse_args()
    runs = [r for f in sorted(glob.glob(os.path.join(a.runs_dir, "runs_*.json"))) for r in json.load(open(f))]
    todo = [r for r in runs if r["reason"] == REASON]
    print(f"{len(todo)} of {len(runs)} runs excluded as '{REASON}'", flush=True)
    out = []
    for r in sorted(todo, key=lambda x: (x["cell"], x["name"])):
        d = diagnose(r["run_dir"])
        d.update(cell=r["cell"], name=r["name"], run_dir=r["run_dir"])
        out.append(d)
        print(f"  {d['cell'][:72]:72s} {d['name'][:10]} "
              + (d["error"] if "error" in d else
                 f"{d['cause']:18s} end/floor {d['late_rise']:.3f} first-under {d['t_first_under']:8.0f} "
                 f"above-after {d['share_above']:.2f} last-above {d['t_last_above']:8.0f} "
                 f"t1.10 {d['t_d10']:8.0f} t1.20 {d['t_d20']:8.0f} act1.10 {d['active_d10']} act1.20 {d['active_d20']}"),
              flush=True)
    json.dump(out, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
