#!/usr/bin/env python3
"""Read B3 (rate penalty x synaptic noise x rescaling, N = 1000) and B4 at N = 4000 under their rules.

Both rules were written before any result: docs/project_trajectory/2026-10-15_1342_open-rules-and-
commands-moved-from-agent-notes.md, from the B3 plan of 13 Oct (`git show eec4a16^:docs/notes/
agent-bplan.md`). How they are applied here was fixed before the read in ptolmachev/dead-relu-paper#2.

B3. Each of the eight cells is read run by run at its own noisy-loss floor (common.read_run; a
penalised run's loss includes the penalty) and at a fixed 75,000 iterations. Per run: active units
(participation probe nearest the read point), held-out R^2 (heldout_r2, R2_DRAWS draws, on the kept
snapshot at or after the read point) and the covariance participation ratio over active units
(run_dims.measure, noise-free, on the same snapshot). Each pair and the triple is then compared with
its best stable single component on the logit of the active fraction (b3_verdict). The floor read
binds; the 75k read is reported with the same bars.

B4 at N = 4000. read_b2_b4.py reads every N present but applies the verdict at N = 1000 only; given
its JSON output, b4_verdict is applied to the N = 4000 cells here, unchanged.

Usage:  python trainRNNbrain/experiments_and_analysis/read_b3_b4_4000.py OUT_DIR [--b4 READ_B2_B4_JSON]
        (repository root, CPU node; writes OUT_DIR/read_b3.json and, with --b4,
        OUT_DIR/b4_verdict_4000.json, and prints the tables)
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H
import run_dims
from common import DATA_DIR, read_run, runs_with_loss, snapshot_at
from read_b2_b4 import b4_verdict

N = 1000
AT_FIXED = 75000          # the same-size read-out iteration (16 Oct rule)
GRID = f"{DATA_DIR}/NBitFlipFlop_remedy_grid/EqType=h_N={N}_arm="
B3_CELLS = {              # cell name -> (folder, set of single remedies it contains)
    "none": (f"{DATA_DIR}/NBitFlipFlop_c1_snapshots_c200000/EqType=h_N={N}_arm=control", frozenset()),
    "rate": (GRID + "frm0.03", frozenset({"rate"})),
    "noise": (GRID + "sw1.0", frozenset({"noise"})),
    "rescale": (GRID + "rsc10", frozenset({"rescale"})),
    "rate+noise": (GRID + "frm0.03_sw1.0", frozenset({"rate", "noise"})),
    "rate+rescale": (GRID + "frm0.03_rsc10", frozenset({"rate", "rescale"})),
    "noise+rescale": (GRID + "sw1.0_rsc10", frozenset({"noise", "rescale"})),
    "rate+noise+rescale": (GRID + "frm0.03_sw1.0_rsc10", frozenset({"rate", "noise", "rescale"})),
}
SD_BAR = 3.0              # combination must beat / sit within this many SDs of its best component
R2_COST = 0.01            # allowed R^2 cost against that component
PR_REPEATS = 8            # run_dims.measure draws an unseeded batch per call (PR of one network varies
                          # by ~0.5-1 between calls, 4-call means by ~0.4, in the smoke test), so PR is
                          # averaged over calls


def logit_active(a, n=N):
    """Logit of the active fraction with a half-count continuity correction.

    Args:
        a: float or array, active-unit counts out of n.
        n: int, network size.
    Returns:
        float or array, log(q / (1 - q)) with q = (a + 0.5) / (n + 1).
    """
    q = (np.asarray(a, float) + 0.5) / (n + 1)
    return np.log(q / (1 - q))


def at_snapshot(snap, cache):
    """Held-out R^2 and covariance participation ratio of one kept snapshot.

    Args:
        snap: str or None, a snapshot folder (LastParams npz + config yaml).
        cache: dict, an open held-out R^2 cache.
    Returns:
        (r2, pr, n_active_pr): floats, nan when snap is None; pr is the mean over PR_REPEATS calls of
        run_dims.measure (each on a fresh task batch), n_active_pr the mean of its own active counts.
    """
    if snap is None:
        return float("nan"), float("nan"), float("nan")
    m = np.array([run_dims.measure(snap) for _ in range(PR_REPEATS)], float)
    return float(H.heldout_r2(snap, draws=H.R2_DRAWS, cache=cache)), float(m[:, 0].mean()), float(m[:, 1].mean())


def b3_run(run, cache):
    """One B3 run read at its floor and at AT_FIXED iterations.

    Args:
        run: dict, one runs_with_loss entry.
        cache: dict, an open held-out R^2 cache.
    Returns:
        dict with name, reason (None when the floor read exists), and two sub-dicts `floor` and `fixed`,
        each with it (the read iteration), active (count of N), r2, pr, n_active_pr, snapshot; `floor`
        also has active_110 and active_120 (counts at the 1.10x and 1.20x crossings, None if absent).
    """
    rd = read_run(run["run_dir"], run=run)
    it_a, act = np.asarray(run["it_act"], float), np.asarray(run["active"], float)
    j = int(np.argmin(np.abs(it_a - AT_FIXED)))
    snap = snapshot_at(run["run_dir"], AT_FIXED)
    r2, pr, npr = at_snapshot(snap, cache)
    fixed = dict(it=float(it_a[j]), active=int(act[j]), r2=r2, pr=pr, n_active_pr=npr, snapshot=snap)
    rec = dict(name=rd["name"], reason=rd["reason"], fixed=fixed, floor=None)
    if rd["reason"] is None and not rd["covered"]:
        rec["reason"] = "participation trace does not cover the crossing"
    if rec["reason"] is not None:
        return rec
    r2, pr, npr = at_snapshot(rd["snapshot"], cache)
    margins = {}
    for d, key in ((0.10, "active_110"), (0.20, "active_120")):
        t = rd["t_stars"].get(d, float("nan")) if rd["t_stars"] else float("nan")
        margins[key] = int(act[int(np.argmin(np.abs(it_a - t)))]) if np.isfinite(t) else None
    rec["floor"] = dict(it=float(rd["t_star"]), active=int(rd["active"]), r2=r2, pr=pr, n_active_pr=npr,
                        snapshot=rd["snapshot"], **margins)
    return rec


def _sd(x):
    """Sample SD (ddof 1) of a 1-D array; nan with fewer than two values."""
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def cell_stats(runs, where):
    """Summary of one cell at one read point.

    Args:
        runs: list of b3_run dicts.
        where: "floor" or "fixed".
    Returns:
        dict with n_read, active (list), logit mean/sd, r2 mean, pr mean/sd (sd nan with < 2 reads).
    """
    rs = [r[where] for r in runs if r[where] is not None]
    a = np.array([r["active"] for r in rs], float)
    lg = logit_active(a)
    r2 = np.array([r["r2"] for r in rs], float)
    pr = np.array([r["pr"] for r in rs], float)
    return dict(n_read=len(rs), active=a.astype(int).tolist(), logit_mean=float(lg.mean()) if len(lg) else float("nan"),
                logit_sd=_sd(lg), r2=r2.tolist(), r2_mean=float(np.nanmean(r2)) if len(r2) else float("nan"),
                pr=pr.tolist(), pr_mean=float(np.nanmean(pr)) if len(pr) else float("nan"), pr_sd=_sd(pr))


def b3_verdict(stats, n_found):
    """Apply B3's pre-registered bars (as fixed in dead-relu-paper#2) at one read point.

    Args:
        stats: dict cell name -> cell_stats dict.
        n_found: dict cell name -> runs in the cell (including diverged ones).
    Returns:
        dict with per-combination comparisons and verdict "supports", "falsifies" or "between the bars".
    """
    stable = {c: stats[c]["n_read"] >= 2 for c in stats}
    comps, support, all_within = {}, False, True
    for c, (_, parts) in B3_CELLS.items():
        if len(parts) < 2:
            continue
        singles = [s for s in parts if stable[s]]
        B = max(singles, key=lambda s: stats[s]["logit_mean"]) if singles else None
        rec = dict(best=B, stable=stable[c], n_read=stats[c]["n_read"], n_found=n_found[c])
        unstable_parts = [s for s in parts if not stable[s]]
        rec["stability_gain"] = bool(stable[c] and any(
            stats[s]["n_read"] == 0 or np.mean(stats[c]["active"]) >= np.mean(stats[s]["active"])
            for s in unstable_parts))
        if B is not None and stats[c]["n_read"]:
            sb = stats[B]
            d_logit = stats[c]["logit_mean"] - sb["logit_mean"]
            d_pr = stats[c]["pr_mean"] - sb["pr_mean"]
            d_r2 = stats[c]["r2_mean"] - sb["r2_mean"]
            rec.update(d_logit=float(d_logit), bar_logit=float(SD_BAR * sb["logit_sd"]),
                       d_pr=float(d_pr), bar_pr=float(SD_BAR * sb["pr_sd"]), d_r2=float(d_r2),
                       d_active_mean=float(np.mean(stats[c]["active"]) - np.mean(sb["active"])))
            rec["beats"] = bool(stable[c] and d_logit > SD_BAR * sb["logit_sd"] and d_r2 >= -R2_COST)
            rec["within"] = bool(abs(d_logit) <= SD_BAR * sb["logit_sd"] and abs(d_pr) <= SD_BAR * sb["pr_sd"])
        else:
            rec.update(beats=False, within=False)
        support |= rec["beats"] or rec["stability_gain"]
        all_within &= rec["within"] and not rec["stability_gain"]
        comps[c] = rec
    v = "supports" if support else ("falsifies" if all_within else "between the bars")
    return dict(verdict=v, combinations=comps)


def main():
    """Read B3 (and B4 N = 4000 with --b4), print the tables, write the JSON outputs."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", help="folder for read_b3.json and the R^2 cache")
    ap.add_argument("--b4", help="read_b2_b4.json from read_b2_b4.py; applies the B4 verdict at N = 4000")
    a = ap.parse_args()
    cache_path = os.path.join(a.out_dir, "read_b3_r2_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    runs, stats, n_found = {}, {"floor": {}, "fixed": {}}, {}
    for c, (path, _) in B3_CELLS.items():
        n_found[c] = len([d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d)])
        runs[c] = [b3_run(r, cache) for r in runs_with_loss(path)]
        json.dump(cache, open(cache_path, "w"))
        for w in stats:
            stats[w][c] = cell_stats(runs[c], w)
    verdict = {w: b3_verdict(stats[w], n_found) for w in stats}

    for w, label in (("floor", "at the floor (binding)"), ("fixed", f"at {AT_FIXED} iterations")):
        print(f"\nB3 {label}\n  cell                  read  active             R2                      PR")
        for c, s in stats[w].items():
            print(f"  {c:<20s} {s['n_read']}/{n_found[c]}  {str(s['active']):<18s} "
                  f"{str(np.round(s['r2'], 3).tolist()):<23s} {np.round(s['pr'], 1).tolist()}")
        print(f"  verdict: {verdict[w]['verdict']}")
        for c, r in verdict[w]["combinations"].items():
            print(f"    {c:<20s} " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                for k, v in r.items()}))
    for c, rs in runs.items():
        for r in rs:
            if r["reason"]:
                print(f"  EXCLUDED {c} {r['name'][:40]}: {r['reason']}")

    out = dict(runs=runs, stats=stats, n_found=n_found, verdict=verdict)
    if a.b4:
        b4 = json.load(open(a.b4))["b4"]
        out_b4 = b4_verdict({k.split("_", 1)[1]: v for k, v in b4.items() if k.startswith("4000_")})
        print("\nB4 N = 4000 verdict:", json.dumps(out_b4))
        json.dump(out_b4, open(os.path.join(a.out_dir, "b4_verdict_4000.json"), "w"), indent=1)
    json.dump(cache, open(cache_path, "w"))
    json.dump(out, open(os.path.join(a.out_dir, "read_b3.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
