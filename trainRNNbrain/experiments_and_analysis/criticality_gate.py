#!/usr/bin/env python3
"""Phase 1 gate of the criticality-projection rules: did lambda1 and gradient propagation deteriorate?

Rules: docs/project_trajectory/2026-11-20_1643_rules-criticality-projection.md. A network has
deteriorated if, from init to the snapshot at or just after its floor crossing t*,
  (1) |lambda1(t*)| - |lambda1(0)| > 0.01 per step, and
  (2) |G100(t*) - G100(0)| > log10(e).
A task passes if at least 2 of its 3 networks deteriorated. Reads criticality_read.py's JSON and each
network's floor crossing (common.read_run, 1.05x).

Usage: PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/criticality_gate.py \
           --json <criticality_read json> --runs <run_dir> [...]
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import read_run

LAM_BAR = 0.01
G_BAR = math.log10(math.e)


def at_or_after(rows, t):
    """The row with the smallest iteration >= t (snapshot or final), or None.

    Args:
        rows: list of criticality_read rows of one network; t: float iteration.
    Returns:
        dict row or None.
    """
    later = sorted((r["iteration"], r) for r in rows if r["iteration"] >= t and r["kind"] != "init")
    return later[0][1] if later else None


def gate_network(rows, t_star):
    """Gate conditions for one network.

    Args:
        rows: its criticality_read rows; t_star: float floor crossing.
    Returns:
        dict: it_read (iteration of the row used), lam0, lam_t, g0, g_t, d_lam (|lam_t| - |lam0|),
        d_g (|g_t - g0|), cond1, cond2, deteriorated.
    """
    r0 = next(r for r in rows if r["kind"] == "init")
    rt = at_or_after(rows, t_star)
    d_lam = abs(rt["lyap_med"]) - abs(r0["lyap_med"])
    d_g = abs(rt["gain100_med"] - r0["gain100_med"])
    return dict(it_read=rt["iteration"], lam0=r0["lyap_med"], lam_t=rt["lyap_med"], g0=r0["gain100_med"],
                g_t=rt["gain100_med"], d_lam=d_lam, d_g=d_g, cond1=d_lam > LAM_BAR, cond2=d_g > G_BAR,
                deteriorated=bool(d_lam > LAM_BAR and d_g > G_BAR), init_ok=r0.get("init_ok"))


def main():
    """Print one line per network and the task verdicts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    rows = [r for f in a.json for r in json.load(open(f))]
    by_task = {}
    for run in a.runs:
        name = os.path.basename(run.rstrip("/"))
        net = [r for r in rows if r["network"] == name]
        t_star = read_run(run)["t_star"]
        g = gate_network(net, t_star)
        by_task.setdefault(net[0]["task"], []).append(g["deteriorated"])
        print(f"{net[0]['task']:13s} s{net[0]['seed']:<11d} t*={t_star:8.0f} read@{g['it_read']:7d} "
              f"lam {g['lam0']:+.4f}->{g['lam_t']:+.4f} (d|lam| {g['d_lam']:+.4f}, >{LAM_BAR}: {g['cond1']}) "
              f"G100 {g['g0']:+.3f}->{g['g_t']:+.3f} (|dG| {g['d_g']:.3f}, >{G_BAR:.3f}: {g['cond2']}) "
              f"deteriorated={g['deteriorated']} init_ok={g['init_ok']}")
    for task, v in by_task.items():
        print(f"{task}: {sum(v)} of {len(v)} deteriorated -> gate {'PASSES' if sum(v) >= 2 else 'FAILS'}")


if __name__ == "__main__":
    main()
