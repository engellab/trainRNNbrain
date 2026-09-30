#!/usr/bin/env python3
"""
The DMTS 7-tau penalty sweep, scored under the read-out its own launcher pre-registered.

WHY THIS IS A TABLE AND NOT A POINT ON AN AXIS. Unpenalised DMTS at this delay is bimodal per
seed: a run either finds the memory solution and scores r2 ~ 0.99, or it sits at the constant-output
solution and scores ~0.43 (0.4278 and 0.4275 to four figures, which is what a constant output
scores on this task). There is nothing in between. A mean r2 over such a cell describes no network
that exists, and a mean active count over it is a selection effect, because the runs that never
learned the task hold systematically fewer active units than the ones that did.

So the read-out fixed in slurm/SilentReLU_dmts_d7_penalties_della.slurm before any job ran is:

  PRIMARY    the SOLVE RATE per (arm, size), solved := clean r2 >= 0.9, n = 3.
  SECONDARY  active units under the scale-free rule, reported SEPARATELY for solvers and for
             failures and never pooled.
  RULE       every cell states its solve rate beside its unit count, and a cell that solves 0/3
             contributes no unit count at all.

WHAT THE SWEEP IS. 4 penalty arms (none; rws at lambda 0.05; frm at lambda 0.2; both) x 3 sizes
(N = 500, 1000, 2000) x 3 seeds, DMTS at a 7-tau delay, 150,000 iterations, equation h. The cells
live on Della under ~/trainRNNbrain/data/trained_RNNs/DMTS_d7_pen and the participation traces are
rsynced here. An N = 4000 unpenalised cell and replacement seeds for the two unpenalised failures
were still training when this was written (SilentReLU_dmts_d7_extend_della.slurm), so the
unpenalised row will gain seeds; the penalised arms are complete.

Usage:  python dmts_d7_solve_table.py [--latex]
Output: the table on stdout; --latex also writes a LaTeX tabular to stdout for the supplementary.
"""

import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, SILENT_REL

ROOT = os.path.join(DATA_DIR, "DMTS_d7_pen")
SOLVED_AT = 0.9          # the pre-registered bar between the two modes
READ_AT = 150_000        # the sweep's own iteration budget
ARM_ORDER = ["none", "rws", "frm", "both"]
ARM_LABEL = {"none": "no penalty", "rws": "rws", "frm": "frm", "both": "frm + rws"}


def active_count(run_dir):
    """Active units of one run, at the last probe at or before READ_AT.

    Args:
        run_dir: one trained-network folder.
    Returns:
        int active-unit count under p_i >= 0.05 q_95(p), or None if the run has no usable trace.
    """
    f = glob.glob(os.path.join(run_dir, "*ParticipationTrace.pkl"))
    if not f:
        return None
    d = pickle.load(open(f[0], "rb"))
    P = np.asarray(d.get("participation", []))
    it = np.asarray(d.get("participation_iters", []))
    if P.ndim != 2 or not len(it):
        return None
    p = P[int(np.argmin(np.abs(it - min(int(it[-1]), READ_AT))))]
    return int((p >= SILENT_REL * np.quantile(p, 0.95)).sum())


def scan():
    """Every cell of the sweep, split into solvers and failures.

    Returns:
        list of (arm, N, solvers, failures) sorted by arm then N, where each of solvers and
        failures is a list of (r2, active count or None).
    """
    rows = []
    for cell in sorted(glob.glob(os.path.join(ROOT, "EqType=h_N=*_pen=*"))):
        b = os.path.basename(cell)
        N = int(re.search(r"N=(\d+)", b).group(1))
        arm = re.search(r"pen=([a-z]+)", b).group(1)
        solved, failed = [], []
        for run in sorted(glob.glob(os.path.join(cell, "*"))):
            if not os.path.isdir(run):
                continue
            m = re.match(r"(-?[0-9.]+|nan)_", os.path.basename(run))
            if m is None or m.group(1) == "nan":
                continue
            r2 = float(m.group(1))
            (solved if r2 >= SOLVED_AT else failed).append((r2, active_count(run)))
        if solved or failed:
            rows.append((arm, N, solved, failed))
    return sorted(rows, key=lambda r: (ARM_ORDER.index(r[0]) if r[0] in ARM_ORDER else 9, r[1]))


def summarise(group):
    """Mean, SD and n of the active counts in one group of runs.

    Args:
        group: list of (r2, active count or None).
    Returns:
        (mean, sd, n) with nan for mean and sd when no run in the group has a count.
    """
    v = [c for _, c in group if c is not None]
    if not v:
        return float("nan"), float("nan"), 0
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)


def main():
    """Print the solve-rate table. Returns the scanned rows."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="also print a LaTeX tabular")
    args = ap.parse_args()
    rows = scan()
    if not rows:
        raise SystemExit(f"no cells under {ROOT} - rsync the traces from Della first")

    print(f"\nDMTS, 7 tau delay, 150,000 iterations. Solved := clean r2 >= {SOLVED_AT}.")
    print(f"{'arm':11s} {'N':>5s}  {'solved':>7s}   {'active, solvers':>16s}   "
          f"{'active, failures':>17s}   {'r2 of failures':>16s}")
    for arm, N, solved, failed in rows:
        ms, ss, ns = summarise(solved)
        mf, sf, nf = summarise(failed)
        a_ok = "-" if ns == 0 else f"{ms:.0f} +- {ss:.0f} ({ns})"
        a_no = "-" if nf == 0 else f"{mf:.0f} +- {sf:.0f} ({nf})"
        bad = ", ".join(f"{r:.3f}" for r, _ in sorted(failed)) or "-"
        print(f"{ARM_LABEL.get(arm, arm):11s} {N:5d}  {len(solved):3d}/{len(solved)+len(failed):<3d}"
              f"  {a_ok:>16s}   {a_no:>17s}   {bad:>16s}")

    tot = {}
    for arm, _, solved, failed in rows:
        s, n = tot.get(arm, (0, 0))
        tot[arm] = (s + len(solved), n + len(solved) + len(failed))
    print("\n  solve rate over all sizes: " +
          ";  ".join(f"{ARM_LABEL.get(a, a)} {tot[a][0]}/{tot[a][1]}"
                     for a in ARM_ORDER if a in tot))

    if args.latex:
        print("\n% --- supplementary table, generated by dmts_d7_solve_table.py ---")
        print("\\begin{tabular}{@{}llrrr@{}}")
        print("\\toprule")
        print("arm & $N$ & solved & active (solvers) & active (failures) \\\\")
        print("\\midrule")
        for arm, N, solved, failed in rows:
            ms, ss, ns = summarise(solved)
            mf, sf, nf = summarise(failed)
            a_ok = "---" if ns == 0 else f"${ms:.0f} \\pm {ss:.0f}$ ({ns})"
            a_no = "---" if nf == 0 else f"${mf:.0f} \\pm {sf:.0f}$ ({nf})"
            print(f"\\texttt{{{ARM_LABEL.get(arm, arm)}}} & {N} & "
                  f"{len(solved)}/{len(solved)+len(failed)} & {a_ok} & {a_no} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
    return rows


if __name__ == "__main__":
    main()
