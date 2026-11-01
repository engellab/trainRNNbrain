#!/usr/bin/env python3
"""Activity-threshold sensitivity on the networks Figure 2a fits (Methods, Participation).

A unit is active if its participation p_i >= f * q95(p) with f = 0.05. This script recounts the
active units of every network Figure 2a fits at f = 0.025, 0.05 and 0.10, at the stored
participation probe nearest the network's 1.05x floor crossing, and refits the size exponent at
each factor. Rule and pass/fail: notebook entry 2026-10-31_2209.

The network set and each crossing come from the Figure 2 collect record
(fig_paper_F2_size.py --collect). The participation traces are read directly from the run folders;
neither common.read_run nor fig_paper_F2_size is imported, so the count at f = 0.05 checks the
record's own count network by network.

Usage:  python threshold_sensitivity.py <collect.pkl> <trained_RNNs root>
"""
import glob
import os
import pickle
import sys

import numpy as np

FACTORS = (0.025, 0.05, 0.10)
FIT_MIN_N = 500                      # Figure 2a fits N = 500-4000 (fig_paper_F2_size.FIT_MIN_N)
FIG2A = {"3BFF": 0.478, "CDDM": 0.443, "DMTS": 0.551}   # Fig. 2a exponents (entry 2026-10-31_1957)
TOL = 0.001                          # the pre-registered tolerance on the exponent at f = 0.05


def fitted_by_fig2a(row):
    """Whether a size-series row of the collect record is one Figure 2a fits.

    Mirrors fig_paper_F2_size.is_read after apply_decisions: an exclusion string other than
    "never learned ..." is cleared, because EXCLUDED_RUNS is empty at draw time.

    Args:
        row: dict, one entry of the record's "series" list.
    Returns:
        bool.
    """
    never_learned = (row["excluded"] or "").startswith("never learned")
    return (row["N"] >= FIT_MIN_N and not never_learned and row["reason"] is None
            and np.isfinite(row["active_at"][0.05]))


def counts_at_crossing(run_dir, t_star):
    """Active counts at the three factors, and q50/q95, at the probe nearest t_star.

    Args:
        run_dir: str, the run folder holding *ParticipationTrace.pkl;
        t_star: float, the 1.05x floor crossing (iterations).
    Returns:
        (dict factor -> int count, float q50/q95, float probe iteration, int N).
    """
    with open(glob.glob(os.path.join(run_dir, "*ParticipationTrace.pkl"))[0], "rb") as fh:
        tr = pickle.load(fh)
    P = np.asarray(tr["participation"])               # (probes, N), the stored dtype
    it = np.asarray(tr["participation_iters"], float)
    j = int(np.argmin(np.abs(it - t_star)))
    p = P[j]
    q95 = np.quantile(p, 0.95)
    return ({f: int((p >= f * q95).sum()) for f in FACTORS}, float(np.quantile(p, 0.5) / q95),
            float(it[j]), int(p.size))


def exponent(by_size):
    """Least-squares slope of log(mean count) against log N.

    Args:
        by_size: dict N -> list of counts.
    Returns:
        float exponent b.
    """
    Ns = sorted(by_size)
    return float(np.polyfit(np.log(Ns), np.log([np.mean(by_size[N]) for N in Ns]), 1)[0])


def main(record_path, root):
    """Print the per-network table, the per-task ranges and exponents, and the pass/fail check.

    Args:
        record_path: str, the Figure 2 collect pickle;
        root: str, the trained_RNNs folder the record's run_dir paths are re-rooted under.
    Returns:
        int, 0 when the check at f = 0.05 passes, 1 otherwise.
    """
    with open(record_path, "rb") as fh:
        rec = pickle.load(fh)
    rows = [r for r in rec["series"] if fitted_by_fig2a(r)]
    ok = True
    print("task  N     seed        t*      probe   q50/q95  f=0.025  f=0.05  f=0.10  record")
    table = {}
    for r in rows:
        run_dir = os.path.join(root, r["run_dir"].split("data/trained_RNNs/")[1])
        c, ratio, probe, N = counts_at_crossing(run_dir, r["t_star"])
        same = c[0.05] == int(r["active_at"][0.05]) and N == r["N"]
        ok &= same
        table.setdefault(r["task"], []).append((r["N"], c, ratio))
        print(f"{r['task']:5s} {r['N']:<5d} {r['seed']:>11s} {r['t_star']:8.0f} {probe:8.0f} "
              f"{ratio:8.3f} {c[0.025]:8d} {c[0.05]:7d} {c[0.10]:7d}  {int(r['active_at'][0.05])}"
              f"{'' if same else '  MISMATCH'}")
    for task, v in table.items():
        up = [100 * (c[0.025] / c[0.05] - 1) for _, c, _ in v]
        down = [100 * (c[0.10] / c[0.05] - 1) for _, c, _ in v]
        b = {f: exponent({N: [c[f] for n, c, _ in v if n == N] for N in {n for n, _, _ in v}})
             for f in FACTORS}
        ok &= abs(b[0.05] - FIG2A[task]) <= TOL
        big = [(n, round(q, 3)) for n, _, q in v if q >= 0.05]
        print(f"{task}: {len(v)} networks; f=0.025 vs 0.05 {min(up):+.0f}% to {max(up):+.0f}%; "
              f"f=0.10 vs 0.05 {min(down):+.0f}% to {max(down):+.0f}%; exponent "
              + " / ".join(f"{b[f]:.3f}" for f in FACTORS)
              + f" (Fig. 2a {FIG2A[task]}); max q50/q95 {max(q for _, _, q in v):.3f}; "
              f"q50/q95 >= 0.05 at {big}")
    print("CHECK at f = 0.05:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
