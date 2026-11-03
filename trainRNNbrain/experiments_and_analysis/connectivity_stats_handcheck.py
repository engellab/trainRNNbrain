#!/usr/bin/env python3
"""
Check M3 of notebook entry 2026-11-02_2203_connectivity-stats-rule.md: W1, W2, W3 and W5 of one
network, over its active units and over all units, recomputed with separately written pure-Python
arithmetic (lists, sorting and the math module; numpy only reads the file) and compared with
data/connectivity_stats.json, written by connectivity_stats.py.

The active indices are read from that JSON (check M1 there ties them to Extended Data Fig. 2). The
initial weights come from gradient_participation.initial_params, the rebuild check M2 validates.

Usage:  python connectivity_stats_handcheck.py [label substring, default s1423687489]
Exit status 1 if any value differs by more than a relative 1e-6.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gradient_participation import initial_params    # noqa: E402
from omegaconf import OmegaConf           # noqa: E402

DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data"))
TOL = 1e-6


def pure_sd(values):
    """Population standard deviation (divide by n) of a list of floats, two-pass."""
    n = len(values)
    mean = math.fsum(values) / n
    return math.sqrt(math.fsum((v - mean) ** 2 for v in values) / n)


def pure_median(values):
    """Median of a list of numbers by sorting (mean of the two middle values for even n)."""
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def hand_stats(W, W0, units, theta):
    """W1, W2, W3 and W5 of the submatrix of W on `units`, in plain Python.

    Args:
        W, W0: N x N nested lists (row i = incoming weights of unit i), trained and initial;
        units: list of int unit indices; theta: float threshold.
    Returns:
        dict W1, W2, W3, W5.
    """
    k = len(units)
    entries = [W[i][j] for i in units for j in units]
    n_big_in = []
    for i in units:
        n_big_in.append(sum(1 for j in units if abs(W[i][j]) >= theta))
    change = []
    for i in units:
        sq = 0.0
        for j in units:
            sq += (W[i][j] - W0[i][j]) ** 2 + (W[j][i] - W0[j][i]) ** 2
        change.append(math.sqrt(sq / (2 * k)))
    return dict(W1=pure_sd(entries),
                W2=sum(1 for v in entries if abs(v) >= theta) / len(entries),
                W3=pure_median([c / k for c in n_big_in]),
                W5=pure_median(change))


def main():
    """Recompute the statistics of one network and compare them with the JSON."""
    key = sys.argv[1] if len(sys.argv) > 1 else "s1423687489"
    with open(os.path.join(DATA, "connectivity_stats.json")) as fh:
        rec = [r for r in json.load(fh)["results"] if key in r["label"]][0]
    d = os.path.join(DATA, "pop_stats_nets", rec["label"])
    params = [f for f in os.listdir(d) if "LastParams" in f][0]
    W = np.load(os.path.join(d, params), allow_pickle=True)["W_rec"].astype(np.float64).tolist()
    cfg = OmegaConf.load(os.path.join(DATA, "connectivity_init", rec["label"], "root_config.yaml"))
    W0 = np.asarray(initial_params(cfg)["W_rec"], dtype=np.float64).tolist()
    N = len(W)
    theta = 2.0 * pure_sd([v for row in W0 for v in row])
    bad = 0
    print(f"{rec['label']}: theta hand {theta:.8f}, script {rec['theta']:.8f}")
    bad += abs(theta - rec["theta"]) > TOL * rec["theta"]
    for name, units in (("act", rec["active"]), ("all", list(range(N)))):
        h = hand_stats(W, W0, units, theta)
        for s, v in h.items():
            ref = rec[name][s]
            r = abs(v - ref) / abs(ref) if ref else abs(v)
            bad += r > TOL
            print(f"  {name:>3} {s}: hand {v:.8f}  script {ref:.8f}  rel {r:.1e}")
    print("M3 passed" if not bad else f"M3 FAILED ({bad} values)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
