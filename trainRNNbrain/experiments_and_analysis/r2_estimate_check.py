#!/usr/bin/env python3
"""Estimated against measured held-out R^2 at the floor crossing, on networks with kept weights.

Where no weights were kept at the floor crossing t*, held-out R^2 there is estimated from the final
weights as 1 - (L(t*) / L_end)(1 - R2_end) (common.r2_at_floor_estimate). This script takes the
networks of the Figure 2 collect record whose R^2 at t* was MEASURED on a kept snapshot (the
Figure 1d networks), computes the estimate for each from the same record's R2_end and from
common.read_run's L(t*) and L_end, and prints estimate minus measurement per network.

Usage:  python r2_estimate_check.py <collect.pkl>     (run from experiments_and_analysis/)
"""
import pickle
import sys

import numpy as np

from common import r2_at_floor_estimate, read_run


def main(record_path):
    """Print one row per network with a measured R^2 at t*, and the largest absolute difference.

    Args:
        record_path: str, the Figure 2 collect pickle (fig_paper_F2_size.py --collect).
    Returns:
        None.
    """
    with open(record_path, "rb") as fh:
        rec = pickle.load(fh)
    rows = [r for key in ("series", "ablation") for r in rec[key]
            if r["reason"] is None and r["r2_measured"] and np.isfinite(r["r2_end"])]
    seen, diffs = set(), []
    for r in rows:
        if r["run_dir"] in seen:
            continue
        seen.add(r["run_dir"])
        rd = read_run(r["run_dir"])
        est = r2_at_floor_estimate(rd["L_star"], rd["L_end"], r["r2_end"])
        diffs.append(est - r["r2_floor"])
        print(f"{r['task']:5s} N={r['N']:<5d} s{r['seed']:>11s} t*={rd['t_star']:8.0f} "
              f"measured {r['r2_floor']:.4f}  end {r['r2_end']:.4f}  L*/Lend {rd['L_star'] / rd['L_end']:.3f}  "
              f"estimate {est:.4f}  diff {est - r['r2_floor']:+.4f}")
    print(f"{len(diffs)} networks; largest |estimate - measured| = {np.max(np.abs(diffs)):.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
