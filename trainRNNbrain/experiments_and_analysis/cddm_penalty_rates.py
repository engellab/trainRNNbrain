#!/usr/bin/env python3
"""Firing-rate distribution of the four CDDM penalty arms, for slide 31's right panel.

The penalty cache behind slides 30 and 31 carries a WEIGHT histogram per network and no rate
histogram, and slide 31's own builder is not in this repository, so the rate version has to be
measured rather than read. The statistics are `f2_rate_lognormal`'s, unchanged, so slides 20 and 31
describe the distribution the same way.

THE FOUR CELLS ARE MATCHED ON THE BUDGET, which is what makes this comparable: all four train
200,000 iterations at N = 1000, gamma 0, weight decay 1e-6. The obvious-looking control
(`CDDM_std_g0/...LmbdRWS=0_LmbdFR=0`) trains 30,000 and is NOT used.

Usage: python cddm_penalty_rates.py [OUT.npz]
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f2_rate_lognormal as R
import f2_remedies_cache as C

OUT_DEFAULT = "data/cddm_penalty_rates.npz"
CELLS = [("control", "CDDM_std_g0_drift/EqType=h_N=1000_iters=200000"),
         ("frm", "CDDM_std_g0_penalties/EqType=h_N=1000_pen=frm"),
         ("rws", "CDDM_std_g0_penalties/EqType=h_N=1000_pen=rws"),
         ("both", "CDDM_std_g0_penalties/EqType=h_N=1000_pen=both")]


def main(out_path=OUT_DEFAULT):
    """Measure every network of the four cells and write one row each.

    Args:
        out_path: where to write the npz.
    Returns:
        the number of networks recorded.
    """
    rows = []
    for arm, cell in CELLS:
        for net in sorted(glob.glob(os.path.join(C.D, cell, "*"))):
            if not os.path.isdir(net):
                continue
            try:
                mean_rate, _sm, part, stored, n_units = R.per_unit_rates(net)
            except Exception as exc:
                print(f"  SKIP {arm} {os.path.basename(net)[:9]}: {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            active = part >= C.SILENT_REL * np.quantile(part, 0.95)
            st = R.rate_stats(mean_rate, active)
            row = {"arm": arm, "cell": cell, "N": n_units, "stored": stored,
                   "n_active": int(active.sum())}
            row.update({f"r_{k}": v for k, v in st.items()})
            rows.append(row)
            print(f"  {arm:8s} active {row['n_active']:4d}  sd ln r {st['sigma_log']:.2f}  "
                  f"skew {st['skew_log']:+.2f}  ks {st['ks']:.3f} (ref {st['ks_ref']:.3f})",
                  flush=True)
    if not rows:
        print("no networks found")
        return 0
    keys = sorted({k for r in rows for k in r})
    np.savez(out_path, log_bins=R.LOG_BINS,
             **{k: np.array([r.get(k, np.nan) for r in rows],
                            dtype=object if isinstance(rows[0].get(k), str) else float)
                for k in keys})
    print(f"\nwrote {out_path}: {len(rows)} networks")
    return len(rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT)
