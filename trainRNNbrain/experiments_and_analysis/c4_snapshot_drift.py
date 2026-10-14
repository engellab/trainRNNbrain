#!/usr/bin/env python3
"""What did training actually do to the silent units the gradient reaches? (claim C4, test g)

The one-step Adam estimate in c4_gradient_mechanism.py could not answer this (its direction is a
network-wide mode shared by active units, and it disagrees with the saved optimizer state). Here the
answer is read from the kept weight snapshots instead: the floor snapshot t*, and t* + 500 and
t* + 5000 iterations of real training.

Per unit, classified at t* by c4_gradient_mechanism.py's output (active by participation, reached by
g_in, both on the same seeded reference batch):
  - relative change of the incoming row, ||dW[i,:]|| / ||W[i,:]|| over [W_rec | W_inp];
  - the real weight change's effect on the unit's own drive, dW_rec[i,:] r(t-1) + dW_inp[i,:] u(t-1),
    averaged over its 1% highest-h samples on the reference batch at t*;
  - whether the unit is active at t* + dt (participation recomputed on the reference batch);
  - the radial part of the change, (dW[i,:] . W[i,:]) / ||W[i,:]||^2 (< 0: the row shrinks), and the
    drive change from the part of dW[i,:] orthogonal to W[i,:] alone (a pure shrink of a row lowers
    the drive wherever that drive is positive, so it must be separated from a change of direction).

Input layout (copied from Della's c1_snapshots by hand): SNAP_DIR/<seed>_d{0,500,5000}.npz,
SNAP_DIR/<seed>_config.yaml, and NPZ_DIR/*__s<seed>__floor.npz from c4_gradient_mechanism.py.
Usage: python c4_snapshot_drift.py SNAP_DIR NPZ_DIR [--trials 256]
Output: NPZ_DIR/<task>_s<seed>__drift.npz with the per-unit arrays (read by c4_mechanism_report.py).
"""

import argparse
import glob
import os
import sys

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c4_gradient_mechanism as C

DTS = (500, 5000)


def over(g, rel=0.05):
    """Boolean mask of units with g >= rel * q95(g)."""
    return g >= rel * np.quantile(g, 0.95)


def main(snap_dir, npz_dir, trials):
    """Print, per run and unit class, row change, drive-change sign and later activity.

    Args:
        snap_dir, npz_dir: see the module docstring; trials: reference-batch trials used here.
    """
    for f in sorted(glob.glob(os.path.join(npz_dir, "*__floor.npz"))):
        seed = f.split("__s")[1].split("__")[0]
        z = np.load(f)
        cfg = OmegaConf.load(os.path.join(snap_dir, f"{seed}_config.yaml"))
        P0 = C.load_params(os.path.join(snap_dir, f"{seed}_d0.npz"))
        u, _, _ = C.batch(cfg, C.REF_SEED)
        u = u[:, :, :trials]
        H0, U0, _, _ = C.forward(P0, u)
        p0 = C.unit_stats(H0, U0, P0["slope"])["p"]
        act = over(z["ref_p"])
        reach = over(z["strict_g_in"])
        cls = {"active": act, "silent-reached": reach & ~act, "silent-unreached": ~reach & ~act}
        task = "FF" if "FlipFlop" in f else "CDDM"
        print(f"\n=== {task} seed {seed} t*={int(z['read_iter'])}: active here {over(p0).sum()} "
              f"(full batch {act.sum()}); classes " + ", ".join(f"{k} {v.sum()}" for k, v in cls.items()))
        W0 = np.hstack([P0["W_rec"], P0["W_inp"]])
        keep = {f"cls_{k}": v for k, v in cls.items()}
        for dt in DTS:
            P1 = C.load_params(os.path.join(snap_dir, f"{seed}_d{dt}.npz"))
            dW = np.hstack([P1["W_rec"], P1["W_inp"]]) - W0
            absn = np.linalg.norm(dW, axis=1)
            rel = absn / np.linalg.norm(W0, axis=1)
            radial = (dW * W0).sum(1) / (W0 * W0).sum(1)
            DD = C.own_drive(P1["W_rec"] - P0["W_rec"], P1["W_inp"] - P0["W_inp"], H0, U0, P0["slope"])
            dd = C.top_mean(DD, H0)
            dW_orth = dW - radial[:, None] * W0
            n_rec = P0["W_rec"].shape[1]
            DD = C.own_drive(dW_orth[:, :n_rec], dW_orth[:, n_rec:], H0, U0, P0["slope"])
            dd_orth = C.top_mean(DD, H0)
            del DD
            H1, U1, _, _ = C.forward(P1, u)
            act1 = over(C.unit_stats(H1, U1, P1["slope"])["p"])
            del H1
            ref = np.median(rel[act])
            for k, m in cls.items():
                print(f"  +{dt:5d} {k:16s}: row change median {np.median(rel[m]):.4f} ({np.median(rel[m]) / ref:.2f}x active)"
                      f"; drive up at top-h samples {(dd[m] > 0).mean():.2f} (median {np.median(dd[m]):+.1e})"
                      f"; active at t*+dt {act1[m].mean():.2f}")
                print(f"  {'':6s} {'':16s}  |dW| median {np.median(absn[m]):.3f}; radial median {np.median(radial[m]):+.4f} "
                      f"(shrinking {(radial[m] < 0).mean():.2f}); orthogonal part raises drive {(dd_orth[m] > 0).mean():.2f}")
            print(f"  +{dt:5d} active count {act1.sum()} (at t* on this batch {over(p0).sum()})")
            keep.update({f"{k}_{dt}": v for k, v in (("rel", rel), ("abs", absn), ("radial", radial),
                                                       ("dd", dd), ("dd_orth", dd_orth), ("act", act1))})
        np.savez(os.path.join(npz_dir, f"{task}_s{seed}__drift.npz"), **keep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snap_dir")
    ap.add_argument("npz_dir")
    ap.add_argument("--trials", type=int, default=256)
    a = ap.parse_args()
    main(a.snap_dir, a.npz_dir, a.trials)
