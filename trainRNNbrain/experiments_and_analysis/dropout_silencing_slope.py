"""Does dropout change the RATE of silencing, or only its offset? 3-bit flip-flop, N=1000.

The pre-registered SECONDARY read-out of the 7-seed dropout sweep
(`slurm/SilentReLU_flipflop_dropout_seeds_spock.slurm`). The 1-seed figure showed dropout shifting
the whole live-unit curve UP without flattening it, i.e. both arms still silencing at 150k at what
looked like the same rate. One seed cannot separate "higher offset, same rate" from "slower
silencing", so this fits the late-training slope per network and compares conditions.

Method. Silencing is roughly linear in log iteration over the last decade, so for each network the
live count (scale-free, and the flip-flop absolute 4e-2) is regressed on log10(iteration) over
[FIT_FROM, 150000]. The slope is units lost per decade of training. Reported as mean +- sd over the
7 seeds per condition, with a Welch test of each dropout cell against the no-dropout cell at the
same penalty.

Only the `none` and `rws` arms are interpretable: frm and both sit at the N ceiling with and
without dropout, so their slope is ~0 by construction and is printed for completeness only.

Usage: python dropout_silencing_slope.py [<trained_RNNs root>] [--fit-from 30000]
"""
import argparse
import glob
import os
import pickle

import numpy as np

from trainRNNbrain.experiments_and_analysis.common import DATA_DIR, SILENT_FLIPFLOP, active_count
from trainRNNbrain.experiments_and_analysis.flipflop_dropout_readout import (DROP_SUB, READ_AT,
                                                                             REFS, welch)
FIT_FROM = 30000


def slopes(net_dir, fit_from):
    """Late-training silencing slope of one network, in live units lost per decade of iteration.

    Args:
        net_dir: per-network folder holding *_ParticipationTrace.pkl;
        fit_from: first iteration included in the fit.
    Returns:
        (slope_scalefree, slope_abs4e-2) as floats (negative = still silencing), or None.
    """
    f = glob.glob(os.path.join(net_dir, "*ParticipationTrace.pkl"))
    if not f:
        return None
    with open(f[0], "rb") as fh:
        tr = pickle.load(fh)
    it = np.asarray(tr["participation_iters"])
    keep = (it >= fit_from) & (it <= READ_AT)
    if keep.sum() < 10:
        return None
    x = np.log10(it[keep].astype(float))
    P = [np.asarray(p, dtype=float) for p, k in zip(tr["participation"], keep) if k]
    out = []
    for crit in ("scalefree", SILENT_FLIPFLOP):
        y = np.array([active_count(p, crit) for p in P], dtype=float)
        out.append(float(np.polyfit(x, y, 1)[0]))
    return tuple(out)


def cell(*dirs):
    """Slopes of every network across one or more cell folders, as an (n, 2) array."""
    rows = []
    for d in dirs:
        rows += [slopes(n, cell.fit_from) for n in sorted(glob.glob(os.path.join(d, "*", "")))]
    return np.array([r for r in rows if r is not None], dtype=float)


def main():
    """Print the per-condition silencing slope and its comparison to the no-dropout cell."""
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=DATA_DIR)
    ap.add_argument("--fit-from", type=int, default=FIT_FROM)
    a = ap.parse_args()
    cell.fit_from = a.fit_from

    print(f"live units lost per DECADE of iteration, fitted over [{a.fit_from}, {READ_AT}]")
    print("negative = still silencing; less negative under dropout = dropout SLOWS silencing\n")
    print(f"{'pen':5} {'dropout':8} {'n':>2} {'slope scale-free':>22} {'slope abs 4e-2':>22}   vs no-dropout")
    for pen in ("none", "rws", "frm", "both"):
        ref = cell(os.path.join(a.root, REFS[pen]),
                   os.path.join(a.root, DROP_SUB, f"EqType=h_k=3_N=1000_pen={pen}_do=none"))
        print(f"{pen:5} {'-- none':8} {len(ref):>2} "
              f"{ref[:,0].mean():10.1f} ± {ref[:,0].std():<9.1f} {ref[:,1].mean():10.1f} ± {ref[:,1].std():<9.1f}")
        for kind in ("mute", "dead"):
            b = cell(os.path.join(a.root, DROP_SUB, f"EqType=h_k=3_N=1000_pen={pen}_do={kind}"))
            if not len(b):
                continue
            _, p_sf = welch(b[:, 0], ref[:, 0])
            _, p_ab = welch(b[:, 1], ref[:, 1])
            verdict = ("SLOWER silencing" if (b[:, 0].mean() > ref[:, 0].mean() and p_sf < 0.05)
                       else "faster silencing" if (b[:, 0].mean() < ref[:, 0].mean() and p_sf < 0.05)
                       else "same rate")
            print(f"{pen:5} {kind:8} {len(b):>2} "
                  f"{b[:,0].mean():10.1f} ± {b[:,0].std():<9.1f} {b[:,1].mean():10.1f} ± {b[:,1].std():<9.1f}"
                  f"   p={p_sf:.2g}/{p_ab:.2g}  {verdict}")
        print()
    print("frm and both sit at the N ceiling with and without dropout; their slopes are ~0 by")
    print("construction and carry no information about the question.")


if __name__ == "__main__":
    main()
