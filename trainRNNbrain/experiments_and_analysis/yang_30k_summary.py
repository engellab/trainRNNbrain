"""Confirmation run read-out: does training on 20 tasks recruit more units for CDDM than training
on CDDM alone? Yang family, N=1000, 30k iterations, 3 seeds per set (job 6209433).

Pre-registered before the run (docs/project_trajectory.md, 2026-09-15 19:05): per SET, the number
of units live on CDDM trials (the union of contextdm1 and contextdm2) as mean +- sd over the three
seeds under all three criteria, the shared-vs-CDDM-private split of that set in the 20-task nets,
and per-rule accuracy with "solved" := accuracy >= 0.95.

Both sets are read at the END of training, i.e. 30000 TRUE iterations for all six networks. Three
of them were warm-started from the 10k pilots, so their folder name says MI=20000 and their trace
covers true iterations 10k-30k; LastParams is nonetheless the 30k network, so the comparison is
matched and no offset is needed here (it is needed for any live-units-vs-iteration figure).

Wraps `multitask_readout.readout` so the single- and multi-task sides use one code path.

Usage: python yang_30k_summary.py [<trained_RNNs root>] [--sub Yang_std_30k] [--n-trials 256]
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

from trainRNNbrain.experiments_and_analysis.common import DATA_DIR, SILENT_FLIPFLOP, active_count
from trainRNNbrain.experiments_and_analysis.multitask_readout import readout

FOCUS = "contextdm1,contextdm2"
SOLVED = 0.95


def ms(v):
    """Format a list of numbers as 'mean ± sd' (sd over seeds, population sd as elsewhere here)."""
    a = np.asarray(v, dtype=float)
    return f"{a.mean():7.1f} ± {a.std():<5.1f}"


def main():
    """Read every network under the sweep, group by task set, print the pre-registered table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=DATA_DIR)
    ap.add_argument("--sub", default="Yang_std_30k")
    ap.add_argument("--n-trials", type=int, default=256)
    a = ap.parse_args()

    by_set = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(a.root, a.sub, "*", "*"))):
        if not os.path.isdir(d):
            continue
        s = re.search(r"_set=([a-z0-9]+)_", d).group(1)
        r = readout(d, FOCUS, a.n_trials)
        r["true_iters"] = 30000
        r["warm"] = "MI=20000" in d          # warm-started from the 10k pilot
        by_set[s].append(r)
        print(f"  read {s} seed={r['seed']}{' (warm start)' if r['warm'] else ''}: "
              f"CDDM-live {r['cddm_active']}, {len(r['p'])} rules", flush=True)

    print(f"\n{'':22} {'CDDM live (scale-free)':>24} {'CDDM live (1e-6)':>24} {'CDDM live (4e-2)':>24}")
    for s, rs in sorted(by_set.items()):
        sf = [r["cddm_active"] for r in rs]
        hard = [active_count(np.maximum(*[r["p"][n] for n in FOCUS.split(",")]), "hard") for r in rs]
        ab = [active_count(np.maximum(*[r["p"][n] for n in FOCUS.split(",")]), SILENT_FLIPFLOP) for r in rs]
        print(f"{s:14} n={len(rs)}  {ms(sf)} {'':6} {ms(hard)} {'':6} {ms(ab)}")

    print(f"\n{'':22} {'whole-batch live (scale-free)':>30}   shared / CDDM-private (scale-free)")
    for s, rs in sorted(by_set.items()):
        allsf = [active_count(r["p_all"], "scalefree") for r in rs]
        print(f"{s:14} n={len(rs)}  {ms(allsf)} {'':10}   "
              f"{ms([r['cddm_shared'] for r in rs])} / {ms([r['cddm_private'] for r in rs])}")

    print("\nper-rule accuracy, mean over seeds (solved := >= 0.95):")
    for s, rs in sorted(by_set.items()):
        names = list(rs[0]["acc"])
        print(f"  [{s}]")
        for n in names:
            v = [r["acc"][n] for r in rs]
            flag = "" if np.mean(v) >= SOLVED else "   <-- below 0.95"
            print(f"    {n:18} {np.mean(v):5.3f} ± {np.std(v):5.3f}{flag}")


if __name__ == "__main__":
    main()
