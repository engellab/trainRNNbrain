"""Live-unit counts for the hyper flip-flop (2^k-1 product read-out) against the plain flip-flop.

Every cell is read at the SAME iteration (the last snapshot at or before 150k, the hyper grid's end) from its
participation trace, so plain-task cells trained to 400-500k are read backward, never at their end.
Both silence criteria are reported side by side, as the project requires (common.CRITERIA).

Usage: python flipflop_hyper_readout.py <trained_RNNs root> [--seeds]
Prints one row per (task, pen, k, N): n seeds, live count (scale-free), live count (absolute 4e-2),
mean folder-prefix r2. With --seeds, one row PER NET instead (task pen k N live_sf live_abs r2).
"""
import glob
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np

from trainRNNbrain.experiments_and_analysis.common import SILENT_FLIPFLOP, active_count

READ_AT = 150000
SUBS = ["NBitFlipFlopHyper_std_hyper", "NBitFlipFlop_std_ksweep", "NBitFlipFlop_std_pen",
        "NBitFlipFlop_std_penlong"]
R2_MIN = 0.0  # drops diverged plain-task runs (nan or negative r2), as pr_matrix.py does


def main(root, per_seed=False):
    """Collect every trace under the sweep folders and print the per-cell table (or one row per net)."""
    cells = defaultdict(list)
    for sub in SUBS:
        task = "hyper" if "Hyper" in sub else "plain"
        for f in sorted(glob.glob(os.path.join(root, sub, "*", "*", "*ParticipationTrace.pkl"))):
            m = re.search(r"_k=(\d+)_N=(\d+)(?:_pen=([a-z]+))?", f)
            pen = m.group(3) or "none"
            if pen not in ("none", "both"):
                continue
            with open(f, "rb") as fh:
                tr = pickle.load(fh)
            it = np.asarray(tr["participation_iters"])
            j = np.flatnonzero(it <= READ_AT)  # snapshots are every 100 up to 149900
            if not j.size or it[j[-1]] < READ_AT - 200:
                continue
            p = np.asarray(tr["participation"][j[-1]], dtype=float)
            try:
                r2 = float(os.path.basename(f).split("_")[0])
            except ValueError:
                r2 = float("nan")
            if not r2 >= R2_MIN:
                continue
            cells[(task, pen, int(m.group(1)), int(m.group(2)))].append(
                (active_count(p, "scalefree"), active_count(p, SILENT_FLIPFLOP), r2))
    if per_seed:
        for key in sorted(cells, key=lambda t: (t[2], t[3], t[0], t[1])):
            for sf, ab, r2 in cells[key]:
                print(f"{key[0]:6} {key[1]:5} {key[2]:>2} {key[3]:>5} {sf:>5d} {ab:>5d} {r2:7.4f}")
        return
    print(f"{'task':6} {'pen':5} {'k':>2} {'N':>5} {'n':>2}  {'live_sf':>12}  {'live_abs':>12}  {'r2':>6}")
    for key in sorted(cells, key=lambda t: (t[2], t[3], t[0], t[1])):
        a = np.array(cells[key])
        print(f"{key[0]:6} {key[1]:5} {key[2]:>2} {key[3]:>5} {len(a):>2}  "
              f"{a[:, 0].mean():6.0f} ± {a[:, 0].std():<4.0f} {a[:, 1].mean():6.0f} ± {a[:, 1].std():<4.0f} "
              f"{a[:, 2].mean():6.3f}")


if __name__ == "__main__":
    main(sys.argv[1], per_seed="--seeds" in sys.argv)
