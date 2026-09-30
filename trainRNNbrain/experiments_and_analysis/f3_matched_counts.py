#!/usr/bin/env python3
"""
Active-unit counts at MATCHED COMPUTE for every cell of Figure 3, read from the participation
traces rather than from a rebuilt network.

WHY THIS EXISTS SEPARATELY FROM f3_penalty_cache.py. Figure 3 needs two read-outs and they cannot
come from the same place.

  dimensionality, rate spread   need the firing rates, so they need the weights, and only the LAST
                                checkpoint of each run was saved. They can only be measured at each
                                run's own endpoint.
  active units                  can be read from the participation trace at any probed iteration,
                                and must be, because the arms of this figure were trained to
                                different budgets: on the flip-flop the `rws` cells stop at 150,000
                                iterations, the `frm` and `frm+rws` cells at 400,000 and the
                                unpenalised cells at 500,000. Units keep falling silent long after
                                the loss plateaus (Fig. 1e), so reading each arm at its own endpoint
                                compares a 150,000-iteration network with a 500,000-iteration one.
                                At k=3, N=1000 that alone flips the sign of the `rws` result: read
                                at its endpoint `rws` holds 218 units against the unpenalised arm's
                                190 and looks like an improvement, while read at a matched 150,000
                                it holds 218 against 263 and is the loss the manuscript reports.

So this script applies the project's standard rule - every cell read at the largest iteration EVERY
seed in it reaches, capped at READ_AT - and Figure 3 uses it for the active-unit panels while the
rebuild cache supplies the other three measures. The caption has to say that the panels differ in
read-out, and it is worth saying alongside that between 150,000 iterations and the endpoint the
penalised arms barely move (979 -> 977 and 1000 -> 1000 at k=3, N=1000); it is the unpenalised arm
that goes on losing units.

Usage:
    python f3_matched_counts.py                # writes data/fig_paper_F3_matched.npz
    python f3_matched_counts.py OUT.npz
"""

import glob
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, SILENT_REL
from f3_penalty_cache import CELL_GLOBS, cell_key

READ_AT = 150_000       # the cap the rest of the project reads penalised cells at
MIN_R2 = 0.5            # a run that never learned its task is dropped, the same rule and the same
                        # threshold the figure applies to the rebuilt networks


def cell_counts(cell, cap=READ_AT):
    """Active units per seed of one cell, at the largest iteration every seed in it reaches.

    Args:
        cell: the cell folder; cap: never read later than this iteration.
    Returns:
        (counts array, iteration read at), or (empty array, 0) if the cell has no usable traces.
    """
    tr = []
    for f in sorted(glob.glob(os.path.join(cell, "*", "*ParticipationTrace.pkl"))):
        m = re.match(r"(-?[0-9.]+|nan)_", os.path.basename(os.path.dirname(f)))
        if m is None or m.group(1) == "nan" or float(m.group(1)) < MIN_R2:
            continue
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        P, it = np.asarray(d.get("participation", [])), np.asarray(d.get("participation_iters", []))
        if P.ndim == 2 and len(it):
            tr.append((it, P))
    if not tr:
        return np.array([]), 0
    read_at = min(min(int(it[-1]) for it, _ in tr), cap)
    out = []
    for it, P in tr:
        p = P[int(np.argmin(np.abs(it - read_at)))]
        out.append(int((p >= SILENT_REL * np.quantile(p, 0.95)).sum()))
    return np.array(out), read_at


def main(out_path):
    """Read every Figure 3 cell at matched compute and write the counts. Returns the path."""
    recs = []
    cells = [(task, c) for task, g in CELL_GLOBS for c in sorted(glob.glob(g)) if os.path.isdir(c)]
    for task, cell in cells:
        counts, read_at = cell_counts(cell)
        if not len(counts):
            continue
        key = cell_key(task, os.path.basename(cell), os.path.basename(os.path.dirname(cell)))
        key.update(n_active_matched=float(counts.mean()),
                   sd=float(counts.std(ddof=1)) if len(counts) > 1 else 0.0,
                   n_seeds=len(counts), read_at=read_at)
        recs.append(key)
        print(f"  {key['task']:9s} k={key['k']} N={key['N']:5d} {key['pen']:5s} "
              f"read at {read_at:7d}: {counts.mean():7.1f} +- {key['sd']:5.1f}  n={len(counts)}",
              flush=True)
    if not recs:
        raise SystemExit("no cell had a usable participation trace - nothing written")
    arrays = {f: np.array([r[f] for r in recs]) for f in sorted(recs[0])}
    np.savez_compressed(out_path, **arrays)
    print(f"\nwrote {out_path}: {len(recs)} cells")
    return out_path


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(DATA_DIR), "fig_paper_F3_matched.npz"))
