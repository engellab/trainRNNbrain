"""The N = 250 3BFF/CDDM networks and the N = 100 Maze networks of the size series, read with the paper's readers.

Rules: docs/project_trajectory/2026-11-09_1925_n250-n100maze-rules.md (written before any job ran).

supervised  Reads every run under NBitFlipFlop_n250 and CDDM_n250 (common.DATA_DIR) with
            fig_paper_F2_size.read_one, the function collect() applies to every size-series network: the floor
            crossing at 1.05x, 1.10x and 1.20x the fitted floor of the noisy training loss, the number of active
            units at the noise-free participation probe nearest each crossing, and held-out R^2 (36 noise draws)
            at the end and at the 1.05x crossing. The other sizes come from Figure 2's collect record
            (fig_paper_F2_size.COLLECT, decisions re-applied). Fits: power_fit and boot_exponent (BOOT_SEED,
            N_BOOT resamples within each size) over N >= 500 (the paper's, unchanged) and over every size,
            without and with N = 250. New held-out scores go to data/size250/r2_cache.json; the shared cache is
            read, never written.
maze        Reads Maze_suite with fig_paper_F2_size.maze_series (as round19_size_exponent.py does) and fits the
            learned-point counts, the counts at update 4,000 (networks learned by then, round19_size_exponent.cut)
            and the end counts at each size's final budget, over N >= 500 and over every size from N = 250 (old)
            and from N = 100 (new).

Both first check that they reproduce the values the rules entry states (3BFF 0.478, CDDM 0.443 over N >= 500 at
1.05x; Maze 0.343 and 0.471 at the learned point).

Usage (repository root):
  python trainRNNbrain/experiments_and_analysis/size250_n100maze_read.py supervised
  python trainRNNbrain/experiments_and_analysis/size250_n100maze_read.py maze [--root data/trained_RNNs_rl_maze]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import heldout_r2 as heldout                                                                     # noqa: E402
from common import DATA_DIR, runs_with_loss                                                      # noqa: E402
from fig_paper_F2_size import (BOOT_SEED, COLLECT, FIT_MIN_N, apply_decisions, boot_exponent,   # noqa: E402
                               is_read, maze_series, power_fit, read_one)
from round19_size_exponent import cut                                                            # noqa: E402

OUT_DIR = os.path.join(REPO, "data", "size250")
NEW_CELLS = {"3BFF": "NBitFlipFlop_n250/EqType=h_N=250_arm=control",
             "CDDM": "CDDM_n250/EqType=h_N=250_arm=control"}
MARGINS = (0.05, 0.10, 0.20)
# values stated in the rules entry, which the readers must reproduce before anything new is read
CHECK = {("3BFF", 0.05): 0.478, ("CDDM", 0.05): 0.443}
MAZE_CHECK = {"N>=500": 0.343, "every size": 0.471}


def exponent(cells, min_n, drop_small):
    """The fitted size exponent of the mean active count and its 95% bootstrap interval.

    Args:
        cells: dict N -> 1-D array of active counts (one per network); min_n: int, the smallest size fitted;
        drop_small: bool, remove sizes below min_n before resampling (as round19_size_exponent.fit does for
            Maze) or resample every size and fit N >= min_n (as fig_paper_F2_size does for 3BFF and CDDM). The
            two give the same b and intervals that differ only through the random stream.
    Returns:
        dict: b, lo, hi (floats), and sizes (list of the N fitted).
    """
    cells = {N: np.asarray(v, float) for N, v in cells.items() if len(v)}
    use = {N: v for N, v in cells.items() if N >= min_n} if drop_small else cells
    b, _ = power_fit(use, min_n=min_n)
    lo, hi = boot_exponent(use, np.random.default_rng(BOOT_SEED), min_n)
    return dict(b=float(b), lo=float(lo), hi=float(hi), sizes=sorted(N for N in use if N >= min_n))


def cells_from(rows, margin):
    """Active counts per size at one floor margin, from read rows that count in Figure 2.

    Args:
        rows: list of size-series dicts (collect record rows or read_one records with task, N, excluded set);
        margin: float, 0.05, 0.10 or 0.20.
    Returns:
        dict N -> float array of active counts.
    """
    out = {}
    for r in rows:
        if is_read(r, margin):
            out.setdefault(int(r["N"]), []).append(float(r["active_at"][margin]))
    return {N: np.array(v) for N, v in sorted(out.items())}


def fmt(f):
    """Format an exponent dict from exponent() as 'b (lo-hi)'."""
    return f"{f['b']:.3f} ({f['lo']:.3f}-{f['hi']:.3f})"


def supervised():
    """Read the N = 250 networks, print every network and the fits old and new, and write data/size250/supervised.json.

    Returns:
        dict: the record written.
    """
    rec = apply_decisions(pickle.load(open(COLLECT, "rb")))
    old = {t: [r for r in rec["series"] if r["task"] == t] for t in NEW_CELLS}
    for (t, m), want in CHECK.items():
        got = power_fit(cells_from(old[t], m), FIT_MIN_N)[0]
        assert abs(got - want) < 5e-4, f"check failed: {t} {m} gives {got:.4f}, the rules entry states {want}"
    print("check passed: 3BFF and CDDM exponents over N >= 500 at 1.05x reproduce 0.478 and 0.443")
    cache = heldout.load_cache()
    private = os.path.join(OUT_DIR, "r2_cache.json")
    if os.path.exists(private):
        cache.update(json.load(open(private)))
    n0 = len(cache)
    new, out = {}, dict(networks=[], fits={})
    for t, cell in NEW_CELLS.items():
        runs = runs_with_loss(os.path.join(DATA_DIR, cell))
        print(f"{t} N = 250: {len(runs)} run(s) under {cell}")
        new[t] = []
        for run in runs:
            r, _ = read_one(run, cache)
            r.update(task=t, N=250, excluded=None)
            new[t].append(r)
            row = dict(task=t, N=250, seed=r["seed"], it_end=r["it_end"], t_star=r["t_star"], reason=r["reason"],
                       t_stars=r["t_stars"], active_at=r["active_at"], active_end=r["active_end"],
                       r2_floor=r["r2_floor"], r2_measured=r["r2_measured"], r2_end=r["r2_end"],
                       n_bursts=r["n_bursts"], floor_kind=r["floor_kind"])
            out["networks"].append(row)
            print(f"  s{r['seed']}: end {r['it_end']:,.0f}; 1.05x crossing {r['t_star']:,.0f} ({r['reason']}); active "
                  f"{r['active_at'][0.05]:.0f} / {r['active_at'][0.10]:.0f} / {r['active_at'][0.20]:.0f} of 250 at "
                  f"1.05/1.10/1.20x; end {r['active_end']}; R2 at crossing {r['r2_floor']:.3f}"
                  f"{' (measured)' if r['r2_measured'] else ' (estimated)'}, end {r['r2_end']:.3f}; bursts {r['n_bursts']}")
        for m in MARGINS:
            c_old, c_new = cells_from(old[t], m), cells_from(old[t] + new[t], m)
            fits = {"N>=500 (paper)": exponent(c_old, FIT_MIN_N, False),
                    "N>=500, with N = 250 in the resampled set": exponent(c_new, FIT_MIN_N, False),
                    "every size, old": exponent(c_old, 0, False), "every size, new": exponent(c_new, 0, False)}
            means = {N: [len(v), float(v.mean()), float(v.min()), float(v.max())] for N, v in c_new.items()}
            out["fits"][f"{t} {m:.2f}"] = dict(fits=fits, cells=means)
            print(f"  {t} at {1 + m:.2f}x: n, mean per size "
                  + "; ".join(f"{N}: {v[0]}, {v[1]:.1f}" for N, v in means.items()))
            for k, f in fits.items():
                print(f"      {k:<42s} b = {fmt(f)} over N = {f['sizes']}")
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({k: v for k, v in cache.items()}, open(private, "w"), indent=0, sort_keys=True)
    path = os.path.join(OUT_DIR, "supervised.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"wrote {path}; {len(cache) - n0} new held-out R2 score(s) in {private}")
    return out


def maze(root):
    """Read the Maze ReLU size series with N = 100, print every N = 100 network and the fits old and new.

    Args:
        root: str, the folder holding Maze_suite (data/trained_RNNs_rl_maze by default).
    Returns:
        dict: the record written to data/size250/maze.json.
    """
    rows = maze_series(os.path.join(root, "Maze_suite"))
    learned = {}
    for r in rows:
        if r["learned"]:
            learned.setdefault(r["N"], []).append(r["active_learned"])
    learned = {N: np.array(v) for N, v in sorted(learned.items())}
    for k, mn in (("N>=500", FIT_MIN_N), ("every size", 250)):
        got = power_fit({N: v for N, v in learned.items() if N >= 250}, mn)[0]
        assert abs(got - MAZE_CHECK[k]) < 5e-4, f"check failed: Maze {k} gives {got:.4f}, stated {MAZE_CHECK[k]}"
    print("check passed: Maze learned-point exponents reproduce 0.343 (N >= 500) and 0.471 (N >= 250)")
    out = dict(networks=[], fits={})
    print("N = 100 networks: seed | finished, budget, last update | learned (update, active there) | "
          "end success, active | at 4,000: learned, active, success | reason")
    for r in rows:
        if r["N"] != 100:
            continue
        c = cut(r, 4000)
        out["networks"].append(dict(N=r["N"], seed=r["seed"], finished=r["finished"], budget=r["budget"],
                                    it_end=r["it_end"], learned=r["learned"], t_star=r["t_star"],
                                    active_learned=r["active_learned"], active_end=r["active_end"],
                                    success_end=r["success_end"], reason=r["reason"], at4000=c))
        print(f"  s{r['seed']} | {r['finished']}, {r['budget']}, {r['it_end']} | {r['learned']} ({r['t_star']:.0f}, "
              f"{r['active_learned']:.0f}) | {r['success_end']:.2f}, {r['active_end']:.0f} | {c['learned']}, "
              f"{c['active_at']:.0f}, {c['success_at']:.2f} | {r['reason']}")
    reads = {"learned point": {}, "update 4,000": {}, "end, final budgets": {}}
    for r in rows:
        c = cut(r, 4000)
        if r["learned"]:
            reads["learned point"].setdefault(r["N"], []).append(r["active_learned"])
            reads["end, final budgets"].setdefault(r["N"], []).append(r["active_end"])
        if c["learned"]:
            reads["update 4,000"].setdefault(r["N"], []).append(c["active_at"])
    for k, d in reads.items():
        cells = {N: np.array(v, float) for N, v in sorted(d.items())}
        fits = {"N>=500": exponent(cells, FIT_MIN_N, True),
                "every size, old (N >= 250)": exponent(cells, 250, True),
                "every size, new (N >= 100)": exponent(cells, 0, True)}
        means = {N: [len(v), float(v.mean()), float(v.min()), float(v.max())] for N, v in cells.items()}
        out["fits"][k] = dict(fits=fits, cells=means)
        print(f"  {k}: n, mean per size " + "; ".join(f"{N}: {v[0]}, {v[1]:.1f}" for N, v in means.items()))
        for kk, f in fits.items():
            print(f"      {kk:<28s} b = {fmt(f)} over N = {f['sizes']}")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "maze.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("wrote", path)
    return out


def main():
    """Parse the command line and run the supervised or the Maze read."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("supervised", "maze"))
    ap.add_argument("--root", default=os.path.join(REPO, "data", "trained_RNNs_rl_maze"))
    a = ap.parse_args()
    if a.what == "supervised":
        supervised()
    else:
        maze(a.root)


if __name__ == "__main__":
    main()
