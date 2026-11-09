#!/usr/bin/env python3
"""Round 19: the Maze size series (ReLU, N = 250-4,000) at its final budgets, and the size exponent refit.

RULES. docs/project_trajectory/2026-11-07_1008_round19-train-all-rules.md: the across-size comparison is read at
the learned point (the 16 Oct rule); the fit on equal 4,000-update budgets is computed at update 4,000, with
N = 4,000 represented only by the networks learned by then; the end read at each size's own final budget has
unequal budgets (4,000 updates for N <= 2,000, 8,000 for N = 4,000) and is labelled so. The fit itself is
round18_training_read.read_exponent's: fig_paper_F2_size.power_fit (least squares of log mean active count on
log N) and boot_exponent (95% bootstrap, BOOT_SEED), at N >= 500 (FIT_MIN_N) and at every size.

FITS PRINTED.
  check     the round 18 equal-budget fit recomputed: N = 1,000 seeds 1-3 only, every record cut at update
            4,000, N = 4,000 by the networks learned at 4,000. Entry 2026-11-08_0859 gave 0.351 (learned point)
            and 0.491 (end), N >= 500, with N = 4,000 seed 1 alone.
  learned   every learned network at its learned point, each on its full record (maze_series).
  u4000     equal budgets: every record cut at update 4,000; learned at 4,000; the learned point and update 4,000.
  final     every learned network's last probe at its own final budget (unequal budgets, labelled).
The learned, u4000 and final fits are given with ReLU N = 1,000 seeds 1-6 (the round 19 N = 1,000 group, one 4,000-update
budget) and, as a sensitivity read, with seeds 1-3 (the round 18 set).

Usage (repository root):
  python trainRNNbrain/experiments_and_analysis/round19_size_exponent.py [--root data/trained_RNNs_rl_maze]
      [--out data/round19/size_exponent.json]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from fig_paper_F2_size import BOOT_SEED, FIT_MIN_N, boot_exponent, maze_series, power_fit   # noqa: E402
from common import maze_learned_index                                                         # noqa: E402


def cut(row, U):
    """A size-series row's state on its record cut at update U.

    Args:
        row: dict from fig_paper_F2_size.maze_series (it_act, success, active arrays per probe).
        U: int, the update at which the record is cut.
    Returns:
        dict: reached (bool), learned (bool), t_star (float, nan if not learned), active_learned and
        active_at (floats, nan where undefined), success_at (float).
    """
    m = row["it_act"] <= U
    if row["it_act"].max() < U:
        return dict(reached=False, learned=False, t_star=np.nan, active_learned=np.nan, active_at=np.nan,
                    success_at=np.nan)
    li = maze_learned_index(row["success"][m])
    return dict(reached=True, learned=li is not None,
                t_star=float(row["it_act"][m][li]) if li is not None else np.nan,
                active_learned=float(row["active"][m][li]) if li is not None else np.nan,
                active_at=float(row["active"][m][-1]), success_at=float(row["success"][m][-1]))


def fit(pairs):
    """The power-law exponent of mean active count on N, with its bootstrap interval.

    Args:
        pairs: list of (N, active count) tuples, one per network.
    Returns:
        dict: cells (N -> list of counts), and for "N>=500" and "every size" the exponent b with lo, hi.
    """
    cells = {}
    for N, a in pairs:
        cells.setdefault(int(N), []).append(float(a))
    arr = {N: np.array(v) for N, v in sorted(cells.items())}
    out = dict(cells={str(N): v for N, v in sorted(cells.items())})
    for lab, mn in (("N>=500", FIT_MIN_N), ("every size", 0)):
        b, _ = power_fit(arr, min_n=mn)
        lo, hi = boot_exponent({N: v for N, v in arr.items() if N >= mn}, np.random.default_rng(BOOT_SEED), mn)
        out[lab] = dict(b=float(b), lo=float(lo), hi=float(hi))
    return out


def keep(row, only13):
    """Whether a size-series row enters a fit.

    Args:
        row: dict from maze_series; only13: bool, restrict N = 1,000 to seeds 1-3 (the round 18 set).
    Returns:
        bool.
    """
    return not (only13 and row["N"] == 1000 and row["seed"] not in ("1", "2", "3"))


def show(name, res):
    """Print one fit: networks and mean count per size, and the two exponents.

    Args:
        name: str, the fit's label; res: dict from fit().
    """
    cells = "; ".join(f"N={N}: n={len(v)}, mean {np.mean(v):.0f}" for N, v in res["cells"].items())
    print(f"{name:<34s} {cells}")
    for lab in ("N>=500", "every size"):
        f = res[lab]
        print(f"{'':<34s} {lab:>10s}: b = {f['b']:.3f} ({f['lo']:.3f}-{f['hi']:.3f})")


def main():
    """Read the ReLU size series, print every network and the four fits, and write the JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.join(REPO, "data", "trained_RNNs_rl_maze"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "round19", "size_exponent.json"))
    a = ap.parse_args()
    rows = maze_series(os.path.join(a.root, "Maze_suite"))
    print("N seed | finished, budget, last probe | learned (update, active there) | end success, active | "
          "at 4,000: learned, active | reason")
    for r in rows:
        c = cut(r, 4000)
        print(f"{r['N']:5d} s{r['seed']} | {r['finished']}, {r['budget']}, {r['it_end']} | {r['learned']} "
              f"({r['t_star']:.0f}, {r['active_learned']:.0f}) | {r['success_end']:.2f}, {r['active_end']:.0f} | "
              f"{c['learned']}, {c['active_at']:.0f} | {r['reason']}")
    res = {}
    c4 = [(r, cut(r, 4000)) for r in rows]
    res["check"] = {k: fit([(r["N"], c[k]) for r, c in c4 if keep(r, True) and c["learned"]])
                    for k in ("active_learned", "active_at")}
    res["learned"] = fit([(r["N"], r["active_learned"]) for r in rows if r["learned"]])
    res["learned_seeds1-3"] = fit([(r["N"], r["active_learned"]) for r in rows if r["learned"] and keep(r, True)])
    for tag, only13 in (("seeds1-6", False), ("seeds1-3", True)):
        res[f"u4000_{tag}"] = {k: fit([(r["N"], c[k]) for r, c in c4 if keep(r, only13) and c["learned"]])
                               for k in ("active_learned", "active_at")}
        res[f"final_{tag}"] = fit([(r["N"], r["active_end"]) for r in rows if keep(r, only13) and r["learned"]])
    print()
    show("check: learned point, cut at 4,000", res["check"]["active_learned"])
    show("check: update 4,000", res["check"]["active_at"])
    show("learned point, final budgets", res["learned"])
    show("learned point, final (seeds1-3)", res["learned_seeds1-3"])
    for tag in ("seeds1-6", "seeds1-3"):
        show(f"u4000 learned point ({tag})", res[f"u4000_{tag}"]["active_learned"])
        show(f"u4000 update 4,000 ({tag})", res[f"u4000_{tag}"]["active_at"])
        show(f"final budgets, end ({tag})", res[f"final_{tag}"])
    res["runs"] = [dict(N=r["N"], seed=r["seed"], finished=r["finished"], budget=r["budget"], it_end=r["it_end"],
                        learned=r["learned"], t_star=r["t_star"], active_learned=r["active_learned"],
                        active_end=r["active_end"], success_end=r["success_end"], reason=r["reason"],
                        at4000=cut(r, 4000)) for r in rows]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
