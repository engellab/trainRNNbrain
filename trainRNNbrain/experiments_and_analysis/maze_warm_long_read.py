"""Read the long warm Maze continuation (Phase B): readings P1-P5 of notebook entry
2026-11-18_0745_rules-maze-long-warm-continuation.md. Written and committed before any record of Phase B was
pulled to this machine; the thresholds are the rules'.

Window W: updates 24,001-34,000 (warm stages 13 and 14), or W' = 24,001-29,000 if fewer than 7 networks reach
update 34,000. Per network:
  U  decrease: minus the least-squares slope of the probe's active count over W, per 100,000 Adam steps
     (1 update = 8 Adam steps); decreasing if U >= 5.
  F  return slope: least-squares slope of the per-update discounted return over W, per 100,000 Adam steps,
     divided by the rise (as maze_flat_return_full.measure computes it); flat if |F| < 3%.
  P1 (primary): the two-thirds rule over the flat networks (at least 5 needed).
  P2: flat by maze_flat_return_full.measure over W (3% of the rise per 1,000 updates), decreasing as in P1.
  P3: P1's flat networks, decreasing if at least 5 units per 1,000 updates (62.5 per 100,000 Adam steps).
  P4: P1 over updates 29,001-34,000 alone (W only).
  P5: jumps J(24,000) and J(29,000) against the noise of maze_warm_start_readings (descriptive; flagged if more than
      2 of 9 networks exceed it at either boundary).
Every record is cut at the window's end first; the learned point (common.maze_learned_index), the rise and the
read condition refer to the cut record. A network is read if its record reaches the window's end and the cut
record has a learned point (which requires its last probe's success >= 0.9).

Usage (from trainRNNbrain/experiments_and_analysis):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python maze_warm_long_read.py <suite root> <out.json>
The root holds the Phase B cells (trained_RNNs_rl_maze_warmlong), the warm cells of the warm-start test and every
earlier stage of the nine networks (links into data/trained_RNNs_rl_maze/Maze_suite), so that
common.maze_load_runs joins each network into one record from update 0.
"""
import json
import sys

import numpy as np

from common import maze_learned_index, maze_load_runs
from fig_paper_F2_size import maze_series
from maze_flat_return_full import BLOCK, FLAT_BAR, measure
from maze_warm_start_readings import jump, noise
from round18_maze import three_way

START, MID, END, END_SHORT = 24000.0, 29000.0, 34000.0, 29000.0   # W = (START, END]; W' = (START, END_SHORT]
MIN_AT_END = 7                         # networks that must reach END for W to be used
ADAM_PER_UPDATE = 8
PER_100K = 100000.0 / ADAM_PER_UPDATE  # updates per 100,000 Adam steps (12,500)
DECREASE_BAR = 5.0                     # units per 100,000 Adam steps (the supervised bar)
FLAT_MATCHED = 0.03                    # |F| below this fraction of the rise per 100,000 Adam steps
OLD_DECREASE_BAR = 5.0                 # P3: units per 1,000 updates (the earlier Maze bar)
MIN_FLAT = 5
SUCCESS_LEVEL = 0.9                    # common.MAZE_LEARNED_SUCCESS; printed in the exclusion reason
N_NETS = 9
MAX_DISRUPTED = 2                      # P5: more than this many networks above their noise flags a disruption


def phase_b_records(root):
    """The joined record of every network whose chain includes a Phase B stage.

    Args:
        root: str, the suite folder.
    Returns:
        dict (N, seed) -> (row, run): row from fig_paper_F2_size.maze_series, run from common.maze_load_runs.
    """
    rows = {r["name"]: r for r in maze_series(root)}
    out = {}
    for run in maze_load_runs(root):
        if not any("_warm_cont13" in c for c in run["chain"]):
            continue
        out[(run["N"], run["seed"])] = (rows["/".join(run["folder"].split("/")[-2:])], run)
    return out


def truncate(row, run, end):
    """Copies of a record cut at update `end`, with the learned point recomputed on the cut record.

    Args:
        row: dict from maze_series; run: dict from maze_load_runs; end: float, last update kept.
    Returns:
        (row, run) copies with it_act/active/success and progress limited to updates <= end, and t_star the
        learned probe's update of the cut record (nan if the cut record has none).
    """
    m = np.asarray(row["it_act"], float) <= end
    suc = np.asarray(row["success"])[m]
    li = maze_learned_index(suc)
    row2 = dict(row, it_act=np.asarray(row["it_act"])[m], active=np.asarray(row["active"])[m], success=suc,
                t_star=float(np.asarray(row["it_act"])[m][li]) if li is not None else float("nan"))
    run2 = dict(run, progress=[p for p in run["progress"] if p["update"] <= end])
    return row2, run2


def rise_of(row, run):
    """The network's rise, as maze_flat_return_full.measure computes it.

    Args:
        row: dict from maze_series (t_star); run: dict from maze_load_runs (progress).
    Returns:
        float, median return over the BLOCK updates up to the learned point minus median over the first BLOCK updates.
    """
    upd = np.array([p["update"] for p in run["progress"]], float)
    ret = np.array([p["discounted_return"] for p in run["progress"]], float)
    t = float(row["t_star"])
    return float(np.median(ret[(upd > t - BLOCK) & (upd <= t)]) - np.median(ret[upd <= BLOCK]))


def window_quantities(row, run, lo, hi, rise):
    """U and F of one network over the window (lo, hi].

    Args:
        row: dict from maze_series; run: dict from maze_load_runs; lo, hi: float updates; rise: float.
    Returns:
        dict: U (units lost per 100,000 Adam steps), F (return slope per 100,000 Adam steps over the rise),
        lost_per_1000 (units per 1,000 updates), active_first, active_last (medians of the window's first and last 5
        probes), success_last (the last probe's success in the window), n_probes, decreasing, flat.
    """
    pu, pa = np.asarray(row["it_act"], float), np.asarray(row["active"], float)
    ps = np.asarray(row["success"], float)
    m = (pu > lo) & (pu <= hi)
    slope_act = float(np.polyfit(pu[m], pa[m], 1)[0])            # units per update
    upd = np.array([p["update"] for p in run["progress"]], float)
    ret = np.array([p["discounted_return"] for p in run["progress"]], float)
    k = (upd > lo) & (upd <= hi)
    slope_ret = float(np.polyfit(upd[k], ret[k], 1)[0])           # return per update
    U, F = -slope_act * PER_100K, slope_ret * PER_100K / rise
    return dict(U=U, F=F, lost_per_1000=-slope_act * 1000.0, active_first=float(np.median(pa[m][:5])),
                active_last=float(np.median(pa[m][-5:])), success_last=float(ps[m][-1]), n_probes=int(m.sum()),
                n_updates=int(k.sum()), decreasing=bool(U >= DECREASE_BAR), flat=bool(abs(F) < FLAT_MATCHED))


def rule(flags, flats, label_yes="decreasing while the return is flat"):
    """The two-thirds rule over the flat networks.

    Args:
        flags: list of bool (decreasing) per read network; flats: list of bool (flat), same order.
    Returns:
        dict with n_flat, n_decreasing_among_flat, verdict ("no verdict" below MIN_FLAT flat networks).
    """
    dec = [d for d, f in zip(flags, flats) if f]
    v = three_way(dec, label_yes, "not decreasing") if len(dec) >= MIN_FLAT else \
        f"no verdict ({len(dec)} flat networks, needs {MIN_FLAT})"
    return dict(n_flat=len(dec), n_decreasing_among_flat=int(sum(dec)), verdict=v)


def main(root, out_path):
    """Read every Phase B network under root, print the tables and verdicts, write the JSON; returns None."""
    recs = phase_b_records(root)
    ends = {k: float(run["progress"][-1]["update"]) for k, (row, run) in recs.items()}
    n_full = sum(e >= END for e in ends.values())
    hi = END if n_full >= MIN_AT_END else END_SHORT
    print(f"{len(recs)} networks with a Phase B record (expected {N_NETS}); {n_full} reach update {END:.0f} -> window "
          f"{START + 1:.0f}-{hi:.0f}" + ("" if hi == END else f" (W', fewer than {MIN_AT_END} reach {END:.0f})"))
    nets, excluded = {}, {}
    for (n, s), (row, run) in sorted(recs.items()):
        label = f"N={n} s{s}"
        row, run = truncate(row, run, hi)
        rise = rise_of(row, run) if np.isfinite(row["t_star"]) else float("nan")
        pu, pa = np.asarray(row["it_act"], float), np.asarray(row["active"], float)
        w = dict(end=ends[(n, s)], rise=rise, chain=[c.split("Maze_suite/")[-1] for c in run["chain"]][-4:],
                 noise=noise(pu, pa), J_24000=jump(pu, pa, START), J_29000=jump(pu, pa, MID))
        if ends[(n, s)] < hi:
            excluded[label] = f"record ends at update {ends[(n, s)]:.0f}, before {hi:.0f}"
            nets[label] = w
            continue
        if not np.isfinite(rise):
            excluded[label] = f"no learned point in the record cut at {hi:.0f} (last probe success " \
                              f"{float(np.asarray(row['success'])[-1]):.2f}, needs {SUCCESS_LEVEL})"
            nets[label] = w
            continue
        w["W"] = window_quantities(row, run, START, hi, rise)
        f2 = measure(row, run, "B", FLAT_BAR, start=START + 1)
        w["P2_flat"], w["P2_D_over_rise"] = f2.get("flat"), f2.get("D_over_rise")
        w["P3_decreasing"] = bool(w["W"]["lost_per_1000"] >= OLD_DECREASE_BAR)
        if hi == END:
            w["stage14"] = window_quantities(row, run, MID, END, rise)
        nets[label] = w
    read = {k: w for k, w in nets.items() if k not in excluded}
    for k, why in excluded.items():
        print(f"  not read: {k}: {why}")

    print(f"\n{'network':<12} {'end':>6} {'active':>11} {'U/100k Adam':>11} {'decr':>5} {'F %/100k':>9} {'flat':>5} "
          f"{'P2 D/rise%/1k':>13} {'P2 flat':>7} {'succ':>5}")
    for k, w in nets.items():
        if "W" not in w:
            print(f"{k:<12} {w['end']:>6.0f} not read")
            continue
        q = w["W"]
        p2 = f"{100 * w['P2_D_over_rise']:>+13.2f}" if w["P2_D_over_rise"] is not None else f"{'n/t':>13}"
        print(f"{k:<12} {w['end']:>6.0f} {q['active_first']:>5.0f}->{q['active_last']:<5.0f} {q['U']:>11.1f} "
              f"{str(q['decreasing']):>5} {100 * q['F']:>+9.2f} {str(q['flat']):>5} {p2} {str(w['P2_flat']):>7} "
              f"{q['success_last']:>5.2f}")
    keys = list(read)
    res = {"window": [START + 1, hi], "n_reach_end": n_full, "excluded": excluded,
           "P1": rule([read[k]["W"]["decreasing"] for k in keys], [read[k]["W"]["flat"] for k in keys]),
           "P2": rule([read[k]["W"]["decreasing"] for k in keys], [bool(read[k]["P2_flat"]) for k in keys]),
           "P3": rule([read[k]["P3_decreasing"] for k in keys], [read[k]["W"]["flat"] for k in keys])}
    if hi == END:
        print(f"\n[P4] stage 14 alone (updates {MID + 1:.0f}-{END:.0f})")
        for k in keys:
            q = read[k]["stage14"]
            print(f"  {k:<12} U {q['U']:>7.1f} {'decr' if q['decreasing'] else '    '}  F {100 * q['F']:>+7.2f}% "
                  f"{'flat' if q['flat'] else ''}")
        res["P4"] = rule([read[k]["stage14"]["decreasing"] for k in keys], [read[k]["stage14"]["flat"] for k in keys])
    print(f"\n[P5] warm boundaries: J and the network's noise (Phase A definition)")
    over = {}
    for b in ("J_24000",) + (("J_29000",) if hi == END else ()):
        ex = [k for k, w in nets.items() if not np.isnan(w[b]) and abs(w[b]) > w["noise"]]
        over[b] = ex
        print(f"  {b}: above noise in {len(ex)} of {len(nets)} {ex}")
    for k, w in nets.items():
        print(f"  {k:<12} noise {w['noise']:>5.1f}  J(24,000) {w['J_24000']:>+6.1f}  J(29,000) {w['J_29000']:>+6.1f}")
    res["P5"] = dict(above_noise={b: v for b, v in over.items()},
                     disruption=bool(any(len(v) > MAX_DISRUPTED for v in over.values())))
    print()
    for r in ("P1", "P2", "P3", "P4", "P5"):
        if r in res:
            print(f"{r}: {res[r]}")
    res["networks"] = nets
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
