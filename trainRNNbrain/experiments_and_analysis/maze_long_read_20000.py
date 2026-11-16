"""Read the Maze networks continued from update 12,000 to 20,000 (rules: notebook entries
2026-11-14_1051_rules-maze-long-training-to-20000.md, readings A-C, and
2026-11-14_1107_rules-maze-flat-return-by-stage-and-stopped-networks.md, reading D and the resumed networks).
Written and committed before any record after update 12,000 was pulled to this machine.

maze_long_read.py (the 4,000-12,000 reading) is left as it was; this script takes its readings with the new
windows and adds readings C and D.

A  decrease over updates 12,000-20,000: least-squares slope of the noise-free probe's number of active units,
   and the medians of the last 10 probes up to 12,000 and up to 20,000; decreasing if at least 5 units are lost
   per 1,000 updates AND the later median is lower. Two-thirds rule over the learned networks.
B  flat return, last quarter (updates 15,000-20,000): maze_flat_return_full.measure, reading "B"; a verdict
   needs at least 4 flat networks.
C  flat success rate over updates 12,000-20,000: the training success rate is flat if the medians of the
   window's first and last 100 updates differ by less than 0.01; decreasing as in A; two-thirds rule over the
   networks with a flat success rate, at least 4 needed.
D  flat return by stage (stages 7-10, 2,000 updates each): the stage is flat if the least-squares slope of the
   per-update return, per 1,000 updates and divided by the network's rise, is below 0.03 in magnitude; it is
   decreasing if the least-squares slope of the number of active units over the stage's probes is a loss of
   at least 5 units per 1,000 updates. Verdict needs at least 8 flat stages from at least 4 networks.
   "decreasing while the return is flat" if at least two thirds of the flat stages decrease AND in at least
   two thirds of the networks with a flat stage at least half of that network's flat stages decrease;
   "not decreasing" if at most one third of the flat stages decrease; "mixed" otherwise.

Each verdict is computed twice: over the networks never stopped (run folder without "_resume_") and over all.

Usage (from trainRNNbrain/experiments_and_analysis):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python maze_long_read_20000.py
Writes data/maze_long_read_20000.json and prints one table per reading.
"""
import json
import os

import numpy as np

from common import maze_load_runs
from fig_paper_F2_size import MAZE_ROOT, maze_series
from maze_flat_return_full import BLOCK, DECREASE_BAR, FLAT_BAR, measure
from round18_maze import REPO, three_way

START, END = 12000.0, 20000.0       # readings A, C: the continued stretch
N_MEDIAN = 10                       # probes in each median of reading A
MIN_FLAT = 4                        # readings B, C: fewer flat networks give no verdict
SUCCESS_FLAT = 0.01                 # reading C: flat if the first and last 100-update medians differ by less
STAGES = [(12000.0, 14000.0), (14000.0, 16000.0), (16000.0, 18000.0), (18000.0, 20000.0)]   # reading D
MIN_FLAT_STAGES, MIN_FLAT_NETS = 8, 4                       # reading D: verdict thresholds
SIZES = (500, 1000)
OUT = os.path.join(REPO, "data", "maze_long_read_20000.json")


def continued_rows():
    """The continued networks whose record reaches update 20,000.

    Returns:
        list of (row, run, resumed): row from fig_paper_F2_size.maze_series (default ReLU networks of the sizes
        in SIZES with it_end >= END), run from common.maze_load_runs, resumed (bool, the chain's last folder
        is a resumed stage, named with "_resume_").
    """
    runs = {os.path.relpath(r["folder"], MAZE_ROOT): r for r in maze_load_runs(MAZE_ROOT)}
    rows = [r for r in maze_series() if r["N"] in SIZES and r["it_end"] >= END]
    return [(r, runs[r["name"]], "_resume_" in r["name"]) for r in rows]


def active_slope(row, lo, hi):
    """Units lost per 1,000 updates: minus the least-squares slope of the probe's active count on (lo, hi].

    Args:
        row: dict from maze_series (it_act, active); lo, hi: float, update bounds.
    Returns:
        float, units lost per 1,000 updates (positive when the count falls).
    """
    upd, act = np.asarray(row["it_act"], float), np.asarray(row["active"], float)
    m = (upd > lo) & (upd <= hi)
    return -float(np.polyfit(upd[m] / 1000.0, act[m], 1)[0])


def reading_a(row):
    """The change of the number of active units over updates 12,000 to 20,000 in one network.

    Args:
        row: dict from maze_series.
    Returns:
        dict with N, seed, learned, success_end, active_start, active_end (medians of N_MEDIAN probes),
        lost_per_1000, decreasing (bool).
    """
    upd, act = np.asarray(row["it_act"], float), np.asarray(row["active"], float)
    early = float(np.median(act[upd <= START][-N_MEDIAN:]))
    late = float(np.median(act[upd <= END][-N_MEDIAN:]))
    lost = active_slope(row, START - 1, END)
    return dict(N=int(row["N"]), seed=int(row["seed"]), learned=bool(row["learned"]),
                success_end=float(row["success"][-1]), active_start=early, active_end=late,
                lost_per_1000=lost, decreasing=bool(lost >= DECREASE_BAR and late < early))


def reading_c(run):
    """Whether the training success rate is flat over updates 12,000 to 20,000.

    Args:
        run: dict from maze_load_runs (progress: one dict per PPO update with update and success).
    Returns:
        dict with success_first, success_last (medians of the window's first and last BLOCK updates), flat (bool).
    """
    upd = np.array([p["update"] for p in run["progress"]], float)
    suc = np.array([p["success"] for p in run["progress"]], float)
    first = float(np.median(suc[(upd > START) & (upd <= START + BLOCK)]))
    last = float(np.median(suc[(upd > END - BLOCK) & (upd <= END)]))
    return dict(success_first=first, success_last=last, flat=bool(abs(last - first) < SUCCESS_FLAT))


def reading_d(row, run, rise):
    """The return slope and the active-unit slope of each stage after update 12,000.

    Args:
        row: dict from maze_series; run: dict from maze_load_runs; rise: float, the network's rise (measure()).
    Returns:
        list of dicts, one per stage: lo, hi, slope_over_rise (per 1,000 updates), flat (bool),
        lost_per_1000, decreasing (bool).
    """
    upd = np.array([p["update"] for p in run["progress"]], float)
    ret = np.array([p["discounted_return"] for p in run["progress"]], float)
    out = []
    for lo, hi in STAGES:
        m = (upd > lo) & (upd <= hi)
        s = float(np.polyfit(upd[m] / 1000.0, ret[m], 1)[0]) / rise
        lost = active_slope(row, lo, hi)
        out.append(dict(lo=lo, hi=hi, slope_over_rise=s, flat=bool(abs(s) < FLAT_BAR),
                        lost_per_1000=lost, decreasing=bool(lost >= DECREASE_BAR)))
    return out


def verdict_d(stages_by_net):
    """Reading D's verdict over a set of networks.

    Args:
        stages_by_net: dict network label -> list of stage dicts from reading_d.
    Returns:
        (n_flat_stages, n_nets_with_flat, n_flat_decreasing, verdict string).
    """
    flat = {k: [s for s in v if s["flat"]] for k, v in stages_by_net.items()}
    flat = {k: v for k, v in flat.items() if v}
    all_flat = [s for v in flat.values() for s in v]
    n_dec = sum(s["decreasing"] for s in all_flat)
    if len(all_flat) < MIN_FLAT_STAGES or len(flat) < MIN_FLAT_NETS:
        return len(all_flat), len(flat), n_dec, (f"no verdict (needs {MIN_FLAT_STAGES} flat stages from "
                                                 f"{MIN_FLAT_NETS} networks)")
    nets_ok = sum(2 * sum(s["decreasing"] for s in v) >= len(v) for v in flat.values())
    if 3 * n_dec >= 2 * len(all_flat) and 3 * nets_ok >= 2 * len(flat):
        v = "decreasing while the return is flat"
    elif 3 * n_dec <= len(all_flat):
        v = "not decreasing"
    else:
        v = "mixed"
    return len(all_flat), len(flat), n_dec, v


def verdicts(rows, label):
    """Apply readings A-D to one set of networks and print the verdicts.

    Args:
        rows: list of per-network dicts (keys A, B, C, D, label, resumed); label: str for the printout.
    Returns:
        dict of verdicts and counts per reading.
    """
    learned = [w for w in rows if w["A"]["learned"]]
    v_a = three_way([w["A"]["decreasing"] for w in learned], "keeps decreasing", "does not keep decreasing")
    b = [w for w in learned if w["B"]["tested"]]
    b_flat = [w for w in b if w["B"]["flat"]]
    v_b = (three_way([w["B"]["decreasing"] for w in b_flat], "decreasing while the return is flat", "not decreasing")
           if len(b_flat) >= MIN_FLAT else f"no verdict ({len(b_flat)} flat networks, needs {MIN_FLAT})")
    c_flat = [w for w in learned if w["C"]["flat"]]
    v_c = (three_way([w["A"]["decreasing"] for w in c_flat], "decreasing while the success rate is flat",
                     "not decreasing")
           if len(c_flat) >= MIN_FLAT else f"no verdict ({len(c_flat)} flat networks, needs {MIN_FLAT})")
    d = verdict_d({w["label"]: w["D"] for w in learned})
    print(f"\n[{label}] networks {len(rows)}, learned {len(learned)}")
    print(f"  A: decreasing {sum(w['A']['decreasing'] for w in learned)} of {len(learned)} -> {v_a}")
    print(f"  B: tested {len(b)}, flat {len(b_flat)}, decreasing among flat "
          f"{sum(w['B']['decreasing'] for w in b_flat)} -> {v_b}")
    print(f"  C: flat success {len(c_flat)}, decreasing among them {sum(w['A']['decreasing'] for w in c_flat)} -> {v_c}")
    print(f"  D: flat stages {d[0]} from {d[1]} networks, decreasing {d[2]} -> {d[3]}")
    return dict(n=len(rows), n_learned=len(learned), A=v_a, B=v_b, n_B_flat=len(b_flat), C=v_c,
                n_C_flat=len(c_flat), D=d[3], n_D_flat_stages=d[0], n_D_nets=d[1], n_D_decreasing=d[2])


def main():
    """Print every reading with its verdicts over never-stopped and all networks; write the JSON; returns None."""
    nets = []
    for row, run, resumed in continued_rows():
        b = measure(row, run, "B")
        nets.append(dict(label=f"N={row['N']} s{row['seed']}" + (" (resumed)" if resumed else ""), resumed=resumed,
                         A=reading_a(row), B=b, C=reading_c(run), D=reading_d(row, run, b["rise"])))
    print(f"{len(nets)} networks with a record to update {END:.0f} "
          f"({sum(w['resumed'] for w in nets)} resumed with the fixed trainer)")
    print(f"\n{'network':<22} {'learned':>7} {'succ end':>8} {'act 12k':>7} {'act 20k':>7} {'lost/1k':>7} {'A':>5} "
          f"{'B D/rise%':>9} {'B flat':>6} {'C first':>7} {'C last':>7} {'C flat':>6}")
    for w in nets:
        a, b, c = w["A"], w["B"], w["C"]
        bd = f"{100 * b['D_over_rise']:+9.2f}" if b["tested"] else f"{'n/t':>9}"
        print(f"{w['label']:<22} {str(a['learned']):>7} {a['success_end']:>8.2f} {a['active_start']:>7.0f} "
              f"{a['active_end']:>7.0f} {a['lost_per_1000']:>7.1f} {str(a['decreasing']):>5} {bd} "
              f"{str(b.get('flat', '-')):>6} {c['success_first']:>7.3f} {c['success_last']:>7.3f} {str(c['flat']):>6}")
    print(f"\n[D by stage] slope/rise % per 1,000 updates (F = flat) and units lost per 1,000 updates (d = decreasing)")
    for w in nets:
        cells = "  ".join(f"{100 * s['slope_over_rise']:+6.1f}{'F' if s['flat'] else ' '} {s['lost_per_1000']:5.1f}"
                          f"{'d' if s['decreasing'] else ' '}" for s in w["D"])
        print(f"  {w['label']:<22} {cells}")
    res = {"never_stopped": verdicts([w for w in nets if not w["resumed"]], "never stopped"),
           "all": verdicts(nets, "all networks"), "networks": nets}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
