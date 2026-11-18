"""Read the Maze warm-start test: readings R1-R5 of notebook entry 2026-11-17_2249_rules-maze-warm-start-test.md.
Written and committed before any record of the test was pulled to this machine; the thresholds are the rules'.

Arms (from update 20,000 of the nine continued networks): warm = Adam state reloaded (stages 11 and 12, updates
20,001-24,000), cold = Adam restarted with the 50-update ramp (stage 11, updates 20,001-22,000).

Jump J(U): median active count of the first 5 probes after update U minus the median of the last 5 probes up to U.
Noise of a network: the largest |J| over the non-boundary points 18,250, 18,500, ..., 19,750 of the parent stage.
  R1  warm start not disruptive: |J_warm(20,000)| <= noise in >= 7 of 9 AND |J_warm(22,000)| <= noise in >= 7 of 9.
  R2  cold restart disruptive (replication): J_cold(20,000) > noise in >= 5 of 9.
  R3  paired: J_warm(20,000) < J_cold(20,000) in >= 7 of 9.
  R4  performance: warm probe success >= 0.9 at every probe of updates 20,001-20,500, in every network.
  R5  descriptive: warm arm over 20,001-24,000, units lost per 100,000 Adam steps (1 update = 8 Adam steps) from a
      least-squares line, against 5 per 100,000; and maze_flat_return_full.measure over the same window, 3% bar.
A network whose arm does not reach the probes a reading needs does not meet that reading's condition (counted).

Usage (from trainRNNbrain/experiments_and_analysis):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python maze_warm_start_read.py <suite root> <out.json>
The root holds the warm and cold cells of trained_RNNs_rl_maze_warmtest and every earlier stage of the nine
networks (links into data/trained_RNNs_rl_maze/Maze_suite), so that common.maze_load_runs joins each arm into one
record from update 0 (the flat-return test needs the rise up to the learned point). Partial records (stages still
running) are read as far as they go; a reading that needs probes beyond them is reported as pending.
"""
import json
import sys

import numpy as np

from fig_paper_F2_size import maze_series
from common import maze_load_runs
from maze_flat_return_full import FLAT_BAR, measure

B0, B1, END_WARM = 20000.0, 22000.0, 24000.0       # the two warm boundaries and the end of the warm arm
NOISE_POINTS = np.arange(18250.0, 19750.0 + 1, 250.0)   # non-boundary points of the parent stage (7)
N_SIDE = 5                                          # probes on each side of a jump
R1_MIN, R2_MIN, R3_MIN = 7, 5, 7                    # pass counts out of 9
R4_LEVEL, R4_END = 0.9, 20500.0                     # R4: probe success level and the last update checked
ADAM_PER_UPDATE = 8                                 # 4 epochs x 2 minibatches
SUPERVISED_BAR = 5.0                                # units per 100,000 Adam steps (the supervised flat-loss bar)
N_NETS = 9


def jump(upd, act, u):
    """The change of the active count across update u.

    Args:
        upd, act: (n_probes,) float arrays, probe updates (increasing) and active counts.
        u: float, the update.
    Returns:
        float, median of the first N_SIDE probes after u minus median of the last N_SIDE probes up to u; nan if
        fewer than N_SIDE probes exist on either side.
    """
    after, before = act[upd > u][:N_SIDE], act[upd <= u][-N_SIDE:]
    if len(after) < N_SIDE or len(before) < N_SIDE:
        return float("nan")
    return float(np.median(after) - np.median(before))


def noise(upd, act):
    """The network's jump noise: the largest |J| over NOISE_POINTS (all inside the parent stage)."""
    return float(max(abs(jump(upd, act, p)) for p in NOISE_POINTS))


def arm_records(root):
    """The warm and the cold record of every network under root.

    Args:
        root: str, the suite folder.
    Returns:
        dict (N, seed) -> {"warm": (row, run) or None, "cold": (row, run) or None}; row from
        fig_paper_F2_size.maze_series, run from common.maze_load_runs, matched by folder.
    """
    runs = {r["folder"]: r for r in maze_load_runs(root)}
    rows = {r["name"]: r for r in maze_series(root)}
    out = {}
    for folder, run in runs.items():
        arm = "warm" if "_warm_cont" in folder else "cold" if "_cold_cont" in folder else None
        if arm is None:
            continue
        name = "/".join(folder.split("/")[-2:])
        out.setdefault((run["N"], run["seed"]), {"warm": None, "cold": None})[arm] = (rows[name], run)
    return out


def read_network(arms):
    """Every quantity of R1-R5 for one network.

    Args:
        arms: {"warm": (row, run) or None, "cold": (row, run) or None} from arm_records.
    Returns:
        dict: warm_end, cold_end (last recorded update, nan when absent), noise, J_warm_20000, J_warm_22000,
        J_cold_20000 (nan when not yet readable), before_equal (the parent's last N_SIDE probes agree between the
        arms), r4_min_success and r4_n_probes (probes of 20,001-20,500 in the warm arm), and the R5 quantities
        when the warm arm reaches 24,000: lost_per_1000_updates, lost_per_100k_adam, decreasing_vs_supervised,
        flat (measure over 20,001-24,000) with D_over_rise and rise.
    """
    out = {}
    probes = {}
    for arm in ("warm", "cold"):
        if arms[arm] is None:
            out[f"{arm}_end"] = float("nan")
            continue
        row, run = arms[arm]
        probes[arm] = (np.asarray(row["it_act"], float), np.asarray(row["active"], float),
                       np.asarray(row["success"], float))
        out[f"{arm}_end"] = float(run["progress"][-1]["update"])
    ref = probes.get("warm", probes.get("cold"))
    out["noise"] = noise(ref[0], ref[1]) if ref is not None else float("nan")
    for arm, u in (("warm", B0), ("warm", B1), ("cold", B0)):
        out[f"J_{arm}_{int(u)}"] = jump(*probes[arm][:2], u) if arm in probes else float("nan")
    if "warm" in probes and "cold" in probes:
        bw = probes["warm"][1][probes["warm"][0] <= B0][-N_SIDE:]
        bc = probes["cold"][1][probes["cold"][0] <= B0][-N_SIDE:]
        out["before_equal"] = bool(np.array_equal(bw, bc))
    if "warm" in probes:
        upd, _, suc = probes["warm"]
        m = (upd > B0) & (upd <= R4_END)
        out["r4_n_probes"] = int(m.sum())
        out["r4_min_success"] = float(suc[m].min()) if m.any() else float("nan")
    if "warm" in probes and out["warm_end"] >= END_WARM:
        row, run = arms["warm"]
        upd, act, _ = probes["warm"]
        m = (upd > B0) & (upd <= END_WARM)
        lost = -float(np.polyfit(upd[m] / 1000.0, act[m], 1)[0])
        per_100k = lost * 100.0 / ADAM_PER_UPDATE      # 1,000 updates = 8,000 Adam steps
        f = measure(row, run, "B", FLAT_BAR, start=B0 + 1)
        out.update(lost_per_1000_updates=lost, lost_per_100k_adam=per_100k,
                   decreasing_vs_supervised=bool(per_100k >= SUPERVISED_BAR),
                   active_first=float(np.median(act[m][:N_SIDE])), active_last=float(np.median(act[m][-N_SIDE:])),
                   flat_tested=f["tested"], flat=f.get("flat"), D_over_rise=f.get("D_over_rise"), rise=f["rise"],
                   measure_lost_per_1000=f.get("lost_per_1000"))
    return out


def verdicts(nets):
    """R1-R4 verdicts over the networks (R5 is descriptive).

    Args:
        nets: dict label -> dict from read_network.
    Returns:
        dict with, per reading, the count meeting its condition, the count still pending, and pass / fail /
        pending (pending when the readable networks alone cannot decide it).
    """
    def count(cond):
        """(met, pending): networks meeting cond, networks whose quantity is nan; cond takes a network dict."""
        met = sum(1 for w in nets.values() if cond(w) is True)
        pend = sum(1 for w in nets.values() if cond(w) is None) + (N_NETS - len(nets))
        return met, pend

    def leq_noise(key):
        """Condition |J| <= noise on one jump key; None when the jump is not readable yet."""
        return lambda w: None if np.isnan(w[key]) else bool(abs(w[key]) <= w["noise"])

    def decide(met, pend, need):
        """pass if met >= need, fail if met + pend < need, else pending."""
        return "pass" if met >= need else "fail" if met + pend < need else "pending"

    res = {}
    a = count(leq_noise("J_warm_20000"))
    b = count(leq_noise("J_warm_22000"))
    r1a, r1b = decide(*a, R1_MIN), decide(*b, R1_MIN)
    res["R1"] = dict(met_20000=a[0], pending_20000=a[1], met_22000=b[0], pending_22000=b[1],
                     verdict="pass" if r1a == r1b == "pass" else "fail" if "fail" in (r1a, r1b) else "pending")
    c = count(lambda w: None if np.isnan(w["J_cold_20000"]) else bool(w["J_cold_20000"] > w["noise"]))
    res["R2"] = dict(met=c[0], pending=c[1], verdict=decide(*c, R2_MIN))
    d = count(lambda w: None if np.isnan(w["J_warm_20000"]) or np.isnan(w["J_cold_20000"])
              else bool(w["J_warm_20000"] < w["J_cold_20000"]))
    res["R3"] = dict(met=d[0], pending=d[1], verdict=decide(*d, R3_MIN))
    # R4: every network; a network is readable once its warm arm has passed update R4_END
    def r4(w):
        """True/False once the warm arm's record passes R4_END, else None."""
        if np.isnan(w["warm_end"]) or w["warm_end"] < R4_END:
            return None
        return bool(w["r4_min_success"] >= R4_LEVEL)
    e = count(r4)
    res["R4"] = dict(met=e[0], pending=e[1], verdict=decide(*e, N_NETS))
    return res


def main(root, out_path):
    """Read every network under root, print the tables and verdicts, write the JSON; returns None."""
    recs = arm_records(root)
    nets = {f"N={n} s{s}": read_network(recs[(n, s)]) for n, s in sorted(recs)}
    print(f"{len(nets)} networks with a warm or cold record (expected {N_NETS})")
    print(f"\n{'network':<12} {'warm end':>8} {'cold end':>8} {'noise':>6} {'Jw20k':>6} {'Jw22k':>6} {'Jc20k':>6} "
          f"{'same before':>11} {'R4 min succ':>11} {'R4 probes':>9}")
    for k, w in nets.items():
        print(f"{k:<12} {w['warm_end']:>8.0f} {w['cold_end']:>8.0f} {w['noise']:>6.1f} {w['J_warm_20000']:>6.1f} "
              f"{w['J_warm_22000']:>6.1f} {w['J_cold_20000']:>6.1f} {str(w.get('before_equal', '-')):>11} "
              f"{w.get('r4_min_success', float('nan')):>11.3f} {w.get('r4_n_probes', 0):>9}")
    print("\n[R5, descriptive] warm arm, updates 20,001-24,000")
    print(f"{'network':<12} {'active':>11} {'lost/1k upd':>11} {'lost/100k Adam':>14} {'>= 5':>5} {'D/rise %':>9} "
          f"{'flat':>5}")
    for k, w in nets.items():
        if "lost_per_100k_adam" not in w:
            print(f"{k:<12} pending (warm arm ends at {w['warm_end']:.0f})")
            continue
        dr = f"{100 * w['D_over_rise']:>+9.2f}" if w["flat_tested"] else f"{'n/t':>9}"
        print(f"{k:<12} {w['active_first']:>5.0f}->{w['active_last']:<5.0f} {w['lost_per_1000_updates']:>11.2f} "
              f"{w['lost_per_100k_adam']:>14.1f} {str(w['decreasing_vs_supervised']):>5} {dr} {str(w['flat']):>5}")
    v = verdicts(nets)
    print()
    for r, d in v.items():
        print(f"{r}: {d}")
    with open(out_path, "w") as f:
        json.dump({"verdicts": v, "networks": nets}, f, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
