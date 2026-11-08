#!/usr/bin/env python3
"""Round 18, the Maze analyses G4 (summary), G8, G9, G10, G13, G14 and G15 (dead-relu-paper#201).

RULES. docs/project_trajectory/2026-11-07_0100_round18-rules.md, committed before this script; each
subcommand below implements the rule of the same name and prints its verdict.

  g4   summary of rl_maze_ablate.py --added (data/round18/g4_maze_added): per network the success
       drop with the k added units removed (inactive units in place), k random active units, the
       570-unit form and its random counterpart; per technique the reading (depends if every drop >=
       0.10, does not depend if every drop < 0.05, else mixed); and the check of the 570-unit values
       against the interpolated values of data/rl_maze_ablation_interventions (|diff| <= 0.03).
  g8   flat training return after the learned point, learned default size-series networks.
  g9   fewer active units at the end than at the learned point, every arm of read_maze_arms.json.
  g10  the size exponent at N >= 500 (fig_paper_F2_size.power_fit / boot_exponent), learned point,
       end, and the count at update 2,000.
  g13  s.d. of the recurrent weights over all units against the active units (W1), N = 1,000 controls.
  g14  silent and low-active units on the clean probe and on the frozen rollout, N = 1,000 controls.
  g15  inactive units active 50 and 700 updates after the learned point, from the stored participation.

Every subcommand writes data/round18/<name>.json and prints its table.
Usage:  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python round18_maze.py {g4,g8,g9,g10,g13,g14,g15}
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import active_mask, maze_load_runs
from fig_paper_F2_size import BOOT_SEED, FIT_MIN_N, MAZE_ROOT, boot_exponent, maze_series, power_fit
from rl_maze_ablate import probe_participation
from trainRNNbrain.trainer.TrainerPPO import load_run

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "data", "round18")
MAZE_JSON = os.path.join(REPO, "data", "read_maze_arms.json")
G4_DIR = os.path.join(OUT, "g4_maze_added")
OLD_ABL = os.path.join(REPO, "data", "rl_maze_ablation_interventions")
CONTROLS = [os.path.join(MAZE_ROOT, "N=1000_act=relu", f"s{s}") for s in (1, 2, 3)]
FROZEN_SEED = 20261107           # the frozen rollout's drops, action noise (recurrent noise is off)
FROZEN_EPISODES = 256
SUBSET_SEED = 20261107           # G13: random subsets
SUBSETS = 200
REINIT_KEYS = ("redraw", "zeroout")


def save(name, obj):
    """Write data/round18/<name>.json.

    Args:
        name: str; obj: JSON-ready object.
    Returns:
        str, the path written.
    """
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    print(f"-> {path}")
    return path


def three_way(values, yes, no):
    """The two-thirds reading of the flat-loss test.

    Args:
        values: list of bool (True: the network shows the effect); yes, no: str verdicts.
    Returns:
        str: `yes` if at least two thirds are True, `no` if at least two thirds are False, "mixed"
        otherwise, "no network" for an empty list.
    """
    if not values:
        return "no network"
    n, t = len(values), sum(values)
    if 3 * t >= 2 * n:
        return yes
    if 3 * (n - t) >= 2 * n:
        return no
    return "mixed"


# ------------------------------------------------------------------------------------------------ G4

def interp_old(abl, n_remove):
    """The quietest-first and median random success of an old full-curve JSON at n_remove units removed.

    Args:
        abl: dict from rl_maze_ablate.ablate; n_remove: int.
    Returns:
        (quiet, random median) interpolated on the JSON's k grid.
    """
    ks = np.asarray(abl["ks"], float)
    q = float(np.interp(n_remove, ks, abl["score_quiet"]))
    rnd = float(np.median([np.interp(n_remove, ks, r) for r in abl["score_random"]]))
    return q, rnd


def g4():
    """G4 summary: the added-unit removals of every technique network and the reading per technique."""
    rows = []
    for path in sorted(glob.glob(os.path.join(G4_DIR, "*.json"))):
        r = json.load(open(path))
        cell = os.path.basename(path).split("__")[0]
        drop = r["score_full"] - r["score_added"] if "score_added" in r else None
        row = dict(cell=cell, file=os.path.basename(path), n_active=r["n_active"], k=r["k"], intact=r["score_full"],
                   no_inactive=r["score_no_inactive"], added=r.get("score_added"),
                   random_active=float(np.median(r["score_random_active"])) if "score_random_active" in r else None,
                   keep=r["score_keep_ctrl"], random_keep=float(np.median(r["score_random_keep"])),
                   n_keep=r["n_remove_keep"], drop=drop)
        old = os.path.join(OLD_ABL, os.path.basename(path))
        if os.path.exists(old):
            q, rnd = interp_old(json.load(open(old)), r["n_remove_keep"])
            row.update(old_keep_interp=q, old_random_interp=rnd, check_ok=abs(q - r["score_keep_ctrl"]) <= 0.03)
        rows.append(row)
    print(f"{'network':>16} {'act':>4} {'k':>4} {'intact':>6} {'-inact':>6} {'-k add':>6} {'-k rnd':>6} "
          f"{'drop':>6} {'-keep':>6} {'-rndN':>6}  check")
    for w in rows:
        chk = "" if "check_ok" not in w else f"old interp {w['old_keep_interp']:.3f} ok={w['check_ok']}"
        f = lambda v: "  --  " if v is None else f"{v:6.3f}"
        print(f"{w['file'][:-5]:>16} {w['n_active']:>4} {w['k']:>4} {w['intact']:6.3f} {w['no_inactive']:6.3f} "
              f"{f(w['added'])} {f(w['random_active'])} {f(w['drop'])} {w['keep']:6.3f} {w['random_keep']:6.3f}  {chk}")
    tech = {}
    for cell in sorted({w["cell"] for w in rows}):
        drops = [w["drop"] for w in rows if w["cell"] == cell and w["drop"] is not None]
        v = ("depends" if drops and all(d >= 0.10 for d in drops) else
             "does not depend" if drops and all(d < 0.05 for d in drops) else "mixed")
        further = [w["random_active"] < w["added"] for w in rows if w["cell"] == cell and w["added"] is not None]
        tech[cell] = dict(drops=drops, verdict=v, random_lower_in=f"{sum(further)} of {len(further)}")
        print(f"  {cell}: drops {np.round(drops, 3).tolist()} -> {v}; k random active lower than k added in "
              f"{sum(further)} of {len(further)}")
    rate = [w["drop"] for w in rows if w["cell"].startswith("frm") and w["drop"] is not None]
    rv = ("depends" if rate and all(d >= 0.10 for d in rate) else
          "does not depend" if rate and all(d < 0.05 for d in rate) else "mixed")
    print(f"  rate penalty, all values pooled: {rv}")
    save("g4_maze_added_summary", dict(rows=rows, techniques=tech, rate_pooled=rv))


# ------------------------------------------------------------------------------------------------ G8

def learned_default_runs():
    """The learned default size-series runs, joined along their chains, with their maze_series row.

    Returns:
        list of (row, run) pairs: row from fig_paper_F2_size.maze_series, run from common.maze_load_runs.
    """
    rows = {r["name"]: r for r in maze_series() if r["learned"]}
    runs = {os.path.relpath(r["folder"], MAZE_ROOT): r for r in maze_load_runs(MAZE_ROOT)}
    return [(rows[n], runs[n]) for n in sorted(rows, key=lambda n: (rows[n]["N"], rows[n]["seed"]))]


def junctions(run):
    """Updates at which a continuation run of the chain starts (Adam restarts there with a warm-up).

    Args:
        run: dict from maze_load_runs.
    Returns:
        list of int, the first update of every chain member after the first.
    """
    out = []
    for d in run["chain"][1:]:
        prog = json.load(open(sorted(glob.glob(os.path.join(d, "*Progress.json")))[0]))
        out.append(int(prog[0]["update"]))
    return out


def flat_return(row, run, drop_after_junction=0):
    """G8 for one network: the return change after the learned point and the active-unit slope.

    Args:
        row: maze_series row; run: maze_load_runs dict; drop_after_junction: int, updates left out
            after each chain junction (0 in the main read, 100 in the sensitivity line).
    Returns:
        dict with learned, end, rise, D_per_1000, flat (bool or None when the window is shorter than
        300 updates), lost_per_1000, decreasing (bool).
    """
    upd = np.array([p["update"] for p in run["progress"]], float)
    ret = np.array([p["discounted_return"] for p in run["progress"]], float)
    keep = np.ones(len(upd), bool)
    for j in junctions(run):
        keep &= ~((upd >= j) & (upd < j + drop_after_junction))
    upd, ret = upd[keep], ret[keep]
    tl, end = float(row["t_star"]), float(upd[-1])
    b0 = np.median(ret[upd <= 100])
    bl = np.median(ret[(upd > tl - 100) & (upd <= tl)])
    win = upd >= tl
    out = dict(N=row["N"], seed=row["seed"], learned=tl, end=end, window=end - tl, rise=float(bl - b0),
               junctions=junctions(run))
    if end - tl < 300:
        out.update(flat=None, D_per_1000=None)
    else:
        first = (upd >= tl) & (upd < tl + 100)
        last = upd > end - 100
        dist = (np.mean(upd[last]) - np.mean(upd[first])) / 1000.0
        D = (np.median(ret[last]) - np.median(ret[first])) / dist
        out.update(D_per_1000=float(D), flat=bool(abs(D) < 0.03 * out["rise"]),
                   first_block=float(np.median(ret[first])), last_block=float(np.median(ret[last])))
    pu, pa = np.asarray(row["it_act"], float), np.asarray(row["active"], float)
    m = pu >= tl
    for j in junctions(run):
        m &= ~((pu >= j) & (pu < j + drop_after_junction))
    x, y = pu[m] / 1000.0, pa[m]
    slope = np.polyfit(x, y, 1)[0] if len(x) >= 3 else float("nan")
    out.update(lost_per_1000=float(-slope), decreasing=bool(-slope >= 5.0))
    return out


def g8():
    """G8: flat training return after the learned point, and the active-unit slope there."""
    res = {}
    for tag, drop in (("main", 0), ("junction_100_dropped", 100)):
        rows = [flat_return(row, run, drop) for row, run in learned_default_runs()]
        print(f"\n[{tag}] {'N':>5} {'seed':>4} {'learned':>7} {'end':>5} {'rise':>7} {'D/1000':>8} {'flat':>5} "
              f"{'lost/1000':>9} {'decr':>5} junctions")
        for w in rows:
            D = "--" if w["D_per_1000"] is None else f"{w['D_per_1000']:+.3f}"
            print(f"       {w['N']:>5} {w['seed']:>4} {w['learned']:>7.0f} {w['end']:>5.0f} {w['rise']:>7.3f} {D:>8} "
                  f"{str(w['flat']):>5} {w['lost_per_1000']:>9.1f} {str(w['decreasing']):>5} {w['junctions']}")
        flat = [w for w in rows if w["flat"]]
        v = three_way([w["decreasing"] for w in flat], "decreasing while the return is flat", "not decreasing")
        print(f"  [{tag}] tested {sum(w['flat'] is not None for w in rows)} of {len(rows)}; flat {len(flat)}; "
              f"decreasing among flat {sum(w['decreasing'] for w in flat)} -> {v}")
        res[tag] = dict(rows=rows, n_flat=len(flat), n_decreasing=sum(w["decreasing"] for w in flat), verdict=v)
    save("g8_maze_flat_return", res)


# ------------------------------------------------------------------------------------------------ G9

def g9():
    """G9: active units at the end against the learned point, every arm with a learned point."""
    d = json.load(open(MAZE_JSON))
    rows = []
    for r in d["runs"]:
        if r["learned"] is None or r["active_learned"] is None:
            continue
        group = ("counted" if r["cls"] in ("control", "setting") or r["key"].startswith(REINIT_KEYS)
                 else "technique")
        rows.append(dict(key=r["key"], cls=r["cls"], seed=r["seed"], group=group, learned=r["learned"],
                         active_learned=r["active_learned"], active_end=r["active_end"],
                         diff=r["active_end"] - r["active_learned"]))
    for grp in ("counted", "technique"):
        sub = [w for w in rows if w["group"] == grp]
        fewer = [w for w in sub if w["diff"] < 0]
        print(f"\n{grp}: {len(fewer)} of {len(sub)} have fewer active units at the end")
        for w in sorted(sub, key=lambda w: (w["key"], w["seed"])):
            print(f"   {w['key']:>16} s{w['seed']} learned {w['learned']:>5} {w['active_learned']:>4} -> "
                  f"{w['active_end']:>4} ({w['diff']:+d})")
    sub = [w for w in rows if w["group"] == "counted"]
    n_fewer = sum(w["diff"] < 0 for w in sub)
    v = "extends to Maze" if 2 * n_fewer > len(sub) else "Maze differs"
    print(f"-> counted networks: {n_fewer} of {len(sub)} fewer at the end: {v}")
    save("g9_maze_learned_vs_end", dict(rows=rows, n_counted=len(sub), n_fewer=n_fewer, verdict=v))


# ------------------------------------------------------------------------------------------------ G10

def g10():
    """G10: the Maze size exponent at N >= 500 at the learned point, the end and update 2,000."""
    rows = maze_series()
    learned = [r for r in rows if r["learned"]]
    res = {}
    for at, key in (("learned", "active_learned"), ("end", "active_end"), ("update 2,000", None)):
        cells = {}
        for r in learned:
            if key is None:
                it = np.asarray(r["it_act"], float)
                j = int(np.argmin(np.abs(it - 2000)))
                v = float(r["active"][j])
            else:
                v = float(r[key])
            cells.setdefault(r["N"], []).append(v)
        cells = {N: np.array(v) for N, v in sorted(cells.items())}
        fit = {}
        for lab, mn in (("N>=500", FIT_MIN_N), ("every size", 0)):
            b, A = power_fit(cells, min_n=mn)
            lo, hi = boot_exponent({N: v for N, v in cells.items() if N >= mn}, np.random.default_rng(BOOT_SEED), mn)
            fit[lab] = dict(b=b, lo=float(lo), hi=float(hi))
        res[at] = dict(cells={str(N): v.tolist() for N, v in cells.items()}, fit=fit,
                       ends={str(r["N"]): r["it_end"] for r in learned})
        print(f"{at:>13}: " + "; ".join(f"N={N}: n={len(v)} mean {v.mean():.0f}" for N, v in cells.items()))
        for lab, f in fit.items():
            print(f"{'':>15}{lab:>11}: b = {f['b']:.3f}, bootstrap 95% [{f['lo']:.3f}, {f['hi']:.3f}]")
    budgets = sorted({(r["N"], r["it_end"]) for r in learned})
    print(f"last update per learned network (N, update): {budgets}")
    ok = all(res[a]["fit"]["N>=500"]["hi"] < 1 for a in ("learned", "end"))
    print(f"-> every N >= 500 upper end below 1 (learned, end): {ok}")
    res["budgets"] = budgets
    res["upper_below_one"] = ok
    save("g10_maze_exponent", res)


# ------------------------------------------------------------------------------------------------ G13 / G14

def load_control(folder):
    """Rebuild a Maze network and its clean-probe participation.

    Args:
        folder: str, a run folder.
    Returns:
        (task, trainer, T, p (N,) participation, active (N,) bool, clean probe success).
    """
    cfg, task, trainer = load_run(folder, torch.device("cpu"))
    T = trainer.T
    p, succ = probe_participation(task, trainer, T, 128, trainer.probe_seed)
    return task, trainer, T, p, active_mask(p), succ


def g13():
    """G13: W1, the s.d. of the recurrent weights over all units against the active units."""
    ss = np.random.SeedSequence(SUBSET_SEED).spawn(len(CONTROLS))
    rows = []
    for folder, sq in zip(CONTROLS, ss):
        _, trainer, _, p, act, succ = load_control(folder)
        W = trainer.RNN.W_rec.detach().cpu().numpy()
        A = np.flatnonzero(act)
        s_all, s_act = float(W.std()), float(W[np.ix_(A, A)].std())
        rng = np.random.default_rng(sq)
        N = W.shape[0]
        sr = [float(W[np.ix_(R, R)].std()) for R in (rng.choice(N, len(A), replace=False) for _ in range(SUBSETS))]
        rel = (s_all - s_act) / s_act
        differs = bool(abs(rel) > 0.10 and not (min(sr) <= s_act <= max(sr)))
        rows.append(dict(folder=folder, n_active=int(len(A)), success=succ, s_all=s_all, s_act=s_act, rel=rel,
                         subset_min=min(sr), subset_max=max(sr), differs=differs))
        print(f"{folder[-22:]}: {len(A)} active (probe success {succ:.2f}); s.d. all {s_all:.4f}, active {s_act:.4f}, "
              f"rel {100 * rel:+.0f}%, random subsets [{min(sr):.4f}, {max(sr):.4f}] -> differs {differs}")
    n = sum(w["differs"] and w["rel"] < 0 for w in rows)
    v = "extends to Maze" if n == len(rows) else ("scoped to 3BFF and CDDM" if not any(w["differs"] for w in rows)
                                                  else "mixed")
    print(f"-> {n} of {len(rows)} differ with S_all < S_act: {v}")
    save("g13_maze_weight_spread", dict(rows=rows, verdict=v))


def silent_on(trainer, T, out, inactive):
    """Silent units of one noise-free rollout: net input below zero on every RNN step of every episode.

    Args:
        trainer: TrainerPPO; T: int, decisions; out: a rollout dict (its u and noise); inactive: (N,) bool.
    Returns:
        (silent (N,) bool, restricted to `inactive`; max net input per unit over the steps used).
    """
    k = trainer.substeps
    with torch.no_grad():
        states = trainer.sequence(out["u"], out["noise"], alive=out["alive"])[0][:, :T * k]
    alive = out["alive"][:-1].bool().repeat_interleave(k, dim=0)
    x = states[:, alive].cpu().numpy()                     # (N, steps while the episode runs, episodes)
    xmax = x.reshape(x.shape[0], -1).max(axis=1)
    return (xmax < 0) & inactive, xmax


def g14():
    """G14: silent and low-active units on the clean probe and on the frozen rollout."""
    rows = []
    for folder in CONTROLS:
        task, trainer, T, p, act, succ = load_control(folder)
        g = torch.Generator(device=task.device)
        g.manual_seed(int(trainer.probe_seed))
        clean = trainer.rollout(128, T, clean=True, generator=g)
        s_clean, _ = silent_on(trainer, T, clean, ~act)
        draw = trainer.draw_noise
        trainer.draw_noise = lambda T_, B, w_noise, generator=None: draw(T_, B, False, generator)   # noise off
        g2 = torch.Generator(device=task.device)
        g2.manual_seed(FROZEN_SEED)
        frozen = trainer.rollout(FROZEN_EPISODES, T, clean=False, generator=g2)    # actions sampled, s.d. action_std
        trainer.draw_noise = draw
        assert float(frozen["noise"]["rec"].abs().max()) == 0.0, "the frozen rollout must have no recurrent noise"
        s_frozen, _ = silent_on(trainer, T, frozen, ~act)
        n_in = int((~act).sum())
        rows.append(dict(folder=folder, success=succ, n_active=int(act.sum()), n_inactive=n_in,
                         silent_clean=int(s_clean.sum()), silent_frozen=int(s_frozen.sum()),
                         silent_both=int((s_clean & s_frozen).sum()), frozen_success=float(frozen["success"].float().mean()),
                         action_std=trainer.action_std))
        print(f"{folder[-22:]}: active {int(act.sum())}, inactive {n_in}; silent on the clean probe {int(s_clean.sum())}, "
              f"on the frozen rollout {int(s_frozen.sum())} (both {int((s_clean & s_frozen).sum())}); low-active on the "
              f"frozen rollout {n_in - int(s_frozen.sum())}; frozen-rollout success {rows[-1]['frozen_success']:.2f}")
    if all(w["silent_frozen"] >= 1 and w["silent_frozen"] >= 0.10 * w["n_inactive"] for w in rows):
        v = "with CDDM (silent units present)"
    elif all(w["silent_frozen"] == 0 for w in rows):
        v = "with 3BFF (no silent unit)"
    else:
        v = "intermediate"
    print(f"-> {v}")
    save("g14_maze_silent", dict(rows=rows, verdict=v, frozen_seed=FROZEN_SEED, episodes=FROZEN_EPISODES))


# ------------------------------------------------------------------------------------------------ G15

def g15():
    """G15: inactive units at the learned point that are active 50 and 700 updates later (logs only)."""
    rows = []
    for row, run in learned_default_runs():
        pu, P = np.asarray(run["probes"]["p_update"], float), np.asarray(run["probes"]["p"], float)
        i0 = int(np.flatnonzero(pu >= row["t_star"])[0])
        ref = ~active_mask(P[i0])
        w = dict(N=row["N"], seed=row["seed"], learned=row["t_star"], ref_update=float(pu[i0]),
                 n_inactive=int(ref.sum()), last_vector=float(pu[-1]))
        for lag in (50, 700):
            j = np.flatnonzero(np.isclose(pu, pu[i0] + lag))
            if len(j) == 0:
                w[f"later_{lag}"] = None
                continue
            back = active_mask(P[j[0]]) & ref
            w[f"later_{lag}"] = dict(update=float(pu[j[0]]), n_active_again=int(back.sum()),
                                     fraction=float(back.sum() / max(ref.sum(), 1)))
        rows.append(w)
        f = lambda x: "not reached" if x is None else f"{x['n_active_again']} ({100 * x['fraction']:.1f}%) at {x['update']:.0f}"
        print(f"N={row['N']:>5} s{row['seed']}: learned {row['t_star']:.0f}, reference vector {pu[i0]:.0f}, "
              f"{int(ref.sum())} inactive; active again +50: {f(w['later_50'])}; +700: {f(w['later_700'])}")
    counted = [w for w in rows if w["later_700"] is not None]
    fr = [w["later_700"]["fraction"] for w in counted]
    if counted and all(x < 0.10 for x in fr):
        v = "inactive units do not become active again (all below a tenth)"
    elif counted and 2 * sum(x >= 0.10 for x in fr) > len(fr):
        v = "contradicted"
    else:
        v = "mixed" if counted else "no network reaches +700"
    print(f"-> counted {len(counted)} of {len(rows)}: {v}")
    save("g15_maze_noreturn", dict(rows=rows, verdict=v))


def main():
    """Run one subcommand."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["g4", "g8", "g9", "g10", "g13", "g14", "g15"])
    a = ap.parse_args()
    globals()[a.what]()


if __name__ == "__main__":
    main()
