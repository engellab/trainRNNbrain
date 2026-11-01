#!/usr/bin/env python3
"""Manuscript Figure 2 (size and training time) and Extended Data Fig. 3 (the read-out rule).

Figure 2: a larger network keeps a smaller share of its units active, and training past the loss
floor makes more units inactive.
  (a) active units against network size N (100-4000), three tasks, one point per run, each run read
      where its noisy training loss reaches 1.05x its own fitted floor (common.read_run); power laws
      fitted to the size means from N = 500 (N = 100 drawn, not fitted).
  (b) the same runs as a percentage of N, guides at 50% and 10%.
  (c) the fitted exponent at the 1.05x, 1.10x and 1.20x margins, with a bootstrap 95% interval
      (runs resampled within each size).
  (d) the noisy training loss over its own start and the inactive-unit count against iteration,
      every run at N = 1000, one column per task; dashed lines at each run's own 1.05x crossing.
  (e) units made inactive per 100,000 iterations from 150,000 iterations on, against the change of
      the loss over the same window, every 3BFF run trained to 200,000 or 300,000 iterations
      (c3_flat_loss_units: loss_change, loss_rate).
Extended Data Fig. 3: (a) the read-out on one N = 1000 run per task (raw loss, Hampel curve, fitted
floor, bars, crossings, bursts); (b) the 1.05x crossing against N, every run of panel (a);
(c) the rule's crossing against the crossing of a snapshot oracle (each kept weight snapshot scored on
fresh batches under training noise, /scratch/gpfs/TENGEL/pt1290/oracle_readout, Della job 15635966,
entry 2026-10-15_2054), every run whose snapshots were scored.
The collect stage also writes the rows of Extended Data Table 2 (every run of Figures 1 and 2).

TWO STAGES, because the networks live on Della:
  --collect   on a Della CPU node, from the checkout root (data/ links the trained networks): reads
              every run and writes data/fig_paper_F2_size_collect.pkl. Held-out R^2 comes from
              heldout_r2 (36 noise draws); the shared cache is read but never written, new scores go
              to data/fig_paper_F2_size_collect_r2_cache.json.
  (no flag)   on the Mac: draws both figures from that record into img/internal_figures/
              (fig_paper_F2_size, fig_ed_readout; PDF + SVG), prints every number drawn, checks text
              overlap, and writes the table body img/internal_figures/edt2_runs_rows.tex.

READ-OUT (Pavel, 16 Oct; docs/AGENT_GUIDE.md "Reading networks"). Across sizes every run is read at
1.05x its own noisy-loss floor, because the learning rate scales as N^(-1/3) and larger networks
converge at different times. A series is drawn at the floor only if every size has at least one
read-out (otherwise it would be drawn at the end of training and labelled; it does not happen here).
Runs without a read-out are counted and listed with their reason in Extended Data Table 2.

Usage:  python fig_paper_F2_size.py --collect      (Della)
        python fig_paper_F2_size.py                (Mac)
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
import heldout_r2 as heldout
from common import (DATA_DIR, FLOOR_DELTAS, HOLD_FRAC, IMG_DIR, floor_curve, hampel_curve,
                    HAMPEL_SPAN, logbin, read_run, runs_with_loss, warm_start_chain)
from c3_flat_loss_units import CELLS as C3_CELLS, T_FLAT, FLAT_PCT, loss_change, loss_rate

REPO = os.path.normpath(os.path.join(DATA_DIR, "..", ".."))
COLLECT = os.path.join(REPO, "data", "fig_paper_F2_size_collect.pkl")
ORACLE_DIR = "/scratch/gpfs/TENGEL/pt1290/oracle_readout"

# ---------------------------------------------------------------------------------------------
# what is read
# ---------------------------------------------------------------------------------------------
# One series per task: the standard network's unpenalised control at every size, one experiment per
# task (same task, model and trainer; only N, the N-scaled learning rate and the seed differ). Ported
# from branch panelc-n100 (b02bba4), which added N = 100 and DMTS N = 4000.
#   3BFF  N = 100: NBitFlipFlop_n100 (200k, Spock job 6965913, synced to Della); N = 500-2000: the
#         remedy grid's control, 40k + 35k + 125k = 200k in one warm-start chain; N = 4000: the paper
#         grid's control continued to 200k.
#   CDDM  N = 100: CDDM_n100; N = 500-4000: the paper grid's control; 100k.
#   DMTS  5 tau delay. N = 100: DMTS_n100; N = 500-4000: the paper grid's control; 150k. N = 4000 was
#         trained on Spock and relayed to Della through the Mac on 31 Oct (this package).
FF200 = "NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N={N}_pen=none_arm=none"
TASKS = ["3BFF", "CDDM", "DMTS"]
TASK_COL = {"3BFF": ps.SLOTS[0], "CDDM": ps.SLOTS[1], "DMTS": ps.SLOTS[2]}
SERIES = {
    "3BFF": {100: "NBitFlipFlop_n100/EqType=h_N=100_arm=control",
             500: FF200.format(N=500), 1000: FF200.format(N=1000), 2000: FF200.format(N=2000),
             4000: "NBitFlipFlop_paper_grid_c200000/EqType=h_N=4000_arm=control"},
    "CDDM": {100: "CDDM_n100/EqType=h_N=100_arm=control",
             **{N: f"CDDM_paper_grid/EqType=h_N={N}_arm=control" for N in (500, 1000, 2000, 4000)}},
    "DMTS": {100: "DMTS_n100/EqType=h_N=100_arm=control",
             **{N: f"DMTS_paper_grid_d5/EqType=h_N={N}_arm=control" for N in (500, 1000, 2000, 4000)}},
}
SIZES = [100, 500, 1000, 2000, 4000]
FIT_MIN_N = 500        # N = 100 is drawn but not fitted: nearly every unit is active there
# A network that never learned the task is no evidence about how many units the task needs. DMTS is
# bimodal per seed: solved (R^2 > 0.99) or stuck at the constant-output solution (R^2 ~ 0.4).
MIN_R2 = {"DMTS": 0.8}
# Runs left out by a recorded decision (folder-name fragment -> reason), as in Figure 1.
EXCLUDED_RUNS = {"seed=3374117298;": "replaced by a fourth seed (board decision)"}
# Panel (d): every run of one N = 1000 cell per task. CDDM uses its longest standard control (200k),
# because the panel is about what happens long after the floor; the size series' CDDM runs stop at 100k.
TRAJ = {"3BFF": FF200.format(N=1000),
        "CDDM": "CDDM_std_g0_drift_bias0/EqType=h_N=1000_iters=200000",
        "DMTS": "DMTS_paper_grid_d5/EqType=h_N=1000_arm=control"}
# Figure 1's ablation networks (panel d of Figure 1), listed in Extended Data Table 2.
ABLATION_CELLS = {"3BFF": "NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control",
                  "CDDM": "CDDM_c1_snapshots/EqType=h_N=1000_arm=control",
                  "DMTS": "DMTS_c1_snapshots/EqType=h_N=1000_arm=control"}
# Panel (e): the lineages c3_flat_loss_units reads, plus the N = 4000 controls continued to 200k.
C3_EXTRA = [("N=4000", "NBitFlipFlop_paper_grid_c200000/EqType=h_N=4000_arm=control")]
TASKNAME = {"NBitFlipFlop": "3BFF", "CDDM": "CDDM", "DMTS": "DMTS"}


def seed_of(name):
    """The training seed in a run folder's name (`...;seed=<int>;...`), as a string, or ''."""
    m = re.search(r"seed=(\d+)", name)
    return m.group(1) if m else ""


def excluded_by_decision(name):
    """The recorded reason a run is left out of Figure 2 (EXCLUDED_RUNS), or None."""
    return next((why for key, why in EXCLUDED_RUNS.items() if key in name), None)


def read_one(run, cache, with_r2=True):
    """Read one run at every floor margin, with held-out R^2 at the end and at the 1.05x crossing.

    Args:
        run: a common.runs_with_loss dict (it_loss, loss, it_act, active, name, run_dir, task);
        cache: an open held-out R^2 cache dict (added to, never saved here);
        with_r2: score R^2 (False skips the scoring, for runs only drawn as trajectories).
    Returns:
        dict: name, seed, run_dir, N (units in the trace), it_end (last loss iteration), t_star and
        reason at 1.05x, t_stars and active_at ({margin: crossing / active count at the probe nearest
        it}, nan where none), covered (probes span t*), active_end, r2_end (held-out R^2 of the final
        weights), r2_floor and r2_measured (floor_r2: on a kept snapshot, else estimated), floor,
        floor_kind, n_bursts, and the trajectory arrays it_act, active (count per probe), tb, lb
        (log-binned noisy loss, 150 bins).
    """
    rd = read_run(run["run_dir"], run=run)
    it_a, act = np.asarray(run["it_act"], float), np.asarray(run["active"], float)
    N = size_of(run)
    active_at = {}
    for d, t in rd["t_stars"].items():
        active_at[d] = (float(act[int(np.argmin(np.abs(it_a - t)))])
                        if np.isfinite(t) and it_a[0] <= t <= it_a[-1] else float("nan"))
    tb, lb = logbin(run["it_loss"], run["loss"], nbins=150)
    rec = dict(name=run["name"], seed=seed_of(run["name"]), run_dir=run["run_dir"], N=N,
               it_end=float(run["it_loss"][-1]), t_star=float(rd["t_star"]), reason=rd["reason"],
               t_stars={d: float(t) for d, t in rd["t_stars"].items()}, active_at=active_at,
               covered=bool(rd["covered"]), active_end=rd["active_end"], floor=rd["floor"],
               floor_kind=rd["floor_kind"], n_bursts=len(rd["bursts"]), it_act=it_a, active=act,
               tb=tb, lb=lb, r2_end=float("nan"), r2_floor=float("nan"), r2_measured=False)
    if rd["reason"] is None and not rd["covered"]:
        rec["reason"] = "participation probes do not span the crossing"
    if with_r2:
        rec["r2_end"] = float(heldout.heldout_r2(run["run_dir"], draws=heldout.R2_DRAWS, cache=cache))
        if rec["reason"] is None:
            rec["r2_floor"], rec["r2_measured"] = heldout.floor_r2(rd, cache)
            rec["r2_floor"] = float(rec["r2_floor"])
    return rec, rd


def size_of(run):
    """Network size N of a run, parsed from its folder name (`...;N=<int>;...`).

    Args:
        run: a runs_with_loss dict.
    Returns:
        int N.
    """
    return int(re.search(r"N=(\d+)", run["name"]).group(1))


def readout_example(rd):
    """The pieces Extended Data Fig. 3a draws for one run, reduced to a few hundred points.

    Args:
        rd: common.read_run's dict for a run with a floor.
    Returns:
        dict: tq/lq10/lq50/lq90 (raw loss percentiles in 200 log bins), ts/ys (the Hampel read-out
        curve on a log grid), tf/yf (the fitted floor curve, None for a late-level floor), floor,
        floor_kind, escape, t_stars, bursts.
    """
    it, L = np.asarray(rd["it_loss"], float), np.asarray(rd["loss"], float)
    ok = np.isfinite(L)
    it, L = it[ok], L[ok]
    edges = np.logspace(0, np.log10(it[-1]), 201)
    idx = np.digitize(it, edges) - 1
    tq, q = [], []
    for b in range(200):
        m = idx == b
        if m.sum() > 4:
            tq.append(np.sqrt(edges[b] * edges[b + 1]))
            q.append(np.percentile(L[m], [10, 50, 90]))
    q = np.array(q)
    sm = np.asarray(rd["smoothed"], float)
    grid = np.unique(np.clip(np.logspace(0, np.log10(it[-1]), 400).astype(int), 1, len(sm)) - 1)
    out = dict(tq=np.array(tq), lq10=q[:, 0], lq50=q[:, 1], lq90=q[:, 2], ts=it[grid], ys=sm[grid],
               tf=None, yf=None, floor=rd["floor"], floor_kind=rd["floor_kind"],
               escape=rd["escape"], t_stars=rd["t_stars"], bursts=rd["bursts"])
    if rd.get("fit") is not None:
        tf = np.logspace(np.log10(max(rd["burst_from"], 1.0)), np.log10(it[-1]), 200)
        out["tf"], out["yf"] = tf, floor_curve(tf - rd["escape"], rd["fit"])
    return out


def oracle_read(folder, rd):
    """The snapshot oracle's sustained crossing for one run, at the rule's own 1.05x bar.

    Written independently of common.noisy_floor: each kept snapshot's value is the mean of its
    per-batch fresh-batch losses after dropping points more than 3 scaled MADs from their median; the
    oracle at a snapshot iteration t is the mean of the values of every snapshot in [t/1.1, 1.1 t]
    outside the rule's detected bursts; its crossing is the first t after which it stays at or under
    1.05x the rule's floor through 1.2 t. A run's snapshots are joined along its warm-start chain.

    Args:
        folder: the run folder whose oracle is read (its chain's oracle folders are found by seed);
        rd: common.read_run's dict for that run (floor, bursts, t_star).
    Returns:
        dict(t_rule, t_oracle, n_snap) or None when no oracle file exists for the run.
    """
    files = []
    for d in warm_start_chain(folder):
        cell = os.path.basename(os.path.dirname(d.rstrip("/")))
        sweep = os.path.basename(os.path.dirname(os.path.dirname(d.rstrip("/"))))
        files += glob.glob(os.path.join(ORACLE_DIR, f"{sweep}__{cell}__s{seed_of(os.path.basename(d))}",
                                        "it*.npy"))
    if not files:
        return None
    t = np.array([int(os.path.basename(f)[2:-4]) for f in files], float)
    v = []
    for f in files:
        x = np.load(f)
        dev = np.abs(x - np.median(x))
        v.append(x[dev <= 3 * 1.4826 * np.median(dev)].mean() if np.median(dev) > 0 else x.mean())
    order = np.argsort(t)
    t, v = t[order], np.array(v)[order]
    calm = np.ones(len(t), bool)
    for b0, b1, _ in rd["bursts"]:
        calm &= (t < b0) | (t > b1)
    curve = np.array([v[calm & (t >= s / 1.1) & (t <= s * 1.1)].mean()
                      if np.any(calm & (t >= s / 1.1) & (t <= s * 1.1)) else np.nan for s in t])
    bar = 1.05 * rd["floor"]
    t_or = float("nan")
    for i, s in enumerate(t):
        hold = (t >= s) & (t <= (1 + HOLD_FRAC) * s) & np.isfinite(curve)
        if (1 + HOLD_FRAC) * s > t[-1]:
            break
        if np.isfinite(curve[i]) and np.all(curve[hold] <= bar):
            t_or = float(s)
            break
    return dict(t_rule=float(rd["t_star"]), t_oracle=t_or, n_snap=len(t))


def collect():
    """Read every input of Figure 2, Extended Data Fig. 3 and Extended Data Table 2; pickle to COLLECT.

    Returns:
        the path written.
    """
    cache = heldout.load_cache()
    n0 = len(cache)
    series, examples = [], {}
    for task in TASKS:
        for N in SIZES:
            pat = os.path.join(DATA_DIR, SERIES[task][N])
            runs = runs_with_loss(pat)
            if not runs:
                print(f"  !! {task} N={N}: no runs at {pat}")
            for r in runs:
                rec, rd = read_one(r, cache)
                rec.update(task=task, cell=SERIES[task][N], N=N, role="size series",
                           excluded=excluded_by_decision(r["name"]))
                if task in MIN_R2 and rec["r2_end"] < MIN_R2[task]:
                    rec["excluded"] = f"never learned the task (held-out R$^2$ {rec['r2_end']:.2f})"
                if N == 1000 and rec["reason"] is None and rec["excluded"] is None:
                    examples.setdefault(task, []).append(readout_example(rd) | dict(name=r["name"]))
                series.append(rec)
                print(f"  .. {task} N={N} s{rec['seed']}: active {rec['active_at'][0.05]} at t* "
                      f"{rec['t_star']:,.0f} ({rec['reason']}), R2 end {rec['r2_end']:.3f}, floor "
                      f"{rec['r2_floor']:.3f}{' measured' if rec['r2_measured'] else ''}; "
                      f"excluded: {rec['excluded']}")
    traj = []
    for task, rel in TRAJ.items():
        for r in runs_with_loss(os.path.join(DATA_DIR, rel)):
            rec, _ = read_one(r, cache, with_r2=(task == "CDDM"))
            rec.update(task=task, cell=rel, role="trajectory", excluded=excluded_by_decision(r["name"]))
            traj.append(rec)
    ablation = []
    for task, rel in ABLATION_CELLS.items():
        for r in runs_with_loss(os.path.join(DATA_DIR, rel)):
            rec, _ = read_one(r, cache)
            rec.update(task=task, cell=rel, role="Fig. 1d ablation", excluded=None)
            ablation.append(rec)
    # panel (e): c3_flat_loss_units' own lineage walk (a run continued further is read once, as its
    # longest continuation), on its cells plus the N = 4000 controls
    c3, seen = [], set()
    for size, rel in C3_CELLS + C3_EXTRA:
        for r in runs_with_loss(os.path.join(DATA_DIR, rel)):
            chain = [os.path.basename(d.rstrip("/")) for d in warm_start_chain(r["run_dir"])]
            if r["name"] in seen:
                continue
            seen.update(chain)
            it, L = np.asarray(r["it_loss"], float), np.asarray(r["loss"], float)
            ita, act = np.asarray(r["it_act"], float), np.asarray(r["active"], float)
            if it[-1] < T_FLAT + 40_000:
                continue
            rd = read_run(r["run_dir"], run=r)
            _, _, pct, pct100 = loss_change(it, L)
            lost, se = loss_rate(ita, act)
            c3.append(dict(N=int(size.split("=")[1].split()[0]), label=size, name=r["name"],
                           seed=seed_of(r["name"]), end=float(it[-1]), t_star=float(rd["t_star"]),
                           reason=rd["reason"], pct=pct, pct100=pct100, lost=lost, se=se,
                           excluded=excluded_by_decision(r["name"])))
            print(f"  .. C3 {size} s{seed_of(r['name'])}: end {it[-1]:,.0f}, t* {rd['t_star']:,.0f}, "
                  f"loss {pct100:+.1f}%/100k, lost {lost:.1f} +- {se:.1f} per 100k")
    # Extended Data Fig. 3c: every run with scored snapshots
    oracle = []
    for folder in sorted(glob.glob(os.path.join(ORACLE_DIR, "*__*__s*"))):
        sweep, cell, s = os.path.basename(folder).split("__")
        hits = [d for d in glob.glob(os.path.join(DATA_DIR, sweep, cell, "*"))
                if seed_of(os.path.basename(d)) == s[1:]]
        if not hits:
            print(f"  !! oracle {os.path.basename(folder)}: run folder not found")
            continue
        rd = read_run(hits[0])
        if rd["reason"] is not None:
            oracle.append(dict(name=os.path.basename(folder), task=TASKNAME.get(sweep.split("_")[0]),
                               t_rule=float("nan"), t_oracle=float("nan"), reason=rd["reason"]))
            continue
        o = oracle_read(hits[0], rd)
        if o is not None:
            oracle.append(o | dict(name=os.path.basename(folder), reason=None,
                                   task=TASKNAME.get(sweep.split("_")[0])))
            print(f"  .. oracle {os.path.basename(folder)}: rule {o['t_rule']:,.0f}, oracle "
                  f"{o['t_oracle']:,.0f} ({o['n_snap']} snapshots)")
    rec = dict(series=series, traj=traj, ablation=ablation, c3=c3, oracle=oracle, examples=examples)
    with open(COLLECT, "wb") as fh:
        pickle.dump(rec, fh)
    private = COLLECT.replace(".pkl", "_r2_cache.json")
    with open(private, "w") as fh:
        json.dump(cache, fh, indent=0, sort_keys=True)
    print(f"wrote {COLLECT}; {len(cache) - n0} new held-out R2 score(s) in {private}")
    return COLLECT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="on Della: read every run, write COLLECT")
    args = ap.parse_args()
    if args.collect:
        collect()
