#!/usr/bin/env python3
"""Manuscript Figure 2 (size and training time) and Supplementary Table 1; collect stage of Extended Data Fig. 3.

Figure 2: a larger network keeps a smaller share of its units active, and training past the loss
floor makes more units inactive.
  (a) active units against network size N (100-4000), the three main-text tasks (MAIN_TASKS: 3BFF,
      CDDM and the maze; round 14), one point per run. A supervised run is read where its noisy
      training loss reaches 1.05x its own fitted floor (common.read_run); a maze run at its "learned"
      probe (maze_series). Power laws fitted to the size means from N = 500 for the supervised tasks
      (N = 100 drawn, not fitted) and over every size with a learned run for the maze.
  (b) the same runs as a percentage of N, guides at 50% and 10%.
  (c) the fitted exponent at the 1.05x, 1.10x and 1.20x margins, with a bootstrap 95% interval
      (runs resampled within each size); 3BFF and CDDM only, because the margins are margins over
      the supervised loss floor. The maze's exponent and interval are printed, not drawn.
  (d) a schematic of the floor crossing (Pavel's feedback item 47; schematic since 3 Nov): an
      idealised loss on the curve of Eq. (floor), its floor, the band to 1.05x the floor, t* and the
      hold to 1.2 t*. No network's data; Extended Data Fig. 3a shows real losses and fits.
  (e) one column per main-text task, every run at N = 1000. Supervised columns: the noisy training
      loss over its own start and the inactive-unit count against iteration; dashed lines at each
      run's own 1.05x crossing. Maze column: the clean probe's success rate and the inactive-unit
      count against PPO update, on a linear axis; dashed lines at each run's learned probe.
  (f) the flat-loss test drawn from its raw quantities (round 12, Pavel's item 55), every 3BFF run
      trained to 200,000 or 300,000 iterations, one column per training length: top, the noisy
      loss as 5,000-iteration block medians, % change from the first block after 150,000, inside
      or outside the flat-loss wedge (3% per 100,000 iterations); bottom, the least-squares line
      of the active count from 150,000 against the decrease of 5 units per 100,000 iterations
      (dashed). Raw traces from --collect-flat (FLAT); both test quantities are recomputed from
      them and checked against the collect record (c3_flat_loss_units: loss_change, loss_rate).
Extended Data Fig. 3 is drawn by fig_ed_readout.py from this script's collect record (examples, size
series and the snapshot oracle of /scratch/gpfs/TENGEL/pt1290/oracle_readout, Della job 15635966,
entry 2026-10-15_2054). The drawing stage also writes the rows of Supplementary Table 1 (every run of
Figures 1 and 2; formerly Extended Data Table 2).

ROUND 14 (dead-relu-paper#182, Pavel's decision of 4 Nov; Figure 1 commit ef5ab33). The maze-running
task (an agent trained by PPO to reach the cheese in a fixed maze; engellab/trainRNNbrain#3)
replaces DMTS as the third task of panels (a), (b) and (e); DMTS moves to Extended Data. MAIN_TASKS
lists the three tasks those panels draw; TASKS keeps DMTS because collect() reads it and
fig_ed_readout.py reads the collect record. The maze is read when the figure is drawn, from this
repo and not from COLLECT: every ReLU run without a goal cue of the uniform series under MAZE_ROOT
(common.maze_load_runs joins a run and its continuations into one record with absolute update
indices). The maze counterpart of the floor crossing is the "learned" probe, the first probe from
which the clean probe's success rate stays at or above MAZE_LEARNED_SUCCESS for MAZE_LEARNED_HOLD
probes (common.maze_learned_index); the end of training is the chain's last probe. A finished run
with no learned probe did not learn: it is a cross at its end-of-training count in (a) and (b) and
enters no mean and no fit, as fig_ed_maze.py draws it. A run whose records end before its budget
(stopped by the trainer's guard, or still running) is listed and not drawn. The maze's sizes are
whatever the suite holds (no size is hard-coded), so N = 4,000 and the N = 2,000 continuations
are drawn when they land. Every maze number drawn is printed.

TWO STAGES, because the networks live on Della:
  --collect   on a Della CPU node, from the checkout root (data/ links the trained networks): reads
              every run and writes data/fig_paper_F2_size_collect.pkl. Held-out R^2 comes from
              heldout_r2 (36 noise draws); the shared cache is read but never written, new scores go
              to data/fig_paper_F2_size_collect_r2_cache.json.
  (no flag)   on the Mac: draws Figure 2 from that record into img/internal_figures/fig_paper_F2_size
              (PDF + SVG, 165.1 mm wide, paperstyle), prints every number drawn, checks text
              overlap, and writes the table body img/internal_figures/edt2_runs_rows.tex.

READ-OUT (Pavel, 16 Oct; docs/AGENT_GUIDE.md "Reading networks"). Across sizes every run is read at
1.05x its own noisy-loss floor, because the learning rate scales as N^(-1/3) and larger networks
converge at different times. A series is drawn at the floor only if every size has at least one
read-out (otherwise it would be drawn at the end of training and labelled; it does not happen here).
Runs without a read-out are counted and listed with their reason in Extended Data Table 2.

Usage:  python fig_paper_F2_size.py --collect      (Della)
        python fig_paper_F2_size.py --collect-flat (Della; panel f's raw traces, CPU, under a minute)
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
import yaml
import matplotlib
matplotlib.use("Agg")
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator
from matplotlib.transforms import ScaledTranslation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
import heldout_r2 as heldout
from common import (DATA_DIR, FLOOR_DELTAS, HOLD_FRAC, IMG_DIR, MAZE_LEARNED_HOLD,
                    MAZE_LEARNED_SUCCESS, floor_curve, logbin, maze_learned_index, maze_load_runs,
                    read_run, runs_with_loss, warm_start_chain)
from c3_flat_loss_units import (BLOCK as FLAT_BLOCK, CELLS as C3_CELLS, DIE_BAR, T_FLAT, FLAT_PCT,
                                loss_change, loss_rate)

REPO = os.path.normpath(os.path.join(DATA_DIR, "..", ".."))
COLLECT = os.path.join(REPO, "data", "fig_paper_F2_size_collect.pkl")
# panel (f)'s raw traces from iteration 150,000 on (collect_flat; Della, --collect-flat)
FLAT = os.path.join(REPO, "data", "fig_paper_F2_size_flat.pkl")
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
TASKS = ["3BFF", "CDDM", "DMTS"]      # the supervised series collect() reads (DMTS: Extended Data)
TASK_COL = ps.TASK_COLORS     # reserved task hues of the style sheet (docs/figure_style.md)
# The maze-running task (PPO; engellab/trainRNNbrain#3; in the main text since dead-relu-paper#182).
# Its runs live in this repo, not on Della, and are read when the figure is drawn (maze_series).
MAZE = "Maze"
MAZE_ROOT = os.path.join(REPO, "data", "trained_RNNs_rl_maze", "Maze_suite")
MAZE_ABLATION_DIR = os.path.join(REPO, "data", "rl_maze_ablation")   # Fig. 1d's maze JSONs (table)
# The three tasks of the main text, in the order panels (a), (b) and (e) draw them (#182: the maze
# in place of DMTS, which stays in TASKS for collect() and fig_ed_readout.py).
MAIN_TASKS = ["3BFF", "CDDM", MAZE]
SUPERVISED_MAIN = [t for t in MAIN_TASKS if t != MAZE]     # panel (c): margins over the loss floor
MAZE_TRAJ_N = 1000            # panel (e)'s maze column: the suite's N = 1,000 ReLU runs
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
# Runs left out by a recorded decision (folder-name fragment -> reason). Empty: the 3BFF N = 1000
# straggler (seed 3374117298), once replaced by a fourth seed, reads out under the current read-out
# (277 active), and dead-relu-paper#20 recommends drawing all four networks; followed until Pavel
# answers. A collect record written under an earlier list is re-flagged by apply_decisions.
EXCLUDED_RUNS = {}
# Panel (e): every run of one N = 1000 cell per task. CDDM uses its longest standard control (200k),
# because the panel is about what happens long after the floor; the size series' CDDM runs stop at 100k.
TRAJ = {"3BFF": FF200.format(N=1000),
        "CDDM": "CDDM_std_g0_drift_bias0/EqType=h_N=1000_iters=200000",
        "DMTS": "DMTS_paper_grid_d5/EqType=h_N=1000_arm=control"}
# Figure 1's ablation networks (panel d of Figure 1), listed in Extended Data Table 2.
ABLATION_CELLS = {"3BFF": "NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control",
                  "CDDM": "CDDM_c1_snapshots/EqType=h_N=1000_arm=control",
                  "DMTS": "DMTS_c1_snapshots/EqType=h_N=1000_arm=control"}
# Panel (f): the lineages c3_flat_loss_units reads, plus the N = 4000 controls continued to 200k.
C3_EXTRA = [("N=4000", "NBitFlipFlop_paper_grid_c200000/EqType=h_N=4000_arm=control")]
# Panel (f) reads two Fig. 1d networks as their 300k continuations, whose folders carry new seeds.
# Continuation seed -> the 200k network's seed (its row in Extended Data Table 2), from each
# continuation's paths.init_from in NBitFlipFlop_c1_snapshots_c300000 (read on Della, 31 Oct).
C3_CONTINUED = {"3204992658": "4113307577", "3194339211": "2461724787"}
TASKNAME = {"NBitFlipFlop": "3BFF", "CDDM": "CDDM", "DMTS": "DMTS"}
# Size-series cells whose networks are the transfer-test sources of Fig. 4e,f (task, N). At N = 4,000
# the old sources are this cell's 200,000-iteration networks and the young ones the same networks at
# 75,000 iterations: paths.transfer_from of NBitFlipFlopHyper_b4_transfer/EqType=h_N=4000_src=
# control_age=old names the three NBitFlipFlop_paper_grid_c200000 N=4000 control folders, and of
# ..._age=young the three NBitFlipFlop_paper_grid_c75000 folders that are those networks'
# paths.init_from (saved configs on Della, read by the PR #119 reviewer, 1 Nov). The young sources'
# active counts at transfer (588, 502, 544; data/remedy_reads_31oct/read_b2_b4.json) equal the
# 75,000-iteration counts (589, 502, 544; data/c1_ablation_seeded) to within one unit
# (dead-relu-paper #111). The N = 1,000 sources are the
# Fig. 1d networks, marked in tex_row.
TRANSFER_SOURCES = {("3BFF", 4000)}
# Supplementary Table 1's Fig. column marks the 35 size-series networks that Extended Data Fig. 1
# (removing inactive units) tests, by task and table seed (dead-relu-paper #97, #89). Set read from
# fig_supp_ablation.py on trainRNNbrain wp/fig1-r3, notebook entry 2026-11-01_1003: every size-series
# network at N = 500-4,000 except 3BFF s4186074845 and DMTS s632794669 and s2627187151, plus DMTS
# s2946835052 (no floor crossing); no N = 100 network.
ED1_NETWORKS = {
    "3BFF": {"822903475", "822939552", "893469585", "2484557599", "2489235792", "2545439171",
             "3374117298", "1914910909", "3866906883", "620846234", "2422919889", "4145304148",
             "3341404959", "3346922328", "3386387268"},
    "CDDM": {"2251851164", "3291291930", "3565681970", "457853", "1272763220", "3833537590",
             "575781518", "3640167262", "4259132547", "292757883", "3046823201", "3572448132"},
    "DMTS": {"1084904964", "2099578504", "2106510642", "86371460", "1167683015", "2946835052",
             "2585017553", "4124951173"},
}


def seed_of(name):
    """The training seed in a run folder's name (`...;seed=<int>;...`), as a string, or ''."""
    m = re.search(r"seed=(\d+)", name)
    return m.group(1) if m else ""


def excluded_by_decision(name):
    """The recorded reason a run is left out of Figure 2 (EXCLUDED_RUNS), or None."""
    return next((why for key, why in EXCLUDED_RUNS.items() if key in name), None)


def apply_decisions(rec):
    """Re-apply EXCLUDED_RUNS to a collect record, so a changed decision needs no new Della read.

    Exclusions for a run that never learned the task (set from its R^2 at collect time) are kept.

    Args:
        rec: the unpickled COLLECT record (modified in place).
    Returns:
        the same record.
    """
    for key in ("series", "traj", "ablation", "c3"):
        for r in rec[key]:
            if not (r.get("excluded") or "").startswith("never learned"):
                r["excluded"] = excluded_by_decision(r["name"])
    return rec


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


def flat_lineages():
    """The 3BFF lineages of panel (f), walked the way c3_flat_loss_units walks them.

    A run continued further is read once, as its longest continuation (the cells are listed longest
    first); a lineage trained less than 40,000 iterations past T_FLAT is skipped.

    Returns:
        list of (size label, common.runs_with_loss dict), one per lineage, in C3_CELLS + C3_EXTRA order.
    """
    out, seen = [], set()
    for size, rel in C3_CELLS + C3_EXTRA:
        for r in runs_with_loss(os.path.join(DATA_DIR, rel)):
            chain = [os.path.basename(d.rstrip("/")) for d in warm_start_chain(r["run_dir"])]
            if r["name"] in seen:
                continue
            seen.update(chain)
            if r["it_loss"][-1] < T_FLAT + 40_000:
                continue
            out.append((size, r))
    return out


def collect_flat():
    """Read the raw traces panel (f) draws, from T_FLAT on, and pickle them to FLAT.

    For each lineage of `flat_lineages`: the noisy training loss (iterations > T_FLAT - BLOCK) and the
    active count at every participation probe (iterations >= T_FLAT - BLOCK), plus the two test
    quantities recomputed with c3_flat_loss_units' own functions, so the drawing stage can check them
    against the collect record's panel-(f) rows.

    Returns:
        the path written.
    """
    rows = []
    for size, r in flat_lineages():
        it, L = np.asarray(r["it_loss"], float), np.asarray(r["loss"], float)
        ita, act = np.asarray(r["it_act"], float), np.asarray(r["active"], float)
        _, _, _, pct100 = loss_change(it, L)
        lost, se = loss_rate(ita, act)
        keep, keep_a = it > T_FLAT - FLAT_BLOCK, ita >= T_FLAT - FLAT_BLOCK
        rows.append(dict(N=int(size.split("=")[1].split()[0]), seed=seed_of(r["name"]), name=r["name"],
                         it_loss=it[keep], loss=L[keep], it_act=ita[keep_a], active=act[keep_a],
                         pct100=float(pct100), lost=float(lost), se=float(se)))
        print(f"  .. flat {size} s{rows[-1]['seed']}: end {it[-1]:,.0f}, loss {pct100:+.2f}%/100k, "
              f"lost {lost:.2f} +- {se:.2f} per 100k")
    with open(FLAT, "wb") as fh:
        pickle.dump(rows, fh)
    print(f"wrote {FLAT} ({len(rows)} lineages)")
    return FLAT


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
    # panel (f): c3_flat_loss_units' own lineage walk (a run continued further is read once, as its
    # longest continuation), on its cells plus the N = 4000 controls
    c3 = []
    for size, r in flat_lineages():
        it, L = np.asarray(r["it_loss"], float), np.asarray(r["loss"], float)
        ita, act = np.asarray(r["it_act"], float), np.asarray(r["active"], float)
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


# ---------------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------------
FIG2_H_MM = 103.0       # printed height of Figure 2 (width is \textwidth, 165.1 mm); 117 until round 15,
                        # when the caption grew by the maze's read-out and n per size (dead-relu-paper#182)
FIG2_BOTTOM_WIDTHS = [2.15, 1.35]       # bottom row: panel e (three tasks), panel f (two columns)
# gap between panels e and f, as a fraction of the mean panel width: about 13 mm, room for the maze
# column's success-rate ticks and label on its right (round 14) beside panel f's left tick labels
# and y label (0.27, about 10.6 mm, left 1 mm between the two rotated labels)
BOTTOM_WSPACE = 0.34
TOP_WIDTHS = [1.2, 1.2, 0.64, 0.86]     # top row: panels a, b, c, d
TOP_WSPACE = 0.37       # gaps of the top row, as a fraction of the mean panel width (about 11 mm)
# x range of panels a-b: 80 to 4,800 gives 20 mm per factor of 10 in N at print size, so the tick
# labels 1,000, 2,000 and 4,000 (5.3 mm wide at 6 pt, 6.2 mm apart) stay apart
SIZE_XLIM = (80, 4800)
N_BOOT = 2000           # bootstrap resamples for the exponent's interval (panel c)
BOOT_SEED = 20261031
# Panel f: one line per network, coloured by N in shades of the 3BFF blue (every network is 3BFF),
# light to dark with size (round 12, Pavel's item 55)
FLAT_SHADE = {500: "#86b3ea", 1000: "#4f91de", 2000: "#2363b4", 4000: "#123f78"}
FLAT_OFF = ps.FAINT     # panel f: a network whose loss is not flat, in both rows (review of PR #177)
# Panels a-b: one network's point is drawn at a display offset beside its size N, never at a moved N
# (decision #83). The offsets are set in millimetres on the page (network_xs): every size's dots form
# one narrow column centred on N, each dot at the slot nearest N (0, then -/+ NET_MIN_MM, -/+ 2
# NET_MIN_MM, ...) at which it is at least NET_MIN_MM from every dot already placed and MEAN_CLEAR_MM
# from every mean marker. At print size the widest offset is 1.5 mm, a quarter of the narrowest gap
# between sizes (1,000 to 2,000, 6.2 mm in panel a). Item 46 of Pavel's 1 Nov feedback: the old rows
# reached more than half of that gap and read as random sizes.
NET_MIN_MM = 0.5        # least distance between two dots' centres (a 2 pt dot is 0.71 mm wide)
MEAN_CLEAR_MM = 0.88    # least distance from a mean marker's centre (radii 0.53 + 0.35 mm): never hidden
# Panel d: a schematic of the floor-crossing definition, not data (Pavel, 3 Nov: the real CDDM loss
# ended at 1.03x its floor and never showed the approach). The curve is Eq. (floor) of Methods,
# L_inf + A exp[-((t - t0)/s)^beta], drawn divided by L_inf. Its shape is the fit of 3BFF N = 1,000
# seed 1914910909 (a read-out example of the collect record), whose parameters were recovered by
# refitting the stored fitted curve (notebook entry 2026-11-03_0851): A/L_inf = 2.55, s = 9,253,
# beta = 0.70, t0 = 145 (rounded below). With these the curve is within 0.15% of the floor at
# 2.5 t*. The three CDDM N = 1,000 example fits have beta of 0.24 to 0.30 and come that close only
# after 10 to 15 t* (270,000 to 420,000 iterations, beyond their 100,000), which would squeeze the
# hold window to under 1 mm of the panel.
SCHEMATIC_A, SCHEMATIC_S, SCHEMATIC_BETA, SCHEMATIC_T0 = 2.55, 9_250.0, 0.70, 145.0
SCHEMATIC_X_END = 2.5   # the iteration axis runs to this multiple of the schematic's t*


def is_read(r, margin=0.05):
    """True when a size-series run counts in Figure 2 at `margin`: it has a read-out there and is not
    excluded (decision or never learned)."""
    return r["excluded"] is None and r["reason"] is None and np.isfinite(r["active_at"][margin])


def cells_of(series, task, margin=0.05):
    """Active counts per size of one task at one margin, read runs only.

    Args:
        series: the collect record's size-series rows; task: "3BFF", "CDDM" or "DMTS";
        margin: floor margin (0.05, 0.10, 0.20).
    Returns:
        dict N -> float array of active counts (sizes with no read run left out).
    """
    out = {}
    for N in SIZES:
        v = [r["active_at"][margin] for r in series if r["task"] == task and r["N"] == N
             and is_read(r, margin)]
        if v:
            out[N] = np.array(v, float)
    return out


def maze_budget(folder):
    """The last PPO update a maze run's budget reaches, from the saved config of its last run.

    A continuation's config records trainer.iter_offset (the parent's last update) and its own
    trainer.n_updates, so the budget is their sum; the earlier layout (run_maze_ppo.py) has the
    budget under ppo.n_updates and no offset.

    Args:
        folder: str, the run folder of the chain's last run (common.maze_load_runs's "folder").
    Returns:
        int, the absolute update index at which the run's budget ends.
    """
    with open(sorted(glob.glob(os.path.join(folder, "*config.yaml")))[0]) as fh:
        cfg = yaml.safe_load(fh)
    block = cfg["trainer"] if "trainer" in cfg else cfg["ppo"]
    offset = block.get("iter_offset", 0)
    return int(block["n_updates"]) + (int(offset) if isinstance(offset, (int, float)) else 0)


def maze_series(root=MAZE_ROOT):
    """The maze's size-series rows, read from this repo when the figure is drawn (#182).

    Every ReLU run without a goal cue under `root` (common.maze_load_runs; a run and its
    continuations are one record with absolute update indices) is read at its "learned" probe,
    the first probe from which the clean probe's success rate stays at or above
    MAZE_LEARNED_SUCCESS for MAZE_LEARNED_HOLD probes in a row (common.maze_learned_index), the
    maze counterpart of the supervised floor crossing, and at the chain's last probe (the end of
    training). A run whose last recorded update is below its budget (maze_budget) is unfinished:
    stopped by the trainer's guard or still running. It is listed with a reason and drawn nowhere.

    Args:
        root: str, the suite folder (<cell>/s<seed>/ run folders).
    Returns:
        list of dicts, one per run, sorted by N and seed: task (MAZE), N, seed (str), name (the
        run folder relative to root), n_chain (runs in the chain), it_end (last probe's update),
        budget, finished (bool), learned (bool), t_star (the learned probe's update, nan when
        none), active_learned, success_learned (nan when none), active_end, success_end, reason
        (None, or why the run is not read at a learned point), and the probe arrays it_act
        (update per probe), active (count per probe) and success (probe success rate).
    """
    rows = []
    for r in maze_load_runs(root):
        if r["act"] != "relu" or r["cue"]:
            continue
        pr = r["probes"]
        budget = maze_budget(r["folder"])
        it_end = int(pr["update"][-1])
        finished = int(r["progress"][-1]["update"]) >= budget
        li = maze_learned_index(pr["probe_success"])
        learned = finished and li is not None
        if not finished:
            reason = (f"unfinished: records end at update {r['progress'][-1]['update']:,} of "
                      f"{budget:,}")
        elif li is None:
            reason = (f"not learned: probe success never stayed at or above {MAZE_LEARNED_SUCCESS} "
                      f"for {MAZE_LEARNED_HOLD} probes")
        else:
            reason = None
        rows.append(dict(task=MAZE, N=int(r["N"]), seed=str(r["seed"]),
                         name=os.path.relpath(r["folder"], root), n_chain=len(r["chain"]),
                         it_end=it_end, budget=budget, finished=finished, learned=learned,
                         t_star=float(pr["update"][li]) if learned else float("nan"),
                         active_learned=float(pr["active"][li]) if learned else float("nan"),
                         success_learned=float(pr["probe_success"][li]) if learned else float("nan"),
                         active_end=float(pr["active"][-1]), success_end=float(pr["probe_success"][-1]),
                         reason=reason, it_act=np.asarray(pr["update"], float),
                         active=np.asarray(pr["active"], float),
                         success=np.asarray(pr["probe_success"], float)))
    return sorted(rows, key=lambda r: (r["N"], int(r["seed"])))


def maze_cells(rows, at="learned"):
    """Active counts per size of the maze's learned runs, at the learned probe or at the end.

    Args:
        rows: maze_series rows; at: "learned" (the count at the learned probe) or "end" (the
            count at the chain's last probe, learned runs only).
    Returns:
        dict N -> float array of active counts (sizes with no learned run left out).
    """
    key = "active_learned" if at == "learned" else "active_end"
    out = {}
    for r in rows:
        if r["learned"]:
            out.setdefault(r["N"], []).append(r[key])
    return {N: np.array(v, float) for N, v in sorted(out.items())}


def power_fit(cells, min_n=FIT_MIN_N):
    """Least-squares line through log(mean active count) against log N, sizes N >= min_n.

    Args:
        cells: dict N -> array of active counts; min_n: the smallest size fitted (FIT_MIN_N for
            the supervised series; 0 for the maze, whose suite has no N = 100 cell).
    Returns:
        (exponent b, prefactor A) with mean count ~ A N^b.
    """
    Ns = np.array([N for N in sorted(cells) if N >= min_n], float)
    ys = np.array([cells[N].mean() for N in sorted(cells) if N >= min_n])
    b, loga = np.polyfit(np.log(Ns), np.log(ys), 1)
    return float(b), float(np.exp(loga))


def boot_exponent(cells, rng, min_n=FIT_MIN_N):
    """Bootstrap 95% interval of the fitted exponent, resampling runs within each size.

    Args:
        cells: dict N -> array of active counts; rng: numpy Generator; min_n: as power_fit.
    Returns:
        (2.5th, 97.5th percentile) of the exponent over N_BOOT resamples.
    """
    bs = []
    for _ in range(N_BOOT):
        bs.append(power_fit({N: rng.choice(v, len(v)) for N, v in cells.items()}, min_n)[0])
    return tuple(np.percentile(bs, [2.5, 97.5]))


def maze_fit(rows, at="learned"):
    """The maze's power law over every size with a learned run, with its bootstrap interval.

    Args:
        rows: maze_series rows; at: "learned" or "end" (maze_cells).
    Returns:
        (b, A, lo, hi, cells) with b and A as power_fit, lo and hi the bootstrap 95% interval
        (BOOT_SEED) and cells the maze_cells fitted; (nan, nan, nan, nan, cells) with fewer than
        two sizes.
    """
    cells = maze_cells(rows, at)
    if len(cells) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), cells
    b, A = power_fit(cells, min_n=0)
    lo, hi = boot_exponent(cells, np.random.default_rng(BOOT_SEED), min_n=0)
    return b, A, float(lo), float(hi), cells


def network_xs(ax, N, values, means):
    """Drawing positions on the x axis of one size's network dots: a narrow column centred on N.

    The dots are placed in order of value. Each takes the first slot, in page millimetres from N,
    of 0, -NET_MIN_MM, +NET_MIN_MM, -2 NET_MIN_MM, ... at which it lies at least NET_MIN_MM from every
    dot already placed and at least MEAN_CLEAR_MM from every mean marker (drawn at N). The y values
    are not changed. The axes' scales and limits must be set before the call.

    Args:
        ax: axes; N: int, the network size;
        values: float array (n,), the dots' y values in data units (every task at this size);
        means: float array, the y values of the mean markers drawn at N.
    Returns:
        float array of shape (n,): x positions in data units, for drawing only.
    """
    px_per_mm = ax.figure.dpi / 25.4
    disp = ax.transData.transform(np.column_stack([np.full(len(values), float(N)), values]))
    y_mm = disp[:, 1] / px_per_mm
    m_mm = ax.transData.transform(np.column_stack([np.full(len(means), float(N)), means]))[:, 1] / px_per_mm
    slots = [0.0] + [sign * k * NET_MIN_MM for k in range(1, 20) for sign in (-1.0, 1.0)]
    x_mm, placed = np.zeros(len(values)), []
    for i in np.argsort(y_mm, kind="stable"):
        x_mm[i] = next(x for x in slots
                       if all(np.hypot(x, y_mm[i] - m) >= MEAN_CLEAR_MM for m in m_mm)
                       and all(np.hypot(x - px, y_mm[i] - py) >= NET_MIN_MM for px, py in placed))
        placed.append((x_mm[i], y_mm[i]))
    x_disp = disp[:, 0] + x_mm * px_per_mm
    return ax.transData.inverted().transform(np.column_stack([x_disp, disp[:, 1]]))[:, 0]


def leave_one_out(series, task, margin):
    """Exponent and bootstrap upper end of one fit with each network left out in turn.

    Only networks at sizes with two or more are left out, so every refit keeps all four sizes
    N >= FIT_MIN_N. Each refit's bootstrap starts from BOOT_SEED.

    Args:
        series: the collect record's size-series rows; task: "3BFF", "CDDM" or "DMTS";
        margin: floor margin (0.05, 0.10, 0.20).
    Returns:
        list of (upper end of the 95% interval, exponent, N, seed), largest upper end first.
    """
    rows = [r for r in series if r["task"] == task and r["N"] >= FIT_MIN_N and is_read(r, margin)]
    out = []
    for drop in rows:
        if sum(r["N"] == drop["N"] for r in rows) < 2:
            continue
        cells = {}
        for r in rows:
            if r is not drop:
                cells.setdefault(r["N"], []).append(r["active_at"][margin])
        cells = {N: np.array(v, float) for N, v in cells.items()}
        hi = boot_exponent(cells, np.random.default_rng(BOOT_SEED))[1]
        out.append((float(hi), power_fit(cells)[0], drop["N"], drop["seed"]))
    return sorted(out, reverse=True)


def panel_size(ax_a, ax_b, series, maze):
    """Panels (a) and (b): active units, and active units as a percentage of N, against N.

    One narrow column of small points per size (network_xs): every supervised run of MAIN_TASKS at
    its floor crossing and every learned maze run at its learned probe, filled dots in the task's
    hue; a finished maze run that did not learn is a cross at its end-of-training count and enters
    no mean and no fit. Per task and size the mean +- SD, and per task a power law: over the sizes
    N >= FIT_MIN_N for the supervised tasks, over every size with a learned run for the maze (its
    suite has no N = 100 cell). The x ticks are the sizes any drawn series has, so a maze size the
    supervised series lack (250) gets a tick and a new size (4,000 continuations) needs no edit.

    Args:
        ax_a, ax_b: axes; series: the collect record's size-series rows; maze: maze_series rows.
    Returns:
        dict task -> (exponent, prefactor, {N: (mean, SD, n)}); the exponent is nan for a task
        with fewer than two fitted sizes (no line, no b in the legend).
    """
    fits, handles = {}, []
    all_cells = {task: cells_of(series, task) for task in SUPERVISED_MAIN}
    all_cells[MAZE] = maze_cells(maze, "learned")
    crosses = [(r["N"], r["active_end"]) for r in maze if r["finished"] and not r["learned"]]
    sizes = sorted(set(SIZES) | set(all_cells[MAZE]) | {N for N, _ in crosses})
    # scales and limits first: network_xs places the dots in millimetres on the page
    ax_a.set(xscale="log", yscale="log", xlim=SIZE_XLIM, ylim=(50, 2500))
    ax_b.set(xscale="log", xlim=SIZE_XLIM, ylim=(0, 100))
    # one point per network, in a narrow column centred on its size (network_xs)
    for ax, pct in ((ax_a, False), (ax_b, True)):
        for N in sizes:
            scale = 100.0 / N if pct else 1.0
            pts = [(task, v, "o") for task in MAIN_TASKS for v in all_cells[task].get(N, [])]
            pts += [(MAZE, v, "x") for n, v in crosses if n == N]
            vals = np.array([v for _, v, _ in pts], float) * scale
            means = np.array([all_cells[task][N].mean() for task in MAIN_TASKS
                              if N in all_cells[task]]) * scale
            for (task, _, marker), x, y in zip(pts, network_xs(ax, N, vals, means), vals):
                if marker == "x":
                    ax.plot([x], [y], "x", ms=ps.SMALL_MARKER_PT + 0.8, mec=TASK_COL[task], mew=0.7,
                            zorder=4)
                else:
                    ax.plot([x], [y], "o", ms=ps.SMALL_MARKER_PT, zorder=4,
                            **ps.mark_floor(TASK_COL[task]))
    for task in MAIN_TASKS:
        col = TASK_COL[task]
        cells = all_cells[task]
        if task == MAZE:
            min_n = 0
        else:
            missing = [N for N in SIZES if N not in cells]
            assert not missing, f"{task}: no read-out at N = {missing}; the series would have to be drawn at end of training"
            min_n = FIT_MIN_N
        Ns = np.array(sorted(cells), float)
        m = np.array([cells[N].mean() for N in Ns])
        sd = np.array([cells[N].std(ddof=1) if len(cells[N]) > 1 else 0.0 for N in Ns])
        for ax, scale in ((ax_a, 1.0), (ax_b, 100.0 / Ns)):
            sc = np.atleast_1d(scale) * np.ones(len(Ns))
            # every value is read at the floor crossing, so every marker is filled (#83); N = 100 is
            # set apart by the fitted line, which starts above it
            # no caps, bars drawn under the mean's marker, so a small s.d. does not turn the round
            # marker into a square
            ax.errorbar(Ns, m * sc, yerr=sd * sc, fmt="none", ecolor=col, lw=1.0, capsize=0,
                        capthick=ps.DATA_LW, zorder=5)
            ax.plot(Ns, m * sc, "o", ms=ps.MARKER_PT, zorder=6, **ps.mark_floor(col))
        ax_b.plot(Ns, m * 100.0 / Ns, "-", color=col, lw=0.9, zorder=3)
        fitted = Ns[Ns >= min_n]
        if len(fitted) >= 2:
            b, A = power_fit(cells, min_n)
            xf = np.array([fitted.min() * 0.85, fitted.max() * 1.15])
            ax_a.plot(xf, A * xf ** b, ls="-", lw=0.9, color=col, zorder=3)
            label = f"{task}, $b$ = {b:.2f}"
        else:
            b, A, label = float("nan"), float("nan"), task
            print(f"  !! panel a: {task} has {len(fitted)} fitted size(s); no power law drawn")
        handles.append(Line2D([], [], color=col, marker="o", ms=3.0, lw=0.9, label=label))
        fits[task] = (b, A, {int(N): (float(mm), float(s), len(cells[N])) for N, mm, s in zip(Ns, m, sd)})
    if crosses:
        handles.append(Line2D([], [], ls="none", marker="x", ms=ps.SMALL_MARKER_PT + 0.8,
                              mec=TASK_COL[MAZE], mew=0.7, label=f"{MAZE}, not learned"))
    nn = np.array([80, 5000.0])
    ax_a.plot(nn, nn, "-", lw=ps.THIN_LW, color=ps.MUTED, zorder=2)
    # the label's angle follows the diagonal on screen, whatever the panel's aspect
    ax_a.text(250, 250, "every unit active", rotation=45, transform_rotates_text=True,
              fontsize=ps.TICK_PT, color=ps.MUTED, ha="center", va="bottom", rotation_mode="anchor")
    ax_a.set(xlabel=ps.LABELS["size"], ylabel=ps.LABELS["active"])
    ax_a.set_yticks([50, 100, 200, 500, 1000, 2000])
    ax_a.yaxis.set_minor_locator(NullLocator())
    ps.count_ticks(ax_a.yaxis)
    ax_a.legend(handles=handles, loc="lower right", handlelength=1.8, borderaxespad=0.2)
    for frac, lab in ((50, "50%"), (10, "10%")):
        ax_b.axhline(frac, ls=(0, (4, 3)), lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
        ax_b.text(SIZE_XLIM[0] * 1.03, frac + 1.5, lab, fontsize=ps.TICK_PT, color=ps.MUTED, ha="left", va="bottom")
    ax_b.set(xlabel=ps.LABELS["size"], ylabel="active units (% of $N$)")
    for ax in (ax_a, ax_b):
        ax.set_xticks(sizes)
        ax.xaxis.set_minor_locator(NullLocator())
        ps.count_ticks(ax.xaxis)
        ps.ygrid(ax)
    return fits


def panel_margins(ax, series, tasks=SUPERVISED_MAIN):
    """Panel (c): the fitted exponent at the 1.05x, 1.10x and 1.20x margins, bootstrap 95% intervals.

    The margins are margins over the supervised loss floor, so the maze (read at its learned
    probe, not at a loss floor) is not drawn here; draw_fig2 prints its exponent and interval.

    Args:
        ax: axes; series: the collect record's size-series rows; tasks: the supervised tasks drawn,
            left to right at each margin (default SUPERVISED_MAIN).
    Returns:
        dict (task, margin) -> (exponent, lo, hi, runs read).
    """
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    for k, task in enumerate(tasks):
        for j, d in enumerate(FLOOR_DELTAS):
            cells = cells_of(series, task, d)
            if any(N not in cells for N in SIZES if N >= FIT_MIN_N):
                print(f"  !! panel c, {task} at {1 + d:.2f}x: a size has no read-out; not drawn")
                continue
            b, _ = power_fit(cells)
            lo, hi = boot_exponent({N: v for N, v in cells.items() if N >= FIT_MIN_N}, rng)
            x = j + (k - (len(tasks) - 1) / 2) * 0.22     # the tasks centred on the margin's tick
            ax.errorbar([x], [b], yerr=[[b - lo], [hi - b]], fmt="none", ecolor=TASK_COL[task],
                        lw=ps.DATA_LW, capsize=1.0, capthick=ps.DATA_LW, zorder=5)   # caps under the marker
            ax.plot([x], [b], "o", ms=ps.MARKER_PT, zorder=6, **ps.mark_floor(TASK_COL[task]))
            out[(task, d)] = (b, lo, hi, int(sum(len(v) for N, v in cells.items() if N >= FIT_MIN_N)))
    ax.axhline(1.0, color=ps.MUTED, lw=ps.THIN_LW, ls="-")
    ax.text(2.4, 1.01, "constant fraction", fontsize=ps.TICK_PT, color=ps.MUTED, ha="right", va="bottom")
    ax.set_xticks(range(len(FLOOR_DELTAS)))
    ax.set_xticklabels([f"{1 + d:.2f}×" for d in FLOOR_DELTAS])
    ax.set(xlim=(-0.45, 2.45), ylim=(0, 1.1), xlabel="margin over the loss floor",
           ylabel="exponent $b$")
    ps.ygrid(ax)
    return out


def panel_maze_trajectories(at, ab, maze, N=MAZE_TRAJ_N):
    """Panel (e)'s maze column: probe success (top) and inactive units (bottom) against PPO update.

    Every finished maze run of size N: the clean probe's success rate (ink, with the learned
    criterion MAZE_LEARNED_SUCCESS as a dashed guide) and N minus the active count (the task's
    hue), on a linear update axis from 0 to the longest run's end rounded up to 500; a dashed
    vertical line at each learned run's learned probe. The success axis has its ticks and label
    on the right, because the column's left neighbour carries the supervised loss axis. An
    unfinished run is not drawn; with no run the column reads "no runs".

    Args:
        at, ab: the top and bottom axes of the column (sharing x); maze: maze_series rows;
        N: int, the size drawn.
    Returns:
        list of (MAZE, seed, learned update, inactive at the learned probe, inactive at the end,
        end update), nan where the run did not learn; one per run drawn.
    """
    rows = []
    runs = [r for r in maze if r["N"] == N and r["finished"]]
    x_max = int(np.ceil(max((r["it_end"] for r in runs), default=2000) / 500.0) * 500)
    for r in runs:
        at.plot(r["it_act"], r["success"], "-", lw=ps.THIN_LW, color=ps.INK, alpha=0.7)
        ab.plot(r["it_act"], N - r["active"], "-", lw=ps.THIN_LW + 0.2, color=TASK_COL[MAZE],
                alpha=0.85)
        if r["learned"]:
            for ax in (at, ab):
                ax.axvline(r["t_star"], color=ps.MUTED, lw=ps.THIN_LW, ls=(0, (3, 2)), zorder=1)
            rows.append((MAZE, r["seed"], r["t_star"], N - r["active_learned"], N - r["active_end"],
                         float(r["it_end"])))
        else:
            rows.append((MAZE, r["seed"], float("nan"), float("nan"), N - r["active_end"],
                         float(r["it_end"])))
    if not runs:
        at.text(0.5, 0.5, "no runs", transform=at.transAxes, ha="center", va="center",
                fontsize=ps.TICK_PT, color=ps.MUTED)
    at.axhline(MAZE_LEARNED_SUCCESS, ls=(0, (4, 3)), lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    at.set(xlim=(0, x_max), ylim=(0, 1.02))
    at.set_yticks([0, 0.5, 1.0])
    at.yaxis.tick_right()
    at.yaxis.set_label_position("right")
    at.set_ylabel("success rate")
    at.tick_params(labelbottom=False)
    at.set_title(f"{MAZE}, $N$ = {ps.fmt_count(N)}", pad=2)
    ab.set(xlim=(0, x_max), ylim=(0, N), xlabel="PPO update")
    for ax in (at, ab):
        ax.set_xticks([0, x_max / 2, x_max])
        ps.count_ticks(ax.xaxis)
    ab.set_yticks([0, 250, 500, 750, 1000])
    ps.count_ticks(ab.yaxis)
    ab.tick_params(labelleft=False)
    ps.ygrid(ab)
    return rows


def panel_trajectories(axes_top, axes_bot, traj, maze):
    """Panel (e): noisy loss over its start (top) and inactive units (bottom) against iteration,
    one column per task of MAIN_TASKS; the maze column is panel_maze_trajectories.

    Args:
        axes_top, axes_bot: one axes per task each; traj: the collect record's trajectory rows;
        maze: maze_series rows.
    Returns:
        list of (task, run seed, t*, inactive at t*, inactive at the end, end iteration).
    """
    rows = []
    x_hi = max(r["it_end"] for r in traj) * 1.15
    for k, task in enumerate(MAIN_TASKS):
        at, ab = axes_top[k], axes_bot[k]
        if task == MAZE:
            rows += panel_maze_trajectories(at, ab, maze)
            continue
        runs = [r for r in traj if r["task"] == task and r["excluded"] is None]
        N = runs[0]["N"]
        for r in runs:
            at.plot(r["tb"], r["lb"] / r["lb"][0], "-", lw=ps.THIN_LW, color=ps.INK, alpha=0.7)
            ab.plot(r["it_act"][1:], N - r["active"][1:], "-", lw=ps.THIN_LW + 0.2,
                    color=TASK_COL[task], alpha=0.85)
            if r["reason"] is None:
                for ax in (at, ab):
                    ax.axvline(r["t_star"], color=ps.MUTED, lw=ps.THIN_LW, ls=(0, (3, 2)), zorder=1)
                j = int(np.argmin(np.abs(r["it_act"] - r["t_star"])))
                rows.append((task, r["seed"], r["t_star"], N - r["active"][j], N - r["active"][-1],
                             r["it_act"][-1]))
            else:
                rows.append((task, r["seed"], float("nan"), float("nan"), N - r["active"][-1],
                             r["it_act"][-1]))
        at.set(xscale="log", yscale="log", xlim=(80, x_hi), ylim=(3e-4, 2.0))
        ab.set(xscale="log", xlim=(80, x_hi), ylim=(0, N), xlabel="training iteration")
        at.tick_params(labelbottom=False)
        at.set_title(f"{task}, $N$ = {ps.fmt_count(N)}", pad=2)
        at.set_yticks([1e-3, 1e-2, 1e-1, 1.0])
        at.set_yticklabels(["0.001", "0.01", "0.1", "1"])
        at.yaxis.set_minor_locator(NullLocator())
        for ax in (at, ab):
            ax.set_xticks([1e2, 1e3, 1e4, 1e5])
            ps.count_ticks(ax.xaxis)
            # label every other decade: four labels per sub-panel ran into one string (#86)
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda v, _pos: ps.fmt_count(v) if round(np.log10(v)) % 2 == 0 else ""))
        ab.set_yticks([0, 250, 500, 750, 1000])
        ps.count_ticks(ab.yaxis)
        if k:
            at.tick_params(labelleft=False)
            ab.tick_params(labelleft=False)
        ps.ygrid(ab)
    axes_top[0].set_ylabel("loss / initial loss")
    axes_bot[0].set_ylabel(ps.LABELS["inactive"])
    return rows


def block_medians(t, y, t0=T_FLAT, block=FLAT_BLOCK, t_end=None):
    """Medians of y over consecutive blocks (a, a + block] from t0 to t_end, written apart from
    c3_flat_loss_units.loss_change so that panel (f) checks the record rather than repeats it.

    Args:
        t, y: float arrays of the same length (iterations, values); t0: start of the first block;
        block: block length in iterations; t_end: end of the last block (default: t's last value).
    Returns:
        (block centres, block medians), float arrays; blocks with no sample are left out.
    """
    t_end = t[-1] if t_end is None else t_end
    starts = np.arange(t0, t_end - block + 1, block)
    centres, meds = [], []
    for a in starts:
        m = (t > a) & (t <= a + block)
        if m.any():
            centres.append(a + block / 2)
            meds.append(np.median(y[m]))
    return np.array(centres), np.array(meds)


def flat_quantities(row):
    """The two quantities of the flat-loss test for one network, from its raw traces.

    Loss: the last 5,000-iteration block median over the first, as a % change per 100,000 iterations
    between the block centres. Active units: the least-squares slope of the count against iteration
    over every probe from T_FLAT on (np.polyfit), per 100,000 iterations.

    Args:
        row: one row of the FLAT record (it_loss, loss, it_act, active).
    Returns:
        dict: tc, rel (block centres and the loss as % change from the first block), pct100 (loss
        change, % per 100,000), ta, da (count block centres, and block medians minus the first block's
        median, so the line starts at zero), slope (fitted change in active units per 100,000 iterations; minus the decrease),
        t_last (last probe), end (last loss iteration).
    """
    it, L = np.asarray(row["it_loss"], float), np.asarray(row["loss"], float)
    tc, med = block_medians(it, L)
    rel = 100.0 * (med / med[0] - 1.0)
    pct100 = rel[-1] / ((tc[-1] - tc[0]) / 1e5)
    ia, act = np.asarray(row["it_act"], float), np.asarray(row["active"], float)
    k = ia >= T_FLAT
    slope = np.polyfit((ia[k] - T_FLAT) / 1e5, act[k], 1)[0]
    ta, ma = block_medians(ia, act, t_end=it[-1])
    return dict(tc=tc, rel=rel, pct100=float(pct100), ta=ta, da=ma - ma[0], slope=float(slope),
                t_last=float(ia[k][-1]), end=float(it[-1]))


def panel_flat(axes, c3, flat):
    """Panel (f): the raw quantities of the flat-loss test, every 3BFF network trained past 190,000.

    Two columns, by training length (left: to 200,000 iterations; right: to 300,000), each with its
    own iteration axis. Top: the noisy training loss as medians over 5,000-iteration blocks, as a %
    change from the first block after 150,000; the grey wedge is the flat-loss band, a change of less
    than FLAT_PCT % per 100,000 iterations, opening from that first block's centre, so a line that
    ends inside the wedge is a flat loss. Bottom: the change in the number of active units since the
    first 5,000-iteration block after 150,000, as block medians (thin, starting at zero), and each
    network's least-squares line of the count from 150,000 on, drawn from zero at 150,000 (thick);
    the dashed line is the decrease of DIE_BAR units per 100,000 iterations, drawn over the data, so
    a thick line that ends below it is a decreasing count. Networks with a flat loss are coloured by
    N in shades of the 3BFF blue; the networks whose loss is not flat are FLAT_OFF grey in both rows
    and drawn under the others (review of PR #177, row 2), so the coloured thick lines are the test's
    15 networks. No marker: every network is a line, so the panel carries no marker encoding.

    The test quantities are recomputed here (`flat_quantities`, separate code) and must equal the
    collect record's panel-(f) rows to 1e-6.

    Args:
        axes: 2 x 2 nested list of axes [[top left, top right], [bottom left, bottom right]];
        c3: the collect record's panel-(f) rows; flat: the FLAT record's rows (raw traces).
    Returns:
        list of (N, seed, end, loss % per 100k, decrease per 100k, flat, decreasing), one per network.
    """
    rec = {r["seed"]: r for r in c3}
    assert sorted(rec) == sorted(r["seed"] for r in flat), "FLAT and the collect record differ"
    rows = []
    for col, (hi, title) in enumerate(((200_000, "trained to 200,000"), (300_000, "trained to 300,000"))):
        at, ab = axes[0][col], axes[1][col]
        t0 = T_FLAT + FLAT_BLOCK / 2                      # centre of the first loss block
        x = np.array([t0, hi])
        at.fill_between(x, -FLAT_PCT * (x - t0) / 1e5, FLAT_PCT * (x - t0) / 1e5, color=ps.GRID,
                        lw=0, zorder=0)
        x = np.array([T_FLAT, hi])
        ab.plot(x, -DIE_BAR * (x - T_FLAT) / 1e5, color=ps.INK, lw=0.9, ls=(0, (3, 1.5)), zorder=6)
        for r in sorted(flat, key=lambda r: r["N"]):
            q = flat_quantities(r)
            if abs(q["end"] - hi) > 1:
                continue
            c = rec[r["seed"]]
            assert abs(q["pct100"] - c["pct100"]) < 1e-6 and abs(-q["slope"] - c["lost"]) < 1e-6, r["seed"]
            flat_loss = abs(q["pct100"]) < FLAT_PCT
            shade, z = (FLAT_SHADE[r["N"]], 3) if flat_loss else (FLAT_OFF, 1)
            at.plot(q["tc"], q["rel"], color=shade, lw=0.7, zorder=z + 1)
            ab.plot(q["ta"], q["da"], color=shade, lw=0.35, alpha=0.6, zorder=z)
            ab.plot([T_FLAT, q["t_last"]], [0.0, q["slope"] * (q["t_last"] - T_FLAT) / 1e5],
                    color=shade, lw=1.0, zorder=z + 2, solid_capstyle="butt")
            rows.append((r["N"], r["seed"], q["end"], q["pct100"], -q["slope"],
                         abs(q["pct100"]) < FLAT_PCT, -q["slope"] >= DIE_BAR))
        at.set_title(title, pad=2)
        for ax in (at, ab):
            ax.set_xlim(T_FLAT, hi)
            ax.set_xticks([T_FLAT, hi] if col == 0 else [200_000, hi])
            ps.count_ticks(ax.xaxis)
            ps.ygrid(ax)
        at.set_ylim(-6, 5.5)
        at.set_yticks([-4, -2, 0, 2, 4])
        ab.set_ylim(-76, 14)              # the block medians span -72 to +10 units
        ab.set_yticks([-60, -40, -20, 0])
        at.tick_params(labelbottom=False)
        if col:
            at.tick_params(labelleft=False)
            ab.tick_params(labelleft=False)
    assert len(rows) == len(flat), "a network of panel (f) is in neither column"
    axes[0][0].set_ylabel("loss change (%)")
    axes[1][0].set_ylabel("change in\nactive units")
    # words on the two thresholds, in the right column where the wedge and the line are widest
    axes[0][1].text(298_000, -4.6, f"flat: under {FLAT_PCT:.0f}%\nper 100,000", fontsize=ps.TICK_PT,
                    color=ps.MUTED, ha="right", va="bottom", linespacing=1.0)
    axes[1][1].text(298_000, -4.0, f"−{DIE_BAR:.0f} per\n100,000", fontsize=ps.TICK_PT,
                    color=ps.INK, ha="right", va="bottom", linespacing=1.0)
    handles = [Line2D([], [], color=FLAT_SHADE[N], lw=1.0, label=f"$N$ = {ps.fmt_count(N)}")
               for N in sorted(FLAT_SHADE)]
    # below every loss line of the left column (none falls under -2%), where the axes is empty
    axes[0][0].legend(handles=handles, loc="lower left", ncol=2, handlelength=1.0,
                      handletextpad=0.3, borderaxespad=0.1, labelspacing=0.1, columnspacing=0.6)
    # the grey key, top right of the right column, over the wedge's fill where no line runs (on the
    # left it touched the loss bump near 207,000)
    axes[0][1].legend(handles=[Line2D([], [], color=FLAT_OFF, lw=1.0, label="loss not flat")],
                      loc="upper right", handlelength=1.0, handletextpad=0.3, borderaxespad=0.1)
    return rows


def panel_floor_schematic(ax):
    """Panel (d): a schematic of the floor-crossing definition, drawn on an idealised loss, not data.

    The curve is Eq. (floor) with L_inf = 1, i.e. 1 + A exp[-((t - t0)/s)^beta] with the SCHEMATIC_*
    parameters, so it is the loss divided by its floor and approaches 1 from above without crossing
    it. The band is [1, 1 + FLOOR_DELTAS[0]] (grey). Drawn: the curve (ink), the floor and the band's
    top, the crossing t* (filled marker: a floor-crossing value, #83) and the hold from t* to
    (1 + HOLD_FRAC) t* (double arrow). The curve falls monotonically, so t* is where it meets the
    band's top and it stays in the band through the hold. The iteration axis has no tick values, so
    the panel cannot be read as one network's data.

    Args:
        ax: matplotlib axes to draw into.
    Returns:
        (t*, axis end, curve value at the axis end), the iterations in the schematic's own units.
    """
    bar = 1.0 + FLOOR_DELTAS[0]
    fit = (1.0, SCHEMATIC_A, SCHEMATIC_S, SCHEMATIC_BETA)
    # the monotone curve meets the band's top where A exp[-((t - t0)/s)^beta] = FLOOR_DELTAS[0]
    t_star = SCHEMATIC_T0 + SCHEMATIC_S * np.log(SCHEMATIC_A / FLOOR_DELTAS[0]) ** (1.0 / SCHEMATIC_BETA)
    t_hold = (1.0 + HOLD_FRAC) * t_star
    x_hi, y_lo, y_hi = SCHEMATIC_X_END * t_star, 0.94, 1.25
    ts = np.linspace(SCHEMATIC_T0, x_hi, 600)
    ys = floor_curve(ts - SCHEMATIC_T0, fit)
    y_star = float(floor_curve(t_star - SCHEMATIC_T0, fit))
    ax.axhspan(1.0, bar, color=ps.GRID, lw=0, zorder=1)
    ax.axhline(1.0, color=ps.INK, lw=ps.THIN_LW, zorder=3)
    ax.axhline(bar, color=ps.INK, lw=ps.THIN_LW, ls=(0, (3, 2)), zorder=3)
    for t in (t_star, t_hold):      # from the bottom up to the curve, clear of the labels
        ax.plot([t, t], [y_lo, np.interp(t, ts, ys)], color=ps.MUTED, lw=ps.THIN_LW,
                ls=(0, (3, 2)), zorder=2)
    ps.arrow(ax, (t_star, 0.985), (t_hold, 0.985), col=ps.INK, lw=ps.THIN_LW, style="<|-|>",
             mutation_scale=5, shrink=0)
    ax.plot(ts, ys, "-", color=ps.INK, lw=ps.DATA_LW, zorder=5)
    ax.plot([t_star], [y_star], "o", ms=ps.MARKER_PT, zorder=6, **ps.mark_floor(ps.INK))
    ax.set(xlim=(0, x_hi), ylim=(y_lo, y_hi), xlabel="training iteration",
           ylabel="smoothed loss / loss floor")
    ax.text(x_hi * 0.02, 1.0 - 0.005, "loss floor", fontsize=ps.TICK_PT, color=ps.INK, ha="left",
            va="top", zorder=7)
    # right, above the band: the schematic's t* lies at 40% of the axis, so the label is clear of the
    # marker (PR #170, row 2), and the curve there is already inside the band, under the label
    ax.text(x_hi * 0.98, bar + 0.005, f"{bar:.2f}× floor", fontsize=ps.TICK_PT, color=ps.INK,
            ha="right", va="bottom", zorder=7)
    # the asterisk is Helvetica's own (raised) glyph: mathtext's \ast would add a second font family
    ax.text(t_star, 0.976, "$t$* ", fontsize=ps.TICK_PT, color=ps.INK, ha="right", va="top")
    ax.text(t_hold, 0.976, f" {1 + HOLD_FRAC:.1f}$\\,t$*", fontsize=ps.TICK_PT,
            color=ps.INK, ha="left", va="top")
    ax.set_title("schematic", pad=2)
    ax.set_xticks([])       # no iteration values: a schematic, not one network's data
    ax.set_yticks([1.0, 1.1, 1.2])
    return t_star, x_hi, float(ys[-1])


def check_overlaps(fig):
    """Print every pair of drawn text artists whose rendered boxes overlap (as fig_paper_F1's check).

    Args:
        fig: the finished figure.
    Returns:
        list of (text a, text b) pairs that overlap.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        leg = ax.get_legend()
        if leg is not None:
            texts += list(leg.get_texts())
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            for tick in axis.get_major_ticks():
                if lo <= tick.get_loc() <= hi and tick.label1.get_visible():
                    texts.append(tick.label1)
    texts = [t for t in texts if t.get_visible() and t.get_text().strip()]
    boxes = [(t, t.get_window_extent(rend)) for t in texts]
    hits = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(boxes) for b, bb in boxes[i + 1:]
            if ba.overlaps(bb) and ba.width > 0 and bb.width > 0]
    for a, b in hits:
        print(f"  !! text overlap: {a[:40]!r} <-> {b[:40]!r}")
    if not hits:
        print(f"  .. no overlapping text ({len(boxes)} text artists checked)")
    return hits


def draw_fig2(rec, flat, maze):
    """Draw Figure 2 from the collect record and the maze rows, and print every number it shows.

    Args:
        rec: the unpickled COLLECT record; flat: the unpickled FLAT record (panel f's raw traces);
        maze: maze_series rows (read from this repo when the figure is drawn).
    Returns:
        path of the PDF written.
    """
    ps.setup()
    fig = ps.figure(h_mm=FIG2_H_MM)
    # margins in figure fractions of 165.1 x FIG2_H_MM mm; panel letters sit 7 mm left of each spine
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.2], hspace=0.36,
                  left=0.075, right=0.975, top=0.955, bottom=0.095)
    # top row: a, b, c and the floor-crossing schematic d beside the margins panel c (item 47)
    top = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], width_ratios=TOP_WIDTHS, wspace=TOP_WSPACE)
    bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=FIG2_BOTTOM_WIDTHS,
                                  wspace=BOTTOM_WSPACE)
    egs = GridSpecFromSubplotSpec(2, 3, subplot_spec=bot[0], height_ratios=[1.0, 1.3], hspace=0.12,
                                  wspace=0.16)
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(top[i]) for i in range(4))
    e_top = [fig.add_subplot(egs[0, k]) for k in range(3)]
    e_bot = [fig.add_subplot(egs[1, k], sharex=e_top[k]) for k in range(3)]
    # panel f: two columns (trained to 200,000 / 300,000) of two rows (loss, active units)
    fgs = GridSpecFromSubplotSpec(2, 2, subplot_spec=bot[1], width_ratios=[1.25, 1.0], hspace=0.12,
                                  wspace=0.24)
    f_ax = [[None, None], [None, None]]
    f_ax[0][0] = fig.add_subplot(fgs[0, 0])
    f_ax[1][0] = fig.add_subplot(fgs[1, 0], sharex=f_ax[0][0])
    f_ax[0][1] = fig.add_subplot(fgs[0, 1], sharey=f_ax[0][0])
    f_ax[1][1] = fig.add_subplot(fgs[1, 1], sharex=f_ax[0][1], sharey=f_ax[1][0])

    fits = panel_size(ax_a, ax_b, rec["series"], maze)
    margins = panel_margins(ax_c, rec["series"])
    schem = panel_floor_schematic(ax_d)
    trows = panel_trajectories(e_top, e_bot, rec["traj"], maze)
    frows = panel_flat(f_ax, rec["c3"], flat)
    f_ax[1][0].set_xlabel("training iteration")
    f_ax[1][0].xaxis.set_label_coords(1.05, 0, transform=f_ax[1][0].transAxes + ScaledTranslation(
        0, -ps.TICK_PT * 1.6 / 72 - ps.LABEL_PAD_PT / 72, fig.dpi_scale_trans))   # under both columns
    for ax, L in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d"), (e_top[0], "e"), (f_ax[0][0], "f")):
        ps.panel_letter(ax, L, x_mm=9.0 if ax is f_ax[0][0] else ps.PANEL_LETTER_X_MM)
    check_overlaps(fig)
    out = ps.save(fig, "fig_paper_F2_size")

    print("\nPanels a-b (supervised: 1.05x floor; maze: learned probe; mean, SD, runs read):")
    for task, (b, A, cells) in fits.items():
        print(f"  {task}: N^{b:.3f}; 1000 active at N = {(1000 / A) ** (1 / b):,.0f}; " +
              "; ".join(f"N={N}: {m:.1f} +- {s:.1f} (n={n}, {100 * m / N:.1f}%)"
                        for N, (m, s, n) in cells.items()))
    print("Maze runs (N, seed, chain, end update, budget, learned update, active and success there, "
          "active and success at the end, status):")
    for r in maze:
        print(f"  N={r['N']} s{r['seed']} chain {r['n_chain']} end {r['it_end']:,} of {r['budget']:,}: "
              f"learned {r['t_star']:,.0f} active {r['active_learned']:.0f} success "
              f"{r['success_learned']:.2f}; end active {r['active_end']:.0f} success "
              f"{r['success_end']:.2f}; {r['reason'] or 'learned'}")
    for at, what in (("learned", "learned-probe counts (panel a's fit)"), ("end", "end-of-training counts")):
        b, _, lo, hi, cells = maze_fit(maze, at)
        print(f"  Maze, {what}, learned runs, every size with one: b = {b:.3f} [{lo:.3f}, {hi:.3f}]; "
              + "; ".join(f"N={N}: {v.mean():.1f} (n={len(v)}, {100 * v.mean() / N:.1f}%)"
                          for N, v in cells.items()))
    print("Panel c (exponent, bootstrap 95%, runs in fit):")
    for (task, d), (b, lo, hi, n) in margins.items():
        print(f"  {task} {1 + d:.2f}x: {b:.3f} [{lo:.3f}, {hi:.3f}] n={n}")
    print("Panel c, one network left out at a time (largest two upper ends: hi, b, N, seed):")
    for task in SUPERVISED_MAIN:
        for d in FLOOR_DELTAS:
            top = leave_one_out(rec["series"], task, d)[:2]
            print(f"  {task} {1 + d:.2f}x: " + "; ".join(f"{hi:.3f} (b {b:.3f}) without N={N} s{sd}"
                                                         for hi, b, N, sd in top))
    print(f"Panel d (schematic, Eq. floor with A/L_inf {SCHEMATIC_A}, s {SCHEMATIC_S:,.0f}, beta "
          f"{SCHEMATIC_BETA}, t0 {SCHEMATIC_T0:.0f}): t* {schem[0]:,.0f}, hold to "
          f"{(1 + HOLD_FRAC) * schem[0]:,.0f}, axis end {schem[1]:,.0f}, loss / floor there {schem[2]:.5f}")
    print("Panel e (task, seed, t* or learned update, inactive there, inactive at end, end):")
    for row in trows:
        print("  ", row, f"-> {row[4] - row[3]:+.0f} after the floor" if np.isfinite(row[3]) else "")
    print("Panel f (N, seed, end, loss %/100k, decrease/100k, flat, decreasing), recomputed from the "
          "raw traces and equal to the collect record to 1e-6:")
    for row in sorted(frows):
        print(f"   N={row[0]} s{row[1]} end {row[2]:,.0f} loss {row[3]:+.2f} decrease {row[4]:.1f}"
              f"{'  FLAT' if row[5] else ''}{'  DECREASING' if row[6] else ''}")
    n_flat = sum(r[5] for r in frows)
    print(f"  {len(frows)} networks, {n_flat} with a flat loss, {sum(r[5] and r[6] for r in frows)} "
          f"of those with a decreasing number of active units")
    return out


def tex_row(r, in_e, ed3a):
    """One LaTeX row of Extended Data Table 2 for one run record.

    Args:
        r: a size-series, trajectory or ablation row of the collect record.
        in_e: dict seed -> training length (iterations) of the panel-(f) networks (3BFF only).
        ed3a: set of (task, seed) of the networks Extended Data Fig. 3a draws (size series).
    Returns:
        str ending in a LaTeX line break.
    """
    read = r["reason"] is None and r.get("excluded") is None
    t = ps.fmt_count(1000 * round(r["t_star"] / 1e3)) if r["reason"] is None else "--"   # nearest 1,000
    a = f"{r['active_at'][0.05]:.0f}" if r["reason"] is None else "--"
    if r["reason"] is None and np.isfinite(r["r2_floor"]):
        r2 = f"{r['r2_floor']:.3f}" + ("" if r["r2_measured"] else "$^{e}$")
    elif np.isfinite(r["r2_end"]):
        r2 = f"({r['r2_end']:.3f})"
    else:
        r2 = "--"
    note = r.get("excluded") or (r["reason"] or "")
    if read:
        note = ""
    fig = {"size series": "2a--c", "trajectory": "2e", "Fig. 1d ablation": "1d"}[r["role"]]
    if r["role"] == "size series" and r["N"] == 1000:
        fig = "1c, 2a--c" + (", 2e" if r["task"] != "CDDM" else "")
    if not read:      # a network with no read-out is drawn only as a panel-(e) trajectory, if at all
        in_d = r["role"] == "trajectory" or (r["role"] == "size series" and r["N"] == 1000
                                              and r["task"] != "CDDM" and r.get("excluded") is None)
        fig = "2e" if in_d else "--"
    if r["task"] not in MAIN_TASKS:
        fig = "--"        # DMTS is in no main-text panel since round 14 (#182); its Extended Data
        # figure adds its own mark here once it exists
    trained = ps.fmt_count(1000 * round(r["it_end"] / 1e3))       # nearest 1,000
    if r["task"] == "3BFF" and r["seed"] in in_e:
        fig = "2f" if fig == "--" else fig + ", 2f"
        if in_e[r["seed"]] > r["it_end"]:      # continued for panel (f)
            trained += "$^{c}$"                # trained on to 300,000 (table footnote)
    if r["role"] == "Fig. 1d ablation" and r["task"] in ("3BFF", "CDDM"):
        # Fig. 4a-c draws the same weights as Fig. 1d for these six networks (dead-relu-paper #100,
        # #91); the 3BFF ones are also the old networks of Fig. 4e,f
        fig += ", 4a--c" + (", 4e,f" if r["task"] == "3BFF" else "")
    if r["role"] == "size series" and (r["task"], r["N"]) in TRANSFER_SOURCES:
        fig += ", 4e,f"      # transfer-test source at N = 4,000, young and old (dead-relu-paper #111)
    if r["role"] == "size series" and r["seed"] in ED1_NETWORKS[r["task"]]:
        fig = "ED1" if fig == "--" else fig + ", ED1"      # tested in Extended Data Fig. 1
    if r["role"] == "size series" and (r["task"], r["seed"]) in ed3a:
        fig = "ED3a" if fig == "--" else fig + ", ED3a"    # read-out example of Extended Data Fig. 3a
    note = re.sub(r" \(held-out R\$\^2\$ [0-9.]+\)", "", note)   # the R^2 column already shows it
    if (r.get("excluded") or "").startswith("never learned"):
        # the paper's wording for the MIN_R2 exclusion (dead-relu-paper #168); the collect-time
        # string stays, because apply_decisions and threshold_sensitivity match its prefix
        note = f"final $R^2$ below {MIN_R2[r['task']]}"
    note = note.replace("_", r"\_").replace("%", r"\%")
    return (f"{r['task']} & {r['N']:,} & {r['seed']} & {fig} & {trained} & {t} & {a} & "
            f"{r2} & {note} \\\\")


def tex_row_maze(r):
    """One LaTeX row of Supplementary Table 1 for one maze run (a maze_series row).

    In the table's columns: Trained is the chain's last PPO update, t* the learned probe's update,
    Active the count there, and the R^2 column holds the probe success rate at the learned probe
    (in brackets at the end of training for a finished run that did not learn). The note names
    gives the run's end of training: the active count there and, for a learned run, the success
    rate there (a run that did not learn has it in the R^2 column); it is kept under 25 characters
    because the table's Note column is the widest and the longtable overflows its line otherwise;
    the table's own notes say that Trained and t* count PPO updates. Fig.: a learned
    run is in 2a,b and, at N = MAZE_TRAJ_N, in 1c (Figure 1's maze_rows reads the same runs at
    the same probe) and 2e; a finished run that did not learn is a cross in 2a,b and a line in 2e;
    1d for a learned run of N = MAZE_TRAJ_N whose rl_maze_ablate.py JSON exists under
    MAZE_ABLATION_DIR (Figure 1's maze_ablation draws the learned N = 1,000 files only); every
    finished run is in the Extended Data maze figure (fig_ed_maze.py), marked EDM; an unfinished
    run is in no panel.

    Args:
        r: a maze_series row.
    Returns:
        str ending in a LaTeX line break.
    """
    fig = []
    if r["finished"]:
        if r["learned"] and r["N"] == MAZE_TRAJ_N:
            fig.append("1c")
        if r["learned"] and r["N"] == MAZE_TRAJ_N and glob.glob(
                os.path.join(MAZE_ABLATION_DIR, f"N={r['N']}_act=relu__s{r['seed']}.json")):
            fig.append("1d")
        fig.append("2a,b")
        if r["N"] == MAZE_TRAJ_N:
            fig.append("2e")
        fig.append("EDM")
    trained = ps.fmt_count(r["it_end"])
    t = ps.fmt_count(r["t_star"]) if r["learned"] else "--"
    a = f"{r['active_learned']:.0f}" if r["learned"] else "--"
    if r["learned"]:
        score = f"{r['success_learned']:.2f}"
        note = f"end: {r['active_end']:.0f} active, {r['success_end']:.2f}"
    elif r["finished"]:
        score, note = f"({r['success_end']:.2f})", f"not learned; {r['active_end']:.0f} at end"
    else:
        score, note = "--", "PPO updates; " + r["reason"]
    return (f"{MAZE} & {r['N']:,} & {r['seed']} & {', '.join(fig) or '--'} & {trained} & {t} & {a} & "
            f"{score} & {note} \\\\")


def write_table(rec, maze):
    """Write the body rows of Supplementary Table 1 to img/internal_figures/edt2_runs_rows.tex.

    The supervised rows come from the collect record (tex_row), grouped by task in TASKS order
    with the CDDM panel-(e) networks after CDDM's size series; the maze rows (tex_row_maze) follow
    the CDDM rows, before DMTS; the Fig. 1d ablation networks close the table.

    Args:
        rec: the unpickled COLLECT record; maze: maze_series rows.
    Returns:
        path written.
    """
    # grouped by task (size series, then the CDDM panel-e networks), seeds sorted as numbers
    grouped = [(r, 0) for r in rec["series"]] + [(r, 1) for r in rec["traj"] if r["task"] == "CDDM"]
    grouped.sort(key=lambda rk: (TASKS.index(rk[0]["task"]), rk[1], rk[0]["N"], int(rk[0]["seed"])))
    rows = [r for r, _ in grouped]
    rows += sorted(rec["ablation"], key=lambda r: (TASKS.index(r["task"]), int(r["seed"])))
    # panel (f): every network it draws must be exactly one 3BFF row of the table
    in_e = {C3_CONTINUED.get(r["seed"], r["seed"]): r["end"] for r in rec["c3"]}
    hits = [r["seed"] for r in rows if r["task"] == "3BFF" and r["seed"] in in_e]
    assert len(in_e) == len(rec["c3"]) and sorted(hits) == sorted(in_e), (sorted(in_e), hits)
    n_ed1 = sum(r["role"] == "size series" and r["seed"] in ED1_NETWORKS[r["task"]] for r in rows)
    assert n_ed1 == sum(len(v) for v in ED1_NETWORKS.values()) == 35, n_ed1
    # Extended Data Fig. 3a draws each task's first read-out example (fig_ed_readout.panel_examples)
    ed3a = {(task, seed_of(rec["examples"][task][0]["name"])) for task in TASKS}
    lines = [tex_row(r, in_e, ed3a) for r in rows]
    figcol = [line.split(" & ")[3] for line in lines]
    # panel d is a schematic and draws no network (3 Nov)
    assert sum("ED3a" in f for f in figcol) == len(ed3a) and sum("2d" in f.split(", ") for f in figcol) == 0
    # Fig. 4e,f draws six 3BFF networks: three at N = 1,000 (Fig. 1d) and three at N = 4,000 (#111)
    n_4ef = [r["N"] for r, line in zip(rows, lines) if "4e,f" in line.split(" & ")[3]]
    assert all(r["task"] == "3BFF" for r, line in zip(rows, lines) if "4e,f" in line.split(" & ")[3])
    assert sorted(n_4ef) == [1000] * 3 + [4000] * 3, n_4ef
    # the maze rows after the last CDDM size-series or panel-(e) row (before DMTS)
    k = 1 + max(i for i, r in enumerate(rows) if r["task"] == "CDDM" and r["role"] != "Fig. 1d ablation")
    lines = lines[:k] + [tex_row_maze(r) for r in maze] + lines[k:]
    path = os.path.join(IMG_DIR, "edt2_runs_rows.tex")
    with open(path, "w") as fh:
        fh.write("% written by fig_paper_F2_size.py from data/fig_paper_F2_size_collect.pkl\n")
        for line in lines:
            fh.write(line + "\n")
    print(f"wrote {path} ({len(lines)} rows, {len(maze)} of them maze runs)")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="on Della: read every run, write COLLECT")
    ap.add_argument("--collect-flat", action="store_true",
                    help="on Della: read panel (f)'s raw traces, write FLAT")
    args = ap.parse_args()
    if args.collect:
        collect()
    elif args.collect_flat:
        collect_flat()
    else:
        with open(COLLECT, "rb") as fh:
            record = apply_decisions(pickle.load(fh))
        with open(FLAT, "rb") as fh:
            flat_record = pickle.load(fh)
        maze_rows = maze_series()
        draw_fig2(record, flat_record, maze_rows)
        write_table(record, maze_rows)
