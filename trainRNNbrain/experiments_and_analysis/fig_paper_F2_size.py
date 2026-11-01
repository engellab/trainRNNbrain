#!/usr/bin/env python3
"""Manuscript Figure 2 (size and training time) and Supplementary Table 1; collect stage of Extended Data Fig. 3.

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
Extended Data Fig. 3 is drawn by fig_ed_readout.py from this script's collect record (examples, size
series and the snapshot oracle of /scratch/gpfs/TENGEL/pt1290/oracle_readout, Della job 15635966,
entry 2026-10-15_2054). The drawing stage also writes the rows of Supplementary Table 1 (every run of
Figures 1 and 2; formerly Extended Data Table 2).

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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
import heldout_r2 as heldout
from common import (DATA_DIR, FLOOR_DELTAS, HOLD_FRAC, IMG_DIR, floor_curve, logbin, read_run,
                    runs_with_loss, warm_start_chain)
from c3_flat_loss_units import CELLS as C3_CELLS, DIE_BAR, T_FLAT, FLAT_PCT, loss_change, loss_rate

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
TASK_COL = ps.TASK_COLORS     # reserved task hues of the style sheet (docs/figure_style.md)
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
# Panel (e) reads two Fig. 1d networks as their 300k continuations, whose folders carry new seeds.
# Continuation seed -> the 200k network's seed (its row in Extended Data Table 2), from each
# continuation's paths.init_from in NBitFlipFlop_c1_snapshots_c300000 (read on Della, 31 Oct).
C3_CONTINUED = {"3204992658": "4113307577", "3194339211": "2461724787"}
TASKNAME = {"NBitFlipFlop": "3BFF", "CDDM": "CDDM", "DMTS": "DMTS"}


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


# ---------------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------------
FIG2_H_MM = 125.0       # printed height of Figure 2 (width is \textwidth, 165.1 mm)
N_BOOT = 2000           # bootstrap resamples for the exponent's interval (panel c)
BOOT_SEED = 20261031
SIZE_MARK = {500: "o", 1000: "s", 2000: "D", 4000: "^"}


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


def power_fit(cells):
    """Least-squares line through log(mean active count) against log N, sizes N >= FIT_MIN_N.

    Args:
        cells: dict N -> array of active counts.
    Returns:
        (exponent b, prefactor A) with mean count ~ A N^b.
    """
    Ns = np.array([N for N in sorted(cells) if N >= FIT_MIN_N], float)
    ys = np.array([cells[N].mean() for N in sorted(cells) if N >= FIT_MIN_N])
    b, loga = np.polyfit(np.log(Ns), np.log(ys), 1)
    return float(b), float(np.exp(loga))


def boot_exponent(cells, rng):
    """Bootstrap 95% interval of the fitted exponent, resampling runs within each size.

    Args:
        cells: dict N -> array of active counts; rng: numpy Generator.
    Returns:
        (2.5th, 97.5th percentile) of the exponent over N_BOOT resamples.
    """
    bs = []
    for _ in range(N_BOOT):
        bs.append(power_fit({N: rng.choice(v, len(v)) for N, v in cells.items()})[0])
    return tuple(np.percentile(bs, [2.5, 97.5]))


def panel_size(ax_a, ax_b, series):
    """Panels (a) and (b): active units, and active units as a percentage of N, against N.

    Args:
        ax_a, ax_b: axes; series: the collect record's size-series rows.
    Returns:
        dict task -> (exponent, prefactor, {N: (mean, SD, n)}).
    """
    fits, handles = {}, []
    for task in TASKS:
        col = TASK_COL[task]
        cells = cells_of(series, task)
        missing = [N for N in SIZES if N not in cells]
        assert not missing, f"{task}: no read-out at N = {missing}; the series would have to be drawn at end of training"
        b, A = power_fit(cells)
        Ns = np.array(sorted(cells), float)
        m = np.array([cells[N].mean() for N in Ns])
        sd = np.array([cells[N].std(ddof=1) if len(cells[N]) > 1 else 0.0 for N in Ns])
        fit = Ns >= FIT_MIN_N
        for ax, scale in ((ax_a, 1.0), (ax_b, 100.0 / Ns)):
            for N, s in zip(Ns, np.atleast_1d(scale) * np.ones(len(Ns))):
                v = cells[N] * s
                ax.plot(np.full(len(v), N), v, "o", ms=ps.SMALL_MARKER_PT, color=col, alpha=0.45, mec="none",
                        zorder=4)
            sc = np.atleast_1d(scale) * np.ones(len(Ns))
            # fitted sizes filled; N = 100 (drawn, not fitted) open
            ax.errorbar(Ns[fit], (m * sc)[fit], yerr=(sd * sc)[fit], fmt="o", color=col, ms=3.0,
                        lw=1.0, capsize=1.6, zorder=5)
            ax.errorbar(Ns[~fit], (m * sc)[~fit], yerr=(sd * sc)[~fit], fmt="o", color=col, mfc="white",
                        ms=3.0, lw=1.0, capsize=1.6, zorder=5)
        ax_b.plot(Ns, m * 100.0 / Ns, "-", color=col, lw=0.9, zorder=3)
        xf = np.array([FIT_MIN_N * 0.85, 4000 * 1.15])
        ax_a.plot(xf, A * xf ** b, ls="-", lw=0.9, color=col, zorder=3)
        handles.append(Line2D([], [], color=col, marker="o", ms=3.0, lw=0.9,
                              label=f"{task}, $b$ = {b:.2f}"))
        fits[task] = (b, A, {int(N): (float(mm), float(s), len(cells[N])) for N, mm, s in zip(Ns, m, sd)})
    nn = np.array([80, 5000.0])
    ax_a.plot(nn, nn, "-", lw=ps.THIN_LW, color=ps.MUTED, zorder=2)
    ax_a.text(135, 175, "every unit active", rotation=37, fontsize=ps.TICK_PT, color=ps.MUTED,
              ha="center", va="bottom", rotation_mode="anchor")
    ax_a.set(xscale="log", yscale="log", xlim=(75, 5500), ylim=(50, 2500),
             xlabel=ps.LABELS["size"], ylabel=ps.LABELS["active"])
    ax_a.set_yticks([50, 100, 200, 500, 1000, 2000])
    ax_a.yaxis.set_minor_locator(NullLocator())
    ps.count_ticks(ax_a.yaxis)
    ax_a.legend(handles=handles, loc="lower right", handlelength=1.8, borderaxespad=0.2)
    for frac, lab in ((50, "50%"), (10, "10%")):
        ax_b.axhline(frac, ls=(0, (4, 3)), lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
        ax_b.text(80, frac + 1.5, lab, fontsize=ps.TICK_PT, color=ps.MUTED, ha="left", va="bottom")
    ax_b.set(xscale="log", xlim=(75, 5500), ylim=(0, 100), xlabel=ps.LABELS["size"],
             ylabel="active share (% of $N$)")
    for ax in (ax_a, ax_b):
        ax.set_xticks(SIZES)
        ax.xaxis.set_minor_locator(NullLocator())
        ps.count_ticks(ax.xaxis)
        ps.ygrid(ax)
    return fits


def panel_margins(ax, series):
    """Panel (c): the fitted exponent at the 1.05x, 1.10x and 1.20x margins, bootstrap 95% intervals.

    Args:
        ax: axes; series: the collect record's size-series rows.
    Returns:
        dict (task, margin) -> (exponent, lo, hi, runs read).
    """
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    for k, task in enumerate(TASKS):
        for j, d in enumerate(FLOOR_DELTAS):
            cells = cells_of(series, task, d)
            if any(N not in cells for N in SIZES if N >= FIT_MIN_N):
                print(f"  !! panel c, {task} at {1 + d:.2f}x: a size has no read-out; not drawn")
                continue
            b, _ = power_fit(cells)
            lo, hi = boot_exponent({N: v for N, v in cells.items() if N >= FIT_MIN_N}, rng)
            x = j + (k - 1) * 0.22
            ax.errorbar([x], [b], yerr=[[b - lo], [hi - b]], fmt="o", color=TASK_COL[task],
                        ms=ps.MARKER_PT, lw=ps.DATA_LW, capsize=1.6)
            out[(task, d)] = (b, lo, hi, int(sum(len(v) for N, v in cells.items() if N >= FIT_MIN_N)))
    ax.axhline(1.0, color=ps.MUTED, lw=ps.THIN_LW, ls="-")
    ax.text(2.45, 1.01, "constant share", fontsize=ps.TICK_PT, color=ps.MUTED, ha="right", va="bottom")
    ax.set_xticks(range(len(FLOOR_DELTAS)))
    ax.set_xticklabels([f"{1 + d:.2f}×" for d in FLOOR_DELTAS])
    ax.set(xlim=(-0.55, 2.55), ylim=(0, 1.1), xlabel="margin over the loss floor",
           ylabel="exponent $b$")
    ps.ygrid(ax)
    return out


def panel_trajectories(axes_top, axes_bot, traj):
    """Panel (d): noisy loss over its start (top) and inactive units (bottom) against iteration.

    Args:
        axes_top, axes_bot: one axes per task each; traj: the collect record's trajectory rows.
    Returns:
        list of (task, run seed, t*, inactive at t*, inactive at the end, end iteration).
    """
    rows = []
    x_hi = max(r["it_end"] for r in traj) * 1.15
    for k, task in enumerate(TASKS):
        at, ab = axes_top[k], axes_bot[k]
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
        ab.set_yticks([0, 250, 500, 750, 1000])
        ps.count_ticks(ab.yaxis)
        if k:
            at.tick_params(labelleft=False)
            ab.tick_params(labelleft=False)
        ps.ygrid(ab)
    axes_top[0].set_ylabel("loss / initial loss")
    axes_bot[0].set_ylabel(ps.LABELS["inactive"])
    return rows


def panel_flat(ax, c3):
    """Panel (e): units made inactive per 100,000 iterations from 150,000 on, against the loss change.

    Every network is 3BFF, so all marks take the 3BFF colour; marker shape codes N.

    Args:
        ax: axes; c3: the collect record's panel-(e) rows (one per 3BFF lineage).
    Returns:
        the rows drawn.
    """
    col = TASK_COL["3BFF"]
    ax.axvspan(-FLAT_PCT, FLAT_PCT, color=ps.GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(0, -13, "flat loss", fontsize=ps.TICK_PT, color=ps.MUTED, ha="center", va="bottom")
    ax.axhline(0, color=ps.MUTED, lw=ps.THIN_LW)
    # the losing-units threshold of the flat-loss test (entry 2026-10-15_0955)
    ax.axhline(DIE_BAR, color=ps.INK, lw=ps.THIN_LW, ls=(0, (3, 2)))
    ax.text(6.9, DIE_BAR + 1.5, "losing units", fontsize=ps.TICK_PT, color=ps.INK, ha="right",
            va="bottom")
    handles = []
    for N in sorted(SIZE_MARK):
        rows = [r for r in c3 if r["N"] == N]
        for r in rows:
            long_run = r["end"] > 250_000
            ax.errorbar([r["pct100"]], [r["lost"]], yerr=[r["se"]], fmt=SIZE_MARK[N], ms=ps.MARKER_PT,
                        color=col, mfc=col, mec=ps.INK if long_run else col, mew=0.6 if long_run else 0,
                        lw=0.8, capsize=1.2, zorder=4, alpha=0.85)
        if rows:
            handles.append(Line2D([], [], ls="none", marker=SIZE_MARK[N], ms=ps.MARKER_PT, color=col,
                                  mew=0, label=f"$N$ = {ps.fmt_count(N)}"))
    handles.append(Line2D([], [], ls="none", marker="s", ms=ps.MARKER_PT, color=col, mec=ps.INK,
                          mew=0.6, label=f"trained to {ps.fmt_count(300_000)}"))
    ax.set(xlim=(-6, 7), ylim=(-15, 135),
           xlabel="loss change after iteration 150,000\n(% per 100,000 iterations)",
           ylabel="units made inactive\nper 100,000 iterations")
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.legend(handles=handles, loc="upper left", handletextpad=0.2, borderaxespad=0.2)
    ps.ygrid(ax)
    return c3


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


def draw_fig2(rec):
    """Draw Figure 2 from the collect record and print every number it shows.

    Args:
        rec: the unpickled COLLECT record.
    Returns:
        path of the PDF written.
    """
    ps.setup()
    fig = ps.figure(h_mm=FIG2_H_MM)
    # margins in figure fractions of 165.1 x FIG2_H_MM mm; panel letters sit 7 mm left of each spine
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.2], hspace=0.36,
                  left=0.075, right=0.99, top=0.965, bottom=0.095)
    top = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], width_ratios=[1.2, 1.2, 0.85], wspace=0.36)
    bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=[2.5, 1.0], wspace=0.30)
    dgs = GridSpecFromSubplotSpec(2, 3, subplot_spec=bot[0], height_ratios=[1.0, 1.3], hspace=0.12,
                                  wspace=0.12)
    ax_a, ax_b, ax_c = (fig.add_subplot(top[i]) for i in range(3))
    d_top = [fig.add_subplot(dgs[0, k]) for k in range(3)]
    d_bot = [fig.add_subplot(dgs[1, k], sharex=d_top[k]) for k in range(3)]
    ax_e = fig.add_subplot(bot[1])

    fits = panel_size(ax_a, ax_b, rec["series"])
    margins = panel_margins(ax_c, rec["series"])
    trows = panel_trajectories(d_top, d_bot, rec["traj"])
    panel_flat(ax_e, rec["c3"])
    for ax, L in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (d_top[0], "d"), (ax_e, "e")):
        ps.panel_letter(ax, L, x_mm=9.0 if ax is ax_e else ps.PANEL_LETTER_X_MM)
    check_overlaps(fig)
    out = ps.save(fig, "fig_paper_F2_size")

    print("\nPanels a-b (1.05x floor; mean, SD, runs read):")
    for task, (b, A, cells) in fits.items():
        print(f"  {task}: N^{b:.3f}; 1000 active at N = {(1000 / A) ** (1 / b):,.0f}; " +
              "; ".join(f"N={N}: {m:.1f} +- {s:.1f} (n={n}, {100 * m / N:.1f}%)"
                        for N, (m, s, n) in cells.items()))
    print("Panel c (exponent, bootstrap 95%, runs in fit):")
    for (task, d), (b, lo, hi, n) in margins.items():
        print(f"  {task} {1 + d:.2f}x: {b:.3f} [{lo:.3f}, {hi:.3f}] n={n}")
    print("Panel d (task, seed, t*, inactive at t*, inactive at end, end):")
    for row in trows:
        print("  ", row, f"-> {row[4] - row[3]:+.0f} after the floor" if np.isfinite(row[3]) else "")
    print("Panel e (N, seed, end, t*, loss %/100k, lost/100k +- se):")
    for r in sorted(rec["c3"], key=lambda r: (r["N"], r["seed"])):
        print(f"   N={r['N']} s{r['seed']} end {r['end']:,.0f} t* {r['t_star']:,.0f} "
              f"loss {r['pct100']:+.2f} lost {r['lost']:.1f} +- {r['se']:.1f}"
              f"{'  FLAT' if abs(r['pct100']) < FLAT_PCT else ''}")
    return out


def tex_row(r, in_e):
    """One LaTeX row of Extended Data Table 2 for one run record.

    Args:
        r: a size-series, trajectory or ablation row of the collect record.
        in_e: dict seed -> training length (iterations) of the panel-(e) networks (3BFF only).
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
    fig = {"size series": "2a--c", "trajectory": "2d", "Fig. 1d ablation": "1d"}[r["role"]]
    if r["role"] == "size series" and r["N"] == 1000:
        fig = "1c, 2a--c" + (", 2d" if r["task"] != "CDDM" else "")
    if not read:      # a network with no read-out is drawn only as a panel-(d) trajectory, if at all
        in_d = r["role"] == "trajectory" or (r["role"] == "size series" and r["N"] == 1000
                                              and r["task"] != "CDDM" and r.get("excluded") is None)
        fig = "2d" if in_d else "--"
    trained = ps.fmt_count(1000 * round(r["it_end"] / 1e3))       # nearest 1,000
    if r["task"] == "3BFF" and r["seed"] in in_e:
        fig = "2e" if fig == "--" else fig + ", 2e"
        if in_e[r["seed"]] > r["it_end"]:      # continued for panel (e)
            trained += "$^{c}$"                # trained on to 300,000 (table footnote)
    note = re.sub(r" \(held-out R\$\^2\$ [0-9.]+\)", "", note)   # the R^2 column already shows it
    note = note.replace("_", r"\_").replace("%", r"\%")
    return (f"{r['task']} & {r['N']:,} & {r['seed']} & {fig} & {trained} & {t} & {a} & "
            f"{r2} & {note} \\\\")


def write_table(rec):
    """Write the body rows of Supplementary Table 1 to img/internal_figures/edt2_runs_rows.tex.

    Args:
        rec: the unpickled COLLECT record.
    Returns:
        path written.
    """
    # grouped by task (size series, then the CDDM panel-d networks), seeds sorted as numbers
    grouped = [(r, 0) for r in rec["series"]] + [(r, 1) for r in rec["traj"] if r["task"] == "CDDM"]
    grouped.sort(key=lambda rk: (TASKS.index(rk[0]["task"]), rk[1], rk[0]["N"], int(rk[0]["seed"])))
    rows = [r for r, _ in grouped]
    rows += sorted(rec["ablation"], key=lambda r: (TASKS.index(r["task"]), int(r["seed"])))
    # panel (e): every network it draws must be exactly one 3BFF row of the table
    in_e = {C3_CONTINUED.get(r["seed"], r["seed"]): r["end"] for r in rec["c3"]}
    hits = [r["seed"] for r in rows if r["task"] == "3BFF" and r["seed"] in in_e]
    assert len(in_e) == len(rec["c3"]) and sorted(hits) == sorted(in_e), (sorted(in_e), hits)
    path = os.path.join(IMG_DIR, "edt2_runs_rows.tex")
    with open(path, "w") as fh:
        fh.write("% written by fig_paper_F2_size.py from data/fig_paper_F2_size_collect.pkl\n")
        for r in rows:
            fh.write(tex_row(r, in_e) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="on Della: read every run, write COLLECT")
    args = ap.parse_args()
    if args.collect:
        collect()
    else:
        with open(COLLECT, "rb") as fh:
            record = apply_decisions(pickle.load(fh))
        draw_fig2(record)
        write_table(record)
