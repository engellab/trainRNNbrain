#!/usr/bin/env python3
"""Absorbing-boundary hypothesis: do units cross a zero-gradient boundary by diffusion and never return?

QUESTION. Pavel's proposal (notebook 2026-11-01_2119_absorbing-boundary-plan.md, which fixes every
prediction and threshold used here): units move "in some force field", there is a zero-gradient
boundary, stochasticity drives some units over it, and they never come back. The plan splits this
into P1 absorption, P2 stochastic crossing, P3 noise sets the rate, P4 the shape of the curve.

THE DISTANCE. For unit i at a weight snapshot, d_i = max over steps t >= 1 and every trial of the
state x_i of the h-equation on the noise-free reference batch. d_i <= 0 is the paper's "silent":
the rate is zero on every step, so on that batch the loss gradient with respect to the unit's
incoming and outgoing weights is zero.

INDEPENDENCE. The forward pass is this file's own numpy Euler integration of
    x[t] = x[t-1] + a (-x[t-1] + W_rec relu(x[t-1]) + W_inp u[t-1] + b),   a = dt / tau,
from the snapshot's parameter file. It does not use RNN_torch, the c4 scripts or any figure script.
The task class builds the input batch, and `common` supplies the warm-start chain and the project's
floor read-out (read_run), because those are the definitions the plan names.

STAGES.
  distances  (Della, CPU)  one network: d, participation and incoming-weight norms at every kept
                           snapshot along its warm-start chain, its floor read-out, and its
                           production participation trace (for oracle o3). One npz per network.
  ladder     (Della, CPU)  the CDDM recurrent-noise ladder (P3a): silent and active counts from each
                           run's participation trace, its floor read-out and smoothed loss.
  oracle     (Mac)         o1/o2 on the Fig. 1d weights copied to data/pop_stats_nets.
  report     (Mac)         P1-P4 from the npz files, per network; figures to img/internal_figures.

Usage:
  python absorb_boundary.py list --groups all                      # print the network run dirs
  python absorb_boundary.py distances RUN_DIR --snap-root DIR --out DIR
  python absorb_boundary.py ladder --data-root DIR --out FILE
  python absorb_boundary.py oracle
  python absorb_boundary.py report --dist DIR --ladder FILE
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
import time

import hydra
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from scipy import stats
from scipy.optimize import least_squares
from scipy.special import ndtr

from trainRNNbrain.training.training_utils import prepare_task_arguments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import paperstyle as ps

REF_SEED = 12345                 # task seed of the noise-free reference batch (C4) for seeded tasks
SILENT_P = 1e-6                  # participation below this on the trace probe = silent (common.SILENT_HARD)
DELLA_DATA = "/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data"

# Network groups (plan, "Networks"). Globs of run folders under <data-root>/trained_RNNs; for 3BFF the
# leaves of the warm-start chains are kept and their ancestors dropped.
GROUPS = {
    "cddm_control": ["CDDM_c1_snapshots/EqType=h_N=1000_arm=control/*seed=*"],
    "cddm_winp": ["CDDM_c1_snapshots/EqType=h_N=1000_arm=winp*/*seed=*"],
    "cddm_screen": ["CDDM_screen3_c75000/EqType=h_N=1000_pen=none_arm=none/*seed=*"],
    "dmts_control": ["DMTS_c1_snapshots/EqType=h_N=1000_arm=control/*seed=*"],
    "ff_control": ["NBitFlipFlop_c1_snapshots/EqType=h_N=1000_arm=control/*seed=*",
                   "NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control/*seed=*",
                   "NBitFlipFlop_c1_snapshots_c300000/EqType=h_N=1000_arm=control/*seed=*"],
    "cddm_noise": ["CDDM_absorb_noise/EqType=h_N=1000_noise=*/*seed=*"],
}
LADDER_GLOB = "CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=*/*seed=*"

# Pre-registered thresholds (plan, "Predictions"). Changing any of them after the run is not allowed.
P1_MIN_SILENT = 50
P1_PASS_STAR, P1_PASS_ALL, P1_FAIL_STAR, P1_FROM = 0.95, 0.90, 0.80, 5000
P2_K, P2_FROM = 20, 10500
P2_S_DIFF, P2_S_DRIFT = 0.4, 0.6
P2_RHO_UNC, P2_RHO_SMOOTH = 0.2, 0.4
P2_LATE_MIN = 0.10
P2_VMR_IND, P2_VMR_BURST = 2.0, 5.0
P3A_RHO, P3A_P = 0.4, 0.05
P3A_WIN = 50000
P3B_PASS, P3B_FAIL = 0.5, 0.75
P3B_FALLBACK = (30000, 75000)
P4_T0, P4_DBIC = 5000, 10.0


# --------------------------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------------------------

def list_networks(data_root, group):
    """Run folders of one network group, leaves of their warm-start chains only.

    Args:
        data_root: str, the folder holding trained_RNNs/ (Della: DELLA_DATA).
        group: str, a key of GROUPS.
    Returns:
        sorted list of run-folder paths; a folder that another listed run was warm-started from
        (an ancestor) is dropped, so each network appears once, as its latest continuation.
    """
    runs = []
    for pat in GROUPS[group]:
        runs += [d for d in glob.glob(os.path.join(data_root, "trained_RNNs", pat)) if os.path.isdir(d)]
    runs = sorted(set(os.path.realpath(d) for d in runs))
    ancestors = set()
    for d in runs:
        for a in common.warm_start_chain(d)[:-1]:
            ancestors.add(os.path.realpath(a))
    return [d for d in runs if d not in ancestors]


def seed_of(run_dir):
    """The seed in a run folder's name ('...;seed=<n>;...'), as an int."""
    return int(re.search(r"seed=(\d+)", os.path.basename(run_dir.rstrip("/"))).group(1))


def chain_snapshots(run_dir, snap_root):
    """Every kept snapshot along a run's warm-start chain, oldest first.

    A run's snapshots sit at <snap_root>/<sweep>/<cell>/s<seed>/it<iters>/, where the run folder is
    .../trained_RNNs/<sweep>/<cell>/<run>. A later run of the chain wins where two share an
    iteration.

    Args:
        run_dir: str, the leaf run folder; snap_root: str, the snapshot tree root (c1_snapshots).
    Returns:
        list of (iteration int, parameter-file path, config path), sorted by iteration.
    """
    found = {}
    for d in common.warm_start_chain(run_dir):
        cell = os.path.dirname(d.rstrip("/"))
        sdir = os.path.join(snap_root, os.path.basename(os.path.dirname(cell)), os.path.basename(cell),
                            f"s{seed_of(d)}")
        for it_dir in glob.glob(os.path.join(sdir, "it*")):
            par = glob.glob(os.path.join(it_dir, "*LastParams*.npz"))
            cfg = glob.glob(os.path.join(it_dir, "*config.yaml"))
            if par and cfg:
                found[int(os.path.basename(it_dir)[2:])] = (par[0], cfg[0])
    return [(t, p, c) for t, (p, c) in sorted(found.items())]


# --------------------------------------------------------------------------------------------
# the independent forward pass
# --------------------------------------------------------------------------------------------

def load_snapshot(path):
    """A snapshot's parameters as float64 arrays.

    Args:
        path: str, a *LastParams*.npz file.
    Returns:
        dict: W_rec (N,N), W_inp (N,n_in), W_out (n_out,N), y0 (N,), b (N,), a (float, dt/tau),
        slope (float, the ReLU slope).
    """
    z = np.load(path, allow_pickle=True)
    act = z["activation_args"].item()
    if act["name"] != "relu":
        raise ValueError(f"{path}: activation {act['name']}, expected relu")
    N = z["W_rec"].shape[0]
    b = np.asarray(z["bias"], float).reshape(-1) if z["bias"].shape else np.zeros(N)
    return dict(W_rec=z["W_rec"].astype(np.float64), W_inp=z["W_inp"].astype(np.float64),
                W_out=z["W_out"].astype(np.float64), y0=z["y_init"].astype(np.float64).reshape(-1),
                b=b, a=float(z["dt"]) / float(z["tau"]), slope=float(act["slope"]))


def reference_batch(cfg_path):
    """The noise-free reference batch of a network's task.

    CDDM builds the same trials on every call; every other task is seeded with REF_SEED.

    Args:
        cfg_path: str, a run or snapshot config (*config.yaml) with task and model.dt.
    Returns:
        (u (n_in, T, B) float64 inputs, taskname str).
    """
    cfg = OmegaConf.load(cfg_path)
    targs = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    targs.seed = REF_SEED
    u, _, _ = hydra.utils.instantiate(targs).get_batch()
    return np.asarray(u, np.float64), str(cfg.task.taskname)


def unit_stats(P, u):
    """Distance to the boundary and participation of every unit on one batch.

    Args:
        P: load_snapshot dict; u: (n_in, T, B) inputs.
    Returns:
        d (N,) max over t >= 1 and trials of x; p (N,) participation std(r) + q0.9(|r|) over all
        (t, trial) samples including t = 0, as the production probe computes it; n_pos (N,) the
        number of (t >= 1, trial) samples with x > 0.
    """
    n_in, T, B = u.shape
    N = P["W_rec"].shape[0]
    a = P["a"]
    x = np.repeat(P["y0"][:, None], B, axis=1)
    R = np.empty((T, N, B), np.float32)                    # rates, kept for the participation
    R[0] = np.maximum(P["slope"] * x, 0.0)
    d = np.full(N, -np.inf)
    n_pos = np.zeros(N, np.int64)
    for t in range(1, T):
        r = np.maximum(P["slope"] * x, 0.0)
        x = x + a * (-x + P["W_rec"] @ r + P["W_inp"] @ u[:, t - 1, :] + P["b"][:, None])
        d = np.maximum(d, x.max(axis=1))
        n_pos += (x > 0).sum(axis=1)
        R[t] = np.maximum(P["slope"] * x, 0.0)
    flat = R.transpose(1, 0, 2).reshape(N, -1).astype(np.float64)
    p = flat.std(axis=1) + np.quantile(np.abs(flat), 0.9, axis=1)
    return d, p, n_pos


def active_bar(p):
    """Active units under the project's rule: p >= 5% of the network's q95 of p (own copy)."""
    return p >= 0.05 * np.quantile(p, 0.95)


# --------------------------------------------------------------------------------------------
# Della stages
# --------------------------------------------------------------------------------------------

def run_distances(run_dir, snap_root, out_dir, every=1):
    """Stage `distances` for one network; writes <out_dir>/<group-free name>.npz.

    Args:
        run_dir: str, the leaf run folder; snap_root: str, snapshot tree root; out_dir: str;
        every: int, use every `every`-th snapshot (1 = all).
    Returns:
        the output path.
    """
    snaps = chain_snapshots(run_dir, snap_root)[::every]
    if not snaps:
        raise SystemExit(f"no snapshots for {run_dir}")
    u, task = reference_batch(snaps[0][2])
    its, D, Pp, NP, WR, WI = [], [], [], [], [], []
    tic = time.time()
    for k, (t, par, _) in enumerate(snaps):
        P = load_snapshot(par)
        d, p, n_pos = unit_stats(P, u)
        its.append(t); D.append(d); Pp.append(p); NP.append(n_pos)
        WR.append(np.linalg.norm(P["W_rec"], axis=1)); WI.append(np.linalg.norm(P["W_inp"], axis=1))
        if k % 20 == 0:
            print(f"  it {t}: silent {(d <= 0).sum()} active {active_bar(p).sum()} "
                  f"({time.time() - tic:.0f} s)", flush=True)
    rd = common.read_run(run_dir)
    # the production trace of every run of the chain (oracle o3, and P3b counts between snapshots)
    tr_it, tr_p = [], []
    for d_ in common.warm_start_chain(run_dir):
        tp = glob.glob(os.path.join(d_, "*ParticipationTrace.pkl"))
        if tp:
            with open(tp[0], "rb") as fh:
                tr = pickle.load(fh)
            tr_it += list(np.asarray(tr["participation_iters"], float))
            tr_p += list(np.asarray(tr["participation"], np.float32))
    cfg = OmegaConf.load(snaps[-1][2])
    name = "__".join(os.path.realpath(run_dir).split(os.sep)[-3:-1] + [f"s{seed_of(run_dir)}"])
    out = os.path.join(out_dir, name + ".npz")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        out, it=np.array(its), d=np.array(D), p=np.array(Pp), n_pos=np.array(NP),
        w_rec_row=np.array(WR), w_inp_row=np.array(WI), task=task, run_dir=run_dir,
        chain=np.array(common.warm_start_chain(run_dir)),
        t_star=float(rd["t_star"]), reason=str(rd["reason"]),
        t_stars=json.dumps({str(k): float(v) for k, v in (rd.get("t_stars") or {}).items()}),
        it_loss=np.asarray(rd["it_loss"], float) if rd["it_loss"] is not None else np.zeros(0),
        smoothed=np.asarray(rd.get("smoothed", np.zeros(0)), float),
        trace_it=np.array(tr_it), trace_p=np.array(tr_p, np.float32),
        sigmas=np.array([float(cfg.model.sigma_rec), float(cfg.model.sigma_inp), float(cfg.model.sigma_out)]),
        n_trials=u.shape[2], T=u.shape[1])
    print(f"wrote {out}: {len(its)} snapshots, t* {rd['t_star']} ({rd['reason']})")
    return out


def run_ladder(data_root, out_path):
    """Stage `ladder`: the CDDM recurrent-noise ladder from participation traces (P3a).

    Args:
        data_root: str, the folder holding trained_RNNs/; out_path: str, a .pkl to write.
    Returns:
        None; writes a list of per-run dicts: sigma_rec, budget, seed, t_star, reason, it (probe
        iterations), n_silent and n_active per probe, it_loss and the smoothed noisy loss.
    """
    rows = []
    for d in sorted(glob.glob(os.path.join(data_root, "trained_RNNs", LADDER_GLOB))):
        cell = os.path.basename(os.path.dirname(d))
        m = re.search(r"sigrec=([0-9.]+)(_iters=(\d+))?", cell)
        tp = glob.glob(os.path.join(d, "*ParticipationTrace.pkl"))
        if not tp or "nan" in os.path.basename(d):
            continue
        with open(tp[0], "rb") as fh:
            tr = pickle.load(fh)
        Pm = np.asarray(tr["participation"], np.float32)
        rd = common.read_run(d)
        cfg = OmegaConf.load(glob.glob(os.path.join(d, "*config.yaml"))[0])
        rows.append(dict(
            sigma_rec=float(m.group(1)), budget=int(m.group(3) or 75000), seed=seed_of(d), run_dir=d,
            sigmas=(float(cfg.model.sigma_rec), float(cfg.model.sigma_inp), float(cfg.model.sigma_out)),
            bias_range=list(cfg.model.bias_range), same_batch=bool(cfg.trainer.same_batch),
            t_star=float(rd["t_star"]), reason=rd["reason"],
            it=np.asarray(tr["participation_iters"], float),
            n_silent=(Pm < SILENT_P).sum(axis=1), n_active=np.array([active_bar(p).sum() for p in Pm]),
            it_loss=np.asarray(rd["it_loss"], float), smoothed=np.asarray(rd.get("smoothed", []), float)))
        print(f"{cell} s{rows[-1]['seed']}: t* {rows[-1]['t_star']} ({rd['reason']}), "
              f"silent end {rows[-1]['n_silent'][-1]}")
    with open(out_path, "wb") as fh:
        pickle.dump(rows, fh)
    print(f"wrote {out_path}: {len(rows)} runs")


# --------------------------------------------------------------------------------------------
# oracle o1/o2 (Mac)
# --------------------------------------------------------------------------------------------

def run_oracle():
    """Oracles o1 and o2 of the plan on the Fig. 1d weights in data/pop_stats_nets.

    Returns:
        True when every network matches (CDDM silent and active exactly; 3BFF active within 5 and 0
        silent); prints one line per network.
    """
    expect = {"s1423687489": (310, 430), "s2657899762": (288, 416), "s2670295995": (253, 438),
              "s1258717025": (0, 269), "s2461724787": (0, 342), "s4113307577": (0, 287)}
    ok = True
    root = os.path.join(common.DATA_DIR, "..", "pop_stats_nets")
    for net, (n_sil, n_act) in expect.items():
        d_ = glob.glob(os.path.join(root, f"*__{net}__floor"))[0]
        P = load_snapshot(glob.glob(os.path.join(d_, "*LastParams*.npz"))[0])
        u, task = reference_batch(glob.glob(os.path.join(d_, "*config.yaml"))[0])
        d, p, _ = unit_stats(P, u)
        s, a = int((d <= 0).sum()), int(active_bar(p).sum())
        good = (s == n_sil and a == n_act) if task == "CDDM" else (s == 0 and abs(a - n_act) <= 5)
        ok &= good
        print(f"{task:13s} {net}: silent {s} (expect {n_sil}), active {a} (expect {n_act}) "
              f"{'ok' if good else 'MISMATCH'}")
    return ok


# --------------------------------------------------------------------------------------------
# report (Mac)
# --------------------------------------------------------------------------------------------

def load_dist(dist_dir):
    """Every `distances` npz in a folder.

    Args:
        dist_dir: str.
    Returns:
        dict name -> dict of arrays (it, d, p, ...), with `group` set from the file name.
    """
    nets = {}
    for f in sorted(glob.glob(os.path.join(dist_dir, "*.npz"))):
        z = dict(np.load(f, allow_pickle=True))
        name = os.path.basename(f)[:-4]
        z["name"] = name
        z["task"] = str(z["task"])
        z["t_star"] = float(z["t_star"])
        z["reason"] = str(z["reason"])
        z["group"] = ("cddm_noise_zero" if "noise=zero" in name else
                      "cddm_noise_default" if "noise=default" in name else
                      "cddm_winp" if "arm=winp" in name else
                      "cddm_screen" if "screen3" in name else
                      "cddm_control" if name.startswith("CDDM") else
                      "dmts_control" if name.startswith("DMTS") else "ff_control")
        nets[name] = z
    return nets


def absorption(it, beyond, t_from):
    """Absorption fraction at every snapshot (P1).

    Args:
        it: (S,) snapshot iterations; beyond: (S, N) bool, True where the unit is past the boundary;
        t_from: int, first snapshot iteration to report.
    Returns:
        (its, A, n): snapshot iterations >= t_from (excluding the last), the fraction of units past
        the boundary there that stay past it at every later snapshot, and how many were past it.
    """
    # stays[s, i] = beyond at s and at every later snapshot
    stays = np.flip(np.logical_and.accumulate(np.flip(beyond, 0), axis=0), 0)
    sel = np.flatnonzero((it >= t_from))[:-1]
    n = beyond[sel].sum(axis=1)
    A = np.where(n > 0, stays[sel].sum(axis=1) / np.maximum(n, 1), np.nan)
    return it[sel], A, n


def nearest(it, t):
    """Index of the snapshot iteration nearest t."""
    return int(np.argmin(np.abs(np.asarray(it, float) - t)))


def p1_network(z):
    """P1 for one network: absorption at the crossing and at every snapshot from P1_FROM.

    Args:
        z: a load_dist record.
    Returns:
        dict with n_silent_star, A_star, A_min, A_med, back_active (share of units silent at s* that
        are active at some later snapshot), n_back (units silent at s* with d > 0 later), verdict.
    """
    it, d, p = z["it"], z["d"], z["p"]
    beyond = d <= 0
    t_ref = z["t_star"] if np.isfinite(z["t_star"]) else np.nan
    its, A, n = absorption(it, beyond, P1_FROM)
    j = nearest(it, t_ref) if np.isfinite(t_ref) else None
    out = dict(n_silent_star=int(beyond[j].sum()) if j is not None else None,
               A_min=float(np.nanmin(A)) if len(A) else np.nan,
               A_med=float(np.nanmedian(A)) if len(A) else np.nan, t_ref=t_ref)
    if j is None or j >= len(it) - 1:
        out.update(A_star=np.nan, back_active=np.nan, n_back=None, verdict="no read-out")
        return out
    sil = beyond[j]
    later = beyond[j + 1:]
    act_later = np.array([active_bar(pp) for pp in p[j + 1:]])
    out["A_star"] = float((sil & later.all(axis=0)).sum() / max(sil.sum(), 1))
    out["n_back"] = int((sil & ~later.all(axis=0)).sum())
    out["back_active"] = float((sil & act_later.any(axis=0)).sum() / max(sil.sum(), 1))
    if sil.sum() < P1_MIN_SILENT:
        out["verdict"] = "not tested (< 50 silent)"
    elif out["A_star"] >= P1_PASS_STAR and out["A_min"] >= P1_PASS_ALL:
        out["verdict"] = "pass"
    elif out["A_star"] < P1_FAIL_STAR:
        out["verdict"] = "falsified"
    else:
        out["verdict"] = "inconclusive"
    return out


def first_crossings(it, d):
    """Each unit's first snapshot index with d <= 0 (P2), counting only units with d > 0 at the first snapshot.

    Args:
        it: (S,) iterations; d: (S, N) distances.
    Returns:
        (N,) int index of the first snapshot at or past the boundary, -1 for never or already past
        at the first snapshot.
    """
    beyond = d <= 0
    idx = np.where(beyond.any(axis=0), beyond.argmax(axis=0), -1)
    idx[beyond[0]] = -1
    return idx


def p2_network(z):
    """P2 for one network: straightness, increment autocorrelation, late crossings, dispersion.

    Args:
        z: a load_dist record.
    Returns:
        dict with n_cross (units with a first crossing at >= P2_FROM and K intervals before it),
        S_med, rho1, late_frac (share of units silent at the last snapshot whose first crossing is
        after t*), vmr (variance/mean of first crossings per interval after t*), and verdicts.
    """
    it, d = z["it"], z["d"]
    fc = first_crossings(it, d)
    K = P2_K
    S, inc_res = [], []
    for i in np.flatnonzero(fc >= 0):
        k = fc[i]
        if it[k] < P2_FROM or k < K:
            continue
        seg = d[k - K:k + 1, i]
        dd = np.diff(seg)
        S.append(abs(seg[-1] - seg[0]) / max(np.abs(dd).sum(), 1e-12))
        inc_res.append(dd - dd.mean())
    if inc_res:
        R = np.array(inc_res)
        a_, b_ = R[:, :-1].ravel(), R[:, 1:].ravel()
        rho1 = float(np.corrcoef(a_, b_)[0, 1])
    else:
        rho1 = np.nan
    t_ref = z["t_star"]
    end_sil = d[-1] <= 0
    if np.isfinite(t_ref):
        crossed_after = (fc >= 0) & (it[np.maximum(fc, 0)] > t_ref)
        late_frac = float((end_sil & crossed_after).sum() / max(end_sil.sum(), 1))
        sel = it > t_ref
        counts = np.array([((fc == k)).sum() for k in np.flatnonzero(sel)])
        vmr = float(counts.var() / counts.mean()) if counts.size and counts.mean() > 0 else np.nan
    else:
        late_frac, vmr = np.nan, np.nan
    S_med = float(np.median(S)) if S else np.nan
    va = ("diffusive" if S_med < P2_S_DIFF else "drift-like" if S_med > P2_S_DRIFT else "inconclusive") \
        if S else "no crossings"
    vb = ("uncorrelated" if abs(rho1) < P2_RHO_UNC else "smooth" if rho1 > P2_RHO_SMOOTH else "inconclusive") \
        if np.isfinite(rho1) else "no crossings"
    vc = ("holds" if late_frac >= P2_LATE_MIN else "fails") if np.isfinite(late_frac) else "no read-out"
    vd = ("independent" if vmr < P2_VMR_IND else "bursts" if vmr > P2_VMR_BURST else "inconclusive") \
        if np.isfinite(vmr) else "no read-out"
    return dict(n_cross=len(S), S_med=S_med, S_q=(np.percentile(S, [25, 75]) if S else (np.nan, np.nan)),
                rho1=rho1, late_frac=late_frac, n_end_silent=int(end_sil.sum()), vmr=vmr,
                va=va, vb=vb, vc=vc, vd=vd, S_all=np.array(S))


def slope_per_10k(t, y, lo, hi):
    """Least-squares slope of y against t over lo <= t <= hi, per 10,000 iterations (nan if < 5 points)."""
    m = (t >= lo) & (t <= hi)
    if m.sum() < 5:
        return np.nan
    return float(np.polyfit(t[m], y[m], 1)[0] * 1e4)


def loss_change(it_loss, smoothed, lo, hi):
    """Relative change of the smoothed noisy loss from lo to hi: L(hi)/L(lo) - 1 (nan if unavailable)."""
    if len(smoothed) != len(it_loss) or not len(smoothed):
        return np.nan
    a, b = np.searchsorted(it_loss, lo), min(np.searchsorted(it_loss, hi), len(it_loss) - 1)
    return float(smoothed[b] / smoothed[a] - 1.0) if a < len(smoothed) else np.nan


def p3a(rows):
    """P3a on the recurrent-noise ladder.

    Args:
        rows: the ladder stage's list of per-run dicts.
    Returns:
        dict budget -> dict(table=list of per-run tuples, rho, p_one_sided, mean_by_sigma, verdict).
    """
    out = {}
    for budget in sorted(set(r["budget"] for r in rows)):
        tab = []
        for r in sorted((r for r in rows if r["budget"] == budget), key=lambda r: (r["sigma_rec"], r["seed"])):
            ts = r["t_star"]
            hi = min(ts + P3A_WIN, r["it"][-1]) if budget > 75000 else r["it"][-1]
            ok = r["reason"] in (None, "None") and np.isfinite(ts)
            sl = slope_per_10k(r["it"], r["n_silent"].astype(float), ts, hi) if ok else np.nan
            sa = slope_per_10k(r["it"], r["n_active"].astype(float), ts, hi) if ok else np.nan
            dl = loss_change(r["it_loss"], r["smoothed"], ts, hi) if ok else np.nan
            j = nearest(r["it"], ts) if ok else -1
            tab.append((r["sigma_rec"], r["seed"], ts, hi, sl, sa, dl, int(r["n_silent"][j]) if ok else None,
                        int(r["n_silent"][-1]), r["reason"]))
        good = [t for t in tab if np.isfinite(t[4])]
        sig = np.array([t[0] for t in good]); sl = np.array([t[4] for t in good])
        rho, p2 = stats.spearmanr(sig, sl) if len(good) > 3 else (np.nan, np.nan)
        p1 = p2 / 2 if rho > 0 else 1 - p2 / 2
        means = {s: float(np.mean(sl[sig == s])) for s in sorted(set(sig))}
        lo_s, hi_s = min(means), max(means)
        if rho >= P3A_RHO and p1 < P3A_P and means[hi_s] > means[lo_s]:
            v = "pass"
        elif rho <= 0 or means[lo_s] >= means[hi_s]:
            v = "falsified"
        else:
            v = "inconclusive"
        out[budget] = dict(table=tab, rho=float(rho), p_one_sided=float(p1), means=means, verdict=v,
                           n_used=len(good), n_all=len(tab))
    return out


def p3b(nets):
    """P3b on the noise-free / default-noise pairs.

    Args:
        nets: load_dist dict.
    Returns:
        list of per-pair dicts (seed, window, slope_zero, slope_default, ratio, crossings after the
        window start for each arm, verdict per pair) and an overall verdict string.
    """
    zero = {seed_of(str(z["run_dir"])): z for z in nets.values() if z["group"] == "cddm_noise_zero"}
    dflt = {seed_of(str(z["run_dir"])): z for z in nets.values() if z["group"] == "cddm_noise_default"}
    pairs = []
    for s in sorted(set(zero) & set(dflt)):
        a, b = zero[s], dflt[s]
        if np.isfinite(a["t_star"]) and np.isfinite(b["t_star"]):
            win = ((a["t_star"], a["it"][-1]), (b["t_star"], b["it"][-1]))
            kind = "own t*"
        else:
            win = (P3B_FALLBACK, P3B_FALLBACK)
            kind = "fallback 30k-75k"
        sz = slope_per_10k(a["it"], (a["d"] <= 0).sum(axis=1).astype(float), *win[0])
        sd = slope_per_10k(b["it"], (b["d"] <= 0).sum(axis=1).astype(float), *win[1])
        fa, fb = first_crossings(a["it"], a["d"]), first_crossings(b["it"], b["d"])
        ca = int(((fa >= 0) & (a["it"][np.maximum(fa, 0)] > win[0][0])).sum())
        cb = int(((fb >= 0) & (b["it"][np.maximum(fb, 0)] > win[1][0])).sum())
        ratio = sz / sd if sd > 0 else np.nan
        pairs.append(dict(seed=s, window=win, kind=kind, slope_zero=sz, slope_default=sd, ratio=ratio,
                          cross_zero=ca, cross_default=cb,
                          silent_end=(int((a["d"][-1] <= 0).sum()), int((b["d"][-1] <= 0).sum())),
                          active_end=(int(active_bar(a["p"][-1]).sum()), int(active_bar(b["p"][-1]).sum())),
                          t_star=(a["t_star"], b["t_star"])))
    r = np.array([q["ratio"] for q in pairs])
    if len(pairs) == 3 and np.all(r <= P3B_PASS):
        v = "pass"
    elif (r >= P3B_FAIL).sum() >= 2:
        v = "falsified"
    else:
        v = "inconclusive"
    return pairs, (v if pairs else "no data")


def fpt_survival(a, tau, mu, sig):
    """Survival probability of Brownian motion with drift mu toward 0 from distance a > 0 (P4 DD model).

    Args:
        a: (n,) start distances (> 0); tau: (m,) elapsed iterations (> 0); mu: drift toward the
        boundary per iteration; sig: diffusion per sqrt(iteration).
    Returns:
        (m,) mean survival over the n units.
    """
    A = a[None, :]; T = tau[:, None]
    s = sig * np.sqrt(T)
    with np.errstate(over="ignore", invalid="ignore"):
        cdf = ndtr((-A + mu * T) / s) + np.exp(np.clip(2 * mu * A / sig ** 2, -700, 700)) * ndtr((-A - mu * T) / s)
    return np.clip(1 - np.nan_to_num(cdf, nan=1.0), 0, 1).mean(axis=1)


def p4_network(z):
    """P4 for one network: BIC of the DD, EXP and STR models of the survival curve.

    Args:
        z: a load_dist record.
    Returns:
        dict with bic (model -> BIC), winner (model name or 'none'), and the fitted curves.
    """
    it, d = z["it"], z["d"]
    j0 = nearest(it, P4_T0)
    alive0 = d[j0] > 0
    a0 = d[j0, alive0]
    crossed = np.logical_or.accumulate(d[:, alive0] <= 0, axis=0)
    t = it[j0 + 1:].astype(float); F = 1 - crossed[j0 + 1:].mean(axis=1)
    tau = t - it[j0]
    n = len(F)
    scale = float(np.median(a0))

    def bic(res, k):
        """BIC of a least-squares fit with Gaussian residuals."""
        rss = max(float((res ** 2).sum()), 1e-300)
        return n * np.log(rss / n) + k * np.log(n)

    fits = {}
    r_dd = least_squares(lambda q: fpt_survival(a0, tau, q[0] * scale / 1e4, np.exp(q[1]) * scale / 100) - F,
                         x0=[0.0, 0.0], bounds=([-50, -10], [50, 10]))
    fits["DD"] = (bic(r_dd.fun, 2), F + r_dd.fun)
    r_ex = least_squares(lambda q: q[0] + (1 - q[0]) * np.exp(-tau / np.exp(q[1])) - F,
                         x0=[F[-1], np.log(tau[-1] / 3)], bounds=([0, 0], [1, 20]))
    fits["EXP"] = (bic(r_ex.fun, 2), F + r_ex.fun)
    r_st = least_squares(lambda q: q[0] + (1 - q[0]) * np.exp(-(tau / np.exp(q[1])) ** q[2]) - F,
                         x0=[F[-1], np.log(tau[-1] / 3), 1.0], bounds=([0, 0, 0.05], [1, 20, 5]))
    fits["STR"] = (bic(r_st.fun, 3), F + r_st.fun)
    b = {k: v[0] for k, v in fits.items()}
    order = sorted(b, key=b.get)
    winner = order[0] if b[order[1]] - b[order[0]] >= P4_DBIC else "none"
    return dict(bic=b, winner=winner, t=t, F=F, curves={k: v[1] for k, v in fits.items()},
                n_alive0=int(alive0.sum()))


def oracle_o3(z):
    """Oracle o3: d <= 0 against trace participation < 1e-6 at the nearest stored iteration (CDDM).

    Args:
        z: a load_dist record.
    Returns:
        (min agreement over snapshots, median agreement, max |iteration offset|), or None without a trace.
    """
    if not len(z["trace_it"]):
        return None
    ti = np.asarray(z["trace_it"], float)
    agree, offs = [], []
    for k, t in enumerate(z["it"]):
        j = int(np.argmin(np.abs(ti - t)))
        offs.append(abs(ti[j] - t))
        agree.append(((z["d"][k] <= 0) == (z["trace_p"][j] < SILENT_P)).mean())
    return float(np.min(agree)), float(np.median(agree)), float(np.max(offs))


def short(name):
    """A short label for a network: group tag and seed."""
    s = re.search(r"s(\d+)$", name).group(1)
    tag = re.search(r"arm=([^_]+)|noise=([a-z]+)", name)
    t = (tag.group(1) or tag.group(2)) if tag else ""
    task = name.split("_")[0]
    return f"{task} {t} s{s}"


def draw(nets, p1, p2, pairs, lad, out_stem):
    """The report figure: d trajectories, absorption, straightness, silent counts with and without noise, the ladder.

    Args:
        nets: load_dist dict; p1, p2: name -> result dicts; pairs: p3b pairs; lad: p3a result;
        out_stem: str, path without extension.
    Returns:
        None; writes <out_stem>.pdf and .svg through paperstyle.save.
    """
    ps.setup()
    fig = ps.figure(120)
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.98, bottom=0.09, top=0.95, wspace=0.38, hspace=0.45)
    cddm = [n for n in nets if nets[n]["group"] == "cddm_control"]
    # a: d of 40 crossing units of one control network
    ax = fig.add_subplot(gs[0, 0])
    if cddm:
        z = nets[cddm[0]]
        fc = first_crossings(z["it"], z["d"])
        idx = np.flatnonzero(fc >= 0)
        rng = np.random.default_rng(0)
        for i in rng.choice(idx, size=min(40, len(idx)), replace=False):
            ax.plot(z["it"] / 1e3, z["d"][:, i], lw=0.4, color=ps.INK, alpha=0.5)
        ax.axhline(0, color="0.5", lw=0.6, ls="--")
        ax.set_ylim(-0.5, 0.6)
        ax.set_xlabel("iteration (thousands)"); ax.set_ylabel("largest state $d_i$")
        ax.set_title(f"{short(cddm[0])}: 40 units that cross", fontsize=6)
    # b: absorption A(s) per CDDM/DMTS network
    ax = fig.add_subplot(gs[0, 1])
    for n, z in nets.items():
        if z["task"] in ("CDDM", "DMTS") and not z["group"].startswith("cddm_noise"):
            its, A, cnt = absorption(z["it"], z["d"] <= 0, P1_FROM)
            ax.plot(its / 1e3, A, lw=0.6, color="0.2" if z["task"] == "CDDM" else "0.6")
    ax.axhline(P1_PASS_STAR, color="0.5", lw=0.5, ls="--")
    ax.set_xlabel("snapshot iteration (thousands)"); ax.set_ylabel("absorption fraction $A(s)$")
    ax.set_title("silent units silent at every later snapshot", fontsize=6)
    # c: straightness distribution
    ax = fig.add_subplot(gs[0, 2])
    allS = np.concatenate([p2[n]["S_all"] for n in p2 if len(p2[n]["S_all"])]) if p2 else np.zeros(0)
    if allS.size:
        ax.hist(allS, bins=np.linspace(0, 1, 26), color="0.4")
    ax.axvline(1 / np.sqrt(P2_K), color="0.5", lw=0.6, ls="--")
    ax.set_xlabel("straightness $S_i$ before crossing"); ax.set_ylabel("units")
    ax.set_title("random walk 0.22, steady approach 1", fontsize=6)
    # d: silent count over training, noise-free vs default pairs
    ax = fig.add_subplot(gs[1, 0])
    for z in nets.values():
        if z["group"].startswith("cddm_noise"):
            ax.plot(z["it"] / 1e3, (z["d"] <= 0).sum(axis=1), lw=0.8,
                    color="0.1" if z["group"].endswith("zero") else "0.6",
                    ls="-" if z["group"].endswith("zero") else "--")
    ax.set_xlabel("iteration (thousands)"); ax.set_ylabel("silent units (of 1,000)")
    ax.set_title("CDDM: no training noise (solid) vs default (dashed)", fontsize=6)
    # e: ladder slopes
    ax = fig.add_subplot(gs[1, 1])
    for budget, mk in ((200000, "o"), (75000, "s")):
        if budget in lad:
            tab = [t for t in lad[budget]["table"] if np.isfinite(t[4])]
            ax.scatter([t[0] for t in tab], [t[4] for t in tab], s=8, marker=mk, color="0.3",
                       label=f"{budget // 1000}k runs")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlabel("recurrent noise $\\sigma_{rec}$"); ax.set_ylabel("silent units gained per 10k iterations")
    ax.legend(frameon=False, fontsize=5)
    ax.set_title("CDDM ladder, after each run's floor crossing", fontsize=6)
    # f: P4 survival of one network
    ax = fig.add_subplot(gs[1, 2])
    if cddm:
        r = p4_network(nets[cddm[0]])
        ax.plot(r["t"] / 1e3, r["F"], color="k", lw=1.0, label="measured")
        for k, ls in (("DD", "--"), ("EXP", ":"), ("STR", "-.")):
            ax.plot(r["t"] / 1e3, r["curves"][k], lw=0.7, ls=ls, label=k)
        ax.legend(frameon=False, fontsize=5)
        ax.set_xlabel("iteration (thousands)"); ax.set_ylabel("share not yet silent")
        ax.set_title(f"{short(cddm[0])}: survival from 5k", fontsize=6)
    ps.save(fig, os.path.basename(out_stem), out_dir=os.path.dirname(out_stem))


def run_report(dist_dir, ladder_path, out_stem):
    """Stage `report`: every prediction, per network, printed as tables; the figure.

    Args:
        dist_dir: folder of `distances` npz files; ladder_path: the `ladder` pkl; out_stem: figure path stem.
    Returns:
        None.
    """
    nets = load_dist(dist_dir)
    print("== oracle o3 (CDDM: d <= 0 vs trace p < 1e-6)")
    for n, z in nets.items():
        if z["task"] == "CDDM":
            print(f"  {short(n):28s} {oracle_o3(z)}")
    print("== networks")
    for n, z in nets.items():
        print(f"  {short(n):28s} {z['group']:20s} snaps {len(z['it'])} ({z['it'][0]}-{z['it'][-1]}) "
              f"t* {z['t_star']:.0f} ({z['reason']}) sigmas {z['sigmas']} "
              f"silent first/t*/end {(z['d'][0] <= 0).sum()}/"
              f"{(z['d'][nearest(z['it'], z['t_star'])] <= 0).sum() if np.isfinite(z['t_star']) else '-'}/"
              f"{(z['d'][-1] <= 0).sum()} active end {active_bar(z['p'][-1]).sum()} "
              f"ever-silent {(z['d'] <= 0).any(axis=0).sum()}")
    print("== P1 absorption")
    p1 = {n: p1_network(z) for n, z in nets.items()}
    for n, r in p1.items():
        print(f"  {short(n):28s} silent@s* {r['n_silent_star']} A* {r['A_star']:.4f} A_min {r['A_min']:.4f} "
              f"A_med {r['A_med']:.4f} back>0 {r['n_back']} later-active {r['back_active']:.4f} -> {r['verdict']}")
    print("== P1 on 3BFF with the participation bar as the boundary (inactive = below 5% of q95)")
    for n, z in nets.items():
        if z["task"] == "NBitFlipFlop":
            inact = ~np.array([active_bar(pp) for pp in z["p"]])
            its, A, cnt = absorption(z["it"], inact, P1_FROM)
            j = nearest(its, z["t_star"]) if np.isfinite(z["t_star"]) else None
            print(f"  {short(n):28s} min d over all units and snapshots {z['d'].min():.4f}; "
                  f"A_inactive at t* {A[j] if j is not None else np.nan:.4f}, min {np.nanmin(A):.4f}, "
                  f"median {np.nanmedian(A):.4f}")
    print("== P2 crossing statistics")
    p2 = {n: p2_network(z) for n, z in nets.items() if z["task"] in ("CDDM", "DMTS")}
    for n, r in p2.items():
        print(f"  {short(n):28s} n_cross {r['n_cross']} S_med {r['S_med']:.3f} IQR {np.round(r['S_q'], 3)} "
              f"rho1 {r['rho1']:.3f} late {r['late_frac']:.3f} of {r['n_end_silent']} VMR {r['vmr']:.2f} "
              f"-> a {r['va']}, b {r['vb']}, c {r['vc']}, d {r['vd']}")
    print("== P3a recurrent-noise ladder")
    with open(ladder_path, "rb") as fh:
        rows = pickle.load(fh)
    lad = p3a(rows)
    for budget, r in lad.items():
        print(f"  budget {budget}: rho {r['rho']:.3f} p(one-sided) {r['p_one_sided']:.4f} "
              f"means {r['means']} used {r['n_used']} of {r['n_all']} -> {r['verdict']}")
        for t in r["table"]:
            print(f"    sigma {t[0]:<5} s{t[1]} t* {t[2]:.0f} window to {t[3]:.0f} silent/10k {t[4]:.2f} "
                  f"active/10k {t[5]:.2f} loss change {t[6]:+.3f} silent@t* {t[7]} end {t[8]} ({t[9]})")
    print("== P3b noise-free pairs")
    pairs, v3b = p3b(nets)
    for q in pairs:
        print(f"  s{q['seed']} {q['kind']} t* {q['t_star']}: slope zero {q['slope_zero']:.2f} default "
              f"{q['slope_default']:.2f} ratio {q['ratio']:.2f}; crossings after start {q['cross_zero']} vs "
              f"{q['cross_default']}; silent end {q['silent_end']} active end {q['active_end']}")
    print(f"  -> {v3b}")
    print("== P2 straightness, noise-free vs default (the noise-free check)")
    for n, r in p2.items():
        if nets[n]["group"].startswith("cddm_noise"):
            print(f"  {short(n):28s} S_med {r['S_med']:.3f} rho1 {r['rho1']:.3f}")
    print("== P4 survival-curve models (CDDM)")
    for n, z in nets.items():
        if z["task"] == "CDDM":
            r = p4_network(z)
            print(f"  {short(n):28s} alive@5k {r['n_alive0']} BIC " +
                  " ".join(f"{k} {v:.1f}" for k, v in r["bic"].items()) + f" -> {r['winner']}")
    draw(nets, p1, p2, pairs, lad, out_stem)
    print(f"figure: {out_stem}.pdf")


def main():
    """Command-line entry point; see the module docstring for the stages."""
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("list"); a.add_argument("--groups", default="all")
    a.add_argument("--data-root", default=DELLA_DATA)
    a = sub.add_parser("distances"); a.add_argument("run_dir"); a.add_argument("--out", required=True)
    a.add_argument("--snap-root", default=os.path.join(DELLA_DATA, "c1_snapshots"))
    a.add_argument("--every", type=int, default=1)
    a = sub.add_parser("ladder"); a.add_argument("--data-root", default=DELLA_DATA)
    a.add_argument("--out", required=True)
    sub.add_parser("oracle")
    a = sub.add_parser("report"); a.add_argument("--dist", required=True); a.add_argument("--ladder", required=True)
    a.add_argument("--fig", default=os.path.join(common.IMG_DIR, "absorb_boundary"))
    args = ap.parse_args()
    if args.cmd == "list":
        groups = list(GROUPS) if args.groups == "all" else args.groups.split(",")
        for g in groups:
            for d in list_networks(args.data_root, g):
                print(d)
    elif args.cmd == "distances":
        run_distances(args.run_dir, args.snap_root, args.out, args.every)
    elif args.cmd == "ladder":
        run_ladder(args.data_root, args.out)
    elif args.cmd == "oracle":
        sys.exit(0 if run_oracle() else 1)
    elif args.cmd == "report":
        run_report(args.dist, args.ladder, args.fig)


if __name__ == "__main__":
    main()
