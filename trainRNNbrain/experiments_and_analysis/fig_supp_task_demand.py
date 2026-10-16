#!/usr/bin/env python3
"""Supplementary panel A5: does a harder task fill the network?

THE OBJECTION. "The task just doesn't need more units." If that were the whole story, a task that
demands more should recruit more units, and a demanding enough task should leave few silent.

THE DATA (all on disk, all unpenalised; the penalised `both` arms of the same grids are left out
because the question is about plain task training):
  - plain k-bit flip-flop, k = 1..8 (`NBitFlipFlop_std_ksweep`): k target functions, k outputs;
  - "parity" (hyper) flip-flop, k = 2, 4, 6, 8 (`NBitFlipFlopHyper_std_hyper`, 150k, and a 500k
    replicate of five cells in `_hyper_long`): the same k bits, but the read-out is the product of
    the bit states over every non-empty subset, 2^k - 1 target functions and outputs;
  - Walsh flip-flop (`NBitFlipFlopWalsh_std_walsh`): one output that is a random combination of all
    products up to `degree`, so 63 (k=6) or 255 (k=8) target functions behind a ONE-channel read-out;
  - Yang et al. (2019) task family (`Yang_std_30k`): the CDDM pair alone (2 rules) against all 20
    rules, same 85-input / 33-output layout, N = 1000.
Demand is put on one axis as the number of target functions the read-out must produce (k, 2^k - 1,
the Walsh basis size) for the flip-flop family, and as the number of rules for the Yang family.

READ-OUT. Each run where its NOISY training loss first reaches 1.05x its own fitted floor
(`common.noisy_floor`), warm-start chains joined; the end of training is a labelled second
read-out. Active units = p_i >= 0.05 q95(p), p = std(r) + q0.9(|r|), from the noise-free
participation trace the trainer stored every 100 iterations, at the probe nearest the crossing.
The q50/q95 health ratio is recorded at every read-out (above 0.05 the relative rule is broken).

R^2. The end value is the run's folder score (final weights, one held-out batch, training noise
on, pooled over output channels). At the floor no weights were saved, so R^2 there is ESTIMATED
from the loss ratio, R2_floor = 1 - (L(t*) / L_end) (1 - R2_end) (common.r2_at_floor_estimate).
⚠️ R2_end here is still the folder score, where heldout_r2.floor_r2 uses held-out R^2 of the final
weights. Moving this panel to held-out R^2 changes its numbers, so it is left for a change of its own.

ARCHITECTURE. Every network here is PRE-STANDARD: ReLU, h equation, self-connections on, no Dale,
gamma 0, but a TRAINED bias in [-1, 1] (the standard network fixes it at 0). Read from each
saved config and printed per sweep.

WHY THE CHAIN IS JOINED HERE. `common.noisy_loss_series` follows `paths.init_from`, which records
the cluster path of the parent (/usr/people/..., /home/...). On this machine those paths do not
exist and the parent would be silently dropped, so the warm-started runs' loss would start at
iteration 1 instead of 10k / 30k. `chain()` maps the parent to the local mirror by the part of the
path after `trained_RNNs/`. For runs without a parent the joined series is checked equal to
`common.noisy_loss_series`.

Usage (from the repository root):
    python trainRNNbrain/experiments_and_analysis/fig_supp_task_demand.py [--recompute] [--out DIR]
Writes data/task_demand_cache.json and <out>/fig_supp_task_demand.{pdf,svg}.
"""

import argparse
import glob
import json
import os
import pickle
import re
import shutil
import sys
from math import comb

import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paperstyle as ps
from common import (DATA_DIR, END_FRAC, SILENT_FLIPFLOP, active_mask, noisy_floor, noisy_loss_series,
                    r2_at_floor_estimate)

CACHE = os.path.join(DATA_DIR, "..", "task_demand_cache.json")

# (family, sweep folder, cell-name regex). The regex names every field the cell folder encodes.
SWEEPS = [
    ("plain", "NBitFlipFlop_std_ksweep", r"k=(?P<k>\d+)_N=(?P<N>\d+)_iters=(?P<iters>\d+)$"),
    ("hyper", "NBitFlipFlopHyper_std_hyper", r"k=(?P<k>\d+)_N=(?P<N>\d+)_pen=(?P<pen>none)$"),
    ("hyper500k", "NBitFlipFlopHyper_std_hyper_long",
     r"k=(?P<k>\d+)_N=(?P<N>\d+)_pen=(?P<pen>none)_iters=500000$"),
    ("walsh", "NBitFlipFlopWalsh_std_walsh",
     r"k=(?P<k>\d+)_deg=(?P<deg>\d+)_N=(?P<N>\d+)_pen=(?P<pen>none)(_iters=\d+(_warm)?)?$"),
    ("yang", "Yang_std_30k", r"set=(?P<set>\w+?)_N=(?P<N>\d+)_pen=(?P<pen>none)$"),
]
# the Walsh k=6 30k run is the PARENT of the 70k warm-started one; read the chain once, at its end
WALSH_PARENT_ONLY = "k=6_deg=6_N=1000_pen=none_iters=30000"


def local_path(p):
    """Map a cluster path under .../trained_RNNs/ to the local mirror.

    Args:
        p: a path string as recorded in a saved config.
    Returns:
        the local folder if it exists, else None.
    """
    if not p:
        return None
    p = p.rstrip("/")
    if os.path.isdir(p):
        return p
    tail = p.split("trained_RNNs/", 1)[-1]
    q = os.path.join(DATA_DIR, tail)
    return q if os.path.isdir(q) else None


def config_of(run_dir):
    """The saved Hydra config of one run folder (OmegaConf DictConfig)."""
    return OmegaConf.load(sorted(glob.glob(os.path.join(run_dir, "*_config.yaml")))[0])


def chain(run_dir):
    """A run and its warm-start ancestors, oldest first, with cluster paths mapped to local ones.

    Args:
        run_dir: a trained-network folder.
    Returns:
        list of folders; raises if a recorded parent is missing locally (never silently dropped).
    """
    out, d = [], run_dir.rstrip("/")
    while d:
        out.append(d)
        parent = (config_of(d).get("paths", {}) or {}).get("init_from", None)
        if parent:
            q = local_path(parent)
            if q is None:
                raise FileNotFoundError(f"warm-start parent of {d} not on disk: {parent}")
            d = q
        else:
            d = None
    return out[::-1]


def joined(run_dir):
    """Noisy loss and participation trace of a run, joined along its warm-start chain.

    Args:
        run_dir: a trained-network folder.
    Returns:
        dict with `it_loss`, `loss` (per-iteration noisy training loss, iterations 1-based and
        absolute), `it_p` (absolute iteration of each participation probe), `P` (probes x N
        participation, noise-free), `n_chain` (number of runs joined).
    """
    its, Ls, itp, Ps, done = [], [], [], [], 0
    dirs = chain(run_dir)
    for d in dirs:
        with open(sorted(glob.glob(os.path.join(d, "*TrainLosses.json")))[0]) as fh:
            L = np.asarray(json.load(fh)["train_losses"], float)
        with open(sorted(glob.glob(os.path.join(d, "*ParticipationTrace.pkl")))[0], "rb") as fh:
            tr = pickle.load(fh)
        its.append(done + np.arange(1, len(L) + 1, dtype=float))
        Ls.append(L)
        itp.append(done + np.asarray(tr["participation_iters"], float))
        Ps.append(np.asarray(tr["participation"], float))
        done += len(L)
    out = dict(it_loss=np.concatenate(its), loss=np.concatenate(Ls), it_p=np.concatenate(itp),
               P=np.concatenate(Ps, axis=0), n_chain=len(dirs))
    if len(dirs) == 1:   # the joined series must be the project's own for an unchained run
        it_c, L_c = noisy_loss_series(run_dir)
        assert np.array_equal(it_c, out["it_loss"]) and np.array_equal(L_c, out["loss"], equal_nan=True)
    return out


def count(p):
    """Active units, the q50/q95 health ratio, and the absolute-threshold cross-check of one probe.

    Args:
        p: (N,) noise-free participation.
    Returns:
        (active count under p >= 0.05 q95(p), q50/q95, active count under the flip-flop's absolute
        threshold p >= 4e-2). The absolute count is a cross-check only: where q50/q95 > 0.05 more
        than half the units pass the relative rule, and the absolute rule shows whether that is real.
    """
    q95 = np.quantile(p, 0.95)
    return (int(active_mask(p).sum()), float(np.quantile(p, 0.5) / q95),
            int((p >= SILENT_FLIPFLOP).sum()))


def architecture(cfg):
    """The architecture fields that define the standard network, read from a saved config.

    Args:
        cfg: a run's config.
    Returns:
        dict of activation, equation, bias_range, self_connections, dale, gamma, and `label`:
        "standard" when every field matches, else "pre-standard (<deviations>)".
    """
    m = cfg.model
    a = dict(activation=str(m.activation_args.name), equation=str(m.equation_type),
             bias_range=[float(v) for v in m.get("bias_range", [0.0, 0.0])],
             self_connections=bool(m.get("self_connections", False)), dale=bool(m.get("dale", False)),
             gamma=float(m.get("gamma", 0)))
    dev = []
    if a["activation"] != "relu":
        dev.append(a["activation"])
    if a["equation"] != "h":
        dev.append(f"eq {a['equation']}")
    if a["bias_range"] != [0.0, 0.0]:
        dev.append("trained bias")
    if not a["self_connections"]:
        dev.append("no self-connections")
    if a["dale"]:
        dev.append("Dale")
    if a["gamma"] != 0:
        dev.append(f"gamma {a['gamma']}")
    a["label"] = "standard" if not dev else "pre-standard (" + ", ".join(dev) + ")"
    return a


def demand(family, f):
    """Number of target functions (flip-flop family) or rules (Yang) of one cell.

    Args:
        family: one of the SWEEPS families; f: the cell's parsed regex fields.
    Returns:
        int.
    """
    k = int(f.get("k") or 0)
    if family == "plain":
        return k
    if family in ("hyper", "hyper500k"):
        return 2 ** k - 1
    if family == "walsh":
        return sum(comb(k, m) for m in range(1, int(f["deg"]) + 1))
    return {"ctxdm": 2, "multi20": 20}[f["set"]]


def read_run(family, sweep, cell, f, run_dir):
    """Floor and end read-out of one run.

    Args:
        family, sweep, cell: where the run sits; f: the cell's regex fields; run_dir: the folder.
    Returns:
        dict of identifiers, architecture, budget, floor fit, active counts and R^2 at both
        read-outs (R^2 at the floor estimated, see module docstring).
    """
    cfg = config_of(run_dir)
    J = joined(run_dir)
    it_l, L, it_p, P = J["it_loss"], J["loss"], J["it_p"], J["P"]
    fl = noisy_floor(it_l, L)
    head = os.path.basename(run_dir).split("_")[0]
    r2_end = float(head) if head != "nan" else float("nan")
    L_end = float(np.median(L[it_l >= it_l[-1] - END_FRAC * (it_l[-1] - it_l[0])]))
    a_end, h_end, abs_end = count(P[-1])
    rec = dict(family=family, sweep=sweep, cell=cell, run=os.path.basename(run_dir),
               N=int(f["N"]), k=int(f.get("k") or 0), deg=int(f["deg"]) if f.get("deg") else None,
               set=f.get("set"), demand=demand(family, f), n_outputs=int(cfg.task.n_outputs),
               seed=int(cfg.seed), budget=int(it_l[-1]), n_chain=J["n_chain"],
               arch=architecture(cfg), floor=fl["floor"], t_star=fl["t_star"],
               late_rise=fl["late_rise"], reason=fl["reason"], active_end=a_end, health_end=h_end,
               abs_end=abs_end, abs_floor=None,
               it_end=float(it_p[-1]), r2_end=r2_end, L_end=L_end,
               active_floor=None, health_floor=None, it_probe=None, r2_floor=None, L_star=None)
    if fl["reason"] is None:
        T = fl["t_star"]
        if not it_p[0] <= T <= it_p[-1]:
            rec["reason"] = "participation trace does not cover the crossing"
            return rec
        j = int(np.argmin(np.abs(it_p - T)))
        rec["active_floor"], rec["health_floor"], rec["abs_floor"] = count(P[j])
        rec["it_probe"] = float(it_p[j])
        L_star = float(fl["smoothed"][np.flatnonzero(it_l >= T)[0]])
        rec["L_star"] = L_star
        rec["r2_floor"] = r2_at_floor_estimate(L_star, L_end, r2_end)
    return rec


def build():
    """Read every run of every sweep in SWEEPS. Returns the list of per-run dicts."""
    rows = []
    for family, sweep, rx in SWEEPS:
        for cell_dir in sorted(glob.glob(os.path.join(DATA_DIR, sweep, "EqType=*"))):
            cell = os.path.basename(cell_dir)
            m = re.search(rx, cell)
            if not m:
                continue
            if family == "walsh" and cell.endswith(WALSH_PARENT_ONLY):
                continue
            for run_dir in sorted(glob.glob(os.path.join(cell_dir, "*", ""))):
                run_dir = run_dir.rstrip("/")
                if not glob.glob(os.path.join(run_dir, "*ParticipationTrace.pkl")):
                    continue
                rows.append(read_run(family, sweep, cell, m.groupdict(), run_dir))
                r = rows[-1]
                print(f"{family:9s} {cell:45s} s={r['seed']:<11d} T*={r['t_star']:>8.0f} "
                      f"act floor/end={r['active_floor']}/{r['active_end']} "
                      f"r2 end={r['r2_end']:.3f} {r['reason'] or ''}", flush=True)
    return rows


def load(recompute=False):
    """The per-run table, from the cache unless asked to recompute."""
    if not recompute and os.path.exists(CACHE):
        with open(CACHE) as fh:
            return json.load(fh)
    rows = build()
    with open(CACHE, "w") as fh:
        json.dump(rows, fh, indent=1)
    return rows


def diverged(r):
    """A run whose final weights score r^2 below 0 (or nan) blew up; it is read at neither point."""
    return not (np.isfinite(r["r2_end"]) and r["r2_end"] >= 0)


def cells(rows, family, N=None):
    """Group the non-diverged runs of one family (and size) by demand.

    Args:
        rows: per-run dicts; family: family name; N: network size, or None for every size.
    Returns:
        dict demand -> list of runs, sorted by demand.
    """
    g = {}
    for r in rows:
        if r["family"] == family and (N is None or r["N"] == N) and not diverged(r):
            g.setdefault(r["demand"], []).append(r)
    return dict(sorted(g.items()))


def vals(runs, key):
    """Finite values of `key` across runs, as an array."""
    return np.array([r[key] for r in runs if r[key] is not None and np.isfinite(r[key])], float)


def table(rows):
    """Print the per-cell summary (mean over seeds) for both read-outs, with drops and health."""
    print(f"\n{'family':9s} {'N':>5s} {'demand':>6s} {'n':>2s} {'drop':>4s} {'T* (k)':>13s} "
          f"{'active floor':>14s} {'active end':>14s} {'r2 floor~':>9s} {'r2 end':>7s} "
          f"{'q50/q95 max':>11s} {'abs4e-2 fl/end':>14s}  arch")
    for fam in ("plain", "hyper", "hyper500k", "walsh", "yang"):
        for N in sorted({r["N"] for r in rows if r["family"] == fam}):
            for d, runs in cells(rows, fam, N).items():
                af, ae = vals(runs, "active_floor"), vals(runs, "active_end")
                ts = vals(runs, "t_star") / 1e3
                hl = np.concatenate([vals(runs, "health_floor"), vals(runs, "health_end")])
                drop = sum(r["reason"] is not None for r in runs)
                ff = f"{af.mean():6.0f} ± {af.std():4.0f}" if len(af) else "—"
                ts_s = f"{ts.min():.0f}–{ts.max():.0f}" if len(ts) else "—"
                print(f"{fam:9s} {N:5d} {d:6d} {len(runs):2d} {drop:4d} {ts_s:>13s} {ff:>14s} "
                      f"{ae.mean():6.0f} ± {ae.std():4.0f} "
                      f"{(vals(runs, 'r2_floor').mean() if len(af) else np.nan):9.3f} "
                      f"{vals(runs, 'r2_end').mean():7.3f} {hl.max():11.4f} "
                      f"{(vals(runs, 'abs_floor').mean() if len(af) else np.nan):6.0f} /"
                      f"{vals(runs, 'abs_end').mean():6.0f}  "
                      f"{runs[0]['arch']['label']}")
    for r in rows:
        if diverged(r):
            print(f"  diverged, read at neither point: {r['family']} {r['cell']} seed {r['seed']} "
                  f"(end r2 {r['r2_end']:.2f})")
        elif r["reason"]:
            print(f"  dropped at floor: {r['family']} {r['cell']} seed {r['seed']}: {r['reason']}")


def size_exponents(rows, key):
    """Fit active = A N^b per demand level (all seeds pooled) for the plain and hyper families.

    Args:
        rows: per-run dicts; key: "active_floor" or "active_end".
    Returns:
        dict (family, demand) -> b, for demand levels read at all three sizes.
    """
    out = {}
    for fam in ("plain", "hyper"):
        for d, runs in cells(rows, fam).items():
            pts = [(r["N"], r[key]) for r in runs if r[key] is not None]
            if len({n for n, _ in pts}) < 3:
                continue
            x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
            out[(fam, d)] = float(np.polyfit(x, y, 1)[0])
    return out


COL = {"plain": ps.BASE, "hyper": ps.SLOTS[1], "hyper500k": ps.SLOTS[1], "walsh": ps.SLOTS[3]}
LAB = {"plain": "plain flip-flop ($k$ targets)", "hyper": "parity flip-flop ($2^k-1$ targets)",
       "walsh": "Walsh flip-flop (1 output)"}


def draw_family(ax, rows, fam, N, key, mk, fill):
    """One flip-flop series at one size: every seed as a dot, the mean as a line.

    Args:
        ax: axes; rows: per-run dicts; fam: family; N: size; key: "active_floor"/"active_end"
        (or an r2 key); mk: marker; fill: True for filled markers (floor), False for open (end).
    Returns:
        None.
    """
    g = cells(rows, fam, N)
    xs, ms = [], []
    for d, runs in g.items():
        v = vals(runs, key)
        if not len(v):
            continue
        jit = np.exp(np.linspace(-0.05, 0.05, len(v))) if len(v) > 1 else np.ones(1)
        ax.scatter(d * jit, v, s=7, marker=mk, lw=0.6, zorder=3,
                   facecolor=COL[fam] if fill else "none", edgecolor=COL[fam], alpha=0.8)
        xs.append(d)
        ms.append(v.mean())
    if len(xs) > 1:
        ax.plot(xs, ms, color=COL[fam], lw=1.0 if fill else 0.7, ls="-" if fill else ":", zorder=2)


def figure(rows, out_dir):
    """Draw the panel: active units and R^2 against demand at N = 500, 1000, 2000, plus Yang."""
    ps.setup()
    fig = plt.figure(figsize=(180 * ps.MM, 92 * ps.MM))
    gs = fig.add_gridspec(2, 4, left=0.065, right=0.995, bottom=0.15, top=0.885, wspace=0.42,
                          hspace=0.42, width_ratios=[1, 1, 1, 0.62])
    Ns = (500, 1000, 2000)
    for j, N in enumerate(Ns):
        ax = fig.add_subplot(gs[0, j])
        ax2 = fig.add_subplot(gs[1, j])
        for fam in ("plain", "hyper", "walsh"):
            for key, fill in (("active_floor", True), ("active_end", False)):
                draw_family(ax, rows, fam, N, key, "o", fill)
            for key, fill in (("r2_floor", True), ("r2_end", False)):
                draw_family(ax2, rows, fam, N, key, "o", fill)
        ax.axhline(N, color="#3a3a36", lw=0.6, ls="--", zorder=1)
        ax.text(1.05, N * 0.97, f"N = {N}", va="top", ha="left", fontsize=6, color="#3a3a36")
        ax.set_ylim(0, N * 1.06)
        for a in (ax, ax2):
            a.set_xscale("log")
            a.set_xlim(0.8, 340)
            a.set_xticks([1, 2, 4, 8, 15, 63, 255])
            a.set_xticklabels(["1", "2", "4", "8", "15", "63", "255"])
            a.minorticks_off()
        ax2.set_ylim(0.5, 1.0)
        ax2.axhline(0.9, color="#3a3a36", lw=0.5, ls=":", zorder=1)
        ax.set_title(f"flip-flop family, N = {N}")
        if j == 0:
            ax.set_ylabel("active units")
            ax2.set_ylabel("$r^2$ (floor: estimated)")
        ps.panel_letter(ax, "abc"[j])
        ps.panel_letter(ax2, "efg"[j])
    # Yang: number of rules, whole training batch
    ax = fig.add_subplot(gs[0, 3])
    ax2 = fig.add_subplot(gs[1, 3])
    for key, fill, a in (("active_floor", True, ax), ("active_end", False, ax),
                         ("r2_floor", True, ax2), ("r2_end", False, ax2)):
        xs, ms = [], []
        for x, (d, runs) in enumerate(cells(rows, "yang", 1000).items()):
            v = vals(runs, key)
            dx = -0.12 if fill else 0.12
            a.scatter(np.full(len(v), x + dx), v, s=7, lw=0.6, zorder=3,
                      facecolor=ps.SLOTS[2] if fill else "none", edgecolor=ps.SLOTS[2])
            xs.append(x + dx)
            ms.append(v.mean())
        a.plot(xs, ms, color=ps.SLOTS[2], lw=1.0 if fill else 0.7, ls="-" if fill else ":")
    ax.axhline(1000, color="#3a3a36", lw=0.6, ls="--", zorder=1)
    ax.set_ylim(0, 1060)
    ax2.set_ylim(0.9, 1.0)
    for a in (ax, ax2):
        a.set_xlim(-0.5, 1.5)
        a.set_xticks([0, 1])
        a.set_xticklabels(["CDDM\npair", "20\ntasks"])
    ax.set_title("Yang tasks, N = 1000", loc="right")
    ax2.set_xlabel("rules trained")
    fig.text(0.37, 0.065, "target functions the read-out must produce (log scale)", ha="center",
             va="center", fontsize=7)
    ps.panel_letter(ax, "d", dx=-0.32)
    ps.panel_letter(ax2, "h", dx=-0.32)
    # legend: series and read-outs
    h = [plt.Line2D([], [], color=COL[f], marker="o", ms=3, lw=1, label=LAB[f])
         for f in ("plain", "hyper", "walsh")]
    h.append(plt.Line2D([], [], color=ps.SLOTS[2], marker="o", ms=3, lw=1, label="Yang family"))
    h.append(plt.Line2D([], [], color="#3a3a36", marker="o", ms=3, lw=1,
                        label="filled, solid: at 1.05x own noisy-loss floor"))
    h.append(plt.Line2D([], [], color="#3a3a36", marker="o", ms=3, lw=0.7, ls=":", mfc="none",
                        label="open, dotted: end of training"))
    fig.legend(handles=h, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), fontsize=5.6,
               columnspacing=1.0, handlelength=1.6)
    fig.text(0.5, 0.005, "All networks pre-standard: ReLU, h equation, self-connections on, no Dale, "
             "$\\gamma$ = 0, bias TRAINED in [-1, 1]; unpenalised. Active: $p_i \\geq 0.05\\,q_{95}(p)$, "
             "noise-free participation.", ha="center", va="bottom", fontsize=5.6, color="#3a3a36")
    ps.save(fig, "fig_supp_task_demand")
    if out_dir:
        for ext in ("pdf", "svg"):
            src = os.path.join(ps.IMG_DIR, f"fig_supp_task_demand.{ext}")
            if os.path.abspath(os.path.dirname(src)) != os.path.abspath(out_dir):
                shutil.copy(src, out_dir)


def main():
    """Command-line entry: build or load the table, print it, print size exponents, draw."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--out", default=None, help="extra directory to copy the PDF/SVG into")
    a = ap.parse_args()
    rows = load(a.recompute)
    table(rows)
    for key in ("active_floor", "active_end"):
        b = size_exponents(rows, key)
        print(f"\nsize exponent b (active = A N^b), {key}: " +
              ", ".join(f"{f} {d}: {v:.2f}" for (f, d), v in b.items()))
    figure(rows, a.out)


if __name__ == "__main__":
    main()
