#!/usr/bin/env python3
"""Extended Data Fig. 3: floor crossings against an independent snapshot estimate.

  (a) the noisy training loss of one N = 1,000 default network per task, with the smoothed loss, the
      fitted decay, the loss floor, the 1.05x, 1.10x and 1.20x levels and their crossings, and the
      loss bursts masked before smoothing;
  (b) the floor crossing t* (1.05x) against N, every network of Fig. 2a;
  (c) t* against the snapshot estimate of the same crossing, for every N = 1,000 network whose weights
      were kept every 1,000 iterations;
  (d) the difference between t* and the snapshot estimate, in percent of the snapshot estimate, per
      task, with the CDDM extremes printed on the panel (the Discussion quotes them); vertical bars
      on the three CDDM default networks show that difference over the snapshot estimate's 90%
      range from batch noise alone (stage --batch-noise), a description of n = 3 networks, not a
      test (review of dead-relu-paper PR #67).

Inputs: the collect record of fig_paper_F2_size.py (data/fig_paper_F2_size_collect.pkl, Della job
16558986) and, for the bars of panel (d), data/fig_ed_readout_batch_noise.json written by the
--batch-noise stage.

TWO STAGES:
  --batch-noise [--oracle-dir DIR]   recompute the snapshot estimate of the three CDDM default networks
              from their per-batch snapshot losses (Della: /scratch/gpfs/TENGEL/pt1290/oracle_readout,
              job 15635966; or a local copy), and its bootstrap over batches; rule and method in
              notebook entry 2026-11-01_0118. Writes data/fig_ed_readout_batch_noise.json.
  (no flag)   draw the figure into img/internal_figures/fig_ed_readout.{pdf,svg} and print its numbers.

Usage:  python fig_ed_readout.py --batch-noise --oracle-dir <copy of oracle_readout>
        python fig_ed_readout.py
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator

import paperstyle as ps
from common import FLOOR_DELTAS, HOLD_FRAC
from fig_paper_F2_size import (COLLECT, ORACLE_DIR, REPO, SIZES, TASK_COL, TASKS, apply_decisions,
                               check_overlaps, is_read, network_offsets, seed_of)

BATCH_NOISE = os.path.join(REPO, "data", "fig_ed_readout_batch_noise.json")
CDDM_DEFAULT = "CDDM_c1_snapshots__EqType=h_N=1000_arm=control__s"   # oracle folder prefix
N_BOOT_BATCH = 1000          # bootstrap resamples over batches (entry 2026-11-01_0118)
BATCH_SEED = 20261101
# panels c-d: every value is read at the floor crossing, so every marker is filled (dead-relu-paper
# #83); default networks and networks with scaled input weights differ by shape
DEFAULT_MARK = "o"
SCALED_MARK = "^"
ED3_H_MM = 102.0             # printed height: at most 0.45 \textheight (102.9 mm), the float cap; width 165.1 mm


# ---------------------------------------------------------------------------------------------
# stage --batch-noise
# ---------------------------------------------------------------------------------------------

def trimmed_mean(x):
    """Mean of per-batch losses after dropping points more than 3 scaled MADs from their median.

    Args:
        x: 1-D array of per-batch losses of one snapshot (or an array of shape (B, n_batch)).
    Returns:
        float (or array of shape (B,)) as fig_paper_F2_size.oracle_read computes it.
    """
    x = np.atleast_2d(x)
    med = np.median(x, axis=1, keepdims=True)
    dev = np.abs(x - med)
    mad = np.median(dev, axis=1, keepdims=True)
    keep = np.where(mad > 0, dev <= 3 * 1.4826 * mad, True)
    out = (x * keep).sum(axis=1) / keep.sum(axis=1)
    return out if out.size > 1 else float(out[0])


def snapshot_crossing(t, v, floor):
    """The snapshot estimate of the 1.05x floor crossing, from per-snapshot values (no burst mask).

    Args:
        t: snapshot iterations, sorted, shape (S,); v: per-snapshot loss values, shape (S,);
        floor: the network's loss floor.
    Returns:
        float iteration of the first snapshot t after which the window mean over [t/1.1, 1.1t] stays
        at or under 1.05 * floor through (1 + HOLD_FRAC) t; nan if none.
    """
    curve = np.array([v[(t >= s / 1.1) & (t <= s * 1.1)].mean() for s in t])
    bar = 1.05 * floor
    for i, s in enumerate(t):
        if (1 + HOLD_FRAC) * s > t[-1]:
            break
        hold = (t >= s) & (t <= (1 + HOLD_FRAC) * s)
        if np.all(curve[hold] <= bar):
            return float(s)
    return float("nan")


def batch_noise(oracle_dir):
    """Snapshot estimate and its 90% batch-noise range for the three CDDM default networks.

    Args:
        oracle_dir: directory holding the oracle folders (`<sweep>__<cell>__s<seed>/it*.npy`).
    Returns:
        dict seed -> {t_oracle, t_oracle_record, lo, hi, n_snap, n_batch}; also written to BATCH_NOISE.
    """
    with open(COLLECT, "rb") as fh:
        rec = pickle.load(fh)
    floors = {r["seed"]: r for r in rec["ablation"] if r["task"] == "CDDM"}
    record = {o["name"].split("__s")[-1]: o["t_oracle"] for o in rec["oracle"]
              if o["name"].startswith(CDDM_DEFAULT)}
    rng = np.random.default_rng(BATCH_SEED)
    out = {}
    for seed in sorted(record):
        r = floors[seed]
        assert r["n_bursts"] == 0, f"s{seed} has loss bursts; the snapshot mask would be needed"
        files = sorted(glob.glob(os.path.join(oracle_dir, CDDM_DEFAULT + seed, "it*.npy")))
        t = np.array([int(os.path.basename(f)[2:-4]) for f in files], float)
        X = np.stack([np.load(f) for f in files])                   # (snapshots, batches)
        t_hat = snapshot_crossing(t, np.array([trimmed_mean(x) for x in X]), r["floor"])
        boots = []
        for _ in range(N_BOOT_BATCH):
            idx = rng.integers(0, X.shape[1], size=X.shape)
            boots.append(snapshot_crossing(t, trimmed_mean(np.take_along_axis(X, idx, 1)), r["floor"]))
        boots = np.array(boots)
        lo, hi = np.nanpercentile(boots, [5, 95])
        out[seed] = dict(t_oracle=t_hat, t_oracle_record=float(record[seed]), lo=float(lo),
                         hi=float(hi), n_nan=int(np.isnan(boots).sum()), n_snap=len(t),
                         n_batch=int(X.shape[1]), t_rule=float(r["t_star"]))
        ok = "MATCHES" if t_hat == record[seed] else "DIFFERS FROM"
        print(f"  CDDM s{seed}: estimate {t_hat:,.0f} ({ok} record {record[seed]:,.0f}); 90% range "
              f"{lo:,.0f}-{hi:,.0f}; t* {r['t_star']:,.0f}; {len(t)} snapshots x {X.shape[1]} batches")
    assert all(o["t_oracle"] == o["t_oracle_record"] for o in out.values()), \
        "rule 1 of entry 2026-11-01_0118 fails: re-implementation does not reproduce the record"
    with open(BATCH_NOISE, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {BATCH_NOISE}")
    return out


# ---------------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------------

def panel_examples(axes, rec):
    """Panel (a): one N = 1,000 default network per task, its loss and how t* is found.

    Args:
        axes: three axes, one per task; rec: the collect record.
    Returns:
        dict task -> seed drawn.
    """
    seeds = {}
    for k, task in enumerate(TASKS):
        ax, ex = axes[k], rec["examples"][task][0]
        col = TASK_COL[task]
        seeds[task] = seed_of(ex["name"])
        ax.fill_between(ex["tq"], ex["lq10"], ex["lq90"], color=ps.GRID, lw=0, zorder=1)
        for b0, b1, _ in ex["bursts"]:
            ax.axvspan(b0, b1, ymin=0.94, ymax=1.0, color=ps.MUTED, lw=0, zorder=5)
        ax.plot(ex["ts"], ex["ys"], "-", color=ps.INK, lw=0.9, zorder=3)
        if ex["tf"] is not None:
            ax.plot(ex["tf"], ex["yf"], ls=(0, (1, 1.2)), color=col, lw=ps.DATA_LW, zorder=4)
        for d, a in zip(FLOOR_DELTAS, (1.0, 0.6, 0.35)):
            ax.axhline((1 + d) * ex["floor"], color=ps.MUTED, lw=ps.THIN_LW, ls=(0, (4, 2)), alpha=a)
            t = ex["t_stars"][d]
            if np.isfinite(t):
                ax.axvline(t, color=col, lw=0.8, alpha=a, zorder=2)
        ax.axhline(ex["floor"], color=ps.INK, lw=ps.THIN_LW)
        lo = ex["floor"] * 0.8
        hi = max(np.nanmax(ex["ys"][ex["ts"] > 200]) * 1.3, ex["floor"] * 3)
        ax.set(xscale="log", yscale="log", xlim=(100, ex["tq"][-1] * 1.1), ylim=(lo, hi),
               xlabel="training iteration")
        ax.set_xticks([1e2, 1e3, 1e4, 1e5])
        ps.count_ticks(ax.xaxis)
        ax.xaxis.set_major_formatter(FuncFormatter(     # every other decade, as Fig. 2d (#86)
            lambda v, _pos: ps.fmt_count(v) if round(np.log10(v)) % 2 == 0 else ""))
        # loss ticks as plain decimals, at 1-2-5 steps inside the range
        cand = np.array([m * 10.0 ** e for e in range(-6, 1) for m in (1, 2, 5)])
        ticks = cand[(cand >= lo) & (cand <= hi)]
        if len(ticks) > 4:
            ticks = ticks[::3] if len(ticks) > 8 else ticks[::2]
        ax.set_yticks(ticks)
        ax.set_yticklabels([np.format_float_positional(v, trim="-") for v in ticks])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{task}, $N$ = 1,000", pad=2)
        ps.ygrid(ax)
    axes[0].set_ylabel("noisy training loss")
    return seeds


def panel_tstar(ax, rec):
    """Panel (b): floor crossing t* against N, every network counted in Fig. 2a (dots) and means.

    Args:
        ax: axes; rec: the collect record.
    Returns:
        dict task -> {N: mean t*}.
    """
    means = {}
    for task in TASKS:
        rows = [r for r in rec["series"] if r["task"] == task and is_read(r)]
        Ns = np.array([r["N"] for r in rows], float)
        ts = np.array([r["t_star"] for r in rows])
        uN = np.unique(Ns)
        for n in uN:      # one point per network beside its size's mean, as in Fig. 2a,b
            v = ts[Ns == n]
            ax.plot(n * 10 ** network_offsets(task, len(v)), v, "o", ms=ps.SMALL_MARKER_PT, mew=0,
                    **ps.mark_floor(TASK_COL[task]))
        m = [ts[Ns == n].mean() for n in uN]
        ax.plot(uN, m, "o-", ms=ps.MARKER_PT, lw=ps.DATA_LW, color=TASK_COL[task], label=task,
                **ps.mark_floor(TASK_COL[task]))
        means[task] = dict(zip(uN.astype(int), m))
    ax.set(xscale="log", yscale="log", xlim=(75, 5500), ylim=(8_000, 160_000),
           xlabel=ps.LABELS["size"], ylabel="floor crossing $t^*$ (iterations)")
    ax.set_xticks(SIZES)
    ax.set_yticks([10_000, 20_000, 50_000, 100_000])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(NullLocator())
        ps.count_ticks(axis)
    ax.legend(loc="lower center", ncol=3, borderaxespad=0.2, columnspacing=1.0, handlelength=1.4)
    ps.ygrid(ax)
    return means


def panel_snapshot(ax_c, ax_d, rec, noise):
    """Panels (c) and (d): t* against the snapshot estimate, and their difference in percent.

    In (d) the CDDM default networks sit left of the CDDM networks with scaled input weights, and
    each default network carries a vertical bar: the difference t* minus the snapshot estimate
    over the snapshot estimate's 90% batch-noise range (t*/hi - 1 to t*/lo - 1).

    Args:
        ax_c, ax_d: axes; rec: the collect record; noise: BATCH_NOISE contents (seed -> range) or {}.
    Returns:
        list of (name, task, t*, snapshot estimate, difference %) for every network with both, and
        dict seed -> (low %, high %) of the bars drawn.
    """
    ok = [o for o in rec["oracle"] if o["reason"] is None and np.isfinite(o["t_oracle"])]
    lim = (3_000, 150_000)
    xx = np.array(lim)
    ax_c.fill_between(xx, xx / 1.1, xx * 1.1, color=ps.GRID, lw=0, zorder=0)
    ax_c.plot(xx, xx, "-", color=ps.MUTED, lw=ps.THIN_LW)
    rows, bars = [], {}
    for k, task in enumerate(TASKS):
        o = [x for x in ok if x["task"] == task]
        default = np.array(["arm=control" in x["name"] for x in o])
        to = np.array([x["t_oracle"] for x in o])
        tr = np.array([x["t_rule"] for x in o])
        diff = (tr / to - 1) * 100
        split = default.any() and (~default).any()      # CDDM: defaults left, scaled inputs right
        for mask, is_default, centre, step in ((default, True, -0.2 if split else 0.0, 0.12),
                                               (~default, False, 0.18, 0.028)):
            mk = DEFAULT_MARK if is_default else SCALED_MARK
            ax_c.plot(to[mask], tr[mask], mk, ms=ps.MARKER_PT, mew=0, zorder=4,
                      **ps.mark_floor(TASK_COL[task]))
            n = int(mask.sum())
            xs = k + centre + step * (np.arange(n) - (n - 1) / 2)
            if is_default and not split:
                xs = k + np.linspace(-0.12, 0.12, max(n, 1))[:n]
            ax_d.plot(xs, diff[mask], mk, ms=ps.MARKER_PT, mew=0, zorder=4,
                      **ps.mark_floor(TASK_COL[task]))
            for x, xo in zip(np.array(o)[mask], xs):
                seed = x["name"].split("__s")[-1]
                if is_default and seed in noise:
                    b = noise[seed]
                    lo, hi = (b["t_rule"] / b["hi"] - 1) * 100, (b["t_rule"] / b["lo"] - 1) * 100
                    ax_d.plot([xo, xo], [lo, hi], "-", color=TASK_COL[task], lw=0.8, zorder=3)
                    bars[seed] = (lo, hi)
        rows += [(x["name"], task, t1, t0, d) for x, t1, t0, d in zip(o, tr, to, diff)]
        if task == "CDDM":
            for y in (diff.min(), diff.max()):
                ax_d.text(k + 0.38, y, f"{y:+.0f}%".replace("-", "\u2212"), fontsize=ps.TICK_PT,
                          ha="left", va="center", color=ps.INK)
    ax_c.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
             xlabel="snapshot estimate (iterations)", ylabel="floor crossing $t^*$ (iterations)")
    for axis in (ax_c.xaxis, ax_c.yaxis):
        axis.set_ticks([3_000, 10_000, 30_000, 100_000])
        axis.set_minor_locator(NullLocator())
        ps.count_ticks(axis)
    ax_c.legend(handles=[Line2D([], [], ls="none", marker=DEFAULT_MARK, ms=ps.MARKER_PT, color=ps.INK,
                                mew=0, label="default network"),
                         Line2D([], [], ls="none", marker=SCALED_MARK, ms=ps.MARKER_PT, color=ps.INK,
                                mew=0, label="input weights scaled")],
                loc="upper left", borderaxespad=0.2, handletextpad=0.2)
    ps.ygrid(ax_c)
    ax_d.axhspan(-10, 10, color=ps.GRID, lw=0, zorder=0)
    ax_d.axhline(0, color=ps.MUTED, lw=ps.THIN_LW)
    if bars:
        ax_d.legend(handles=[Line2D([], [], ls="none", marker="|", ms=7, mew=0.8,
                                    color=TASK_COL["CDDM"], label="batch-noise\n90% range")],
                    loc="lower left", borderaxespad=0.2, handletextpad=0.3)
    ax_d.set_xticks(range(len(TASKS)))
    ax_d.set_xticklabels(TASKS)
    ax_d.set(xlim=(-0.6, 2.6), ylim=(-60, 80),
             ylabel="$t^*$ minus snapshot estimate\n(% of snapshot estimate)")
    ps.ygrid(ax_d)
    return rows, bars


def draw(rec, noise):
    """Draw Extended Data Fig. 3 and print every number it shows.

    Args:
        rec: the collect record (decisions applied); noise: BATCH_NOISE contents, or {} if absent.
    Returns:
        path of the PDF written.
    """
    ps.setup()
    fig = ps.figure(h_mm=ED3_H_MM)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.05], hspace=0.42,
                  left=0.09, right=0.99, top=0.905, bottom=0.10)
    agrid = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.28)
    bgrid = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], width_ratios=[1.1, 1.1, 0.8], wspace=0.42)
    ax_a = [fig.add_subplot(agrid[k]) for k in range(3)]
    ax_b, ax_c, ax_d = (fig.add_subplot(bgrid[k]) for k in range(3))
    seeds = panel_examples(ax_a, rec)
    fig.legend(handles=[
        Line2D([], [], color=ps.GRID, lw=4),
        Line2D([], [], color=ps.MUTED, lw=2.5),
        Line2D([], [], color=ps.INK, lw=0.9),
        tuple(Line2D([], [], color=TASK_COL[t], lw=ps.DATA_LW, ls=(0, (1, 1.2))) for t in TASKS
              if rec["examples"][t][0]["tf"] is not None),      # DMTS has no fitted decay
        Line2D([], [], color=ps.INK, lw=ps.THIN_LW),
        Line2D([], [], color=ps.MUTED, lw=ps.THIN_LW, ls=(0, (4, 2))),
        tuple(Line2D([], [], color=TASK_COL[t], lw=0.8) for t in TASKS)],
        labels=["raw loss, 10th-90th percentile", "loss burst (masked)", "smoothed loss",
                "fitted decay", "loss floor", "1.05, 1.10, 1.20 × floor", "crossings"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        loc="upper center", bbox_to_anchor=(0.53, 0.998), ncol=7, handlelength=1.6,
        columnspacing=1.0, frameon=False)
    means = panel_tstar(ax_b, rec)
    rows, bars = panel_snapshot(ax_c, ax_d, rec, noise)
    for ax, L in ((ax_a[0], "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")):
        ps.panel_letter(ax, L, x_mm=9.0 if ax is ax_d else ps.PANEL_LETTER_X_MM)
    check_overlaps(fig)
    out = ps.save(fig, "fig_ed_readout")

    print("\nED3 a examples:", seeds)
    print("ED3 b mean t* per size:")
    for task, m in means.items():
        print(f"  {task}: " + "; ".join(f"N={N}: {v:,.0f}" for N, v in m.items()))
    print("ED3 c-d t* vs snapshot estimate (networks with both):")
    for name, task, tr, to, d in rows:
        print(f"  {task} {name.split('__', 1)[1][:60]}: t* {tr:,.0f} snapshot {to:,.0f} ({d:+.0f}%)")
    for task in TASKS:
        d = [r[4] for r in rows if r[1] == task]
        print(f"  {task}: n = {len(d)}, {min(d):+.0f}% to {max(d):+.0f}%")
    print(f"  networks with kept weights: {len(rec['oracle'])}; no t*: "
          f"{sum(o['reason'] is not None for o in rec['oracle'])}; no snapshot crossing: "
          f"{sum(o['reason'] is None and not np.isfinite(o['t_oracle']) for o in rec['oracle'])}")
    for seed, b in noise.items():
        print(f"  CDDM default s{seed}: snapshot {b['t_oracle']:,.0f}, 90% batch range "
              f"{b['lo']:,.0f}-{b['hi']:,.0f}, t* {b['t_rule']:,.0f}; panel d bar "
              f"{bars[seed][0]:+.0f}% to {bars[seed][1]:+.0f}%")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-noise", action="store_true", help="recompute the snapshot estimate's range")
    ap.add_argument("--oracle-dir", default=ORACLE_DIR, help="folder of oracle snapshot losses")
    args = ap.parse_args()
    if args.batch_noise:
        batch_noise(args.oracle_dir)
    else:
        with open(COLLECT, "rb") as fh:
            record = apply_decisions(pickle.load(fh))
        bn = {}
        if os.path.exists(BATCH_NOISE):
            with open(BATCH_NOISE) as fh:
                bn = json.load(fh)
        draw(record, bn)
