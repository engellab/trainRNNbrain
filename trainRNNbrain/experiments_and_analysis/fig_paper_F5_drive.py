#!/usr/bin/env python3
"""
Manuscript Figure 5 - A PER-UNIT PARTICIPATION DRIVE RECRUITS UNITS, AT A COST.

Every rule that adds a per-unit drive raises the 3-bit flip-flop's active count above its control's;
the multiplier and the cost in held-out R^2 and in stable training differ by rule; the recruited
units carry task signal; and combining two rules raises the count further at a larger R^2 cost
(B3, read as between the bars: dead-relu-paper#17).

  (a) THE RULES          each remedy, its glyph and one line on what it does, grouped bio-inspired /
                         engineering (criteria: Extended Data Table 4).
  (b) ONE SIZE, 75k      N = 1000: active units against held-out R^2 for every network of every remedy
                         and of the control, read at 75,000 iterations (the same-size rule). Source:
                         A3 (the intact scores of a3_remedy_ablation.py, 9 seeded draws).
  (c) ACROSS SIZES       top: multiplier (run's active count / its control's mean, same N, same
                         read-out) against N = 500-4000; bottom: R^2 minus the control's mean R^2.
                         Read at 1.05x each run's own noisy-loss floor (RMtop200,
                         read_remedies.py). Rescaling and read-out dropout have no floor read at
                         N >= 2000, so their whole series is drawn at 75k, open markers, dashed
                         (dead-relu-paper#16, as decided on 31 Oct).
  (d) RATE PENALTY ALONE (B2), N = 1000: active units over training, every run, against the
                         control's floor-read range; the noisy loss of the same runs above. No run
                         has a floor read.
  (e) A3                 left: R^2 after removing the k recruited units (k = active count minus the
                         control's mean; the k least participating active units) against removing k
                         random active units; N = 1000 filled, 4000 open. Right: R^2 after removing a
                         random 10/30/50% of all units, N = 1000, median over networks.
  (f) B3                 N = 1000 factorial of rate penalty (lambda 0.03), synaptic noise (sigma_w 1)
                         and rescaling: active units, R^2 and participation ratio per run, at the
                         floor (filled) and at 75k (open). Verdicts as pre-registered.

DATA. Read outputs copied from Della into data/remedy_reads_31oct/ (gitignored):
  remedies_catalogue.json  ~/rm_read/   (read_remedies.py 2d32b3b, Della array 16553347 + 16555725)
  a3_N<N>.json             ~/a3_read/   (a3_remedy_ablation.py 30efe65, Della jobs 16553440-43)
  read_b3.json             ~/b3b4_read/ (read_b3_b4_4000.py, Della job 16553903)
  rm_diag.json             ~/rm_diag/   (rm_exclusion_diag.py cb36d82, Della job 16554963)
and the B2 runs' loss and participation traces (TrainLosses.json, ParticipationTrace.pkl, config)
rsynced from Della ~/trainRNNbrain_c1ana/data/trained_RNNs/NBitFlipFlop_remedy_grid/ into the same
place under data/trained_RNNs. Panel (d) re-reads those runs with common.read_run.
Notebook: docs/project_trajectory/2026-10-31_1532, _1732, _1750 (the reads) and the entry that
records this figure.

Usage:  python fig_paper_F5_drive.py
Output: img/internal_figures/fig_paper_F5_drive.pdf (+ .svg), and a printout of every number the
        caption and Results 5 quote.
"""

import collections
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from common import DATA_DIR, load_run, read_run

READS = os.path.join(DATA_DIR, "..", "remedy_reads_31oct")
B2_DIR = os.path.join(DATA_DIR, "NBitFlipFlop_remedy_grid")
SIZES = (500, 1000, 2000, 4000)

# The remedies in the main figure, in legend order. Rate + sparsity is parked (board, "Decisions
# taken": the rate penalty alone replaces it) and appears only in the Extended Data.
MAIN = ("frm", "synnoise", "rescale", "duplicate", "dropout")
BIO = ("frm", "synnoise", "rescale")
NAME = {"control": "Control", "frm": "Rate penalty", "synnoise": "Noisy synapses",
        "rescale": "Rescaling", "duplicate": "Prune-and-duplicate", "dropout": "Read-out dropout",
        "frm+rws": "Rate + sparsity"}
WHAT = {"control": "task loss only",
        "frm": "penalises a peak rate above or below a target",
        "synnoise": "every weight $\\times (1 + \\sigma_w\\,\\xi)$, redrawn each step",
        "rescale": "tilts an inactive unit's inputs to excitation",
        "duplicate": "replaces an inactive unit by a copy",
        "dropout": "hides busy units from the read-out"}
COL = {"control": ps.BASE, "frm": ps.COND_COL["frm"], "synnoise": ps.COND_COL["synnoise"],
       "rescale": ps.COND_COL["rescale"], "duplicate": ps.COND_COL["duplicate"],
       "dropout": ps.COND_COL["dropout"], "frm+rws": ps.COND_COL["frm + rws"]}
MARK = {"control": "o", "frm": "o", "synnoise": "s", "rescale": "D", "duplicate": "^",
        "dropout": "v", "frm+rws": "P"}
# Remedies whose size series is drawn at 75k because they have no floor read at N >= 2000
# (dead-relu-paper#16, decided on 31 Oct).
AT_75K = ("rescale", "dropout")

B3_CELLS = ("none", "rate", "noise", "rescale", "rate+noise", "rate+rescale", "noise+rescale",
            "rate+noise+rescale")
B3_LABEL = {"none": "none", "rate": "rate", "noise": "noise", "rescale": "rescale",
            "rate+noise": "rate + noise", "rate+rescale": "rate + rescale",
            "noise+rescale": "noise + rescale", "rate+noise+rescale": "all three"}
B3_COL = {"none": ps.BASE, "rate": COL["frm"], "noise": COL["synnoise"], "rescale": COL["rescale"]}


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------

def load_catalogue():
    """The RMtop200 catalogue (read_remedies.py output), one dict per cell.

    Returns:
        list of dicts with keys cell, role, family, signature, N, n_runs, n_read, excluded, active,
        r2, r2_measured, t_star, active_d10, active_d20, active_75k, r2_75k, mult_*, dr2, dr2_75k.
        Arrays over runs: `active`, `r2`, `active_d10/d20` cover the runs read at 1.05x;
        `active_75k` and `r2_75k` cover every run with weights at 75k.
    """
    with open(os.path.join(READS, "remedies_catalogue.json")) as fh:
        return json.load(fh)


def size_series(cat):
    """Each family's size series: the one (family, signature) present at three or more sizes.

    The same selection a3_remedy_ablation.py uses, so panels (c) and (e) draw the same cells.

    Args:
        cat: load_catalogue() list.
    Returns:
        dict family -> {N: cell dict}; "control" maps to the role == "control" cells (Figure 1's).
    Raises:
        AssertionError if a family has more than one signature at three or more sizes, or a size
        appears twice.
    """
    groups = collections.defaultdict(list)
    for c in cat:
        if c["role"] in ("topup", "b2"):
            groups[(c["family"], c["signature"])].append(c)
    out = {"control": {c["N"]: c for c in cat if c["role"] == "control"}}
    for (fam, _), cells in groups.items():
        if len({c["N"] for c in cells}) < 3:
            continue
        assert fam not in out, f"two size series for {fam}"
        assert len({c["N"] for c in cells}) == len(cells), f"duplicate size in {fam}"
        out[fam] = {c["N"]: c for c in cells}
    return out


def load_a3():
    """The A3 ablation records (a3_remedy_ablation.py output) for every N.

    Returns:
        dict N -> list of per-network dicts (family, cell, n_active, r2_full, random_10/30/50,
        surplus, r2_recruited, r2_random_active, r2_keep_ctrl, r2_no_inactive).
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(READS, "a3_N*.json"))):
        with open(f) as fh:
            recs = json.load(fh)
        out[recs[0]["N"]] = recs
    return out


def load_b3():
    """The B3 read (read_b3_b4_4000.py output): runs, stats, n_found and verdicts, per read-out."""
    with open(os.path.join(READS, "read_b3.json")) as fh:
        return json.load(fh)


def load_b2(N):
    """Every B2 run (rate penalty alone, lambda 0.1) at size N, re-read with common.read_run.

    Args:
        N: network size (500, 1000 or 2000 are on the Mac).
    Returns:
        list of read_run dicts (it_loss, loss, it_act, active_series, reason, floor, t_star,
        bursts, ...), sorted by folder name.
    """
    runs = []
    for d in sorted(glob.glob(os.path.join(B2_DIR, f"EqType=h_N={N}_arm=frm0.1", "*", ""))):
        rec = load_run(d)
        runs.append(read_run(d, run=rec))
    return runs


# --------------------------------------------------------------------------------------------
# derived quantities
# --------------------------------------------------------------------------------------------

def per_run(series, fam, N, readout):
    """One family's runs at one size under one read-out, as multipliers and R^2 differences.

    Args:
        series: size_series() dict; fam: family key; N: size;
        readout: "floor" (1.05x each run's own floor) or "75k".
    Returns:
        dict with active (runs' counts), mult (count / control mean, same read-out), r2, dr2 (R^2
        minus the control's mean R^2, same read-out; every run kept, non-finite R^2 included),
        n_read, n_runs.
        Empty arrays when the cell is missing.
    """
    c, ctrl = series.get(fam, {}).get(N), series["control"][N]
    if c is None:
        return dict(active=np.array([]), mult=np.array([]), r2=np.array([]), dr2=np.array([]),
                    n_read=0, n_runs=0)
    ka, kr = ("active", "r2") if readout == "floor" else ("active_75k", "r2_75k")
    act, r2 = np.asarray(c[ka], float), np.asarray(c[kr], float)
    base_a, base_r = np.mean(ctrl[ka]), np.mean(ctrl[kr])
    return dict(active=act, mult=act / base_a, r2=r2, dr2=r2 - base_r,
                n_read=len(c[ka]), n_runs=c["n_runs"])


def clip(v, lo, hi):
    """Clip an array to [lo, hi] and return it with a mask of the clipped entries.

    Args:
        v: array-like; lo, hi: bounds.
    Returns:
        (clipped float array, bool array True where a value was outside the bounds).
    """
    v = np.asarray(v, float)
    return np.clip(v, lo, hi), (v < lo) | (v > hi)


# --------------------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------------------

def panel_rules(ax):
    """Panel (a): the remedies as a key - glyph, name, what it does - grouped by class.

    Args:
        ax: a blank axes.
    Returns:
        None.
    """
    ps.blank(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rows = [("head", "Bio-inspired: uses only the unit's own synapses and rate")]
    rows += [("fam", f) for f in BIO]
    rows += [("head", "Engineering: copies units or ranks the population")]
    rows += [("fam", f) for f in MAIN if f not in BIO]
    rows += [("fam", "control")]
    y, dy = 0.96, 0.118
    for kind, val in rows:
        if kind == "head":
            ax.text(0.0, y, val, fontsize=6.5, color=ps.MUTED, style="italic", va="center")
        else:
            ax.plot([0.035], [y], marker=MARK[val], ms=4.6, color=COL[val], mec=COL[val],
                    ls="none", clip_on=False)
            ax.text(0.075, y, NAME[val], fontsize=6.8, fontweight="bold", color=ps.INK, va="center")
            ax.text(0.47, y, WHAT[val], fontsize=6.3, color=ps.INK, va="center")
        y -= dy


def panel_one_size(ax, a3):
    """Panel (b): N = 1000 at 75k, active units against held-out R^2, every network.

    Args:
        ax: axes; a3: load_a3() dict.
    Returns:
        dict family -> (active list, r2 list) as drawn, for the printout.
    """
    drawn = collections.defaultdict(lambda: ([], []))
    for r in a3[1000]:
        fam = r["family"]
        if fam not in MAIN and fam != "control":
            continue
        drawn[fam][0].append(r["n_active"])
        drawn[fam][1].append(r["r2_full"])
    for fam in ("control",) + MAIN:
        a, r2 = drawn[fam]
        open_ = fam == "frm"          # rate alone: the count at 75k is one moment of a swing (d)
        ax.scatter(a, r2, s=16, marker=MARK[fam], facecolor="none" if open_ else COL[fam],
                   edgecolor=COL[fam], lw=0.8, zorder=3)
    ax.set_xlim(0, 1030)          # rate-penalty networks reach 995: whole markers, not halves
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_ylim(0.920, 0.956)
    ax.set_xlabel("Active units (of 1000)")
    ax.set_ylabel("Held-out $R^2$")
    ax.text(0.02, 0.97, "$N$ = 1000, 75k iterations", transform=ax.transAxes, fontsize=6.3,
            color=ps.MUTED, va="top")
    ax.annotate("rate penalty: count\nswings (see d)", xy=(935, 0.9268), xytext=(720, 0.9235),
                fontsize=5.8, color=COL["frm"], ha="center",
                arrowprops=dict(arrowstyle="-", color=COL["frm"], lw=0.5))
    return drawn


def panel_sizes(ax_m, ax_r, series):
    """Panel (c): multiplier (top) and R^2 difference (bottom) against N, per remedy.

    Floor reads filled with solid lines; the AT_75K families at 75k, open markers, dashed lines.
    A size with no read is left out of that family's line.

    Args:
        ax_m, ax_r: the two axes (shared x); series: size_series() dict.
    Returns:
        dict family -> {N: per_run dict} as drawn, for the printout.
    """
    offs = dict(zip(MAIN, np.linspace(-0.06, 0.06, len(MAIN))))
    drawn = {}
    for fam in MAIN:
        ro = "75k" if fam in AT_75K else "floor"
        drawn[fam] = {}
        xs, mm, rr = [], [], []
        for N in SIZES:
            d = per_run(series, fam, N, ro)
            drawn[fam][N] = d
            if not len(d["mult"]):
                continue
            x = N * 10 ** offs[fam]
            fc = "none" if ro == "75k" else COL[fam]
            ax_m.scatter(np.full(len(d["mult"]), x), d["mult"], s=7, marker=MARK[fam], facecolor=fc,
                         edgecolor=COL[fam], lw=0.6, alpha=0.75, zorder=3)
            dr, cut = clip(d["dr2"], -0.06, 0.01)
            ax_r.scatter(np.full(len(dr), x), dr, s=7, marker=MARK[fam], facecolor=fc,
                         edgecolor=COL[fam], lw=0.6, alpha=0.75, zorder=3)
            xs.append(x)
            mm.append(d["mult"].mean())
            rr.append(np.median(d["dr2"]))
        ls = "--" if fam in AT_75K else "-"
        ax_m.plot(xs, mm, ls=ls, color=COL[fam], lw=1.0, zorder=2)
        ax_r.plot(xs, rr, ls=ls, color=COL[fam], lw=1.0, zorder=2)
    for ax in (ax_m, ax_r):
        ax.set_xscale("log")
        ax.set_xticks(SIZES)
        ax.set_xticklabels([str(n) for n in SIZES])
        ax.minorticks_off()
        ax.set_xlim(400, 5000)
    ax_m.axhline(1, color=ps.FAINT, lw=0.6, zorder=1)
    ax_r.axhline(0, color=ps.FAINT, lw=0.6, zorder=1)
    ax_m.set_ylim(0.8, 7.0)
    ax_m.set_ylabel("Multiplier\n(active / control's)")
    ax_r.set_ylim(-0.062, 0.008)
    ax_r.set_ylabel("$R^2$ minus\ncontrol's")
    ax_r.set_xlabel("Network size $N$")
    plt.setp(ax_m.get_xticklabels(), visible=False)
    ax_m.text(430, 3.9, "rate penalty: no floor\nread at $N \\leq$ 1000", fontsize=5.6,
              color=COL["frm"], ha="left", va="center")
    h = [Line2D([], [], color=ps.MUTED, ls="-", marker="o", ms=3, lw=0.9, label="at the floor"),
         Line2D([], [], color=ps.MUTED, ls="--", marker="o", mfc="none", ms=3, lw=0.9,
                label="at 75k (no floor read\nat $N \\geq$ 2000)")]
    ax_m.legend(handles=h, loc="upper left", fontsize=5.6, handlelength=2.2, borderaxespad=0.1)
    return drawn


def panel_rate_alone(ax_l, ax_a, runs, ctrl_floor):
    """Panel (d): B2 at N = 1000 - noisy loss (top) and active units (bottom) over training.

    Args:
        ax_l, ax_a: the two axes (shared x); runs: load_b2(1000);
        ctrl_floor: the N = 1000 control's active counts at the floor (catalogue).
    Returns:
        dict with the per-run active range after 3k iterations and burst counts, for the printout.
    """
    info = []
    shades = [COL["frm"], "#a8341b", "#f08a6a"]
    for r, col in zip(runs, shades):
        it, L = np.asarray(r["it_loss"]), np.asarray(r["loss"])
        # vector-friendly: the per-bin maximum (the bursts) and median (the trend), 250-iteration bins
        nb = len(L) // 250
        Lb = L[:nb * 250].reshape(nb, 250)
        xb = it[:nb * 250].reshape(nb, 250)[:, 0] / 1000
        ax_l.plot(xb, np.clip(Lb.max(1), None, 1e3), color=col, lw=0.35, alpha=0.55)
        ax_l.plot(xb, np.median(Lb, 1), color=col, lw=0.8)
        ia, act = np.asarray(r["it_act"]), np.asarray(r["active_series"])
        ax_a.plot(ia / 1000, act, color=col, lw=0.45)
        late = act[ia >= 3000]
        info.append(dict(name=r["name"], reason=r["reason"], n_bursts=len(r["bursts"]),
                         lo=int(late.min()), hi=int(late.max()), loss_max=float(np.nanmax(L))))
    ax_a.axhspan(min(ctrl_floor), max(ctrl_floor), color=ps.BASE, alpha=0.25, lw=0, zorder=0)
    ax_a.text(198, min(ctrl_floor) - 15, "control at its floor", fontsize=5.6, color=ps.MUTED,
              ha="right", va="top")
    ax_l.set_yscale("log")
    ax_l.set_ylim(0.03, 1e3)
    ax_l.set_yticks([0.1, 10, 1000])
    ax_l.set_ylabel("Noisy loss")
    ax_a.set_ylim(0, 1020)
    ax_a.set_ylabel("Active units")
    ax_a.set_xlabel("Iteration (thousands)")
    for ax in (ax_l, ax_a):
        ax.set_xlim(0, 200)
    plt.setp(ax_l.get_xticklabels(), visible=False)
    ax_l.text(0.02, 0.95, "Rate penalty alone, $N$ = 1000: no floor read in 3 of 3 runs",
              transform=ax_l.transAxes, fontsize=6.0, color=ps.INK, va="top",
              bbox=dict(facecolor=ps.PAPER, edgecolor="none", pad=0.6, alpha=0.9))
    return info


def panel_a3(ax_s, ax_f, a3):
    """Panel (e): removing the recruited units (left) and random fractions of all units (right).

    Args:
        ax_s, ax_f: axes; a3: load_a3() dict.
    Returns:
        dict with the per-cell medians drawn, for the printout.
    """
    lo = -0.05
    ax_s.plot([lo, 1], [lo, 1], color=ps.FAINT, lw=0.6, zorder=1)
    out = {}
    for N, filled in ((1000, True), (4000, False)):
        for r in a3[N]:
            fam = r["family"]
            if fam not in MAIN:
                continue
            x = np.median(r["r2_random_active"])
            y, cut = clip([r["r2_recruited"]], lo, 1)
            ax_s.scatter([max(x, lo)], y, s=12, marker=MARK[fam],
                         facecolor=COL[fam] if filled else "none", edgecolor=COL[fam], lw=0.7,
                         zorder=3)
            out.setdefault((N, fam), []).append((float(x), float(r["r2_recruited"])))
    ax_s.set_xlim(lo, 0.45)
    ax_s.set_ylim(lo, 0.8)
    ax_s.set_xlabel("$R^2$, $k$ random active\nunits removed")
    ax_s.set_ylabel("$R^2$, $k$ recruited\nunits removed")
    ax_s.text(0.44, 0.40, "equal cost", fontsize=5.6, color=ps.MUTED, ha="right", va="top")
    h = [Line2D([], [], marker="o", color=ps.MUTED, ls="none", ms=3.5, label="$N$ = 1000"),
         Line2D([], [], marker="o", mfc="none", color=ps.MUTED, ls="none", ms=3.5,
                label="$N$ = 4000")]
    ax_s.legend(handles=h, loc="lower right", fontsize=5.6, handletextpad=0.2, borderaxespad=0.1)

    fr = np.array([0, 10, 30, 50])
    meds = {}
    for fam in ("control",) + MAIN:
        nets = [r for r in a3[1000] if r["family"] == fam]
        if not nets:
            continue
        y = [np.median([r["r2_full"] for r in nets])]
        for k in ("random_10", "random_30", "random_50"):
            y.append(np.median([np.median(r[k]) for r in nets]))
        meds[fam] = y
        ax_f.plot(fr, np.clip(y, 0, 1), marker=MARK[fam], ms=3, color=COL[fam],
                  lw=1.4 if fam == "control" else 0.8, mfc=COL[fam], zorder=3 if fam == "control" else 2)
    ax_f.set_xticks(fr)
    ax_f.set_xlim(-3, 53)
    ax_f.set_ylim(0, 1.0)
    ax_f.set_xlabel("Units removed at\nrandom (% of $N$)")
    ax_f.set_ylabel("Held-out $R^2$")
    ax_f.text(0.97, 0.97, "$N$ = 1000", transform=ax_f.transAxes, fontsize=6.0, color=ps.MUTED,
              ha="right", va="top")
    out["random"] = meds
    return out


def panel_b3(axs, b3):
    """Panel (f): B3, per run, active units / R^2 / participation ratio, at the floor and at 75k.

    Args:
        axs: three axes sharing y (one row per cell); b3: load_b3() dict.
    Returns:
        list of (cell, read-out, active, r2, pr) rows drawn, for the printout.
    """
    rows = []
    ypos = {c: len(B3_CELLS) - 1 - i for i, c in enumerate(B3_CELLS)}
    lims = [(200, 1000), (0.83, 0.96), (0, 14)]
    for cell in B3_CELLS:
        col = B3_COL.get(cell, ps.INK)
        for ro, dy, filled in (("floor", 0.14, True), ("fixed", -0.14, False)):
            for r in b3["runs"][cell]:
                rec = r[ro]
                if rec is None:
                    continue
                vals = [rec["active"], rec["r2"], rec["pr"]]
                rows.append((cell, ro, *vals))
                for ax, v, (lo, hi) in zip(axs, vals, lims):
                    v = np.nan if v is None else float(v)
                    if not np.isfinite(v) or v < lo:
                        # off-scale (an R^2 taken inside a loss burst): a marker at the left edge
                        ax.scatter([lo], [ypos[cell] + dy], s=10, marker="<", color=col, lw=0.6,
                                   facecolor=col if filled else "none", clip_on=False, zorder=3)
                        continue
                    ax.scatter([v], [ypos[cell] + dy], s=9, marker="o", lw=0.6, edgecolor=col,
                               facecolor=col if filled else "none", zorder=3)
    for ax, (lo, hi), lab in zip(axs, lims, ("Active units", "Held-out $R^2$", "Participation\nratio")):
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.6, len(B3_CELLS) - 0.4)
        ax.set_xlabel(lab)
        for y in range(len(B3_CELLS)):
            ax.axhline(y, color=ps.GRID, lw=0.4, zorder=0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    axs[0].set_yticks([ypos[c] for c in B3_CELLS])
    axs[0].set_yticklabels([B3_LABEL[c] for c in B3_CELLS], fontsize=6.2)
    axs[0].tick_params(axis="y", length=0)
    axs[1].set_xticks([0.85, 0.90, 0.95])
    axs[0].set_xticks([300, 600, 900])
    h = [Line2D([], [], marker="o", color=ps.MUTED, ls="none", ms=3.2, label="floor"),
         Line2D([], [], marker="o", mfc="none", color=ps.MUTED, ls="none", ms=3.2, label="75k"),
         Line2D([], [], marker="<", color=ps.MUTED, ls="none", ms=3.2, label="off scale")]
    axs[1].set_title("B3 factorial, $N$ = 1000", fontsize=6.5, pad=3)
    axs[1].legend(handles=h, loc="upper left", fontsize=5.6, handletextpad=0.1,
                  borderaxespad=0.1, labelspacing=0.25)
    return rows


# --------------------------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------------------------

def build():
    """Draw Figure 5 and print every number its caption and Results 5 quote.

    Returns:
        the path of the PDF written by paperstyle.save.
    """
    ps.setup()
    cat = load_catalogue()
    series = size_series(cat)
    a3, b3 = load_a3(), load_b3()
    b2 = load_b2(1000)

    fig = plt.figure(figsize=(180 * ps.MM, 190 * ps.MM))
    gs = GridSpec(3, 2, figure=fig, left=0.085, right=0.985, top=0.975, bottom=0.068,
                  wspace=0.30, hspace=0.30, height_ratios=[0.85, 1.15, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    panel_rules(ax_a)
    ax_b = fig.add_subplot(gs[0, 1])
    drawn_b = panel_one_size(ax_b, a3)

    g_c = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], hspace=0.12, height_ratios=[1.5, 1])
    ax_cm, ax_cr = fig.add_subplot(g_c[0]), fig.add_subplot(g_c[1])
    drawn_c = panel_sizes(ax_cm, ax_cr, series)

    g_d = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.12, height_ratios=[1, 1.5])
    ax_dl, ax_da = fig.add_subplot(g_d[0]), fig.add_subplot(g_d[1])
    info_d = panel_rate_alone(ax_dl, ax_da, b2, series["control"][1000]["active"])

    g_e = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 0], wspace=0.55)
    ax_es, ax_ef = fig.add_subplot(g_e[0]), fig.add_subplot(g_e[1])
    drawn_e = panel_a3(ax_es, ax_ef, a3)

    g_f = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, 1], wspace=0.12)
    axs_f = [fig.add_subplot(g_f[i]) for i in range(3)]
    rows_f = panel_b3(axs_f, b3)

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_cm, "c"), (ax_dl, "d"), (ax_es, "e"),
                       (axs_f[0], "f")):
        ps.panel_letter(ax, letter, dx=-0.30 if ax is axs_f[0] else (-0.02 if ax is ax_a else (-0.42 if ax is ax_es else -0.16)),
                        dy=1.04 if ax not in (ax_cm, ax_dl) else 1.06)

    report(series, drawn_b, drawn_c, info_d, drawn_e, rows_f, b3)
    return ps.save(fig, "fig_paper_F5_drive", w_mm=180)


def report(series, drawn_b, drawn_c, info_d, drawn_e, rows_f, b3):
    """Print the numbers the caption and Results 5 quote, from exactly what was drawn.

    Args:
        series: size_series(); drawn_b, drawn_c, info_d, drawn_e, rows_f: the panel functions'
        returns; b3: load_b3().
    Returns:
        None.
    """
    print("\n(b) N = 1000, 75k (A3 intact scores):")
    for fam, (a, r2) in drawn_b.items():
        print(f"  {fam:9s} n={len(a)} active {min(a)}-{max(a)} (mean {np.mean(a):.0f})"
              f"  R2 {min(r2):.3f}-{max(r2):.3f}")
    print("\n(c) multiplier and R2 difference (run - control mean), by N:")
    for N in SIZES:
        c = series["control"][N]
        print(f"  control N={N}: floor active {c['active']} R2 {np.round(c['r2'], 3).tolist()};"
              f" 75k active {c['active_75k']}")
    for fam, byN in drawn_c.items():
        for N, d in byN.items():
            if not len(d["mult"]):
                print(f"  {fam:9s} N={N}: no read ({d['n_runs']} runs)")
                continue
            print(f"  {fam:9s} N={N}: {d['n_read']}/{series[fam][N]['n_runs']} runs, mult "
                  f"{d['mult'].mean():.2f} ({d['mult'].min():.2f}-{d['mult'].max():.2f}), dR2 median "
                  f"{np.median(d['dr2']):+.3f} (range {d['dr2'].min():+.3f} to {d['dr2'].max():+.3f})")
    print("\n(d) B2, N = 1000:")
    for i in info_d:
        print(f"  {i['name'][:40]}: {i['reason']}; {i['n_bursts']} bursts; active after 3k "
              f"{i['lo']}-{i['hi']}; raw loss max {i['loss_max']:.3g}")
    print("\n(e) A3, (median random-k, recruited) per network:")
    for k, v in drawn_e.items():
        if k == "random":
            continue
        rec = [round(y, 2) for _, y in v]
        rnd = [round(x, 2) for x, _ in v]
        print(f"  N={k[0]} {k[1]:9s} recruited {rec}  random-k {rnd}")
    for fam, y in drawn_e["random"].items():
        print(f"  random removal N=1000 {fam:9s} intact/10/30/50%: {np.round(y, 2).tolist()}")
    print("\n(f) B3 verdicts:", b3["verdict"]["floor"]["verdict"], "(floor);",
          b3["verdict"]["fixed"]["verdict"], "(75k)")
    for cell in B3_CELLS:
        for ro in ("floor", "fixed"):
            rr = [r for r in rows_f if r[0] == cell and r[1] == ro]
            print(f"  {cell:20s} {ro:5s} n={len(rr)} active {[r[2] for r in rr]} "
                  f"R2 {[None if r[3] is None else round(r[3], 3) for r in rr]} "
                  f"PR {[round(r[4], 1) for r in rr]}")


if __name__ == "__main__":
    build()
