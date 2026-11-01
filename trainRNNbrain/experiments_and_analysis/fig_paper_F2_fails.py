#!/usr/bin/env python3
"""Extended Data Fig. 4 (was manuscript Figure 2) - every standard-setting sweep, the input-weight cap
and the re-initialisation of inactive units, on the default network (ReLU, bias 0, N = 1000), at BOTH
measurement points: each run's own floor crossing and a fixed 75,000 iterations; and, in its last
panel pair, the optimizer comparison (3BFF) at the floor crossing and at 200,000 iterations (its
75,000-iteration view is Figure 3d, so it is not repeated here). Figure 3 (`fig_paper_F3_settings.py`)
draws the 75k read from this script's cache. The old `fig_paper_F2.py` is kept because other code
imports it.

ROUND 2 (dead-relu-paper#54). Drawn at the printed width (165.1 mm) in the house style of
`paperstyle.py`: colours from the setting palette (default grey, other values on the teal ramp,
the rate and sparsity penalties in the violet rule family), never a task colour; counts with a comma
every three digits; the old bottom-line panel l (every value's mean at both points) is dropped, since
Figure 3a draws the 75k means and each pair here draws both points; panel l is now the optimizer.

TWO READ-OUTS (31 Oct, work package fig3-settings). Networks of one size are compared at a fixed
75,000 iterations (Pavel, 16 Oct); a ladder whose common budget is shorter (CDDM input scale, 50k) is
read at that budget (`fixed_point`). Each run's 75k read is `read_fixed`: R^2 measured on weights
from that iteration where they exist, else estimated from the loss ratio. In the drawing, open
symbols are the floor read and filled symbols the 75k read.

WHY TWO VIEWS PER KNOB. Each run is read where its noisy training loss first reaches 1.05x its own
fitted floor (`common.noisy_floor`), the project's rule. Its blind spot: a knob that speeds
convergence is read EARLIER, while more units are still active. CDDM input scale 2 reads ~700
active at its floor (crossing ~4,500) against the control's ~430 (crossing ~25,000), yet has FEWER
active units at every matched iteration from 25,000 on. So every ladder is drawn as a PAIR of panels
sharing one active-units axis:
  left   held-out R^2 at each run's floor against its active units there (the trade-off scatter:
         dose ladders coloured in dose order, categorical ones by marker shape, the standard network
         ringed in black; the axes are transposed from `fig_knobs_at_floor --tradeoff` so that the
         active-units axis is shared with the traces);
  right  active units against training iteration for every run, each run's floor crossing marked
         with the same symbol, so a point on the left and its crossing on the right sit at the
         same height. A run with no usable crossing is dashed. The dotted vertical line is the
         ladder's common budget.
Panels a-i are the nine ladders of `fig_knobs_at_floor.LADDERS`; j and k the revival screens
(input-weight cap and re-initialising inactive units); l the optimizer comparison, held-out R^2
against active units at each run's floor crossing and at 200,000 iterations (B5 cache,
data/b5_read_cache.pkl, written by `b5_read.py`).

BUDGETS DIFFER between ladders and are never compared across them: weight decay, recurrent noise and
activation 200k; metabolic, spectral radius, W_rec penalty, rate/sparsity and the CDDM revival
screen 75k (the metabolic ladder's 200k control is read at 75k via `end_at`); CDDM input scale 50k
(its 100k controls read at 50k); flip-flop input scale 150k; flip-flop revival 200k.

Floor counts and R^2 come from `heldout_r2.read_rung` unchanged (R^2 measured on kept
snapshots where they exist - only the CDDM input-scale ladder - otherwise estimated from the run's
end; on CDDM an R^2 difference below ~0.01 is scoring noise). Traces from
`common.runs_with_loss`.

Usage (repository root):
    python trainRNNbrain/experiments_and_analysis/fig_paper_F2_fails.py --collect   # Della CPU node:
                       read every run (R^2 from data/heldout_r2_cache.json), write the cache, draw
    python trainRNNbrain/experiments_and_analysis/fig_paper_F2_fails.py --probe     # per-run table
    python trainRNNbrain/experiments_and_analysis/fig_paper_F2_fails.py             # draw from the
                       cache (data/fig_paper_F2_fails_cache.pkl; copy it from Della to draw locally)
Output: img/internal_figures/fig_paper_F2_fails.{pdf,svg} (vector only, paperstyle.save), with a
layout check printed (legends covering data, overlapping text) and each value's mean active count
at its floor crossing and at 75,000 iterations.
"""

import argparse
import os
import pickle
import sys

import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fig_knobs_at_floor as K
import paperstyle as ps
from common import DATA_DIR, END_FRAC, logbin, noisy_floor, r2_at_floor_estimate, runs_with_loss, snapshot_at
from heldout_r2 import R2_DRAWS, heldout_r2, load_cache, read_rung, save_cache
from read_remedies import chain_member_ending_at

CACHE = os.path.normpath(os.path.join(DATA_DIR, "..", "fig_paper_F2_fails_cache.pkl"))
# The optimizer comparison's cache, written by `b5_read.py` (the same path as b5_read.CACHE; not
# imported from there because b5_read imports this module).
B5_CACHE = os.path.normpath(os.path.join(DATA_DIR, "..", "b5_read_cache.pkl"))
# Networks of the SAME size are compared at a fixed iteration (Pavel, 16 Oct); the floor read is for
# comparisons across sizes. A ladder whose common budget (`end_at`) is shorter is read at that budget.
FIXED_IT = 75_000
PROBE_STEP = 100            # iterations between participation probes and between logged losses

# The revival screens, continued to a budget where they can reach their floor (trajectory
# 2026-10-13: top-ups A and B). Each cell is read only when its folder holds finished runs; until
# then the panel is a placeholder. The CDDM control is the screen's own in-sweep control; the
# flip-flop screen has none, and its config differs from the claim-1 flip-flop control
# (`ff_revive_c200000`, arm none) only in the prune keys, so that control is its reference.
REVIVAL = [
    dict(stem="revival_cddm", knob="interventions to raise the active count", ref=0, categorical=True,
         task="CDDM, $N$ = 1000, 75,000 iterations",
         rungs=[("none", f"{DATA_DIR}/CDDM_screen3_c75000/EqType=h_N=1000_pen=none_arm=none"),
                ("input-weight\ncap 0.3", f"{DATA_DIR}/CDDM_screen3_c75000/EqType=h_N=1000_pen=none_arm=cap"),
                ("re-initialise\ninactive units", f"{DATA_DIR}/CDDM_screen3_c75000/EqType=h_N=1000_pen=none_arm=prune")]),
    dict(stem="revival_ff", knob="interventions to raise the active count", ref=0, categorical=True,
         task="3-bit flip-flop, $N$ = 1000, 200,000 iterations",
         rungs=[("none", f"{DATA_DIR}/NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=1000_pen=none_arm=none"),
                ("zero-out\ninactive units", f"{DATA_DIR}/NBitFlipFlop_replace_variants_c200000/EqType=h_k=3_N=1000_mode=zero_out_rate=0.025_mat=1000"),
                ("random\nredraw", f"{DATA_DIR}/NBitFlipFlop_replace_variants_c200000/EqType=h_k=3_N=1000_mode=random_rate=0.025_mat=1000")]),
]


def fixed_point(spec):
    """The iteration a ladder is read at for the same-size comparison.

    Args:
        spec: one entry of `fig_knobs_at_floor.LADDERS` or REVIVAL.
    Returns:
        float: FIXED_IT, or the ladder's `end_at` where that is shorter (CDDM input scale, 50,000).
    """
    e = spec.get("end_at")
    return float(min(FIXED_IT, e)) if e else float(FIXED_IT)


def read_fixed(r, t, cache):
    """One run read at a fixed iteration: its active units there and its held-out R^2 there.

    Ported from `fig_cosyne2027.read_fixed` (branch panelc-n100), with R^2 MEASURED wherever weights
    from iteration t exist, in this order: the run's own final weights when it ends at t; the member
    of its warm-start chain that ends at t (a top-up's parent); a kept snapshot within
    `common.SNAP_MAX_LAG` after t. Only where none exists is R^2 estimated as
    `common.r2_at_floor_estimate` does at the floor: the final weights' unexplained variance scaled by
    the ratio of the noisy loss at t (mean over the 2% of iterations up to t) to the loss at the end
    (median of the last END_FRAC, as `common.read_run`). All R^2 are held-out, R2_DRAWS seeded draws.
    A run that ends before t is read at its end and flagged short.

    Args:
        r: one `common.runs_with_loss` dict (it_loss, loss, it_act, active, run_dir);
        t: float, the iteration to read at; cache: an open heldout_r2 cache.
    Returns:
        dict(t_read: iteration read, short: True if the run ended before t, active: active units at
        the probe nearest t_read, probe: that probe's iteration, r2: held-out R^2 at t_read,
        r2_how: "final", "parent", "snapshot" or "estimate").
    """
    it = np.asarray(r["it_loss"], float)
    L = np.asarray(r["loss"], float)
    short = bool(it[-1] < t - PROBE_STEP)
    t_read = float(it[-1]) if short else float(t)
    it_a = np.asarray(r["it_act"], float)
    k = int(np.argmin(np.abs(it_a - t_read)))
    out = dict(t_read=t_read, short=short, active=float(r["active"][k]), probe=float(it_a[k]))

    def r2_of(d):
        """Held-out R^2 of the weights saved in folder d."""
        return heldout_r2(d, draws=R2_DRAWS, cache=cache)

    if short or it[-1] <= t + PROBE_STEP:                 # the run's own final weights are at t
        out.update(r2=r2_of(r["run_dir"]), r2_how="final")
        return out
    member = chain_member_ending_at(r["run_dir"], int(t))
    if member:
        out.update(r2=r2_of(member), r2_how="parent")
        return out
    snap = snapshot_at(r["run_dir"], t)
    if snap:
        out.update(r2=r2_of(snap), r2_how="snapshot")
        return out
    win = (it <= t_read) & (it > t_read - max(0.02 * t_read, 50.0))
    L_t = float(np.mean(L[win])) if win.any() else float(L[np.argmin(np.abs(it - t_read))])
    L_end = float(np.median(L[it >= it[-1] - END_FRAC * (it[-1] - it[0])]))
    out.update(r2=float(r2_at_floor_estimate(L_t, L_end, r2_of(r["run_dir"]))), r2_how="estimate")
    return out


def collect_spec(spec, r2cache):
    """Read one ladder: the floor read-out of every rung, and every run's participation trace.

    The floor counts and R^2 come from `heldout_r2.read_rung`, unchanged. The traces come
    from `common.runs_with_loss`, and each run's crossing is recomputed with the same
    `common.noisy_floor` call so it can be marked on its own trace; the two must agree, and an
    AssertionError says so if they do not.

    Args:
        spec: one entry of `fig_knobs_at_floor.LADDERS` or REVIVAL; r2cache: open heldout_r2 cache.
    Returns:
        dict(spec=spec, rungs=[dict(label, path, read, runs)]), with `read` read_rung's dict and
        `runs` one dict per run: name, it_act (iterations of the participation probes), active
        (active units at each probe), t_star (floor crossing, nan when the run has no usable
        crossing), reason (why not, or None) and fixed (read_fixed at fixed_point(spec)).
    """
    rungs = []
    for lab, path in spec["rungs"]:
        if not os.path.isdir(path):
            rungs.append(dict(label=lab, path=path, read=None, runs=[]))
            continue
        read = read_rung(path, spec.get("end_at"), r2cache)
        runs = []
        for r in runs_with_loss(path):
            it_a = np.asarray(r["it_act"], float)
            f = noisy_floor(r["it_loss"], r["loss"], task=r["task"])
            t, why = f["t_star"], f["reason"]
            if why is None and not it_a[0] <= t <= it_a[-1]:
                t, why = float("nan"), "participation trace does not cover the crossing"
            if why is not None:
                t = float("nan")
            runs.append(dict(name=r["name"], it_act=it_a, active=np.asarray(r["active"], float),
                             t_star=float(t), reason=why,
                             fixed=read_fixed(r, fixed_point(spec), r2cache)))
        mine = sorted(r["t_star"] for r in runs if np.isfinite(r["t_star"]))
        assert np.allclose(mine, sorted(read["cross"])), f"crossings disagree in {path}"
        rungs.append(dict(label=lab, path=path, read=read, runs=runs))
    return dict(spec=spec, rungs=rungs)


def collect():
    """Read every ladder and revival screen and pickle the result to CACHE. Returns CACHE."""
    r2cache = load_cache()
    out = dict(ladders=[collect_spec(s, r2cache) for s in K.LADDERS],
               revival=[collect_spec(s, r2cache) for s in REVIVAL])
    save_cache(r2cache)
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh)
    print(f"wrote {CACHE}")
    return CACHE


def active_at(run, t):
    """A run's active-unit count at the probe nearest iteration t, or nan if t is outside its trace.

    Args:
        run: one run dict from collect_spec; t: an absolute iteration.
    Returns:
        float.
    """
    it = run["it_act"]
    if not it[0] - 500 <= t <= it[-1] + 500:
        return float("nan")
    return float(run["active"][int(np.argmin(np.abs(it - t)))])


def probe(data):
    """Print, per run, its trace span, crossing, and counts at the floor and at fixed iterations."""
    for group in ("ladders", "revival"):
        for lad in data[group]:
            spec = lad["spec"]
            print(f"\n=== {spec['knob']}  ({spec['task']})  end_at={spec.get('end_at')}")
            for rg in lad["rungs"]:
                lab = rg["label"].replace("\n", " ")
                if rg["read"] is None:
                    print(f"  {lab:<26} no folder")
                    continue
                print(f"  {lab:<26} n={rg['read']['n']}  floor {rg['read']['floor'].astype(int).tolist()}"
                      f"  R2 {np.round(rg['read']['r2'], 3).tolist()}"
                      f"  measured {rg['read']['measured'].tolist()}")
                for r in rg["runs"]:
                    fx = r["fixed"]
                    print(f"     {r['name'][:9]:<10} at {fx['t_read'] / 1e3:.0f}k{' (short)' if fx['short'] else ''}:"
                          f" {fx['active']:.0f} active, R2 {fx['r2']:.4f} ({fx['r2_how']})")
                    at = " ".join(f"{active_at(r, t):5.0f}" for t in (5e3, 12.5e3, 25e3, 50e3, 75e3, 150e3, 200e3))
                    print(f"     {r['name'][:9]:<10} trace {r['it_act'][0]:>7.0f}-{r['it_act'][-1]:<7.0f}"
                          f" ({len(r['it_act'])} probes)  t*={r['t_star']:>7.0f}"
                          f"  @5k,12.5k,25k,50k,75k,150k,200k: {at}"
                          f"  {r['reason'] or ''}")


# ------------------------------------------------------------------------------------------------
# drawing
# ------------------------------------------------------------------------------------------------

# Panel order, by `stem`: CDDM dose ladders first, then the two categorical CDDM ladders and the
# input scale on both tasks, so the two input-scale ladders sit side by side.
ORDER = ["slide_k_weightdecay", "slide_k_recnoise", "slide_k_metabolic",
         "slide_k_recgain", "slide_k_rwm", "slide_k_frm_rws",
         "slide_k_activation", "slide_k_inputscale_cddm", "slide_k_inputscale"]
N_UNITS = 1000
Y_MAX = 1040                # room for markers at exactly 1000 (the rate penalty)
X_MIN = 2000                # first iteration drawn on the log axis of the traces. Below ~2,500 a log
                            # bin holds fewer than three probes (one every 100 iterations), and no
                            # run reaches its floor before 4,400.
K_TICKS = [1e4, 5e4, 2e5]   # labelled iterations; 2,000, 5,000, 20,000 and 100,000 get unlabelled ticks
K_MINOR = [2e3, 5e3, 2e4, 1e5]
TRACE_BINS = 40             # log-spaced bins the drawn traces are median-reduced into (see panel_pair)
# Legend text of every value, by stem, in value order. Parameters left out here (leak 0.01, beta 25,
# the sigmoid's gain and offset, the penalty weights) are in Methods.
VALUE_LABELS = {
    "slide_k_weightdecay": ["0", "$10^{-6}$ (default)", "$10^{-5}$", "$10^{-4}$"],
    "slide_k_recnoise": ["0", "0.01", "0.05 (default)", "0.1"],
    "slide_k_metabolic": ["0 (default)", "0.01", "0.1", "1", "10"],
    "slide_k_recgain": ["0.5", "1.2 (default)", "2.0"],
    "slide_k_rwm": ["0 (default)", "0.05", "0.2"],
    "slide_k_activation": ["ReLU", "leaky", "softplus", "sigmoid"],   # ReLU is the default (grey)
    "slide_k_inputscale_cddm": ["default", "0.5", "2", "5", "20"],
    "slide_k_inputscale": ["default", "0.5", "2", "5", "20"],
    "slide_k_frm_rws": ["default", "rate", "sparsity", "both"],
    "revival_cddm": ["none (default)", "input-weight cap", "re-initialised"],
    "revival_ff": ["none (default)", "outgoing zeroed", "outgoing kept"],
}
# Panel titles: the setting by its glossary name, then the task.
TITLES = {
    "slide_k_weightdecay": "weight decay, CDDM",
    "slide_k_recnoise": "recurrent noise $\\sigma_{rec}$, CDDM",
    "slide_k_metabolic": "metabolic penalty $\\lambda_{met}$, CDDM",
    "slide_k_recgain": "initial spectral radius, CDDM",
    "slide_k_rwm": "recurrent-weight magnitude penalty, CDDM",
    "slide_k_activation": "activation function, CDDM",
    "slide_k_inputscale_cddm": "initial input-weight scale, CDDM",
    "slide_k_inputscale": "initial input-weight scale, 3BFF",
    "slide_k_frm_rws": "rate and sparsity penalties, CDDM",
    "revival_cddm": "revival methods, CDDM",
    "revival_ff": "re-initialising inactive units, 3BFF",
}
# A ladder whose traces leave no room for a legend borrows the legend of a ladder with the same
# values and colours; its dashed traces still mark the runs with no floor crossing.
BORROW_LEGEND = {"slide_k_inputscale": "slide_k_inputscale_cddm"}
# Legends placed by hand (matplotlib loc, anchor in axes fractions of the traces panel) where the
# automatic search finds no free spot; `check_layout` still tests that they cover no data. The third
# item is the number of legend columns. "r2": one column in the empty upper half of the R^2 panel.
LEGEND_AT = {"slide_k_activation": "r2",
             "slide_k_frm_rws": ("lower left", (0.02, 0.02), 1)}
# Rate-penalty values, a per-unit rule, drawn in the violet rule family (paperstyle.RULE_COLORS), by
# (stem, value index). The sparsity penalty alone is a standard setting and stays on the teal ramp.
PENALTY_COLORS = {("slide_k_frm_rws", 1): ps.RATE_PENALTY, ("slide_k_frm_rws", 3): ps.RATE_SPARSITY}
# Revival methods: one marker per method across both screens (cap: square; re-initialising: diamond),
# the shade telling the two re-initialising variants apart. The CDDM screen re-initialises with the
# outgoing weights kept (no reinit_mode in its configs: Trainer's default "random"), so it shares the
# 3BFF "outgoing kept" style.
REVIVAL_STYLE = {("revival_cddm", 1): (ps.SETTING_RAMP[1], "s"), ("revival_cddm", 2): (ps.SETTING_RAMP[2], "D"),
                 ("revival_ff", 1): (ps.SETTING_RAMP[3], "D"), ("revival_ff", 2): (ps.SETTING_RAMP[2], "D")}
MARKERS = "osD^v"
# The optimizer comparison (panel l). SGD learning rates take the teal ramp, light to dark; the two
# Adam variants are SETTING teal and differ from SGD in marker shape (circle, square) and position.
SGD_LRS = ["0.03", "0.1", "0.3", "1", "3"]
SGD_MARKER = "^"
OPT_R2_LO = 0.86            # R^2 below this is drawn on the bottom edge as a downward triangle


def rung_style(spec, i, n):
    """Colour and marker of value i of a sweep, by meaning (paperstyle palettes).

    The default value is BASELINE grey. The rate penalty (alone or with the sparsity penalty) takes
    the violet rule family. The revival methods follow REVIVAL_STYLE. Every other value takes the teal
    setting ramp: ordered values light to dark in sweep order (circles); unordered values (activation
    function, sparsity penalty) ramp steps 1-3 with one marker shape each. No task colour is used.

    Args:
        spec: a sweep (fig_knobs_at_floor.LADDERS or REVIVAL entry); i: value index; n: number of values.
    Returns:
        (colour, marker).
    """
    if i == spec["ref"]:
        return ps.BASELINE, "o"
    if (spec["stem"], i) in PENALTY_COLORS:
        return PENALTY_COLORS[(spec["stem"], i)], MARKERS[i]
    if (spec["stem"], i) in REVIVAL_STYLE:
        return REVIVAL_STYLE[(spec["stem"], i)]
    k = i - (i > spec["ref"])                       # position among the non-default values
    if spec.get("categorical", False):
        return ps.SETTING_RAMP[1 + k], MARKERS[i]
    steps = {1: [2], 2: [1, 3], 3: [1, 2, 3], 4: [0, 1, 2, 3]}[n - 1]
    return ps.SETTING_RAMP[steps[k]], "o"


def b5_style(opt, wd, lr):
    """Colour, marker, fill and legend label of one optimizer-comparison arm.

    Args:
        opt, wd, lr: the arm, as strings ('adam', '1e-6', '1e-3').
    Returns:
        dict(color, marker, filled, label); `filled` is False for SGD without weight decay.
    """
    if opt == "adam" and wd == "1e-6":
        return dict(color=ps.BASELINE, marker="o", filled=True, label="Adam (default)")
    if opt == "adam":
        return dict(color=ps.SETTING, marker="o", filled=True, label="Adam, no weight decay")
    if opt == "adamw":
        return dict(color=ps.SETTING, marker="s", filled=True, label="AdamW")
    # learning rate 3 shares the darkest shade with 1 (four ramp steps); its runs all sit below the
    # R^2 axis, drawn as downward triangles on the bottom edge
    col = ps.SETTING_RAMP[min(SGD_LRS.index(lr), 3)]
    return dict(color=col, marker=SGD_MARKER, filled=wd != "0", label=f"SGD, learning rate {lr}")


def b5_points(rows, key):
    """(active, R^2) of every run of one arm at one measurement point, finite values only.

    Args:
        rows: the arm's run dicts from b5_read; key: "floor", "fixed" (75,000 iterations) or "end"
            (200,000 iterations).
    Returns:
        list of (active units, held-out R^2).
    """
    if key == "floor":
        pts = [(r["act_floor"], r["r2_floor"]) for r in rows]
    elif key == "fixed":
        pts = [(r["fixed"]["active"], r["fixed"]["r2"]) for r in rows]
    else:
        pts = [(r["act_end"], r["r2_end"]) for r in rows]
    return [(a, v) for a, v in pts if np.isfinite(a) and np.isfinite(v)]


def b5_scatter(ax, b5, key, skip_lr=()):
    """Every optimizer-comparison run on one axes: held-out R^2 against active units.

    A run below OPT_R2_LO is drawn on the bottom edge as a downward triangle in its arm's colour.

    Args:
        ax: axes; b5: b5_read rows, a list of ((opt, wd, lr), runs); key: "floor", "fixed" or "end";
        skip_lr: SGD learning rates (strings) not drawn.
    Returns:
        (list of (arm, [active], [R^2]) per arm drawn, list of legend handles, one per arm with
        weight decay, plus one for "open: SGD without weight decay").
    """
    out, handles = [], []
    for (opt, wd, lr), rows in b5:
        if opt == "sgd" and lr in skip_lr:
            continue
        st = b5_style(opt, wd, lr)
        pts = b5_points(rows, key)
        out.append(((opt, wd, lr), [a for a, _ in pts], [v for _, v in pts]))
        is_ref = opt == "adam" and wd == "1e-6"
        for a, v in pts:
            low = v < OPT_R2_LO
            ax.scatter([a], [OPT_R2_LO if low else v], s=10, marker="v" if low else st["marker"],
                       facecolors=st["color"] if st["filled"] else "none",
                       edgecolors=ps.INK if is_ref else st["color"], linewidths=0.6,
                       zorder=4 + is_ref, clip_on=False)
        if wd != "0" or opt != "sgd":
            handles.append(Line2D([], [], ls="none", marker=st["marker"], ms=3.0, color=st["color"],
                                  mec=ps.INK if is_ref else st["color"], mew=0.5, label=st["label"]))
    handles.append(Line2D([], [], ls="none", marker=SGD_MARKER, ms=3.0, mfc="none", mec=ps.MUTED,
                          mew=0.6, label="open: SGD, no weight decay"))
    ax.set_xlim(0, N_UNITS)
    ax.set_xticks([0, 500, 1000])
    ps.count_ticks(ax.xaxis)
    ax.set_ylim(OPT_R2_LO, 0.97)
    return out, handles


def budget(lad):
    """A ladder's common budget: `end_at` when set, else the shortest trace among its runs.

    Args:
        lad: one collected ladder.
    Returns:
        the iteration (float), or nan if no rung has runs.
    """
    if lad["spec"].get("end_at"):
        return float(lad["spec"]["end_at"])
    ends = [r["it_act"][-1] for rg in lad["rungs"] for r in rg["runs"]]
    return float(min(ends)) if ends else float("nan")


def panel_pair(ax_r2, ax_tr, lad):
    """One ladder as two panels sharing the active-units axis.

    Left (narrow): held-out R^2 (x) against active units (y), one point per run and measurement
    point: open at the run's floor crossing, filled at the ladder's fixed point (fixed_point: 75,000
    iterations, or a shorter budget). Right: active units over training for every run (log iteration
    axis), each run's floor crossing marked with the open symbol and its fixed-point measurement with
    the filled one, so a point on the left and its mark on the right sit at the same height. A run
    with no floor crossing is drawn dashed, with only the filled symbol. A dashed vertical line marks
    75,000 iterations where the budget is longer; the dotted one the budget.

    Each trace is drawn as medians in TRACE_BINS log-spaced bins (common.logbin). Probe-to-probe
    noise is small on both tasks (median step 1-4 units), but a 3BFF count drops by 100-350 units
    within one probe 4-10 times per run on the input-scale ladder, and drawn raw that ladder's 15
    traces cannot be told apart; the medians keep the level a trace stays at and lose the one-probe
    spikes. The crossing symbol is the RAW count at the probe nearest the crossing, the number the
    left panel reports, so on 3BFF it can sit off its line.

    Args:
        ax_r2, ax_tr: the two axes; lad: one collected ladder.
    Returns:
        list of legend handles (one per value).
    """
    spec = lad["spec"]
    n = len(lad["rungs"])
    labels = VALUE_LABELS[spec["stem"]]
    handles = []
    for i, rg in enumerate(lad["rungs"]):
        col, mk = rung_style(spec, i, n)
        ref = i == spec["ref"]
        edge = ps.INK if ref else "none"
        rd = rg["read"]
        if rd is not None and len(rd["floor"]):
            ax_r2.scatter(rd["r2"], rd["floor"], s=12 if ref else 9, marker=mk, facecolors="none",
                          edgecolors=col, linewidths=0.6, zorder=3)
        fx = [r["fixed"] for r in rg["runs"]]
        ax_r2.scatter([f["r2"] for f in fx], [f["active"] for f in fx], s=13 if ref else 10, marker=mk,
                      color=col, edgecolors=edge, linewidths=0.6, zorder=4 + ref)
        for r in rg["runs"]:
            keep = (r["it_act"] >= X_MIN) & (r["it_act"] <= budget(lad))
            tb, yb = logbin(r["it_act"][keep], r["active"][keep], nbins=TRACE_BINS)
            ax_tr.plot(tb, yb, color=col, lw=0.8 if ref else ps.THIN_LW + 0.1, alpha=0.9,
                       ls="-" if np.isfinite(r["t_star"]) else (0, (2, 1.2)), zorder=2 + ref)
        for r in rg["runs"]:
            if np.isfinite(r["t_star"]):
                ax_tr.scatter([r["t_star"]], [active_at(r, r["t_star"])], s=12 if ref else 9,
                              marker=mk, facecolors="none", edgecolors=col, linewidths=0.6, zorder=5)
            ax_tr.scatter([r["fixed"]["probe"]], [r["fixed"]["active"]], s=13 if ref else 10,
                          marker=mk, color=col, edgecolors=edge, linewidths=0.6, zorder=6 + ref)
        # runs with no floor crossing are dashed; the caption says so, the legend does not count them
        handles.append(Line2D([], [], color=col, marker=mk, ms=3.0, lw=0.8,
                              mec=ps.INK if ref else col, mew=0.5 if ref else 0, label=labels[i]))
    B = budget(lad)
    ax_tr.set_xscale("log")
    ax_tr.set_xlim(X_MIN, B * 1.25)
    ax_tr.set_xticks([t for t in K_TICKS if t <= B * 1.25])
    ax_tr.set_xticks([t for t in K_MINOR if t <= B * 1.25], minor=True)
    ps.count_ticks(ax_tr.xaxis)
    ax_tr.axvline(B, color=ps.FAINT, lw=0.6, ls=":", zorder=1)
    if fixed_point(spec) < B:
        ax_tr.axvline(fixed_point(spec), color=ps.FAINT, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for ax in (ax_r2, ax_tr):
        ax.set_ylim(0, Y_MAX)
        ax.set_yticks([0, 500, 1000])
        ps.count_ticks(ax.yaxis)
        ax.axhline(N_UNITS, color=ps.GRID, lw=0.6, zorder=0)
    ax_tr.tick_params(labelleft=False)
    return handles


def value_means(lads):
    """Each value's mean active count at its floor crossing and at the ladder's fixed point.

    Args:
        lads: collected ladders.
    Returns:
        list of (stem, fixed point, [(value label, floor mean or nan, fixed-point mean)]).
    """
    rows = []
    for lad in lads:
        spec = lad["spec"]
        out = []
        for i, rg in enumerate(lad["rungs"]):
            fl = rg["read"]["floor"] if rg["read"] is not None else []
            ends = [r["fixed"]["active"] for r in rg["runs"]]
            out.append((VALUE_LABELS[spec["stem"]][i], float(np.mean(fl)) if len(fl) else float("nan"),
                        float(np.mean(ends)) if ends else float("nan")))
        rows.append((spec["stem"], fixed_point(spec), out))
    return rows


def covered(ax, leg):
    """Number of drawn trace vertices and symbols that fall inside a legend's box; a legend that
    reaches outside its axes counts as covering everything.

    Args:
        ax: a traces axes, or a tuple of axes for a legend spanning a panel pair (then every axes'
            data is tested and the outside-the-axes rule does not apply); leg: its legend.
    Returns:
        int.
    """
    axes = ax if isinstance(ax, tuple) else (ax,)
    fig = axes[0].figure
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer()).padded(2)
    box = axes[0].get_window_extent()
    if len(axes) == 1 and (bb.x0 < box.x0 - 1 or bb.x1 > box.x1 + 1 or bb.y0 < box.y0 - 1 or bb.y1 > box.y1 + 1):
        return 10 ** 6
    pts = []
    for a in axes:
        pts += [a.transData.transform(ln.get_xydata()) for ln in a.lines if len(ln.get_xydata()) > 2]
        pts += [a.transData.transform(c.get_offsets()) for c in a.collections]
    pts = np.concatenate(pts) if pts else np.zeros((0, 2))
    return int(((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) &
                (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).sum())


def place_legend(ax, handles, alt=None):
    """Put the value legend where it covers no data: inside the traces axes, else inside `alt`.

    The legend's size is measured once per column count; candidate positions are then tested
    arithmetically against every drawn vertex and symbol: the four corners first (one column, then
    two, then three), then a grid over the whole axes. When the traces axes has no free spot (dense
    traces), the same search runs on `alt`, the paired R^2 panel, whose points sit in one column.
    Falls back to the traces axes' lower-left corner, which `check_layout` then reports.

    Args:
        ax: the traces axes; handles: legend handles; alt: a second axes to try, or None.
    Returns:
        (the axes holding the legend - a tuple (ax, alt) for a legend spanning both - , the legend).
    """
    kw = dict(handles=handles, fontsize=ps.TICK_PT, handlelength=1.1, borderaxespad=0.0,
              labelspacing=0.15, handletextpad=0.3, columnspacing=0.6, loc="lower left")
    fig = ax.figure
    rend = fig.canvas.get_renderer()
    m = 0.015                                   # margin from the axes edge, in axes fractions

    def search(a):
        """The legend placed on axes `a` covering no data, or None (no legend left on `a`)."""
        fig.canvas.draw()
        pts = [a.transData.transform(ln.get_xydata()) for ln in a.lines if len(ln.get_xydata()) > 2]
        pts += [a.transData.transform(c.get_offsets()) for c in a.collections]
        pts = np.concatenate(pts) if pts else np.zeros((0, 2))
        box = a.get_window_extent()
        for grid in (False, True):
            for ncol in (1, 2, 3):
                leg = a.legend(ncol=ncol, bbox_to_anchor=(0, 0), **kw)
                fig.canvas.draw()
                bb = leg.get_window_extent(rend)
                w, h = (bb.width + 4) / box.width, (bb.height + 4) / box.height
                if w > 1 - 2 * m or h > 1 - 2 * m:
                    continue
                if grid:
                    cand = [(x, y) for y in np.arange(m, 1 - h - m, 0.025)
                            for x in np.arange(m, 1 - w - m, 0.025)]
                else:
                    cand = [(m, m), (1 - w - m, 1 - h - m), (1 - w - m, m), (m, 1 - h - m)]
                for x, y in cand:
                    x0, y0 = box.x0 + x * box.width, box.y0 + y * box.height
                    x1, y1 = x0 + w * box.width, y0 + h * box.height
                    hit = ((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)).any()
                    if not hit:
                        return a.legend(ncol=ncol, bbox_to_anchor=(x + 2 / box.width, y + 2 / box.height), **kw)
        if a.get_legend() is not None:
            a.get_legend().remove()
        return None

    for a in (ax, alt):
        if a is not None:
            leg = search(a)
            if leg is not None:
                return a, leg
    if alt is not None:
        # neither panel alone has room (the R^2 panel is narrower than the legend): search the
        # rectangle spanning both panels, testing the data of both, with the legend in figure units
        fig.canvas.draw()
        pts = []
        for a in (ax, alt):
            pts += [a.transData.transform(ln.get_xydata()) for ln in a.lines if len(ln.get_xydata()) > 2]
            pts += [a.transData.transform(c.get_offsets()) for c in a.collections]
        b1, b2 = alt.get_window_extent(), ax.get_window_extent()
        # the panels' left and bottom axis lines are obstacles too: a legend must not straddle them
        for b in (b1, b2):
            ys = np.linspace(b.y0, b.y1, 200)
            xs = np.linspace(b.x0, b.x1, 200)
            pts += [np.c_[np.full_like(ys, b.x0), ys], np.c_[xs, np.full_like(xs, b.y0)]]
        pts = np.concatenate(pts)
        X0, X1, Y0, Y1 = min(b1.x0, b2.x0), max(b1.x1, b2.x1), max(b1.y0, b2.y0), min(b1.y1, b2.y1)
        W, H = fig.bbox.width, fig.bbox.height
        for ncol in (1, 2, 3):
            leg = fig.legend(ncol=ncol, bbox_to_anchor=(0, 0), **kw)
            fig.canvas.draw()
            bb = leg.get_window_extent(rend)
            w, h = bb.width + 4, bb.height + 4
            leg.remove()
            for y0 in np.arange(Y1 - h - 3, Y0 + 3, -0.02 * (Y1 - Y0)):
                for x0 in np.arange(X0 + 3, X1 - w - 3, 0.02 * (X1 - X0)):
                    hit = ((pts[:, 0] > x0) & (pts[:, 0] < x0 + w) & (pts[:, 1] > y0) & (pts[:, 1] < y0 + h)).any()
                    if not hit:
                        return (ax, alt), fig.legend(ncol=ncol, bbox_to_anchor=((x0 + 2) / W, (y0 + 2) / H), **kw)
    return ax, ax.legend(ncol=1, bbox_to_anchor=(m, m), **kw)


def optimizer_pair(ax_fl, ax_end, b5):
    """Panel l: the optimizer comparison at each run's floor crossing and at 200,000 iterations.

    Its 75,000-iteration view is Figure 3d and is not repeated. Markers and colours are those of
    Figure 3d (b5_style); the two groups are labelled directly ("Adam, AdamW", "SGD") because no
    corner of either axes is free for a legend. The R^2 axis runs to 1.0 to make room for the label.

    Args:
        ax_fl, ax_end: the two axes; b5: b5_read rows.
    Returns:
        dict(floor=..., end=...) of b5_scatter's per-arm points.
    """
    nums = {}
    for ax, key, title in ((ax_fl, "floor", "floor crossing"),
                           (ax_end, "end", "200,000 iterations")):
        nums[key], _ = b5_scatter(ax, b5, key)
        ax.set_ylim(OPT_R2_LO, 1.0)
        ax.set_title(title, fontsize=ps.LABEL_PT, pad=2)
        ax.set_xlabel(ps.LABELS["active"], labelpad=1)
    ax_fl.set_xticks([0, 500])
    ps.count_ticks(ax_fl.xaxis)
    for ax in (ax_fl, ax_end):
        ax.set_yticks([0.88, 0.92, 0.96, 1.0])
    ax_fl.set_ylabel(ps.LABELS["r2"], labelpad=1)
    ax_end.tick_params(labelleft=False)
    # direct labels instead of a legend (no free corner holds one); the markers are those of Fig. 3d
    for ax in (ax_fl, ax_end):
        ax.text(30, 0.985, "Adam, AdamW", fontsize=ps.TICK_PT, color=ps.INK, va="center")
    ax_end.text(560, 0.93, "SGD", fontsize=ps.TICK_PT, color=ps.INK, va="center")
    return nums


def draw(data, b5):
    """Assemble Extended Data Fig. 4 from the collected data and write it (PDF + SVG).

    Layout, 165.1 mm wide (the printed width): four rows of three slots. Each of the nine sweeps and
    the two revival screens takes one slot as a PAIR of panels that share the active-units axis (the
    narrow R^2 panel and the traces panel, see panel_pair), so a run's point on the trade-off and its
    floor crossing on its trace sit at the same height. The last slot holds the optimizer comparison
    (optimizer_pair).

    Args:
        data: the dict `collect` writes; b5: the optimizer comparison's rows (B5_CACHE).
    Returns:
        (output path, value_means rows, optimizer points).
    """
    # the specs (titles, labels) come from this file, not from the cache, so a wording change
    # needs no re-collect; the rung folders must still be the ones the cache was read from
    current = {s["stem"]: s for s in K.LADDERS + REVIVAL}
    for lad in data["ladders"] + data["revival"]:
        spec = current[lad["spec"]["stem"]]
        # compared below the data root: the cache may have been written on Della
        assert ([p.split("trained_RNNs/")[-1] for _, p in spec["rungs"]]
                == [rg["path"].split("trained_RNNs/")[-1] for rg in lad["rungs"]]), \
            f"{spec['stem']}: rungs changed since the cache was collected; re-run --collect"
        lad["spec"] = spec
    by_stem = {lad["spec"]["stem"]: lad for lad in data["ladders"]}
    lads = [by_stem[s] for s in ORDER]
    rev = data["revival"]

    ps.setup()
    fig = ps.figure(178)
    gs = GridSpec(4, 3, figure=fig, left=0.062, right=0.976, top=0.950, bottom=0.042,
                  hspace=0.46, wspace=0.26)
    slots = [gs[i // 3, i % 3] for i in range(12)]
    letters = iter("abcdefghijkl")
    pairs = []
    for slot, lad in zip(slots, lads + rev):
        sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=slot, width_ratios=[0.42, 1.0], wspace=0.14)
        ax_r2, ax_tr = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])
        ps.panel_letter(ax_r2, next(letters), x_mm=7.5, y_mm=4.0)
        handles = panel_pair(ax_r2, ax_tr, lad)
        ax_r2.set_xlabel(ps.LABELS["r2"], labelpad=1)
        ax_tr.set_xlabel(ps.LABELS["iteration"], labelpad=1)
        pairs.append((ax_r2, ax_tr, lad, handles))

    # one R^2 range for every CDDM pair so they compare by eye; 3BFF gets its own
    for task_ff, ticks in ((False, [0.80, 0.83]), (True, [0.950, 0.953])):
        group = [p for p in pairs if ("flip" in p[2]["spec"]["task"]) == task_ff]
        r2 = np.concatenate([rg["read"]["r2"] for p in group for rg in p[2]["rungs"]
                             if rg["read"] is not None]
                            + [[r["fixed"]["r2"] for p in group for rg in p[2]["rungs"] for r in rg["runs"]]])
        pad = 0.08 * (r2.max() - r2.min())
        for p in group:
            p[0].set_xlim(r2.min() - pad, r2.max() + pad)
            p[0].set_xticks(ticks)
    for i, (ax_r2, ax_tr, lad, handles) in enumerate(pairs):
        pos_l, pos_r = ax_r2.get_position(), ax_tr.get_position()
        fig.text((pos_l.x0 + pos_r.x1) / 2, pos_l.y1 + 0.006, TITLES[lad["spec"]["stem"]],
                 ha="center", va="bottom", fontsize=ps.LABEL_PT, color=ps.INK)
        if i % 3 == 0:
            ax_r2.set_ylabel(ps.LABELS["active"], labelpad=1)
    # legends last, once every axis has its final limits, so the coverage test reads the real layout
    legends = []
    for ax_r2, ax_tr, lad, handles in pairs:
        lender = BORROW_LEGEND.get(lad["spec"]["stem"])
        if lender:
            handles = [Line2D([], [], ls="none",
                              label=f"values as in {'abcdefghi'[ORDER.index(lender)]}")]
        at = LEGEND_AT.get(lad["spec"]["stem"])
        if at == "r2":
            holder, leg = ax_r2, ax_r2.legend(
                handles=handles, loc="upper left", bbox_to_anchor=(0.025, 0.97),
                fontsize=ps.TICK_PT, handlelength=0.7, borderaxespad=0.0, labelspacing=0.15,
                handletextpad=0.3)
        elif at:
            holder, leg = ax_tr, ax_tr.legend(handles=handles, loc=at[0], bbox_to_anchor=at[1], ncol=at[2],
                                              columnspacing=0.8,
                                              fontsize=ps.TICK_PT, handlelength=0.9, borderaxespad=0.0,
                                              labelspacing=0.15, handletextpad=0.3)
        else:
            holder, leg = place_legend(ax_tr, handles, alt=ax_r2)
        legends.append((holder, lad, leg))

    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=slots[11], wspace=0.10, width_ratios=[0.8, 1.2])
    ax_fl, ax_end = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])
    opt = optimizer_pair(ax_fl, ax_end, b5)
    ps.panel_letter(ax_fl, "l", x_mm=7.5, y_mm=4.0)
    pos_l, pos_r = ax_fl.get_position(), ax_end.get_position()
    fig.text((pos_l.x0 + pos_r.x1) / 2, pos_l.y1 + 0.006 + 0.012, "optimizer, 3BFF",
             ha="center", va="bottom", fontsize=ps.LABEL_PT, color=ps.INK)
    check_layout(fig, legends)
    return ps.save(fig, "fig_paper_F2_fails"), value_means(lads + rev), opt


def drawn_texts(fig):
    """Every text artist that is actually drawn: titles, labels, legend entries, free text, and the
    tick labels whose tick lies inside its axis' view limits.

    Args:
        fig: the figure.
    Returns:
        list of Text artists.
    """
    out = [t for t in fig.texts if t.get_text().strip()]
    for leg in fig.legends:                      # legends spanning a panel pair live on the figure
        out += list(leg.get_texts())
    for ax in fig.axes:
        out += [t for t in ax.texts if t.get_text().strip()]
        out += [t for t in (ax.title, ax.xaxis.label, ax.yaxis.label) if t.get_text().strip()]
        leg = ax.get_legend()
        if leg is not None:
            out += list(leg.get_texts())
        for axis, (lo, hi) in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lo, hi), max(lo, hi)
            for tk in axis.get_major_ticks():
                if lo <= tk.get_loc() <= hi and tk.label1.get_visible() and tk.label1.get_text():
                    out.append(tk.label1)
    return out


def check_layout(fig, legends):
    """Print every legend that covers data and every pair of drawn texts whose tight bounding boxes
    intersect, plus any text outside the canvas. Read the output; do not trust the eye.

    Args:
        fig: the figure, fully drawn; legends: (traces axes, ladder, legend) tuples.
    Returns:
        number of problems found.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bad = 0
    for ax, lad, leg in legends:
        c = covered(ax, leg)
        if c:
            bad += 1
            print(f"  OVERLAP: legend of {lad['spec']['stem']} covers {c} data points")
    texts = drawn_texts(fig)
    boxes = [(t, t.get_window_extent(rend)) for t in texts]
    W, H = fig.bbox.width, fig.bbox.height
    for t, b in boxes:
        if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H:
            bad += 1
            print(f"  OUTSIDE: '{t.get_text()[:40]}'")
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            if boxes[a][1].overlaps(boxes[b][1]):
                bad += 1
                print(f"  OVERLAP: '{boxes[a][0].get_text()[:30]}' x '{boxes[b][0].get_text()[:30]}'")
    print(f"layout check: {len(boxes)} texts, {len(legends)} legends, {bad} problem(s)")
    return bad


def main():
    """Collect, probe, and draw Extended Data Fig. 4; print each value's means and the optimizer runs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect", action="store_true", help="read every run and write the cache")
    ap.add_argument("--probe", action="store_true", help="print the per-run table from the cache")
    a = ap.parse_args()
    if a.collect:
        collect()
    with open(CACHE, "rb") as fh:
        data = pickle.load(fh)
    if a.probe:
        probe(data)
        return
    with open(B5_CACHE, "rb") as fh:
        b5 = pickle.load(fh)
    _, rows, opt = draw(data, b5)
    print("\n--- mean active units at the floor crossing -> at 75,000 iterations (or the shorter budget) ---")
    for stem, B, rr in rows:
        print(f"  {stem} ({fmt(B)}): " + "; ".join(f"{lab} {f:.0f}->{e:.0f}" for lab, f, e in rr))
    for key in ("floor", "end"):
        print(f"\n--- optimizer comparison, {key} ---")
        for (o, wd, lr), act, r2 in opt[key]:
            print(f"  {o:<6} wd {wd:<5} lr {lr:<5} n={len(act)} active {sorted(int(x) for x in act)}"
                  f"  R2 {[round(v, 3) for v in r2]}")


def fmt(t):
    """An iteration count as the paper prints it ('75,000')."""
    return ps.fmt_count(t)


if __name__ == "__main__":
    main()
