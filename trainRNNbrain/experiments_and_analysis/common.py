#!/usr/bin/env python3
"""
Shared primitives for the analysis scripts in this folder.

WHY THIS EXISTS. The scripts here used to import each other: 14 files did `from plot_drift_curves
import IMG_DIR`, which executes a 625-line analysis module to obtain one directory string, and 9 more
pulled `load_losses`/`logbin` out of `plot_loss_fit`. Alongside that, the same primitives were
re-typed in several places - `logbin` in 3 files, `active_count` in 2, `aicc` in 3, `r2_from_dir` in
3, the stretched-exponential `curve` in 2, and the "keep only the longest budget per size" dedupe in
4. Divergent copies of an analysis primitive are how a project silently reports two different numbers
for the same quantity, which has already happened here more than once.

Everything in this module is either a pure function of its arguments or a loader that reads from
disk. Nothing here plots, and nothing here has module-level side effects, so importing it is free.

CONVENTION FOR THE SCRIPTS THAT USE IT. Each analysis script:
  - states its question, its method, and its output path in the module docstring
  - imports shared primitives from here rather than from a sibling script
  - keeps its own domain loader when its folder layout is specific to it
  - guards execution behind `if __name__ == "__main__": main()`

Self-check: `python common.py`
"""

import os
import re
import json
import glob
import pickle
import numpy as np
from omegaconf import OmegaConf
from scipy.ndimage import median_filter
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "../../img/internal_figures")
DATA_DIR = os.path.join(HERE, "../../data/trained_RNNs")

# The two silence criteria. Both are ALWAYS reported together: they disagree, and which one is used
# has flipped a conclusion in this project (a 60.5% vs 86.0% silent fraction at N=2000 read as a
# penalty "rescue" until both were shown side by side). Quoting one alone is not acceptable.
CRITERIA = [("hard", r"$p_i<10^{-6}$"), ("scalefree", r"$p_i<0.05\,q_{95}(p)$")]

SILENT_HARD = 1e-6          # absolute participation below which a unit counts as silent
SILENT_REL = 0.05           # ... or this fraction of the 95th percentile, for the scale-free rule
SILENT_FLIPFLOP = 4e-2      # task-calibrated absolute threshold for the n-bit flip-flop, from
                            # Otsu's method on log participation (flipflop_hard_threshold.py).
                            # SILENT_HARD=1e-6 is a CDDM number and sits BELOW both flip-flop
                            # modes, so it reports ~0% silence on this task - do not reuse it here.


# --------------------------------------------------------------------------------------------
# participation and silence
# --------------------------------------------------------------------------------------------

def participation(fr):
    """Per-unit participation p_i = std(r_i) + q_0.9(|r_i|), reduced over every axis but the first.

    Both terms are needed: the standard deviation alone calls a unit with a large constant offset
    silent, and the quantile alone calls a unit with a small but genuine oscillation silent.

    Args:
        fr: firing rates with units on axis 0 and any number of trailing axes - (N, T) for a single
            flattened trial set, (N, T, B) for time x condition. All trailing axes are reduced.
    Returns:
        (N,) float array of participation values, >= 0.
    """
    fr = np.asarray(fr, dtype=float)
    axes = tuple(range(1, fr.ndim))
    return fr.std(axis=axes) + np.quantile(np.abs(fr), 0.9, axis=axes)


def active_count(p, criterion):
    """Number of active units in a participation vector under the named criterion.

    ⚠️ THE ABSOLUTE THRESHOLD IS NOT TASK-PORTABLE. `SILENT_HARD = 1e-6` was calibrated on CDDM,
    whose silent mode sits at exactly 0. On the n-bit flip-flop the silent mode sits at ~1e-3, so
    1e-6 falls below BOTH modes and reports ~0% silence where the scale-free rule reports ~80%. Pass
    a float to use a threshold calibrated for the task in hand (see flipflop_hard_threshold.py, which
    derives 4e-2 for the flip-flop by Otsu's method on log participation).

    Args:
        p: (N,) participation values;
        criterion: "hard" (p >= SILENT_HARD, the CDDM-calibrated absolute floor), "scalefree"
            (p >= 5% of the 95th percentile of p), or a FLOAT giving an explicit absolute threshold.
    Returns:
        int count of active units.
    """
    p = np.asarray(p, dtype=float)
    if isinstance(criterion, (int, float)) and not isinstance(criterion, bool):
        return int((p >= float(criterion)).sum())
    if criterion == "hard":
        return int((p >= SILENT_HARD).sum())
    return int(active_mask(p).sum())


def active_mask(p, rel=SILENT_REL):
    """Which units are active under the scale-free rule: p >= rel x q95(p).

    This is the one implementation of the rule; `active_count(p, "scalefree")` counts its True
    entries. It is computed in p's own dtype (a float32 participation trace stays float32), as every
    inline copy of the rule it replaced was, so moving a script onto it changes no count.
    `active_count` casts to float64 first; on 23,700 stored probes of 15 CDDM and flip-flop runs the
    two dtypes gave the same count at every probe.

    Args:
        p: (N,) participation values, or any non-negative per-unit measure the scale-free bar is put
            on (the c4 scripts put it on per-unit gradients);
        rel: the bar as a fraction of the 95th percentile, SILENT_REL (5%) for the project's rule.
    Returns:
        (N,) bool array, True for an active unit.
    """
    p = np.asarray(p)
    return p >= rel * np.quantile(p, 0.95)


def hhi(v):
    """Herfindahl concentration of a non-negative vector: sum of squared shares.

    Args:
        v: (N,) non-negative values, e.g. per-unit participation. Non-finite entries are dropped.
    Returns:
        float in [1/N, 1] - 1/N when perfectly even, larger when concentrated in fewer units;
        nan if the total is non-positive or nothing finite remains.
    """
    p = np.asarray(v, dtype=float)
    p = p[np.isfinite(p)]
    tot = p.sum()
    if p.size == 0 or tot <= 0:
        return float("nan")
    s = p / tot
    return float(np.sum(s * s))


def otsu_threshold(p, nbins=200):
    """Data-derived silence threshold: the antimode of the log-participation distribution.

    WHY THIS EXISTS. The absolute rule `p < 1e-6` is not task-portable. It was calibrated on CDDM,
    whose silent mode sits at exactly 0, and on the n-bit flip-flop it reports ~0% silence because
    that task's silent mode sits at ~2e-4 - three to four orders below its own active mode, but far
    above 1e-6. Both tasks are cleanly BIMODAL in log participation; only the location of the silent
    mode differs. Otsu's method finds the split between the two modes by maximising between-class
    variance, so it adapts to the task's dynamic range without a hand-picked constant.

    Units with p == 0 are definitionally silent and are excluded from the search (log undefined)
    rather than allowed to drag the threshold.

    Args:
        p: (N,) participation values; nbins: histogram resolution in log10 units.
    Returns:
        float threshold in the ORIGINAL units (not log), or nan if the distribution has too few
        distinct non-zero values to split.
    """
    p = np.asarray(p, dtype=float)
    v = np.log10(p[np.isfinite(p) & (p > 0)])
    if v.size < 20 or np.ptp(v) < 1e-9:
        return float("nan")
    hist, edges = np.histogram(v, bins=nbins)
    w = hist.astype(float) / hist.sum()
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(w)
    w1 = 1.0 - w0
    m0 = np.cumsum(w * centres) / np.maximum(w0, 1e-12)
    m1 = (np.sum(w * centres) - np.cumsum(w * centres)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(10 ** centres[int(np.argmax(between))])


def participation_ratio(v):
    """Effective number of participating units: PR = (sum p)^2 / sum p^2.

    ⚠️ THIS IS EXACTLY 1/hhi(v) - the two are algebraically identical, since
    HHI = sum (p_i / sum p)^2 = (sum p^2)/(sum p)^2. Provided under both names because they are read
    differently: HHI is "how concentrated" (0..1), PR is "how many units effectively participate"
    (1..N). Do not report both as if they were independent evidence.

    WHY IT IS NEEDED. The thresholded active-unit count M SATURATES: under frm and frm+rws every unit
    clears any silence threshold, so M/N = 0.99-1.00 and the measure cannot discriminate those two
    conditions at all, nor fit an exponent. PR is graded - measured at the same read-out it gives
    0.97 (frm) vs 0.93 (both) vs 0.39 (none) as a fraction of N - because it asks how EVENLY activity
    is spread rather than how many units clear a cut.

    Args:
        v: (N,) non-negative participation values; non-finite entries dropped.
    Returns:
        float in [1, N], or nan if nothing finite remains or the total is non-positive.
    """
    h = hhi(v)
    return float("nan") if not np.isfinite(h) or h <= 0 else 1.0 / h


# --------------------------------------------------------------------------------------------
# loss curves: smoothing and read-out times
# --------------------------------------------------------------------------------------------

def smooth_loss(L, t):
    """Training loss averaged over a +-2% window ending at iteration t (the raw loss is very noisy).

    Args:
        L: per-iteration loss array; t: iteration to read at.
    Returns:
        float mean over [t - max(0.02t, 50), t).
    """
    w = max(int(0.02 * t), 50)
    return float(L[max(t - w, 0):t].mean())


def T_at_loss(L, target):
    """First iteration at which the smoothed loss reaches `target`.

    Args:
        L: per-iteration loss array; target: loss level to reach.
    Returns:
        int iteration, or None if the run never reaches the level.
    """
    ts = np.arange(2000, len(L), 200)
    sm = np.array([smooth_loss(L, t) for t in ts])
    hit = ts[sm <= target]
    return int(hit[0]) if len(hit) else None


def stable_crossing(L, thr, window=201, base=0):
    """Last index at which the smoothed loss is still above `thr` - a STABLE crossing, not the first.

    "Stably below" rather than "first time below" because the loss is not monotone and a
    first-crossing rule fires on a transient dip. Returns None when the run never gets below the
    threshold, or is still above it at the end, which would put the crossing outside the run.

    Args:
        L: loss array, indexed either per probe or per iteration - the caller decides what the
            returned index means; thr: threshold; window: centred running-mean width, in the same
            units as L's index; base: 0 to return a 0-based index into L, 1 for a 1-based iteration
            count (the two call sites historically differed only in this).
    Returns:
        int index, or None.
    """
    L = np.asarray(L, dtype=float)
    h = window // 2
    if len(L) < window:
        return None
    s = np.convolve(L, np.ones(window) / window, mode="valid")
    idx = np.arange(h + base, len(L) - h + base)
    above = idx[s > thr]
    if not len(above) or above[-1] >= idx[-1]:
        return None
    return int(above[-1])


# --------------------------------------------------------------------------------------------
# fitting helpers
# --------------------------------------------------------------------------------------------

def logbin(t, y, nbins=60, t_min=None, centre="median"):
    """Median-reduce (t, y) into log-spaced bins, so a fit is not dominated by late iterations.

    A loss trace has tens of thousands of probes near its floor and only a few hundred in the two
    decades above it; fitting the raw trace weights the floor overwhelmingly and the approach barely
    at all. Log-binning equalises them.

    Args:
        t, y: equal-length arrays with t > 0; nbins: number of log-spaced bins;
        t_min: left edge of the binning, default t[0];
        centre: "median" to place each bin at the median t of its members, "geometric" to place it at
            the geometric mean of the bin edges (used where bins must be independent of the sampling).
    Returns:
        (tb, yb) bin-centre and bin-median arrays, with bins holding <= 2 points dropped.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    lo = t[0] if t_min is None else t_min
    edges = np.logspace(np.log10(lo), np.log10(t[-1]), nbins + 1)
    idx = np.digitize(t, edges) - 1
    tb, yb = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() > 2:
            tb.append(np.sqrt(edges[b] * edges[b + 1]) if centre == "geometric" else np.median(t[m]))
            yb.append(np.median(y[m]))
    return np.array(tb), np.array(yb)


def stretched(t, Li, A, tau, beta):
    """Stretched-exponential approach to a floor: L_inf + A*exp(-(t/tau)^beta).

    beta < 1 means a broad spectrum of relaxation timescales rather than a single rate.

    ⚠️ A and tau are NOT individually identified once tau falls below the fit's start iteration: the
    form degenerates toward a power law and A becomes an extrapolated t=0 amplitude the data never
    constrains. Compare fits through Li and through excess-decay times inside the fitted range
    (see `excess_time`), never through A or tau alone.

    Args:
        t: iterations; Li: floor; A: amplitude; tau: timescale; beta: stretch exponent.
    Returns:
        array of predicted losses, same shape as t.
    """
    return Li + A * np.exp(-np.power(np.clip(t / tau, 1e-12, None), beta))


def excess_time(A, tau, beta, frac, t0):
    """Iteration at which the excess over the floor has fallen to `frac` of its value at t0.

    This is the identified alternative to quoting tau: it is an interpolation inside the fitted
    range, so it does not inherit the A/tau degeneracy documented in `stretched`.

    Args:
        A, tau, beta: stretched-exponential parameters; frac: target fraction of the excess at t0
            (0.5 for a half-life, 0.1 to remove 90%); t0: reference iteration.
    Returns:
        float iteration, or nan if beta or tau is degenerate.
    """
    if not (beta > 1e-6 and tau > 0):
        return float("nan")
    lhs = np.power(t0 / tau, beta) - np.log(frac)
    return float(tau * np.power(lhs, 1.0 / beta))


def aicc(rss, n, k):
    """Akaike information criterion with the small-sample correction.

    Args:
        rss: residual sum of squares; n: number of data points; k: number of fitted parameters.
    Returns:
        float AICc. Only DIFFERENCES between models fitted to the SAME data are meaningful.
    """
    return n * np.log(rss / n) + 2 * k + (2 * k * (k + 1)) / max(n - k - 1, 1e-9)


# --------------------------------------------------------------------------------------------
# THE READ-OUT: where a run's NOISY loss first reaches 1.05x its own fitted floor
# --------------------------------------------------------------------------------------------
#
# The rule since 2026-10-12 (Pavel): performance and convergence are read on the loss the network
# was trained on - the masked MSE under its training noise - never on the noise-free probe. On the
# standard (bias-0) flip-flop the noise-free loss bottoms out near 10k iterations and climbs back
# 2-4x while the noisy loss keeps falling; these networks come to depend on their noise, so the
# noise-free loss is the wrong yardstick.
#
# THE SERIES COSTS NOTHING EXTRA. The trainer computes the noisy loss at every step, on a fresh
# batch with fresh noise, and saves it (TrainLosses.json). Smoothing it averages over noise draws
# and batches, so no evaluation is added to training.
#
# THE SMOOTHER (Pavel, 2026-10-15) is a Hampel-filtered mean over the iterations [t/1.1, 1.1t]:
# points more than 3 MADs (median absolute deviations) from the window's median are dropped and
# the rest averaged. Outlier-robust because DMTS spikes back to chance for brief episodes
# (1,300-3,900 spiking iterations per run) and a plain mean is dragged up by them. The window grows
# with t so every network gets the same RELATIVE smoothing: the 2,001-iteration running median used
# until then was a large share of training for a fast run and a sliver of it for a slow one. The
# floor is fitted to this same curve (it was fitted to log-binned medians), so floor and crossing
# are one statistic. (A robust soft-L1 fit loss was tried and dropped: from the shared start point it
# converged to a degenerate floor ~0, beta on its lower bound, on 231 CDDM and flip-flop runs.)
#
# THE CROSSING IS SUSTAINED (Pavel, 2026-10-15): the read-out is the first t after which the curve
# stays at or below (1+delta) x floor through (1+HOLD_FRAC) x t. A first crossing takes the first
# noise dip under the bar, so the noisier a network's loss the earlier it was read - a bias across
# sizes and hyperparameters, not just scatter. It is read at every FLOOR_DELTAS margin; 1.05x is the
# tightest margin every network can be held to (Pavel), the others show whether a comparison depends
# on where it is read.
#
# BURSTS ARE COUNTED, AND MASKED OUT OF THE READ-OUT. A burst is a stretch where the 51-iteration
# running median of the raw loss sits above BURST_RATIO x a wide reference ([t/1.5, 1.5t] Hampel
# curve, wide so that a burst of a few thousand iterations cannot pull its own reference up). Its
# iterations, padded on each side by max(51, half the burst's length) because the 51-iteration median
# blurs its edges and long bursts ramp, are dropped before the read-out curve is computed:
# DMTS bursts last thousands of iterations and recur every few thousand, so at t ~ 20-30k they
# fill a quarter of the [t/1.1, 1.1t] window, the Hampel cut keeps them, and the unmasked floor
# came out up to 1.6x the between-burst level. How often a training method, size or hyperparameter
# bursts is a property of it, and is reported per cell (burst_stats.py). A run with more than
# MAX_BURST_SHARE of its iterations after the fit start inside bursts gets no read-out: there is no
# stable stretch to read (DMTS, the burstiest task that learns, sits at 9% median, 23% 90th pct).
#
# VALIDATED on synthetic traces with a known floor and crossing (trajectory 2026-10-15): floor within
# 1% and the 1.05x crossing within 6% at bulk CV (SD/mean of the per-iteration loss) 0.03-0.3, with
# 1% of iterations x10 and planted 3-20x bursts; the old rule read 2-19% early on the same traces.
#
# A TASK THAT STALLS (DMTS sits at the constant-output solution for 200 to 155,000 iterations
# before it escapes) is fitted from its escape, not from iteration 0: a stretched exponential
# cannot describe a plateau followed by a drop. The escape is where the running median first
# falls below half the level of the first 50 iterations; for the flip-flop and CDDM that is within
# the first few hundred iterations, and the fit is unchanged.
#
# THE GUARDS ARE BLIND TO EPISODES (Pavel, 2026-10-12). After reaching its floor DMTS has bursts:
# the loss climbs 2-10x, sometimes back to chance, for a few thousand iterations, then returns.
# Judged by its final value, a run was excluded whenever training happened to stop inside one (4 of
# 6 DMTS controls). So the end of a run is the MEDIAN of its last 10% of iterations, and a run is
# excluded when that end sits more than 1.5x above its fitted floor - the loss really climbed back
# and stayed. (A second test, floor against the run's best long-run level, was tried and dropped:
# DMTS sits flat between bursts at ~7e-5 while its fitted floor, ~1e-4, averages troughs and
# bursts, so the test rejected 7 of 8 runs that had plainly reached their floor.)

HAMPEL_SPAN = 1.1       # the read-out curve at t averages the iterations [t/1.1, 1.1t]
HAMPEL_K = 3.0          # a point more than this many MADs from its window median is dropped
GRID_RATIO = 1.01       # the curve is computed on t-grid points 1% apart, then interpolated
HOLD_FRAC = 0.2         # a crossing at t counts once the curve stays under the bar through 1.2t
FLOOR_DELTA = 0.05      # read at 1.05x the fitted floor (1.07x before 2026-10-12)
FLOOR_DELTAS = (0.05, 0.10, 0.20)   # every margin the read-out is also taken at
BURST_RATIO = 2.0       # a burst: the short running median above this x the wide reference
BURST_SPAN = 1.5        # the burst reference at t averages [t/1.5, 1.5t]
BURST_MEDIAN = 51       # iterations in the short running median, so a lone bad batch is no burst
BURST_GAP = 200         # bursts closer than this many iterations are one burst
MAX_BURST_SHARE = 0.5   # more than this share of post-fit-start iterations in bursts: no read-out
MAX_LATE_RISE = 1.5     # end of the run (median of its last 10%) over the fitted floor; above it, no
                        # read-out
END_FRAC = 0.10         # the "end of the run" is its last 10% of iterations
FIT_START = 2000        # the floor is fitted from this iteration, or 200 after the escape if later
FLOOR_BOUND = 1e-6      # the fit's lower bound on the floor; a fit pinned there is no floor


def warm_start_chain(run_dir):
    """A run and every run it was warm-started from, oldest first.

    Follows `paths.init_from` in each saved config. A parent that is not at its recorded path (a
    local copy) is looked for as a nested `parent_*` folder, the way the top-up sync stores it.

    Args:
        run_dir: a trained-network folder.
    Returns:
        list of run folders, the original run first and `run_dir` last.
    """
    chain, d = [], run_dir.rstrip("/")
    while d and d not in chain:
        chain.append(d)
        cfgs = sorted(glob.glob(os.path.join(d, "*_config.yaml")))
        parent = None
        if cfgs:
            paths = OmegaConf.load(cfgs[0]).get("paths", {}) or {}
            parent = paths.get("init_from", None)
        if parent and not os.path.isdir(parent):
            nested = sorted(glob.glob(os.path.join(d, "parent_*", "")))
            parent = nested[0].rstrip("/") if nested else None
        d = parent.rstrip("/") if parent else None
    return chain[::-1]


def noisy_loss_series(run_dir):
    """The per-iteration noisy training loss of a run, its whole warm-start chain joined.

    Args:
        run_dir: a trained-network folder.
    Returns:
        (iterations, loss) float arrays, iterations absolute (1-based), or (None, None).
    """
    its, Ls = [], []
    done = 0
    for d in warm_start_chain(run_dir):
        tl = sorted(glob.glob(os.path.join(d, "*TrainLosses.json")))
        if not tl:
            return None, None
        with open(tl[0]) as fh:
            L = np.asarray(json.load(fh).get("train_losses", []), float)
        its.append(done + np.arange(1, len(L) + 1, dtype=float))
        Ls.append(L)
        done += len(L)
    if not Ls:
        return None, None
    return np.concatenate(its), np.concatenate(Ls)


def hampel_curve(it, L, span, ratio=GRID_RATIO, k=HAMPEL_K, min_n=11):
    """Outlier-trimmed mean of L over the iterations [t/span, t*span], on a log-spaced grid of t.

    In each window, points more than k scaled MADs (1.4826 x median absolute deviation, which equals
    the SD for Gaussian noise) from the window median are dropped and the rest averaged.

    Args:
        it: (n,) increasing iterations; L: (n,) loss at those iterations;
        span: window half-width as a ratio (1.1 averages [t/1.1, 1.1t]);
        ratio: spacing of the grid points; k: the outlier cut, in MADs;
        min_n: grid points whose window holds fewer samples are skipped.
    Returns:
        (tg, yg): grid iterations and the trimmed mean at each, both (m,) float arrays.
    """
    tg = np.exp(np.arange(np.log(it[0]), np.log(it[-1]), np.log(ratio)))
    a = np.searchsorted(it, tg / span, "left")
    b = np.searchsorted(it, tg * span, "right")
    keep = b - a >= min_n
    tg, a, b = tg[keep], a[keep], b[keep]
    yg = np.empty(len(tg))
    for j in range(len(tg)):
        x = L[a[j]:b[j]]
        med = np.median(x)
        dev = np.abs(x - med)
        yg[j] = x[dev <= k * 1.4826 * np.median(dev)].mean()
    return tg, yg


def sustained_crossing(it, y, bar, t_from, hold=HOLD_FRAC):
    """First iteration t >= t_from after which y stays <= bar through (1+hold) x t.

    Args:
        it: (n,) increasing iterations; y: (n,) smoothed loss; bar: the threshold;
        t_from: earliest iteration a crossing may be read at; hold: the hold window as a fraction of t.
    Returns:
        the iteration as a float; nan if no crossing holds, or if every hold window runs past the end.
    """
    above = y > bar
    nxt = np.minimum.accumulate(np.where(above, np.arange(len(y)), len(y))[::-1])[::-1]
    t_next = np.append(it, np.inf)[nxt]          # iteration of the next above-bar sample, or inf
    ok = (it >= t_from) & ~above & (t_next > (1 + hold) * it) & ((1 + hold) * it <= it[-1])
    hit = np.flatnonzero(ok)
    return float(it[hit[0]]) if len(hit) else float("nan")


def find_bursts(it, L, ref, t_from):
    """Stretches where the loss climbs above BURST_RATIO x its reference level.

    Args:
        it: (n,) increasing iterations; L: (n,) raw noisy loss; ref: (n,) reference level per
        iteration (the wide Hampel curve); t_from: bursts are only looked for from this iteration.
    Returns:
        list of (first iteration, last iteration, peak ratio of short running median to ref).
    """
    r = median_filter(L, size=BURST_MEDIAN, mode="nearest") / ref
    d = np.diff(np.r_[0, ((r > BURST_RATIO) & (it >= t_from)).astype(int), 0])
    segs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))   # [start, stop) indices
    merged = []
    for s, e in segs:
        if merged and it[s] - it[merged[-1][1] - 1] < BURST_GAP:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(float(it[s]), float(it[e - 1]), float(r[s:e].max())) for s, e in merged]


def noisy_floor(it, L, delta=FLOOR_DELTA):
    """Where a run's smoothed noisy loss reaches (1+delta) x its own fitted floor and stays there.

    Args:
        it, L: the per-iteration noisy loss and its iterations (noisy_loss_series).
        delta: the margin `t_star` is read at; every FLOOR_DELTAS margin is read too.
    Returns:
        dict: t_star (nan when there is no read-out), t_stars ({margin: crossing} for FLOOR_DELTAS and
        delta, nan where none holds), floor, floor_kind ("fit", or "smoothed_min" when the fit's beta
        sits on its upper bound and the floor falls back to the minimum of the smoothed curve),
        escape, late_rise (median of the smoothed curve over the run's last 10%, over the floor),
        reason (None when the read-out is usable), `smoothed` (the Hampel curve the crossing is read
        on, one value per iteration of `it`), `fit` (the stretched-exponential parameters, fitted in
        t - escape), `bursts` (find_bursts list) and `burst_from` (the iteration bursts are counted
        from).
    """
    it, L = np.asarray(it, float), np.asarray(L, float)
    ok = np.isfinite(L)
    it, L = it[ok], L[ok]
    deltas = sorted(set(FLOOR_DELTAS) | {delta})
    out = dict(t_star=float("nan"), t_stars={d: float("nan") for d in deltas}, floor=None,
               floor_kind=None, escape=float("nan"), late_rise=float("nan"), reason=None,
               bursts=[], burst_from=float("nan"))
    if len(L) < 2 * FIT_START:
        out["reason"] = "series too short"
        return out
    tg, yg = hampel_curve(it, L, HAMPEL_SPAN)
    sm = np.interp(np.log(it), np.log(tg), yg)
    # the starting level is the first 50 iterations: CDDM halves its loss within ~100, so a later
    # window already sits near the bottom and the escape is never found; DMTS's stall (~0.07) is
    # still above half of its first-50 level (~0.09), so its escape is caught where it happens
    start = np.median(L[:50])
    esc = np.flatnonzero(sm < 0.5 * start)
    t0 = float(it[esc[0]]) if len(esc) else float("nan")
    fit_from = max(FIT_START, t0 + 200) if len(esc) else FIT_START
    tr, yr = hampel_curve(it, L, BURST_SPAN, ratio=1.05)
    out["bursts"] = find_bursts(it, L, np.interp(np.log(it), np.log(tr), yr), fit_from)
    out["burst_from"] = float(fit_from)
    if out["bursts"]:                            # the read-out curve without the burst iterations
        calm = np.ones(len(it), bool)
        for b0, b1, _ in out["bursts"]:
            pad = max(BURST_MEDIAN, 0.5 * (b1 - b0))
            calm &= (it < b0 - pad) | (it > b1 + pad)
        tg, yg = hampel_curve(it[calm], L[calm], HAMPEL_SPAN)
        sm = np.interp(np.log(it), np.log(tg), yg)
    out["smoothed"] = sm
    share = sum(b1 - b0 + 1 for b0, b1, _ in out["bursts"]) / max(it[-1] - fit_from + 1, 1)
    if share > MAX_BURST_SHARE:
        out["reason"] = f"bursts cover {share:.0%} of training"
        return out
    if not len(esc):
        out["reason"] = "never leaves its starting level"
        return out
    out["escape"] = t0
    g = tg >= fit_from
    tb, yb = tg[g] - t0, yg[g]
    if len(tb) < 8:
        out["reason"] = "too little training after the escape"
        return out
    lo, hi = [FLOOR_BOUND, 1e-6, 1e2, .05], [1., 1e3, 1e9, 3.]
    p0 = np.clip([yb.min() * .9, float(yb.max()), 2e4, .4], np.array(lo) * 1.01, np.array(hi) * .99)
    try:
        fit = least_squares(lambda p: np.log(np.clip(stretched(tb, *p), 1e-12, None)) - np.log(yb),
                            p0, bounds=(lo, hi), max_nfev=20000).x
    except Exception:
        out["reason"] = "floor fit failed"
        return out
    floor = float(fit[0])
    out["floor"], out["fit"], out["floor_kind"] = floor, [float(v) for v in fit], "fit"
    end_level = float(np.median(sm[it >= it[-1] - END_FRAC * (it[-1] - it[0])]))
    out["late_rise"] = end_level / floor
    if floor <= FLOOR_BOUND * 1.01:
        out["reason"] = "fit pinned at its lower bound"
        return out
    # A stretch exponent on its UPPER bound is no fitted floor (Pavel, 2026-10-13). On the flip-flop
    # such a fit puts the "floor" at a level the loss later falls through (it ends at 0.73-0.92x of
    # it), so the crossing fired early (27k instead of ~52k on one claim-1 run). These runs have
    # converged - two claim-1 flip-flop runs sit at ~0.034 from 52-60k to 300k - and it is the fit
    # that fails on a fast drop followed by a flat plateau. So (Pavel, 2026-10-14) such a run is read
    # against the MINIMUM of its smoothed curve over the fit window instead, and the crossing is
    # where the smoothed curve reaches and holds 1.05x that minimum; floor_kind records which floor was
    # used. tau on its LOWER bound is not checked on purpose: on CDDM and DMTS it only means the fast
    # decay is over before the fit window opens, and those floors hold (CDDM ends at 1.00-1.01x).
    # Guarding every bound dropped 64 of 123 runs; the beta guard touches 6, the suspicious ones.
    if fit[3] >= hi[3] * 0.99:
        floor = float(yb.min())
        out["floor"], out["floor_kind"] = floor, "smoothed_min"
        out["late_rise"] = end_level / floor
    if out["late_rise"] > MAX_LATE_RISE:
        out["reason"] = f"loss ends {out['late_rise']:.1f}x above its floor"
        return out
    out["t_stars"] = {d: sustained_crossing(it, sm, (1.0 + d) * floor, fit_from) for d in deltas}
    out["t_star"] = out["t_stars"][delta]
    if np.isnan(out["t_star"]):
        out["reason"] = "never stays under the bar"
    return out


# --------------------------------------------------------------------------------------------
# READING A RUN AT ITS FLOOR: the active count, the kept weights and the loss there
# --------------------------------------------------------------------------------------------
#
# Every script that reads a network at its floor goes through `read_run` (one run) or
# `runs_with_loss` + `at_own_floor` (a cell). These lived in fig_paper_F1 (runs_with_loss,
# floor_crossing, at_own_floor), fig_knobs_at_floor (the R^2 estimate) and heldout_r2
# (snapshot_at) until 2026-10-15, and other scripts imported them from there. Held-out R^2 itself
# needs torch and stays in heldout_r2 (`heldout_r2.floor_r2`).

# kept weight snapshots, by <sweep>/<cell>/s<seed>/it<absolute iteration>/
SNAP_DIR = os.path.join(DATA_DIR, "..", "c1_snapshots")
SNAP_MAX_LAG = 1000     # a snapshot more than this many iterations after the crossing is not "at" it


def load_run(run_dir):
    """One run's noisy loss series and its active-unit count at every participation probe.

    Args:
        run_dir: a trained-network folder holding *ParticipationTrace.pkl and *TrainLosses.json.
    Returns:
        dict with run_dir, name (the folder name), it_loss/loss (noisy_loss_series, the warm-start
        chain joined), it_act (probe iterations) and active (float active count per probe, under
        active_mask); None when the trace is missing or unreadable, the loss is missing or empty, or
        the trace holds no 2-D participation array.
    """
    tp = glob.glob(os.path.join(run_dir, "*ParticipationTrace.pkl"))
    if not tp:
        return None
    try:
        with open(tp[0], "rb") as fh:
            tr = pickle.load(fh)
    except Exception:
        return None
    it_l, L = noisy_loss_series(run_dir)
    if L is None or not len(L):
        return None
    P = np.asarray(tr["participation"])
    it_a = np.asarray(tr["participation_iters"], float)
    if P.ndim != 2 or not len(it_a):
        return None
    act = np.array([active_mask(p).sum() for p in P], float)
    return dict(it_loss=it_l, loss=L, it_act=it_a, active=act, name=os.path.basename(run_dir),
                run_dir=run_dir)


def runs_with_loss(pattern, min_r2=None):
    """Every non-diverged run under a glob, with its NOISY loss series and active-unit series.

    Args:
        pattern: glob matching run folders; min_r2: drop runs scoring below this, or None.
    Returns:
        list of load_run dicts (it_loss/loss, it_act/active, name, run_dir), one per run folder
        whose name does not start with "nan", whose folder-name R^2 is at least min_r2, and which
        load_run can read.
    """
    out = []
    for d in sorted(glob.glob(os.path.join(pattern, "*"))):
        if not os.path.isdir(d):
            continue
        head = os.path.basename(d).split("_")[0]
        if head == "nan":
            continue
        if min_r2 is not None:
            try:
                if float(head) < min_r2:
                    continue
            except ValueError:
                continue
        r = load_run(d)
        if r is not None:
            out.append(r)
    return out


def floor_crossing(it_l, L, delta=None):
    """Where one run's noisy loss first reaches (1+delta) x its OWN fitted floor (common.noisy_floor).

    Args:
        it_l, L: the run's per-iteration noisy loss; delta: margin, FLOOR_DELTA when None.
    Returns:
        (crossing iteration or nan, reason string or None when the crossing is usable).
    """
    r = noisy_floor(it_l, L, FLOOR_DELTA if delta is None else delta)
    return r["t_star"], r["reason"]


def at_own_floor(cells, delta=None, why=None):
    """Active units per size, read where each run's own loss first reaches (1+delta) x its OWN floor.

    The project's read-out (common.noisy_floor): each run's floor is fitted on its own noisy loss
    and the count is taken AT the crossing. A run with no usable crossing - fit failed or pinned at
    its bound, loss never reaching the bar, or a loss that climbs back more than 1.5x after its
    minimum - is DROPPED and counted, never clamped to its last probe.

    Args:
        cells: dict N -> list of run dicts from runs_with_loss;
        delta: the excess margin, FLOOR_DELTA (0.05) when None;
        why: optional list; (N, run name, reason) is appended for every dropped run.
    Returns:
        (dict N -> (counts array, mean crossing), n_dropped).
    """
    out, dropped = {}, 0
    for N, v in cells.items():
        counts, crossings = [], []
        for r in v:
            T, reason = floor_crossing(r["it_loss"], r["loss"], delta)
            if reason is not None:
                dropped += 1
                if why is not None:
                    why.append((N, r["name"], reason))
                continue
            counts.append(r["active"][int(np.argmin(np.abs(np.asarray(r["it_act"], float) - T)))])
            crossings.append(T)
        if counts:
            out[N] = (np.array(counts, float), float(np.mean(crossings)))
    return out, dropped


def snapshot_at(run_dir, t):
    """The kept weight snapshot at or just after iteration t, if the run (or a run it was
    warm-started from) kept snapshots.

    A continuation's floor can fall inside its parent, whose snapshots sit under the PARENT's
    sweep, cell and seed, so every run of the warm-start chain is searched.

    Args:
        run_dir: a run folder (its name carries `seed=<seed>`); t: an absolute iteration.
    Returns:
        the snapshot folder, or None when there is none within SNAP_MAX_LAG after t.
    """
    snaps = []
    for d in warm_start_chain(run_dir):
        cell = os.path.dirname(d.rstrip("/"))
        seed = re.search(r"seed=(\d+)", os.path.basename(d.rstrip("/")))
        if seed:
            snaps += glob.glob(os.path.join(SNAP_DIR, os.path.basename(os.path.dirname(cell)),
                                            os.path.basename(cell), f"s{seed.group(1)}", "it*"))
    later = sorted((int(os.path.basename(d)[2:]), d) for d in snaps if int(os.path.basename(d)[2:]) >= t)
    return later[0][1] if later and later[0][0] - t <= SNAP_MAX_LAG else None


def r2_at_floor_estimate(L_star, L_end, r2_end):
    """R^2 at the floor crossing, estimated from the end of the run when no weights were kept there.

    R2_floor = 1 - (L(t*) / L_end) (1 - R2_end): the unexplained variance is scaled by the ratio of
    the noisy loss at the crossing to the loss at the end. Against held-out R^2 measured on weights
    saved at the crossing (nine claim-1 networks, trajectory 2026-10-13): eight within 0.006, one CDDM
    network off by 0.0115.

    Args:
        L_star: smoothed noisy loss at the crossing (read_run's L_star); L_end: the end-of-run loss
            (read_run's L_end); r2_end: R^2 of the final weights - held-out R^2
            (heldout_r2.floor_r2) for every number the project reports.
    Returns:
        float estimated R^2 at the crossing.
    """
    return 1.0 - (L_star / L_end) * (1.0 - r2_end)


def read_run(run_dir, delta=FLOOR_DELTA, run=None):
    """THE read-out of one run: where its noisy loss reaches (1+delta) x its own floor, and what is there.

    Args:
        run_dir: a trained-network folder;
        delta: the margin t_star is read at (noisy_floor's `delta`);
        run: the run's load_run / runs_with_loss dict, to skip reading its trace and loss again;
            None reads them.
    Returns:
        dict holding every key noisy_floor returns (t_star, t_stars, floor, floor_kind, escape,
        late_rise, reason, bursts, burst_from, and `smoothed` / `fit` where it got that far), plus
          run_dir, name;
          it_loss, loss: the noisy loss series (None when the run has none; reason then says so);
          it_act, active_series: probe iterations and the active count at each (None without a
              usable participation trace);
          active, probe_iter: the active count (int) at the probe nearest t_star and that probe's
              iteration; None when there is no read-out (reason is not None) or no trace;
          covered: True when the probes span t_star (it_act[0] <= t_star <= it_act[-1]). Where they
              do not, `active` is an edge probe, not the crossing; callers that need the crossing
              drop such a run;
          active_end, it_end: the count at the last probe and its iteration (None without a trace);
          L_star: the smoothed noisy loss at the first iteration >= t_star, and L_end: the median
              raw noisy loss over the last END_FRAC of iterations - the inputs of
              r2_at_floor_estimate (None without a read-out);
          snapshot: the kept weight snapshot at the crossing (snapshot_at), or None.
    """
    rec = run if run is not None else load_run(run_dir)
    it, L = (rec["it_loss"], rec["loss"]) if rec is not None else noisy_loss_series(run_dir)
    if L is not None:
        out = noisy_floor(it, L, delta)
    else:
        out = dict(t_star=float("nan"), t_stars={}, floor=None, floor_kind=None, escape=float("nan"),
                   late_rise=float("nan"), reason="no noisy loss", bursts=[], burst_from=float("nan"))
    out.update(run_dir=run_dir, name=os.path.basename(run_dir.rstrip("/")), it_loss=it, loss=L,
               it_act=None, active_series=None, active=None, probe_iter=None, covered=False,
               active_end=None, it_end=None, L_star=None, L_end=None, snapshot=None)
    if rec is not None:
        it_a, act = np.asarray(rec["it_act"], float), np.asarray(rec["active"], float)
        out.update(it_act=it_a, active_series=act, active_end=int(act[-1]), it_end=float(it_a[-1]))
    if out["reason"] is not None:
        return out
    T = out["t_star"]
    if rec is not None:
        j = int(np.argmin(np.abs(it_a - T)))
        out.update(active=int(act[j]), probe_iter=float(it_a[j]), covered=bool(it_a[0] <= T <= it_a[-1]))
    # indexed on the series as given, as fig_knobs_at_floor and fig_supp_task_demand did
    it, L = np.asarray(it, float), np.asarray(L, float)
    out["L_star"] = float(out["smoothed"][np.flatnonzero(it >= T)[0]])
    out["L_end"] = float(np.median(L[it >= it[-1] - END_FRAC * (it[-1] - it[0])]))
    out["snapshot"] = snapshot_at(run_dir, T)
    return out


# --------------------------------------------------------------------------------------------
# sweep loading
# --------------------------------------------------------------------------------------------

def keep_longest_budget(files):
    """Drop every run of a cell except those at that cell's longest iteration budget.

    A cell routinely exists at more than one budget because a sweep was extended (N=100 at 50k and
    again at 200k; flip-flop k-cells at 300k and 1M). Pooling them treats runs trained 3x apart as
    extra seeds of one condition - the mixed-budget error that invalidated an early CDDM penalty
    table. This is the one place that rule is implemented.

    Args:
        files: dict {(cell_key..., budget): [paths]} where the LAST element of each key tuple is the
            integer budget and the leading elements identify the cell.
    Returns:
        dict {cell_key: [paths]} with cell_key the leading elements, keeping only the longest budget.
    """
    best = {}
    for key in files:
        cell, budget = key[:-1], key[-1]
        if budget >= best.get(cell, -1):
            best[cell] = budget
    out = {}
    for key, fs in sorted(files.items()):
        cell, budget = key[:-1], key[-1]
        if budget == best[cell]:
            out.setdefault(cell, []).extend(fs)
    return out


def load_traces(sweep, pattern=r"_N=(\d+)_iters=(\d+)"):
    """Load every ParticipationTrace under a sweep folder, grouped by network size.

    Args:
        sweep: path to the sweep folder holding `EqType=..._N=<N>_iters=<I>/<net>/` subfolders;
        pattern: regex whose groups are (cell key..., budget); the default keys on N.
    Returns:
        dict {N (int): list of trace dicts} as written by Trainer (keys `iters`, `participation`,
        `participation_iters`, `metrics`), each gaining a `loss` key holding TrainLosses.json as a
        float array, or None where that file is absent.
    """
    files = {}
    for f in sorted(glob.glob(os.path.join(sweep, "*", "*", "*ParticipationTrace.pkl"))):
        m = re.search(pattern, f)
        if m:
            files.setdefault(tuple(int(g) for g in m.groups()), []).append(f)
    out = {}
    for cell, fs in keep_longest_budget(files).items():
        for f in fs:
            with open(f, "rb") as fh:
                tr = pickle.load(fh)
            lf = glob.glob(os.path.join(os.path.dirname(f), "*TrainLosses.json"))
            tr["loss"] = np.array(json.load(open(lf[0]))["train_losses"], dtype=float) if lf else None
            out.setdefault(cell[0] if len(cell) == 1 else cell, []).append(tr)
    return out


def load_losses(sweep, pattern=r"_N=(\d+)_iters=(\d+)"):
    """Load every training-loss curve under a sweep folder, grouped by network size.

    ⚠️ TrainLosses.json is the OPTIMISER objective: task loss + lambda*penalty, evaluated with noise
    ON. It is not comparable across penalty conditions and has inverted a conclusion twice in this
    project. For anything cross-condition use the noise-free `loss_clean_train` metric inside the
    participation trace instead.

    Args:
        sweep: path to the sweep folder; pattern: regex whose groups are (cell key..., budget).
    Returns:
        dict {N (int): list of (tag, loss array)}, loss indexed from iteration 1.
    """
    files = {}
    for f in sorted(glob.glob(os.path.join(sweep, "*", "*", "*TrainLosses.json"))):
        m = re.search(pattern, f)
        if m:
            files.setdefault(tuple(int(g) for g in m.groups()), []).append(f)
    out = {}
    for cell, fs in keep_longest_budget(files).items():
        for f in fs:
            L = np.array(json.load(open(f))["train_losses"], dtype=float)
            out.setdefault(cell[0] if len(cell) == 1 else cell, []).append(
                (os.path.basename(f)[:9], L))
    return out


def r2_from_dir(leaf_dir):
    """Final R^2 of a network, parsed from the score prefix of its folder name.

    Args:
        leaf_dir: per-network folder, named `<r2>_<taskname>;<params>`.
    Returns:
        float R^2.
    """
    return float(os.path.basename(leaf_dir).split("_")[0])



# --------------------------------------------------------------------------------------------
# weight drift: is the optimiser still making directed progress, or just jittering?
# --------------------------------------------------------------------------------------------

LAGS = (100, 1000, 10000)       # the lags at which Trainer logs weight displacement


def series(trace, key):
    """Extract one metric as (iterations, values) with the NaN padding removed.

    Metrics defined only once per lag (the drift distances) are stored NaN-padded on the full probe
    grid; the cosines are defined at every probe. Dropping NaNs handles both.

    Args:
        trace: a trace dict from load_traces; key: metric name, e.g. "drift_W_rec_lag1000".
    Returns:
        (iters, vals) float arrays of equal length, possibly empty.
    """
    it = np.asarray(trace["iters"], dtype=float)
    v = np.asarray(trace["metrics"][key], dtype=float)
    ok = np.isfinite(v)
    return it[ok], v[ok]


def _drift_key(var, lag):
    """Metric key for one drift variable at one lag.

    Args:
        var: "W_rec"/"W_inp"/"W_out" for a weight matrix, or "p" for the participation vector
            (logged under `dp_lag*`, not `drift_p_lag*`).
        lag: lag in iterations.
    Returns:
        str metric name.
    """
    return f"dp_lag{lag}" if var == "p" else f"drift_{var}_lag{lag}"


def drift_alpha(trace, W="W_rec"):
    """Lag-scaling exponent of weight displacement over training: d(L) ~ L^alpha.

    This is the threshold-free way to ask whether the optimiser is still travelling or merely
    jittering. Displacement over a lag L grows as L^1 for motion in a fixed direction and as L^0.5
    for an unbiased random walk, so alpha reads directly:

        alpha ~ 1.0   ballistic - updates are BIASED, the network is still going somewhere
        alpha ~ 0.5   diffusive - updates have decorrelated, the network is jittering in place
        alpha < 0.5   confined  - mean-reverting, held inside a basin

    Fitted per probe as the slope of log d against log L across whichever lags have a value there.

    Args:
        trace: a trace dict from load_traces; W: "W_rec", "W_inp", "W_out", or "p" for the
            participation vector.
    Returns:
        (iters, alpha) float arrays; empty when fewer than two lags are available.
    """
    grids = {}
    for L in LAGS:
        it, v = series(trace, _drift_key(W, L))
        for i, val in zip(it, v):
            if val > 0:
                grids.setdefault(i, {})[L] = val
    its, al = [], []
    for i in sorted(grids):
        d = grids[i]
        if len(d) < 2:
            continue
        x = np.log(np.array(sorted(d)))
        y = np.log(np.array([d[L] for L in sorted(d)]))
        its.append(i)
        al.append(np.polyfit(x, y, 1)[0])
    return np.array(its), np.array(al)


def drift_alpha_pairwise(trace, W="W_rec"):
    """Lag exponent computed separately over each ADJACENT pair of lags.

    A single alpha from a 3-point log-log fit hides whether one power law actually describes the
    whole lag range. If alpha(100->1000) and alpha(1000->10000) disagree, the displacement is not a
    power law in lag and any single alpha - including the one `drift_alpha` returns - is an average
    over two different regimes rather than a measurement of one.

    Args:
        trace: a trace dict; W: drift variable, as in `drift_alpha`.
    Returns:
        dict {(l1, l2): (iters, alpha)} for each adjacent lag pair present.
    """
    out = {}
    for l1, l2 in zip(LAGS[:-1], LAGS[1:]):
        i1, d1 = series(trace, _drift_key(W, l1))
        i2, d2 = series(trace, _drift_key(W, l2))
        if len(i1) < 2 or len(i2) < 1:
            continue
        ok1, ok2 = d1 > 0, d2 > 0
        if ok1.sum() < 2 or ok2.sum() < 1:
            continue
        d1i = np.exp(np.interp(np.log(i2[ok2]), np.log(i1[ok1]), np.log(d1[ok1])))
        out[(l1, l2)] = (i2[ok2], np.log(d2[ok2] / d1i) / np.log(l2 / l1))
    return out


def scalar_alpha(iters, x, lags=LAGS):
    """Lag-scaling exponent of a SCALAR trajectory, e.g. the active-unit count M(t).

    Same question as `drift_alpha` but for a one-dimensional walk: the mean absolute displacement
    over a lag L grows as L^alpha, with 1.0 directed, 0.5 an unbiased random walk and <0.5 confined.
    Computed directly from the trajectory rather than from pre-logged displacements, so it works for
    any series recorded on a regular grid.

    Args:
        iters: (T,) iteration of each sample, assumed evenly spaced;
        x: (T,) the scalar series; lags: lags in ITERATIONS.
    Returns:
        (iters, alpha) evaluated on a sliding window, or empty arrays if the series is too short.
    """
    iters = np.asarray(iters, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(x) < 10:
        return np.array([]), np.array([])
    step = float(np.median(np.diff(iters)))
    if not np.isfinite(step) or step <= 0:
        return np.array([]), np.array([])
    steps = [max(1, int(round(L / step))) for L in lags]
    steps = sorted({s for s in steps if s < len(x) // 3})
    if len(steps) < 2:
        return np.array([]), np.array([])
    win = 4 * max(steps)
    its, al = [], []
    for start in range(0, len(x) - win, max(1, win // 4)):
        seg = x[start:start + win]
        d = []
        for sp in steps:
            dd = np.abs(seg[sp:] - seg[:-sp])
            d.append(np.mean(dd) if len(dd) else np.nan)
        d = np.array(d, dtype=float)
        if np.any(~np.isfinite(d)) or np.any(d <= 0):
            continue
        its.append(iters[start + win // 2])
        al.append(np.polyfit(np.log(np.array(steps, float) * step), np.log(d), 1)[0])
    return np.array(its), np.array(al)


def diffusive_onset(trace, W="W_rec", thresh=0.6, smooth=5, persist=5, alpha=None):
    """First iteration at which weight motion stops being directed AND stays that way.

    Reading every network here compares them at a matched DYNAMICAL state rather than a matched
    budget or a matched score: each is read once its updates stop carrying it somewhere new.

    ⚠️ The definition has to be the FIRST SUSTAINED crossing, not the last. An earlier version
    required alpha to stay below threshold all the way to the end of the run, which made a single
    late noisy excursion above threshold reset the answer to near the final iteration - sibling seeds
    of one cell returned 40k, 44k and 458k, and cells at high k returned ~470k against a 500k budget,
    i.e. the criterion was reporting the BUDGET rather than the dynamics. Requiring `persist`
    consecutive probes below threshold gives the first genuine transition and is stable across seeds.

    Args:
        trace: a trace dict; W: weight matrix; thresh: alpha below which motion is no longer directed
            (0.6 sits between the 0.5 diffusive and 1.0 ballistic references);
        smooth: running-median window in probes, since alpha is noisy at the longest lag;
        persist: consecutive smoothed probes that must stay below thresh for the crossing to count;
        alpha: optional precomputed (iters, alpha) pair, to avoid recomputing it per threshold.
    Returns:
        float iteration, or nan if the run never settles below the threshold.
    """
    it, al = drift_alpha(trace, W) if alpha is None else alpha
    if len(al) < max(smooth, persist) + 1:
        return float("nan")
    h = smooth // 2
    sm = np.array([np.median(al[max(0, i - h):i + h + 1]) for i in range(len(al))])
    below = sm < thresh
    # ⚠️ An ONSET presupposes a prior state. If alpha is already below threshold when the trace
    # begins, the run never shows directed motion inside the recorded window and there is no
    # transition to find - returning the first probe would report where the TRACE STARTS, not a
    # property of the network. That is exactly what happened on CDDM, where alpha is ~0.5 by the
    # second probe for every N >= 500 and the criterion returned iteration 1000 with the loss still
    # at 0.033-0.040 against a final 0.019-0.025, yielding a meaningless b = 0.94.
    if not (~below[:persist]).all():
        return float("nan")
    # ⚠️ THE RUN MUST STILL BE SETTLED AT THE END. `persist` consecutive probes below threshold is
    # not enough on its own: alpha is noisy per probe, so a run whose alpha PLATEAUS at 0.74 will
    # still produce runs of 5 probes under 0.6 by chance. That is exactly what happened on the frm
    # networks - `diffusive_onset` reported 6/8 W_rec "settled" while the 50k-block medians showed a
    # plateau at 0.74 that never sustained below the threshold. Requiring the tail median to be below
    # thresh too rejects a transient dip without rejecting a genuine settling.
    tail = sm[-max(persist, len(sm) // 10):]
    if np.median(tail) >= thresh:
        return float("nan")
    for i in range(persist, len(below) - persist + 1):
        if below[i:i + persist].all():
            return float(it[i])
    return float("nan")


def _self_check():
    """Assert the invariants that make these primitives safe to share. Raises on failure."""
    # participation reduces every trailing axis, so (N, T, B) and its (N, T*B) flattening agree
    rng = np.random.default_rng(0)
    fr = rng.normal(size=(7, 11, 5))
    assert np.allclose(participation(fr), participation(fr.reshape(7, -1))), "participation rank"

    # otsu_threshold lands between two well-separated log-spaced modes, wherever they sit
    rng2 = np.random.default_rng(1)
    for lo, hi in [(-4.0, 0.0), (-8.0, -2.0), (-1.0, 1.0)]:
        pp = np.concatenate([10 ** rng2.normal(lo, 0.25, 800), 10 ** rng2.normal(hi, 0.25, 200)])
        t = otsu_threshold(pp)
        assert 10 ** lo < t < 10 ** hi, (lo, hi, t)
        # and it recovers the right split: ~800 below, ~200 above
        assert 700 < (pp < t).sum() < 900, (pp < t).sum()
    assert np.isnan(otsu_threshold(np.zeros(100))), "all-zero input must return nan"

    # the two criteria bracket each other: hard is an absolute floor, scale-free a relative one
    p = np.array([0.0, 1e-9, 0.5, 1.0, 2.0])
    assert active_count(p, "hard") == 3, active_count(p, "hard")
    assert active_count(p, "scalefree") == 3
    # a float criterion is an explicit absolute threshold
    assert active_count(p, 0.75) == 2 and active_count(p, 1e-12) == 4
    # active_mask is the rule as it was typed inline, in the input's own dtype, at any bar
    for q in (rng.gamma(0.3, size=500).astype(np.float32), rng.gamma(0.3, size=500)):
        for rel in (SILENT_REL, 0.01, 0.2):
            assert np.array_equal(active_mask(q, rel), q >= rel * np.quantile(q, 0.95)), rel
        assert active_count(q, "scalefree") == active_mask(q.astype(float)).sum()
    assert r2_at_floor_estimate(0.02, 0.01, 0.99) == 1.0 - 2.0 * (1.0 - 0.99)

    # hhi: even -> 1/N, fully concentrated -> 1, degenerate -> nan
    assert abs(hhi(np.ones(4)) - 0.25) < 1e-12
    # PR is exactly 1/HHI, and equals the unit count when activity is perfectly even
    assert abs(participation_ratio(np.ones(7)) - 7.0) < 1e-9
    assert abs(participation_ratio(np.array([0., 0., 3.])) - 1.0) < 1e-9
    for q in (np.array([1., 2., 3., 4.]), np.abs(np.random.default_rng(4).normal(size=20))):
        assert abs(participation_ratio(q) * hhi(q) - 1.0) < 1e-9, "PR must equal 1/HHI"
    assert abs(hhi(np.array([0.0, 0.0, 3.0])) - 1.0) < 1e-12
    assert np.isnan(hhi(np.zeros(3)))

    # stable_crossing takes the LAST time above threshold, not the first dip below it
    L = np.concatenate([np.full(300, 1.0), np.full(50, 0.1), np.full(300, 1.0), np.full(600, 0.1)])
    c = stable_crossing(L, 0.5, window=51)
    assert c is not None and 600 < c < 700, c
    assert stable_crossing(np.full(500, 1.0), 0.5, window=51) is None      # never crosses
    assert stable_crossing(np.full(500, 0.1), 0.5, window=51) is None      # never above

    # logbin drops sparse bins and returns matched lengths
    t = np.arange(1, 10001, dtype=float)
    tb, yb = logbin(t, 1.0 / t, nbins=20)
    assert len(tb) == len(yb) and len(tb) > 5 and np.all(np.diff(tb) > 0)

    # stretched hits its floor at large t and its floor+A at t << tau
    assert abs(stretched(1e12, 0.02, 1.0, 1e3, 0.5) - 0.02) < 1e-9
    assert abs(stretched(1e-9, 0.02, 1.0, 1e3, 0.5) - 1.02) < 1e-6

    # excess_time inverts stretched: at the returned t the excess really is frac of its t0 value
    A, tau, beta, t0 = 1.0, 5e3, 0.4, 2000.0
    for frac in (0.5, 0.1):
        t_f = excess_time(A, tau, beta, frac, t0)
        e0 = stretched(t0, 0.0, A, tau, beta)
        assert abs(stretched(t_f, 0.0, A, tau, beta) / e0 - frac) < 1e-9, frac

    # aicc penalises the extra parameter when the fit is not improved
    assert aicc(1.0, 50, 3) < aicc(1.0, 50, 4)

    # keep_longest_budget drops the short run and keeps every seed of the long one
    got = keep_longest_budget({(500, 100): ["a"], (500, 300): ["b", "c"], (1000, 50): ["d"]})
    assert got == {(500,): ["b", "c"], (1000,): ["d"]}, got

    # drift_alpha recovers the exponent of a synthetic power law d(L) = c * L^a
    for a_true in (0.5, 1.0):
        fake = {"iters": np.arange(1, 4, dtype=float),
                "metrics": {f"drift_W_rec_lag{L}": np.full(3, (L ** a_true)) for L in LAGS}}
        _, al = drift_alpha(fake)
        assert np.allclose(al, a_true), (a_true, al)

    # diffusive_onset fires on a sustained drop, not on a single dip
    it = np.arange(1, 41, dtype=float)
    al = np.where(it > 25, 0.5, 1.0)
    al[5] = 0.5                                        # a lone dip that must NOT trigger
    fake = {"iters": it, "metrics": {f"drift_W_rec_lag{L}": (L ** al) for L in LAGS}}
    on = diffusive_onset(fake, thresh=0.6, smooth=5, persist=5)
    assert 24 <= on <= 28, on

    # ... and it reports the FIRST sustained crossing, not the last: a late excursion back above
    # threshold must NOT push the answer to the end of the run (the bug that made sibling seeds
    # of one cell return 40k, 44k and 458k).
    al2 = np.where(it > 10, 0.4, 1.0)
    al2[30:33] = 1.0                                   # a late blip back to ballistic
    fake2 = {"iters": it, "metrics": {f"drift_W_rec_lag{L}": (L ** al2) for L in LAGS}}
    on2 = diffusive_onset(fake2, thresh=0.6, smooth=5, persist=5)
    assert 9 <= on2 <= 14, on2
    flat = {"iters": it, "metrics": {f"drift_W_rec_lag{L}": np.full(40, L ** 1.0) for L in LAGS}}
    assert np.isnan(diffusive_onset(flat)), "never-diffusive run must return nan"

    # a run that PLATEAUS above threshold but dips transiently must NOT count as settled
    rng4 = np.random.default_rng(3)
    al4 = np.full(40, 0.74) + rng4.normal(0, 0.12, 40)
    al4[12:18] = 0.45                                   # a sustained-looking transient dip
    fake4 = {"iters": it, "metrics": {f"drift_W_rec_lag{L}": (L ** al4) for L in LAGS}}
    assert np.isnan(diffusive_onset(fake4, thresh=0.6)), "plateau above thresh must return nan"

    # already diffusive at the first probe -> no onset exists, must be nan rather than probe 0
    al3 = np.full(40, 0.5)
    fake3 = {"iters": it, "metrics": {f"drift_W_rec_lag{L}": (L ** al3) for L in LAGS}}
    assert np.isnan(diffusive_onset(fake3)), "no prior directed phase must return nan"

    # scalar_alpha recovers the exponent of a synthetic self-affine walk
    rng3 = np.random.default_rng(2)
    n = 4000
    itr = np.arange(n, dtype=float) * 100
    walk = np.cumsum(rng3.normal(size=n))                 # alpha = 0.5 by construction
    _, a_w = scalar_alpha(itr, walk, lags=(100, 1000, 10000))
    assert len(a_w) and abs(np.median(a_w) - 0.5) < 0.15, np.median(a_w)
    ramp = np.arange(n, dtype=float)                      # pure drift -> alpha = 1
    _, a_r = scalar_alpha(itr, ramp, lags=(100, 1000, 10000))
    assert len(a_r) and abs(np.median(a_r) - 1.0) < 0.05, np.median(a_r)

    # pairwise alphas equal the pooled one when the law really is a single power law
    it4 = np.arange(1, 60, dtype=float)
    fake4 = {"iters": it4, "metrics": {f"drift_W_rec_lag{L}": np.full(59, L ** 0.7) for L in LAGS}}
    pw = drift_alpha_pairwise(fake4)
    assert pw and all(abs(np.median(a) - 0.7) < 1e-6 for _, a in pw.values()), pw

    # noisy_floor recovers a known floor and crossing under noise, and finds planted bursts without
    # letting them move either. Truth: floor 0.01, crossing of 1.05x at tau*ln(A/(0.05 floor))^2.
    rng5 = np.random.default_rng(5)
    it5 = np.arange(1, 150_001, dtype=float)
    L5 = (0.01 + 0.5 * np.exp(-np.sqrt(it5 / 1000))) * rng5.gamma(1 / 0.03**2, 0.03**2, len(it5))
    for s, w, h in ((80_000, 500, 5.0), (110_000, 3000, 10.0)):
        L5[s:s + w] *= h
    f5 = noisy_floor(it5, L5)
    assert abs(f5["floor"] / 0.01 - 1) < 0.01, f5["floor"]
    assert abs(f5["t_star"] / (1000 * np.log(1000) ** 2) - 1) < 0.1, f5["t_star"]
    assert len(f5["bursts"]) == 2, f5["bursts"]

    print("common.py self-check passed")


if __name__ == "__main__":
    _self_check()
