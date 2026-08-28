#!/usr/bin/env python3
"""
The flip-flop grid, sliced every way: loss floors and active units vs (N, k), under four read-out
criteria.

WHY FOUR CRITERIA AND NOT ONE. "How many units are active?" has no answer until you say WHEN the
network is read, and every choice of when is a modelling assumption that can carry the result:

  iter    fixed iteration. Naive but assumption-free about the loss. Confounded because cells reach
          very different performance in the same budget, and silencing tracks training depth - so a
          cell that has trained further looks more silent for reasons unrelated to N or k.
  loss    fixed absolute clean loss L*. Comparable across N, and across k too because the flip-flop's
          target variance is k-independent (0.720-0.738 measured over k=1..8), so MSE is already
          per-channel-normalised. ⚠️ but the FLOOR rises with k, so a single L* sits at a different
          distance above the floor for each k - deep for k=1, barely reached for k=8.
  excess  fixed RELATIVE excess over each cell's OWN fitted floor, L = (1+delta)*L_inf. This is the
          k-fair version of `loss`: every cell is read at the same fraction of the way down to its
          own achievable optimum. Costs a floor fit, and inherits that fit's uncertainty.
  drift   the iteration at which weight motion stops being DIRECTED - the lag exponent alpha of
          |W(t+L)-W(t)| ~ L^alpha falls below 0.6 and stays there (1.0 ballistic, 0.5 diffusive,
          <0.5 confined). Reads every network at a matched DYNAMICAL state rather than a matched
          budget or a matched score, and needs no threshold on the loss at all.

If the ordering in N and k survives all four, it does not depend on the choice and needs no further
defence. Where they disagree, THAT is the finding and it is reported rather than resolved by picking
a favourite.

A FIFTH VIEW WITH NO READ-OUT AT ALL. Figure 4 plots M against R^2 as a trajectory, one curve per
cell. Because R^2 is comparable across k, this shows the whole M-vs-performance relation instead of
one point on it, and any criterion above is just a vertical slice through it.

Floors and learning-dynamics parameters come from a stretched-exponential fit
    L(t) = L_inf + A*exp(-(t/tau)^beta)
fitted in log space on log-binned medians. ⚠️ A and tau are not individually identified once tau
falls below the fit's start (see common.stretched); compare through L_inf and common.excess_time.

Losses are `loss_clean_train`, the noise-free forward pass. Never TrainLosses.json, which is
task + lambda*penalty with noise on and is not comparable across penalty conditions.

Outputs: img/internal_figures/flipflop_fig{1_curves,2_floor,3_active,4_criteria}.png

Usage:  python flipflop_figures.py [T_ITER]
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import f as fdist
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (IMG_DIR, active_count, diffusive_onset, excess_time, logbin, stretched)
import plotstyle as ps

KSWEEP = "data/trained_RNNs/NBitFlipFlop_std_ksweep"
PROBE_EVERY = 10          # trainer.track_every; loss_clean_train is indexed in probes
T_ITER = 150000           # the `iter` criterion: largest budget EVERY cell in the grid reaches
T_START = 2000            # fit start; skips the initial collapse, which no smooth model describes
TGT_VAR = 0.735           # k-independent target variance, for R^2 = 1 - MSE/TGT_VAR
EXCESS_DELTA = 0.10       # `excess` criterion: read where L = (1 + delta) * L_inf
LOSS_MARGIN = 1.10        # `loss` criterion: L* = margin * the WORST floor, so every cell reaches it
ALPHA_THRESH = 0.6        # `drift` criterion: alpha below this counts as no longer directed
CRITS = ["endpoint", "iter", "loss", "excess", "drift"]

# Absolute silence threshold CALIBRATED FOR THIS TASK. Derived in flipflop_hard_threshold.py by
# Otsu's method on log10(participation), validated against three checks fixed in advance: spread
# across cells 0.50 decades (<1), median |dM|/N vs the scale-free rule 0.030 (<0.05), and 41000x
# larger than the CDDM-calibrated 1e-6. The old 1e-6 sits below BOTH of the flip-flop's modes and
# reports ~0% silence, which is why the `hard` row of these figures is degenerate.
ABS_THRESH = 4e-2

SILENCE_RULES = [
    ("scalefree", "flipflop_fig5_law",
     "SCALE-FREE active units  ($p_i \\geq 0.05\\,q_{95}$, relative to the network's own scale)"),
    (ABS_THRESH, "flipflop_fig5_law_abs",
     f"ABSOLUTE active units, TASK-CALIBRATED  ($p_i \\geq {ABS_THRESH:g}$, Otsu antimode of the "
     f"bimodal log-participation distribution)"),
    ("hard", "flipflop_fig5_law_hard",
     "ABSOLUTE active units, CDDM-calibrated  ($p_i \\geq 10^{-6}$) — ⚠️ DEGENERATE on this task: "
     "$10^{-6}$ lies below both modes, so $M = N$ and nothing is measured"),
]

# Stated concretely, because "matched performance" is ambiguous until you say matched on WHAT.
CRIT_DESC = {
    "endpoint": "ENDPOINT\neach run at its own final iteration\n⚠️ budgets differ (400k vs 500k)",
    "iter":     f"FIXED COMPUTE\nevery run read at t = {T_ITER:,}\n(equal budget for all)",
    "loss":     "MATCHED PERFORMANCE\nsame absolute noise-free loss $L^*$\n(equal score)",
    "excess":   "MATCHED PERFORMANCE, k-fair\n$L=1.10\\times$ each run's OWN floor $L_\\infty$\n(equal distance to own optimum)",
    "drift":    "MATCHED DYNAMICS\nwhere weight motion stops being\ndirected ($\\alpha<0.6$, diffusion onset)",
}


def load():
    """Load every completed unpenalised flip-flop run as a flat list of run records.

    Returns:
        (runs, dropped): runs is a list of dicts with keys k, N, loss (noise-free loss indexed by
        probe), part (participation snapshots), piters (their iterations), trace (the raw dict, for
        the drift criterion), budget (int iterations); dropped lists (reason, k, N) for runs whose
        loss contains a NaN or that never reach T_ITER.
    """
    runs, dropped = [], []
    for f in sorted(glob.glob(os.path.join(KSWEEP, "*", "*", "*ParticipationTrace.pkl"))):
        m = re.search(r"_k=(\d+)_N=(\d+)", f)
        if not m:
            continue
        k, N = int(m.group(1)), int(m.group(2))
        with open(f, "rb") as fh:
            tr = pickle.load(fh)
        L = np.asarray(tr["metrics"].get("loss_clean_train", []), dtype=float)
        if L.size == 0 or np.isnan(L).any():
            dropped.append(("diverged (NaN loss)", k, N))
            continue
        if len(L) * PROBE_EVERY < T_ITER:
            dropped.append((f"budget {len(L)*PROBE_EVERY} < {T_ITER}", k, N))
            continue
        runs.append(dict(k=k, N=N, loss=L, trace=tr, budget=len(L) * PROBE_EVERY,
                         part=tr["participation"],
                         piters=np.asarray(tr["participation_iters"], dtype=float)))
    if not runs:
        raise SystemExit(f"no completed runs under {KSWEEP}")
    return runs, dropped


def fit_run(L, t_end):
    """Stretched-exponential fit of one noise-free loss trace, in log space on log-binned medians.

    Args:
        L: loss indexed by probe (probe i is iteration (i+1)*PROBE_EVERY);
        t_end: last iteration included, so every run is fitted over an identical range.
    Returns:
        (L_inf, A, tau, beta) or None if the fit fails or the range holds too few bins.
    """
    t = (np.arange(len(L)) + 1) * PROBE_EVERY
    m = (t >= T_START) & (t <= t_end)
    tb, yb = logbin(t[m], L[m])
    if len(tb) < 8:
        return None
    try:
        sol = least_squares(
            lambda p: np.log(np.clip(stretched(tb, *p), 1e-12, None)) - np.log(yb),
            [yb.min() * 0.9, float(yb.max()), 2e4, 0.4],
            bounds=([1e-6, 1e-6, 1e2, 0.05], [1.0, 1e3, 1e8, 3.0]), max_nfev=20000)
    except Exception:
        return None
    return tuple(sol.x)


def loss_at(L, t):
    """Noise-free loss at an iteration, from the probe grid.

    Args:
        L: loss indexed by probe; t: iteration.
    Returns:
        float loss at the nearest probe at or before t, or nan if t precedes the first probe.
    """
    i = int(t // PROBE_EVERY) - 1
    if i < 0 or len(L) == 0:
        return float("nan")
    return float(L[min(i, len(L) - 1)])


def first_below(L, target):
    """First iteration at which the noise-free loss falls to `target`.

    Uses a short running mean, because the raw clean loss is not monotone and a bare first-crossing
    fires on a transient dip.

    Args:
        L: loss indexed by probe; target: loss level.
    Returns:
        float iteration, or nan if never reached.
    """
    w = 21
    if len(L) < w:
        return float("nan")
    s = np.convolve(L, np.ones(w) / w, mode="valid")
    hit = np.flatnonzero(s <= target)
    if not len(hit):
        return float("nan")
    return float((hit[0] + w // 2 + 1) * PROBE_EVERY)


def readout_times(runs, Lstar):
    """Read-out iteration for every run under each of the four criteria.

    Args:
        runs: run records, each already carrying a fitted "L_inf"; Lstar: the common absolute loss
            level for the `loss` criterion.
    Returns:
        None; each run gains a "T" dict {criterion: iteration or nan}.
    """
    for r in runs:
        T = {"endpoint": float(r["budget"]),
             "iter": float(T_ITER),
             "loss": first_below(r["loss"], Lstar),
             "excess": (first_below(r["loss"], (1 + EXCESS_DELTA) * r["L_inf"])
                        if r.get("L_inf") else float("nan")),
             "drift": diffusive_onset(r["trace"], "W_rec", thresh=ALPHA_THRESH)}
        # A criterion that lands beyond what the run actually has is not usable for that run.
        for c, t in T.items():
            if not np.isfinite(t) or t > r["budget"]:
                T[c] = float("nan")
        r["T"] = T


def M_at(run, t, criterion="scalefree"):
    """Active-unit count at the last participation snapshot at or before an iteration.

    Args:
        run: run record; t: iteration (nan gives nan); criterion: "hard" or "scalefree".
    Returns:
        float count, or nan when t is nan or precedes the first snapshot.
    """
    if not np.isfinite(t):
        return float("nan")
    ok = np.flatnonzero(run["piters"] <= t)
    if ok.size == 0:
        return float("nan")
    return float(active_count(np.asarray(run["part"][int(ok[-1])], dtype=float), criterion))


def cellwise(runs, field, ks, Ns):
    """Mean/sd/count of a per-run field over seeds, as a (len(Ns), len(ks)) grid plus a lookup.

    Args:
        runs: run records; field: callable run -> float; ks, Ns: sorted axes.
    Returns:
        (mu, sd, lookup) where mu and sd are (len(Ns), len(ks)) arrays with nan for empty cells and
        lookup is {(k, N): (mean, sd, n)}.
    """
    box = {}
    for r in runs:
        v = field(r)
        if v is not None and np.isfinite(v):
            box.setdefault((r["k"], r["N"]), []).append(v)
    mu = np.full((len(Ns), len(ks)), np.nan)
    sd = np.full((len(Ns), len(ks)), np.nan)
    look = {}
    for (k, N), vs in box.items():
        look[(k, N)] = (float(np.mean(vs)), float(np.std(vs)), len(vs))
        mu[Ns.index(N), ks.index(k)] = np.mean(vs)
        sd[Ns.index(N), ks.index(k)] = np.std(vs)
    return mu, sd, look


def sqrt_law(ks, floors):
    """Fit the per-channel floor law f(k) = a + b*sqrt(k).

    Args:
        ks: bit counts; floors: matching per-channel floors.
    Returns:
        (a, b, max relative residual), or nans if fewer than three points.
    """
    ks, floors = np.asarray(ks, float), np.asarray(floors, float)
    ok = np.isfinite(floors)
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(ok.sum()), np.sqrt(ks[ok])])
    (a, b), *_ = np.linalg.lstsq(X, floors[ok], rcond=None)
    pred = a + b * np.sqrt(ks[ok])
    return a, b, float(np.max(np.abs(pred - floors[ok]) / floors[ok]))


def slope_b(look, k, Ns):
    """Scaling exponent b of M ~ N^b at one complexity, by log-log least squares.

    Args:
        look: {(k, N): (mean, sd, n)}; k: complexity; Ns: sizes to use.
    Returns:
        float b, or nan if fewer than two sizes carry data.
    """
    pts = [(N, look[(k, N)][0]) for N in Ns if (k, N) in look and np.isfinite(look[(k, N)][0])]
    if len(pts) < 2:
        return float("nan")
    return float(np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0])


def fit_law(runs, crit, silence="scalefree"):
    """Fit M = A*N^b*k^c at one read-out criterion and test it against the saturated model.

    THE BASELINE IS A FREE MEAN PER CELL, not zero. The saturated model gives every (k, N) cell its
    own mean log M - 22 free numbers here - and its residual is pure seed scatter, which no formula
    can beat. A 3-parameter law is credible only if its residual is no worse, i.e. if what the
    saturated model captures beyond the law is nothing but noise.

    The statistic is the classical lack-of-fit F with replication:

        F = [(RSS_law - RSS_sat)/(ncell - npar)] / [RSS_sat/(n - ncell)]
              lack of fit per dof                    pure error per dof

    H0 is that the law is EXACTLY right - every cell mean lies on the surface and all scatter is seed
    noise - so a LARGE p supports the law. That is inverted from the usual convention and is the one
    thing about this test that is easy to read backwards.

    ⚠️ Large p can also mean low power. With 3 seeds per cell, failing to reject is weaker evidence
    than rejecting.

    ⚠️ A single power law cannot express an N x k INTERACTION. If the k-trend changes sign with N,
    c averages over that sign change and reports ~0 while neither size is actually flat - so read c
    together with the per-cell table, never alone.

    Args:
        runs: run records carrying "T"; crit: read-out criterion name;
        silence: which silence rule counts a unit as active - "scalefree" (relative to the network's
            own scale) or "hard" (absolute, p >= 1e-6). BOTH are always reported: they disagree, and
            which one is used has flipped a conclusion in this project before.
    Returns:
        dict with A, b, c, their bootstrap CIs, F, p, r2_law, r2_sat, ncell, n - or None if the
        criterion yields fewer than 3 usable cells.
    """
    K, NN, M = [], [], []
    for r in runs:
        m = M_at(r, r["T"][crit], silence)
        if np.isfinite(m) and m > 0:
            K.append(r["k"]); NN.append(r["N"]); M.append(m)
    if len(M) < 8:
        return None
    K, NN, M = np.array(K, float), np.array(NN, float), np.array(M, float)
    y = np.log(M)
    X = np.column_stack([np.ones_like(y), np.log(NN), np.log(K)])
    if np.linalg.matrix_rank(X) < 3:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss_law = float(np.sum((y - X @ beta) ** 2))

    cells = {}
    for i in range(len(y)):
        cells.setdefault((K[i], NN[i]), []).append(y[i])
    rss_sat = float(sum(np.sum((np.array(v) - np.mean(v)) ** 2) for v in cells.values()))
    tot = float(np.sum((y - y.mean()) ** 2))
    ncell, n = len(cells), len(y)
    d1, d2 = ncell - 3, n - ncell
    if d1 > 0 and d2 > 0 and rss_sat > 0:
        F = ((rss_law - rss_sat) / d1) / (rss_sat / d2)
        pval = float(1 - fdist.cdf(F, d1, d2))
    else:
        F, pval = np.nan, np.nan

    bs = []                                   # seed-resampled CIs, seeded for reproducibility
    rng = np.random.default_rng(0)
    for _ in range(1000):
        i = rng.integers(0, n, n)
        if len(np.unique(K[i])) < 3 or len(np.unique(NN[i])) < 2:
            continue
        Xi = np.column_stack([np.ones(n), np.log(NN[i]), np.log(K[i])])
        try:
            bs.append(np.linalg.lstsq(Xi, y[i], rcond=None)[0])
        except np.linalg.LinAlgError:
            continue
    bs = np.array(bs)
    ci = (lambda j: (float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5)))
          ) if len(bs) > 50 else (lambda j: (np.nan, np.nan))
    return dict(A=float(np.exp(beta[0])), b=float(beta[1]), c=float(beta[2]),
                b_ci=ci(1), c_ci=ci(2), F=F, p=pval, ncell=ncell, n=n,
                r2_law=1 - rss_law / tot, r2_sat=1 - rss_sat / tot,
                K=K, NN=NN, M=M)


def fig_law(runs, ks, Ns, crits, silence="scalefree"):
    """Figure 5: the joint power law under each comparison criterion, before and after collapsing."""
    fits = [(c, fit_law(runs, c, silence)) for c in crits]
    fits = [(c, f) for c, f in fits if f]
    fig, ax = plt.subplots(2, len(fits), figsize=(4.1 * len(fits), 9.8), squeeze=False)
    for j, (crit, f) in enumerate(fits):
        K, NN, M = f["K"], f["NN"], f["M"]
        law = (f"$M = {f['A']:.1f}\\,N^{{{f['b']:.2f}}}k^{{{f['c']:.2f}}}$")
        verdict = ("law OK: matches a free mean per cell" if f["p"] > 0.05
                   else "LAW REJECTED: misses cell structure")

        for k in ks:                                        # BEFORE: raw M vs N, one colour per k
            m = K == k
            if m.sum():
                ax[0][j].plot(NN[m], M[m], "o", color=ps.col_k(k, ks), ms=6, alpha=.85)
        ax[0][j].set(xscale="log", yscale="log", xlabel="$N$", ylabel="active units $M$",
                     xticks=Ns, xticklabels=[str(n) for n in Ns],
                     title=f"CRITERION: {CRIT_DESC[crit]}\n"
                           f"BEFORE — raw $M$ vs $N$, one colour per $k$")
        ax[0][j].title.set_fontsize(8.5)
        if j == 0:
            ps.legend_k(ax[0][j], ks, loc="upper left")

        alpha = f["c"] / f["b"] if f["b"] else 0.0          # AFTER: collapse onto u = N k^(c/b)
        u = NN * K ** alpha
        for k in ks:
            m = K == k
            if m.sum():
                ax[1][j].plot(u[m], M[m], "o", color=ps.col_k(k, ks), ms=6, alpha=.85)
        uu = np.logspace(np.log10(u.min()), np.log10(u.max()), 100)
        ax[1][j].plot(uu, f["A"] * uu ** f["b"], "k-", lw=1.5, alpha=.8)
        ax[1][j].set(xscale="log", yscale="log", ylabel="active units $M$",
                     xlabel=f"$u = N\\,k^{{{alpha:.2f}}}$",
                     title=f"LAW: {law}\n"
                           f"$b$={f['b']:.2f} [{f['b_ci'][0]:.2f},{f['b_ci'][1]:.2f}]   "
                           f"$c$={f['c']:.2f} [{f['c_ci'][0]:.2f},{f['c_ci'][1]:.2f}]\n"
                           f"lack-of-fit p={f['p']:.2g} — {verdict}")
        ax[1][j].title.set_fontsize(8.5)
    sname = next(t for r, _, t in SILENCE_RULES if r == silence)
    fig.suptitle(f"{sname}\n"
                 "Is $M = A\\,N^b k^c$ a real law, and does that depend on HOW networks are compared?\n"
                 "Columns = comparison criterion. Top: before collapsing. Bottom: after rescaling "
                 "$x$ to $u=Nk^{c/b}$ — a real law fuses the colours into one line.\n"
                 "$c>0$ means more bits recruit more units; $c=0$ means the count is set by size "
                 "alone. LARGE p supports the law (it is a lack-of-fit test).", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return ps.save(fig, next(n for r, n, _ in SILENCE_RULES if r == silence), tight=False)


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------

def fig_curves(runs, ks, Ns):
    """Figure 1: learning curves with their stretched-exponential fits, and the fit residuals."""
    fig, ax = plt.subplots(2, len(Ns), figsize=(4.7 * len(Ns), 8.0), squeeze=False)
    for c, N in enumerate(Ns):
        for k in ks:
            sel = [r for r in runs if r["k"] == k and r["N"] == N and r.get("L_inf")]
            if not sel:
                continue
            col = ps.col_k(k, ks)
            for r in sel:
                t = (np.arange(len(r["loss"])) + 1) * PROBE_EVERY
                m = (t >= T_START) & (t <= T_ITER)
                tb, yb = logbin(t[m], r["loss"][m])
                ax[0][c].plot(tb, yb, "-", color=col, alpha=.55, lw=1.1)
                pred = stretched(tb, r["L_inf"], r["A"], r["tau"], r["beta"])
                ax[0][c].plot(tb, pred, "--", color="k", alpha=.45, lw=0.8)
                ax[1][c].plot(tb, 100 * (yb - pred) / pred, "-", color=col, alpha=.6, lw=1.0)
            r0 = sel[0]
            ax[0][c].axhline(r0["L_inf"], color=col, ls=":", lw=0.8, alpha=.7)
        ax[0][c].set(xscale="log", yscale="log", xlabel="iteration", ylabel="noise-free loss",
                     title=f"N={N}: curves + fits\n(black dashed = fit, dotted = $L_\\infty$)")
        ax[1][c].axhline(0, color="k", lw=0.8)
        ax[1][c].set(xscale="log", xlabel="iteration", ylabel="residual (%)", ylim=(-8, 8),
                     title=f"N={N}: fit residual")
        ps.legend_k(ax[0][c], ks, loc="lower left")
    fig.suptitle(r"Stretched-exponential fits  $L(t)=L_\infty + A\,e^{-(t/\tau)^\beta}$"
                 f"   (fitted on [{T_START}, {T_ITER}], log-binned, log-space residuals)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return ps.save(fig, "flipflop_fig1_curves", tight=False)


def fig_floor(runs, ks, Ns):
    """Figure 2: the fitted floor sliced against k, against N, and as a contour over both."""
    mu, sd, look = cellwise(runs, lambda r: r.get("L_inf"), ks, Ns)
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.2))

    for i, N in enumerate(Ns):                                   # (a) floor vs k, one line per N
        ps.band(ax[0][0], ks, mu[i], sd[i], ps.col_n(N), label=f"N={N}")
        a, b, _ = sqrt_law(ks, mu[i])
        if np.isfinite(a):
            xx = np.linspace(min(ks), max(ks), 100)
            ax[0][0].plot(xx, a + b * np.sqrt(xx), ":", color=ps.col_n(N), lw=1.1, alpha=.9)
    ax[0][0].set(xlabel="k (bits)", ylabel=r"per-channel floor $L_\infty$", xticks=ks,
                 title="(a) floor vs complexity\ndotted: $a+b\\sqrt{k}$ fitted per N")
    ax[0][0].legend()

    for j, k in enumerate(ks):                                   # (b) floor vs N, one line per k
        ps.band(ax[0][1], Ns, mu[:, j], sd[:, j], ps.col_k(k, ks), label=f"k={k}")
    ax[0][1].set(xscale="log", xlabel="N (units)", ylabel=r"per-channel floor $L_\infty$",
                 xticks=Ns, xticklabels=[str(n) for n in Ns],
                 title="(b) floor vs size — FLAT means N-independent")
    ps.legend_k(ax[0][1], ks, loc="best")

    ps.contour(ax[0][2], ks, Ns, mu, r"per-channel floor $L_\infty$")   # (c) contour
    ax[0][2].set_title("(c) floor over the (k, N) grid\nо = measured cell, x = missing")

    for i, N in enumerate(Ns):                                   # (d) total floor
        ps.band(ax[1][0], ks, np.array(ks) * mu[i], np.array(ks) * sd[i], ps.col_n(N), f"N={N}")
    ax[1][0].set(xlabel="k (bits)", ylabel=r"total floor $k\,L_\infty$", xticks=ks,
                 title=r"(d) total floor $= k a + b k^{1.5}$")
    ax[1][0].legend()

    for i, N in enumerate(Ns):                                   # (e) residual to the law
        a, b, _ = sqrt_law(ks, mu[i])
        if np.isfinite(a):
            ax[1][1].plot(ks, 100 * (mu[i] - (a + b * np.sqrt(ks))) / mu[i], "-o",
                          color=ps.col_n(N), label=f"N={N}")
    ax[1][1].axhline(0, color="k", lw=0.8)
    ax[1][1].set(xlabel="k (bits)", ylabel="residual to $a+b\\sqrt{k}$ (%)", xticks=ks,
                 title="(e) is the law exact?\nresidual at the seed-noise level = yes")
    ax[1][1].legend()

    rows = []                                                    # (f) the fitted law per N
    for i, N in enumerate(Ns):
        a, b, res = sqrt_law(ks, mu[i])
        rows.append((N, a, b, res))
    ax[1][2].axis("off")
    txt = ["fitted  $L_\\infty(k) = a + b\\sqrt{k}$", ""]
    txt += [f"N={N:<5d}  a={a:.5f}   b={b:.5f}   max resid {100*res:.2f}%"
            for N, a, b, res in rows if np.isfinite(a)]
    txt += ["", "a = single-channel floor (no interference)",
            "b = interference amplitude",
            "", "a and b constant across N means the floor is",
            "a TASK property, not a capacity limit."]
    ax[1][2].text(0.02, 0.95, "\n".join(txt), va="top", family="monospace", fontsize=8.5,
                  transform=ax[1][2].transAxes)
    fig.suptitle("Loss floor vs task complexity and network size", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return ps.save(fig, "flipflop_fig2_floor", tight=False)


def fig_active(runs, ks, Ns, crit_used):
    """Figure 3: active units vs N and k, one row per read-out criterion."""
    fig, ax = plt.subplots(len(crit_used), 3, figsize=(16.5, 4.3 * len(crit_used)), squeeze=False)
    for r_i, crit in enumerate(crit_used):
        mu, sd, look = cellwise(runs, lambda r, c=crit: M_at(r, r["T"][c]), ks, Ns)
        _, _, lab = ps.CRIT_STYLE[crit]

        for j, k in enumerate(ks):                               # M vs N, log-log
            ps.band(ax[r_i][0], Ns, mu[:, j], sd[:, j], ps.col_k(k, ks), label=f"k={k}")
        if len(Ns) > 1:
            xx = np.array(Ns, float)
            base = np.nanmedian(mu[0])
            ax[r_i][0].plot(xx, base * xx / xx[0], "k--", lw=1.1, label=r"$M\propto N$")
        ax[r_i][0].set(xscale="log", yscale="log", xlabel="N (units)", ylabel="active units M",
                       xticks=Ns, xticklabels=[str(n) for n in Ns],
                       title=f"[{lab}] M vs N — slope is the exponent b")
        ps.legend_k(ax[r_i][0], ks, loc="upper left")

        for i, N in enumerate(Ns):                               # M vs k
            ps.band(ax[r_i][1], ks, mu[i], sd[i], ps.col_n(N), label=f"N={N}")
        ax[r_i][1].set(xlabel="k (bits)", ylabel="active units M", xticks=ks,
                       title=f"[{lab}] M vs complexity")
        ax[r_i][1].legend()

        ps.contour(ax[r_i][2], ks, Ns, mu, "active units M", cmap="magma", fmt="%.0f")
        ax[r_i][2].set_title(f"[{lab}] M over the (k, N) grid")
    fig.suptitle("Active units vs size and complexity, under four read-out criteria\n"
                 "(scale-free criterion $p_i \\geq 0.05\\,q_{95}$; an ordering that survives all "
                 "four rows does not depend on when the network is read)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return ps.save(fig, "flipflop_fig3_active", tight=False)


def r2_curve(run, nbins=40):
    """Median active-unit count per R^2 bin for one run - the criterion-free M-vs-performance curve.

    Plotted directly against time-ordered R^2 this is unreadable spaghetti, because R^2 is NOT
    monotone in training time: the clean loss fluctuates, so the trajectory doubles back on itself
    many times. Binning in R^2 and taking the median M per bin gives the underlying relation.

    Args:
        run: run record; nbins: number of R^2 bins.
    Returns:
        (r2_centres, M_median) arrays over non-empty bins, ascending in R^2.
    """
    R2, M = [], []
    for i, t in enumerate(run["piters"]):
        if t < T_START:
            continue
        L = loss_at(run["loss"], t)
        if not np.isfinite(L):
            continue
        R2.append(1 - L / TGT_VAR)
        M.append(active_count(np.asarray(run["part"][i], dtype=float), "scalefree"))
    if len(R2) < 5:
        return np.array([]), np.array([])
    R2, M = np.asarray(R2), np.asarray(M, float)
    edges = np.linspace(R2.min(), R2.max(), nbins + 1)
    idx = np.clip(np.digitize(R2, edges) - 1, 0, nbins - 1)
    xs, ys = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() >= 2:
            xs.append(0.5 * (edges[b] + edges[b + 1]))
            ys.append(np.median(M[m]))
    return np.array(xs), np.array(ys)


def fig_criteria(runs, ks, Ns, crit_used, Lstar):
    """Figure 4: how the read-out choice moves the answer, plus the criterion-free M-vs-R^2 view."""
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.2))

    # ⚠️ Panels (a) and (c) must NOT average over sizes: k=1 and k=8 currently exist at N=500 only,
    # while k=2..7 have all three sizes, and M/N falls with N - so a mean over "whatever sizes this k
    # happens to have" puts the incomplete k values artificially high and manufactures a U shape that
    # is about coverage, not complexity. Both panels therefore use ONE size, the one present at every
    # k, and say so in the title.
    ref_N = min(Ns, key=lambda N: -sum(1 for k in ks if any(r["k"] == k and r["N"] == N
                                                            for r in runs)))
    ks_ref = [k for k in ks if any(r["k"] == k and r["N"] == ref_N for r in runs)]

    for crit in crit_used:                                       # (a) read-out iteration
        _, _, look = cellwise(runs, lambda r, c=crit: r["T"][c], ks, Ns)
        ys = [look[(k, ref_N)][0] if (k, ref_N) in look else np.nan for k in ks_ref]
        ls, mk, lab = ps.CRIT_STYLE[crit]
        ax[0][0].plot(ks_ref, ys, ls, marker=mk, color=f"C{CRITS.index(crit)}", label=lab)
    ax[0][0].set(yscale="log", xlabel="k (bits)", ylabel="read-out iteration T", xticks=ks_ref,
                 title=f"(a) WHEN each criterion reads the network  (N={ref_N})")
    ax[0][0].legend()

    for crit in crit_used:                                       # (b) exponent b per criterion
        _, _, look = cellwise(runs, lambda r, c=crit: M_at(r, r["T"][c]), ks, Ns)
        bs = [slope_b(look, k, Ns) for k in ks]
        ls, mk, lab = ps.CRIT_STYLE[crit]
        ax[0][1].plot(ks, bs, ls, marker=mk, color=f"C{CRITS.index(crit)}", label=lab)
    ax[0][1].axhline(1.0, color="k", lw=1, alpha=.6)
    ax[0][1].text(ks[0], 1.02, "no saturation", fontsize=7.5)
    ax[0][1].set(xlabel="k (bits)", ylabel="exponent b in $M\\sim N^b$", xticks=ks, ylim=(0, 1.15),
                 title="(b) does the CONCLUSION depend on the criterion?\n"
                       "curves close together = robust")
    ax[0][1].legend()

    for crit in crit_used:                                       # (c) active fraction
        _, _, look = cellwise(runs, lambda r, c=crit: M_at(r, r["T"][c]) / r["N"], ks, Ns)
        ys = [look[(k, ref_N)][0] if (k, ref_N) in look else np.nan for k in ks_ref]
        ls, mk, lab = ps.CRIT_STYLE[crit]
        ax[0][2].plot(ks_ref, ys, ls, marker=mk, color=f"C{CRITS.index(crit)}", label=lab)
    ax[0][2].set(xlabel="k (bits)", ylabel="active fraction M/N", xticks=ks_ref, ylim=(0, 0.75),
                 title=f"(c) active FRACTION  (N={ref_N})\n"
                       "fixed-iteration RISES with k; matched criteria are FLAT")
    ax[0][2].legend()

    for j, (which, _vals) in enumerate([("k", ks), ("N", Ns)]):   # (d, e) criterion-free M vs R^2
        a = ax[1][j]
        for r in runs:
            x, y = r2_curve(r)
            if not len(x):
                continue
            col = ps.col_k(r["k"], ks) if which == "k" else ps.col_n(r["N"])
            a.plot(x, y, "-", color=col, alpha=.55, lw=1.2)
        for crit in crit_used:            # mark where each criterion slices this plane
            lv = [1 - loss_at(r["loss"], r["T"][crit]) / TGT_VAR
                  for r in runs if np.isfinite(r["T"][crit])]
            if lv:
                a.axvline(float(np.median(lv)), color=f"C{CRITS.index(crit)}",
                          ls=ps.CRIT_STYLE[crit][0], lw=1.2, alpha=.9,
                          label=ps.CRIT_STYLE[crit][2] if j == 0 else None)
        a.set(xlabel="$R^2 = 1 - \\mathrm{MSE}/\\sigma^2_{\\mathrm{target}}$",
              ylabel="active units M", xlim=(0.80, 0.97), yscale="log",
              title=f"({'de'[j]}) M vs $R^2$ — each criterion is a VERTICAL LINE\ncoloured by {which}")
        if which == "k":
            ps.legend_k(a, ks, loc="upper right")
        else:
            ps.legend_n(a, Ns, loc="upper right")

    ax[1][2].axis("off")                                          # (f) the criterion table
    lines = ["read-out criterion          mean T   seedCV   b (mean over k)", ""]
    for crit in crit_used:
        muT, _, _ = cellwise(runs, lambda r, c=crit: r["T"][c], ks, Ns)
        _, _, look = cellwise(runs, lambda r, c=crit: M_at(r, r["T"][c]), ks, Ns)
        bs = [slope_b(look, k, Ns) for k in ks]
        lines.append(f"{ps.CRIT_STYLE[crit][2]:<26s} {np.nanmean(muT):>8.0f}   "
                     f"{_seed_cv(runs, crit):>5.2f}   {np.nanmean(bs):>5.2f} +- {np.nanstd(bs):.2f}")
    lines += ["", f"`loss` level  L* = {Lstar:.5f}",
              f"`excess` level  L = {1+EXCESS_DELTA:.2f} x own L_inf",
              f"`drift` threshold  alpha < {ALPHA_THRESH}",
              "", "b < 1 under EVERY criterion: added units are",
              "increasingly not recruited, whenever you look.",
              "", "But M vs k is NOT robust - see (c). At a fixed",
              "iteration M rises with k; at matched performance",
              "it is flat. The rise is a convergence-depth",
              "artifact: high-k cells are less converged at 150k,",
              "and less-converged networks have more active units.",
              "", f"seedCV = within-cell scatter of T across seeds;",
              "a criterion above ~0.3 is measuring noise."]
    ax[1][2].text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=8,
                  transform=ax[1][2].transAxes)

    fig.suptitle("Does the answer depend on WHEN the network is read?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return ps.save(fig, "flipflop_fig4_criteria", tight=False)


def _seed_cv(runs, crit):
    """Mean within-cell coefficient of variation of a criterion's read-out iteration.

    A read-out rule whose T scatters wildly across sibling seeds of one cell is measuring noise, not
    a state of the network - the symptom that exposed the first `drift` definition, where one cell
    returned 40k, 44k and 458k. Anything above ~0.3 deserves suspicion.

    Args:
        runs: run records carrying "T"; crit: criterion name.
    Returns:
        float mean CV over cells with at least two finite read-outs, or nan.
    """
    cells = {}
    for r in runs:
        t = r["T"][crit]
        if np.isfinite(t):
            cells.setdefault((r["k"], r["N"]), []).append(t)
    cvs = [np.std(v) / np.mean(v) for v in cells.values() if len(v) > 1 and np.mean(v) > 0]
    return float(np.mean(cvs)) if cvs else float("nan")


def main():
    """Fit every run, resolve all four read-out criteria, and emit the four figures."""
    global T_ITER
    if len(sys.argv) > 1:
        T_ITER = int(sys.argv[1])
    ps.setup()

    runs, dropped = load()
    for r in runs:
        f = fit_run(r["loss"], T_ITER)
        r["L_inf"], r["A"], r["tau"], r["beta"] = f if f else (None,) * 4
        if f:
            r["t_half"] = excess_time(*f[1:], 0.5, T_START)
    ks = sorted({r["k"] for r in runs})
    Ns = sorted({r["N"] for r in runs})
    print(f"loaded {len(runs)} runs; k={ks}; N={Ns}")
    for d in sorted(set(dropped)):
        print(f"  dropped {d}  x{dropped.count(d)}")

    # The `loss` level must be reachable by EVERY cell, so it is set from the worst floor present
    # rather than chosen by hand - otherwise the criterion silently drops the hardest cells.
    worst = max(r["L_inf"] for r in runs if r.get("L_inf"))
    Lstar = LOSS_MARGIN * worst
    readout_times(runs, Lstar)
    print(f"`loss` level L* = {Lstar:.5f}  ({LOSS_MARGIN:g} x worst floor {worst:.5f})")

    print("\ncriterion coverage (runs with a usable read-out) and mean read-out iteration:")
    crit_used = []
    for c in CRITS:
        ok = [r["T"][c] for r in runs if np.isfinite(r["T"][c])]
        if ok:
            print(f"  {c:8s} {len(ok):3d}/{len(runs)} runs   mean T = {np.mean(ok):9.0f}"
                  f"   seed CV within cells = {_seed_cv(runs, c):.2f}")
        else:
            print(f"  {c:8s}   0/{len(runs)} runs  UNUSABLE")
        if len(ok) >= 0.5 * len(runs):
            crit_used.append(c)
        else:
            print("           -> dropped from the figures: fewer than half the runs reach it")

    print("\nper-criterion exponent b in M ~ N^b (scale-free criterion):")
    print("%-28s %s" % ("criterion", "  ".join(f"k={k}" for k in ks)))
    for c in crit_used:
        _, _, look = cellwise(runs, lambda r, cc=c: M_at(r, r["T"][cc]), ks, Ns)
        bs = [slope_b(look, k, Ns) for k in ks]
        print("%-28s %s   mean %.2f +- %.2f"
              % (ps.CRIT_STYLE[c][2], "  ".join(f"{b:4.2f}" for b in bs),
                 np.nanmean(bs), np.nanstd(bs)))

    print("\nfitted floor law per N:")
    mu, _, _ = cellwise(runs, lambda r: r.get("L_inf"), ks, Ns)
    for i, N in enumerate(Ns):
        a, b, res = sqrt_law(ks, mu[i])
        if np.isfinite(a):
            print(f"  N={N:<5d}  a={a:.5f}  b={b:.5f}  max resid {100*res:.2f}%")

    fig_curves(runs, ks, Ns)
    fig_floor(runs, ks, Ns)
    fig_active(runs, ks, Ns, crit_used)
    for silence, _, _ in SILENCE_RULES:
        sl = silence if isinstance(silence, str) else f"absolute p>={silence:g}"
        print(f"\njoint power law  M = A N^b k^c   [{sl} active units]")
        print("%-26s %18s %22s %11s  %s"
              % ("criterion", "b [95% CI]", "c [95% CI]", "lack-fit p", "verdict"))
        for c in crit_used:
            f = fit_law(runs, c, silence)
            if not f:
                continue
            star = "" if (f["c_ci"][0] <= 0 <= f["c_ci"][1]) else "  <- c != 0"
            print("%-26s %5.2f [%.2f,%.2f] %8.3f [%.3f,%.3f] %10.2g  %s%s"
                  % (CRIT_DESC[c].split(chr(10))[0], f["b"], f["b_ci"][0], f["b_ci"][1],
                     f["c"], f["c_ci"][0], f["c_ci"][1], f["p"],
                     "law OK" if f["p"] > 0.05 else "REJECTED", star))

    fig_criteria(runs, ks, Ns, crit_used, Lstar)
    for silence, _, _ in SILENCE_RULES:
        fig_law(runs, ks, Ns, crit_used, silence)


if __name__ == "__main__":
    main()
