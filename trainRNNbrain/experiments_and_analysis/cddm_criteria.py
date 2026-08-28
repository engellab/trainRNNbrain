#!/usr/bin/env python3
"""
Does the CDDM saturating law survive the same read-out criteria the flip-flop was tested against?

The flip-flop result is that M ~ N^b with b ~ 0.41-0.46 under four different ways of deciding WHEN to
read a network, so the sublinear exponent is not an artifact of the read-out choice. CDDM's saturating
law was only ever measured at matched performance. This applies the same battery to CDDM so the two
tasks are compared on identical terms rather than on whatever each was originally analysed with.

CRITERIA, identical in spirit to flipflop_figures.py:
  endpoint  each run at its own final iteration.  ⚠️ budgets differ across this sweep, so this mixes
            reading depths and is reported but not trusted - exactly the defect that made the
            flip-flop endpoint fit spuriously reject its law.
  iter      every run at a common iteration, the largest budget every size reaches.
  loss      every run at a common absolute loss L*.
  excess    every run at (1 + delta) x its OWN fitted floor - the size-fair version of `loss`.
  drift     every run where weight motion stops being directed (lag exponent alpha < 0.6).

⚠️ THE LOSS USED HERE IS NOT NOISE-FREE. CDDM traces predate `loss_clean_train`, so the only loss on
disk is TrainLosses.json, the optimiser objective evaluated with noise ON. That is acceptable HERE
only because this entire sweep is unpenalised (lambda = 0), so the objective IS the task loss and the
comparison is within one condition. It must never be used to compare across penalty conditions - the
standing rule in README.md - and the flip-flop numbers it is set beside come from a clean forward
pass, so the two are not interchangeable at the level of an absolute loss value. What IS comparable
is the EXPONENT b, which is what this script reports.

Both silence criteria are reported. On CDDM they broadly agree, unlike on the flip-flop where the
absolute rule reports ~0% silence because that task's silent mode sits at 2e-4 rather than at zero.

Output: img/internal_figures/cddm_criteria.png

Usage:  python cddm_criteria.py [SWEEP_FOLDER]
"""

import os
import sys
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (IMG_DIR, active_count, diffusive_onset, load_traces, logbin, stretched)
import plotstyle as ps

DEFAULT_SWEEP = "data/trained_RNNs/CDDM_std_g0_drift"
T_START = 2000
EXCESS_DELTA = 0.10
LOSS_MARGIN = 1.10
ALPHA_THRESH = 0.6
CRITS = ["endpoint", "iter", "loss", "excess", "drift"]


def fit_floor(L, t_end):
    """Stretched-exponential fit of one loss trace over [T_START, t_end], in log space.

    Args:
        L: per-iteration loss array; t_end: last iteration included.
    Returns:
        (L_inf, A, tau, beta) or None if the fit fails.
    """
    t = np.arange(1, len(L) + 1, dtype=float)
    m = (t >= T_START) & (t <= t_end)
    if m.sum() < 100:
        return None
    tb, yb = logbin(t[m], L[m])
    if len(tb) < 8:
        return None
    try:
        sol = least_squares(
            lambda p: np.log(np.clip(stretched(tb, *p), 1e-12, None)) - np.log(yb),
            [yb.min() * 0.9, float(yb.max()), 2e4, 0.4],
            bounds=([1e-9, 1e-9, 1e2, 0.05], [1.0, 1e3, 1e8, 3.0]), max_nfev=20000)
    except Exception:
        return None
    return tuple(sol.x)


def first_below(L, target, w=201):
    """First iteration at which the smoothed loss reaches `target`, or nan.

    Args:
        L: per-iteration loss; target: level; w: running-mean width, since the raw loss is noisy.
    Returns:
        float iteration or nan.
    """
    if len(L) < w:
        return float("nan")
    s = np.convolve(L, np.ones(w) / w, mode="valid")
    hit = np.flatnonzero(s <= target)
    return float(hit[0] + w // 2 + 1) if len(hit) else float("nan")


def M_at(trace, t, criterion):
    """Active-unit count at the last participation snapshot at or before iteration t.

    Args:
        trace: a CDDM trace dict; t: iteration (nan gives nan); criterion: "hard" or "scalefree".
    Returns:
        float count or nan.
    """
    if not np.isfinite(t):
        return float("nan")
    I = np.asarray(trace["participation_iters"], dtype=float)
    ok = np.flatnonzero(I <= t)
    if ok.size == 0:
        return float("nan")
    return float(active_count(np.asarray(trace["participation"][int(ok[-1])], dtype=float),
                              criterion))


def main():
    """Read CDDM at every criterion and report the M ~ N^b exponent for each."""
    ps.setup()
    sweep = (sys.argv[1:] or [DEFAULT_SWEEP])[0]
    by = load_traces(sweep)
    by = {N: [t for t in ts if t.get("loss") is not None and not np.isnan(t["loss"]).any()]
          for N, ts in by.items()}
    by = {N: ts for N, ts in by.items() if ts}
    if not by:
        raise SystemExit(f"no usable traces under {sweep}")
    Ns = sorted(by)
    budgets = {N: min(len(t["loss"]) for t in by[N]) for N in Ns}
    T_ITER = min(budgets.values())
    print(f"sizes {Ns}; per-size budget {budgets}; common iteration T_ITER = {T_ITER}")
    print("⚠️ loss is TrainLosses.json (noise ON); valid here only because lambda = 0 throughout.\n")

    rows = []
    for N in Ns:
        for t in by[N]:
            f = fit_floor(t["loss"], T_ITER)
            rows.append(dict(N=N, trace=t, L=t["loss"], budget=len(t["loss"]),
                             L_inf=f[0] if f else None))
    worst = max(r["L_inf"] for r in rows if r["L_inf"])
    Lstar = LOSS_MARGIN * worst
    for r in rows:
        T = {"endpoint": float(r["budget"]),
             "iter": float(T_ITER),
             "loss": first_below(r["L"], Lstar),
             "excess": (first_below(r["L"], (1 + EXCESS_DELTA) * r["L_inf"])
                        if r["L_inf"] else float("nan")),
             "drift": diffusive_onset(r["trace"], "W_rec", thresh=ALPHA_THRESH)}
        for c, v in T.items():
            if not np.isfinite(v) or v > r["budget"]:
                T[c] = float("nan")
        r["T"] = T
    print(f"`loss` level L* = {Lstar:.5f}  ({LOSS_MARGIN:g} x worst fitted floor {worst:.5f})")

    # ⚠️ GUARD: a criterion that fires on the FIRST available probe is not measuring a transition,
    # it is reporting where the trace starts. On CDDM the lag exponent alpha is already ~0.5 by the
    # second probe for every N >= 500, so `drift` returns the first probe and the resulting b just
    # says "nothing has silenced yet". Detect that and disqualify the criterion rather than quoting
    # a number that looks like a finding.
    disq = {}
    for c in CRITS:
        Ts = [r["T"][c] for r in rows if np.isfinite(r["T"][c])]
        if not Ts:
            disq[c] = "no run reaches it"
            continue
        floors = [min(r["trace"]["participation_iters"]) for r in rows]
        at_first = np.mean([t <= 1.5 * f for t, f in zip(Ts, floors)])
        if c != "iter" and at_first > 0.5:
            disq[c] = (f"fires at the first probe for {100*at_first:.0f}% of runs — reports where "
                       f"the trace starts, not a transition")
    for c, why in disq.items():
        print(f"⚠️  DISQUALIFIED `{c}`: {why}")

    res = {}
    for silence in ("scalefree", "hard"):
        print(f"\n=== CDDM,  M ~ N^b   [{silence} active units] ===")
        print("%-10s %8s %10s   %s" % ("criterion", "mean T", "b", "M per N"))
        for c in CRITS:
            if c in disq:
                continue
            pts = {}
            for r in rows:
                m = M_at(r["trace"], r["T"][c], silence)
                if np.isfinite(m):
                    pts.setdefault(r["N"], []).append(m)
            if len(pts) < 2:
                print("%-10s   UNUSABLE (fewer than two sizes reach it)" % c)
                continue
            xs = sorted(pts)
            mu = [np.mean(pts[N]) for N in xs]
            b = float(np.polyfit(np.log(xs), np.log(mu), 1)[0])
            Ts = [r["T"][c] for r in rows if np.isfinite(r["T"][c])]
            res[(silence, c)] = (xs, mu, [np.std(pts[N]) for N in xs], b)
            print("%-10s %8.0f %10.2f   %s"
                  % (c, np.mean(Ts), b, "  ".join(f"{N}:{m:.0f}" for N, m in zip(xs, mu))))

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for j, silence in enumerate(("scalefree", "hard")):
        for c in CRITS:
            if (silence, c) not in res:
                continue
            xs, mu, sd, b = res[(silence, c)]
            ls, mk, lab = ps.CRIT_STYLE[c]
            ax[j].errorbar(xs, mu, yerr=sd, fmt=ls + mk, color=f"C{CRITS.index(c)}",
                           label=f"{lab}: b={b:.2f}", capsize=2.5)
        xx = np.array(sorted({N for (s, c), v in res.items() for N in v[0]}), float)
        if len(xx) > 1:
            base = min(v[1][0] for (s, c), v in res.items() if s == silence)
            ax[j].plot(xx, base * xx / xx[0], "k--", lw=1.1, label=r"$M\propto N$ (no saturation)")
        ax[j].set(xscale="log", yscale="log", xlabel="N (units)", ylabel="active units M",
                  title=f"CDDM — {silence} criterion")
        ax[j].legend(fontsize=7)
    fig.suptitle("Does the CDDM saturating law depend on the read-out criterion?\n"
                 "Same battery as the flip-flop (which gave b = 0.41–0.46 across criteria). "
                 "⚠️ loss here is the noisy optimiser objective; only λ=0 runs, so it is the task loss.",
                 fontsize=11)
    return ps.save(fig, "cddm_criteria")


if __name__ == "__main__":
    main()
