#!/usr/bin/env python3
"""Power law or saturation? Whether the active-unit count keeps growing with N or hits a ceiling.

THE QUESTION.  M = A*N^b says a bigger network always recruits proportionally more units, forever.
A saturating law says recruitment approaches a hard ceiling M_max set by the task, not the network.
With only N = 500/1000/2000 the two are indistinguishable - a power law with b < 1 and a saturating
curve well below its ceiling trace nearly the same path. N=4000 is what separates them, which is
why those runs exist.

TWO TESTS, one assumption-light and one parametric:

 (1) LOCAL LOG-SLOPE between adjacent sizes, d log M / d log N. A pure power law has a CONSTANT
     slope, b, at every size. Saturation has a slope that FALLS monotonically toward zero. This
     needs no model and no fit, so it cannot be an artefact of a chosen functional form.

 (2) AICc between  M = A*N^b*k^c  and  M = M_max*(N/(N+N0))*k^c  - both 3 parameters, so the
     comparison is not a complexity trade. Fits are in log M, so residuals are relative.

⚠️ N=4000 RUNS ONLY 100k ITERATIONS against 400-500k elsewhere. An under-trained network would show
a LOW M and fake a ceiling, so M must NOT be read at the endpoint. Everything here is read at the
`excess` criterion - the iteration where the loss first reaches (1+delta) x that run's OWN fitted
floor - which is budget-independent. Verified reachable: the N=4000 cells hit it at 33-46% of their
budget, so they are not censored.

⚠️ ONLY `none` AND `rws` HAVE N=4000. frm and both stop at 2000 and cannot be tested for a ceiling.

Output: img/internal_figures/flipflop_ceiling.png

Usage:  python flipflop_ceiling.py [EXCESS_DELTA] [criterion]
        criterion: hard | scalefree | pr        (default hard)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, active_count, participation_ratio, aicc, SILENT_FLIPFLOP
import plotstyle as ps
import pr_matrix as P

NS = (500, 1000, 2000, 4000)


def power(X, logA, b, c):
    """log M under M = A N^b k^c.  X is (log N, log k) stacked."""
    return logA + b * X[0] + c * X[1]


def satur(X, logMmax, logN0, c):
    """log M under M = M_max (N/(N+N0)) k^c - a hard ceiling M_max approached with scale N0."""
    N, k = np.exp(X[0]), np.exp(X[1])
    return logMmax + np.log(N / (N + np.exp(logN0))) + c * np.log(k)


def fit_pair(N, k, M):
    """Fit both laws to (N, k, M) in log space.

    Args:
        N, k, M: 1-D arrays of equal length, one entry per run.
    Returns:
        dict with each law's parameters, rss and AICc, plus the AICc difference.
    """
    X = np.vstack([np.log(N), np.log(k)])
    y = np.log(M)
    out = {}
    for name, f, p0 in (("power", power, [np.log(M.mean()), 0.5, 0.0]),
                        ("satur", satur, [np.log(2 * M.max()), np.log(2000.0), 0.0])):
        try:
            p, _ = curve_fit(f, X, y, p0=p0, maxfev=200_000)
            rss = float(np.sum((f(X, *p) - y) ** 2))
            out[name] = dict(p=p, rss=rss, aicc=aicc(rss, len(y), len(p)))
        except Exception as e:
            out[name] = dict(p=None, rss=np.inf, aicc=np.inf)
    out["dAICc"] = out["satur"]["aicc"] - out["power"]["aicc"]
    return out


def main():
    """Compare power-law and saturating fits to the active-unit count, per penalty."""
    delta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
    crit = sys.argv[2] if len(sys.argv) > 2 else "hard"
    P.EXCESS_DELTA = delta
    fn = {"hard": lambda p: active_count(p, SILENT_FLIPFLOP),
          "scalefree": lambda p: active_count(p, "scalefree"),
          "pr": participation_ratio}[crit]

    runs = P.load()
    for r in runs:
        r["floor"] = P.fit_floor(r["loss"], r["budget"])
        r["T"] = P.excess_time(r["loss"], r["floor"], delta) if r["floor"] else float("nan")
        r["M"] = P.measure(r, fn)
    runs = [r for r in runs if np.isfinite(r["M"]) and r["M"] > 0]
    print(f"read-out: loss reaches {1+delta:.2f}x each run's OWN floor   criterion: {crit}")
    print(f"{len(runs)} runs\n")

    # ---- (1) local log-slope between adjacent sizes -------------------------------------------
    print("LOCAL SLOPE  d log M / d log N  between adjacent sizes")
    print("  constant => power law   ·   falling => saturation\n")
    print(f"{'pen':6s} " + "".join(f"{f'{a}->{b}':>14}" for a, b in zip(NS, NS[1:])))
    med = {}
    for pen in ("none", "rws", "frm", "both"):
        for N in NS:
            v = [r["M"] for r in runs if r["pen"] == pen and r["N"] == N]
            if v:
                med[(pen, N)] = float(np.median(v))
        row = []
        for a, b in zip(NS, NS[1:]):
            if (pen, a) in med and (pen, b) in med:
                s = np.log(med[(pen, b)] / med[(pen, a)]) / np.log(b / a)
                row.append(f"{s:14.3f}")
            else:
                row.append(f"{'-':>14}")
        print(f"{pen:6s} " + "".join(row))

    # ---- (2) power vs saturation, AICc --------------------------------------------------------
    print(f"\n{'pen':6s} {'sizes':>6} {'n':>5} {'b (power)':>11} {'M_max (sat)':>12} "
          f"{'N0':>8} {'dAICc':>9}  verdict")
    for pen in ("none", "rws", "frm", "both"):
        sel = [r for r in runs if r["pen"] == pen]
        if len(sel) < 12:
            continue
        N = np.array([r["N"] for r in sel], float)
        k = np.array([r["k"] for r in sel], float)
        M = np.array([r["M"] for r in sel], float)
        f = fit_pair(N, k, M)
        nsz = len(set(N))
        b = f["power"]["p"][1] if f["power"]["p"] is not None else np.nan
        if f["satur"]["p"] is not None:
            mmax, n0 = np.exp(f["satur"]["p"][0]), np.exp(f["satur"]["p"][1])
        else:
            mmax = n0 = np.nan
        d = f["dAICc"]
        verdict = ("power favoured" if d > 2 else
                   "saturation favoured" if d < -2 else "indistinguishable")
        if nsz < 4:
            verdict += "  (only 3 sizes - cannot separate)"
        print(f"{pen:6s} {nsz:6d} {len(sel):5d} {b:11.3f} {mmax:12.0f} {n0:8.0f} {d:+9.1f}  {verdict}")

    # ---- figure --------------------------------------------------------------------------------
    ps.setup()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for pen, col in (("none", "C0"), ("rws", "C1"), ("frm", "C2"), ("both", "C3")):
        xs = [N for N in NS if (pen, N) in med]
        if not xs:
            continue
        ys = [med[(pen, N)] for N in xs]
        ax[0].plot(xs, ys, "o-", color=col, label=pen)
        sl = [np.log(med[(pen, b)] / med[(pen, a)]) / np.log(b / a)
              for a, b in zip(xs, xs[1:])]
        ax[1].plot([np.sqrt(a * b) for a, b in zip(xs, xs[1:])], sl, "o-", color=col, label=pen)
    ax[0].set(xscale="log", yscale="log", xlabel="N (units)",
              ylabel=f"active units ({crit})", title="M vs N  (log-log)")
    ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set(xscale="log", xlabel="N (geometric mean of the pair)",
              ylabel=r"local slope  $d\log M/d\log N$",
              title="constant = power law   ·   falling = saturation")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Power law or ceiling?  active units read at {1+delta:.2f}x each run's own floor "
                 f"({crit})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    print("\nfigure ->", ps.save(fig, "flipflop_ceiling", tight=False))


if __name__ == "__main__":
    main()
