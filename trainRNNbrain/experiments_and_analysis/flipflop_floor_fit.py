#!/usr/bin/env python3
"""Loss floor per (penalty, N, k), from a stretched exponential with tau FIXED to 1 iteration.

MODEL.  L(t) = L_inf + A * exp(-(t/tau)^beta) with tau == 1, i.e. L_inf + A*exp(-t^beta).
Fixing tau removes the A/tau degeneracy that makes the free 4-parameter form unidentifiable once
tau falls below the fit start. It is a genuine constraint, not a reparameterisation: with the time
unit pinned to one iteration, beta has to absorb the entire timescale.

⚠️ THE USABLE beta RANGE IS NARROW. t reaches 4e5, so t^beta = 13 at beta=0.2 but 48 at beta=0.3,
where exp(-t^beta) ~ 1e-21 and the exponential is identically zero past t~100 - at which point A
and beta stop being identifiable and the degeneracy tau=1 was meant to remove comes back. Measured
fits land at beta ~ 0.19-0.21, right at that edge, so every fit is checked for collapse
(L_inf pinned at 0, or beta on a bound) and refitted from several starts before being accepted.

⚠️ FIT ON A LOG-BINNED TRACE. The raw trace is 40000 points sampled every 10 iterations, so ~90%
of them sit in the last decade of time; unweighted least squares would let the tail dictate the
shape and bias L_inf. Binning uniformly in log t gives each decade equal say.

Per run -> L_inf. Per cell -> mean +/- sd over the 3 seeds, so a single bad seed is visible rather
than silently averaged in.

Output: img/internal_figures/flipflop_floor_fit.png  (+ a CSV of per-run fits)

Usage:  python flipflop_floor_fit.py [T_MIN]        (default T_MIN = 500)
"""

import os
import re
import sys
import glob
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, DATA_DIR, logbin
import plotstyle as ps

ROOTS = ("NBitFlipFlop_std_ksweep", "NBitFlipFlop_std_pen",
         "NBitFlipFlop_std_penlong", "NBitFlipFlop_std_bigN")
PENS = ("none", "rws", "frm", "both")
NS = (500, 1000, 2000, 4000)
KS = tuple(range(1, 9))
R2_MIN = 0.5          # below this the run never learned the task; see the score gap -0.32 .. 0.857
MIN_ITERS = 50_000    # anything shorter is a timing-calibration run, not an experiment
T_MIN = 500           # fit start; excludes the initial transient
NBINS = 80
BETA_STARTS = (0.08, 0.15, 0.20, 0.30, 0.50)


def model(t, L_inf, A, beta):
    """Stretched exponential with tau fixed to 1: L_inf + A*exp(-t^beta).

    Args:
        t: iteration numbers (array, >0).
        L_inf: asymptotic loss floor.
        A: amplitude of the decaying part.
        beta: stretching exponent (dimensionless).
    Returns:
        Predicted loss, same shape as t. The exponent is clipped at 700 to avoid overflow.
    """
    return L_inf + A * np.exp(-np.clip(np.power(t, beta), 0.0, 700.0))


def fit_one(t, y):
    """Fit the tau=1 stretched exponential to one binned loss curve, with multi-start.

    Args:
        t: binned iteration centres (1-D array).
        y: binned loss values (1-D array, same length).
    Returns:
        dict with L_inf, A, beta, rmse, ok (bool), why (str). `ok` is False when the fit
        collapsed - L_inf driven to ~0, or beta pinned on a bound - which means the curve
        carried no usable curvature in the window.
    """
    # ⚠️ L_inf CANNOT EXCEED THE SMALLEST OBSERVED LOSS. The model decays monotonically down to
    # L_inf, so any fit placing the floor above min(y) is unphysical. Without this bound the
    # optimiser happily returns L_inf = 4.5 or 291 on a curve whose data never leaves 0.02-0.6,
    # letting the exponential run the wrong way; those fits then poison the cell mean.
    # ⚠️ BOUND L_inf BY THE OBSERVED RANGE. The curve decays monotonically down to L_inf, so a
    # floor above the largest observed loss is unphysical; unbounded, the optimiser returns
    # L_inf = 4.5 or 291 on data that never leaves 0.02-0.6 and poisons the cell mean.
    # Do NOT bound by min(y): the binned curve dips below its own asymptote on noise, so that
    # bound pins every fit and rejects almost everything (232/319 when tried).
    # ⚠️ BOUND L_inf BY THE OBSERVED TAIL, NOT BY max(y). The curve decays down to L_inf, so the
    # floor must sit near the end of the trace. Bounding by max(y) is not enough for frm runs near
    # instability: their loss SPIKES to 1e5, so max(y) is huge and a fit returned L_inf = 440.
    # Bounding by min(y) is too tight (the binned curve dips below its own asymptote on noise and
    # 232/319 fits got pinned). The tail median is robust to both.
    lo = [0.0, 0.0, 0.01]
    hi = [2.0 * float(np.median(y[-10:])), np.inf, 1.0]
    best = None
    for b0 in BETA_STARTS:
        # p0 must lie inside the bounds or curve_fit refuses to start ("no convergence").
        p0 = [float(np.clip(y[-1], 1e-9, hi[0])), max(y[0] - y[-1], 1e-9), b0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, _ = curve_fit(model, t, y, p0=p0, bounds=(lo, hi), maxfev=100_000)
        except Exception:
            continue
        rmse = float(np.sqrt(np.mean((model(t, *p) - y) ** 2)))
        if best is None or rmse < best[1]:
            best = (p, rmse)
    if best is None:
        return dict(L_inf=np.nan, A=np.nan, beta=np.nan, rmse=np.nan, ok=False, why="no convergence")
    p, rmse = best
    # The decaying term must still be alive at the start of the window, else the "fit" is a
    # flat line through the mean and L_inf is just min(y): beta near 1 gives exp(-t_min^beta) ~ 0
    # at t_min=500, so the exponential explains nothing and beta/A are meaningless.
    alive = p[1] * float(np.exp(-min(t[0] ** p[2], 700.0)))
    span = float(y[0] - y[-1])
    why = ""
    if p[0] <= 1e-6 * max(y.max(), 1e-12):
        why = "L_inf collapsed to 0"
    elif p[2] <= 0.011 or p[2] >= 0.999:
        why = f"beta pinned at bound ({p[2]:.3f})"
    elif span > 0 and alive < 0.01 * span:
        why = f"exponential dead at t_min (beta={p[2]:.3f}); fit is a flat line"
    elif p[0] >= hi[0] * 0.999:
        why = "L_inf pinned at 2x tail median; no asymptote below the tail"
    return dict(L_inf=float(p[0]), A=float(p[1]), beta=float(p[2]),
                rmse=rmse, ok=(why == ""), why=why)


def parse_cell(cell):
    """Pull (penalty, N, k, iters) out of a cell directory name, or None if it is not a grid cell."""
    m = re.search(r"_k=(\d+)_N=(\d+)", cell)
    if not m:
        return None
    pm = re.search(r"pen=([a-z]+)", cell)
    im = re.search(r"iters=(\d+)", cell)
    return (pm.group(1) if pm else "none", int(m.group(2)), int(m.group(1)),
            int(im.group(1)) if im else 0)


def load_runs(t_min):
    """Fit every usable run in the local mirror.

    Args:
        t_min: first iteration included in the fit.
    Returns:
        list of dicts, one per run: penalty, N, k, r2, plus the fit fields from `fit_one`.
    """
    out = []
    for R in ROOTS:
        for f in sorted(glob.glob(os.path.join(DATA_DIR, R, "*", "*", "*ParticipationTrace.pkl"))):
            cell = f.split(os.sep)[-3]
            key = parse_cell(cell)
            if key is None:
                continue
            pen, N, k, iters = key
            if R == "NBitFlipFlop_std_pen" and pen == "frm":
                continue                      # the 150k frm sweep is retracted (not converged)
            if iters and iters < MIN_ITERS:
                continue
            if N not in NS:
                continue
            try:
                r2 = float(os.path.basename(f).split("_")[0])
            except ValueError:
                continue
            if not (r2 >= R2_MIN):
                continue                      # failed run: never learned the task
            with open(f, "rb") as fh:
                tr = pickle.load(fh)
            L = np.asarray(tr["metrics"]["loss_clean_train"], dtype=float)
            if L.size * 10 < MIN_ITERS or not np.isfinite(L).any():
                continue
            t = (np.arange(L.size) + 1) * 10
            tb, yb = logbin(t, L, nbins=NBINS, t_min=t_min)
            m = np.isfinite(tb) & np.isfinite(yb)
            if m.sum() < 10:
                continue
            rec = dict(pen=pen, N=N, k=k, r2=r2)
            rec.update(fit_one(tb[m], yb[m]))
            out.append(rec)
    return out


def main():
    """Fit every run, aggregate per cell, and report floors with seed-to-seed error bars."""
    t_min = float(sys.argv[1]) if len(sys.argv) > 1 else T_MIN
    runs = load_runs(t_min)
    bad = [r for r in runs if not r["ok"]]
    print(f"fitted {len(runs)} runs (t_min={t_min:g}, tau=1, log-binned to {NBINS} bins)")
    print(f"degenerate fits: {len(bad)}")
    for r in bad[:12]:
        print(f"   {r['pen']:5s} N={r['N']:<5d} k={r['k']}  r2={r['r2']:.3f}  {r['why']}")
    good = [r for r in runs if r["ok"]]
    b = np.array([r["beta"] for r in good])
    print(f"\nbeta over accepted fits: median {np.median(b):.3f}  "
          f"range {b.min():.3f}-{b.max():.3f}   (usable range is beta <~0.25)")

    print(f"\n{'pen':5s} {'N':>5s} " + "".join(f"{'k='+str(k):>14}" for k in KS))
    cells = {}
    for pen in PENS:
        for N in NS:
            row = []
            any_ = False
            for k in KS:
                v = [r["L_inf"] for r in good if (r["pen"], r["N"], r["k"]) == (pen, N, k)]
                if v:
                    any_ = True
                    cells[(pen, N, k)] = (float(np.mean(v)), float(np.std(v)), len(v))
                    row.append(f"{np.mean(v):.4f}±{np.std(v):.4f}")
                else:
                    row.append("·")
            if any_:
                print(f"{pen:5s} {N:5d} " + "".join(f"{c:>14}" for c in row))

    # ---- floor law: floor(k) = a + b*sqrt(k), fitted within each (penalty, N) ----------------
    print(f"\n{'pen':5s} {'N':>5s} {'a (1-channel floor)':>21} {'b (interference)':>18}")
    for pen in PENS:
        for N in NS:
            kk = [k for k in KS if (pen, N, k) in cells]
            if len(kk) < 4:
                continue
            y = np.array([cells[(pen, N, k)][0] for k in kk])
            X = np.column_stack([np.ones(len(kk)), np.sqrt(kk)])
            (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
            print(f"{pen:5s} {N:5d} {a:21.5f} {b:18.5f}")

    # ---- figure ------------------------------------------------------------------------------
    ps.setup()
    fig, ax = plt.subplots(1, len(PENS), figsize=(4.2 * len(PENS), 4.0), sharey=True, squeeze=False)
    for c, pen in enumerate(PENS):
        a_ = ax[0][c]
        for N in NS:
            kk = [k for k in KS if (pen, N, k) in cells]
            if not kk:
                continue
            m = np.array([cells[(pen, N, k)][0] for k in kk])
            e = np.array([cells[(pen, N, k)][1] for k in kk])
            a_.errorbar(kk, m, yerr=e, marker="o", ms=4, capsize=3, lw=1.4,
                        color=ps.col_n(N), label=f"N={N}")
        a_.set(xlabel="k (bits)", xticks=list(KS), title=pen)
        if c == 0:
            a_.set_ylabel(r"fitted loss floor $L_\infty$")
        a_.legend(fontsize=7)
    fig.suptitle(r"Loss floor from $L(t)=L_\infty+A\,e^{-t^{\beta}}$ ($\tau$ fixed to 1), "
                 f"fit from t={t_min:g}\n"
                 "error bars = sd over 3 seeds  ·  per-run fits, then averaged", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    print("\nfigure ->", ps.save(fig, "flipflop_floor_fit", tight=False))

    outp = os.path.join(IMG_DIR, "flipflop_floor_fit_runs.csv")
    os.makedirs(IMG_DIR, exist_ok=True)
    with open(outp, "w") as fh:
        fh.write("pen,N,k,r2,L_inf,A,beta,rmse,ok,why\n")
        for r in runs:
            fh.write(f"{r['pen']},{r['N']},{r['k']},{r['r2']:.6f},{r['L_inf']:.6g},"
                     f"{r['A']:.6g},{r['beta']:.6g},{r['rmse']:.6g},{int(r['ok'])},{r['why']}\n")
    print(f"\nper-run fits -> {outp}")
    return cells, runs


if __name__ == "__main__":
    main()
