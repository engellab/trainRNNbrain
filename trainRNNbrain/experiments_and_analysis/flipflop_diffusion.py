#!/usr/bin/env python3
"""
How robust is the diffusion read-out? Multiple drift VARIABLES x multiple THRESHOLDS.

The drift criterion is the only one of the five that needs no threshold on the loss and no fitted
floor: it reads each network at the point where its updates stop carrying it somewhere new. That
makes it the most independent check on M(N, k) - and also the one whose definition has already
produced a wrong-looking answer twice, so it earns the most scrutiny rather than the least.

Displacement over a lag L grows as L^alpha, with alpha ~ 1 for directed motion, 0.5 for an unbiased
random walk and < 0.5 for confined motion. This script varies everything the read-out depends on:

  VARIABLES   W_rec, W_inp, W_out  - the three weight matrices, each of which could settle at a
                                     different time (the input weights grow 54x on this task while
                                     the recurrent ones barely move, so they need not agree).
              p                    - the PARTICIPATION vector (logged as dp_lag*). This is the
                                     quantity the conclusion is actually about, so its own settling
                                     time is the most direct criterion available.
              M(t)                 - the active-unit COUNT as a scalar trajectory, via
                                     common.scalar_alpha. Uses the task-calibrated threshold, since
                                     M(t) under p<1e-6 is a flat line on this task and has no
                                     dynamics to measure.
  THRESHOLDS  0.50 .. 0.80 on alpha.

ROBUSTNESS OF ALPHA ITSELF, checked before any of it is used. A single alpha from three lags hides
whether one power law describes the whole range, so alpha is also computed separately over each
adjacent decade (100->1000, 1000->10000). If those disagree, the pooled alpha is an average over two
regimes and every threshold on it inherits that.

EVERY COMBINATION IS SCORED BY THE SEED CV OF ITS READ-OUT TIME. A rule whose T scatters across
sibling seeds of one cell is measuring noise - that is exactly how the first drift definition failed,
returning 40k, 44k and 458k for one cell. Combinations above CV_MAX are reported and excluded.

Output: img/internal_figures/flipflop_diffusion.png

Usage:  python flipflop_diffusion.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (IMG_DIR, active_count, diffusive_onset, drift_alpha, drift_alpha_pairwise,
                    scalar_alpha)
import plotstyle as ps
import flipflop_figures as F

VARS = ["W_rec", "W_inp", "W_out", "p", "M"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.70, 0.80]
CV_MAX = 0.35            # seed CV of the read-out time above which a combination is not trusted
VAR_LABEL = {"W_rec": "$W_{rec}$", "W_inp": "$W_{inp}$", "W_out": "$W_{out}$",
             "p": "participation vector", "M": "active-unit count $M(t)$"}


def alphas(runs):
    """Precompute the alpha trajectory of every variable for every run.

    Args:
        runs: run records (must carry `part`, `piters`, `trace`).
    Returns:
        None; each run gains `alpha` = {variable: (iters, alpha)}.
    """
    for r in runs:
        a = {v: drift_alpha(r["trace"], v) for v in ("W_rec", "W_inp", "W_out", "p")}
        M = np.array([active_count(np.asarray(p, dtype=float), F.ABS_THRESH) for p in r["part"]],
                     dtype=float)
        a["M"] = scalar_alpha(r["piters"], M)
        r["alpha"] = a


def report_alpha_robustness(runs):
    """Print how well-determined alpha is: per-decade agreement and seed spread.

    Args:
        runs: run records carrying `alpha`.
    Returns:
        dict {variable: median |alpha(100->1000) - alpha(1000->10000)| at the end of training}.
    """
    print("ALPHA ROBUSTNESS — is one power law enough, and do seeds agree?")
    print("%-22s %14s %16s %18s" % ("variable", "final alpha", "seed sd", "decade disagreement"))
    dis = {}
    for v in VARS:
        fin = [r["alpha"][v][1][-1] for r in runs if len(r["alpha"][v][1])]
        if v == "M":
            d = [np.nan]
        else:
            d = []
            for r in runs:
                pw = drift_alpha_pairwise(r["trace"], v)
                if len(pw) == 2:
                    (a1, a2) = [np.median(a[1][-5:]) for a in pw.values()]
                    d.append(abs(a1 - a2))
        dis[v] = float(np.nanmedian(d)) if d else np.nan
        flag = "" if not np.isfinite(dis[v]) or dis[v] < 0.15 else "   <- one power law is NOT enough"
        print("%-22s %14.3f %16.3f %18s%s"
              % (v, np.median(fin) if fin else np.nan, np.std(fin) if fin else np.nan,
                 f"{dis[v]:.3f}" if np.isfinite(dis[v]) else "n/a (scalar)", flag))
    return dis


def slope_bc(runs, key):
    """Fit M = A N^b k^c at one read-out, returning (b, c, n_used).

    Args:
        runs: run records; key: name of the per-run read-out iteration stored in r["Tdiff"].
    Returns:
        (b, c, n) or (nan, nan, 0) if too few usable points.
    """
    K, NN, M = [], [], []
    for r in runs:
        t = r["Tdiff"].get(key, np.nan)
        m = F.M_at(r, t, "scalefree")
        if np.isfinite(m) and m > 0:
            K.append(r["k"]); NN.append(r["N"]); M.append(m)
    if len(M) < 8 or len(set(NN)) < 2 or len(set(K)) < 3:
        return np.nan, np.nan, len(M)
    X = np.column_stack([np.ones(len(M)), np.log(NN), np.log(K)])
    beta, *_ = np.linalg.lstsq(X, np.log(M), rcond=None)
    return float(beta[1]), float(beta[2]), len(M)


def seed_cv(runs, key):
    """Mean within-cell coefficient of variation of a read-out time.

    Args:
        runs: run records carrying `Tdiff`; key: read-out name.
    Returns:
        float mean CV over cells with >= 2 finite read-outs, or nan.
    """
    cells = {}
    for r in runs:
        t = r["Tdiff"].get(key, np.nan)
        if np.isfinite(t):
            cells.setdefault((r["k"], r["N"]), []).append(t)
    cvs = [np.std(v) / np.mean(v) for v in cells.values() if len(v) > 1 and np.mean(v) > 0]
    return float(np.mean(cvs)) if cvs else float("nan")


def fit_settling_law(runs, key, n_boot=1000):
    """Fit the SETTLING TIME itself as T = A N^beta k^gamma, with bootstrap CIs.

    Everything else in this project asks how the active-unit COUNT depends on N and k. This asks how
    the DYNAMICS do: how much longer does a network keep moving directionally when the task carries
    more bits, or when the network is larger. gamma is the direct measurement of the mechanism that
    was inferred earlier from the fact that compute-based read-outs find c > 0 while matched ones
    find c ~ 0 - "harder tasks converge slower" - so it should come out clearly positive.

    Args:
        runs: run records carrying "Tdiff"; key: "<variable>@<threshold>"; n_boot: resamples.
    Returns:
        dict with beta, gamma, their 95% CIs, n, and the implied T(k=8)/T(k=1) ratio; or None.
    """
    K, NN, T = [], [], []
    for r in runs:
        t = r["Tdiff"].get(key, np.nan)
        if np.isfinite(t) and t > 0:
            K.append(r["k"]); NN.append(r["N"]); T.append(t)
    if len(T) < 10 or len(set(NN)) < 2 or len(set(K)) < 3:
        return None
    K, NN, T = np.array(K, float), np.array(NN, float), np.array(T, float)
    y = np.log(T)
    X = np.column_stack([np.ones(len(y)), np.log(NN), np.log(K)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(K[i])) < 3 or len(np.unique(NN[i])) < 2:
            continue
        Xi = np.column_stack([np.ones(len(y)), np.log(NN[i]), np.log(K[i])])
        try:
            bs.append(np.linalg.lstsq(Xi, y[i], rcond=None)[0])
        except np.linalg.LinAlgError:
            continue
    bs = np.array(bs)
    ci = lambda j: (float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5)))
    kmax = K.max() / K.min()
    return dict(beta=float(beta[1]), gamma=float(beta[2]), beta_ci=ci(1), gamma_ci=ci(2),
                n=len(y), kratio=kmax ** float(beta[2]),
                kratio_ci=(kmax ** ci(2)[0], kmax ** ci(2)[1]), kmax=kmax)


def main():
    """Sweep drift variable x threshold, score each combination, and report b and c."""
    ps.setup()
    runs, _ = F.load()
    for r in runs:
        f = F.fit_run(r["loss"], F.T_ITER)
        r["L_inf"] = f[0] if f else None
    worst = max(r["L_inf"] for r in runs if r["L_inf"])
    F.readout_times(runs, F.LOSS_MARGIN * worst)
    alphas(runs)
    ks = sorted({r["k"] for r in runs})
    Ns = sorted({r["N"] for r in runs})
    print(f"{len(runs)} runs, k={ks}, N={Ns}\n")

    dis = report_alpha_robustness(runs)

    for r in runs:
        r["Tdiff"] = {}
        for v in VARS:
            it, al = r["alpha"][v]
            for th in THRESHOLDS:
                t = (diffusive_onset(r["trace"], v, thresh=th, alpha=(it, al))
                     if len(al) else float("nan"))
                r["Tdiff"][f"{v}@{th}"] = t if (np.isfinite(t) and t <= r["budget"]) else np.nan

    print(f"\nREAD-OUT SWEEP — coverage, seed CV, and the fitted law "
          f"(combinations with CV > {CV_MAX} are not trusted)")
    print("%-22s %6s %8s %9s %8s %8s %s"
          % ("variable @ threshold", "n/64", "mean T", "seed CV", "b", "c", ""))
    grid_b = np.full((len(VARS), len(THRESHOLDS)), np.nan)
    grid_c = np.full((len(VARS), len(THRESHOLDS)), np.nan)
    for i, v in enumerate(VARS):
        for j, th in enumerate(THRESHOLDS):
            key = f"{v}@{th}"
            Ts = [r["Tdiff"][key] for r in runs if np.isfinite(r["Tdiff"][key])]
            if len(Ts) < 0.5 * len(runs):
                print("%-22s %6d %8s %9s %8s %8s  too few runs reach it"
                      % (key, len(Ts), "-", "-", "-", "-"))
                continue
            cv = seed_cv(runs, key)
            b, c, n = slope_bc(runs, key)
            trusted = np.isfinite(cv) and cv <= CV_MAX
            if trusted:
                grid_b[i, j], grid_c[i, j] = b, c
            print("%-22s %6d %8.0f %9.2f %8.2f %8.3f  %s"
                  % (key, len(Ts), np.mean(Ts), cv, b, c,
                     "" if trusted else "<- CV too high, excluded"))

    ok_b = grid_b[np.isfinite(grid_b)]
    ok_c = grid_c[np.isfinite(grid_c)]
    print(f"\nacross all {len(ok_b)} trusted (variable, threshold) combinations:")
    print(f"  b = {ok_b.mean():.3f} +- {ok_b.std():.3f}   range [{ok_b.min():.2f}, {ok_b.max():.2f}]")
    print(f"  c = {ok_c.mean():.3f} +- {ok_c.std():.3f}   range [{ok_c.min():.3f}, {ok_c.max():.3f}]")
    print(f"  for reference, the loss-based criteria gave b = 0.41-0.43, c = -0.04 to -0.02")

    # ---- how the SETTLING TIME itself depends on N and k ---------------------------------------
    print("\nSETTLING TIME as a law:  T = A N^beta k^gamma")
    print("gamma > 0 means harder tasks keep moving directionally for longer - the mechanism that")
    print("makes compute-based read-outs find c > 0 while matched ones find c ~ 0.")
    print("%-22s %6s %20s %20s %s"
          % ("variable @ threshold", "n", "beta (size)", "gamma (complexity)", "T(k=8)/T(k=1)"))
    settle = {}
    for v in VARS:
        for th in THRESHOLDS:
            key = f"{v}@{th}"
            if not np.isfinite(seed_cv(runs, key)) or seed_cv(runs, key) > CV_MAX:
                continue
            f = fit_settling_law(runs, key)
            if not f:
                continue
            settle[key] = f
            print("%-22s %6d %7.3f [%6.3f,%6.3f] %7.3f [%6.3f,%6.3f]  %.2fx [%.2f, %.2f]"
                  % (key, f["n"], f["beta"], f["beta_ci"][0], f["beta_ci"][1],
                     f["gamma"], f["gamma_ci"][0], f["gamma_ci"][1],
                     f["kratio"], f["kratio_ci"][0], f["kratio_ci"][1]))
    winp = {k: v for k, v in settle.items() if k.startswith("W_inp")}
    if winp:
        g = np.array([v["gamma"] for v in winp.values()])
        print(f"\nW_inp specifically ({len(winp)} thresholds): gamma = {g.mean():.3f} +- {g.std():.3f}")
        r = np.array([v["kratio"] for v in winp.values()])
        print(f"  -> going k=1 to k=8 lengthens the W_inp settling time by "
              f"{100*(r.mean()-1):+.0f}% ({r.min():.2f}x to {r.max():.2f}x)")

    # ---- figure ------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 10))
    for v in VARS:
        med = []
        grid = np.linspace(2000, min(r["budget"] for r in runs), 60)
        for r in runs:
            it, al = r["alpha"][v]
            if len(al) > 3:
                med.append(np.interp(grid, it, al))
        if med:
            ax[0][0].plot(grid, np.median(med, axis=0), lw=1.8, label=VAR_LABEL[v])
    for y, lab in [(1.0, "ballistic (directed)"), (0.5, "diffusive"), ]:
        ax[0][0].axhline(y, color="k", ls="--" if y == 1 else "-", lw=1, alpha=.5)
        ax[0][0].text(2500, y + 0.02, lab, fontsize=7.5)
    ax[0][0].axhspan(min(THRESHOLDS), max(THRESHOLDS), color="C0", alpha=.10)
    ax[0][0].set(xscale="log", xlabel="iteration", ylabel=r"lag exponent $\alpha$",
                 title="(a) when does each variable stop moving directionally?\n"
                       "shaded = the thresholds swept")
    ax[0][0].legend(fontsize=7.5)

    for i, (grid, lab, ctr) in enumerate([(grid_b, "exponent $b$ in $M\\sim N^b$", 0.42),
                                          (grid_c, "exponent $c$ in $M\\sim k^c$", 0.0)]):
        a = ax[0][1] if i == 0 else ax[1][0]
        im = a.imshow(grid, cmap="RdBu_r", aspect="auto",
                      vmin=ctr - (0.25 if i == 0 else 0.15), vmax=ctr + (0.25 if i == 0 else 0.15))
        a.set(xticks=range(len(THRESHOLDS)), xticklabels=[f"{t:.2f}" for t in THRESHOLDS],
              yticks=range(len(VARS)), yticklabels=[VAR_LABEL[v] for v in VARS],
              xlabel=r"$\alpha$ threshold", title=f"({'bc'[i]}) {lab}")
        for r_ in range(len(VARS)):
            for c_ in range(len(THRESHOLDS)):
                if np.isfinite(grid[r_, c_]):
                    a.text(c_, r_, f"{grid[r_, c_]:.2f}", ha="center", va="center", fontsize=8)
                else:
                    a.text(c_, r_, "—", ha="center", va="center", fontsize=8, color="0.5")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)

    ax[1][1].axis("off")
    txt = ["ROBUSTNESS SUMMARY", "",
           f"trusted combinations: {len(ok_b)} of {len(VARS)*len(THRESHOLDS)}",
           f"  (seed CV of the read-out time <= {CV_MAX})", "",
           f"b = {ok_b.mean():.3f} +- {ok_b.std():.3f}"
           f"   range [{ok_b.min():.2f}, {ok_b.max():.2f}]",
           f"c = {ok_c.mean():.3f} +- {ok_c.std():.3f}"
           f"   range [{ok_c.min():.3f}, {ok_c.max():.3f}]", "",
           "loss-based criteria, for comparison:",
           "  b = 0.41-0.43,  c = -0.04 to -0.02", "",
           "per-decade disagreement in alpha",
           "(>0.15 means one power law is not enough):"]
    txt += [f"  {v:<8s} {dis[v]:.3f}" if np.isfinite(dis[v]) else f"  {v:<8s} n/a" for v in VARS]
    ax[1][1].text(0.02, 0.98, "\n".join(txt), va="top", family="monospace", fontsize=8.5,
                  transform=ax[1][1].transAxes)

    fig.suptitle("How much does the diffusion read-out depend on WHICH variable and WHICH threshold?",
                 fontsize=12)
    return ps.save(fig, "flipflop_diffusion")


if __name__ == "__main__":
    main()
