#!/usr/bin/env python3
"""
Which read-out criterion actually works across N, k AND penalty? A scored search, not an argument.

THE PROBLEM. Comparing M(N, k) needs every network read at a comparable stage of training, or the
comparison measures convergence depth instead. Every criterion tried so far fails somewhere:

  iter      fixed iteration        - CONFOUNDED. Harder tasks converge later (settling time scales as
                                     k^0.458), so c > 0 appears even unpenalised, and compensating by
                                     k^0.458 removes 100% of it.
  endpoint  each run's own end     - budgets differ (150k/400k/500k); disqualified.
  loss      common absolute loss   - the achievable floor rises with k, so one L* sits at a different
                                     depth for each k.
  excess    (1+d) x own floor      - k-fair, but needs a floor fit, hence a converged run.
  drift     W_inp alpha < 0.6      - DOES NOT EXIST for frm (1/27) or both (1/9): their input weights
                                     never stop moving directionally. Also carries an N x k
                                     interaction of its own.

So the search is for something defined for EVERY condition, robust across seeds, and free of the
convergence confound. Two new candidates are floor-free by construction:

  slope     read where the log-log learning rate rho = -dlog(L)/dlog(t) first falls below a threshold
            and stays. "The network has stopped improving", measured on the loss itself, needing no
            fit, no floor and no convergence assumption.
  hess      read where the loss has flattened RELATIVE TO ITS OWN EARLY DESCENT: rho(t)/max(rho)
            below a threshold. Scale-free in the same way, and immune to conditions whose overall
            descent rate differs.

Scored against thresholds fixed in advance (see scratchpad/criteria_spec.md):
  1 COVERAGE      defined for >= 90% of runs in every condition
  2 ROBUSTNESS    within-cell seed CV of T <= 0.30
  3 DECONFOUNDING for `none`, fitted c must straddle 0 (three independent matched criteria agree it
                  should, and the k-compensation test showed fixed-compute's c = +0.179 is entirely
                  convergence depth)
  4 SANITY        median T must RISE with k

1+2 passing        -> usable WITHIN a penalty condition  (program minimum)
1+2+3+4 everywhere -> usable ACROSS penalties            (program maximum)

Output: img/internal_figures/criterion_search.png

Usage:  python criterion_search.py
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, active_count, diffusive_onset, logbin, stretched
import plotstyle as ps

ROOTS = {"ksweep": "data/trained_RNNs/NBitFlipFlop_std_ksweep",
         "pen": "data/trained_RNNs/NBitFlipFlop_std_pen",
         "penlong": "data/trained_RNNs/NBitFlipFlop_std_penlong"}
SKIP = {("pen", "frm")}          # the retracted 150k frm cells
PROBE = 10
T_START = 2000
MIN_ITERS = 50_000
PENS = ["none", "rws", "frm", "both"]
COV_MIN, CV_MAX = 0.90, 0.30


def load():
    """Load every usable run. Returns list of dicts with pen, k, N, loss, trace, part, piters, budget."""
    runs = []
    for tag, root in ROOTS.items():
        for f in sorted(glob.glob(os.path.join(root, "*", "*", "*ParticipationTrace.pkl"))):
            m = re.search(r"_k=(\d+)_N=(\d+)(?:_pen=([a-z]+))?", f)
            if not m:
                continue
            pen = m.group(3) or "none"
            if (tag, pen) in SKIP:
                continue
            with open(f, "rb") as fh:
                tr = pickle.load(fh)
            L = np.asarray(tr["metrics"].get("loss_clean_train", []), dtype=float)
            if L.size * PROBE < MIN_ITERS or np.isnan(L).any():
                continue
            runs.append(dict(pen=pen, k=int(m.group(1)), N=int(m.group(2)), loss=L, trace=tr,
                             part=tr["participation"], budget=L.size * PROBE,
                             piters=np.asarray(tr["participation_iters"], dtype=float)))
    return runs


def smooth_loglog(L):
    """Log-binned (t, L) for one run, the common input to every loss-based criterion.

    Returns:
        (t, y) arrays over [T_START, run end], log-binned so the tail does not dominate.
    """
    t = (np.arange(len(L)) + 1) * PROBE
    m = t >= T_START
    return logbin(t[m], L[m], nbins=80)


def rho_series(L):
    """Relative learning rate rho = -dlog(L)/dlog(t): how fast the loss still improves per decade.

    Floor-free and fit-free. rho ~ 1 early (loss falling like 1/t), rho -> 0 once the loss flattens
    onto its floor, whatever that floor happens to be. Because it is a LOG derivative it is invariant
    to the overall scale of the loss, so conditions with different achievable floors are directly
    comparable - which is exactly what an absolute loss threshold cannot do.

    Args:
        L: noise-free loss indexed by probe.
    Returns:
        (t, rho) arrays, or empty if the trace is too short.
    """
    t, y = smooth_loglog(L)
    if len(t) < 8:
        return np.array([]), np.array([])
    lt, ly = np.log(t), np.log(np.clip(y, 1e-12, None))
    rho = -np.gradient(ly, lt)
    w = 5                                            # median filter: the raw derivative is noisy
    rho = np.array([np.median(rho[max(0, i - w // 2):i + w // 2 + 1]) for i in range(len(rho))])
    return t, rho


def first_sustained(x, vals, thresh, persist=3):
    """First x where vals is below thresh for `persist` consecutive points and stays below at the end.

    The tail check is the same guard that `diffusive_onset` needed: without it a noisy series that
    merely dips below the threshold reports a spurious crossing.
    """
    below = vals < thresh
    if not below.any() or np.median(vals[-max(persist, len(vals) // 10):]) >= thresh:
        return float("nan")
    for i in range(len(below) - persist + 1):
        if below[i:i + persist].all():
            return float(x[i])
    return float("nan")


def fit_floor(L, t_end):
    """Stretched-exponential floor over [T_START, t_end]; None if the fit fails."""
    t = (np.arange(len(L)) + 1) * PROBE
    m = (t >= T_START) & (t <= t_end)
    tb, yb = logbin(t[m], L[m])
    if len(tb) < 8:
        return None
    try:
        s = least_squares(lambda p: np.log(np.clip(stretched(tb, *p), 1e-12, None)) - np.log(yb),
                          [yb.min() * .9, float(yb.max()), 2e4, .4],
                          bounds=([1e-6, 1e-6, 1e2, .05], [1., 1e3, 1e8, 3.]), max_nfev=20000)
    except Exception:
        return None
    return float(s.x[0])


def first_below_loss(L, target):
    """First iteration where the smoothed loss reaches `target`, else nan."""
    w = 21
    if len(L) < w:
        return float("nan")
    s = np.convolve(L, np.ones(w) / w, mode="valid")
    hit = np.flatnonzero(s <= target)
    return float((hit[0] + w // 2 + 1) * PROBE) if len(hit) else float("nan")


def build_criteria(runs):
    """Attach every candidate criterion's read-out iteration to each run."""
    for r in runs:
        r["L_inf"] = fit_floor(r["loss"], r["budget"])       # own budget: each condition converged
    worst = max(r["L_inf"] for r in runs if r["L_inf"])
    Lstar = 1.10 * worst
    for r in runs:
        t_r, rho = rho_series(r["loss"])
        T = {"iter@150k": 150_000.0,
             "loss": first_below_loss(r["loss"], Lstar),
             "excess": (first_below_loss(r["loss"], 1.10 * r["L_inf"]) if r["L_inf"] else np.nan),
             "drift(W_inp)": diffusive_onset(r["trace"], "W_inp", thresh=0.6),
             "drift(W_rec)": diffusive_onset(r["trace"], "W_rec", thresh=0.6)}
        for th in (0.02, 0.05, 0.10):
            T[f"slope<{th}"] = first_sustained(t_r, rho, th) if len(rho) else np.nan
        if len(rho):
            peak = np.max(rho[:max(3, len(rho) // 3)])
            for fr in (0.05, 0.10):
                T[f"rho<{fr:.2f}peak"] = first_sustained(t_r, rho, fr * peak) if peak > 0 else np.nan
        for c, v in T.items():
            if not np.isfinite(v) or v > r["budget"]:
                T[c] = np.nan
        r["T"] = T
    return sorted(runs[0]["T"].keys()), Lstar


def M_at(run, t):
    """Scale-free active-unit count at the last snapshot at or before t."""
    if not np.isfinite(t):
        return np.nan
    ok = np.flatnonzero(run["piters"] <= t)
    if ok.size == 0:
        return np.nan
    return float(active_count(np.asarray(run["part"][int(ok[-1])], dtype=float), "scalefree"))


def fit_c(runs, pen, crit):
    """Fitted c in M = A N^b k^c for one condition, with a bootstrap CI. None if unfittable."""
    K, NN, M = [], [], []
    for r in runs:
        if r["pen"] != pen:
            continue
        m = M_at(r, r["T"][crit])
        if np.isfinite(m) and m > 0:
            K.append(r["k"]); NN.append(r["N"]); M.append(m)
    if len(M) < 8 or len(set(NN)) < 2 or len(set(K)) < 3:
        return None
    K, NN, M = np.array(K, float), np.array(NN, float), np.array(M, float)
    y = np.log(M)
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(y), np.log(NN), np.log(K)]), y,
                               rcond=None)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(1500):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(K[i])) < 3 or len(np.unique(NN[i])) < 2:
            continue
        bs.append(np.linalg.lstsq(np.column_stack([np.ones(len(y)), np.log(NN[i]), np.log(K[i])]),
                                  y[i], rcond=None)[0])
    bs = np.array(bs)
    return dict(b=float(beta[1]), c=float(beta[2]),
                c_ci=(float(np.percentile(bs[:, 2], 2.5)), float(np.percentile(bs[:, 2], 97.5))),
                n=len(M))


def score(runs, crit, pen):
    """Coverage, seed CV and k-monotonicity of one criterion in one condition."""
    sel = [r for r in runs if r["pen"] == pen]
    if not sel:
        return None
    ok = [r for r in sel if np.isfinite(r["T"][crit])]
    cov = len(ok) / len(sel)
    cells = {}
    for r in ok:
        cells.setdefault((r["k"], r["N"]), []).append(r["T"][crit])
    cvs = [np.std(v) / np.mean(v) for v in cells.values() if len(v) > 1 and np.mean(v) > 0]
    ks = sorted({r["k"] for r in ok})
    med = [np.median([r["T"][crit] for r in ok if r["k"] == k]) for k in ks]
    slope = np.polyfit(np.log(ks), np.log(med), 1)[0] if len(ks) > 2 and all(
        m > 0 for m in med) else np.nan
    return dict(cov=cov, cv=float(np.mean(cvs)) if cvs else np.nan, kslope=float(slope),
                n=len(sel))


def main():
    """Score every candidate criterion against the pre-registered thresholds."""
    ps.setup()
    runs = load()
    crits, Lstar = build_criteria(runs)
    pens = [p for p in PENS if any(r["pen"] == p for r in runs)]
    print(f"{len(runs)} runs; conditions {pens};  `loss` level L* = {Lstar:.5f}\n")

    print("1+2  COVERAGE and SEED CV per condition   (need cov >= %.0f%%, CV <= %.2f)"
          % (100 * COV_MIN, CV_MAX))
    hdr = "%-16s" % "criterion" + "".join(f"{p:>22s}" for p in pens)
    print(hdr); print("%-16s" % "" + "".join(f"{'cov   CV   dlogT/dlogk':>22s}" for p in pens))
    keep = {}
    for c in crits:
        row = "%-16s" % c
        ok_all = True
        for p in pens:
            s = score(runs, c, p)
            if s is None:
                row += f"{'-':>22s}"; continue
            good = s["cov"] >= COV_MIN and (np.isnan(s["cv"]) or s["cv"] <= CV_MAX)
            ok_all &= good
            row += "%9.0f%% %5.2f %6.2f" % (100 * s["cov"], s["cv"], s["kslope"])
        keep[c] = ok_all
        print(row + ("   PASS" if ok_all else ""))

    print("\n3  DECONFOUNDING — fitted c for `none` must straddle 0")
    print("%-16s %26s %s" % ("criterion", "c [95% CI]", "verdict"))
    for c in crits:
        f = fit_c(runs, "none", c)
        if not f:
            print("%-16s %26s" % (c, "unfittable")); continue
        straddles = f["c_ci"][0] <= 0 <= f["c_ci"][1]
        print("%-16s   %+.3f [%+.3f, %+.3f]   %s"
              % (c, f["c"], f["c_ci"][0], f["c_ci"][1],
                 "PASS" if straddles else "FAIL — confound not removed"))
        keep[c] = keep.get(c, False) and straddles

    print("\nSUMMARY")
    winners = [c for c, v in keep.items() if v]
    print("  criteria passing ALL of coverage, seed CV and deconfounding in every condition:")
    print("   ", ", ".join(winners) if winners else "NONE")

    print("\n4  what each surviving criterion says, per condition:")
    print("%-16s %-6s %6s %10s %26s" % ("criterion", "pen", "n", "median T", "c [95% CI]"))
    for c in (winners or crits):
        for p in pens:
            f = fit_c(runs, p, c)
            Ts = [r["T"][c] for r in runs if r["pen"] == p and np.isfinite(r["T"][c])]
            if not Ts:
                continue
            if f:
                print("%-16s %-6s %6d %10.0f   %+.3f [%+.3f, %+.3f]"
                      % (c, p, f["n"], np.median(Ts), f["c"], f["c_ci"][0], f["c_ci"][1]))
            else:
                print("%-16s %-6s %6s %10.0f   %s" % (c, p, "-", np.median(Ts), "unfittable"))

    # ---- figure ------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for i, (metric, lab, ref) in enumerate([
            ("cov", "coverage (fraction of runs with a read-out)", COV_MIN),
            ("cv", "within-cell seed CV of the read-out time", CV_MAX)]):
        for j, p in enumerate(pens):
            vals = [(score(runs, c, p) or {}).get(metric, np.nan) for c in crits]
            ax[i].plot(range(len(crits)), vals, "o-", color=ps.PCOL_PEN.get(p, f"C{j}")
                       if hasattr(ps, "PCOL_PEN") else f"C{j}", label=p, ms=6)
        ax[i].axhline(ref, color="k", ls="--", lw=1.2)
        ax[i].set(xticks=range(len(crits)), ylabel=lab, title=f"({'ab'[i]}) {lab}")
        ax[i].set_xticklabels(crits, rotation=40, ha="right", fontsize=7.5)
        ax[i].legend(fontsize=8)
    for c in crits:
        f = fit_c(runs, "none", c)
        if f:
            ax[2].errorbar(crits.index(c), f["c"],
                           yerr=[[f["c"] - f["c_ci"][0]], [f["c_ci"][1] - f["c"]]],
                           fmt="o", ms=7, capsize=3,
                           color="C2" if f["c_ci"][0] <= 0 <= f["c_ci"][1] else "C3")
    ax[2].axhline(0, color="k", lw=1)
    ax[2].set(xticks=range(len(crits)), ylabel="fitted c for `none`",
              title="(c) deconfounding: c must straddle 0\ngreen = passes, red = confound remains")
    ax[2].set_xticklabels(crits, rotation=40, ha="right", fontsize=7.5)
    fig.suptitle("Which read-out criterion works across N, k and penalty?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return ps.save(fig, "criterion_search", tight=False)


if __name__ == "__main__":
    main()
