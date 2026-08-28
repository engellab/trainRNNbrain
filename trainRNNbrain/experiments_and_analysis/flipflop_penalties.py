#!/usr/bin/env python3
"""
The three flip-flop measurements, per penalty condition: floor vs k, W_inp diffusion, and M(N, k).

⚠️ THE BUDGETS ARE NOT EQUAL, so nothing here is read at each run's own endpoint. Measured:

    none      400-500k   (the k-sweep)
    rws       150k       (the original penalty grid; converges well inside it)
    frm       400k       (the re-run; 150k was NOT converged and its results were retracted)
    both      none yet   (72 tasks queued, none started)

Every trace is therefore TRUNCATED to a common T_CAP before anything is fitted or read. Comparing a
150k rws cell against a 500k none cell at their own endpoints would compare training depth, not
penalty - the same defect that made the unpenalised `endpoint` criterion spuriously reject its own
law, and the reason the 150k frm floors had to be withdrawn.

WHAT EACH SECTION MEASURES

  1. FLOOR vs k      L_inf from a stretched-exponential fit, then the law floor(k) = a + b*sqrt(k).
                     `a` is the single-channel floor, `b` the interference amplitude. The
                     pre-registered penalty test (docs/experiments/penalty_vs_interference.md) is
                     whether a penalty lowers `b` (interference) WITHOUT lowering `a`.
                     ⚠️ At T_CAP = 150k the floor is identified to ~2% (checked against the
                     full-budget fit on the unpenalised cells), which is coarse next to the ~0.7%
                     residuals the law achieves. Differences smaller than that are not resolvable.

  2. W_inp DIFFUSION the iteration at which input-weight motion stops being directed (lag exponent
                     alpha < 0.6, sustained, and still below at the end). rws settles FASTER than no
                     penalty at all; frm never settles, so this criterion does not exist for frm and
                     the table says so rather than printing a number.

  3. M(N, k)         active units under the scale-free rule, fitted as M = A*N^b*k^c at a common
                     iteration. b is the size exponent, c the complexity exponent. Unpenalised and
                     matched, c is indistinguishable from zero; the question here is whether either
                     penalty changes that.

Silence is scale-free throughout. The absolute rule needs a task-calibrated threshold (4e-2 for the
flip-flop, not the CDDM-calibrated 1e-6) and is reported by flipflop_hard_threshold.py.

Output: img/internal_figures/flipflop_penalties.png

Usage:  python flipflop_penalties.py [T_CAP]
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

ROOTS = {"none": "data/trained_RNNs/NBitFlipFlop_std_ksweep",
         "rws": "data/trained_RNNs/NBitFlipFlop_std_pen",
         "frm": "data/trained_RNNs/NBitFlipFlop_std_penlong",
         "both": "data/trained_RNNs/NBitFlipFlop_std_penlong"}
PROBE_EVERY = 10
T_CAP = 150_000        # common truncation: the shortest budget any condition has
T_START = 2000
MIN_ITERS = 50_000     # anything shorter is a timing-calibration run, not an experiment
PENS = ["none", "rws", "frm", "both"]
PCOL = {"none": "k", "rws": "C0", "frm": "C3", "both": "C2"}


def load():
    """Load every run of every penalty condition, excluding calibration-length traces.

    ⚠️ The bigN sweep folder contains 400- and 600-iteration TIMING CALIBRATION runs alongside real
    ones. They carry a complete trace and are indistinguishable from an experiment unless the trace
    length is checked, so anything shorter than MIN_ITERS is dropped and counted.

    Returns:
        (runs, dropped): runs is a list of dicts with pen, k, N, loss (noise-free, indexed by probe),
        trace, part, piters, budget; dropped is a list of (reason, pen, k, N).
    """
    runs, dropped = [], []
    seen = set()
    for pen, root in ROOTS.items():
        for f in sorted(glob.glob(os.path.join(root, "*", "*", "*ParticipationTrace.pkl"))):
            m = re.search(r"_k=(\d+)_N=(\d+)(?:_pen=([a-z]+))?", f)
            if not m:
                continue
            this_pen = m.group(3) or "none"
            if this_pen != pen or f in seen:
                continue
            seen.add(f)
            with open(f, "rb") as fh:
                tr = pickle.load(fh)
            L = np.asarray(tr["metrics"].get("loss_clean_train", []), dtype=float)
            k, N = int(m.group(1)), int(m.group(2))
            if L.size == 0 or np.isnan(L).any():
                dropped.append(("diverged/empty", pen, k, N)); continue
            if L.size * PROBE_EVERY < MIN_ITERS:
                dropped.append(("calibration run", pen, k, N)); continue
            if L.size * PROBE_EVERY < T_CAP:
                dropped.append((f"budget < T_CAP", pen, k, N)); continue
            runs.append(dict(pen=pen, k=k, N=N, loss=L, trace=tr, part=tr["participation"],
                             piters=np.asarray(tr["participation_iters"], dtype=float),
                             budget=L.size * PROBE_EVERY))
    return runs, dropped


def fit_floor(L, t_end):
    """Stretched-exponential floor of one run over [T_START, t_end].

    Args:
        L: noise-free loss indexed by probe; t_end: last iteration included.
    Returns:
        float L_inf, or None if the fit fails.
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
    return float(sol.x[0])


def sqrt_law(ks, floors, n_boot=2000):
    """Fit floor(k) = a + b*sqrt(k) with bootstrap CIs.

    Args:
        ks, floors: per-CELL complexity and mean floor; n_boot: resamples over cells.
    Returns:
        dict with a, b, their CIs and the max relative residual, or None if fewer than 3 cells.
    """
    ks, floors = np.asarray(ks, float), np.asarray(floors, float)
    if ks.size < 3:
        return None
    X = np.column_stack([np.ones_like(ks), np.sqrt(ks)])
    (a, b), *_ = np.linalg.lstsq(X, floors, rcond=None)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(ks), len(ks))
        if len(np.unique(ks[i])) < 3:
            continue
        Xi = np.column_stack([np.ones(len(ks)), np.sqrt(ks[i])])
        try:
            bs.append(np.linalg.lstsq(Xi, floors[i], rcond=None)[0])
        except np.linalg.LinAlgError:
            continue
    bs = np.array(bs)
    # ⚠️ With exactly 3 distinct k, every bootstrap resample that keeps 3 unique k IS the original
    # sample, so the "CI" collapses onto the point estimate and looks spuriously precise. Report nan
    # in that case rather than a zero-width interval.
    nk = len(set(ks))
    degenerate = nk < 4 or len(bs) < 50 or np.allclose(bs.std(axis=0), 0)
    ci = ((lambda j: (np.nan, np.nan)) if degenerate else
          (lambda j: (float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5)))))
    res = float(np.max(np.abs(a + b * np.sqrt(ks) - floors) / floors))
    return dict(a=float(a), b=float(b), a_ci=ci(0), b_ci=ci(1), resid=res, nk=nk,
                degenerate=degenerate)


def truncate(tr, tcap):
    """Trace copy with every full-length metric cut at tcap, so drift onsets are range-matched."""
    it = np.asarray(tr["iters"], dtype=float)
    keep = it <= tcap
    return {"iters": it[keep].tolist(),
            "metrics": {k: list(np.asarray(v, dtype=float)[keep])
                        for k, v in tr["metrics"].items() if len(v) == len(it)}}


def M_at(run, t):
    """Scale-free active-unit count at the last snapshot at or before iteration t."""
    ok = np.flatnonzero(run["piters"] <= t)
    if ok.size == 0:
        return float("nan")
    return float(active_count(np.asarray(run["part"][int(ok[-1])], dtype=float), "scalefree"))


def fit_law(runs, pen, t):
    """Fit M = A N^b k^c at iteration t for one penalty condition, with bootstrap CIs."""
    K, NN, M = [], [], []
    for r in runs:
        if r["pen"] != pen:
            continue
        m = M_at(r, t)
        if np.isfinite(m) and m > 0:
            K.append(r["k"]); NN.append(r["N"]); M.append(m)
    if len(M) < 8 or len(set(NN)) < 2 or len(set(K)) < 3:
        return None
    K, NN, M = np.array(K, float), np.array(NN, float), np.array(M, float)
    y = np.log(M)
    X = np.column_stack([np.ones_like(y), np.log(NN), np.log(K)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
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
    return dict(b=float(beta[1]), c=float(beta[2]), b_ci=ci(1), c_ci=ci(2), n=len(M),
                K=K, NN=NN, M=M)


def main():
    """Run all three per-penalty analyses at a common truncation and plot them."""
    global T_CAP
    if len(sys.argv) > 1:
        T_CAP = int(sys.argv[1])
    ps.setup()
    runs, dropped = load()
    have = sorted({r["pen"] for r in runs}, key=PENS.index)
    print(f"T_CAP = {T_CAP:,} — every trace truncated here before any fit or read-out\n")
    print("coverage (runs per condition):")
    for pen in PENS:
        sel = [r for r in runs if r["pen"] == pen]
        if not sel:
            print(f"  {pen:5s}  NO DATA"); continue
        cells = sorted({(r["k"], r["N"]) for r in sel})
        print(f"  {pen:5s}  {len(sel):3d} runs, {len(cells)} cells; "
              f"k={sorted({c[0] for c in cells})}, N={sorted({c[1] for c in cells})}")
    for d in sorted(set(dropped)):
        print(f"  dropped {d} x{dropped.count(d)}")

    # ---- 1. floor vs k ------------------------------------------------------------------------
    print(f"\n{'='*78}\n1. LOSS FLOOR vs COMPLEXITY   floor(k) = a + b*sqrt(k)\n{'='*78}")
    print("a = single-channel floor, b = interference amplitude.")
    print("Pre-registered test: a penalty supports the interference hypothesis only if it lowers b")
    print("WITHOUT lowering a.\n")
    floors = {}
    for r in runs:
        f = fit_floor(r["loss"], T_CAP)
        if f:
            floors.setdefault((r["pen"], r["k"], r["N"]), []).append(f)
    cellf = {g: float(np.mean(v)) for g, v in floors.items()}
    laws = {}
    print("%-6s %6s %4s %24s %24s %9s" % ("pen", "N", "nk", "a [95% CI]", "b [95% CI]", "resid"))
    for pen in have:
        for N in sorted({g[2] for g in cellf if g[0] == pen}):
            ks = [g[1] for g in cellf if g[0] == pen and g[2] == N]
            fl = [cellf[(pen, k, N)] for k in ks]
            law = sqrt_law(ks, fl)
            if not law:
                print(f"{pen:<6s} {N:6d} {len(set(ks)):4d}   only {len(set(ks))} k values — need 3")
                continue
            laws[(pen, N)] = law
            if law["degenerate"]:
                print("%-6s %6d %4d  %.5f [  too few k  ]  %.5f [  too few k  ] %8.2f%%   "
                      "<- CI not estimable at %d k values"
                      % (pen, N, law["nk"], law["a"], law["b"], 100 * law["resid"], law["nk"]))
            else:
                print("%-6s %6d %4d  %.5f [%.5f,%.5f]  %.5f [%.5f,%.5f] %8.2f%%"
                      % (pen, N, law["nk"], law["a"], law["a_ci"][0], law["a_ci"][1],
                         law["b"], law["b_ci"][0], law["b_ci"][1], 100 * law["resid"]))

    # ---- 2. W_inp diffusion -------------------------------------------------------------------
    print(f"\n{'='*78}\n2. W_inp DIFFUSION   (alpha < 0.6, sustained, still below at the end)\n{'='*78}")
    print("%-6s %6s %14s %14s %12s" % ("pen", "n", "settled", "median onset", "alpha at cap"))
    from common import drift_alpha
    diff = {}
    for pen in have:
        sel = [r for r in runs if r["pen"] == pen]
        ons, ends = [], []
        for r in sel:
            t = truncate(r["trace"], T_CAP)
            ons.append(diffusive_onset(t, "W_inp", thresh=0.6))
            _, al = drift_alpha(t, "W_inp")
            ends.append(al[-1] if len(al) else np.nan)
        ok = [o for o in ons if np.isfinite(o)]
        diff[pen] = (len(ok), len(sel), np.median(ok) if ok else np.nan, np.nanmedian(ends))
        print("%-6s %6d %10d/%-3d %14s %12.2f"
              % (pen, len(sel), len(ok), len(sel),
                 f"{np.median(ok)/1000:.0f}k" if ok else "NEVER", np.nanmedian(ends)))
    if "frm" in have:
        print("\n⚠️ frm does not settle: alpha plateaus above the threshold, so the diffusion")
        print("   criterion does not exist for it. frm must be compared by matched LOSS.")

    # ---- 3. M(N, k) ----------------------------------------------------------------------------
    print(f"\n{'='*78}\n3. ACTIVE UNITS   M = A N^b k^c  at iteration {T_CAP:,}\n{'='*78}")
    print("%-6s %5s %24s %26s" % ("pen", "n", "b (size) [95% CI]", "c (complexity) [95% CI]"))
    mlaws = {}
    for pen in have:
        law = fit_law(runs, pen, T_CAP)
        if not law:
            print(f"{pen:<6s}   too few cells to fit"); continue
        mlaws[pen] = law
        star = "" if law["c_ci"][0] <= 0 <= law["c_ci"][1] else "   <- c != 0"
        print("%-6s %5d   %.3f [%.3f, %.3f]      %+.3f [%+.3f, %+.3f]%s"
              % (pen, law["n"], law["b"], law["b_ci"][0], law["b_ci"][1],
                 law["c"], law["c_ci"][0], law["c_ci"][1], star))

    # ---- 3b. the same law at MATCHED DIFFUSION, which fixed compute cannot substitute for ---
    print(f"\n{'='*78}\n3b. THE SAME LAW AT MATCHED W_inp DIFFUSION (not fixed compute)\n{'='*78}")
    print("Fixed compute is CONFOUNDED: harder tasks converge slower, so c > 0 appears there even")
    print("unpenalised. Only conditions whose W_inp actually settles can be read this way — frm")
    print("cannot. rws lacks N=2000, so `none` is refitted on the SAME two sizes as a control.")
    print("\n%-30s %5s %22s %26s" % ("condition", "n", "b [95% CI]", "c [95% CI]"))
    for pen, Nset, lab in [("none", {500, 1000, 2000}, "none  all sizes"),
                           ("none", {500, 1000}, "none  N=500,1000 (control)"),
                           ("rws", {500, 1000}, "rws   N=500,1000")]:
        K, NN, M = [], [], []
        for r in runs:
            if r["pen"] != pen or r["N"] not in Nset:
                continue
            t = diffusive_onset(truncate(r["trace"], T_CAP), "W_inp", thresh=0.6)
            if not np.isfinite(t):
                continue
            m = M_at(r, t)
            if np.isfinite(m) and m > 0:
                K.append(r["k"]); NN.append(r["N"]); M.append(m)
        if len(M) < 8 or len(set(NN)) < 2:
            print("%-30s   too few runs settle to fit" % lab); continue
        K, NN, M = np.array(K, float), np.array(NN, float), np.array(M, float)
        y = np.log(M)
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(y), np.log(NN), np.log(K)]),
                                   y, rcond=None)
        rng = np.random.default_rng(0); bs = []
        for _ in range(3000):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(K[i])) < 3 or len(np.unique(NN[i])) < 2:
                continue
            bs.append(np.linalg.lstsq(
                np.column_stack([np.ones(len(y)), np.log(NN[i]), np.log(K[i])]), y[i],
                rcond=None)[0])
        bs = np.array(bs)
        lb, hb = np.percentile(bs[:, 1], [2.5, 97.5])
        lc, hc = np.percentile(bs[:, 2], [2.5, 97.5])
        print("%-30s %5d  %.3f [%.3f,%.3f]   %+.3f [%+.3f,%+.3f]%s"
              % (lab, len(M), beta[1], lb, hb, beta[2], lc, hc,
                 "" if lc <= 0 <= hc else "   <- c != 0"))

    # ---- 3c. frm abolishes silence, so its exponent is a ceiling effect ----------------------
    print(f"\n{'='*78}\n3c. ACTIVE FRACTION M/N — why frm's exponent is not comparable\n{'='*78}")
    print("%-6s %8s %12s %14s" % ("pen", "N", "mean M/N", "range"))
    for pen in have:
        for N in sorted({r["N"] for r in runs if r["pen"] == pen}):
            v = [M_at(r, T_CAP) / N for r in runs if r["pen"] == pen and r["N"] == N]
            v = [x for x in v if np.isfinite(x)]
            if v:
                print("%-6s %8d %12.3f %14s"
                      % (pen, N, np.mean(v), f"{min(v):.2f}-{max(v):.2f}"))
    print("\n⚠️ frm drives M/N to 0.95-0.99: it abolishes silence BY CONSTRUCTION, which is what it")
    print("   is for. Its b ~ 0.91 is therefore a CEILING EFFECT at M = N, not a scaling exponent,")
    print("   and must not be compared against none/rws's b as though it measured the same thing.")
    print("⚠️ frm's FLOOR at this truncation is also invalid: frm needs ~400k to converge, so a fit")
    print("   over [2000, 150000] is fitted to a still-descending curve — the exact defect that")
    print("   forced the earlier retraction. Compare frm floors only against `none` at 400k.")

    # ---- figure ---------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))
    for pen in have:
        cells = sorted({(g[1], g[2]) for g in cellf if g[0] == pen})
        Ns = sorted({c[1] for c in cells})
        for N in Ns:
            ks = sorted(k for k, n in cells if n == N)
            if len(ks) < 2:
                continue
            ax[0].plot(ks, [cellf[(pen, k, N)] for k in ks], "-o", color=PCOL[pen],
                       alpha=0.45 + 0.5 * (N == max(Ns)), ms=4,
                       label=f"{pen} N={N}" if N == max(Ns) else None)
    ax[0].set(xlabel="k (bits)", ylabel=r"per-channel floor $L_\infty$",
              title=f"(a) floor vs complexity\nfitted on [{T_START}, {T_CAP:,}] for every condition")
    ax[0].legend(fontsize=7)

    xs = [p for p in have if np.isfinite(diff[p][3])]
    ax[1].bar(range(len(xs)), [diff[p][3] for p in xs], color=[PCOL[p] for p in xs], alpha=.85)
    ax[1].axhline(0.6, color="k", ls="--", lw=1.2)
    ax[1].text(-0.4, 0.62, "threshold 0.6", fontsize=8)
    ax[1].axhline(0.5, color="grey", ls=":", lw=1)
    for i, p in enumerate(xs):
        ax[1].text(i, diff[p][3] + .02, f"{diff[p][0]}/{diff[p][1]}", ha="center", fontsize=8)
    ax[1].set(xticks=range(len(xs)), xticklabels=xs, ylabel=r"$\alpha(W_{inp})$ at the cap",
              ylim=(0, 1.05), title=f"(b) W_inp diffusion at {T_CAP:,}\n"
                                    "below 0.6 = settled; label = fraction settled")

    for pen in mlaws:
        law = mlaws[pen]
        ax[2].errorbar(law["b"], law["c"],
                       xerr=[[law["b"] - law["b_ci"][0]], [law["b_ci"][1] - law["b"]]],
                       yerr=[[law["c"] - law["c_ci"][0]], [law["c_ci"][1] - law["c"]]],
                       fmt="o", ms=9, color=PCOL[pen], capsize=3, label=pen)
    ax[2].axhline(0, color="k", lw=1, alpha=.6)
    ax[2].set(xlabel="b — size exponent", ylabel="c — complexity exponent",
              title=f"(c) $M = A\\,N^b k^c$ at {T_CAP:,}\nc = 0 means M does not depend on k")
    ax[2].legend(fontsize=8)
    fig.suptitle(f"Flip-flop: floor, input-weight diffusion and active units, per penalty "
                 f"(all truncated to {T_CAP:,} iterations)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return ps.save(fig, "flipflop_penalties", tight=False)


if __name__ == "__main__":
    main()
