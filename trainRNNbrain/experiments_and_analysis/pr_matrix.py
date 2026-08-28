#!/usr/bin/env python3
"""
Participation ratio PR/N over the (N, k) grid, per penalty, at the `excess` read-out.

READ-OUT: every network is read where its noise-free loss reaches 1.10 x ITS OWN fitted floor - i.e.
90% of the way to the best that network can do. This is the criterion that won the scored search in
criterion_search.py: 100% coverage in all four penalty conditions, within-cell seed CV 0.04-0.06,
read-out time rising with k as convergence requires, and it reports c ~ 0 for unpenalised networks,
which three independent criteria agree is the right answer.

⚠️ EACH CONDITION'S FLOOR IS FITTED OVER ITS OWN BUDGET, NOT A COMMON RANGE. Range-matching is only
ever a proxy for "the floor is estimated correctly"; forcing every condition onto a common 150k is
what made frm's floor invalid and forced a retraction, because frm needs ~400k to converge. Verify
convergence per condition, then let each use the range where it is converged.

WHY PR RATHER THAN THE ACTIVE-UNIT COUNT. The thresholded count SATURATES: under frm and frm+rws
essentially every unit clears any silence threshold (M/N = 0.99-1.00), so M cannot separate those two
conditions or support an exponent. PR = (sum p)^2 / sum p^2 is the EFFECTIVE number of participating
units - it asks how evenly activity is spread rather than how many units clear a bar - and stays
graded (frm 0.97, both 0.93, none 0.39 as fractions of N).

⚠️ M AND PR DISAGREE, AND BOTH ARE REPORTED. Under all six deconfounding criteria the thresholded
count is k-INDEPENDENT (c straddles 0) while PR RISES with k (c = +0.05..+0.12, every CI excluding
0). Reading: at higher complexity the same number of units stay active but activity spreads more
evenly among them. Row 3 of the figure carries M/N so the two are never quoted in isolation.

Output: img/internal_figures/pr_matrix.png

Usage:  python pr_matrix.py [EXCESS_DELTA]
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
from common import IMG_DIR, active_count, logbin, participation_ratio, stretched
import plotstyle as ps

ROOTS = {"ksweep": "data/trained_RNNs/NBitFlipFlop_std_ksweep",
         "pen": "data/trained_RNNs/NBitFlipFlop_std_pen",
         "penlong": "data/trained_RNNs/NBitFlipFlop_std_penlong"}
SKIP = {("pen", "frm")}          # retracted 150k frm cells; frm comes from penlong
PENS = ["none", "rws", "frm", "both"]
PROBE = 10
T_START = 2000
MIN_ITERS = 50_000
EXCESS_DELTA = 0.10


def load():
    """Load every usable run; drops calibration-length traces and diverged runs."""
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
            runs.append(dict(pen=pen, k=int(m.group(1)), N=int(m.group(2)), loss=L,
                             part=tr["participation"], budget=L.size * PROBE,
                             piters=np.asarray(tr["participation_iters"], dtype=float)))
    return runs


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


def excess_time(L, floor, delta):
    """Iteration where the smoothed loss first reaches (1+delta) x floor, else nan.

    ⚠️ `delta` is REQUIRED, not defaulted. It was previously `delta=EXCESS_DELTA`, which binds the
    module-level value at DEFINITION time - so reassigning the global from the command line left the
    threshold frozen at 0.10 and every "swept" figure came out identical.
    """
    w = 21
    if floor is None or len(L) < w:
        return float("nan")
    s = np.convolve(L, np.ones(w) / w, mode="valid")
    hit = np.flatnonzero(s <= (1 + delta) * floor)
    return float((hit[0] + w // 2 + 1) * PROBE) if len(hit) else float("nan")


def measure(run, fn):
    """Apply a participation statistic at this run's read-out iteration."""
    t = run["T"]
    if not np.isfinite(t):
        return float("nan")
    ok = np.flatnonzero(run["piters"] <= t)
    if ok.size == 0:
        return float("nan")
    return fn(np.asarray(run["part"][int(ok[-1])], dtype=float))


def cells(runs, pen, fn, ks, Ns):
    """(mean, sd, n) grids of a per-run statistic over (N, k) for one penalty."""
    Z = np.full((len(Ns), len(ks)), np.nan)
    S = np.full((len(Ns), len(ks)), np.nan)
    C = np.zeros((len(Ns), len(ks)), dtype=int)
    box = {}
    for r in runs:
        if r["pen"] != pen:
            continue
        v = measure(r, fn)
        if np.isfinite(v):
            box.setdefault((r["k"], r["N"]), []).append(v / r["N"])
    for (k, N), v in box.items():
        if k in ks and N in Ns:
            i, j = Ns.index(N), ks.index(k)
            Z[i, j], S[i, j], C[i, j] = np.mean(v), np.std(v), len(v)
    return Z, S, C


def fit_law(runs, pen, fn):
    """Fit Y = A N^b k^c for a per-run statistic (NOT divided by N), with bootstrap CIs."""
    K, NN, Y = [], [], []
    for r in runs:
        if r["pen"] != pen:
            continue
        v = measure(r, fn)
        if np.isfinite(v) and v > 0:
            K.append(r["k"]); NN.append(r["N"]); Y.append(v)
    if len(Y) < 8 or len(set(NN)) < 2 or len(set(K)) < 3:
        return None
    K, NN, Y = np.array(K, float), np.array(NN, float), np.array(Y, float)
    ly = np.log(Y)
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(ly), np.log(NN), np.log(K)]), ly,
                              rcond=None)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(ly), len(ly))
        if len(np.unique(K[i])) < 3 or len(np.unique(NN[i])) < 2:
            continue
        bs.append(np.linalg.lstsq(np.column_stack([np.ones(len(ly)), np.log(NN[i]), np.log(K[i])]),
                                  ly[i], rcond=None)[0])
    bs = np.array(bs)
    q = lambda j: (float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5)))
    return dict(A=float(np.exp(beta[0])), b=float(beta[1]), c=float(beta[2]),
                b_ci=q(1), c_ci=q(2), n=len(Y))


def main():
    """Compute PR/N and M/N at the excess read-out and plot them over the (N, k) grid."""
    global EXCESS_DELTA
    if len(sys.argv) > 1:
        EXCESS_DELTA = float(sys.argv[1])
    ps.setup()
    runs = load()
    for r in runs:
        r["floor"] = fit_floor(r["loss"], r["budget"])       # OWN budget, not a common range
        r["T"] = excess_time(r["loss"], r["floor"], EXCESS_DELTA)
    n_before = {p: sum(1 for r in runs if r["pen"] == p) for p in PENS}
    runs = [r for r in runs if np.isfinite(r["T"])]
    ks = sorted({r["k"] for r in runs})
    Ns = sorted({r["N"] for r in runs})
    have = [p for p in PENS if any(r["pen"] == p for r in runs)]
    print(f"read-out: loss reaches {1+EXCESS_DELTA:.2f} x each run's OWN floor "
          f"(floor fitted over that run's own budget)\n")
    for p in PENS:
        sel = [r for r in runs if r["pen"] == p]
        if not sel:
            print(f"  {p:5s}  NO DATA — panel will be drawn empty"); continue
        print(f"  {p:5s}  {len(sel):3d}/{n_before[p]:3d} runs reach it "
              f"({100*len(sel)/max(n_before[p],1):.0f}% coverage), "
              f"k={sorted({r['k'] for r in sel})}, N={sorted({r['N'] for r in sel})}, "
              f"median read-out {np.median([r['T'] for r in sel]):.0f}")

    STATS = [(participation_ratio, "PR/N", "effective fraction of units participating"),
             (lambda p: active_count(p, "scalefree"), "M/N", "fraction above the silence threshold")]

    for fn, lab, _ in STATS:
        print(f"\n{'='*74}\n{lab}:  fitted  Y = A N^b k^c   (Y is the raw count, not /N)\n{'='*74}")
        print("%-6s %5s %24s %26s" % ("pen", "n", "b (size)", "c (complexity)"))
        for p in have:
            f = fit_law(runs, p, fn)
            if not f:
                print("%-6s %5s   not fittable (needs k>=3 at >=2 sizes)" % (p, "-")); continue
            star = "" if f["c_ci"][0] <= 0 <= f["c_ci"][1] else "   <- c != 0"
            print("%-6s %5d   %.3f [%.3f, %.3f]      %+.3f [%+.3f, %+.3f]%s"
                  % (p, f["n"], f["b"], f["b_ci"][0], f["b_ci"][1],
                     f["c"], f["c_ci"][0], f["c_ci"][1], star))

    # ---- figure: matrix, then the same data as curves vs k, then M/N for contrast -------------
    laws = {(p, lab): fit_law(runs, p, fn) for p in PENS for fn, lab, _ in STATS}

    def law_text(p, lab):
        """One-line rendering of the fitted law, or why it could not be fitted."""
        f = laws.get((p, lab))
        if not f:
            return "law not fittable\n(needs k>=3 at >=2 sizes)"
        return (f"${lab.split('/')[0]} = {f["A"]:.2f}\\,N^{{{f['b']:.2f}}}k^{{{f['c']:+.2f}}}$\n"
                f"$b$={f['b']:.2f} [{f['b_ci'][0]:.2f},{f['b_ci'][1]:.2f}]   "
                f"$c$={f['c']:+.2f} [{f['c_ci'][0]:+.2f},{f['c_ci'][1]:+.2f}]"
                + ("" if f["c_ci"][0] <= 0 <= f["c_ci"][1] else "   c≠0"))

    fig, ax = plt.subplots(3, len(PENS), figsize=(4.3 * len(PENS), 11.2), squeeze=False)
    for c_i, pen in enumerate(PENS):
        Z, S, C = cells(runs, pen, participation_ratio, ks, Ns)
        a = ax[0][c_i]
        if not np.isfinite(Z).any():
            a.text(.5, .5, f"no {pen} data yet", ha="center", va="center", transform=a.transAxes,
                   color="0.5", fontsize=11)
            a.set_xticks([]); a.set_yticks([])
        else:
            im = a.imshow(Z, cmap="magma", vmin=0, vmax=1, aspect="auto")
            for i in range(len(Ns)):
                for j in range(len(ks)):
                    if np.isfinite(Z[i, j]):
                        col = "white" if Z[i, j] < 0.6 else "black"
                        a.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="bottom", fontsize=7.5,
                               color=col)
                        a.text(j, i, f"±{S[i, j]:.2f}", ha="center", va="top", fontsize=5.6,
                               color=col, alpha=.85)
                    else:
                        a.text(j, i, "·", ha="center", va="center", color="0.6", fontsize=9)
            a.set(xticks=range(len(ks)), xticklabels=ks, yticks=range(len(Ns)),
                  yticklabels=[str(n) for n in Ns])
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
        f = laws.get((pen, "PR/N"))
        sub = (f"$PR = {f['A']:.2f}N^{{{f['b']:.2f}}}k^{{{f['c']:+.2f}}}$" if f else "law not fittable")
        a.set_title(f"{pen}\nPR/N over the (N, k) grid\n{sub}", fontsize=10.5, fontweight="bold")
        if c_i == 0:
            a.set_ylabel("N (units)")

        for row, (fn, lab, note) in enumerate(STATS, start=1):
            b = ax[row][c_i]
            Zg, Sg, _ = cells(runs, pen, fn, ks, Ns)
            if not np.isfinite(Zg).any():
                b.text(.5, .5, "—", ha="center", va="center", transform=b.transAxes, color="0.6")
                b.set_xticks([]); b.set_yticks([]); continue
            for i, N in enumerate(Ns):
                if np.isfinite(Zg[i]).any():
                    ps.band(b, ks, Zg[i], Sg[i], ps.col_n(N), label=f"N={N}")
            # overlay the fitted law, dividing by N since the panel plots Y/N not Y
            f = laws.get((pen, lab))
            if f:
                kk = np.linspace(min(ks), max(ks), 100)
                for i, N in enumerate(Ns):
                    if np.isfinite(Zg[i]).any():
                        b.plot(kk, f["A"] * N ** (f["b"] - 1) * kk ** f["c"], "--",
                               color=ps.col_n(N), lw=1.1, alpha=.75)
            b.set(xlabel="k (bits)", ylabel=lab, xticks=ks, ylim=(0, 1.05),
                  title=f"{lab} vs k — {note}\ndashed = fitted law")
            b.text(.03, .04, law_text(pen, lab), transform=b.transAxes, fontsize=6.8,
                   va="bottom", ha="left",
                   bbox=dict(fc="white", ec="0.7", alpha=.85, boxstyle="round,pad=0.3"))
            b.legend(fontsize=7, loc="upper right")
    fig.suptitle("Participation ratio over the (N, k) grid, per penalty\n"
                 f"every network read where its loss reaches {1+EXCESS_DELTA:.2f}x its OWN floor  ·  "
                 "PR/N = effective fraction of units participating (1.0 = perfectly even)",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return ps.save(fig, f"pr_matrix_d{1+EXCESS_DELTA:.2f}", tight=False)


if __name__ == "__main__":
    main()
