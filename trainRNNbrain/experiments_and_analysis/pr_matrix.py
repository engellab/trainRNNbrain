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

Usage:  python pr_matrix.py [EXCESS_DELTA | iter=150000 | rho=0.10 | r2=0.98]
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from trainRNNbrain.tasks.TaskNBitFlipFlop import TaskNBitFlipFlop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, active_count, logbin, participation_ratio, stretched
import plotstyle as ps
from criterion_search import rho_series, first_sustained

ROOTS = {"ksweep": "data/trained_RNNs/NBitFlipFlop_std_ksweep",
         "pen": "data/trained_RNNs/NBitFlipFlop_std_pen",
         "penlong": "data/trained_RNNs/NBitFlipFlop_std_penlong",
         # ⚠️ bigN carries the ONLY N=4000 cells (none and rws). It was missing here while
         # drift_matrix had it, so every PR figure silently fitted N over 500/1000/2000 only -
         # three points - and the size exponent b was reported without the size that most
         # constrains it. N=4000 also runs a 100k budget against 400-500k elsewhere, so read-outs
         # must be budget-independent (the excess criterion), never the endpoint.
         "bigN": "data/trained_RNNs/NBitFlipFlop_std_bigN"}
SKIP = {("pen", "frm")}          # retracted 150k frm cells; frm comes from penlong
PENS = ["none", "rws", "frm", "both"]
PROBE = 10
T_START = 2000
MIN_ITERS = 50_000
R2_MIN = 0.5           # below this the run never solved the task; see load()
EXCESS_DELTA = 0.10
R2_FRAC = 0.98         # r2= mode: read where R^2 reaches this fraction of its fitted ceiling


def target_variance():
    """R^2 denominator: global variance of the target over the masked window.

    R^2(t) = 1 - loss_clean_train(t) / V EXACTLY, because `loss_clean_train` is the numerator of
    Trainer.r2_score - same mask, same noise-free probe, same fresh-batch distribution. No R^2 trace
    is stored during training, so this affine map is the only way to get one. Validated against an
    independent oracle (RNN_numpy forward + training_utils.r2, a different code path) on 5 runs
    spanning k=1..8 and N=500..2000: max |diff| = 0.0016.

    ⚠️ NOT the folder-name r2, which comes from eval_step(noise=True) and so reads ~0.010 LOWER
    than any noise-free number. It is a biased oracle and must not be used to check this.

    Task params are read from a run config, never hardcoded. V is k-INDEPENDENT to <1% (0.7192 over
    k=1..8) because every bit channel is an i.i.d. copy of the same pulse process, so one value
    serves the whole grid.

    Returns:
        float: population variance of the target, ~0.724 for this grid.
    """
    cfg = OmegaConf.load(sorted(glob.glob(
        os.path.join(ROOTS["ksweep"], "*", "*", "*_config.yaml")))[0])
    T = int(cfg.task.T)
    # The affine map holds only if the mask is the whole trial; CDDM's two-window mask would need
    # the target restricted before taking the variance. Fail loudly rather than silently mis-scale.
    assert [eval(m) for m in cfg.task.mask_params] == [(0, T)], \
        f"target_variance() assumes a full-trial mask, got {cfg.task.mask_params}"
    t = TaskNBitFlipFlop(n_steps=T, n_inputs=int(cfg.task.n_inputs),
                         n_outputs=int(cfg.task.n_outputs), mu=float(cfg.task.mu),
                         n_flip_steps=int(cfg.task.n_flip_steps), batch_size=20000, seed=0)
    _, tgt, _ = t.get_batch()
    return float(((tgt - tgt.mean()) ** 2).mean())


def r2_time(L, floor, V, frac):
    """Iteration where the smoothed R^2 first reaches `frac` x this run's fitted R^2 ceiling.

    The ceiling is R2_max = 1 - floor/V, so the condition R^2 >= frac * R2_max rearranges to
    L <= (1 - frac) * V + frac * floor: frac of the run's own floor PLUS an absolute slack of
    (1 - frac) * V set by the task, not by the run. That is what makes it different from `excess`,
    whose whole tolerance scales with the floor and so hands a run with a poor floor a
    proportionally wider window.

    Args:
        L: (n_probes,) noise-free loss trace, indexed in probes of PROBE iterations.
        floor: fitted loss floor L_inf for this run, or None if the fit failed.
        V: target variance from target_variance().
        frac: fraction of the ceiling to read at (0.98 as requested).
    Returns:
        float: iteration of first sustained crossing, or nan if never reached / no floor.
    """
    w = 21
    if floor is None or len(L) < w:
        return float("nan")
    thr = (1.0 - frac) * V + frac * floor
    s = np.convolve(L, np.ones(w) / w, mode="valid")
    hit = np.flatnonzero(s <= thr)
    return float((hit[0] + w // 2 + 1) * PROBE) if len(hit) else float("nan")


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
            # ⚠️ DROP RUNS THAT NEVER LEARNED THE TASK. 11 runs in the grid ended with NEGATIVE r2
            # (to -12.4) yet perfectly FINITE losses, so the isnan() check below misses them. Their
            # floors and participation ratios describe a network that never solved the task.
            # Score is the folder-name prefix; the success/failure gap is -0.32 .. 0.857, so any
            # cut inside it gives the same answer.
            try:
                r2 = float(os.path.basename(f).split("_")[0])
            except ValueError:
                r2 = float("nan")
            if not (r2 >= R2_MIN):
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
    return fit_power_law(K, NN, Y)


def fit_power_law(K, NN, Y):
    """Least-squares Y = A N^b k^c in logs, with 2000-resample bootstrap CIs on b and c.

    Args:
        K, NN, Y: equal-length sequences of k, N and the positive statistic being fitted.
    Returns:
        dict with A, b, c, b_ci, c_ci, n; or None if the design cannot support the fit
        (needs >=8 points, >=2 distinct N and >=3 distinct k).
    """
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
    fixed_iter = rho_frac = r2_frac = None
    if len(sys.argv) > 1:
        if sys.argv[1].startswith("iter="):
            fixed_iter = float(sys.argv[1].split("=")[1])
        elif sys.argv[1].startswith("rho="):
            rho_frac = float(sys.argv[1].split("=")[1])
        elif sys.argv[1].startswith("r2="):
            r2_frac = float(sys.argv[1].split("=")[1])
        else:
            EXCESS_DELTA = float(sys.argv[1])
    ps.setup()
    V = target_variance() if r2_frac is not None else None
    runs = load()
    for r in runs:
        r["floor"] = fit_floor(r["loss"], r["budget"])       # OWN budget, not a common range
        if rho_frac is not None:
            # rho = -dlog(L)/dlog(t): the local log-log slope of the loss, i.e. how fast the loss
            # still improves per decade. Floor-free and fit-free, and because it is a LOG
            # derivative it is invariant to the overall loss scale, so conditions with different
            # achievable floors are directly comparable. Read where rho first falls below
            # `rho_frac` of its OWN early peak and stays there (self-referential, so the level is
            # reachable by construction - unlike an absolute alpha < 0.5).
            t_r, rho = rho_series(r["loss"])
            if len(rho):
                peak = float(np.max(rho[:max(3, len(rho) // 3)]))
                r["T"] = (first_sustained(t_r, rho, rho_frac * peak) if peak > 0 else float("nan"))
            else:
                r["T"] = float("nan")
        elif r2_frac is not None:
            # R^2 = 1 - L/V, so the fitted loss floor IS the fitted R^2 ceiling; no second fit.
            r["r2max"] = 1.0 - r["floor"] / V if r["floor"] is not None else float("nan")
            # A ceiling <= 0 means the fitted floor is no better than predicting the target mean,
            # i.e. the floor fit failed - one none/k=1/N=1000 run fits L_inf = 0.728 against a
            # final loss of 0.023. The excess criterion cannot see this (1.01 x a huge floor is
            # crossed at t=590) and carries the garbage read-out into every pr_matrix_dX figure.
            # Expressing the read-out in R^2 makes the failure checkable, so check it.
            r["T"] = (r2_time(r["loss"], r["floor"], V, r2_frac)
                      if r["r2max"] > 0 else float("nan"))
        elif fixed_iter is not None:
            # ⚠️ FIXED-ITERATION READ-OUT, kept only as the confounded reference. Harder tasks
            # settle later (settling time ~ k^0.458), so every k is read at a DIFFERENT depth of
            # its own convergence and the fitted c inherits that difference. It is also UNDEFINED
            # for any run whose budget is shorter than the chosen iteration - at 150k that silently
            # removes every N=4000 cell, whose budget is 100k.
            r["T"] = fixed_iter if fixed_iter <= r["budget"] else float("nan")
        else:
            r["T"] = excess_time(r["loss"], r["floor"], EXCESS_DELTA)
    n_before = {p: sum(1 for r in runs if r["pen"] == p) for p in PENS}
    runs = [r for r in runs if np.isfinite(r["T"])]
    ks = sorted({r["k"] for r in runs})
    Ns = sorted({r["N"] for r in runs})
    have = [p for p in PENS if any(r["pen"] == p for r in runs)]
    if r2_frac is not None:
        print(f"read-out: R^2 = 1 - L/V reaches {r2_frac:.3f} x each run's OWN fitted ceiling\n"
              f"  V (target variance) = {V:.5f}, k-independent\n"
              f"  ceiling R2_max = 1 - floor/V, per run; threshold on the loss is "
              f"{1-r2_frac:.3f}*V + {r2_frac:.3f}*floor\n")
        for p in PENS:
            sel = [r for r in runs if r["pen"] == p and np.isfinite(r.get("r2max", np.nan))]
            if sel:
                rm = np.array([r["r2max"] for r in sel])
                print(f"  {p:5s}  fitted R2_max  median {np.median(rm):.4f}  "
                      f"[{np.min(rm):.4f}, {np.max(rm):.4f}]  ->  read at "
                      f"{r2_frac*np.median(rm):.4f}")
        print()
    else:
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
    readout_txt = (f"every network read where its $R^2$ reaches {r2_frac:.3f}x its OWN fitted ceiling"
                   if r2_frac is not None else
                   f"every network read where its loss reaches {1+EXCESS_DELTA:.2f}x its OWN floor")
    fig.suptitle("Participation ratio over the (N, k) grid, per penalty\n"
                 f"{readout_txt}  ·  "
                 "PR/N = effective fraction of units participating (1.0 = perfectly even)",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    name = (f"pr_matrix_r2_{r2_frac:.3f}" if r2_frac is not None
            else f"pr_matrix_rho{rho_frac:.2f}peak" if rho_frac is not None
            else f"pr_matrix_iter{int(fixed_iter/1000)}k" if fixed_iter is not None
            else f"pr_matrix_d{1+EXCESS_DELTA:.2f}")
    return ps.save(fig, name, tight=False)


if __name__ == "__main__":
    main()
