#!/usr/bin/env python3
"""
The flip-flop grid: loss floor, active-unit count, and learning-dynamics parameters vs (k, N, penalty).

Three questions, one loader, because all three read the SAME per-run objects and the comparability
argument is identical for all of them.

  (1) FLOOR vs k and N. L_inf from a stretched-exponential fit to the noise-free loss. The
      unpenalised grid gives floor_per_channel(k) = a + b*sqrt(k) with a = 0.0208, b = 0.0029 and
      residuals at the seed-noise level, exactly N-independent. Both properties are re-checked here
      against the completed grid and against each penalty condition, since the pre-registered test in
      docs/experiments/penalty_vs_interference.md turns on whether the penalty moves b (interference)
      or a (single-channel accuracy).

  (2) ACTIVE UNITS vs k and N. M ~ N^b with b ~ 0.37 unpenalised. frm should drive the silent
      fraction to ~0 by construction, so the question is what happens to the EXPONENT.

  (3) LEARNING DYNAMICS. The same fit's A, tau, beta - amplitude, timescale, and stretch. beta < 1
      means a broad spectrum of timescales rather than a single relaxation rate.

THE ONE THING THAT MAKES THIS COMPARABLE AT ALL: A COMMON READ-OUT ITERATION.
Budgets across the grid are NOT equal - 500k (unpenalised N=500,1000), 400k (unpenalised N=2000),
150k (k=1 and the whole penalty grid). That does not matter for a quantity that converges and is
fatal for one that does not:

  - the floor converges by ~25k, so any budget past that identifies it - but the fit RANGE is still
    forced identical for every cell, because mismatched ranges manufactured a spurious trend three
    separate times in the CDDM analysis.
  - the active-unit count does NOT converge; it was still moving +5 pp per doubling at 490k. Reading
    a 150k penalised cell against a 500k unpenalised one would compare training depth, not penalty.

So every cell is read at T_READ = 150000, the largest budget every cell has, by reading the longer
traces BACKWARD. A cell that never reaches T_READ is dropped and named rather than silently
compared.

MEASURED CONSEQUENCE, which splits the two quantities apart. Truncating the floor fit to T_READ
costs <= 1.9% on most cells but 4.0% on k=5 N=2000, where the truncated fit falls into the A/tau
degeneracy and drags that cell's sqrt-law residual to 3.5% against 0.6% for the same cell fitted over
its full budget. Bounding tau to the fitted range was tried as a fix and rejected: it binds on most
cells and pushes a at N=2000 to 0.0218, breaking the N-independence that the untruncated fits show
cleanly. So BOTH fits are reported and each comparison uses the one it needs:
  - cross-k and cross-N WITHIN a penalty condition -> full-budget fits. Those budgets are already
    near-matched (500k/500k/400k) and the floor has converged, so nothing is gained by truncating.
  - cross-PENALTY (none vs rws/frm/both) -> matched T_READ fits on both sides, because the penalty
    grid has only 150k and this is the one comparison where the ranges genuinely differ.
  - active units -> matched T_READ always. That count has not converged and never gets the full
    budget treatment.

Losses come from `loss_clean_train` (the noise-free forward pass, recorded every probe), never
TrainLosses.json, which is task + lambda*penalty with noise on and is not comparable across penalty
conditions at all.

Output: img/internal_figures/flipflop_grid_{floor,active,dynamics}.png

Usage:  python flipflop_grid.py [T_READ]
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, active_count, excess_time, logbin, stretched

KSWEEP = "data/trained_RNNs/NBitFlipFlop_std_ksweep"
PENSWEEP = "data/trained_RNNs/NBitFlipFlop_std_pen"
PROBE_EVERY = 10        # trainer.track_every; loss_clean_train is indexed in probes, not iterations
T_READ = 150000         # common read-out iteration: the largest budget EVERY cell has
T_START = 2000          # skips the initial collapse, which no smooth model describes
NBINS = 60
PENS = ["none", "rws", "frm", "both"]
PCOL = {"none": "k", "rws": "C0", "frm": "C3", "both": "C2"}


def load():
    """Load every completed run from both sweeps into a flat list of run records.

    Reads the unpenalised k-sweep and the penalty grid, keying each run by its penalty condition,
    bit count k and size N. Runs whose noise-free loss contains a NaN, or which never reach T_READ,
    are dropped and counted.

    Returns:
        (runs, dropped) where runs is a list of dicts with keys pen, k, N, loss (float array indexed
        by probe), part (list of participation vectors), piters (their iterations), budget (int);
        and dropped is a list of (reason, pen, k, N) tuples.
    """
    runs, dropped = [], []
    pats = [(KSWEEP, r"_k=(\d+)_N=(\d+)_iters=(\d+)", None),
            (PENSWEEP, r"_k=(\d+)_N=(\d+)_pen=(\w+)", None)]
    for root, pat, _ in pats:
        for f in sorted(glob.glob(os.path.join(root, "*", "*", "*ParticipationTrace.pkl"))):
            m = re.search(pat, f)
            if not m:
                continue
            k, N = int(m.group(1)), int(m.group(2))
            pen = "none" if root == KSWEEP else m.group(3)
            with open(f, "rb") as fh:
                tr = pickle.load(fh)
            L = np.asarray(tr["metrics"].get("loss_clean_train", []), dtype=float)
            if L.size == 0 or np.isnan(L).any():
                dropped.append(("nan/empty loss", pen, k, N))
                continue
            budget = len(L) * PROBE_EVERY
            if budget < T_READ:
                dropped.append((f"budget {budget} < T_READ", pen, k, N))
                continue
            runs.append(dict(pen=pen, k=k, N=N, loss=L, budget=budget,
                             part=tr["participation"],
                             piters=np.asarray(tr["participation_iters"], dtype=float)))
    if not runs:
        raise SystemExit(f"no completed runs under {KSWEEP} or {PENSWEEP}")
    return runs, dropped


def fit_run(L, t_end):
    """Fit one run's noise-free loss trace with a stretched exponential, in log space.

    Args:
        L: noise-free loss indexed by probe (probe i is iteration (i+1)*PROBE_EVERY);
        t_end: last iteration included, so every run is fitted over the identical range
            [T_START, t_end].
    Returns:
        (L_inf, A, tau, beta) or None if the fit fails or the range holds too few bins.
    """
    t = (np.arange(len(L)) + 1) * PROBE_EVERY
    m = (t >= T_START) & (t <= t_end)
    tb, yb = logbin(t[m], L[m])
    if len(tb) < 8:
        return None
    p0 = [yb.min() * 0.9, float(yb.max()), 2e4, 0.4]
    lo, hi = [1e-6, 1e-6, 1e2, 0.05], [1.0, 1e3, 1e8, 3.0]
    try:
        sol = least_squares(lambda p: np.log(np.clip(stretched(tb, *p), 1e-12, None)) - np.log(yb),
                            p0, bounds=(lo, hi), max_nfev=20000)
    except Exception:
        return None
    return tuple(sol.x)


def M_at(run, criterion, t):
    """Active-unit count for one run at the last participation snapshot at or before iteration t.

    Args:
        run: run record from load(); criterion: "hard" or "scale-free"; t: read-out iteration.
    Returns:
        (count, iteration actually used) or (None, None) if no snapshot is at or before t.
    """
    ok = np.flatnonzero(run["piters"] <= t)
    if ok.size == 0:
        return None, None
    i = int(ok[-1])
    return active_count(np.asarray(run["part"][i], dtype=float), criterion), float(run["piters"][i])


def sqrt_law(ks, floors):
    """Least-squares fit of the per-channel floor law f(k) = a + b*sqrt(k).

    Args:
        ks: array of bit counts; floors: matching per-channel floor values.
    Returns:
        (a, b, max relative residual) or (nan, nan, nan) if fewer than three points.
    """
    ks, floors = np.asarray(ks, float), np.asarray(floors, float)
    if ks.size < 3:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones_like(ks), np.sqrt(ks)])
    (a, b), *_ = np.linalg.lstsq(X, floors, rcond=None)
    pred = a + b * np.sqrt(ks)
    return a, b, float(np.max(np.abs(pred - floors) / floors))


def boot_law(per_seed, n_boot=2000, rng=None):
    """Bootstrap 95% CIs on (a, b) of f(k) = a + b*sqrt(k) by resampling seeds within each k.

    The pre-registered threshold in docs/experiments/penalty_vs_interference.md is non-overlapping
    bootstrap 95% CIs, so the CI is the deciding statistic and not a decoration. Seeds are resampled
    WITHIN each k (the unit of replication is the network), then the law is refitted on the resampled
    cell means.

    Args:
        per_seed: dict {k: [L_inf per seed]}; n_boot: resamples; rng: numpy Generator.
    Returns:
        dict with point estimates and (lo, hi) 2.5/97.5 percentiles for a and b, or None if fewer
        than three k values carry data.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    ks = sorted(k for k, v in per_seed.items() if len(v))
    if len(ks) < 3:
        return None
    a0, b0, _ = sqrt_law(ks, [np.mean(per_seed[k]) for k in ks])
    A, B = [], []
    for _ in range(n_boot):
        mu = [np.mean(rng.choice(per_seed[k], size=len(per_seed[k]), replace=True)) for k in ks]
        a, b, _ = sqrt_law(ks, mu)
        A.append(a); B.append(b)
    return dict(a=a0, b=b0, nk=len(ks),
                a_ci=(float(np.percentile(A, 2.5)), float(np.percentile(A, 97.5))),
                b_ci=(float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))))


def agg(rows, key):
    """Mean and sd of one field over runs, grouped by (pen, k, N).

    Args:
        rows: list of per-run dicts already carrying fitted/derived fields; key: field name.
    Returns:
        dict {(pen, k, N): (mean, sd, n)} skipping runs whose field is None/NaN.
    """
    box = {}
    for r in rows:
        v = r.get(key)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        box.setdefault((r["pen"], r["k"], r["N"]), []).append(v)
    return {g: (float(np.mean(v)), float(np.std(v)), len(v)) for g, v in box.items()}


def panel_vs_k(ax, table, pen, Ns, ks, ylabel, title, logy=False):
    """Plot one aggregated quantity against k, one line per N, for a single penalty condition."""
    for N in Ns:
        xs = [k for k in ks if (pen, k, N) in table]
        if not xs:
            continue
        mu = [table[(pen, k, N)][0] for k in xs]
        sd = [table[(pen, k, N)][1] for k in xs]
        ax.errorbar(xs, mu, yerr=sd, fmt="o-", lw=1.8, ms=5, capsize=3,
                    color={500: "C0", 1000: "C3", 2000: "C2"}.get(N, "C7"), label=f"N={N}")
    if logy:
        ax.set_yscale("log")
    ax.set(xlabel="k (bits)", ylabel=ylabel, title=title, xticks=ks)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)


def main():
    """Fit every run, then emit the floor, active-unit and learning-dynamics figures."""
    global T_READ
    if len(sys.argv) > 1:
        T_READ = int(sys.argv[1])

    runs, dropped = load()
    print(f"loaded {len(runs)} runs; T_READ = {T_READ} (common budget, longer traces read backward)")
    if dropped:
        print(f"dropped {len(dropped)}:")
        for d in sorted(set(dropped)):
            print("   ", d, f"x{dropped.count(d)}")

    # Fit every run over the IDENTICAL range [T_START, T_READ], plus a halved range as the
    # pre-registered budget check: if L_inf moves when the range is halved, the run has not reached
    # its floor and the fitted value is tracking the last data point, not an asymptote.
    for r in runs:
        f = fit_run(r["loss"], T_READ)                  # primary: identical range for every cell
        h = fit_run(r["loss"], T_READ // 2)             # halved range: the budget check
        g = fit_run(r["loss"], r["budget"])             # this cell's full budget: truncation check
        r["L_inf"], r["A"], r["tau"], r["beta"] = f if f else (None,) * 4
        r["L_inf_half"] = h[0] if h else None
        r["beta_half"] = h[3] if h else None
        r["L_inf_full"] = g[0] if g else None
        if f:
            r["t_half"] = excess_time(*f[1:], 0.5, T_START)           # iterations to halve the excess over L_inf
            r["t_90"] = excess_time(*f[1:], 0.1, T_START)             # ... and to remove 90% of it
        for crit, tag in [("scale-free", "M_sf"), ("hard", "M_hard")]:
            r[tag], r["t_read_actual"] = M_at(r, crit, T_READ)
        r["silent_frac"] = (1 - r["M_sf"] / r["N"]) if r["M_sf"] is not None else None

    floors = agg(runs, "L_inf")
    halves = agg(runs, "L_inf_half")
    taus, betas, amps = agg(runs, "tau"), agg(runs, "beta"), agg(runs, "A")
    thalf, t90 = agg(runs, "t_half"), agg(runs, "t_90")
    fulls, bhalf = agg(runs, "L_inf_full"), agg(runs, "beta_half")
    Msf, Mhard = agg(runs, "M_sf"), agg(runs, "M_hard")
    sfrac = agg(runs, "silent_frac")

    pens = [p for p in PENS if any(g[0] == p for g in floors)]
    Ns = sorted({g[2] for g in floors})
    ks = sorted({g[1] for g in floors})

    # ---- budget check, before any interpretation of the floors ------------------------------
    print(f"\nBUDGET CHECK. L_inf must not move when the fit range is halved; if it does, the fit is")
    print(f"tracking the last data point rather than an asymptote and floors must not be compared.")
    print(f"  primary range   [{T_START}, {T_READ}]     (identical for every cell)")
    print(f"  halved range    [{T_START}, {T_READ//2}]")
    print(f"  full range      [{T_START}, cell budget]  (checks what truncation to {T_READ} costs)")
    rows = []
    for g in sorted(floors):
        d_half = abs(halves[g][0] - floors[g][0]) / floors[g][0] if g in halves else np.nan
        d_full = abs(fulls[g][0] - floors[g][0]) / floors[g][0] if g in fulls else np.nan
        d_beta = abs(bhalf[g][0] - betas[g][0]) if g in bhalf else np.nan
        rows.append((d_half, d_full, d_beta, g))
    rows.sort(reverse=True)
    wh = np.nanmax([r[0] for r in rows]); wf = np.nanmax([r[1] for r in rows])
    print(f"  worst |dL_inf| on halving: {100*wh:5.2f}%   "
          f"({'identified' if wh < 0.05 else 'NOT identified'} at a 5% threshold)")
    print(f"  worst |dL_inf| vs full budget: {100*wf:5.2f}%   "
          f"(cost of truncating to the common {T_READ})")
    print("  worst five cells on halving:")
    for d_half, d_full, d_beta, g in rows[:5]:
        print("    %-5s k=%d N=%-5d  halve %5.2f%%  full %5.2f%%  dbeta %.3f"
              % (g[0], g[1], g[2], 100 * d_half, 100 * d_full, d_beta))

    # ---- table ------------------------------------------------------------------------------
    print("\nA and tau are reported for completeness but are NOT individually identified (see")
    print("t_frac docstring); t_half and t_90 are the identifiable timescales and beta the shape.")
    print("\n%5s %3s %6s %2s %10s %8s %9s %7s %9s %9s %8s %8s %7s"
          % ("pen", "k", "N", "n", "L_inf", "A", "tau", "beta", "t_half", "t_90",
             "M_sf", "M_hard", "silent"))
    for pen in pens:
        for k in ks:
            for N in Ns:
                g = (pen, k, N)
                if g not in floors:
                    continue
                print("%5s %3d %6d %2d %8.5f+-%.5f %8.4f %9.0f %7.3f %9.0f %9.0f %8.1f %8.1f %6.1f%%"
                      % (pen, k, N, floors[g][2], floors[g][0], floors[g][1], amps[g][0],
                         taus[g][0], betas[g][0],
                         thalf.get(g, (np.nan,))[0], t90.get(g, (np.nan,))[0],
                         Msf.get(g, (np.nan,))[0], Mhard.get(g, (np.nan,))[0],
                         100 * sfrac.get(g, (np.nan,))[0]))

    # ---- the a + b*sqrt(k) law per penalty condition -----------------------------------------
    print("\nper-channel floor law  f(k) = a + b*sqrt(k)   (a = single-channel, b = interference)")
    print("%5s %6s %4s %11s %11s %10s" % ("pen", "N", "nk", "a", "b", "max resid"))
    for pen in pens:
        for N in Ns:
            xs = [k for k in ks if (pen, k, N) in floors]
            if len(xs) < 3:
                continue
            a, b, res = sqrt_law(xs, [floors[(pen, k, N)][0] for k in xs])
            print("%5s %6d %4d %11.5f %11.5f %9.2f%%" % (pen, N, len(xs), a, b, 100 * res))

    # ---- M ~ N^b per (pen, k) ----------------------------------------------------------------
    # Bootstrap CIs on the matched-T_READ fits: the pre-registered cross-penalty comparison.
    print("\nbootstrap 95% CIs on the MATCHED-{} fits — the pre-registered cross-penalty test.".format(T_READ))
    print("H predicts penalised b BELOW none's with non-overlapping CIs; a is the control and")
    print("should NOT move (if it does, the penalty changes single-channel accuracy, not interference).")
    print("%5s %6s %3s %28s %28s" % ("pen", "N", "nk", "a  [95% CI]", "b  [95% CI]"))
    rng_b = np.random.default_rng(0)
    for pen in pens:
        for N in Ns:
            per_seed = {}
            for r in runs:
                if r["pen"] == pen and r["N"] == N and r.get("L_inf") is not None:
                    per_seed.setdefault(r["k"], []).append(r["L_inf"])
            bl = boot_law(per_seed, rng=rng_b)
            if bl:
                print("%5s %6d %3d   %.5f [%.5f, %.5f]   %.5f [%.5f, %.5f]"
                      % (pen, N, bl["nk"], bl["a"], bl["a_ci"][0], bl["a_ci"][1],
                         bl["b"], bl["b_ci"][0], bl["b_ci"][1]))

    print("\nsame law on the FULL-budget fits, as a robustness column: if a and b move between the two")
    print("blocks the difference is fit noise from truncation, not a property of the network.")
    print("%5s %6s %4s %11s %11s %10s" % ("pen", "N", "nk", "a", "b", "max resid"))
    for pen in pens:
        for N in Ns:
            xs = [k for k in ks if (pen, k, N) in fulls]
            if len(xs) < 3:
                continue
            a, b, res = sqrt_law(xs, [fulls[(pen, k, N)][0] for k in xs])
            print("%5s %6d %4d %11.5f %11.5f %9.2f%%" % (pen, N, len(xs), a, b, 100 * res))

    print("\nactive-unit scaling  M ~ N^b  (scale-free criterion, read at T_READ)")
    print("%5s %3s %5s %s" % ("pen", "k", "b", "  M per N"))
    for pen in pens:
        bs = []
        for k in ks:
            pts = [(N, Msf[(pen, k, N)][0]) for N in Ns if (pen, k, N) in Msf]
            if len(pts) < 2:
                continue
            x = np.log([p[0] for p in pts])
            y = np.log([p[1] for p in pts])
            b = float(np.polyfit(x, y, 1)[0])
            bs.append(b)
            print("%5s %3d %5.2f   %s" % (pen, k, b,
                  "  ".join(f"{int(p[0])}:{p[1]:.0f}" for p in pts)))
        if bs:
            print("%5s  mean b = %.2f +- %.2f over %d k values" % (pen, np.mean(bs), np.std(bs), len(bs)))

    # ---- FIGURE 1: floor ---------------------------------------------------------------------
    fig, ax = plt.subplots(len(pens), 3, figsize=(16, 4.4 * len(pens)), squeeze=False)
    for r, pen in enumerate(pens):
        panel_vs_k(ax[r][0], floors, pen, Ns, ks, "per-channel floor $L_\\infty$",
                   f"({pen}) floor vs complexity")
        # overlay the sqrt law fitted on the first N with enough k values, as a shape guide
        for N in Ns:
            xs = [k for k in ks if (pen, k, N) in floors]
            if len(xs) < 3:
                continue
            a, b, _ = sqrt_law(xs, [floors[(pen, k, N)][0] for k in xs])
            xx = np.linspace(min(xs), max(xs), 100)
            ax[r][0].plot(xx, a + b * np.sqrt(xx), "--", lw=1.2, alpha=.7,
                          color={500: "C0", 1000: "C3", 2000: "C2"}.get(N, "C7"))
            break
        for N in Ns:                    # open markers: same fit over each cell's FULL budget
            xs = [k for k in ks if (pen, k, N) in fulls]
            if not xs:
                continue
            ax[r][0].plot(xs, [fulls[(pen, k, N)][0] for k in xs], "s", ms=4, mfc="none",
                          color={500: "C0", 1000: "C3", 2000: "C2"}.get(N, "C7"), alpha=.9)
        ax[r][0].text(.03, .95, "dashed: $a+b\\sqrt{k}$\nfilled: fit to 150k (matched)\n"
                                "open: fit to full budget", transform=ax[r][0].transAxes,
                      va="top", fontsize=7)

        for k in ks:                                    # floor vs N: the N-independence check
            xs = [N for N in Ns if (pen, k, N) in floors]
            if len(xs) < 2:
                continue
            ax[r][1].errorbar(xs, [floors[(pen, k, N)][0] for N in xs],
                              yerr=[floors[(pen, k, N)][1] for N in xs], fmt="o-", lw=1.5, ms=5,
                              capsize=3, color=plt.cm.viridis((k - min(ks)) / max(1, max(ks) - min(ks))),
                              label=f"k={k}")
        ax[r][1].set(xlabel="N (units)", ylabel="per-channel floor $L_\\infty$", xscale="log",
                     title=f"({pen}) floor vs size — flat = N-independent")
        ax[r][1].grid(alpha=.3)
        if ax[r][1].get_legend_handles_labels()[0]:
            ax[r][1].legend(fontsize=6, ncol=2)

        for N in Ns:                                    # total floor = k * per-channel
            xs = [k for k in ks if (pen, k, N) in floors]
            if not xs:
                continue
            ax[r][2].plot(xs, [k * floors[(pen, k, N)][0] for k in xs], "o-", lw=1.8, ms=5,
                          color={500: "C0", 1000: "C3", 2000: "C2"}.get(N, "C7"), label=f"N={N}")
        ax[r][2].set(xlabel="k (bits)", ylabel="total floor $k\\cdot L_\\infty$", xticks=ks,
                     title=f"({pen}) total floor — $ka + bk^{{1.5}}$")
        ax[r][2].grid(alpha=.3); ax[r][2].legend(fontsize=7)
    fig.suptitle(f"Flip-flop loss floor vs complexity, size and penalty  "
                 f"(stretched-exponential $L_\\infty$, fitted on [{T_START}, {T_READ}])", fontsize=12)
    fig.tight_layout()
    out1 = os.path.join(IMG_DIR, "flipflop_grid_floor.png")
    fig.savefig(out1, dpi=150); plt.close(fig)

    # ---- FIGURE 2: active units --------------------------------------------------------------
    fig, ax = plt.subplots(len(pens), 3, figsize=(16, 4.4 * len(pens)), squeeze=False)
    for r, pen in enumerate(pens):
        for k in ks:                                    # M vs N, log-log: the scaling exponent
            xs = [N for N in Ns if (pen, k, N) in Msf]
            if len(xs) < 2:
                continue
            c = plt.cm.viridis((k - min(ks)) / max(1, max(ks) - min(ks)))
            ax[r][0].errorbar(xs, [Msf[(pen, k, N)][0] for N in xs],
                              yerr=[Msf[(pen, k, N)][1] for N in xs], fmt="o-", lw=1.5, ms=5,
                              capsize=3, color=c, label=f"k={k}")
        if Ns:
            xx = np.array(Ns, float)
            ax[r][0].plot(xx, 150 * (xx / xx[0]), "k--", lw=1.2, label="M ∝ N (no saturation)")
        ax[r][0].set(xlabel="N (units)", ylabel="active units M (scale-free)", xscale="log",
                     yscale="log", title=f"({pen}) M vs N — slope = scaling exponent")
        ax[r][0].grid(alpha=.3, which="both")
        if ax[r][0].get_legend_handles_labels()[0]:
            ax[r][0].legend(fontsize=6, ncol=2)

        panel_vs_k(ax[r][1], Msf, pen, Ns, ks, "active units M (scale-free)",
                   f"({pen}) M vs complexity")
        panel_vs_k(ax[r][2], sfrac, pen, Ns, ks, "silent fraction (scale-free)",
                   f"({pen}) silent fraction vs complexity")
    fig.suptitle(f"Flip-flop active units vs complexity, size and penalty  "
                 f"(all cells read at iteration {T_READ})", fontsize=12)
    fig.tight_layout()
    out2 = os.path.join(IMG_DIR, "flipflop_grid_active.png")
    fig.savefig(out2, dpi=150); plt.close(fig)

    # ---- FIGURE 3: learning dynamics ---------------------------------------------------------
    fig, ax = plt.subplots(len(pens), 3, figsize=(16, 4.4 * len(pens)), squeeze=False)
    for r, pen in enumerate(pens):
        panel_vs_k(ax[r][0], thalf, pen, Ns, ks, "iterations",
                   f"({pen}) $t_{{1/2}}$ — halve the excess over $L_\\infty$", logy=True)
        panel_vs_k(ax[r][1], betas, pen, Ns, ks, r"$\beta$ (stretch)",
                   f"({pen}) stretch $\\beta$ — <1 = broad spectrum")
        ax[r][1].axhline(1.0, color="k", lw=1, alpha=.5)
        panel_vs_k(ax[r][2], t90, pen, Ns, ks, "iterations",
                   f"({pen}) $t_{{90}}$ — remove 90% of the excess", logy=True)
    fig.suptitle(r"Flip-flop learning dynamics: $L(t) = L_\infty + A\,e^{-(t/\tau)^\beta}$",
                 fontsize=12)
    fig.tight_layout()
    out3 = os.path.join(IMG_DIR, "flipflop_grid_dynamics.png")
    fig.savefig(out3, dpi=150); plt.close(fig)

    print(f"\nwrote {out1}\n      {out2}\n      {out3}")


if __name__ == "__main__":
    main()
