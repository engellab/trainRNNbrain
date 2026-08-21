#!/usr/bin/env python3
"""
Is the per-channel loss floor the same for every k? Stretched-exponential fits to the flip-flop.

WHY THIS IS THE PER-CHANNEL FLOOR. The training loss is a masked MSE averaged over ALL axes,
channels included, so it is already a per-channel quantity. Each bit is generated i.i.d. and the
target variance is k-independent (0.727-0.738 measured over k=2..6, a 1.5% spread), so L_inf is
directly comparable across k with no normalisation and no per-k rescaling. Fitting it answers "does
one bit cost the same accuracy regardless of how many other bits share the network".

MODEL. L(t) = L_inf + A exp(-(t/tau)^beta), fitted in LOG space on log-binned medians so the two
decades before the floor are not swamped by the 30000 probes near it. The stretched form won by AICc
on CDDM against a power law and a single exponential; it is used here for continuity, and the
residual panel shows whether that carries over to this task.

THE TEST. Per N, two nested fits over all k at once:
    separate  every k gets its own L_inf   (4k parameters)
    shared    one L_inf for all k, A/tau/beta still free per k   (1 + 3k parameters)
compared by F-test and AICc. A shared floor that is not rejected means matching on an absolute loss
level is legitimate across k; a rejected one means levels must be set per k.

FIT RANGE IS IDENTICAL FOR EVERY CELL by construction - all 45 runs have the same 300k budget and the
same T_START. This matters: mismatched fit ranges manufactured a spurious trend three separate times
in the CDDM analysis, and it is the single easiest way to fake a floor difference.

Losses come from `loss_clean_train` (noise-free, every probe), never TrainLosses.json.

Output: img/internal_figures/flipflop_floor.png

Usage:  python flipflop_floor.py [T_START]
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import f as fdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_drift_curves import IMG_DIR

SWEEP = "data/trained_RNNs/NBitFlipFlop_std_ksweep"
PROBE_EVERY = 10
T_START = 2000     # iterations; skips the initial collapse, which no smooth model describes
NBINS = 60


def load():
    """Load noise-free loss traces, dropping diverged runs.

    Returns:
        dict {(k, N): [loss arrays indexed by probe]}. Runs with a NaN anywhere in the trace are
        dropped (one of 45, k=2 N=2000, diverged despite carrying no penalty).
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(SWEEP, "*", "*", "*ParticipationTrace.pkl"))):
        m = re.search(r"_k=(\d+)_N=(\d+)", f)
        if not m:
            continue
        with open(f, "rb") as fh:
            tr = pickle.load(fh)
        L = np.asarray(tr["metrics"].get("loss_clean_train", []), dtype=float)
        if L.size == 0 or np.isnan(L).any():
            continue
        out.setdefault((int(m.group(1)), int(m.group(2))), []).append(L)
    return out


def logbin(t, y, nbins=NBINS):
    """Median-reduce (t, y) into log-spaced bins so the fit is not dominated by late iterations.

    Args:
        t, y: equal-length arrays, t > 0; nbins: number of log-spaced bins.
    Returns:
        (tb, yb) bin-centre and bin-median arrays, with empty bins dropped.
    """
    edges = np.logspace(np.log10(t[0]), np.log10(t[-1]), nbins + 1)
    idx = np.digitize(t, edges) - 1
    tb, yb = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() > 2:
            tb.append(np.median(t[m]))
            yb.append(np.median(y[m]))
    return np.array(tb), np.array(yb)


def curve(t, Li, A, tau, beta):
    """Stretched-exponential approach to a floor: L_inf + A exp(-(t/tau)^beta)."""
    return Li + A * np.exp(-np.power(np.clip(t / tau, 1e-12, None), beta))


def prep(L, t_end=None):
    """Log-binned (t, loss) for one run, over [T_START, t_end].

    Args:
        L: noise-free loss indexed by probe (probe i is iteration i*PROBE_EVERY);
        t_end: last iteration to include, or None for the whole run.
    Returns:
        (t, y) log-binned arrays in iteration units.
    """
    t = (np.arange(len(L)) + 1) * PROBE_EVERY
    m = (t >= T_START) if t_end is None else ((t >= T_START) & (t <= t_end))
    return logbin(t[m], L[m])


def fit_joint(data, mode):
    """Fit every curve at one N simultaneously, tying the floor at one of three granularities.

    Three nested models are needed, not two. Comparing "one floor for everything" directly against
    "one floor per seed" conflates the question asked (does the floor depend on k?) with pure
    seed-to-seed scatter, which is large here - one k=3 N=1000 cell spans +-57% across seeds. The
    per-k model is the missing middle term, and it is the one the k-question actually needs.

    Args:
        data: list of (k, t, y) log-binned curves, several seeds per k.
        mode: "one" (a single L_inf), "per_k" (one L_inf per complexity, shared across its seeds),
            or "per_seed" (one L_inf per curve).
    Returns:
        (residual vector, n_params, list of per-curve L_inf). Residuals are in log space, which
        equalises the weight of the decades above the floor against the floor itself.

    A/tau/beta are always free per curve; only the floor's granularity changes, so the comparison
    isolates the floor.
    """
    n = len(data)
    ks = sorted({k for k, _, _ in data})
    if mode == "one":
        slot = [0] * n
    elif mode == "per_k":
        slot = [ks.index(k) for k, _, _ in data]
    else:
        slot = list(range(n))
    nfloor = max(slot) + 1

    p0 = [min(y.min() for _, _, y in data) * 0.9] * nfloor
    for _, t, y in data:
        p0 += [float(y.max()), 2e4, 0.4]
    lo = [1e-6] * nfloor + [1e-6, 1e2, 0.05] * n
    hi = [1.0] * nfloor + [1e3, 1e8, 3.0] * n

    def resid(p):
        r = []
        for i, (_, t, y) in enumerate(data):
            A, tau, beta = p[nfloor + 3 * i: nfloor + 3 * i + 3]
            r.append(np.log(np.clip(curve(t, p[slot[i]], A, tau, beta), 1e-12, None)) - np.log(y))
        return np.concatenate(r)

    sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=200000)
    return sol.fun, len(p0), [sol.x[slot[i]] for i in range(n)]


def main():
    """Fit the floor per (k, N), test whether it is shared across k, and plot."""
    global T_START
    if len(sys.argv) > 1:
        T_START = int(sys.argv[1])
    by = load()
    Ns = sorted({N for _, N in by})
    ks = sorted({k for k, _ in by})
    print(f"stretched-exponential fits, T_START={T_START}, identical range for every cell")

    fig, ax = plt.subplots(len(Ns), 3, figsize=(19, 5.0 * len(Ns)), squeeze=False)
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(ks)))
    summary = {}

    for r, N in enumerate(Ns):
        data, owner = [], []
        for k in ks:
            for L in by.get((k, N), []):
                t, y = prep(L)
                data.append((k, t, y))
                owner.append(k)
        if not data:
            continue

        # Pre-registered budget check: is L_inf stable when the fit range is HALVED? If the floor
        # moves, the run did not reach it and the fitted value is tracking the last data point
        # rather than an asymptote - so any cross-k comparison of floors at that N is meaningless.
        half = []
        for k in ks:
            for L in by.get((k, N), []):
                half.append((k, *prep(L, t_end=150000)))
        fl_half = fit_joint(half, "per_k")[2]
        half_by_k = {k: fl_half[[o for o in owner].index(k)] for k in ks if k in owner}

        fits = {mo: fit_joint(data, mo) for mo in ("one", "per_k", "per_seed")}
        rss = {mo: float(r @ r) for mo, (r, _, _) in fits.items()}
        npar = {mo: p for mo, (_, p, _) in fits.items()}
        m = len(fits["one"][0])
        aicc = lambda s, kk: m * np.log(s / m) + 2 * kk + 2 * kk * (kk + 1) / max(m - kk - 1, 1e-9)

        def ftest(small, big):
            """F-test of nested floor models; returns (F, p, df1, df2)."""
            d1, d2 = npar[big] - npar[small], m - npar[big]
            F = ((rss[small] - rss[big]) / d1) / (rss[big] / d2)
            return F, 1 - fdist.cdf(F, d1, d2), d1, d2

        fk, pk, d1k, d2k = ftest("one", "per_k")          # does the floor depend on k?
        fs, ps, d1s, d2s = ftest("per_k", "per_seed")     # is there seed scatter on top?

        fl_k = fits["per_k"][2]
        fl_s = fits["per_seed"][2]
        per_k = {k: fl_k[owner.index(k)] for k in ks if k in owner}
        seeds = {k: [fl_s[i] for i in range(len(data)) if owner[i] == k] for k in ks}
        print(f"\n=== N={N} ===")
        print("%3s %14s %14s %9s %24s" % ("k", "L_inf [300k]", "L_inf [150k]", "shift", "per-seed fits"))
        for k in ks:
            v = np.array(seeds[k])
            if v.size:
                sh = 100 * (per_k[k] - half_by_k[k]) / half_by_k[k]
                print("%3d %14.5f %14.5f %8.0f%% %24s"
                      % (k, per_k[k], half_by_k[k], sh, "  ".join(f"{x:.5f}" for x in v)))
        shifts = np.array([abs(100 * (per_k[k] - half_by_k[k]) / half_by_k[k]) for k in ks if k in per_k])
        ok = shifts.max() < 20
        print("  BUDGET CHECK: max |shift| when halving the fit range = %.0f%%  ->  %s"
              % (shifts.max(),
                 "300k sufficient, L_inf identified" if ok else
                 "300k NOT sufficient; L_inf is NOT identified at this N and the k-comparison below is void"))
        print("  does the floor depend on k?   F(%d,%d) = %.2f, p = %.2g   dAICc(one - per_k) = %+.1f"
              % (d1k, d2k, fk, pk, aicc(rss["one"], npar["one"]) - aicc(rss["per_k"], npar["per_k"])))
        print("  -> floors %s across k" % ("DIFFER" if pk < 0.05 else "are INDISTINGUISHABLE"))
        print("  seed scatter on top of k:     F(%d,%d) = %.2f, p = %.2g"
              % (d1s, d2s, fs, ps))
        summary[N] = (seeds, pk, dict(per_k), dict(zip(ks, shifts)))
        fl_sep = fl_s

        for i, (k, t, y) in enumerate(data):
            c = cols[ks.index(k)]
            lab = f"k={k}" if owner[:i].count(k) == 0 else None
            ax[r][0].plot(t, y, "-", color=c, lw=1.2, alpha=.85, label=lab)
            Li = fl_sep[i]
            ax[r][1].plot(t, np.clip(y - Li, 1e-8, None), "-", color=c, lw=1.2, alpha=.85, label=lab)
        for k in ks:
            v = np.array(seeds[k])
            if v.size:
                ax[r][2].plot([k] * v.size, v, "o", color=cols[ks.index(k)], ms=5, alpha=.55)
                ax[r][2].plot([k], [per_k[k]], "s", color=cols[ks.index(k)], ms=11,
                              label="per-$k$ fit" if k == ks[0] else None)
        one = fits["one"][2][0]
        ax[r][2].axhline(one, color="grey", ls="--", lw=1.4,
                         label=f"single shared floor {one:.5f}")

        ax[r][0].set(xscale="log", yscale="log", xlabel="iteration", ylabel="noise-free loss",
                     title=f"N={N}  (a) training dynamics, log-binned")
        ax[r][1].set(xscale="log", yscale="log", xlabel="iteration",
                     ylabel="$L(t)-L_\\infty$",
                     title=f"N={N}  (b) floor subtracted\nstraight = the fitted floor is right")
        verdict = ("floor IDENTIFIED" if shifts.max() < 20
                   else f"floor NOT identified (shifts {shifts.max():.0f}%)")
        ax[r][2].set(xlabel="task complexity $k$", ylabel="$L_\\infty$ per channel", xticks=ks,
                     title=f"N={N}  (c) floor vs $k$   shared-floor p={summary[N][1]:.2g}\n"
                           f"halved-range check: {verdict}")
        for a in ax[r]:
            a.legend(fontsize=8)
            a.grid(alpha=.3)

    fig.suptitle("Flip-flop: is the per-channel loss floor the same for every task complexity?\n"
                 "$L(t)=L_\\infty+A\\,e^{-(t/\\tau)^\\beta}$, log space, identical range per cell. "
                 "A p-value in (c) is only meaningful where the halved-range check passes.",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "flipflop_floor.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")

    # ---- summary figure: L_inf vs k, all three N on one axes ------------------------------------
    # Filled markers are floors that PASS the halved-range budget check (<20% shift) and can be
    # compared; open markers failed it and are plotted only so the failure is visible rather than
    # silently omitted. A line is drawn solid only through the identified points.
    fig2, a2 = plt.subplots(figsize=(8.2, 6))
    ncol = {500: "#1f77b4", 1000: "#d62728", 2000: "#2ca02c"}
    for N in Ns:
        if N not in summary:
            continue
        seeds, _, pk_floor, shift = summary[N]
        xs = [k for k in ks if k in pk_floor]
        ys = [pk_floor[k] for k in xs]
        good = [shift[k] < 20 for k in xs]
        a2.plot(xs, ys, "-", color=ncol.get(N, "grey"), lw=1.6, alpha=.45)
        for x, y, g in zip(xs, ys, good):
            a2.plot([x], [y], "o" if g else "o", color=ncol.get(N, "grey"), ms=11,
                    mfc=ncol.get(N, "grey") if g else "white", mew=2)
        for k in xs:
            v = np.array(seeds[k])
            a2.plot([k] * v.size, v, ".", color=ncol.get(N, "grey"), ms=6, alpha=.4)
        a2.plot([], [], "o-", color=ncol.get(N, "grey"), lw=1.6, ms=9, label=f"N={N}")
    a2.plot([], [], "o", color="grey", ms=10, label="filled = floor identified")
    a2.plot([], [], "o", color="grey", ms=10, mfc="white", mew=2,
            label="open = NOT identified (>20% shift)")
    a2.set(xlabel="task complexity $k$ (bits)", ylabel="$L_\\infty$ per channel", xticks=ks,
           yscale="log",
           title="Per-channel loss floor vs task complexity\n"
                 "small dots = individual seeds; only filled points are comparable")
    a2.legend(fontsize=9)
    a2.grid(alpha=.3, which="both")
    fig2.tight_layout()
    out2 = os.path.join(IMG_DIR, "flipflop_floor_vs_k.png")
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
