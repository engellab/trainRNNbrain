#!/usr/bin/env python3
"""
How does the ACTIVE UNIT COUNT scale with network size and task complexity?

The companion test (flipflop_collapse.py) asks whether the silent FRACTION collapses onto a single
variable. This asks the same question of the absolute count M, which is a different target and cannot be inferred
from the fraction result:

    M = N (1 - f),  so if f = F(x) with x = N/k^a, then M = N G(x) - still carrying an explicit N.

Three ansatze, in increasing generality:

  (A) plain power law     M = A N^b k^c
      Directly interpretable: b is the return on network size, c the return on task complexity. For
      a pure power law M IS a function of the single variable u = N k^(c/b), so this also yields the
      collapse exponent alpha = -c/b as a by-product rather than a search.

  (B) literal collapse    M = G(N/k^alpha)     for an arbitrary smooth G
      What was asked for. Note it forces G to absorb the N prefactor, so it is a genuine constraint,
      not a reparametrisation of (A).

  (C) scaling form        M/k^beta = G(N/k^alpha)
      The standard finite-size-scaling shape, which (B) is the beta = 0 special case of.

Each is compared against the SATURATED model - a free mean per (k, N) cell - because the question is
whether a low-dimensional law captures what 15 free numbers capture. Fits are in log M, so residuals
are relative and a cell near M = 2000 does not dominate one near M = 300.

Reported at several matched performance levels and at the endpoint, both silence criteria, because a
law that holds at one reading is a coincidence.

ALWAYS REPORT b WITH THE READING DEPTH IT WAS MEASURED AT, expressed as excess over that task's own
floor. b is not a property of the task alone: it varies with how close to the floor the reading is
taken, so a bare exponent is not comparable across tasks or across levels.

Output: img/internal_figures/flipflop_M_scaling.png

Usage:  python flipflop_M_scaling.py
NOTE ON PROVENANCE. Every number that this file previously quoted from the flip-flop came from the
first sweep, which ran `same_batch=True` and therefore trained on 256 frozen trials - memorisation,
not the task. Those numbers are RETRACTED and have been stripped rather than updated; the data is
quarantined in `data/trained_RNNs/RETRACTED_samebatch_NBitFlipFlop_ksweep/`. Nothing here has yet
been run against the corrected fresh-batch sweep, so this file currently states METHOD only, with no
results. Do not reintroduce a remembered figure into these docstrings.

"""

import os
import sys
import numpy as np
from scipy.stats import f as fdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, stable_crossing
from flipflop_ksweep import load, silence_at

N_BOOT = 300
# Bootstrap RNG. Seeded, because an unseeded global np.random makes the reported CIs
# irreproducible: two runs of identical code returned b = [0.353, 0.430] and [0.349, 0.431].
BOOT_RNG = np.random.default_rng(0)
GRID_A = np.linspace(-6.0, 6.0, 481)     # alpha, signed: M may rise OR fall with k
GRID_B = np.linspace(-3.0, 3.0, 241)     # beta


def points(by, ks, Ns, thr, crit):
    """Seed-level (k, N, M) at one performance level.

    Args:
        by: loaded traces; ks, Ns: axes; thr: loss level or None for the endpoint;
        crit: 2 for the hard active count, 3 for scale-free.
    Returns:
        (k, N, M) float arrays, one entry per seed reaching the level.
    """
    K, NN, M = [], [], []
    for k in ks:
        for N in Ns:
            for t in by.get((k, N), []):
                x = None if thr is None else stable_crossing(t["clean"], thr)
                if thr is not None and x is None:
                    continue
                h, s, a = silence_at(t, N, x)
                K.append(k)
                NN.append(N)
                M.append(a if crit == 2 else N * (1 - s / 100))
    return np.array(K, float), np.array(NN, float), np.maximum(np.array(M, float), 1.0)


def rss_poly(x, y, deg=2):
    """RSS of a degree-`deg` polynomial fit of y on x; falls back to the mean if x is degenerate."""
    if len(np.unique(x)) <= deg:
        return float(np.sum((y - y.mean()) ** 2))
    r = y - np.polyval(np.polyfit(x, y, deg), x)
    return float(r @ r)


def fit_powerlaw(K, NN, M):
    """Least squares for log M = log A + b log N + c log k.

    Returns:
        (b, c, RSS, R^2).
    """
    y = np.log(M)
    X = np.column_stack([np.ones_like(y), np.log(NN), np.log(K)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    rss = float(r @ r)
    return coef[1], coef[2], rss, 1 - rss / float(((y - y.mean()) ** 2).sum())


def fit_collapse(K, NN, M, betas=(0.0,)):
    """Best (alpha, beta) for M/k^beta = G(N/k^alpha), G a quadratic in log.

    Args:
        K, NN, M: seed-level arrays; betas: beta values to search (just 0 gives ansatz B).
    Returns:
        (alpha, beta, RSS).
    """
    y0 = np.log(M)
    best = (np.nan, np.nan, np.inf)
    for beta in betas:
        y = y0 - beta * np.log(K)
        for a in GRID_A:
            rss = rss_poly(np.log(NN) - a * np.log(K), y)
            if rss < best[2]:
                best = (a, beta, rss)
    return best


def main():
    """Fit and test three scaling laws for the active-unit count."""
    by, _ = load()
    ks = sorted({k for k, _ in by})
    Ns = sorted({N for _, N in by})
    readings = [(None, "endpoint (300k)"), (0.0375, "R^2 = 0.949"),
                (0.0220, "R^2 = 0.970"), (0.0150, "R^2 = 0.980")]
    store = {}

    for crit, cname in ((3, "scale-free"), (2, "hard  p<1e-6")):
        print(f"\n{'=' * 84}\nACTIVE UNIT COUNT M — criterion: {cname}\n{'=' * 84}")
        for thr, lab in readings:
            K, NN, M = points(by, ks, Ns, thr, crit)
            ncell = len({(a, b) for a, b in zip(K, NN)})
            if len(M) < 12 or ncell < 8:
                print(f"\n{lab}: only {ncell} cells reachable — skipped")
                continue
            y = np.log(M)
            tot = float(((y - y.mean()) ** 2).sum())
            rss_sat = float(sum(((y[m] - y[m].mean()) ** 2).sum()
                                for m in [(K == a) & (NN == b) for a in ks for b in Ns]
                                if m.sum()))
            n = len(M)

            b, c, rss_pl, r2_pl = fit_powerlaw(K, NN, M)
            a_B, _, rss_B = fit_collapse(K, NN, M)
            a_C, beta_C, rss_C = fit_collapse(K, NN, M, betas=GRID_B)

            bb, bc = [], []
            for _ in range(N_BOOT):
                i = BOOT_RNG.integers(0, n, n)
                if len(np.unique(K[i])) < 3 or len(np.unique(NN[i])) < 2:
                    continue
                r = fit_powerlaw(K[i], NN[i], M[i])
                bb.append(r[0])
                bc.append(r[1])
            blo, bhi = np.percentile(bb, [2.5, 97.5])
            clo, chi = np.percentile(bc, [2.5, 97.5])

            def vs_sat(rss, npar):
                """F-test of a model against the saturated per-cell model."""
                d1, d2 = ncell - npar, n - ncell
                if d1 <= 0 or d2 <= 0:
                    return np.nan, np.nan
                F = ((rss - rss_sat) / d1) / (rss_sat / d2)
                return F, 1 - fdist.cdf(F, d1, d2)

            F_pl, p_pl = vs_sat(rss_pl, 3)
            F_B, p_B = vs_sat(rss_B, 4)
            F_C, p_C = vs_sat(rss_C, 5)

            print(f"\n{lab}   ({ncell} cells, {n} seed points)")
            print(f"  (A) power law   M = A N^b k^c :  b = {b:.3f} [{blo:.3f}, {bhi:.3f}]   "
                  f"c = {c:.3f} [{clo:.3f}, {chi:.3f}]")
            print(f"      variance explained {r2_pl:.3f}   saturated {1 - rss_sat / tot:.3f}   "
                  f"vs saturated: F = {F_pl:.2f}, p = {p_pl:.3g}")
            print(f"      implied collapse variable  u = N * k^{c / b:.2f}   "
                  f"(i.e. alpha = {-c / b:.2f} in N/k^alpha)")
            print(f"  (B) M = G(N/k^a)          :  a = {a_B:+.2f}   "
                  f"variance explained {1 - rss_B / tot:.3f}   F = {F_B:.2f}, p = {p_B:.3g}")
            print(f"  (C) M/k^B = G(N/k^a)      :  a = {a_C:+.2f}, B = {beta_C:+.2f}   "
                  f"variance explained {1 - rss_C / tot:.3f}   F = {F_C:.2f}, p = {p_C:.3g}")
            verdict = [nm for nm, pp in (("A", p_pl), ("B", p_B), ("C", p_C)) if pp > 0.05]
            print(f"  -> matches the saturated model: {', '.join(verdict) if verdict else 'NONE'}")
            store[(crit, thr)] = (K, NN, M, b, c, lab, r2_pl, p_pl)

    keys = [q for q in store if q[0] == 3]
    if not keys:
        return
    fig, ax = plt.subplots(2, len(keys), figsize=(5.6 * len(keys), 9.2), squeeze=False)
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(ks)))
    for j, key in enumerate(keys):
        K, NN, M, b, c, lab, r2, p = store[key]
        for k in ks:
            m = K == k
            if m.sum():
                ax[0][j].plot(NN[m], M[m], "o", color=cols[ks.index(k)], ms=7, label=f"k={k}")
        # ⚠️ BEFORE/AFTER refer to the CHANGE OF X-VARIABLE, not to two datasets or two times.
        # The reading depth (`lab`) identifies the COLUMN and must appear in both titles, or the pair
        # reads as "BEFORE = endpoint, AFTER = a power law", which is meaningless.
        ax[0][j].set(xscale="log", yscale="log", xlabel="$N$", ylabel="active units $M$",
                     title=f"[{lab}]  BEFORE collapse\n"
                           f"raw $M$ vs $N$ — one curve per $k$")
        u = NN * K ** (c / b)
        for k in ks:
            m = K == k
            if m.sum():
                ax[1][j].plot(u[m], M[m], "o", color=cols[ks.index(k)], ms=7, label=f"k={k}")
        uu = np.logspace(np.log10(u.min()), np.log10(u.max()), 100)
        A = np.exp(np.mean(np.log(M) - b * np.log(NN) - c * np.log(K)))
        ax[1][j].plot(uu, A * uu ** b, "k-", lw=1.6, alpha=.75,
                      label=f"$M \\propto u^{{{b:.2f}}}$")
        ax[1][j].set(xscale="log", yscale="log",
                     xlabel=f"$u = N\\,k^{{{c / b:.2f}}}$", ylabel="active units $M$",
                     title=f"[{lab}]  AFTER collapse onto $u=N k^{{{c / b:.2f}}}$\n"
                           f"$M = A\\,N^{{{b:.2f}}}k^{{{c:.2f}}}$;  lack-of-fit p = {p:.3g} "
                           f"({'law OK' if p > 0.05 else 'LAW REJECTED'})")
        for a_ in (ax[0][j], ax[1][j]):
            a_.legend(fontsize=8)
            a_.grid(alpha=.3, which="both")
    fig.suptitle("Active-unit count as a joint power law in size and task complexity\n"
                 "scale-free criterion. Rows: same points BEFORE vs AFTER rescaling the x-axis — a "
                 "real law fuses the per-$k$ curves into one.\n"
                 "Lack-of-fit p tests the law against a FREE MEAN PER CELL; LARGE p = the law is as "
                 "good as 20 free numbers, small p = it is not.", fontsize=11)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "flipflop_M_scaling.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
