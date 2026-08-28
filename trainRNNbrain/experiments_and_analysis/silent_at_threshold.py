#!/usr/bin/env python3
"""
Silent-unit fraction at the moment each network STABLY reaches a common loss threshold.

This is the matched-performance comparison in its simplest form: every size is read at its own
iteration T_N, defined as the last time its smoothed loss is above L*, so all networks are equally
good at the task when M is counted. It is licensed by the measured result that the loss floor is
common to within ~1% across a 20-fold size range (and what difference exists runs the safe way -
larger networks sit slightly HIGHER, so they are not secretly under-trained at equal loss).

"STABLY" is doing real work. The per-iteration loss comes from a NOISY forward pass, and a single
favourable draw dips below any threshold long before the network is actually there - at L=0.025 the
raw trace first touches the line ~7x earlier than the smoothed one. T_N is therefore taken from a
centred running mean over VALID windows only; zero-padded edges otherwise manufacture a crossing at
iteration 1.

Both silent-unit criteria are reported because they disagree by an order of magnitude early in
training and converge only once a network is heavily silenced:
    hard        p_i < 1e-6            a unit that has genuinely stopped firing
    scale-free  p_i < 0.05*q95(p)     negligible relative to the network's own scale

Panels (a) and (b) give the silent FRACTION under each criterion; panel (c) gives the ABSOLUTE
active count M, which is the quantity the M* question is actually about. The two say different
things and both are needed: the fraction rising toward 100% is the reason silencing matters for
treating an RNN as a testbed, while M still growing is the reason there is no hard ceiling - it just
grows so slowly (M ~ N^0.28) that buying active units by enlarging the network is hopeless.

Output: img/internal_figures/silent_at_threshold.png

Usage:  python silent_at_threshold.py [L* ...]        (default: 0.023 0.025)
"""

import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, active_count, aicc, load_traces, stable_crossing

WINDOW = 2001
N_FIT_MIN = 500      # N=100 sits at 99.3% active, hard against the M <= N ceiling; including it
                     # would inflate the low-N slope and manufacture the curvature saturation predicts


def m_power(N, A, k):
    """Unbounded power law M = A N^k. Non-saturating: M grows without limit."""
    return A * np.power(N, k)


def m_hyper(N, Mstar, N0):
    """Hyperbolic saturation M = M* N/(N+N0); M -> M* as N grows."""
    return Mstar * N / (N + N0)


def m_expsat(N, Mstar, N0):
    """Exponential saturation M = M*(1 - exp(-N/N0)); M -> M* as N grows."""
    return Mstar * (1.0 - np.exp(-N / N0))


MODELS = [("power law", m_power, [10.0, 0.5], ([1e-6, -1], [1e6, 3])),
          ("saturating (hyperbolic)", m_hyper, None, ([1, 1], [1e7, 1e8])),
          ("saturating (exponential)", m_expsat, None, ([1, 1], [1e7, 1e8]))]


def fit_models(Ns, Ms):
    """Fit every candidate M(N) law to per-seed points and rank by AICc.

    All three have TWO parameters, so AICc differences are a like-for-like comparison of shape.

    Args:
        Ns, Ms: per-seed network size and active count (N >= N_FIT_MIN only).
    Returns:
        list of (name, params, aicc, callable), best first.
    """
    out = []
    for name, fn, p0, bounds in MODELS:
        guess = p0 if p0 is not None else [float(max(Ms)) * 1.5, 500.0]
        try:
            p, _ = curve_fit(fn, Ns, Ms, p0=guess, bounds=bounds, maxfev=200000)
            r = Ms - fn(Ns, *p)
            out.append((name, p, aicc(float(r @ r), len(Ns), 2), (lambda x, f=fn, q=p: f(x, *q))))
        except Exception:
            out.append((name, None, float("inf"), None))
    return sorted(out, key=lambda z: z[2])


def main():
    """Report and plot the silent fraction at each size's stable crossing of each threshold."""
    thrs = [float(a) for a in sys.argv[1:] if not a.startswith("--")] or [0.023, 0.025]
    by = load_traces("data/trained_RNNs/CDDM_std_g0_drift")
    Ns = sorted(by)
    out, per_seed = {}, {}

    for thr in thrs:
        print(f"\n=== L* = {thr:.4f} (stable crossing) ===")
        print("%7s %16s %20s %20s %16s"
              % ("N", "T_N", "silent, hard (%)", "silent, scalefree (%)", "ACTIVE (hard)"))
        for N in Ns:
            T, h, s = [], [], []
            for t in by[N]:
                x = stable_crossing(t["loss"], thr, window=WINDOW, base=1)
                P = np.array(t["participation"])
                I = np.array(t["participation_iters"])
                if x is None or x > I[-1]:
                    continue
                p = P[np.argmin(np.abs(I - x))]
                T.append(x)
                h.append(100 * (N - active_count(p, "hard")) / N)
                s.append(100 * (N - active_count(p, "scalefree")) / N)
            if not T:
                continue
            act = [N * (1 - x / 100) for x in h]
            out[(thr, N)] = (np.mean(T), np.mean(h), np.std(h), np.mean(s), np.std(s), len(T),
                             np.mean(act), np.std(act))
            per_seed[(thr, N)] = act
            print("%7d %8.0f +- %-5.0f %9.1f +- %-8.1f %9.1f +- %-8.1f %8.1f +- %-5.1f"
                  % (N, np.mean(T), np.std(T), np.mean(h), np.std(h), np.mean(s), np.std(s),
                     np.mean(act), np.std(act)))

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    cols = plt.cm.viridis(np.linspace(0.15, 0.75, len(thrs)))
    for col, (crit, idx, ttl) in enumerate([("hard", 1, r"$p_i<10^{-6}$"),
                                            ("scalefree", 3, r"$p_i<0.05\,q_{95}(p)$")]):
        for i, thr in enumerate(thrs):
            xs = [N for N in Ns if (thr, N) in out]
            mu = [out[(thr, N)][idx] for N in xs]
            sd = [out[(thr, N)][idx + 1] for N in xs]
            ax[col].errorbar(xs, mu, yerr=sd, fmt="o-", color=cols[i], ms=8, capsize=4, lw=2,
                             label=f"$L^*$={thr:.4f}  ($T_N$={out[(thr, xs[0])][0]:.0f}"
                                   f"–{out[(thr, xs[-1])][0]:.0f})")
        ax[col].set(xscale="log", xlabel="$N$", ylabel="silent units (% of N)", ylim=(0, 100),
                    title=f"({'ab'[col]}) {ttl}")
        ax[col].legend(fontsize=9)
        ax[col].grid(alpha=.3)
    # (c) absolute active count: the quantity M* is about. Log-log with the M=N diagonal, because a
    # power law M ~ N^k is a straight line there and the gap to the diagonal is the waste.
    grid = np.logspace(np.log10(80), np.log10(2.5e4), 100)
    for i, thr in enumerate(thrs):
        xs = [N for N in Ns if (thr, N) in out]
        mu = [out[(thr, N)][6] for N in xs]
        sd = [out[(thr, N)][7] for N in xs]
        ax[2].errorbar(xs, mu, yerr=sd, fmt="o", color=cols[i], ms=9, capsize=4, lw=2,
                       label=f"$L^*$={thr:.4f}")
        fN = np.array([N for N in xs for _ in per_seed[(thr, N)] if N >= N_FIT_MIN], float)
        fM = np.array([v for N in xs if N >= N_FIT_MIN for v in per_seed[(thr, N)]], float)
        if len(set(fN)) < 3:
            continue
        ranked = fit_models(fN, fM)
        best = ranked[0][2]
        print(f"\n  M(N) fits at L*={thr:.4f}  (N >= {N_FIT_MIN}, n={len(fN)} seed points)")
        for name, p, a, pred in ranked:
            if pred is None:
                print(f"    {name:26s} FAILED")
                continue
            ps = ", ".join(f"{v:.4g}" for v in p)
            print(f"    {name:26s} dAICc {a-best:+7.1f}   params [{ps}]")
        pw = [r for r in ranked if r[0] == "power law"][0]
        sat = [r for r in ranked if r[0].startswith("saturating")][0]
        if pw[3] is not None:
            ax[2].plot(grid, pw[3](grid), "-", color=cols[i], lw=1.8,
                       label=f"   power law $k$={pw[1][1]:.2f} ($\\Delta$AICc {pw[2]-best:+.1f})")
        if sat[3] is not None:
            ax[2].plot(grid, sat[3](grid), ":", color=cols[i], lw=2.2,
                       label=f"   saturating $M^*$={sat[1][0]:.0f} ($\\Delta$AICc {sat[2]-best:+.1f})")
    if Ns:
        ax[2].plot(Ns, Ns, "k--", lw=1.2, alpha=.6, label="$M=N$ (nothing silent)")
    ax[2].set(xscale="log", yscale="log", xlabel="$N$", ylabel="ACTIVE units $M$",
              title=f"(c) absolute active count, with fits\n"
                    f"(fitted on $N\\geq${N_FIT_MIN}; both models have 2 parameters)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3, which="both")

    fig.suptitle("Silent fraction and absolute active count when each size STABLY reaches a common "
                 "loss threshold\n(matched performance; $T_N$ from a centred 2001-iteration mean, "
                 "not the raw trace)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    p = os.path.join(IMG_DIR, "silent_at_threshold.png")
    fig.savefig(p, dpi=150)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
