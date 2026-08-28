#!/usr/bin/env python3
"""
Is the achievable loss floor L_inf independent of network size, or does it fall with N?

WHY THIS IS THE LOAD-BEARING FACT. The whole cross-size comparison matches networks at equal
training loss. That is only a valid notion of "equally trained" if every size is heading for the SAME
floor: if a bigger network could reach a lower loss, then two networks at equal loss are NOT equally
far from their own optima, and every M(N) number inherits the confound.

There is also a concrete mechanism that could make the floor size-dependent, so this is not a
formality. Learning rate is scaled as lr = 1e-3 (100/N)^(1/3), and a noisy optimiser equilibrates in
a noise ball whose width grows with lr. Smaller lr (larger N) should therefore give a LOWER floor,
all else equal. If no size dependence is found, that mechanism is either absent or cancelled.

HYPOTHESES:
  M0  constant     L_inf = c                       (1 param)  -- the floor is a property of the TASK
  M1  linear       L_inf = c + b N                 (2 params)
  M2  log-linear   L_inf = c + b log10(N)          (2 params)
  M3  power law    L_inf = a N^(-b)                (2 params) -- unbounded improvement with size

THE ESTIMATION TRAP THIS CONTROLS FOR. L_inf is obtained by extrapolating L(t) = L_inf + A t^(-gamma),
and that extrapolation is biased by how much data it sees: a curve that has not yet flattened makes
the asymptote look too low. Runs of different length are therefore NOT comparable. Every fit here
uses the same first t_max iterations for every size, so the bias is common to all of them, and the
whole comparison is repeated at several t_max to confirm the verdict does not depend on it.

A model-free cross-check is also reported: the raw measured loss at matched iteration count, which
involves no extrapolation at all. It answers a slightly different question (it confounds the floor
with how far along each size is) but it cannot be wrong about the data.

CRITERION FIXED BEFORE RUNNING: a size dependence is accepted only if (a) the slope differs from zero
at p < 0.05, (b) the best model beats M0 by dAICc > 4, and (c) both hold at every t_max tested.

Output: img/internal_figures/floor_vs_N.png

Usage:  python test_floor_vs_N.py [SWEEP_FOLDER]
"""

import os
import re
import sys
import glob
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, aicc, load_losses
from plot_loss_fit import fit_loss

TMAXES = [25000, 50000, 100000, 150000, 200000]
LOSS_RX = re.compile(r"iteration (\d+)/(\d+), train: ([0-9.eE+-]+)")
HEAD_RX = re.compile(r"N=(\d+) iters=(\d+)")


def losses_from_logs(logdir):
    """Recover loss traces for runs that have not finished yet, from their SLURM stdout.

    The saved TrainLosses.json is only written when a run completes, but the per-iteration loss is
    printed to stdout as it goes, so an in-progress run's curve is fully recoverable. Verified
    against a completed run: the parsed values match the saved JSON to 5e-7, i.e. to the 6 decimals
    the log prints. Runs that reached their declared max_iter are SKIPPED, because those already have
    a JSON and would otherwise be counted twice.

    Args:
        logdir: directory of StdDrift.*.out files.
    Returns:
        dict {N: [(tag, loss array)]} for in-progress runs only.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(logdir, "*.out"))):
        head = HEAD_RX.search(open(f, errors="ignore").read(4000))
        if not head:
            continue
        N, target = int(head.group(1)), int(head.group(2))
        tr, last = [], 0
        for line in open(f, errors="ignore"):
            m = LOSS_RX.search(line)
            if m:
                last = int(m.group(1))
                tr.append(float(m.group(3)))
        if not tr or last >= target:          # finished -> the JSON copy is authoritative
            continue
        out.setdefault(N, []).append((os.path.basename(f)[9:19] + "*", np.array(tr)))
    return out


def floors(by, tmax):
    """Per-seed L_inf estimates, all fitted on the same first `tmax` iterations.

    Args:
        by: {N: [(tag, loss array)]}; tmax: iterations used for every fit.
    Returns:
        (Ns, Ls) arrays with one entry per seed, restricted to runs at least tmax long.
    """
    Ns, Ls = [], []
    for N, entries in sorted(by.items()):
        for _, L in entries:
            if len(L) >= tmax:
                Li, _, _ = fit_loss(L[:tmax], 2000)
                if np.isfinite(Li):
                    Ns.append(float(N))
                    Ls.append(Li)
    return np.array(Ns), np.array(Ls)


def fit_models(Ns, Ls):
    """Fit the four candidate shapes of L_inf(N) and return their statistics.

    Args:
        Ns, Ls: per-seed size and floor estimates.
    Returns:
        dict {name: {"rss", "k", "aicc", "p", "pred"}}, p = two-sided p-value on the size term
        (None for the constant model), pred = callable for plotting.
    """
    n = len(Ns)
    out = {}
    rss0 = float(np.sum((Ls - Ls.mean()) ** 2))
    out["constant"] = {"rss": rss0, "k": 1, "aicc": aicc(rss0, n, 1), "p": None,
                       "pred": (lambda x, m=Ls.mean(): np.full_like(np.asarray(x, float), m))}

    for name, xf in (("linear", lambda z: z), ("log-linear", lambda z: np.log10(z))):
        x = xf(Ns)
        X = np.column_stack([np.ones(n), x])
        beta, *_ = np.linalg.lstsq(X, Ls, rcond=None)
        r = Ls - X @ beta
        rss = float(r @ r)
        dof = n - 2
        se = np.sqrt(rss / dof * np.linalg.pinv(X.T @ X)[1, 1])
        t = beta[1] / se if se > 0 else 0.0
        out[name] = {"rss": rss, "k": 2, "aicc": aicc(rss, n, 2),
                     "p": float(2 * (1 - stats.t.cdf(abs(t), dof))), "slope": beta[1], "se": se,
                     "pred": (lambda z, b=beta, f=xf: b[0] + b[1] * f(np.asarray(z, float)))}

    try:
        p, cov = curve_fit(lambda z, a, b: a * z ** (-b), Ns, Ls, p0=[0.03, 0.01],
                           bounds=([1e-6, -1], [1e3, 3]), maxfev=60000)
        r = Ls - p[0] * Ns ** (-p[1])
        rss = float(r @ r)
        se = float(np.sqrt(np.diag(cov))[1])
        t = p[1] / se if se > 0 else 0.0
        out["power law"] = {"rss": rss, "k": 2, "aicc": aicc(rss, n, 2),
                            "p": float(2 * (1 - stats.t.cdf(abs(t), n - 2))),
                            "slope": p[1], "se": se,
                            "pred": (lambda z, q=p: q[0] * np.asarray(z, float) ** (-q[1]))}
    except Exception:
        pass
    return out


def main():
    """Run the floor-vs-size test at several matched fit lengths and draw the figure."""
    sweep = ([a for a in sys.argv[1:] if not a.startswith("--")] or
             ["data/trained_RNNs/CDDM_std_g0_drift"])[0]
    by = load_losses(sweep)
    logdir = ([a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--logs=")] or [None])[0]
    if logdir:
        extra = losses_from_logs(logdir)
        for N, v in extra.items():
            by.setdefault(N, []).extend(v)
        print("added in-progress runs from logs: " +
              ", ".join(f"N={N}: {len(v)} seed(s), up to {max(len(L) for _, L in v)} iters"
                        for N, v in sorted(extra.items())))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.3))
    cols = plt.cm.plasma(np.linspace(0.1, 0.72, len(TMAXES)))
    verdicts = []

    for i, tmax in enumerate(TMAXES):
        Ns, Ls = floors(by, tmax)
        if len(np.unique(Ns)) < 3:
            continue
        res = fit_models(Ns, Ls)
        best = min(res, key=lambda z: res[z]["aicc"])
        print(f"\n=== fits on the first {tmax} iterations   "
              f"(n={len(Ns)} seeds, sizes {sorted(set(Ns.astype(int)))}) ===")
        print(f"  {'model':<12} {'AICc':>9} {'dAICc vs const':>15} {'size-term p':>13}")
        for nm in ("constant", "linear", "log-linear", "power law"):
            if nm not in res:
                continue
            d = res[nm]["aicc"] - res["constant"]["aicc"]
            pv = "-" if res[nm]["p"] is None else f"{res[nm]['p']:.3f}"
            print(f"  {nm:<12} {res[nm]['aicc']:>9.1f} {d:>15.1f} {pv:>13}")
        sig = (res["log-linear"]["p"] < 0.05 and
               res["log-linear"]["aicc"] - res["constant"]["aicc"] < -4)
        verdicts.append(sig)
        print(f"  best by AICc: {best};  size dependence accepted at this t_max: {sig}")
        # spread relative to seed noise
        gm = [Ls[Ns == u].mean() for u in np.unique(Ns)]
        within = np.mean([Ls[Ns == u].std() for u in np.unique(Ns) if (Ns == u).sum() > 1])
        print(f"  across-size spread = {max(gm)-min(gm):.5f}   "
              f"typical within-size seed sd = {within:.5f}   ratio = {(max(gm)-min(gm))/within:.1f}")

        for u in np.unique(Ns):
            v = Ls[Ns == u]
            ax[0].errorbar([u], [v.mean()], yerr=[v.std()], fmt="o", color=cols[i], ms=6, capsize=3,
                           label=f"$t_{{max}}$={tmax//1000}k" if u == Ns.min() else None)
        xs = np.logspace(np.log10(90), np.log10(2500), 50)
        ax[0].plot(xs, res["constant"]["pred"](xs), "-", color=cols[i], lw=1.2, alpha=.8)
        ax[0].plot(xs, res["log-linear"]["pred"](xs), "--", color=cols[i], lw=1.2, alpha=.8)
        ax[1].errorbar([tmax], [res["log-linear"]["slope"]], yerr=[1.96 * res["log-linear"]["se"]],
                       fmt="o", color=cols[i], ms=7, capsize=4)

    ax[0].set(xscale="log", xlabel="$N$", ylabel=r"estimated $L_\infty$",
              title="(a) floor vs size at matched fit length\nsolid = constant fit, dashed = log-linear")
    ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="k", lw=1.2)
    ax[1].set(xscale="log", xlabel="$t_{max}$ used for the fit",
              ylabel=r"slope of $L_\infty$ vs $\log_{10}N$",
              title="(b) is the size term nonzero?\n95% CI; zero = no size dependence")

    # model-free cross-check: raw measured loss at matched iteration count
    for i, tmax in enumerate(TMAXES):
        Ns, Ms = [], []
        for N, entries in sorted(by.items()):
            v = [L[max(tmax - 4000, 0):tmax].mean() for _, L in entries if len(L) >= tmax]
            if v:
                Ns.append(N)
                Ms.append(np.mean(v))
        ax[2].plot(Ns, Ms, "-o", color=cols[i], ms=6, label=f"at {tmax//1000}k iters")
    ax[2].set(xscale="log", xlabel="$N$", ylabel="measured training loss",
              title="(c) model-free check: raw loss\nat matched iteration count")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.25)
    fig.suptitle(r"Is the loss floor $L_\infty$ independent of network size?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(IMG_DIR, "floor_vs_N.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")
    print(f"\nOVERALL: size dependence accepted at {sum(verdicts)}/{len(verdicts)} fit lengths "
          f"-> {'SIZE-DEPENDENT' if all(verdicts) and verdicts else 'NOT established; floor treated as CONSTANT'}")


if __name__ == "__main__":
    main()
