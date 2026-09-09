#!/usr/bin/env python3
"""
Effective dimensionality of the population activity at the END of training, over the (N, k) grid.

DIMENSIONALITY, NOT PARTICIPATION. `pr_matrix.py` reports the participation ratio of the per-unit
participation vector: how EVENLY the units are engaged. This reports the participation ratio of the
activity COVARIANCE EIGENSPECTRUM: how many directions the population trajectory actually occupies.
A network can have every unit active (M/N = 1) while its activity lives in six dimensions; the two
numbers answer different questions and are not interchangeable.

    D_PR = (sum lambda)^2 / sum lambda^2        lambda = eigenvalues of cov(rates over samples)
    D_95 = smallest number of PCs carrying 95% of the variance
    VE_n = fraction of total variance carried by the leading n PCs (n = 1, 5, 10)

Rates are the noise-free forward pass on a fresh batch, relu applied (equation_type "h" stores
pre-activations), pooled over time and trials into (samples, neurons) and centred per neuron. Silent
units contribute zero variance and so cannot inflate either measure.

⚠️ SAMPLING WAS CHECKED, NOT ASSUMED. D_PR is a ratio of spectrum moments and is biased when the
sample count approaches the true dimensionality. Measured on N=2000 runs, D_PR moves 6.62 -> 6.71
(+1.4%) and 5.64 -> 5.75 (+2%) going from 960 to 9600 samples, so the 4800 used here is far into the
converged regime for the D ~ 5-15 seen on this task.

Output: img/internal_figures/dimensionality_matrix.png
        data/dimensionality_cache.npz  (delete to recompute)

Usage:  python flipflop_dimensionality.py [--recompute]
"""

import os
import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR
import plotstyle as ps
from pr_matrix import ROOTS, SKIP, PENS, R2_MIN, MIN_ITERS, fit_power_law
from flipflop_fixedpoints import load_net
from flipflop_bouquet import run_trials

N_TRIALS = 32          # trials simulated per net
T_STEP = 2             # keep every T_STEP-th timepoint -> 150 * 32 = 4800 samples
VAR_FRAC = 0.95        # D_95 threshold
CACHE = "data/dimensionality_cache.npz"


def run_folders():
    """Every usable trained-net folder in the grid, with its (penalty, k, N).

    Applies the same two gates as pr_matrix.load(): r2 >= R2_MIN (runs that never solved the task
    have meaningless activity spectra) and a budget of at least MIN_ITERS.

    ⚠️ THE BUDGET GATE IS NOT OPTIONAL. std_bigN holds calibration folders trained for 400-600
    iterations (`_iters=600` in the name); pr_matrix drops them by trace length, but globbing
    LastParams instead picks them up, and an untrained net's spectrum is not a result. Without this
    they added a spurious N=3000 row and a 25th none/N=4000 run.

    Returns:
        list of (folder, pen, k, N) tuples.
    """
    out = []
    for tag, root in ROOTS.items():
        for f in sorted(glob.glob(os.path.join(root, "*", "*", "*LastParams*.npz"))):
            folder = os.path.dirname(f)
            m = re.search(r"_k=(\d+)_N=(\d+)(?:_pen=([a-z]+))?", folder)
            if not m:
                continue
            pen = m.group(3) or "none"
            if (tag, pen) in SKIP:
                continue
            try:
                r2 = float(os.path.basename(f).split("_")[0])
            except ValueError:
                continue
            if not (r2 >= R2_MIN):
                continue
            cfgs = glob.glob(os.path.join(folder, "*_config.yaml"))
            if not cfgs or int(OmegaConf.load(cfgs[0]).trainer.max_iter) < MIN_ITERS:
                continue
            out.append((folder, pen, int(m.group(1)), int(m.group(2))))
    return out


def spectrum(rates):
    """Centred activity covariance eigenvalues, largest first.

    Args:
        rates: (N, T, B) noise-free firing rates.
    Returns:
        (min(N, samples),) non-negative eigenvalues in descending order.
    """
    N = rates.shape[0]
    X = rates[:, ::T_STEP, :].reshape(N, -1).T.astype(np.float64)     # (samples, neurons)
    X -= X.mean(0, keepdims=True)
    # Gram over whichever axis is smaller; its non-zero eigenvalues match the covariance's.
    G = X @ X.T if X.shape[0] < X.shape[1] else X.T @ X
    lam = np.clip(np.linalg.eigvalsh(G), 0.0, None)[::-1]
    return lam / max(X.shape[0] - 1, 1)


def measures(lam):
    """Dimensionality summaries of one activity covariance spectrum.

    Args:
        lam: (n,) non-negative eigenvalues, descending.
    Returns:
        dict with d_pr (participation ratio), d_95 (PCs to reach VAR_FRAC) and ve1/ve5/ve10, the
        cumulative variance fraction carried by the leading 1, 5 and 10 PCs. The VE numbers are the
        direct, assumption-free reading of the spectrum: D_PR and D_95 are both single summaries of
        a shape that VE reports at a fixed cut.
    """
    tot = lam.sum()
    if tot <= 0:
        return dict(d_pr=np.nan, d_95=np.nan, ve1=np.nan, ve5=np.nan, ve10=np.nan)
    cum = np.cumsum(lam) / tot
    at = lambda n: float(cum[min(n, cum.size) - 1])
    return dict(d_pr=float(tot ** 2 / (lam ** 2).sum()),
                d_95=float(np.searchsorted(cum, VAR_FRAC) + 1),
                ve1=at(1), ve5=at(5), ve10=at(10))


def compute():
    """Simulate every run and measure its dimensionality, caching the result.

    Returns:
        dict of arrays with keys pen, k, N, d_pr, d_95 (one entry per run).
    """
    FIELDS = ("pen", "k", "N", "d_pr", "d_95", "ve1", "ve5", "ve10")
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        # A cache written before a field was added would silently make that field unavailable, so
        # recompute rather than half-answer.
        if all(f in z.files for f in FIELDS):
            return {kk: z[kk] for kk in z.files}
        print("cache predates the variance-explained fields; recomputing")
    fold = run_folders()
    print(f"simulating {len(fold)} nets ({N_TRIALS} trials each, every {T_STEP}nd timepoint)")
    rec = {kk: [] for kk in FIELDS}
    for i, (folder, pen, k, N) in enumerate(fold, 1):
        net, _ = load_net(folder)
        m = measures(spectrum(run_trials(net, folder, N_TRIALS)))
        for kk, v in zip(("pen", "k", "N"), (pen, k, N)):
            rec[kk].append(v)
        for kk in ("d_pr", "d_95", "ve1", "ve5", "ve10"):
            rec[kk].append(m[kk])
        if i % 25 == 0 or i == len(fold):
            print(f"  {i}/{len(fold)}")
    rec = {kk: np.array(v) for kk, v in rec.items()}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **rec)
    return rec


def grid(rec, pen, key, ks, Ns):
    """(mean, sd, n) of a measure over the (N, k) grid for one penalty.

    Args:
        rec: dict from compute(); pen: penalty name; key: "d_pr" or "d_95";
        ks, Ns: sorted axis values.
    Returns:
        (Z, S, C) arrays shaped (len(Ns), len(ks)).
    """
    Z = np.full((len(Ns), len(ks)), np.nan)
    S = np.full((len(Ns), len(ks)), np.nan)
    C = np.zeros((len(Ns), len(ks)), dtype=int)
    for i, N in enumerate(Ns):
        for j, k in enumerate(ks):
            v = rec[key][(rec["pen"] == pen) & (rec["k"] == k) & (rec["N"] == N)]
            v = v[np.isfinite(v)]
            if v.size:
                Z[i, j], S[i, j], C[i, j] = v.mean(), v.std(), v.size
    return Z, S, C


def main():
    """Plot D_PR and D_95 over the (N, k) grid, one column per penalty."""
    if "--recompute" in sys.argv and os.path.exists(CACHE):
        os.remove(CACHE)
    ps.setup()
    rec = compute()
    ks = sorted(set(rec["k"].tolist()))
    Ns = sorted(set(rec["N"].tolist()))

    print(f"\n{'='*74}\nD_PR:  fitted  D = A N^b k^c\n{'='*74}")
    print("%-6s %5s %24s %26s" % ("pen", "n", "b (size)", "c (complexity)"))
    laws = {}
    for p in PENS:
        m = (rec["pen"] == p) & np.isfinite(rec["d_pr"]) & (rec["d_pr"] > 0)
        f = fit_power_law(rec["k"][m], rec["N"][m], rec["d_pr"][m])
        laws[p] = f
        if not f:
            print("%-6s   not fittable" % p); continue
        star = "" if f["c_ci"][0] <= 0 <= f["c_ci"][1] else "   <- c != 0"
        print("%-6s %5d   %+.3f [%+.3f, %+.3f]      %+.3f [%+.3f, %+.3f]%s"
              % (p, f["n"], f["b"], f["b_ci"][0], f["b_ci"][1],
                 f["c"], f["c_ci"][0], f["c_ci"][1], star))

    ROWS = [("d_pr", "$D_{PR}$", "participation ratio of the activity spectrum"),
            ("d_95", "$D_{95}$", f"PCs carrying {VAR_FRAC:.0%} of the variance")]
    vmax = {kk: np.nanpercentile(rec[kk], 99) for kk, _, _ in ROWS}
    fig, ax = plt.subplots(3, len(PENS), figsize=(4.3 * len(PENS), 11.2), squeeze=False)
    for c_i, pen in enumerate(PENS):
        for r_i, (key, lab, note) in enumerate(ROWS):
            a = ax[r_i][c_i]
            Z, S, C = grid(rec, pen, key, ks, Ns)
            if not np.isfinite(Z).any():
                a.text(.5, .5, f"no {pen} data", ha="center", va="center", transform=a.transAxes,
                       color="0.5"); a.set_xticks([]); a.set_yticks([]); continue
            im = a.imshow(Z, cmap="viridis", vmin=0, vmax=vmax[key], aspect="auto")
            for i in range(len(Ns)):
                for j in range(len(ks)):
                    if np.isfinite(Z[i, j]):
                        col = "white" if Z[i, j] < 0.6 * vmax[key] else "black"
                        a.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="bottom", fontsize=7.5,
                               color=col)
                        a.text(j, i, f"±{S[i, j]:.1f}", ha="center", va="top", fontsize=5.6,
                               color=col, alpha=.85)
                    else:
                        a.text(j, i, "·", ha="center", va="center", color="0.6", fontsize=9)
            a.set(xticks=range(len(ks)), xticklabels=ks, yticks=range(len(Ns)),
                  yticklabels=[str(n) for n in Ns], xlabel="k (bits)")
            if c_i == 0:
                a.set_ylabel("N (units)")
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
            a.set_title((f"{pen}\n" if r_i == 0 else "") + f"{lab} — {note}",
                        fontsize=10.5, fontweight="bold" if r_i == 0 else "normal")

        b = ax[2][c_i]
        Z, S, _ = grid(rec, pen, "d_pr", ks, Ns)
        if np.isfinite(Z).any():
            for i, N in enumerate(Ns):
                if np.isfinite(Z[i]).any():
                    ps.band(b, ks, Z[i], S[i], ps.col_n(N), label=f"N={N}")
            f = laws.get(pen)
            if f:
                kk = np.linspace(min(ks), max(ks), 100)
                for i, N in enumerate(Ns):
                    if np.isfinite(Z[i]).any():
                        b.plot(kk, f["A"] * N ** f["b"] * kk ** f["c"], "--", color=ps.col_n(N),
                               lw=1.1, alpha=.75)
                b.text(.03, .96, f"$D = {f['A']:.2f}N^{{{f['b']:+.2f}}}k^{{{f['c']:+.2f}}}$\n"
                                 f"$b$={f['b']:+.2f} [{f['b_ci'][0]:+.2f},{f['b_ci'][1]:+.2f}]   "
                                 f"$c$={f['c']:+.2f} [{f['c_ci'][0]:+.2f},{f['c_ci'][1]:+.2f}]",
                       transform=b.transAxes, fontsize=6.8, va="top",
                       bbox=dict(fc="white", ec="0.7", alpha=.85, boxstyle="round,pad=0.3"))
            b.set(xlabel="k (bits)", ylabel="$D_{PR}$", xticks=ks, ylim=(0, None),
                  title="$D_{PR}$ vs k\ndashed = fitted law")
            b.legend(fontsize=7, loc="lower right")
            b.grid(alpha=.25)
    fig.suptitle("Effective dimensionality of the population activity at the END of training\n"
                 "$D_{PR}$ = (Σλ)²/Σλ² over the activity covariance spectrum  ·  "
                 "how many directions the trajectory occupies, NOT how many units are active",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return ps.save(fig, "dimensionality_matrix", tight=False)


if __name__ == "__main__":
    main()
