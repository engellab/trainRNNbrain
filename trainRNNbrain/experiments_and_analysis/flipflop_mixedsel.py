#!/usr/bin/env python3
"""
Task-free mixed selectivity: how many independent things does each unit respond to, measured
WITHOUT regressing on the task variables.

Two measures, both computed from the activity matrix alone (units x samples, samples = time x trial):

  n_eff     NMF (non-negative matrix factorisation) of the rate matrix into d non-negative factor
            time courses and non-negative per-unit loadings h_i in R^d. The factors are discovered
            from the population, not given. With factors scaled to unit norm, h_ij^2 is the share
            of unit i's trace carried by factor j, and n_eff_i = (sum_j h_ij^2)^2 / sum_j h_ij^4 is
            the effective number of factors carrying it: 1 = pure, 2 = two equal factors, d = evenly
            mixed. The participation ratio of the SQUARED loadings, not of the loadings: the L1 form
            reads 2.1 for a unit with one loading of 1 and five of 0.1, which is 95% one factor. This is the
            task-free twin of the Hoyer sparsity on rectified-bit regression loadings in
            flipflop_arms.py. d is swept; 2k is the value the task suggests but nothing here uses
            the bits.
  n_eff (nnICA)  the same statistic from a second, unrelated decomposition: Plumbley's non-negative
            ICA. Whiten the population to d dimensions (PCA with the mean kept), then rotate those
            d whitened components until every rotated component is non-negative. Sources with
            disjoint support (the arms) make that rotation unique up to permutation. No
            reconstruction objective, no multiplicative updates: if it orders the penalties the
            same way as NMF, the ordering is not an artefact of one algorithm.
            Factor time courses are scaled to unit L2 norm in BOTH methods before loadings are
            read, so a loading measures how much of the unit's trace a factor carries.
  c_ent     entropy (bits, 50 bins on [-1, 1]) of the off-diagonal unit-correlation values. In a pure
            population the correlation between two units takes a few discrete values (same arm ~ 1,
            antipodal arm negative, unrelated bits ~ 0), so the histogram is a few spikes and the
            entropy is low; mixed units fill in the values between. This is the number behind the
            "block structure" of the correlation matrix. Computed on a common subsample of units.

⚠️ A 'nearest twin' correlation was tried and dropped: on a synthetic one-parameter continuum of
mixed units it read 0.998, the same as on pure units. It measures local density, not mixing.
⚠️ NMF IS RUN FROM SIX INITIALISATIONS AND THE LOWEST-ERROR FIT IS KEPT. On some trial batches a
single fit lands in a local optimum where every pure unit loads on ~2 factors; that optimum has
~6x the reconstruction error of the true one, so best-of-restarts removes it.

Block-structure figure: the unit correlation matrix with units sorted by their dominant NMF factor
(then by purity within it), one panel per penalty. Pure populations give crisp blocks whose entries
take a few discrete values; mixed populations smear them.

⚠️ CALIBRATION RUNS FIRST AND GATES EACH METHOD. Synthetic populations built from the real bit
traces: 'pure' (every unit relu(+-b_j) with a random gain) must read n_eff < PURE_NEFF_MAX; a
population of pure units plus 'sum2' units (relu(s1 b_i) + relu(s2 b_j), equal gains, i != j, so
the true answer is exactly 2) must read n_eff > MIXED_NEFF_MIN on the sum2 half. The pure half is
there because NMF is only identifiable when each factor has a pure unit, as real nets do. A method
that fails is dropped from the table for that k, not silently reported.
⚠️ A relu(w1 b_i + w2 b_j) synthetic with random w was used first and read ~1.35 under the
variance-share statistic. That is the RIGHT answer for it (median split of a random angle is
85/15), so it cannot serve as a calibration target; sum2 has a known answer.
⚠️ nnICA DOES NOT IDENTIFY THE ARMS AT k=8. On a pure synthetic population in d=16 a rotation with
LOWER negativity than the true arms exists (J 0.069 vs 0.100), so no optimiser can be expected to
return the arms and it reads 1.5-2.3. It passes at k=3 (1.05) and is a cross-check there only. Thresholds were fixed before the real nets were looked at. If the synthetic
checks fail, the measure is not trusted and the real numbers are not printed.

Usage:  python flipflop_mixedsel.py [k] [N]        (defaults k=3, N=2000)
Output: img/internal_figures/mixedsel_N{N}_k{k}.png
"""

import os
import sys
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP
import plotstyle as ps
from pr_matrix import PENS
from flipflop_dimensionality import run_folders
from flipflop_diversity import rates_and_targets

N_TRIALS = 32
SEED = 0
D_SWEEP = (1, 2, 3, 4)          # multiples of k for the NMF factor count
PURE_NEFF_MAX, MIXED_NEFF_MIN = 1.15, 1.7   # calibration pass thresholds
METHODS = (("NMF", lambda X, d: nmf_loadings(X, d)), ("nnICA", lambda X, d: nnica_loadings(X, d)))
N_RESTARTS = 5
COL = {"none": "#7f7f7f", "rws": "#2ca02c", "frm": "#d62728", "both": "#1f77b4"}


def live_matrix(rates):
    """Rates of the live units as a (n_live, S) matrix, each unit scaled to unit L2 norm.

    Args:
        rates: (N, T, B) non-negative rates.
    Returns:
        (n_live, S) float64 matrix; live = participation >= SILENT_FLIPFLOP.
    """
    X = rates.reshape(rates.shape[0], -1).astype(np.float64)
    live = (X.std(1) + np.quantile(X, 0.9, axis=1)) >= SILENT_FLIPFLOP
    X = X[live]
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-300)


def nmf_loadings(X, d):
    """Non-negative loadings of each unit on d discovered factors.

    Args:
        X: (n, S) non-negative unit-by-sample matrix; d: number of factors.
    Returns:
        (n, d) loadings H^T from X^T ~ W H (W: S x d factor time courses).
    """
    fits = [NMF(n_components=d, init="nndsvda", max_iter=2000, random_state=SEED, tol=1e-6).fit(X.T)]
    fits += [NMF(n_components=d, init="random", max_iter=2000, random_state=r, tol=1e-6).fit(X.T)
             for r in range(N_RESTARTS)]
    best = min(fits, key=lambda m: m.reconstruction_err_)
    W = best.transform(X.T)                                   # (S, d) factor time courses
    return best.components_.T * np.linalg.norm(W, axis=0)     # loadings in unit-norm-factor scale


def nnica_loadings(X, d, iters=1000, lr=0.3):
    """Non-negative ICA (Plumbley 2003 / Plumbley & Oja 2004) loadings of each unit on d sources.

    Whitens X to d components using the covariance over samples but WITHOUT subtracting the mean
    (the sources must stay non-negative), then does projected gradient descent on the orthogonal
    group minimising the squared negative part of the rotated components. Best of N_RESTARTS+1
    starts by that objective.

    Args:
        X: (n, S) non-negative unit-by-sample matrix; d: number of sources; iters, lr: optimiser.
    Returns:
        (n, d) mixing matrix A with X ~ A F, F the unit-L2-norm non-negative sources; entries are
        clipped at zero (a well-fit model has none negative; the clipped mass is printed by
        calibrate()).
    """
    S = X.shape[1]
    Xc = X - X.mean(1, keepdims=True)
    lam, U = np.linalg.eigh(Xc @ Xc.T / S)
    lam, U = lam[::-1][:d], U[:, ::-1][:, :d]
    Q = U.T / np.sqrt(lam)[:, None]                            # (d, n) whitening
    Z = Q @ X                                                  # (d, S), unit covariance, mean kept
    rng = np.random.default_rng(SEED)
    best = None
    for r in range(N_RESTARTS + 1):
        R = np.eye(d) if r == 0 else np.linalg.qr(rng.normal(size=(d, d)))[0]
        for _ in range(iters):
            neg = np.minimum(R @ Z, 0.0)
            G = neg @ Z.T / S                                  # dJ/dR, J = mean ||neg||^2 / 2
            step = R - lr * (G - R @ G.T @ R)                  # tangent step on the orthogonal group
            u, _, vt = np.linalg.svd(step); R = u @ vt         # polar retraction
        J = float((np.minimum(R @ Z, 0.0) ** 2).mean())
        if best is None or J < best[0]:
            best = (J, R)
    R = best[1]
    F = R @ Z                                                  # (d, S) sources
    A = (U * np.sqrt(lam)) @ R.T                               # (n, d) mixing, X ~ A F
    return np.maximum(A * np.linalg.norm(F, axis=1), 0.0)


def n_eff(H):
    """Effective number of factors carrying each unit: PR of the squared loadings per row.

    Args:
        H: (n, d) non-negative loadings, factors at unit norm.
    Returns:
        (n,) values in [1, d].
    """
    E = H ** 2
    return E.sum(1) ** 2 / np.maximum((E ** 2).sum(1), 1e-300)


def corr_entropy(X):
    """Entropy of the off-diagonal unit-correlation values, and the correlation matrix.

    Args:
        X: (n, S) rates.
    Returns:
        (entropy in bits over 50 bins on [-1, 1], (n, n) correlation matrix, (n(n-1)/2,) values).
    """
    Xc = X - X.mean(1, keepdims=True)
    Xc /= np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-300)
    C = Xc @ Xc.T
    off = C[np.triu_indices(C.shape[0], k=1)]
    p = np.histogram(off, bins=50, range=(-1, 1))[0] / off.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()), C, off


def synthetic(targets, kind, n=300, noise=0.03, rng=None):
    """Synthetic unit population built from the real bit traces.

    Args:
        targets: (k, T, B) bit traces in {-1, 0, +1}; kind: 'pure' (relu(+-b_j)) or 'sum2'
                 (relu(s1 b_i) + relu(s2 b_j), i != j, equal gains: exactly two arms); n: units;
                 noise: sd of additive Gaussian noise before rectification, relative to unit gain.
    Returns:
        (n, S) rate matrix.
    """
    rng = rng or np.random.default_rng(SEED)
    k = targets.shape[0]
    B = targets.reshape(k, -1)
    out = np.zeros((n, B.shape[1]))
    for i in range(n):
        if kind == "pure":
            j = rng.integers(k); s = rng.choice([-1.0, 1.0])
            out[i] = np.maximum(s * B[j] + noise * rng.normal(size=B.shape[1]), 0.0)
        else:
            j1, j2 = rng.choice(k, 2, replace=False); s1, s2 = rng.choice([-1.0, 1.0], 2)
            out[i] = (np.maximum(s1 * B[j1] + noise * rng.normal(size=B.shape[1]), 0.0)
                      + np.maximum(s2 * B[j2] + noise * rng.normal(size=B.shape[1]), 0.0))
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-300)


def calibrate(targets, k):
    """Run the synthetic checks; return the list of (name, fn) methods that pass both."""
    Xp = synthetic(targets, "pure")
    Xm = np.vstack([Xp, synthetic(targets, "sum2", rng=np.random.default_rng(SEED + 1))])
    n = Xp.shape[0]
    passing = []
    print(f"{'calibration':<12}{'method':<8}{'n_eff(d=2k)':>12}   requirement")
    for name, fn in METHODS:
        ne_p = float(np.median(n_eff(fn(Xp, 2 * k))))
        ne_m = float(np.median(n_eff(fn(Xm, 2 * k))[n:]))
        ok = ne_p < PURE_NEFF_MAX and ne_m > MIXED_NEFF_MIN
        print(f"{'pure':<12}{name:<8}{ne_p:>12.3f}   n_eff < {PURE_NEFF_MAX}")
        print(f"{'sum2':<12}{name:<8}{ne_m:>12.3f}   n_eff > {MIXED_NEFF_MIN} (truth 2)   "
              f"{'PASS' if ok else 'FAIL - dropped'}")
        if ok:
            passing.append((name, fn))
    return passing


def main():
    """Calibrate, then measure task-free mixed selectivity for every net at (k, N)."""
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    ps.setup()
    runs = [(f, p) for f, p, kk, nn in run_folders() if kk == k and nn == N]
    data = {p: [] for p in PENS}
    targets = None
    for f, p in runs:
        r, t = rates_and_targets(f, N_TRIALS)
        targets = targets if targets is not None else t
        data[p].append(live_matrix(r))
    print(f"N={N}, k={k}: " + ", ".join(f"{p} x{len(v)}" for p, v in data.items()))
    methods = calibrate(targets, k)
    if not methods:
        print("CALIBRATION FAILED for every method - real nets not reported")
        sys.exit(1)
    use_ica = any(name == "nnICA" for name, _ in methods)

    n_common = min(X.shape[0] for v in data.values() for X in v)
    rng = np.random.default_rng(SEED)
    print(f"\ncommon subsample for correlations/blocks: {n_common} live units per net\n")
    hdr = f"{'pen':<6}{'live':>6}" + "".join(f"{f'NMF d={m}k':>10}" for m in D_SWEEP) \
          + f"{'NMF pure':>10}{'nnICA 2k':>10}{'nnICA pure':>12}"
    print(hdr)
    rows, figs = {}, {}
    for p in PENS:
        per_seed = []
        for X in data[p]:
            sub = X[rng.choice(X.shape[0], n_common, replace=False)]
            ne = {m: n_eff(nmf_loadings(X, m * k)) for m in D_SWEEP}
            ni = n_eff(nnica_loadings(X, 2 * k)) if use_ica else np.full(X.shape[0], np.nan)
            ent, C, off = corr_entropy(sub)
            H = nmf_loadings(sub, 2 * k)
            per_seed.append(dict(live=X.shape[0], ne={m: float(np.median(v)) for m, v in ne.items()},
                                 pure=float((ne[2] < 1.2).mean()), ni=float(np.median(ni)),
                                 ni_pure=float((ni < 1.2).mean())))
            if p not in figs:                       # first seed: block figure + distributions
                order = np.lexsort((-n_eff(H), H.argmax(1)))
                figs[p] = (C[np.ix_(order, order)], ne[2], off, ent)
        rows[p] = per_seed
        for s in per_seed:
            print(f"{p:<6}{s['live']:>6}" + "".join(f"{s['ne'][m]:>10.3f}" for m in D_SWEEP)
                  + f"{s['pure']:>10.3f}{s['ni']:>10.3f}{s['ni_pure']:>12.3f}")

    fig, ax = plt.subplots(3, 4, figsize=(18, 12))
    for j, p in enumerate(PENS):
        C, ne, off, ent = figs[p]
        im = ax[0, j].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax[0, j].set(title=f"{p}: unit correlations, sorted by NMF factor", xticks=[], yticks=[])
        ax[1, j].hist(ne, bins=np.linspace(1, 2 * k, 40), color=COL[p])
        ax[1, j].set(xlabel="effective number of NMF factors (d = 2k)", ylabel="units",
                     title=f"{p}: median {np.median(ne):.2f}, {np.mean(ne < 1.2):.0%} pure")
        ax[2, j].hist(off, bins=np.linspace(-1, 1, 50), color=COL[p])
        ax[2, j].set(xlabel="pairwise unit correlation", ylabel="pairs",
                     title=f"{p}: entropy {ent:.2f} bits")
    fig.colorbar(im, ax=ax[0, :].tolist(), fraction=0.01)
    fig.suptitle(f"Task-free mixed selectivity, N={N}, k={k}, seed 0 of each penalty\n"
                 f"NMF factors discovered from activity (no bit regressors); "
                 f"correlations on {n_common} matched live units", fontsize=12.5)
    return ps.save(fig, f"mixedsel_N{N}_k{k}", tight=False)


if __name__ == "__main__":
    main()
