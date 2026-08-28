#!/usr/bin/env python3
"""
Cross-channel interference vs task complexity: structural coupling and linear response.

The loss-floor fit says the total floor grows as k*f1 + g*[C(k,2)]^0.81, i.e. interference grows as
~k^1.6 rather than the k^2 that all-to-all pairwise interference would give. Two mechanism families
survive that, and they are distinguished by measuring interference DIRECTLY rather than inferring it
from the shape of a five-point curve:

  weaker pairs   every channel pair stays coupled, but each pair weakens as k grows.
                 Quantitatively: per-channel excess ~ k^0.6 spread over (k-1) partners gives
                 eps^2 ~ k^-0.4, so coupling ~ k^-0.2 - only a ~20% drop from k=2 to k=6. Small
                 signal, so the controls below matter more than the statistic.
  fewer pairs    coupling per connected pair stays flat, but the between-block distribution becomes
                 BIMODAL - a few strongly coupled pairs, the rest near zero - with the count of
                 effectively coupled pairs growing as k^1.6 instead of k^2.

Telling those apart needs the DISTRIBUTION of between-channel coupling, not only its mean, so the
spread and the bimodality are reported alongside.

TWO METRICS, because structural and functional coupling are not the same thing: a network can keep
raw recurrent connectivity and cancel its effect at the readout.

  (1) STRUCTURAL.  Assign each active unit to the channel it encodes most strongly, then compare
      mean |W_rec[i,j]| for i, j in DIFFERENT channel blocks against the same within blocks. Reported
      as the ratio between/within, which is 1 at chance and < 1 if the network segregates, plus a
      label-permutation null that absorbs any block-size artifact.

  (2) LINEAR RESPONSE - the quantity the floor law is actually about. Linearising at an operating
      point with activation pattern d = 1[y > 0], the steady-state gain from input channel A to
      output channel B is

          G[B, A] = W_out[B,:] @ diag(d) @ (I - W_rec @ diag(d))^-1 @ W_inp[:, A]

      The diagonal is the intended gain, the off-diagonal is cross-talk, and the interference measure
      is mean|off-diagonal| / mean|diagonal| - interference per unit of signal, dimensionless and
      comparable across k and N. Evaluated at real trajectory states, each with its own d, and
      averaged, since the network is piecewise linear and d varies over the trial.

CONTROLS, without which a 20% effect is not measurable:
  - per CONNECTION, not summed: block sizes shrink as ~N/k, so totals fall trivially.
  - ratios, not absolutes: the overall weight scale may drift with k for unrelated reasons.
  - a label-permutation null, recomputed per network.
  - a purity floor on block assignment: argmax|beta| is arbitrary for genuinely mixed units near the
    origin, so units below the floor are excluded and the excluded count is reported.

Output: img/internal_figures/flipflop_interference.png

Usage:  python flipflop_interference.py [purity_floor]
"""

import os
import sys
import glob
import numpy as np
from scipy.linalg import solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, participation
from flipflop_fixedpoints import SWEEP, load_net
from flipflop_selectivity import rates_and_targets, selectivity

N_PERM = 200          # label permutations for the structural null
N_STATES = 16         # operating points at which the linear response is evaluated
PURITY = 0.60         # min max|beta|/|beta| for a unit to be assigned to a channel


def cells():
    """Every (k, N) cell with completed networks, as {(k, N): [folders]}."""
    out = {}
    for d in sorted(glob.glob(os.path.join(SWEEP, "EqType=h_k=*_N=*_iters=*", "*"))):
        if not os.path.isdir(d) or not glob.glob(os.path.join(d, "*LastParams*.npz")):
            continue
        import re
        m = re.search(r"_k=(\d+)_N=(\d+)", d)
        out.setdefault((int(m.group(1)), int(m.group(2))), []).append(d)
    return out


def assign(beta, active):
    """Channel label per unit by strongest encoding weight, restricted to pure-enough units.

    Args:
        beta: (N, k) selectivity vectors; active: (N,) bool mask of non-silent units.
    Returns:
        (labels, mask) - labels in 0..k-1, and the mask of units actually assigned.
    """
    norm = np.linalg.norm(beta, axis=1)
    purity = np.max(np.abs(beta), axis=1) / np.maximum(norm, 1e-12)
    mask = active & (purity >= PURITY) & (norm > 0)
    return np.argmax(np.abs(beta), axis=1), mask


def structural(W_rec, lab, mask, k, rng):
    """Between- vs within-channel recurrent coupling, against a label-permutation null.

    Args:
        W_rec: (N, N); lab: channel label per unit; mask: which units are assigned;
        k: number of channels; rng: for the permutation null.
    Returns:
        dict with the between/within ratio, the null ratio, and the spread of per-pair values.
    """
    idx = np.flatnonzero(mask)
    if len(idx) < 4 * k:
        return None
    A = np.abs(W_rec[np.ix_(idx, idx)])
    l = lab[idx]
    same = l[:, None] == l[None, :]
    off = ~np.eye(len(idx), dtype=bool)
    within = A[same & off].mean()
    between = A[~same].mean()

    # per-PAIR between-block means, to see whether the distribution is bimodal (fewer pairs) or
    # uniformly shifted (weaker pairs).
    pair = [A[np.ix_(l == a, l == b)].mean()
            for a in range(k) for b in range(a + 1, k)
            if (l == a).sum() and (l == b).sum()]
    pair = np.array(pair) / max(within, 1e-12)

    null = []
    for _ in range(N_PERM):
        lp = rng.permutation(l)
        s = lp[:, None] == lp[None, :]
        null.append(A[~s].mean() / max(A[s & off].mean(), 1e-12))
    return dict(ratio=between / max(within, 1e-12), null=float(np.mean(null)),
                pair_cv=float(np.std(pair) / max(np.mean(pair), 1e-12)),
                n_assigned=len(idx))


def linear_response(p, states, k):
    """Cross-talk / signal from the steady-state input-output gain at real operating points.

    Args:
        p: parameter dict (W_rec, W_inp, W_out); states: (N, n_states) pre-activations;
        k: number of channels.
    Returns:
        (mean |off-diagonal| / mean |diagonal|, mean |diagonal|) averaged over operating points.
    """
    W_rec, W_inp, W_out = p["W_rec"], p["W_inp"], p["W_out"]
    N = W_rec.shape[0]
    eye = np.eye(N)
    ratios, gains = [], []
    for j in range(states.shape[1]):
        d = (states[:, j] > 0).astype(float)
        try:
            X = solve(eye - W_rec * d[None, :], W_inp, assume_a="gen")   # (N, k)
        except np.linalg.LinAlgError:
            continue
        G = W_out @ (d[:, None] * X)                                      # (k, k)
        diag = np.abs(np.diag(G)).mean()
        offd = np.abs(G[~np.eye(k, dtype=bool)]).mean()
        if diag > 1e-12:
            ratios.append(offd / diag)
            gains.append(diag)
    if not ratios:
        return np.nan, np.nan
    return float(np.median(ratios)), float(np.median(gains))


def main():
    """Measure both interference metrics for every completed cell and plot against k."""
    global PURITY
    if len(sys.argv) > 1:
        PURITY = float(sys.argv[1])
    rng = np.random.default_rng(0)
    by = cells()
    if not by:
        raise SystemExit(f"no completed networks under {SWEEP}")

    res = {}
    print(f"purity floor {PURITY}, {N_PERM} permutations, {N_STATES} operating points")
    print("%3s %6s %8s %10s %10s %9s %12s %10s"
          % ("k", "N", "n_assig", "struct", "null", "pair CV", "lin.resp", "gain"))
    for (k, N) in sorted(by):
        if k < 2:
            continue
        S, Snull, CV, LR, GA = [], [], [], [], []
        for folder in by[(k, N)]:
            rnn, p = load_net(folder)
            r, b = rates_and_targets(rnn, folder, batch=32)
            beta, _ = selectivity(r, b)
            pp = participation(r)
            active = pp >= 0.05 * np.quantile(pp, 0.95)
            lab, mask = assign(beta, active)
            st = structural(p["W_rec"], lab, mask, k, rng)
            states = np.array(rnn.get_history()).reshape(p["W_rec"].shape[0], -1)
            sel = rng.choice(states.shape[1], size=min(N_STATES, states.shape[1]), replace=False)
            lr, ga = linear_response(p, states[:, sel], k)
            if st:
                S.append(st["ratio"]); Snull.append(st["null"]); CV.append(st["pair_cv"])
            LR.append(lr); GA.append(ga)
        if not S:
            continue
        res[(k, N)] = (np.mean(S), np.std(S), np.mean(Snull), np.mean(CV),
                       np.nanmean(LR), np.nanstd(LR), np.nanmean(GA))
        print("%3d %6d %8d %5.3f+-%.3f %10.3f %9.3f %6.4f+-%.4f %10.4f"
              % (k, N, st["n_assigned"], res[(k, N)][0], res[(k, N)][1], res[(k, N)][2],
                 res[(k, N)][3], res[(k, N)][4], res[(k, N)][5], res[(k, N)][6]))

    Ns = sorted({N for _, N in res})
    ks = sorted({k for k, _ in res})
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for N in Ns:
        xs = [k for k in ks if (k, N) in res]
        if not xs:
            continue
        c = {500: "C0", 1000: "C3", 2000: "C2"}.get(N, "C7")
        ax[0].errorbar(xs, [res[(k, N)][0] for k in xs], yerr=[res[(k, N)][1] for k in xs],
                       fmt="o-", color=c, lw=2, ms=7, capsize=3, label=f"N={N}")
        ax[0].plot(xs, [res[(k, N)][2] for k in xs], ":", color=c, lw=1.4, alpha=.8)
        ax[1].errorbar(xs, [res[(k, N)][4] for k in xs], yerr=[res[(k, N)][5] for k in xs],
                       fmt="o-", color=c, lw=2, ms=7, capsize=3, label=f"N={N}")
        ax[2].plot(xs, [res[(k, N)][3] for k in xs], "o-", color=c, lw=2, ms=7, label=f"N={N}")

    # k^-0.2 is what "weaker pairs" predicts; anchor it at the smallest k of the largest N.
    ref = [k for k in ks if (k, Ns[-1]) in res]
    if ref:
        a0 = res[(ref[0], Ns[-1])][4]
        xx = np.array(ref, float)
        ax[1].plot(xx, a0 * (xx / xx[0]) ** -0.2, "k--", lw=1.4,
                   label=r"$k^{-0.2}$ (weaker-pairs prediction)")
    ax[0].axhline(1.0, color="k", lw=1, alpha=.5)
    ax[0].set(xlabel="k (bits)", ylabel="between / within coupling", xticks=ks,
              title="(a) STRUCTURAL: recurrent coupling\ndotted = label-permutation null; 1 = chance")
    ax[1].set(xlabel="k (bits)", ylabel="mean |off-diag| / mean |diag| gain", xticks=ks,
              title="(b) LINEAR RESPONSE: cross-talk per unit of signal\n"
                    r"$G=W_{out}D(I-W_{rec}D)^{-1}W_{inp}$")
    ax[2].set(xlabel="k (bits)", ylabel="CV of per-pair coupling", xticks=ks,
              title="(c) spread across channel PAIRS\nrising = bimodal (fewer pairs); "
                    "flat = uniform (weaker pairs)")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=.3)
    fig.suptitle("Cross-channel interference vs task complexity — structural and functional",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "flipflop_interference.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
