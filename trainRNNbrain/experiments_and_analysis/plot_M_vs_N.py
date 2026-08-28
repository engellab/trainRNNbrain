#!/usr/bin/env python3
"""
M(N): does the number of ACTIVE units saturate as the network grows?

Every size is compared at MATCHED PERFORMANCE — each network is read out at its own iteration T_N,
the first at which its training loss reaches a common level L*. This is licensed by the measured
result that the achievable loss floor is shared across sizes (see plot_shared_floor.py), so equal
loss means equal distance from the achievable optimum. It requires no convergence claim, no fit and
no asymptote.

Levels come from the ladder in the trajectory doc: the deepest is the worst final loss across all
seeds (so every seed reaches it), the shallowest keeps T >= 5000 (past the transient), and the rungs
halve the reference size's budget.

Both silent-unit definitions are reported, because they disagree by an order of magnitude at N=100
and might also disagree about saturation:
    hard        p_i < 1e-6                 truly silent
    scale-free  p_i < 0.05 * q95(p)        silent relative to the network's own scale

Panels, one row per criterion:
  (a) M vs N, one line per level, with the M=N diagonal
  (b) active FRACTION M/N vs N
  (c) local scaling exponent k = dlog M / dlog N between consecutive sizes; k=1 means M keeps up
      with N, k=0 means full saturation

Output: img/internal_figures/M_vs_N.png

Usage:  python plot_M_vs_N.py [SWEEP_FOLDER]
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CRITERIA, IMG_DIR, T_at_loss, active_count, load_traces, smooth_loss


def ladder(by, n_rungs=4, t_floor=5000):
    """Build the matching levels: deepest reachable by every seed, halving the reference budget.

    Args:
        by: {N: [traces]}; n_rungs: maximum number of levels; t_floor: minimum reference budget.
    Returns:
        list of L* values, deepest first.
    """
    L_deep = max(smooth_loss(t["loss"], len(t["loss"])) for N in by for t in by[N])
    ref = max(by)
    T_ref = max(T_at_loss(t["loss"], L_deep) for t in by[ref])
    out = []
    for k in range(n_rungs):
        Tr = int(T_ref / 2 ** k)
        if Tr < t_floor:
            break
        out.append(float(np.mean([smooth_loss(t["loss"], Tr) for t in by[ref]])))
    return out


def plot_single_level(by, Lstar, out):
    """One clean figure of M vs N at a single matching level, both criteria.

    Args:
        by: {N: [traces]}; Lstar: the matching loss level; out: output png path.
    Returns:
        dict {criterion: (Ns, means, sds)}.
    """
    Ns = sorted(by)
    res = {}
    for crit, _ in CRITERIA:
        ms, es = [], []
        for N in Ns:
            v = []
            for t in by[N]:
                T = T_at_loss(t["loss"], Lstar)
                P, I = np.array(t["participation"]), np.array(t["participation_iters"])
                if T is not None and T <= I[-1]:
                    v.append(active_count(P[np.argmin(abs(I - T))], crit))
            ms.append(np.mean(v))
            es.append(np.std(v))
        res[crit] = (np.array(Ns), np.array(ms), np.array(es))

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))
    style = {"hard": ("#1f77b4", "o", r"truly silent  $p_i<10^{-6}$"),
             "scalefree": ("#d62728", "s", r"scale-free  $p_i<0.05\,q_{95}(p)$")}
    for crit, _ in CRITERIA:
        Nsa, m, e = res[crit]
        c, mk, lab = style[crit]
        for a in ax:
            a.errorbar(Nsa, m, yerr=e, fmt="-" + mk, color=c, ms=8, lw=2, capsize=4, label=lab)
    for a in ax:
        a.plot(Ns, Ns, "k--", lw=1.2, alpha=.6, label="$M=N$ (nothing silent)")
        a.set_xlabel("network size $N$")
        a.set_ylabel("active units $M$")
        a.grid(alpha=.3)
        a.legend(fontsize=9)
    ax[0].set(xscale="log", yscale="log", title="log-log")
    ax[1].set(title="linear")
    fig.suptitle(f"Active units vs network size at matched performance, $L^*$={Lstar:.5f}\n"
                 f"(each size read at its own $T_N$; mean $\\pm$ sd over 3 seeds)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for crit, _ in CRITERIA:
        Nsa, m, e = res[crit]
        print(f"  {crit:10s} " + "  ".join(f"N={n}: {mm:.1f}+-{ee:.1f} ({100*mm/n:.1f}%)"
                                           for n, mm, ee in zip(Nsa, m, e)))
    return res


def main():
    """Compute and plot M(N) at every matching level, under both silent-unit criteria."""
    sweep = ([a for a in sys.argv[1:] if not a.startswith("--")] or
             ["data/trained_RNNs/CDDM_std_g0_drift"])[0]
    by = load_traces(sweep)
    Ns = sorted(by)
    levels = ladder(by)
    cols = plt.cm.viridis(np.linspace(0.08, 0.78, len(levels)))
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))

    for r, (crit, cname) in enumerate(CRITERIA):
        print(f"\n=== criterion: {crit} ({cname}) ===")
        for i, Lstar in enumerate(levels):
            Ms, Es, Ts = [], [], []
            for N in Ns:
                vals, ts = [], []
                for t in by[N]:
                    T = T_at_loss(t["loss"], Lstar)
                    P = np.array(t["participation"])
                    I = np.array(t["participation_iters"])
                    if T is None or T > I[-1]:
                        continue
                    vals.append(active_count(P[np.argmin(abs(I - T))], crit))
                    ts.append(T)
                Ms.append(np.mean(vals) if vals else np.nan)
                Es.append(np.std(vals) if vals else np.nan)
                Ts.append(np.mean(ts) if ts else np.nan)
            Ms, Es = np.array(Ms, float), np.array(Es, float)
            lbl = f"$L^*$={Lstar:.5f}"
            ax[r, 0].errorbar(Ns, Ms, yerr=Es, fmt="-o", color=cols[i], ms=6, capsize=3, label=lbl)
            ax[r, 1].errorbar(Ns, Ms / np.array(Ns), yerr=Es / np.array(Ns), fmt="-o",
                              color=cols[i], ms=6, capsize=3, label=lbl)
            ks = [np.log(Ms[j + 1] / Ms[j]) / np.log(Ns[j + 1] / Ns[j]) for j in range(len(Ns) - 1)]
            mids = [np.sqrt(Ns[j] * Ns[j + 1]) for j in range(len(Ns) - 1)]
            ax[r, 2].plot(mids, ks, "-o", color=cols[i], ms=6, label=lbl)
            print(f"  L*={Lstar:.5f}: " + "  ".join(
                f"N={N}: T={T:.0f} M={m:.1f}+-{e:.1f} ({100*m/N:.1f}%)"
                for N, T, m, e in zip(Ns, Ts, Ms, Es)) +
                "   k=" + ", ".join(f"{k:.2f}" for k in ks))

        ax[r, 0].plot(Ns, Ns, "k--", lw=1, alpha=.5, label="$M=N$")
        ax[r, 0].set(xscale="log", yscale="log", xlabel="$N$", ylabel="active units $M$",
                     title=f"(a) $M$ vs $N$, {cname}")
        ax[r, 1].set(xscale="log", xlabel="$N$", ylim=(0, 1.05), ylabel="$M/N$",
                     title=f"(b) active fraction, {cname}")
        ax[r, 2].axhline(1, color="k", ls="--", lw=1, alpha=.6)
        ax[r, 2].axhline(0, color="r", ls="--", lw=1, alpha=.6)
        ax[r, 2].text(Ns[0] * 1.05, 1.02, "keeps up with $N$", fontsize=8)
        ax[r, 2].text(Ns[0] * 1.05, 0.03, "saturated", color="r", fontsize=8)
        ax[r, 2].set(xscale="log", xlabel="$N$ (geometric mean of the pair)", ylim=(-0.15, 1.15),
                     ylabel=r"$k=\mathrm{d}\log M/\mathrm{d}\log N$",
                     title=f"(c) local exponent, {cname}")
        for c in range(3):
            ax[r, c].legend(fontsize=8)
            ax[r, c].grid(alpha=.25)

    fig.suptitle("Active units vs network size, compared at MATCHED PERFORMANCE "
                 "(each size read at its own $T_N$ where the loss reaches $L^*$)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(IMG_DIR, "M_vs_N.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")
    plot_single_level(by, levels[0], os.path.join(IMG_DIR, "M_vs_N_deepest.png"))


if __name__ == "__main__":
    main()
