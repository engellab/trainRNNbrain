#!/usr/bin/env python3
"""
A task-appropriate ABSOLUTE silence threshold for the n-bit flip-flop, and the M(N, k) law under it.

WHY. `p < 1e-6` reports ~0-1% silence on the flip-flop while the scale-free rule reports ~80%, so the
absolute criterion looked like it disagreed with the relative one and with CDDM. It does not. Both
tasks are cleanly BIMODAL in log participation and both silence ~80% of units; they differ only in
where the silent mode sits - exactly 0 on CDDM, ~2e-4 on the flip-flop, each ~3-4 orders below its own
active mode. 1e-6 was calibrated to CDDM's dynamic range and simply falls below the flip-flop's
silent mode. The fix is to derive the threshold from the data instead of importing a constant.

METHOD. Otsu's method on log10(p) (`common.otsu_threshold`): the split that maximises between-class
variance, i.e. the antimode between the two modes. Parameter-free and scale-adaptive.

VALIDATION, with the falsifier fixed first. A per-network threshold is only worth promoting to a
per-task constant if it is STABLE. Pre-registered checks:

  (1) spread across cells      adoptable if the 10-90th percentile spans < 1 decade. A threshold that
                               wanders more than that is tracking each network, not the task, and the
                               honest conclusion is to stay scale-free.
  (2) agreement with scale-free  the two rules should now select nearly the same units, since both are
                               trying to find the same bimodal split. Pre-set: median |ΔM|/N < 0.05.
  (3) it must NOT reproduce 1e-6  if the derived threshold came out near 1e-6 the whole premise would
                               be wrong and the original criterion vindicated.

Failing (1) or (2) is reported as a failure, not patched with a different statistic.

Then M ~ A N^b k^c is refitted under the new threshold at every read-out criterion, to see whether
the absolute rule - once calibrated - agrees with the scale-free one about b and c.

Output: img/internal_figures/flipflop_hard_threshold.png

Usage:  python flipflop_hard_threshold.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, otsu_threshold, active_count
import plotstyle as ps
import flipflop_figures as F

SPREAD_MAX = 1.0        # decades; wider than this and a per-task constant is not defensible
DELTA_MAX = 0.05        # max median |M_new - M_scalefree| / N for the two rules to count as agreeing


def per_net_thresholds(runs, crit="excess"):
    """Otsu threshold for every run, at one read-out iteration.

    Args:
        runs: run records carrying "T"; crit: read-out criterion naming when to take the snapshot.
    Returns:
        list of (k, N, threshold, participation vector) with non-finite thresholds dropped.
    """
    out = []
    for r in runs:
        t = r["T"][crit]
        if not np.isfinite(t):
            continue
        ok = np.flatnonzero(r["piters"] <= t)
        if ok.size == 0:
            continue
        p = np.asarray(r["part"][int(ok[-1])], dtype=float)
        th = otsu_threshold(p)
        if np.isfinite(th):
            out.append((r["k"], r["N"], th, p))
    return out


def main():
    """Derive the threshold, validate it against the pre-set checks, then refit the law under it."""
    ps.setup()
    runs, dropped = F.load()
    for r in runs:
        f = F.fit_run(r["loss"], F.T_ITER)
        r["L_inf"] = f[0] if f else None
    worst = max(r["L_inf"] for r in runs if r["L_inf"])
    F.readout_times(runs, F.LOSS_MARGIN * worst)
    ks = sorted({r["k"] for r in runs})
    Ns = sorted({r["N"] for r in runs})

    rows = per_net_thresholds(runs)
    th = np.array([r[2] for r in rows])
    lo, hi = np.percentile(np.log10(th), [10, 90])
    spread = hi - lo
    med = float(np.median(th))
    print(f"derived Otsu thresholds over {len(rows)} networks")
    print(f"  median          {med:.3e}   (log10 = {np.log10(med):.2f})")
    print(f"  10-90 pct       {10**lo:.2e} .. {10**hi:.2e}   spread {spread:.2f} decades")
    print(f"  min / max       {th.min():.2e} / {th.max():.2e}")
    print("  by N: " + "  ".join(
        f"N={N}: {np.median([r[2] for r in rows if r[1]==N]):.2e}" for N in Ns))
    print("  by k: " + "  ".join(
        f"k={k}: {np.median([r[2] for r in rows if r[0]==k]):.2e}" for k in ks))

    print(f"\nCHECK 1  spread < {SPREAD_MAX} decade: "
          f"{'PASS' if spread < SPREAD_MAX else 'FAIL'}  ({spread:.2f})")
    d = [abs(active_count(p, "scalefree") - (p >= med).sum()) / N for _, N, _, p in rows]
    print(f"CHECK 2  agrees with scale-free (median |dM|/N < {DELTA_MAX}): "
          f"{'PASS' if np.median(d) < DELTA_MAX else 'FAIL'}  ({np.median(d):.3f})")
    print(f"CHECK 3  differs from 1e-6: "
          f"{'PASS' if med > 1e-5 else 'FAIL'}  (median is {med/1e-6:.0f}x larger)")
    ok = spread < SPREAD_MAX and np.median(d) < DELTA_MAX and med > 1e-5

    # Round to a clean decade-ish constant so the paper can quote one number rather than a fit.
    TH = float(10 ** np.round(np.log10(med), 1))
    print(f"\nadopted flip-flop absolute threshold: p < {TH:.1e}"
          f"   ({'validated' if ok else 'NOT VALIDATED - do not adopt'})")

    print(f"\nsilent fraction under each rule, at the `excess` read-out:")
    print("%4s %6s %10s %10s %10s" % ("k", "N", "p<1e-6", f"p<{TH:.0e}", "scale-free"))
    for N in Ns:
        for k in ks:
            sel = [r for r in rows if r[0] == k and r[1] == N]
            if not sel:
                continue
            f_old = np.mean([1 - active_count(p, "hard") / N for _, _, _, p in sel])
            f_new = np.mean([1 - (p >= TH).sum() / N for _, _, _, p in sel])
            f_sf = np.mean([1 - active_count(p, "scalefree") / N for _, _, _, p in sel])
            print("%4d %6d %10.3f %10.3f %10.3f" % (k, N, f_old, f_new, f_sf))

    # ---- refit the law under the new threshold, at every criterion ---------------------------
    def M_new(r, t):
        """Active-unit count under the adopted absolute threshold at iteration t."""
        if not np.isfinite(t):
            return float("nan")
        okk = np.flatnonzero(r["piters"] <= t)
        if okk.size == 0:
            return float("nan")
        return float((np.asarray(r["part"][int(okk[-1])], dtype=float) >= TH).sum())

    print(f"\nM ~ A N^b k^c under p >= {TH:.0e}, per read-out criterion "
          f"(scale-free values in brackets):")
    print("%-26s %18s %22s %11s" % ("criterion", "b", "c", "lack-fit p"))
    laws = {}
    for c in F.CRITS:
        Ks, NNs, Ms = [], [], []
        for r in runs:
            m = M_new(r, r["T"][c])
            if np.isfinite(m) and m > 0:
                Ks.append(r["k"]); NNs.append(r["N"]); Ms.append(m)
        if len(Ms) < 8:
            continue
        Ks, NNs, Ms = np.array(Ks, float), np.array(NNs, float), np.array(Ms, float)
        X = np.column_stack([np.ones_like(Ms), np.log(NNs), np.log(Ks)])
        beta, *_ = np.linalg.lstsq(X, np.log(Ms), rcond=None)
        sf = F.fit_law(runs, c, "scalefree")
        laws[c] = (beta[1], beta[2], sf)
        print("%-26s %6.2f  [sf %5.2f] %10.3f  [sf %6.3f] %11s"
              % (F.CRIT_DESC[c].split("\n")[0], beta[1], sf["b"], beta[2], sf["c"], ""))

    # ---- figure -------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
    allp = np.concatenate([p for _, _, _, p in rows])
    nz = allp[allp > 0]
    ax[0].hist(np.log10(nz), bins=120, color="0.6")
    for x, lab, col in [(np.log10(1e-6), "old  $p<10^{-6}$", "C3"),
                        (np.log10(TH), f"adopted  $p<{TH:.0e}$", "C0")]:
        ax[0].axvline(x, color=col, lw=2, label=lab)
    ax[0].set(xlabel="$\\log_{10}$ participation", ylabel="units (all nets pooled)",
              title="(a) the flip-flop is bimodal;\n$10^{-6}$ sits below BOTH modes")
    ax[0].legend()

    for N in Ns:
        v = [r[2] for r in rows if r[1] == N]
        ax[1].scatter([N] * len(v), v, color=ps.col_n(N), s=18, alpha=.75)
    ax[1].axhline(med, color="k", ls="--", lw=1.2, label=f"median {med:.1e}")
    ax[1].axhline(1e-6, color="C3", ls=":", lw=1.2, label="old $10^{-6}$")
    ax[1].set(xscale="log", yscale="log", xlabel="N (units)", ylabel="derived threshold",
              xticks=Ns, xticklabels=[str(n) for n in Ns],
              title=f"(b) is it a task constant?\nspread {spread:.2f} decades "
                    f"({'PASS' if spread < SPREAD_MAX else 'FAIL'})")
    ax[1].legend()

    xs = [1 - active_count(p, "scalefree") / N for _, N, _, p in rows]
    ys = [1 - (p >= TH).sum() / N for _, N, _, p in rows]
    zs = [1 - active_count(p, "hard") / N for _, N, _, p in rows]
    ax[2].plot([0, 1], [0, 1], "k--", lw=1, alpha=.6)
    ax[2].scatter(xs, ys, s=20, color="C0", label=f"adopted $p<{TH:.0e}$")
    ax[2].scatter(xs, zs, s=20, color="C3", label="old $p<10^{-6}$")
    ax[2].set(xlabel="silent fraction, scale-free", ylabel="silent fraction, absolute rule",
              xlim=(0, 1), ylim=(0, 1),
              title="(c) does the absolute rule now agree\nwith the relative one?")
    ax[2].legend()

    fig.suptitle("A task-calibrated absolute silence threshold for the n-bit flip-flop", fontsize=12)
    return ps.save(fig, "flipflop_hard_threshold")


if __name__ == "__main__":
    main()
