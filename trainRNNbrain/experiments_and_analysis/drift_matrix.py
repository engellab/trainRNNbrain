#!/usr/bin/env python3
"""
Final lag exponent alpha over the whole (N, k) grid: 3 weight matrices x 4 penalty conditions.

alpha is the exponent in |W(t+L) - W(t)| / |W(t)| ~ L^alpha:
    alpha ~ 1.0   ballistic - updates still BIASED, the network is travelling somewhere
    alpha ~ 0.5   diffusive - updates decorrelated, jittering in place
    alpha < 0.5   confined  - mean-reverting inside a basin

⚠️ "ALPHA AT THE END" IS THE TAIL MEDIAN, NOT THE LAST PROBE. Per-probe alpha is spiky because
`drift_alpha` fits across whichever lags are finite at that probe, and the Trainer logs lag100 every
100 iterations, lag1000 every 1000 and lag10000 every 10000 - so 9 probes in 10 use two lags and
every tenth uses three, with different noise (W_inp: 2-lag median 0.849 sd 0.224 vs 3-lag 0.795 sd
0.128). Quoting alpha[-1] therefore returns whatever the final probe happened to be: on one `both`
network it gave 0.05 against a tail median of 0.78. Every value here is the median over the final
TAIL_WINDOW iterations.

⚠️ TWO FIGURES, BECAUSE THE BUDGETS ARE NOT EQUAL. none runs 400-500k, rws 150k, frm and both 400k.
    _own  each run at its OWN final iterations - what "at the end of the runs" literally means, but
          cross-panel comparison is then confounded with budget, since alpha keeps falling with
          training in the conditions that settle at all.
    _cap  every run truncated to a common 150k first - the only cross-penalty-valid version.
Within a panel the budget is uniform (except `none`, 400k at N=2000 vs 500k elsewhere), so the (N, k)
STRUCTURE inside a panel is readable in both.

Output: img/internal_figures/drift_matrix_own.png, drift_matrix_cap.png

Usage:  python drift_matrix.py [TAIL_WINDOW] [T_CAP]
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, drift_alpha
import plotstyle as ps

ROOTS = {"ksweep": "data/trained_RNNs/NBitFlipFlop_std_ksweep",
         "pen": "data/trained_RNNs/NBitFlipFlop_std_pen",
         "penlong": "data/trained_RNNs/NBitFlipFlop_std_penlong",
         "bigN": "data/trained_RNNs/NBitFlipFlop_std_bigN"}
# frm at 150k is RETRACTED (not converged); its cells come from penlong only.
SKIP = {("pen", "frm")}
VARS = ["W_inp", "W_rec", "W_out"]
VLAB = {"W_inp": "$W_{inp}$", "W_rec": "$W_{rec}$", "W_out": "$W_{out}$"}
PENS = ["none", "rws", "frm", "both"]
TAIL_WINDOW = 50_000
T_CAP = 150_000
MIN_ITERS = 50_000     # shorter traces are timing-calibration runs, not experiments


def load():
    """Load every usable trace, keyed by (pen, k, N).

    ⚠️ Drops timing-calibration runs (400/600 iterations) - they carry a full trace and are
    indistinguishable from an experiment unless the LENGTH is checked - and drops the retracted
    150k frm cells.

    Returns:
        (dict {(pen, k, N): [traces]}, dict {(pen, k, N): budget}, list of dropped descriptions).
    """
    out, buds, dropped = {}, {}, []
    for tag, root in ROOTS.items():
        for f in sorted(glob.glob(os.path.join(root, "*", "*", "*ParticipationTrace.pkl"))):
            m = re.search(r"_k=(\d+)_N=(\d+)(?:_pen=([a-z]+))?", f)
            if not m:
                continue
            pen = m.group(3) or "none"
            if (tag, pen) in SKIP:
                continue
            with open(f, "rb") as fh:
                tr = pickle.load(fh)
            L = tr["metrics"].get("loss_clean_train", [])
            n = len(L) * 10
            k, N = int(m.group(1)), int(m.group(2))
            if n < MIN_ITERS:
                dropped.append(f"calibration run ({n} it) {pen} k={k} N={N}"); continue
            if np.isnan(np.asarray(L, dtype=float)).any():
                dropped.append(f"diverged {pen} k={k} N={N}"); continue
            out.setdefault((pen, k, N), []).append(tr)
            buds[(pen, k, N)] = max(buds.get((pen, k, N), 0), n)
    return out, buds, dropped


def tail_alpha(tr, var, t_cap=None, window=TAIL_WINDOW):
    """Median alpha over the final `window` iterations, optionally after truncating at t_cap.

    Args:
        tr: trace dict; var: "W_inp"/"W_rec"/"W_out"; t_cap: truncate here first, or None for the
            run's own end; window: width of the tail median, in iterations.
    Returns:
        float alpha, or nan if the trace has too few probes.
    """
    it, al = drift_alpha(tr, var)
    if len(al) < 5:
        return float("nan")
    if t_cap is not None:
        keep = it <= t_cap
        it, al = it[keep], al[keep]
        if len(al) < 5:
            return float("nan")
    return float(np.median(al[it > it.max() - window]))


def grid(data, pen, var, ks, Ns, t_cap, min_budget=0, buds=None):
    """Mean and sd of tail alpha across seeds, as two (len(Ns), len(ks)) arrays.

    Args:
        data: {(pen,k,N): [traces]}; pen, var: condition and weight matrix; ks, Ns: axes;
        t_cap: truncate every trace here first, or None for its own end;
        min_budget: skip any cell whose runs are shorter than this - a cell read "at 400k" must
            actually HAVE 400k, otherwise the tail median is taken at whatever the run reached;
        buds: {(pen,k,N): budget}, required when min_budget > 0.
    Returns:
        (mean, sd, n) arrays, nan / 0 where the cell is missing.
    """
    Z = np.full((len(Ns), len(ks)), np.nan)
    S = np.full((len(Ns), len(ks)), np.nan)
    C = np.zeros((len(Ns), len(ks)), dtype=int)
    for (p, k, N), trs in data.items():
        if p != pen or k not in ks or N not in Ns:
            continue
        if min_budget and buds is not None and buds.get((p, k, N), 0) < min_budget:
            continue
        v = [tail_alpha(t, var, t_cap) for t in trs]
        v = [x for x in v if np.isfinite(x)]
        if v:
            i, j = Ns.index(N), ks.index(k)
            Z[i, j], S[i, j], C[i, j] = np.mean(v), np.std(v), len(v)
    return Z, S, C


def figure(data, buds, ks, Ns, t_cap, name, subtitle, min_budget=0):
    """Draw the 3 x 4 matrix of final alpha and save it."""
    fig, ax = plt.subplots(len(VARS), len(PENS), figsize=(4.3 * len(PENS), 3.5 * len(VARS)),
                           squeeze=False)
    im = None
    for r, var in enumerate(VARS):
        for c, pen in enumerate(PENS):
            a = ax[r][c]
            Z, S, C = grid(data, pen, var, ks, Ns, t_cap, min_budget, buds)
            if not np.isfinite(Z).any():
                a.text(.5, .5, f"no {pen} data", ha="center", va="center", transform=a.transAxes,
                       fontsize=10, color="0.5")
                a.set_xticks([]); a.set_yticks([])
                if r == 0:
                    a.set_title(pen, fontsize=12, fontweight="bold")
                continue
            im = a.imshow(Z, cmap="RdYlBu_r", vmin=0.0, vmax=1.1, aspect="auto")
            for i in range(len(Ns)):
                for j in range(len(ks)):
                    if np.isfinite(Z[i, j]):
                        col = "white" if (Z[i, j] > 0.85 or Z[i, j] < 0.2) else "black"
                        a.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="bottom", fontsize=7.5,
                               color=col)
                        a.text(j, i, f"±{S[i, j]:.2f} (n{C[i, j]})", ha="center", va="top",
                               fontsize=5.6, color=col, alpha=.85)
                    else:
                        a.text(j, i, "·", ha="center", va="center", fontsize=9, color="0.6")
            a.set(xticks=range(len(ks)), xticklabels=ks, yticks=range(len(Ns)),
                  yticklabels=[str(n) for n in Ns])
            if r == 0:
                bs = sorted({buds[g] for g in buds if g[0] == pen
                             and (not min_budget or buds[g] >= min_budget)}) or [0]
                lab = f"{bs[0]//1000}k" if len(bs) == 1 else f"{min(bs)//1000}-{max(bs)//1000}k"
                a.set_title(f"{pen}\n" + (f"budget {lab}" if t_cap is None else f"read at {t_cap//1000}k"),
                            fontsize=11, fontweight="bold")
            if c == 0:
                a.set_ylabel(f"{VLAB[var]}\n\nN (units)", fontsize=10)
            if r == len(VARS) - 1:
                a.set_xlabel("k (bits)")
    if im is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.015)
        cb.set_label(r"final $\alpha$   (1.0 ballistic  ·  0.6 threshold  ·  0.5 diffusive)",
                     fontsize=9)
        for y in (0.5, 0.6, 1.0):
            cb.ax.axhline(y, color="k", lw=1.1)
    fig.suptitle(r"Final lag exponent $\alpha$ over the (N, k) grid — "
                 r"$\|\Delta W\|/\|W\| \sim L^{\alpha}$" + f"\n{subtitle}", fontsize=12.5)
    return ps.save(fig, name, tight=False)


def main():
    """Build both versions of the matrix and print the numbers behind them."""
    global TAIL_WINDOW, T_CAP
    if len(sys.argv) > 1:
        TAIL_WINDOW = int(sys.argv[1])
    if len(sys.argv) > 2:
        T_CAP = int(sys.argv[2])
    ps.setup()
    data, buds, dropped = load()
    ks = sorted({g[1] for g in data})
    Ns = sorted({g[2] for g in data})
    print(f"alpha = median over the final {TAIL_WINDOW:,} iterations (NOT the last probe)\n")
    print("coverage:")
    for pen in PENS:
        g = sorted({(k, N) for p, k, N in data if p == pen})
        if not g:
            print(f"  {pen:5s}  no data"); continue
        n = sum(len(v) for kk, v in data.items() if kk[0] == pen)
        bs = sorted({buds[x] for x in buds if x[0] == pen})
        print(f"  {pen:5s}  {n:3d} runs, {len(g)} cells, k={sorted({x[0] for x in g})}, "
              f"N={sorted({x[1] for x in g})}, budget {min(bs)//1000}-{max(bs)//1000}k")
    for d in sorted(set(dropped)):
        print(f"  dropped: {d} x{dropped.count(d)}")

    for t_cap, minb, name, sub in [
            (None, 0, "drift_matrix_own",
             "each run at its OWN end — within-panel structure is valid; "
             "⚠️ cross-panel is confounded with budget"),
            (T_CAP, 0, "drift_matrix_cap",
             f"every run truncated to {T_CAP:,} first — the cross-penalty-valid comparison"),
            (400_000, 400_000, "drift_matrix_400k",
             "read at 400,000 iterations; cells with a shorter budget are DISCARDED "
             "(this removes rws entirely — it was only ever run to 150k)")]:
        hdr = ("own end" if t_cap is None else f"capped at {t_cap:,}"
               + (f", cells with budget < {minb:,} discarded" if minb else ""))
        print(f"\n{'='*88}\n{hdr}\n{'='*88}")
        print("%-8s %-7s " % ("variable", "pen") + "".join(f"{'k='+str(k):>10s}" for k in ks))
        for var in VARS:
            for pen in PENS:
                Z, S, C = grid(data, pen, var, ks, Ns, t_cap, minb, buds)
                if not np.isfinite(Z).any():
                    continue
                col = np.nanmean(Z, axis=0)
                sdc = np.nanmean(S, axis=0)
                print("%-8s %-7s " % (var, pen)
                      + "".join(f"{x:6.2f}±{d:.2f}" if np.isfinite(x) else "         ·"
                                for x, d in zip(col, sdc)))
        figure(data, buds, ks, Ns, t_cap, name, sub, minb)


if __name__ == "__main__":
    main()
