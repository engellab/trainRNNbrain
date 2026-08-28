#!/usr/bin/env python3
"""
Do frm/rws penalties keep units active, and do they make silencing SETTLE?

Two claims are tested, and the second is the sharper one.

  1. MORE UNITS STAY ACTIVE. The original motivation.
  2. SILENCING STOPS. Unpenalised networks never settle: at the end of every run the silent fraction
     is still moving 4-10 percentage points per DOUBLING of the budget, at every size. That means
     there is no principled iteration at which to read the unit count, and three separate stopping
     criteria were tried and all failed (directional cosine: floors on Adam's momentum; lag-scaling
     exponent: the diffusive transition never arrives; caging timescale: size-dependent). If penalties
     drive that rate to zero, "you cannot say when to stop" becomes a measured contrast rather than a
     limitation of the analysis.

The penalty sweep uses lambda_frm = 0.1, HALF the configured 0.2, because a smoke test at 0.2 drove
the silent count from 5 to 0 within 150 iterations - saturating, and so uninformative about dynamics.

A second figure compares LOSS dynamics across the same conditions, which is the "do the penalties
change learning, and do they cost task performance" question. It is drawn in the coordinates the
decay analysis settled on:

  log-log loss            the conventional view; a power law is straight only after the floor is
                          subtracted, so this one is curved by construction
  doubling difference     D(t) = L(t/2) - L(t), which scales as t^(-gamma) under a power law and
                          contains NO L_inf. This matters here because L_inf differs between
                          conditions and is model-dependent, so any coordinate that subtracts a
                          fitted floor would compare fit choices rather than networks.

Output: img/internal_figures/penalty_comparison.png, img/internal_figures/penalty_loss_dynamics.png

Usage:  python plot_penalty_comparison.py
"""

import os
import re
import sys
import glob
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_traces, IMG_DIR

PEN_SWEEP = "data/trained_RNNs/CDDM_std_g0_penalties"
NONE_SWEEP = "data/trained_RNNs/CDDM_std_g0_drift"
COND = ["none", "rws", "frm", "both"]
COL = {"none": "C7", "rws": "C0", "frm": "C3", "both": "C2"}


def load_penalty(sweep):
    """Load penalty-sweep traces, keyed by (N, condition).

    The penalty folders are named `EqType=h_N=<N>_pen=<name>` with no `iters=` field, so the
    drift-sweep loader's regex does not match them.

    Args:
        sweep: path to the penalty sweep folder.
    Returns:
        dict {(N, cond): [trace dicts]} with a "loss" key added to each trace.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(sweep, "*", "*", "*ParticipationTrace.pkl"))):
        m = re.search(r"_N=(\d+)_pen=(\w+)", f)
        if not m:
            continue
        with open(f, "rb") as fh:
            tr = pickle.load(fh)
        lf = glob.glob(os.path.join(os.path.dirname(f), "*TrainLosses.json"))
        tr["loss"] = np.array(json.load(open(lf[0]))["train_losses"], dtype=float) if lf else None
        out.setdefault((int(m.group(1)), m.group(2)), []).append(tr)
    return out


def silent_series(trace, N):
    """Silent-unit percentage over training, from the every-probe online counter."""
    it = np.asarray(trace["iters"], dtype=float)
    return it, 100 * np.asarray(trace["metrics"]["silent_1em6"], dtype=float) / N


def per_doubling(it, pct, ts):
    """Change in silent percentage over the last doubling of the budget, at each t in ts."""
    return np.array([np.interp(t, it, pct) - np.interp(t / 2, it, pct) for t in ts])


def logbin(t, y, nbins=90, t_min=20):
    """Median of y in log-spaced bins of t, so a log x-axis is sampled evenly."""
    edges = np.logspace(np.log10(t_min), np.log10(t[-1]), nbins + 1)
    idx = np.digitize(t, edges) - 1
    c, m = [], []
    for b in range(nbins):
        sel = idx == b
        if sel.sum() >= 3:
            c.append(np.sqrt(edges[b] * edges[b + 1]))
            m.append(np.median(y[sel]))
    return np.array(c), np.array(m)


def doubling_diff(L, ts, frac=0.02):
    """L(t/2) - L(t): loss removed over the last doubling. Scales as t^-gamma; no L_inf needed."""
    out_t, out_d = [], []
    for t in ts:
        if t > len(L):
            continue
        w = max(int(frac * t), 50)
        lo = L[max(int(t / 2) - w, 0):int(t / 2) + w].mean()
        hi = L[max(int(t) - w, 0):int(t)].mean()
        if lo - hi > 0:
            out_t.append(float(t))
            out_d.append(float(lo - hi))
    return np.array(out_t), np.array(out_d)


def plot_loss(pen, none, N, out):
    """Loss dynamics across penalty conditions at one size, all seeds drawn.

    Args:
        pen: {(N, cond): traces}; none: {N: traces}; N: size; out: output png path.
    """
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for cond in COND:
        traces = none[N] if cond == "none" else pen.get((N, cond), [])
        if not traces:
            continue
        finals = []
        for j, tr in enumerate(traces):
            L = tr["loss"]
            t = np.arange(1, len(L) + 1, dtype=float)
            c, m = logbin(t, L)
            ax[0].plot(c, m, "-", color=COL[cond], lw=1.5, alpha=.8,
                       label=cond if j == 0 else None)
            ts = np.unique(np.round(np.logspace(np.log10(4000), np.log10(len(L)), 34)).astype(int))
            td, dd = doubling_diff(L, ts)
            if len(td):
                ax[1].plot(td, dd, "-", color=COL[cond], lw=1.5, alpha=.8,
                           label=cond if j == 0 else None)
            finals.append(L[-4000:].mean())
        print(f"  N={N} {cond:5s}: final loss {np.mean(finals):.5f} +- {np.std(finals):.5f}  "
              f"(n={len(traces)})")
    ax[0].set(xscale="log", yscale="log", xlabel="iteration", ylabel="training loss",
              title="(a) loss, log-log\ncurved because the floor is not subtracted")
    ax[1].set(xscale="log", yscale="log", xlabel="iteration $t$",
              ylabel="$L(t/2)-L(t)$",
              title=r"(b) $L_\infty$-free power-law coordinate"
                    r"\nstraight = power law; slope $=-\gamma$")
    for a in ax:
        a.legend(fontsize=9)
        a.grid(alpha=.3, which="both")
    fig.suptitle(f"Loss dynamics across penalty conditions, N={N} — every seed drawn\n"
                 r"$\lambda_{rws}=0.05$, $\lambda_{frm}=0.1$", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    """Compare silencing level and its rate across penalty conditions."""
    pen = load_penalty(PEN_SWEEP)
    none = load_traces(NONE_SWEEP)
    Ns = sorted({n for n, _ in pen})
    if not Ns:
        sys.exit(f"no penalty traces under {PEN_SWEEP}")

    fig, ax = plt.subplots(2, len(Ns), figsize=(7 * len(Ns), 9), squeeze=False)
    for col, N in enumerate(Ns):
        for cond in COND:
            traces = none[N] if cond == "none" else pen.get((N, cond), [])
            if not traces:
                continue
            end = min(t["iters"][-1] for t in traces)
            ts = np.unique(np.round(np.logspace(np.log10(2000), np.log10(end), 40)).astype(int))
            lv, rt = [], []
            for t in traces:
                it, p = silent_series(t, N)
                ax[0][col].plot(it, p, "-", color=COL[cond], lw=1.3, alpha=.65)
                lv.append(np.interp(end, it, p))
                rt.append(per_doubling(it, p, ts))
            mu = np.mean(rt, axis=0)
            ax[1][col].plot(ts, mu, "-o", color=COL[cond], ms=3.5, lw=1.8,
                            label=f"{cond}  (final {np.mean(lv):.1f}%)")
            ax[1][col].fill_between(ts, mu - np.std(rt, axis=0), mu + np.std(rt, axis=0),
                                    color=COL[cond], alpha=.18)
            print(f"  N={N:5d} {cond:5s}: final silent {np.mean(lv):5.1f} +- {np.std(lv):4.1f} %"
                  f"   rate at end {mu[-1]:+6.2f} pp/doubling   n={len(traces)}")
        ax[0][col].set(xscale="log", xlabel="iteration", ylabel="silent units (% of N)",
                       ylim=(-2, 100), title=f"(a) N={N}: silencing trajectory")
        ax[1][col].axhline(0, color="k", lw=1.4)
        ax[1][col].axhline(1, color="grey", ls="--", lw=1)
        ax[1][col].set(xscale="log", xlabel="iteration $t$",
                       ylabel="change over $t/2\\to t$ (pp of N)",
                       title=f"(b) N={N}: has it STOPPED?\n0 = settled; dashed = 1 pp target")
        ax[1][col].legend(fontsize=9)
        for r in (0, 1):
            ax[r][col].grid(alpha=.3)
    handles = [plt.Line2D([], [], color=COL[c], lw=2, label=c) for c in COND]
    ax[0][0].legend(handles=handles, fontsize=9)
    fig.suptitle(r"Penalties vs none: $\lambda_{rws}=0.05$, $\lambda_{frm}=0.1$, 200k iterations"
                 "\nLEVEL is the original claim; RATE (bottom) is whether training can be stopped "
                 "at all", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = os.path.join(IMG_DIR, "penalty_comparison.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}\n")
    for N in Ns:
        plot_loss(pen, none, N, os.path.join(IMG_DIR, f"penalty_loss_dynamics_N{N}.png"))


if __name__ == "__main__":
    main()
