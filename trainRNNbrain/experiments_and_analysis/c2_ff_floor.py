#!/usr/bin/env python3
"""
Claim C2 check: the 3-bit flip-flop size series read at each network's own noisy-loss floor.

For every run in the flip-flop cells of fig_paper_F1.SCALING this prints the joined warm-start
chain, where its noisy loss first reaches 1.05x its own fitted floor (common.noisy_floor, through
fig_paper_F1.floor_crossing), the active count at that crossing (fig_paper_F1.runs_with_loss), and
the reason for every run without a crossing. It then fits active ~ N^a over the sizes that have a
floor read-out at every run that has one.

Sanity checks printed per run, each with the result that would show the read-out is wrong:
  * participation probes span the WHOLE chain (last probe within 1% of the chain length). A last
    probe near 125k on a 200k chain would mean the trace is in segment-relative iterations and every
    active count read at an absolute crossing is taken at the wrong time.
  * the probe nearest the crossing is within 1,000 iterations of it.
  * the fitted floor sits at or below the end-of-run smoothed loss and above half of it.
For a run with no crossing, the fitted curve is extrapolated to where it would reach the bar.

Usage:  python c2_ff_floor.py [task label, default "3-bit flip-flop"]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import noisy_floor, warm_start_chain, stretched, FLOOR_DELTA
from fig_paper_F1 import SCALING, runs_with_loss


def extrapolated_crossing(res, delta=FLOOR_DELTA, t_max=2e6):
    """Iteration at which a run's FITTED curve would reach (1+delta) x its fitted floor.

    Args:
        res: dict returned by common.noisy_floor, with `fit` = [floor, A, tau, beta] and `escape`.
        delta: the margin over the floor; t_max: search horizon, absolute iterations.
    Returns:
        float absolute iteration, or nan if the fit never gets there before t_max.
    """
    if "fit" not in res:
        return float("nan")
    Li, A, tau, beta = res["fit"]
    t = np.logspace(2, np.log10(t_max), 20000)            # time since the escape
    hit = np.flatnonzero(stretched(t, Li, A, tau, beta) <= (1 + delta) * Li)
    return float(t[hit[0]] + res["escape"]) if len(hit) else float("nan")


def read_cell(N, pattern):
    """Floor read-out of every run in one cell, printed run by run.

    Args:
        N: network size (int); pattern: the cell's run-folder glob from SCALING.
    Returns:
        (list of active counts at the crossing, list of crossings, list of (run, reason) dropped).
    """
    counts, crossings, dropped = [], [], []
    for r in runs_with_loss(pattern):
        chain = warm_start_chain(os.path.join(pattern, r["name"]))
        res = noisy_floor(r["it_loss"], r["loss"])
        it_a = np.asarray(r["it_act"], float)
        n_loss = len(r["loss"])
        span_ok = abs(it_a[-1] - n_loss) <= 0.01 * n_loss
        end_med = res["smoothed"][-1] if "smoothed" in res else float("nan")
        floor = res["floor"] if res["floor"] is not None else float("nan")
        seed = r["name"].split("seed=")[1].split(";")[0]
        line = (f"N={N:5d} seed={seed:>10}  chain={len(chain)} segs, {n_loss:,} it; probes "
                f"{it_a[0]:,.0f}-{it_a[-1]:,.0f} ({'whole chain' if span_ok else '!! NOT whole chain'})"
                f"; floor={floor:.3e} end/floor={end_med / floor:.3f}")
        if res["reason"] is not None:
            dropped.append((r["name"], res["reason"]))
            print(line + f"\n      DROPPED: {res['reason']}; fitted curve reaches the bar at "
                         f"{extrapolated_crossing(res):,.0f}")
            continue
        T = res["t_star"]
        k = int(np.argmin(np.abs(it_a - T)))
        flag = "" if abs(it_a[k] - T) <= 1000 else "  !! nearest probe > 1,000 it away"
        if not (0.5 * end_med < floor <= end_med):
            flag += "  !! floor outside (end/2, end]"
        counts.append(float(r["active"][k]))
        crossings.append(T)
        print(line + f"\n      crossing {T:,.0f}, probe {it_a[k]:,.0f}: active {r['active'][k]:.0f} "
                     f"(end of chain: {r['active'][-1]:.0f}){flag}")
    return counts, crossings, dropped


def main():
    """Read every size of the chosen SCALING series at the floor and print the summary and fit."""
    task = sys.argv[1] if len(sys.argv) > 1 else "3-bit flip-flop"
    Ns, pats, _budget, _col = SCALING[task]
    summary = {}
    for N in Ns:
        print(f"\n=== {task} N = {N}: {pats[N]}")
        if not os.path.isdir(pats[N]):
            print("  no cell on disk")
            continue
        summary[N] = read_cell(N, pats[N])
    print("\n--- summary (active units at 1.05x own floor) ---")
    xs, ys = [], []
    for N, (c, t, d) in summary.items():
        if not c:
            print(f"N={N}: 0 read, {len(d)} dropped")
            continue
        sd = np.std(c, ddof=1) if len(c) > 1 else 0.0
        print(f"N={N}: {len(c)} read, {len(d)} dropped {[x[1] for x in d]}; crossings "
              f"{min(t):,.0f}-{max(t):,.0f}; active {np.mean(c):.1f} +- {sd:.1f} "
              f"(per run {[int(v) for v in c]})")
        xs.append(N)
        ys.append(np.mean(c))
    if len(xs) >= 2:
        a, _ = np.polyfit(np.log(xs), np.log(ys), 1)
        print(f"fit on cell means, N = {xs}: active ~ N^{a:.2f}")


if __name__ == "__main__":
    main()
