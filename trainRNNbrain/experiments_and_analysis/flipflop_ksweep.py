#!/usr/bin/env python3
"""
Does silencing depend on task complexity? The n-bit flip-flop k-sweep.

CDDM alone cannot separate the paper's claim from its deflationary reading: a task-set ceiling of
~880 active units is exactly what "the task is easy and the rest is spare capacity" predicts. The
flip-flop makes complexity a dial - k bits demand 2^k stable states and a readout of dimension k -
so silencing can be measured AS A FUNCTION of task demand rather than at a single point.

TWO READINGS ARE REPORTED, AND BOTH MUST AGREE BEFORE ANY ORDERING IN k IS BELIEVED.

  endpoint    every cell at the same 300k iterations. Confounded: the cells reach very different
              performance (clean loss 0.0037 to 0.021), and silencing tracks training depth, so a
              cell that trained further will silence more for reasons unrelated to k.
  matched     every cell read at the iteration where its smoothed noise-free loss stably crosses a
              common level L*. Comparable across k WITHOUT any floor fitting, because the flip-flop's
              target variance is k-independent (0.727-0.738 measured over k=2..6, a 1.5% spread):
              each bit is generated i.i.d. and the loss averages over channels, so MSE is already
              per-channel-normalised and R^2 = 1 - MSE/0.735 means the same thing at every k.

If the k-ordering is the same under both, it does not matter which is "correct" and the cross-k
comparison needs no further defence. If they disagree, that is the finding and it gets reported.

L* must be reachable by the WORST cell in the grid, which drags the common level shallow - the
regime where CDDM's saturation verdict was weakest. Levels are therefore swept, not fixed.

Losses come from `loss_clean_train`: the NOISE-FREE task loss, recorded at every probe. Not
TrainLosses.json, which is the noisy optimiser objective and has inverted a conclusion twice here.

Output: img/internal_figures/flipflop_ksweep.png

Usage:  python flipflop_ksweep.py [L* ...]
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_drift_curves import IMG_DIR
from plot_M_vs_N import active_count

SWEEP = "data/trained_RNNs/NBitFlipFlop_std_ksweep"
PROBE_EVERY = 10       # trainer.track_every; loss_clean_train is indexed in probes, not iterations
WINDOW = 201           # probes -> 2010 iterations, matching the CDDM protocol's 2001-iteration window
TGT_VAR = 0.735        # target variance, k-independent; only used to convert loss to R^2 for display


def load():
    """Load every flip-flop trace, dropping runs that diverged.

    A run is diverged if its noise-free loss trace contains any NaN or its final participation
    vector has no active unit at all. One of the 45 runs (k=2, N=2000) diverged despite carrying no
    penalty, so this is plain training instability rather than the penalty-driven self-excitation
    mode documented elsewhere in the project.

    Returns:
        (dict {(k, N): [trace dicts]}, list of dropped (k, N, seed) tuples). Each trace gains a
        "clean" key holding the noise-free loss as a float array indexed by probe.
    """
    out, dropped = {}, []
    for f in sorted(glob.glob(os.path.join(SWEEP, "*", "*", "*ParticipationTrace.pkl"))):
        m = re.search(r"_k=(\d+)_N=(\d+)", f)
        if not m:
            continue
        k, N = int(m.group(1)), int(m.group(2))
        with open(f, "rb") as fh:
            tr = pickle.load(fh)
        clean = np.asarray(tr["metrics"].get("loss_clean_train", []), dtype=float)
        p_end = np.array(tr["participation"])[-1]
        if clean.size == 0 or np.isnan(clean).any() or (p_end >= 1e-6).sum() == 0:
            dropped.append((k, N, os.path.basename(os.path.dirname(f))[:10]))
            continue
        tr["clean"] = clean
        out.setdefault((k, N), []).append(tr)
    return out, dropped


def stable_crossing(loss, thr, window=WINDOW):
    """Last probe index at which the smoothed loss is still above `thr`.

    "Stably" rather than "first time below" because the loss is not monotone: a first-crossing rule
    fires on a transient dip. Returns None if the run never gets below thr, or if it is still above
    at the end (which would mean the crossing is not inside the run).

    Args:
        loss: noise-free loss per probe; thr: threshold; window: centred running-mean width in probes.
    Returns:
        probe index, or None.
    """
    h = window // 2
    s = np.convolve(loss, np.ones(window) / window, mode="valid")
    idx = np.arange(h, len(loss) - h)
    above = idx[s > thr]
    if not len(above) or above[-1] >= idx[-1]:
        return None
    return int(above[-1])


def silence_at(trace, N, probe):
    """Silent-unit percentages and active count at the participation probe nearest `probe`.

    Args:
        trace: a loaded trace; N: network size; probe: index into the loss (track_every) grid, or
            None for the endpoint.
    Returns:
        (hard %, scale-free %, active count under the hard criterion).
    """
    P = np.array(trace["participation"])
    I = np.array(trace["participation_iters"])
    p = P[-1] if probe is None else P[np.argmin(np.abs(I - probe * PROBE_EVERY))]
    a = active_count(p, "hard")
    return 100 * (N - a) / N, 100 * (N - active_count(p, "scalefree")) / N, a


def table(by, ks, Ns, thr, title):
    """Print one silence table and return {(k, N): (hard, sf, active, T)} of per-cell means.

    Args:
        by: loaded traces; ks, Ns: sorted axes; thr: L* level, or None for the endpoint reading;
        title: header line.
    Returns:
        dict of per-cell (mean hard %, mean sf %, mean active, mean read-out iteration).
    """
    print(f"\n=== {title} ===")
    print("%3s %6s %15s %15s %13s %11s" % ("k", "N", "silent hard %", "silent sf %", "ACTIVE", "T_read"))
    out = {}
    for k in ks:
        for N in Ns:
            rows = []
            for t in by.get((k, N), []):
                x = None if thr is None else stable_crossing(t["clean"], thr)
                if thr is not None and x is None:
                    continue
                h, s, a = silence_at(t, N, x)
                rows.append((h, s, a, len(t["clean"]) * PROBE_EVERY if x is None else x * PROBE_EVERY))
            if not rows:
                print("%3d %6d %15s" % (k, N, "  (unreachable)"))
                continue
            v = np.array(rows, dtype=float)
            out[(k, N)] = tuple(v.mean(axis=0))
            print("%3d %6d %8.1f +-%-4.1f %8.1f +-%-4.1f %7.0f +-%-4.0f %11.0f"
                  % (k, N, v[:, 0].mean(), v[:, 0].std(), v[:, 1].mean(), v[:, 1].std(),
                     v[:, 2].mean(), v[:, 2].std(), v[:, 3].mean()))
        print()
    return out


def main():
    """Report silencing vs task complexity at the endpoint and at matched performance."""
    levels = [float(a) for a in sys.argv[1:]] or [0.022, 0.015, 0.010]
    by, dropped = load()
    ks = sorted({k for k, _ in by})
    Ns = sorted({N for _, N in by})

    print(f"loaded {sum(len(v) for v in by.values())} runs, k={ks}, N={Ns}")
    if dropped:
        print(f"DROPPED {len(dropped)} diverged run(s) (NaN loss or zero active units):")
        for k, N, s in dropped:
            print(f"    k={k} N={N} seed {s}")
    fin = np.array([t["clean"][-1] for v in by.values() for t in v])
    print(f"endpoint clean loss across all runs: {fin.min():.5f} - {fin.max():.5f} "
          f"(R^2 {1 - fin.max() / TGT_VAR:.3f} - {1 - fin.min() / TGT_VAR:.3f})")
    print(f"a common L* must be >= {fin.max():.5f} to be reachable by every cell")

    res = {None: table(by, ks, Ns, None, "ENDPOINT: all cells at 300k iterations (CONFOUNDED)")}
    for L in levels:
        res[L] = table(by, ks, Ns, L,
                       f"MATCHED PERFORMANCE: L* = {L:.4f}  (R^2 = {1 - L / TGT_VAR:.3f})")

    fig, ax = plt.subplots(4, len(Ns), figsize=(6 * len(Ns), 18), squeeze=False)
    cols = plt.cm.viridis(np.linspace(0.1, 0.8, len(levels) + 1))
    for j, N in enumerate(Ns):
        for row, (idx, lab) in enumerate([(0, "hard  $p_i<10^{-6}$"), (1, "scale-free")]):
            for i, L in enumerate([None] + levels):
                xs = [k for k in ks if (k, N) in res[L]]
                if not xs:
                    continue
                ax[row][j].plot(xs, [res[L][(k, N)][idx] for k in xs], "o-", color=cols[i], lw=2,
                                ms=7, label="endpoint (300k)" if L is None else f"$L^*$={L:.3f}")
            ax[row][j].set(xlabel="task complexity $k$ (bits)", ylabel="silent units (% of N)",
                           title=f"N={N}  —  {lab}", ylim=(-3, 100), xticks=ks)
            ax[row][j].legend(fontsize=8)
            ax[row][j].grid(alpha=.3)

        # (c, d) the ABSOLUTE active count under each criterion. The percentage panels above answer
        # "what fraction is wasted"; these answer "how many units does the task actually recruit",
        # which is the quantity M* is about and the one comparable to CDDM's ~880 ceiling. Both
        # criteria get their own panel because they disagree materially here: the hard count reaches
        # M=N by k~4, while the scale-free count is still well short of N at k=6. "Non-zero" and
        # "doing work" are not the same claim, and the hard criterion is the flattering one.
        for row, (idx, crit) in enumerate([(2, "hard  $p_i<10^{-6}$"),
                                           (3, "scale-free  $p_i<0.05\\,q_{95}(p)$")]):
            a = ax[idx][j]
            for i, L in enumerate([None] + levels):
                xs = [k for k in ks if (k, N) in res[L]]
                if not xs:
                    continue
                # row 0 reads the stored hard active count; row 1 converts the scale-free percentage
                ys = ([res[L][(k, N)][2] for k in xs] if row == 0
                      else [N * (1 - res[L][(k, N)][1] / 100) for k in xs])
                a.plot(xs, ys, "o-", color=cols[i], lw=2, ms=7,
                       label="endpoint (300k)" if L is None else f"$L^*$={L:.3f}")
            a.axhline(N, color="k", ls="--", lw=1.2, alpha=.6, label=f"$M=N$ ({N})")
            a.set(xlabel="task complexity $k$ (bits)", ylabel="active units $M$",
                  title=f"N={N}  —  active count, {crit}", xticks=ks, ylim=(0, N * 1.12))
            a.legend(fontsize=8)
            a.grid(alpha=.3)
    fig.suptitle("Flip-flop: does silencing fall as the task gets harder?\n"
                 "agreement between endpoint and matched-performance readings is what licenses "
                 "any ordering in $k$", fontsize=12)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "flipflop_ksweep.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
