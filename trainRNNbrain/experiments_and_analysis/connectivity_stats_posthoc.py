#!/usr/bin/env python3
"""
Post-hoc diagnostics of connectivity_stats.py, written AFTER its results were seen (notebook entry
2026-11-02_2203_connectivity-stats-rule.md fixed the rule; the result entry names this script). Nothing
here changes the rule's verdict.

(1) Block structure of the trained W_rec, A active and I inactive units: the s.d. of W[A,A], W[A,I]
    (incoming to active units from inactive ones), W[I,A] and W[I,I] against the initial s.d., the
    regression slope of W[I,I] on its initial values W0[I,I] (1 = initial values kept, 0 = gone), and
    the fraction of W[I,I] with |w| < 1e-3.
(2) The size control. For 200 random subsets R of |A| units: the correlation between rel_R =
    (S_all - S_R) / S_R and the number m of active units that R contains, and how many subsets reach
    the active set's value S_act, for W1-W4.

Usage:  python connectivity_stats_posthoc.py   (reads data/connectivity_stats.json)
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import connectivity_stats as C            # noqa: E402

DRAWS = 200
SEED = 1                                  # a stream separate from the rule's size control


def block_sd(W, rows, cols):
    """s.d. of the block W[rows, cols]; rows are postsynaptic units, cols presynaptic units."""
    return float(W[np.ix_(rows, cols)].std())


def diagnose(x):
    """Both diagnostics for one network.

    Args:
        x: one record of data/connectivity_stats.json.
    Returns:
        dict sd0, sd_AA, sd_AI, sd_IA, sd_II, slope_II, near0_II, and per statistic W1-W4:
        corr (rel_R against m), n_reach (subsets with S_R >= S_act), r_max (largest S_R), m range.
    """
    d = os.path.join(C.DATA, "pop_stats_nets", x["label"])
    W = C.load_w(os.path.join(d, [f for f in os.listdir(d) if "LastParams" in f][0]))
    W0 = C.rebuild_initial(os.path.join(C.INIT, x["label"], "root_config.yaml"))
    N = W.shape[0]
    A = np.array(x["active"])
    I = np.setdiff1d(np.arange(N), A)
    wii, w0ii = W[np.ix_(I, I)].ravel(), W0[np.ix_(I, I)].ravel()
    out = dict(sd0=float(W0.std()), sd_AA=block_sd(W, A, A), sd_AI=block_sd(W, A, I),
               sd_IA=block_sd(W, I, A), sd_II=block_sd(W, I, I),
               slope_II=float(wii @ w0ii / (w0ii @ w0ii)), near0_II=float((np.abs(wii) < 1e-3).mean()))
    rng = np.random.default_rng(SEED)
    is_act = np.zeros(N, bool)
    is_act[A] = True
    s_all = C.statistics(W, None, np.arange(N), x["theta"])
    vals = {s: [] for s in ("W1", "W2", "W3", "W4")}
    ms = []
    for _ in range(DRAWS):
        R = np.sort(rng.choice(N, size=len(A), replace=False))
        ms.append(int(is_act[R].sum()))
        sr = C.statistics(W, None, R, x["theta"])
        for s in vals:
            vals[s].append(sr[s])
    ms = np.array(ms)
    for s, v in vals.items():
        v = np.array(v)
        relR = np.where(v != 0, (s_all[s] - v) / np.where(v != 0, v, 1), np.nan)
        ok = np.isfinite(relR)
        corr = float(np.corrcoef(relR[ok], ms[ok])[0, 1]) if ok.sum() > 2 and relR[ok].std() > 0 else float("nan")
        out[s] = dict(corr=corr, n_reach=int((v >= x["act"][s]).sum()), r_max=float(v.max()))
    out["m_range"] = [int(ms.min()), int(ms.max())]
    return out


def main():
    """Print both diagnostics for the nine networks."""
    with open(os.path.join(C.DATA, "connectivity_stats.json")) as fh:
        res = json.load(fh)["results"]
    print(f"{'network':<18}{'sd0':>7}{'AA':>7}{'A<-I':>7}{'I<-A':>7}{'I<-I':>7}{'slope II':>9}{'|w|<1e-3':>9}"
          f"{'m':>9}  per W1-W4: corr(rel_R, m) / subsets reaching S_act / max S_R vs S_act")
    for x in res:
        g = diagnose(x)
        stats = "  ".join(f"{s}: {g[s]['corr']:+.2f}/{g[s]['n_reach']}/{g[s]['r_max']:.4f} vs {x['act'][s]:.4f}"
                          for s in ("W1", "W2", "W3", "W4"))
        print(f"{x['task'] + ' ' + x['seed'][1:]:<18}{g['sd0']:>7.4f}{g['sd_AA']:>7.4f}{g['sd_AI']:>7.4f}"
              f"{g['sd_IA']:>7.4f}{g['sd_II']:>7.4f}{g['slope_II']:>9.3f}{g['near0_II']:>9.3f}"
              f"{g['m_range'][0]:>5}-{g['m_range'][1]:<3}  {stats}")


if __name__ == "__main__":
    main()
