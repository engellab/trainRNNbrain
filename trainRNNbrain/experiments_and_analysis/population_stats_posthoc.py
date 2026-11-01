#!/usr/bin/env python3
"""
POST-HOC checks of the population statistics (dead-relu-paper PR #66 review), defined in notebook
entry 2026-11-01_0739_population-stats-posthoc-plan.md after the pre-registered results were seen.
They do not change the pre-registered labels of population_stats.py; they say what produces them.

  P1  S2 with a soft normaliser: rates / (sd + c), c = f x median sd of the active units
  P2  what each class of the all-unit S4 run holds (size, inactive units, selective units, centroid)
  P3  S4 on selective units only
  P4  S4 at fixed k = 3 and k = 4 in both runs
  P5  size of the inactive selective units' responses, and their count under added noise

Usage:  python population_stats_posthoc.py
Output: data/population_stats_posthoc.json (gitignored) and a printed summary per network.
"""

import json
import os
import sys

import numpy as np
# sklearn before torch (population_stats imports torch): see the note in population_stats.py
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import population_stats as P                                           # noqa: E402

OUT = os.path.join(P.DATA, "population_stats_posthoc.json")
SOFT_FRACTIONS = (0.0, 0.01, 0.05, 0.10)    # c / median active-unit sd; 0 is the rule's S2
FIXED_K = (3, 4)
NOISE_FRACTIONS = (0.01, 0.05)              # added noise sd / median active-unit response sd


def soft_pr(C, var, keep, c):
    """Participation ratio of the covariance of rates divided by (sd + c), over the units in `keep`.

    Args:
        C: (N, N) rate covariance; var: (N,) variances; keep: (N,) bool; c: soft-normaliser constant.
    Returns:
        float PR.
    """
    s = np.sqrt(var[keep]) + c
    return P.pr_from_cov(C[np.ix_(keep, keep)] / np.outer(s, s))


def fixed_k(profiles, k):
    """k-means with k fixed (same restarts and seed as the rule), in kmeans_classes' format.

    Args:
        profiles: (n, p) profiles; k: number of classes.
    Returns:
        dict k_star, centroids, shares.
    """
    km = KMeans(n_clusters=k, n_init=P.KMEANS_RESTARTS, random_state=P.POP_SEED).fit(profiles)
    return dict(k_star=k, centroids=km.cluster_centers_,
                shares=np.bincount(km.labels_, minlength=k) / len(km.labels_))


def posthoc(net):
    """P1-P5 for one network of the manifest.

    Returns:
        JSON-ready dict of every post-hoc quantity.
    """
    sim = P.simulate(net)
    rates, act, Y, X = sim["rates"], sim["act"], sim["Y"], sim["X"]
    out = dict(task=P.TASK_LABEL[sim["name"]], seed=net["seed"], n_active=int(act.sum()))

    # P1: soft normalisation
    C, var = P.covariance(rates)
    nz = var > 0
    med_sd = float(np.median(np.sqrt(var[act])))
    out["P1"] = {}
    for f in SOFT_FRACTIONS:
        c = f * med_sd
        a, b = soft_pr(C, var, nz, c), soft_pr(C, var, act, c)
        out["P1"][str(f)] = dict(all=a, act=b, rel=(a - b) / b)

    # profiles, exactly as population_statistics builds them
    pr2 = P.partial_r2(Y, X)
    sel, _mix = P.selectivity(pr2)
    tot = np.nansum(pr2, axis=1)
    ok = ~np.isnan(pr2).any(axis=1) & (tot > 0)
    prof = np.zeros_like(pr2)
    prof[ok] = pr2[ok] / tot[ok, None]

    # P2: classes of the all-unit run
    k_all = P.kmeans_classes(prof[ok])
    idx = np.where(ok)[0]
    classes = []
    for j in range(k_all["k_star"]):
        members = idx[k_all["labels"] == j]
        classes.append(dict(size=int(len(members)), inactive=int((~act[members]).sum()),
                            selective=int(sel[members].sum()),
                            centroid=np.round(k_all["centroids"][j], 3).tolist()))
    out["P2"] = dict(k_all=k_all["k_star"], classes=classes,
                     median_sum_pr2_inactive=float(np.median(tot[ok & ~act])) if (ok & ~act).any() else None,
                     median_sum_pr2_active=float(np.median(tot[ok & act])))

    # P3: selective units only
    s_all = P.kmeans_classes(prof[ok & sel])
    s_act = P.kmeans_classes(prof[ok & sel & act])
    out["P3"] = dict(k_all=s_all["k_star"], k_act=s_act["k_star"], n_all=int((ok & sel).sum()),
                     n_act=int((ok & sel & act).sum()),
                     max_diff_pp=P.match_classes(s_all, s_act)["max_diff_pp"])

    # P4: fixed k
    out["P4"] = {str(k): P.match_classes(fixed_k(prof[ok], k), fixed_k(prof[ok & act], k))["max_diff_pp"]
                 for k in FIXED_K}

    # P5: response size of selective units, and their count under added noise
    rsd = Y.astype(np.float64).std(axis=1)
    med_resp = float(np.median(rsd[act]))
    out["P5"] = dict(median_resp_sd_inactive_selective=float(np.median(rsd[sel & ~act])) if (sel & ~act).any() else None,
                     median_resp_sd_active_selective=float(np.median(rsd[sel & act])),
                     n_inactive_selective=int((sel & ~act).sum()), noise={})
    rng = np.random.default_rng(P.POP_SEED)
    for f in NOISE_FRACTIONS:
        Yn = Y.astype(np.float64) + rng.normal(0.0, f * med_resp, size=Y.shape)
        sel_n, _ = P.selectivity(P.partial_r2(Yn, X))
        out["P5"]["noise"][str(f)] = dict(inactive_selective=int((sel_n & ~act).sum()),
                                          active_selective=int((sel_n & act).sum()))
    return out


def report(r):
    """Print one network's post-hoc results on a few lines."""
    p1 = "  ".join(f"c={f}: {v['all']:.1f}/{v['act']:.1f} ({100 * v['rel']:+.0f}%)" for f, v in r["P1"].items())
    print(f"{r['task']} {r['seed']} (active {r['n_active']})")
    print(f"  P1 S2 all/act  {p1}")
    print(f"  P2 k*={r['P2']['k_all']}, median summed partial R2 inactive "
          f"{r['P2']['median_sum_pr2_inactive']}, active {r['P2']['median_sum_pr2_active']:.3f}")
    for c in r["P2"]["classes"]:
        print(f"     class: {c['size']} units, {c['inactive']} inactive, {c['selective']} selective, centroid {c['centroid']}")
    print(f"  P3 selective only: k* {r['P3']['k_all']}/{r['P3']['k_act']} over {r['P3']['n_all']}/{r['P3']['n_act']} units, "
          f"matched diff {r['P3']['max_diff_pp']:.1f} pp")
    print(f"  P4 fixed k: " + ", ".join(f"k={k}: {v:.1f} pp" for k, v in r["P4"].items()))
    p5 = r["P5"]
    print(f"  P5 inactive selective {p5['n_inactive_selective']}, median response sd {p5['median_resp_sd_inactive_selective']} "
          f"vs active selective {p5['median_resp_sd_active_selective']:.4f}; with noise "
          + ", ".join(f"{f}: {v['inactive_selective']} inactive / {v['active_selective']} active" for f, v in p5["noise"].items()))


def main():
    """Run P1-P5 on the nine networks, print and write OUT."""
    res = []
    for net in P.manifest():
        r = posthoc(net)
        report(r)
        res.append(r)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1, default=P._jsonable)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
