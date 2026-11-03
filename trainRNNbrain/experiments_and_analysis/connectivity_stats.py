#!/usr/bin/env python3
"""
Connectivity statistics of the recurrent weights over all N units and over the active units only
(dead-relu-paper #127, round 12, package r12-weights; Pavel's item 52).

RULE. Notebook entry 2026-11-02_2203_connectivity-stats-rule.md, committed (4f25b25) before any
value was computed. For a set S of units (all units, or the active set A), on W[S, S]:

  W1  s.d. of the k^2 recurrent weights
  W2  fraction of the k^2 weights with |w| >= theta
  W3  median over units of the normalised in-degree  #{j in S: |W_ij| >= theta} / k
  W4  median over units of the normalised out-degree #{j in S: |W_ji| >= theta} / k
  W5  median over units of the RMS change from initialisation of the unit's incoming and outgoing
      weights to the units of S

theta = 2 x s.d. of the network's own rebuilt initial W_rec. rel = (S_all - S_act) / S_act. A statistic
differs if |rel| > 0.10 in all nine networks and, in all nine, the 95th percentile of
|(S_all - S_R) / S_R| over 200 random subsets R of |A| units is below 0.05.

NETWORKS. The nine networks of Fig. 1d/1e and Extended Data Fig. 2 (data/pop_stats_nets/manifest.tsv),
with the active set of Extended Data Fig. 2 (population_stats.simulate). Initial weights are rebuilt
from the first run of each warm-start chain (data/connectivity_init/<label>/root_config.yaml) and
checked against that run's kept snapshot at iteration 500 (same folder).

Usage:  python connectivity_stats.py
Output: data/connectivity_stats.json (gitignored) and the printed per-network table.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import population_stats as PS            # noqa: E402  (imports sklearn before torch; see that file)
from gradient_participation import initial_params    # noqa: E402
from omegaconf import OmegaConf           # noqa: E402

DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data"))
INIT = os.path.join(DATA, "connectivity_init")
OUT = os.path.join(DATA, "connectivity_stats.json")

STATS = ("W1", "W2", "W3", "W4", "W5")
THETA_MULT = 2.0                 # theta = THETA_MULT x s.d. of the initial W_rec
THETA_MULT_SECONDARY = (1.0, 3.0)  # reported for W2-W4, outside the pass condition
DIFFERS = 0.10                   # |rel| above this in every network
SIZE_DRAWS = 200                 # random subsets of |A| units per network
SIZE_LIMIT = 0.05                # 95th percentile of |rel| over the random subsets must be below this
SIZE_SEED = 20261102
M2_MIN_CORR, M2_MAX_WRONG = 0.3, 0.01   # rebuilt W0 vs iteration-500 snapshot; seed + 1 twin
SPECTRAL_RAD = 1.2               # fallback theta scale if M2 fails: THETA_MULT x 1.2 / sqrt(N)


def load_w(path):
    """W_rec of a parameter file as float64.

    Args:
        path: a *LastParams*.npz.
    Returns:
        (N, N) float64 array; row i holds unit i's incoming weights (the forward pass is W_rec @ r).
    """
    return np.load(path, allow_pickle=True)["W_rec"].astype(np.float64)


def rebuild_initial(cfg_path, seed_offset=0):
    """The untrained W_rec of a run, rebuilt from its config by gradient_participation.initial_params.

    Args:
        cfg_path: the first run's *_config.yaml; seed_offset: added to cfg.seed (0 for the real
            twin, 1 for the wrong-draw control of check M2).
    Returns:
        (N, N) float64 array.
    """
    cfg = OmegaConf.load(cfg_path)
    cfg.seed = int(cfg.seed) + seed_offset
    return np.asarray(initial_params(cfg)["W_rec"], dtype=np.float64)


def statistics(W, W0, idx, theta):
    """W1-W5 over the units `idx`, and W2-W4 at the secondary thresholds.

    Args:
        W: (N, N) trained W_rec; W0: (N, N) initial W_rec or None (then W5 is nan);
        idx: (k,) int indices of the set S; theta: float threshold for W2-W4.
    Returns:
        dict W1..W5 (floats), and "secondary": {multiple: {W2, W3, W4}} for theta scaled by
        multiple / THETA_MULT.
    """
    sub = W[np.ix_(idx, idx)]
    k = len(idx)

    def thresholded(th):
        """W2, W3, W4 of the submatrix at threshold th."""
        big = np.abs(sub) >= th
        return dict(W2=float(big.mean()),
                    W3=float(np.median(big.sum(axis=1) / k)),     # row i: incoming to unit i
                    W4=float(np.median(big.sum(axis=0) / k)))     # column i: outgoing from unit i

    out = dict(W1=float(sub.std()), **thresholded(theta))
    if W0 is None:
        out["W5"] = float("nan")
    else:
        d2 = (sub - W0[np.ix_(idx, idx)]) ** 2
        out["W5"] = float(np.median(np.sqrt((d2.sum(axis=1) + d2.sum(axis=0)) / (2 * k))))
    out["secondary"] = {str(m): thresholded(theta * m / THETA_MULT) for m in THETA_MULT_SECONDARY}
    return out


def rel(s_all, s_set):
    """(S_all - S_set) / S_set, the definition of population_stats.rel_diff (nan for a zero base)."""
    return (s_all - s_set) / s_set if s_set != 0 else float("nan")


def size_control(W, W0, k, theta, rng):
    """The 95th percentile of |rel| between all units and random subsets of k units, per statistic.

    Args:
        W, W0: trained and initial W_rec (W0 may be None); k: subset size (|A|); theta: threshold;
        rng: numpy Generator.
    Returns:
        dict statistic -> 95th percentile of |(S_all - S_R) / S_R| over SIZE_DRAWS subsets.
    """
    N = W.shape[0]
    s_all = statistics(W, W0, np.arange(N), theta)
    rels = {s: [] for s in STATS}
    for _ in range(SIZE_DRAWS):
        R = np.sort(rng.choice(N, size=k, replace=False))
        s_r = statistics(W, W0, R, theta)
        for s in STATS:
            rels[s].append(abs(rel(s_all[s], s_r[s])))
    return {s: float(np.nanpercentile(rels[s], 95)) if np.isfinite(rels[s]).any() else float("nan")
            for s in STATS}


def measure(net, stored, rng):
    """All checks and statistics for one network of the manifest.

    Args:
        net: a population_stats.manifest() row; stored: that network's record in
            data/population_stats.json; rng: numpy Generator for the size control.
    Returns:
        dict with label, task, seed, n_active, active (indices), checks M1, M2, M4, theta, sigma0,
        all / act statistics, rel, size-control percentiles.
    """
    sim = PS.simulate(net)
    act = np.flatnonzero(sim["act"])
    m1 = int(len(act)) == int(stored["n_active"])
    if not m1:
        raise SystemExit(f"FATAL M1: {net['label']} has {len(act)} active units, "
                         f"population_stats.json has {stored['n_active']}")
    W = load_w(net["params"])
    N = W.shape[0]
    d = os.path.join(INIT, net["label"])
    W0 = rebuild_initial(os.path.join(d, "root_config.yaml"))
    Wwrong = rebuild_initial(os.path.join(d, "root_config.yaml"), seed_offset=1)
    W500 = load_w(os.path.join(d, "it0000500_LastParams.npz"))
    c_true = float(np.corrcoef(W0.ravel(), W500.ravel())[0, 1])
    c_wrong = float(np.corrcoef(Wwrong.ravel(), W500.ravel())[0, 1])
    m2 = c_true > M2_MIN_CORR and abs(c_wrong) < M2_MAX_WRONG
    sigma0 = float(W0.std())
    if m2:
        theta = THETA_MULT * sigma0
    else:
        theta = THETA_MULT * SPECTRAL_RAD / np.sqrt(N)
        W0 = None
    everyone = np.arange(N)
    s_all = statistics(W, W0, everyone, theta)
    s_act = statistics(W, W0, act, theta)
    r = {s: rel(s_all[s], s_act[s]) for s in STATS}
    r_sec = {m: {s: rel(s_all["secondary"][m][s], s_act["secondary"][m][s]) for s in ("W2", "W3", "W4")}
             for m in s_all["secondary"]}
    size = size_control(W, W0, len(act), theta, rng)
    # M4: the same comparison on the initial weights (W5 is identically 0 there and is skipped)
    m4 = None
    if W0 is not None:
        i_all, i_act = statistics(W0, None, everyone, theta), statistics(W0, None, act, theta)
        m4 = {s: rel(i_all[s], i_act[s]) for s in ("W1", "W2", "W3", "W4")}
    return dict(label=net["label"], task=PS.TASK_LABEL[sim["name"]], seed=net["seed"],
                iteration=net["iteration"], N=N, n_active=int(len(act)), active=act.tolist(),
                M1=m1, M2=dict(passed=bool(m2), corr_true=c_true, corr_wrong=c_wrong),
                sigma0=sigma0, theta=float(theta), all=s_all, act=s_act, rel=r, rel_secondary=r_sec,
                size_p95=size, M4=m4)


def verdict(results):
    """Apply the pass condition to every statistic.

    Args:
        results: list of measure() dicts.
    Returns:
        dict statistic -> dict differs (bool), n_over (networks with |rel| > DIFFERS),
        n_size_ok (networks passing the size control), signs (per task, the set of signs of rel).
    """
    out = {}
    for s in STATS:
        over = [np.isfinite(x["rel"][s]) and abs(x["rel"][s]) > DIFFERS for x in results]
        size_ok = [np.isfinite(x["size_p95"][s]) and x["size_p95"][s] < SIZE_LIMIT for x in results]
        signs = {}
        for x in results:
            signs.setdefault(x["task"], set()).add("+" if x["rel"][s] > 0 else "-")
        out[s] = dict(differs=bool(all(over) and all(size_ok)), n_over=int(sum(over)),
                      n_size_ok=int(sum(size_ok)), signs={t: "".join(sorted(v)) for t, v in signs.items()})
    return out


def table(results, v):
    """Print the per-network values, the checks and the verdict."""
    print(f"\n{'network':<26}{'act':>5}{'M2 r':>7}{'M2 wrong':>10}{'theta':>8}")
    for x in results:
        print(f"{x['task'] + ' ' + x['seed']:<26}{x['n_active']:>5}{x['M2']['corr_true']:>7.3f}"
              f"{x['M2']['corr_wrong']:>10.4f}{x['theta']:>8.4f}")
    for s in STATS:
        print(f"\n{s}: {'network':<24}{'all':>10}{'active':>10}{'rel':>8}{'size p95':>10}{'M4 rel':>8}")
        for x in results:
            m4 = x["M4"][s] if x["M4"] and s in x["M4"] else float("nan")
            print(f"    {x['task'] + ' ' + x['seed']:<24}{x['all'][s]:>10.5f}{x['act'][s]:>10.5f}"
                  f"{x['rel'][s]:>8.3f}{x['size_p95'][s]:>10.4f}{m4:>8.3f}")
        print(f"    -> differs: {v[s]['differs']} (|rel| > {DIFFERS} in {v[s]['n_over']}/9, size control "
              f"ok in {v[s]['n_size_ok']}/9, signs {v[s]['signs']})")
    print("\nsecondary thresholds (rel of W2, W3, W4):")
    for x in results:
        print(f"    {x['task'] + ' ' + x['seed']:<24}" + "  ".join(
            f"{m}sd: " + " ".join(f"{x['rel_secondary'][m][s]:.3f}" for s in ("W2", "W3", "W4"))
            for m in x["rel_secondary"]))


def main():
    """Measure the nine networks, print the table and write OUT."""
    with open(PS.OUT) as fh:
        stored = {r["label"]: r for r in json.load(fh)["results"]}
    nets = PS.manifest()
    streams = np.random.SeedSequence(SIZE_SEED).spawn(len(nets))   # one stream per network, manifest order
    results = []
    for net, ss in zip(nets, streams):
        x = measure(net, stored[net["label"]], np.random.default_rng(ss))
        print(f"  {x['label']}: active {x['n_active']} (M1 ok), M2 {x['M2']['passed']}", flush=True)
        results.append(x)
    order = {t: i for i, t in enumerate(PS.TASK_LABEL.values())}
    results.sort(key=lambda x: (order[x["task"]], x["seed"]))
    v = verdict(results)
    table(results, v)
    with open(OUT, "w") as fh:
        json.dump(dict(results=results, verdict=v), fh, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
