#!/usr/bin/env python3
"""
Confirmatory test of connectivity statistics on 39 unseen default networks of N = 1,000, and the test of
dead-relu-paper#172 (r12-weights, dead-relu-paper #127, item 52).

RULE. Notebook entry 2026-11-02_2218_connectivity-confirm-rule.md, committed (44cd160) before any of
these networks was computed. For each network, on its read weights W, its active set A
(population_stats.simulate) and its rebuilt initial weights W0:

  W1  s.d. of the recurrent weights over W[S, S]
  W3  median normalised in-degree, W4 median normalised out-degree, at theta = 2 x s.d.(W0)

rel = (S_all - S_act) / S_act. A statistic differs if, in all 39 networks, |rel| > 0.10 and S_act lies
outside [min, max] of S_R over 200 random subsets R of |A| units (SeedSequence(20261103).spawn(39), one
stream per network in the rule's order). Checks: T (rebuilt W0 row norms against the first run's stored
iteration-0 row norms, max relative difference < 1e-4) and M1 (active count within 10% of the trace's
count nearest the read iteration; a failure stops the run). Amendment 2026-11-02_2223 (cd92893): for
a CDDM network whose first-run trace has no row norms, the twin is also valid if check T' passes (the
twin's noise-free participation on the population_stats batch against the first run's stored
iteration-0 participation, max relative difference < 1e-3).

#172. Slope b = sum(W W0) / sum(W0^2) over the inactive-inactive block of the final weights: weight
decay 0 (jobs 40-45) against weight decay 1e-6 (networks 11-13); DMTS weight decay 0 (46-48) secondary.

Inputs: data/connectivity_confirm/<id>/ (copied from Della by the rule's fetch step; each folder's
source.txt holds the Della paths).
Usage:  python connectivity_confirm.py
Output: data/connectivity_confirm.json (gitignored) and the printed tables.
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import population_stats as PS            # noqa: E402  (imports sklearn before torch; see that file)
import ablate_silent_units as A           # noqa: E402
import connectivity_stats as C            # noqa: E402
from gradient_participation import initial_params    # noqa: E402
import torch                              # noqa: E402  (after population_stats: sklearn must load first)
from omegaconf import OmegaConf           # noqa: E402

DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data"))
NETS = os.path.join(DATA, "connectivity_confirm")
OUT = os.path.join(DATA, "connectivity_confirm.json")

STATS = ("W1", "W3", "W4")
N_MAIN = 39
DIFFERS = 0.10
SUBSETS = 200
SUBSET_SEED = 20261103
TWIN_TOL = 1e-4                     # check T: max |row norm difference| / mean row norm
TWIN_P_TOL = 1e-3                   # check T' (CDDM without row norms): max |participation difference| / mean
M1_TOL = 0.10                       # check M1: |simulate count - trace count| / trace count
COMPARISON = ("11_", "12_", "13_")  # #172: weight decay 1e-6, CDDM_std_g0_drift_bias0
SUPPORT_HI, SUPPORT_LO, FALSIFY = 0.9, 0.1, 0.5


def folders():
    """The network folders in the rule's order: (id, folder, job dict from source.txt)."""
    out = []
    for d in sorted(glob.glob(os.path.join(NETS, "*_*_*"))):
        with open(os.path.join(d, "source.txt")) as fh:
            out.append((os.path.basename(d), d, json.load(fh)))
    return out


def twin_participation(d):
    """Check T' input: the rebuilt twin's noise-free participation on the population_stats batch.

    Args:
        d: network folder (link *_config.yaml for the architecture and batch, root.yaml for the draw).
    Returns:
        (N,) float64 participation p = std(r) + q0.9(|r|) of the untrained twin.
    """
    params = {k: np.asarray(v) for k, v in initial_params(OmegaConf.load(os.path.join(d, "root.yaml"))).items()}
    cfg, rnn = A.build(d, params)
    inp = PS.batch(cfg)[0]
    with torch.no_grad():
        states = rnn(inp.to(rnn.device), w_noise=False)[0]
        return A.participation(states).cpu().numpy().astype(np.float64)


def load_network(d):
    """Read weights, initial weights and the check-T inputs of one network folder.

    Args:
        d: a folder of data/connectivity_confirm.
    Returns:
        dict params (path), W (N, N) float64, W0 (N, N) float64, twin_rel (float, check T; nan when
        the first-run trace has no row norms), twin_p_rel (float, check T'; nan where not run),
        twin_ok (bool), trace (dict from trace.json).
    """
    params = sorted(glob.glob(os.path.join(d, "*LastParams*")))[0]
    W = np.asarray(A.load_params(params)["W_rec"], dtype=np.float64)
    W0 = C.rebuild_initial(os.path.join(d, "root.yaml"))
    with open(os.path.join(d, "trace.json")) as fh:
        tr = json.load(fh)
    if tr["root_has_rownorm"]:       # older first-run traces store no row norms: check T cannot run, so it fails
        stored = np.load(os.path.join(d, "root_rownorm0.npy"))
        twin_rel = float(np.abs(np.linalg.norm(W0, axis=1) - stored).max() / stored.mean())
    else:
        twin_rel = float("nan")
    twin_ok = bool(twin_rel < TWIN_TOL) and tr["root_first_probe_iter"] == 0
    twin_p_rel = float("nan")
    with open(os.path.join(d, "source.txt")) as fh:
        task = json.load(fh)["task"]
    if not tr["root_has_rownorm"] and task == "CDDM" and tr["root_first_probe_iter"] == 0:
        stored_p = np.load(os.path.join(d, "root_partic0.npy"))
        twin_p_rel = float(np.abs(twin_participation(d) - stored_p).max() / stored_p.mean())
        twin_ok = twin_p_rel < TWIN_P_TOL
    return dict(params=params, W=W, W0=W0, twin_rel=twin_rel, twin_p_rel=twin_p_rel, twin_ok=bool(twin_ok),
                trace=tr)


def active_set(d, params):
    """Active units of the read weights: population_stats.simulate with the folder's link config.

    Args:
        d: network folder (holds the link's *_config.yaml); params: the read-weights file.
    Returns:
        (k,) int indices of the active units.
    """
    sim = PS.simulate(dict(dir=d, params=params))
    return np.flatnonzero(sim["act"])


def main_test(idx, d, job, rng):
    """All values of the main test for one network.

    Args:
        idx: network id; d: folder; job: its source.txt dict; rng: numpy Generator for the subsets.
    Returns:
        dict with the checks, theta, all / act values, rel, subset min / max, and the per-statistic
        pass of conditions 1 and 2.
    """
    net = load_network(d)
    W = net["W"]
    N = W.shape[0]
    act = active_set(d, net["params"])
    tc = net["trace"]["trace_active"]
    m1 = abs(len(act) - tc) / tc
    if m1 > M1_TOL:
        raise SystemExit(f"FATAL M1: {idx} has {len(act)} active units, its trace {tc} ({m1:.1%})")
    theta = C.THETA_MULT * net["W0"].std() if net["twin_ok"] else C.THETA_MULT * C.SPECTRAL_RAD / np.sqrt(N)
    s_all = C.statistics(W, None, np.arange(N), theta)
    s_act = C.statistics(W, None, act, theta)
    sub = {s: [] for s in STATS}
    for _ in range(SUBSETS):
        R = np.sort(rng.choice(N, size=len(act), replace=False))
        sr = C.statistics(W, None, R, theta)
        for s in STATS:
            sub[s].append(sr[s])
    rec = dict(id=idx, task=job["task"], seed=job["seed"], cell=job["cell"], read_iter=job["read_iter"],
               t_star=job["t_star"], N=N, n_active=int(len(act)), active=act.tolist(), trace_active=tc,
               trace_probe=net["trace"]["probe_iter"], M1_rel=float(m1), twin_rel=net["twin_rel"],
               twin_p_rel=net["twin_p_rel"],
               twin_ok=net["twin_ok"], theta=float(theta), all={}, act={}, rel={}, sub_min={}, sub_max={},
               cond1={}, cond2={})
    for s in STATS:
        r = C.rel(s_all[s], s_act[s])
        lo, hi = float(min(sub[s])), float(max(sub[s]))
        rec["all"][s], rec["act"][s], rec["rel"][s] = s_all[s], s_act[s], r
        rec["sub_min"][s], rec["sub_max"][s] = lo, hi
        rec["cond1"][s] = bool(np.isfinite(r) and abs(r) > DIFFERS)
        rec["cond2"][s] = bool(s_act[s] > hi or s_act[s] < lo)
    return rec


def slope_test(idx, d, job):
    """The #172 statistic on one network's read (final) weights.

    Returns:
        dict id, task, seed, read_iter, n_active, twin_ok, twin_rel, slope (b on W[I, I]), sd_ratio
        (s.d. of W[I, I] over s.d. of W0[I, I]).
    """
    net = load_network(d)
    act = active_set(d, net["params"])
    N = net["W"].shape[0]
    I = np.setdiff1d(np.arange(N), act)
    w, w0 = net["W"][np.ix_(I, I)].ravel(), net["W0"][np.ix_(I, I)].ravel()
    return dict(id=idx, task=job["task"], seed=job["seed"], read_iter=job["read_iter"], n_active=int(len(act)),
                twin_ok=net["twin_ok"], twin_rel=net["twin_rel"], twin_p_rel=net["twin_p_rel"], slope=float(w @ w0 / (w0 @ w0)),
                sd_ratio=float(w.std() / w0.std()))


def main():
    """Run both tests, print the tables and write OUT."""
    nets = folders()
    main_nets = [n for n in nets if n[2]["group"] == "main"]
    assert len(main_nets) == N_MAIN, len(main_nets)
    streams = np.random.SeedSequence(SUBSET_SEED).spawn(N_MAIN)
    res = []
    for (idx, d, job), ss in zip(main_nets, streams):
        r = main_test(idx, d, job, np.random.default_rng(ss))
        print(f"  {idx}: active {r['n_active']} (trace {r['trace_active']}), twin T {r['twin_rel']:.1e} "
              f"T' {r['twin_p_rel']:.1e} valid {r['twin_ok']}", flush=True)
        res.append(r)
    verdict = {}
    for s in STATS:
        c1 = sum(r["cond1"][s] for r in res)
        c2 = sum(r["cond2"][s] for r in res)
        both = sum(r["cond1"][s] and r["cond2"][s] for r in res)
        signs = {t: "".join(sorted({"+" if r["rel"][s] > 0 else "-" for r in res if r["task"] == t}))
                 for t in ("3BFF", "CDDM", "DMTS")}
        verdict[s] = dict(differs=both == N_MAIN, n_cond1=c1, n_cond2=c2, n_both=both, signs=signs)
    print(f"\n{'network':<22}{'read':>8}{'act':>5}{'trace':>6}{'twin ok':>8}" +
          "".join(f"{s + ' all':>9}{s + ' act':>9}{'rel':>7}{'sub range':>18}" for s in STATS))
    for r in res:
        print(f"{r['id']:<22}{r['read_iter']:>8}{r['n_active']:>5}{r['trace_active']:>6}{str(r['twin_ok']):>8}" +
              "".join(f"{r['all'][s]:>9.4f}{r['act'][s]:>9.4f}{r['rel'][s]:>7.2f}"
                      f"{r['sub_min'][s]:>9.4f}-{r['sub_max'][s]:<8.4f}" for s in STATS))
    for s in STATS:
        print(f"{s}: differs {verdict[s]['differs']} (|rel| > 0.10 in {verdict[s]['n_cond1']}/39, outside "
              f"subset range in {verdict[s]['n_cond2']}/39, both in {verdict[s]['n_both']}/39, signs {verdict[s]['signs']})")

    # #172
    cmp_ids = [n for n in main_nets if n[0].startswith(COMPARISON)]
    wd0 = [n for n in nets if n[2]["group"] == "wd0"]
    slopes = [dict(slope_test(*n), arm="wd 1e-6 (comparison)") for n in cmp_ids]
    slopes += [dict(slope_test(*n), arm="wd 0" + (" (secondary)" if n[2]["task"] == "DMTS" else "")) for n in wd0]
    print(f"\n#172: {'network':<22}{'arm':<24}{'read':>8}{'act':>5}{'twin ok':>8}{'slope':>8}{'sd ratio':>9}")
    for x in slopes:
        print(f"      {x['id']:<22}{x['arm']:<24}{x['read_iter']:>8}{x['n_active']:>5}{str(x['twin_ok']):>8}"
              f"{x['slope']:>8.3f}{x['sd_ratio']:>9.3f}")
    prim = [x for x in slopes if x["arm"] == "wd 0" and x["twin_ok"]]
    comp = [x for x in slopes if x["arm"].startswith("wd 1e-6") and x["twin_ok"]]
    if len(prim) == 6 and all(x["slope"] > SUPPORT_HI for x in prim) and len(comp) == 3 and \
            all(x["slope"] < SUPPORT_LO for x in comp):
        outcome = "supported"
    elif prim and all(x["slope"] < FALSIFY for x in prim) and len(prim) == 6:
        outcome = "falsified"
    else:
        outcome = "partial"
    print(f"#172 outcome: {outcome} ({len(prim)} weight-decay-0 networks with a valid twin, {len(comp)} comparison)")
    with open(OUT, "w") as fh:
        json.dump(dict(results=res, verdict=verdict, slopes=slopes, outcome_172=outcome), fh, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
