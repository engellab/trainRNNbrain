#!/usr/bin/env python3
"""
Population statistics over all N units and over the active units only (dead-relu-paper #34, #56).

QUESTION. Does counting the inactive units change the population statistics an analysis computes
over a network? The rule (notebook entry 2026-10-31_2212_population-stats-rule.md, fixed before any
value was computed) names four statistics, each computed over all units and over the active units:

  S1  participation-ratio dimensionality PR = (sum l)^2 / sum l^2 of the rate covariance
  S2  the same on rates divided by their standard deviation (zero-variance units dropped)
  S3  selective and mixed-selective units: partial R^2 >= 0.10 against the task variables
  S4  k-means functional classes on partial-R^2 profiles, k* in 2..8 by mean silhouette

and labels each statistic per task "changes", "unchanged" or "mixed" from the three networks.

NETWORKS. The nine N = 1000 default networks of Fig. 1d, at the weights the ablation used (the
snapshot at each network's floor crossing), copied from Della into data/pop_stats_nets/ with a
manifest.tsv (label, run_dir, params_file, iteration, active count in the ablation file).

Usage:
  python population_stats.py oracle      the oracle tests V1-V4 of the rule entry, nothing else
  python population_stats.py run         the oracle tests, then the nine networks; writes OUT
Output: data/population_stats.json (gitignored) and the printed per-network table.
"""

import argparse
import json
import os
import sys

import numpy as np
# sklearn before torch: the other order loads two OpenMP runtimes on the Mac and KMeans aborts
# ("OMP: Error #179: Function pthread_mutex_init failed")
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import hydra

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ablate_silent_units as A                                        # noqa: E402
from common import active_mask                                         # noqa: E402
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask  # noqa: E402
from trainRNNbrain.trainer.Trainer import Trainer                      # noqa: E402

DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data"))
NETS = os.path.join(DATA, "pop_stats_nets")
OUT = os.path.join(DATA, "population_stats.json")

POP_SEED = 20261031          # task generator, global numpy/torch seeds, k-means
SEL_THR = 0.10               # partial R^2 at or above this: selective for that variable
K_RANGE = range(2, 9)        # k-means cluster counts tried
KMEANS_RESTARTS = 20
CHANGES, UNCHANGED = 0.10, 0.02          # relative-difference limits of the decision rule
S4_CHANGES_PP, S4_UNCHANGED_PP = 10.0, 2.0   # matched-share limits for S4, percentage points
N_TRIALS = 256               # DMTS (64 per sample-test pair) and 3BFF trials
CDDM_EPOCH = slice(100, 200)  # stimulus onset to decision onset
DMTS_EPOCH = slice(110, 130)  # test-stimulus offset to decision onset
TASK_LABEL = {"NBitFlipFlop": "3BFF", "CDDM": "CDDM", "DMTS": "DMTS"}


# ------------------------------------------------------------------------------------------------
# the statistics
# ------------------------------------------------------------------------------------------------

def pr_from_cov(C):
    """Participation ratio (sum l)^2 / sum l^2 of a covariance matrix, from its eigenvalues.

    Args:
        C: (n, n) symmetric positive semi-definite matrix.
    Returns:
        float PR; nan for an empty or all-zero matrix.
    """
    if C.shape[0] == 0:
        return float("nan")
    lam = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.sum() > 0 else float("nan")


def pr_from_trace(C):
    """Participation ratio as tr(C)^2 / tr(C^2) = tr(C)^2 / ||C||_F^2, without eigenvalues.

    Args:
        C: (n, n) symmetric matrix.
    Returns:
        float PR.
    """
    return float(np.trace(C) ** 2 / (C * C).sum())


def covariance(rates, chunk=32):
    """Covariance of unit rates over every step and trial, accumulated in float64 trial chunks.

    Args:
        rates: (N, T, B) array (float32 or float64) of rates.
        chunk: trials per accumulation step (bounds memory to N x T x chunk doubles).
    Returns:
        (N, N) float64 covariance (normalised by the sample count), and (N,) float64 variances.
    """
    N, T, B = rates.shape
    S = np.zeros((N, N))
    m = np.zeros(N)
    n = 0
    for b in range(0, B, chunk):
        X = rates[:, :, b:b + chunk].reshape(N, -1).astype(np.float64)
        S += X @ X.T
        m += X.sum(axis=1)
        n += X.shape[1]
    m /= n
    C = S / n - np.outer(m, m)
    C = 0.5 * (C + C.T)
    # exact zero-variance units (rate identically constant, in practice identically 0) get exact
    # zeros, so rounding in the accumulation cannot give them a tiny variance
    flat = rates.reshape(N, -1)
    zero = flat.max(axis=1) == flat.min(axis=1)
    C[zero, :] = 0.0
    C[:, zero] = 0.0
    return C, np.diag(C).copy()


def correlation_submatrix(C, var, keep):
    """The covariance of rates divided by their standard deviation, over the units in `keep`.

    Args:
        C: (N, N) covariance; var: (N,) variances; keep: (N,) bool, units with var > 0 only.
    Returns:
        (k, k) correlation matrix of the kept units.
    """
    s = np.sqrt(var[keep])
    return C[np.ix_(keep, keep)] / np.outer(s, s)


def partial_r2(Y, X):
    """Partial R^2 of every regressor for every unit: the drop in residual variance when the
    regressor is added last, over the residual variance without it. An intercept is always included.

    Args:
        Y: (n_units, n_samples) responses; X: (n_samples, p) regressors (no intercept column).
    Returns:
        (n_units, p) float64 partial R^2; rows of units with zero response variance are nan.
    """
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    sd = Y.std(axis=1)
    ok = sd > 0
    Z = np.zeros_like(Y)
    Z[ok] = (Y[ok] - Y[ok].mean(axis=1, keepdims=True)) / sd[ok, None]   # scale-free, centred

    def rss(cols):
        """Residual sum of squares of every row of Z on [intercept, X[:, cols]]."""
        D = np.column_stack([np.ones(n), X[:, cols]])
        Q, _ = np.linalg.qr(D)
        proj = Z @ Q
        return (Z * Z).sum(axis=1) - (proj * proj).sum(axis=1)

    full = rss(list(range(p)))
    out = np.full((Y.shape[0], p), np.nan)
    for j in range(p):
        red = rss([c for c in range(p) if c != j])
        out[ok, j] = np.clip((red[ok] - full[ok]) / red[ok], 0.0, 1.0)
    return out


def selectivity(pr2, thr=SEL_THR):
    """Selective and mixed-selective units from partial R^2.

    Args:
        pr2: (n_units, p) partial R^2, nan rows for units without one.
    Returns:
        (selective (n_units,) bool, mixed-selective (n_units,) bool); nan rows are neither.
    """
    n_sel = np.nan_to_num(pr2, nan=0.0) >= thr
    k = n_sel.sum(axis=1)
    return k >= 1, k >= 2


def kmeans_classes(profiles, seed=POP_SEED):
    """k-means classes of selectivity profiles with k chosen by the highest mean silhouette.

    Args:
        profiles: (n, p) profiles (partial R^2 divided by its sum).
    Returns:
        dict k_star, silhouettes {k: mean silhouette}, labels (n,), centroids (k*, p),
        shares (k*,) class size / n.
    """
    sil, fits = {}, {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=KMEANS_RESTARTS, random_state=seed).fit(profiles)
        fits[k] = km
        sil[k] = float(silhouette_score(profiles, km.labels_)) if len(set(km.labels_)) > 1 else -1.0
    k_star = max(sil, key=sil.get)
    km = fits[k_star]
    shares = np.bincount(km.labels_, minlength=k_star) / len(km.labels_)
    return dict(k_star=int(k_star), silhouettes=sil, labels=km.labels_,
                centroids=km.cluster_centers_, shares=shares)


def match_classes(a, b):
    """Match the classes of two k-means runs one to one by minimum total centroid distance.

    Args:
        a, b: kmeans_classes results ("all" and "active").
    Returns:
        dict pairs [(class in a, class in b, share a, share b)], max_diff_pp (largest absolute
        share difference over matched pairs, percentage points).
    """
    D = np.linalg.norm(a["centroids"][:, None, :] - b["centroids"][None, :, :], axis=2)
    ia, ib = linear_sum_assignment(D)
    pairs = [(int(i), int(j), float(a["shares"][i]), float(b["shares"][j])) for i, j in zip(ia, ib)]
    return dict(pairs=pairs, max_diff_pp=float(max(abs(sa - sb) for _, _, sa, sb in pairs) * 100))


def population_statistics(rates, Y, X, act):
    """S1-S4 over all units and over the active units.

    Args:
        rates: (N, T, B) rates for S1 and S2; Y: (N, n_samples) responses for S3 and S4;
        X: (n_samples, p) task variables; act: (N,) bool, the active units.
    Returns:
        dict of statistics, each with an "all" and an "act" value, plus diagnostics.
    """
    N = rates.shape[0]
    C, var = covariance(rates)
    nz = var > 0
    s1_all, s1_act = pr_from_cov(C), pr_from_cov(C[np.ix_(act, act)])
    if np.any(act & ~nz):
        raise SystemExit("FATAL: an active unit has zero variance")
    s2_all = pr_from_cov(correlation_submatrix(C, var, nz))
    s2_act = pr_from_cov(correlation_submatrix(C, var, act))
    pr2 = partial_r2(Y, X)
    sel, mix = selectivity(pr2)
    n_act = int(act.sum())
    has = ~np.isnan(pr2).any(axis=1)
    tot = np.nansum(pr2, axis=1)
    prof_ok = has & (tot > 0)
    prof = np.zeros_like(pr2)
    prof[prof_ok] = pr2[prof_ok] / tot[prof_ok, None]
    k_all = kmeans_classes(prof[prof_ok])
    k_act = kmeans_classes(prof[prof_ok & act])
    m = match_classes(k_all, k_act)
    return dict(
        N=N, n_active=n_act, n_zero_var=int((~nz).sum()), n_inactive_nonzero_var=int((~act & nz).sum()),
        pr_check_rel=float(abs(pr_from_trace(C) - s1_all) / s1_all),
        S1=dict(all=s1_all, act=s1_act),
        S2=dict(all=s2_all, act=s2_act, n_all=int(nz.sum())),
        S3a=dict(all=float(sel.sum() / N), act=float(sel[act].sum() / n_act),
                 n_all=int(sel.sum()), n_act=int(sel[act].sum()), n_inactive=int(sel[~act].sum())),
        S3b=dict(all=float(mix.sum() / N), act=float(mix[act].sum() / n_act),
                 n_all=int(mix.sum()), n_act=int(mix[act].sum()), n_inactive=int(mix[~act].sum())),
        S4=dict(k_all=k_all["k_star"], k_act=k_act["k_star"],
                shares_all=k_all["shares"].tolist(), shares_act=k_act["shares"].tolist(),
                sil_all=k_all["silhouettes"], sil_act=k_act["silhouettes"],
                n_all=int(prof_ok.sum()), n_act=int((prof_ok & act).sum()),
                pairs=m["pairs"], max_diff_pp=m["max_diff_pp"]),
        n_no_profile=int((~prof_ok).sum()),
    )


def network_flag(stat, s):
    """One network's verdict on one statistic: "changes", "unchanged" or "between".

    Args:
        stat: "S1", "S2", "S3a", "S3b" or "S4"; s: that statistic's dict from population_statistics.
    Returns:
        str flag under the rule's per-network limits.
    """
    if stat == "S4":
        if s["k_all"] != s["k_act"] or s["max_diff_pp"] >= S4_CHANGES_PP:
            return "changes"
        return "unchanged" if s["max_diff_pp"] < S4_UNCHANGED_PP else "between"
    d, base = abs(s["all"] - s["act"]), abs(s["act"])
    if base == 0:
        return "unchanged" if d == 0 else "changes"
    if d >= CHANGES * base:
        return "changes"
    return "unchanged" if d < UNCHANGED * base else "between"


def task_label(flags):
    """The rule's label for one statistic on one task from its networks' flags."""
    if all(f == "changes" for f in flags):
        return "changes"
    if all(f == "unchanged" for f in flags):
        return "unchanged"
    return "mixed"


def rel_diff(s):
    """(S_all - S_act) / S_act for a scalar statistic dict (nan for a zero baseline)."""
    return (s["all"] - s["act"]) / s["act"] if s["act"] != 0 else float("nan")


# ------------------------------------------------------------------------------------------------
# oracle tests (Part 3 of the rule entry)
# ------------------------------------------------------------------------------------------------

def cddm_design():
    """The CDDM regressors (context, motion coh, colour coh, choice) of the 450 trained conditions.

    Built from the task class with the trained 15-value coherence grid, so the oracle uses the real
    correlated design. Returns (450, 4) float64.
    """
    cohs = [-1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625, 0.0,
            0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
    rows = []
    for context in ["motion", "color"]:
        for c1 in cohs:
            for c2 in cohs:
                motion = c1 if context == "motion" else c2
                color = c1 if context == "color" else c2
                rel = motion if context == "motion" else color
                rows.append([1.0 if context == "motion" else -1.0, motion, color, 1.0 if rel > 0 else -1.0])
    return np.asarray(rows)


def planted_unit(X, targets, rng, orthogonal=True):
    """A response with a prescribed sample partial R^2 on each regressor.

    y = sum_j a_j x_j + e. With e orthogonal to [1, X], adding x_j last lowers the residual sum of
    squares by a_j^2 |x_j residualised on the other columns|^2, so a_j is solved in closed form;
    this construction does not use partial_r2.

    Args:
        X: (n, p) design; targets: {column: partial R^2}; rng: numpy Generator;
        orthogonal: project e orthogonal to the design (exact values) or leave it raw (sampling).
    Returns:
        (n,) float64 response.
    """
    n, p = X.shape
    e = rng.standard_normal(n)
    D = np.column_stack([np.ones(n), X])
    if orthogonal:
        e = e - D @ np.linalg.lstsq(D, e, rcond=None)[0]
    ee = float(e @ e)
    y = e.copy()
    for j, t in targets.items():
        others = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
        xr = X[:, j] - others @ np.linalg.lstsq(others, X[:, j], rcond=None)[0]
        a = np.sqrt(t / (1 - t) * ee / float(xr @ xr))
        y += a * X[:, j]
    return y


def oracle(verbose=True):
    """Run the oracle tests V1a, V1b, V2, V3 and V4 with the thresholds of the rule entry.

    Returns:
        dict test name -> (passed bool, details dict); V4 records expected behaviour (no pass/fail
        beyond the stated predictions).
    """
    rng = np.random.default_rng(POP_SEED)
    res = {}

    # V1a: two PR formulas
    M = rng.standard_normal((50, 80))
    C = M @ M.T / 80
    r = abs(pr_from_cov(C) - pr_from_trace(C)) / pr_from_cov(C)
    res["V1a"] = (r < 1e-8, dict(rel=r))

    # V1b: k sources, 200 mixed units, 800 zero units
    det = {}
    ok = True
    for k in (3, 10):
        src = rng.standard_normal((k, 20000))
        src = (src - src.mean(1, keepdims=True)) / src.std(1, keepdims=True)
        Q, _ = np.linalg.qr(rng.standard_normal((200, k)))
        for kind, Mx in (("orthonormal", Q), ("gaussian", rng.standard_normal((200, k)))):
            R = np.zeros((1000, 20000))
            R[:200] = Mx @ src
            rates = R[:, :, None]                  # (N, T, B = 1)
            Cc, var = covariance(rates, chunk=1)
            all_ = pr_from_cov(Cc)
            sub = pr_from_cov(Cc[:200, :200])
            nz = var > 0
            s2 = pr_from_cov(correlation_submatrix(Cc, var, nz))
            det[f"k={k} {kind}"] = dict(S1_all=all_, S1_200=sub, S2=s2, n_nonzero=int(nz.sum()))
            if kind == "orthonormal":
                ok &= all(abs(v - k) < 0.05 * k for v in (all_, sub, s2))
                ok &= abs(all_ - sub) / sub < 1e-8
    res["V1b"] = (ok, det)

    # V2: planted partial R^2 on the real CDDM design, then a sampling version
    X = cddm_design()
    det, ok = {}, True
    for j in range(4):
        for t in (0.05, 0.2, 0.5):
            y = planted_unit(X, {j: t}, rng)
            pr2 = partial_r2(y[None], X)[0]
            sel, mix = selectivity(pr2[None])
            want = np.zeros(4)
            want[j] = t
            good = np.all(np.abs(pr2 - want) < 0.02) and bool(sel[0]) == (t >= SEL_THR) and not mix[0]
            ok &= good
            det[f"x{j} {t}"] = dict(recovered=pr2.round(4).tolist(), selective=bool(sel[0]),
                                     mixed=bool(mix[0]), ok=bool(good))
    for js in ((1, 2), (0, 3)):
        y = planted_unit(X, {js[0]: 0.3, js[1]: 0.3}, rng)
        pr2 = partial_r2(y[None], X)[0]
        sel, mix = selectivity(pr2[None])
        good = bool(mix[0]) and all(abs(pr2[j] - 0.3) < 0.02 for j in js)
        ok &= good
        det[f"mixed {js}"] = dict(recovered=pr2.round(4).tolist(), mixed=bool(mix[0]), ok=bool(good))
    Xs = np.where(rng.random((20000, 3)) < 0.5, -1.0, 1.0)
    for t in (0.05, 0.2, 0.5):
        y = planted_unit(Xs, {0: t}, rng, orthogonal=False)
        pr2 = partial_r2(y[None], Xs)[0]
        want = np.array([t, 0.0, 0.0])
        good = bool(np.all(np.abs(pr2 - want) < 0.02))
        ok &= good
        det[f"sampling {t}"] = dict(recovered=pr2.round(4).tolist(), ok=good)
    res["V2"] = (ok, det)

    # V3 and V4: three planted classes of 100 units, plus 700 zero (V3) or weak (V4) units
    def classes_population(extra):
        """(rates (1000, 450, 1), responses (1000, 450), active mask) for V3/V4."""
        Y = np.zeros((1000, X.shape[0]))
        plan = [{1: 0.4}, {2: 0.4}, {1: 0.4, 2: 0.4}]
        for c, tg in enumerate(plan):
            for u in range(100):
                Y[100 * c + u] = planted_unit(X, tg, rng)
        if extra == "weak":
            for u in range(300, 1000):
                Y[u] = 1e-3 * planted_unit(X, {int(rng.integers(4)): 0.4}, rng)
        act = np.zeros(1000, bool)
        act[:300] = True
        return Y[:, :, None], Y, act

    rates, Y, act = classes_population("zero")
    st = population_statistics(rates, Y, X, act)
    exp_share = lambda s: abs(s["all"] - s["act"] * 300 / 1000) < 1e-12
    ok = (abs(st["S1"]["all"] - st["S1"]["act"]) / st["S1"]["act"] < 1e-8
          and abs(st["S2"]["all"] - st["S2"]["act"]) / st["S2"]["act"] < 1e-8
          and st["S3a"]["n_all"] == st["S3a"]["n_act"] and st["S3b"]["n_all"] == st["S3b"]["n_act"]
          and exp_share(st["S3a"]) and exp_share(st["S3b"])
          and st["S4"]["k_all"] == 3 and st["S4"]["k_act"] == 3
          and all(abs(s - 1 / 3) < 0.02 for s in st["S4"]["shares_all"] + st["S4"]["shares_act"])
          and st["S4"]["max_diff_pp"] < 1e-9)
    res["V3"] = (bool(ok), _brief(st))

    rates, Y, act = classes_population("weak")
    st = population_statistics(rates, Y, X, act)
    ok = (abs(st["S1"]["all"] - st["S1"]["act"]) / st["S1"]["act"] < 0.02
          and abs(st["S2"]["all"] - st["S2"]["act"]) / st["S2"]["act"] >= 0.10
          and st["S3a"]["n_inactive"] > 0)
    res["V4"] = (bool(ok), _brief(st))

    if verbose:
        for name, (passed, d) in res.items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}: {json.dumps(d, default=_jsonable)}")
    return res


def _brief(st):
    """The printable core of a population_statistics result (no per-unit arrays)."""
    return {k: st[k] for k in ("n_active", "n_zero_var", "S1", "S2", "S3a", "S3b")} | {
        "S4": {k: st["S4"][k] for k in ("k_all", "k_act", "shares_all", "shares_act", "max_diff_pp")}}


def _jsonable(o):
    """json.dumps fallback for numpy scalars and arrays."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    raise TypeError(type(o))


# ------------------------------------------------------------------------------------------------
# the networks
# ------------------------------------------------------------------------------------------------

def manifest():
    """The nine networks: list of dicts label, task, seed, run_dir (Della), params (local), iteration,
    n_active_ablation, config (local)."""
    rows = []
    with open(os.path.join(NETS, "manifest.tsv")) as fh:
        for line in fh:
            label, run_dir, pf, it, na = line.rstrip("\n").split("\t")
            d = os.path.join(NETS, label)
            rows.append(dict(label=label, task=label.split("_")[0], seed=label.split("__")[2],
                             run_dir=run_dir, params=os.path.join(d, os.path.basename(pf)),
                             iteration=int(it), n_active_ablation=int(na), dir=d))
    return rows


def batch(cfg):
    """The fixed noise-free batch of the rule: inputs, targets, conditions and the task variables.

    Args:
        cfg: the run's saved config.
    Returns:
        (inputs (n_in, T, B) float32 tensor, targets (n_out, T, B) array, mask, conditions list).
    """
    np.random.seed(POP_SEED)
    torch.manual_seed(POP_SEED)
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    tcfg.seed = POP_SEED
    name = cfg.task.taskname
    if name == "DMTS":
        tcfg.num_rep = N_TRIALS // 4
    elif name == "NBitFlipFlop":
        tcfg.batch_size = N_TRIALS
    task = hydra.utils.instantiate(tcfg)
    inp, tgt, cond = task.get_batch()
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
    return torch.as_tensor(np.asarray(inp), dtype=torch.float32), np.asarray(tgt), mask, cond


def responses(name, rates, tgt, cond):
    """Per-sample responses and task variables of the rule's S3.

    Args:
        name: taskname; rates: (N, T, B) float32 array; tgt: (n_out, T, B) targets; cond: conditions.
    Returns:
        (Y (N, n_samples), X (n_samples, p), variable names).
    """
    if name == "CDDM":
        Y = rates[:, CDDM_EPOCH, :].mean(axis=1)
        X = np.array([[1.0 if c["context"] == "motion" else -1.0, c["motion_coh"], c["color_coh"],
                       float(c["correct_choice"])] for c in cond])
        return Y, X, ["context", "motion coherence", "colour coherence", "choice"]
    if name == "DMTS":
        Y = rates[:, DMTS_EPOCH, :].mean(axis=1)
        s = np.array([1.0 if c["num_sample_channel"] == 0 else -1.0 for c in cond])
        t = np.array([1.0 if c["num_match_channel"] == 0 else -1.0 for c in cond])
        return Y, np.column_stack([s, t, s * t]), ["sample", "test", "match"]
    # 3BFF: every (step, trial) at which all three bit targets are nonzero
    ok = np.all(tgt != 0, axis=0)                     # (T, B)
    Y = rates[:, ok]                                  # (N, n_samples); boolean mask over (T, B)
    X = np.stack([tgt[c][ok] for c in range(tgt.shape[0])], axis=1)
    return Y, X, ["bit 1", "bit 2", "bit 3"]


def measure(net):
    """All statistics for one network of the manifest.

    Returns:
        dict of population_statistics plus gates (active counts, noise-free R^2 on the batch).
    """
    cfg, rnn = A.build(net["dir"], A.load_params(net["params"]))   # config read from net["dir"]
    inp, tgt, mask, cond = batch(cfg)
    with torch.no_grad():
        states, out = rnn(inp.to(rnn.device), w_noise=False)[:2]
        p = A.participation(states).cpu().numpy()
        rates = torch.relu(states).cpu().numpy()
        r2 = Trainer.r2_score(out, torch.as_tensor(tgt, dtype=torch.float32, device=rnn.device),
                              torch.as_tensor(np.asarray(mask), device=rnn.device))
    act = active_mask(p)
    name = cfg.task.taskname
    Y, X, names = responses(name, rates, tgt, cond)
    st = population_statistics(rates, Y, X, act)
    st.update(label=net["label"], task=TASK_LABEL[name], seed=net["seed"], iteration=net["iteration"],
              n_active_ablation=net["n_active_ablation"], r2_noise_free=float(r2),
              variables=names, n_samples=int(Y.shape[1]), batch_shape=list(rates.shape))
    return st


def table(results):
    """Print one row per network and the per-task labels; return the labels dict."""
    hdr = (f"{'network':<22}{'active':>7}{'abl.':>6}{'zero':>6}{'R2':>7} | {'S1 all':>7}{'act':>7}"
           f" | {'S2 all':>7}{'act':>7} | {'sel all':>8}{'act':>6}{'inact':>6} | {'mix all':>8}{'act':>6}"
           f"{'inact':>6} | {'k*':>5}{'dpp':>6}")
    print(hdr)
    for s in results:
        print(f"{s['task'] + ' ' + s['seed']:<22}{s['n_active']:>7}{s['n_active_ablation']:>6}"
              f"{s['n_zero_var']:>6}{s['r2_noise_free']:>7.3f} | {s['S1']['all']:>7.2f}{s['S1']['act']:>7.2f}"
              f" | {s['S2']['all']:>7.1f}{s['S2']['act']:>7.1f} | {s['S3a']['n_all']:>8}{s['S3a']['n_act']:>6}"
              f"{s['S3a']['n_inactive']:>6} | {s['S3b']['n_all']:>8}{s['S3b']['n_act']:>6}{s['S3b']['n_inactive']:>6}"
              f" | {s['S4']['k_all']}/{s['S4']['k_act']:<3}{s['S4']['max_diff_pp']:>6.1f}")
    labels = {}
    for task in ("3BFF", "CDDM", "DMTS"):
        rows = [s for s in results if s["task"] == task]
        labels[task] = {}
        for stat in ("S1", "S2", "S3a", "S3b", "S4"):
            flags = [network_flag(stat, s[stat]) for s in rows]
            labels[task][stat] = dict(label=task_label(flags), flags=flags)
        print(f"  {task}: " + "; ".join(f"{k} {v['label']} ({','.join(v['flags'])})"
                                         for k, v in labels[task].items()))
    return labels


def main():
    """Command line: `oracle` or `run` (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["oracle", "run"])
    a = ap.parse_args()
    orc = oracle()
    if a.what == "oracle":
        return
    if not all(passed for passed, _ in orc.values()):
        raise SystemExit("FATAL: an oracle test failed; the networks are not measured")
    results = []
    for net in manifest():
        st = measure(net)
        print(f"  {st['label']}: active {st['n_active']} (ablation file {st['n_active_ablation']}), "
              f"zero-variance {st['n_zero_var']}, noise-free R2 {st['r2_noise_free']:.4f}, "
              f"{st['n_samples']} S3 samples, PR check {st['pr_check_rel']:.1e}")
        if st["task"] == "CDDM" and st["n_active"] != st["n_active_ablation"]:
            raise SystemExit("FATAL: CDDM active count differs from the ablation file on the same batch")
        results.append(st)
    labels = table(results)
    with open(OUT, "w") as fh:
        json.dump(dict(results=results, labels=labels,
                       oracle={k: dict(passed=bool(p), details=d) for k, (p, d) in orc.items()}),
                  fh, default=_jsonable, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
