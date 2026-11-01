#!/usr/bin/env python3
"""
Which held-out R^2 values of Figure 5 and Supplementary Table 2 were computed from weights saved inside
a loss burst or a stretch of non-finite loss (dead-relu-paper#18; rule: notebook entry
2026-11-01_0745, fixed before this script ran). Also collects the active-count trace of every
N = 1,000 per-unit-rule and control network, for Figure 5a.

For every network the script joins the noisy training loss along the warm-start chain
(common.noisy_loss_series), finds its loss bursts with common.noisy_floor and its non-finite
stretches, widens each by max(BURST_MEDIAN, half its length) (noisy_floor's own masking pad), and
asks whether the iteration t_w of the scored weights falls inside one:
  - R^2 at the floor crossing, measured on a kept snapshot: t_w = the snapshot's iteration;
  - R^2 at the floor crossing, estimated from the end of training: t_w = the run's last iteration;
  - R^2 at 75,000 iterations and every removal score: t_w = 75,000.

INPUTS (Della): ~/rm_read/runs_*.json (read_remedies.py 2d32b3b, entry 2026-10-31_1732),
~/a3_read/a3_N*.json (a3_remedy_ablation.py 30efe65, same entry), ~/b3b4_read/read_b3.json
(read_b3_b4_4000.py, entry 2026-10-31_1750), and the CDDM rate-penalty sweep of Fig. 3a
(CDDM_std_g0_bias0_c75000, lambda_frm 0.2 and its unpenalised neighbour).

Usage (Della, through sbatch; CPU only):
  python fig5_burst_check.py --out ~/fig5r2_read --cddm <dir of CDDM_std_g0_bias0_c75000>
Output: <out>/burst_check.json (one record per network: bursts, non-finite stretches and every
scored value with t_w and its verdict), <out>/traces_N1000.json (it_act, active per N = 1,000
catalogue network) and a printed summary.
"""

import argparse
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BURST_MEDIAN, DATA_DIR, load_run, noisy_floor, noisy_loss_series, task_of)

HOME = os.path.expanduser("~")
FIXED_ITER = 75000.0


def padded(first, last):
    """Widen an interval [first, last] (iterations) by noisy_floor's burst-masking pad.

    Args:
        first, last: the interval's first and last iteration (floats).
    Returns:
        (lo, hi): the widened interval.
    """
    pad = max(BURST_MEDIAN, 0.5 * (last - first))
    return first - pad, last + pad


def nonfinite_stretches(it, L):
    """Maximal runs of consecutive iterations whose loss is NaN or infinite.

    Args:
        it: (n,) iterations; L: (n,) raw noisy loss.
    Returns:
        list of (first iteration, last iteration) of each run.
    """
    bad = ~np.isfinite(L)
    d = np.diff(np.r_[0, bad.astype(int), 0])
    return [(float(it[s]), float(it[e - 1])) for s, e in zip(np.flatnonzero(d == 1),
                                                             np.flatnonzero(d == -1))]


def where_is(t, bursts, nonfinite):
    """Which kind of stretch iteration t lies in.

    Args:
        t: iteration of the scored weights; bursts, nonfinite: lists of padded (lo, hi) intervals.
    Returns:
        "non-finite", "burst" or "calm" (non-finite wins when both apply).
    """
    if any(lo <= t <= hi for lo, hi in nonfinite):
        return "non-finite"
    if any(lo <= t <= hi for lo, hi in bursts):
        return "burst"
    return "calm"


def snapshot_iter(path):
    """The iteration of a kept snapshot from its folder name (.../it0058500 -> 58500.0), or None."""
    m = re.search(r"/it(\d+)/?$", path or "")
    return float(m.group(1)) if m else None


def check_run(job):
    """Find one network's bursts and non-finite stretches and judge each of its scored values.

    Args:
        job: dict with run_dir, source (str), key (identifier), and values: list of dicts with
            what (label), r2 (float or None) and t_w (iteration; None means the run's last one).
    Returns:
        dict: the job's fields plus n_iter, n_nonfinite, n_bursts, bursts and nonfinite (padded
        intervals), and each value given t_w and where ("calm", "burst", "non-finite");
        error (str) instead when the loss cannot be read.
    """
    out = dict(job)
    it, L = noisy_loss_series(job["run_dir"])
    if L is None:
        out["error"] = "no noisy loss"
        return out
    nf = noisy_floor(it, L, task=task_of(job["run_dir"]))
    bursts = [padded(b0, b1) for b0, b1, _ in nf["bursts"]]
    nonfin = [padded(a, b) for a, b in nonfinite_stretches(it, L)]
    out.update(n_iter=int(len(L)), it_last=float(it[-1]), n_nonfinite=int((~np.isfinite(L)).sum()),
               n_bursts=len(nf["bursts"]), bursts=bursts, nonfinite=nonfin)
    vals = []
    for v in job["values"]:
        v = dict(v)
        t = float(it[-1]) if v["t_w"] is None else float(v["t_w"])
        v.update(t_w=t, where=where_is(t, bursts, nonfin))
        vals.append(v)
    out["values"] = vals
    return out


def find_dir(name):
    """The run folder with this name anywhere under DATA_DIR/<sweep>/<cell>/, or None."""
    hits = glob.glob(os.path.join(DATA_DIR, "*", "*", glob.escape(name)))
    return hits[0] if hits else None


def jobs_catalogue():
    """One job per run of the per-unit-rule catalogue: R^2 at the floor crossing and at 75,000."""
    jobs = []
    for f in sorted(glob.glob(os.path.join(HOME, "rm_read", "runs_*.json"))):
        with open(f) as fh:
            for r in json.load(fh):
                vals = []
                if r.get("r2") is not None:
                    t = snapshot_iter(r.get("snapshot")) if r.get("r2_measured") else None
                    vals.append(dict(what="floor, measured" if r.get("r2_measured") else
                                     "floor, estimated", r2=r["r2"], t_w=t))
                if r.get("r2_75k") is not None:
                    vals.append(dict(what="75k", r2=r["r2_75k"], t_w=FIXED_ITER))
                if os.path.basename(r["run_dir"].rstrip("/")).startswith("nan"):
                    continue
                jobs.append(dict(source="catalogue", key=r["run_dir"], cell=r["cell"],
                                 run_dir=r["run_dir"], values=vals))
    return jobs


def jobs_a3():
    """One job per network of the removal analysis: intact and every removal score, at 75,000."""
    jobs = []
    for f in sorted(glob.glob(os.path.join(HOME, "a3_read", "a3_N*.json"))):
        with open(f) as fh:
            for r in json.load(fh):
                vals = [dict(what="intact", r2=r["r2_full"], t_w=FIXED_ITER)]
                for k in ("r2_recruited", "r2_random_active", "random_10", "random_30", "random_50"):
                    if r.get(k) is not None:
                        v = r[k]
                        vals.append(dict(what=k, r2=float(np.median(v)) if isinstance(v, list)
                                         else v, t_w=FIXED_ITER))
                jobs.append(dict(source="a3", key=r["run_dir"], cell=r["cell"], family=r["family"],
                                 N=r["N"], run_dir=r["run_dir"], values=vals))
    return jobs


def jobs_b3():
    """One job per network of the combination test: R^2 at the floor crossing and at 75,000."""
    with open(os.path.join(HOME, "b3b4_read", "read_b3.json")) as fh:
        b3 = json.load(fh)
    jobs = []
    for cell, runs in b3["runs"].items():
        for r in runs:
            if r["name"].startswith("nan"):
                continue
            d = find_dir(r["name"])
            vals = []
            for ro in ("floor", "fixed"):
                rec = r.get(ro)
                if rec is not None and rec.get("r2") is not None:
                    vals.append(dict(what=ro, r2=rec["r2"], t_w=snapshot_iter(rec["snapshot"])))
            jobs.append(dict(source="b3", key=r["name"], cell=cell, run_dir=d, values=vals))
    return jobs


def jobs_cddm(cddm_dir):
    """One job per CDDM network of Fig. 3a's rate-penalty value and its unpenalised neighbour."""
    jobs = []
    for cell in ("EqType=h_N=1000_LmbdRWS=0_LmbdFR=0.2", "EqType=h_N=1000_LmbdRWS=0_LmbdFR=0"):
        for d in sorted(glob.glob(os.path.join(cddm_dir, cell, "*", ""))):
            if os.path.basename(d.rstrip("/")).startswith("nan"):
                continue
            jobs.append(dict(source="cddm", key=d.rstrip("/"), cell=cell, run_dir=d.rstrip("/"),
                             values=[dict(what="75k, final weights", r2=None, t_w=None)]))
    return jobs


def trace_n1000(run_dir):
    """Active count at every participation probe of one network (common.load_run).

    Args:
        run_dir: a run folder.
    Returns:
        (run_dir, it_act list, active list) or (run_dir, None, None) when unreadable.
    """
    r = load_run(run_dir)
    if r is None:
        return run_dir, None, None
    return run_dir, np.asarray(r["it_act"]).tolist(), np.asarray(r["active"]).tolist()


def summary(res):
    """Print the counts the rule asks for and every affected or suspicious value.

    Args:
        res: list of check_run outputs.
    Returns:
        None.
    """
    n_val = n_aff = 0
    for r in res:
        if "error" in r:
            print(f"ERROR {r['source']} {r['key']}: {r['error']}")
            continue
        for v in r["values"]:
            n_val += 1
            bad = v["where"] != "calm"
            n_aff += bad
            low = v["r2"] is not None and v["what"] in ("floor, measured", "floor, estimated",
                                                        "75k", "intact", "floor", "fixed") \
                and (not np.isfinite(v["r2"]) or v["r2"] < 0.5)
            if bad or low:
                r2 = "None" if v["r2"] is None else f"{v['r2']:.4g}"
                flag = "MISSED (R2 < 0.5, calm)" if low and not bad else v["where"]
                print(f"{flag:24s} {r['source']:9s} {v['what']:16s} t_w {v['t_w']:9.0f} R2 {r2:>12s}"
                      f"  {r['cell'][:60]} {os.path.basename(r['run_dir'] or '')[:40]}")
    print(f"\n{n_aff} of {n_val} values affected")
    for src in ("cddm",):
        for r in res:
            if r["source"] == src and "error" not in r:
                print(f"CDDM {r['cell']}: {r['n_bursts']} bursts, {r['n_nonfinite']} non-finite, "
                      f"last iteration {r['it_last']:.0f} {r['values'][0]['where']}")


def main():
    """Build the jobs, run them in parallel, write the two JSON files and print the summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cddm", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = jobs_catalogue() + jobs_a3() + jobs_b3() + jobs_cddm(a.cddm)
    missing = [j for j in jobs if j["run_dir"] is None]
    for j in missing:
        print(f"no run folder for {j['source']} {j['key']}")
    jobs = [j for j in jobs if j["run_dir"] is not None]
    print(f"{len(jobs)} networks to check")
    with Pool(a.procs) as pool:
        res = pool.map(check_run, jobs)
    with open(os.path.join(a.out, "burst_check.json"), "w") as fh:
        json.dump(res, fh)
    n1000 = sorted({j["run_dir"] for j in jobs if j["source"] == "catalogue" and "_N=1000_" in j["cell"]})
    with Pool(a.procs) as pool:
        tr = pool.map(trace_n1000, n1000)
    with open(os.path.join(a.out, "traces_N1000.json"), "w") as fh:
        json.dump({d: dict(it_act=i, active=v) for d, i, v in tr if i is not None}, fh)
    summary(res)


if __name__ == "__main__":
    main()
