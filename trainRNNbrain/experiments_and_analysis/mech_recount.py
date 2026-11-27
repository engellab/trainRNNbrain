#!/usr/bin/env python3
"""Independent recount of the mechanism tests (rules entry 2026-11-25_1311). Numpy only, no import from this repository.

Written from the task and network definitions, not from the project's code: its own CDDM conditions, its own Euler pass in
double precision, its own participation, its own file reading. It recounts, for every finished network of parts A and B, the
active units on the final weights, the active units of the last stored participation vector, the noise-free held-out R^2, and
(part B) the units inactive on the parent's weights and active at the end; then the gate, R0, S and the part B net shares with
the bars of the rules typed here a second time. With --reads it compares every number with mech_read.py's JSON.

Usage:  python mech_recount.py TRAINED_RNNS_DIR OUT_JSON [--reads READ_JSON]
"""
import argparse
import glob
import json
import os
import pickle
import re

import numpy as np

COH = [-1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625, 0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
ALPHA = 0.1


def conditions(coh):
    """CDDM inputs and targets for every context x coherence x coherence.

    Two contexts; input channels: 2 context cues (on throughout), then motion right/left and colour right/left (on from step 99),
    each (1 +- coherence)/2; target 1 on the channel of the sign of the attended coherence from step 199, 0.5 on both if it is 0.

    Args:
        coh: list of coherences.
    Returns:
        (u (6, 300, B), y (2, 300, B)) float64 arrays.
    """
    U, Y = [], []
    for ctx in range(2):
        for m in coh:
            for c in coh:
                u = np.zeros((6, 300))
                u[ctx] = 1.0
                u[2:, 99:] = np.array([(1 + m) / 2, (1 - m) / 2, (1 + c) / 2, (1 - c) / 2])[:, None]
                a = (m, c)[ctx]
                y = np.zeros((2, 300))
                y[:, 199:] = 0.5 if a == 0 else np.array([a > 0, a < 0], float)[:, None]
                U.append(u)
                Y.append(y)
    return np.stack(U, -1), np.stack(Y, -1)


def simulate(W, J, O, x0, u):
    """Noise-free Euler pass of tau dx/dt = -x + W relu(x) + J u, double precision.

    Args:
        W (N,N), J (N,6), O (2,N), x0 (N,): weights and initial state; u (6,T,B).
    Returns:
        (outputs (2,T,B), rates (N,T,B) float32).
    """
    N, T, B = len(x0), u.shape[1], u.shape[2]
    x = np.repeat(x0[:, None], B, 1)
    z, R = np.empty((2, T, B)), np.empty((N, T, B), np.float32)
    for t in range(T):
        r = np.maximum(x, 0.0)
        z[:, t], R[:, t] = O @ r, r
        x = (1 - ALPHA) * x + ALPHA * (W @ r + J @ u[:, t])
    return z, R


def partic(R):
    """Participation: std plus 90th percentile of each unit's rate over all steps and trials."""
    return np.array([np.std(r.ravel().astype(np.float64)) + np.percentile(r.ravel(), 90) for r in R])


def is_active(p):
    """Participation at least 5% of the network's 95th percentile."""
    return p >= 0.05 * np.percentile(p, 95)


def params(d):
    """(W, J, O, x0) float64 from the run folder's LastParams npz."""
    z = np.load(glob.glob(os.path.join(glob.escape(d), "*LastParams*.npz"))[0], allow_pickle=True)
    return [np.asarray(z[k], dtype=np.float64) for k in ("W_rec", "W_inp", "W_out", "y_init")]


def cell(root, name):
    """Finished runs of a cell.

    Args:
        root: trained_RNNs directory; name: cell path.
    Returns:
        dict seed -> run folder.
    """
    out = {}
    for d in sorted(glob.glob(os.path.join(glob.escape(os.path.join(root, name)), "*seed=*"))):
        if glob.glob(os.path.join(glob.escape(d), "*LastParams*.npz")):
            out[re.search(r"seed=(\d+);", os.path.basename(d)).group(1)] = d
    return out


def main():
    """Command line: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("out_json")
    ap.add_argument("--reads", default=None)
    a = ap.parse_args()
    grid, _ = conditions(COH)
    uh, yh = conditions([(x + y) / 2 for x, y in zip(COH[:-1], COH[1:])])
    scored = np.r_[0:100, 200:300]
    A_names = {f"{n}/{w}": f"CDDM_mech_noise_decay/EqType=h_N=1000_noise={n}_wd={w}" for n in ("default", "zero") for w in ("1e-6", "0")}
    B_names = {b: f"CDDM_mech_switch/EqType=h_N=1000_arm={b}" for b in ("none", "ent0.1", "rate0.1")}
    out, act = dict(A={}, B={}), {}
    for part, names in (("A", A_names), ("B", B_names)):
        for g, name in names.items():
            out[part][g] = {}
            for s, d in cell(a.root, name).items():
                W = params(d)
                act[(part, g, s)] = is_active(partic(simulate(*W, grid)[1]))
                z = simulate(*W, uh)[0][:, scored]
                t = yh[:, scored]
                r2 = 1 - ((z - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
                with open(glob.glob(os.path.join(glob.escape(d), "*ParticipationTrace.pkl"))[0], "rb") as fh:
                    tr = pickle.load(fh)
                rec = dict(run_dir=d, active=int(act[(part, g, s)].sum()), r2_clean=float(r2),
                           last_vector=int(is_active(np.asarray(tr["participation"][-1], np.float64)).sum()),
                           last_it=int(tr["participation_iters"][-1]))
                if part == "B":
                    start = act[("A", "default/1e-6", s)]
                    end = act[(part, g, s)]
                    rec.update(n_inactive=int((~start).sum()), resurrected=int((end & ~start).sum()),
                               share=float(end[~start].mean()))
                out[part][g][s] = rec
                print(part, g, s, rec["active"], rec["last_vector"], round(rec["r2_clean"], 4), rec.get("resurrected"), flush=True)
    m = {g: np.mean([r["active"] for r in v.values()]) for g, v in out["A"].items() if v}
    if len(m) == 4:
        r1, r0 = m["zero/1e-6"] / m["default/1e-6"], m["zero/0"] / m["default/0"]
        pr = [out["A"]["zero/0"][s]["active"] / out["A"]["default/0"][s]["active"] for s in out["A"]["zero/0"] if s in out["A"]["default/0"]]
        text = ("no verdict (gate)" if r1 > 0.75 else "supported" if r0 >= 0.9 and min(pr) > 0.75 else
                "refuted" if r0 <= 0.75 and max(pr) < 0.9 else "inconclusive")
        out["verdict_A"] = dict(R1=float(r1), R0=float(r0), pairs=[float(x) for x in pr], text=text,
                                S=float((m["default/0"] - m["default/1e-6"]) / (1000 - m["default/1e-6"])))
        print("A:", out["verdict_A"])
    for b in ("ent0.1", "rate0.1"):
        net = [out["B"][b][s]["share"] - out["B"]["none"][s]["share"] for s in out["B"][b] if s in out["B"]["none"]]
        if net:
            out["verdict_" + b] = dict(net=net, text="resurrects" if min(net) >= 0.2 else "does not resurrect" if max(net) <= 0.1 else "partial")
            print(b, out["verdict_" + b])
    if a.reads:
        R, bad, n = json.load(open(a.reads)), 0, 0
        for part in ("A", "B"):
            for g, nets in out[part].items():
                for s, r in nets.items():
                    q = R[part][g][s]
                    checks = dict(active=abs(q["active"] - r["active"]) <= 2, r2=abs(q["r2_clean"] - r["r2_clean"]) <= 1e-4,
                                  vector=q["trace_active"][-1] == r["last_vector"] and q["trace_it"][-1] == r["last_it"])
                    if part == "B":
                        checks["resurrected"] = abs(q["resurrected"] - r["resurrected"]) <= 2
                    n += len(checks)
                    for k, ok in checks.items():
                        if not ok:
                            bad += 1
                            print("DISAGREES", part, g, s, k, {x: q[x] for x in ("active", "r2_clean")}, r)
        out["comparison"] = dict(checks=n, disagreements=bad)
        print(f"comparison with the reader: {n} checks, {bad} disagreements")
    json.dump(out, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
