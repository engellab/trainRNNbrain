#!/usr/bin/env python3
"""How fast does Adam's coupled weight decay shrink a weight, and how far do weights move? (mechanism tests, part C)

Question. A reviewer estimated, from saved Adam states, that the default decay of 1e-6 alone shrinks a weight onto an
inactive unit e-fold in 350 to 500 iterations. This script recomputes the estimate from the Adam states (independent code)
and then MEASURES the time constant from weights saved during training, where the loss gradient and the changing gradient
size act too.

Mode `adam` (one saved network with an AdamState file). With v, m Adam's bias-corrected moments of one weight, lr and eps the
optimizer's values and lam = 1e-6:
  gradient size  sqrt(v);   step  |m| / (sqrt(v) + eps), in units of lr;   predicted decay time  (sqrt(v) + eps) / (lr lam).
Medians over W_rec by class of the postsynaptic and the presynaptic unit: A active, I inactive (common.active_mask on the last
stored participation vector). Also the median incoming row norm of W_rec, inactive / active.

Mode `weights` (one network with kept snapshots). For base iterations t and lags D:
  slope b(D) of w(t+D) on w(t) through the origin, over the weights of one class, classes fixed at t and t+D;
  measured decay time  -D / ln b  (a pure exponential decay gives b = exp(-D/tau); added uncorrelated change leaves b unchanged).
  norm of the change of a unit's incoming weights, ||(W_rec, W_inp) row (t+D) - row (t)||, median over the units of a class
  (active; low = inactive but participation >= 1e-6; silent = participation < 1e-6, at t), and, at the first base iteration,
  the norm of the task-loss gradient on the same rows (rms over GRAD_BATCHES training batches under the training noise).
The forward pass and the noise are mech_removal's, not RNN_torch's.

Mode `summary`: the part C verdicts of the rules (entry 2026-11-25_1311) from a folder of the JSON files above.

Usage:  python mech_optimizer.py adam OUT_JSON RUN_DIR [RUN_DIR ...]
        python mech_optimizer.py weights OUT_JSON RUN_DIR SNAP_DIR --base 25000 50000 --lags 500 1000 2000 5000 5500
        python mech_optimizer.py summary OUT_JSON JSON_DIR
"""
import argparse
import glob
import json
import os
import pickle
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import active_mask  # noqa: E402
from mech_removal import forward, load, noise_draw  # noqa: E402
from trainRNNbrain.training.training_utils import get_training_mask, prepare_task_arguments  # noqa: E402

LAM = 1e-6              # the default weight decay, also used for the what-if of networks trained without it
SILENT = 1e-6           # participation below which a unit is silent (the trace's silent_1em6)
GRAD_BATCHES = 4
GRAD_SEED = 47000       # noise seed of gradient batch k is GRAD_SEED + k


def only(folder, pattern):
    """The single file of `folder` matching `pattern`; raises if there is not exactly one."""
    f = glob.glob(os.path.join(glob.escape(folder), pattern))
    if len(f) != 1:
        raise FileNotFoundError(f"{len(f)} files match {pattern} in {folder}")
    return f[0]


def trace(run_dir):
    """Stored participation vectors of a run.

    Args:
        run_dir: a run folder.
    Returns:
        dict iteration (int) -> (N,) float32 participation vector.
    """
    with open(only(run_dir, "*ParticipationTrace.pkl"), "rb") as fh:
        t = pickle.load(fh)
    return {int(i): np.asarray(p) for i, p in zip(t["participation_iters"], t["participation"])}


def block_medians(x, act):
    """Median of a (post, pre) matrix in the four blocks AA, IA, AI, II (post first).

    Args:
        x: (N, N) array; act: (N,) bool, active units.
    Returns:
        dict block -> float.
    """
    rows = {"A": act, "I": ~act}
    return {a + b: float(np.median(x[np.ix_(rows[a], rows[b])])) for a in "AI" for b in "AI"}


def do_adam(runs):
    """Mode `adam`, every run.

    Args:
        runs: list of run folders.
    Returns:
        list of dicts, one per run.
    """
    out = []
    for d in runs:
        cfg = OmegaConf.load(only(d, "*_config.yaml"))
        st = torch.load(only(d, "*AdamState*.pt"), map_location="cpu", weights_only=True)
        g = st["param_groups"][0]
        N = int(cfg.model.N)
        s = [x for x in st["state"].values() if tuple(x["exp_avg"].shape) == (N, N)][0]
        n = float(s["step"])
        m = s["exp_avg"].double().numpy() / (1 - g["betas"][0] ** n)
        root_v = np.sqrt(s["exp_avg_sq"].double().numpy() / (1 - g["betas"][1] ** n))
        tr = trace(d)
        p = tr[max(tr)]
        act = active_mask(p)
        (Wr, _, _, _), _ = load(d)
        row = np.linalg.norm(Wr.numpy().astype(np.float64), axis=1)
        grad, step = block_medians(root_v, act), block_medians(np.abs(m) / (root_v + g["eps"]), act)
        tau = block_medians((root_v + g["eps"]) / (g["lr"] * LAM), act)
        rec = dict(run_dir=d, task=str(cfg.task.taskname), N=N, optimizer=str(cfg.trainer.get("optimizer", "adam")),
                   weight_decay=float(g["weight_decay"]), sigma_rec=float(cfg.model.sigma_rec), lr=float(g["lr"]),
                   step=int(n), trace_it=max(tr), active=int(act.sum()), silent=int((p < SILENT).sum()),
                   grad=grad, adam_step=step, tau=tau, grad_ratio=grad["AA"] / max(grad["IA"], 1e-300),
                   step_ratio=step["AA"] / max(step["IA"], 1e-300),
                   row_ratio=float(np.median(row[~act]) / np.median(row[act])), row_active=float(np.median(row[act])),
                   row_inactive=float(np.median(row[~act])))
        out.append(rec)
        print(f"{rec['task']} N={N} {rec['optimizer']} wd={rec['weight_decay']:g} sigma={rec['sigma_rec']:g} "
              f"active {rec['active']} | gradient AA/IA {rec['grad_ratio']:.1f}, step AA/IA {rec['step_ratio']:.2f} "
              f"(IA/AA {1 / rec['step_ratio']:.2f}) | decay time AA {tau['AA']:.0f} IA {tau['IA']:.0f} AI {tau['AI']:.0f} "
              f"II {tau['II']:.0f} | row ratio {rec['row_ratio']:.3f}", flush=True)
    return out


def row_gradients(snap_dir, run_dir):
    """Norm of the task-loss gradient on each unit's incoming weights, rms over GRAD_BATCHES noisy training batches.

    Args:
        snap_dir: folder with the weights; run_dir: the run, for its saved config.
    Returns:
        (N,) float64 array.
    """
    W, _ = load(snap_dir)
    cfg = OmegaConf.load(only(run_dir, "*_config.yaml"))
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    tcfg.seed = 12345 if tcfg.get("seed", None) is None else int(tcfg.seed)
    task = hydra.utils.instantiate(tcfg)
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
    alpha = float(cfg.model.dt) / float(cfg.model.tau)
    sig = [float(cfg.model[k]) for k in ("sigma_rec", "sigma_inp", "sigma_out")]
    acc = 0.0
    for k in range(GRAD_BATCHES):
        u, y, _ = task.get_batch()
        u, y = (torch.from_numpy(np.asarray(a, dtype=np.float32)) for a in (u, y))
        Wr, Wi = W[0].clone().requires_grad_(True), W[1].clone().requires_grad_(True)
        z = forward((Wr, Wi, W[2], W[3]), u, alpha, noise_draw(GRAD_SEED + k, Wr.shape[0], u.shape[0], u.shape[1], u.shape[2], alpha, sig))
        (z[:, mask] - y[:, mask]).pow(2).mean().backward()
        acc = acc + (Wr.grad.double().pow(2).sum(1) + Wi.grad.double().pow(2).sum(1)).numpy()
    return np.sqrt(acc / GRAD_BATCHES)


def do_weights(run_dir, snap_dir, bases, lags):
    """Mode `weights`, one network.

    Args:
        run_dir: the run folder (trace, config); snap_dir: its it<iteration>/ snapshots; bases, lags: ints.
    Returns:
        dict with one record per (base, lag) and the gradient norms at the first base.
    """
    def weights_at(it):
        """(W_rec, W_inp) at iteration `it` as float64 arrays."""
        (Wr, Wi, _, _), _ = load(os.path.join(snap_dir, f"it{it:07d}"))
        return Wr.double().numpy(), Wi.double().numpy()

    tr = trace(run_dir)
    out = dict(run_dir=run_dir, snap_dir=snap_dir, records=[])
    for k, t in enumerate(bases):
        Wr0, Wi0 = weights_at(t)
        p0 = tr[t]
        a0 = active_mask(p0)
        cls = dict(active=a0, low=~a0 & (p0 >= SILENT), silent=p0 < SILENT)
        if k == 0:
            g = row_gradients(os.path.join(snap_dir, f"it{t:07d}"), run_dir)
            out["gradient"] = dict(base=t, **{c: float(np.median(g[ix])) if ix.any() else None for c, ix in cls.items()},
                                   inactive=float(np.median(g[~a0])))
        for D in lags:
            if not os.path.isdir(os.path.join(snap_dir, f"it{t + D:07d}")) or (t + D) not in tr:
                continue
            Wr1, Wi1 = weights_at(t + D)
            a1 = active_mask(tr[t + D])
            A, I = a0 & a1, ~a0 & ~a1
            slope = {}
            for name, post, pre in (("AA", A, A), ("IA", I, A), ("AI", A, I), ("II", I, I)):
                w0, w1 = Wr0[np.ix_(post, pre)], Wr1[np.ix_(post, pre)]
                slope[name] = float((w0 * w1).sum() / (w0 * w0).sum())
            change = np.sqrt(((Wr1 - Wr0) ** 2).sum(1) + ((Wi1 - Wi0) ** 2).sum(1))
            norm0 = np.sqrt((Wr0 ** 2).sum(1) + (Wi0 ** 2).sum(1))
            rec = dict(base=t, lag=D, n=dict(active=int(a0.sum()), low=int(cls["low"].sum()), silent=int(cls["silent"].sum())),
                       slope=slope, tau={q: (float(-D / np.log(b)) if 0 < b < 1 else None) for q, b in slope.items()},
                       change={c: float(np.median(change[ix])) if ix.any() else None for c, ix in cls.items()},
                       change_inactive=float(np.median(change[~a0])),
                       row_norm={c: float(np.median(norm0[ix])) if ix.any() else None for c, ix in cls.items()})
            out["records"].append(rec)
            print(f"{os.path.basename(snap_dir)} t={t} D={D} n={rec['n']} slope "
                  + " ".join(f"{q} {b:.3f}" for q, b in slope.items())
                  + f" | change active {rec['change']['active']:.4f} inactive {rec['change_inactive']:.4f}", flush=True)
    return out


def within(x, lo, hi, slack=0.1):
    """Is x inside [lo, hi] widened by `slack` on each side?"""
    return lo * (1 - slack) <= x <= hi * (1 + slack)


def summarise(folder):
    """The part C verdicts, with the bars of the rules entry.

    Args:
        folder: directory holding adam_*.json and weights_*.json.
    Returns:
        dict of the verdicts' numbers and texts.
    """
    def js(pattern):
        """All JSON files of `folder` matching `pattern`, as name -> content."""
        return {os.path.basename(f): json.load(open(f)) for f in sorted(glob.glob(os.path.join(folder, pattern)))}

    out = {}
    adam = [r for v in js("adam_*.json").values() for r in (v if isinstance(v, list) else [v])]
    ctrl = [r for r in adam if "CDDM_paper_grid" in r["run_dir"]]
    ctrl1 = [r for r in ctrl if r["N"] == 1000]
    row = lambda q, rs: [q(r) for r in rs]
    out["C1"] = {
        "gradient ratio AA/IA": (row(lambda r: r["grad_ratio"], ctrl1), 28, 52),
        "step ratio IA/AA (nine controls)": (row(lambda r: 1 / r["step_ratio"], ctrl), 0.15, 0.34),
        "decay time AA": (row(lambda r: r["tau"]["AA"], ctrl1), 13000, 18500),
        "decay time IA": (row(lambda r: r["tau"]["IA"], ctrl1), 345, 496),
        "decay time II": (row(lambda r: r["tau"]["II"], ctrl1), 31, 54),
        "row ratio": (row(lambda r: r["row_ratio"], ctrl1), 0.17, 0.20)}
    for k, (v, lo, hi) in out["C1"].items():
        out["C1"][k] = dict(values=[round(float(x), 4) for x in v], range=[lo, hi], replicated=bool(v) and all(within(x, lo, hi) for x in v))

    def tau_of(pattern, lags=(500, 1000)):
        """Measured IA decay times of every (network, base, lag); None entries (slope not in (0, 1)) are kept as None."""
        return {n: {(r["base"], r["lag"]): r["tau"]["IA"] for r in w["records"] if r["lag"] in lags} for n, w in js(pattern).items()}

    for name, pat in (("C2 absorb default", "weights_absorb_default_*.json"), ("C2 c1 control", "weights_c1control_*.json"),
                      ("absorb zero", "weights_absorb_zero_*.json"), ("new default/1e-6", "weights_noise=default_wd=1e-6_*.json"),
                      ("new default/0", "weights_noise=default_wd=0_*.json"), ("new zero/1e-6", "weights_noise=zero_wd=1e-6_*.json"),
                      ("new zero/0", "weights_noise=zero_wd=0_*.json"), ("3BFF", "weights_3bff_*.json")):
        t = tau_of(pat)
        v = [x for d in t.values() for x in d.values() if x is not None]
        out[name] = dict(n=sum(len(d) for d in t.values()), n_decay=len(v), median=float(np.median(v)) if v else None,
                         lag500=float(np.median([x for d in t.values() for (b, L), x in d.items() if L == 500 and x is not None])) if v else None,
                         per_network={n: {f"{b}+{L}": (None if x is None else round(x, 1)) for (b, L), x in d.items()} for n, d in t.items()})
    m = out["C2 absorb default"]["median"]
    out["C2"] = None if m is None else ("SUPPORTED" if 115 <= m <= 1500 else "REFUTED" if m > 5000 else "RIGHT DIRECTION, WRONG SIZE")
    pd = [r["tau"]["IA"] for r in adam if "CDDM_absorb_noise" in r["run_dir"] and r["sigma_rec"] > 0]
    pz = [r["tau"]["IA"] for r in adam if "CDDM_absorb_noise" in r["run_dir"] and r["sigma_rec"] == 0]
    mz, md = out["absorb zero"]["median"], out["C2 absorb default"]["median"]
    out["C3"] = dict(predicted_default=pd, predicted_zero=pz, predicted_ratio=float(np.median(pz) / np.median(pd)) if pd and pz else None,
                     measured_ratio=None if not (mz and md) else mz / md)
    rr = [out["C3"]["predicted_ratio"], out["C3"]["measured_ratio"]]
    out["C3"]["text"] = None if None in rr else ("REFUTED" if max(rr) >= 1 else "SUPPORTED" if max(rr) < 0.5 else "PARTIAL")
    out["C4"] = {}
    for pat in ("weights_c1control_*.json", "weights_3bff_*.json"):
        for n, w in js(pat).items():
            g = w["gradient"]
            ch = {r["lag"]: r["change"]["active"] / r["change_inactive"] for r in w["records"] if r["base"] == g["base"] and r["lag"] in (500, 5500)}
            out["C4"][n] = dict(gradient_ratio=g["active"] / g["inactive"], change_ratio=ch)
    cd = [v for n, v in out["C4"].items() if "c1control" in n]
    sup = [all(x <= 10 and x <= v["gradient_ratio"] / 3 for x in v["change_ratio"].values()) for v in cd]
    ref = [all(x >= v["gradient_ratio"] / 2 for x in v["change_ratio"].values()) for v in cd]
    out["C4 verdict"] = None if len(cd) < 3 else "SUPPORTED" if all(sup) else "REFUTED" if all(ref) else "INCONCLUSIVE"
    print(json.dumps(out, indent=1))
    return out


def main():
    """Command line: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("adam", "weights", "summary"))
    ap.add_argument("out_json")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--base", type=int, nargs="+", default=[25000, 50000, 69000])
    ap.add_argument("--lags", type=int, nargs="+", default=[500, 1000, 2000, 5000, 5500])
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    res = (do_adam(a.dirs) if a.mode == "adam" else summarise(a.dirs[0]) if a.mode == "summary"
           else do_weights(a.dirs[0], a.dirs[1], a.base, a.lags))
    with open(a.out_json, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
