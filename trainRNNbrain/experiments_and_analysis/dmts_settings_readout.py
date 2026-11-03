#!/usr/bin/env python3
"""DMTS standard-setting sweeps (DMTS_std_settings, Della job 16628171): read-out and pass/fail rule.

Question (entry 2026-11-01_2124_dmts-sweeps-plan): does any of eight standard settings stop the
decrease of active units after the floor crossing on DMTS, as Results 3 found for none on CDDM and
3BFF? 27 networks: the default and one value of each of eight settings, 3 networks per arm, N = 1000,
150,000 iterations, a kept snapshot every 1,000 iterations.

Read-out, as the plan fixes it and with the code path of the CDDM and 3BFF sweeps:
  - floor crossing t*: `common.read_run` (1.05x the run's own noisy-loss floor, DMTS fitted from its
    escape, the standard exclusions with their reason);
  - active units: the Trainer's participation trace (noise-free probe, vector every 100 iterations),
    `common.active_mask` (p >= 0.05 q95(p)); at the probe nearest t* (primary), the last probe at or
    before t* (variant), the probe nearest 150,000 (primary end), the median over the probes of the
    last 2% of training (variant end), and the probe nearest 75,000;
  - held-out R^2: `heldout_r2.heldout_r2` (36 seeded draws, CPU) on the final weights (150,000), on
    the kept snapshot nearest t* and on the snapshot at 75,000;
  - learned: held-out R^2 at 150,000 >= 0.8; a network that did not learn is excluded and counted.
Rule: every learned network with a crossing must have fewer active units at the end than at the
crossing (primary comparison). One counterexample fails the extension to DMTS; an arm with no such
network is untested.

Independent check of the two counts the rule compares (not part of the plan's rule; threshold set
before the run): active units recomputed from the saved weights (snapshot nearest t*, final weights)
with a numpy forward pass written here, on a noise-free DMTS batch, sharing only the task's input
generator with the production path. Pass: within max(5 units, 3%) of the trace's count at both
points, and the same sign of (end - crossing) in every network that enters the rule.

Usage:
  python dmts_settings_readout.py collect <arm> [<arm> ...]   # on Della (CPU); data/dmts_settings/<arm>.pkl
  python dmts_settings_readout.py report                      # on the Mac; tables, rule, figure
"""
import glob
import os
import pickle
import re
import sys

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paperstyle as ps
from common import DATA_DIR, load_run, read_run, SNAP_DIR
from heldout_r2 import R2_DRAWS, heldout_r2
from trainRNNbrain.tasks.TaskDMTS import TaskDMTS
from trainRNNbrain.training.training_utils import prepare_task_arguments

SWEEP = "DMTS_std_settings"
ARMS = ["default", "wd0", "sigrec0.1", "met1", "sr0.5", "rwm0.2", "softplus25", "rws0.05", "winp20"]
ARM_LABEL = {"default": "default", "wd0": "weight decay 0", "sigrec0.1": "recurrent noise 0.1",
             "met1": "metabolic penalty 1", "sr0.5": "spectral radius 0.5",
             "rwm0.2": "rec.-weight penalty 0.2", "softplus25": "softplus, beta 25",
             "rws0.05": "sparsity penalty 0.05", "winp20": "input-weight scale 20"}
BUDGET = 150_000
FIXED_IT = 75_000
LEARNED_R2 = 0.8          # the paper's rule for DMTS
LAST_FRAC = 0.02          # variant end: median over the last 2% of training
SNAP_EVERY = 1000
OUT_DIR = os.path.normpath(os.path.join(DATA_DIR, "..", "dmts_settings"))
INDEP_TOL_UNITS, INDEP_TOL_FRAC = 5, 0.03


def arm_of(run_dir):
    """The arm name in a run's cell folder (`..._arm=<arm>_iters=...`)."""
    return re.search(r"_arm=(.+?)_iters=", os.path.basename(os.path.dirname(run_dir))).group(1)


def seed_of(run_dir):
    """The seed in a run folder name (`...;seed=<seed>;...`), as an int."""
    return int(re.search(r"seed=(\d+)", os.path.basename(run_dir)).group(1))


def snapshot_dir(run_dir, it):
    """The kept snapshot folder of a run at iteration `it`, the run folder itself at the budget.

    Args:
        run_dir: a trained-network folder; it: a multiple of SNAP_EVERY in [SNAP_EVERY, BUDGET].
    Returns:
        path of the folder holding that iteration's *LastParams* and config, or None if missing.
    """
    if it >= BUDGET:
        return run_dir
    d = os.path.join(SNAP_DIR, SWEEP, os.path.basename(os.path.dirname(run_dir)),
                     f"s{seed_of(run_dir)}", f"it{it:07d}")
    return d if os.path.isdir(d) else None


def nearest_snapshot_it(t):
    """The kept-snapshot iteration nearest t (snapshots every SNAP_EVERY, final weights at BUDGET)."""
    return int(min(max(round(t / SNAP_EVERY), 1) * SNAP_EVERY, BUDGET))


def indep_active(weights_dir):
    """Active units of saved weights, from a numpy forward pass written independently of RNN_torch.

    Dynamics (equation h, dt/tau = alpha): h <- h + alpha (-h + W_rec f(h) + W_inp u + b), started at
    y_init, noise-free, on the run's own noise-free DMTS batch (its config's task, seeded as saved).
    Participation p = std(f(h)) + q0.9(|f(h)|) over every (time, trial) sample; active if
    p >= 0.05 q95(p).

    Args:
        weights_dir: a folder holding one *LastParams*.npz and one *_config.yaml.
    Returns:
        int number of active units.
    """
    z = np.load(glob.glob(os.path.join(weights_dir, "*LastParams*.npz"))[0], allow_pickle=True)
    cfg = OmegaConf.load(glob.glob(os.path.join(weights_dir, "*_config.yaml"))[0])
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    task = TaskDMTS(**{k: v for k, v in dict(tcfg).items() if k != "_target_"})
    u = np.asarray(task.get_batch()[0], np.float64)                     # (n_in, T, B)
    W_rec = z["W_rec"].astype(np.float64)
    W_inp = z["W_inp"].astype(np.float64)
    b = z["bias"].astype(np.float64)[:, None]
    alpha = float(z["dt"]) / float(z["tau"])
    act = z["activation_args"].item()
    if act["name"] == "relu":
        f = lambda x: np.maximum(x, 0.0)
    elif act["name"] == "softplus":
        beta, slope = float(act["beta"]), float(act.get("slope", 1.0))
        # torch Softplus: (1/beta) log(1 + exp(beta x)), linear where beta x > 20
        f = lambda x: np.where(beta * slope * x > 20, slope * x,
                               np.log1p(np.exp(np.minimum(beta * slope * x, 20))) / beta)
    else:
        raise ValueError(act["name"])
    T, B = u.shape[1], u.shape[2]
    h = np.repeat(z["y_init"].astype(np.float64)[:, None], B, axis=1)
    rates = np.empty((W_rec.shape[0], T, B))
    rates[:, 0] = f(h)
    for t in range(1, T):
        h = h + alpha * (-h + W_rec @ f(h) + W_inp @ u[:, t - 1] + b)
        rates[:, t] = f(h)
    r = rates.reshape(rates.shape[0], -1)
    p = r.std(axis=1) + np.quantile(np.abs(r), 0.9, axis=1)
    return int((p >= 0.05 * np.quantile(p, 0.95)).sum())


def collect_run(run_dir):
    """Every read-out quantity of one network.

    Args:
        run_dir: a trained-network folder of the sweep.
    Returns:
        dict: arm, seed, name; t_star, t_stars, reason, floor_kind, escape, covered; it_act, active
        (trace, float per probe); act_cross (probe nearest t*), probe_cross, act_cross_before (last
        probe <= t*), act_end (probe nearest BUDGET), probe_end, act_end_med (median over the last
        LAST_FRAC), act_75k; r2_end, r2_cross (snapshot nearest t*), it_r2_cross, r2_75k;
        indep_cross, indep_end (indep_active at the snapshot nearest t* and at BUDGET).
    """
    rec = load_run(run_dir)
    rd = read_run(run_dir, run=rec)
    it_a, act = np.asarray(rec["it_act"], float), np.asarray(rec["active"], float)
    j_end = int(np.argmin(np.abs(it_a - BUDGET)))
    j_75 = int(np.argmin(np.abs(it_a - FIXED_IT)))
    cache = {}                                    # in memory only: never write the shared cache
    out = dict(arm=arm_of(run_dir), seed=seed_of(run_dir), name=os.path.basename(run_dir),
               t_star=float(rd["t_star"]), t_stars=rd["t_stars"], reason=rd["reason"],
               floor_kind=rd["floor_kind"], escape=float(rd["escape"]), covered=rd["covered"],
               it_act=it_a, active=act, act_end=float(act[j_end]), probe_end=float(it_a[j_end]),
               act_end_med=float(np.median(act[it_a >= BUDGET * (1 - LAST_FRAC)])),
               act_75k=float(act[j_75]), act_cross=None, probe_cross=None, act_cross_before=None,
               r2_end=heldout_r2(run_dir, draws=R2_DRAWS, cache=cache),
               r2_75k=heldout_r2(snapshot_dir(run_dir, FIXED_IT), draws=R2_DRAWS, cache=cache),
               r2_cross=None, it_r2_cross=None, indep_cross=None,
               indep_end=indep_active(run_dir))
    if rd["reason"] is None:
        T = rd["t_star"]
        before = np.flatnonzero(it_a <= T)
        k = nearest_snapshot_it(T)
        out.update(act_cross=float(rd["active"]), probe_cross=float(rd["probe_iter"]),
                   act_cross_before=float(act[before[-1]]) if len(before) else None,
                   it_r2_cross=k,
                   r2_cross=heldout_r2(snapshot_dir(run_dir, k), draws=R2_DRAWS, cache=cache),
                   indep_cross=indep_active(snapshot_dir(run_dir, k)))
    return out


def collect(arms):
    """Read every network of the named arms and write one pickle per arm to OUT_DIR."""
    os.makedirs(OUT_DIR, exist_ok=True)
    for arm in arms:
        runs = sorted(glob.glob(os.path.join(DATA_DIR, SWEEP, f"EqType=h_N=1000_arm={arm}_iters={BUDGET}", "*")))
        rows = [collect_run(d) for d in runs if os.path.isdir(d)]
        with open(os.path.join(OUT_DIR, f"{arm}.pkl"), "wb") as fh:
            pickle.dump(rows, fh)
        print(arm, len(rows), "networks", flush=True)


def load_rows():
    """Every collected network, in ARMS order (one list of collect_run dicts)."""
    rows = []
    for arm in ARMS:
        with open(os.path.join(OUT_DIR, f"{arm}.pkl"), "rb") as fh:
            rows += pickle.load(fh)
    return rows


def status(r):
    """'rule' if the network enters the pass/fail rule, else why it does not."""
    if r["r2_end"] < LEARNED_R2:
        return "not learned"
    if r["reason"] is not None:
        return f"no crossing ({r['reason']})"
    if not r["covered"]:
        return "trace does not cover the crossing"
    return "rule"


def slope_last_third(r):
    """Change of the active count per 10,000 iterations over the last third of training (OLS)."""
    m = r["it_act"] >= BUDGET * 2 / 3
    return float(np.polyfit(r["it_act"][m], r["active"][m], 1)[0] * 1e4)


def fmt(v, f="{:.0f}"):
    """A number formatted with f, or a dash for None/nan."""
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)


def report():
    """Print the per-network, per-arm and overall tables and the verdict; draw the figure."""
    rows = load_rows()
    print("| arm | seed | R2 end | learned | crossing | t*1.10 | t*1.20 | act@cross | act@<=cross | act@75k "
          "| act@end | act end med2% | end-cross | R2@cross (it) | R2@75k | slope/10k | indep cross/end | status |")
    print("|" + "---|" * 18)
    for r in rows:
        s = status(r)
        d = r["act_end"] - r["act_cross"] if s == "rule" else None
        t10, t20 = (r["t_stars"] or {}).get(0.1), (r["t_stars"] or {}).get(0.2)
        print(f"| {r['arm']} | {r['seed']} | {r['r2_end']:.4f} | {'yes' if r['r2_end'] >= LEARNED_R2 else 'no'} "
              f"| {fmt(r['t_star'] if r['reason'] is None else None)} | {fmt(t10)} | {fmt(t20)} "
              f"| {fmt(r['act_cross'])} | {fmt(r['act_cross_before'])} | {fmt(r['act_75k'])} | {fmt(r['act_end'])} "
              f"| {fmt(r['act_end_med'])} | {fmt(d, '{:+.0f}')} "
              f"| {fmt(r['r2_cross'], '{:.4f}')} ({fmt(r['it_r2_cross'])}) | {r['r2_75k']:.4f} "
              f"| {slope_last_third(r):+.1f} | {fmt(r['indep_cross'])}/{r['indep_end']} | {s} |")
    print("\nkeys of t_stars:", list(rows[0]["t_stars"]))
    print("reasons:", sorted({str(r["reason"]) for r in rows}))
    print("end probes:", sorted({r["probe_end"] for r in rows}))

    # independent check
    bad = []
    for r in rows:
        tol = lambda n: max(INDEP_TOL_UNITS, INDEP_TOL_FRAC * n)
        if abs(r["indep_end"] - r["act_end"]) > tol(r["act_end"]):
            bad.append((r["arm"], r["seed"], "end", r["indep_end"], r["act_end"]))
        if r["indep_cross"] is not None and abs(r["indep_cross"] - r["act_cross"]) > tol(r["act_cross"]):
            bad.append((r["arm"], r["seed"], "cross", r["indep_cross"], r["act_cross"]))
        if status(r) == "rule" and np.sign(r["indep_end"] - r["indep_cross"]) != np.sign(r["act_end"] - r["act_cross"]):
            bad.append((r["arm"], r["seed"], "sign", r["indep_end"] - r["indep_cross"], r["act_end"] - r["act_cross"]))
    print("\nindependent check:", "PASS" if not bad else f"FAIL {bad}")

    # rule, per arm and overall
    print("\n| arm | networks | learned | with crossing (enter rule) | mean crossing | act@cross | act@end "
          "| R2 end | fewer at end (primary / <=cross / med2%) | verdict |")
    print("|" + "---|" * 10)
    counter = []
    for arm in ARMS:
        a = [r for r in rows if r["arm"] == arm]
        ru = [r for r in a if status(r) == "rule"]
        prim = [r["act_end"] < r["act_cross"] for r in ru]
        v_before = [r["act_end"] < r["act_cross_before"] for r in ru]
        v_med = [r["act_end_med"] < r["act_cross"] for r in ru]
        counter += [(arm, r["seed"]) for r, ok in zip(ru, prim) if not ok]
        verdict = "untested" if not ru else ("pass" if all(prim) else "FAIL")
        r2s = ", ".join(f"{r['r2_end']:.3f}" for r in a)
        print(f"| {arm} | {len(a)} | {sum(r['r2_end'] >= LEARNED_R2 for r in a)} | {len(ru)} "
              f"| {fmt(np.mean([r['t_star'] for r in ru]) if ru else None)} "
              f"| {', '.join(fmt(r['act_cross']) for r in ru) or '-'} | {', '.join(fmt(r['act_end']) for r in ru) or '-'} "
              f"| {r2s} "
              f"| {sum(prim)}/{len(ru)} / {sum(v_before)}/{len(ru)} / {sum(v_med)}/{len(ru)} | {verdict} |")
    n_rule = sum(status(r) == "rule" for r in rows)
    print(f"\noverall: {n_rule} networks enter the rule; counterexamples: {counter or 'none'}")
    draw(rows)


def draw(rows):
    """Active units over training, one panel per arm, floor crossings marked (img/internal_figures)."""
    ps.setup()
    fig = ps.figure(h_mm=150, w_mm=ps.W_FULL_MM)
    gs = fig.add_gridspec(3, 3, left=0.08, right=0.98, bottom=0.07, top=0.95, hspace=0.45, wspace=0.28)
    cols = ps.SLOTS[:3]
    for i, arm in enumerate(ARMS):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        for j, r in enumerate([r for r in rows if r["arm"] == arm]):
            learned = r["r2_end"] >= LEARNED_R2
            c = cols[j] if learned else ps.INACTIVE
            ax.plot(r["it_act"] / 1e3, r["active"], color=c, lw=ps.THIN_LW,
                    ls="-" if learned else "--",
                    label=f"s{r['seed']}, R² {r['r2_end']:.2f}" + ("" if learned else " (not learned)"))
            if status(r) == "rule":
                ax.plot(r["probe_cross"] / 1e3, r["act_cross"], "o", ms=ps.MARKER_PT + 1, color=c,
                        mec="k", mew=0.4, zorder=5)
        ax.axvline(FIXED_IT / 1e3, color=ps.BASELINE, lw=ps.THIN_LW, ls=":")
        ax.set_title(ARM_LABEL[arm], fontsize=ps.LABEL_PT)
        ax.set_xlim(0, BUDGET / 1e3)
        ax.set_ylim(0, None)
        ax.legend(fontsize=ps.TICK_PT - 1, frameon=False, loc="upper right", handlelength=1.2)
        ps.despine(ax)
        if i // 3 == 2:
            ax.set_xlabel("iteration (thousands)")
        if i % 3 == 0:
            ax.set_ylabel("active units (of 1,000)")
    fig.suptitle("DMTS setting sweeps, N = 1,000: active units over training; "
                 "filled circle = floor crossing (1.05x floor); dotted line = 75,000", fontsize=ps.LABEL_PT)
    ps.save(fig, "dmts_settings_active_over_training")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2:] or ARMS)
    else:
        report()
