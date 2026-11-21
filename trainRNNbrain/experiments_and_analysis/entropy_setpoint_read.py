#!/usr/bin/env python3
"""Entropy and correlation set points on 3BFF: the 75,000-iteration read, the verdict and the figure.

RULES. docs/project_trajectory/2026-11-20_2307_rules-entropy-correlation-setpoint.md, committed before any run.

MODE `read` (Della CPU node, where the runs are). For every network of
NBitFlipFlop_entropy_setpoint/EqType=h_N=1000_arm={ent0.01,ent0.1,cor0.01,cor0.1}, of the four default
controls (round18_g1g2.CTRL_CELL) and of the rate penalty at 0.1 (round18_g1g2.FRM_CELL at 0.1):
  - active units at the participation probe nearest 75,000 and held-out R^2 of the weights at 75,000,
    36 draws (round18_g1g2.read_run, the read behind data/round18/g1g2.json);
  - active units at every stored participation vector, joined along the warm-start chain, to 75,000;
  - revival: units inactive at some stored vector after iteration 1,000 and active at the read probe;
  - manipulation check: median over units of |H_i - 0.75| and of |c_i| on the 75,000-iteration weights
    (held-out batch, training noise, noise seed fixed);
  - skipped updates, from the run's Slurm log (the `gradient spikes` line; none printed = 0).
Writes OUT_DIR/entropy_setpoint_reads.json (the R^2 cache is kept in OUT_DIR).

MODE `verdict` (Mac). Prints each arm's networks, the criterion (fig_paper_F5_tasks.rule, tolerance 0.01)
against the controls, the rate penalty beside it, the manipulation check and revival; draws the figure
(active units against iterations, every network, with the controls) to --fig (PDF and PNG).

Usage:
  python entropy_setpoint_read.py read OUT_DIR [--logs DIR]     (Della, repository root on PYTHONPATH)
  python entropy_setpoint_read.py verdict READS_JSON --fig PATH_STEM
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H  # noqa: E402
import fig_paper_F5_tasks as F5T  # noqa: E402
from common import active_mask, warm_start_chain  # noqa: E402
from criticality_read import build_rnn, load_params  # noqa: E402
from read_remedies import SHARED_FIELDS, cfg_of, sel  # noqa: E402
from round18_g1g2 import CTRL_CELL, FIXED_ITER, FRM_CELL, read_run, run_dirs, state_at  # noqa: E402

ARMS = ("ent0.01", "ent0.1", "cor0.01", "cor0.1")
CELL = "NBitFlipFlop_entropy_setpoint/EqType=h_N=1000_arm={arm}"
TARGET_BITS = 0.75          # configs/experiment/entropy_setpoint.yaml, ent_args.target_bits
TARGET_CORR = 0.0           # cor_args.target_corr
STAT_NOISE_SEED = 20261121  # noise draw of the manipulation-check pass
TOL = 0.01
COLORS = {"control": "0.45", "frm0.1": "#9b59b6", "ent0.01": "#7fb3e6", "ent0.1": "#2a78d6",
          "cor0.01": "#f2a07b", "cor0.1": "#eb6834"}


def unit_stats(state_dir):
    """Per-unit on/off entropy and mean pairwise correlation of h on one weight state.

    Args:
        state_dir: folder holding *LastParams* and *_config.yaml of the state.
    Returns:
        dict med_dev_H (median |H_i - TARGET_BITS|, bits), med_abs_c (median |c_i - TARGET_CORR|),
        med_H, med_c (medians), all floats.
    """
    cfg = OmegaConf.load(sorted(glob.glob(os.path.join(state_dir, "*_config.yaml")))[0])
    rnn = build_rnn(cfg, seed=int(cfg.seed))
    params, _ = H._params(state_dir)
    load_params(rnn, params)
    bi = H.heldout_batch(cfg)[0]
    g = rnn.random_generator.get_state()
    rnn.random_generator.manual_seed(STAT_NOISE_SEED)
    with torch.no_grad():
        h, _ = rnn(bi, w_noise=True)
    rnn.random_generator.set_state(g)
    x = h.reshape(h.shape[0], -1).double()
    z = x.mean(1) / x.std(1)
    lp, lq = torch.special.log_ndtr(z), torch.special.log_ndtr(-z)
    Hb = (-(lp.exp() * lp + lq.exp() * lq) / np.log(2.0)).numpy()
    xc = x - x.mean(1, keepdim=True)
    zz = xc / xc.pow(2).mean(1, keepdim=True).sqrt()
    c = (((zz * zz.sum(0)).mean(1) - zz.pow(2).mean(1)) / (zz.shape[0] - 1)).numpy()
    return dict(med_dev_H=float(np.median(np.abs(Hb - TARGET_BITS))), med_abs_c=float(np.median(np.abs(c - TARGET_CORR))),
                med_H=float(np.median(Hb)), med_c=float(np.median(c)))


def chain_trace(run_dir, until=FIXED_ITER):
    """Active units at every stored participation vector, joined along the warm-start chain.

    Args:
        run_dir: a run folder; until: last iteration kept.
    Returns:
        (iterations list, active-count list, list of (N,) bool active masks), sorted by iteration.
    """
    its, acts, masks = [], [], []
    for d in warm_start_chain(run_dir):
        f = glob.glob(os.path.join(d, "*ParticipationTrace.pkl"))
        if not f:
            continue
        tr = pickle.load(open(f[0], "rb"))
        for it, p in zip(tr["participation_iters"], np.asarray(tr["participation"])):
            if it <= until and it not in its:
                m = active_mask(p)
                its.append(float(it)); acts.append(int(m.sum())); masks.append(m)
    o = np.argsort(its)
    return [its[i] for i in o], [acts[i] for i in o], [masks[i] for i in o]


def revived(its, masks, probe):
    """Units inactive at some stored vector after iteration 1,000 and active at the read probe.

    Args:
        its: iterations; masks: (N,) bool active masks at those iterations; probe: the read iteration.
    Returns:
        int count.
    """
    j = int(np.argmin(np.abs(np.asarray(its) - probe)))
    late = [m for i, m in zip(its, masks) if 1000 < i <= its[j]]
    if not late:
        return 0
    ever_off = ~np.logical_and.reduce(late)
    return int((ever_off & masks[j]).sum())


def skipped(run_dir, logs):
    """Skipped updates of one run from its Slurm log, or None when no log names its seed.

    Args:
        run_dir: run folder (its name holds seed=<seed>); logs: folder of *.out files.
    Returns:
        int (0 when the log prints no `gradient spikes` line) or None.
    """
    seed = re.search(r"seed=(\d+)", os.path.basename(run_dir)).group(1)
    for f in glob.glob(os.path.join(logs, "EntSP.*.out")):
        txt = open(f, errors="replace").read()
        if f"'seed': {seed}," in txt:
            m = re.search(r"gradient spikes: (\d+) updates skipped", txt)
            return int(m.group(1)) if m else 0
    return None


def do_read(out_dir, logs):
    """Mode `read`: every network of the four arms, the controls and the rate penalty at 0.1.

    Args:
        out_dir: str, output folder (created); logs: folder of the Slurm logs.
    Returns:
        None.
    """
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "r2cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    c0 = cfg_of(run_dirs(CTRL_CELL)[0])
    ctrl_shared = {k: sel(c0, k) for k in SHARED_FIELDS}
    out = {}
    groups = [("control", CTRL_CELL), ("frm0.1", FRM_CELL.format(lam="0.1"))] + [(a, CELL.format(arm=a)) for a in ARMS]
    for name, cell in groups:
        out[name] = []
        for d in run_dirs(cell):
            rec = read_run(d, cache, ctrl_shared)
            its, acts, masks = chain_trace(d)
            rec.update(trace_it=its, trace_active=acts, revived=revived(its, masks, rec["probe_75k"]) if its else None,
                       stats=unit_stats(rec["state"]) if rec["state"] else None,
                       skipped=skipped(d, logs) if name in ARMS else None, cell=cell)
            out[name].append(rec)
            print(f"{name} {os.path.basename(d)[:50]} active {rec['active_75k']} R2 {rec['r2_75k']} "
                  f"revived {rec['revived']} stats {rec['stats']} skipped {rec['skipped']} "
                  f"config differs {rec['config_differs']}", flush=True)
            json.dump(cache, open(cache_path, "w"))
            json.dump(out, open(os.path.join(out_dir, "entropy_setpoint_reads.json"), "w"), indent=1)


def do_verdict(reads_json, fig_stem):
    """Mode `verdict`: print the criterion per arm and term, the descriptions, and draw the figure.

    Args:
        reads_json: str, the file `read` wrote; fig_stem: str, figure path without extension.
    Returns:
        dict arm -> fig_paper_F5_tasks.rule result.
    """
    R = json.load(open(reads_json))
    ca = [r["active_75k"] for r in R["control"]]
    cr = [r["r2_75k"] for r in R["control"] if r["r2_75k"] is not None]
    print(f"controls: active {ca} (largest {max(ca)}), R2 {np.round(cr, 4).tolist()} mean {np.mean(cr):.4f}; "
          f"R2 bar {np.mean(cr) - TOL:.4f}")
    verdicts = {}
    for name in ("frm0.1",) + ARMS:
        recs = R.get(name, [])
        act = [r["active_75k"] for r in recs]
        r2 = [r["r2_75k"] for r in recs if r["r2_75k"] is not None]
        v = F5T.rule(act, r2, ca, cr, TOL) if recs else dict(works=False, text="no finished network", fold=float("nan"),
                                                               dperf=float("nan"))
        verdicts[name] = v
        print(f"\n{name}{' (description only)' if name == 'frm0.1' else ''}: {v['text']}; fold {v['fold']:.2f}, "
              f"dR2 {v['dperf']:+.4f}")
        for r in recs:
            st = r["stats"] or {}
            print(f"   {os.path.basename(r['run_dir'])[:40]}  active {r['active_75k']} (probe {r['probe_75k']:.0f})  "
                  f"R2 {r['r2_75k']}  revived {r['revived']}  skipped {r['skipped']}  "
                  f"|H-0.75| {st.get('med_dev_H', float('nan')):.3f}  |c| {st.get('med_abs_c', float('nan')):.3f}  "
                  f"config differs {r['config_differs']}")
    cs = [r["stats"] for r in R["control"] if r["stats"]]
    print(f"\ncontrols, manipulation stats: |H-0.75| {[round(s['med_dev_H'], 3) for s in cs]}  "
          f"|c| {[round(s['med_abs_c'], 3) for s in cs]}  revived {[r['revived'] for r in R['control']]}")
    for term, arms in (("entropy (A)", ARMS[:2]), ("correlation (B)", ARMS[2:])):
        print(f"{term}: {'PASSES' if any(verdicts[a]['works'] for a in arms) else 'fails'} on 3BFF "
              f"({', '.join(f'{a}: ' + ('passes' if verdicts[a]['works'] else 'fails') for a in arms)})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
    for ax, arms, title in ((axes[0], ARMS[:2], "(A) entropy set point"), (axes[1], ARMS[2:], "(B) correlation set point")):
        for name in ("control", "frm0.1") + arms:
            for k, r in enumerate(R.get(name, [])):
                lab = {"control": "default (controls)", "frm0.1": "rate penalty, λ = 0.1"}.get(name, f"λ = {name[3:]}")
                ax.plot(np.asarray(r["trace_it"]) / 1000, r["trace_active"], color=COLORS[name],
                        lw=1.2 if name in ("control", "frm0.1") else 1.6, alpha=0.9, label=lab if k == 0 else None)
        ax.axhline(max(ca), color="0.45", lw=0.8, ls=":")
        ax.set_xlim(0, 75)
        ax.set_ylim(0, 1000)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("iteration (thousands)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    axes[0].set_ylabel("active units (of 1,000)")
    fig.savefig(fig_stem + ".pdf")
    fig.savefig(fig_stem + ".png", dpi=150)
    print(f"figure: {fig_stem}.pdf, .png")
    return verdicts


def main():
    """Parse the command line and run the selected mode."""
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("read", "verdict"))
    ap.add_argument("path")
    ap.add_argument("--logs", default=os.path.expanduser("~/trainRNNbrain_logs"))
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    if a.mode == "read":
        do_read(a.path, a.logs)
    else:
        do_verdict(a.path, a.fig)


if __name__ == "__main__":
    main()
