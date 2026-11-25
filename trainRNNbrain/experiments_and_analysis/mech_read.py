#!/usr/bin/env python3
"""Mechanism tests, parts A and B: the read and the verdicts (rules entry of 25 Nov 2026, docs/project_trajectory/).

Part A, training noise x Adam's coupled weight decay on CDDM (four cells, three seeds each). Part B, a penalty switched on in
the three default networks of part A (arms none, ent0.1, rate0.1). Every bar is a constant below and was written in the rules
entry before any run.

MODE `read` (Della CPU node). Per finished network, on its FINAL weights: active, low-active and silent units on a noise-free
pass over the training grid (mech_removal.participation); held-out R^2 noise-free and under the default training noise
(mech_removal.forward, 36 draws) and, for the default-noise networks, heldout_r2 (the project's measure); incoming row norms;
the active count along training from the stored participation vectors. Part B networks also get the unit classes at the
switch, recomputed from the parent's final weights, and the share of each class that is active at the end under two
thresholds (the network's own 95th percentile, and the 95th percentile at the switch).
MODE `grad` (Della CPU node). At the switch weights of the three part B parents: the gradient of each penalty (and of the task
loss) with respect to every unit's incoming weights, on a noise-free and on a noisy training batch, with the production
penalties (Trainer.Penalties).
MODE `verdict` (any CPU). Prints the tables and the verdicts from the two JSON files and the removal JSONs, and draws the figure.

Usage:
  python mech_read.py read OUT_JSON         python mech_read.py grad OUT_JSON       (repository root on PYTHONPATH)
  python mech_read.py verdict READ_JSON --removal DIR --fig STEM
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H  # noqa: E402
import mech_removal as MR  # noqa: E402
from common import DATA_DIR, active_mask  # noqa: E402
from criticality_read import build_rnn, load_params  # noqa: E402
from mech_optimizer import SILENT, only, trace  # noqa: E402
from trainRNNbrain.trainer.Trainer import Penalties  # noqa: E402

SEEDS = ("1153179695", "2873231781", "2425511753")
A_CELLS = [(n, w) for n in ("default", "zero") for w in ("1e-6", "0")]
A_DIR = "CDDM_mech_noise_decay/EqType=h_N=1000_noise={0}_wd={1}"
B_ARMS = ("none", "ent0.1", "rate0.1")
B_DIR = "CDDM_mech_switch/EqType=h_N=1000_arm={0}"
DEFAULT_NOISE = (0.05, 0.05, 0.03)
SWITCH_IT, END_IT, LATE = 75000, 100000, 2000      # part B: the switch, the end, the window of "lasting"
DRAWS = 36
READ_NOISE = 52000       # noise draw d of the read uses numpy seed READ_NOISE + d
GRAD_NOISE = 61000       # noise seed of the gradient read's noisy batch

# ---- the bars (rules entry) ----
R_SUPPORT, R_REFUTE = 0.90, 0.75          # part A: active(zero, wd 0) / active(default, wd 0)
R_PHENOMENON = 0.75                       # part A gate: active(zero, wd 1e-6) / active(default, wd 1e-6)
S_MAIN, S_MINOR = 0.50, 0.25              # part A: share of the default network's inactive units that decay accounts for
NET_RESURRECT, NET_NOT = 0.20, 0.10       # part B: share of the inactive units made active, above the arm `none`
REV_ENT, REV_RATE = 0.50, 0.30            # the reviewer's own bars, reported beside the verdict
R2_TOL = 0.01                             # M09's tolerance
FALL_UNIT = 1e-4                          # part B: a unit's removal costs performance above this fall in R^2
COSTLY_YES, COSTLY_NO = 0.5, 0.1          # part B: share of costly resurrected units, relative to the kept units (amendment 2026-11-25)
SILENT_REFUTE = 0.30                      # part B: the zero-gradient account is refuted by a net share of silent units above this
FALL_GROUP = 0.01                         # part B: the whole group's removal costs performance above this fall


def finished(cell):
    """Finished run folders of a cell, by seed.

    Args:
        cell: path below DATA_DIR.
    Returns:
        dict seed -> folder.
    """
    out = {}
    for s in SEEDS:
        d = [x for x in glob.glob(os.path.join(glob.escape(os.path.join(DATA_DIR, cell)), f"*seed={s};*"))
             if glob.glob(os.path.join(glob.escape(x), "*LastParams*"))]
        if len(d) > 1:
            raise RuntimeError(f"{len(d)} runs of seed {s} in {cell}")
        if d:
            out[s] = d[0]
    return out


def performance(W, cfg):
    """Held-out R^2 of one set of weights, noise-free and under the default training noise.

    Args:
        W: mech_removal.load's weights; cfg: the run's config.
    Returns:
        (float noise-free R^2, float mean R^2 over DRAWS draws at DEFAULT_NOISE).
    """
    u, y, mask = H.heldout_batch(cfg)
    alpha = float(cfg.model.dt) / float(cfg.model.tau)
    clean = MR.r2(MR.forward(W, u, alpha), y, mask)
    noisy = [MR.r2(MR.forward(W, u, alpha, MR.noise_draw(READ_NOISE + d, W[0].shape[0], u.shape[0], u.shape[1], u.shape[2],
                                                         alpha, DEFAULT_NOISE)), y, mask) for d in range(DRAWS)]
    return clean, float(np.mean(noisy))


def classes(p):
    """Active, low-active and silent masks of a participation vector."""
    a = active_mask(p)
    return dict(active=a, silent=p < SILENT, low=~a & ~(p < SILENT))


def read_net(d, cache):
    """Everything the verdict needs from one run folder's final weights and stored trace.

    Args:
        d: run folder; cache: open heldout_r2 cache.
    Returns:
        dict of plain Python values.
    """
    W, cfg = MR.load(d)
    p = MR.participation(W, cfg)
    c = classes(p)
    clean, noisy = performance(W, cfg)
    tr = trace(d)
    its = sorted(tr)
    row = torch.cat([W[0], W[1]], 1).double().norm(dim=1).numpy()
    loss = json.load(open(only(d, "*TrainLosses.json")))["train_losses"]
    return dict(run_dir=d, p=p.tolist(), active=int(c["active"].sum()), low=int(c["low"].sum()), silent=int(c["silent"].sum()),
                r2_clean=clean, r2_noise=noisy, r2_project=float(H.heldout_r2(d, draws=H.R2_DRAWS, cache=cache)),
                row_ratio=float(np.median(row[~c["active"]]) / np.median(row[c["active"]])) if (~c["active"]).any() else None,
                trace_it=its, trace_active=[int(active_mask(tr[i]).sum()) for i in its],
                trace_silent=[int((tr[i] < SILENT).sum()) for i in its], n_iter=len(loss),
                loss_end=float(np.mean(loss[-1000:])))


def do_read(out_json):
    """Mode `read`: parts A and B.

    Args:
        out_json: str, output file.
    """
    cache_path = os.path.join(os.path.dirname(os.path.abspath(out_json)), "r2cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    out = dict(A={}, B={})
    for n, w in A_CELLS:
        out["A"][f"{n}/{w}"] = {s: read_net(d, cache) for s, d in finished(A_DIR.format(n, w)).items()}
    parents = finished(A_DIR.format("default", "1e-6"))
    for arm in B_ARMS:
        out["B"][arm] = {}
        for s, d in finished(B_DIR.format(arm)).items():
            r = read_net(d, cache)
            p1, p0 = np.asarray(r["p"]), np.asarray(out["A"]["default/1e-6"][s]["p"])
            tr = trace(d)
            start = classes(p0)
            inact = ~start["active"]
            own, fixed = active_mask(p1), p1 >= 0.05 * np.quantile(p0, 0.95)
            late = [i for i in tr if END_IT - LATE <= i < END_IT]
            stay = np.mean([active_mask(tr[i]) for i in late], 0)
            # share of the units inactive at the switch that are active, at every stored vector of the continuation
            cont = sorted(i for i in tr if i >= SWITCH_IT)
            r.update(parent=parents[s], p_switch=p0.tolist(), n_inactive=int(inact.sum()), n_silent=int(start["silent"].sum()),
                     share=dict(inactive=float(own[inact].mean()), low=float(own[start["low"]].mean()) if start["low"].any() else None,
                                silent=float(own[start["silent"]].mean()) if start["silent"].any() else None,
                                active=float(own[~inact].mean())),
                     share_fixed=dict(inactive=float(fixed[inact].mean()), silent=float(fixed[start["silent"]].mean()) if start["silent"].any() else None),
                     resurrected=int((own & inact).sum()), resurrected_lasting=int((own & inact & (stay >= 0.5)).sum()),
                     n_late=len(late), switch_trace_agrees=int((active_mask(tr[SWITCH_IT]) == start["active"]).sum()),
                     course_it=cont, course_share=[float(active_mask(tr[i])[inact].mean()) for i in cont])
            out["B"][arm][s] = r
    json.dump(cache, open(cache_path, "w"))
    json.dump(out, open(out_json, "w"))
    for part in ("A", "B"):
        for g, nets in out[part].items():
            for s, r in nets.items():
                print(part, g, s, "active", r["active"], "silent", r["silent"], "R2 clean %.4f noise %.4f project %.4f" %
                      (r["r2_clean"], r["r2_noise"], r["r2_project"]), "row ratio", r["row_ratio"],
                      ("share of inactive now active %.3f (%d units)" % (r["share"]["inactive"], r["resurrected"])) if part == "B" else "", flush=True)


def do_grad(out_json):
    """Mode `grad`: penalty and task gradients on each unit's incoming weights at the switch.

    Args:
        out_json: str, output file.
    """
    out = {}
    for s, d in finished(A_DIR.format("default", "1e-6")).items():
        cfg = OmegaConf.load(only(d, "*_config.yaml"))
        rnn = build_rnn(cfg, seed=int(cfg.seed))
        P, _ = H._params(d)
        load_params(rnn, P)
        u, y, mask = training_batch(cfg)
        c = classes(MR.participation(MR.load(d)[0], cfg))
        st = torch.load(only(d, "*AdamState*.pt"), map_location="cpu", weights_only=True)
        N = int(cfg.model.N)
        root_v = [x for x in st["state"].values() if tuple(x["exp_avg_sq"].shape) == (N, N)][0]["exp_avg_sq"].sqrt()
        pen = Penalties(RNN=rnn)
        out[s] = dict(n={k: int(v.sum()) for k, v in c.items()}, rms_v_silent=float(root_v[c["silent"]].median()))
        for noise in (False, True):
            rnn.random_generator.manual_seed(GRAD_NOISE)
            states, z = rnn(u, w_noise=noise)
            terms = dict(ent=0.1 * pen.entropy_setpoint_penalty(states, target_bits=0.75, eps=1e-6),
                         rate=0.1 * pen.fr_magnitude_penalty(states, **OmegaConf.to_container(cfg.trainer.frm_args)),
                         task=(z[:, mask] - y[:, mask]).pow(2).mean())
            rec = dict(on_silent=float((states[c["silent"]] > 0).float().mean()) if c["silent"].any() else None)
            for k, v in terms.items():
                g = torch.autograd.grad(v, (rnn.W_rec, rnn.W_inp), retain_graph=True)
                row = torch.cat(g, 1).double().norm(dim=1).detach().numpy()
                rec[k] = {q: dict(median=float(np.median(row[ix])), zero=float((row[ix] == 0).mean())) for q, ix in c.items() if ix.any()}
            out[s]["noisy" if noise else "clean"] = rec
            print(s, "noisy" if noise else "clean", json.dumps(rec), flush=True)
    json.dump(out, open(out_json, "w"), indent=1)


def training_batch(cfg):
    """The task's full training grid as float32 tensors.

    Args:
        cfg: the run's config.
    Returns:
        (inputs, targets, mask).
    """
    tcfg = H.prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    task = H.hydra.utils.instantiate(tcfg)
    u, y, _ = task.get_batch()
    return (torch.from_numpy(np.asarray(u, dtype=np.float32)), torch.from_numpy(np.asarray(y, dtype=np.float32)),
            H.get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt))


def verdict_a(R):
    """Part A: the gate, the verdict on the noise x decay interaction, and the share of inactive units.

    Args:
        R: the read's "A" block.
    Returns:
        dict of the verdict's numbers and text.
    """
    M = {g: {s: r["active"] for s, r in nets.items()} for g, nets in R.items()}
    mean = {g: float(np.mean(list(v.values()))) for g, v in M.items()}
    r1 = mean["zero/1e-6"] / mean["default/1e-6"]
    r0 = mean["zero/0"] / mean["default/0"]
    pairs = {s: M["zero/0"][s] / M["default/0"][s] for s in M["zero/0"] if s in M["default/0"]}
    share = (mean["default/0"] - mean["default/1e-6"]) / (1000 - mean["default/1e-6"])
    if r1 > R_PHENOMENON:
        text = "gate fails: the new noise-free networks do not have the deficit the test is about; no verdict"
    elif r0 >= R_SUPPORT and min(pairs.values()) > R_REFUTE:
        text = "SUPPORTED: without decay the noise-free networks have as many active units"
    elif r0 <= R_REFUTE and max(pairs.values()) < R_SUPPORT:
        text = "REFUTED: the noise-free deficit remains without decay"
    else:
        text = "INCONCLUSIVE (between the bars, or the pairs disagree)"
    removed = 1 - np.log(r0) / np.log(r1) if r1 < 1 else float("nan")
    size = "main cause" if share >= S_MAIN else "minor cause" if share < S_MINOR else "partial cause"
    return dict(M=M, mean=mean, r_decay=r1, r_nodecay=r0, pairs=pairs, text=text, deficit_removed=float(removed),
                share=float(share), share_text=size)


def verdict_b(R):
    """Part B: net share of the units inactive at the switch that are active at the end, per arm.

    Args:
        R: the read's "B" block.
    Returns:
        dict arm -> numbers and text.
    """
    out = {}
    for arm in B_ARMS[1:]:
        seeds = [s for s in R[arm] if s in R["none"]]
        net = {pop: {s: R[arm][s]["share"][pop] - R["none"][s]["share"][pop] for s in seeds
                     if R[arm][s]["share"][pop] is not None} for pop in ("inactive", "silent", "low")}
        v = list(net["inactive"].values())
        text = ("RESURRECTS" if v and min(v) >= NET_RESURRECT else "DOES NOT RESURRECT" if v and max(v) <= NET_NOT else "PARTIAL")
        dr2 = float(np.mean([R[arm][s]["r2_project"] for s in seeds]) - np.mean([R["none"][s]["r2_project"] for s in seeds]))
        out[arm] = dict(net=net, text=text, dr2=dr2, within=dr2 >= -R2_TOL)
    return out


def removal_table(folder):
    """Part B: what removing single units and groups does to R^2, per arm and seed.

    Args:
        folder: directory of mech_removal JSONs named removal_<arm>_s<seed>.json.
    Returns:
        dict (arm, seed) -> per-set counts, share of units whose removal costs more than FALL_UNIT, median fall, group falls.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, "removal_*_s*.json"))):
        arm, seed = os.path.basename(f)[len("removal_"):-len(".json")].rsplit("_s", 1)
        J = json.load(open(f))
        row = {}
        for k, fall in J["fall"].items():
            x = np.array([np.mean(v) for v in fall.values()])
            row[k] = dict(n=len(x), costly=float((x > FALL_UNIT).mean()) if len(x) else None, median=float(np.median(x)) if len(x) else None)
        row["group"] = {k: [(g["n"], float(np.mean(g["fall"]))) for g in v] for k, v in J["group"].items()}
        out[(arm, seed)] = row
    return out


def verdict_zero_gradient(B):
    """Part B: the zero-gradient account, from the net shares of the units silent at the switch.

    Args:
        B: verdict_b's output.
    Returns:
        str.
    """
    ent, rate = list(B["ent0.1"]["net"]["silent"].values()), list(B["rate0.1"]["net"]["silent"].values())
    if len(ent) < 3 or len(rate) < 3:
        return "no verdict (fewer than three networks)"
    if min(rate) >= SILENT_REFUTE or max(ent) <= NET_NOT:
        return "REFUTED"
    if B["ent0.1"]["text"] == "RESURRECTS" and max(rate) <= NET_NOT:
        return "SUPPORTED"
    return "INCONCLUSIVE"


def verdict_removal(table, arm):
    """Part B: do the resurrected units of one arm contribute to the output?

    Args:
        table: removal_table's output; arm: str.
    Returns:
        str.
    """
    rows = [r for (a, _), r in table.items() if a == arm]
    if len(rows) < 3 or any(not r["resurrected"]["n"] for r in rows):
        return "no verdict (fewer than three networks, or a network without resurrected units)"
    rel = [r["resurrected"]["costly"] / max(r["kept"]["costly"], 1e-12) for r in rows]
    grp = [r["group"]["resurrected"][0][1] for r in rows]
    tail = f" (share of costly units relative to the kept ones {np.round(rel, 2).tolist()}; group falls {np.round(grp, 4).tolist()})"
    if min(rel) >= COSTLY_YES and min(grp) > FALL_GROUP:
        return "CONTRIBUTE" + tail
    if max(rel) <= COSTLY_NO and max(grp) < FALL_GROUP:
        return "DO NOT CONTRIBUTE" + tail
    return "MIXED" + tail


def do_verdict(read_json, removal, fig):
    """Mode `verdict`: tables, verdicts, figure.

    Args:
        read_json: the `read` output; removal: folder of removal JSONs or None; fig: figure path without extension, or None.
    """
    R = json.load(open(read_json))
    n = {g: len(v) for part in ("A", "B") for g, v in R[part].items()}
    print("networks read per cell:", n)
    if any(v != 3 for v in n.values()):
        print("!! not every cell has three finished networks")
    for part in ("A", "B"):
        for g, nets in R[part].items():
            for s, r in nets.items():
                print(f"{part} {g:14s} s{s:>10s} active {r['active']:4d} low {r['low']:4d} silent {r['silent']:4d} R2 clean {r['r2_clean']:.4f} "
                      f"noise {r['r2_noise']:.4f} project {r['r2_project']:.4f} row ratio {r['row_ratio'] if r['row_ratio'] is None else round(r['row_ratio'], 3)} loss {r['loss_end']:.2e}")
    a = verdict_a(R["A"])
    print("\nPART A", json.dumps({k: v for k, v in a.items() if k != "M"}, indent=1))
    if not all(R["B"].get(x) for x in B_ARMS):
        return
    vb = verdict_b(R["B"])
    for arm, v in vb.items():
        print(f"\nPART B {arm}: {v['text']}; R2 minus arm none {v['dr2']:+.4f} (within tolerance: {v['within']})")
        print("  net share, per population and network:", json.dumps(v["net"]))
        raw = {s: r["share"] for s, r in R["B"][arm].items()}
        print("  raw share:", json.dumps(raw), "fixed threshold:", json.dumps({s: r["share_fixed"] for s, r in R["B"][arm].items()}))
        print("  resurrected", {s: (r["resurrected"], r["resurrected_lasting"]) for s, r in R["B"][arm].items()}, "(total, lasting)")
        bar = REV_ENT if arm == "ent0.1" else REV_RATE
        print(f"  reviewer's bar ({bar:.0%} of the silent units):", {s: r["share"]["silent"] for s, r in R["B"][arm].items()})
    print("\narm none raw share:", {s: r["share"] for s, r in R["B"]["none"].items()})
    print("\nzero-gradient account:", verdict_zero_gradient(vb))
    if removal:
        table = removal_table(removal)
        print("\nremoval")
        for (arm, seed), row in table.items():
            print(arm, seed, json.dumps(row))
        for arm in B_ARMS[1:]:
            print(f"resurrected units of {arm}:", verdict_removal(table, arm))
    if fig:
        draw(R, fig)


def draw(R, stem):
    """Figure: active units over training in the four cells (a), and share of the inactive units active after the switch (b).

    Args:
        R: the read; stem: path without extension.
    """
    col = {"default/1e-6": "#333333", "default/0": "#2a78d6", "zero/1e-6": "#eb6834", "zero/0": "#9b59b6"}
    lab = {"default/1e-6": "noise, decay (default)", "default/0": "noise, no decay", "zero/1e-6": "no noise, decay", "zero/0": "no noise, no decay"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for g, nets in R["A"].items():
        for k, r in enumerate(nets.values()):
            ax[0].plot(np.asarray(r["trace_it"]) / 1000, r["trace_active"], color=col[g], lw=1.2, label=lab[g] if k == 0 else None)
    ax[0].set(xlabel="iteration (thousands)", ylabel="active units (of 1,000)", ylim=(0, 1000), title="a  CDDM, N = 1,000: training noise x weight decay")
    cb = {"none": "#333333", "ent0.1": "#2a78d6", "rate0.1": "#eb6834"}
    for arm, nets in R["B"].items():
        for k, r in enumerate(nets.values()):
            ax[1].plot((np.asarray(r["course_it"]) - SWITCH_IT) / 1000, r["course_share"], color=cb[arm], lw=1.2,
                       label={"none": "no penalty", "ent0.1": "entropy set point 0.1", "rate0.1": "rate penalty 0.1"}[arm] if k == 0 else None)
    ax[1].set(xlabel="iterations after the switch (thousands)", ylabel="share active, of units inactive at the switch", ylim=(0, 1),
              title="b  penalty switched on at 75,000")
    for x in ax:
        x.grid(alpha=0.25, lw=0.5)
        x.legend(fontsize=8, frameon=False)
    for ext in (".pdf", ".png"):
        fig.savefig(stem + ext, dpi=150)
    print(f"figure: {stem}.pdf, .png")


def main():
    """Command line: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("read", "grad", "verdict"))
    ap.add_argument("path")
    ap.add_argument("--removal", default=None)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    if a.mode == "read":
        do_read(a.path)
    elif a.mode == "grad":
        do_grad(a.path)
    else:
        do_verdict(a.path, a.removal, a.fig)


if __name__ == "__main__":
    main()
