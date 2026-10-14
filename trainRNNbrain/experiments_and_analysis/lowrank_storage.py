#!/usr/bin/env python3
"""Can a weight snapshot be STORED at low rank without changing what the paper measures on it?

Storage only: the networks stay full-rank and are trained as now. A snapshot of W_rec costs N^2
float32 numbers; this script reconstructs W_rec from a cheaper stored form and re-measures the
network, against the original weights of the same snapshot. Pass bar and design:
docs/notes/agent-lowrank.md (committed before any result was computed).

SCHEMES (W_inp, W_out, y_init stay exact in all of them; they are N x 2-6 and cost no more than a
rank-6 factor):
  init   W ~ a W0 + rank-r SVD of (W - a W0). W0 is the untrained matrix, regenerated from the chain's
         FIRST run's seed (gradient_participation.initial_params); a = <W, W0> / <W0, W0>.
  svd    W ~ rank-r SVD of W.
  fp16   W rounded to float16 (full rank, half the bytes).
  q<b>   W quantised to b-bit integers, one float32 scale per row (absmax: the largest |entry| of the
         row maps to the largest integer), e.g. q8 = N^2 bytes + 4N.
  exact  W itself, written and read back through the same folder path: every difference must be 0.
         This is the check that the write/score path adds nothing.

MEASURES, each on the original and on the reconstruction, same batch, same seeds:
  r2        heldout_r2.score_net on a folder holding the reconstructed parameters, 36 draws
  p, active participation std(r) + q0.9(|r|) on a noise-free pass of a fixed training batch;
            active if p >= 0.05 q95(p) (gradient_participation.unit_gradients / count)
  loss      noisy training loss (masked MSE), one noise draw with a fixed seed
  g_in/out  per-unit incoming / outgoing weight-gradient norms from the same noise-free pass

Usage (Della, CPU node):
  python lowrank_storage.py run RUN_DIR --at floor|end --schemes init,svd,q8 --out OUT.json --scratch DIR
  python lowrank_storage.py chain RUN_DIR --ranks 10,50 --every 1000 --out OUT.json
  python lowrank_storage.py figure 'results/*.json' --fig-dir DIR
"""

import argparse
import glob
import json
import os
import shutil
import sys

import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from common import warm_start_chain
from fig_knobs_at_floor import SNAP_DIR
from gradient_participation import count, initial_params, read_point, unit_gradients
from heldout_r2 import score_net
from trainRNNbrain.trainer.Trainer import scored_
from trainRNNbrain.training.training_utils import get_training_mask, prepare_task_arguments
from trainRNNbrain.utils import filter_kwargs, import_any

R2_DRAWS = 36                 # held-out noise draws, as fig_knobs_at_floor
TRAIN_SEED = 20261014         # seeds a sampled task's generator: one fixed training batch per run
LOSS_SEED = 7                 # the noise seed of the noisy training loss
RANKS = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 700]

# the pass bar (docs/notes/agent-lowrank.md)
BAR_R2 = 0.002                # |delta held-out R^2|
BAR_ACTIVE = 0.01             # relative error of the active count
BAR_AGREE = 0.99              # fraction of units with the same active/silent label
BAR_LOSS = 0.01               # relative error of the noisy training loss
BAR_CORR = 0.99               # Pearson correlation of g_in and of g_out


def load_npz(path):
    """A saved parameter file as a dict of numpy values (activation_args unpacked to a dict)."""
    z = np.load(path, allow_pickle=True)
    W = {k: z[k] for k in z.files}
    W["activation_args"] = W["activation_args"].item()
    return W


def for_gradients(W):
    """The parameter dict in the form unit_gradients expects (a zero bias where none is stored)."""
    W = dict(W)
    W["bias"] = W["bias"] if W["bias"].shape else np.zeros(int(W["N"]))
    return W


def truncated(U, s, Vt, r):
    """U[:, :r] diag(s[:r]) Vt[:r] as float64 (N, N)."""
    return (U[:, :r] * s[:r]) @ Vt[:r]


def build_rnn(cfg, params):
    """The run's RNN_torch, built as heldout_r2.score_net builds it, with these parameters loaded.

    Args:
        cfg: the run's saved config; params: dict of numpy parameters (W_rec, W_inp, W_out, y_init).
    Returns:
        the RNN_torch module.
    """
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    if "_target_" in tcfg:
        del tcfg._target_
    rnn = hydra.utils.instantiate(
        filter_kwargs(import_any(cfg.model._target_), OmegaConf.merge(OmegaConf.create(cfg.model), tcfg)))
    with torch.no_grad():
        for nm, prm in rnn.named_parameters():
            if nm in params:
                prm.copy_(torch.from_numpy(np.asarray(params[nm], dtype=np.float32)))
        if getattr(rnn, "y_init", None) is not None:
            rnn.y_init.copy_(torch.from_numpy(np.asarray(params["y_init"], dtype=np.float32)))
    return rnn


def noisy_loss(cfg, params, u, y, mask):
    """Masked MSE of one noisy forward pass (the run's own noise levels, noise seeded with LOSS_SEED).

    Args:
        cfg: run config; params: numpy parameter dict; u, y: (n_in, T, B), (n_out, T, B) arrays;
        mask: scored time indices.
    Returns:
        float loss, the quantity train_step records as the task loss.
    """
    rnn = build_rnn(cfg, params)
    rnn.random_generator.manual_seed(LOSS_SEED)
    with torch.no_grad():
        out = rnn(torch.as_tensor(u, dtype=torch.float32), w_noise=True)[1]
        Y = torch.as_tensor(y, dtype=torch.float32)
        m = torch.as_tensor(mask, dtype=torch.long)
        return float(((scored_(out, m) - scored_(Y, m)) ** 2).mean())


def measure(cfg, params, cfg_path, folder, u, y, mask, r2_dir=None):
    """Every measure of the pass bar for one parameter set.

    Args:
        cfg, cfg_path: the run config and its file (copied next to the parameters for score_net);
        params: numpy parameter dict as loaded from the npz; folder: where to write it for score_net;
        u, y, mask: the fixed training batch; r2_dir: score this existing folder instead of
        writing one (folder is then unused).
    Returns:
        dict: r2, loss_noisy, loss_clean, p, g_in, g_out (per-unit lists), active (count).
    """
    if r2_dir is None:
        os.makedirs(folder, exist_ok=True)
        np.savez(os.path.join(folder, "recon_LastParams.npz"), **params)
        shutil.copy(cfg_path, os.path.join(folder, "recon_config.yaml"))
    g = unit_gradients(for_gradients(params), u, y, mask, torch.device("cpu"))
    return dict(r2=score_net(r2_dir or folder, draws=R2_DRAWS), loss_noisy=noisy_loss(cfg, params, u, y, mask),
                loss_clean=g["loss"], p=g["p"].tolist(), g_in=g["g_in"].tolist(),
                g_out=g["g_out"].tolist(), active=count(g["p"]))


def compare(ref, rec):
    """The pass-bar statistics of a reconstruction against the original.

    Args:
        ref, rec: measure() dicts of the original and of the reconstruction.
    Returns:
        dict of differences and a `pass` flag (all bars met).
    """
    pr, pc = np.asarray(ref["p"]), np.asarray(rec["p"])
    lab_r, lab_c = pr >= 0.05 * np.quantile(pr, 0.95), pc >= 0.05 * np.quantile(pc, 0.95)
    corr = lambda k: float(np.corrcoef(ref[k], rec[k])[0, 1])
    d = dict(d_r2=rec["r2"] - ref["r2"], active=rec["active"],
             d_active_rel=(rec["active"] - ref["active"]) / ref["active"],
             agree=float((lab_r == lab_c).mean()), n_flip=int((lab_r != lab_c).sum()),
             d_loss_rel=(rec["loss_noisy"] - ref["loss_noisy"]) / ref["loss_noisy"],
             corr_gin=corr("g_in"), corr_gout=corr("g_out"), r2=rec["r2"], loss_noisy=rec["loss_noisy"])
    d["pass"] = bool(abs(d["d_r2"]) <= BAR_R2 and abs(d["d_active_rel"]) <= BAR_ACTIVE
                     and d["agree"] >= BAR_AGREE and abs(d["d_loss_rel"]) <= BAR_LOSS
                     and min(d["corr_gin"], d["corr_gout"]) >= BAR_CORR)
    return d


def energy(s):
    """Cumulative fraction of squared Frobenius norm captured by the top-r singular values."""
    e = np.cumsum(np.asarray(s, float) ** 2)
    return (e / e[-1]).tolist()


def run(run_dir, at, schemes, ranks, out, scratch):
    """Reconstruct one snapshot under each scheme and rank, and measure each against the original.

    Args:
        run_dir: a run folder; at: "floor" or "end" (gradient_participation.read_point);
        schemes: subset of init, svd, fp16, exact; ranks: ranks for init and svd;
        out: JSON path written; scratch: parent folder for the reconstructed parameter folders.
    """
    run_dir = run_dir.rstrip("/")
    npz, read, read_iter = read_point(run_dir, at)
    if at == "floor" and read != "floor":
        raise SystemExit(f"no snapshot at the floor for {run_dir}")
    cfg_path = sorted(glob.glob(os.path.join(run_dir, "*_config.yaml")))[0]
    cfg = OmegaConf.load(cfg_path)
    Wt = load_npz(npz)
    N = int(Wt["N"])
    tag = f"{cfg.task.taskname}_N{N}_s{cfg.seed}_{read}{read_iter or ''}"
    base = os.path.join(scratch, tag)

    # the fixed training batch (a sampled task's generator seeded, so every scheme sees the same trials)
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    tcfg.seed = TRAIN_SEED
    u, y, _ = hydra.utils.instantiate(tcfg).get_batch()
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)

    # the untrained matrix, from the chain's first run; checked against that run's iteration-0 probe
    root = warm_start_chain(run_dir)[0]
    W0p = initial_params(OmegaConf.load(sorted(glob.glob(os.path.join(root, "*_config.yaml")))[0]))
    W0 = np.asarray(W0p["W_rec"], np.float64)
    W = np.asarray(Wt["W_rec"], np.float64)
    a = float((W * W0).sum() / (W0 * W0).sum())
    dW = W - a * W0
    Us, ss, Vts = np.linalg.svd(W)
    Ud, sd, Vtd = np.linalg.svd(dW)
    sr = np.linalg.svd(np.random.default_rng(0).standard_normal((N, N)), compute_uv=False)
    print(f"{tag}: a={a:.4f}  |dW|/|W|={np.linalg.norm(dW) / np.linalg.norm(W):.3f}  "
          f"|W0|/|W|={np.linalg.norm(W0) / np.linalg.norm(W):.3f}", flush=True)

    # the original is scored from its own snapshot folder; "exact" goes through the write path, so
    # its differences being 0 shows the write/score path adds nothing
    ref = measure(cfg, Wt, cfg_path, None, u, y, mask, r2_dir=os.path.dirname(npz))
    print(f"  orig r2={ref['r2']:.4f} active={ref['active']} loss={ref['loss_noisy']:.5f}", flush=True)
    rec = dict(run=run_dir, task=str(cfg.task.taskname), N=N, seed=int(cfg.seed), read=read,
               read_iter=read_iter, npz=npz, a=a, fro_W=float(np.linalg.norm(W)),
               fro_W0=float(np.linalg.norm(W0)), fro_dW=float(np.linalg.norm(dW)),
               s_W=ss.tolist(), s_dW=sd.tolist(), s_rand=sr.tolist(),
               energy_W=energy(ss), energy_dW=energy(sd), energy_rand=energy(sr),
               orig={k: ref[k] for k in ("r2", "loss_noisy", "loss_clean", "active")}, results=[])

    jobs = []
    for sch in schemes:
        if sch in ("init", "svd"):
            jobs += [(sch, r) for r in ranks if r < N]
        else:
            jobs.append((sch, N))
    for sch, r in jobs:
        if sch == "init":
            Wr, nbytes = a * W0 + truncated(Ud, sd, Vtd, r), 4 * (2 * r * N + r + 1)
        elif sch == "svd":
            Wr, nbytes = truncated(Us, ss, Vts, r), 4 * (2 * r * N + r)
        elif sch == "fp16":
            Wr, nbytes = Wt["W_rec"].astype(np.float16), 2 * N * N
        elif sch.startswith("q"):
            levels = 2 ** (int(sch[1:]) - 1) - 1
            scale = np.abs(W).max(axis=1, keepdims=True) / levels
            Wr, nbytes = np.round(W / scale) * scale, N * N * int(sch[1:]) // 8 + 4 * N
        else:
            Wr, nbytes = Wt["W_rec"], 4 * N * N
        P = dict(Wt)
        P["W_rec"] = np.asarray(Wr, np.float32)
        m = measure(cfg, P, cfg_path, os.path.join(base, f"{sch}_r{r}"), u, y, mask)
        d = compare(ref, m)
        d.update(scheme=sch, rank=r, bytes=nbytes,
                 rel_fro_err=float(np.linalg.norm(P["W_rec"] - W) / np.linalg.norm(W)))
        rec["results"].append(d)
        print(f"  {sch:5s} r={r:4d} {nbytes / 1e6:6.2f}MB err={d['rel_fro_err']:.4f} "
              f"dR2={d['d_r2']:+.4f} act={d['active']} ({d['d_active_rel']:+.3f}) "
              f"flip={d['n_flip']} dL={d['d_loss_rel']:+.4f} cin={d['corr_gin']:.4f} "
              f"cout={d['corr_gout']:.4f} {'PASS' if d['pass'] else 'fail'}", flush=True)
        json.dump(rec, open(out, "w"))
    json.dump(rec, open(out, "w"))


def chain_snapshots(run_dir, every, extra=None):
    """Every kept snapshot of a run's warm-start chain, every `every` iterations, then its final weights.

    Args:
        run_dir: a run folder; every: keep snapshots whose iteration is a multiple of this;
        extra: one more snapshot npz to include whatever its iteration (the floor snapshot).
    Returns:
        list of (iteration, npz path), oldest first; the last entry is the run's LastParams.
    """
    out = []
    for d in warm_start_chain(run_dir):
        cell = os.path.dirname(d.rstrip("/"))
        seed = OmegaConf.load(sorted(glob.glob(os.path.join(d, "*_config.yaml")))[0]).seed
        for s in glob.glob(os.path.join(SNAP_DIR, os.path.basename(os.path.dirname(cell)),
                                        os.path.basename(cell), f"s{seed}", "it*")):
            it = int(os.path.basename(s)[2:])
            f = glob.glob(os.path.join(s, "*LastParams*.npz"))[0]
            if it % every == 0 or f == extra:
                out.append((it, f))
    out.sort()
    end = glob.glob(os.path.join(run_dir.rstrip("/"), "*LastParams*.npz"))[0]
    return out + [(out[-1][0] + every, end)]


def chain(run_dir, ranks, every, out, scratch):
    """Delta coding along the snapshot series: W_k ~ What_{k-1} + rank-r SVD of (W_k - What_{k-1}).

    The keyframe is the untrained W0, regenerated from the seed, so it costs nothing. Each delta is
    taken against the previous RECONSTRUCTION, as a decoder holding only the stored factors would
    rebuild it, so the part of W a step leaves out is still there to be coded at the next step. The
    snapshots at the run's floor and at its end are then measured against the original weights.

    Args:
        run_dir: a run folder; ranks: ranks per delta; every: snapshot spacing in iterations;
        out: JSON path; scratch: parent folder for reconstructed parameter folders.
    """
    run_dir = run_dir.rstrip("/")
    floor_npz = read_point(run_dir, "floor")[0]
    snaps = chain_snapshots(run_dir, every, extra=floor_npz)
    root = warm_start_chain(run_dir)[0]
    W0 = np.asarray(initial_params(OmegaConf.load(sorted(glob.glob(os.path.join(root, "*_config.yaml")))[0]))
                    ["W_rec"], np.float64)
    cfg_path = sorted(glob.glob(os.path.join(run_dir, "*_config.yaml")))[0]
    cfg = OmegaConf.load(cfg_path)
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    tcfg.seed = TRAIN_SEED
    u, y, _ = hydra.utils.instantiate(tcfg).get_batch()
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
    targets = {floor_npz: "floor", snaps[-1][1]: "end"}
    rec = dict(run=run_dir, task=str(cfg.task.taskname), N=int(W0.shape[0]), seed=int(cfg.seed), every=every,
               iters=[it for it, _ in snaps], step_norm=[], results=[], err={})
    Wprev = W0
    for it, f in snaps:
        W = np.load(f)["W_rec"].astype(np.float64)
        if Wprev is not None:
            rec["step_norm"].append(float(np.linalg.norm(W - Wprev) / np.linalg.norm(W)))
        Wprev = W
    refs = {}
    for f, lab in targets.items():
        P = load_npz(f)
        refs[f] = (P, measure(cfg, P, cfg_path, None, u, y, mask, r2_dir=os.path.dirname(f)))
    for r in ranks:
        What, errs = W0.copy(), []
        for it, f in snaps:
            W = np.load(f)["W_rec"].astype(np.float64)
            U, s, Vt = np.linalg.svd(W - What)
            What = What + truncated(U, s, Vt, r)
            errs.append(float(np.linalg.norm(W - What) / np.linalg.norm(W)))
            if f in targets:
                P = dict(refs[f][0])
                P["W_rec"] = What.astype(np.float32)
                m = measure(cfg, P, cfg_path, os.path.join(scratch, f"chain_{rec['task']}_s{rec['seed']}",
                                                            f"{targets[f]}_r{r}"), u, y, mask)
                d = compare(refs[f][1], m)
                d.update(scheme="delta", rank=r, read=targets[f], read_iter=it, rel_fro_err=errs[-1],
                         bytes=4 * (2 * r * W.shape[0] + r))
                rec["results"].append(d)
                print(f"  delta r={r:4d} {targets[f]:5s}@{it} err={errs[-1]:.4f} dR2={d['d_r2']:+.4f} "
                      f"flip={d['n_flip']} dL={d['d_loss_rel']:+.4f} cin={d['corr_gin']:.4f} "
                      f"cout={d['corr_gout']:.4f} {'PASS' if d['pass'] else 'fail'}", flush=True)
        rec["err"][str(r)] = errs
        json.dump(rec, open(out, "w"))


def rows_and_spectra(pattern):
    """Every measured reconstruction as one flat row, and the spectra of each snapshot.

    Args:
        pattern: glob of JSON records written by run() and chain().
    Returns:
        (rows, spectra): rows are the compare() dicts with task, N, seed, read added; spectra maps
        (task, N, seed, read) to that snapshot's run() record (energy_W, energy_dW, energy_rand, a).
    """
    rows, spectra = [], {}
    for f in sorted(glob.glob(pattern)):
        rec = json.load(open(f))
        N = rec.get("N", 1000)          # chain records written before N was recorded are all N = 1000
        for d in rec["results"]:
            rows.append(dict(d, task=rec["task"], N=N, seed=rec["seed"], read=d.get("read", rec.get("read"))))
        if "energy_dW" in rec:
            spectra[(rec["task"], N, rec["seed"], rec["read"])] = rec
    return rows, spectra


def table(rows):
    """Per task, N and scheme: snapshots measured, and the smallest passing rank (or pass count).

    The smallest passing rank is the smallest tested rank at which every snapshot of the group
    passes, with every larger tested rank passing too (docs/notes/agent-lowrank.md).

    Args:
        rows: from rows_and_spectra.
    Returns:
        list of (task, N, scheme, n_snapshots, smallest passing rank or None, bytes there,
        number of snapshots passing at full rank schemes).
    """
    out = []
    for task, N, sch in sorted({(r["task"], r["N"], r["scheme"]) for r in rows}):
        g = [r for r in rows if (r["task"], r["N"], r["scheme"]) == (task, N, sch)]
        snaps = {(r["seed"], r["read"]) for r in g}
        best, nb = None, None
        for k in sorted({r["rank"] for r in g}, reverse=True):
            at = [r for r in g if r["rank"] == k]
            if len(at) < len(snaps) or not all(r["pass"] for r in at):
                break
            best, nb = k, at[0]["bytes"]
        out.append((task, N, sch, len(snaps), best, nb, sum(r["pass"] for r in g if r["rank"] == N)))
    return out


def figure(pattern, fig_dir):
    """Pass-bar statistics against rank, the worst snapshot of each task; plus the spectra.

    Low-rank schemes are lines (solid: a W0 + low rank; dashed: plain SVD; dash-dot: delta coding
    along the snapshot chain, one run per task). Full-rank formats (float16, q8, q6, q4) are
    markers placed at the rank that costs the same bytes, r/N = bytes / (8 N^2); the top axis gives
    storage as a fraction of the float32 matrix.

    Args:
        pattern: glob of JSON records; fig_dir: where the PDF and SVG are written.
    Returns:
        the PDF path.
    """
    rows, spectra = rows_and_spectra(pattern)
    ps.setup()
    names = {"CDDM": "CDDM", "DMTS": "DMTS", "NBitFlipFlop": "3-bit flip-flop"}
    keys = sorted({(r["task"], r["N"]) for r in rows}, key=lambda k: (k[1], k[0]))
    col = {k: ps.SLOTS[i % len(ps.SLOTS)] for i, k in enumerate(keys)}
    styles = {"init": "-", "svd": "--", "delta": "-."}
    marks = {"fp16": "s", "q12": "P", "q10": "X", "q8": "D", "q6": "^", "q4": "v"}
    fig, axes = plt.subplots(2, 3, figsize=(180 * ps.MM, 128 * ps.MM))
    panels = [("|$\\Delta$ held-out $R^2$|", lambda d: abs(d["d_r2"]), BAR_R2),
              ("|active-count error| (fraction)", lambda d: abs(d["d_active_rel"]), BAR_ACTIVE),
              ("units relabelled (fraction of $N$)", lambda d: 1 - d["agree"], 1 - BAR_AGREE),
              ("|noisy-loss error| (fraction)", lambda d: abs(d["d_loss_rel"]), BAR_LOSS),
              ("1 $-$ gradient-norm correlation\n(worse of incoming, outgoing)",
               lambda d: 1 - min(d["corr_gin"], d["corr_gout"]), 1 - BAR_CORR)]
    # an exact 0 sits on the bottom edge (FLOOR); a reconstruction whose dynamics diverge (|dR2| up
    # to 1e29) is clipped to the top edge (CEIL) so it does not set the scale
    FLOOR, CEIL = 1e-6, 10.0
    clip = lambda v: np.clip(v, FLOOR, CEIL)
    for ax, (lab, fn, bar) in zip(axes.flat, panels):
        for k in keys:
            g = [r for r in rows if (r["task"], r["N"]) == k]
            for sch, ls in styles.items():
                ranks = sorted({r["rank"] for r in g if r["scheme"] == sch})
                ys = [max(fn(r) for r in g if r["scheme"] == sch and r["rank"] == q) for q in ranks]
                if ranks:
                    ax.plot(np.asarray(ranks) / k[1], clip(ys), ls, color=col[k],
                            lw=0.9 if sch != "delta" else 0.7, marker="o", ms=1.8)
            for sch, mk in marks.items():
                at = [r for r in g if r["scheme"] == sch]
                if at:
                    ax.plot(at[0]["bytes"] / (8 * k[1] ** 2), clip(max(fn(r) for r in at)), mk,
                            color=col[k], ms=3.2, mfc="none", mew=0.8)
        ax.axhline(bar, color=ps.INK, lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(7e-4, 1.0)
        ax.set_ylim(FLOOR * 0.5, CEIL * 2)
        ax.set_ylabel(lab, fontsize=6.4)
        ax.set_xlabel("rank $r$ / $N$")
        top = ax.secondary_xaxis("top", functions=(lambda x: 2 * x, lambda x: x / 2))
        top.set_xlabel("storage / float32 matrix", fontsize=5.6)
        top.tick_params(labelsize=5.4)
    hs = [plt.Line2D([], [], color=col[k], lw=1.2, label=f"{names[k[0]]}, $N$ = {k[1]:,}") for k in keys]
    hs += [plt.Line2D([], [], color=ps.MUTED, ls=ls, lw=0.9, label=lb)
           for ls, lb in (("-", "$a W_0$ + low rank"), ("--", "plain SVD"), ("-.", "delta coding"))]
    hs += [plt.Line2D([], [], color=ps.MUTED, ls="", marker=mk, mfc="none", ms=3.2, label=sch)
           for sch, mk in marks.items()]
    hs += [plt.Line2D([], [], color=ps.INK, ls=":", lw=0.7, label="pass bar")]
    fig.legend(handles=hs, fontsize=5.6, frameon=False, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 0.995), handlelength=2.4, columnspacing=1.2, labelspacing=0.3)
    fig.text(0.5, 0.905, "each point: the worst snapshot (N = 1,000: 3 runs, each at its floor and at its end; "
             "N = 2,000 and 4,000: one run, end only); bottom edge = exactly 0, top edge = 10 or more",
             ha="center", fontsize=5.6, color=ps.MUTED)
    ax = axes.flat[5]
    for (task, N, seed, read), r in spectra.items():
        x = np.arange(1, N + 1) / N
        c = col[(task, N)]
        ax.plot(x, np.maximum(1 - np.asarray(r["energy_dW"]), 1e-9), color=c, lw=0.6,
                ls="-" if read == "end" else (0, (1, 1)))
    k0 = next(iter(spectra.values()))
    ax.plot(np.arange(1, k0["N"] + 1) / k0["N"], np.maximum(1 - np.asarray(k0["energy_rand"]), 1e-9),
            color=ps.BASE, lw=1.0, ls="--")
    ax.text(0.03, 0.04, "solid: end of training; dotted: at the floor\ngrey dashed: Gaussian matrix",
            transform=ax.transAxes, fontsize=5.0, color=ps.MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7e-4, 1.0)
    ax.set_ylim(1e-5, 1.5)
    ax.set_xlabel("rank $r$ / $N$")
    ax.set_ylabel("fraction of $\\|W - aW_0\\|^2$\nleft out by rank $r$", fontsize=6.4)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.08, wspace=0.42, hspace=0.62)
    os.makedirs(fig_dir, exist_ok=True)
    stem = os.path.join(fig_dir, "lowrank_storage")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".svg")
    for t in table(rows):
        print(f"{t[0]:13s} N={t[1]:5d} {t[2]:6s} snapshots={t[3]:2d} smallest passing r={t[4]} "
              f"({'-' if t[5] is None else f'{t[5] / 1e6:.2f} MB'})  passing at full rank: {t[6]}")
    return stem + ".pdf"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "chain", "figure"])
    ap.add_argument("target", help="run folder (run) or JSON glob (figure)")
    ap.add_argument("--at", choices=["floor", "end"], default="end")
    ap.add_argument("--schemes", default="exact,fp16,init,svd")
    ap.add_argument("--ranks", default=",".join(map(str, RANKS)))
    ap.add_argument("--every", type=int, default=1000, help="chain: snapshot spacing in iterations")
    ap.add_argument("--out")
    ap.add_argument("--scratch", default="/scratch/gpfs/TENGEL/pt1290/agent_lowrank/params")
    ap.add_argument("--fig-dir", default=os.path.expanduser("~/Documents/GitHub/trainRNNbrain/img/internal_figures"))
    a = ap.parse_args()
    if a.mode == "run":
        run(a.target, a.at, a.schemes.split(","), [int(v) for v in a.ranks.split(",")], a.out, a.scratch)
    elif a.mode == "chain":
        chain(a.target, [int(v) for v in a.ranks.split(",")], a.every, a.out, a.scratch)
    else:
        print(figure(a.target, a.fig_dir))
