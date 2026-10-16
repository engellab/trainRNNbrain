#!/usr/bin/env python3
"""Which units still receive a gradient? A silence read-out that does not care about the activation.

WHY. The participation rule (p_i = std(r_i) + q90(|r_i|) >= 0.05 q95(p)) looks for units sitting at
0, which is what a dead ReLU unit does. A tanh unit cannot sit at 0 and stay out of the computation;
it fails by pinning at +-1, where it is loud and the participation rule counts it. What a dead ReLU
unit and a pinned tanh unit share is that the task loss no longer reaches them. So measure that
directly: run the trained network noise-free on its own training batch, backpropagate the training
loss (masked MSE), and ask, per unit, how large the gradient is that arrives at it.

THREE PER-UNIT GRADIENTS, because they answer different questions.
  g_state_i = RMS over time and trials of dL/dh_i(t), the gradient on the unit's hidden state (h is
              the pre-activation; the networks use the h equation). For ReLU this is exactly 0 when
              h_i < 0 at every step: every path from h_i to the loss goes through f'(h_i) = 0.
  g_in_i    = ||dL/dW_rec[i,:]|| and ||dL/dW_inp[i,:]|| together: can the unit's INPUTS learn.
  g_out_j   = ||dL/dW_rec[:,j]|| and ||dL/dW_out[:,j]|| together: can its OUTPUTS learn. A silent
              ReLU unit has r = 0, so 0. A pinned tanh unit has r = +-1, so this need NOT vanish --
              its output weights keep learning as if it were a constant bias source.
A unit counts as receiving a gradient if its g >= 0.05 x q95(g), the same scale-free bar the
participation rule uses.

⚠️ READ g_in AND g_out, NOT g_state, FOR ReLU (found 2026-10-13). Every unit starts a trial at h = 0
exactly, and on the flip-flop stays there until the first input pulse (~36 of 300 steps). The
network's ReLU (torch.clamp) passes the gradient at exactly 0, so g_state counted every flip-flop
unit as reached (997-1000 of 1000) through steps on which no weight can learn (all rates and inputs
are 0). The weight gradients g_in and g_out are untouched by this. `slope` counts only h > 0 for ReLU.

THE UNTRAINED TWIN is rebuilt from the saved seed exactly as run_experiment.py builds it (model cfg
merged with the task cfg, seed + (65537**3 % 7309), torch global seed set first). The reconstruction
is checked against the participation trace's iteration-0 probe and the correlation is printed.

WHERE A RUN IS READ (--at). `end` reads the final weights (LastParams). `floor` reads the kept
weight snapshot at or just after the run's noisy-loss floor crossing (1.05x, common.noisy_floor),
and falls back to the final weights, labelled `end`, for a run that kept no snapshots; every record
says which (`read`, `read_iter`).

Usage (on Della, needs the data and the package):
  PYTHONPATH=<checkout> python gradient_participation.py CELL [CELL ...] --out DIR [--at floor]
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask
from trainRNNbrain.utils import filter_kwargs, import_any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import active_mask, read_run, warm_start_chain


def act_fn(args):
    """The network's own activation (RNN_torch.configure_activation_) and its derivative.

    Args:
        args: the saved activation_args dict (name, slope, and leak_slope / shift / beta as used).
    Returns:
        (f, f'), both elementwise on tensors; f' by autograd, so every activation shares one path.
    """
    f = RNN_torch.configure_activation_(dict(args))

    def fp(x):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(f(x).sum(), x)[0]
        # clamp's gradient at exactly 0 is 1; a ReLU unit at h = 0 is not "on"
        return g * (x > 0) if args["name"] == "relu" else g
    return f, fp


def unit_gradients(W, u, target, mask, dev):
    """Noise-free forward + backward of the masked MSE; per-unit gradient summaries.

    Args:
        W: dict of numpy W_rec, W_inp, W_out, y_init, bias, tau, dt, activation_args;
        u: (n_in, T, B) inputs; target: (n_out, T, B); mask: timepoint indices scored.
    Returns:
        dict of per-unit numpy arrays: g_state, g_in, g_out, p (participation), slope (mean f'),
        plus the scalar loss.
    """
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev)
    Wr, Wi, Wo = (t(W[k]).requires_grad_(True) for k in ("W_rec", "W_inp", "W_out"))
    b, y0 = t(W["bias"]).reshape(-1, 1), t(W["y_init"]).reshape(-1, 1)
    alpha = float(W["dt"]) / float(W["tau"])
    f, fp = act_fn(W["activation_args"])
    U, Y = t(u), t(target)
    B, T = U.shape[2], U.shape[1]
    xs = [y0.repeat(1, B).requires_grad_(True)]
    for k in range(1, T):
        x = xs[-1]
        xs.append(x + alpha * (-x + Wr @ f(x) + Wi @ U[:, k - 1, :] + b))
    H = torch.stack(xs, 1)                                # (N, T, B)
    R = f(H)
    out = torch.einsum("oj,jtk->otk", Wo, R)
    m = torch.as_tensor(np.asarray(mask), device=dev, dtype=torch.long)
    loss = ((out[:, m, :] - Y[:, m, :]) ** 2).mean()
    grads = torch.autograd.grad(loss, [Wr, Wi, Wo] + xs)
    gWr, gWi, gWo, gX = grads[0], grads[1], grads[2], torch.stack(grads[3:], 1)
    with torch.no_grad():
        Rn = R.cpu().numpy()
        res = dict(
            g_state=gX.pow(2).mean(dim=(1, 2)).sqrt(),
            g_in=(gWr.pow(2).sum(1) + gWi.pow(2).sum(1)).sqrt(),
            g_out=(gWr.pow(2).sum(0) + gWo.pow(2).sum(0)).sqrt(),
            slope=fp(H).mean(dim=(1, 2)),
        )
        res = {k: v.cpu().numpy().astype(float) for k, v in res.items()}
    res["p"] = Rn.std(axis=(1, 2)) + np.quantile(np.abs(Rn), 0.9, axis=(1, 2))
    res["loss"] = float(loss)
    return res


def initial_params(cfg):
    """The untrained twin's parameters, built the way run_experiment.py builds the network."""
    task_cfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    del task_cfg._target_
    rnn_cfg = OmegaConf.create(cfg.model)
    rnn_args = filter_kwargs(import_any(rnn_cfg._target_), OmegaConf.merge(rnn_cfg, task_cfg))
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rnn_args.seed = seed + (0 * 14653 + (0 + 65537) ** 3) % 7309
    net = hydra.utils.instantiate(rnn_args)
    return net.get_params()


def count(g):
    """Units over the scale-free bar, g >= 0.05 x q95(g) (common.active_mask).

    Args:
        g: (N,) per-unit participation or gradient magnitude.
    Returns:
        int count.
    """
    return int(active_mask(g).sum())


def read_point(d, at):
    """Which saved weights of run folder d to read, per --at.

    Args:
        d: a run folder; at: "floor" for the kept snapshot at the floor crossing (common.read_run),
            anything else for the final weights.
    Returns:
        (path of a *LastParams*.npz, "floor" or "end", the iteration it stands for or None).
    """
    end = glob.glob(os.path.join(d, "*LastParams*.npz"))[0]
    if at != "floor":
        return end, "end", None
    snap = read_run(d)["snapshot"]
    if snap is None:
        return end, "end", None
    return glob.glob(os.path.join(snap, "*LastParams*.npz"))[0], "floor", int(os.path.basename(snap)[2:])


def main(cells, out_dir, dev, at="end"):
    """Per-unit gradients, untrained twin and trained network, for every run of every cell.

    Args:
        cells: cell folders (one subfolder per run); out_dir: one JSON per run is written here;
        dev: torch device; at: "end" or "floor" (see the module docstring).
    """
    os.makedirs(out_dir, exist_ok=True)
    for cell in cells:
        for d in sorted(glob.glob(os.path.join(cell, "*/"))):
            name = os.path.basename(d.rstrip("/"))
            if name.startswith("nan"):
                continue
            cfg = OmegaConf.load(glob.glob(os.path.join(d, "*_config.yaml"))[0])
            task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
            u, y, _ = task.get_batch()
            mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
            params, read, read_iter = read_point(d, at)
            z = np.load(params, allow_pickle=True)
            Wt = {k: z[k] for k in z.files}
            Wt["activation_args"] = Wt["activation_args"].item()
            Wt["bias"] = Wt["bias"] if Wt["bias"].shape else np.zeros(int(Wt["N"]))
            # the untrained twin is the chain's FIRST run: a continuation's own seed is a fresh draw
            root = warm_start_chain(d)[0]
            W0 = initial_params(OmegaConf.load(glob.glob(os.path.join(root, "*_config.yaml"))[0]))
            W0["activation_args"] = dict(W0["activation_args"])
            if W0["bias"] is None:
                W0["bias"] = np.zeros(int(W0["N"]))
            tr = pickle.load(open(glob.glob(os.path.join(d, "*ParticipationTrace.pkl"))[0], "rb"))
            P = np.asarray(tr["participation"], float)
            it_a = np.asarray(tr["participation_iters"], float)
            k = len(P) - 1 if read_iter is None else int(np.argmin(np.abs(it_a - read_iter)))
            rec = {"cell": os.path.basename(cell), "run": name, "read": read, "read_iter": read_iter,
                   "r2_saved": float(re.match(r"([0-9.]+)_", name).group(1))}
            for tag, W in (("init", W0), ("trained", Wt)):
                r = unit_gradients(W, u, y, mask, dev)
                rec[tag] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
            # the reconstruction check: our init participation against the trace's iteration-0 probe
            rec["check_init_corr"] = float(np.corrcoef(rec["init"]["p"], P[0])[0, 1])
            rec["check_final_corr"] = float(np.corrcoef(rec["trained"]["p"], P[k])[0, 1])
            rec["trace_active"] = count(P[k])
            sweep = os.path.basename(os.path.dirname(cell.rstrip("/")))
            json.dump(rec, open(os.path.join(out_dir, f"{sweep}__{rec['cell']}__{name[:40]}__{at}.json"), "w"))
            s = lambda tag, k: count(np.asarray(rec[tag][k]))
            print(f"{rec['cell'][-35:]:35s} {name[:9]} {read}@{read_iter} loss0={rec['init']['loss']:.4f} "
                  f"loss={rec['trained']['loss']:.5f} corr0={rec['check_init_corr']:.3f} "
                  f"corrT={rec['check_final_corr']:.3f} | trace={rec['trace_active']} "
                  f"p={s('trained','p')} | INIT st={s('init','g_state')} in={s('init','g_in')} "
                  f"out={s('init','g_out')} | TRAINED st={s('trained','g_state')} "
                  f"in={s('trained','g_in')} out={s('trained','g_out')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--at", choices=["end", "floor"], default="end")
    a = ap.parse_args()
    main(a.cells, a.out, torch.device("cuda" if torch.cuda.is_available() else "cpu"), a.at)
