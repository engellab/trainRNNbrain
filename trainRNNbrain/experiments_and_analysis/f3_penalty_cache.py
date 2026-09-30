#!/usr/bin/env python3
"""
Build the cache behind manuscript Figure 3: what each penalty does to a trained network, measured
on every penalised cell on disk rather than on one cell at one size.

WHAT IS MEASURED, per network. Four quantities, because a remedy that fixes one and wrecks another
has not fixed anything:

  r2             task performance. Recomputed twice - once WITH the network's own noise, which is
                 the quantity the folder name stores and therefore the only one that can be checked
                 against it, and once noise-free, which is the number the figure reports.
  active units   scale-free rule, p_i >= 0.05 q_95(p) on p_i = std(r_i) + q_0.9(r_i), measured on
                 the noise-free run so that it matches the participation traces Figure 1 reads.
  dimensionality participation ratio (sum ev)^2 / sum ev^2 of the noise-free rate covariance over
                 the ACTIVE units: how many directions the population uses. The count of components
                 reaching 95% of the variance is recorded beside it, since the two can disagree.
  rate spread    the firing-rate distribution's shape. Cortical rates are lognormal over roughly a
                 hundredfold range; a trained network departs from that in two different ways, and
                 both are recorded: the ZERO ATOM (the silent fraction, which no lognormal has) and,
                 over the active units only, the sd and skew of log(mean rate) and the q99/q01 range.
                 The weight distribution's same three statistics are recorded for comparison -
                 the weights are lognormal in every arm, so a change in the rate distribution is not
                 inherited from them.

⚠️ SCORE THE WHOLE BATCH, NOT ITS HEAD. `task.get_batch()` returns conditions in a fixed order, so
the first 128 trials of a CDDM batch are one corner of the coherence grid. Scoring that slice puts
the recomputed r2 0.03 below the stored value and, worse, calls 310 of 1000 units silent in a
network where every unit is active on the full batch. Every measure here uses the complete batch,
strided across conditions if it has to be cut at all.

⚠️ GATE. Every network is rebuilt from ITS OWN saved config and scored; anything whose recomputed
noisy r2 misses the stored value by more than R2_TOL is dropped, because a wrong forward pass still
yields a plausible-looking dimensionality and a perfectly plausible rate histogram.

⚠️ DMTS IS NOT HERE. Those runs saved a participation trace and a config but no weights, so they
cannot be re-simulated at all, and DMTS_d7_pen holds only the unpenalised arm. Figure 3 cannot
carry that task, and its caption must not claim it does.

Usage:
    python f3_penalty_cache.py                 # writes data/fig_paper_F3_cache.npz
    python f3_penalty_cache.py OUT.npz
"""

import glob
import os
import re
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, SILENT_REL, participation
# ⚠️ `participation_ratio` exists TWICE in this folder under one name and two meanings:
# common's takes a vector of participation VALUES and returns 1/HHI, the effective number of
# participating units, while f2_remedies_cache's takes a rate MATRIX and returns the participation
# ratio of its covariance spectrum, the number of directions the population uses. This figure wants
# the second. Importing the first and handing it a matrix returns ~10^7 for a 1000-unit network.
from f2_remedies_cache import load_net, n_comp_95, participation_ratio
from trainRNNbrain.trainer.Trainer import Trainer

R2_TOL = 0.05           # noisy r2 is itself a noisy estimate: on CDDM one seed's score moves by
                        # ~0.01 sd between noise draws and the stored value came from a single
                        # training-time batch, so a tolerance tighter than this rejects networks
                        # that are perfectly well rebuilt
MAX_TRIALS = 512        # if a batch is larger than this it is STRIDED, never truncated
ZERO_TOL = 1e-12

# Every cell that has trained weights on disk, as (task, glob). The penalty arm and the size are
# parsed out of the folder name, so a new cell appears here by existing rather than by being listed.
CELL_GLOBS = [
    ("CDDM", f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N=*_iters=*"),
    ("CDDM", f"{DATA_DIR}/CDDM_std_g0_penalties/EqType=h_N=*_pen=*"),
    ("flip-flop", f"{DATA_DIR}/NBitFlipFlop_std_ksweep/EqType=h_k=*_N=*_iters=*"),
    ("flip-flop", f"{DATA_DIR}/NBitFlipFlop_std_pen/EqType=h_k=*_N=*_pen=*"),
    ("flip-flop", f"{DATA_DIR}/NBitFlipFlop_std_penlong/EqType=h_k=*_N=*_pen=*_iters=*"),
]


def cell_key(task, cell, sweep):
    """Parse a cell folder name into the coordinates the figure plots against.

    ⚠️ `sweep` is part of the key, not decoration. NBitFlipFlop_std_pen and
    NBitFlipFlop_std_penlong both hold `pen=frm` cells at the same k and N, trained to different
    budgets; keyed on (task, k, N, pen) alone they merge, and a cell then silently reports n=6
    seeds pooled across two conditions.

    Args:
        task: "CDDM" or "flip-flop"; cell: the cell folder's basename; sweep: its parent folder.
    Returns:
        dict with task, sweep, N (int), k (int, 0 for CDDM) and pen ("none"/"rws"/"frm"/"both").
    """
    N = int(re.search(r"N=(\d+)", cell).group(1))
    k = re.search(r"k=(\d+)", cell)
    pen = re.search(r"pen=([a-z]+)", cell)
    return dict(task=task, sweep=sweep, N=N, k=int(k.group(1)) if k else 0,
                pen=pen.group(1) if pen else "none")


def shape_stats(v):
    """Lognormal shape statistics of a set of positive magnitudes.

    Args:
        v: 1-D array of magnitudes; non-positive entries are dropped.
    Returns:
        dict with n, sigma_log (sd of log v), skew_log (0 for an exact lognormal) and spread,
        the q99/q01 ratio as a FOLD range, or None if fewer than 100 values survive.
    """
    m = np.asarray(v, dtype=np.float64).ravel()
    m = m[m > ZERO_TOL]
    if m.size < 100:
        return None
    lg = np.log(m)
    z = (lg - lg.mean()) / lg.std(ddof=1)
    return dict(n=int(m.size), sigma_log=float(lg.std(ddof=1)), skew_log=float((z ** 3).mean()),
                spread=float(np.quantile(m, 0.99) / max(np.quantile(m, 0.01), 1e-300)))


def batch_of(task_obj):
    """The task's full batch as tensors, strided across conditions if it exceeds MAX_TRIALS.

    Striding rather than truncating matters: a CDDM batch is ordered by condition, so its head is
    one corner of the coherence grid and scoring it misreports both r2 and the active count.

    Args:
        task_obj: an instantiated task.
    Returns:
        (inputs, targets) float32 tensors, each (., T, B).
    """
    bi, bt, _ = task_obj.get_batch()
    step = max(1, int(np.ceil(bi.shape[2] / MAX_TRIALS)))
    return (torch.tensor(bi[:, :, ::step], dtype=torch.float32),
            torch.tensor(bt[:, :, ::step], dtype=torch.float32))


def analyse(net_dir):
    """Every measure for one trained network.

    Args:
        net_dir: path to one trained-network folder.
    Returns:
        dict of scalars, or None if the network has no saved weights.
    """
    rnn, task, mask, stored, d = load_net(net_dir)
    # the training budget this run actually reached. The arms of this figure were NOT trained to a
    # common budget - the unpenalised flip-flop cells run to 500k iterations and the penalised ones
    # to 400k - and silencing continues long after the loss plateaus (Fig. 1e), so the budget has
    # to travel with the measurement rather than be assumed equal.
    cfg = OmegaConf.load(glob.glob(os.path.join(net_dir, "*_config.yaml"))[0])
    bi, bt = batch_of(task)
    with torch.no_grad():
        _, out = rnn(bi, w_noise=True)
        r2_noisy = float(Trainer.r2_score(out, bt, mask))
        srec, sinp = float(rnn.sigma_rec), float(rnn.sigma_inp)
        rnn.sigma_rec = rnn.sigma_inp = 0.0
        states, out_c = rnn(bi, w_noise=False)
        rnn.sigma_rec, rnn.sigma_inp = srec, sinp
        r2_clean = float(Trainer.r2_score(out_c, bt, mask))
        rates = torch.relu(states).numpy().astype(np.float64)      # (N, T, B)

    flat = rates.reshape(rnn.N, -1)
    p = participation(rates)
    live = p >= SILENT_REL * np.quantile(p, 0.95)
    row = dict(stored=stored, r2_noisy=r2_noisy, r2_clean=r2_clean,
               max_iter=int(cfg.trainer.max_iter), n_active=int(live.sum()), N=int(rnn.N),
               dim_pr=float("nan"), dim95=float("nan"))
    if live.sum() >= 2:
        row["dim_pr"] = float(participation_ratio(flat[live]))
        row["dim95"] = float(n_comp_95(flat[live]))

    # the rate distribution, over the ACTIVE units only: the silent units are the zero atom, and a
    # lognormal has no atom at zero, so mixing them in would describe neither piece
    mean_rate = flat.mean(axis=1)
    for tag, vals in (("rate", mean_rate[live]), ("w", np.abs(np.asarray(d["W_rec"], float)))):
        st = shape_stats(vals)
        for key in ("sigma_log", "skew_log", "spread"):
            row[f"{tag}_{key}"] = float("nan") if st is None else st[key]
    return row


def main(out_path):
    """Score every network of every cell and write the cache. Returns the output path."""
    recs = []
    cells = [(task, c) for task, g in CELL_GLOBS for c in sorted(glob.glob(g)) if os.path.isdir(c)]
    print(f"{len(cells)} cells to score")
    for task, cell in cells:
        key = cell_key(task, os.path.basename(cell), os.path.basename(os.path.dirname(cell)))
        for nd in sorted(glob.glob(os.path.join(cell, "*"))):
            if not os.path.isdir(nd) or not glob.glob(os.path.join(nd, "*LastParams*.npz")):
                continue
            try:
                r = analyse(nd)
            except Exception as e:
                print(f"  SKIP {os.path.basename(cell)[:38]:38s} {type(e).__name__}: {e}",
                      flush=True)
                continue
            gate = abs(r["stored"] - r["r2_noisy"]) < R2_TOL
            print(f"  {key['task']:9s} k={key['k']} N={key['N']:5d} {key['pen']:5s} "
                  f"stored {r['stored']:7.4f} recomp {r['r2_noisy']:7.4f} "
                  f"{'PASS' if gate else 'FAIL'}  active {r['n_active']:5d}  "
                  f"dim {r['dim_pr']:6.2f}  rate spread {r['rate_spread']:8.1f}x", flush=True)
            if not gate:
                continue
            r.update(key)
            recs.append(r)
    if not recs:
        raise SystemExit("no network passed the gate - nothing written")
    fields = sorted(recs[0])
    arrays = {f: np.array([r[f] for r in recs]) for f in fields}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    print(f"\nwrote {out_path}: {len(recs)} networks over "
          f"{len({(r['task'], r['sweep'], r['k'], r['N'], r['pen']) for r in recs})} cells")
    return out_path


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(DATA_DIR), "fig_paper_F3_cache.npz"))
