#!/usr/bin/env python3
"""
Measure everything the talk's "why is this hard?" section claims, once, and cache it.

Three claims, three blocks, nothing typed in by hand:

  FROZEN      A ReLU unit held below threshold gets exactly zero gradient on every weight into it
              AND on every weight out of it, so no term added to the loss can move it. Measured by
              running autograd on a small network wired silent, which is the same premise the
              repository's own `tests/test_prune_and_reinit.py` asserts; this block records the
              numbers so the figure shows a measurement rather than an algebraic claim.

  TREADMILL   Force those units back on and the network turns them off again. Read off the three
              CDDM networks of the pruning arm: how many times the rule fired, over how many
              distinct units, and what it bought in working units against its own matched control.

  DECOMPOSE   Copying a working unit DOES recruit. Four matched flip-flop cells separate what the
              copy carries -- the donor's outgoing wires, the sizes of its incoming weights, and
              the places those weights sit in -- so the recruitment can be split three ways.

EVERY ACTIVE-UNIT COUNT IS NOISE-FREE. Injected noise puts a floor under every unit's variance and
the relative silence criterion then counts units the task never drives; see the project's
participation-is-measured-noise-free rule. Networks are rebuilt from their own saved config and the
recomputed score is checked against the one stored in the folder name, so a cell whose task class
has moved under it drops out instead of quietly scoring at chance.

Usage:  python why_hard_cache.py [--refresh]
Output: data/why_hard_cache.npz
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "..")))
from common import DATA_DIR, participation, active_count
from f2_remedies_cache import load_net
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

OUT = "data/why_hard_cache.npz"
R2_TOL = 0.04         # how far a rebuilt network's recomputed score may miss its stored one

# ⚠️ COUNTS ARE READ ON THE FULL TASK BATCH, never a subsample. CDDM's batch enumerates its
# coherence x context grid in a fixed order, so taking k of 450 trials samples one corner of that
# grid whenever the stride collapses to 1: at k = 256 the three control networks counted 309, 319
# and 337 active units against the full batch's 399, 412 and 414, and the trajectory's own figure of
# 410.3 is the full-batch number. The full batch at N = 1000 is 1000 x 300 x 450 float32, about
# half a gigabyte, which is why this is said explicitly rather than left as a default.

# ---- TREADMILL: the pruning screen on CDDM. Both arms are the same 30,000-iteration run at
# N = 1000 with no penalties, differing only in whether silent units are redrawn.
TREADMILL = {
    "control": f"{DATA_DIR}/CDDM_screen3/EqType=h_N=1000_pen=none_arm=none",
    "redraw":  f"{DATA_DIR}/CDDM_screen3/EqType=h_N=1000_pen=none_arm=prune",
}

# ---- DECOMPOSE: 3-bit flip-flop, N = 1000, 40,000 iterations, replacement rate 0.025, maturity
# 1000, three seeds each, identical but for what a replaced unit is handed. The `copy` cells under
# NBitFlipFlop_replace_variants are excluded on purpose: they carry the detuned self-weight block
# (named __DETUNED_SELFWEIGHT on disk) that was fixed on 2026-09-24.
DECOMPOSE = {
    "control":  f"{DATA_DIR}/NBitFlipFlop_ff_revive/EqType=h_k=3_N=1000_pen=none_arm=none",
    "random":   f"{DATA_DIR}/NBitFlipFlop_replace_variants/"
                "EqType=h_k=3_N=1000_mode=random_rate=0.025_mat=1000",
    "zero_out": f"{DATA_DIR}/NBitFlipFlop_replace_variants/"
                "EqType=h_k=3_N=1000_mode=zero_out_rate=0.025_mat=1000",
    "copy_iid": f"{DATA_DIR}/NBitFlipFlop_copy_iid/EqType=h_k=3_N=1000_cn=0",
    "permute":  f"{DATA_DIR}/NBitFlipFlop_copy_permute/EqType=h_k=3_N=1000_cn=0",
    "copy":     f"{DATA_DIR}/NBitFlipFlop_copy_perturb/EqType=h_k=3_N=1000_cn=0",
    # the bias kick belongs to the treadmill story, not the decomposition, but it is a flip-flop
    # cell from the same array and is measured here against the same control.
    "bias_kick": f"{DATA_DIR}/NBitFlipFlop_replace_variants/"
                 "EqType=h_k=3_N=1000_mode=bias_kick_rate=0.025_mat=1000",
}


# -------------------------------------------------------------------------------------------------
# FROZEN: the gradient on a silent unit's weights, measured
# -------------------------------------------------------------------------------------------------

def frozen_gradients(n=30, t_steps=12, batch=4, n_silent=4, seed=0):
    """Largest weight gradient into and out of silent units, and out of firing ones, by autograd.

    A small ReLU recurrent network is built, a few of its units are wired with strongly negative
    incoming weights so their input stays below threshold for every input, the squared output is
    backpropagated, and the largest absolute gradient on the recurrent weights is read off for the
    silent rows/columns and the firing ones. The silent entries come out as exact zeros, which is
    the premise the whole intervention section rests on.

    Args:
        n: units in the test network; t_steps: timesteps; batch: trials;
        n_silent: how many units are wired permanently below threshold; seed: weight-draw seed.
    Returns:
        dict with keys "silent_in", "silent_out", "firing_in", "firing_out" (float, largest
        absolute gradient of the loss with respect to a recurrent weight in that group) and
        "silent_rate_max" (float, largest rate reached by a silent unit; 0 if the wiring worked).
    """
    rnn = RNN_torch(N=n, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=1, equation_type="h", seed=seed)
    silent = np.arange(n_silent)
    firing = np.setdiff1d(np.arange(n), silent)
    with torch.no_grad():
        rnn.W_rec[silent, :] = -5.0
        rnn.W_inp[silent, :] = -5.0
    inp = torch.abs(torch.randn(2, t_steps, batch, generator=torch.Generator().manual_seed(2)))
    states, out = rnn(inp, w_noise=False)
    r = torch.relu(states)
    out.pow(2).mean().backward()
    g = rnn.W_rec.grad
    return dict(silent_in=float(g[silent, :].abs().max()),
                silent_out=float(g[:, silent].abs().max()),
                firing_in=float(g[firing, :].abs().max()),
                firing_out=float(g[:, firing].abs().max()),
                silent_rate_max=float(r[silent].detach().abs().max()))


# -------------------------------------------------------------------------------------------------
# counts from trained networks
# -------------------------------------------------------------------------------------------------

def noise_free_rates(net_dir):
    """Noise-free rates of one trained network, with the score gate that says the rebuild is right.

    The network is rebuilt from its own saved config, scored in the noisy condition it trained in
    (the quantity the trainer wrote into the folder name), and only then run noise-free for the
    rates. A task class that has changed since training scores a good network near chance while
    still producing a perfectly plausible-looking rate matrix, which is why the gate exists.

    Args:
        net_dir: path to one trained-network folder.
    Returns:
        ((N, T, B) float array of noise-free rates over the FULL task batch, stored r2 read off the
        folder name, recomputed r2 in the trained noise condition).
    """
    rnn, task, mask, stored, _ = load_net(net_dir)
    bi_np, bt_np, _ = task.get_batch()
    bi = torch.tensor(np.asarray(bi_np), dtype=torch.float32)
    bt = torch.tensor(np.asarray(bt_np), dtype=torch.float32)
    with torch.no_grad():
        got = float(Trainer.r2_score(rnn(bi, w_noise=True)[1], bt, mask))
        rnn.sigma_rec = rnn.sigma_inp = 0.0
        states, _ = rnn(bi, w_noise=False)
    r = rnn.activation(states) if rnn.equation_type == "h" else states
    return r.detach().cpu().numpy(), stored, got


def redraw_counters(net_dir):
    """How many times a revival rule fired in one run, and over how many distinct units.

    The trainer logs both counters at every checkpoint; the last entry is the run's total. A run
    with no revival rule has neither counter and returns NaNs.

    Args:
        net_dir: path to one trained-network folder.
    Returns:
        (events, units_ever) as floats, NaN where the run logged no such counter.
    """
    hit = glob.glob(os.path.join(net_dir, "*ParticipationTrace.pkl"))
    if not hit:
        return float("nan"), float("nan")
    with open(hit[0], "rb") as fh:
        trace = pickle.load(fh)
    m = trace.get("metrics")
    m = m.item() if hasattr(m, "item") else m
    if not isinstance(m, dict):
        return float("nan"), float("nan")
    out = []
    for key in ("reinit_events", "reinit_units_ever"):
        v = m.get(key)
        out.append(float("nan") if v is None else float(np.asarray(v)[-1]))
    return out[0], out[1]


def training_curves(cell):
    """How the unit counts and the revival counter move over training, for every seed of a cell.

    Read out of the participation trace the trainer writes during the run. That trace is the
    project's canonical record of these counts -- it is logged from a noise-free forward pass, and
    it is the source the project trajectory's own screen table was built from -- so it is kept here
    beside the independent end-of-training recompute in `measure_cell`. The two routes share no
    code: one reads the trainer's log, the other rebuilds the network from its weights and runs the
    task again. Agreement between them is the check that neither is wrong.

    Args:
        cell: a sweep-cell directory holding one folder per seed.
    Returns:
        dict with "it" (checkpoint iterations, shape (K,)), "active" and "never" (seeds x K unit
        counts), "ev_it" (metric iterations, shape (M,)) and "events" (seeds x M cumulative count of
        times the revival rule fired). The event arrays are absent when the runs logged no counter.
        Empty dict if the cell holds no trace.
    """
    act, nev, evs, it, ev_it = [], [], [], None, None
    for net_dir in sorted(glob.glob(os.path.join(cell, "*"))):
        hit = glob.glob(os.path.join(net_dir, "*ParticipationTrace.pkl"))
        if not hit:
            continue
        with open(hit[0], "rb") as fh:
            trace = pickle.load(fh)
        p = np.asarray(trace["participation"], float)            # (K, N), noise-free
        it = np.asarray(trace["participation_iters"], float)
        act.append([active_count(p[k], "scalefree") for k in range(p.shape[0])])
        nev.append([p.shape[1] - active_count(p[k], "hard") for k in range(p.shape[0])])
        m = trace.get("metrics")
        m = m.item() if hasattr(m, "item") else m
        if isinstance(m, dict) and m.get("reinit_events") is not None:
            evs.append(np.asarray(m["reinit_events"], float))
            ev_it = np.asarray(trace["iters"], float)
    if not act:
        return {}
    out = dict(it=it, active=np.asarray(act, float), never=np.asarray(nev, float))
    if evs and ev_it is not None:
        out["ev_it"] = ev_it
        out["events"] = np.asarray(evs, float)
    return out


def measure_cell(cell, want_counters=False):
    """Active units, never-firing units and (optionally) revival counters for every seed of a cell.

    Args:
        cell: a sweep-cell directory holding one folder per seed;
        want_counters: also read the revival counters out of each run's participation trace.
    Returns:
        dict of 1-D arrays over the seeds that passed the score gate -- "active" (units clearing the
        scale-free criterion), "never" (units whose participation is below the 1e-6 floor, i.e. that
        never fire at all), "n_units", "r2_stored", "r2_got" (the recomputed score), and with
        want_counters also "events" and "units_ever". Empty arrays if no network passed.
    """
    seeds = [d for d in sorted(glob.glob(os.path.join(cell, "*"))) if os.path.isdir(d)]
    keys = ("active", "never", "n_units", "r2_stored", "r2_got", "events", "units_ever")
    got = {k: [] for k in keys}
    for net_dir in seeds:
        try:
            r, stored, scored = noise_free_rates(net_dir)
        except Exception as exc:                       # a cell whose task class has moved
            print(f"    skip {os.path.basename(net_dir)[:28]}: {type(exc).__name__} {exc}")
            continue
        if not np.isfinite(scored) or abs(scored - stored) > R2_TOL:
            print(f"    GATE {os.path.basename(net_dir)[:28]}: stored {stored:.4f}, "
                  f"recomputed {scored:.4f}")
            continue
        p = participation(r)
        n = r.shape[0]
        got["active"].append(active_count(p, "scalefree"))
        got["never"].append(n - active_count(p, "hard"))
        got["n_units"].append(n)
        got["r2_stored"].append(stored)
        got["r2_got"].append(scored)
        ev, ue = redraw_counters(net_dir) if want_counters else (float("nan"),) * 2
        got["events"].append(ev)
        got["units_ever"].append(ue)
    return {k: np.asarray(v, float) for k, v in got.items()}


def main(refresh=False):
    """Build the cache. Returns the dict that was written."""
    if os.path.exists(OUT) and not refresh:
        print(f"{OUT} exists; pass --refresh to rebuild")
        return dict(np.load(OUT, allow_pickle=True))

    out = {}
    g = frozen_gradients()
    assert g["silent_rate_max"] == 0.0, "the silent units are not actually silent"
    for k, v in g.items():
        out[f"grad|{k}"] = np.asarray(v, float)
    print(f"frozen: silent in {g['silent_in']:.3g}, out {g['silent_out']:.3g}; "
          f"firing in {g['firing_in']:.3g}, out {g['firing_out']:.3g}")

    for block, cells in (("treadmill", TREADMILL), ("decompose", DECOMPOSE)):
        for arm, cell in cells.items():
            if not os.path.isdir(cell):
                print(f"  SKIP {block}/{arm}: {cell} missing")
                continue
            got = measure_cell(cell, want_counters=True)
            if not len(got["active"]):
                print(f"  SKIP {block}/{arm}: no readable network")
                continue
            for k, v in got.items():
                out[f"{block}|{arm}|{k}"] = v
            if block == "treadmill":
                for k, v in training_curves(cell).items():
                    out[f"treadmill|{arm}|curve_{k}"] = v
                end = out.get(f"treadmill|{arm}|curve_active")
                if end is not None:
                    print(f"    cross-check {arm}: trace says {end[:, -1].mean():.1f} "
                          f"+/- {end[:, -1].std(ddof=1):.1f} active at the end, "
                          f"rebuild-and-rerun says {got['active'].mean():.1f} "
                          f"+/- {got['active'].std(ddof=1):.1f}")
            ev = got["events"]
            print(f"  {block:9s} {arm:10s} n={len(got['active'])} "
                  f"active {got['active'].astype(int).tolist()} "
                  f"never {got['never'].astype(int).tolist()}"
                  + ("" if not np.isfinite(ev).any()
                     else f" | fired {np.nanmean(ev):.0f} over "
                          f"{np.nanmean(got['units_ever']):.0f} units"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-measure even if the cache exists")
    main(refresh=ap.parse_args().refresh)
