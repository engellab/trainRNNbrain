#!/usr/bin/env python3
"""A3: do the units a remedy recruits do work? Ablate them, and compare with random ablations.

NETWORKS. Every run of each remedy's size series (a remedy setting present at three or more sizes,
discovered with read_remedies.all_cells / cell_info; no cell list) and of Figure 1's flip-flop
control at the same N. Each network is read at 75,000 iterations: these are same-size comparisons,
which the 16 Oct rule reads at a fixed 75k, and every run has weights saved there (the end of its
_c75000 parent, or a kept snapshot for runs trained in one piece).

PER NETWORK (ablate_silent_units machinery: build from the saved config, seeded held-out batch,
participation on a noise-free training batch, a unit removed by zeroing its columns of W_rec and
W_out, held-out R^2 under the training noise with sigma_w = 0, DRAWS = 9 seeded draws):
  r2_full           the intact network;
  random_<f>        RANDOM_ORDERS random sets of a fraction f = 10, 30, 50% of all N units removed;
  n_active          units with p >= 0.05 q95(p) (offline count);
  surplus           n_active minus the mean offline active count of the control at the same N: the
                    units the remedy recruited beyond the control (none for a control or k <= 0);
  r2_recruited      the `surplus` least-participating active units removed;
  r2_random_active  `surplus` active units drawn at random removed, RANDOM_ORDERS sets;
  r2_keep_ctrl      every unit but the n_ctrl most-participating removed, n_ctrl = the control's mean
                    active count: the remedy network cut to the control's active count;
  r2_no_inactive    every inactive unit removed;
  r2_random_keep    N - n_ctrl units drawn at random from all N removed, RANDOM_ORDERS sets, from a
                    generator of its own so the keys above are unchanged (round 18, G3 and G5: the
                    random counterpart of r2_keep_ctrl, the removal the maze analysis makes).
Reading: if the recruited units carry no task signal, r2_recruited stays near r2_full while
r2_random_active falls; if they do work, the two are close.

Usage (repository root; GPU or CPU node):
  python trainRNNbrain/experiments_and_analysis/a3_remedy_ablation.py run OUT_DIR --N 1000
  python trainRNNbrain/experiments_and_analysis/a3_remedy_ablation.py summary OUT_DIR
`run` writes OUT_DIR/a3_N<N>.json (controls first, so the surplus is known for the arms);
`summary` prints one row per cell from every a3_N*.json.
"""

import argparse
import glob
import json
import os
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ablate_silent_units as A
from common import DATA_DIR, active_mask
from read_remedies import all_cells, cell_info

FIXED_ITER = 75000
FRACS = (0.1, 0.3, 0.5)
RANDOM_ORDERS = A.RANDOM_ORDERS


def size_series_cells():
    """Cells to ablate, grouped by N.

    Returns:
        dict N -> list of (cell path relative to DATA_DIR, family, role), Figure 1's control first; the
        arms are every remedy cell whose (family, signature) appears at three or more sizes.
    """
    info = {os.path.relpath(c, DATA_DIR): (role, cell_info(os.path.relpath(c, DATA_DIR)))
            for c, role in all_cells()}
    sizes = {}
    for cell, (role, i) in info.items():
        if role in ("topup", "b2"):
            sizes.setdefault((i["family"], i["signature"]), set()).add(i["N"])
    series = {k for k, ns in sizes.items() if len(ns) >= 3}
    out = {}
    for cell, (role, i) in sorted(info.items()):
        if role == "control":
            out.setdefault(i["N"], []).insert(0, (cell, "control", role))
        elif role in ("topup", "b2") and (i["family"], i["signature"]) in series:
            out.setdefault(i["N"], []).append((cell, i["family"], role))
    return out


def weights_at(run_dir, it=FIXED_ITER):
    """The parameter file of a run's weight state at iteration `it` (segment end or kept snapshot), or None."""
    its, files = A.chain_snapshots(run_dir)
    hit = [f for i, f in zip(its, files) if int(i) == it]
    return hit[0] if hit else None


def ablate_net(run_dir, n_ctrl, it=FIXED_ITER, pf=None):
    """All A3 scores for one network at iteration `it` (75k unless another is given).

    Args:
        run_dir: a run folder; n_ctrl: the control's mean offline active count at this N, or None
            while the control itself is being read; it: int, the iteration whose weights are read
            (round 18 reads CDDM networks that kept weights only at 100k); pf: str or None, a
            parameter file to read instead of looking one up at `it` (a run whose warm-start parent
            is not on this machine, so its chain's iterations cannot be resolved).
    Returns:
        JSON-ready dict of the scores listed in the module docstring, or None without weights at `it`.
    """
    pf = weights_at(run_dir, it) if pf is None else pf
    if pf is None:
        return None
    cfg, rnn = A.build(run_dir, A.load_params(pf))
    batches = A.seeded_batches(cfg, rnn.device)
    with torch.no_grad():
        states = rnn(batches["tr"][0], w_noise=False)[0]
        p = A.participation(states).cpu().numpy()
        hmax = states.amax(dim=(1, 2)).cpu().numpy()
        del states
    N = len(p)
    live = active_mask(p)
    n_active = int(live.sum())
    order = np.lexsort((hmax, p))                       # least participating first
    rng = np.random.default_rng(zlib.crc32(run_dir.encode()))
    score = A.Scorer(rnn, batches["ho"])
    res = dict(run_dir=run_dir, params_file=pf, N=N, n_active=n_active, r2_full=score()[0])
    perms = [rng.permutation(N) for _ in range(RANDOM_ORDERS)]
    for f in FRACS:
        res[f"random_{int(100 * f)}"] = [score(o[:int(round(f * N))])[0] for o in perms]
    res["r2_no_inactive"] = score(np.flatnonzero(~live))[0]
    if n_ctrl is not None:
        k = n_active - int(round(n_ctrl))
        res["surplus"] = k
        if k > 0:
            act_order = [u for u in order if live[u]]           # active units, least participating first
            res["r2_recruited"] = score(act_order[:k])[0]
            act_idx = np.flatnonzero(live)
            res["r2_random_active"] = [score(rng.choice(act_idx, size=k, replace=False))[0]
                                       for _ in range(RANDOM_ORDERS)]
            res["r2_keep_ctrl"] = score(order[:N - int(round(n_ctrl))])[0]
            rng_keep = np.random.default_rng(zlib.crc32((run_dir + "|keep").encode()))
            res["r2_random_keep"] = [score(rng_keep.choice(N, size=N - int(round(n_ctrl)), replace=False))[0]
                                     for _ in range(RANDOM_ORDERS)]
    return res


def run(out_dir, N):
    """Ablate every network at size N, control first; write OUT_DIR/a3_N<N>.json."""
    cells = size_series_cells().get(N, [])
    out, n_ctrl = [], None
    for cell, family, role in cells:
        for d in sorted(x for x in glob.glob(os.path.join(DATA_DIR, cell, "*")) if os.path.isdir(x)):
            if os.path.basename(d).startswith("nan"):
                continue
            r = ablate_net(d, None if role == "control" else n_ctrl)
            if r is None:
                print(f"  {cell[:70]} {os.path.basename(d)[:10]}: no weights at {FIXED_ITER}", flush=True)
                continue
            r.update(cell=cell, family=family, role=role)
            out.append(r)
            print(f"  {family:9s} {cell[:70]:70s} active {r['n_active']:5d} R2 {r['r2_full']:.4f} "
                  f"rand10/30/50 {np.mean(r['random_10']):.3f}/{np.mean(r['random_30']):.3f}/"
                  f"{np.mean(r['random_50']):.3f}"
                  + (f" surplus {r['surplus']} recruited {r.get('r2_recruited', np.nan):.4f} "
                     f"random-active {np.mean(r.get('r2_random_active', [np.nan])):.4f} "
                     f"keep-ctrl {r.get('r2_keep_ctrl', np.nan):.4f}" if 'surplus' in r else ""), flush=True)
        if role == "control":
            n_ctrl = float(np.mean([r["n_active"] for r in out if r["role"] == "control"]))
            print(f"  control mean active at N={N}: {n_ctrl:.1f}", flush=True)
        json.dump(out, open(os.path.join(out_dir, f"a3_N{N}.json"), "w"), indent=1)


def summary(out_dir):
    """Print one row per cell (medians over its networks) from every a3_N*.json."""
    rows = [r for f in sorted(glob.glob(os.path.join(out_dir, "a3_N*.json"))) for r in json.load(open(f))]
    cells = {}
    for r in rows:
        cells.setdefault((r["N"], r["family"], r["cell"]), []).append(r)
    m = lambda rs, k: float(np.median([np.median(r[k]) for r in rs if k in r])) if any(k in r for r in rs) else float("nan")
    print("medians over each network's random sets, then over the cell's networks (a removal can make the\n"
          "dynamics diverge and R^2 run to -1e6, so means are not used)")
    print(f"{'N':>5s} {'family':9s} n  active  R2full  -10%   -30%   -50%   -inact  surplus  -recruited  -rand.active  keep_ctrl")
    for (N, fam, cell), rs in sorted(cells.items()):
        print(f"{N:5d} {fam:9s} {len(rs)} {m(rs, 'n_active'):7.0f} {m(rs, 'r2_full'):7.4f} "
              f"{m(rs, 'random_10'):6.3f} {m(rs, 'random_30'):6.3f} {m(rs, 'random_50'):6.3f} "
              f"{m(rs, 'r2_no_inactive'):7.4f} {m(rs, 'surplus'):7.0f}  {m(rs, 'r2_recruited'):10.4f}  "
              f"{m(rs, 'r2_random_active'):12.4f}  {m(rs, 'r2_keep_ctrl'):9.4f}   {cell}")


def main():
    """Parse the command line and run `run` or `summary`."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("out_dir")
    ap.add_argument("--N", type=int)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.mode == "run":
        run(a.out_dir, a.N)
    else:
        summary(a.out_dir)


if __name__ == "__main__":
    main()
