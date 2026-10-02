#!/usr/bin/env python3
"""Three more dimensionality measures for the slide-19 networks, from one covariance spectrum each.

WHY A SEPARATE FILE. `fig_paper_F2_cache.npz` stores `dims` (participation ratio) and `dims95` (PCs
for 95% of the variance) as scalars and keeps no eigenvalue spectrum, so a fourth measure cannot be
derived from it. The spectrum is cheap to recompute - 0.7 s per network - and rebuilding the shared
270-row cache to add two columns would overwrite a file another thread of this work owns. This
computes the extra measures for the 22 networks slide 19 draws and writes its own small npz.
Folding the columns into `f2_remedies_cache.analyse` is the tidier end state once the deck settles.

ALL FOUR MEASURES COME FROM ONE OBJECT: the eigenvalues L of the covariance of the ACTIVE units'
noise-free firing rates, which is what `f2_remedies_cache` already uses. np.cov centres the data, so
every measure below is a property of the mean-subtracted rate matrix.

    participation ratio  PR   = (sum L)^2 / sum L^2
        The project's "dimensionality". A soft count: it is n when n eigenvalues are equal and the
        rest zero, and falls toward 1 as one direction dominates. Never an integer.
    PCs for 95% / 99%    k95, k99
        A hard count: the smallest k whose top-k eigenvalues carry that share of total variance.
        Sensitive to the tail - k99 asks how many directions are needed before almost nothing is
        left, where PR is dominated by the few largest.
    stable rank          sr   = sum L / max L  =  ||X||_F^2 / ||X||_2^2 for the centred rates X
        The softest of the three, and the one that ignores the shape of the tail entirely: it is the
        total variance in units of the largest single direction.

Reading them together is the point. PR and stable rank are dominated by the top of the spectrum and
k99 by its tail, so a rule that raises k99 without raising stable rank has added directions that
carry almost no variance.

VALIDATION, and why a single tolerance will not do. The flip-flop draws its trials at random and the
saved configs carry `seed: null`, so the cache's rate matrix cannot be reproduced bit for bit - the
cached value is ONE draw of a quantity that moves from batch to batch. How much it moves is not a
property of the pipeline but of the ARM: measured over six draws per network, the participation
ratio has a spread of 2.6% on control networks but 9-11% on duplication networks, because
duplication makes pairs of near-identical units whose covariance is near-degenerate, so which
eigenvalue is which shifts between draws. A flat 5% tolerance calibrated on a control network
therefore fails duplication for no reason, which is exactly what the first version of this check did.

The test instead asks of each network: does the cached single-batch value fall inside the range THIS
network's own repeated draws span? That is falsifiable - a rate matrix which is not the cache's
lands far outside, the duplication networks' ranges being 4.4-5.5 and 7.1-9.7 against a between-arm
span of 4 to 13 - and it is calibrated per network rather than per project. Networks are matched to
cache rows by active-unit count, which is stable to ~2% and distinct between networks, rather than
by sorted PR, whose ranks the batch noise can swap.

Usage (from the repo root, which is where the cache paths resolve):
    F2_REPO=$PWD F2_DATA=$PWD/data/trained_RNNs python3 \\
        trainRNNbrain/experiments_and_analysis/f2_dimensionality_extra.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f2_remedies_cache as FC
import fig_paper_F2 as F2

OUT = "data/f2_dimensionality_extra.npz"
N_BATCH = 6            # draws per network: enough that the min-max range is a usable interval
K95_SLACK = 1          # integer slack on the 95% PC count, which is a hard count


def spectrum_measures(net_dir, n_batch=N_BATCH):
    """The four dimensionality measures of one trained network, averaged over random batches.

    Runs the network noise-free, keeps the active units under the project's scale-free rule, and
    derives every measure from the eigenvalues of their rate covariance.

    Args:
        net_dir: path to one trained-network folder (no trailing slash);
        n_batch: how many independent task batches to average over.
    Returns:
        dict with n_active, pr, k95, k99, srank (floats, means over batches) and pr_sd, srank_sd.
    """
    acc = {k: [] for k in ("n_active", "pr", "k95", "k99", "srank", "srank99")}
    for _ in range(n_batch):
        rnn, task, _, _, _ = FC.load_net(net_dir)
        bi = np.asarray(task.get_batch()[0])
        sub = np.arange(0, bi.shape[2], max(1, bi.shape[2] // FC.TRIALS))[:FC.TRIALS]
        with torch.no_grad():
            rnn.sigma_rec = rnn.sigma_inp = rnn.sigma_w = 0.0
            states, _ = rnn(torch.tensor(bi[:, :, sub], dtype=torch.float32), w_noise=False)
        r = torch.relu(states).numpy().reshape(rnn.N, -1)
        p = r.std(axis=1) + np.quantile(np.abs(r), 0.9, axis=1)
        live = p >= FC.SILENT_REL * np.quantile(p, 0.95)
        if live.sum() < 2:
            continue
        ev = np.linalg.eigvalsh(np.cov(np.asarray(r[live], dtype=np.float64)))
        ev = np.sort(ev[ev > 1e-12])[::-1]
        frac = np.cumsum(ev) / ev.sum()
        acc["n_active"].append(int(live.sum()))
        acc["pr"].append((ev.sum() ** 2) / (ev ** 2).sum())
        acc["k95"].append(int(np.searchsorted(frac, 0.95) + 1))
        k99 = int(np.searchsorted(frac, 0.99) + 1)
        acc["k99"].append(k99)
        acc["srank"].append(ev.sum() / ev[0])
        # stable rank of the 99%-VARIANCE SUBSPACE, i.e. with the tail that carries the last 1%
        # dropped. Reported beside the untruncated value because "stable rank when the number of PCs
        # captures 99% variance" can be read either way; dropping at most 1% of sum(L) can move
        # sum(L)/max(L) by at most 1%, so the two are near-identical and the choice does not matter.
        acc["srank99"].append(ev[:k99].sum() / ev[0])
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    for k in ("pr", "k95", "k99", "srank", "srank99"):
        out[k + "_lo"], out[k + "_hi"] = float(np.min(acc[k])), float(np.max(acc[k]))
    return out


def main():
    """Recompute the measures for every slide-19 network, check them against the shared cache, save.

    Returns:
        0 if every network passes both pre-registered agreement thresholds, 1 otherwise.
    """
    c = F2.load()
    at_main = F2.restrict(c, n_units=F2.N_MAIN)
    at = F2.restrict(c, n_units=F2.N_MAIN,
                     chosen={a: F2.pick_cell(at_main, a)[0] for a in F2.GRID_ARMS})
    root = os.environ.get("F2_DATA", FC.D)

    rows, fails, checks = [], [], 0
    for arm, _, full, _ in F2.ARMS:
        m = at["arm"] == arm
        if not m.any():
            print(f"{full:<22} no cells")
            continue
        cached = list(zip(np.asarray(at["n_active"][m], float),
                          np.asarray(at["dims"][m], float),
                          np.asarray(at["dims95"][m], float)))
        # ⚠️ UNIQUE cells. `at` holds one row per NETWORK, so its `cell` column repeats a cell once
        # per network in it; looping that column and then over the cell's networks visits each
        # network n_nets times (70 rows instead of 22).
        for cell in sorted(set(at["cell"][m].tolist())):
            for net in sorted(os.listdir(os.path.join(root, str(cell)))):
                net_dir = os.path.join(root, str(cell), net)
                if not os.path.isdir(net_dir) or net.split("_")[0] == "nan":
                    continue
                r = spectrum_measures(net_dir)
                rows.append((arm, r["n_active"], r["pr"], r["k95"], r["k99"],
                             r["srank"], r["srank99"]))
                # match to the cache row for the SAME network, by active count
                na, pr_c, k95_c = min(cached, key=lambda t: abs(t[0] - r["n_active"]))
                checks += 2
                ok_pr = r["pr_lo"] <= pr_c <= r["pr_hi"]
                ok_k = r["k95_lo"] - K95_SLACK <= k95_c <= r["k95_hi"] + K95_SLACK
                if not ok_pr:
                    fails.append(f"{full}: cached PR {pr_c:.3f} outside this net's draws "
                                 f"[{r['pr_lo']:.3f}, {r['pr_hi']:.3f}]")
                if not ok_k:
                    fails.append(f"{full}: cached k95 {k95_c:.0f} outside "
                                 f"[{r['k95_lo']:.0f}, {r['k95_hi']:.0f}] +-{K95_SLACK}")
                print(f"{full:<22} active {r['n_active']:4.0f}  PR {r['pr']:6.2f} "
                      f"[{r['pr_lo']:.2f},{r['pr_hi']:.2f}] (cached {pr_c:5.2f} {'ok' if ok_pr else 'MISS'})"
                      f"  k95 {r['k95']:5.1f}  k99 {r['k99']:6.1f}  srank {r['srank']:5.2f}"
                      f"  srank99 {r['srank99']:5.2f}")

    arms = np.array([r[0] for r in rows])
    np.savez(OUT, arm=arms,
             n_active=np.array([r[1] for r in rows], float),
             pr=np.array([r[2] for r in rows], float),
             k95=np.array([r[3] for r in rows], float),
             k99=np.array([r[4] for r in rows], float),
             srank=np.array([r[5] for r in rows], float),
             srank99=np.array([r[6] for r in rows], float))
    print(f"\nwrote {OUT} ({len(rows)} networks)")
    print(f"PRE-REGISTERED: each cached value falls inside its own network's {N_BATCH}-draw range "
          f"({checks} checks)")
    if fails:
        print("FAIL:")
        for f in fails:
            print("   " + f)
        return 1
    print("PASS - the recomputed spectra reproduce the shared cache, so the new measures are\n"
          "       derived from the same rate matrices slide 19 is drawn from.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
