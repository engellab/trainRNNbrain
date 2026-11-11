"""Fraction of ALL inactive units that are active 5,500 iterations later, in the networks of Fig. 4a-c.

Results 4 compares the Maze count (17% and 19% of all units inactive at the learned point are active
700 updates later) with 3BFF and CDDM networks. The 3BFF/CDDM numbers printed so far are fractions of
subclasses (silent units; low-active units above the gradient threshold). This script gives the
3BFF/CDDM fraction over the same denominator as the Maze one: every inactive unit.

Inputs (written by fig_paper_F4_noreturn.py fig1d, entry 2026-11-01_1021):
  data/fig4_noreturn/activity_fig1d/<task>_s<seed>__act.npz   participation p_0, p_500, p_5500
  data/fig4_noreturn/c4_fig1d/*s<seed>__floor.npz             ref_n_on (steps with x > 0), strict_g_in
The active rule is the paper's: p >= 0.05 * q95(p) over the network's units. The 3BFF values use the one
reference batch (task seed 12345), not the twelve batches of the 3BFF ranges in the text.

Usage: python trainRNNbrain/experiments_and_analysis/r23_reactivation_all_inactive.py
"""
import glob

import numpy as np

ACT_DIR = "data/fig4_noreturn/activity_fig1d"
C4_DIR = "data/fig4_noreturn/c4_fig1d"
SEEDS = {"CDDM": ["1423687489", "2657899762", "2670295995"],
         "3BFF": ["1258717025", "2461724787", "4113307577"]}


def active(p):
    """Return the active mask of one network.

    Args:
        p (np.ndarray): participation of each unit, shape (N,).
    Returns:
        np.ndarray of bool, shape (N,): True where p >= 5% of the 95th percentile of p.
    """
    return p >= 0.05 * np.quantile(p, 0.95)


def one_network(task, seed):
    """Count the unit classes of one network and how many of each are active 5,500 iterations later.

    Args:
        task (str): "CDDM" or "3BFF".
        seed (str): network seed, as in the file names.
    Returns:
        dict: active, inactive, silent and above-threshold low-active counts on the reference weights,
            and the number of inactive, silent and above-threshold low-active units active at +5,500.
    """
    act = np.load(f"{ACT_DIR}/{task}_s{seed}__act.npz")
    c4 = np.load(glob.glob(f"{C4_DIR}/*s{seed}__floor.npz")[0])
    a0, a5 = active(act["p_0"]), active(act["p_5500"])
    inactive = ~a0
    silent = inactive & (c4["ref_n_on"] == 0)
    g = c4["strict_g_in"]
    above = inactive & ~silent & (g > 0.05 * np.quantile(g, 0.95))
    return {"active": int(a0.sum()), "inactive": int(inactive.sum()), "silent": int(silent.sum()),
            "above": int(above.sum()), "inactive_later": int((inactive & a5).sum()),
            "silent_later": int((silent & a5).sum()), "above_later": int((above & a5).sum())}


def main():
    """Print one row per network and the per-task range of the all-inactive fraction."""
    for task, seeds in SEEDS.items():
        fracs = []
        for seed in seeds:
            r = one_network(task, seed)
            frac = r["inactive_later"] / r["inactive"]
            fracs.append(frac)
            print(f"{task} s{seed}: active {r['active']}, silent {r['silent']} ({r['silent_later']} later), "
                  f"above-threshold low-active {r['above']} ({r['above_later']} later), "
                  f"inactive {r['inactive']} -> {r['inactive_later']} active at +5,500 ({100 * frac:.1f}%)")
        print(f"{task}: {100 * min(fracs):.1f} to {100 * max(fracs):.1f}% of all inactive units active at +5,500")


if __name__ == "__main__":
    main()
