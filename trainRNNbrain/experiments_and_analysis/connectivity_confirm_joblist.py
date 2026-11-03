#!/usr/bin/env python3
"""
The job list of the r12-weights confirmatory test (notebook 2026-11-02_2218_connectivity-confirm-rule.md):
for each network, the weights to read, the config of the run that wrote them, and the first run of its
warm-start chain. Built from the inventory written by connectivity_inventory.py, in the rule's order.

Read point: the saved weights nearest the floor crossing t* (a kept snapshot, or the final weights of
any chain link); nearest 75,000 iterations without a crossing; the final weights for the #172 networks.

Usage:  python connectivity_confirm_joblist.py <inventory.json> <joblist.json>
"""

import json
import os
import re
import sys

# the rule's order: main test (39 networks), then the #172 weight-decay-0 networks
MAIN_CELLS = ["NBitFlipFlop_ff_revive_c200000/", "NBitFlipFlop_std_dropfix_c200000/", "CDDM_paper_grid/",
              "CDDM_std_g0_drift_bias0/", "sigrec=0.05_iters=200000", "sigrec=0.05", "CDDM_std_g0_recinit/",
              "CDDM_std_g0_bias0_c75000/", "CDDM_screen3_c75000/", "DMTS_paper_grid_d5/", "arm=default_iters"]
WD0_CELLS = ["CDDM_std_g0_weightdecay_bias0/", "CDDM_std_g0_wd0_long/", "arm=wd0_iters"]
NO_FLOOR_TARGET = 75000.0


def seed_of(path):
    """The seed in a run folder's name."""
    return re.search(r"seed=(\d+)", os.path.basename(path)).group(1)


def in_cell(record, key):
    """Whether an inventory record belongs to the cell named by `key` (a path fragment; the bare
    'sigrec=0.05' matches only the cell that ends with it)."""
    if key == "sigrec=0.05":
        return record["cell"].endswith("sigrec=0.05")
    return key in record["cell"] + "/"


def read_point(record, final):
    """The weights to read for one network.

    Args:
        record: an inventory record; final: True for the #172 networks (final weights).
    Returns:
        (iteration, params folder or None for the run's own *LastParams*, run folder, config path).
    """
    t = record["read"]["t_star"]
    target = 1e12 if final else (t if t == t else NO_FLOOR_TARGET)
    best = min(((abs(l["end_iter"] - target), l["end_iter"], None, l["run"], l["config"]) for l in record["links"]),
               key=lambda c: c[0])
    snap = record["read"].get("snapshot")
    if snap and not final:
        si = int(os.path.basename(snap)[2:])
        if abs(si - target) < best[0]:
            last = record["links"][-1]
            best = (abs(si - target), si, snap, last["run"], last["config"])
    return best[1:]


def main():
    """Write the job list."""
    inv = json.load(open(sys.argv[1]))
    jobs = []
    for group, keys in (("main", MAIN_CELLS), ("wd0", WD0_CELLS)):
        for key in keys:
            for r in (x for x in inv if in_cell(x, key)):
                it, params, run, cfg = read_point(r, final=group == "wd0")
                jobs.append(dict(group=group, task=r["cell"].split("_")[0].replace("NBitFlipFlop", "3BFF"),
                                 cell=r["cell"], seed=seed_of(r["run"]), t_star=r["read"]["t_star"], read_iter=it,
                                 params=params, link_run=run, link_config=cfg, root_run=r["links"][0]["run"],
                                 root_config=r["links"][0]["config"]))
    for i, j in enumerate(jobs, 1):
        j["id"] = f"{i:02d}_{j['task']}_{j['seed']}"
    json.dump(jobs, open(sys.argv[2], "w"), indent=1)
    print(f"{len(jobs)} jobs written to {sys.argv[2]}")


if __name__ == "__main__":
    main()
