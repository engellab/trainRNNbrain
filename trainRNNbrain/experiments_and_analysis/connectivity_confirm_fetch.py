#!/usr/bin/env python3
"""Della side of the r12-weights confirmatory test: copy each network's read weights and configs, and
extract the two trace quantities the checks need (no connectivity statistic is computed here).

Per job of joblist.json, into <out>/<id>/:
  params file (the read weights, *LastParams*.npz or .json), the link's *_config.yaml (architecture and
  batch), root.yaml (the first run's config, for the initial weights), root_rownorm0.npy (w_rec_row at
  the first run's iteration-0 probe, check T), root_partic0.npy (participation there, check T'), trace.json (active count at the probe of the link's
  participation trace nearest the read iteration, check M1), source.txt.

Usage (Della, CPU job): PYTHONPATH=<analysis checkout> python connectivity_confirm_fetch.py joblist.json <out dir>
"""

import glob
import json
import os
import pickle
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.environ["PYTHONPATH"].split(":")[0], "trainRNNbrain", "experiments_and_analysis"))
from common import active_mask    # noqa: E402


def trace(run):
    """The participation trace of a run folder (dict)."""
    with open(glob.glob(os.path.join(run, "*ParticipationTrace.pkl"))[0], "rb") as fh:
        return pickle.load(fh)


def main():
    """Copy and extract for every job."""
    jobs = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    for j in jobs:
        d = os.path.join(out, j["id"])
        os.makedirs(d, exist_ok=True)
        src = j["params"] if j["params"] else j["link_run"]
        pf = sorted(glob.glob(os.path.join(src, "*LastParams*")))[0]
        shutil.copy(pf, d)
        shutil.copy(j["link_config"], d)
        shutil.copy(j["root_config"], os.path.join(d, "root.yaml"))
        rt = trace(j["root_run"])
        has_rows = "w_rec_row" in rt and len(rt["w_rec_row"]) > 0      # older traces do not store it
        if has_rows:
            np.save(os.path.join(d, "root_rownorm0.npy"), np.asarray(rt["w_rec_row"][0], dtype=np.float64))
        np.save(os.path.join(d, "root_partic0.npy"), np.asarray(rt["participation"][0], dtype=np.float64))  # check T'
        lt = trace(j["link_run"])
        its = np.asarray(lt["participation_iters"], float)
        k = int(np.argmin(np.abs(its - j["read_iter"])))
        info = dict(root_has_rownorm=bool(has_rows), root_first_probe_iter=float(rt["participation_iters"][0]), probe_iter=float(its[k]),
                    trace_active=int(active_mask(np.asarray(lt["participation"][k], float)).sum()))
        json.dump(info, open(os.path.join(d, "trace.json"), "w"))
        with open(os.path.join(d, "source.txt"), "w") as fh:
            fh.write(json.dumps(dict(j, params_file=pf), indent=1))
        print(j["id"], os.path.basename(pf), info, flush=True)


if __name__ == "__main__":
    main()
