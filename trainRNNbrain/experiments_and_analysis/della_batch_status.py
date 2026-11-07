#!/usr/bin/env python3
"""Build the per-task status table of the uniform-programme Della arrays (issue #6) from sacct and the job logs.

WHAT IT MAKES. One row per array task of the eleven arrays submitted on 5 Nov (entries
2026-11-05_0903, _2256, _2259): job_task, the Slurm state, the seed and experiment tag the job log
printed at its start, and the trainer's "gradient spikes" line (updates skipped) from the end of the
log. read_uniform_cells_2.py reads this table to tie each run folder on the Mac to its task, to
list the tasks still pending, and to apply the exclusion rule on skipped updates. The index decode
of every array (which task index is which cell) is DECODE below, copied from the three entries
(the launcher writes the metabolic values 1 and 10 as `LmbdMet=1.0` and `LmbdMet=10.0`, which entry
_2259's table abbreviates); a log whose tag disagrees with the decode is printed and the log's tag
is used.

INPUTS, fetched from Della by hand (the login node runs both in a second):
  ssh della 'sacct -j 16955456,16955457,16957418,16957419,16957556,16957557,16957558,16957559,16957560,16957561,16957562 -X --format=JobID%24,State%14 -n -p' > sacct.txt
  ssh della 'cd ~/trainRNNbrain_logs && for f in WdB0.16955456_*.out TgtLadder.16955457_*.out B5opt.16957418_*.out StdPen.16957419_*.out NoiseB0.16957556_*.out StdRecInit.16957557_*.out StdRecInit.16957558_*.out MetB0.16957559_*.out FFpenD.16957560_*.out ActB0.16957561_*.out Screen3.16957562_*.out; do t=${f%.out}; seed=$(grep -o -m1 "'"'"'seed'"'"': [0-9]*" $f | grep -o "[0-9]*$"); tag=$(grep -o -m1 "'"'"'experiment_tag'"'"': '"'"'[^'"'"']*'"'"'" $f | sed "s/.*: '"'"'//; s/'"'"'$//"); sk=$(grep -m1 "gradient spikes" $f | sed "s/gradient spikes: //"); echo -e "$t\\t$seed\\t$tag\\t${sk:-none}"; done' > logs.tsv
  (the one-line version used on 6 Nov is in entry 2026-11-06_0900.)

Usage:  python trainRNNbrain/experiments_and_analysis/della_batch_status.py --sacct sacct.txt --logs logs.tsv --out data/della_batch2_status.tsv
"""

import argparse
import csv


def rng(lo, hi, tag):
    """{index: tag} for the array indices lo..hi inclusive."""
    return {i: tag for i in range(lo, hi + 1)}


FF = "NBitFlipFlop"
# array id -> {task index: experiment tag (the cell path without the task prefix)}
DECODE = {
    "16955456": rng(4, 6, "std_g0_weightdecay_bias0/EqType=h_N=1000_wd=1e-5_iters=75000")
                | rng(7, 9, "std_g0_weightdecay_bias0/EqType=h_N=1000_wd=1e-4_iters=75000"),
    "16955457": rng(9, 11, "target_ladder/EqType=h_N=1000_tgt=10.0"),
    "16957418": rng(7, 9, "b5_optimizer/EqType=h_N=1000_opt=adamw_wd=1e-6_lr=1e-3")
                | rng(10, 12, "b5_optimizer/EqType=h_N=1000_opt=sgd_wd=1e-6_lr=0.03")
                | rng(13, 15, "b5_optimizer/EqType=h_N=1000_opt=sgd_wd=1e-6_lr=0.1")
                | rng(16, 18, "b5_optimizer/EqType=h_N=1000_opt=sgd_wd=1e-6_lr=0.3")
                | rng(19, 21, "b5_optimizer/EqType=h_N=1000_opt=sgd_wd=1e-6_lr=1"),
    "16957419": rng(4, 6, "std_g0_penalties_bias0/EqType=h_N=1000_pen=frm0.1_iters=75000")
                | rng(7, 9, "std_g0_penalties_bias0/EqType=h_N=1000_pen=frm0.1_rws0.05_iters=75000"),
    "16957556": rng(1, 3, "std_g0_noise_bias0/EqType=h_N=1000_sigrec=0_iters=75000")
                | rng(6, 8, "std_g0_noise_bias0/EqType=h_N=1000_sigrec=0.01_iters=75000")
                | rng(16, 18, "std_g0_noise_bias0/EqType=h_N=1000_sigrec=0.1_iters=75000"),
    "16957557": rng(1, 3, "std_g0_recinit/EqType=h_N=1000_sr=0.5_iters=75000")
                | rng(7, 9, "std_g0_recinit/EqType=h_N=1000_sr=2.0_iters=75000")
                | rng(10, 12, "std_g0_recinit/EqType=h_N=1000_rwm=0.05_iters=75000")
                | rng(13, 15, "std_g0_recinit/EqType=h_N=1000_rwm=0.2_iters=75000"),
    "16957558": {1: "std_g0_recinit/EqType=h_N=1000_sr=0.5_iters=75000", 2: "std_g0_recinit/EqType=h_N=1000_sr=1.2_iters=75000",
                 3: "std_g0_recinit/EqType=h_N=1000_sr=2.0_iters=75000", 4: "std_g0_recinit/EqType=h_N=1000_rwm=0.05_iters=75000",
                 5: "std_g0_recinit/EqType=h_N=1000_rwm=0.2_iters=75000"},
    "16957559": rng(7, 9, "std_g0_metabolic_bias0/EqType=h_N=1000_LmbdMet=0.01_iters=75000")
                | rng(16, 18, "std_g0_metabolic_bias0/EqType=h_N=1000_LmbdMet=0.1_iters=75000")
                | rng(25, 27, "std_g0_metabolic_bias0/EqType=h_N=1000_LmbdMet=1.0_iters=75000")
                | rng(34, 36, "std_g0_metabolic_bias0/EqType=h_N=1000_LmbdMet=10.0_iters=75000"),
    "16957560": rng(166, 168, "std_penlong/EqType=h_k=3_N=1000_pen=rws_iters=75000"),
    "16957561": rng(1, 3, "std_g0_activations_bias0/EqType=h_N=1000_act=softplus25_iters=75000")
                | rng(4, 6, "std_g0_activations_bias0/EqType=h_N=1000_act=leakyrelu_iters=75000")
                | rng(7, 9, "std_g0_activations_bias0/EqType=h_N=1000_act=sigmoid_iters=75000")
                | rng(10, 12, "std_g0_activations_bias0/EqType=h_N=1000_act=tanh_iters=75000"),
    "16957562": rng(4, 6, "screen3/EqType=h_N=1000_pen=none_arm=cap_cap=0.3_iters=75000"),
}
TASK = {"16955456": FF, "16955457": "CDDM", "16957418": "CDDM", "16957419": "CDDM", "16957556": FF, "16957557": FF,
        "16957558": "CDDM", "16957559": FF, "16957560": FF, "16957561": FF, "16957562": FF}


def read_sacct(path):
    """Slurm states per (array id, index) from `sacct -X -n -p` output, folded index ranges expanded.

    Args:
        path: a file of lines `<job>_<index or [a-b,c-d]>|<state>|`.
    Returns:
        dict (job str, index int) -> state str.
    """
    state = {}
    with open(path) as fh:
        for line in fh:
            if "|" not in line:
                continue
            jid, st = line.split("|")[:2]
            job, idx = jid.strip().split("_")
            for part in idx.strip("[]").split(","):
                a, *b = part.split("-")
                for i in range(int(a), int(b[0]) + 1) if b else [int(a)]:
                    state[(job, i)] = st.strip()
    return state


def read_logs(path):
    """(seed, tag, skipped line) per (array id, index) from the logs TSV (`<name>.<job>_<index>\\t<seed>\\t<tag>\\t<skipped>`)."""
    logs = {}
    with open(path) as fh:
        for line in fh:
            t, seed, tag, sk = line.rstrip("\n").split("\t")
            job, idx = t.split(".")[1].split("_")
            logs[(job, int(idx))] = (seed, tag, sk)
    return logs


def main():
    """Join the decode, the sacct states and the logs into the TSV; print tag mismatches and completed tasks without a log."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sacct", required=True, help="output of sacct -X -n -p over the arrays")
    ap.add_argument("--logs", required=True, help="the logs TSV (see the module docstring)")
    ap.add_argument("--out", required=True, help="TSV to write")
    a = ap.parse_args()
    state, logs = read_sacct(a.sacct), read_logs(a.logs)
    rows = []
    for job, dec in DECODE.items():
        for i, tag in sorted(dec.items()):
            st = state.get((job, i), "PENDING")
            seed, ltag, sk = logs.get((job, i), ("", "", ""))
            if ltag and ltag != tag:
                print(f"TAG MISMATCH {job}_{i}: log says {ltag}, decode says {tag}")
            if st == "COMPLETED" and not seed:
                print(f"COMPLETED WITHOUT LOG {job}_{i}")
            rows.append(dict(job_task=f"{job}_{i}", state=st, seed=seed, tag=f"{TASK[job]}_{ltag or tag}",
                             skipped_updates=sk if st == "COMPLETED" else ""))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["job_task", "state", "seed", "tag", "skipped_updates"], delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    done = sum(r["state"] == "COMPLETED" for r in rows)
    print(f"wrote {a.out}: {len(rows)} tasks, {done} completed, {len(rows) - done} not")


if __name__ == "__main__":
    main()
