"""Falsification checks for a trained TaskYang network: is a high accuracy real, or a shortcut?

Written when a 600-iteration network scored 0.95-1.00 accuracy on 19 of 20 rules (2026-09-15),
which is either a fast-learning family or a leak. Three checks that a shortcut would fail:
  1. CONTEXT: contextdm1 / contextdm2 accuracy on CONFLICT trials only (the two rings favour
     different options). A network that follows the strongest bump anywhere, ignoring the rule,
     is at ~50% here; a network that solves the task is at its overall accuracy.
  2. MEMORY: delaydm1 / delaygo / dms accuracy split by delay length (shortest vs longest third).
     Nothing is on the rings during the delay, so a network without memory is at chance on the
     long delays; a leak from the inputs would show no dependence on delay AND high accuracy.
  3. RULE ABLATION: accuracy with the 20 rule channels zeroed. Tasks whose answer depends on the
     rule (anti vs go, dnms vs dms, contextdm1 vs contextdm2) must collapse; if they do not, the
     network never used the rule and the family is being solved by something else.
Prints one table. Usage: python yang_checks.py <net_dir> [--n 256]
"""
import argparse

import numpy as np

from trainRNNbrain.experiments_and_analysis.multitask_readout import accuracy, build, run_noise_free


def acc_subset(task, out, Y, C, keep):
    """Accuracy restricted to the trials where `keep` is True."""
    idx = np.flatnonzero(keep)
    return accuracy(task, out[..., idx], Y[..., idx], [C[i] for i in idx]) if idx.size else float("nan")


def main():
    """Run the three checks on one network and print the table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("net_dir")
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    rnn, task, cfg = build(a.net_dir)
    print(f"network: N={cfg.model.N}, rules={len(task.rules)}, {a.n} trials per rule")

    print("\n1. CONTEXT tasks, accuracy on congruent vs CONFLICT trials (a strongest-bump shortcut is ~0.5 on conflict):")
    for rule in ("contextdm1", "contextdm2", "contextdelaydm1", "contextdelaydm2"):
        if rule not in task.rules:
            continue
        X, Y, C = task.task_batch(rule, a.n)
        _, out = run_noise_free(rnn, X)
        c1 = np.array([c["sub"]["coh1"] for c in C]); c2 = np.array([c["sub"]["coh2"] for c in C])
        conflict = np.sign(c1) != np.sign(c2)
        print(f"   {rule:16} congruent {acc_subset(task, out, Y, C, ~conflict):.2f}   conflict {acc_subset(task, out, Y, C, conflict):.2f}"
              f"   (n conflict = {conflict.sum()})")

    print("\n2. MEMORY tasks, accuracy by delay length (shortest third vs longest third; chance 0.10 for a responding trial):")
    for rule in ("delaygo", "delayanti", "delaydm1", "contextdelaydm1", "dms", "dnms", "dmc"):
        if rule not in task.rules:
            continue
        X, Y, C = task.task_batch(rule, a.n)
        _, out = run_noise_free(rnn, X)
        on1 = X[task.i_mod1].max(axis=0) > 0.05                 # (T, B)
        delay = np.array([c["sub"]["t_go"] - (np.flatnonzero(on1[:c["sub"]["t_go"], b])[-1] + 1) for b, c in enumerate(C)])
        lo, hi = np.quantile(delay, [1 / 3, 2 / 3])
        print(f"   {rule:16} delay <= {lo:3.0f} steps: {acc_subset(task, out, Y, C, delay <= lo):.2f}   "
              f"delay >= {hi:3.0f} steps: {acc_subset(task, out, Y, C, delay >= hi):.2f}")

    print("\n3. RULE ABLATION, accuracy with the rule channels zeroed (rule-dependent tasks must collapse):")
    print(f"   {'rule':16} {'intact':>7} {'no rule':>8}")
    for i, rule in enumerate(task.rules):
        X, Y, C = task.task_batch(i, a.n)
        _, out = run_noise_free(rnn, X)
        X0 = X.copy(); X0[task.i_rule:] = 0.0
        _, out0 = run_noise_free(rnn, X0)
        print(f"   {rule:16} {accuracy(task, out, Y, C):7.2f} {accuracy(task, out0, Y, C):8.2f}")


if __name__ == "__main__":
    main()
