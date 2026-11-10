#!/usr/bin/env python3
"""Expected chance directions of the setting rule (dead-relu-paper #218, committee #217 item R3).

The direction rule (paper, Methods: Standard training settings): a setting value increases (decreases) the number
of active units on a task when every network of the value that learned the task has more (fewer) active units
than every control network. If the n value networks and the m control networks are exchangeable, all n lie above
all m with probability n! m! / (n + m)!, and the value gets an increase or a decrease with twice that.

The numbers of networks that learned the task per value and task are typed in from the paper's Supplementary
Table 2 (floats/ST_verdicts.tex at dead-relu-paper main ad52b64); None means the value was not run on that task,
0 that no network learned it (no direction possible). Control networks: 3 on 3BFF and CDDM, 5 on Maze (the five
default networks that learned by update 2,000). The CDDM activity-noise and sparsity sweeps have five networks
per value; they are scored with three controls, which gives the larger probability. The CDDM tanh networks have
no measurement at 75,000 iterations and count as 0.

Usage: python trainRNNbrain/experiments_and_analysis/chance_directions.py
"""
from math import factorial

CONTROLS = {"3BFF": 3, "CDDM": 3, "Maze": 5}
TASKS = ("3BFF", "CDDM", "Maze")
# value: networks that learned the task on (3BFF, CDDM, Maze)
LEARNED = {
    "weight decay 0": (3, 3, 3), "weight decay 1e-5": (3, 3, 3), "weight decay 1e-4": (3, 3, 1),
    "input norm 0.5": (3, 3, 3), "input norm 2": (3, 3, 2), "input norm 5": (3, 3, 0), "input norm 20": (3, 3, 0),
    "cap 0.1": (3, None, None), "cap 0.3": (3, 3, 3), "cap 0.5": (3, None, None), "cap 2": (3, None, None),
    "noise 0": (3, 5, 3), "noise 0.01": (3, 5, 3), "noise 0.1": (3, 5, 0),
    "radius 0.5": (3, 3, 3), "radius 1.5": (None, None, 2), "radius 2.0": (3, 3, None),
    "sparsity 0.05": (3, 5, 1),
    "metabolic 0.01": (3, 3, 2), "metabolic 0.1": (3, 3, 1), "metabolic 1": (3, 3, 0), "metabolic 10": (3, 0, 0),
    "magnitude 0.05": (3, 3, 3), "magnitude 0.2": (3, 3, 2),
    "AdamW": (3, 3, 2), "SGD 0.03": (3, 3, 0), "SGD 0.1": (3, 3, 0), "SGD 0.3": (3, 3, 0), "SGD 1": (2, 3, 0),
    "leaky ReLU": (3, 3, 3), "softplus": (3, 3, 2), "shifted sigmoid": (3, 3, 0), "tanh": (3, 0, 0),
}


def p_above(n, m):
    """Probability that all n value networks exceed all m control networks under exchangeability.

    Args:
        n: int, networks of the value that learned the task (0 gives 0: no direction is possible).
        m: int, control networks.
    Returns:
        float in [0, 1].
    """
    return 0.0 if n == 0 else factorial(n) * factorial(m) / factorial(n + m)


def main():
    """Print the expected chance increases or decreases per task and the expected chance conjunctions."""
    for t, task in enumerate(TASKS):
        run = [v[t] for v in LEARNED.values() if v[t] is not None]
        expected = sum(2 * p_above(n, CONTROLS[task]) for n in run)
        print(f"{task}: {len(run)} values run, expected chance increases or decreases {expected:.2f}")
    common = {k: v for k, v in LEARNED.items() if None not in v}
    conj = {k: 2 * p_above(v[0], 3) * p_above(v[1], 3) * p_above(v[2], 5) for k, v in common.items()}
    total = sum(conj.values())
    print(f"{len(common)} values run on all three tasks: expected chance same-direction conjunctions "
          f"{total:.4f} (1 in {1 / total:.0f})")
    print(f"weight decay 1e-4: same direction on all three tasks by chance, 1 in {1 / conj['weight decay 1e-4']:.0f}")


if __name__ == "__main__":
    main()
