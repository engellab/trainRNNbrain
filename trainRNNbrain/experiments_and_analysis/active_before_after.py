"""Active units before and after training in default 3BFF networks of N = 1,000 (Results 1, item 40).

For each network, two independent measurements of the number of active units:

  (A) the participation trace logged during training (Methods, Participation): noise-free probe on
      the training batch of that iteration, per-unit vector stored every 100 iterations. The probe
      at iteration 0 precedes the first update, so it is the untrained network. Read at iteration 0
      and at the last stored iteration.
  (B) a numpy re-simulation, the path of fig_paper_F1.example_network_pair: the untrained twin
      rebuilt from the chain's root config (gradient_participation.initial_params) and the final
      weights (LastParams), both run noise-free on one 24-trial batch from a task seeded with 11.

Both use common.participation and common.active_mask. The paper reports (A); (B) is the check.

Input layout (one folder per network, copied from Della by the r11-results1 package):
    <root>/<seed>/final/{*_config.yaml, *LastParams*.npz, *ParticipationTrace.pkl}
    <root>/<seed>/root_config.yaml       the config of the chain's first run (carries the init seed)

Usage:
    python active_before_after.py <root> [--out results.json] [--batch-seed 11]
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import hydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import participation, active_mask
from flipflop_fixedpoints import load_net
from gradient_participation import initial_params
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.training.training_utils import prepare_task_arguments
from trainRNNbrain.utils import filter_kwargs

N_TRIALS = 24      # fig_paper_F1.N_TRIALS
TRACE_SEED = 11    # fig_paper_F1.TRACE_SEED


def trace_counts(final_dir):
    """Active-unit counts from the logged participation trace (measurement A).

    Args:
        final_dir: folder of the chain's last run, holding *ParticipationTrace.pkl.
    Returns:
        dict with the first and last stored iterations and the active counts there (ints), and N.
    """
    t = pickle.load(open(glob.glob(os.path.join(final_dir, "*ParticipationTrace.pkl"))[0], "rb"))
    P = np.asarray(t["participation"])
    it = np.asarray(t["participation_iters"], int)
    return dict(it_first=int(it[0]), active_first=int(active_mask(P[0]).sum()),
                it_last=int(it[-1]), active_last=int(active_mask(P[-1]).sum()), N=int(P.shape[1]))


def simulate_counts(final_dir, root_cfg_path, batch_seed=TRACE_SEED):
    """Active-unit counts from a numpy re-simulation of the untrained twin and the final weights (B).

    Args:
        final_dir: folder of the chain's last run (its config and LastParams);
        root_cfg_path: config of the chain's first run, whose seed drew the initial weights;
        batch_seed: int, seed of the task generator that draws the 24 trials.
    Returns:
        dict with the active counts before and after training (ints).
    """
    cfg = OmegaConf.load(glob.glob(os.path.join(final_dir, "*_config.yaml"))[0])
    cfg.task.batch_size = N_TRIALS
    # The task draws its trials from its own generator (self.rng, seeded from cfg.task.seed, which
    # the training configs leave null), so np.random.seed alone does NOT fix the batch: that is how
    # fig_paper_F1.example_network_pair draws it, and its 24 trials differ on every call. Seed the
    # task itself so this evaluation batch is the same on every run.
    cfg.task.seed = int(batch_seed)
    np.random.seed(int(batch_seed))
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    inputs, _, _ = task.get_batch()

    def run(net):
        """Simulate `net` noise-free on `inputs`; return its (N, T, B) ReLU rates."""
        net.clear_history()
        net.y = net.y_init
        net.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)
        return np.maximum(np.array(net.get_history()), 0.0)

    rates = run(load_net(final_dir)[0])
    root = OmegaConf.load(root_cfg_path)
    p0 = dict(initial_params(root))
    p0["activation_name"] = "relu"
    p0.pop("activation_args", None)
    p0.pop("equation_type", None)
    rates0 = run(RNN_numpy(**filter_kwargs(RNN_numpy, p0),
                           equation_type=str(root.model.equation_type), seed=0))
    return dict(active_before=int(active_mask(participation(rates0)).sum()),
                active_after=int(active_mask(participation(rates)).sum()))


def main():
    """Measure every network under <root>, check (A) against (B), print and save the table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-seed", type=int, default=TRACE_SEED)
    args = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(args.root, "*", ""))):
        seed = os.path.basename(d.rstrip("/"))
        a = trace_counts(os.path.join(d, "final"))
        b = simulate_counts(os.path.join(d, "final"), os.path.join(d, "root_config.yaml"),
                            args.batch_seed)
        rows.append(dict(seed=seed, **a, **{f"sim_{k}": v for k, v in b.items()}))
        r = rows[-1]
        print(f"{seed:>11}  trace it {r['it_first']:>6}: {r['active_first']:>4}  it {r['it_last']:>6}: "
              f"{r['active_last']:>4}   sim before {r['sim_active_before']:>4}  after "
              f"{r['sim_active_after']:>4}", flush=True)
    for key in ("active_first", "active_last", "sim_active_before", "sim_active_after"):
        v = np.array([r[key] for r in rows], float)
        print(f"{key:>18}: mean {v.mean():.1f}  s.d. {v.std(ddof=1):.1f}  n {len(v)}  "
              f"range {v.min():.0f}-{v.max():.0f}")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
