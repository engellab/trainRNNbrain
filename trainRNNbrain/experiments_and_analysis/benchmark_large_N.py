#!/usr/bin/env python3
"""
Benchmark peak GPU memory and per-iteration time for large-N RNN training, so that a large-N sweep
can be sized without OOMs or timeouts.

Runs a handful of REAL training steps (the production Trainer.train_step on the production task and
model) for each N, and reports:
  - peak GPU memory actually allocated / reserved
  - seconds per iteration, extrapolated to a full run
  - the size the saved parameter files would reach

Usage:  python benchmark_large_N.py [N1 N2 ...]   (default: 1000 2000 5000 10000)
        ITERS=8 python benchmark_large_N.py       (iterations timed per size, after warm-up)
"""

import os
import sys
import time
import gc
import numpy as np
import torch
import hydra
from hydra import compose, initialize_config_dir

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask
from trainRNNbrain.utils import filter_kwargs
from omegaconf import OmegaConf

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "configs")


def benchmark_one(N, iters=8, warmup=3):
    """Time and memory-profile real training steps at a given network size.

    Args:
        N: number of units.
        iters: timed iterations (after warm-up).
        warmup: untimed iterations first, so allocator caching and cuDNN autotune settle.
    Returns:
        dict with peak memory (GB), seconds per iteration, and parameter-count info,
        or {"oom": True} if the size does not fit.
    """
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="base", overrides=[
            "model=rnn_relu_standard", "trainer=trainer_ptrack_plain",
            f"model.N={N}", "trainer.max_iter=1"])
    task_cfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    task = hydra.utils.instantiate(task_cfg)
    del task_cfg._target_

    rnn_args = filter_kwargs(RNN_torch, OmegaConf.merge(OmegaConf.create(cfg.model), task_cfg))
    rnn_args.seed = 1234
    rnn = hydra.utils.instantiate(rnn_args)
    lr = cfg.trainer.lr * (100.0 / N) ** cfg.trainer.lr_scale_exp
    opt = torch.optim.Adam(rnn.parameters(), lr=lr, weight_decay=cfg.trainer.weight_decay)
    trainer = hydra.utils.instantiate(filter_kwargs(Trainer, OmegaConf.create(cfg.trainer)),
                                      RNN=rnn, Task=task, optimizer=opt, _convert_="none")

    inp, tgt, _ = task.get_batch()
    dev = rnn.device
    inp = torch.from_numpy(inp.astype("float32")).to(dev)
    tgt = torch.from_numpy(tgt.astype("float32")).to(dev)
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)

    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(warmup):
            trainer.train_step(input=inp, target_output=tgt, mask=mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            trainer.train_step(input=inp, target_output=tgt, mask=mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    except torch.cuda.OutOfMemoryError as e:
        return {"N": N, "oom": True, "msg": str(e).split("\n")[0][:120]}

    n_par = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
    out = {
        "N": N, "oom": False, "s_per_iter": dt,
        "peak_alloc_GB": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan"),
        "peak_resv_GB": torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else float("nan"),
        "n_params_M": n_par / 1e6,
        # each float in the saved json costs ~20 bytes of text; two files are written per net
        "json_GB": 2 * n_par * 20 / 1e9,
    }
    del trainer, rnn, opt, inp, tgt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    """Benchmark every requested N and print a sizing table."""
    sizes = [int(x) for x in sys.argv[1:]] or [1000, 2000, 5000, 10000]
    iters = int(os.environ.get("ITERS", 8))
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}, {p.total_memory/1e9:.1f} GB total\n")
    print(f"{'N':>6s} {'params':>9s} {'peak alloc':>11s} {'peak resv':>10s} {'s/iter':>8s} "
          f"{'30k iters':>10s} {'5k iters':>9s} {'json out':>9s}")
    for N in sizes:
        r = benchmark_one(N, iters=iters)
        if r["oom"]:
            print(f"{N:6d} {'—':>9s} {'OOM':>11s}  {r['msg']}")
            continue
        print(f"{N:6d} {r['n_params_M']:8.1f}M {r['peak_alloc_GB']:10.2f}G {r['peak_resv_GB']:9.2f}G "
              f"{r['s_per_iter']:8.3f} {r['s_per_iter']*30000/3600:9.1f}h {r['s_per_iter']*5000/3600:8.1f}h "
              f"{r['json_GB']:8.2f}G")


if __name__ == "__main__":
    main()
