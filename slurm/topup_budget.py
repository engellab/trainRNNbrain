#!/usr/bin/env python3
"""Print "<parent total> <standard> <top-up>" in iterations, for one finished run.

A top-up should train the experiment line's standard minus whatever the parent actually trained, so
that no launcher carries an iteration count of its own to drift out of sync with the config. The
parent's total is its own max_iter plus any iter_offset it inherited, so a chain of continuations
accumulates correctly.

Args (argv):
    1: a finished run's folder (holding its *_config.yaml).
    2: the experiment config yaml whose trainer.max_iter is the standard.
Prints:
    three space-separated integers - parent total, standard, and standard minus parent total. The
    third may be <= 0, which means the parent is already at or past the standard and the caller
    should skip it rather than shorten it.
"""
import sys
from pathlib import Path
from omegaconf import OmegaConf

src, expcfg = sys.argv[1], sys.argv[2]
cfgs = sorted(Path(src).glob("*_config.yaml"))
if not cfgs:
    sys.exit(f"topup_budget: no *_config.yaml in {src}")
pc = OmegaConf.load(cfgs[0])
off = pc.trainer.get("iter_offset", 0)
parent_total = (int(off) if isinstance(off, (int, float)) else 0) + int(pc.trainer.max_iter)
standard = int(OmegaConf.load(expcfg).trainer.max_iter)
print(parent_total, standard, standard - parent_total)
