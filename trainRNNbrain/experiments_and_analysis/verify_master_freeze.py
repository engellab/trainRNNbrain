#!/usr/bin/env python3
"""
Verify the frozen-master-inhibitor experiment: the master unit's weights must not move during training.

The master-inhibitor run (CDDM_*_g0_masterinhib) showed frm rescues the clamped units *indirectly* — it
suppresses the always-active, over-cap master (a valid gradient, since the master fires), which releases
the clamp. To make the clamp genuinely gradient-proof (Pavel's true "no gradient -> no rescue" test) we
FREEZE the master: `RNN_torch(freeze_master=True)` registers gradient hooks that zero the gradient on the
master's input row (W_inp[m,:]) and its recurrent input row + output column (W_rec[m,:], W_rec[:,m]), so
its context drive and its deep inhibition onto the target set stay fixed at their init values.

This script checks that the hooks actually fire on the Trainer's gradient path (it uses
`torch.autograd.grad`, not `loss.backward()`):
  - frozen:   gradient at every master entry is EXACTLY 0, and non-master entries are (mostly) non-zero;
  - unfrozen: the master entries DO receive non-zero gradient (control — confirms the test can fail).

Run from this directory:  python3 verify_master_freeze.py
"""
import glob
import os
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.utils import filter_kwargs, import_any
from trainRNNbrain.training.training_utils import prepare_task_arguments
import calibrate_inhibitory_boost as C

OmegaConf.register_new_resolver("eval", eval, replace=True)


def build_rnn(cfgp, seed, frac, freeze):
    """Instantiate an RNN_torch with the master inhibitor (and optional freeze) from a baseline config."""
    cfg = OmegaConf.load(cfgp)
    task_cfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    if "_target_" in task_cfg:
        del task_cfg._target_
    rnn_cfg = OmegaConf.create(cfg.model)
    rnn_cls = import_any(getattr(rnn_cfg, "_target_", None)) or RNN_torch
    args = filter_kwargs(rnn_cls, OmegaConf.merge(rnn_cfg, task_cfg))
    args.seed = int(seed)
    args.master_inhib_frac = frac
    args.freeze_master = freeze
    return hydra.utils.instantiate(args), cfg


def master_grad(cfgp, seed, frac, freeze):
    """Return (max |grad| on master entries, median |grad| off master) for one forward/grad pass."""
    rnn, cfg = build_rnn(cfgp, seed, frac, freeze)
    m = rnn.master_idx
    task_cfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    task = hydra.utils.instantiate(task_cfg)
    inp, tgt, _ = task.get_batch()
    inp_t = torch.as_tensor(inp, dtype=torch.float32, device=rnn.device)
    tgt_t = torch.as_tensor(tgt, dtype=torch.float32, device=rnn.device)
    _states, out = rnn(inp_t, w_noise=False)
    loss = ((out - tgt_t) ** 2).mean()
    # exactly the Trainer's gradient path: autograd.grad w.r.t. the trainable params
    g_inp, g_rec = torch.autograd.grad(loss, [rnn.W_inp, rnn.W_rec], allow_unused=True)
    gi, gr = g_inp.abs(), g_rec.abs()
    master_entries = torch.cat([gi[m, :].flatten(), gr[m, :].flatten(), gr[:, m].flatten()])
    off = gr.clone(); off[m, :] = float("nan"); off[:, m] = float("nan")
    off_med = float(torch.nanmedian(off).item())
    return float(master_entries.max().item()), off_med


def main():
    cfgp = sorted(glob.glob(os.path.join(C.DATA, C.BASELINE, C.COND["h"], "*", "*_config.yaml")))[0]
    seed, frac = 4242, 0.5
    print("Master-weight gradient on the Trainer's autograd.grad path (frac=0.5):\n")
    for freeze in (False, True):
        gmax, off = master_grad(cfgp, seed, frac, freeze)
        tag = "FROZEN " if freeze else "unfrozen"
        print(f"  {tag}: max|grad| on master entries = {gmax:.3e}   (median |grad| off-master = {off:.3e})")
    print("\nExpect: unfrozen -> master grad > 0 (frm/task can move it);  FROZEN -> master grad == 0.")


if __name__ == "__main__":
    main()
