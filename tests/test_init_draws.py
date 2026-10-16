"""A network's initial weight matrices are independent draws (`init_draws: independent`), and the
old shared draws are still available to rebuild networks trained before 2026-10-15.

Until 2026-10-15 `RNN_torch.sparse` reseeded a fresh generator for every matrix, so W_out was a
rescaled copy of W_inp in draw order (correlation 1.0) and W_rec's first rows repeated the same draws.
Run: python tests/test_init_draws.py
"""
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from trainRNNbrain.rnns.RNN_torch import RNN_torch   # noqa: E402
from trainRNNbrain.utils import filter_kwargs         # noqa: E402

KW = dict(N=1000, activation_args={"name": "relu", "slope": 1.0}, equation_type="h", dale=False,
          io_nonnegativity=False, self_connections=True, gamma=0, bias_init="zeros",
          n_inputs=3, n_outputs=3, seed=12345)


def corr(a, b):
    """Pearson correlation of two tensors' flattened values.

    Args:
        a, b: tensors with the same number of elements.
    Returns:
        float.
    """
    return float(np.corrcoef(a.detach().cpu().numpy().ravel(), b.detach().cpu().numpy().ravel())[0, 1])


def test_independent_draws_are_uncorrelated():
    """With independent draws, W_inp is unrelated to W_out and to W_rec's first rows (|r| < 0.1)."""
    rnn = RNN_torch(**KW, init_draws="independent")
    n = rnn.W_inp.numel()
    r_out, r_rec = corr(rnn.W_inp, rnn.W_out), corr(rnn.W_inp, rnn.W_rec.reshape(-1)[:n])
    assert abs(r_out) < 0.1 and abs(r_rec) < 0.1, (r_out, r_rec)
    print(f"ok  independent: corr(W_inp, W_out) = {r_out:.3f}, corr(W_inp, W_rec rows) = {r_rec:.3f}")


def test_shared_draws_reproduce_the_old_coupling():
    """The legacy mode still gives W_out as a rescaled copy of W_inp, so old networks rebuild exactly."""
    rnn = RNN_torch(**KW, init_draws="shared")
    r = corr(rnn.W_inp, rnn.W_out)
    assert r > 0.999, r
    print(f"ok  shared (legacy): corr(W_inp, W_out) = {r:.6f}")


def test_draws_are_reproducible_per_seed():
    """The same seed gives identical weights; another seed gives different ones."""
    a, b = RNN_torch(**KW, init_draws="independent"), RNN_torch(**KW, init_draws="independent")
    c = RNN_torch(**dict(KW, seed=54321), init_draws="independent")
    assert all(torch.equal(getattr(a, k), getattr(b, k)) for k in ("W_rec", "W_inp", "W_out"))
    assert not torch.equal(a.W_inp, c.W_inp)
    print("ok  same seed -> same weights; another seed -> different weights")


def test_model_configs_select_independent_draws():
    """Every model config sets init_draws: independent, and it reaches the network's constructor."""
    folder = os.path.join(REPO, "configs", "model")
    for name in sorted(os.listdir(folder)):
        cfg = OmegaConf.to_container(OmegaConf.load(os.path.join(folder, name)), resolve=True)
        assert cfg.get("init_draws") == "independent", name
    cfg = OmegaConf.to_container(OmegaConf.load(os.path.join(folder, "rnn_relu_standard.yaml")), resolve=True)
    # built the way run_experiment builds it: filter_kwargs keeps _target_, hydra instantiates
    rnn = hydra.utils.instantiate(filter_kwargs(RNN_torch, dict(cfg, N=50, n_inputs=3, n_outputs=3, seed=1)))
    assert rnn.init_draws == "independent", rnn.init_draws
    print("ok  every model config selects independent draws, and the constructor receives it")


if __name__ == "__main__":
    test_independent_draws_are_uncorrelated()
    test_shared_draws_reproduce_the_old_coupling()
    test_draws_are_reproducible_per_seed()
    test_model_configs_select_independent_draws()
    print("all init-draw contracts hold")
