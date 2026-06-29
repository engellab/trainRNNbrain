#!/usr/bin/env python3
"""Sanity checks for the weight_boundary feature (sticky vs reflective Dale enforcement).

Run:  python3 trainRNNbrain/rnns/check_weight_boundary.py   (asserts; prints OK on success)

Verifies:
  1. reflective effective W_rec obeys Dale (exc cols >= 0, inh cols <= 0, zero diagonal) and
     W_inp >= 0;
  2. get_params exports the EFFECTIVE weights (reflective: |param|*sign*mask; sticky: raw);
  3. a reflective net's saved (effective) weights reproduce its dynamics both when reloaded into a
     sticky RNN_torch and via RNN_numpy reconstruction -> the analysis path is boundary-agnostic;
  4. legacy fallback: a params dict lacking weight_boundary reconstructs as sticky / eps=1e-12.
"""
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.utils import filter_kwargs, jsonify, unjsonify

torch.manual_seed(0)
np.random.seed(0)
N, T, B = 30, 40, 4
u = torch.randn(6, T, B)


def build(boundary, seed=0):
    """A small gamma=0 ReLU-Dale RNN_torch with the given weight_boundary."""
    return RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, equation_type="h",
                     gamma=0.0, weight_boundary=boundary, seed=seed)


def firing_rates_numpy(params, inp):
    """Reconstruct via RNN_numpy (the analysis path) and return noise-free firing rates (N,T,B)."""
    net_cfg = filter_kwargs(RNN_numpy, unjsonify(jsonify(params)))
    if isinstance(net_cfg, DictConfig):
        net_cfg = OmegaConf.to_container(net_cfg, resolve=True)
    rn = RNN_numpy(**net_cfg, seed=0)
    rn.clear_history()
    rn.y = rn.y_init
    rn.run(input_timeseries=inp.detach().cpu().numpy(), sigma_rec=0, sigma_inp=0)
    return rn.get_firing_rate_history()


def main():
    # 1) reflective Dale compliance
    r = build("reflective")
    Wr, Wi, Wo = r._constrained_weights()
    Wr_np = Wr.detach().cpu().numpy()
    dm = r.dale_mask.detach().cpu().numpy()
    exc, inh = dm > 0, dm < 0
    assert (Wr_np[:, exc] >= -1e-9).all(), "reflective: exc columns must be >= 0"
    assert (Wr_np[:, inh] <= 1e-9).all(), "reflective: inh columns must be <= 0"
    assert np.allclose(np.diag(Wr_np), 0.0), "reflective: diagonal must be zero"
    assert (Wi.detach().cpu().numpy() >= -1e-9).all(), "reflective: W_inp must be >= 0"

    # 2) get_params exports effective weights
    p = r.get_params()
    expected = (np.abs(r.W_rec.detach().cpu().numpy())
                * dm[None, :] * r.recurrent_mask.detach().cpu().numpy())
    assert np.allclose(p["W_rec"], expected), "reflective get_params must export |param|*sign*mask"
    assert p["weight_boundary"] == "reflective"
    s = build("sticky")
    assert np.allclose(s.get_params()["W_rec"], s.W_rec.detach().cpu().numpy()), \
        "sticky get_params must export the raw param unchanged"

    # 3) effective weights reproduce the dynamics in a reloaded sticky net + via RNN_numpy
    uu = u.to(r.device)  # match the model device (cpu or cuda)
    fr_r = r.get_firing_rates(r.forward(uu, w_noise=False)[0]).detach().cpu().numpy()
    s2 = build("sticky", seed=123)
    s2.set_params(p)
    fr_s = s2.get_firing_rates(s2.forward(uu, w_noise=False)[0]).detach().cpu().numpy()
    assert np.allclose(fr_r, fr_s, atol=1e-5), "effective weights must reproduce dynamics in sticky mode"
    cc = np.corrcoef(fr_r.ravel(), firing_rates_numpy(p, uu).ravel())[0, 1]
    assert cc > 0.999, f"RNN_numpy reconstruction must match torch dynamics (corr={cc:.5f})"

    # 4) legacy fallback
    p_legacy = {k: v for k, v in p.items() if k not in ("weight_boundary", "weight_boundary_eps")}
    s3 = build("reflective", seed=7)
    s3.set_params(p_legacy)
    assert s3.weight_boundary == "sticky", "legacy params must fall back to sticky"
    assert abs(float(s3.weight_boundary_eps) - 1e-12) < 1e-20, "legacy eps must default to 1e-12"

    print("OK: weight_boundary checks passed "
          "(Dale compliance, effective export, reconstruction, legacy fallback)")


if __name__ == "__main__":
    main()
