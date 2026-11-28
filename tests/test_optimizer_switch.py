"""The optimizer switch (trainer.optimizer) and one float32 fact about AdamW at weight decay 1e-6.

Run:  python tests/test_optimizer_switch.py   (or under pytest)
"""
import torch
from omegaconf import OmegaConf

from trainRNNbrain.training.run_rl import build_optimizer

LR = 1e-3 * 0.1 ** (1 / 3)      # the base learning rate at N = 1,000


def trained(name, wd, momentum=0.0, steps=200):
    """Weights after `steps` updates on a fixed gradient sequence; the first 20 rows get no gradient.

    Args:
        name: str, trainer.optimizer; wd: float, weight decay; momentum: float (sgd); steps: int.
    Returns:
        (60, 60) float32 tensor, and the optimizer.
    """
    torch.manual_seed(0)
    layer = torch.nn.Linear(60, 60, bias=False)
    opt = build_optimizer(layer, OmegaConf.create(dict(optimizer=name, weight_decay=wd, momentum=momentum)), LR)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        grad = torch.randn(60, 60, generator=g) * 1e-3
        grad[:20] = 0
        layer.weight.grad = grad
        opt.step()
    return layer.weight.detach(), opt


def test_adamw_1e6_is_no_decay_in_float32():
    """1 - lr*1e-6 rounds to 1 in float32, so AdamW at 1e-6 is Adam without decay; at 1e-2 it is not."""
    ref, _ = trained("adam", 0.0)
    assert torch.equal(trained("adamw", 0.0)[0], ref)
    assert torch.equal(trained("adamw", 1e-6)[0], ref)
    assert not torch.equal(trained("adamw", 1e-2)[0], ref)
    assert not torch.equal(trained("adam", 1e-6)[0], ref)


def test_amsgrad_keeps_the_largest_second_moment():
    """amsgrad is Adam with the running maximum of v, and it differs from adam."""
    w, opt = trained("amsgrad", 1e-6)
    assert opt.param_groups[0]["amsgrad"]
    assert "max_exp_avg_sq" in next(iter(opt.state.values()))
    assert not torch.equal(w, trained("adam", 1e-6)[0])


def test_unknown_optimizer_is_refused():
    """A misspelt name fails instead of training with the default."""
    try:
        trained("adamax", 0.0)
    except ValueError:
        return
    raise AssertionError("an unknown optimizer name was accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
