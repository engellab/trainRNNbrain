"""Falsification test for the frm NaN diagnosis and its fix.

Mechanism under test:
  clip_grad_norm_ with an inf gradient gives total_norm=inf -> clip_coef=max_norm/inf=0.
  Each grad is multiplied by that 0: a clean grad becomes 0.0 (harmless), but the grad
  carrying the overflow becomes inf*0 = NaN. Adam then writes NaN into that parameter.
  On the NEXT forward pass that NaN parameter contaminates the whole network.
"""
import torch

def step(guard, grads, params):
    """Apply one clipped Adam step to `params` using `grads`. Returns nothing; mutates params."""
    opt = torch.optim.Adam(params, lr=1e-3)
    opt.zero_grad(set_to_none=True)
    for p, g in zip(params, grads):
        p.grad = g
    if guard and any(g is not None and not torch.isfinite(g).all() for g in grads):
        opt.zero_grad(set_to_none=True)          # skip: weights untouched
    else:
        torch.nn.utils.clip_grad_norm_(params, max_norm=50.0)
        opt.step()

def trial(guard):
    """Overflow on W2 only. Returns (W1 finite, W2 finite) after the step."""
    W1 = torch.nn.Parameter(torch.ones(3))
    W2 = torch.nn.Parameter(torch.ones(3))
    step(guard, [torch.ones(3), torch.full((3,), float("inf"))], [W1, W2])
    return bool(torch.isfinite(W1).all()), bool(torch.isfinite(W2).all())

off_1, off_2 = trial(guard=False)
on_1, on_2 = trial(guard=True)
print(f"WITHOUT guard: W1 finite={off_1}  W2 finite={off_2}")
print(f"WITH    guard: W1 finite={on_1}  W2 finite={on_2}")

assert off_1, "unexpected: clean param should survive (its grad is scaled to 0)"
assert not off_2, "DIAGNOSIS REFUTED: the overflowing param did NOT become NaN"
assert on_1 and on_2, "FIX FAILED: guard did not protect the weights"

# Propagation: a single NaN weight poisons the whole network on the next forward pass.
Wbad = torch.tensor([float("nan"), 1.0, 1.0])
h = torch.relu(torch.ones(3) @ torch.diag(Wbad)) @ torch.ones(3)
assert not torch.isfinite(h).all(), "NaN weight failed to contaminate the forward pass"
print("\nPASS: inf grad + clip -> NaN in that param; it spreads on the next forward; guard blocks it.")
