"""Committee round 3 (rigor): independent recompute of Results 4 / Fig. 4a-c on the Fig. 1d weights.

My own forward model (Methods M01) and my own task batches (Methods M02), PyTorch autograd only for
the backward pass (no trainRNNbrain import). Classes per Methods M04/M07: active p >= 0.05 q95(p);
silent = inactive with x < 0 on every step after step 0 of every trial; low-active = other inactive;
reached = incoming-weight gradient norm (rows of W_rec and W_inp) >= 0.05 q95(g) of that network.
Later activity: participation on the same noise-free batch with the weights 500 (snap d0) and 5,500
(snap d5000) iterations later (mapping as stated in notebook entry 2026-11-01_1021).

Pass rules, fixed before running (paper values: R4 text and notebook 2026-11-01_1021):
  CDDM (fixed 450-condition batch, so the batch matches): silent = 310, 288, 253 exactly; reached
    low-active within +-3 of 40, 44, 60; share of reached low-active active at +5,500 within +-5
    percentage points of 23, 39, 32%; silent units active at +500 and at +5,500 at most 1 per network.
  3BFF (my own random batch, 1,024 trials, so counts can differ): silent = 0; reached low-active
    within +-15% of 136, 155, 161 and at least 100; share active at +5,500 at most 5% ("at most 5%").
  Any failure is traced before it is reported.
"""
import glob
import os
import sys

import numpy as np
import torch

DATA = os.path.expanduser("~/Documents/GitHub/trainRNNbrain/data")
COH = np.array([-1, -.5, -.25, -.125, -.0625, -.03125, -.015625, 0, .015625, .03125, .0625, .125, .25, .5, 1.0])


def cddm_batch(on=99):
    """CDDM noise-free batch (M02): u (450,300,6), y (450,300,2), mask (300,) bool.

    Args: on, first step of the stimulus (decision from on + 100).
    """
    u = np.zeros((450, 300, 6)); y = np.zeros((450, 300, 2)); k = 0
    for ci in range(2):
        for c1 in COH:
            for c2 in COH:
                mot, col = (c1, c2) if ci == 0 else (c2, c1)
                u[k, :, ci] = 1.0
                u[k, on:, 2] = (1 + mot) / 2; u[k, on:, 3] = (1 - mot) / 2
                u[k, on:, 4] = (1 + col) / 2; u[k, on:, 5] = (1 - col) / 2
                rel = mot if ci == 0 else col
                if rel == 0:
                    y[k, on + 100:, :] = 0.5
                else:
                    y[k, on + 100:, 0 if rel > 0 else 1] = 1
                k += 1
    m = np.zeros(300, bool); m[:100] = True; m[200:] = True
    return u, y, m


def ff_batch(B=1024, seed=7, T=300, flip=10, mean_gap=75.0):
    """3-bit flip-flop batch (M02): pulses of +-1 for 10 steps, gaps 10 + Exp(mean 75) steps.

    Returns: u (B,T,3), y (B,T,3), mask (T,) all True.
    """
    rng = np.random.default_rng(seed)
    u = np.zeros((B, T, 3)); y = np.zeros((B, T, 3))
    for b in range(B):
        for c in range(3):
            t = 0; last = 0.0; ev = []
            while True:
                t = t + flip + int(rng.exponential(mean_gap))
                if t >= T:
                    break
                ev.append((t, 1.0 if rng.random() < 0.5 else -1.0))
            for (t0, s) in ev:
                u[b, t0:t0 + flip, c] = s
            for i, (t0, s) in enumerate(ev):
                t1 = ev[i + 1][0] if i + 1 < len(ev) else T
                y[b, t0:t1, c] = s
    return u, y, np.ones(T, bool)


def run(Wr, Wi, u, tau=10.0):
    """Forward Euler from x = 0 (M01), no noise. Torch tensors. Returns x (B,T,N) after each step."""
    B, T, _ = u.shape
    x = torch.zeros(B, Wr.shape[0], dtype=Wr.dtype)
    xs = []
    for t in range(T):
        x = x + (-x + torch.relu(x) @ Wr.T + u[:, t] @ Wi.T) / tau
        xs.append(x)
    return torch.stack(xs, 1)


def participation(x):
    """p_i = std(r) + q0.9(|r|) over every step and trial (M04). x: (B,T,N) numpy. Returns (N,)."""
    r = np.maximum(x, 0).reshape(-1, x.shape[-1])
    return r.std(0) + np.quantile(np.abs(r), 0.9, axis=0)


def load(path):
    """Weights from an npz: (W_rec, W_inp, W_out) float64 numpy."""
    w = np.load(path, allow_pickle=True)
    return w["W_rec"].astype(float), w["W_inp"].astype(float), w["W_out"].astype(float)


def analyse(task, seed, base, u, y, m, chunk):
    """Classes, gradient reach and later activity for one network. Prints and returns a dict."""
    Wr, Wi, Wo = load(glob.glob(os.path.join(base, "ckpt*.npz"))[0])
    Wr_t = torch.tensor(Wr, requires_grad=True); Wi_t = torch.tensor(Wi, requires_grad=True)
    Wo_t = torch.tensor(Wo)
    mt = torch.tensor(m)
    xs = []
    nscore = len(u) * m.sum() * Wo.shape[0]
    for s in range(0, len(u), chunk):
        ut = torch.tensor(u[s:s + chunk]); yt = torch.tensor(y[s:s + chunk])
        x = run(Wr_t, Wi_t, ut)
        z = torch.relu(x) @ Wo_t.T
        loss = ((z - yt)[:, mt] ** 2).sum() / nscore
        loss.backward()
        xs.append(x.detach().numpy().astype(np.float32))
    x = np.concatenate(xs)
    g = np.sqrt((Wr_t.grad.numpy() ** 2).sum(1) + (Wi_t.grad.numpy() ** 2).sum(1))
    p = participation(x)
    act = p >= 0.05 * np.quantile(p, 0.95)
    silent = ~act & np.all(x < 0, axis=(0, 1))  # x holds the states after each step, so step 0 (x = 0) is not in it
    low = ~act & ~silent
    reached = g >= 0.05 * np.quantile(g, 0.95)
    out = dict(active=int(act.sum()), silent=int(silent.sum()), low=int(low.sum()),
               reached_low=int((reached & low).sum()), silent_gmax=float(g[silent].max()) if silent.any() else None)
    for tag, f in (("+500", "d0"), ("+5500", "d5000")):
        W2 = load(os.path.join(DATA, "fig4_noreturn/snap", f"{seed}_{f}.npz"))
        with torch.no_grad():
            x2 = run(torch.tensor(W2[0]), torch.tensor(W2[1]), torch.tensor(u)).numpy()
        p2 = participation(x2)
        a2 = p2 >= 0.05 * np.quantile(p2, 0.95)
        out[tag] = dict(reached_low_active_share=round(100 * (a2 & reached & low).sum() / max((reached & low).sum(), 1), 1),
                        silent_active=int((a2 & silent).sum()), active=int(a2.sum()))
    print(task, seed, out, flush=True)
    return out


if __name__ == "__main__":
    torch.set_num_threads(10)
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("CDDM", "both"):
        u, y, m = cddm_batch()
        for d in sorted(glob.glob(os.path.join(DATA, "pop_stats_nets", "CDDM_*"))):
            analyse("CDDM", d.split("__s")[1].split("__")[0], d, u, y, m, chunk=150)
    if which in ("3BFF", "both"):
        u, y, m = ff_batch(seed=int(sys.argv[2]) if len(sys.argv) > 2 else 7)
        for d in sorted(glob.glob(os.path.join(DATA, "pop_stats_nets", "NBitFlipFlop_*"))):
            analyse("3BFF", d.split("__s")[1].split("__")[0], d, u, y, m, chunk=128)
