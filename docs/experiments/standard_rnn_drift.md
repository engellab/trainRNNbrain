# Experiment: when does systematic drift become jitter?

*Copied into the sweep's data folder as `EXPERIMENT.md` by the launcher.*

## The problem

Strict convergence is unreachable. The relative change of the participation vector decays as a
**power law** in iteration (exponent −0.29 at N=100 to −0.57 at N=1000), so reaching a 1% criterion
extrapolates to **0.5–5.6 million iterations** — weeks to months of GPU time per network. Meanwhile
the training loss is flat from early on, so it says nothing about whether the network has settled.

What *is* reachable is the transition from **systematic drift** (the network still marching in a
consistent direction, silencing more units) to **jitter** (weights random-walking around a solution
because noise is injected every step). That transition is the defensible definition of "trained
enough", and it is what lets the size sweep match training across N **by drift rather than by
iteration count** — necessary because `lr ∝ N^(−1/3)` means a fixed iteration budget buys less
progress at larger N.

## Why one distance is not enough

With noise injected every step the weights never stop moving, so `‖W(t) − W(t−L)‖` plateaus at a
**noise floor** rather than reaching zero. A plateau alone cannot distinguish "settled and jittering"
from "still drifting". Two measures separate them:

| | diffusion (jitter) | systematic drift |
|---|---|---|
| distance vs lag `L` | grows as **√L** | grows as **L** |
| cosine between consecutive displacements | ≈ **0** | **> 0** |

## Design — 12 jobs

`4 sizes {100, 500, 1000, 2000} × 3 seeds`, h equation, **no penalties**, standard RNNs,
**200000 iterations** (6.7× the original horizon). Runtime 2.8 / 4.4 / 8.0 / 17.6 h.

## What is recorded, and how it is stored

**Nothing about the weights reaches disk.** Reference snapshots live in CPU RAM — one per lag plus
one previous displacement, per matrix, ~80 MB at N=2000 — and every quantity is reduced to a scalar
during training. Memory is O(number of lags), not O(training length).

Per probe (every 10 iterations), into `metrics` of the trace pickle:

| key | meaning |
|---|---|
| `drift_{W_inp,W_rec,W_out}_lag{100,1000,10000}` | `‖W(t) − W(t−L)‖_F / ‖W(t)‖_F` |
| `cos_{W_inp,W_rec,W_out}` | cosine between consecutive displacements |
| `dp_lag{100,1000,10000}` | `‖p(t) − p(t−L)‖ / ‖p(t)‖` |
| `silent_1em6` | silent-unit count (participation < 1e-6) |

Lagged entries are NaN on probes where that lag is not due; each lag is measured at exactly its own
separation. `bias` is excluded — it is normally not trained in this project.

The per-unit participation matrix is the only bulky item and gets a coarser cadence
(`store_participation_every = 100` → 2000 vectors, 16 MB at N=2000). The scalar series are ~1 MB
regardless of training length.

## Read-out

For each N, the iteration at which (a) the cosine falls to ≈0 and (b) the lag-scaling exponent
approaches 0.5 — i.e. where systematic drift ends. Then: how that horizon scales with N, and whether
the silent count has plateaued by then. That yields either a stated convergence criterion (option 1)
or a drift-matched training protocol for the size sweep (option 3).
