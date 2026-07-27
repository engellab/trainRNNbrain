# Experiment: the STANDARD RNN reference sweep

*Copied into the sweep's data folder as `EXPERIMENT.md` by the launcher.*

## What this is

The reference condition for the project from 2026-07-27 onwards. Every previous sweep carried at
least one project-specific or biologically-motivated constraint; this one carries none, so results
can be stated for **RNNs in general** rather than for a niche variant.

| Property | Value | Note |
|---|---|---|
| Dynamics | `dx/dt = −x + W_rec·ReLU(x) + W_inp·u + b`, α = dt/τ = 0.1 | textbook continuous-time RNN, forward Euler |
| Dale's law | **off** | signed `W_rec`, no E/I split, every unit reads out |
| I/O positivity | **off** | `W_inp`, `W_out` unconstrained in sign |
| Self-connections | **on** | `W_rec` diagonal free and trainable |
| Cubic term | **γ = 0** | no saturation term in the dynamics |
| Bias | trainable, `[-1, 1]`, zero init | |
| Gradient handling | `task_safe_gradients=false` | penalties combine with the task loss the ordinary way |
| Monitoring | `monitor=false` | no per-penalty breakdown; total loss still saved as `*_TrainLosses.json` |
| Deliberate non-vanilla choice | training noise σ_rec = σ_inp = 0.05, σ_out = 0.03 | standard in neuroscience RNNs; state in Methods |

Configs: `model=rnn_relu_standard`, `trainer=trainer_ptrack_plain`, `+experiment=silent_units_std`.

## Questions

1. **Does the silent population survive in a fully standard RNN?** Every constraint that could have
   been blamed is now gone. Prior result in the constrained architectures: 42–58% silent under
   `none`/`rws`, 0% under `frm`.
2. **Do self-connections rescue?** This is the cheapest possible escape from silence — a unit that
   can excite itself needs no help from the network. If self-recurrence mattered, it shows up here.
3. **Does removing the task-safe gradient projection change the penalty results?** The projection
   guaranteed penalties could never hurt the task; without it, `frm` and the task loss compete
   normally. If the `frm` rescue survives, it was never an artifact of the projection.

## Prediction

Silence persists at ~40–55% under `none`/`rws`; `frm` still drives it to 0%; R² ≈ 0.85 throughout.
Self-connections change little, because a positive diagonal helps a unit stay active only if it is
already active — a unit sitting below ReLU threshold gets nothing from self-excitation.

**Falsifier:** a silent fraction below ~15% under `none` would mean one of the constraints we just
removed (most plausibly the zeroed diagonal) was producing the effect all along.

## Grid — 40 jobs

`2 equation types {h, s} × 2 λ_rws {0, 0.05} × 2 λ_frm {0, 0.2} × 5 seeds`, N = 1000, 30000
iterations, participation logged every 10 iterations.

Launcher `slurm/SilentReLU_std_gamma0_N1000_della.slurm`, run from its own git worktree
(`$HOME/trainRNNbrain_std`) with a PYTHONPATH guard so concurrent sweeps keep their own code.
