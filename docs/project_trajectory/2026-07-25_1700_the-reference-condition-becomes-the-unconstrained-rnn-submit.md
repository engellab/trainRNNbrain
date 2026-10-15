## 2026-07-25 (17:00) — the reference condition becomes the *unconstrained* RNN (submitted)

### Decision

Pavel's call, and it changes the paper's frame rather than just adding a control: **Dale's law, the
excitatory-only readout and the I/O sign clamps make these networks a biologically-motivated special
case, and a result stated only for that case is easy to dismiss as niche.** Most RNN work trains
unconstrained networks with biases. So the reference architecture becomes the unconstrained one, and
the Dale sweep becomes the constrained comparison rather than the main result.

Concretely: array `11610299` (Dale + trainable bias) was **cancelled** at ~16:50 EDT — 4 of 40 tasks
had run 9–13 min and written no data, since outputs are saved only at the end — and replaced by the
same bias manipulation without the constraints.

### The three-point constraint ladder now in flight

| Array | Folder | Architecture | Code (worktree) |
|---|---|---|---|
| `11609846` | `CDDM_ptrack_g0` | Dale + non-negative I/O, no bias — links to every earlier result | `1492041` (`~/trainRNNbrain`) |
| `11610813` | `CDDM_ptrack_g0_nodale` | unconstrained, no bias | `a5bde03` (`~/trainRNNbrain_nodale`) |
| `11610886` | `CDDM_ptrack_g0_nodale_trainablebias` | **unconstrained + trainable bias** — the field-standard architecture | `733c021` (`~/trainRNNbrain_nodalebias`) |

Each is 40 jobs (2 eq × 4 penalties × 5 seeds), N=1000, γ=0, 30000 iterations, participation tracked
every 10 iterations. If the low-activity population appears at all three rungs, no architectural
constraint explains it and the claim is about trained RNNs in general; if it collapses at the
unconstrained rungs, the phenomenon belongs to constrained networks and the paper narrows honestly.

Config `configs/model/rnn_relu_noDale_trainablebias.yaml`; launcher
[`slurm/SilentReLU_ptrack_nodalebias_gamma0_N1000_della.slurm`](../slurm/SilentReLU_ptrack_nodalebias_gamma0_N1000_della.slurm);
descriptor [`docs/experiments/participation_trace_nodale_bias.md`](experiments/participation_trace_nodale_bias.md).
The pre-registered readout is unchanged from the cancelled sweep: a constant bias fakes participation,
so **rescue is scored on `std(fr)`, not peak rate**.

**One worktree per sweep is now the standing practice.** Three concurrent arrays run three different
commits with no interference: `git worktree add`, plus a `PYTHONPATH` guard in each launcher that
**aborts the job** if `trainRNNbrain` does not resolve inside its own worktree. That guard is not
theoretical — the editable install resolves the package to `~/trainRNNbrain`, so without it a job
launched from a worktree silently trains with the other sweep's code.

### Caveat on the design

Cancelling the Dale+bias cell leaves the 2×2 of {Dale, unconstrained} × {no bias, bias} incomplete: we
will be able to say what a bias does *without* Dale, but not whether it interacts *with* Dale. That is
the right trade given the framing decision — the unconstrained+bias cell is the one a general reader
cares about — but if a reviewer asks specifically whether biases rescue Dale networks, that cell has to
be run.

