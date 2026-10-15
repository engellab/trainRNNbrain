## 2026-07-25 (16:45) — removing Dale's law and I/O positivity: is the silent mode an artifact of the constraints? (submitted)

### Why this was missing

Every network in this document is **Dale-constrained**, and "Dale" here bundles three separate
restrictions: (i) `W_rec` is sign-split (excitatory units excite, inhibitory units inhibit, ratio
`exc2inhR = 4`) with the sign re-projected after every optimizer step; (ii) the readout is restricted to
the excitatory subpopulation (the inhibitory columns of `W_out` are structurally masked to zero); and
(iii) `W_inp` and `W_out` are clamped non-negative. The 2026-06-29 boundary control (`sticky` vs
`reflective`) tested only two *implementations* of (i)'s projection — it never removed the constraint.
This matters twice over: most of the RNN literature uses unconstrained networks, so as it stands the
result describes a special case; and a sign constraint could plausibly *cause* silence, since a weight
cannot change role and a unit needing mixed-sign output is stuck.

### The finding that motivated it — silence is an *excitatory* phenomenon

Free diagnostic on existing trained nets (`CDDM_4a031e_g0`, `h`, N=1000, γ=0, 3 nets/condition),
splitting the silent population by unit type:

| Condition | overall silent | E units (800) | I units (200) |
|---|---|---|---|
| `none` | 43–45% | **53–55%** | **3.5–5.0%** |
| `rws` | 52–54% | 58–60% | 27–29% |

Under `none`, inhibitory units are almost **never** silent while more than half the excitatory units
are. This **falsifies the natural Dale hypothesis**: the excitatory-only readout gives inhibitory units
the weaker gradient path, so they should have been the ones to go quiet — the opposite happens. The
likely explanation is load-bearing redundancy — 200 inhibitory units supply the whole network's
inhibition at 4× weight, so each is individually indispensable, while 800 excitatory units are mutually
redundant and half can be dropped. This is a new descriptive result and belongs in the paper regardless
of the sweep; it is also a Dale-specific structure, which is precisely why the unconstrained control is
needed.

### Implementation — two independent switches (commit `a5bde03`)

- **`model.dale`** (default `true`). `false` → new `get_connectivity_unconstrained`: signed zero-mean
  weights, no E/I split, `dale_mask = None`, **every** unit reads out; identical 1/√N scale, zero
  diagonal and spectral-radius rescaling to the Dale version, so the two are comparable.
  `Trainer.enforce_dale_` is skipped.
- **`model.io_nonnegativity`** (default `true`). `false` → the `W_inp ≥ 0`, `W_out ≥ 0` clamps are
  skipped. Independent of `dale`, so "Dale recurrence with a signed readout" is one override away.
- `_constrained_weights` (the `reflective` path) and `get_params`/`set_params` honour both flags;
  networks saved before this commit have no flags and default to both-on.
- `run_experiment.py` grouped units by the sign of their outgoing recurrent weights for the sorted /
  clustered figures — meaningless without Dale, so it now falls back to a single group when
  `model.dale` is false.

**Verification.** (1) The Dale path is **bit-identical** to a pre-change snapshot at the same seed
(`W_rec`, `W_inp`, `W_out`, `dale_mask`) — no existing result is disturbed. (2) The unconstrained init
has `dale_mask = None`, 50% negative weights, **no zeroed readout columns**, spectral radius exactly
1.200, zero diagonal. (3) Over 30 training iterations the unconstrained net shows **648 `W_rec` sign
crossings** (Dale net: 0), 167 negative `W_inp` and 54 negative `W_out` entries — the projections really
are skipped. (4) The switches compose. (5) `get_params`→`set_params` round-trips the flags, and legacy
params default to both-on. (Two apparent anomalies were chased to ground and are pre-existing, not new:
the Dale net's "negative" `W_out` entries are `enforce_dale_` writing `eps·dale_mask = −1e-12` on
inhibitory columns, and the "entries at eps" count is the zeroed diagonal.)

### The sweep (submitted — Della array `11610813`)

40 jobs = 2 equations {h, s} × 4 penalties {none, rws, frm, both} × 5 seeds, N=1000, γ=0, 30000
iterations, participation tracked every 10 iterations. Identical to `CDDM_ptrack_g0` except
`model=rnn_relu_noDale` (`dale: false`, `io_nonnegativity: false`).

Config `configs/model/rnn_relu_noDale.yaml`; launcher
[`slurm/SilentReLU_ptrack_nodale_gamma0_N1000_della.slurm`](../slurm/SilentReLU_ptrack_nodale_gamma0_N1000_della.slurm);
descriptor [`docs/experiments/participation_trace_nodale.md`](experiments/participation_trace_nodale.md).
Output → `.../CDDM_ptrack_g0_nodale/`, kept separate from the two sibling sweeps
(`CDDM_ptrack_g0`, `CDDM_ptrack_g0_trainablebias`).

**Isolation from in-flight jobs (the fix for this morning's repro caveat).** This sweep runs from a
**separate git worktree**, `$HOME/trainRNNbrain_nodale`, so `$HOME/trainRNNbrain` — the checkout that
the pending tasks of arrays `11609846` and `11610299` will import when they start — is never touched.
Because the editable install resolves `trainRNNbrain` to `$HOME/trainRNNbrain`, the launcher prepends
the worktree to `PYTHONPATH` **and refuses to run** if `trainRNNbrain.__file__` does not resolve inside
it: training with the wrong code is exactly the failure this guard exists to prevent.

### Prediction

Silence **persists** at a broadly similar level (~40–55% under `none`/`rws`): the apparent driver is
that CDDM needs far fewer units than the network has (unpenalized nets solve it with an effective
~60–150 units at any N), which sign constraints do not touch. The E/I asymmetry disappears by
construction.

**Falsifier:** a silent fraction under `none` below ~15% would mean Dale was doing the work, and every
claim in this document narrows to Dale-constrained networks.

