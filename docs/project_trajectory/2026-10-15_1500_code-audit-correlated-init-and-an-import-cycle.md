## 2026-10-15 15:00 — Code audit: correlated weight initialisation, an import cycle I introduced, and what confuses a newcomer

Pavel: "run a separate audit for the code itself - send sub agents … if they are confused thats a
sign things could be improved." Five read-only agents, each given a newcomer's job:
- package core: explain one training run end to end;
- read-out pipeline: read a run at its floor and count active units and R² there;
- figure builders: rebuild Figure 1c and Figure 2;
- launchers and configs: prepare a Della flip-flop size series to 200k with snapshots;
- onboarding: answer 12 questions starting from `CLAUDE.md` alone.

### Every network's initial weights share one random stream (confirmed)

`RNN_torch.sparse` (`trainRNNbrain/rnns/RNN_torch.py:32-38`) builds a fresh CPU generator on each
call, seeded with the network generator's `initial_seed()`. Each of W_rec, W_inp and W_out therefore
starts from the same sequence of normal draws.

**Check, written independently of the auditor.** Pass/fail was set before running: a correlation
above 0.99 confirms the bug, a correlation near 0 refutes it. Standard model config, seed 12345:
- Flattened W_inp against flattened W_out: correlation 1.000000 at N = 100 and at N = 1000.
- Flattened W_inp against the first 3N entries of W_rec (its first three rows): 0.92 at N = 100,
  0.995 at N = 1000.
- Control: two different seeds give unrelated W_inp (correlation −0.03).

The coupling is element by element in draw order, not unit by unit. Unit j's own input-row norm and
output-column norm correlate only 0.09-0.17. Every network in the project was initialised this way.
**Untested:** whether the coupling changes silencing. A fix changes every future initialisation, so
it marks a version boundary; it is an open decision on the status board.

### An import cycle I introduced, and a test that would have caught it

Hoisting the function-level imports (308f63e) closed the cycle
`fig_paper_F1 → gradient_participation → fig_knobs_at_floor → fig_paper_F1`. Seven scripts died
with `ImportError` when run as scripts: fig_paper_F1, fig_paper_F2_fails, gradient_participation,
gradient_sensitivity, c4_gradient_mechanism, c4_snapshot_drift and c4b_convergence. My check had
imported each one as a module, and that does not trigger the cycle.

Fixed in d41c8f2: `snapshot_at` moved to `heldout_r2`. The new `tests/test_scripts_start.py` runs
every argparse script with `--help`; it fails all 7 at 308f63e and passes all 30 after the fix.

### Other defects the auditors verified

- **Truncated chains on the Mac.** `common.warm_start_chain` silently stops when a parent path is a
  Della path that does not exist on the Mac. Verified on `CDDM_std_g0_bias0_c75000`: chain length
  1, read as "never leaves its starting level". On Della the paths resolve.
- **Resume breaks the read-out.** A run resumed from `resume/` loses its loss history and its trace,
  because `write_checkpoint` saves neither. `noisy_loss_series` then returns nothing for the whole
  chain, so the run has no floor read-out. The guide and the 1342 entry recommend this resume path.
- **Double-counted offset.** `fig_supp_task_demand.joined` adds the iteration offset twice to an
  already-absolute trace: probes 0-124,900 for a 75k chain (the A5 panel).
- **Two meanings of the folder score.** The leading number in a run folder's name is computed two
  ways. With `light_outputs` it is a noisy R² pooled over the batch; otherwise it is a numpy mean of
  per-trial R² with NaN trials dropped. The same network read 0.22 one way and 0.31 the other.
- **Competing read-outs.** Five read-out routes on one CDDM run put the read-out anywhere from
  9,600 to 32,224 iterations and count 416 to 548 active units. Canonical: `common.noisy_floor`.
  The active-unit rule is re-typed inline in about 20 places instead of calling
  `common.active_count`.

### What confused the newcomers most

- **No config describes the standard network.** All four trainer configs switch the rate and
  sparsity penalties on (`lambda_frm` 0.2, `lambda_rws` 0.05), and `base.yaml` defaults to a Dale
  network. The standard control exists only because 39 launchers override those values.
- **No launcher for a fresh 200k run.** Four launchers train the standard flip-flop control, with
  defaults of 40k, 40k, 40k and 75k iterations. Every 200k network came from topping up a 75k run.
- **Figure numbering disagrees.** Paper `main` includes `fig_paper_F2.pdf` (the remedies). The
  analysis docs call `fig_paper_F2_fails.py` Figure 2. Another section points at a third figure.
- **Comments contradict code.** Examples: the dropout sampler's docstring describes a different
  algorithm; checkpoint comments give the wrong spacing and contents; the read-out docstrings
  describe the rule being replaced today.
- **Missing docstrings and dead code.** Missing docstrings include `run_training`, `train_step`,
  `forward` and all of `training_utils`. Unused code includes 14 task classes, three RNN variants
  and the deleted-penalty helpers.
