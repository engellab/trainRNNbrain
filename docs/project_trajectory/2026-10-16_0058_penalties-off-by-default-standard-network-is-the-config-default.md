## 2026-10-16 00:58 — Penalties off by default: the standard network is what the configs give

Pavel approved audit item 4a on 2026-10-15. Merged as 2c4e71e (branch commit 542b9f0, made by an
agent in a separate worktree).

**Change:**
- All four trainer configs now set `lambda_rws: 0` and `lambda_frm: 0`. They were 0.05 and 0.2.
- `configs/base.yaml` now defaults to `model: rnn_relu_standard`; it was `rnn_relu_Dale`. The trainer
  default stays `trainer`, because `slurm/b0_noise_topup.slurm` relies on it.
- 39 launchers drop their now-redundant `trainer.lambda_rws=0 trainer.lambda_frm=0`. Penalised arms
  still set their lambdas on the command line.

**Check, pass condition set before results.** Every (launcher, documented variant, array index)
was resolved with Hydra before and after: 2,112 cases from 53 launchers.
- All 2,112 resolved configs are identical, field by field.
- Negative control: the new launchers against the old configs differ in exactly 851 cases, the
  39 edited launchers.
- The resolved lambdas agree with each launcher's penalty tag in all 1,345 tagged cases.
- The three launchers that build from a saved run's own config (b4_transfer ×2, ff_topup200k) do not
  read `configs/` and are unaffected.
- All 23 test scripts pass after the merge.

**Caveats:**
- An edited launcher run from a checkout older than 542b9f0 trains with the penalties on
  (0.05 / 0.2). The pinned sweep worktrees on Della hold their own launcher copies, so this happens
  only if someone copies a new launcher into an old tree.
- `benchmark_large_N.py` now sizes an unpenalised run. For worst-case sizing of a penalised sweep,
  pass the lambdas on its command line.
- Not changed by this commit: `slurm/b0_noise_topup.slurm` continues its parents under `trainer`,
  but its parent launcher `b0_noise` trained them under `trainer_ptrack_plain`. The two differ at
  least in `task_safe_gradients` and `monitor`, so that sweep's continuations ran with different
  trainer settings from their parents. That affects how the recurrent-noise ladder is read: compare
  its 75k cells with care.
