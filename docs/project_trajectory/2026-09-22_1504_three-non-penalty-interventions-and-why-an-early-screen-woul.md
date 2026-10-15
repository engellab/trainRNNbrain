## 2026-09-22, 15:04 — Three non-penalty interventions, and why an early screen would have picked the worst one

### The finding that shaped the design: `rws` is a late-collapsing impostor

Re-reading the archived `CDDM_ptrack_*` traces (120 nets, 30k iterations, full per-unit
participation every 10 iterations) to ask whether a cheap early read-out can substitute for a full
run. Pre-registered: Spearman ρ ≥ 0.9 against the final count, plus preserved arm ordering.

**The ordering test failed at every probe iteration, and the reason is not noise.** Split by
equation type — the earlier pass pooled `h` and `s`, which is wrong, they behave differently:

| arm | active @30k, `h` | active @30k, `s` |
|---|---|---|
| `none` | 417.1 ± 19.0 | 411.5 ± 17.1 |
| `rws` | **357.7 ± 13.2** | 425.6 ± 57.2 |
| `frm` | 1000.0 ± 0.0 | 1000.0 ± 0.0 |
| `both` | 1000.0 ± 0.0 | 1000.0 ± 0.0 |

For `equation_type: h` — the config `rnn_relu_standard` uses — **`rws` rises to 918 of 1000 active
units at iteration 3620, the best-looking arm on offer, and then collapses to 357.7 by 30k, BELOW
the 417.1 of no penalty at all.** Welch t = 9.96, p = 3.5e-10, n = 15 each. Sparsity does not merely
fail to help; it costs 59.4 active units relative to doing nothing.

For `s` the same arm ends at 425.6, ABOVE `none`. The impostor behaviour is specific to `h`.

**Consequence for any automated search: silencing has long latency, and an intervention that
DELAYS it is indistinguishable from one that PREVENTS it until late.** A screen stopping at 5k
would have selected `rws`. Hyperband-style aggressive early pruning would be worse than no
automation here. Coarse binary screening ("does it recruit at all?") is fine — 91.7% of
recruiting/non-recruiting net pairs order correctly at iteration 500, 100% by 10k — but ranking two
similar interventions early is actively anti-informative. A slope feature was tried as a rescue and
failed its pre-registered test (slope ordering at 3k: `frm > both > rws > none`; final:
`frm > both > none > rws`).

### The premise behind all three interventions, now measured

A silent ReLU unit is **frozen, not merely quiet**. With r_i = 0 at every timestep,
dL/dW_rec[i,j] ~ relu'(h_i)·r_j = 0 and dL/dW_rec[j,i] ~ r_i = 0 — every weight into AND out of the
unit has exactly zero gradient. Asserted directly in
`tests/test_prune_and_reinit.py::test_dead_units_have_exactly_zero_gradient`.

**No term added to the loss can revive a dead unit**, because a penalty acts through the same
vanished gradient. It can only act before the unit dies. That is why all three interventions below
act outside the loss, on the weights directly.

### The three interventions

- **`cap`** — `inp_weight_cap`, hard clamp on |W_inp| post-step. Already existed. Screen uses 0.3,
  the measured `frm+rws` q90 at N=1000 (free fit 41.7·N^−0.709 = 0.31), ~10× the init scale.
- **`prune`** — `prune_and_reinit_` (new, `2892b74`). Redraws the INCOMING weights of a unit silent
  for `patience` consecutive checks, from the initialisation distribution. Outgoing weights are left
  alone: they took no gradient while the unit was dead. **Adam's moments for redrawn entries are
  zeroed** — without that, stale momentum from before the unit died pushes the weights straight back
  into the dead configuration. Runs before the Dale/mask projections so they clean up fresh rows.
- **`scale`** — `synaptic_scaling_` (new, `58d76f5`). Multiplicative, toward the median
  participation of the live pool.

### Why `scale` splits on synapse sign, with numbers

Naive multiplicative scaling has the **wrong sign of effect** on the case that matters. A unit
silenced because its net drive is negative gets a MORE negative drive when its whole incoming row is
scaled up. Measured on a net-inhibited unit:

| rule | h before | h after |
|---|---|---|
| sign-split (W>0 ×α, W<0 ÷α) | −30.25 | **−19.90** |
| naive uniform (×α) | −30.25 | −45.37 |

Biology does the sign split: synaptic scaling acts on excitatory synapses while inhibitory synapses
scale the opposite way under the same deprivation. Relative weights are preserved WITHIN the
excitatory set and within the inhibitory set separately — the property that keeps scaling from
destroying learned selectivity, and the property a weight-magnitude penalty does not have.

**Documented limit, tested:** `scale` CANNOT rescue a unit whose excitatory input is gone — shrinking
inhibition drives h toward 0 from below but never across (h = −0.0000 after 30 scalings, still
negative). `prune` can. **They address disjoint failure modes, so `prune+scale` should beat either
alone** — a prediction with a real way to lose.

### What is running

`Screen3` on Della, job `14277821_[1-12]`, worktree `trainRNNbrain_screen3` pinned at `58d76f5`.
CDDM, N=1000, no penalties, bias-free, 30,000 iterations (~80 min/job, measured).
8 arms × 3 seeds; **only arms 0–3 submitted** (control, `cap`, `prune`, `scale`).
Arms 4–7 (pairs + triple) are `--array=13-24` and stay unsubmitted until the singles report —
greedy forward selection, so an intervention that adds nothing is visibly dropped rather than
absorbed into a stack of knobs that collectively "works". The eventual claim is about the MINIMAL
set, so the ablation table has to be built, not reconstructed.

**Pre-registered** (in the launcher, before any job existed): primary = active units at 30k
(scale-free 0.05·q95); co-primary = `loss_clean_train` (dropout-off, noise-free — NOT the
training-pass loss, which is structurally blind to this failure class). Advance at
control + 2·sd (2 not 3 because at screening stage a false negative costs a real mechanism);
confirmatory bar stays at 3·sd. Named failure modes: **treadmill** for `prune`
(`reinit_events` ≫ `reinit_units_ever` with no count gain = redrawn units re-dying, a FAILURE not
partial success — both counters are logged to the trace), and **homogenisation trade** for `scale`
(two-sided scaling pulls busy units down; count up with loss degraded = buying units by flattening
the representation). Control is run, not read off disk: archived CDDM h/none cells disagree across
architecture (438.8 ± 6.8 / 408.8 ± 11.4 / 403.8 ± 13.6) and all predate the bias-fixing change.

A null on `cap` would be informative on its own — it would settle the causal direction the
input-weight analysis left open, i.e. that spread-out input drive is a CONSEQUENCE of having many
active units rather than a cause.

### ⚠️ Two Della traps found

1. **The main checkout `~/trainRNNbrain` fails the provenance guard.** `data` there is a SYMLINK to
   scratch, and `.gitignore` has `/data/` with a trailing slash, which matches directories but not
   symlinks — so `git status --porcelain` is never empty and any job run from it dies in seconds.
   Use a pinned worktree (the existing convention). The gitignore line is worth fixing; untouched.
2. **Two stray files in that checkout** from May 2026: `trainRNNbrain/rnns/Trainer.py` (35 KB, wrong
   directory — the real one is `trainer/Trainer.py`) and
   `trainRNNbrain/analyzers/TrainGTRNNs_tanh_slopes.slurm`. Left alone, but the misplaced
   `Trainer.py` is an import-shadowing hazard.

Calibration artifacts (a 400-iteration run that landed in the real `CDDM_screen3/arm=all3` folder,
and `CDDM_screen3_gpucheck`) were deleted before submission — a short run sitting in a real arm
folder would have silently contaminated the mean.

---

