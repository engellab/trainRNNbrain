## 2026-11-19 17:49 — Rules, gap B: B2, the rate-penalty size test, on CDDM (N = 500, 1,000, 2,000)

Written and committed before any run of this gap is submitted. Ranking and cluster choice:
`2026-11-19_1749_claim-gaps-ranking.md`.

**Claim at stake.** The paper reports that the rate penalty alone fails the stable-training test across sizes
(B2, entry 2026-10-31_1532), shown on 3BFF only. This entry applies B2's rule, unchanged, to CDDM.

**Terms.** *Rate penalty*: a two-sided penalty around a target firing rate, weight λ_frm (`trainer.lambda_frm`).
*Floor crossing*, *active unit*: as in the ranking entry. *Held-out R²*: R² on the seeded held-out batch under
the training noise, 36 noise draws (`heldout_r2`). *Updates skipped*: optimiser steps the supervised trainer
skips because of a gradient spike, counted from the training log line `gradient spikes: … updates skipped`, as
B2 counted them.

### Networks

| N | Rate penalty λ_frm = 0.1 (B2's value; the value run on all three tasks) | Controls |
|---|---|---|
| 500 | new: `CDDM_std_g0_penalties_bias0/EqType=h_N=500_pen=frm0.1_iters=75000`, 3 runs (`seed=random`) | `CDDM_paper_grid/EqType=h_N=500_arm=control`, 3 runs |
| 1,000 | existing: `CDDM_std_g0_penalties_bias0/EqType=h_N=1000_pen=frm0.1_iters=75000`, 3 runs (Della 16957419_4-6) | `CDDM_paper_grid/EqType=h_N=1000_arm=control`, 3 runs |
| 2,000 | new: `..._N=2000_pen=frm0.1_iters=75000`, 3 runs | `CDDM_paper_grid/EqType=h_N=2000_arm=control`, 3 runs |

The new runs come from the launcher of the N = 1,000 cell (`slurm/SilentReLU_std_penalties_della.slurm`, indices
4-6) with `N_OVERRIDE`, at the commit named in the submission entry, written to a new tree
(`SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs_cg`). 75,000 iterations, the same-size budget of the N = 1,000
cell. The paper-grid controls (100,000 iterations) are the controls of G6 and of the N = 1,000 rate-penalty
verdict; each run is read at its own floor, so the budgets need not match.

### Read-out (B2's)

Every run at its own noisy-loss floor crossing, 1.05× (`heldout_r2.read_rung`, which `read_b2_b4.py` uses):
the active count at the crossing; held-out R² at the crossing, measured on a kept snapshot where one exists
and otherwise estimated from the final weights as 1 − (L(t*)/L_end)(1 − R²_end); updates skipped from the
training log; whether the loss stays finite. The penalised run's noisy loss includes λ·penalty, so its floor
is task plus penalty (`docs/AGENT_GUIDE.md`). A run is *stable* if it has a floor read, fewer than 5% of its
updates skipped and a finite loss. Also at 1.10× and 1.20× the floor (on CDDM the crossing is uncertain by about
±30%); 1.05× binds, and the results entry says so if a looser margin gives another verdict.

### Rule (B2's, unchanged)

- **Supports** (the rate penalty trains stably across sizes on CDDM) if at every N (500, 1,000, 2,000): at least
  2 of 3 runs are stable; the mean active count at the floor is at least 1.5× the control's mean; and the mean
  held-out R² at the floor is within 0.02 of the control's mean.
- **Falsifies** (the rate penalty fails B2 on CDDM, as on 3BFF) if at any N ≥ 1,000 (1,000 or 2,000): fewer
  than 2 of 3 runs are stable, or the mean R² at the floor is more than 0.02 below the control's, or the
  multiplier is below synaptic noise's at the same N. Synaptic noise on the standard CDDM network exists only at
  N = 1,000 (`CDDM_synnoise_ext/EqType=h_N=1000_sw=1.0_iters=100000`, σ_w = 1.0), read at its floor by the same
  script; at N = 500 and 2,000 that clause cannot be evaluated and is reported as such.
- **Neither:** numbers, no verdict. A failure at N = 500 alone keeps "supports" from holding but does not
  falsify (B2's falsification clause names N ≥ 1,000).

### What each outcome does to the paper

- **Falsifies:** the stable-training failure holds on 3BFF and CDDM (N = 500-2,000 on CDDM).
- **Supports:** the failure is specific to 3BFF. The paper keeps the 3BFF result and states that on CDDM the
  rate penalty passes the same test at N = 500-2,000.
- **Neither:** the claim stays scoped to 3BFF; the CDDM numbers are reported.

N = 4,000 is not run (44-54 h for 75,000 iterations, past the 23 h jobs that start before the deadline). On
3BFF, B2 failed at N = 1,000 (no stable run) and at N = 2,000 (R² 0.024 below the control), so the sizes run
here include both sizes at which 3BFF failed.

### Jobs (Della)

`REPO=<pinned clone> N_OVERRIDE=<N> SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs_cg TASK=cddm sbatch --chdir=$REPO
--array=4-6 slurm/SilentReLU_std_penalties_della.slurm`, with `--time=08:00:00` at N = 500 and
`--time=23:00:00` at N = 2,000 (0.62-0.75 s per iteration plus the penalty: about 14-17 h).
