## 2026-11-05 09:03 — Uniform settings: 3BFF weight decay 1e-5 and 1e-4, CDDM rescaling (issue #6)

Part of the uniform-interventions programme (engellab/trainRNNbrain issue #6,
`docs/interventions_matrix.md`; Pavel's instruction of 5 Nov: the same settings and rules on all
three tasks). Two cells at N = 1,000 were missing on the supervised tasks. Both are prepared here
on the standard network (`configs/model/rnn_relu_standard.yaml`), 75,000 iterations
(`configs/experiment/silent_units_largeN.yaml`, the same-size budget), 3 seeds each
(`seed=random`), Della.

1. **3BFF weight decay 1e-5 and 1e-4** (6 runs). wd 0 exists
   (`NBitFlipFlop_b5_optimizer/EqType=h_N=1000_opt=adam_wd=0_lr=1e-3`) and 1e-6 is the default.
   Launcher `slurm/b0_wd.slurm` (the CDDM weight-decay ladder) with a new `TASK` switch:
   `TASK=ff` passes `task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3`, selects
   `trainer=trainer_ptrack_freshbatch` and sets the budget to 75,000; TASK unset or `cddm` is the
   ladder as it ran (CDDM, `trainer_ptrack_plain`, 200,000). Output
   `NBitFlipFlop_std_g0_weightdecay_bias0/EqType=h_N=1000_wd={1e-5,1e-4}_iters=75000/`.
   Why the trainer is part of the switch: the saved configs of the three CDDM cells record
   `same_batch: true` (`trainer_ptrack_plain` reuses one batch of trials for every iteration), and
   every 3BFF control and the B5 wd 0 cell were trained with `trainer_ptrack_freshbatch`
   (`same_batch: false`, a fresh draw of pulse trains each iteration). On CDDM the flag changes
   nothing, because `TaskCDDM.get_batch` builds the same context-by-coherence grid on every call;
   on the flip-flop, whose trials are random draws, the two trainers train different networks, so
   the 3BFF cells match their own controls. A correction to the matrix: `CDDM_std_g0_tanh` is not
   the only standard cell with `same_batch: true`; the CDDM weight-decay ladder and
   `CDDM_std_g0_drift_bias0` record it too.

2. **CDDM rescaling, the paper's arm** (3 runs): `reinit_mode rescale`, `rescale_norm_mode self`,
   target 10 times the rate cap, `rescale_active_only`, `rescale_normalize`, alpha 1.002, cap 64,
   protect 0.20 N, check every step, patience 1. Launcher
   `slurm/SilentReLU_target_ladder_della.slurm`, cell 2 (`tgt=10.0`), with a `TASK` switch:
   `TASK=cddm` passes `task=CDDM` in place of the flip-flop words and sets the budget to 75,000;
   TASK unset or `ff` is the ladder as it ran (40,000 iterations). Output
   `CDDM_target_ladder/EqType=h_N=1000_tgt=10.0/` (no `k` in the folder: CDDM's task config
   carries its 6 inputs and 2 outputs).

Only the swept keys and the task words are on the command line; everything else comes from the
configs. Commit 99530cf (launchers are gitignored, added with `git add -f`).

**Smoke tests, pass conditions set before the runs.** Each launcher was run as `sbatch` runs it,
with `module`, `conda` and `srun` stubbed, the `srun` stub appending
`trainer.max_iter=30 model.N=50`, `REPO` at a clean detached worktree of 3e0595e, outputs under the
scratchpad. Six runs: the unedited launchers at indices 4 and 9, the edited launchers at the same
indices with TASK unset, and the two new arms. All exit 0. (1) The saved config of each unchanged
arm is identical to the unedited launcher's apart from the seed, the derived `task.seed` and
`save_to`: pass. (2) `b0_wd` TASK=ff index 4 records `taskname NBitFlipFlop`, 3 inputs and
3 outputs, `weight_decay 1e-05`, `same_batch false`, `optimizer adam`, folder
`NBitFlipFlop_std_g0_weightdecay_bias0/EqType=h_N=50_wd=1e-5_iters=75000`: pass. (3) The ladder
TASK=cddm index 9 records `taskname CDDM` (6 inputs, 2 outputs), `prune_reinit true`, the ten
rescale keys at the values above, `same_batch false`, folder
`CDDM_target_ladder/EqType=h_N=1000_tgt=10.0`: pass. (4) Between the two arms of each launcher the
configs differ only in the task block, the tag and, for `b0_wd`, the keys that differ between the
two trainer configs: pass. `max_iter: 30` in every saved config shows that Hydra took the trailing
override, so the launchers' own command lines are what ran.

**Submission.** Della has been unreachable from the Mac since 08:54 (ssh times out); it is polled
every 10 minutes for up to 4 hours. The commands, from the Della home directory (the main checkout
`~/trainRNNbrain` is on another branch and dirty, so a detached worktree at the commit; `--chdir`
because the provenance check's `python -c` puts the working directory ahead of `PYTHONPATH`;
`--constraint=nomig` as the guide says; 8 h for about 5.5 h of training at N = 1,000):

```
git -C ~/trainRNNbrain fetch origin main
git -C ~/trainRNNbrain worktree add --detach ~/trainRNNbrain_wd 99530cf
ls ~/trainRNNbrain_wd/slurm/b0_wd.slurm ~/trainRNNbrain_wd/slurm/SilentReLU_target_ladder_della.slurm
cd ~/trainRNNbrain_wd
REPO=$HOME/trainRNNbrain_wd TASK=ff   sbatch --chdir=$HOME/trainRNNbrain_wd --time=08:00:00 --constraint=nomig --array=4-9  slurm/b0_wd.slurm
REPO=$HOME/trainRNNbrain_wd TASK=cddm sbatch --chdir=$HOME/trainRNNbrain_wd --time=08:00:00 --constraint=nomig --array=9-11 slurm/SilentReLU_target_ladder_della.slurm
```

`SAVE_TO` stays at its default, `~/trainRNNbrain/data/trained_RNNs`. Indices: `b0_wd` decodes
`WD = WDS[(i-1)/3]` over `(0 1e-5 1e-4)`, so 4-6 are wd 1e-5 and 7-9 are wd 1e-4; the ladder
decodes `CELL = (i-1)/4` over `(6.0 8.0 10.0)` with four seeds per cell, so 9-11 are three seeds
of target 10.0. Job ids: a follow-up entry names them once the jobs are submitted.

**Pre-registered read-out, written before any run exists.** At 75,000 iterations, the end of
these runs, under the same-size rule of `docs/AGENT_GUIDE.md`: (a) the number of active units,
`common.active_mask` on the stored participation trace at the probe nearest 75,000 (the trainer
measures participation on a noise-free pass), as `fig_paper_F2_fails.read_fixed` does; (b)
held-out R² under the training noise from `heldout_r2.py` (36 draws) on the final weights, which
are the 75,000-iteration weights, so no estimate from the loss is needed. Controls, read the same
way: for 3BFF the B5 default cell `NBitFlipFlop_b5_optimizer/EqType=h_N=1000_opt=adam_wd=1e-6_lr=1e-3`
(3 runs, the same launcher family and trainer as the wd 0 cell; read at 75k in
`data/b5_read_cache.pkl`), with the wd 0 cell as the ladder's zero rung; for CDDM
`CDDM_paper_grid/EqType=h_N=1000_arm=control` (3 runs, 100,000 iterations, the same trainer,
`same_batch false`), with `CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0` (5 runs
ending at 75,000, the Figure 3a control) as a second reference. Verdicts: rescaling **works on
CDDM** if each of its three seeds has more active units than every control run and its mean
held-out R² is within 0.01 of the controls' mean (one-sided: a higher R² passes), the per-unit-rule
criterion of `docs/interventions_matrix.md` and `read_uniform_cells.py`; otherwise it does not
work, and the entry says which of the two conditions failed. A weight decay **changes the number of
active units on 3BFF** if all three of its seeds lie outside the range of the wd 1e-6 control's three
seeds, on the same side; the uniform question is whether the direction of the change over 0, 1e-6,
1e-5 and 1e-4 matches the CDDM ladder's at 75,000 (Figure 3b), with held-out R² reported beside
every count. A run whose loss diverges or that discards more than 5% of its gradient updates is
reported as broken and excluded, with the number of such runs given.
