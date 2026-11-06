## 2026-11-05 22:56 — Uniform settings on CDDM, package 2b: the optimiser comparison, the rate penalty 0.1 alone and with the sparsity penalty (issue #6)

Part of the uniform-interventions programme (engellab/trainRNNbrain issue #6,
`docs/interventions_matrix.md`, section "CDDM (Della, standard network, N = 1,000, 75,000
iterations)"; Pavel's go-ahead of 5 Nov: "Go ahead on additional jobs"). Three CDDM cells are
submitted here, every one on the standard network (`configs/model/rnn_relu_standard.yaml`: ReLU,
bias fixed at 0, weight decay 1e-6, Adam unless the arm says otherwise), N = 1,000, 75,000
iterations (the same-size budget of `configs/experiment/silent_units_largeN.yaml`), 3 seeds each
(`seed=random`), `trainer_ptrack_freshbatch` (the trainer of the `CDDM_paper_grid` control,
`same_batch false`), Della. The fourth cell of the package, the threshold lift (bias kick), is
dropped on Pavel's instruction of 5 Nov evening: no launcher work, no check and no submission for
it, and the status board records the decision (commit f057218). For the record, since the matrix
asked for the check: `Trainer.prune_and_reinit_` writes the bias offset into the fixed bias tensor
when the bias is not a Parameter (`b[idx] = offs`), the saved parameters carry `bias` whenever
`bias_range` is set, and the 3BFF cell `NBitFlipFlop_replace_variants/..._mode=bias_kick_...`
ran with `bias_range [0, 0]` and ends with 457 non-zero biases, so the mode would have acted on
the standard network. It is not run.

**Cells, launchers and the commit (4621b9c; launchers are gitignored, added with `git add -f`).**

1. **Optimiser comparison, 15 runs.** AdamW at weight decay 1e-6, and SGD at weight decay 1e-6
   over the base learning rates 0.03, 0.1, 0.3 and 1 (arms 3-7 of the B5 table; base lr is scaled
   by (100/N)^(1/3) = 0.464 at N = 1,000, as on 3BFF). The Spock launcher
   `SilentReLU_b5_optimizer_spock.slurm` is 3BFF-only through its experiment config and carries
   Spock's module loads and partition, so a Della copy stands beside it, unchanged on the flip-flop:
   `slurm/SilentReLU_b5_optimizer_della.slurm`, with Della's module loads, the git provenance check
   of the other Della launchers and a `TASK` switch (`TASK` unset or `ff`: `+experiment=b5_optimizer_ff`,
   the Spock launcher's run; `TASK=cddm`: `+experiment=b5_optimizer_cddm`). The new config
   `configs/experiment/b5_optimizer_cddm.yaml` is `b5_optimizer_ff.yaml` with `override /task: CDDM`
   and `max_iter: 75000`; it keeps the B5 family's `checkpoint_every: 5000` (the trainer config's default is 1,000; the on-disk control
   predates checkpointing; snapshots do not change training). Output
   `CDDM_b5_optimizer/EqType=h_N=1000_opt={adamw,sgd}_wd=1e-6_lr={1e-3,0.03,0.1,0.3,1}/`. Adam at
   wd 0 is not rerun: it exists in the CDDM weight-decay ladder
   (`CDDM_std_g0_weightdecay_bias0/EqType=h_N=1000_wd=0_iters=200000`, read at 75,000 in the
   Figure 3 cache).
2. **Rate penalty 0.1 alone, 3 runs** (`trainer.lambda_frm=0.1 trainer.lambda_rws=0`), the 3BFF
   and maze value; the standard CDDM cell that exists has 0.2.
3. **Rate + sparsity 0.1 + 0.05, 3 runs** (`trainer.lambda_frm=0.1 trainer.lambda_rws=0.05`), the
   3BFF and maze pair.
   Cells 2 and 3 run from `slurm/SilentReLU_std_penalties_della.slurm`, the Della counterpart of
   `SilentReLU_std_penalties_spock.slurm`, which is unchanged. A copy with a switch was not enough:
   the Spock launcher trained `CDDM_std_g0_penalties` before 2026-09-21 with a trainable bias,
   `trainer_ptrack_plain` and 200,000 iterations, has a size axis (its indices 1-9 are N = 500, so
   the matrix's `--array=4-6` and `7-9` would have been N = 500 cells), and its provenance check
   reads the `CODE_VERSION.txt` of a tree copied by rsync, which a git worktree does not have. The
   Della launcher keeps the three arms (`rws0.05`, `frm0.1`, `frm0.1_rws0.05`, indices 1-3, 4-6
   and 7-9, `PEN = (i-1)/3`) at N = 1,000 (`N_OVERRIDE` otherwise), the standard network,
   `trainer_ptrack_freshbatch`, 75,000 iterations (`ITERS_OVERRIDE` otherwise), `track_valid false`
   as the control, and a `TASK` switch (`cddm` by default; `ff` passes the 3-bit flip-flop words).
   Output `CDDM_std_g0_penalties_bias0/EqType=h_N=1000_pen={frm0.1,frm0.1_rws0.05}_iters=75000/`.

Only the swept keys and the task words are on the command line; everything else comes from the
configs. Runtime: on `CDDM_std_g0_bias0` (50,000 iterations, N = 1,000, `sacct`) the frm 0.2 arms
took 2:04-2:30 and the unpenalised arms 1:52-1:59, so the rate penalty adds 10-25% on the standard
trainer (the penalty is added to the task loss; no second backward pass), and 75,000 iterations fit
in the 8 h requested.

**Smoke tests, pass conditions set before the runs.** Each launcher was run as `sbatch` runs it,
from a clean detached worktree of 4621b9c in the scratchpad, with `module`, `conda` and `srun`
stubbed, the `srun` stub appending `trainer.max_iter=30 model.N=50`, outputs under
`/private/tmp/claude-502/-Users-pt1290-Documents-GitHub-trainRNNbrain/f99068d5-e5b3-4db6-8bd2-d256366ab285/scratchpad/smoke/out/`.
Eight runs, all exit 0 with one saved config each: the Spock B5 launcher at index 7 (AdamW), the
Della B5 launcher at index 7 with `TASK` unset, the Della B5 launcher with `TASK=cddm` at indices 7
(AdamW) and 10 (SGD 0.03), the paper-grid launcher with `TASK=cddm` at index 4 (the N = 1,000
control, the reference), and the penalties launcher at indices 4 (frm) and 7 (both) with `TASK`
unset and at index 4 with `TASK=ff`. (P1) The Della B5 launcher on the flip-flop saves a config
identical to the Spock launcher's apart from `seed`, `task.seed` and `save_to`: pass. (P2) The
CDDM optimiser arms record `taskname CDDM` (6 inputs, 2 outputs), `optimizer adamw` or `sgd`,
`weight_decay 1e-06`, `lr 0.001` or `0.03`, `same_batch false`, `bias_range [0, 0]`,
`lambda_rws 0`, `lambda_frm 0`; the config composed without the stub (Hydra `--cfg job`,
`+experiment=b5_optimizer_cddm`) has `max_iter 75000`: pass. (P3) Against the paper-grid control
the optimiser arms differ only in `optimizer`, `lr` (SGD), `checkpoint_every` (5000 against 1000),
the tag, the seed and `save_to`: pass. (P4) The penalty arms differ from the control only in
`lambda_frm` and, for `both`, `lambda_rws`, the tag, the seed and `save_to`; their echo lines say
`iters=75000` and name the folders above: pass. (P5) The `ff` arm differs from the `cddm` arm only
in the task block, the seeds and `save_to`: pass. (P6) `max_iter: 30` in every saved config, so
Hydra took the trailing override and the launchers' own command lines ran: pass.

**Submission (Della, 2026-11-05, about 22:54 Mac time).** From a detached worktree at the commit, the main
checkout untouched; `--chdir` because the provenance check's `python -c` puts the working
directory ahead of `PYTHONPATH`; `SAVE_TO` at its default `~/trainRNNbrain/data/trained_RNNs`;
12 jobs were queued before, 33 after, under the 60-job cap:

```
git -C ~/trainRNNbrain fetch origin main
git -C ~/trainRNNbrain worktree add --detach ~/trainRNNbrain_p2b 4621b9c
cd ~/trainRNNbrain_p2b
REPO=$HOME/trainRNNbrain_p2b TASK=cddm sbatch --chdir=$HOME/trainRNNbrain_p2b --time=08:00:00 --constraint=nomig --array=7-21 slurm/SilentReLU_b5_optimizer_della.slurm
REPO=$HOME/trainRNNbrain_p2b TASK=cddm sbatch --chdir=$HOME/trainRNNbrain_p2b --time=08:00:00 --constraint=nomig --array=4-9  slurm/SilentReLU_std_penalties_della.slurm
```

Job ids: **16957418** (`B5opt`, indices 7-21: 7-9 AdamW, 10-12 SGD 0.03, 13-15 SGD 0.1, 16-18
SGD 0.3, 19-21 SGD 1; `ARM = (i-1)/3` over the 15-arm table) and **16957419** (`StdPen`, indices
4-6 frm 0.1, 7-9 frm 0.1 + rws 0.05). Both in `gpu-short`, 8 h each; 21 runs, about 115 GPU-h.
Logs `~/trainRNNbrain_logs/B5opt.16957418_<i>.out` and `StdPen.16957419_<i>.out`.

**Pre-registered read-out, written before any run exists.** At 75,000 iterations, the end of
these runs, under the same-size rule of `docs/AGENT_GUIDE.md`: (a) the number of active units,
`common.active_mask` on the stored participation trace at the probe nearest 75,000 (the trainer
measures participation on a noise-free pass), as `fig_paper_F2_fails.read_fixed` does; (b) held-out
R² under the training noise from `heldout_r2.py` (36 draws) on the final weights, which are the
75,000-iteration weights. The control, read the same way: `CDDM_paper_grid/EqType=h_N=1000_arm=control`
(3 runs, 100,000 iterations, the same trainer; at 75,000 it has 342, 334 and 333 active units and
a mean held-out R² of 0.834, entry 2026-11-05_0926), with
`CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0` (5 runs ending at 75,000, 325 active
on average, R² 0.833) as a second reference; `read_uniform_cells.py` is the script to extend.
Verdicts. An arm **has more active units at no cost in R²** if each of its three seeds has more
active units than the largest control count and its mean held-out R² is at least the controls'
mean minus 0.01 (the criterion of `docs/interventions_matrix.md`); it **has more active units only
at lower R²** if the counts pass and the R² condition fails; otherwise it does not increase the
number of active units. The optimiser question is whether CDDM reproduces the 3BFF pattern, in
which networks trained with Adam at wd 0 or with AdamW have more active units at a higher held-out
R² than the Adam wd 1e-6 control, and networks trained with SGD have more active units only at a
lower R²: CDDM reproduces it if AdamW (and the ladder's Adam wd 0 cell, read at 75,000) is in the
first class and every SGD arm whose counts pass is in the second class; the entry that reads the
runs states the class of every arm with its three counts and mean R² beside the control's. An SGD
run with held-out R² below 0.8 is reported as not having learned the task and is excluded from the
pattern, with the number of such runs given (on 3BFF, SGD at lr 3 and 10 did not learn, which is
why they are not run). The rate-penalty rule as the matrix states it: the 0.1 arm **works on CDDM**
if each of its three seeds has more active units than every control run and its mean held-out R²
is within 0.01 of the controls' mean (one-sided: a higher R² passes); the same for the 0.1 + 0.05
pair; the entry says which of the two conditions failed otherwise, and puts the 0.2 cells
(`CDDM_std_g0_bias0_c75000`, LmbdFR 0.2 alone and with LmbdRWS 0.05: all 1,000 active, R² 0.004
and 0.008 lower) beside them. A run whose loss diverges or that discards more than 5% of its
gradient updates is reported as broken and excluded, with the number of such runs given.
