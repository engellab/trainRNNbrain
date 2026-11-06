## 2026-11-05 22:59 — Uniform settings, package 2a: the 3BFF settings ladders and the CDDM third seeds (issue #6)

Part of the uniform-interventions programme (engellab/trainRNNbrain issue #6,
`docs/interventions_matrix.md`, section "3BFF (Della, standard network, N = 1,000, 75,000
iterations)"; Pavel approved the package on 5 Nov). Package 1 (3BFF weight decay 1e-5 and 1e-4,
CDDM rescaling) is entry 2026-11-05_0903; its jobs are running (WdB0 16955456, TgtLadder 16955457).
This package fills every other missing 3BFF settings cell at N = 1,000 on the standard network
(`configs/model/rnn_relu_standard.yaml`, bias fixed at 0), 75,000 iterations (the same-size budget
of `configs/experiment/silent_units_largeN.yaml`), `trainer_ptrack_freshbatch` (`same_batch false`,
the trainer of every 3BFF control; entry 2026-11-05_0903 says why the trainer is part of the task
switch), 3 seeds each (`seed=random`), plus the third seed of the five CDDM recurrent-init cells.
56 runs on Della, about 310 GPU-h at 5.5 h per run.

**The cells.** Only the swept keys and the task words are on the command line; everything else
comes from the configs. Indices are array task ids of the named launcher. Folders are under
`~/trainRNNbrain/data/trained_RNNs/` on Della.

| Setting | Values | Launcher and switch | Indices | Output folder | Runs |
|---|---|---|---|---|---|
| Activity noise sigma_rec | 0, 0.01, 0.1 | `b0_noise.slurm`, `TASK=ff` | 1-3, 6-8, 16-18 | `NBitFlipFlop_std_g0_noise_bias0/EqType=h_N=1000_sigrec=<v>_iters=75000` | 9 |
| Spectral radius | 0.5, 2.0 | `SilentReLU_std_recinit_della.slurm`, `TASK=ff NSEED=3` | 1-3, 7-9 | `NBitFlipFlop_std_g0_recinit/EqType=h_N=1000_sr=<v>_iters=75000` | 6 |
| Recurrent-weight penalty lambda_rwm | 0.05, 0.2 | the same | 10-15 | `NBitFlipFlop_std_g0_recinit/EqType=h_N=1000_rwm=<v>_iters=75000` | 6 |
| CDDM third seed: spectral radius 0.5, 1.2, 2.0; lambda_rwm 0.05, 0.2 | one seed per cell | the same, `TASK=cddm NSEED=1` | 1-5 | `CDDM_std_g0_recinit/EqType=h_N=1000_<cell>_iters=75000` (the existing folders, two seeds each) | 5 |
| Metabolic penalty lambda_met | 0.01, 0.1, 1, 10 | `b0_met.slurm`, `TASK=ff` | 7-9, 16-18, 25-27, 34-36 | `NBitFlipFlop_std_g0_metabolic_bias0/EqType=h_N=1000_LmbdMet=<v>_iters=75000` | 12 |
| Sparsity penalty alone lambda_rws | 0.05 (lambda_frm 0) | `SilentReLU_flipflop_frmboth_long_della.slurm`, arm `rws`, `ITERS_OVERRIDE=75000` | 166-168 | `NBitFlipFlop_std_penlong/EqType=h_k=3_N=1000_pen=rws_iters=75000` | 3 |
| Activation | leaky ReLU (leak 0.01), softplus beta 25, shifted sigmoid, tanh | `b0_act.slurm`, `TASK=ff` | 1-12 | `NBitFlipFlop_std_g0_activations_bias0/EqType=h_N=1000_act={softplus25,leakyrelu,sigmoid,tanh}_iters=75000` | 12 |
| Input-weight cap | 0.3 | `SilentReLU_screen3_della.slurm`, arm `cap`, `TASK=ff` | 4-6 | `NBitFlipFlop_screen3/EqType=h_N=1000_pen=none_arm=cap_cap=0.3_iters=75000` | 3 |

The rungs that are not run (sigma_rec 0.05, spectral radius 1.2, no penalty, ReLU, no cap) are
the 3BFF N = 1,000 controls named under the read-out.

**What changed in the launchers (commit 56d0338; launchers are gitignored, added with `git add -f`).**
Each CDDM launcher got the `TASK` switch of `b0_wd.slurm` (entry 2026-11-05_0903): `TASK=ff`
passes `task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3`, selects `trainer_ptrack_freshbatch`,
sets the 75,000-iteration budget and the `NBitFlipFlop_` folder prefix; TASK unset or `cddm` is
the launcher as it ran. Per launcher:

- `b0_noise.slurm`: the switch; the ff arm runs under `+experiment=silent_units_largeN` with
  `model.N=1000` on the command line in place of the CDDM arm's `silent_units_N1000`. The two
  experiment configs set the same size and budget; `silent_units_largeN` also sets
  `light_outputs: true` (no CPU analysis after training, parameters saved as .npz). Every 3BFF
  cell at N = 1,000 so far (`b0_winp`, `b0_ffref`, B5) ran that way, and no 3BFF run has gone
  through the CPU analysis, whose trajectory array is 2.3 times CDDM's (1,024 trials against
  450). The ff folder is `NBitFlipFlop_std_g0_noise_bias0`, named like the other 3BFF bias-0
  ladders; the CDDM folder carries the commit hash instead (`CDDM_77fbd9_g0_noise_ptrack`).
- `SilentReLU_std_recinit_della.slurm`: the switch, and `NSEED` read from the environment
  (default 2, the value that was hard-coded), so one more index per cell gives the CDDM third
  seeds. The CDDM arm keeps `trainer_ptrack_plain`, which its two existing seeds used.
- `b0_met.slurm`: the switch; the ff arm runs under `silent_units_largeN` in place of
  `silent_units_std` (the same size and budget, light outputs, as for `b0_noise`), and its folder
  carries `_iters=75000`; the CDDM folders carry no budget, because the original runs were 30,000
  iterations and the 75,000 reads are the `_c75000` top-ups. The index decode is unchanged, so the
  N = 1,000 cells are the indices in the table.
- `SilentReLU_flipflop_frmboth_long_della.slurm`, chosen over `SilentReLU_std_penalties_spock.slurm`:
  it already is 3BFF, the standard network, `trainer_ptrack_freshbatch` and Della, whereas the Spock
  launcher is CDDM with `trainer_ptrack_plain`, Spock modules and an rsync provenance check. A
  third arm `rws` (`lambda_rws 0.05`, `lambda_frm 0`) as `P_IDX=2`, tasks 145-216, with the same
  inner decode, so tasks 1-144 are what they were.
- `b0_act.slurm`: the switch, and a fourth activation `tanh` (`model=rnn_tanh_standard`, indices
  10-12, leaving 1-9 as they were). No new model config was needed: `configs/model/rnn_tanh_standard.yaml`
  exists since commit e00924b (`rnn_relu_standard.yaml` with the activation changed: bias fixed
  at 0, the same noise, `init_draws: independent`). The matrix proposed adding it; it was there.
- `SilentReLU_screen3_della.slurm`: the switch (`task=NBitFlipFlop ...` in place of the hard-coded
  `task=CDDM`); the trainer was already `trainer_ptrack_freshbatch`. The ff folder carries the cap
  value and the budget (`arm=cap_cap=0.3_iters=75000`), so a second cap value or budget on 3BFF
  would not pool with this one.

**Smoke tests, pass conditions written before the runs.** Each launcher was run as `sbatch` runs
it, with `module`, `conda` and `srun` stubbed, the `srun` stub appending
`trainer.max_iter=30 model.N=50`, `REPO` at a clean detached worktree of 665b568 (the commit
before the launcher edits), outputs under the scratchpad. 22 runs: each launcher at one index
unedited and edited with TASK unset (12 runs), and every kind of new arm (sigma_rec 0.01;
spectral radius 0.5; lambda_rwm 0.05; the CDDM third seed of lambda_rwm 0.2; lambda_met 0.01 and
10; rws; leaky ReLU and tanh; cap 0.3). All 22 exit 0. (1) For each launcher the saved config of
the edited launcher with TASK unset is identical to the unedited launcher's apart from `seed`,
the derived `task.seed` (flip-flop only) and `paths.save_to`: pass, six of six. (2) Each new
arm's saved config records `taskname NBitFlipFlop` (CDDM for the third seed), 3 inputs and
3 outputs, `same_batch false` (true for the CDDM third seed, as its two existing seeds),
`bias_range [0, 0]`, the setting at its value, `max_iter 30` and `N 50` (Hydra took the trailing
stub overrides, so the launcher's own command line is what ran) and the folder in the table: pass,
ten of ten. (3) Between the unchanged arm and the new arm of each launcher the configs differ only
in the task block, the tag, the setting, the keys that differ between the two trainer configs
(`same_batch`, and the optimiser and gradient-spike keys that only `trainer_ptrack_freshbatch`
declares) and, for `b0_noise` and `b0_met`, `light_outputs`; the tanh arm also has
`model.inp_weight_cap: null` where the leaky-ReLU config lacks the key (both mean no cap): pass.
(4) The composed config of each new arm without the stub overrides (Hydra `--cfg job`) has
`max_iter 75000` and `N 1000`: pass, ten of ten.

**Submission.** From the Della home directory: a detached worktree at the commit (the main
checkout `~/trainRNNbrain` is on another branch and dirty), `--chdir` at the worktree (the
provenance check's `python -c` puts the working directory ahead of `PYTHONPATH`),
`--constraint=nomig`, `SAVE_TO` the default `~/trainRNNbrain/data/trained_RNNs` (passed
explicitly, because `b0_noise.slurm` defaults to `$REPO/data/trained_RNNs`), 8 h for about 5.5 h
of training at N = 1,000. The worktree was clean (`git status --porcelain` empty) and held the six
launchers and `rnn_tanh_standard.yaml` before the first `sbatch`.

```
git -C ~/trainRNNbrain fetch origin main
git -C ~/trainRNNbrain worktree add --detach ~/trainRNNbrain_p2a 56d0338
cd ~/trainRNNbrain_p2a
R=$HOME/trainRNNbrain_p2a; ST=$HOME/trainRNNbrain/data/trained_RNNs
C="--chdir=$R --time=08:00:00 --constraint=nomig"
REPO=$R SAVE_TO=$ST TASK=ff              sbatch $C --array=1-3,6-8,16-18         slurm/b0_noise.slurm                                # 16957556
REPO=$R SAVE_TO=$ST TASK=ff   NSEED=3    sbatch $C --array=1-3,7-15              slurm/SilentReLU_std_recinit_della.slurm           # 16957557
REPO=$R SAVE_TO=$ST TASK=cddm NSEED=1    sbatch $C --array=1-5                   slurm/SilentReLU_std_recinit_della.slurm           # 16957558
REPO=$R SAVE_TO=$ST TASK=ff              sbatch $C --array=7-9,16-18,25-27,34-36 slurm/b0_met.slurm                                  # 16957559
REPO=$R SAVE_TO=$ST ITERS_OVERRIDE=75000 sbatch $C --array=166-168               slurm/SilentReLU_flipflop_frmboth_long_della.slurm  # 16957560
REPO=$R SAVE_TO=$ST TASK=ff              sbatch $C --array=1-12                  slurm/b0_act.slurm                                  # 16957561
REPO=$R SAVE_TO=$ST TASK=ff              sbatch $C --array=4-6                   slurm/SilentReLU_screen3_della.slurm               # 16957562
```

Job ids 16957556 to 16957562, 56 tasks (9 + 12 + 5 + 12 + 3 + 12 + 3), all pending at 22:59.
The queue then held 89 tasks of this account in all, including package 1 and the CDDM arrays of
other agents (B5opt 16957418, StdPen 16957419); gpu-short runs 24 per user at a time, so the last
of these start after about three waves. Logs: `~/trainRNNbrain_logs/<name>.<id>_<task>.out`.

**Pre-registered read-out, written before any run exists.** At 75,000 iterations, the end of
these runs, under the same-size rule of `docs/AGENT_GUIDE.md`: (a) the number of active units,
`common.active_mask` on the stored participation trace at the probe nearest 75,000 (the trainer
measures participation on a noise-free pass), as `fig_paper_F2_fails.read_fixed` does; (b)
held-out R² under the training noise from `heldout_r2.py` (36 draws) on the final weights, which
are the 75,000-iteration weights. Controls, read the same way: the 3BFF N = 1,000 controls
`NBitFlipFlop_b5_optimizer/EqType=h_N=1000_opt=adam_wd=1e-6_lr=1e-3` (3 runs, the same trainer,
read at 75,000 in `data/b5_read_cache.pkl`), with
`NBitFlipFlop_std_ksweep_bias0/EqType=h_k=3_N=1000_iters=150000` (3 runs) as a second reference.
A setting **changes the number of active units on 3BFF** if all three of its seeds lie outside
the range of the controls' three seeds, on the same side; a ladder (noise, spectral radius,
lambda_rwm, lambda_met) is read rung by rung. The uniform question per setting is whether the
direction of the change matches CDDM's at 75,000 (Figure 3a,b for noise, the metabolic penalty,
the activations, rws and the cap; `CDDM_std_g0_recinit` for the spectral radius and lambda_rwm),
with held-out R² reported beside every count; a cell whose mean held-out R² is below 0.7 has its
count reported but not interpreted (the recinit launcher's gate). The tanh cell is also read for
saturated units (rates pinned near plus or minus 1), which participation scores as active
(`rnn_tanh_standard.yaml` header). The CDDM third seeds join their cells, which are then read as
the matrix's CDDM table says: active units at 75,000 and held-out R² against the sr 1.2 cell. A
run whose loss diverges or that discards more than 5% of its gradient updates is reported as
broken and excluded, with the number of such runs given.
