# Interventions and settings across 3BFF, CDDM and the maze: what exists, what is missing

Inventory written 2026-11-05 08:33 (repo at edc46f5) for Pavel's instruction of 5 Nov: report the
interventions that work on every task first, then the ones that work on some tasks only; so the
same training settings and per-unit rules have to be tested on all three tasks. The three tasks are
the 3-bit flip-flop (3BFF), context-dependent decision making (CDDM) and the maze (a reinforcement
learning task trained by PPO, proximal policy optimisation; it replaced DMTS on 4 Nov,
dead-relu-paper#182). Every comparison is at N = 1,000 units. The supervised tasks are read at a
fixed 75,000 iterations (the same-size rule, `docs/AGENT_GUIDE.md`); the maze is read at update 2,000
and at its "learned" probe (`common.maze_learned_index`: the first probe from which the clean probe's
success stays at or above 0.9 for five probes).

**How the cells were counted.** Directories were listed on Della (`~/trainRNNbrain/data/trained_RNNs`),
on Spock (`~/trainRNNbrain/data/trained_RNNs` and `~/trainRNNbrain/data/trained_RNNs_rl_maze`) and on
the Mac (`data/trained_RNNs`, the only copy of some deleted sweeps), with one seed folder counted as
one run; a run's saved config was read where the folder name does not say the setting or the bias.
Read-out status comes from the caches the figure scripts use (`data/fig_paper_F2_fails_cache.pkl`,
`data/b5_read_cache.pkl`, `data/remedy_reads_31oct/remedies_catalogue.json`) and from issue
engellab/trainRNNbrain#5 and the notebook entries of 5 Nov.

**Cell notation.** `n` runs; location D = Della, S = Spock, M = Mac only; `floor k/n` = k of n runs
have a read at 1.05 times their noisy-loss floor; `75k` = read at 75,000 iterations; "standard" =
the standard network (ReLU, bias fixed at 0, weight decay 1e-6, Adam); "pre-standard" = a trainable
bias in [-1, 1], which the paper does not compare with standard networks. Launchers are under
`slurm/`; most were last changed in the repair commit 542b9f0 (15 Oct), so a launcher's commit is
not the commit a sweep ran at; the run commit is given where the notebook or the guide records it.
The maze control cells are the uniform series' N = 1,000 ReLU runs (`Maze_suite/N=1000_act=relu`,
seeds 1-3, Spock array 7383494 at 9dce02e): 426, 449 and 415 active units at update 2,000, end
success 0.99, 0.98 and 0.73 (seed 3 did not learn).

## A. Standard training settings (Figure 3 and the Extended Data figure of every setting sweep)

| Setting | 3BFF | CDDM | Maze |
|---|---|---|---|
| Weight decay 0, 1e-5, 1e-4 (default 1e-6) | **wd 0 only**: `NBitFlipFlop_b5_optimizer/EqType=h_N=1000_opt=adam_wd=0_lr=1e-3`, n = 3, 200k, S + M; read at floor, 75k and 200k (Fig. 3d); launcher `SilentReLU_b5_optimizer_spock.slurm` (42e8811). **wd 1e-5, 1e-4: MISSING** | `CDDM_std_g0_weightdecay_bias0/EqType=h_N=1000_wd={0,1e-5,1e-4}_iters=200000`, n = 3 each, D + M; floor 3/3 each and 75k (Fig. 3a,b); default `CDDM_std_g0_drift_bias0`, n = 3; also `CDDM_std_g0_wd0_long` (wd 0, 300k, n = 3). Launcher `b0_wd.slurm`, ran from `~/trainRNNbrain_stdfix` at 77fbd97 plus uncommitted trainer edits (guide) | **MISSING** (default 1e-6 in every run). `trainer.weight_decay` is a key of `configs/trainer/trainer_ppo.yaml`; launcher arm only |
| Activity noise σ_rec 0, 0.01, 0.1 (default 0.05) | **MISSING** (every 3BFF run has σ_rec 0.05) | `CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec={0,0.01,0.05,0.1}_iters=200000`, n = 5 each, D + M; floor 5/5 each and 75k. Also `CDDM_absorb_noise/EqType=h_N=1000_noise={zero,default}` (all noise off), n = 3 each, 75k, D (entry 2026-11-02_0058). Launcher `b0_noise.slurm` (+ `b0_noise_topup.slurm`), 77fbd97 as above | **MISSING**. `model.sigma_rec` (and `model.sigma_inp`) are model keys; TrainerPPO's rollouts draw the recurrent and input noise as `RNN_torch.forward` does (its module docstring, lines 17-27); launcher arm only |
| Initial input-weight row norm 0.5, 2, 5, 20 (default draw) | `NBitFlipFlop_std_winp_bias0/EqType=h_k=3_N=1000_s={0.5,2,5,20}_iters=150000`, n = 3 each, D + M; floor 3/3 each and 75k (Fig. 3a,c); default `NBitFlipFlop_std_ksweep_bias0/EqType=h_k=3_N=1000_iters=150000`, n = 3. Launcher `b0_winp.slurm` / `SilentReLU_flipflop_winp_spock.slurm`, 77fbd97 as above | `CDDM_c1_snapshots/EqType=h_N=1000_arm=winp{0.5,2,5,20}`, n = 3 each, 50k, D + M; floor 2-3/3; read at 50k (both ends; Extended Data only; counts 166-336 vs 355-376). Launcher `SilentReLU_c1_snapshots_della.slurm` (entry 2026-10-13_1523) | **MISSING**. `model.input_row_norm` is set in `RNN_torch.__init__` (line 412), so the maze model takes it; launcher arm only. The maze has 9 inputs (8 sensors and the efference copy), so a row norm drives a unit harder than on 3BFF (3 inputs) |
| Input-weight cap (clamp on \|W_inp\| after every step) | `NBitFlipFlop_std_inpcap/EqType=h_k=3_N=1000_cap={0.1,0.5,2.0}_pen=none`, n = 3 each, 150k, standard (bias [0, 0] in the saved config), S + M. **Not read by any paper script** (`table_rule_screen.py` skips `std_inpcap`; no ladder in `fig_knobs_at_floor.py`). CDDM used cap 0.3; no 3BFF cell at 0.3 | `CDDM_screen3_c75000/EqType=h_N=1000_pen=none_arm=cap` (cap 0.3), n = 3, 75k, D + M; floor 3/3, 75k (Fig. 3e: 390-400 vs 332-340). Launcher `SilentReLU_screen3_della.slurm` | **MISSING, needs code**: the clamp is `Trainer.py` line 1421 (after an optimiser step), not in the model, so TrainerPPO has no cap |
| Spectral radius 0.5, 2.0 (default 1.2) | **MISSING** | `CDDM_std_g0_recinit/EqType=h_N=1000_sr={0.5,1.2,2.0}_iters=75000`, **n = 2 each**, D + M; floor 2/2, 75k. Launcher `SilentReLU_std_recinit_della.slurm` (job 14919348). A third seed is on the board's "Should" list | **MISSING**. `model.spectral_rad` is a model key; launcher arm only |
| Recurrent-weight magnitude penalty λ_rwm 0.05, 0.2 | **MISSING** | `CDDM_std_g0_recinit/EqType=h_N=1000_rwm={0.05,0.2}_iters=75000`, **n = 2 each**, D + M; floor 2/2, 75k; control = the sr 1.2 cell. Same launcher (`+trainer.lambda_rwm=<v> +trainer.rwm_args={}`) | **MISSING, needs code**: TrainerPPO has no λ_rwm term |
| Sparsity penalty alone λ_rws 0.05 | **MISSING on the standard network**. Pre-standard runs: `NBitFlipFlop_std_dropout/EqType=h_k=3_N=1000_pen=rws_do=none`, n = 4, S + M (trainable bias) | `CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0.05_LmbdFR=0`, n = 5, 75k, D + M; floor 0/5 (the loss ends below its fitted floor), 75k (Fig. 3a, among the "15 other values"). Launcher `SilentReLU_std_penalties_spock.slurm` (A4, entry 2026-10-14_0019) | **MISSING**. `trainer.lambda_rws` is implemented (`TrainerPPO.rec_weights_sparsity_penalty`); launcher arm only |
| Metabolic (one-sided rate) penalty λ_met 0.01, 0.1, 1, 10 | **MISSING** | `CDDM_std_g0_metabolic_bias0_c75000/EqType=h_N=1000_LmbdMet={0.01,0.1,1.0,10.0}`, n = 3 each, 75k, D + M; floor 3/3 each, 75k (Fig. 3a). Launcher `b0_met.slurm` (+ `b0_met_topup.slurm`) | **MISSING, needs code**: TrainerPPO has no λ_met term |
| Activation: leaky ReLU (0.01), softplus (β 25), shifted sigmoid, tanh | **MISSING on the standard network**. Pre-standard (bias [-1, 1], commit a831629): `NBitFlipFlop_std_activations/EqType=h_k=3_N=1000_act={leakyrelu,softplus25}_iters=150000` and `NBitFlipFlop_std_sigmoid/EqType=h_k=3_N=1000_iters=150000`, n = 3 each, **M only** (on neither cluster). No 3BFF tanh cell. Launcher `SilentReLU_flipflop_activations_della.slurm` (pinned to a831629 on purpose: its header says a bias-0 rerun would import a bias change) | `CDDM_std_g0_activations_bias0/EqType=h_N=1000_act={leakyrelu,softplus25,sigmoid}_iters=200000`, n = 3 each, D + M; floor 3/3 each, 75k (Fig. 3a). tanh: `CDDM_std_g0_tanh/EqType=h_N=1000_act=tanh_iters=150000`, n = 9 on D (3 on M), end of 150k, not a floor read (Fig. 4, the tanh Extended Data figure); its saved config has `same_batch: true`, unlike every other standard cell. Launchers `b0_act.slurm`, `SilentReLU_std_tanh_della.slurm` | `Maze_suite/N=1000_act={leaky_relu,softplus_beta25,sigmoid_shifted,tanh}`, **n = 2 each** (seeds 1-2), S-maze (and the pre-warm-up first suite on D, array 16856486, not comparable); read at update 2,000 (entry 2026-11-05_0355): leaky ReLU 600-603 active, softplus 420-437, tanh and sigmoid do not learn in 2,000 updates. Launcher `rl_maze_suite_spock.slurm` (9dce02e, array 7383494). **A third seed is missing** |
| Optimiser: Adam (wd 1e-6), Adam wd 0, AdamW, SGD (lr 0.03-10, wd 1e-6 and 0) | `NBitFlipFlop_b5_optimizer/EqType=h_N=1000_opt={adam,adamw,sgd}_wd={1e-6,0}_lr=...`, 15 cells, n = 3 each, 200k, S + M (the sgd wd 1e-6 lr 10 cell has 0 runs on M); read at floor, 75k and 200k in `b5_read_cache.pkl` (13 cells; Fig. 3d and the Extended Data figure). Launcher `SilentReLU_b5_optimizer_spock.slurm` (42e8811) with `configs/experiment/b5_optimizer_ff.yaml` | **AdamW and SGD MISSING** (Adam wd 0 is the weight-decay ladder). The B5 launcher is 3BFF-only through its experiment config | **MISSING, needs code**: `run_rl.py` line 283 builds `torch.optim.Adam` with no `trainer.optimizer` switch (`run_experiment.py` lines 388-397 have one for adam, adamw, sgd) |

## B. Per-unit rules (Figure 5 and the rule tables)

| Rule | 3BFF | CDDM | Maze |
|---|---|---|---|
| Rate penalty alone λ_frm (3BFF 0.1; CDDM 0.2; maze 0.03, 0.1, 0.3) | `NBitFlipFlop_remedy_grid/EqType=h_N=1000_arm=frm0.1`, n = 3, 200k, D + M; floor 0/3 (loss still falling), 75k (Fig. 5d); also arms frm0.01 and frm0.03, n = 3 each (the combination test's rungs). Launcher `SilentReLU_remedy_grid_della.slurm`, `~/trainRNNbrain_redo` at 709c17d | λ 0.2: `CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0.2`, n = 5, 75k, D + M; floor 5/5, 75k (Fig. 3a; all 1,000 active, R² 0.004 lower). **λ 0.1 on the standard network: MISSING** (`CDDM_std_g0_penalties/EqType=h_N=1000_pen=frm`, n = 3, 200k, is pre-standard, M only) | `Maze_interventions/frm{0.03,0.1,0.3}`, n = 3 each, S-maze, Spock array 7383696 at 55fab3e; read (issue #5, entry 2026-11-05_0812): 0.03 and 0.3 **work** (792-962 active, success cost at most 0.003), 0.1 fails the success condition (seed 1 ends at 0.63). Launcher `rl_maze_interventions_spock.slurm` |
| Rate + sparsity (λ_frm 0.1 + λ_rws 0.05; CDDM 0.2 + 0.05) | `NBitFlipFlop_ff_both40k_c200000/EqType=h_N=1000_arm=both`, n = 3, D + M; floor 3/3 (3.51×); launcher `SilentReLU_flipflop_frmboth_long_della.slurm`, top-up 709c17d | 0.2 + 0.05: `CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0.05_LmbdFR=0.2`, n = 5, D + M; floor 0/5, 75k (all 1,000 active, R² 0.008 lower). The 3BFF/maze pair 0.1 + 0.05 exists only pre-standard (M) | `Maze_interventions/frm0.1_rws0.05`, n = 3, S-maze, 7383696; fails the success condition (seeds 2 and 3 did not learn; counts 901-946) |
| Noisy synapses σ_w 1.0 | `NBitFlipFlop_synnoise_c200000/EqType=h_k=3_N=1000_sw=1.0`, n = 3, D + M; floor 3/3 (2.03×, R² 0.023 lower); also σ_w 0.1, 0.3 (`synnoise`), 2.0, 3.0 (`synnoise_ext`), n = 3 each. Launchers `SilentReLU_synnoise_della.slurm`, `SilentReLU_synnoise_ext_della.slurm` (launched with 84b4485), top-ups 709c17d | `CDDM_synnoise_ext/EqType=h_N=1000_sw=1.0_iters=100000`, n = 3, standard (bias [0, 0]), 100k, **M only** (indices 13-15 of `SilentReLU_synnoise_ext_della.slurm`; the folder is on neither cluster). **Not read for the paper**: only the old deck's `fig_slides.py` (`slide_26d_synnoise_cddm`) reads it; the catalogue `read_remedies.py` is 3BFF-only | `Maze_interventions/sigmaw1.0`, n = 3, S-maze, Spock array 7383697 at 55fab3e, **running** (updates 1,200-1,300 of 2,000 at 08:30, 24 h cap, about 17 h per run); interim counts 428-467, inside the controls' range |
| Rescaling incoming weights (the paper's arm: target 10× the rate cap, α 1.002, cap 64, protect 0.2 N, check every step) | `NBitFlipFlop_target_ladder_c200000/EqType=h_k=3_N=1000_tgt=10.0`, n = 4, D + M; floor 1/4 (1.51×), 75k; 21 other rescaling settings at N = 1,000 (`norm_mode`, `recgrow`, `rescale_*`, `revive_*`, `target_ladder` 6-30), seven of which lower the count. Launcher `SilentReLU_target_ladder_della.slurm`, top-up 709c17d | **MISSING** (the `CDDM_screen3` arm `scale` is synaptic scaling, a different rule, n = 3 at 30k, no top-up) | `Maze_interventions/rescale`, n = 3, S-maze, array 7383755 at 1033e5c: **all three stopped** by the rollback guard at updates 52-60 (non-finite rollouts at the end of the warm-up); counts against the rule at the paper's values. Two gentler arms submitted 5 Nov (entry 2026-11-05_0832, array 7384816 at 4c65946, running): `rescale_mild` (α 1.0005, cap 8) and `rescale_few` (protect 0.05 N), n = 3 each |
| Read-out dropout (mute, rate 0.2, β 4) | `NBitFlipFlop_dropout_sizes_c200000/EqType=h_k=3_N=1000_pen=none_do=mute_rate=0.20_beta=4`, n = 3, D + M; floor 1/3 (1.81×), 75k; plus the 12-cell ladder `NBitFlipFlop_std_bernoulli_c200000` (rates 0.05-0.25 × β 1, 2, 4), n = 3 each. Launchers `SilentReLU_dropout_sizes_della.slurm`, `SilentReLU_flipflop_dropfix_della.slurm`, top-ups 709c17d | `CDDM_paper_grid/EqType=h_N=1000_arm=mute`, n = 3, standard, 100k, D + M (saved config: kind mute, drop_rate 0.2, β 4, sampling participation). **Not read for the paper** (no script reads this arm; the catalogue is 3BFF-only). Launcher `SilentReLU_paper_grid_della.slurm` with `TASK=cddm` | `Maze_interventions/dropout0.2_beta4`, n = 3, S-maze, 7383696; **fails** both conditions (413-461 active, success 0.76-0.84) |
| Prune-and-duplicate (copy, copy noise 1.0, rate 0.025, maturity 1,000) | `NBitFlipFlop_paper_grid_c200000/EqType=h_N=1000_arm=duplication`, n = 3, D + M; floor 3/3 (2.77×, R² 0.005 lower); plus `copy_perturb` (copy noise 0-3), `replace_variants_fix` (rates 0.001-0.005), n = 3 each. Launcher `SilentReLU_paper_grid_della.slurm`, top-up 709c17d | `CDDM_paper_grid/EqType=h_N=1000_arm=duplication`, n = 3, standard, 100k, D + M (same prune_args as the 3BFF arm). Read only in the old deck (`fig_slides.py` size panel); **not in the paper's catalogue**. Also `CDDM_screen3/..._arm=prune_copy` (no replacement cap), n = 3, 30k, no top-up | **MISSING, needs code**: TrainerPPO refuses `reinit_mode` other than `rescale` (constructor, lines 233-235) |
| Redraw / re-initialise silent units (random; zero_out) | `NBitFlipFlop_replace_variants_c200000/EqType=h_k=3_N=1000_mode={random,zero_out}_rate=0.025_mat=1000`, n = 3 each, D + M; floor 3/3 (Fig. 3e; 0.84-0.96×). Launcher `SilentReLU_replace_variants_della.slurm`, top-up 709c17d | `CDDM_screen3_c75000/EqType=h_N=1000_pen=none_arm=prune` (redraw the incoming weights after `patience` 5 silent checks), n = 3, 75k, D + M; floor 3/3, 75k (Fig. 3e; 330-346). Launcher `SilentReLU_screen3_della.slurm` | **MISSING, needs code** (random and zero_out refused, as above) |
| Threshold lift (bias kick) | `NBitFlipFlop_replace_variants/EqType=h_k=3_N=1000_mode=bias_kick_rate=0.025_mat=1000`, n = 3, 40k screen, D + M; 1.36× at 40k (Supplementary Table 3b) | **MISSING** | **MISSING, needs code** (bias_kick refused) |
| Full dropout (kind dead; parked) | `NBitFlipFlop_dropout_sizes/EqType=h_k=3_N=1000_pen=none_do=dead_rate=0.20_beta=4`, n = 3, 40k, D + M; and the 12-cell `std_bernoulli` dead ladder, n = 3 each, 40k. Parked (904 of 1,000 active at a cost of 12-17 R² points) | **MISSING**; parked | **MISSING, needs code** (kind dead refused); parked |
| Other screened rules (synaptic scaling, synaptogenesis, disinhibition, orthogonal-row and donor-mix replacement) | 3BFF screens at N = 1,000, 40k (`revive_ops2`, `norm_mode`, `replace_variants` orth and mix), n = 3 each; Supplementary Table 3b | synaptic scaling: `CDDM_screen3/..._arm=scale` and the pairs `cap+prune`, `cap+scale`, `prune+scale`, all three, n = 3 each, 30k, no top-up, not in the paper | **MISSING, needs code** |

**What the maze trainer can run today** (`trainRNNbrain/trainer/TrainerPPO.py`, constructor
docstring lines 144-205): the rate penalty (`trainer.lambda_frm`, `frm_args`), the sparsity penalty
(`trainer.lambda_rws`), noisy synapses (`model.sigma_w`), read-out dropout of kind mute
(`trainer.dropout` with the `dropout_args` block of `trainer_ppo.yaml`), and the rescale rule
(`trainer.prune_reinit` with `reinit_mode: rescale`, `rescale_norm_mode: self`, `revive_op: rescale`,
no refractory period). Every other mode of `prune_reinit` (random, zero_out, orth, mix, bias_kick,
copy), dropout of kind dead, the metabolic and recurrent-weight penalties, synaptic scaling, the
input-weight cap and any optimiser but Adam need code before a job can run. Weight decay
(`trainer.weight_decay`), the activity noise (`model.sigma_rec`, `model.sigma_inp`), the input row
norm (`model.input_row_norm`) and the spectral radius (`model.spectral_rad`) are config keys the
launcher can pass today.

## Missing cells and how to make them

Costs: a supervised run at N = 1,000 for 75,000 iterations takes about 5.5 GPU-h (the guide's
0.24-0.30 s per iteration gives 5.0-6.3 h; the B5 launcher's 2.9 h per 40,000 gives 5.4 h; the
DMTS settings array 16628171, 150,000 iterations at N = 1,000, took 6.4-6.9 h per run with DMTS's
shorter trials). A maze run at N = 1,000 takes about 5.5 h on a Spock 40 GB card (noisy synapses
about 17 h). Every cell below is 3 seeds unless it says otherwise; the controls are the existing
N = 1,000 standard networks of each task, read at the same point. New supervised cells use the
75,000-iteration budget of `configs/experiment/silent_units_largeN.yaml` (the same-size read-out
point); a top-up to 200,000 (`SilentReLU_ff_topup200k_della.slurm`) can follow where a floor read is
wanted. The supervised launchers named here are task-specific through a hard-coded `task=` or
folder prefix, so each needs one `TASK` switch of the kind `SilentReLU_paper_grid_della.slurm`
already has (lines 40-45: `ff`, `dmts`, `cddm` set the task arguments, budget and folder prefix);
the override words are the ones the existing cell of the other task was made with. Launchers are
gitignored: `git add -f`, then check the file exists on the cluster at the pinned commit before
`sbatch`, and validate each launcher with a 30-iteration local run at N = 50 first (guide).

### 3BFF (Della, standard network, N = 1,000, 75,000 iterations)

| Cell | Launcher to extend and overrides | sbatch | Runs | GPU-h |
|---|---|---|---|---|
| Weight decay 1e-5, 1e-4 | `b0_wd.slurm`: add `task=NBitFlipFlop task.n_inputs=3 task.n_outputs=3`, folder prefix `NBitFlipFlop_`; its `WDS=(0 1e-5 1e-4)` with `IT=75000`; wd 0 exists (B5) | `ITERS_OVERRIDE=75000 sbatch --array=4-9 slurm/b0_wd.slurm` | 6 | 33 |
| Activity noise σ_rec 0, 0.01, 0.1 | `b0_noise.slurm`: add the 3BFF task words; `SIGS=(0 0.01 0.05 0.1)` with `model.sigma_rec=<v>`, `EQ=h`, `IT=75000`; skip the 0.05 rung (controls exist) | `sbatch --array=1-5,6-10,16-20 slurm/b0_noise.slurm` (its decode has 5 seeds per rung; use 3) | 9 | 50 |
| Spectral radius 0.5, 2.0 | `SilentReLU_std_recinit_della.slurm`: add the 3BFF task words; cells `model.spectral_rad=0.5` and `2.0`; `NSEED=3` | `sbatch --array=1-3,7-9 slurm/SilentReLU_std_recinit_della.slurm` (after `NSEED=3`) | 6 | 33 |
| Recurrent-weight penalty λ_rwm 0.05, 0.2 | same launcher; cells `+trainer.lambda_rwm=0.05 +trainer.rwm_args={}` and `0.2` | `sbatch --array=10-15 slurm/SilentReLU_std_recinit_della.slurm` | 6 | 33 |
| Metabolic penalty λ_met 0.01, 0.1, 1, 10 | `b0_met.slurm`: add the 3BFF task words; `trainer.lambda_met=<v>` with `NS=(1000)` | `sbatch --array=1-12 slurm/b0_met.slurm` | 12 | 66 |
| Sparsity penalty alone λ_rws 0.05 | `SilentReLU_std_penalties_spock.slurm` (arm `rws`: `trainer.lambda_rws=0.05 trainer.lambda_frm=0`) with the 3BFF task words and `IT=75000`; or `SilentReLU_flipflop_frmboth_long_della.slurm` with a `rws` arm | `sbatch --array=1-3 slurm/SilentReLU_std_penalties_spock.slurm` | 3 | 17 |
| Activations on the bias-0 network: leaky ReLU, softplus β 25, shifted sigmoid, tanh | `b0_act.slurm` / `SilentReLU_std_activations_spock.slurm`: add the 3BFF task words and a fourth model `rnn_tanh_standard`; `model=rnn_{leaky_relu,softplus_beta25,sigmoid_shifted,tanh}_standard`, `IT=75000` (the a831629 runs have a trainable bias and cannot be compared with bias-0 controls) | `sbatch --array=1-12 slurm/b0_act.slurm` | 12 | 66 |
| Input-weight cap 0.3 (CDDM's value) | `SilentReLU_screen3_della.slurm` arm `cap` (`model.inp_weight_cap=0.3`) with the 3BFF task words; the existing caps 0.1, 0.5, 2.0 need only a read-out (below) | `sbatch --array=4-6 slurm/SilentReLU_screen3_della.slurm` | 3 | 17 |
| Optimiser, per-unit rules | nothing missing | | | |
| **3BFF total** | | | **57** | **315** |

### CDDM (Della, standard network, N = 1,000, 75,000 iterations)

| Cell | Launcher to extend and overrides | sbatch | Runs | GPU-h |
|---|---|---|---|---|
| Spectral radius 0.5, 1.2, 2.0: third seed | `SilentReLU_std_recinit_della.slurm` as is (`NSEED=2` makes seeds `random`; one more index per cell) | `NSEED=1 sbatch --array=1-3 slurm/SilentReLU_std_recinit_della.slurm` (after making NSEED an env variable) | 3 | 17 |
| λ_rwm 0.05, 0.2: third seed | same | `NSEED=1 sbatch --array=4-5 ...` | 2 | 11 |
| Optimiser: AdamW (wd 1e-6), SGD lr 0.03, 0.1, 0.3, 1 (wd 1e-6) | `SilentReLU_b5_optimizer_spock.slurm`: a copy of `configs/experiment/b5_optimizer_ff.yaml` for CDDM (`task=CDDM`, `max_iter: 75000`); arms 3-7 of its table (`trainer.optimizer=adamw`, `trainer.optimizer=sgd trainer.lr=<v>`); SGD lr 3 and 10 did not learn on 3BFF; Adam wd 0 exists (weight-decay ladder) | `sbatch --array=7-9,10-21 slurm/SilentReLU_b5_optimizer_spock.slurm` (Spock, `--partition=all`) | 15 | 83 |
| Rate penalty λ_frm 0.1 (the 3BFF and maze value) | `SilentReLU_std_penalties_spock.slurm` arm `frm` (`trainer.lambda_frm=0.1 trainer.lambda_rws=0`) on `trainer_ptrack_freshbatch`, `IT=75000`, with bias 0 (the existing cell has a trainable bias) | `sbatch --array=4-6 slurm/SilentReLU_std_penalties_spock.slurm` | 3 | 17 |
| Rate + sparsity 0.1 + 0.05 (the 3BFF and maze pair; optional, 0.2 + 0.05 exists) | same launcher, arm `both` | `sbatch --array=7-9 ...` | 3 | 17 |
| Rescaling, the paper's arm | `SilentReLU_target_ladder_della.slurm` cell `tgt=10.0` with `task=CDDM` in place of `task=NBitFlipFlop` (the overrides are lines 124-133 of `SilentReLU_size_series_della.slurm`: `trainer.prune_reinit=true trainer.prune_args.reinit_mode=rescale trainer.prune_args.rescale_norm_mode=self trainer.prune_args.rescale_target_frac=10.0 trainer.prune_args.rescale_active_only=true trainer.prune_args.rescale_normalize=true trainer.prune_args.rescale_alpha=1.002 trainer.prune_args.rescale_cap=64.0 trainer.prune_args.rescale_protect_frac=0.20 trainer.prune_args.check_every=1 trainer.prune_args.patience=1`), `IT=75000` | `sbatch --array=9-11 slurm/SilentReLU_target_ladder_della.slurm` | 3 | 17 |
| Threshold lift (bias kick) | `SilentReLU_replace_variants_della.slurm` arm 10 (`mode=bias_kick`, rate 0.025, maturity 1,000) with `task=CDDM`, `IT=75000` | `sbatch --array=31-33 slurm/SilentReLU_replace_variants_della.slurm` | 3 | 17 |
| Full dropout | parked; not proposed without a decision issue | | | |
| **CDDM total** | | | **32** (29 without the optional pair) | **179** (162) |

### Maze (Spock, N = 1,000, 2,000 updates; `rl_maze_interventions_spock.slurm`, one `ARM_TAG`/`ARM_OVERRIDES` entry per cell, seeds 1-3 at indices 3a+1 to 3a+3)

Launcher arm only (no code):

| Cell | Arm overrides | sbatch (after adding the arm at index a) | Runs | GPU-h |
|---|---|---|---|---|
| Weight decay 0, 1e-5, 1e-4 | `trainer.weight_decay=0`, `=1e-5`, `=1e-4` (three arms) | `sbatch --array=<3a+1>-<3a+9> slurm/rl_maze_interventions_spock.slurm` | 9 | 50 |
| Activity noise σ_rec 0, 0.01, 0.1 | `model.sigma_rec=0`, `=0.01`, `=0.1` | same form | 9 | 50 |
| Input-weight row norm 0.5, 2, 5, 20 | `model.input_row_norm=0.5`, `=2`, `=5`, `=20` | same form | 12 | 66 |
| Spectral radius 0.5, 2.0 | `model.spectral_rad=0.5`, `=2.0` (the warm-up tamed the full-rate divergence at N = 2,000; radius 2.0 may still trip the rollback guard, which then reports it) | same form | 6 | 33 |
| Sparsity penalty alone λ_rws 0.05 | `trainer.lambda_rws=0.05` | same form | 3 | 17 |
| Activations, third seed (tanh, sigmoid, softplus, leaky ReLU) | `rl_maze_suite_spock.slurm`: four indices added to `TASK_N`/`TASK_ACT`/`TASK_SEED` with seed 3 | `sbatch --array=24-27 --time=12:00:00 slurm/rl_maze_suite_spock.slurm` | 4 | 22 |
| **Maze, launcher only** | | | **43** | **238** |

Code first (TrainerPPO or `run_rl.py`), then the same launcher:

| Cell | Code needed | Runs | GPU-h |
|---|---|---|---|
| Input-weight cap 0.3 | port `Trainer.py` line 1421 (the clamp after each optimiser step) into `TrainerPPO.ppo_update` | 3 | 17 |
| Recurrent-weight penalty λ_rwm 0.05, 0.2 | add the λ_rwm term to the PPO loss as `lambda_frm` is added | 6 | 33 |
| Metabolic penalty λ_met 0.01, 0.1, 1, 10 | add the λ_met term | 12 | 66 |
| Optimiser: AdamW; SGD lr 0.03, 0.1, 0.3, 1 | a `trainer.optimizer` switch in `run_rl.py` as in `run_experiment.py` lines 388-397 | 15 | 83 |
| Prune-and-duplicate | mirror `reinit_mode: copy` (issue #5: "not built, the paper's DMTS result makes it the least transferable") | 3 | 17 |
| Redraw silent units (random, zero_out) | mirror the two modes | 6 | 33 |
| Threshold lift | mirror `bias_kick` | 3 | 17 |
| Full dropout | parked; not proposed | | |
| **Maze, code first** | | **48** | **264** |

### Read-outs that cost no GPU time

- 3BFF input-weight cap 0.1, 0.5, 2.0 (`NBitFlipFlop_std_inpcap`, Spock and Mac): held-out R² and
  active units at 75,000 iterations against the N = 1,000 controls, as `fig_paper_F2_fails.read_fixed`
  does; the runs are 150,000 iterations with kept final weights only, so R² at 75k is an estimate.
- CDDM noisy synapses σ_w 1.0 (`CDDM_synnoise_ext`, Mac only): copy to Della, read at 75k against
  `CDDM_std_g0_bias0_c75000/..._LmbdRWS=0_LmbdFR=0` or the `CDDM_paper_grid` control.
- CDDM read-out dropout and prune-and-duplicate (`CDDM_paper_grid` arms `mute` and `duplication`,
  Della): a `read_remedies.py`-style read at 75k; both are standard networks at the 3BFF arms'
  settings. These two reads settle "works on CDDM?" for two of the five Figure 5 rules at no cost.

### Totals

| | Runs | GPU-h |
|---|---|---|
| 3BFF | 57 | 315 |
| CDDM | 32 | 179 |
| Maze, launcher only | 43 | 238 |
| **Sum, no code needed** | **132** | **732** |
| Maze, code first | 48 | 264 |
| **Sum, everything** | **180** | **996** |

The four settings Pavel named (weight decay, input-weight norm, noisy synapses, dropout) across all
three tasks cost 3BFF weight decay 33 GPU-h, maze weight decay 50 GPU-h and maze input norm
66 GPU-h, 149 GPU-h in all, plus the three CPU read-outs above; noisy synapses on the maze is
running and read-out dropout on the maze is read. Under the 31 Oct rule (`agents-may-submit-jobs`)
a package of up to about 200 GPU-h needs no approval; the full set needs Pavel's go-ahead.

### Open points found while counting

- `CDDM_std_g0_tanh` (the paper's tanh cell) was trained with `same_batch: true`; every other
  standard cell has `same_batch: false`.
- The 3BFF non-ReLU cells exist only with a trainable bias and only on the Mac; the paper's
  activation claim on 3BFF rests on them, pinned to a831629 by their launcher on purpose.
- The CDDM spectral-radius and recurrent-weight-penalty ladders have two seeds each (on the board's
  "Should" list since 31 Oct).
- The maze noisy-synapse arm needs about 17 h per run; a budget above 2,000 updates on the maze
  (issue #3) would raise every maze cost here in proportion.
