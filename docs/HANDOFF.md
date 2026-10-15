# Hand-off: finishing the silent-units paper

*Written 13 October 2026, 13:20 EDT, when Pavel switched laptops. Read this first; it points to
everything else.*

## Prompt for the next agent

> You are picking up a research project that is being turned into a paper: task-trained RNNs leave
> most of their units silent, the network does not need them, and the silent share grows with
> network size and with training. Work methodically toward submission.
>
> 1. Read, in order: `CLAUDE.md` (the commit rules: several agents share this tree), this file
>    (`docs/HANDOFF.md`), the status board and plan in `docs/paper_audit.md`, and the newest
>    entries in `docs/project_trajectory/` (one file per entry, sorted by date and time).
> 2. If this is a fresh laptop, restore it with `~/laptop_backup_2026-10-13/README.txt` on Della
>    (`ssh della`). If `ssh della` asks for a password, ask Pavel to run `ssh -fN della`; you cannot
>    authenticate yourself.
> 3. Check the two running jobs (§4). When the flip-flop N = 4000 top-ups have finished, do §5
>    step 1: Figure 1 panel c on the flip-flop's floor, then sync it to the paper.
> 4. Then work down §5. Before any new training run, tell Pavel what you will run and what it
>    costs.
>
> Read networks only at 1.05× the floor of their noisy training loss, never at a fixed iteration
> or on the noise-free loss. Commit only files you changed, by name. Record every result as a new
> file in `docs/project_trajectory/`, and update the status board in `docs/paper_audit.md` as items
> close. Pavel prefers a picture to an explanation, and short answers with few caveats.

## 1. What the paper says, and where the plan lives

**Framing** (Pavel, round 1). RNNs trained on cognitive tasks are used as model organisms for
neuroscience. Scale them up and most units become computationally useless: ReLU units go silent,
tanh units saturate, and what unites the two is the loss of gradient. tanh only links the paper to
Dohare et al. 2024 (loss of plasticity). Losing neuronal participation is not brain-like. It gets
worse the longer you train, and it does not stop. Every way to alleviate it is listed: biologically
plausible ones (a rate penalty, the "urge to spike, but not to overspike"; noisy synapses;
rescaling of incoming weights) and engineering hacks. The conclusion follows Sutton: gradient
descent on the task is not enough, so participation needs a drive of its own, likely several.

**The plan:** `docs/paper_audit.md`.
- Its *Status board* (top) is current; §1–§8 below it are the round-1 audit.
- §2 holds claims C1–C7, §5 the remedies catalogue, §7 the to-do list (A* analyses, B* new
  training, W* writing), §8 the open decisions.

**Figures (Pavel, 12 Oct):**
- **F1:** the phenomenon. Done.
- **F2:** everything that fails (the knob ladders, re-initialising silent units, and so on). To build.
- **F3:** a bird's-eye view of what worked. To build.

## 2. Where everything is

| Thing | Location |
|---|---|
| Analysis code | GitHub `engellab/trainRNNbrain`, **branch `main`** (the only branch since 15 Oct). It is a shared lab repo: push branches, never force. Locally `~/Documents/GitHub/trainRNNbrain` |
| Paper | GitHub `ptolmachev/dead-relu-paper`, branch `main`. Overleaf syncs it: *Menu → GitHub → Pull*. Locally `~/Documents/GitHub/dead-relu-paper`, next to the analysis repo (`sync_figures.sh` expects that) |
| Trained networks | Della `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs`, also reachable as `~/trainRNNbrain/data/trained_RNNs`. Spock `~/trainRNNbrain/data/trained_RNNs` holds older, pre-standard sweeps |
| Kept weight snapshots (claim-1 runs) | Della `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots` |
| Ablation results (one JSON per network and read-out) | Della `/scratch/gpfs/TENGEL/pt1290/c1_ablation`; lists in `c1_ablation/lists`; logs in `c1_ablation_logs` |
| Prune-and-retrain results | Della `/scratch/gpfs/TENGEL/pt1290/c1_retrain`. `c1_retrain_bug_noise_reseed` holds the invalid pre-fix results |
| Fine-tune investigation (the RNG bug) | Della `/scratch/gpfs/TENGEL/pt1290/c1_finetune_probe` |
| Laptop backup | Della `~/laptop_backup_2026-10-13` (home is backed up; scratch is not) |
| Training logs | Della `~/trainRNNbrain_logs/<jobname>.<id>_<task>.out` |

**Della checkouts:**

| Checkout | State | Rule |
|---|---|---|
| `~/trainRNNbrain` | another agent's checkout (`dimensionality-analysis`, dirty) | Do not touch |
| `~/trainRNNbrain_c1abl` | `b524ac7`, clean; trained the claim-1 snapshot runs | Leave clean |
| `~/trainRNNbrain_c1top` | `7739a60`, clean; the 200k top-ups run from it | **Must stay clean**: the launcher refuses a dirty tree |
| `~/trainRNNbrain_c1ana` | analysis checkout, detached at `7608099`; holds a `data` symlink to the scratch data, uncommitted copies of `fig_paper_F1.py` equal to the committed `e68a591`, and a scratch `_panel_c_numbers.py` | To update: `git fetch origin main && git checkout -- . && git checkout --detach FETCH_HEAD`. Della can fetch from GitHub; the git bundles used before are no longer needed |

**Compute:**
- Everything runs on Della; this laptop has no Python environment with numpy or torch.
- Environment: `module load anaconda3/2024.10; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch-env; export PYTHONPATH=<checkout>`.
- **Login nodes kill anything heavy (about 200 s of CPU).** Submit even small builds as jobs. CPU builds: `sbatch --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=01:00:00 --wrap "..."`. GPU work: add `--gres=gpu:1 --constraint=nomig --mem-per-gpu=32G`.

## 3. Rules that are not up for re-derivation

- **Standard network:** ReLU, equation `h`, bias fixed at 0, self-connections on, no Dale, no I/O
  positivity, γ = 0, weight decay 1e-6, lr = 1e-3·(100/N)^(1/3).
- **Read-out:** where a network's NOISY training loss first reaches 1.05× its own fitted floor
  (`common.noisy_floor` on `common.noisy_loss_series`).
  - The series is the per-iteration training loss (`TrainLosses.json`), joined along the
    warm-start chain, with a running median over 2,001 iterations.
  - The floor is fitted from the *escape* for a task that stalls (DMTS).
  - A run is excluded and counted when its fit sits on the bound, when it never reaches the bar,
    or when the median of its last 10% ends more than 1.5× above its floor.
  - A size series is read at the floor only if every size has a read-out; otherwise it is drawn
    at the end of training and labelled as such.
- **Never the noise-free loss.** Never fixed iterations as the headline comparison.
- **Performance** is held-out r² under the training noise (σ_w = 0), averaged over 9 draws
  (`heldout_r2.py`; the ablation code scores the same way).
- **Active unit:** p = std(r) + q0.9(|r|) on a noise-free pass, and active if p ≥ 0.05·q95(p).
- **Budgets:** the 3-bit flip-flop needs 200k (its noisy loss is still falling at 75k). CDDM
  converges by about 25–32k, inside its 100k. DMTS converges within about 15k of its escape, and
  its escape comes later with size.
- **Learning-rate confound** in the size series: left in, and handled by the floor read-out.
- **Commits:** follow `CLAUDE.md`. Name every path, read every diff, and never stage files you
  did not write. Two untracked files belong to other agents and are left alone:
  `docs/presentation_backup.md` and `trainRNNbrain/experiments_and_analysis/ff75k_atfloor.py`.

## 4. Done, and running now

**Done** (details: trajectory entries 2026-10-12 and 2026-10-13):
- **The round-1 audit and plan:** `docs/paper_audit.md`.
- **Claim 1, closed on standard networks:**
  - Ablation of 36 networks at the end of training, N = 500–4000, and of 9 claim-1 networks at
    their floor, N = 1000.
  - Prune-and-retrain at the end of training and at the floor.
  - Every task does without its silent units. CDDM's silent units do nothing at all. The
    flip-flop's, and DMTS's just after its escape, are wired in, but a short retrain without them
    restores the full score; random pruning does not.
- **Figure 1** (`fig_paper_F1.py`):
  - (a) the example network; (b) participation untrained (945 of 1000 active) and after 200k (183);
  - (c) size, with CDDM N^0.46 and DMTS N^0.51 at the floor, and the flip-flop still at the end of
    75k;
  - (d) the ablation with the retrain stars; (e) silencing after the floor.
  - Plus a supplementary ablation figure (`fig_supp_ablation.py`).
- **The noisy-floor read-out:** `common.py`.
- **Kept weight snapshots:** `+trainer.checkpoint_keep=true` in `run_experiment.py`.
- **200k top-ups** of the flip-flop controls (N = 500, 1000, 2000) and of the claim-1 runs.
- **The paper repo:**
  - Figure 1 and its caption.
  - Claim 1 in the results, Methods (Ablation; Prune and retrain; the read-out), the abstract and
    the intro.
  - The supplementary ablation figure.
  - Every stale Fig. 1d and old-exponent reference removed. The knob paragraphs keep their own
    subsection (`sec:knobs`) and are flagged as pre-standard.

**Running on Della:**

| Job | What | ETA | Output |
|---|---|---|---|
| `FFtop2000` 15431474 (×3) | flip-flop N = 2000 controls, 75k → 200k | finishing about 13:30 on 13 Oct | `trained_RNNs/NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=2000_pen=none_arm=none/` |
| `FFtop4000` 15431475 (×3) | flip-flop N = 4000 controls, 75k → 200k (5-day wall limit) | about 15:30 on 15 Oct | `trained_RNNs/NBitFlipFlop_paper_grid_c200000/EqType=h_N=4000_arm=control/` |

Check them with `squeue -u pt1290`, and progress with `tail -c 300 ~/trainRNNbrain_logs/FFtop4000.15431475_1.out`.

## 5. Next steps, in order

1. **Figure 1 panel c on the flip-flop's floor** (when `FFtop4000` is done):
   - Check that every N has at least one noisy-floor read-out. N = 500: 6 of 6 at 200k. N = 1000:
     2 of 3; the third ends 7% above its floor.
   - Point the flip-flop entries of `SCALING` in `fig_paper_F1.py` at the c200000 cells:
     `NBitFlipFlop_ff_revive_c200000/...N={500,1000,2000}...` and
     `NBitFlipFlop_paper_grid_c200000/EqType=h_N=4000_arm=control`.
   - Rebuild on Della (`python trainRNNbrain/experiments_and_analysis/fig_paper_F1.py` from
     `~/trainRNNbrain_c1ana`), then copy the PDF/SVG to `img/internal_figures/`. Render a PNG on
     the Mac with `qlmanage -t -s 2000`; `sips` gives a black background.
   - Sync: `bash sync_figures.sh fig_paper_F1.pdf fig_paper_F1.svg` in the paper repo.
   - Update in the paper: the flip-flop sentence in `sections/02_problem.tex` (and its
     `% PENDING` comment), the exponent range in `01_intro.tex`, and the panel c sentence of the
     Figure 1 caption in `main.tex`. Commit, push, and tell Pavel to pull in Overleaf.
   - Ask Pavel whether to continue the N = 1000 straggler to 300k.
2. **Figure 2, what fails (C5):**
   - Move `fig_knobs_at_floor.py` to `common.noisy_floor` at 1.05×. It currently reads the
     noise-free loss at 1.07×, and so does `pr_matrix.EXCESS_DELTA`.
   - Data: weight decay (`CDDM_std_g0_weightdecay_bias0`, with `CDDM_std_g0_drift_bias0` as the
     1e-6 reference), activations (`CDDM_std_g0_activations_bias0`), recurrent noise
     (`CDDM_77fbd9_g0_noise_ptrack`), weight cap and spectral radius (`CDDM_std_g0_recinit`),
     metabolic (`CDDM_std_g0_metabolic_bias0`, plus its unread `_c75000` top-up), input scale
     (`NBitFlipFlop_std_winp_bias0`), and the re-initialisation and bias-kick screens.
   - The old knob panel is `panel_knobs` in `fig_paper_F1.py`.
   - Then replace the flagged `sec:knobs` text in the paper.
3. **Figure 3, what worked (C6):**
   - Every flip-flop remedy run stopped at 75k, before convergence. Top them up to 200k with
     `SilentReLU_ff_topup200k_della.slurm` (set `PARENTS`). That is costly at N = 4000 (about
     80 GPU-h per run), so ask Pavel first.
   - Run B2 (the rate penalty alone on the standard flip-flop grid).
   - Run A3 (ablate the recruited units, with `ablate_silent_units.py`).
   - The remedy cache code is `f2_remedies_cache.py`, and the old builders are
     `fig_paper_F2.py` and `fig_paper_F3.py`.
4. **The remaining must-dos:**
   - **A2, gradient reach (C4):** `gradient_participation.py` and `gradient_sensitivity.py`.
   - **B5:** the AdamW/SGD optimiser control.
   - **B3:** the combination test of the rate penalty, noisy synapses and rescaling.
   - **B4:** the plasticity test.
   - **A4, A5:** see `docs/paper_audit.md` §7.
5. **Writing:**
   - **W1:** rewrite the abstract, intro and discussion to the framing above. Claim 1 is already
     written.
   - **W2:** pin "not brain-like" to measured properties.
   - **W3:** criteria for bio-inspired versus engineering.
   - Fill the `% NUMBER NEEDED` notes in the paper repo, among them the Otsu-criterion agreement
     on standard networks.

**Open decisions for Pavel:** decisions 2–5 in `docs/paper_audit.md` §8 (rate penalty alone or
paired; the conclusion's wording; which tasks; whether the plasticity test is in this paper),
the N = 1000 straggler, and the compute for the remedy top-ups.

## 6. Gotchas learned the hard way

- **Slurm stores a copy of the batch script when a job is submitted.** A queued job runs the
  launcher as it was then, not as it is now. Changing the Python it calls is fine; changing the
  launcher's own logic is not.
- **The fine-tune loop once reseeded the generator its training noise comes from**, so training
  replayed the same noise and CDDM memorised it. Save and restore `rnn.random_generator`'s state
  around any seeded scoring done during training.
- **Rebuild a trained network from its saved config** and copy only the learned tensors. The
  parameter file carries no architecture; `set_params` alone silently builds the wrong network
  (see the trajectory, 2026-09-24).
- **The flip-flop's noise-free loss climbs back after its minimum** because the networks rely on
  their noise. That is why only the noisy loss is used.
- **DMTS stalls at the constant-output solution,** for 200 to 155,000 iterations, then escapes,
  and afterwards has brief bursts back toward chance. One N = 2000 seed never learns (r² ≈ 0.42)
  and is excluded at r² < 0.8.
- **The SSH connection to Della drops** when the laptop sleeps. Ask Pavel to run `ssh -fN della`.
  Prefer Slurm dependencies (`--dependency=afterok:<id>`) to local watcher loops for chaining jobs.
- **Della scratch is not backed up.** Keep anything irreplaceable in home or in git.

## 7. File index

**Code** (`trainRNNbrain/experiments_and_analysis/` unless noted):

| File | What |
|---|---|
| `common.py` | `participation`, `noisy_loss_series`, `noisy_floor`, `warm_start_chain`, `logbin`, `stretched` |
| `ablate_silent_units.py` | ablation (`end`, `floor`), `retrain [--at floor]`, `summary` |
| `fig_paper_F1.py` | Figure 1; also `panel_knobs`, the old knob panel kept for F2 |
| `fig_supp_ablation.py` | supplementary ablation figure |
| `heldout_r2.py` | the performance measure |
| `gradient_participation.py` | `initial_params`, the untrained twin; also gradient reach |
| `pr_matrix.py`, `fig_knobs_at_floor.py` | older floor code, still on the noise-free loss at 1.07× |
| `f2_remedies_cache.py`, `fig_paper_F2.py` … `fig_paper_F5.py` | old figure builders, to be rebuilt |
| `fig_slides.py` | the talk deck (`docs/presentation.md`); imports `fig_paper_F1` |
| `trainRNNbrain/training/run_experiment.py` | training; warm start (`paths.init_from`, `init_adam`); `checkpoint_every` and `checkpoint_keep` |
| `trainRNNbrain/trainer/Trainer.py` | the training loop |
| `tests/test_periodic_checkpoint.py` | checkpoint contracts, including the kept-snapshot series |

**Launchers** (`slurm/`; force-added to git, since `slurm/*.slurm` is gitignored):

| File | What |
|---|---|
| `SilentReLU_c1_snapshots_della.slurm` | standard controls with kept snapshots |
| `SilentReLU_c1_ablation_della.slurm` | ablation and retrain, from a `LIST` or a `SNAPSHOT=<Task>` discovery |
| `SilentReLU_ff_topup200k_della.slurm` | continue runs to 200k from their own config (`PARENTS` glob; `KEEP=1` keeps snapshots) |
| `SilentReLU_paper_grid_della.slurm` | the original paper-grid launcher |

**Docs:**

| File | What |
|---|---|
| `docs/HANDOFF.md` | this file |
| `docs/paper_audit.md` | plan and status board |
| `docs/project_trajectory/` | the full chronological record |
| `docs/presentation.md` | talk deck |
| `docs/paper.md` | old paper plan, superseded by the audit |
| `docs/paper_repo_setup.md` | how the paper repo and Overleaf sync were set up |

**Figures** (`img/internal_figures/`, gitignored; backed up on Della):
- `fig_paper_F1.{pdf,svg,png}`, `fig_supp_ablation.{pdf,svg,png}`.
- Diagnostic plots: `diag_loss_floors_standard.png` (noise-free vs noisy loss),
  `diag_noisy_loss_budget.png` (where each run reaches its floor),
  `diag_dmts_noisy_floor.png` (DMTS bursts and the guard).

**Paper repo:** `main.tex` (Figure 1 at about line 98), `sections/00_abstract` … `08_methods`,
`S1_supplementary.tex`, `figures/` (with `SOURCES.md` provenance), `sync_figures.sh`.
