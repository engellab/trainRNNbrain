# Agent guide: how this project works

Read this after `CLAUDE.md`. It states how things are done now. Where an older notebook entry in
`docs/project_trajectory/` describes a different rule, this guide and the status board win: the
notebook is a history, and old entries are kept as they were written.

## What the project is

A paper: task-trained RNNs leave most of their units silent, the network does not need them, and
the number of active units falls further, relative to N, with network size and with training.

**Framing (Pavel).** RNNs trained on cognitive tasks serve as model organisms for neuroscience. Scale
them up and most units become computationally useless: ReLU units go silent, tanh units saturate,
and both lose their gradient. tanh links the paper to Dohare et al. 2024 (loss of plasticity).
Losing neuronal participation is not brain-like. It gets worse the longer you train, and it does
not stop. Every way to alleviate it is listed: biologically plausible ones (a rate penalty, "the
urge to spike, but not to overspike"; noisy synapses; rescaling of incoming weights) and engineering
hacks. The conclusion follows Sutton: gradient descent on the task is not enough, so participation
needs a drive of its own, likely several.

**Figures.** F1, the phenomenon. F2, everything that fails (the knob ladders, re-initialising silent
units). F3, a bird's-eye view of what works. Task schematics are supplementary (`fig_supp_tasks.py`,
`fig:Stasks`), not Figure 1.

**Where things are written down:**

| What | Where |
|---|---|
| Commit and notebook rules | `CLAUDE.md` |
| How things are done (this file) | `docs/AGENT_GUIDE.md` |
| Claims C1-C7, status board, to-do list (A* analyses, B* training, W* writing), open decisions | `docs/paper_audit.md` (its status board, at the top, is current) |
| Every experiment: why, how, result | `docs/project_trajectory/`, one file per entry |
| Open rules and commands for running experiments | `docs/project_trajectory/2026-10-15_1342_open-rules-and-commands-moved-from-agent-notes.md` |
| Frozen pre-registrations that launchers copy into sweep folders as `EXPERIMENT.md` | `docs/experiments/` (do not edit or delete; launchers `cp` them) |
| Talk deck | `docs/presentation.md`, built by `fig_slides.py`, `deck_pngs.py`, `deck_html.py` |
| An agent's working notes while its branch is open | `docs/notes/agent-<name>.md`; once merged, the results go into a new notebook entry and the note is deleted |
| What each config parameter does | the comments in the config file itself (`configs/model/rnn_relu_standard.yaml`, `configs/trainer/trainer_ptrack_freshbatch.yaml`, ...) |
| Which script does what | `trainRNNbrain/experiments_and_analysis/README.md` |

**Working with Pavel.** He prefers a picture to an explanation, and short answers. Before any new
training run, tell him what it runs and what it costs (GPU-hours); large runs need his go-ahead.

## Repos and paths

- **Analysis code:** one checkout, `~/Documents/GitHub/trainRNNbrain`, one branch, `main`. Origin is
  the lab repo `engellab/trainRNNbrain`: never force-push. Short-lived worktrees for an agent's
  branch are fine; merge, then delete the branch and the worktree.
- **Paper:** `~/Documents/Projects/Dead_ReLU_paper` → `ptolmachev/dead-relu-paper`, branch `main`.
  `sections/*.tex` hold the text; `main.tex` holds the figure floats and captions; `make pdf` runs
  latexmk; check `main.log` for "Float too large". Never commit `.aux` or `.fls`.
- **Overleaf** follows the paper's `main` only, and syncs by hand (*Menu → GitHub → Pull/Push*).
  Edit in one place at a time and sync before switching. After pushing, tell Pavel to pull in
  Overleaf.
- **Della data:** runs in `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs` (also
  `~/trainRNNbrain/data/trained_RNNs`). Kept snapshots in
  `.../data/c1_snapshots/<prefix>_<tag>/s<seed>/it<iter>/` plus `resume/`. Ablation and retrain
  results in `/scratch/gpfs/TENGEL/pt1290/c1_ablation_seeded`, `c1_retrain_seeded2`. Logs in
  `~/trainRNNbrain_logs/<job>.<id>_<task>.out`.
- Never point `paths.checkpoint_to` inside `save_to`: cells are inventoried by globbing.
- Della scratch is not backed up; home is. TENGEL is an 8 TiB group fileset with ~1.4 TB free.
- **Spock:** home quota 95 G soft / 100 G hard; `/scratch` is not writable; large outputs go to
  `/mnt/cup/labs/tengel/home/tolmach`. Spock holds older sweeps; new training runs on Della.
- **Figures:** working figures go to `img/internal_figures/` and are linked from the notebook entry
  that made them. `img/` is gitignored; add a figure with `git add -f`.

## Environments

- **Mac:** `/opt/miniconda3/envs/rnns` (Python 3.13, torch 2.10, hydra 1.3.2). Runs the tests and
  local smoke runs.
- **Della:** `module load anaconda3/2024.10; source "$(conda info --base)/etc/profile.d/conda.sh";
  conda activate torch-env; export PYTHONPATH=<checkout>`. torch 2.7.0, so a test can pass on the
  Mac and fail on Della (`tests/test_prune_rescale_mode.py` does).
- **Spock:** `module load anacondapy/2024.02`, `torch-env`.

## Submitting on Della

- **Access** is a cached Duo session. If `ssh della` asks for a password, ask Pavel to run
  `ssh -fN della`; an agent cannot authenticate. Prepare everything locally first, so submission is
  one command.
- **Login nodes kill anything heavy** (~200 s of CPU). Run even small analyses through `sbatch`:
  `sbatch --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=01:00:00 --wrap "..."`. For GPU
  work add `--gres=gpu:1 --constraint=nomig --mem-per-gpu=32G`. A 20-30-iteration launcher check on
  the login node with `srun` stubbed is fine.
- **Omit `--partition`;** GPU routing is automatic. The QOS (queue tier) follows `--time` alone: up
  to 1,441 min is gpu-short (44 GPUs per user), up to 4,321 min gpu-medium (20 GPUs, 24 jobs), up to
  8,641 min gpu-long (16 GPUs, 10 jobs).
- **`--constraint=nomig`** keeps a job off the MIG slices (partial A100s), which run three times
  slower. **`--constraint=gpu80`** (A100 80 GB) is required at N = 4000; a 40 GB card runs out of
  memory. H100 nodes are not usable from this account.
- **`nice` cannot be lowered.** A job stuck behind a high `nice` must be cancelled and resubmitted.
- **Slurm copies the batch script at submission:** editing a launcher does not change queued jobs;
  editing the Python it calls does. Chain jobs with `--dependency=afterok:<id>`, not local watcher
  loops.
- **Fonts:** Della's default font makes legends wider than on the Mac. Collect data on Della, draw
  figures on the Mac.
- `rm -rf` over ssh and crontab installs are blocked for agents. Write a script with explicit paths
  and have Pavel run `! ssh della 'bash -s' < script`.
- **Speeds (s/iteration):** N = 500 0.16-0.20; N = 1000 0.24-0.30; N = 2000 0.62-0.75; N = 4000
  2.1-2.6 on gpu80.

**Spock:** requires `--partition=all`; the longest accepted `--time` is 96 h. Spock and Della have no
direct route; relay through the Mac (`slurm/sync_b5_spock_to_della.sh`).

## Launchers, version pinning, gitignore

- **`.gitignore` hides files you expect tracked.** Before naming a new file, run
  `git check-ignore -v <path>`; a match means git will silently skip it. The rules that bite:
  - `/slurm/*.slurm`: add every new launcher with `git add -f`, then check that it exists on the
    cluster at the pinned commit before `sbatch`.
  - `/img/`, `*.png`, `*.svg`, `*.pdf`: figures need `git add -f`.
  - `*_test*`, `*[_-]test*`: a script named `heldout_test.py` or a config `x_test.yaml` is ignored.
    Name tests `tests/test_<thing>.py` (that folder is exempt) and avoid "test" elsewhere.
  - `*.json`, `*.pkl`: ignored everywhere, so a JSON config or cache outside `data/` never reaches git.
- **Launchers record where they ran.** `slurm/b0_*.slurm` produced the `_bias0` knob ladders behind
  Figure 2 (`CDDM_std_g0_{drift,weightdecay,activations,metabolic}_bias0`, `CDDM_std_g0_bias0`,
  `CDDM_77fbd9_g0_noise_ptrack`, `NBitFlipFlop_std_{winp,ksweep}_bias0`); they ran on Della from
  `~/trainRNNbrain_stdfix` (77fbd97 plus uncommitted trainer edits), with `REPO` set there. The
  committed `SilentReLU_paper_grid_della.slurm` hard-codes `task=DMTS_d7`, so it does not regenerate
  `DMTS_paper_grid_d5`.
- **Sweeps run from detached worktrees on Della** (`~/trainRNNbrain_<name>`), pinned to the commit
  the sweep started from. Launchers refuse a dirty tree. To change one knob for an existing sweep,
  branch from the pinned commit plus one config file, check `git diff --stat <pin> HEAD`, and write to
  a new `SWEEP` folder. Never move a pinned worktree that has jobs running or pending.
- **Current pins:** `~/trainRNNbrain_redo` 709c17d (B2, B3, B4, remedy top-ups);
  `~/trainRNNbrain_c1top` 7739a60 (FFtop4000); `~/trainRNNbrain_b2della` 2816d14 (B2 N = 4000).
  The analysis checkout `~/trainRNNbrain_c1ana` is updated with
  `git fetch origin main && git checkout -- . && git checkout --detach FETCH_HEAD`.
- **Validate with a fast local run, never a queued test job.** Run the launcher itself with
  `module`/`conda`/`srun` stubbed at N = 50 for 30 iterations, then diff each arm's saved config
  against the control's. Set the pass condition before the diff.
- **Rebuild a trained network from its own saved config;** the parameter file carries no
  architecture. A task re-instantiated from current code is invalid if the task class changed after
  training.
- **Warm start:** `paths.init_from` (+ `init_adam`) continues a run; its loss is joined along
  `common.warm_start_chain`. `paths.transfer_from` copies W_rec/W_inp/y_init into a new task with a
  fresh W_out and starts its own chain. Top-ups: `SilentReLU_ff_topup200k_della.slurm` (`PARENTS`,
  `TARGET`, `KEEP=1`). The rescale rule's internal state is not saved, so every warm start resets it.
- **Snapshots:** every 1000 iterations, kept (`checkpoint_every: 1000`, `checkpoint_keep: true`),
  W_rec in float16, compressed with `np.savez_compressed`; an exact float32 copy with Adam state in
  `resume/`. float16 changes R² by less than 1e-5 and leaves the active set identical.
- **Reproducibility:** flip-flop, hyper flip-flop and DMTS task configs use `seed: ${seed}`, so a
  rerun is bitwise identical, as CDDM's always was. Runs made at commits before 9f10955 (including
  the 709c17d and 7739a60 pins) cannot be regenerated exactly on those tasks.

## Reading networks

- **Standard network:** ReLU, equation `h`, bias fixed at 0 (`bias_range: [0, 0]`), self-connections
  on, no Dale's law, no input/output positivity, γ = 0, weight decay 1e-6, lr = 1e-3·(100/N)^(1/3),
  Adam, `trainer_ptrack_freshbatch` (`same_batch: False`). "Pre-standard" means a trainable bias in
  [-1, 1].
- **Which read-out (Pavel, 2026-10-16):** a comparison ACROSS network sizes reads each network at
  its loss floor (the read-out point below), because the learning rate scales as N^(-1/3) and larger
  networks converge later. Networks of the SAME size (knob ladders, optimizers, interventions against
  their control) are compared at one fixed iteration, 75,000 (`fig_cosyne2027.read_fixed`); where no
  weights were kept there, R² at 75k is estimated from the end of the run by
  `common.r2_at_floor_estimate`, and a run that stops earlier is read at its end and labelled so.
- **Read-out point (since 2026-10-15):** `common.noisy_floor(*common.noisy_loss_series(run))` on the
  noisy training loss (`TrainLosses.json`, joined along the warm-start chain). The curve is a Hampel
  mean over [t/1.1, 1.1·t] (points more than 3 MADs from the window median dropped), with burst
  iterations masked out first; the stretched-exponential floor is fitted to that same curve. The
  read-out `t_star` is the first t after which the curve stays at or below 1.05× the floor through
  1.2·t (a sustained crossing). `t_stars` also gives 1.10× and 1.20×; 1.05× is the tightest margin
  used, and a comparison should be checked at the looser ones. If the fit's β sits on its upper bound
  (3), the floor is the curve's minimum (`floor_kind="smoothed_min"`). DMTS is fitted from its escape.
  A run is excluded, and counted, if its floor is on the lower bound, if the curve never stays under
  the bar (including a crossing too late for its 1.2·t hold), if the curve's median over its last 10%
  ends more than 1.5× above the floor, or if bursts cover more than half of training. Never on the noise-free
  loss. The previous rule (2,001-iteration running median,
  first crossing) read 2-19% early on synthetic traces with a known answer. Checked against the
  loss of kept snapshots on fresh batches: t* agrees within 3% on the flip-flop; on CDDM the loss
  nears its floor so slowly that t* is uncertain by about ±30% under any rule, so compare CDDM at
  a matched loss level or across all three margins.
- **Bursts** (the loss climbing above 2× its local level and coming back) are returned in
  `noisy_floor(...)["bursts"]` and tabulated per cell by `burst_stats.py` (`data/burst_stats.csv`):
  report them as a property of the training method, size or hyperparameter.
- A size series is drawn at the floor only if every size has a read-out; otherwise draw it at the end
  of training and label it so.
- A penalised run's noisy loss includes λ·penalty, so its floor is task plus penalty.
- **Active unit:** participation p = std(r) + q0.9(|r|) on a noise-free pass; active if
  p ≥ 0.05·q95(p). Health check: q50/q95 below 0.05. Report active counts, never silent fractions.
- **Performance:** held-out R² under the training noise, from `heldout_r2.py`, on CPU
  (`CUDA_VISIBLE_DEVICES=`). Seeds: held-out batch `HELDOUT_SEED = 20261012`, noise draw d
  `NOISE_SEED + d`, ordering batch `ORDER_SEED = 20261013`, fine-tune batches
  `FINETUNE_SEED = 20261014`; cache key ends `|seeded-v1`. Figures use 36 draws. Where no snapshot
  exists at the floor, R² there is estimated as 1 − (L(t*)/L_end)(1 − R²_end).
- **Budgets:** 3-bit flip-flop 200k iterations (its noisy loss still falls at 75k); CDDM converges by
  ~25-32k; DMTS converges within ~15k of escaping a 155-200k-iteration stall. One DMTS N = 2000 seed
  never learns and is excluded at r² < 0.8.
- The flip-flop's noise-free loss rises again after ~10k iterations, because the networks use their
  noise. That is why the read-out uses the noisy loss.
- Save and restore `rnn.random_generator` around any seeded scoring done during training.
- List the sweep directories and read their configs before every figure; never reuse a cell list.

**Falsified, do not re-derive:** "silence is caused by ReLU scale symmetry". Sigmoid networks silence
as many units.

## Figures

- Manuscript figures: `paperstyle.save()` writes PDF + SVG, no PNG; `pdfimages -list` must show no
  rasters. Do not set `mathtext.fontset: custom`. Preview on the Mac with `qlmanage -t -s 2000`
  (`sips` renders a black background).
- Set GridSpec margins explicitly and check overlaps with `get_tightbbox`, not by eye.
- **Syncing a figure to the paper is two commits.** Commit trainRNNbrain first (`sync_figures.sh`
  records its `git rev-parse HEAD`), then in the paper repo `bash sync_figures.sh fig.pdf fig.svg`,
  update the caption, build, commit and push.
- **Figure 2:** on Della, `fig_paper_F2_fails.py --collect` (CPU, ~18 min) writes
  `data/fig_paper_F2_fails_cache.pkl`; copy it back and draw on the Mac.
- **Figure 1:** `fig_paper_F1.py`, run on Della from `~/trainRNNbrain_c1ana`.
