## 2026-10-16 11:51 — One loss-trajectory figure per network

Pavel asked to see every network's loss trajectory and its fit, with the run's path, task and
non-standard hyperparameters beside it, so he can judge the read-out himself.

**Where.** `img/internal_figures/loss_trajectories/<sweep>/<cell>__s<seed>_<hash>.png`, 1,149
figures in 41 sweep folders (140 MB). `img/internal_figures/loss_trajectories/index.md` lists them
all with sweep, cell, seed, task, N, status, reason and non-standard settings.

**Which networks.** The same 1,149 as `2026-10-16_1122_topup-census.md`: every sweep a paper,
supplementary or COSYNE figure script names, plus its continuations, last run of each warm-start
chain. Della runs carry their Della path; Spock-era runs carry their Mac path, the only copy left.

**What each figure shows.**
- The raw noisy loss: its 5-95% band and median, in 600 log-spaced bins.
- The read-out curve (Hampel mean, bursts masked).
- The fitted stretched exponential, extrapolated to 3x the end of training.
- The floor; the 1.05x, 1.10x and 1.20x bars with their read-outs; bursts (orange); warm-start joins
  (purple); the escape from the starting level (blue); the end of training.
- The status under the top-up rule, with the top-up target where there is one.
- "Non-standard": every model, trainer and task setting that differs from the task's standard
  control config, ignoring bookkeeping keys and the settings of switched-off features. The
  references are CDDM_paper_grid control N=1000, DMTS_paper_grid control N=1000,
  NBitFlipFlop_c1_snapshots control N=1000, and NBitFlipFlopHyper_std_hyper k=2 N=1000 pen=none.

**How.** `trainRNNbrain/experiments_and_analysis/run_trajectories.py` (new, uncommitted):
- `extract` ran on Della (job 15691535, 22 s) for the Della runs, and on the Mac for the rest.
- `draw` ran on the Mac.

**First thing the figures showed.** The DMTS paper-grid mute and duplication arms are not warm-started.
They trained from scratch and never learned. That corrects the census entry, in place.
