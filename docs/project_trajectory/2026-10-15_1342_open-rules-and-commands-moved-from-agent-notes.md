## 2026-10-15 13:42 — Open rules and commands, moved here from docs/notes/

`docs/notes/agent-*.md` were deleted in the repository cleanup. Their results already sit in
earlier entries and on the status board. What follows was still open, and it existed only in those
notes. The rules were all written before any of their results existed.

### B3 pilot: choosing λ (from agent-b23 §2)

Pilot jobs on Della: 15598507 (`frm0.01`) and 15598508 (`frm0.03`).

1. Read each pilot run where its noisy training loss first reaches 1.05× its own fitted floor
   (`common.noisy_floor` on `common.noisy_loss_series`, with the usual exclusions). Active units
   come from the participation trace at that crossing (noise-free, 5%-of-q95 rule). The noisy loss
   of a penalised run includes the penalty; read it that way.
2. A rung's multiplier = its mean active units at the floor ÷ the mean active units at the floor of
   the N = 1000 control, `NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control` (Della),
   read the same way. Its active fraction = its mean active units ÷ 1000.
3. Among the rungs whose active fraction is at or below 0.80, take the one whose multiplier is
   closest to 2. If both are equally close, take 0.01, which leaves more headroom.
4. If both rungs exceed 0.80, add a rung at 0.003 (3 runs, ~50 GPU-h). That rung goes past B3's
   budget, so it waits for Pavel.
5. If both multipliers fall below 1.3, use 0.03 anyway and say so.
6. A rung where fewer than 2 of 3 runs reach a read-out cannot be chosen on its numbers. If neither
   rung reaches a read-out, the choice goes to Pavel.

### B3 factorial: the three rate cells, once λ is chosen

On Della, from the pinned worktree `~/trainRNNbrain_redo` (709c17d). Do not move that worktree to
another commit: the cells must match the rest of B3. At 709c17d the flip-flop task seed is not tied
to the run seed, so these runs, like the rest of B3, cannot be regenerated bit for bit.

```
R=$HOME/trainRNNbrain_redo; cd $R; export REPO=$R; L=slurm/SilentReLU_remedy_grid_della.slurm; LAM=<chosen>
M="--chdir=$R --export=ALL --job-name=B3fact --constraint=nomig --mem-per-gpu=32G --time=1-12:00:00 --array=1-3"
LAMBDA_FRM=$LAM SIGMA_W=1.0           N=1000 sbatch $M $L
LAMBDA_FRM=$LAM RESCALE=1             N=1000 sbatch $M $L
LAMBDA_FRM=$LAM SIGMA_W=1.0 RESCALE=1 N=1000 sbatch $M $L
```

The factorial's "none" cell is the claim-1 control above; its "rate" cell is the chosen pilot rung.
Outputs: `~/trainRNNbrain/data/trained_RNNs/NBitFlipFlop_remedy_grid/EqType=h_N=<N>_arm=<arm>/`.
A run that times out continues from `.../s<seed>/resume/` with `paths.init_from`.

**B3 verdict thresholds** (from agent-bplan):
- *Supports "several sources"* if the all-three cell (or any pair) beats its best single component by
  more than 3 SD of that component on active units at the floor, with an R² cost no more than 0.01
  worse than that component's, at N = 1000. A pair or triple that matches the best single remedy on
  units but trains stably where that remedy does not (fewer than 2 of 3 runs) also counts.
- *Falsifies "several sources"* if every pair and the triple sit within 3 SD of their best single
  component on active units and on participation ratio, with no stability gain. The paper then says
  "a drive of its own".
- Effects are estimated on the logit of the active fraction, so a cell near N does not look
  sub-additive by ceiling alone.
- Stage 2 (all three at N = 4000, ~810 GPU-h) runs only if stage 1 shows an additive effect, and
  only with Pavel's go-ahead.

### B4 plasticity transfer: read-out and verdict (from agent-b4 §5)

Jobs on Della: 15598512-14. B4 at N = 4000 on gpu80 is approved and not yet submitted; its source is
`NBitFlipFlop_paper_grid_c200000/EqType=h_N=4000_arm=control` (the FFtop4000 runs).

Read every run at its own noisy-loss floor: `common.noisy_floor(common.noisy_loss_series(run))`,
the run's chain being the run alone. A run with no valid read-out is excluded and counted.

- S₀ = units silent at iteration 0 (`participation[0]`, noise-free, 5%-of-q95 rule). A* = units
  active in the stored participation vector at the first stored iteration at or after the crossing.
  Revived = |S₀ ∩ A*| out of |S₀|.
- Also per run: newly silenced (active at 0, silent at the floor); active count at the floor against
  fresh; held-out R² at the floor (first kept snapshot after the crossing); |S₀ ∩ A(200k)|.
- **Gate.** Informative at N = 1000 only if the fresh networks' mean active count at their floor
  exceeds the old networks' mean count at transfer by more than 3 fresh-network SDs.
- **Silent units do not come back** (supports the claim) if fewer than 10% of S₀ revive in every old
  seed, AND the old transferred networks end more than 3 fresh SDs below the fresh networks' count.
- **Silent units can be recruited** (falsifies the claim) if 30% or more of S₀ revive in at least 2
  of 3 old seeds, OR the old transferred networks' count at the floor comes within 1 fresh SD of the
  fresh networks'.
- Between the two bars: numbers, no verdict. A verdict needs at least 2 of 3 old and 2 of 3 fresh
  runs with a floor read-out; otherwise report at 200k, labelled end of training, without a verdict.
- Young/old pairing is by network: young s1542248578 ↔ old s2461724787; young s2851654641 ↔ old
  s4113307577; young s2670297180 ↔ old s1258717025. Young runs are a comparison with old, no
  separate verdict.
- Known risk: `noisy_floor` needs the smoothed loss to fall below half its level at iterations
  100-400. A transfer run that learns the single-bit outputs in its first 100 iterations may be
  rejected as "never leaves its starting level". Report it; do not patch.
- Zero-cost baseline, no new training: on the old task, how many units silent at the young N = 1000
  networks' floor become active again by 200k (claim-1 chains' traces).

### B2 at N = 4000

Della job 15527679 from `~/trainRNNbrain_b2della` (2816d14). If it times out on a slow node,
continue it with `paths.init_from` from its last kept snapshot.

### Figure 1 rebuild

When FFtop4000 lands, set `RETRAIN_DIR` in `fig_paper_F1.py` to
`/scratch/gpfs/TENGEL/pt1290/c1_retrain_seeded2` (seeded fine-tune batches; flip-flop silent
pruned 0.948, random pruned 0.921-0.929, intact 0.951). The paper branch `numbers-seeded` holds the
updated text numbers; Figure 1's caption "272" (panel e) differs from the text's 277 until Figure 1
is synced.

### Waiting on Pavel

- **C4 wording, CDDM:** "About half of the silent units (271 to 322 of 1000) never cross threshold on
  the task and receive exactly zero gradient on their incoming weights; the other silent units fire
  briefly and are reached."
- **C4 wording, flip-flop:** "Silence is not loss of gradient. Every unit crosses threshold; the units
  the participation rule calls silent fire for a few steps after nearly every input pulse, and their
  incoming weights receive gradient and keep changing as much as active units' do. Training keeps
  them silent anyway: over the next 5000 iterations 2 to 4% become active, and for about three
  quarters of them the drive at the moments they fire falls." Add, for claim 1, that "silent" on the
  flip-flop means "above threshold for under 10% of the time".
- **C4, non-convergence ruled out:** "The gradient that reaches silent flip-flop units is not a sign
  of unfinished training. From 150k iterations on, two of three flip-flop networks' noisy loss
  changes by less than 1% per 100k iterations (CDDM's still falls by 3% per 100k at 100k), the silent
  units' incoming weights move back and forth rather than travel, as on CDDM, and the gradient
  reaches 150-175 silent units from 50k to 300k iterations and after a ten-fold learning-rate drop,
  while on CDDM that number falls from about 40 to under 15. The flip-flop differs in its gradient,
  not its convergence: at every point read, most of the per-batch gradient is shared by all batches
  (83-99% against 22-69% on CDDM), yet training does not follow it."
- **N = 1000 flip-flop straggler s3374117298:** its smoothed loss sits at ~1.055× its floor from 125k
  to 175k and rises to 1.09× at 199k. Continuing to 300k costs ~7 GPU-h, and a crossing is not
  assured. Not submitted.
- **Paper branches:** merge `numbers-seeded`, `figure2-methods` and `w1-draft` into the paper's
  `main`. The first two merge cleanly; `w1-draft` conflicts only in `01_intro.tex` (keep W1's).

### Loose ends

- `tests/test_prune_rescale_mode.py` fails on Della (torch 2.7.0, CPU) and passes on the Mac
  (torch 2.10). Not investigated.
- The R² cache on Della scratch is written without a lock; concurrent scorers can drop each other's
  entries (they are recomputed, not wrong).
