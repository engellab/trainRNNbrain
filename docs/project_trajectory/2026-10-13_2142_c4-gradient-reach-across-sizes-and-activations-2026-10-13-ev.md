### C4, gradient reach across sizes and activations (2026-10-13 evening)

`gradient_participation.py --at floor` on 19 cells (15521043, CPU, 14 min; JSONs in
/scratch/gpfs/TENGEL/pt1290/gradreach and gradreach_ff). Units the gradient reaches, counted by the
gradient on each unit's incoming weights (in) and outgoing weights (out), 5%-of-q95 bar, against
active units on the same weights and batch (ranges over seeds; r² < 0.8 excluded):

| cell | read | active | grad in | grad out |
|---|---|---|---|---|
| CDDM N = 1000 (claim-1 snapshots) | floor | 415-434 | 428-443 | 419-431 |
| CDDM N = 500 / 1000 / 2000 / 4000 (paper grid, 100k) | end | 213-227 / 311-319 / 403-415 / 567-576 | 201-218 / 296-297 / 368-394 / 499-523 | 217-229 / 301-306 / 379-399 / 506-517 |
| CDDM leaky ReLU / softplus / sigmoid (200k) | end | 273-275 / 238-260 / 180-193 | 243-254 / 211-227 / 115-139 | 279-407 / 220-242 / 156-189 |
| DMTS N = 1000 (claim-1 snapshots) | floor | 291-441 | 337-518 | 245-325 |
| DMTS N = 500 / 1000 / 2000 (paper grid) | end | 129-141 / 155-187 / 192-321 | 136-154 / 161-199 / 239-300 | 107-129 / 136-180 / 194-296 |
| flip-flop N = 1000 (claim-1, 200k chain) | floor (27k, 39.5k, 69.5k) | 265-355 | 401-615 | 346-445 |
| flip-flop N = 500 / 1000 / 2000 (200k) | end | 129-158 / 171-185 / 249-255 | 207-245 / 332-408 / 962-1232 | 157-171 / 211-235 / 271-277 |

Check set before running: on ReLU networks the reached count within 10% of the active count.
- **CDDM passes at every size and at the floor**: the gradient reaches the active units and no
  others, so on CDDM "silent" and "out of the gradient's reach" are the same population at
  N = 500-4000. The other activations agree as well (sigmoid's incoming count is 30% lower).
- **DMTS is close** (incoming +5-25%, outgoing -10-25%).
- **The flip-flop fails**: outgoing weights 1.1-1.3x the active count, incoming weights 1.5-4.5x
  (962-1232 of 2000 at N = 2000). So on the flip-flop the gradient still reaches the INCOMING weights
  of many silent units. Claim C4 as written ("the gradient stops reaching the unit") holds for CDDM
  and roughly for DMTS, not for the flip-flop's incoming weights. Why is not tested.

⚠️ **A wrong turn, caught before it was reported.** The first pass counted the gradient on the hidden
state (g_state), which said the gradient reaches 995-1000 of 1000 flip-flop units, and a per-unit
"on-fraction" seemed to show silent flip-flop units sitting at the ReLU knee (input > 0 on ~12% of
steps at negligible rate). An independent forward pass of the same weights disagreed (silent units on
for < 0.4% of steps after t = 50), and the cause was the derivative: torch.clamp passes the gradient
at exactly h = 0, every unit starts a trial at h = 0, and on the flip-flop stays there until the first
input pulse (~36 of 300 steps; 1 of 300 on CDDM, which made a spike at 1/300). Those steps carry no
weight gradient (all rates and inputs are 0). g_state is now flagged in the docstring, `slope` counts
only h > 0 for ReLU, and the knee figure was deleted. The weight-gradient counts above are unaffected.

Also fixed: `snapshot_at` now searches the whole warm-start chain (the flip-flop claim-1 runs cross
their floor inside the parent, whose snapshots sit under the parent's sweep and seed; they had fallen
back to the end). Not yet fixed: the untrained twin of a warm-started run is rebuilt from the leaf's
seed, not the chain root's, so INIT counts for the flip-flop 200k runs are for the wrong network
(corr0 ≈ 0); the trained counts are unaffected.
Untrained twin fixed (gradient_participation builds it from the warm-start chain's first run): on the
flip-flop 200k runs its participation now correlates 0.999-1.000 with the trace's iteration-0 probe
(was ≈ 0). Re-running showed the flip-flop's trained counts move by up to ~15% between runs (one
network: incoming 408 → 477), because the task draws a fresh random batch on every call; the
flip-flop ranges in the table above carry that noise. CDDM's batch is a fixed grid.

