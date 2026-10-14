# agent-c1ff: the claim-1 flip-flop networks re-read at their floor on the 300k chain

Branch `agent-c1ff` (from `restore-oct5` at a5b4f98). Della checkout `~/trainRNNbrain_c1ff`.

## 0. The three networks

The claim-1 flip-flop runs (standard 3-bit flip-flop, N = 1000, control). Each was trained 75k with
kept snapshots, continued to 200k, and two are now continued to 300k (FFtop300 15526698, 15526699).

| 200k folder seed | 75k seed (snapshots) | 300k seed | longest chain | floor read-out before the beta guard |
|---|---|---|---|---|
| 2461724787 | 1542248578 | 3194339211 | 300k | crossing 26,949, 354 active; beta = 3 (now rejected) |
| 4113307577 | 2851654641 | 3204992658 | 300k | crossing 39,142, 337 active; beta = 3 (now rejected) |
| 1258717025 | 2670297180 | none | 200k | crossing 69,050, 266 active; fit valid |

(The 300k seed for 4113307577 is read from the FFtop300 15526699 log once the run folder is written.)

## 1. Predictions and pass/fail checks, written 2026-10-14 05:45, before any 300k result was read

### 1a. The refitted floors

A floor read-out is VALID when all of these hold (thresholds fixed now):
- `common.noisy_floor` returns `reason = None` on the joined chain: floor above its lower bound,
  beta below 0.99 x 3, late_rise <= 1.5, the smoothed loss reaches 1.05x the floor.
- The loss does not end below the floor: late_rise (median of the last 10% over the floor) >= 0.95.
  `noisy_floor` does not check this side; I check it here. A fit that the loss ends below by more
  than 5% is the failure mode the beta guard was built for, so it counts as no floor even if
  beta is free.
- The weight snapshot used lies within 500 iterations of the crossing.
- tau on its bound is reported, not used to reject (the project rule).

Predictions:
- P1. Both 300k fits pass, with beta well below 3 (guess 0.3-1.5). Confidence moderate (~60%):
  the loss was still falling at 200k (ending at 0.84x and 0.91x of the rejected floors), and
  100k more iterations may not flatten it enough for a clean fit.
- P2. Both crossings land later than 27k and 39k, near the 1.05x-smoothed-minimum crossings on the
  200k chain (52k, 60k). Since the minimum can only fall with 100k more iterations, the
  1.05x-minimum crossing on the 300k chain can only move later; I expect the fitted crossings in
  50-90k. Pass: within +-15k of the 1.05x-minimum crossing recomputed on the 300k chain by
  independent code (a running median and its minimum, not the fit).
- P3. Active at the new crossings 260-345 (at 52k and 60k on the 200k chain: 341 and 292).
- P4. 1258717025 on its 200k chain reproduces the current read-out: crossing 69,050, snapshot
  69,000, 266 active. Its ablation JSON, re-run on the same device type (GPU), matches the existing
  one to within 0.002 of r2 (a mismatch means the code path or the environment changed and every
  new number is suspect until explained).

### 1b. Ablation and prune-and-retrain at the new floors

Expected from the paper's claim 1 (flip-flop at its floor): r2 intact 0.94-0.95; all silent removed
0.25-0.48; as many at random 0.01-0.09; silent pruned and retrained 3,000 iterations 0.937-0.947
(intact fine-tuned 0.943-0.950); random pruned and retrained 0.920-0.924; 266-354 active.

- P5. Intact held-out r2 0.94-0.95; removing every silent unit 0.2-0.5; as many at random < 0.15.
- P6. Pruning the silent units and retraining restores within 0.01 of the intact fine-tuned
  network; random pruning plus retraining stays at least 0.01 below the silent-pruned network.

What would change claim 1 on the flip-flop (decided now):
- C-a. For any valid network, silent-pruned-and-retrained ends more than 0.01 below the intact
  fine-tuned control: the network needs its silent units at the floor; "does not need them" fails.
- C-b. For any valid network, random-pruned-and-retrained ends within 0.005 of silent-pruned: the
  silent units are no more dispensable than any others after retraining; the contrast the paper
  draws disappears for that network.
- C-c. Removing the silent units costs nothing (r2 within 0.02 of intact): the flip-flop's
  "wired in but not needed" wording would become "not needed at all", a change of wording only.
- Otherwise claim 1 stands, and only the quoted ranges are updated.

If neither 300k fit is valid, the flip-flop's at-floor ablation rests on one network
(1258717025), and that is reported as such, not papered over with the 1.05x-minimum read-out.

## 2. Method

(filled in as the work goes)

## 3. Results

(filled in after the runs)
