## 2026-10-15 20:59 — Flip-flop size series at the floor: N = 4000 landed, every size reads out

FFtop4000 (15431475) finished: the three N = 4000 flip-flop controls reached 200k
(`NBitFlipFlop_paper_grid_c200000/EqType=h_N=4000_arm=control`).

**Read**, using Figure 1's own functions (`fig_paper_F1.runs_with_loss`, `at_own_floor`) under the
committed `common.noisy_floor` (597403f). Della CPU job 15655360, script `~/f2_logs/ffc_read.py`.
Rule, from `panel_c`: the series is drawn at the floor only if every listed size has a floor
read-out; otherwise it is drawn at end of training.

| N | runs read at the floor | active units at the floor | mean crossing |
|---|---|---|---|
| 500 | 6 of 6 | 187, 189, 197, 197, 199, 237 | 69,600 |
| 1000 | 2 of 3 | 287, 294 | 64,700 |
| 2000 | 3 of 3 | 363, 387, 410 | 63,600 |
| 4000 | 3 of 3 | 500, 538, 594 | 70,100 |

The excluded N = 1000 run is the straggler s3374117298 ("never reaches the bar"), whose
continuation to 300k is an open decision.

Every size reads out, so panel c can show the flip-flop at the floor. The active count grows as
N^0.47: the share of active units falls from about 40% at N = 500 to about 14% at N = 4000. The
exponent matches the provisional three-size value.

**Caveat.** This is the committed rule. The rewrite of `noisy_floor` awaiting Pavel (Hampel
smoothing, a sustained crossing; entry 1400, uncommitted) may move these numbers, so re-read once it
lands. Figure 1 itself is being rebuilt on the cosyne2027-fig branch; I build it from `main` after
that branch merges.

**Also submitted:** B4 at N = 4000, minimal plan approved 14 Oct. The young, old and fresh cells, 3
seeds each, 100k iterations: jobs 15655301 (old: the 200k controls), 15655302 (young: their 75k
parents) and 15655303 (fresh). Submitted from `~/trainRNNbrain_redo` (709c17d, the commit of the
N = 1000 B4 runs) with `--constraint=gpu80 --mem-per-gpu=64G --time=3-12:00:00`. A run is continued
to 200k only if its loss has not reached the floor bar by 100k.
