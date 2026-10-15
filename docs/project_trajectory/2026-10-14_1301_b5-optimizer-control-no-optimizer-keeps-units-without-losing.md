### B5, optimizer control: no optimizer keeps units without losing performance (read 14 Oct, local copy of Spock data)

`NBitFlipFlop_b5_optimizer` (another session's run, branch b5-optimizer; 3-bit flip-flop, N = 1000,
200k, 3 seeds), read on the merged code (seeded held-out R², smoothed-minimum fallback). Finished arms:

| arm | active at floor (crossing) | active at 200k | held-out R² |
|---|---|---|---|
| Adam, wd 1e-6 (standard) | 237-271 (69-77k) | 169-203 | 0.952-0.953 |
| Adam, wd 0 | 225, 254 (125-166k; 1 no read) | 208-221 | 0.962 |
| AdamW, wd 1e-6 | 282, 292 (117-118k; 1 no read) | 203-226 | 0.962 |
| SGD lr 0.03 | 703-736 (105-112k) | 683-695 | 0.893-0.896 |
| SGD lr 0.1 | 483-585 | 451-523 | 0.909-0.914 |
| SGD lr 0.3 | 254-331 | 229-250 | 0.923-0.925 |
| SGD lr 1 | 249-271 | 212-247 | 0.908-0.917 |
| SGD lr 3 | 174-175 | 198-260 | 0.30-0.44 (fails) |

SGD keeps more units only where it learns the task worse: at lr 0.03, ~700 active at R² 0.895 against
Adam's ~250 at 0.952; at its best lr (0.3, R² 0.924) it has as few units as Adam (254-331 at the
floor, 229-250 at 200k). Adam with coupled weight decay 1e-6 costs 0.01 of R² against Adam wd 0 and
AdamW (0.952 vs 0.962) and ends with ~20-40 fewer units; decoupling it (AdamW) or removing it gives
the same network. So "silence is an Adam + L2 artefact" is ruled out on the flip-flop. Figure:
`img/internal_figures/diag_b5_optimizer.png`. Still running on Spock: SGD wd 0 at lr 0.1-3; Pavel
asked to drop lr 10 (cancellation needs Pavel: the auto-mode classifier blocked cancelling another
session's jobs).
Follow-up (agent-fix ff994a9, merged): fine-tune batches seeded (FINETUNE_SEED); three repeats on
flip-flop 1258717025 agree to the 5th decimal (silent pruned 0.94697, intact 0.95058, random
0.92122). The 0.0103 gap was noise (four earlier unseeded retrains: -0.0033, -0.0048, -0.0103,
-0.0036); the other two flip-flop networks end 0.0031 and 0.0026 below intact, random pruning
0.019-0.023 below silent pruning. Paper edits drafted on paper branch `numbers-seeded` (710de69,
compiles), NOT merged: held until Figure 1 is rebuilt with N = 4000 (~15 Oct), because the
caption's panel e count (272) would disagree with the text (277). When rebuilding Figure 1, point
`RETRAIN_DIR` at `c1_retrain_seeded2`.

**B5 re-read at the floor (Pavel: "you read it at the end - not a fair comparison").** The table above
mixed a floor active count with end-of-training R². Re-read with BOTH at each run's 1.05x noisy-loss
floor (R² estimated by the loss ratio - the B5 runs kept only a rolling restart checkpoint, no
snapshot series; 36 draws, seeded):

| arm | floor crossing | active at floor | R² at floor (est.) |
|---|---|---|---|
| Adam, wd 1e-6 (standard) | 69-77k | 237-271 | 0.950-0.951 |
| Adam, wd 0 | 125, 166k (1 no read) | 225, 254 | 0.960-0.961 |
| AdamW, wd 1e-6 | 117, 118k* (1 no read) | 282, 292 | 0.961 |
| SGD lr 0.03 | 105-112k | 718-736 | 0.882-0.884 |
| SGD lr 0.1 | 97-120k | 483-585 | 0.901-0.905 |
| SGD lr 0.3 | 51-94k | 254-331 | 0.903-0.911 |
| SGD lr 1 | 20-87k* | 249-271 | 0.894-0.908 |
| SGD lr 3 | - | - | fails (0.30-0.44 at 200k) |

(* smoothed-minimum fallback.) The conclusion is the same at the floor: SGD keeps more units only
where it converges to a worse solution (~720 units at R² 0.883 against Adam's ~250 at 0.951); at
SGD's best learning rates (0.3-1) it has Adam's count at R² 0.90-0.91. Adam with coupled weight decay
reaches its floor ~1.6-2x sooner and 0.01 lower in R² than Adam wd 0 / AdamW, with a similar count.
Figure `diag_b5_optimizer.png` redrawn (left: both at the floor; right: both at 200k).

**C3 updated (Pavel agreed: "the Adam optimizer is not the issue of silent units - SGD still does the
same").** Claim wording now in `docs/paper_audit.md` §2 C3: "Silence is not an artefact of the
optimizer: SGD at its best learning rates leaves as many units silent as Adam, and keeps more only
when it converges to a worse solution." Nuance kept in the audit: even at its best, SGD reaches R²
0.90-0.91 against Adam's 0.95, so the comparison is between networks that solve the task to slightly
different degrees. Status board row C3 rewritten; SGD without weight decay pending on Spock.

