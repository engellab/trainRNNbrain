## 2026-10-31 17:02 — Figure 1: panel c read, and panel d's ablations re-run at the current floor (rule set before the jobs)

Work package fig1-inactive (ptolmachev/dead-relu-paper#5), trainRNNbrain branch `wp/fig1-inactive`.

### Panel c, read

`fig_paper_F1.py --collect` at 538bd77, Della CPU job 16554564 (5.5 min), from the worktree
`~/trainRNNbrain_fig1` (detached at 538bd77). Output `data/fig_paper_F1_collect.pkl`. Each run of the
N = 1000 cell of each size series is read with `common.read_run` at 1.05x its own noisy-loss floor
(read-out 79932ef); held-out R² at the crossing from `heldout_r2.floor_r2`, 36 draws. None of these
cells kept snapshots, so every R² below is estimated from the end of the run (1 − (L(t*)/L_end)(1 − R²_end)).

| task | run (folder seed) | active at the floor | t* | R² at the floor |
|---|---|---|---|---|
| 3BFF | 3374117298 (the straggler, excluded) | 277 | 58,968 | 0.949 |
| 3BFF | 1914910909 | 296 | 59,917 | 0.950 |
| 3BFF | 3866906883 | 297 | 62,703 | 0.950 |
| 3BFF | 4186074845 (FFseed4) | 240 | 67,871 | 0.950 |
| CDDM | 3833537590 | 436 | 27,259 | 0.827 |
| CDDM | 1272763220 | 420 | 26,535 | 0.829 |
| CDDM | 457853 | 437 | 23,929 | 0.828 |
| DMTS 5τ | 86371460 | 422 | 18,599 | 0.999 |
| DMTS 5τ | 1167683015 | 218 | 75,721 | 0.999 |
| DMTS 5τ | 2946835052 | no read-out: loss ends below its floor | | |

The board's decision is followed (the straggler stays excluded; FFseed4 replaces it), so the 3BFF cell
drawn is 296, 297 and 240: 278 ± 33 (mean ± SD). With all four runs it would be 278 ± 27, so the
decision does not move the mean.

### Panel d: the ablated weights are not at the current floor

The collect stage re-reads every network of panel d under the current read-out and compares its
crossing with the iteration of the snapshot that was ablated on 14 Oct (`c1_ablation_seeded`,
`c1_retrain_seeded2`, `c1_retrain_seeded`):

| task | network | ablated at | current t* |
|---|---|---|---|
| 3BFF | s1258717025 | 69,000 | 67,191 |
| 3BFF | s2461724787 | 52,000 | 50,073 |
| 3BFF | s4113307577 | 60,000 | 58,035 |
| CDDM | s1423687489 | 26,000 | 25,020 |
| CDDM | s2657899762 | 25,000 | 26,028 |
| CDDM | s2670295995 | 25,000 | 24,219 |
| DMTS | s198893703 | 21,000 | 23,106 |
| DMTS | s3322309204 | 17,000 | 20,477 |
| DMTS | s3753404946 | 42,000 | **65,778** |

Eight networks are within 3,500 iterations of their current crossing. DMTS s3753404946 was ablated
24k iterations early: 79932ef moved DMTS crossings later ("DMTS at its late level", entry
2026-10-16_1631), and the ablation predates it. Its active count at the current crossing is 261,
against 291 in the ablation file.

**Re-run.** All nine networks, ablation (`floor`) and prune-and-retrain (`retrain --at floor`), with
`ablate_silent_units.py` at 538bd77 (unchanged from main 484e0a4), which takes the kept snapshot
nearest the current crossing. Outputs go to new folders, `c1_ablation_79932ef` and
`c1_retrain_79932ef`; nothing existing is overwritten. Cost: 18 GPU tasks, each under 1 h, so at most
18 GPU-h.

**Rule, written before the jobs run.** Claim C1's "the network does not need its inactive units" survives the
re-run if, in every one of the nine networks:
1. on CDDM, removing every inactive unit changes held-out R² by less than 0.01;
2. on all three tasks, the network pruned of its inactive units and retrained for 3,000 iterations
   ends within 0.01 R² of the intact network retrained the same way;
3. on all three tasks, the network pruned of as many random units and retrained ends below the
   network pruned of its inactive units and retrained, or within 0.01 of it.

Any network that breaks rule 1 or 2 is reported by name in the paper and on the board, and C1's
wording is narrowed to the networks that pass. Panel d is drawn from the new folders whatever the
outcome.
