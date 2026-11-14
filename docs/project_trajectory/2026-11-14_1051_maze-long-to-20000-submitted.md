# Maze long training to update 20,000: submitted on Della

Rules: the entry `*_rules-maze-long-training-to-20000.md` of the same day, committed (94e9366) before submission.

Six networks (indices 4, 6, 7, 9, 26, 27), stages 7-10, chained with `--dependency=aftercorr`, checkout
`~/trainRNNbrain_mazelong` at 1a80f5e with a clean tree, parents `N=<N>_act=relu_cont6/s<seed>` (all six have
their parameter files):

| Stage | Updates | Array job | Output cell |
|---|---|---|---|
| 7 | 12,001-14,000 | 17546732 | `_cont7` |
| 8 | 14,001-16,000 | 17546733 | `_cont8` |
| 9 | 16,001-18,000 | 17546734 | `_cont9` |
| 10 | 18,001-20,000 | 17546735 | `_cont10` |

About 122 GPU-h, about 5 h per stage. If a stage fails, the later stages of that network stay pending with
`DependencyNeverSatisfied` and are cancelled by hand.
