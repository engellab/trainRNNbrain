## 2026-11-25 13:24 — Optimizer remedy: stage 1 submitted on Della

Rules: `2026-11-25_1318_rules-optimizer-remedy.md`. Clean clone `~/trainRNNbrain_optrem` at cdb57fa, gpu-short, launcher
`slurm/optimizer_remedy_della.slurm` (arm = (index − 1) / 3: 1 nodecay, 2 adamw1e-2, 3 amsgrad, 4 sgdm0.03, 5 sgdm0.1; 0 control).

| Job | Array | What | Time limit |
|---|---|---|---|
| 18377208 | 4-18 | 3BFF, five candidates, 75,000 iterations | 10 h |
| 18377209 | 4-12 | CDDM, nodecay, adamw1e-2, amsgrad, 75,000 iterations | 7 h |
| 18377210 | 4-9 | Maze, nodecay and adamw1e-2, updates 1-2,000 (80 GB cards) | 14 h |
| 18377211 | 4-9 | the same six, updates 2,001-4,000, after the matching task of 18377210 | 14 h |
| 18377212 | 1-3 | the matched default Maze networks (18374320), updates 2,001-4,000 | 14 h |
| 18377213 | - | GPU check that AdamW at 0 and 1e-6 equal Adam at 0 (`firstmate/data/optimizer-remedy/scripts/cuda_equal.py`) | 15 min |

Expected about 200 GPU-hours of the 250 allowed. The training code (`rnns`, `trainer`, `training`, `tasks`, configs) differs from
that of the 3BFF and CDDM control networks' submission commit (e4c2ae3) only in the AMSGrad option and its comments.
Data: `<task>_optimizer_remedy` beside the other supervised sweeps, and `trained_RNNs_rl_maze_optrem`.
