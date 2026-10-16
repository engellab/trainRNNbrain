## 2026-10-15 22:06 — Initial weights are now drawn independently (`init_draws: independent`)

Pavel, on the correlated initialisation found in entry 1500: "fix it!".

**Change.** `RNN_torch.sparse` takes a `cpu_generator`. With it, every matrix of a network draws
from one continuing stream (`init_stream`, seeded from the network's generator), so W_rec, W_inp
and W_out are independent. A new model key `init_draws` selects the behaviour:
- `independent`: set in all six `configs/model/*.yaml`.
- `shared`: the original reseed-per-matrix draws. It is the code default when the key is absent, so
  every config saved before today (none has the key) rebuilds its network exactly as trained. That
  matters for `gradient_participation.initial_params`, the untrained twin behind Figure 1 panel b.

**Version boundary.** Runs launched from main after this commit start from different initial weights
than earlier runs with the same seed:
- Standard (non-Dale) network: W_rec is unchanged, because it takes the stream's first draws; only
  W_inp and W_out differ.
- Dale networks: W_rec's later blocks differ too.

Jobs already queued or running on Della come from pinned worktrees, so they keep the old draws.

**Tests, pass/fail set before running** (`tests/test_init_draws.py`; 23 of 23 test scripts pass):
- Independent mode at N = 1000: corr(W_inp, W_out) = −0.03 and corr(W_inp, W_rec's first rows) =
  −0.004. The bar was |r| < 0.1.
- Shared mode reproduces the old coupling: correlation 1.000000. A separate check, loading the HEAD
  copy of the module, showed shared mode gives bit-identical W_rec, W_inp and W_out to the pre-fix
  code, with and without Dale's law.
- The same seed gives identical weights, another seed different ones. The composed standard config
  delivers `init_draws: independent` to the constructor through `hydra.utils.instantiate`.

**Not yet done: does the coupling change silencing?** The test would be 3 seeds at N = 1000 on CDDM
and the flip-flop, trained with independent draws and compared at the floor with the current
controls (~45 GPU-h). It waits for Pavel's go-ahead.

**Also today** (Pavel's decisions):
- The paper branches `numbers-seeded`, `figure2-methods` and `w1-draft` are merged into the paper's
  `main` (e6e13d6). The `01_intro.tex` conflict went to W1's version. The paper builds clean, and
  the merged branches were deleted.
- A new N = 1000 flip-flop control seed is submitted to replace the straggler (Della 15657728_4,
  `ff_revive` launcher, `SWEEP=ff_revive_c200000`, 200k, from `~/trainRNNbrain_redo` 709c17d so it
  matches the cell's other runs and the old initial draws; ~15 GPU-h).
