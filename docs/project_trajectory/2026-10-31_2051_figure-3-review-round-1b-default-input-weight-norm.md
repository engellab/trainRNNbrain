## 2026-10-31 20:51 — Figure 3, second round-1 review: the default input-weight norm and the "largest gain" sentence

Work package fig3-settings (dead-relu-paper#7, PR #26), branch `wp/fig3-settings`. Answers the
second reviewer's round-1 comment on PR #26. No new runs. The active counts and R² values below are
from the `fig_paper_F3_settings.py` printout on the caches of entry 2026-10-31_2010, unchanged since
entry 2026-10-31_2035.

**The default input-weight draw.** The input-scale ladders set every unit's input-weight vector
(row of W_inp) to an absolute norm of 0.5, 2, 5 or 20 at initialisation (`model.input_row_norm`).
The default runs have `input_row_norm: null`, `dale: false`, so W_inp is drawn by
`get_connectivity_unconstrained` with normal entries of mean 0 and s.d. 1/sqrt(N); its row norm is
about sqrt(n_inputs/N). To check the number rather than trust the formula, I rebuilt `RNN_torch`
from each default run's saved config and seed (model block of the config, `n_inputs`/`n_outputs`
from the task block) and took the L2 norm of each of the 1000 initial W_inp rows:

| task (n_inputs) | runs rebuilt | mean row norm per run | median | s.d. over units |
|---|---|---|---|---|
| 3BFF (3) | `NBitFlipFlop_std_ksweep_bias0/EqType=h_k=3_N=1000_iters=150000`, 3 seeds | 0.0510, 0.0498, 0.0503 | 0.048-0.049 | 0.021-0.022 |
| CDDM (6) | `CDDM_std_g0_drift_bias0/EqType=h_N=1000_iters=200000`, 3 seeds | 0.0738, 0.0740, 0.0745 | 0.073-0.074 | 0.021-0.023 |

The CDDM ladder's own controls (`CDDM_c1_snapshots/EqType=h_N=1000_arm=control`) are on Della only;
one of their configs has the same model settings that set the draw (`n_inputs: 6`, `dale: false`,
`input_row_norm: null`), checked over ssh. So the default mean norm is 0.050 on 3BFF and 0.074 on
CDDM. Norm 0.5 is 10 times the 3BFF default and 7 times the CDDM default; norm 20 is 400 and 270
times. Every rung of both ladders is above the default draw, so "rescaled to 0.5" is an increase.
The measuring script (scratch, not committed) is 20 lines: load the config with `yaml`, call
`RNN_torch(n_inputs=..., n_outputs=..., seed=cfg["seed"], **model)` and take
`np.linalg.norm(net.W_inp, axis=1)`.

**"Largest gain" is now qualified.** Results 3 had said the input-weight scale gives the largest
gain, two sentences after saying weight decay 0 gives the largest gain on CDDM, and with SGD at
learning rate 0.03 adding about 500 units over Adam (mean 749 against 246) in panel d. The sentence
now reads "On the 3-bit flip-flop (3BFF), the initial input-weight scale gives the largest gain at
equal R²": norm 20 has 455-500 active units (mean 477) against 245-296 (mean 275), +203, at an
estimated R² of 0.953-0.954 against 0.950-0.952. The Figure 3 title now gives a number: "by at most
about 200 units at equal R²" (3BFF norm 20, +203; CDDM weight decay 0, +116). SGD at learning rate
0.03 (+about 500) is at R² 0.873-0.875, and the rate penalty (all 1000 active) is the drive of
Section 5, not a standard setting.

**Estimated against measured R².** No 3BFF ladder or B5 run kept weights at 75k, so every 3BFF R²
at 75k is the loss-ratio estimate; Results 3 now says "estimated" in both 3BFF paragraphs. For the
Adam/AdamW gap it also cites the R² of the final weights at 200k, which is measured: 0.962 for Adam
without weight decay and for AdamW, against 0.952-0.953 for the default Adam.

**One name per thing.** "Recurrent-weight magnitude penalty" in Results 3, Figure 3a, Extended Data
Fig. 4e and l, Methods and the board (was also "W_rec magnitude penalty", "recurrent
weight-magnitude penalty", "W_rec penalty"). "Input-weight scale" with the hyphen in Figure 3 and
Extended Data Fig. 4h, i and l. The Extended Data Fig. 4 titles come from a new
`fig_paper_F2_fails.PAPER_KNOB` map, so `fig_knobs_at_floor.py` (slides) is untouched.

Board C5 updated: the default norm, the means, "estimated", and the CDDM range at 50k now covers norm
0.5 (166-336 against 355-376; it said 2-20, 166-268).
