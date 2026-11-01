## 2026-11-01 09:25 — Correction to 2026-11-01_0913: three failing colour pairs had no stated limitation

Package board-style-r3 (dead-relu-paper#63), after the review of PR #75
(https://github.com/ptolmachev/dead-relu-paper/pull/75#issuecomment-6561319506).

`2026-11-01_0913_board-matches-merged-paper-and-colour-limitations.md` says that every failing pair
now carries a stated limitation. That was false. I had checked every teal against every rule and
grey, but not the teals against the task colours, and I left INACTIVE against the lightest teal out
of the written list although my own run printed it. Three pairs had no limitation:

| pair | normal | protanopia | deuteranopia | limitation added |
|---|---|---|---|---|
| INACTIVE - SETTING_RAMP[0] | 10.0 | 3.8 | 7.6 | INACTIVE never shares a panel with the setting teals |
| DMTS - SETTING_RAMP[1] | 8.3 | 8.0 | 7.7 | tasks and setting teals never share colour coding in one panel |
| 3BFF - SETTING_RAMP[1] | 12.8 | 12.5 | 12.9 | same |

This time I ran every pair of the 17 series colours: 3 tasks, 7 rules, 4 teals, BASELINE, ACTIVE
and INACTIVE, 136 pairs. The run used the independent script (Machado 2009 severity-1 matrices,
OKLab distance × 100). Fifteen pairs fall under 8 in a colour-blind simulation or under 15 in
normal vision. Each one now appears, with its vision type, in the `paperstyle.py` docstring and in
the paper's `docs/figure_style.md`. The dataviz validator gives the same value, to 0.1, for every
pair whose number those texts quote. The fifteen pairs:
- Tasks against rules (rule "tasks and rules never share colour coding"): DMTS-rescale 2.7
  (deuteranopia), 3BFF-sparsity 2.9 (protanopia), 3BFF-rate penalty 7.2 (deuteranopia) and 14.8
  (normal), CDDM-dropout 13.2 (normal). The last two were covered by the rule but not named before.
- Tasks against teals: the two in the table above.
- Teals against rules: SETTING-rescale 3.7 (protanopia), ramp step 1-rescale 6.4 (deuteranopia),
  ramp step 3-duplicate 5.2 (protanopia), SETTING-synaptic noise 13.7 (normal). The 0913 entry and
  the first version of the style text gave the 6.4 and 5.2 without their vision types.
- Greys: BASELINE-INACTIVE 8.6, INACTIVE-ramp step 0 (table), BASELINE-ramp step 0 4.4
  (deuteranopia) and 8.3 (normal), ACTIVE-ramp step 3 12.4 (normal).
- Within the teal ramp: ramp steps 2 and 3, 10.7 (normal). The teals are ordered values of one
  setting, read by lightness, and every value is named in a legend or a direct label (checked on
  renders of Fig. 3 and the Extended Data figure of every standard-setting sweep).

No figure on paper main breaks the new limitations. The scripts that draw a teal (`fig_paper_F3_settings.py`,
`fig_paper_F2_fails.py`, and the tanh figure in `fig_paper_F4_noreturn.py`) draw no task colour and
no INACTIVE in the same panel, so no figure needs a redraw. The reviewer reached the same result.

Board: the excluded DMTS network's R² is now 0.417, the value the Supplementary table of every
network prints (the removal script had printed 0.418).
