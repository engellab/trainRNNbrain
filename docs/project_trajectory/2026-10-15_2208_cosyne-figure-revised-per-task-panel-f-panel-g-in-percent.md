## 2026-10-15 22:08 — COSYNE figure revised (Pavel's comments): per-task panel F, panel G in percent

`fig_cosyne2027.py` (commit e018694, merged here). Numbers drawn are unchanged; the agent diffed the
printed numbers block against the previous build. Panel letters stay A-G.

- B: titled "3-bit flip-flop, N = 1000". C: plain tick labels at 500/1000/2000/4000 (log axes kept).
- D: title "ablation at the loss floor"; legend "low-participation units removed, retrained (3k iter.)".
- E: loss axis cropped to the data (0.044-1.25); label "noisy loss / starting loss".
- F: split into CDDM and 3-bit flip-flop panels sharing axes; "metabolic penalty" renamed "rate L2
  penalty" (`Trainer.metabolic_penalty` is the mean squared rate); the CDDM fr-magnitude arm labelled
  "homeostatic rate penalty".
- G: `G_MODE = "percent"` (default): active units as % of N against N for the control and each 3BFF
  intervention at the end of 75k. Control 39.5% at N = 500 -> 13.6% at 4000; interventions at N =
  4000 from 22.5% (synaptic noise) to 92.7% (rate + sparsity). `G_MODE = "ratio"` keeps the previous
  panel (active / control's), written as `fig_cosyne2027_Gratio` on every build.
- `check_layout` now also tests left/right titles and right-hand tick labels; 0 problems.

Abstract: paper repo `cosyne2027/` (Fig.pdf = percent version, Fig_Gratio_backup.pdf = ratio version).
