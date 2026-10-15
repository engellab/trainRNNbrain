### Does input scale recruit on CDDM too? (submitted 2026-10-13)

The flip-flop's input-scale ladder is the one knob that adds units at the floor at no R² cost
(~300 → 614-913 active at scale 20). It was never run on CDDM. Pavel approved the CDDM repeat:
standard CDDM, N = 1000, `model.input_row_norm` = 0.5 / 2 / 5 / 20 (each unit's input weights rescaled
to that norm at initialisation, then trained), 3 seeds each, 50k iterations (CDDM's floor is at
14-35k), a snapshot every 500 so R² at the floor is MEASURED, not estimated. Code: `~/trainRNNbrain_c1abl`
at b524ac7, the same commit and launcher as the reference (the three CDDM claim-1 snapshot controls).
Jobs 15502206-09, ~25 GPU-h. Output: `CDDM_c1_snapshots/EqType=h_N=1000_arm=winp<s>/`; snapshots
under `c1_snapshots/CDDM_c1_snapshots/`. Launcher change 939e4a9 (INPUT_ROW_NORM), smoke-tested
locally: input rows at norm 5.00 (4.98-5.02) after 10 iterations.

