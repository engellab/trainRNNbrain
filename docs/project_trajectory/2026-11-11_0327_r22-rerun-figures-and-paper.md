## 2026-11-11 03:27 — r22-rerun: Fig. 5a,b and Extended Data Fig. 6g-i draw the Maze read-out dropout and rescaling reruns; the paper's caveat sentences replaced by the result (dead-relu-paper #228)

**Why.** The Maze read-out dropout and rescaling networks were retrained with the corrected trainer
(`trainer.freeze_after_end=true`): rules `2026-11-09_1513_round19-dropout-rescale-fz-rules.md`, read at update 2,000 in
`2026-11-10_2014_round19-dropout-rescale-fz-results-2000.md` and at update 4,000 in
`2026-11-11_0306_round19-dropout-rescale-fz-results-4000.md`. Both technique verdicts on Maze stay "fails", so the
paper's caveat ("whether read-out dropout meets the criterion on all three tasks is not known") is replaced by the
result, and the figures draw the rerun networks in place of the runs made before the correction. No new training or
analysis run; every number below is from `data/round19/fz_reruns.json` (`round19_read_fz.py`) or from the figure
printouts named here.

### Code changes (branch wp/r22-rerun)

- `maze_arms_r19.py`: `FZ_ARMS` adds dropout0.2_beta4, dropout0.05_beta4, rescale, rescale_mild, rescale_few and
  rescale_tgt4, so these six arms are read from `data/trained_RNNs_rl_maze_fz` only. The reader's cross-check of every
  seed against `fz_reruns.json` (read from the raw progress records, independently of that file) prints "every read
  matches".
- `fig_paper_F5_tasks.py`: `MAZE_ARMS` takes the same six arms from source "fz" (`fz_reruns.json`, each record cut at
  update 2,000); before, read-out dropout 0.05 and rescaling m = 4 came from the round 18 parity file and read-out
  dropout 0.2 and the other three rescaling values from `read_maze_arms.json` (the first runs). Noisy synapses keep
  their first runs (not rerun).
- `fig_ed_maze_settings.py`: panel i gains the two continued rescaling values (m = 10, m = 4), eight columns in all;
  panel h is 56 mm wide (was 66) and panel i starts at 92 mm and is 68.5 mm wide (was 103.5 and 57), because with six
  columns' spacing the labels "rate 0.1 / + sparsity" and "dropout / 0.2" overlapped (`check_overlaps`).
- `tests/check_f5_maze_reruns.py` (new): an independent check of the drawn Fig. 5 Maze marks, below.

### What the figures now draw (Maze, N = 1,000; controls 426, 449, 415 active, mean 430, success 0.898)

From `data/round19/fig_paper_F5_tasks_printout_r22rerun.txt` (Fig. 5) and
`data/round19/fig_ed_maze_settings_printout_r22rerun.txt` (Extended Data Fig. 6):

| Value | Update 2,000: active units; success | Verdict at 2,000 | Update 4,000 (ED Fig. 6i) |
|---|---|---|---|
| read-out dropout 0.05 | 444, 416, 498; 1.00, 0.98, 0.97 (fold 1.05; mean success +0.083) | fails (a) | not continued (all learned by 2,000) |
| read-out dropout 0.2 | 428, 456, 436; 0.92, 0.84, 0.93 (fold 1.02; +0.000) | fails (a, l) | 397, 388; seed 3 stopped at 2,100 (incomplete) |
| rescaling m = 4 | 370, 251, 375; 0.62, 0.36, 0.38 (fold 0.77; -0.445) | fails (a, p, l) | 333, 190, 241 (fails) |
| rescaling m = 10 | 233, 239; seed 3 stopped at 500 (fold 0.55; -0.695) | incomplete | every seed stopped (no verdict) |
| smaller factor | 252; seeds 2, 3 stopped at 1,100 (fold 0.59; -0.281) | incomplete | 297 (incomplete) |
| fewer units | 240, 251, 233; 0.54, 0.81, 0.48 (fold 0.56; -0.289) | fails (a, p, l) | 231, 200, 203 (fails) |

The verdict table, the column order, and both tolerance ranges in the Fig. 5 printout are unchanged: prune-and-duplicate
alone meets the criterion on all three tasks from 0.0032 to 0.0161 (one-value rule) and from 0.0046 to 0.0213
(matched values). The printout gains one line, "rescaling matched value 4.0: 3BFF 0.0097, CDDM never, maze never",
because m = 4 now has three finished Maze networks. Fig. 3 (`fig_paper_F3_tasks.py`, which also uses
`maze_arms_r19`) prints the same output before and after the change (diff empty).

**Fig. 5d needs no new removal run.** Its 12 Maze networks are prune-and-duplicate and the rate penalty at 0.03, 0.1
and 0.3 (`data/round19/fz_maze_added_summary.json`); no Maze read-out dropout network is drawn there, so the counts
"43 of the 45" and "42 of the 45" do not change. In the rerun networks, k (active units minus 430) is negative in 11 of
the 15 that reached update 2,000 (two read-out dropout networks, 416 and 428; all nine rescaling networks), which the
Supplementary Methods now state.

### Checks (pass rules set before each run)

- `tests/check_f5_maze_reruns.py`: for each of the six arms, the mean over the networks read at 2,000 and a cross at
  each network without a learned point must be drawn at the `fz_reruns.json` value (fold and success, to 1e-9); panel a
  must hold a Maze marker at 0 per stopped network (3); no pre-correction mean may remain. **PASS.** The first version
  of this check expected one marker per learned network, which the figure does not draw (a learned network is inside
  its value's mean marker); the rule was corrected before the second run and the second run passed.
- `tests/scan_open_markers.py`: Fig. 5 PASS (100 of 100 filled centres dark, 0 rule failures); ED Fig. 6 PASS (218 of
  218, 0 failures). `tests/scan_marker_edges.py`: Fig. 5 218 markers pass, 0 fail; ED Fig. 6 316 pass, 0 fail.
  `check_overlaps` (text against text and the canvas edge): Fig. 5 115 texts, ED Fig. 6 167 texts, no overlap.
  `pdfimages -list`: no raster in either PDF. Both rendered at 300 dpi and inspected.
  Scan outputs: `data/round19/fig_paper_F5_tasks_scan_r22rerun.txt`, `data/round19/fig_ed_maze_settings_scan_r22rerun.txt`.

### Paper (dead-relu-paper branch wp/r22-rerun)

Results 5 and the Discussion state the result in place of the caveat; Methods: Training (defect paragraph and the
list of stage exceptions), Methods: Training techniques (trainer version, the rescaling variants, the Maze removal
set), Supplementary Table 2 (Maze rows of read-out dropout and rescaling, notes), Supplementary Methods (removal,
the 0.9 condition, two measurement points), Supplementary Table 7 (rescaling variants row) and Supplementary Table 8
(Extended Data Fig. 6 g, i) are updated. Re-initialisation, not rerun, keeps its caveat in Methods: Training.
