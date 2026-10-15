### 2026-10-15 — the remedy top-ups to 200k resubmitted with snapshots (Pavel)

The other session's `RMtop200` batch (flip-flop remedy runs 75k → 200k, the Figure 3 data) had 183
tasks left at `nice=10000` - behind every job on the cluster; users cannot lower nice. Pavel chose to
cancel and resubmit. Cancelled (all 183 were pending). Which parents still lack a finished 200k
continuation was rebuilt from the data (`rm_missing.py`: a `_c75000` parent is done when a `_c200000`
run names it in `paths.init_from` and has finished weights): 192 = 183 cancelled + 3 that hit the
batch's 24 h limit (15500305_3, 15500313_2, 15500318_1) + 3 N = 4000 controls already being run by
FFtop4000 (excluded) + 3 not traced to the batch (included; a few GPU-h). Resubmitted 189 tasks as 70
arrays named `RMtop200` from `~/trainRNNbrain_redo` (709c17d, trainer identical to 7739a60), KEEP=1
(snapshots every 1,000, float16), time limits sized from measured speeds (N ≤ 1000: 30 h, 2000: 60 h,
4000: 5 days on gpu80). Array indices checked against the launcher's own sorted parent list (dry
run). Job ids and cells: Della `~/f2_logs/rm_submit_go.txt`. By N: 500: 13, 1000: 139, 2000: 19,
4000: 18. Cost ≈ 3,000 GPU-h; snapshots ≈ 120 GB.

**Paper: Figure 2 and Methods (branch `figure2-methods`, 70faada; not merged).** New Figure 2 with a
full caption (panels a-l, read-out rule incl. the smoothed-minimum fallback, R² measured on snapshots
in h and k, estimated elsewhere, seeds per rung, budgets). Methods: the bias is "fixed at zero and not
trained" on every standard network (the trainable-bias configuration named as earlier); corrected
"spectral radius never varied" (also Supplementary S3), the metabolic penalty's place, λ_frm = 0.2 in
Fig. 2f; a paragraph on R² at the floor. Old Figure-2 references to the remedies figure are now
`\todo` markers until Figure 3 exists. Figure 2 shrank to 0.54 of the text height (~4.4 in wide) to
fit its caption; Figure 1 to 0.565 (it overflowed on main). Merge order: `numbers-seeded` and this
branch merge cleanly; `w1-draft` conflicts only in `01_intro.tex` (keep W1's). Not fixed: S3 still
says the noise block is read under a peak-rate rule.

