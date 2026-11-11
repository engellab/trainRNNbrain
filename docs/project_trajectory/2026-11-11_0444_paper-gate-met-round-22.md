## 2026-11-11 04:44 — The paper meets Pavel's gate: round-22 committee passes; final paper dead-relu-paper main d887629

**The gate** (Pavel, 6 Nov): an independent committee rates the paper equal to or better than Tolmachev &
Engel 2025 (Nat. Mach. Intell.) on writing, rigor, structure and figures.

**Round 22** (dead-relu-paper #231, on main b49b481, every experiment in): three fresh reviewers.

| Reviewer | Writing | Rigor | Structure | Figures |
|---|---|---|---|---|
| W | equals | equals | equals | equals |
| R | equals | surpasses | equals | equals |
| S | equals | equals | equals | equals |

Reviewer R spot-checked 18 groups of numbers against this notebook and refitted the size exponents with
independent code; every group matched.

**After the pass**, the factual and scope errors the reviewers listed were fixed in two packages, each
reviewed independently with a no-regression check: PR #234 (text: Results 1 scope for Maze; the rate-penalty
size test's pass rule and the one-value rule's timing restored in Results 5; three criterion conditions;
"at the values tested"; Supplementary Table 7 rows for the bend definition and the all-inactive-units count;
learning rates for N = 100 and 250; Results 4 fractions over the same denominator, entry 2026-11-11_0418)
and PR #236 (figures: Fig. 1d caption range 0.26-0.52 and 0.15-0.44; Fig. 5c/5d overlap; Fig. 4e means 244
and 508; one encoding for networks without a learned point in Fig. 3c and Extended Data Fig. 2e; Fig. 3b
empty column; λ_frm defined; Fig. 3 title scoped).

**Final paper:** dead-relu-paper main d887629, 87 pages, 0 undefined references, Extended Data order OK;
trainRNNbrain main d02bfb1. PDF copy: ~/Documents/Projects/paper_push/silent_units_paper_final_d887629.pdf.

**Rounds since the round-21 pass** (all merged after independent review): #225 polish, #227 new networks at
Pavel's request (N = 250 3BFF/CDDM, N = 100 Maze; entries 2026-11-09_1925, 2026-11-10_0806), #229 Maze
read-out dropout and rescaling retrained with the corrected trainer (both fail; entries 2026-11-10_2014,
2026-11-11_0306), #234, #236.

**Open, none a factual error:** #224 (triangle still in Extended Data Figs. 4c and 9), #230 (Supplementary
Table 2 wider than the text block), #235 (Extended Data Fig. 8d label at the canvas edge; five dark-centred
open markers in Extended Data Figs. 7 and 8). Untested and stated in the paper: the noisy-synapse Maze
networks and the Maze controls were trained with the first trainer version. Authors' points (grant,
contributions, corresponding author, language-model statement, data deposit, target journal) are listed in
~/Documents/Projects/paper_push/AUTHORS_TO_CONFIRM.md for Pavel.
