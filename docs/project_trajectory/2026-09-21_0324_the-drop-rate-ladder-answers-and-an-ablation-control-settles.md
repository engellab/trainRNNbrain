## 2026-09-21 03:24 — The drop-rate ladder answers, and an ablation control settles the capacity question

Both overnight jobs landed. Two results, one negative and one positive, and the negative is the more
useful of the two.

### The drop-rate ladder (Della 14214329) — pre-registered, and no arm clears the bar

3-bit flip-flop, N = 1000, pen=none, `dead`, 40,000 iterations, read against the bar fixed before
submission (reference + 3 reference SD under BOTH criteria: 571 scale-free, 567 absolute):

| arm | active units (scale-free) | n |
|---|---|---|
| no dropout | 387.0 ± 37.3 | 4 |
| ρ = 0.05 (standard) | 487.0 ± 28.0 | 7 |
| ρ = 0.10 | 498.3 ± 13.1 | 3 |
| ρ = 0.20 | 471.0 ± 26.9 | 2 |
| ρ = 0.40 | 511.0 ± 12.5 | 3 |
| ρ = 0.05, β = 4 | **394.7 ± 71.7** | 3 |

**An eightfold increase in the drop rate is worth +24 units**, inside the reference's own seed
spread. And sharper targeting is worse than useless: β = 4 falls to 394.7, barely above the 387.0 of
no dropout at all.

This is the outcome the launcher's pre-registration named as the falsifying one, and it is coherent
with the sampler-blindness result measured yesterday. Raising ρ draws more units from the same
badly-ordered list, so roughly half the extra drops still land on units that were already silent.
Sharpening β concentrates the drops onto the units the sampler scores highest — which are not the
units that are firing. **The limiting factor is not how much dropout removes but what it chooses.**
The implied fix, scoring units by rate rather than by |x|, is untested and is flagged as such.

One cell is n = 2 (ρ = 0.20); the twelfth task was still running. The Spock twin (6307716) is still
pending and can be cancelled or left as a replicate.

### The ablation control — the capacity question, answered offline

`trainRNNbrain/experiments_and_analysis/ablate_rescued_units.py`. Units are deleted from a trained
network least-active-first by zeroing each unit's column of `W_rec` and of `W_out` — its influence on
the network and on the read-out is gone, the survivors' dynamics untouched — and the task is
re-scored with **no retraining**. CDDM, N = 1000, n = 3 per arm:

| fraction deleted | unpenalised r² | frm r² |
|---|---|---|
| 0% | 0.9245 | 0.9290 |
| 30% | 0.9245 | 0.7828 |
| 60% | 0.9245 | 0.1457 |
| 80% | **0.9220** | −0.3189 |

**The unpenalised network loses 80% of its units for 0.003 in r².** Roughly 730 of 1000 units can be
deleted at no measurable cost — which is what "silent" means, demonstrated causally rather than by
counting. The rate-penalised network has no such reserve: 30% costs 0.15 and 60% costs the task.

Both networks start at the same performance (0.9245 vs 0.9290), so this is **not** a claim that the
penalised network is better. It is a claim about where the work is done. This is the experiment the
2026-09-20 figure review called "the highest-value missing experiment" and that the referee report
this session called decisive; it is now Fig. 5e.

### Two corrections forced by review, both in the paper's favour

**The mechanism was ReLU-specific and the observation is not** (Pavel). Checked against the
activation sweep: leaky ReLU removes the absorbing zero entirely and moves the count by +6 ± 14,
nothing; softplus and sigmoid remove the scale symmetry and silence gets *worse* (−22, −66). The two
activations that HAVE the symmetry have the MOST active units. Neither ReLU-specific ingredient is
causal. What survives is general: the task loss reads only the output, so a solution on 260 units
scores exactly as well as one on 1000, and nothing in the objective asks for the rest.

**Figure 3d was applying one read-out rule to the treatment and another to the control** (referee).
At 36 tau it plotted frm's BEST checkpoint (+0.2368) against the control's plateau. At matched
compute frm ends **0.409 BELOW** a baseline that never learned the task. Both are now plotted, the
convention is in Methods, and the 997 ± 4 active units at that delay are flagged as measured on a
network whose clean r² has already collapsed.

### Circularity check, which the draft had been asserting in the wrong place

A rate penalty compresses the rate distribution, so a *relative* silence criterion could count
everyone by construction. The absolute threshold — Otsu-calibrated on the UNPENALISED task, blind to
the penalty — gives the same answer: frm 979.0 ± 8.0 under both criteria, frm+rws 1000.0 ± 0.0 under
both. The earlier draft demonstrated criterion agreement only at the baseline cell, i.e. exactly
where it was not needed.

Burst accounting at the main-text cell (N = 1000, n = 3): none 9.0 ± 1.5%, frm 25.4 ± 1.7%,
frm+rws 5.1 ± 0.4% of active units. So of frm's 979 units ~730 are sustained — still 3× the
unpenalised network's ~240 even if every transient unit is discarded.

### Manuscript state

33 pages, 0 LaTeX errors, 0 undefined references, 0 undefined citations, 0 float warnings. Five
figures, all regenerable. refs.bib 114 entries, 63 cited. Methods 3,759 words. TODO markers 27 → 10
(DOI, funding, CRediT, a task-schematic supplementary figure). `docs/measured_facts.md` is the
single numerical source; `docs/critique_clarity.md` and `docs/critique_referee.md` hold the two
independent reviews, with the accuracy review still to run (it hit a session limit).

**Open:** main text is 6,691 words against the journal's 5,000. The overage is content, not padding;
`critique_clarity.md` §F identifies where it is and which results overlap.

