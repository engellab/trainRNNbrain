# Paper audit: claims, status, and what is left

The status board for the silent-units paper. Update it whenever a result changes a claim; record the
result itself as a new entry in `docs/project_trajectory/`. How networks are read (the 1.05×
noisy-loss floor, active units, held-out R²) is in `docs/AGENT_GUIDE.md`; every number here is read
that way unless it says otherwise.

## The paper in one paragraph

Neuroscientists train RNNs on tasks animals do, then read the units like a recorded population.
After training, most units take no part in the computation, and that is not brain-like. The task
loss reads only the output, so nothing in it asks a unit to stay involved, and training sheds the
surplus. A shed ReLU unit goes silent; a shed tanh unit saturates. Bigger networks keep a smaller
share of their units active, and longer training keeps shedding them after the loss has stopped
improving. The standard knobs change how many units are lost, not the trend. Rules that give
participation a drive of its own recruit units: bio-inspired ones (a two-sided firing-rate target,
"the urge to spike, but not to overspike"; noisy synapses; rescaling a silent unit's incoming
weights) and engineering ones (prune-and-duplicate, read-out dropout). As Dohare et al. (2024)
concluded for continual learning, gradient descent on the task is not enough: participation needs
its own drive, probably several.

**Terms.** *Standard network*: see the guide. *Pre-standard*: older sweeps with a trainable bias.
*Multiplier*: an arm's active-unit count ÷ its own control's count, at the same N and read-out.
*Rate penalty* (`frm`): a two-sided penalty around a target firing rate. *Sparsity penalty* (`rws`):
charges a unit for spreading its input over many partners. *Tasks*: 3-bit flip-flop, CDDM
(context-dependent decision making), DMTS (delayed match-to-sample).

**Figures.** F1, the phenomenon: (a) an example network, (b) the participation distribution,
(c) size, (d) the ablation, (e) silencing after the floor. F2, everything that fails. F3, a
bird's-eye view of what works.

## Claims

| # | Claim | Status | What remains |
|---|---|---|---|
| C1 | Training leaves most units silent, and silent units do no work | **Done** | Rebuild F1 with `RETRAIN_DIR` → `c1_retrain_seeded2` |
| C2 | Bigger networks keep a smaller share of units active | **Done**; Figure 1 synced to the paper 15 Oct | — |
| C3 | Longer training keeps shedding units, and no run has stopped | **Done** except one B5 arm | SGD lr 3, wd 0 (Spock, another session) |
| C4 | On CDDM ~300 of 1,000 units are silent (never fire, no gradient); on 3BFF none is, and low-active units get gradient yet stay below threshold | **Done**: in the paper as sec:mechanism (16 Oct) | Paper-wide pass for the silent / low-active terms (Pavel to confirm) |
| C5 | Settings move the active count at the floor by ≤ ~120 units; input-weight scale raises it but those networks then lose units faster | **Done**: sec:knobs rewritten, Figure 2 synced (16 Oct) | — |
| C6 | Rules that give participation its own drive recruit units at a small cost | **Partly done**: remedies read at the floor and A3 done (31 Oct, entry 2026-10-31_1732); B2 failed its rule | Build Figure 3; Pavel to decide how rescaling and read-out dropout are reported above N = 1000 (needs-pavel issue dead-relu-paper#16) |
| C7 | Participation needs its own drive, likely several; dead units cannot be called back | **Running**; B4 supports "cannot be called back" at N = 1000 and 4000 (31 Oct); B3 waits on Pavel (dead-relu-paper#17) | Pavel's B3 decision |

**C1.** N = 1000, three networks per task, each read at its floor. Active units: CDDM about 430 of
1000, DMTS about 350, flip-flop 266-341 (crossings 52-69k on the 200k runs). Removing every silent
unit leaves CDDM unchanged (0.83 → 0.83). On DMTS and the flip-flop the silent units are wired in:
removing them costs R² (flip-flop to 0.22-0.53), but a 3,000-iteration retrain without them
restores it (flip-flop 0.948 against 0.951 for the intact network fine-tuned the same way), and
pruning as many units at random does not (0.921-0.929). The same holds at every size from 500 to
4000 at the end of training (36 networks, supplementary figure). Text, Methods and the seeded numbers
are in the paper.

**C2.** Figure 2 (entry 2026-10-31_1957; Della job 16558986, read-out 79932ef). Read at 1.05× each
run's floor and fitted on the size means from N = 500, the active count grows as N^0.48 on the 3-bit
flip-flop (41% of 500 units active, 14% of 4000), N^0.44 on CDDM (298 of 500, 767 of 4000) and N^0.55 on
DMTS (four sizes now; 191 of 500, 620 of 4000 from the one N = 4000 network that reads out). At
N = 100, 66-90% of units are active (drawn, not fitted). At 1.05×, 1.10× and 1.20× the exponents lie
between 0.44 and 0.61 and every bootstrap 95% upper bound is below 0.79, so the rule of entry
2026-10-31_1900 says C2 holds on all three tasks. 1,000 active units would need N ≈ 7,300 on CDDM,
9,200 on DMTS and 14,200 on the flip-flop. The learning-rate confound (lr ∝ N^−1/3) is left in and
handled by reading each network at its own floor; mean crossings move from 59k to 65k (flip-flop) and
27k to 32k (CDDM) between N = 500 and 4000. The flip-flop N = 1000 straggler is counted, per
dead-relu-paper#20's recommendation, until Pavel decides.

**C3.** CDDM, N = 1000: every weight-decay setting reaches its floor early and keeps losing units to
200-300k with no gain in loss. Flip-flop (Figure 2e, entry 2026-10-31_1957): from 150k, 13 of 15 runs
with a flat loss (< 3% change per 100k) lose 8-121 units per 100k: N = 500 4 of 5, N = 1000 5 of 6
(the two 300k runs 23 and 37 per 100k, slowing to 11-22 in their last 100k, entry 2026-10-15_0955),
N = 4000 3 of 3 (31-121); N = 2000 has one flat run (57 per 100k), so it is untested there. **Not the optimizer (B5):** at the floor, Adam (wd 1e-6) keeps 237-271 active at R²
0.951, AdamW 282-292 at 0.961; SGD keeps more units only where it converges to a worse solution
(703-736 active at R² 0.883, lr 0.03) and has Adam's count at its best learning rates. Figure
`diag_b5_optimizer`.

**C4.** On CDDM about 300 of a network's 1,000 units are silent: their net input current is
negative on every time step of every trial, so their rate is always zero and gradient descent cannot
change their incoming weights. On 3BFF no unit is silent; low-active units (below the activity
threshold, but positive at least transiently) get gradient and their weights change, yet they stay
below the threshold. The 3BFF networks are not still learning: in two of three, the loss changes by
under 1% per 100k from 150k. Text: entry 2026-10-16 (second draft).

**C5.** Changing a hyperparameter shifts the active count at the loss floor, but on CDDM by no more
than about 120 units from the default 418 (sigmoid: 235-255), and the count still falls after the
floor (weight decay 0: 412 active at 200k; 1e-4: 86). On 3BFF, a 5-fold larger initial input-weight
scale raises the count from 298-334 to 547-660 at unchanged R². A metabolic penalty lowers R² without
raising the count. Figure 2 (`fig_paper_F2_fails.py`) shows the nine ladders and the revival screens;
re-initialising, zeroing or randomly redrawing silent units does not raise the count. Details: entry
2026-10-15_2207.

**C6.** The remedy top-ups (RMtop200, 267 runs in 83 cells) are read at 1.05× each run's own
noisy-loss floor (entry 2026-10-31_1732; `read_remedies.py` 2d32b3b). 146 runs are read and 121
excluded. 92 of the exclusions never stay under the bar: their loss is still falling at 200k
(`rm_exclusion_diag.py`). Noisy synapses (σ_w = 1) are read in every run at every size and give
1.81×, 2.03×, 1.54× and 1.60× the control's active count at N = 500, 1000, 2000 and 4000, at an R²
cost of 0.021-0.025 for N ≥ 1000 (0.043 at 500). Prune-and-duplicate gives 2.06-3.59× at a cost of
at most 0.018. Rescaling and read-out dropout have no floor read at N = 2000 or 4000 (0 of 16 runs);
at 75k they give 1.7-2.0×. B2 (rate penalty alone, entry 2026-10-31_1532) fails its pre-registered
rule: no run at N = 500 or 1000 reaches a floor read, and at N = 2000 it costs 0.024 in held-out R²
(0.926 against 0.950). B2's third test, against synaptic noise, does not falsify: 4.81× against
1.54× at N = 2000, and 5.64× against 1.60× at 4000. A3 (75k): removing the recruited units (active
units beyond the control's count, least participating first) lowers R² from 0.93-0.95 to 0.18-0.62
in every remedy at N = 1000 and 4000. For read-out dropout and the rate penalty, that cost equals
removing as many random active units.

**C7.** B4 at N = 1000 (entry 2026-10-31_1532): when a trained flip-flop network learns the 3-bit
hyper flip-flop, 5-9% of its inactive units become active, and it ends with 232-259 active against
421-457 for an untrained network trained on the new task; R² is the same (0.91). Verdict, by the
pre-registered rule: inactive units do not come back. B4 at N = 4000 (entry 2026-10-31_1750) gives
the same verdict: old networks revive 2.3-7.8% of their inactive units and end with 446-613 active
against 806-956 for untrained networks; R² 0.909-0.914 for both. B3 (N = 1000, rate × noise ×
rescaling): the rule binds at the floor, where its literal verdict is "supports several sources",
reached only through its stability clause. Rate+noise reads at the floor and rate alone does not,
because its loss is still falling at 200k. Rate+noise has 970 active units against 556 for noise
alone, at a cost of 0.04 in R². At 75k the verdict is between the bars. Decision with Pavel
(dead-relu-paper#17).

## The remedies catalogue

*Bio-inspired*: uses only the unit's own activity and synapses and could run online. *Engineering*:
copies units, ranks the whole population, or acts outside the dynamics. Flip-flop multipliers are
read at 1.05× each run's own floor (RMtop200, entry 2026-10-31_1732), at N = 1000 and then 4000;
'75k' marks a cell with no floor read at that size.

| Remedy | Class | What it does | Active units | Status |
|---|---|---|---|---|
| Rate penalty | Bio-inspired | Two-sided penalty around a target rate | CDDM: all 1000; flip-flop 4.8× at N = 2000, 5.6× at 4000 | Flip-flop alone (B2): unstable at N ≤ 1000, −0.024 R² at N = 2000; its multiplier is above synaptic noise's |
| Noisy synapses | Bio-inspired | Each weight × (1 + σ_w·noise), every step | 2.03×; 1.60× at N = 4000 (read in 14 of 14 runs, N = 500-4000) | −0.021 to −0.025 R² at N ≥ 1000 |
| Rescaling incoming weights | Bio-inspired | Tilts a silent unit's input row toward its positive weights, at fixed length, until a release target | 1.51× (1 of 4 runs read); 1.95× at N = 4000 at 75k, no floor read | No floor read at N ≥ 2000: loss still falling at 200k (dead-relu-paper#16) |
| Threshold lift | Bio-inspired | Raises a silent unit's bias once | 1.36× | Screen only |
| Prune-and-duplicate | Engineering | Overwrites a silent unit with a jittered copy of a working one | 2.77×; 3.59× at N = 4000 | −0.005 R² at N = 1000, −0.018 at 4000; fails on DMTS above N = 500 |
| Read-out dropout | Engineering | Randomly hides busy units from the read-out | 1.81× (1 of 3 runs read); 2.01× at N = 4000 at 75k, no floor read | No floor read at N ≥ 2000 (dead-relu-paper#16); fails on DMTS |
| Full dropout | Engineering | Randomly removes busy units from the dynamics | 904/1000 | Parked: costs 12-17 R² points |
| Redraw silent units | Engineering (≈ continual backprop / ReDo) | Re-initialises a silent unit | 0.84-0.96× | Negative (Figure 2) |
| Rate + sparsity | Bio-inspired pair | Both penalties | 3.51×; 5.87× at N = 4000 (1 run) | Parked; the rate penalty alone replaces it |

## Parked (do not restart without Pavel)

- The selective/modular solution under rate + sparsity: burst units, assemblies, penalty switching.
- Single-unit selectivity, clustering, cell types, NMF, wiring-versus-activity.
- The task-demand ceiling (M*, M ∝ N^b k^c), the Walsh and hyper flip-flops as a demand series,
  MultiRule15, Yang 20-task (except the A5 supplementary panel).
- Long-delay DMTS.
- Master-inhibitor, frozen-clamp and silent-at-init rescue tests.
- Dale-constrained and sign-constrained networks (one supplementary panel at most).
- Loss-floor laws, convergence criteria, drift exponents, lognormality of rates and weights.

## Open work

**Analyses (no new training)**
- **A3.** Done 31 Oct (entry 2026-10-31_1732). Recruited units carry task signal in every remedy.
  Remedy networks lose less to random 10% removal (median R² 0.58-0.86 against 0.48 for the control
  at N = 1000).
- **A5.** Supplementary task-demand panel (`fig_supp_task_demand`) is drafted on pre-standard data:
  harder tasks recruit more units, no solved task fills the network. Open: a re-run on the standard
  network.

**Training**
- **B2.** Done, read 31 Oct: fails its rule (entry 2026-10-31_1532). Third test (against synaptic
  noise) does not falsify.
- **B3.** Read 31 Oct (entry 2026-10-31_1750, Della job 16553903). At the floor: literal "supports",
  through the stability clause only. At 75k: between the bars. Rate+rescale skips 44-67% of updates
  and the triple 8-11%. Under today's read-out the λ = 0.03 pilot rung reads at 0 of 3 runs (2 of 3
  when it was chosen). Stage 2 not run; waits on dead-relu-paper#17.
- **B4.** Done. N = 1000 and N = 4000 both: inactive units do not come back (entries
  2026-10-31_1532 and 2026-10-31_1750).
- **RMtop200.** Read 31 Oct (entry 2026-10-31_1732).
- **Redo with snapshots (waiting on Pavel; after the initialisation decision):** retrain the size-series controls with snapshots every
  1000 iterations (~800 GPU-h). CDDM regenerates exactly; flip-flop and DMTS give new networks.

**Writing**
- **W1.** Manuscript rewrite to this framing: merged into the paper's `main` (15 Oct); the C4
  passage in `01_intro.tex` still awaits its wording.
- **W2, W3.** "Not brain-like" pinned to measured properties, and the bio-inspired vs engineering
  criteria: merged into the paper's `main`.

**Should**
- Published continual backprop and ReDo as baselines on the flip-flop (N = 1000, 4000).
- A third seed on the weight-cap and spectral-radius ladders.
- One Dale panel from existing runs.

## Open decisions (Pavel)

- **Which biological penalty goes in the main text.** The decision "rate penalty alone" rested on
  CDDM. On the flip-flop (B2) the same penalty trains unstably at N ≤ 1000 and costs 0.024 R² at
  N = 2000. Options: keep it with the flip-flop failure stated; lower λ_frm (B3's pilot chose 0.03
  at N = 1000); or lead with synaptic noise, the only remedy read at the floor in all 14 runs at
  N = 500-4000 (1.5-2.0×, R² cost 0.021-0.025 at N ≥ 1000). Rescaling has no floor read above
  N = 1000 (dead-relu-paper#16).
- **Rescaling and read-out dropout above N = 1000** (dead-relu-paper#16): report at 75k, labelled, with the 1.20×
  crossing in the supplement (recommended), or continue the 16 runs past 200k.
- **Figure 1, flip-flop N = 1000:** FFseed4 landed; the cell now reads 4 runs at the floor (240,
  277, 296, 297). Rebuild Figure 1 with all four, or keep the straggler excluded.

- **Conclusion wording.** Recommended: "gradient descent on the task loss alone does not keep units
  participating; participation needs a drive of its own". B3's literal floor verdict "supports
  several sources" rests only on rate+noise reaching a floor read where rate alone, whose loss is
  still falling at 200k, does not; the cost is 0.04 R². At 75k it is between the bars. Options and
  recommendation (keep "a drive of its own") in dead-relu-paper#17.
- **Read-out limits found in B3, not patched:** non-finite losses are dropped before bursts are
  counted, and floor R² can be taken on a snapshot inside a burst (dead-relu-paper#18). They may
  affect the bursting remedy cells (B2, B3, RMtop200), not B4.
- **C4 wording:** candidates in the 2026-10-15_1342 entry.
- **Does the old correlated initialisation change silencing?** 3 seeds at N = 1000, CDDM and the
  flip-flop, independent draws against the current controls (~45 GPU-h); entry 2026-10-15_2206.
- **Redo with snapshots** (above). It should start after the initialisation decision, so the redo
  is not run twice.

## Decisions taken

- Read-out: the noisy training loss, 1.05× its fitted floor, with the smoothed-minimum fallback for
  plateaued fits; never the noise-free loss.
- Standard network only; the 3-bit flip-flop trains to 200k.
- The biological penalty in the main text is the rate penalty alone; rate + sparsity goes to the
  supplement.
- Two main tasks: CDDM carries the knobs, the gradient analysis and tanh; the flip-flop carries the
  remedies. DMTS appears as a third series on the size curve and in claim 1.
- B4 is in the paper, in a minimal form (N = 1000 and 4000, one new task, 3 seeds).
- The rate penalty's R² cost on CDDM (< 1%) is treated as negligible.
- Held-out scoring is seeded project-wide; new runs keep snapshots every 1000 iterations.
- Initial weight matrices are drawn independently from 15 Oct (`init_draws: independent`).
- The paper branches numbers-seeded, figure2-methods and w1-draft are merged (15 Oct).
- The N = 1000 flip-flop straggler stays excluded; a new seed replaces it (Della 15657728_4).
