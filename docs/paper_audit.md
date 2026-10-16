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
| C4 | On CDDM ~300 of 1,000 units are silent (never fire, so no gradient); on 3BFF none is, and low-active units get gradient yet stay below threshold | **Tested**; second draft 16 Oct | Pavel: approve the C4 text, and the active / low-active / silent terms paper-wide |
| C5 | Hyperparameters shift the active count at the floor by ≤ ~120 units on CDDM; input-weight scale raises it on 3BFF | **Done**, Figure 2 built | Sync Figure 2 into the paper; update old numbers in `04_ratepenalty`, `06_whyitmatters` |
| C6 | Rules that give participation its own drive recruit units at a small cost | **Running** | Read the remedy top-ups (RMtop200) at the floor; B2; A3; build Figure 3 |
| C7 | Participation needs its own drive, likely several; dead units cannot be called back | **Running** | B3 rate cells after the pilot; B4 N = 4000; read all at the floor |

**C1.** N = 1000, three networks per task, each read at its floor. Active units: CDDM about 430 of
1000, DMTS about 350, flip-flop 266-341 (crossings 52-69k on the 200k runs). Removing every silent
unit leaves CDDM unchanged (0.83 → 0.83). On DMTS and the flip-flop the silent units are wired in:
removing them costs R² (flip-flop to 0.22-0.53), but a 3,000-iteration retrain without them
restores it (flip-flop 0.948 against 0.951 for the intact network fine-tuned the same way), and
pruning as many units at random does not (0.921-0.929). The same holds at every size from 500 to
4000 at the end of training (36 networks, supplementary figure). Text, Methods and the seeded numbers
are in the paper.

**C2.** Read at 1.05× each run's floor under the new read-out (entry 2026-10-15_2219), the active
count grows as N^0.47 on the 3-bit flip-flop (42% of 500 units active, 14% of 4000), N^0.44 on CDDM
(298 of 500, 767 of 4000) and N^0.43 on DMTS (three sizes). The exponents stay between 0.41 and 0.57
at margins of 1.05×, 1.10× and 1.20×. 1,000 active units would need N ≈ 7,300 on CDDM and ≈ 15,100 on
the flip-flop. The learning-rate confound (lr ∝ N^−1/3) is left in and handled by reading each
network at its own floor.

**C3.** CDDM, N = 1000: every weight-decay setting reaches its floor early and keeps losing units to
200-300k with no gain in loss. Flip-flop: units keep dying while the loss is flat (N = 1000: 22-83
units per 100k in 4 of 5 flat runs, slowing to 11-22 by 300k without stopping; N = 2000 is not yet
flat at 200k). **Not the optimizer (B5):** at the floor, Adam (wd 1e-6) keeps 237-271 active at R²
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

**C6.** The flip-flop remedy cells are being continued to 200k with snapshots (RMtop200), so each
can be read at its floor. Until then the remedy numbers below are end-of-training reads at 75k.

**C7.** B3 (rate × synaptic noise × rescaling at N = 1000) and B4 (plasticity transfer to the 3-bit
hyper flip-flop) are running on Della; their pre-registered rules are in the 2026-10-15_1342 entry.

## The remedies catalogue

*Bio-inspired*: uses only the unit's own activity and synapses and could run online. *Engineering*:
copies units, ranks the whole population, or acts outside the dynamics. Flip-flop multipliers are
N = 1000 at 75k (end of training, to be replaced by floor reads from RMtop200).

| Remedy | Class | What it does | Active units | Status |
|---|---|---|---|---|
| Rate penalty | Bio-inspired | Two-sided penalty around a target rate | CDDM: all 1000 | B2 runs it alone on the standard flip-flop grid |
| Noisy synapses | Bio-inspired | Each weight × (1 + σ_w·noise), every step | 2.08×; 1.65× at N = 4000 | In RMtop200 |
| Rescaling incoming weights | Bio-inspired | Tilts a silent unit's input row toward its positive weights, at fixed length, until a release target | 1.57×; 1.94× at N = 4000 | In RMtop200 |
| Threshold lift | Bio-inspired | Raises a silent unit's bias once | 1.36× | Screen only |
| Prune-and-duplicate | Engineering | Overwrites a silent unit with a jittered copy of a working one | 2.92×; 3.70× at N = 4000 | In RMtop200; fails on DMTS above N = 500 |
| Read-out dropout | Engineering | Randomly hides busy units from the read-out | 1.95× | In RMtop200; fails on DMTS |
| Full dropout | Engineering | Randomly removes busy units from the dynamics | 904/1000 | Parked: costs 12-17 R² points |
| Redraw silent units | Engineering (≈ continual backprop / ReDo) | Re-initialises a silent unit | no gain | Negative (Figure 2) |
| Rate + sparsity | Bio-inspired pair | Both penalties | 3.39× | Parked; the rate penalty alone replaces it |

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
- **A3.** Ablate the recruited units for every remedy, plus random 10/30/50% ablation curves, in
  held-out R², once RMtop200 is read at the floor.
- **A5.** Supplementary task-demand panel (`fig_supp_task_demand`) is drafted on pre-standard data:
  harder tasks recruit more units, no solved task fills the network. Open: a re-run on the standard
  network.

**Training**
- **B2.** Rate penalty alone on the standard flip-flop grid, N = 500-2000 on Della (15598509-11),
  N = 4000 (15527679).
- **B3.** Pilot read (16 Oct): λ = 0.03 chosen (72% of units active, 2.4× the control). Rate cells
  submitted 15689235-37; noise and rescaling cells 15598515-17 running.
- **B4.** N = 1000 on Della (15598512-14); N = 4000 on gpu80 is approved, to submit when FFtop4000
  lands.
- **RMtop200.** 189 remedy top-up tasks to 200k with snapshots, from `~/trainRNNbrain_redo`
  (709c17d); job ids in Della `~/f2_logs/rm_submit_go.txt`.
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

- **Conclusion wording.** Recommended: "gradient descent on the task loss alone does not keep units
  participating; participation needs a drive of its own". Say "several sources" only if B3 shows a
  combination beats the rate penalty alone on stability, cost, dimensionality or ablation.
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
