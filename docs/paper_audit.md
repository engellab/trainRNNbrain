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
| C2 | Bigger networks keep a smaller share of units active | **Done** (read under the committed rule) | Rebuild F1 panel c (after the cosyne2027-fig merge); re-read if the new read-out rule lands |
| C3 | Longer training keeps shedding units, and no run has stopped | **Done** except one B5 arm | SGD lr 3, wd 0 (Spock, another session) |
| C4 | Silence is a loss of participation that the task gradient does not repair | **Tested** | Pavel chooses the wording |
| C5 | The standard knobs change how many units are lost, not the trend | **Done**, Figure 2 built | Sync Figure 2 into the paper; update old numbers in `04_ratepenalty`, `06_whyitmatters` |
| C6 | Rules that give participation its own drive recruit units at a small cost | **Running** | Read the remedy top-ups (RMtop200) at the floor; B2; A3; build Figure 3 |
| C7 | Participation needs its own drive, likely several; dead units cannot be called back | **Running** | B3 rate cells after the pilot; B4 N = 4000; read all at the floor |

**C1.** N = 1000, three networks per task, each read at its floor. Active units: CDDM about 430 of
1000, DMTS about 350, flip-flop 266-341 (crossings 52-69k on the 200k runs). Removing every silent
unit leaves CDDM unchanged (0.83 → 0.83). On DMTS and the flip-flop the silent units are wired in:
removing them costs R² (flip-flop to 0.22-0.53), but a 3,000-iteration retrain without them
restores it (flip-flop 0.948 against 0.951 for the intact network fine-tuned the same way), and
pruning as many units at random does not (0.921-0.929). The same holds at every size from 500 to
4000 at the end of training (36 networks, supplementary figure). Text and Methods are in the paper;
the seeded numbers are on paper branch `numbers-seeded`.

**C2.** Flip-flop at its floor on the 200k runs (entry 2026-10-15_2059): N = 500 187-237 active
(6 of 6 runs), N = 1000 287-294 (2 of 3), N = 2000 363-410 (3 of 3), N = 4000 500-594 (3 of 3).
The active share falls from about 40% at N = 500 to about 14% at N = 4000; the count grows as
N^0.47. CDDM and DMTS are read at the floor from the paper-grid controls. The learning-rate
confound (lr ∝ N^−1/3) is left in and handled by reading each network at its own floor.

**C3.** CDDM, N = 1000: every weight-decay setting reaches its floor early and keeps losing units to
200-300k with no gain in loss. Flip-flop: units keep dying while the loss is flat (N = 1000: 22-83
units per 100k in 4 of 5 flat runs, slowing to 11-22 by 300k without stopping; N = 2000 is not yet
flat at 200k). **Not the optimizer (B5):** at the floor, Adam (wd 1e-6) keeps 237-271 active at R²
0.951, AdamW 282-292 at 0.961; SGD keeps more units only where it converges to a worse solution
(703-736 active at R² 0.883, lr 0.03) and has Adam's count at its best learning rates. Figure
`diag_b5_optimizer`.

**C4.** On CDDM, 271-322 of 1000 units never cross threshold and get exactly zero gradient on their
incoming weights; the other silent units fire briefly and are reached. On the flip-flop every unit
crosses threshold: silent units fire for a few steps after nearly every pulse, their incoming weights
get gradient, and training keeps them silent anyway. The flip-flop is converged (its loss changes
< 1% per 100k from 150k; the gradient keeps reaching 150-175 silent units from 50k to 300k and after a
ten-fold learning-rate drop). Activations: leaky ReLU, softplus and sigmoid also shed units; tanh
saturates instead. Candidate wordings: `docs/project_trajectory/2026-10-15_1342_open-rules-and-commands-moved-from-agent-notes.md`.

**C5.** Figure 2 (`fig_paper_F2_fails.py`): nine knob ladders plus the CDDM and flip-flop revival
screens, each as R² against active units at the floor beside active units over training. No standard
knob adds more than ~120 units at the floor, except sigmoid, which loses them. CDDM input scale
reaches its floor sooner, then sheds faster; it does not recruit. The rate penalty keeps all 1000
CDDM units at < 1% R² (p ≈ 0.02, n = 5; treated as negligible). Re-initialisation, zero-out and random
redraw of silent units do not rescue participation.

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
- **B3.** Pilot 15598507-08, then the three rate cells by the pre-registered rule; noise and
  rescaling cells 15598515-17.
- **B4.** N = 1000 on Della (15598512-14); N = 4000 on gpu80 is approved, to submit when FFtop4000
  lands.
- **RMtop200.** 189 remedy top-up tasks to 200k with snapshots, from `~/trainRNNbrain_redo`
  (709c17d); job ids in Della `~/f2_logs/rm_submit_go.txt`.
- **Redo with snapshots (waiting on Pavel; after the initialisation decision):** retrain the size-series controls with snapshots every
  1000 iterations (~800 GPU-h). CDDM regenerates exactly; flip-flop and DMTS give new networks.

**Writing**
- **W1.** Manuscript rewrite to this framing: drafted on paper branch `w1-draft`.
- **W2, W3.** "Not brain-like" pinned to measured properties, and the bio-inspired vs engineering
  criteria: merged into the paper's `main`.
- Paper branches `numbers-seeded`, `figure2-methods` and `w1-draft` wait for Pavel's approval to
  merge.

**Should**
- Published continual backprop and ReDo as baselines on the flip-flop (N = 1000, 4000).
- A third seed on the weight-cap and spectral-radius ladders.
- One Dale panel from existing runs.

## Open decisions (Pavel)

- **Conclusion wording.** Recommended: "gradient descent on the task loss alone does not keep units
  participating; participation needs a drive of its own". Say "several sources" only if B3 shows a
  combination beats the rate penalty alone on stability, cost, dimensionality or ablation.
- **C4 wording:** candidates in the 2026-10-15_1342 entry.
- **N = 1000 flip-flop straggler s3374117298:** continue to 300k (~7 GPU-h) or leave excluded.
- **Correlated initialisation** (found 15 Oct): every network's W_out is a rescaled copy of its
  W_inp in draw order, and W_rec's first rows repeat the same draws (`RNN_torch.sparse`; entry
  2026-10-15_1500). Fix it now (a version boundary for all new runs), and test first whether it
  changes silencing (N = 1000 controls with independent draws against the current ones)?
- **Redo with snapshots** (above). It should start after the initialisation decision, so the redo
  is not run twice.
- **Merging the paper branches** (above).

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
