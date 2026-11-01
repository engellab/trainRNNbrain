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
| C1 | Training leaves most units inactive, and a network retrained without them scores within 0.004 R² of the intact network retrained the same way | **Done**: Figure 1 rebuilt (a-d) and panel d re-run at the current floor, rule passed (31 Oct) | Pavel: confirm the 3BFF N = 1000 straggler stays excluded (it now reads out: 277 active) |
| C2 | Bigger networks keep a smaller share of units active | **Done**; Figure 1 synced to the paper 15 Oct | — |
| C3 | Longer training keeps shedding units, and no run has stopped | **Done** except one B5 arm | SGD lr 3, wd 0 (Spock, another session) |
| C4 | On CDDM ~300 of 1,000 units are silent (never fire, no gradient); on 3BFF none is, and low-active units get gradient yet stay below threshold | **Done**: in the paper as sec:mechanism (16 Oct) | Paper-wide pass for the silent / low-active terms (Pavel to confirm) |
| C5 | Settings move the active count at the floor by ≤ ~120 units; input-weight scale raises it but those networks then lose units faster | **Done**: sec:knobs rewritten, Figure 2 synced (16 Oct) | — |
| C6 | Rules that give participation its own drive recruit units at a small cost | **Running**; B2 failed its rule (31 Oct) | Read the remedy top-ups (RMtop200) at the floor; A3; build Figure 3 |
| C7 | Participation needs its own drive, likely several; dead units cannot be called back | **Running**; B4 N = 1000 supports it (31 Oct) | B3 rate cells; B4 N = 4000; read all at the floor |

**C1.** Figure 1 (round 1, entries 2026-10-31_1702 and 1755). Active units at N = 1000, each run of
the size series read at 1.05× its floor (read-out 79932ef): 3BFF 278 ± 33 (296, 297, 240; the
straggler s3374117298 stays excluded and would read 277), CDDM 431 ± 10, DMTS 218 and 422 (a third
run has no read-out), at held-out R² 0.95, 0.83 and 0.999 (estimated from the end of training).
Ablation, re-run on 31 Oct at each network's current crossing on
three snapshot networks per task: removing every inactive unit leaves CDDM at 0.83 (changes of
−0.003 to 0.000) and lowers 3BFF from 0.95 to 0.26-0.52 and DMTS from 0.999 to 0.75-0.99; as many
random units take every network to 0.09 or below. A 3,000-iteration retrain without the inactive
units ends within 0.004 of the intact network retrained the same way on every task (3BFF 0.945-0.948
against 0.949-0.950); random pruning plus retrain reaches 0.920-0.930 on 3BFF but costs nothing on
CDDM (0.821-0.828); on DMTS it reaches 0.998 in two networks and 0.419 in s3322309204. The pre-registered rule (entry 1702) passes in all nine networks. At the end of
training, 36 networks of 500-4000 units (Extended Data Fig. 2): removing the inactive units changes
R² by −0.010 to +0.001 on CDDM and −0.046 to −0.001 on DMTS.

**C2.** Read at 1.05× each run's floor under read-out 79932ef (entry 2026-10-16, Figures 1 and 2
under read-out 79932ef), the active count grows as N^0.47 on the 3-bit flip-flop (41% of 500 units
active, 14% of 4000), N^0.44 on CDDM (298 of 500, 767 of 4000) and N^0.58 on DMTS (three sizes; 2 of 3
runs read at N = 1000). The exponents stay between 0.44 and 0.60 at margins of 1.05×, 1.10× and 1.20×.
1,000 active units would need N ≈ 7,300 on CDDM and ≈ 14,300 on the flip-flop. The learning-rate
confound (lr ∝ N^−1/3) is left in and handled by reading each network at its own floor.

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
B2 (rate penalty alone, flip-flop, entry 2026-10-31_1532) fails its pre-registered rule: no run at
N = 500 or 1000 reaches a floor read (the loss bursts 57-67 times per run and the active count swings
between ~300 and 1,000), and at N = 2000 it holds 4.8× the control's active count at a cost of 0.024
in held-out R² (0.926 against 0.950).

**C7.** B4 at N = 1000 (entry 2026-10-31_1532): when a trained flip-flop network learns the 3-bit
hyper flip-flop, 5-9% of its inactive units become active, and it ends with 232-259 active against
421-457 for an untrained network trained on the new task; R² is the same (0.91). Verdict, by the
pre-registered rule: inactive units do not come back. B4 at N = 4000 (1 of 9 runs done) and B3 are
running on Della; their rules are in the 2026-10-15_1342 entry.

## The remedies catalogue

*Bio-inspired*: uses only the unit's own activity and synapses and could run online. *Engineering*:
copies units, ranks the whole population, or acts outside the dynamics. Flip-flop multipliers are
N = 1000 at 75k (end of training, to be replaced by floor reads from RMtop200).

| Remedy | Class | What it does | Active units | Status |
|---|---|---|---|---|
| Rate penalty | Bio-inspired | Two-sided penalty around a target rate | CDDM: all 1000; flip-flop 4.8× at N = 2000 | Flip-flop alone (B2): unstable at N ≤ 1000, −0.024 R² at N = 2000 |
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
- **B2.** Done, read 31 Oct: fails its rule (entry 2026-10-31_1532).
- **B3.** Pilot read (16 Oct): λ = 0.03 chosen (72% of units active, 2.4× the control). Rate cells
  submitted 15689235-37; noise and rescaling cells 15598515-17 running.
- **B4.** N = 1000 read 31 Oct: inactive units do not come back (entry 2026-10-31_1532). N = 4000
  (15655301-03): 1 of 9 runs done; read with `read_b2_b4.py` when the fresh runs land.
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

- **Which biological penalty goes in the main text.** The decision "rate penalty alone" rested on
  CDDM. On the flip-flop (B2) the same penalty trains unstably at N ≤ 1000 and costs 0.024 R² at
  N = 2000. Options: keep it with the flip-flop failure stated; lower λ_frm (B3's pilot chose 0.03
  at N = 1000); or lead with synaptic noise or rescaling once RMtop200 is read.
- **Figure 1, flip-flop N = 1000:** FFseed4 landed; the cell now reads 4 runs at the floor (240,
  277, 296, 297). Rebuild Figure 1 with all four, or keep the straggler excluded.

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
