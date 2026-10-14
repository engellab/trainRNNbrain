# Paper audit: where the project stands and what is left to publish

*Round 1, 9 October 2026. This document supersedes the abstract and storyline in `paper.md`, which
tell the older penalty/selectivity story. The full record stays in `project_trajectory.md`.*

## Round 2 (12 October): decisions taken, and claim 1 on standard networks

**Decisions (PI).**
- **Three figures.**
  - **F1** is the phenomenon only: (a) an example network, (b) the participation distribution, (c) size, (d) the ablation, (e) silencing after the floor.
  - **F2** is everything that failed. The old F1 panel d ("no knob fixes it") moves there.
  - **F3** is a bird's-eye view of what worked to some extent.
- **Read-out** is tightened to **1.05×** each network's own fitted loss floor (was 1.07×).
- **Learning-rate confound** in the size series (lr ∝ N^−1/3) is left in. It is handled by reading each network at its own floor.

- **Read-out is on the noisy loss only** (the loss the network trains on; `common.noisy_floor`). The
  noise-free loss is retired: on the standard flip-flop it bottoms out near 10k iterations and climbs
  back, while the noisy loss keeps falling.

**Claim 1, done on standard networks** (details in the trajectory, 2026-10-12):

| N = 1000 | Silent | Remove all silent units | Remove as many at random | Prune silent units, retrain 3k iterations |
|---|---|---|---|---|
| CDDM, at its floor (~25k) | 57% | no change (0.83 → 0.83) | −0.03 | stays at 0.83, same as the intact network fine-tuned |
| DMTS, at its floor (17–42k) | 65% | 0.999 → 0.95 | −0.12 | back to **0.999** |
| Flip-flop, at its floor (27–69k, 200k runs) | 68% | 0.95 → 0.38 | 0.05 | back to **0.94** (random: 0.92) |

- **No task needs its silent units.** CDDM's do nothing. The flip-flop's, and DMTS's right after
  its escape, are wired in: removing them costs something, but a short retrain without them restores
  the full score. Random pruning of the same number does not.
- The same holds at every size from 500 to 4000 at the end of training (36 networks, supplementary
  figure).
- **Figure 1 is rebuilt** on standard networks, with panel d as the ablation (`fig_paper_F1.py`,
  `fig_supp_ablation.py`).

**New finding: the flip-flop had not converged at 75k.** Its noisy loss was still falling in every
run. All standard flip-flop controls are being continued to 200k: at N = 500 and 1000, 8 of 9 runs
now reach their floor (38–111k). Panel c's flip-flop series waits on N = 2000 and 4000 (~2 days). CDDM converges by about 25–30k and DMTS
within about 15k of escaping its stall.

## Status board at hand-off (13 October)

The plan below (§1–§8) is round 1's; this board says where each piece stands. Hand-off notes:
`docs/HANDOFF.md`.

**Rules now in force** (Pavel): read every network where its *noisy* training loss first reaches
1.05× its own fitted floor (`common.noisy_floor`; fit from the escape for DMTS; excluded if the
median of the last 10% ends > 1.5× the floor). Performance is held-out r² under the training noise.
Never the noise-free loss. Standard network only (ReLU, bias fixed at 0, self-connections on, no
Dale, γ = 0). The 3-bit flip-flop is trained to 200k (it has not converged at 75k).

**Figures:** F1 = the phenomenon (done for claim 1). F2 = what fails (to build). F3 = bird's-eye
view of what worked (to build).

| Item | Status | What remains |
|---|---|---|
| C1 silent, and not needed | **Done.** Figure 1 a, b, d; supplementary ablation; text and Methods in the paper repo | — |
| C2 size (B1) | **Partly.** CDDM and DMTS read at the noisy floor from the paper-grid controls | The flip-flop series at its floor once its 200k top-ups land (N = 2000 done 13 Oct, N = 4000 ~15 Oct); rebuild panel c; update the numbers in the text. The learning-rate crossover is replaced by the floor read-out (Pavel). |
| C3 never stops | **Partly.** Panel e on standard runs (flip-flop 200k, CDDM 200k, DMTS 150k). **B5 submitted 13 Oct** on Spock (job 6928407, branch `b5-optimizer` at b8221f8): 3-bit flip-flop, N = 1000, 200k, 3 seeds each of Adam wd 1e-6 (control) and wd 0, AdamW wd 1e-6, and SGD at wd 1e-6 and at wd 0 (job 6928465, added later), each at base lr 0.03 / 0.1 / 0.3 / 1 / 3 / 10. Output `NBitFlipFlop_b5_optimizer`, mirrored to Della by `slurm/sync_b5_spock_to_della.sh`. One control seed blew up at 1.4k (loss 5e6) and recovered by 2.1k (0.16) | Read B5 at the noisy floor and at the end (~1 day); replacement seeds for any arm left with fewer than 3 that train |
| C4 loss of gradient (A2) | **Partly.** Gradient reach run on every size and activation (2026-10-13): on CDDM the units the gradient reaches are the active units at N = 500-4000 and at the floor; DMTS close; on the flip-flop the gradient still reaches the incoming weights of 1.5-4.5x as many units as are active | Decide how C4 is worded for the flip-flop; test why its silent units keep incoming gradient; fix the untrained twin for warm-started runs |
| C5 knobs → Figure 2 | **Partly.** All seven knob ladders read at the noisy floor (`fig_knobs_at_floor.py`, 2026-10-13): at the floor every CDDM knob but sigmoid (245) is within ~120 of the reference's 418; they spread only at the end; flip-flop input scale is the exception (316 → 717 at the floor). Held-out R² at the floor is drawn against active units for every run (`--tradeoff`, `fig_f2_knob_tradeoff`): no CDDM knob adds more than ~120 units at the floor; flip-flop input scale adds ~300-600 at no R² cost. The revival screens (`CDDM_screen3`, `NBitFlipFlop_replace_variants`) stopped at 30k/40k and have no floor read-out | Revival-screen top-ups running (15499043-48, CDDM → 75k, flip-flop → 200k); read them into the scatter; build F2; `sec:knobs` redrafted on paper branch `knobs-at-floor` (Pavel to review); fix the old knob numbers quoted in `04_ratepenalty`, `06_whyitmatters` |
| C6 remedies → Figure 3 | Not started under the new rule. A1 is obsolete. **Top-ups to 200k submitted 13 Oct** (Della, `RMtop200` 15500244-15500425): all 71 non-control flip-flop cells `f2_remedies_cache.discover` finds at 75k, 227 runs, from `~/trainRNNbrain_c1top` (7739a60; its training code equals the `stdfix` tree that ran 40k → 75k). Kept snapshots (`KEEP=1`) only on the five N = 1000 operating points (dropout 0.20/β4, duplication, rescale target 10, synaptic noise 1.0, rate + sparsity); N = 4000 runs need up to ~5 days | Read at the floor when they land; B2 (rate penalty alone); A3 (ablate the recruited units, using `ablate_silent_units.py`) |
| C7 several drives (B3), plasticity (B4) | Not started | B3, B4 |
| A4 unread top-ups, A5 demand panel | Not started | — |
| W1 manuscript | **Partly.** Claim 1 is written; the abstract, intro and discussion still tell the old story | Rewrite to the framing in §1 |
| W2 not brain-like, W3 bio vs engineering | Not started | — |
| Decision 1 (read-out) | **Resolved:** noisy loss, 1.05×, the guards above | — |
| Decisions 2–5 | Open | Ask Pavel |

## 1. The paper in one paragraph

Neuroscientists train RNNs on tasks animals do, then read the units like a recorded population.
After training, most units take no part in the computation, and that is not brain-like. The task
loss reads only the output, so nothing in it asks any unit to stay involved, and training sheds the
surplus. A shed ReLU unit goes silent; a shed tanh unit saturates. Either way the gradient stops
reaching it, so backpropagation cannot bring it back. **Bigger networks are emptier:** the share of
working units falls from 52–65% at N = 500 to 13–19% at the largest sizes. **Longer training makes it
worse and it does not stop:** CDDM reaches its loss floor by ~7,000 iterations and is still losing
units at 300,000. The standard knobs change how many units are lost, not the trend. Rules that give
participation a drive of its own do recruit units. Bio-inspired ones are a two-sided firing-rate
target ("urge to spike, but not to overspike"), noisy synapses, rescaling a silent unit's incoming
weights, and lifting its threshold. Engineering ones are prune-and-duplicate and read-out dropout.
As Dohare et al. (2024) concluded for continual learning, gradient descent on the task is not
enough: participation needs its own source, probably several.

**Terms.** *Standard network*: ReLU, bias fixed at 0, self-connections on, no Dale constraint.
*Pre-standard*: older sweeps that trained the bias. *Multiplier*: an arm's active-unit count ÷ its
own control's count, at the same N and read-out. *Rate penalty* (`frm`): a two-sided penalty around
a target firing rate. *Sparsity penalty* (`rws`): charges a unit for spreading its input over many
partners. *Tasks*: 3-bit flip-flop, CDDM (context-dependent decision making), DMTS (delayed
match-to-sample).

## 2. The storyline

| # | Claim | Status | Next (§7) |
|---|---|---|---|
| C1 | Training leaves most units silent, and silent units do no work | Partly | A3 |
| C2 | Bigger networks are emptier | Partly | B1, A2 |
| C3 | Longer training makes it worse, and no run has stopped | Done on CDDM | B5 |
| C4 | ReLU silence and tanh saturation are one failure: the gradient stops reaching the unit | Partly | A2 |
| C5 | The standard knobs change how many units are lost, not the trend | Done | A4 |
| C6 | Rules that give participation its own drive recruit units at a small cost | Partly | A1, B2, A3 |
| C7 | The task gradient cannot hold a unit up; participation needs its own drive, likely several | Partly | B3, B4 |

**C1. Training leaves most units silent, and silent units do no work.**
- Flip-flop, N = 1000: 940 of 1000 units work before training, 262 after 150k iterations. CDDM ends at 271.
- On all three tasks the units split into a working group and a quiet one that fires thousands of times less.
- The dormancy score of Sokar et al. (2023), from deep RL, picks the same units: 316 vs our 262 (flip-flop), 276 vs 271 (CDDM).
- Deleting the least-active 80% of a CDDM network (N = 1000, 3 networks) moves r² only from 0.9245 to 0.9220.
- Figures: `slide_m2_training_empties`, `slide_02_participation_by_task_matched`, `slide_m3_two_criteria`.
- *Gap:* the ablation was run once, on pre-standard networks, and has no figure.

**C2. Bigger networks are emptier.**
- Four series (3-bit and 6-bit flip-flop, CDDM, DMTS), 2–3 seeds per size, each run read at its own loss floor.
- The working share falls from 52–65% at N = 500 to 13–19% at the largest size (4000 flip-flop, 5000 CDDM, 2000 DMTS).
- The count still grows, but far slower than N (fitted exponents 0.32–0.45): CDDM goes from 327 working units at N = 500 to 945 at N = 5000.
- The extra units buy nothing: the flip-flop's four sizes reach the same loss floor to within 3%.
- Figures: `slide_06_scaling_fraction`, `fig_paper_F1` panel c.
- *Gap:* every series is pre-standard, and the learning rate falls with N (lr ∝ N^−1/3), so size and step size are confounded.

**C3. Longer training makes it worse, and no run has stopped.**
- CDDM, N = 1000, standard network: every weight-decay setting reaches its floor by ~7k iterations with 482–580 units working.
- At 200k the loss is no better, and the counts are 412 / 266 / 166 / 86 for weight decay 0 / 1e-6 (default) / 1e-5 / 1e-4.
- With no weight decay (3 seeds) the count goes from 430–436 at 100k to 384–397 at 300k, still falling 10–14 units per 100k.
- Flip-flop, N = 1000: 262 at 150k → 190 at 500k. On the matched grid the control goes 296 → 274 between 40k and 75k, all 6 seeds falling.
- Figure: `slide_k_weightdecay`.
- *Gap:* "loss flat, units still dying" is shown on CDDM only. On the flip-flop the score is still creeping up over that window.

**C4. ReLU silence and tanh saturation are one failure: the gradient stops reaching the unit.**
- A ReLU unit at zero rate gets exactly zero gradient on every weight in and out (`slide_wh_zero_gradient`).
- CDDM, N = 1000, 9 ReLU seeds: counting units the loss gradient reaches gives 274–319. That is the same population participation counts (281–301).
- It is not ReLU's hard zero. Leaky ReLU, softplus and sigmoid also end at 276 / 250 / 188 working, against 266 for ReLU (`slide_k_activation`).
- tanh (the Sutton link): no unit goes silent, but the median unit's gradient per unit of error is 27× below its untrained value. Only 25–147 of 1000 tanh units measurably move the output, against 277–293 for ReLU (`slide_k_unit_gradient`, `slide_k_unit_sensitivity_in`).
- *Gap:* measured at one task and one size. The paper's claim is about scale, so the gradient measure must be run across N.

**C5. The standard knobs change how many units are lost, not the trend.**
- Seven knobs at N = 1000: weight decay, recurrent noise, metabolic penalty, recurrent-weight cap, spectral radius and activation on CDDM, plus input scale on the flip-flop.
- They end between 0.32× and 1.6× their default's count, and none gets above 448 of 1000.
- Every ladder keeps losing units after its loss floor. Weight decay goes from 482–580 at the floor to 86–412 at the end.
- Figures: the `slide_k_*` ladders, `slide_33_bottom_line`. Numbers per knob are in §5.

**C6. Rules that give participation its own drive recruit units at a small cost.**
- Flip-flop grid, N = 500–4000, 3–6 seeds, read at 75k: every rule beats the control at every N.
- At N = 1000 the multipliers run from 1.57× (rescaling) to 2.92× (duplication), for 0.1–4.3% of held-out r².
- Between 40k and 75k the control loses 10% of its units, while dropout and synaptic noise hold steady and duplication gains.
- The rate penalty alone keeps every CDDM unit working at every N tested, at no cost (r² 0.8356 vs 0.8358 at N = 1000). On the pre-standard flip-flop it keeps 979 of 1000.
- Recruited units add few new directions: participation ratio 5.5 for the control, 6–7 for the rules.
- Figures: `slide_f2_active`, `slide_f2_size_active`, `slide_f2_r2_vs_active`, `slide_f2_dims` (all still drawn at 40k).
- *Gap:* the figure is at 40k and the rate penalty alone was never run on the standard flip-flop grid.

**C7. The task gradient cannot hold a unit up; participation needs its own drive.**
- Re-initialising silent units (the continual-backprop move) does nothing: 26,012 redraws on CDDM leave 414 working, against the control's 410.
- A one-off threshold lift revives 475 units, but training puts 80% of them back. Net 373 vs 275.
- Rescaling with no release target ends at 258, below the control. Any target from 2.5 to 30 gives 384–450 (`slide_27c_rescale_target`).
- So a unit stays up only while something outside the task gradient holds it there.
- *Gap:* no two working rules have ever been combined, so "several sources at once" is untested.

## 3. The metrics we report

| Metric | Definition | Answers |
|---|---|---|
| **Active units** | pᵢ = std(rᵢ) + q₀.₉(\|rᵢ\|) on a noise-free pass; unit is active if pᵢ ≥ 0.05 · q₉₅(p) of its own network | How many units work |
| **Held-out r²** | Unseen inputs, training noise on, 8–9 draws | What a remedy costs |
| **Read-out point** | First iteration where the smoothed loss reaches 1.07× the run's own fitted floor. End of budget is a second, labelled read-out | Where arms are compared |
| **Gradient reach** | gᵢ = RMS of ∂L/∂hᵢ, ÷ √L; bar = 5% of the untrained network's q₉₅ | Does the gradient still reach the unit (ReLU and tanh)? |
| **Participation ratio** | (Σλ)² / Σλ² of the active units' rate covariance | Do recruited units add directions or copy existing ones? |
| Sokar dormancy | mean rate ÷ population mean ≤ 0.025 | One-time cross-check against the ML literature |

**Retired**: fixed absolute silence thresholds, noise-free r², matched absolute loss, convergence
criteria and drift exponents, M* / task-demand laws, and all population-structure measures
(selectivity, modularity, clustering, lognormality).

## 4. The data we have

Everything is on Della (`/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs`) unless
marked Spock (`~/trainRNNbrain/data/trained_RNNs`). "Pre" = pre-standard.

| Claim | Sweep | Task | N | Arms | Seeds | Budget |
|---|---|---|---|---|---|---|
| C1 | `CDDM_std_g0_drift`, `CDDM_std_g0_penalties` (Pre) | CDDM | 1000 | control, rate penalty | 3 | 200k |
| C1–C3 | `NBitFlipFlop_std_ksweep`, `std_bigN` (Pre, Spock) | 3-, 6-bit flip-flop | 500–4000 | control | 3 | 100k–500k |
| C2, C3 | `CDDM_std_g0_drift` (Pre; N = 5000 on Spock), `_drift_bias0` | CDDM | 100–5000 | control | 3 | 50k–300k |
| C2 | `DMTS_d7_pen`, `DMTS_paper_grid_d5` | DMTS | 500–2000 | control, duplication, dropout | 3 | 150k |
| C3, C5 | `CDDM_std_g0_weightdecay_bias0`, `_wd0_long` | CDDM | 1000 | weight decay ladder | 3 | 200k; 300k |
| C4 | `CDDM_std_g0_activations_bias0` | CDDM | 1000 | leaky ReLU, softplus, sigmoid | 3 | 200k |
| C4 | `CDDM_std_g0_tanh` + gradient files in `/scratch/gpfs/TENGEL/pt1290/figbuild_recinit/gradpart/` | CDDM | 1000 | tanh vs ReLU | 9 + 9 | 150k |
| C5 | `CDDM_std_g0_recinit` | CDDM | 1000 | spectral radius, weight cap | 2 | 75k |
| C5 | `CDDM_77fbd9_g0_noise_ptrack` | CDDM | 1000 | recurrent noise | 5 | 75k |
| C5 | `CDDM_std_g0_metabolic_bias0` | CDDM | 100–1000 | metabolic λ | 3 | 50k (+25k top-up, unread) |
| C5 | `NBitFlipFlop_std_winp_bias0` | flip-flop | 1000 | input scale | 3 | 150k |
| C5, C6 | `CDDM_std_g0_bias0` (unread) | CDDM | 1000 | rate, sparsity | 5 | 75k |
| C6 | Figure 2 grid: `ff_revive`, `dropout_sizes`, `paper_grid`, `size_series`, `synnoise`, `ff_both40k`; 75k cache in `/scratch/gpfs/TENGEL/pt1290/f2_75k/shards_v2` | 3-bit flip-flop | 500–4000 | control + 5 rules | 3–6 | 40k → 75k |
| C6 | `CDDM_paper_grid`, `synnoise_ext` | CDDM | 500–4000 | duplication, dropout, synaptic noise | 3 | 100k |
| C6 | `CDDM_std_g0_penalties` (Pre, Spock) | CDDM | 500–5000 | rate, sparsity, both | 3 | 150k–200k |
| C6 | `NBitFlipFlop_std_pen`, `_penlong` (Pre, Spock) | flip-flop | 500–2000 | rate, sparsity, both | 3 | 150k; 400k |
| C7 | `CDDM_screen3`, `CDDM_prune_reinit`, `NBitFlipFlop_replace_variants` | CDDM, flip-flop | 1000 | redraw, threshold lift | 3 | 30k–40k |

**On this laptop:** 18 GB in `data/trained_RNNs`. That is 16 flip-flop sweeps continued to 75k: the
Figure 2 grid and the revival-rule variants. There are no analysis caches, so most figures can only
be rebuilt on Della.

## 5. The remedies catalogue

*Bio-inspired*: uses only the unit's own activity and synapses and could run online.
*Engineering*: copies units, ranks the whole population, or acts outside the dynamics. Flip-flop
rows are N = 1000, with the multiplier read at 75k and the r² cost at 40k. Knob rows are CDDM,
N = 1000, standard network, read at the end of budget against their own default.

| Remedy | Class | What it does | Active units | r² cost | Tested at | Status |
|---|---|---|---|---|---|---|
| Rate penalty | Bio-inspired | Two-sided penalty around a target rate | CDDM: every unit; flip-flop (Pre): 979/1000 | none (CDDM) | CDDM 500–5000; flip-flop 500–2000 (Pre) | Needs the standard flip-flop grid (B2) |
| Noisy synapses | Bio-inspired | Each weight × (1 + σ_w·noise), every step | 2.08×; 1.65× at N = 4000 | −1.8% | flip-flop 500–4000; CDDM 1000 | Done |
| Rescaling incoming weights | Bio-inspired | Tilts a silent unit's input row toward its positive weights, at fixed length, until a release target | 1.57×; 1.94× at N = 4000 | −1.4% | flip-flop 500–4000 | Done |
| Threshold lift | Bio-inspired (one-off intrinsic plasticity) | Raises a silent unit's bias once | 1.36× | none | flip-flop, 40k | Screen only |
| Prune-and-duplicate | Engineering | Overwrites a silent unit with a jittered copy of a working one | 2.92×; 3.70× at N = 4000 | −0.2% | flip-flop, CDDM 500–4000 | Done; fails on DMTS above N = 500 |
| Read-out dropout | Engineering | Randomly hides busy units from the read-out | 1.95× | −1.9% | flip-flop, CDDM 500–4000 | Done; fails on DMTS |
| Full dropout | Engineering | Randomly removes busy units from the dynamics | 904/1000 | 12–17 r² points | flip-flop 500–4000 | Parked: too costly |
| Redraw silent units | Engineering (≈ continual backprop / ReDo) | Re-initialises a silent unit | none | none | CDDM, flip-flop | Negative |
| Rate + sparsity penalties | Bio-inspired pair | Both penalties together | 3.39× | −2.8% | flip-flop 500–4000 | Parked; rate penalty alone replaces it |
| Weight decay 0 / 1e-5 / 1e-4 | Knob | L2 on weights (default 1e-6) | 1.55 / 0.62 / 0.32× | none | CDDM | Done |
| Recurrent noise σ 0 / 0.01 / 0.1 | Knob | Noise on unit states (default 0.05) | 0.74 / 0.78 / 1.14× | — | CDDM | Done |
| Recurrent-weight cap | Knob | Soft penalty on recurrent weight size | 1.29–1.32× | none | CDDM | 2 seeds |
| Spectral radius 0.5 / 2.0 | Knob | Initial recurrent gain (default 1.2) | 1.06 / 0.92× | none | CDDM | 2 seeds |
| Metabolic penalty | Knob | Penalises firing | 1.34–1.57× | −0.014 at λ = 1 | CDDM 100–1000 | Control budget mismatch (A4) |
| Leaky ReLU / softplus / sigmoid | Knob | Replaces ReLU | 1.04 / 0.94 / 0.71× | none | CDDM | Done |
| Input scale | Knob | Input weight-row length at initialisation | up to 1.60× | none | flip-flop, 150k | Done |

## 6. Set aside (parked, not lost)

- The selective/modular solution under rate + sparsity: burst units, assemblies, penalty switching.
- Single-unit selectivity, clustering, cell types, NMF, wiring-versus-activity.
- The task-demand ceiling (M*, M ∝ N^b k^c), the Walsh and hyper flip-flops, MultiRule15, Yang 20-task (except one supplementary panel, A5).
- Long-delay DMTS.
- Master-inhibitor, frozen-clamp and silent-at-init rescue tests.
- Dale-constrained and sign-constrained networks (one supplementary panel at most).
- Loss-floor laws, convergence criteria, drift exponents, lognormality of rates and weights.

## 7. To-do to publish

### Must — analyses on networks we already have (no new training)
- **A1. Redraw Figure 2 (the remedies) at 75k** from the existing cache, once Decision 1 is made. It is the paper's main remedy figure and is still drawn at 40k.
- **A2. Run the gradient-reach measure across sizes and activations.** Cover the scaling networks (all N), the activation ladder, and the rate-penalty CDDM networks. This turns "loss of gradient grows with scale" from an argument into a result, and the rate-penalty networks show it is not just "sitting at a minimum".
- **A3. Ablate the recruited units for every remedy,** plus random 10/30/50% ablation curves, all in held-out r². Otherwise each remedy is validated only on the count it was built to raise.
- **A4. Read the unread 75k top-ups** (metabolic ladder, `CDDM_std_g0_bias0`). This gives the metabolic arm a matched control and the rate penalty a standard-network CDDM cell.
- **A5. One supplementary "task demand" panel** from the hyper flip-flop and Yang 20-task data on disk. It answers the first objection: "the task just doesn't need more units".

### Must — new training
- **B1. Scaling series on the standard network,** with one learning-rate crossover. Run CDDM at 500–5000 and the 3-bit flip-flop at 500–4000, 3 seeds, 75k, read at the floor. In the crossover, train N = 4000 at N = 500's rate and vice versa. C2 is the headline and is currently pre-standard and confounded with step size.
- **B2. The rate penalty alone in the standard flip-flop grid** (N = 500–4000, 3 seeds, 75k). Report the share of runs that finish. It is the headline biological remedy and was unstable at N = 2000 on the old network.
- **B3. Combination test:** rate penalty (reduced strength) × synaptic noise × rescaling, 2×2×2 at N = 1000, plus all three at N = 4000. Without it, "several sources at once" is untested.
- **B4. Plasticity test.** Take old controls (40k and 150k; N = 1000 and 4000), rate-penalty and noise networks, and fresh networks. Train them on a new task and count how many silent units come back. This is the direct link to Dohare et al., and the evidence that dead units cannot be called back.
- **B5. Optimizer control:** AdamW and SGD on CDDM, N = 1000, 3 seeds, 200k. Adam with coupled L2 is a known unit-shedding artefact; "never stops" should rest on the no-weight-decay runs plus this.

### Must — writing
- **W1. Write the manuscript from this framing.** `paper.md`'s abstract and figure plan are the old story.
- **W2. Pin "not brain-like" to measured properties:** a bimodal rate distribution, where cortex is lognormal; no return to a set point; no recruitment (B4). Cite and answer the "silent/dark neurons" literature (Shoham et al. 2006; Barth & Poulet 2012).
- **W3. State the bio-inspired vs engineering criteria** (as in §5). Cite prior rate regularisers (Zenke & Ganguli 2018; Bellec et al. 2020) and the ML remedies (ReDo, continual backprop, shrink-and-perturb).

### Should
- Run the published continual backprop and ReDo as baselines on the flip-flop (N = 1000, 4000). Our redraw rule only approximates them.
- Add a third seed to the weight-cap and spectral-radius ladders.
- Add learning rate as a knob (can share runs with B1).
- Make one Dale panel from existing runs (49% silent at N = 1000).
- Reproducibility: commit the figure modules, seed the scoring noise in the Figure 2 cache, and test a rebuild from a clean clone.
- Bookkeeping: reconcile tanh seed counts (3 vs 9), the N = 4000 rate + sparsity cell, and the scaling exponents under the own-floor rule.

### Nice
- tanh at a second task and size.
- Remedies on DMTS at the short delay.
- A continuous threshold rule and a purely multiplicative synaptic-scaling rule, to earn the "intrinsic plasticity" and "synaptic scaling" labels.
- 5 seeds on the headline cells.

## 8. Decisions for the next round

1. **How do we read out the flip-flop remedies?** The flip-flop's loss rises after its minimum, so the 1.07× floor rule fires within the first few hundred iterations.
   *Recommended:* report them at 75k as a labelled end-of-training read-out, next to the control's still-falling count. Keep the floor rule for CDDM and the scaling panel.
2. **Is the biological penalty the rate penalty alone, or rate + sparsity?**
   *Recommended:* rate alone in the main text. It is the "urge to spike, but not to overspike", and it costs nothing on CDDM. Move the pair to the supplement.
3. **How do we word the conclusion?** The rate penalty is itself a loss term trained by backprop, and on CDDM it alone keeps every unit.
   *Recommended:* "gradient descent on the task loss alone does not keep units participating; participation needs a drive of its own". Say "several sources" only if B3 shows a combination beats the rate penalty alone on stability, cost, dimensionality or ablation.
4. **Which tasks?**
   *Recommended:* two main tasks. CDDM carries the knobs, the gradient analysis and tanh; the 3-bit flip-flop carries the remedies. DMTS (short delay) appears only as a third series on the scaling curve.
5. **Is the plasticity test (B4) in this paper?** It is the largest new experiment.
   *Recommended:* yes, in a minimal form (N = 1000 and 4000, one new task, 3 seeds). Without it, the Sutton link and "not brain-like" stay motivation rather than results.
