## 2026-11-25 13:18 — Rules: is swapping the optimizer a remedy for inactive units?

Written and committed before any run of this test. Branch `wp/optimizer-remedy`. Pavel's ask, via firstmate (25 Nov): "If
we can solve this problem simply swapping the optimizer - all the best, that's another remedy". A remedy is a change that
meets the paper's technique criterion (M09) on 3BFF, CDDM and Maze.

**Terms.**
- *3BFF*: the 3-bit flip-flop task. *CDDM*: the context-dependent decision-making task. *Maze*: the maze task, trained by
  PPO (proximal policy optimisation); one *update* is one rollout of 256 episodes and 8 optimizer steps.
- *M09*: `dead-relu-paper/methods/M09_remedies.tex`, the paper's methods paragraph on training techniques.
- *Active unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass; active if p ≥ 0.05 × the 95th percentile of p.
- *Coupled decay*: the decay term is added to the gradient before Adam divides by the gradient's running size
  (`torch.optim.Adam(weight_decay=…)`; the default networks, 1e-6). *Decoupled decay* (AdamW): each step multiplies the
  weights by 1 − learning rate × decay, outside Adam's division.
- *AMSGrad*: Adam with the largest running squared gradient so far in the denominator instead of the current one.
- *Probe*: the noise-free test of 128 Maze episodes every 10th update. *Learned*: probe success at 0.9 or above for five
  probes in a row and at the last probe (`common.maze_learned_index`).
- *Held-out R²*: `heldout_r2.heldout_r2` under the training noise, 36 seeded noise draws, CPU.

### What was already known when these rules were written

- **The outcome of one candidate on every task.** Decay-free networks (Adam at decay 0, AdamW at 1e-6) in
  `ST_verdicts`: 3BFF 252-367 and 306-358 active units (controls 233-270), CDDM 442-469 and 431-453 (controls 325-345 and
  333-342), Maze 404-452 and 429, 452 (two of three learned) against 394-449. So I expect the no-decay candidate to pass 3BFF
  and CDDM and to fail Maze. On 3BFF at 200,000 iterations the two groups come close (203-226 and 208-221 against 169-203).
- **SGD without momentum:** within tolerance on CDDM at 0.1 and 0.3, 0.05 to 0.16 below the controls in R² on 3BFF, and no
  Maze network learned.
- **These comparisons were not like for like.** The CDDM optimizer networks have independent initial draws
  (`2026-11-25_1303_rules-independent-init.md`) and their controls shared ones; the CDDM controls' R² is an estimate; the
  Maze AdamW networks used the first trainer version.
- AMSGrad, AdamW at a decay above 1e-6 and SGD with momentum have never been run in this project.

### AdamW at 1e-6 and Adam without decay are the same algorithm in float32

At N = 1,000 the learning rate is 4.64e-4, so AdamW at 1e-6 multiplies each weight by 1 − 4.64e-10 per step. The nearest float32
number is 1 (float32 resolves 6e-8 below 1), and the weights are float32. The decay step is therefore exactly nothing, and AdamW at
0, AdamW at 1e-6 and Adam at 0 give bitwise equal weights (2,000 steps on a fixed gradient sequence, a third of the rows with
zero gradient; AdamW at 1e-2, AdamW at 1e-4 and Adam at 1e-6 differ, so the check could have failed; `tests/test_optimizer_switch.py`).
On a GPU the check is repeated before the first read. **The brief's four arms "AdamW at 0, 1e-6 and one larger value; Adam with no
decay" are therefore two candidates.** One network group stands for the three no-decay names.

### Candidates: five, fixed now

Everything else is the default network: ReLU, equation `h`, bias 0, N = 1,000, learning rate × 0.464 (the size rule).

| Candidate | Optimizer keys | Base learning rate | Reason |
|---|---|---|---|
| `nodecay` | Adam, decay 0 (= AdamW at 0 and at 1e-6) | 1e-3 (default) | removes the decay that Adam rescales |
| `adamw1e-2` | AdamW, decay 0.01 | 1e-3 | a decay that is not erased by rounding and is the same for every weight; 0.01 is the default of `torch.optim.AdamW`. Without a gradient a weight keeps 71% of its size after 75,000 iterations (under coupled 1e-6 it vanishes) |
| `amsgrad` | AMSGrad, coupled decay 1e-6 | 1e-3 | the swap alone: a weight whose gradient becomes small keeps its old, large denominator, so the decay is not amplified as in Adam |
| `sgdm0.03` | SGD, momentum 0.9, coupled decay 1e-6 | 0.03 | the same long-run step, 0.03/(1 − 0.9) = 0.3, as the plain-SGD rate with the best 3BFF R² (0.3) |
| `sgdm0.1` | SGD, momentum 0.9, coupled decay 1e-6 | 0.1 | three times that step; plain SGD was unstable on 3BFF at 1, momentum raises the stable range |

RMSprop, NAdam, Adagrad and second learning rates for AMSGrad are not run (GPU-hours).

### Control networks

Same code, same settings, same three seeds as the arms; they differ in the optimizer keys.
- **3BFF and CDDM:** the independent-draw default networks of task independent-init (`NBitFlipFlop_independent_init`,
  `CDDM_independent_init`, Della 18375654 and 18375655, not read by me when these rules were written). My launcher's 3BFF and CDDM
  commands are that task's. Their CDDM networks run to 100,000 iterations and mine to 75,000; they are read at their kept
  75,000 snapshot. `max_iter` enters training only through noise annealing, which is off (check 2).
- **Maze:** the matched default networks of task maze-matched-controls
  (`trained_RNNs_rl_maze_mctrl_N1000/Maze_interventions/control`, Della 18374320, corrected trainer, seeds 1-3, 2,000 updates). I continue
  them to update 4,000 with Adam's state restored, into my own tree.
- **If a control network is missing** (failed, or its wall clock ended it), I train that seed with arm `control` of my launcher.
- A control is used only after a saved-config diff shows that it differs from each arm in `optimizer`, `weight_decay`,
  `momentum`, `lr`, the tag, the paths and, on CDDM, `max_iter`, and in nothing else.

### Runs and order

Launcher `slurm/optimizer_remedy_della.slurm`, Della gpu-short, from a clean clone `~/trainRNNbrain_optrem` at the commit named in
the submission entry. Snapshots every 1,000 iterations.
1. 3BFF, all five candidates, 75,000 iterations (15 networks).
2. CDDM, `nodecay`, `adamw1e-2`, `amsgrad`, 75,000 iterations (9 networks). An SGD-momentum candidate runs on CDDM **only if it
   passes 3BFF**: a candidate that fails 3BFF cannot be a remedy, and 3BFF is the task plain SGD failed. This changes the brief's
   Stage 1 to save about 20 GPU-hours. A candidate skipped this way counts as failing.
3. Maze, `nodecay` and `adamw1e-2`, 2,000 updates, then 2,000 more with the optimizer state restored (6 networks), and the
   controls' continuation (3 networks).
4. Stage 2: any other candidate that passes 3BFF and CDDM runs on Maze, to update 2,000 and, if the cap allows, to 4,000.
5. Late decline (below), if the cap allows after steps 1-4.

**Cost.** 3BFF about 5 h per network, CDDM 3.3 h, Maze 6-7 h per 2,000 updates on an 80 GB card: 75 + 30 + 78 + 20 = about
200 GPU-hours for steps 1-3. **Cap: 250 GPU-hours** (elapsed × 1 GPU, every attempt counted). Before each later submission I
add the hours used and the projection of every job not yet ended; a step that would pass 250 is not submitted. If the hours used
reach 245, running jobs are cancelled, latest submitted first, and reported. Time limits: 10 h (3BFF), 7 h (CDDM), 14 h
(Maze, per stage). A network ended by its wall clock or a node failure is continued from its last resume state, if the cap
allows. A network that stopped by the rollback guard did not finish.

### Criterion, per task (M09), and when an optimizer swap is a remedy

Every finished network counts. A candidate **passes a task** when every finished network has more active units than every
control network, and its mean performance is at least the controls' mean minus 0.01 (held-out R², 3BFF and CDDM) or 0.05
(probe success, Maze); on Maze also no network ends below 0.9. Evaluated by `fig_paper_F5_tasks.rule`. Read points: 75,000
iterations; Maze update 2,000 (M09's) and, beside it, update 4,000 against the continued controls. A candidate with fewer
than three finished networks on a task is reported as incomplete and does not pass.

- **An optimizer swap is a remedy** if at least one candidate passes 3BFF, CDDM and Maze at update 2,000.
- **Remedy on Maze only with longer training** (a weaker, separate outcome) if at least one candidate passes 3BFF and CDDM and
  passes Maze at update 4,000 but not at 2,000. M09 reads a value at 4,000 only where a network had not learned by 2,000;
  if every network in the candidate and in its controls had learned by 2,000, the 4,000 read is reported and this outcome does
  not apply.
- **An optimizer swap is not a remedy among the five tried** if no candidate does either. The entry then says so in its first
  sentence. A pass on two tasks is not called a remedy.
- **Undetermined** if no candidate passes and some candidate's missing network could still change that.

The bar is not changed and no network is added after a read. **Beside the verdict, the same rule against M09's own controls** (3BFF 236-288,
CDDM 333-342, Maze 415-449), which have shared draws on 3BFF and CDDM. A candidate that passes against the matched controls and
fails against M09's, or the reverse, is reported as such; the matched controls decide.

**Size-test reading (description only).** The pre-registered rate-penalty size test asks, at each N, for at least 2 of 3 stable networks
(fewer than 5% of updates skipped, finite loss), a fold change of at least 1.5 and R² within 0.02. Its size and
floor-crossing parts cannot be read at one N and 75,000 iterations. I report the three remaining numbers per candidate and task.

### Choosing the best of five

- If the optimizer had no effect and the six counts of a candidate and its controls were exchangeable, the chance that all
  three of the candidate's lie above all three controls is 1/20 per task, so at least one of five candidates would meet the count condition
  on a given task with probability up to 5/20 = 0.25 (Bonferroni bound). **A pass on one task is therefore weak evidence, and it is
  reported with that number.** Same-seed networks start from the same weights, which makes the ordering less likely by chance, not more.
- On all three tasks the chance is 1/8,000 per candidate, at most 5/8,000 = 0.0006 for five. With the four plain-SGD rates
  already in the paper (nine distinct optimizer settings tried in total), at most 9/8,000 = 0.0011. A three-task pass survives the correction.
- The correction does not cover `nodecay` on its known outcome: its verdict is a replication, and is reported as one.
- The effect size of a candidate picked as the largest of several is biased upward. If several pass, all are reported, and none is
  named best on its count alone. If one passes, its fold changes are quoted as estimates from the networks that selected it.

### For any candidate that passes both supervised tasks (a pass on all three is the full set)

- **Late decline.** Every candidate and control, from the stored vectors: the median active count over the 2,000 iterations before
  50,000 and before 75,000 (Maze: the mean of the last 10 probes before updates 2,000 and 4,000). *Still falling* in a network:
  the second is below the first. For a candidate that passes 3BFF and CDDM, if the cap allows: 3BFF continued to 125,000 and
  CDDM to 100,000 iterations with the optimizer state restored, beside the control networks (CDDM controls run to 100,000;
  3BFF controls to 200,000 only if that task's GPU-hour rule allows; otherwise the 3BFF comparison is missing and said so).
- **Removal test (M09).** k = the network's active count minus the controls' mean. The k least participating active units are
  removed (inactive units left in place) and compared with five random sets of k active units, with `a3_remedy_ablation` (3BFF,
  CDDM, 9 noise draws) and `rl_maze_ablate --added` (Maze). *Depends*: the fall is at least 0.10 in every network of the
  candidate. *Does not depend*: every fall is below the tolerance (0.01; 0.05 on Maze). Else *mixed*. A network with k ≤ 0 is not removed and is named.
- **Copies or distinct.** For each active unit, the largest cosine between its incoming weights and those of any other active unit.
  Incoming weights: W_inp plus the columns of W_rec from active units. Weights from inactive units carry no input, and in a network
  without decay they keep their random initial values, which would lower every cosine; the full-row value is reported beside it. A unit is
  a *near copy* if that cosine exceeds 0.9. Share of near copies over all active units and over the k least participating ones.
  *Copies*: at least half of the k units in every network. *Distinct*: at most 0.15 in every network, and not above the
  controls' largest share by more than 0.05. Else *mixed*. The controls' own k-unit share uses their least participating third.
- A candidate that fails a supervised task is not analysed further.

### Checks, thresholds fixed now

Checks 1 to 3 were done on the smoke runs before this entry was committed (`firstmate/data/optimizer-remedy/smoke/`): check 1
passes (maximum difference 0.0 on 3BFF and CDDM; every candidate differs from the control arm), check 2 passes (the first 30 losses
and the first stored participation vector are equal), check 3 passes on the five candidates of both tasks (only the keys of the table
and the tag). They are repeated on the real configs at the read.

1. **Code.** At N = 50, 30 iterations, CPU, the `control` arm of my launcher and the same command at 513623a give equal final weights
   (maximum difference 0) on 3BFF and CDDM; each candidate's weights differ from them. *Pass:* both.
2. **`max_iter`.** The CDDM control arm at N = 50 with max_iter 30 and 60 (same seed): the participation trace and losses of the
   first 30 iterations are equal, exactly.
3. **Configs.** The smoke runs' saved configs against the control arm's: only the candidate's optimizer keys, the tag and the paths
   differ (the tag and the paths follow from the arm). Repeated on the real configs at the read, against the control networks.
4. **Same start.** The stored initial probe count of a same-seed pair (iteration 0 or the first probe; Maze: update-1 return, success
   and steps, as in `2026-11-25_1237`) must be equal between a candidate and its control. The Maze values depend on the card model:
   the controls ran on 40 GB cards and mine run on 80 GB cards, so a Maze network may instead equal the stored same-seed `ent0.1`
   or `cor` networks of N = 1,000 on an 80 GB card. Any other mismatch is traced before any verdict.
5. **Independent recount.** A script that shares no code with `common.py`, the reader or the rule function recounts active units at the
   read points from the stored participation vectors and evaluates the criterion. *Pass:* exact equality for every network and every verdict.
6. **Read gate.** Before any verdict the reader prints the number of finished networks per cell against the number expected and
   refuses a verdict on a cell it cannot account for.

Out of scope: dead-relu-paper, Adam's epsilon, combinations with other techniques, other sizes.
