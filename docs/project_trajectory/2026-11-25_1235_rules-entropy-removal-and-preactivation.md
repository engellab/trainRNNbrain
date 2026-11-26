## 2026-11-25 12:35 — Rules: the entropy set point joins Figure 5; unit removal and pre-activation check at λ = 0.1

Written and committed before any measurement of parts 1 and 2 below. Branch `wp/entropy-fig5`. No network is trained.
Pavel's ask, via firstmate (25 Nov): include the per-unit activity-entropy set point at its stronger setting,
λ_ent = 0.1, among the paper's techniques.

**What is already known.** At λ_ent = 0.1 the term meets the criterion of M09 (`dead-relu-paper/methods/M09_remedies.tex`)
in networks of N = 1,000 units on 3BFF, CDDM and Maze (entries `2026-11-21_0755_...`, `2026-11-24_1848_...`,
`2026-11-21_2238_...` and `2026-11-22_1808_...`). M09 also removes the additional active units of a network trained with a
technique and measures the effect on its output. That removal has not been made on the entropy networks. A post hoc
check on one 3BFF network per arm (entry `2026-11-21_0755_...`) found that the pre-activations h of *inactive* units have
the target ratio of mean to standard deviation, but are above 0 on only 5-7% of noisy samples. No such measurement
exists for CDDM, for Maze, or per network.

**Terms.**
- *Active unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass, active if p ≥ 0.05 × the 95th percentile of p.
- *k*: a network's number of active units minus the mean number in its control networks, rounded (M09).
- *Removing a unit*: its outgoing weights (its columns of W_rec and W_out) are set to zero.
- *Drop*: the score of the intact network minus its score after the removal.
- *H_i*: the binary entropy, in bits, of Φ(μ_i/s_i), from the mean and s.d. of unit i's h over steps and trials. The
  term acts on this quantity, with target 0.75.
- *Actual on-fraction of a unit*: the share of the samples (steps and trials) of a noisy pass on which h > 0. *Actual
  entropy*: the binary entropy, in bits, of the actual on-fraction. The two agree with Φ(μ/s) and H_i when h is Gaussian.

### Networks

The nine networks with λ_ent = 0.1, N = 1,000: `NBitFlipFlop_entropy_setpoint/EqType=h_N=1000_arm=ent0.1` (3),
`CDDM_entropy_setpoint/EqType=h_N=1000_arm=ent0.1` (3) and `trained_RNNs_rl_maze_entsp_N1000/Maze_interventions/ent0.1/s1-s3`.
All nine finished and all nine are counted.

### Part 1. Removing the k least participating active units

**Procedure: M09's, with the code M09 used.**
- 3BFF and CDDM: `a3_remedy_ablation.ablate_net` on the 75,000-iteration weights. Held-out R², training noise, 9 draws.
- Maze: `rl_maze_ablate.py --added 430` on the update-2,000 weights, on CPU, as for the 12 Maze networks of Figure 5d.
  Success rate under the training noise, 256 fixed drops × 4 noise draws.
- Scores: intact; every inactive unit removed; the k least participating active units removed with the inactive units left
  in place; k random active units removed (median of 5 sets); and the two larger removals the code also makes.

**Controls for k.**
- 3BFF: the three default networks of A3 (`data/remedy_reads_31oct/a3_N1000.json`: 278, 288 and 242 active units; mean
  269.3), as for the 19 networks of Figure 5d.
- CDDM: the five control networks of Figure 3a at 75,000 iterations (335, 326, 325, 325, 334; mean 329.0), as in the
  removal of the CDDM rate-penalty networks (G3). The criterion's own CDDM controls (342, 334 and 333 at 75,000) have weights only
  at 100,000 iterations, and G3's rule is that arm and control are read at the same iteration. The two means differ by 7 units.
- Maze: 430, the mean of the three controls.

**Bar: M09's, unchanged.** A network's output depends on its k least participating active units if the drop is at least
0.10. Per task: *depends* if the drop is at least 0.10 in every network; *does not depend* if every drop is smaller than the
tolerance of the criterion (0.01 in R², 0.05 in success rate); *mixed* otherwise. Beside it, as in M09: in how many
networks the removal of k random active units lowers the score further. The verdict over the nine networks is the list of
the three task verdicts. It is not pooled.

**What would count against the entropy term.**
- *Does not depend* on any task: the units the term adds do not carry the output there. The paper then may not say that the
  term adds units that the network uses on that task.
- *Mixed* on any task: the statement "the output depends on the added units" does not extend to the entropy term on that
  task, and the paper lists it with prune-and-duplicate on CDDM, the one mixed result so far.
- Neither outcome changes the M09 criterion, which reads counts and performance only.
- Prediction, so that the test can fail: k on 3BFF is expected to be 110 to 185, smaller than for most techniques of
  Figure 5d, so 3BFF is the task where a drop under 0.10 is most likely.

**Gates, before any verdict is read.**
- *Environment, supervised.* The same job ablates the three 3BFF controls and one prune-and-duplicate 3BFF network of A3,
  and one CDDM rate-penalty 0.1 network of G3. Each must reproduce the stored number of active units exactly and the stored
  intact, inactive-removed and k-removed R² within 0.002 (the G5 gate).
- *Environment, Maze.* The same job ablates `trained_RNNs_rl_maze_fz/Maze_interventions/duplicate/s1`, which was scored on
  the Mac CPU: active units within 2, the intact, inactive-removed and k-removed success within 0.03 (the G4 check).
- *Independent check.* A separate numpy script (kept in `firstmate/data/entropy-fig5/scripts/`, sharing no removal code)
  recomputes each of the six supervised networks: the set of removed units must be identical, unit by unit; R² intact and
  after the removal, with its own noise draws, within 0.01 and 0.03 of the read. For Maze it repeats the removal by clamping
  the units' rates to zero instead of zeroing weights, for the three networks: success within 0.03.
- A gate that fails is investigated to its cause before any verdict is read.

### Part 2. Pre-activations, descriptive only

**Measured per network, separately over inactive and active units:** the number of units; median (and quartiles) over units of
the actual on-fraction, of the actual entropy and of the skewness of h; beside them the median of Φ(μ/s), of H_i, the share of units
with |H_i − 0.75| < 0.05, and the share of units with |actual entropy − 0.75| < 0.05.
- *Samples.* 3BFF and CDDM: the seeded held-out batch, training noise, noise seed 20261121 (`unit_stats`' pass). Maze: 256
  noisy episodes with sampled actions (a training rollout), generator seed 20261121, on GPU, only the RNN steps before each
  episode's end, which are the samples the term sees. The noise-free on-fraction is reported beside each.
- *Active and inactive* are read from the stored participation vector nearest the weights' iteration (update 2,000 on Maze).
  On the supervised tasks the entropy networks' nearest vector is 74,900.
- *Controls.* 3BFF: the three default networks with weights at 75,000 (the fourth has none). CDDM: the three controls of the
  criterion at their 100,000-iteration weights, the only ones kept, and the five Figure 3a controls at 75,000. Maze: the
  three default networks at update 2,000.

**What would count against the entropy term (a limit on what the paper may say, no pass or fail).**
- The statement "the term sets the on/off entropy of the units" holds for a class of units (active, inactive) on a task if,
  in each of the three networks, the median actual entropy is within 0.15 bits of 0.75.
- If it does not hold for a class, the paper may say only that the term sets the ratio of mean to standard deviation of h in that
  class. For inactive units of 3BFF this is the expected outcome (post hoc values 0.29 to 0.37 bits).
- If, on a task, the active units' median actual entropy in the entropy networks is no nearer to 0.75 than in the
  controls in every network, the term did not change the active units' on/off entropy there.

**Check.** For one entropy network per task, a separate numpy recomputation of on-fraction and skewness of every unit from the
saved states: agreement within 10⁻⁶ per unit.

### Part 3. Figure 5

- `fig_paper_F5_tasks.py` gains the entropy set point at λ_ent = 0.1 on the three tasks (panels a, b) and its nine removal rows
  (panel d). λ_ent = 0.01 is exported with the data and listed in the printout and the rule tables; it is not drawn.
- Data: a JSON in `data/entropy_fig5/`, written from the three stored reads (3BFF `entropy_setpoint_reads.json`, CDDM `cddm_reads.json`,
  Maze `maze_reads.json`) and compared with the numbers of the three result entries.
- **Test, set now:** every cell, floor read, verdict, panel-c CDDM row and panel-d row that exists at 513623a is bit-identical
  (float hex) afterwards; the baseline dump was made at 513623a before any edit.

### Compute

Della only, gpu-short: one GPU job for the supervised removal and pre-activations and the Maze pre-activations
(`--time=02:30:00`, `--constraint=nomig`; expected under 1 h), and CPU jobs for the Maze removal and the independent
checks (4 networks, about 25 min each on the Mac; limit 4 h). No training. An ssh failure is not a job failure.
