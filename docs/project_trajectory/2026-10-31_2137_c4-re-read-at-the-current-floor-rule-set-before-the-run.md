## 2026-10-31 21:37 — C4 re-read at the current floor (Figure 4, dead-relu-paper#8): rule set before the run

**Why.** The C4 numbers in the paper (entry 2026-10-14_0006: CDDM 271-322 silent units of 1000;
3BFF 145, 273 and 195 low-active units reached by the gradient; 2-4% of them active 5,000
iterations later) were measured on 13 Oct at the floor read-out of that day. Under the current
read-out (79932ef) two of the three 3BFF networks cross their floor much later: s2461724787 at
50,500 instead of 27,000 and s4113307577 at 58,500 instead of 39,500 (s1258717025: 67,500 against
69,500; the CDDM networks move by at most 1,500). Figure 4 draws these networks, so the analysis is
re-run at the current floor snapshots instead of quoting the 13 Oct values.

**What runs.** Della CPU job, from `~/trainRNNbrain_c1ana` (ed11dfa; its read-out code is identical to
trainRNNbrain main 802f17f), output in `/scratch/gpfs/TENGEL/pt1290/fig4_noreturn/`:
1. `c4_gradient_mechanism.py <6 runs> --out c4_floor --at floor` (unchanged script);
2. `fig_paper_F4_noreturn.py collect` (branch wp/fig4-no-return; copied to the scratch folder):
   links the snapshots at t*, t* + 500 and t* + 5,000 and keeps the noise-free x(t) of 16 trials;
3. `c4_snapshot_drift.py snap c4_floor` (unchanged script).
Floor snapshots found by `gradient_participation.read_point` on the login node: CDDM 25,500,
26,500, 24,500; 3BFF 67,500, 50,500, 58,500.

**Classes.** Active: p ≥ 0.05 q95(p) on the noise-free reference batch (task seed 12345).
Silent: inactive and x_i ≤ 0 on every step of every trial of that batch. Low-active: the other
inactive units. Reached: g_in ≥ 0.05 q95(g_in) of the same network.

**Rule, written before the job ran.** The C4 statement in the paper ("on CDDM about 300 of 1,000
units are silent and get exactly zero gradient; on 3BFF no unit is silent; low-active units get
gradient yet stay inactive") stands at the current floor if all of these hold:
1. every 3BFF network has 0 silent units;
2. every CDDM network has at least 200 silent units, and the largest g_in among them is exactly 0;
3. in every 3BFF network at least 100 low-active units are reached;
4. in every 3BFF network fewer than 10% of the reached low-active units are active 5,000 iterations
   after the floor snapshot.
If any fails, the paper reports the measured numbers and the statement is rewritten to match them;
it is not re-run with other settings.
