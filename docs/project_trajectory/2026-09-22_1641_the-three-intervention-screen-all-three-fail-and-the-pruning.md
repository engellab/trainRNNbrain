## 2026-09-22, 16:41 — The three-intervention screen: all three fail, and the pruning failure is the informative one

`Screen3` on Della, job `14277821_[1-12]`, worktree `trainRNNbrain_screen3` pinned at `58d76f5`.
CDDM, N=1000, no penalties, bias-free, 30,000 iterations, 3 seeds per arm.

| arm | n | active units | clean loss | units that never fire |
|---|---|---|---|---|
| none (control) | 3 | 410.3 ± 10.6 | 0.0051 | 34.7% |
| cap (0.3) | 3 | 420.7 ± 9.0 | 0.0061 | 31.0% |
| prune (patience 5) | 3 | 414.3 ± 9.9 | 0.0051 | 21.8% |
| scale (eta 0.05) | 2 | 467.0 ± 161.2 | 0.0126 | 6.7% |

Pre-registered bar: 431.5 active units (control + 2 sd) with clean loss below 0.0056.
**No arm advances.** `scale` clears the count but at two and a half times the control's loss, which
the pre-registration calls a trade rather than a win.

### The treadmill falsifier fired on `prune`

26,012 redraws spread over 591 distinct units, which is **44 redraws per unit**, and the active
count moved by 4 units — well inside one standard deviation of the control. Units are redrawn,
they re-die, and they are redrawn again, 44 times each.

The mechanism reaches the frozen units: `tests/test_prune_and_reinit.py` shows a redrawn unit
regains a nonzero gradient, and the 21.8% never-fire figure against the control's 34.7% shows it
genuinely revives them. The network puts them straight back.

**That is the informative result of the whole screen.** The silent state is not an accident the
network cannot escape, because we escaped it 26,012 times and it returned every time. It is where
this network settles. Whatever explains the silence has to explain the return, and no intervention
that acts only on a unit's own weights will hold.

### `scale` is unstable, and the set-point design is why

One seed of three diverged to NaN between iteration 27,500 and 30,000. It was still finite at
27,500 with 483 active units. Peak participation over training went 0.47 → 4.5 → 10.4 → 22.3
before the blow-up.

The set-point is the population's own median. Scaling everything below the median upward drags the
median up, which raises the set-point, which scales further. The clip at (1+eta) bounds each step
but not the loop, because the target moves with the population. There is no anchor.

⚠️ **The rank-matched demonstration arm inherits part of this.** `scale_ln` takes its scale from the
same live median. It should be better behaved, since units above the median get targets above it
rather than being dragged toward it, so the average multiplicative push is near zero once the shape
matches. It is not immune. The tests written for it check the target's SHAPE and say nothing about
its stability over 30,000 iterations, which is the property that actually failed here.

### `cap` answers the question the input-weight analysis left open

420.7 against the control's 410.3, inside two standard deviations, and the clean loss is worse.
Forcing input drive to spread does not recruit units. The correlation reported on 2026-09-22
(input row norm predicts participation at r = +0.65 without penalties, inverting to −0.61 under
frm+rws) runs the other way round: units that are already active end up with spread-out input
drive, not the reverse.

That also settles what to do with the unverifiable claim in `enforce_inp_cap_`'s docstring — the
"99th percentile 8.57 against a median of 0.002" figure that could not be reproduced from any of
815 saved networks. The intervention it motivated does nothing, so the claim should be removed
rather than chased.

### What this closes and what it leaves

Closed: capping input weights, and reviving units by redrawing their incoming weights. Neither
recruits units in an unpenalised CDDM network at N=1000.

Open: why `frm` succeeds where all three of these fail. `frm` takes every unit to active and holds
it there for 30,000 iterations; pruning takes units to active 26,012 times and holds none of them.
The difference is that `frm` acts through the loss on every unit at every step, while pruning acts
once per unit and then lets the task gradient decide. The task gradient decides silence.

Arms 4–7 of the screen (the pairs and the triple, `--array=13-24`) are NOT submitted and should not
be on this evidence. Pairing two interventions that each do nothing is not a use of 9 GPU-hours.

---

