## 2026-10-16 11:15 — C4 and C5, second draft: "silent" now means never fires

Revises `2026-10-15_2207_c4-and-c5-rewritten-in-plain-english.md` with Pavel's edits (16 Oct):
- end the first sentence after "zero or negative";
- define "silent" before using it;
- simplify the CDDM count;
- write "at least transiently";
- **reserve "silent" for units that do not fire at all**, and call the rest of the below-threshold
  units "low-active";
- rewrite C5 in shorter sentences.

**Terms (proposed for the whole paper; Pavel to confirm):**
- **Active:** participation ≥ 5% of the 95th-percentile unit's (Methods).
- **Low-active:** below that threshold, with a positive net input current at least transiently.
- **Silent:** net input current negative on every time step of every trial, so the rate is always
  zero.

### C4

A ReLU unit's firing rate is zero whenever its net input current is zero or negative. We call a
unit silent if its net input current is negative on every time step of every trial. A silent unit's
rate is always zero, and so is the gradient of the loss with respect to each of its incoming
weights, so gradient descent cannot change them. On CDDM, about 300 of a network's 1,000 units are
silent (271 to 322 in three networks). The remaining units below the activity threshold have a
positive net input current at least transiently. We call them low-active. Their incoming weights
have nonzero gradients.

On the 3-bit flip-flop (3BFF), no unit is silent: every unit's net input current is positive at
least transiently. A low-active unit has a positive current on fewer than 10% of time steps. We
counted a unit as reached by the gradient when the root-mean-square gradient of the loss with
respect to its net input current was at least 5% of the 95th-percentile unit's in the untrained
network. By that count, the gradient reaches 145 to 273 low-active units per network. The median
such unit has a positive current on 2 to 4% of time steps, in most trials, almost always within a
few steps after an input pulse. These units' incoming weights change as much during training as
those of active units. Even so, over the next 5,000 iterations only 2 to 4% of them become active.
For about three quarters of them, the current at the time steps where it is positive decreases.

The 3BFF networks are not still learning. From 150,000 iterations on, the training loss of two of
the three networks changes by less than 1% per 100,000 iterations. At 100,000 iterations, CDDM's
loss still falls by 3% per 100,000. On 3BFF, the gradient reaches between 150 and 175 low-active
units from 50,000 to 300,000 iterations, and the number holds after a ten-fold reduction of the
learning rate. On CDDM, the same number falls from about 40 to fewer than 15. The two tasks differ
in how consistent the gradient is across batches. On 3BFF, the gradient averaged over training
batches carries 83 to 99% of a single batch's squared gradient norm; on CDDM it carries 22 to 69%.
Even so, the 3BFF low-active units' incoming weights move back and forth over training rather than
drift in one direction.

### C5

We varied seven hyperparameters one at a time: weight decay, recurrent noise, a metabolic penalty
on firing rates, the initial spectral radius, a recurrent-weight penalty, the activation function,
and the initial input-weight scale. Each network was read at its loss floor. On CDDM, the default
network has 418 active units, and no hyperparameter raised that count by more than about 120. With
a sigmoid activation, 235 to 255 units are active. The metabolic penalty lowers R² from 0.83 to
0.78 at λ = 10 and does not raise the active count. On 3BFF, a 5-fold larger input-weight scale
raises the active count from about 300 to 550-660, with R² unchanged at 0.95. After the floor, the
active count keeps falling. At 200,000 iterations, CDDM networks trained without weight decay have
412 active units; with the strongest weight decay (1e-4), they have 86.
