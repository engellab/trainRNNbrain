## 2026-10-15 22:07 — C4 and C5 rewritten in plain English

Replaces the C4 candidate wordings in `2026-10-15_1342_open-rules-and-commands-moved-from-agent-notes.md`.
Pavel on the CDDM candidate: "really bad wording. which threshold? what does it mean that the
'receive zero gradient on their incoming weights'? 'fire briefly and are reached'? - what are you
talking about." Rewritten with the plain-english skill. Every number comes from the C4 and C4b
analyses (entries 2026-10-14_0006 and 1218) or the Figure 2 knob ladders (entry 2026-10-14_0534);
no number is new.

**Terms used below.** These are defined once, here; the paper defines them in Methods:
- **Input current**: a unit's summed input, before the ReLU.
- **Silent unit**: one whose participation, std(r) + q0.9(|r|), is below 5% of the 95th-percentile
  unit's in the same network.

### C4, the claim (status board)

On CDDM, about 300 of a network's 1,000 units have a negative input current on every trial, so
gradient descent cannot change the weights onto them. On 3BFF, silent units have nonzero gradients,
and their weights change during training, yet the units stay silent.

### C4, text for the paper (candidate, for `01_intro.tex` and Results)

A ReLU unit's firing rate is zero whenever its input current is zero or negative, and over that
range the derivative of the rate with respect to the current is also zero. On context-dependent
decision making (CDDM), 271 to 322 of the 1,000 units in each of three trained networks have a
negative input current on every time step of every trial. For these units the gradient of the loss
with respect to every incoming weight is exactly zero, so gradient descent leaves those weights
unchanged. The remaining silent units of a CDDM network have a positive input current on some time
steps, and their incoming weights have nonzero gradients.

On the 3-bit flip-flop (3BFF), every unit has a positive input current on at least some time steps,
and a silent unit has one on fewer than 10% of them. We counted a unit as reached by the gradient
when the root-mean-square gradient of the loss with respect to its input current was at least 5% of
the 95th-percentile unit's in the untrained network. By that count, the gradient reaches 145 to 273
silent units per network. The median such unit has a positive input current on 2 to 4% of time
steps, in most trials, almost always within a few steps after an input pulse. These units' incoming
weights change as much during training as those of active units. Even so, over the next 5,000
iterations only 2 to 4% of them become active, and for about three quarters of them the input
current at the time steps where it is positive decreases.

The 3BFF networks are not still learning. From 150,000 iterations on, the training loss of two of
the three networks changes by less than 1% per 100,000 iterations; at 100,000 iterations CDDM's
loss is still falling by 3% per 100,000. The number of reached silent units stays between 150 and
175 from 50,000 to 300,000 iterations, and stays there after a ten-fold reduction of the learning
rate; on CDDM the same number falls from about 40 to fewer than 15. The two tasks differ in how
consistent the gradient is. On 3BFF, the gradient averaged over training batches carries 83 to 99%
of a single batch's squared gradient norm (22 to 69% on CDDM), yet the silent units' incoming
weights move back and forth over training rather than drifting in one direction.

### C5, the claim (status board)

Changing a hyperparameter shifts the number of active units at the loss floor, but on CDDM by no
more than about 120 units, and the active count still falls after the floor. On 3BFF, a larger
initial input-weight scale raises the active count.

### C5, details

We varied seven hyperparameters one at a time: weight decay, recurrent noise, a metabolic penalty
on firing rates, the initial spectral radius of the recurrent weights, a penalty on recurrent
weight magnitude, the activation function, and the scale of the initial input weights. Each network
was read where its noisy training loss first reached 1.05× its fitted floor.
- **CDDM (N = 1,000):** the default network has 418 active units at the floor. No setting of weight
  decay, recurrent noise, spectral radius, recurrent-weight penalty, metabolic penalty or
  input-weight scale raises that count by more than about 120. With a sigmoid instead of a ReLU,
  235 to 255 units are active.
- **Metabolic penalty:** it lowers performance (R² 0.78 at λ = 10, against 0.83 by default) without
  raising the active count.
- **Input-weight scale on CDDM:** a larger scale brings the floor earlier, after which the active
  count falls faster.
- **3BFF:** scaling the initial input weights up 5-fold raises the active count at the floor from
  298-334 to 547-660, at unchanged R² (0.95).
- **Weight decay after the floor:** at 200,000 iterations, CDDM networks trained without weight
  decay have 412 active units, and those trained at the strongest setting (1e-4) have 86.
