## 2026-09-27 — synaptic rescaling, quantified: it trades dimensionality for unit count

The one arm left out of the 99-network dimensionality run, measured on the same footing. N=1000,
gamma 0.1, three seeds per cell, gated on a recomputed r2; 9 of 15 networks passed, the other 6
being the NaN runs, which show the divergence signature of 0 units clearing the activity criterion.

| cell | active | dimensions | r2 | seeds surviving |
|---|---|---|---|---|
| control | 314 | 6.39 | 0.952 | 3/3 |
| scaling, row-norm preserved, eta 0.05 | 391 | **5.14** | 0.943 | 3/3 |
| scaling, row-norm preserved, eta 0.2 | 437 | **3.67** | 0.922 | 1/3 |
| scaling, raw, eta 0.05 | 345 | **2.39** | 0.929 | 1/3 |
| scaling, raw, eta 0.2 | 465 | **1.62** | 0.546 | 1/3 |

**The rule does not do nothing -- it actively destroys dimensionality, monotonically in how hard it
is pushed.** Its one stable configuration recruits 1.24x the control's units while the population
spans 20% fewer directions. At the most aggressive setting that survived at all, 465 active units
span 1.62 dimensions: the network has collapsed onto essentially a single direction while every
unit reports as active.

That is the rule working as designed. Driving every unit toward a common set-point is a
homogenising operation, and a homogeneous population spans fewer directions. The active-unit count
rises because homogenised units all clear a relative threshold, not because the network is using
them for anything.

**The two homeostatic rules now agree.** `rescale`, the per-unit gradual unsuppression, ends at 252
units and 4.09 dimensions -- also below the control. Both members of the homeostatic family lower
dimensionality; both rules that work (duplication, dropout) raise it. Ranked at N=1000:

| arm | active | dimensions |
|---|---|---|
| rescale | 252 | 4.09 |
| scaling, best stable cell | 391 | 5.14 |
| control | 277-314 | 5.6-6.4 |
| bias_kick | 373 | 6.04 |
| mix | 336 | 6.18 |
| mute dropout | 510 | 7.06 |
| duplication | 692-706 | 6.7-8.1 |
| dead dropout | 919 | 30.6 |

This is what makes the rescaling work worth keeping in the paper as a negative control rather than
dropping: it is not a failed attempt at the same goal, it is a family of interventions that moves
the population the WRONG WAY on the measure that matters, and it says why -- homeostasis equalises,
and equalised units are redundant units.





