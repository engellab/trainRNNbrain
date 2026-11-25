## 2026-11-25 13:17 — Amendment to the mechanism rules: the bar for "resurrected units contribute" (part B)

Amends `2026-11-25_1311_rules-mechanism-noise-decay-and-switch.md`, part B, removal. Written before any part B network exists
(its parents are still training: one part A job had run for 11 seconds when the calibration below was read, and nothing from it has
been read).

**Why.** The rules calibrated the per-unit bar (a fall in held-out R² above 1e-4) on 100 inactive units of one older control
(seed 457853), and required that at least half of the resurrected units exceed it. The active units of that network had not been
measured. They are now: removing one of its 311 active units lowers R² by a median of 6e-5, and 43% of them exceed 1e-4
(0 of the 100 inactive units do). Ordinary active units would fail "at least half", so the bar asked more of a resurrected unit
than of an active one.

**Change.** The per-unit bar stays at 1e-4. The shares are now compared with the share of costly units among the units active
throughout (`kept`) in the same network:
- *contribute:* the share of costly resurrected units is at least 0.5 times that share, in all three networks, **and** removing all
  resurrected units at once lowers R² by more than 0.01, in all three;
- *do not contribute:* at most 0.1 times that share, **and** a group fall below 0.01, in all three;
- otherwise *mixed*.

A unit whose removal raises R² counts as not costly. In the calibration network, a quarter of the active units raise R² by more than
5e-5 when removed, so a share of costly units can be low for reasons other than inactivity. The falls of all units are stored.
