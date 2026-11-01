## 2026-11-01 12:20 — markers-r4, review of PR #118: no marker leaves its measured value on a data axis (corrects 2026-11-01_1154)

Corrects `2026-11-01_1154_markers-r4-open-markers-fig2e-fig5.md`. That entry said markers of two
rules were set side by side in three places (5a, 5e, 5f). The independent review of
dead-relu-paper PR #118 logged `set_apart` while it drew the figure and found 12 markers moved in
four panels:

| panel | marker: measured x → drawn x |
|---|---|
| a | noisy synapses 75,000 → 77,460; read-out dropout 75,000 → 72,540 iterations |
| b | noisy synapses 590 → 594; read-out dropout 569 → 565 active units |
| e, N = 1,000 | rate (ringed) 0.128 → 0.149; rate 0.106 → 0.128; duplicate 0.104 → 0.085; duplicate 0.054 → 0.051; read-out dropout 0.127 → 0.106 (R², k random units removed) |
| f, R² rate row | 0.931 (ringed) → 0.928; 0.932 → 0.936; 0.923 → 0.921 |

Two of the moves in e and one in f were chain moves. Panel b's markers did not need to move: their
centres are 1.3 mm apart, more than half a diameter. Decision (editor, on the review): never move a
measured value on a continuous data axis.

Redrawn on `wp/markers-r4`:
- `set_apart` is removed.
- **a:** this is the one remaining offset. The noisy-synapse and read-out-dropout medians are drawn
  exactly 2,500 iterations after and before 75,000 (`A_OFFSET`); the caption says so.
- **b:** every marker is at its measured position.
- **e:** every marker is at its measured position. Two pairs overlap at their values: a rate-penalty
  network on a read-out-dropout network (0.2 mm apart) and a rate-penalty network on a
  prune-and-duplicate network (0.45 mm apart). In each pair, both markers get a white face at 50%
  opacity (`see_through`), and the larger shape is drawn first, so each pair prints as a circle over
  a triangle with both outlines visible.
- **f:** ringed and unringed networks of one row share one group, as in panel c's lower axes.
  - In the rate row of the R² column, the three networks (0.923, 0.931 ringed, 0.932) form one
    marker labelled "1/3". It is one group of three, not of two, because 0.923 lies 1.35 mm from the
    ringed network, inside the 2.4 mm join distance for ringed points.
  - The "2" of the off-scale open group in the rate + rescaling row now sits left of its ring,
    outside the axes, away from the filled 0.849 marker.

Scan change. Rule 2 of `tests/scan_open_markers.py` now lists a pair whose two faces are both
half-transparent white as "see-through" and does not count it, because such a pair is a deliberate
overlap at the measured values. The two pairs in 5e are listed. Rule 1 still holds for all of them:
the lowest centre value in 5e at N = 1,000 is 204.

Result: 0 rule failures in Figs 2 and 5. Mapping check: 129 of 129 and 97 of 97 filled-marker
centres read dark. The script's printout is identical to main's. `tests/test_paperstyle.py` passes,
and no two text spans overlap.

Not done (non-blocking in the review): drawing Fig. 2e's s.e. bars over the white faces. A bar
through a marker's centre would fail rule 1 by design and print as a crossed marker. On 11 networks
the bar is shorter than the marker radius and stays hidden under the white inside.
