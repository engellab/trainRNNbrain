### B5 complete except SGD lr 3 / wd 0, and C3 on the flip-flop (agent-c3, merged; notes `docs/notes/agent-c3.md`)

**SGD without weight decay** matches SGD with it at every learning rate (floor: lr 0.03 703-727
active, R² 0.883-0.884; lr 0.1 524-555, 0.888-0.902; lr 0.3 230-295, 0.904-0.911; lr 1 272, 0.911,
1 of 3 runs). No SGD arm comes within 0.02 of Adam's R²; the SGD arms with > 400 active units are
0.045-0.07 below Adam at the floor. The conclusion stands: the optimizer is not the cause.
Figure `diag_b5_optimizer.png` redrawn with every arm.

**C3 on the flip-flop: units keep dying while the loss is flat** (test written first: < 5 units lost
per 100k in at least 2 of 3 runs per size would mean they stop). 15 lineages read along their chains
from 150k; flat = loss change < 3% per 100k. N = 1000: 5 of 6 runs flat, 4 of those lose 22-83
units per 100k; the two 300k runs slow from ~63 to 11-22 per 100k at an unchanged loss but do not
stop. N = 500: 4 of 5 flat runs lose 7.6-67 per 100k (a second, independently written window-mean
check reads one at 0, which would make it 3 of 5). N = 2000: only 1 of 3 runs is flat (it loses 57
per 100k); the other two are still falling 3.6-4.3% per 100k in loss.

