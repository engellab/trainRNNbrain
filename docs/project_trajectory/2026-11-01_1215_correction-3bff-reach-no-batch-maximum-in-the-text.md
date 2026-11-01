## 2026-11-01 12:15 — Correction to 2026-11-01_1204: the text gives no batch maximum; "fewer than a tenth" stands

Corrects `2026-11-01_1204_correction-3bff-reach-bound-over-twelve-batches.md`, which put "fewer than
a tenth (at most 6.3%)" into Results 4. The independent review of dead-relu-paper PR #117 (comment
6563279856) ran network s4113307577 on five new 3BFF batches of 1,024 trials. Its code shares
nothing with trainRNNbrain: its own batch generator, and a numpy float64 backward pass checked
against finite differences to 2e-8. It used seeds 1001-1005, with the rule fixed before the run.
Active counts were 284-286 against the paper's 287, and no unit was silent. The reached shares were
21.7-23.9%. The share of reached units active 5,500 iterations later was 5.6%, 5.8%, 5.2%, 6.9% and
5.5%. Seed 1004 gives 6.9% (11 of 159), so "at most 6.3%" fails, as "at most 6%" failed before it.

The maximum over batches is not a property of the network. About 10 of 160 reached units become
active, and one unit more or fewer moves the share by 0.6 percentage points. Results 4 therefore
states no maximum. It keeps the pre-registered C4 bound (`2026-11-01_0948`): fewer than a tenth of
the reached units are active later. That bound holds in all 17 batches measured: the paper's, five
from the production code (`2026-11-01_1200`, wp/methods-r4), six from the committee's code
(`2026-11-01_1150`) and five from the reviewer's. The reached share is 16-30% over the twelve
batches of the PR and 22-24% in the reviewer's five, and Results 4 now says "a sixth to
three-tenths". Over all 17 batches the share active later is 0.7-6.9%.

Board row C4's detail paragraph is updated to match on this branch.
