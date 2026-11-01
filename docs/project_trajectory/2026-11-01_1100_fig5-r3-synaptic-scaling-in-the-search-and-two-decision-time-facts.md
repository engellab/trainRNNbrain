## 2026-11-01 11:00 — fig5-r3: synaptic scaling joins the search; two facts as known at the decision

This entry corrects `2026-11-01_1046_fig5-r3-correction-launch-evidence-and-the-full-search.md`. The
corrections follow the second review of dead-relu-paper PR #99 (comment 6562409988). That review
also withdrew its claim that the run files predate the commits: it had compared Pacific file times
with Eastern commit times.

### Synaptic scaling was missing from the search

`table_rule_screen.py` had skipped the sweep `NBitFlipFlop_scaling_gamma` as one that "varies no
per-unit rule". Its 4 cells all have `trainer.synaptic_scaling: true`, on 3BFF, at N = 1,000, with
γ = 0.1, trained to 40,000 iterations. Synaptic scaling scales each unit's excitatory inputs up and
its inhibitory inputs down according to the unit's own rate, so it is a per-unit rule.

The script now includes the sweep. It counts a network whose activity is non-finite at the read-out
as diverged and leaves it out of the mean. Fold change is against the γ = 0.1 control networks
(323 active at 40,000 iterations).

| Setting | Diverged | Active | Fold change |
|---|---|---|---|
| norm=true, eta=0.05 | 0 of 3 | 408 (432, 417, 374) | 1.26 |
| norm=false, eta=0.2 | 2 of 3 | 547 | 1.69 |
| norm=true, eta=0.2 | 2 of 3 | 462 | 1.43 |
| norm=false, eta=0.05 | 3 of 3 | -- | -- |

These counts match the reviewer's independent ones. Earlier γ = 0 settings exist only in the
notebook (2026-09-23_1309, 2026-09-24), and the Mac folder `ff_revive/..._arm=scale` is empty.
Supplementary Table 3b now has 64 cells.

The count of rules changes with this addition. Eleven per-unit rules were tried: five were trained
at four sizes and six were screened. Random redraw and zeroing also act on single units; they are
counted apart from the eleven, as the revival methods of Fig. 3e.

`std_switch` is now excluded by name. Its networks (N = 2,000) were warm-started from rate-penalty
networks and trained 50,000 more iterations with the sparsity penalty switched on or off; that tests
a switch of penalty, not a rule trained from the start. The table footnote says so, and says that
the inventory covers the copies on the Mac.

### Two facts, now stated as they were known at the decision

- **σ_w = 2 and 3.** When σ_w = 1 was kept, these networks had been measured at 40,000 iterations.
  Their noise-free R² was 0.784 and 0.655 against the control networks' 0.894, a cost of 0.11 and
  0.24 (99c877a, 879e707). Entry 1046 and M09 had quoted 0.17-0.31, which are 75,000-iteration
  values of the continuations, measured later. M09 now gives both and says which is which.
- **λ_frm = 0.2.** The August notebook (2026-08-18_1547) records that a smoke test at 0.2 "drove
  silent units 5 → 0 within 150 iterations". That means no silent unit, which is not the same as
  every unit active. M09 now quotes what was recorded.

### Board

Row C6 of `docs/paper_audit.md` is updated: the claim is scoped to the settings shown, Supplementary
Tables 3a and 3b are cited, the eleven rules are listed, and the seven rescaling settings at
0.67-0.93 of the control are named.
