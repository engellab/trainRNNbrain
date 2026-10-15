## 2026-08-21 15:47 — Flip-flop k-sweep lands: silencing FALLS steeply with task complexity

> # ⛔ RETRACTED — trained with `same_batch=True`
> **Every number in this entry is void.** These flip-flop networks reused ONE batch of 256
> trials for all 300k iterations, so they measure memorisation, not the task. See the
> `same_batch=True` entry below for the full account. The data is quarantined in
> `data/trained_RNNs/RETRACTED_samebatch_NBitFlipFlop_ksweep/` and the figures were deleted.
> Kept here only so the reasoning trail is not falsified by deletion.


All 45 jobs COMPLETED (Spock `5671899`/`5671900`/`5672178`), 300k iterations each. Output landed in
`NBitFlipFlop_std_ksweep`, not `FlipFlop_std_ksweep` — Hydra prefixes the folder with the task name,
so the launcher's `mkdir -p` created a stray empty directory that briefly looked like total data loss.
Synced (2.5 GB). New script: `experiments_and_analysis/flipflop_ksweep.py` →
`img/internal_figures/flipflop_ksweep.png`.

**1 of 45 runs diverged** — k=2, N=2000, one seed, NaN loss from 16% of the way in, zero active units
at the end. It carried NO penalty, so this is plain training instability on the flip-flop, not the
frm self-excitation mode. Excluded from all numbers below; the loader drops any run with a NaN in
`loss_clean_train` or no active unit at the endpoint.

### The result

Scale-free silent fraction, matched performance L\*=0.022 (R²=0.970), N=2000:

| k | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| silent, scale-free | 67.5% | 46.6% | 21.5% | 18.2% | **8.0%** |
| silent, hard | 8.0% | 6.4% | 1.5% | 0.5% | **0.0%** |

**Monotone decreasing in k, and it holds under every reading** — endpoint at 300k and matched
performance at L\* ∈ {0.022, 0.015, 0.010}, at all three N. That agreement is what licenses the
ordering; it was the pre-stated criterion.

### This settles the branch flagged on 2026-08-20

The draft claim "silencing remains an issue even for complex tasks like 6-bit flip-flop" is **not
supported**. At k=6 the network is essentially fully active. The honest claim is the other branch:
**silencing is severe when the network is over-provisioned relative to task demand**, and it
disappears as task demand rises. This supports the spare-capacity reading rather than a pathology
reading, and paper.md §1 and §6.4 need rewriting accordingly.

Silencing on the flip-flop is far weaker than on CDDM throughout: hard-silent peaks at ~24% here
versus 70–79% for CDDM at N=2000. CDDM behaves like a LOW-k task, which is consistent — it is a
2-choice decision with a 2-dimensional readout.

### Where the two readings disagree, and why

Hard criterion, N=2000: endpoint gives 7.8 → 11.4 → 17.7 → 23.7 → 16.7 (rising with k), matched
gives 8.0 → 6.4 → 1.5 → 0.5 → 0.0 (falling). Cause is in the `T_read` column: at L\*=0.022 the k=6
cell crosses at **13,423** iterations while k=2 takes **139,380**. At a fixed 300k budget the harder
tasks have therefore trained ~20× longer past their crossing, and silencing grows with depth.

⚠️ **The reverse caveat is real and must be stated in the paper.** At matched loss the k=6 cell is
read very early in training, before its silencing has developed — so matched-performance may
UNDERSTATE silencing at high k just as the endpoint reading overstates it. Neither reading is clean;
the scale-free ordering survives both, the hard-criterion ordering does not.

### Unexplained observation worth following up

`T_read` falls monotonically with k at every level and size — more bits reach a given loss FASTER
(139k / 81k / 34k / 26k / 13k at N=2000, L\*=0.022) — while the endpoint losses converge to similar
values (~0.005). More output channels appear to speed early learning, plausibly through more error
signal per step. Not needed for any claim, but it is what drives the endpoint/matched disagreement.

---

