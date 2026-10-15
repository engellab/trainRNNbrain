### The revival-screen top-ups failed: their parent sweeps were deleted from Della scratch

All 18 top-up tasks (15499043-48) exited within 5 s: "no parent number N". Between ~15:00 and 16:24
on 2026-10-13 (the data directory's last change), `CDDM_screen3`, `NBitFlipFlop_replace_variants`
and the 40k parents of 19 continuation sweeps were removed from
/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs, by a process outside this session.
Among them `NBitFlipFlop_ff_revive` and `NBitFlipFlop_paper_grid`, the first 40k of the flip-flop
size series: `common.noisy_loss_series` joins the loss along the warm-start chain, so the floor
read-out of every `_c75000`/`_c200000` flip-flop run is broken on Della until they are back. The
knob-ladder sweeps, the claim-1 runs and snapshots, and the ablation results are intact.
Copies of all 21 deleted sweeps are on the laptop (`~/Documents/GitHub/trainRNNbrain/data/trained_RNNs`,
~15 GB). Pavel dropped the threshold lift (bias_kick) from option B. Pending tasks cancelled.

**Restore (Pavel: the deletion was an accident).** The 21 sweeps are being uploaded from the laptop
with `rsync --ignore-existing` (nothing on Della overwritten). `CDDM_screen3` checked first, against
criteria set before looking: 0 files differ from the laptop copy (rsync checksum dry run), and the
end-of-training active counts per run equal this afternoon's reads of the Della originals (none
412/420/399, cap 416/431/415, prune 421/419/403). Option A resubmitted: 15509858-60.

**C4, gradient reach (A2) started.** `gradient_participation.py` (c74b08e) now takes every
activation through the network's own definition, with f' by autograd, and reads a run at its floor
snapshot when it has one (`--at floor`; otherwise the final weights, labelled `end`). Check set
before running: on a ReLU network the count of units the gradient reaches must be within 10% of the
active count on the same weights. Smoke test, one CDDM N = 1000 control (end): 414 active; gradient
reaches 448 by state (8%), 427 by incoming and 425 by outgoing weights. Pass. Job 15509818 runs it
on 19 cells: the three claim-1 snapshot controls (at the floor), the CDDM / DMTS / flip-flop size
series (end: no snapshots at N ≠ 1000), the activation ladder and tanh.
All 21 restored sweeps: 0 files differ from the laptop copies (rsync checksum dry run); flip-flop
random / zero_out end counts equal this afternoon's reads (261/250/275, 278/312/278); the flip-flop
chains are whole again (ff_revive_c200000 N = 1000: 3 segments, iterations 1-200,000). Option B
without bias_kick resubmitted (random, zero_out at 0.025 → 200k, KEEP=1).

**Paper: the knob section redrafted** (paper repo, branch `knobs-at-floor`, 257328a, NOT merged to
main, so Overleaf is unchanged). `sec:knobs` in `sections/02_problem.tex` rewritten from the floor
reads above: no setting moves the standard CDDM network's 418 by more than ~120 at its floor; the
metabolic penalty costs R² (0.78 at λ = 10) for nothing; weight decay acts after the floor
(412/266/166/86 at 200k); flip-flop input scale recruits at no R² cost. Still stale elsewhere and
not touched: `04_ratepenalty.tex` (lines 10, 30, 44) and `06_whyitmatters.tex` (line 10) quote the
old trainable-bias knob numbers (+123 for removing weight decay, the old activation deltas).
Della unreachable 18:30-20:20 (laptop off the VPN); jobs there unaffected.

**Queue (21:20).** Della is reachable again. Every GPU job of this session was pending on
`QOSMaxJobsPerUserLimit`: the approved `RMtop200` batch holds all 44 of the account's running
slots, with 77 more of its tasks queued first. `scontrol top` is disabled for users. Pavel chose to
deprioritise the batch: `nice=10000` on its 67 pending array jobs (running tasks untouched), so the
CDDM input-scale ladder, the CDDM screen top-ups and the random/zero-out top-ups (~110 GPU-h) take
the next free slots. Those come as the running tasks end (14 h and 24 h limits, started ~16:50), so
from late tonight. The gradient-reach run was moved to a CPU node (15521043), outside that cap.

