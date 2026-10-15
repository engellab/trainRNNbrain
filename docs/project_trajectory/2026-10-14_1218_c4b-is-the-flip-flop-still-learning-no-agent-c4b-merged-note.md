### C4b: is the flip-flop still learning? No (agent-c4b, merged; notes `docs/notes/agent-c4b.md`, figure `img/internal_figures/c4b_convergence.pdf`)

Pavel's hypothesis (the gradient reaching silent flip-flop units means training has not converged)
was tested on the claim-1 networks, predictions committed first (4bee7bb), scope cut at Pavel's
request (DMTS and intermediate reads dropped). Gradient code checked against C4's numpy backward
(2e-6). Falsified on every convergence test:
- Loss slope (5k-block medians): flip-flop at 150-200k: -2.6%, +2.8%, +0.2% per 100k (two flat);
  at 250-300k -0.8%, -0.5%. CDDM at 50-100k still falls 3% per 100k.
- Weight motion: late drift exponent 0.39-0.42 (flip-flop), 0.33-0.34 (CDDM) - sub-diffusive on
  both (bar for "still learning" ≥ 0.7); silent units' incoming rows reverse between 10k steps
  (median cosine -0.41 to -0.47 vs CDDM -0.38 to -0.46).
- Reach does not shrink with convergence: silent units reached 199→174, 155→152, 175→156 from 50k to
  the end (CDDM ~40 → 6-11), and stays after an lr/10 probe (152→155, 156→215).
- Adam at lr/10 for 300 iterations: flip-flop loss -0.4 to -0.75%, CDDM -0.7 to -1.0%.
What does differ: 83-99% of the flip-flop's per-batch gradient is shared across 32 seeded batches at
every read (CDDM 22-69%), but its direction reverses between reads (silent units whose gradient
lowers their drive: 2% → 96% → 1% → 89% on one chain). Interpretation, untested: Adam steps back and
forth across a steep valley; falsifiable by mean gradients 500 iterations apart pointing opposite on
the flip-flop and not on CDDM.

