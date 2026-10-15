### Snapshot storage (14 Oct, Pavel)

**When is the floor estimate good?** Refitting each of 145 runs on its loss truncated at fractions
of its own final crossing (`floor_settle.py`, /scratch/gpfs/TENGEL/pt1290/floor_settle.json): the
fitted floor only drifts DOWN as data accumulates and settles only after the crossing. With data up
to the crossing itself it is still 5-6% too high (flip-flop 1.052, CDDM 1.057 x the final floor),
and the predicted crossing comes at 0.65-0.74 of the true one; at 1.25x the crossing the flip-flop
reaches 1.023 / 0.88. Online fits often fail (CDDM: ~1 in 8 usable before the crossing). Final floor
over final smoothed minimum: flip-flop ≤ 1.07, CDDM ≤ 1.44 (median 1.02), DMTS ≤ 2.25 (bursts). A
fit-free deletion rule (delete s once its smoothed loss > 1.05·c·current minimum) would be safe but
frees little before the floor.
**Decision (Pavel): compression plus one snapshot every 1,000 iterations.** `np.savez_compressed` for
snapshots and final parameters (lossless; the constant masks were half of each file: 8.1 → 3.8 MB at
N = 1000); launchers' kept-snapshot spacing 500 → 1000. New runs only (running jobs keep their
pinned code). Next: can snapshots be stored as low-rank approximations (W ≈ a·W0 + rank-r ΔW, W0
regenerated from the seed) without changing the loss and the statistics? Agent dispatched.

