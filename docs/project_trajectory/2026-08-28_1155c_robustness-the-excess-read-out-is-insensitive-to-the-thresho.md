## 2026-08-28 — δ robustness: the `excess` read-out is insensitive to the threshold

Swept δ ∈ {0.03, 0.05, 0.10, 0.15} (read at 1.03/1.05/1.10/1.15 × each run's own floor).

**This is a strong test because δ moves the read-out time by ~2x**: median T at δ=0.03 is 1.5-1.9x the
value at δ=0.10 (none 1.52, rws 1.59, frm 1.94). If the conclusions depended on read-out depth they
would move.

### PR exponents — stable

| pen | c at 1.03 | 1.05 | 1.10 | 1.15 |
|---|---|---|---|---|
| none | +0.097 \* | +0.092 \* | +0.084 \* | +0.104 \* |
| rws | +0.214 \* | +0.211 \* | +0.187 \* | +0.163 \* |
| frm | −0.008 | −0.008 | −0.008 | −0.008 |

*(\* = CI excludes 0)*. frm's c is identical to three decimals at every δ.

### M exponents — also stable, with one marginal cell

| pen | c at 1.03 | 1.05 | 1.10 | 1.15 |
|---|---|---|---|---|
| none | **−0.039 [−0.078, −0.000] \*** | −0.032 | −0.029 | −0.001 |
| rws | +0.080 \* | +0.087 \* | +0.074 \* | +0.057 \* |
| frm | −0.005 \* | −0.002 | −0.003 | −0.002 |

⚠️ At δ=0.03 `none`'s M just clears zero, with an upper CI bound of −0.000. That is the only δ at
which M is non-zero for `none`, it is marginal, and it is NEGATIVE — the opposite sign to PR's +0.097
at the same δ. So it sharpens the dissociation rather than threatening it. frm's −0.005 at δ=0.03
likewise clears zero but is scientifically zero at that magnitude (significance without magnitude,
the same pattern as the hard-criterion fits).

### The M vs PR dissociation holds at EVERY δ

M straddles zero (or is marginally negative); PR is positive and excludes zero. Four thresholds, two
outcome measures, same answer.

### A prediction of mine that was WRONG

I expected coverage to degrade at δ=0.03, reasoning that the fitted floor is only identified to ~1-2%
so a 3% target sits barely outside its own uncertainty. **Coverage is 100% at every δ, and the seed
CV does not rise either** (none 0.072/0.069/0.056/0.066, rws 0.053/0.049/0.049/0.045, frm
0.091/0.101/0.060/0.081). The floor-fit noise does not propagate into the read-out time at the level
that matters. Recorded because the reasoning was sound and the prediction still failed.

**Conclusion: δ = 0.10 is not a tuned choice.** Anything in 1.03-1.15 gives the same exponents, the
same signs, the same dissociation, and equally good coverage and seed stability.

---

