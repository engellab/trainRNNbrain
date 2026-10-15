## 2026-08-25 (evening, 2) — ⚠️ On the flip-flop, silence exists ONLY in the scale-free sense

Prompted by the question "can we check the absolute active units, not the scale-free ones?".
`flipflop_figures.py` now emits `flipflop_fig5_law.png` and `flipflop_fig5_law_hard.png`.

**Active fraction M/N under the two silence rules, unpenalised flip-flop:**

| N | hard (p ≥ 1e-6) | scale-free (p ≥ 0.05·q95) |
|---|---|---|
| 500 | 0.967 – 1.000 | 0.33 – 0.61 |
| 1000 | 0.990 – 0.999 | 0.24 – 0.39 |
| 2000 | 0.997 – 1.000 | 0.15 – 0.27 |

**Under the absolute criterion 97–100% of units are active in every cell**, so the joint power law
returns `b = 1.00–1.01` with CIs like [1.00, 1.01] at all five comparison criteria: M = N exactly, no
saturation, no k-dependence, nothing to explain. **The entire flip-flop silent-unit result rests on
the scale-free (relative) criterion.** That has to be stated in `paper.md` as a limitation, not
discovered by a referee.

⚠️ This is a REAL DIFFERENCE FROM CDDM, where hard silence is substantial (N=10000 read 1463/10000
active under `hard`). Unpenalised flip-flop networks keep every unit fractionally above 1e-6; the
`rws` penalty does create hard silence here (M_hard 484 → 309 at k=1 N=500). Why the two tasks differ
is open and not resolved by anything run so far.

⚠️ Two hard-criterion fits come back "REJECTED" at p = 0.017 and p = 0.0049 with c = +0.007 and
−0.003. Statistically real, scientifically empty: with M/N ≈ 1 the residual variance is tiny and the
F-test resolves sub-1% structure. **Significance without magnitude — always read c and p together.**

### The conclusion on which law wins

The five columns are NOT five rival models of one dataset; they are one law form fitted to five
DIFFERENT measurements, because each criterion reads the networks at a different time. Ranking them
by whether the comparison is fair:

1. **ENDPOINT — disqualified.** Mixed budgets (400k vs 500k), a defect rather than a choice.
2. **FIXED COMPUTE — answers "same training budget".** `c = 0.179 [0.152, 0.208]`, law fits.
3. **MATCHED PERFORMANCE / k-fair / DYNAMICS — answer "equally good networks".** `c ≈ 0`.

The paper's claim must survive the deflationary objection *"your big networks are just less trained"*,
which requires comparing equally-good networks. So **matched performance is the criterion**, and the
winning law is

> **M ≈ 21 · N^0.42, independent of k**  (scale-free criterion; b = 0.41–0.45 across the three
> matched criteria, c straddling 0 in all three).

**This is the PRE-REGISTERED outcome for pathology.** The k-sweep launcher fixed in advance: "if M\*
tracks task demand it must move with k... Flat M\*(k) would mean a genuine pathology rather than
spare capacity." At matched performance M\*(k) is flat, so the spare-capacity reading is the one that
fails. The apparent rise at fixed compute (+27–33% from k=2 to k=8) is convergence depth.

Remaining caveats, both unresolved: the scale-free-only nature of the effect (above), and the N × k
interaction under matched loss (+7% at N=500 vs −15% at N=2000) that a single power law averages to
c ≈ 0 and the lack-of-fit test lacks the power to resolve at 3 seeds/cell.

---

