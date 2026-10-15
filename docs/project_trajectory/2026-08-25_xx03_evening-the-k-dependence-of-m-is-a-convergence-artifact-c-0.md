## 2026-08-25 (evening) — The k-dependence of M is a convergence artifact: c = 0.18 at fixed compute, c = 0 at matched performance

Full grid now local (k=2..8 × N ∈ {500,1000,2000}, 3 seeds; k=1 at N=500 only; one k=8 N=2000 seed
diverged). `flipflop_figures.py` gained `fit_law` + `fig_law` → `flipflop_fig5_law.png`: the joint
power law `M = A·N^b·k^c` fitted under FIVE comparison criteria, each tested against the saturated
model (a free mean per cell) by a lack-of-fit F-test.

| criterion | b | c [95% CI] | lack-of-fit p | verdict |
|---|---|---|---|---|
| endpoint (own final iteration) | 0.39 | 0.137 [0.075, 0.222] | 1.7e-08 | **REJECTED** |
| fixed compute, t = 150k | 0.41 | **0.179 [0.152, 0.208]** | 0.77 | law OK |
| matched loss L\* | 0.43 | −0.023 [−0.064, 0.019] | 0.50 | law OK |
| matched excess (1.10 × own floor) | 0.41 | −0.038 [−0.074, 0.002] | 0.53 | law OK |
| matched dynamics (α < 0.6) | 0.45 | 0.004 [−0.042, 0.053] | 0.55 | law OK |

**The conclusion, stated plainly: at equal compute more bits DO recruit more units (c = 0.18,
CI excludes 0, law fits at p = 0.77); at equal performance or equal dynamical state they do not
(c ≈ 0, CI straddles 0). b stays 0.39–0.45 throughout.** The k-dependence is entirely accounted for
by convergence depth — high-k cells are further from their floor at any fixed budget, and
less-converged networks carry more active units. Confirmed independently by the per-cell means:
M rises +27–33% from k=2 to k=8 under fixed compute at every N, but is +7%/−4%/−15% (N=500/1000/2000)
under matched loss.

⚠️ **Correction to the earlier reading of `flipflop_M_scaling`.** The endpoint column's rejection
(p = 1.7e-08) was attributed to the law's functional form. It is not: "endpoint" reads N=2000 at 400k
and the other sizes at 500k, so it mixes reading depths and manufactures per-cell structure no smooth
law can absorb. **Endpoint is not a clean test of anything and should not be quoted as one.** With
budgets equalised (`fixed compute`) the identical law fits the same data at p = 0.77.

⚠️ **A single power law cannot express an N × k interaction.** Under matched loss the k-trend is
+7% at N=500 but −15% at N=2000 — a sign change that `c ≈ 0` averages over. The lack-of-fit test does
not flag it (p = 0.50), but with 3 seeds/cell that test has limited power, so the interaction is
UNRESOLVED, not absent. Read c together with the per-cell table, never alone.

Figure convention fixed: BEFORE/AFTER now say "before/after collapsing", each column names its
criterion concretely (what is held equal), and the p-value is labelled a lack-of-fit test with its
verdict — the previous titles let "endpoint" read as the meaning of "BEFORE".

---

