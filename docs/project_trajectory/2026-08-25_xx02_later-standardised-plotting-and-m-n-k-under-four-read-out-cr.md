## 2026-08-25 (later) — Standardised plotting, and M(N,k) under four read-out criteria

New: [`plotstyle.py`](../trainRNNbrain/experiments_and_analysis/plotstyle.py) (colours, bands,
contours, save; one variable = one visual channel across every figure) and
[`flipflop_figures.py`](../trainRNNbrain/experiments_and_analysis/flipflop_figures.py), which emits
`flipflop_fig{1_curves,2_floor,3_active,4_criteria}.png`. `common.py` gained `series`, `drift_alpha`
and `diffusive_onset`.

### The headline: b is robust, the k-dependence of M is NOT

Four read-out criteria, 58 unpenalised runs, k=1..8, N ∈ {500,1000,2000}:

| criterion | matches | mean T | seed CV of T | b in M~N^b |
|---|---|---|---|---|
| `iter` | training budget | 150 000 | 0.00 | 0.41 ± 0.05 |
| `loss` | absolute clean loss L* = 0.03204 | 27 700 | 0.06 | 0.42 ± 0.08 |
| `excess` | 1.10 × each cell's OWN fitted floor | 39 600 | 0.10 | 0.42 ± 0.07 |
| `drift` | weight motion stops being directed (α<0.6) | 31 000 | 0.13 | 0.46 ± 0.06 |

**b < 1 under every criterion, and the four agree to within their spread.** The sublinear-recruitment
claim does not depend on when the network is read, which is the objection it most needed to survive.

⚠️ **But M vs k is criterion-dependent, and that is a finding, not a nuisance.** At N=500 the active
fraction M/N *rises* with k under `iter` (0.33 → 0.48 over k=1..8) and is *flat* under all three
matched criteria (0.51–0.61, no trend). Reading: at a fixed 150k budget the high-k cells are less
converged — `loss`/`excess`/`drift` all read them 2–3× later than k=1 — and less-converged networks
have more active units. **The apparent "harder task recruits more units" is a convergence-depth
artifact.** At matched performance, active-unit count is essentially k-independent while it remains
strongly N-dependent. Anything in `paper.md` claiming M tracks task demand has to be checked against
this.

### Floors, now at three sizes with the √k law fitted per N

| N | a | b | max resid |
|---|---|---|---|
| 500 | 0.02069 | 0.00286 | 0.67% |
| 1000 | 0.01999 | 0.00305 | 0.66% |
| 2000 | 0.02059 | 0.00281 | 3.46% ⚠️ |

The N=2000 residual is the known k=5 truncation artifact (its full-budget fit gives 0.59%); it is
left visible in fig2 panel (e) rather than smoothed away.

### Two measurement bugs found and fixed

**The first `drift` definition was wrong.** Requiring α to stay below threshold until the end of the
run let one late noisy excursion reset the answer to near the final iteration: sibling seeds of k=4
N=500 returned 40k, 44k and **458k**, and k=8 cells returned ~470k against a 500k budget — the
criterion was reporting the BUDGET, not the dynamics. Redefined as the first crossing that persists
for 5 probes; seed CV fell to 0.13 and onset now rises monotonically with k (24k at k=1 → ~45k at
k=8). **Standing rule added: report the within-cell seed CV of any new read-out rule.** A rule whose
T scatters across sibling seeds is measuring noise.

**Averaging over an inconsistent set of sizes.** The first version of fig4(c) averaged M/N over
"whatever sizes this k has". k=1 and k=8 exist at N=500 only; since M/N falls with N, both ends of
the curve were pushed up and a spurious U shape appeared that was about coverage, not complexity.
Both panels now use a single size and say which.

### Data gaps

k=8 at N=1000 and N=2000 have finished on Spock but their folders synced EMPTY (directory names
only), so k=8 has no b value here. k=1 was only ever run at N=500 by design. The last unpenalised
job, k=7 N=2000, was still running. Re-sync needed once the SSH session is back.

---

