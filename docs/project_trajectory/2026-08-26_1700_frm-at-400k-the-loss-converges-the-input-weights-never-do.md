## 2026-08-26 17:00 — frm at 400k: the LOSS converges, the INPUT WEIGHTS never do

First 8 cells of the 400k re-run landed (frm, N=500, k=1..3). Both convergence tests re-run.

### Test 1 — loss: PASSES

| budget | loss fall over the last third (median) | per-net values |
|---|---|---|
| frm @150k | 3.7% | 0.7, 1.9, 22.6, 5.4% |
| **frm @400k** | **1.0%** | −5.4, 1.1, 1.3, −1.9, 1.7, 3.5, 1.0, 0.4% |
| none (reference) | ~0% | −0.3 … +0.1% |

Values now straddle zero. **Fitted floors and matched-loss read-outs are valid for frm at 400k**, so
the retracted "frm worsens the floor" question becomes answerable once enough cells land — but it must
be re-derived from these runs, not recovered from the 150k numbers.

### Test 2 — W_inp diffusion: FAILS, and NOT because the budget is short

| variable | reached α<0.6 | α at end | onsets |
|---|---|---|---|
| **W_rec** | **6/8** | 0.84 | 47k–183k |
| **W_inp** | **0/8** | 0.87 (range 0.61–1.17) | — |
| W_out | 1/8 | 0.17 | 9k |
| participation vector | 1/8 | 0.52 | 11k |

*(unpenalised reference: W_inp 64/64 at median 63k, W_rec 62/64.)*

**This is a property of frm, not a budget problem.** The loss has stopped improving while the input
weights are still in directed motion — α stays at 0.61–1.17 through 400k, with no sign of decay.
Consistent with what frm penalises: it constrains firing-rate magnitude, so it keeps driving weights
after the task term has plateaued. Extending the budget further would not fix it; there is no reason
to expect W_inp ever settles under frm.

### Methodological consequence

⚠️ **A cross-condition diffusion comparison (none vs frm vs both) must use `W_rec`, not `W_inp`.**
W_inp is the criterion of record for the UNPENALISED analysis because it is the last variable to
settle there — the most conservative choice. It does not exist as a read-out under frm. W_rec is the
only drift variable available in both conditions, and it was already flagged as the best-*founded*
one anyway (per-decade α disagreement 0.091 vs W_inp's 0.302).

The headline unpenalised result stands unchanged: it is an internal comparison among unpenalised
networks, where W_inp is available and appropriate, and b/c were shown stable across W_rec, W_inp and
the participation vector at five thresholds each.

### Queue at 17:00

41 FFpenL running, 3 pending ranges (~85 tasks: 30 frm + 55 both). All old 150k FFpen jobs have now
finished. ⚠️ **FFbigN (N=4000, 48 jobs) has still not started after 21 hours** — it sits behind the
entire FFpenL backlog on priority.

---

