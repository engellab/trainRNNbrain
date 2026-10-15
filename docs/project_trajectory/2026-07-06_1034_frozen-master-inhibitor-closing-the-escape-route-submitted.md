## 2026-07-06 — frozen master inhibitor: closing the escape route (submitted)

The unfrozen result showed `frm` rescues the clamped units *indirectly*, by suppressing the over-cap, **active**
master (a legitimate gradient, since the master fires). To realise the genuinely gradient-proof clamp — Pavel's
true thought experiment — we now **freeze the master's weights** so `frm` cannot touch it.

**Implementation (`RNN_torch(freeze_master=True)`, no Trainer changes).** Two mechanisms together: (1) a gradient
hook zeros the gradient on the master's input row `W_inp[m,:]` and its recurrent input row + output column
`W_rec[m,:]`, `W_rec[:,m]`; (2) a `forward_pre_hook` restores those entries to their init values before *every*
forward pass. (1) alone is **insufficient** — Adam + `weight_decay=1e-6` reintroduce a tiny effective gradient
inside `step()` that Adam's normalization amplifies into a full ~`lr` update (empirically the master drifted ~1e-2
over 25 iters with the hook alone); (2) guarantees the master is exactly at init during every training/eval
forward. Verified: [`verify_master_freeze.py`](../trainRNNbrain/experiments_and_analysis/verify_master_freeze.py)
shows master gradient `0.374` (unfrozen) → `0.000` (frozen) on the Trainer's `autograd.grad` path; an end-to-end
25-iter run leaves the master weights unchanged (4.6e-4 save-time artifact) while the rest of the net trains
normally (~1e-2), targets 100% silent, master peak 1.0.

**The run (submitted — job `5115568`, commit `931680`).** Same grid as the unfrozen run: 48 jobs = 2 eq (h, s) ×
4 fractions `master_inhib_frac ∈ {0.25,0.5,0.75,1.0}` × 2 penalties (`none`, `frm=0.2`) × 3 seeds. Config
`configs/model/rnn_relu_Dale_masterinhib_frozen.yaml`; launcher `slurm/SilentReLU_masterinhib_frozen_gamma0_N1000.slurm`.
Output → a **distinct** folder `data/trained_RNNs/CDDM_931680_g0_masterinhib_frozen/`.

**Prediction.** With the clamp frozen, `frm` can no longer tame the inhibitor, so the target units should **stay
silent even under `frm`** (a sharp contrast with the unfrozen 100% rescue) — and at `frac=1.0` the network should
fail the task (R²≈0), since the always-on inhibition is now immovable. Read out with the
`plot_masterinhib_rescue.py` machinery (target-active% under none vs frm across fractions, plus R²).

### Result (2026-07-06) — freeze verified, `none` arm clean, but the `frm` arm is wrecked by divergence

`plot_masterinhib_rescue.py CDDM_931680_g0_masterinhib_frozen`. Freeze confirmed in the data: **master peak is
pinned at exactly 1.0 in every net** (vs 1–4.5 in the unfrozen run), and targets are 100% silent at init. But
**22/48 nets diverged to NaN**, and the divergence is almost entirely in the `frm` arm — every `frm` condition
lost a net (leaving **only 1 valid `frm` net per condition**), while every `none` condition kept all 3.

**`none` arm (clean, 3 nets/cond):**

| eq | frac=0.25 | 0.5 | 0.75 | 1.0 (R²) |
|----|----|----|----|----|
| h — targets active | (diverged) | 0.2% | 0.1% | 0.0%  (R²=**−0.38**) |
| s — targets active | 15.2% | 16.5% | 14.8% | 0.0%  (R²=**−0.38**) |

- **The frozen clamp holds under `none` for h** — targets stay ~0% active (vs ~10–29% in the *unfrozen* run):
  freezing the master makes the clamp materially harder to escape without rescue pressure. For **s** it still
  leaks ~15% (equation-dependent — the `s` dynamics let some targets escape via built-up excitation).
- **`frac=1.0` breaks the task** (R² ≈ **−0.38**, worse than predicting the mean) under `none` — an immovable
  all-unit clamp prevents learning, close to the extreme-case prediction.

**`frm` arm — NOT interpretable (n=1/cond, survivorship-biased).** The surviving `frm` nets show 100% target
active with genuine participation (~0.09–0.13) and, at frac 0.25–0.75, a solved task (R²≈0.83–0.86); at frac=1.0
even the survivor fails (R²=−0.38). Taken at face value this would say `frm` *still* rescues the frozen-clamp
targets at frac<1.0 — plausibly by a **different** route than the unfrozen case: since the master can no longer
be tamed, `frm` would have to drive enough compensating **excitation** onto the targets to overcome the fixed −5
inhibition. But with only one non-diverged net per `frm` condition (the seed that happened to train stably), this
is a hint, not a result.

**Why divergence is worse when frozen.** In the unfrozen run `frm` could suppress the destabilizing, over-cap
master (that was both the rescue route *and* a stabilizer). Freezing removes that outlet, so the hyperactive
master (peak 1.0 ≫ cap 0.3) and its deep −5 currents persist and the `frm`-driven dynamics blow up more often
(22 NaN vs 12 unfrozen), concentrated in `frm`.

**Verdict / next step.** The freeze mechanism works and the `none` results are informative (clamp holds for h;
frac=1.0 kills the task). But the **central question — can `frm` rescue a genuinely gradient-proof clamp — is not
yet answered**, because the `frm` arm mostly diverged. Needs a **stabilized rerun** to get ≥3 valid `frm` nets/cond:
lower `master_ctx_drive` toward the cap (~0.3 so the master isn't over-cap) and/or `master_inhib_strength`,
tighten `max_grad_norm` (50→~5), and add seeds. (See `TODO.md`.)

**Stabilized rerun submitted (job `5116848`, `CDDM_f4b706_masterinhib_frozen_gamma`).** First stabilizer tried
(Pavel's suggestion): turn the **cubic term back on** (`gamma=0.1`) — the `-gamma*x^3` soft-saturation bounds the
runaway activity that produced the NaNs. It only bites at large positive `x`, so it does not lift the dead (`x<0`)
targets and therefore leaves the rescue question intact (only removes the blow-ups). Caveat: `gamma≠0` departs from
the gamma=0 experiment line, acceptable for a mechanism probe. Smoke test confirmed targets 100% silent and the
master saturating slightly (peak 1.0→0.92). If gamma alone doesn't cut the divergence enough, next levers are a
lower `master_ctx_drive` and a tighter `max_grad_norm`.

### Result (2026-07-07) — stabilized run answers it: frm overcomes even a frozen clamp, *unless the whole net is clamped*

`plot_masterinhib_rescue.py CDDM_f4b706_masterinhib_frozen_gamma`. gamma helped modestly (**18/48 NaN vs 22**) but
did not fix the `frm` arm — every `frm` cell still lost ≥1 net (mostly 1 valid; `s`/frac=1.0 kept all 3). Master
peak pinned at **0.921** (frozen + cubic-saturated). Reporting both peak-based active% and the more honest
**median participation** of the target set (`part`, active-mode ≈ 0.1):

| eq | penalty | frac=0.25 | 0.5 | 0.75 | 1.0 |
|----|----|----|----|----|----|
| h | none | 0.8% | 0.3% | 0.0% | 0.0% (R²=**−0.38**) |
| h | frm  | (div) | (div) | 100%, part 0.056, R²=0.84 | 100%, **part 0.004**, R²=**−0.38** |
| s | none | 9.5% | 8.7% | 9.7% | 0.0% (R²=**−0.38**) |
| s | frm  | 100%, part 0.10, R²=0.83 | (div) | 100%, part 0.12, R²=0.82 | 100%, part 0.11, R²=**−0.38** (n=3) |

**The answer (with the frac=1.0 nuance):**

1. **`frm` overcomes even the frozen, gradient-proof clamp at frac<1.0 — genuinely.** Targets reach real active-mode
   participation (`s`: 0.10–0.12; `h`/0.75: 0.056) and the task is solved (R²≈0.82–0.86). Since the master can no
   longer be tamed (frozen, verified), `frm` must be routing around it: driving the *non-clamped* units to build
   **compensating excitation** onto the dead targets, strong enough to overcome the fixed −5 inhibition and pull
   them active. So freezing the inhibitor does **not** make the targets unrescuable — as long as a functional
   surrounding network remains, `frm` recruits it to rescue them. (The stark `none` contrast — h ~0%, s ~9% — is
   clean at 3 nets/cond and confirms the clamp genuinely holds without `frm` pressure.)
2. **frac=1.0 is the true failure regime — this confirms Pavel's thought experiment.** With *every* unit clamped
   there is no scaffold left to build compensating drive, and the task cannot be solved under either penalty
   (R²≈**−0.38**, a degenerate constant-output solution, consistent across the 3 `s`/1.0/frm nets). `frm` still
   forces *nominal* threshold-crossing — genuinely for `s` (part 0.11) but negligibly for `h` (part 0.004, i.e.
   not really rescued) — yet the network is functionally dead either way.

**Synthesis of the whole rescue arc.** Per unit, a truly-dead ReLU gets no direct gradient (Pavel's argument holds
at the single-unit level). But rescue is a *network-level* phenomenon: `frm` makes "all units active" the trained
solution and reaches it from any init, and it can even dismantle a structured, gradient-proof silencing mechanism
by recruiting the rest of the network to compensate. The **only** way silence survives `frm` is to remove the
network's capacity to compute entirely (clamp everything, frac=1.0) — which is not a per-unit property, and which
also destroys task performance. In short: **you cannot keep an individual unit silent under `frm` if the rest of
the network still works.**

**Caveats.** The frac<1.0 `frm` cells rest on 1 surviving net each (gamma cut but didn't eliminate divergence),
though they are mutually consistent and agree with the unfrozen (100% rescue) and gamma=0-frozen survivors; the
frac=1.0 failure is solid (`s` n=3). The peak≥0.01 "active%" overstates rescue at frac=1.0 (see `h` part=0.004) —
participation is the honest readout. `gamma=0.1` regime. Firming up the frac<1.0 `frm` counts would need a further
stability pass, but the qualitative conclusion is already supported.

### Root cause of the NaN divergence (2026-07-07) — a forward-dynamics instability frm builds; the cubic is not the cause

Diagnosed from diverged nets' saved `LossBreakdown.json` / `GradsRaw.json` (no Spock needed), in **both** the
gamma=0 and gamma=0.1 runs. (An earlier note pinned this on the cubic term — corrected here after checking: the
gamma=0 runs, which have *no* cubic term, diverged too, with the identical signature.)

**The signature is the same everywhere and is sudden, not gradual.** Losses are *normal* until the last finite
step and then go NaN in a *single* step (γ=0.1 net: iter 1595 task 0.090/frm 0.109 → iter 1596 NaN; γ=0 net: iter
801 → 802 NaN), and the gradient norms just before are **modest** — `g_task ≈ 0.1–0.3`, `g_fr_magnitude ≲ 1.4`,
nowhere near `max_grad_norm=50`. So it is **not** a gradient explosion and **not** gradual weight growth, and
gradient clipping cannot catch it — it is a **forward-pass dynamical instability of the learned weights**, which
crosses a stability boundary in one training step and then overflows within the 300-step trial.

**The common cause: `frm` builds a self-exciting recurrent loop with gain > 1.** To activate the deeply-inhibited
targets against the fixed −5 clamp, `frm` drives the network to build strong compensating excitation onto them.
That creates recurrent loops whose effective gain can exceed 1; the continuous system `dx/dt = −x + W·ReLU(x)+…`
is then unstable (activity grows), and explicit forward Euler (`x_{t+1}=x_t+α(−x_t+W_rec·ReLU(x_t)+input−γx_t³)`,
`α=dt/τ=0.1`) integrates that growth over 300 steps. What (if anything) bounds it depends on γ:

- **γ=0 (no saturation):** nothing stops it — a loop with gain > 1 grows unbounded → overflow → NaN. Verified
  numerically: the self-exciting map `x ← x + α(−x + g·ReLU(x))` stays bounded for g ≤ 1 but blows up for g > 1
  (g=1.5 → ~10⁶ in 300 steps).
- **γ=0.1 (cubic on):** the cubic *does* bound moderate growth (helping — NaN rate 22→18), **but** the cubic
  itself is unstable under explicit Euler for large x: `x ← x + α(−x − γx³)` decays for x₀ ≤ 13 and diverges for
  x₀ ≥ 14 (threshold `√(2/(αγ)) ≈ 14`). So γ trades unbounded linear growth for a bounded-then-overshoot regime —
  a partial fix, not a cure.

**This explains every feature:** **sudden** (a stability-boundary crossing, not growth); **disproportionately
under `frm`** (`frm` is what builds the destabilizing excitation; `none` doesn't, so `none` almost never diverges);
**why γ only helped 22→18** (it removes the unbounded route but adds the cubic-overshoot route); and **why
`max_grad_norm` is irrelevant** (the blow-up is in the forward dynamics, gradients are ~1). The divergence and the
rescue are two faces of the same thing — `frm` overcoming the clamp by building excitation, pushed to the point of
instability.

**Correct fixes (for any firm-up rerun), most-direct first:** (1) **milder clamp** — `master_inhib_strength`
−5→−2 (still silences the targets) so far less compensating excitation is needed and loop gains stay < 1 — attacks
the source; (2) **hard-bound the state** in `forward` (clip `x` / saturating map) so no loop can overflow; (3)
**smaller integration step** (dt↓ / τ↑, or sub-step Euler) — raises the explicit-Euler stability threshold; (4)
lower `lambda_frm` so the excitation is built less aggressively. Gradient clipping and γ tweaks act on the wrong
layer.

**Fix search (2026-07-07), by cheap 16–32-job frm-only tests before any full run:**
- **Milder clamp `−2`:** did **not** help — same ~50% NaN (6/12), only *delayed* onset (iter ~4430 vs 800–1600).
  Lowering the clamp just changes how long `frm` takes to build a supercritical loop; it still gets there.
- **`dt=0.5` (finer integration; α 0.1→0.05, raising the loop-gain boundary g<11→g<21), two γ variants:**
  - `dt=0.5 + γ=0.1` → **3/16 NaN (~19%)** — a big drop from ~50%.
  - `dt=0.5 + γ=0` (dt alone) → **16/16 NaN (100%)** — *worse*.
  So you need **both**: γ provides the bounded fixed point (γ=0 has none → any gain>1 grows unbounded, and the
  larger per-step noise at small α, ∝√(2/α), makes γ=0 blow up every time), and the smaller step keeps the
  explicit-Euler integration of that cubic stable. This also corrects the earlier "γ value doesn't matter" note:
  γ=0 vs γ>0 matters enormously (bounded vs unbounded); it's only the *magnitude* of γ>0 that doesn't move the
  boundary. **Chosen fix: `dt=0.5 + γ=0.1`.**

**Full stabilized rescue run submitted (job `5127540`, `CDDM_b5fafb_masterinhib_frozen_dt05`).** `dt=0.5 + γ=0.1`,
frozen master, clamp −5; 64 jobs = 2 eq × 4 frac × (3 `none` + 5 `frm`) seeds (5 `frm` seeds → ~4 valid/cond after
the ~19% divergence; `none` never diverges so 3 suffice). Config
`rnn_relu_Dale_masterinhib_frozen_dt05_g01.yaml`, launcher `SilentReLU_masterinhib_frozen_dt05_N1000.slurm`,
wall 12 h (dt=0.5 ≈ 6 h/job measured). This should finally give a clean, well-powered read on whether `frm`
rescues a genuinely gradient-proof clamp (the earlier answer — yes at frac<1.0, no at frac=1.0 — but on n≈1).

### Result (2026-07-08) — well-powered, and it CONFIRMS the thin answer

Stabilization worked: only **6/61 nets diverged (~10%)**, and every `frm` condition retained **3–5 valid nets**
(vs n≈1 before). Master frozen (peak 0.921). `plot_masterinhib_rescue.py CDDM_b5fafb_masterinhib_frozen_dt05`.
Target-active% (peak≥0.01), target median participation, and R² (mean over valid nets):

| eq | penalty | frac=0.25 | 0.5 | 0.75 | 1.0 |
|----|----|----|----|----|----|
| h | none | 0.8% | 0.2% | 0% | 0% (R²=**−0.38**) |
| h | frm  | 100%, part 0.073, R²=0.84 | 100%, 0.042, 0.84 | 100%, 0.019, 0.85 | 100%, 0.048, **R²=−0.38** |
| s | none | 0% | 0% | 0% | 0% (R²=**−0.38**) |
| s | frm  | 100%, part 0.105, R²=0.83 | 100%, 0.081, 0.85 | 100%, 0.089, 0.83 | 100%, 0.094, **R²=0.08** |

The conclusion from 2026-07-07 holds, now robustly:

1. **`frm` overcomes even the frozen, gradient-proof clamp at frac<1.0** — 100% of targets active with genuine
   participation (`s`: 0.08–0.10; `h`: 0.02–0.07), task solved (R²≈0.83–0.86), consistent across 3–5 nets/cond.
   Since the master is immovable (frozen, verified: peak fixed at 0.921), `frm` rescues by recruiting the
   *non-clamped* units to build compensating excitation onto the dead targets.
2. **frac=1.0 (whole net clamped) fails the task** — with no scaffold left, the network cannot solve CDDM under
   `frm` (h R²=**−0.38**, s R²=**0.08**; both essentially failed) even though `frm` still forces the targets to
   nominal activity. This is the genuinely-unrescuable regime — Pavel's thought experiment confirmed.
3. **Cleaner `none` arm at dt=0.5:** the clamp now holds `s` targets at exactly 0% (they leaked ~9–15% in the
   dt=1 runs) — the finer integration removed the spurious leak, so the `none` control is unambiguous.

**Bottom line for the whole rescue arc:** a single dead ReLU has no direct gradient (per-unit), but silence is a
*network-level* property — `frm` makes "all units active" the trained solution and will dismantle even a
structured, gradient-proof silencing mechanism by recruiting the rest of the network, **unless** you remove the
network's capacity to compute entirely (frac=1.0), which also destroys the task. You cannot keep an individual
unit silent under `frm` while the rest of the network still works. (Method note: getting here required fixing an
explicit-Euler forward-integration instability — `dt=0.5 + γ=0.1` — not the clamp magnitude or gradient clipping.)

