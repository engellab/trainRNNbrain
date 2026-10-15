## 2026-07-01 — deliberate silent-at-init experiment: can frm RESURRECT? (submitted)

**The gap this closes.** The tracking above shows that on the *natural* init nothing is silent, so `frm`'s role is
prevention — but that leaves the sharper question unanswered: **if some units are *already* silent at init, can
`frm` resurrect them, or does it only keep still-active units alive?** Prevention vs resurrection is only
decidable if silent-at-init units actually exist. So we build an init that has them, by construction, and track
those specific units through training.

**Why we can't just "boost inhibition globally".** We first tried the intuitive knob — scale *all* inhibitory
recurrent weights up so more units go net-negative. It fails: the untrained net is nearly homogeneous (every
unit's peak firing rate is within ±6% of ~0.05), so a global inhibitory boost dims the whole network *together*
— the entire population slides through the 0.01 "silent" line as one block (0% silent at boost 1.4 → 71% at 1.6),
with no robustly-active units left (max rate ~0.013). That produces "everything is half-dead", not "a distinct
25% silent against a healthy-active 75%". Any *global scalar* preserves the init symmetry and can only dim
collectively; carving out a silent *subpopulation* requires a *unit-targeted* perturbation.
(Calibration: [`calibrate_inhibitory_boost.py`](../trainRNNbrain/experiments_and_analysis/calibrate_inhibitory_boost.py).)

**The perturbation (targeted inhibition).** Pick a fixed random 25% of units — the set **S** — and over-inhibit
*only them*: at init, after the standard Dale construction, multiply the inhibitory columns (synapses from
I-units, `dale_mask == -1`) of the **S rows** of `W_rec` by a factor **c = inhibitory_boost**. This drives each
S unit's total input net-negative → S is silent at init, while the other 75% keep their normal ~0.05 activity. S
is drawn from the net's seed (`numpy.default_rng(seed)`), so it is **exactly reconstructable from the saved
config** — we know which units were silenced and can track them. Implemented as a no-op-by-default one-liner in
`RNN_torch.__init__` (`inhibitory_boost=None, silent_init_frac=0.25`); all prior runs are unaffected.

**Calibration result.** With `|S| = 25%` fixed, the boost is a near-binary switch: `c = 1.0` → 0% of S silent,
`c ≥ 1.25` → **100% of S silent, 0% collateral on the other 75%, exactly 25% total** (robust across 4 seeds, h
and s; saturates by c≈2 with S peak ~0.003). Verified end-to-end: `boost=None` gives 0% silent, `boost=2.0`
gives exactly 25% and the silent set equals S reconstructed from the seed. **We use c = 2.0** (see the note on
what `c` means below).

**What `c` is.** `c` is *not* the silent fraction — that is fixed at 25% by `|S|` for any `c ≥ ~1.25`. `c` sets
**how hard the silencing is baked in**, i.e. the initial magnitude of the inhibitory weights onto S that training
must undo to bring an S unit back:
- `c` just above threshold (~1.25): S is *marginally* silent (input barely negative) — trivially rescuable, a
  weak test.
- larger `c` (2, 3, 6…): S is *deeply* silent (input strongly negative), the S-inhibition weights start 2–6×
  their natural value, so training must walk them much further down to re-activate an S unit — a progressively
  harder rescue.
So `c` is the **rescue-difficulty dial**. At init the *activity* saturates by c≈2 (more inhibition can't make a
floored unit more floored), but the *weight magnitude* keeps growing with `c`, which is what matters for whether
training can climb back out. We chose c = 2.0: solidly silent (unambiguously below the line, with margin above
the 1.25 threshold) yet only 2× the natural inhibition → clearly rescuable, so the test is fair and not rigged
toward "prevention" by an impossible-to-undo init. A follow-up `c ∈ {1.5, 3, 6}` sweep would trace how rescue
depends on silencing depth.

**The run (submitted — job `5103664`, commit `d9e0ec7`).** ReLU-Dale, γ=0, N=1000, c=2.0 targeted init.
Grid = 20 jobs: 2 equations (h, s) × 2 penalties × 5 seeds, with penalty being
- **`none`** (`lambda_rws=0, lambda_frm=0`) — the control: do the S units stay silent without any rescue
  pressure? (they should — this fixes the "no-rescue" reference), and
- **`frm`** (`lambda_frm=0.2`) — the rescuer: the firing-rate-magnitude penalty that collapses the low-activity
  mode in every earlier sweep.

Config `configs/model/rnn_relu_Dale_silentinit.yaml`; launcher `slurm/SilentReLU_silentinit_gamma0_N1000.slurm`.
Output → `data/trained_RNNs/CDDM_d9e0ec_g0_silentinit/` (folder uses the 6-char commit hash).

**Read-out (planned).** Reconstruct each net's init, identify S from the seed, and follow S through training with
the per-unit machinery of
[`plot_init_vs_trained.py`](../trainRNNbrain/experiments_and_analysis/plot_init_vs_trained.py) (now the
init-silent branch is populated). The decisive comparison, restricted to the S units:
- under **`none`**, S is expected to remain silent (no force pulling it up) — the prevention/no-rescue floor;
- under **`frm`**, if S **stays silent** → `frm` works purely by **prevention** (it keeps active units alive but
  cannot revive dead ones); if S **climbs into the active mode** → `frm` can **resurrect**. Hunch from the earlier
  tracking: mostly prevention, possibly partial resurrection for the `h` equation.

### Result (2026-07-02) — frm RESURRECTS, and the init silencing is not durable

All 20 nets completed (job `5103664`); analysed by
[`plot_silentinit_rescue.py`](../trainRNNbrain/experiments_and_analysis/plot_silentinit_rescue.py). Init check
passes in every net: **S = 100% silent, non-S = 0% silent** — the perturbation did exactly what it should.
Per-condition (mean over 5 nets), "active" = trained peak firing rate ≥ 0.01:

| eq | penalty | init S silent | **trained S active** | trained non-S active | S median participation | non-S median participation |
|----|----|----|----|----|----|----|
| h | none      | 100% | **57.0%** | 57.0% | 0.009 | 0.008 |
| h | frm=0.2   | 100% | **100.0%** | 100.0% | 0.134 | 0.230 |
| s | none      | 100% | **49.7%** | 47.1% | 0.000 | 0.000 |
| s | frm=0.2   | 100% | **100.0%** | 100.0% | 0.342 | 0.382 |

![Silenced-at-init units through training — h](../img/internal_figures/silentinit_rescue_scatter_h.png)

**Two conclusions:**

1. **`frm` can RESURRECT.** Under `frm`, **100% of the units that were dead at init become active** (both h and
   s), landing in the same active mode as the never-silenced units. The earlier natural-init experiment could only
   demonstrate prevention because nothing was silent at init; forcing 25% silent shows `frm`'s reach is broader —
   it drives *every* unit into the active mode regardless of whether it started active (non-S) or dead (S). So
   `frm`'s mechanism is best stated as **"make the all-units-active state the trained solution, reachable from any
   init"** — which subsumes both prevention (keep active units alive) and resurrection (revive dead ones).

2. **The init silencing is not durable — fate is decided during training.** Under `none`, the deliberately-dead S
   units end up **exactly as active as the never-silenced non-S units** (57.0% vs 57.0% for h; 49.7% vs 47.1% for
   s), i.e. S is statistically indistinguishable from the general population after training, which itself carries
   the usual ~50% silent mode. Training reshuffles which units are silent; being silenced at init confers no
   lasting disadvantage. (Consistent with the natural-init finding that init activity only weakly predicts trained
   fate, `corr ≈ 0.3–0.5`.) In the scatter this is the red (S) cloud starting at the far left (init ~2e-4) and,
   under `none`, splitting ~half above / half below the silent guide — the same split as the grey non-S cloud.

**Caveat / scope.** This used `c = 2.0` — a *moderate*, deliberately rescuable silencing (the `none` control
confirms S is not durably stuck even without `frm`). It shows `frm` *can* resurrect units that were dead at init;
it does **not** claim `frm` could revive an arbitrarily deeply-locked unit. The next experiment (master inhibitor)
tests exactly that limit.

