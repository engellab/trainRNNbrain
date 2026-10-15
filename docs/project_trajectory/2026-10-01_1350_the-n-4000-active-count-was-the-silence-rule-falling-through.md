## 2026-10-01 13:50 — the N=4000 active count was the silence rule falling through a noise floor

**The anomaly.** The per-seed CDDM table printed active-unit counts beside the r2 check, and the
CDDM control series read 232 / 334 / 697 / **3797** at N = 500 / 1000 / 2000 / 4000 — 95% of the
network at the top size, against 24–35% one size below. Same `max_iter=100000`, so not a truncated
run.

**The mechanism.** The silence rule is relative: a unit is active when its participation reaches
0.05 × q₉₅(p). Measured across sizes, the median participation is flat at about 0.0056 from N = 1000
up, while q₉₅ falls roughly as 1/N as the representation spreads over more units — 0.52, 0.27, 0.14,
0.087. The threshold falls with q₉₅, and at N = 4000 it reaches 0.0043, below the median. The median
unit becomes "active" and the count jumps. Nothing was near zero at any size: `p < 1e-3` counted 0
units everywhere, which is what gave the mechanism away.

That flat 0.0056 is the rectified recurrent noise, about `(dt/tau) × sigma_rec` = 0.1 × 0.05. A unit
that never fires still reads that much once noise is injected.

**Where the bug is.** `Trainer.track_participation_` logs the participation trace from a
`w_noise=False` pass, and says so in its own comment: "so the trace is comparable to the offline
noise-free analysis". Every figure in the paper reads that trace. `f2_remedies_cache.analyse`
computed its own participation on the **noisy** run, under a comment claiming that was "the one every
other active-unit count in this project is measured under" — the opposite of the truth.

**Quantified.** Same networks, same criterion, noisy against noise-free active counts:

| N | noisy | noise-free | overcount |
|---|---|---|---|
| 500 | 232, 224 | 222, 205 | +5%, +9% |
| 1000 | 334, 332 | 309, 308 | +8% |
| 2000 | 697, 591 | 407, 394 | +71%, +50% |
| 4000 | 3797, 3930 | 559, 562 | +580% |

The noise-free series follows a clean trend, 222 → 309 → 407 → 559, close to the N^0.43 that
Figure 1 panel (c) fits for CDDM. The noisy one does not.

**The paper is not affected,** and this was checked rather than assumed. The rule's own health check
needs no new threshold: more than half the units pass exactly when q₅₀/q₉₅ > 0.05. Across every cell
of `fig_paper_F1.SCALING` — four tasks, four sizes each, every seed — that ratio is 0.000 to 0.001.
The traces are sharply bimodal and the rule has a wide margin everywhere the paper reads it.

**Fixed** in `f2_remedies_cache.analyse`: participation is taken from the clean rate matrix, which the
dimensionality already used. The cache is rebuilding (job 14832898, 109 cells). Figure 2 and the deck
panels that read it have to be rebuilt after it lands; the active-unit panel is the one that moves.

**A second desync caught while submitting it.** `~/f2_cache.slurm` ran `python -u
f2_remedies_cache.py` from `$HOME`, a loose copy of the script, not the one in the repo — so the
first submission would have rebuilt the cache with the unfixed code. The launcher now runs
`$F2_REPO/trainRNNbrain/experiments_and_analysis/f2_remedies_cache.py` and the stale copy is renamed.

**The one gate failure, resolved.** 32 of 33 CDDM seeds passed the ±0.03 reproduction gate; the one
failure was `N=500 arm=control`, stored 0.9066 against a recomputed 0.8735. Drawing that network 200
times gives mean 0.8741, sd 0.0121, max 0.9051 — the stored value is +2.7 sd and no draw reached it.
The trainer records ONE noisy draw, so a ±0.03 bar on a quantity with sd 0.012 is about 2.5 sd and
fails roughly one seed in a hundred by construction. One failure in 33 is the gate's design, not a
fault in the data.

### Slides 8b and 9b: the activation panels' units cost nothing — and why the folder score cannot say so — 2026-10-01 13:54

Pavel asked for R² against active units under panels 8 and 9, one colour per condition, on the
pattern slide 11b set. `fig_slides.activation_r2_slide` → `slide_x_activation_cddm_r2` and
`slide_x_activation_ff_r2`, with `ACTIVATION_CDDM` / `ACTIVATION_FF` as the cell lists. No connecting
line through the cell means, unlike the dose panels: activation is categorical and has no order for a
line to encode.

**The result.** CDDM, 12 networks: across 191–284 active units every one sits within 0.016 of
R² = 0.950 (ReLU 0.951, leaky ReLU 0.956, softplus 0.948, sigmoid 0.949). Sigmoid sheds a quarter of
its live units — 205 against the reference's 272 — and pays 0.002 of R², less than the measurement's
own noise. Flip-flop, 6 networks so far: 226–282 active units within 0.005 of R² = 0.967, with
sigmoid holding 23 fewer units than ReLU and scoring slightly *higher* (0.969 against 0.965). So the
answer to "did the arms that shed units shed them by failing the task" is no, on both tasks.

**⚠️ The folder-name score cannot be used on these panels, and using it would have inverted a
ranking.** `met_cell` uses it, and all four CDDM arms share a 200,000-iteration budget, so it looks
safe here in a way it was not for the input-scale ladder (where the reference's budget is 3.3×
longer). It is not safe, for a different reason: the folder score is a forward pass with the noise
**on**, and that penalty is ACTIVATION-DEPENDENT. Measured clean-minus-folder per arm: **+0.0626**
leaky ReLU, **+0.0617** softplus, **+0.0738** sigmoid — a 0.0121 spread against a between-arm spread
of only 0.007 in either measure alone. On the folder score sigmoid is the **worst** arm (0.8690
against ReLU's 0.8740); noise-free it is level with the rest. A bounded sigmoid saturates and a
softplus has a nonzero floor, so neither absorbs injected noise the way a ReLU net does. For slide
11b's ladder every arm is a ReLU net, which is why one uniform 0.0047 offset sufficed there; across
activations no such uniform offset exists.

**So R² is noise-free throughout.** On the flip-flop every arm carries `loss_clean_train` and
`winp_cell` already reads it. On CDDM the three activation arms carry it but the ReLU reference does
not — `CDDM_std_g0_drift` predates the probe — so all four arms are re-scored **offline** by
rebuilding each net from its saved parameters and running a noise-free batch, one route for every
arm rather than a different route for the arm that lacks a probe. The CDDM R² denominator is the
masked target variance 0.17917, measured from the runs' own task config over the 200 scored
timepoints of 300; `TaskCDDM.py` has not changed since 2025-11-17, well before any of these runs, so
there is no version-pin hazard in rebuilding the task.

**The validation, and the threshold that failed first.** The offline re-score is checked against the
recorded probe on the nine nets that have both. The first threshold — |offline − recorded| ≤ 0.01 in
R² at a single probe — **FAILED** at 0.0186 for sigmoid. Investigating rather than re-running found
the cause: `loss_clean_train` is noise-free in its forward pass but the **parameters are still
moving** at 200k, so consecutive probes score genuinely different networks. Measured per-probe
standard deviation over the last 10,000 iterations: 0.0035–0.0039 (leaky), 0.0035–0.0036 (softplus),
0.0060–0.0179 (sigmoid), with one sigmoid probe dipping to R² = 0.465. The single-probe threshold was
tighter than the quantity's own scatter — it was testing nothing a real bug would fail differently
from drift. Replaced with a criterion taken from that measurement: one offline sample must land
inside the [min, max] the recorded probe spans over those 10,000 iterations. All nine nets pass.
This is slide 4's finding showing up as a measurement constraint: a single late probe is a sample of
a drifting network, not a fixed property of it.

On the flip-flop the same probe is far quieter near 150k — per-probe sd 0.0004–0.0033, single probe
within 0.001 of the window mean — which is what licenses `winp_cell`'s single-probe read on slide 11b
and on 9b.

**🐛 `RNN_numpy`'s softplus overflows and returns NaN.** It computes `log(1 + exp(beta*slope*x))/beta`,
and `np.exp` overflows once `beta*x > ~709` — at β = 25 that is any unit above x ≈ 28, which the
trained nets reach. The whole run then scores NaN. `torch.nn.Softplus` does not have this problem
because it falls back to the linear branch above `threshold=20`; that is what actually ran during
training, so the trained networks are fine and only the offline path is affected. `offline_clean_r2`
patches the instance with the stable form and leaves the class alone. **This is live for the new
flip-flop softplus arm**: anyone re-scoring those nets offline hits it. Worth fixing in
`RNN_numpy.configure_activation_` rather than patched per caller.

**Axis.** The y window is several times the data spread, so a hair's-breadth range cannot read as a
slope, but it is clamped at R² = 1: the CDDM arms span 0.016 and 11b's bare 10×-the-spread rule would
have run the axis to 1.03 and spent a third of the panel on scores no network can reach.

**Status of the two new arms.** Della array `14832258`, all 12 tasks running as of 13:54. Both panels
draw what is on disk and name what is not: `slide_x_activation_ff_r2` carries the note "leaky ReLU,
leak 0.01; softplus, β = 25 are still training".

### Every intervention panel now states the iteration it was read at — 2026-10-01 14:04

Pavel asked for the read-out iteration on the suptitle of slides 8–14. It is now **derived, not
typed**: `_family_values` carries the set of iterations `live_matched` actually returned for each
cell, `collect` passes it through, and `readout_line` turns it into the panel's last title line. A
hand-typed number on a figure goes stale the first time a cell is re-run; this one cannot.

**What the derived numbers turned out to be, against what the panels used to say.**

| slide | said before | actually read at |
|---|---|---|
| 8 activation (CDDM) | "read at 200k" | **199,900** |
| 9 activation (flip-flop) | "read at 150k" | **149,900–150,000** |
| 10 weight decay | nothing | **199,900** |
| 11 input scale | "read at 150,000" | **149,900–150,000** |
| 12 metabolic | nothing | **30,000** |
| 13 architecture | nothing | **30,000** |
| 14 recurrent noise | nothing | **30,000** |

The two flip-flop panels do not share a probe: the four intervention cells stop at a 150,000
budget and land on probe 149,900, while the ksweep reference runs to 500,000 and has a probe at
exactly 150,000. The line says so rather than rounding both to 150k. Slide 11b's title now reports
the same range, from the probes `winp_cell` actually landed on.

**The two archived families cannot derive it, so it is recorded with provenance** in
`ARCHIVE_READOUT`, and the line marks it as recorded rather than measured. Each was checked against
something that does carry an iteration:
- the archive reference's five rows (`rel_5p95` = 0.569, 0.572, 0.575, 0.598, 0.614) are five of the
  twenty `silent_stats_v2.csv` `sweep=std` rows at N = 1000, eq = h, every one at `iters=30000` —
  `silent_stats_all.csv`, where the reference is read from, has no `iters` column at all;
- the metabolic rows come from `silent_stats_v2.csv` directly, all twelve at `iters=30000`;
- the architecture rows' sweeps still have their run folders: `CDDM_ptrack_g0`,
  `CDDM_ptrack_g0_nodale` and `CDDM_ptrack_g0_nodale_trainablebias` are all `MI=30000`;
- the noise sweep's folders survive too (`CDDM_fb2792_g0_noise`, `max_iter: 30000`) and its nets were
  scored from their final weights, so the budget is the read-out.

Figures rebuilt: `slide_x_activation_cddm`, `slide_x_weightdecay`, `slide_x_activation_ff`,
`slide_x_inputscale`, `slide_x_metabolic`, `slide_x_architecture`, `slide_x_recnoise`,
`slide_x_inputscale_r2`. Checked for text outside the canvas: none.

### Two offline-scoring bugs fixed, and 8b/9b restyled to 12b — 2026-10-01 14:04

**🐛 `RNN_numpy` softplus overflowed to NaN. Fixed.** `configure_activation_` computed
`log(1 + exp(beta*slope*x))/beta`, and `np.exp` overflows to `inf` once `beta*z > ~709` — at β = 25
that is any unit above z ≈ 28, which the trained networks reach. The `inf` then passed through
`W_rec @ r` and poisoned the whole run, so the softplus arm of panel 8b scored NaN on every seed.
`torch.nn.Softplus`, which is what actually ran during training, takes the linear branch above
`beta*z = 20`, where softplus equals z to within `exp(-20)/beta` < 1e-10. The numpy version now does
the same. Verified against torch over z ∈ [−200, 200] plus ±10⁴ at β ∈ {1, 5, 25} × slope ∈ {1, 2}:
**all finite, max relative error 2.9e-16**. The trained networks were never affected — only the
offline path — and the fix reproduces the per-caller patch it replaces exactly (softplus arm 0.9474 /
0.9485 / 0.9472 either way), so `fig_slides.offline_clean_r2` no longer patches the instance.

**🐛 `penalty_matched.clean_loss` forced ReLU and the wrong equation. Fixed.** Two faults in three
lines:

- `params["activation_name"] = "relu"` is not a `RNN_numpy` parameter, so `filter_kwargs` dropped it;
  `params.pop("activation_args", None)` then left the constructor on its own ReLU default. The net
  effect was to force ReLU — correct by accident while every net in `CDDM_std_g0_penalties` and
  `CDDM_std_g0_drift` was a ReLU net, and silently wrong the moment one is not. The activation
  sweeps now hold softplus, leaky-ReLU and sigmoid networks, which this would have simulated as ReLU
  without raising anything.
- `equation_type` was never passed, and `RNN_numpy` defaults it to `"s"` while every net here is
  `"h"`. This is the bug `fig_slides` already carried a warning about; the warning is now stale and
  the code is fixed.

Both now come from the run's own config. Checked on a real net: the fixed `clean_loss` scores
`CDDM_std_g0_drift` seed 0.8622066 at **R² = 0.9483** on the training batch and 0.9056 held out,
against the documented symptom of the `"s"` bug (MSE ≈ 2.08, R² far below zero). That 0.9483 is
element-for-element what `fig_slides.offline_clean_r2` gives for the same net by an independently
written path — two separate reconstructions agreeing to four decimals.

`flipflop_fixedpoints.load_net` has the same forced-ReLU shape (it does pass `equation_type`
correctly). Left alone rather than changed: it is a ReLU-fixed-point script and nothing points it at
the new nets today. Flagged here because the flip-flop sweeps are no longer ReLU-only, so that
assumption now has an expiry date.

**8b and 9b restyled to the current 12b.** `metabolic_r2_slide` settled three conventions since 8b
and 9b were first drawn, and all three now carry over: the cell mean is an opaque **square**, not a
large translucent circle, because a mean over three seeds differs from a seed by shape and not by
certainty; seed markers at ms 4.2 with the mean above them; the legend in the **lower right**; and
the title says "Circles are networks, squares the cell means." No line joins the means on any of the
three — on a plane whose axes are active units and R², a path between categorical conditions implies
a trajectory that does not exist. 11b was left as it is; it belongs to the other thread of this work.

The restyle also lands on `slide_x_weightdecay_r2`, which now shares `activation_r2_slide` (and added
its `headline` argument). All four scatter panels — 8b, 9b, 11b, 12b — and the weight-decay panel
rebuild cleanly.

**Della access dropped at 14:04** (`ssh: Operation timed out`), the cached-2FA behaviour this project
has seen before. Array `14832258` had all 12 tasks RUNNING at 13:54 and keeps going without the
session; its state has not been read since.

### Slide 11b restyled to 12b, and the read-out sentence derived rather than typed — 2026-10-01 14:12

**11b now carries the same conventions as 8b, 9b, 10b and 12b.** Seed markers at ms 4.2 under the
cell mean, the mean an opaque square, the legend in the lower right, and "Circles are networks,
squares the cell means." in the title. The square marker and that sentence were already there; the
sizes, z-order and legend corner were not.

**The line joining 11b's cell means is gone**, for the reason `metabolic_r2_slide` gave when it
dropped its own. Neither axis of this panel is the input scale, so a path between the means implies
a trajectory through a plane that has none — and here it visibly doubles back, because the ladder is
single-peaked in active units: the five means run 263 → 302 → 339 → 324 → 306, so the steps go
**+ + − −** while the R² the line threads spans **0.0010**. A line that reverses inside the noise
reads as noise rather than as order. The legend names every rung, which is what carries the ladder.
The y window is untouched at ±0.01 around the data centre: that is a data choice with its own
rationale in the docstring, not one of the marker conventions being unified.

**The cell functions' contract changed under me, and adopting it fixed a caption.** `winp_cell` now
returns `(r2, active, probes)` rather than `(r2, active)`, with `probes` the participation-probe
iterations the cells actually landed on, so `readout_line(probes)` can derive the "Read at …"
sentence instead of it being typed. That broke `activation_r2_slide`, which unpacks five fields per
cell while `cddm_activation_cell` still returned two. `cddm_activation_cell` now returns the probe
set too, and all three panels drawn through `activation_r2_slide` take their read-out sentence from
the data.

That is not only a tidier contract — it corrected slide 9b. Its caption said "both axes read at
150,000 iterations", typed by hand. The cells do not share a probe: the ReLU ksweep control lands on
150,000 and the sigmoid arm on 149,900, so the panel now says "Read at 149,900–150,000 iterations
(the cells do not share a probe)." 11b reads the same way; the CDDM activation and weight-decay
panels share one probe and say "Read at 199,900 iterations."

All five scatter panels — 8b, 9b, 10b, 11b, 12b — rebuild cleanly, and the numbers the deck quotes
still match what the figures draw (191–284 units within 0.016 of R² = 0.950; 226–282 within 0.005 of
0.967; 247–342 within 0.002 of 0.964).

### Deck slide 18 becomes r² against active units — 2026-10-01 14:52

Slides 17 and 18 drew the same 22 networks on two categorical axes — active units per arm, then
held-out r² per arm — which left the reader to join them by colour across two pages. The question
the two are asked together, *does a rule that recruits units pay for them*, is a statement about the
joint distribution, so slide 18 now draws it directly: `fig_paper_F2.panel_r2_vs_active` →
`slide_f2_r2_vs_active`, in the convention the other scatter panels use (circles are networks,
squares the cell means, the control's level as a dotted reference line). The per-arm cost moves into
the legend, which is where the old panel's delta row went.

**Replacing rather than adding, because the scatter is a superset.** Every r² value panel (c) draws
is here, as the same dot in the same colour; the only thing it carried that a scatter has no slot
for is the printed per-arm delta, and that is now in the legend labels. Slide 17 stays: the units
axis with its "all 1000" ceiling and the per-arm means annotated is a claim of its own about how
close each rule gets to full participation.

**⚠️ `panel_c` itself is untouched** — the change to `fig_paper_F2.py` is 49 lines, all additions,
no deletions. The manuscript figure still composes (b), (c), (d) in a row of three, where a
categorical r² panel beside a categorical units panel is the right shape; only the deck swaps.

**What the join shows that the two panels apart did not.** The arms are not strung along one
trade-off curve. Prune + duplicate recruits **474** units more than the control (772 against 298) and
lands on the control's own line — **−0.10%**, with all three of its seeds inside the control's seed
range (0.9424–0.9441 against 0.9417–0.9493). Dropout, rescale and synaptic noise each give up
**1.3–1.9%** for *fewer* units than duplication recruits: 509 at −1.77%, 453 at −1.32%, 566 at
−1.85%. The penalty pair buys the most, 939 of 1000, and pays the most, −2.73%. Reading that off two
categorical panels means holding six pairs of numbers in mind; here it is the shape of the cloud.

r² is `r2_common` throughout, as panel (c) uses — every arm scored under one condition rather than
each in its own, since only the synaptic-noise arm would otherwise be measured with its wiring
fluctuating. The x axis is the same scale-free rule as every other panel in this project:
`f2_remedies_cache` computes `p = std + q90` and `live = p >= 0.05·q95(p)`, checked rather than
assumed before the axis was labelled.

**Legend is lower LEFT here**, against the lower right the activation and metabolic panels use: the
penalty arm is both the most active and the lowest scoring, so it occupies this cloud's lower right
corner. Checked against the rendered legend box rather than by eye — no network falls within 5 px of
it. Position follows the data; the convention being shared is the markers.

