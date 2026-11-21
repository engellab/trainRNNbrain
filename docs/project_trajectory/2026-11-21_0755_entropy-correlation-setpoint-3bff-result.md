## 2026-11-21 07:55 — Entropy and correlation set points on 3BFF: both terms pass the M09 criterion at 75,000 iterations

Rules: `2026-11-20_2307_rules-entropy-correlation-setpoint.md` (committed at 0ca7929 before any run). Pilot and
submission: `2026-11-20_2322_entropy-setpoint-pilot-and-screen-submitted.md`. Literature:
`2026-11-20_2305_entropy-correlation-setpoint-literature.md`. Branch `wp/entropy-setpoint`.

**Verdict.** Both terms pass on 3BFF at both of their λ values:
- (A), the per-unit activity-entropy set point, passes at λ_ent = 0.01 and at 0.1;
- (B), the pairwise-correlation set point, passes at λ_cor = 0.01 and at 0.1.

In all 12 networks the number of active units is above 288, the largest control count. Each arm's mean held-out R²
is within 0.0005 of the controls' mean, and the bar was 0.01. Stage 2 (Maze) therefore runs; its rules come in a
separate entry, committed before any Maze run.

**Terms used below.**
- *Active unit*: p = std(r) + q0.9(|r|) on a noise-free pass, and active if p ≥ 0.05 × the 95th percentile of p.
- *H_i*: the binary entropy, in bits, of Φ(μ_i/s_i), where μ_i and s_i are the mean and standard deviation of
  unit i's pre-activation h over steps and trials. Term (A) penalises (H_i − 0.75)².
- *c_i*: the mean Pearson correlation of unit i's h with every other unit's h. Term (B) penalises c_i².
- *Revived*: a unit inactive at some stored participation vector after iteration 1,000 and active at 75,000.

### Runs

Della job 18047724, array tasks 1-12, gpu-short. All 12 ended COMPLETED (exit 0) after 5 h 11 min to 6 h 43 min.
GPU time: 68.7 GPU-h for the screen plus 0.33 GPU-h for the pilot, 69 of the 100 allowed. Spock was not used.
Every network reached iteration 75,000, so all 12 are counted. Network `ent0.1` seed 4168822465 was rolled back
once by the rollback guard and then finished; the rules count a network that finished.

**Check 1 (submission config).** Among the 12 saved configs, the only differences are `lambda_ent`, `lambda_cor`,
the seed, the tag and the output paths. `model.init_draws` is `shared` in all 12. The comparison with the control
command at main was done before the rules entry (Check 1 there).

### Read

The read ran as Della CPU job 18067909 from `~/trainRNNbrain_entsp_ana` at 0ca7929, using
`entropy_setpoint_read.py read`. The verdict was then computed on the Mac with `entropy_setpoint_read.py verdict`.
The output, `entropy_setpoint_reads.json`, is kept outside the repository in `firstmate/data/entropy-setpoint/`.

The control and rate-penalty values reproduce the numbers the rules quote:
- controls: 279, 288, 242 and 236 active units, R² 0.9514, 0.9517 and 0.9509;
- rate penalty: 985, 995 and 935 active units, R² 0.9329, 0.9304 and 0.9276.

The arm networks store their participation every 100 iterations, from 0 to 74,900. Their probe nearest 75,000 is
therefore 74,900, while the controls' probe is 75,000. Held-out R² is measured on the final weights, at 75,000.

**Check 2 (independent).** The check script, `firstmate/data/entropy-setpoint/scripts/check_reads.py`, was run as
Della job 18068824.
- It recounts every network's active units from the stored participation vector with its own numpy rule. The
  recount equals the reader's count in all 19 networks: 4 controls, 3 rate-penalty networks and the 12 arm networks.
- For one network per arm, it runs the h equation forward in numpy and compares with the torch pass on the same
  weights. The noise-free R² agrees within 6 × 10⁻⁹; the pass threshold was 10⁻⁵.

### Per arm

| Arm | Active units at 75,000 (per network) | Mean | Mean over controls' mean (261) | Mean held-out R² | Difference from controls' 0.9513 | Verdict |
|---|---|---|---|---|---|---|
| controls (not retrained) | 279, 288, 242, 236 | 261 | 1.00 | 0.9513 | | |
| `ent0.01` | 383, 424, 411 | 406 | 1.55 | 0.9512 | −0.0002 | passes |
| `ent0.1` | 384, 380, 454 | 406 | 1.55 | 0.9512 | −0.0002 | passes |
| `cor0.01` | 491, 422, 460 | 458 | 1.75 | 0.9514 | +0.0001 | passes |
| `cor0.1` | 659, 596, 611 | 622 | 2.38 | 0.9509 | −0.0005 | passes |
| rate penalty λ_frm = 0.1 (description only) | 985, 995, 935 | 972 | 3.72 | 0.9303 | −0.0211 | fails on R² |

The two entropy arms have the same mean count and mean R² (sums of 1,218 units in both). This match is a
coincidence of the per-network values, which differ.

Compared with the rate penalty at λ_frm = 0.1: the rate-penalty networks have 1.6 to 2.4 times as many active units
as the set-point arms. Their R² is 0.021 below the controls' mean, while every set-point arm is within 0.0005.

### Per network

| Arm | Seed | Active units | Held-out R² | Revived | Skipped updates | Median \|H_i − 0.75\| (bits) | Median \|c_i\| |
|---|---|---|---|---|---|---|---|
| control | 3374117298 | 279 | 0.9514 | 191 | | 0.662 | 0.431 |
| control | 1914910909 | 288 | 0.9517 | 203 | | 0.607 | 0.372 |
| control | 3866906883 | 242 | 0.9509 | 179 | | 0.684 | 0.394 |
| control | 4186074845 | 236 | (no 75,000 weights) | 165 | | | |
| `frm0.1` | 1832668511 | 985 | 0.9329 | 876 | | 0.218 | 0.042 |
| `frm0.1` | 15698289 | 995 | 0.9304 | 951 | | 0.200 | 0.026 |
| `frm0.1` | 1571321706 | 935 | 0.9276 | 894 | | 0.212 | 0.058 |
| `ent0.01` | 2069412482 | 383 | 0.9509 | 332 | 499 | 0.030 | 0.145 |
| `ent0.01` | 4001044797 | 424 | 0.9514 | 398 | 998 | 0.015 | 0.133 |
| `ent0.01` | 3509941279 | 411 | 0.9512 | 332 | 0 | 0.020 | 0.110 |
| `ent0.1` | 4200852863 | 384 | 0.9503 | 277 | 0 | 0.009 | 0.414 |
| `ent0.1` | 1256299004 | 380 | 0.9515 | 281 | 0 | 0.006 | 0.421 |
| `ent0.1` | 4168822465 | 454 | 0.9516 | 392 | 500 (1 rollback) | 0.007 | 0.121 |
| `cor0.01` | 4164253393 | 491 | 0.9507 | 450 | 499 | 0.153 | 0.011 |
| `cor0.01` | 500037542 | 422 | 0.9520 | 383 | 991 | 0.127 | 0.008 |
| `cor0.01` | 1499528544 | 460 | 0.9515 | 386 | 0 | 0.115 | 0.011 |
| `cor0.1` | 2905538948 | 659 | 0.9502 | 619 | 499 | 0.126 | 0.006 |
| `cor0.1` | 1744357236 | 596 | 0.9514 | 531 | 0 | 0.121 | 0.002 |
| `cor0.1` | 3160364235 | 611 | 0.9510 | 560 | 0 | 0.169 | 0.001 |

The H_i and c_i columns are measured at 75,000 iterations, on the held-out batch, with the training noise and noise
seed 20261121.

**Skipped updates.** The spike guard skips an update whose gradient norm is more than 10⁶ times the median of the
last 1,000. Six of the 12 networks had skips: 499 or 500 updates in four of them, and 991 and 998 in the other two. The
log does not record which updates were skipped. The counts fit blocks of about 500 consecutive skips, which the guard
would produce: its reference is the median of the last 1,000 gradient norms, so a run of large gradients is skipped
until it fills half of that window. That cause was not traced update by update.

Skipping is not specific to the set-point terms. The training logs of control seed 4186074845 and of the three
rate-penalty networks (FFseed4.15657728_4 and B2frm.15598510_1-3, runs of 200,000 iterations whose 75,000-iteration
states are read here) record 499 skips for the control and 2,393, 2,285 and 2,864 for the rate-penalty networks. The
logs of the other three controls have no skip-summary line.

### Descriptive read-outs

**Manipulation check: both terms reached their own targets.**
- Median |H_i − 0.75| is 0.006 to 0.030 bits in the six entropy networks, against 0.61 to 0.68 in the controls.
- Median |c_i| is 0.001 to 0.011 in the six correlation networks, against 0.37 to 0.43 in the controls.

**Revival.** The controls revive 165 to 203 units, because units cross the activity bar back and forth (the
"flicker baseline"). The set-point arms revive more:
- entropy arms: 277 to 398 units;
- correlation arms: 383 to 619 units;
- rate penalty: 876 to 951 units.

**The prediction was wrong in direction.** The rules and the literature entry predicted that (A) would revive units
and (B) fewer or none. The reasoning was that a Pearson correlation does not change when one unit's h is shifted or
scaled. The outcome is the reverse: at λ = 0.1, (B) has 622 active units on average and (A) 406.

**The active counts are still falling at 75,000 iterations** (figure). Median active count in the 2,000 iterations
before 50,000 and before 75,000, per network:
- entropy arms: 460 → 382, 553 → 431, 457 → 430, 469 → 388, 478 → 387 and 621 → 460, a fall of 27 to 161 units;
- correlation arms: 513 → 512, 557 → 441, 559 → 473, 654 → 638, 672 → 587 and 686 → 625, a fall of 1 to 116 units;
- controls: changes of +5, −14, −52 and −57 units;
- rate penalty: rises of 45 to 54 units.

The entropy arms peak at 700 to 800 active units between 10,000 and 30,000 iterations. The pass criterion is read at
75,000 iterations, as fixed in the rules, and holds there. The figure gives no evidence that it would still hold
later.

**Post hoc, not in the rules: the entropy term is met by skewed pre-activations.** The script
`firstmate/data/entropy-setpoint/scripts/unit_detail.py` (Della jobs 18068945 and 18069049) examined one network per
arm, plus a second `ent0.1` network and one control. In the three entropy networks:
- 89% to 99% of the inactive units have H_i within 0.05 bits of 0.75;
- their μ_i/s_i is about −0.8, so Φ(μ_i/s_i) is about 0.21, the on-fraction the term aims for;
- their h is above 0 on only 5% to 7% of the noisy samples (median per network), and on 1.5% to 3.5% of the
  noise-free ones;
- their h has a skewness of −1.2 to −1.8, so a long negative tail lowers the mean and raises the standard deviation.

The binary entropy of the actual on-fraction is 0.29 to 0.37 bits, against the term's 0.75. Term (A) depends on h
only through μ/s, which gives the on-fraction only when h is Gaussian; these units are not. In the control network,
the inactive units are above 0 on 1.8% of noisy samples, with skewness +0.67. A version of (A) built on an empirical
on-fraction (for example the batch mean of a sigmoid of h/τ) would not have this gap. That version was not run.

In the correlation networks, the inactive units have |c_i| ≈ 0.002 to 0.011, the same as the active units. Their h
is above 0 on about 6% of noisy samples.

### Figure

`img/internal_figures/entropy_setpoint_active.{pdf,png}` plots active units against iterations for every network
of every arm. Panel A shows the entropy arms and panel B the correlation arms. Both panels show the four controls
and the three rate-penalty networks, with each trace joined along its warm-start chain.

### What follows

- Stage 2 runs, as firstmate pre-authorised (21 Nov): both terms are tested on Maze at N = 1,000 and 2,000, three
  networks per size, under a Maze rules entry committed before any Maze run.
- The paper is unchanged. The set-point terms are not among the paper's techniques, and no claim in
  `docs/paper_audit.md` rests on this result.
