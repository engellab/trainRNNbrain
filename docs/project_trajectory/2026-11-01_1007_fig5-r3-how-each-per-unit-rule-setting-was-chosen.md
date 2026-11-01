## 2026-11-01 10:07 — fig5-r3: how each per-unit-rule setting was chosen, and every setting tested

Work package fig5-r3 (dead-relu-paper#90, #92, #84, #83, #88, #81, #77). Della asked for a password,
so everything below comes from the Mac copies in `data/remedy_reads_31oct/` and from git.

### How the setting shown in Figure 5 was chosen, rule by rule

The committee asked whether the shown settings were fixed before the results or picked from them
(#90). I reconstructed the history from commit messages, file times of the sweep folders and the
notebook. One notebook entry that recorded the noisy-synapse and rescaling choices, "2026-09-30
11:25 — synaptic noise, a maturity-target ladder, and dropout across sizes", was deleted by commit
65d6aa8 forty minutes after commit 879e707 wrote it; it survives only as `git show 879e707`.

| Rule | Shown | How chosen | Source | Verdict |
|---|---|---|---|---|
| Rate penalty | λ_frm = 0.1 | Taken over from the CDDM sweep of August, where it was half the configured 0.2 because 0.2 made every unit active within 150 iterations; written into the B2 rule (13 Oct) before any B2 network was trained | entry 2026-08-18_1547; `git show eec4a16^:docs/notes/agent-bplan.md` §B2 | pre-specified |
| Noisy synapses | σ_w = 1 | Largest of the first three levels (0.1, 0.3, 1: 369, 370, 545 active), each above the one below; put into the size series at 07:19 on 30 Sep, before σ_w = 2 and 3 were read (R² cost 0.17-0.31) | commits a1eaf02, 84b4485, 99c877a | after N = 1,000 results, before its size series |
| Rescaling | target m = 10 | Peak of the participation ratio over targets 6, 8, 10, 14, 20, 30 (5.93, 6.37, 7.32, 7.01, 6.52, 6.26); active count and R² nearly flat; 20 settings of earlier forms read before | commits 85aea77, 84b4485, 99c877a | after N = 1,000 results, before its size series |
| Prune-and-duplicate | s = 1, 0.025N per check, wait 1,000 | Capped version for its R² (0.944 against 0.914 uncapped); s = 1 because the count rose with s (685 to 752) and s = 3 was no longer a copy; an uncapped variant's size series had been read | commit 852a17f | after N = 1,000 results, before its size series |
| Read-out dropout | q = 0.2, β = 4 | Midpoint of the steep region (q = 0.175-0.25 at β = 4) of the whole-pass ("dead") variant, chosen to keep N = 2,000 below every unit active; the read-out-only variant shown uses it without its own tuning | commit 8bfd030; entry 2026-09-23_1309 | after N = 1,000 results (of a related variant), before its size series |

The combination test's λ_frm = 0.03 came from a pilot over 0.01 and 0.03 by a rule written before the
pilot (entry 2026-10-15_1342, read in 2026-10-16_1053). Its noisy synapses and rescaling are the
shown settings (launcher `slurm/SilentReLU_remedy_grid_della.slurm`: SIGMA_W=1.0, the rescale-target-10
block).

No setting was chosen on the results of its own size series. Four of the five were chosen after
results at N = 1,000, so Methods (Per-unit rules) labels them post hoc.

### Every setting carried to 200,000 iterations

`trainRNNbrain/experiments_and_analysis/table_rule_settings.py` prints one row per catalogue cell for
the five rules (57 rows; Supplementary Table 3 of the paper). Fold change is against the size-series
control networks at the same N and measurement point, as in Supplementary Table 2, and every fold
and ΔR² that the committee recomputed for #90 matches. Settings at N = 1,000: 2 for the rate penalty
(λ_frm = 0.1, and 0.03 from the combination test), 5 for noisy synapses, 26 for rescaling, 9 for
prune-and-duplicate (plus a 0.025N variant at N = 500 and 2,000), and 13 for read-out dropout.

Seven rescaling settings lower the count at 75,000 iterations: `rescale_revive` (the rule's first
form) 0.67-0.86, `norm_mode` m = 2.5 with the active-median length 0.82, `rescale_states` m = 0.2
and 0.5 0.91 and 0.93. At the floor crossing, where it exists, the range is 0.61-0.80. One rescaling
setting without row-length normalisation has 998 active units on average, with R² −45, 0.55 and 0.58.
The `norm_mode` cells with the active-median length predate commit 4287801, so their input rows
grew as well; the `recgrow` cells repeat them after that fix.

### Rate-penalty ΔR² at 75,000 iterations (#92)

No 36-draw score exists for the rate-penalty weights at 75,000 iterations, and Della was unavailable.
Supplementary Table 2 now prints the removal analysis's intact scores (9 draws, `a3_N*.json`) against
the control networks scored by the same analysis, marked: −0.151 (N = 500; one network inside a loss
burst at R² 0.557, the others 0.920-0.923), −0.021 (N = 1,000), −0.027 (N = 2,000) and −0.018
(N = 4,000). The #92 acceptance test asks for −0.021 ± 0.005 at N = 1,000. A 36-draw score would
replace these values when Della is back.

### Figures

Fig. 5 and Extended Data Figs. 7-8 were redrawn on wp/fig5-r3 after merging wp/style-r3 (39e5a1e).
Filled markers mean floor crossing and open markers 75,000 iterations, and fill means nothing else
(#83). The changes:
- Fig. 5a: the trace markers are gone; each median carries one open marker at 75,000 iterations.
- Fig. 5b, e: all markers open.
- Fig. 5e: N = 1,000 and 4,000 are now two sub-panels, where fill used to separate them. The
  random-removal curves (10, 30, 50% of units) were dropped; Results 5 never cited them, and the
  notebook entry 2026-10-31_1732 keeps their values.
- Fig. 5f: rows are named in words. The tick labels of neighbouring columns are now at least 3 mm
  apart.
- Rate-penalty shades: no violet darker than RATE_PENALTY.
- Extended Data Fig. 8d: measured and estimated values differ by shape, and the tick labels are
  horizontal.

`tests/test_paperstyle.py` passes on all three figures. The gap between rows of panels is at least
6 mm.
