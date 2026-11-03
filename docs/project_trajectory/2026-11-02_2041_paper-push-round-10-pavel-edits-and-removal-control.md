## Paper push, round 10: Pavel's edits of 2 Nov, and the removal result reframed

Pavel's items 27-36 (`~/Documents/Projects/paper_push/PAVEL_FEEDBACK_2026-11-01.md`), applied
paper-wide in PRs #159, #160, #162, #163 and #164 (each approved by an independent reviewer; paper
main b2d4978, 61 pages, clean build; figure code trainRNNbrain 57d9b2c).

- "Training techniques" replaces "per-unit rules" everywhere (text, captions, tables, Methods and
  Supplementary headings, Fig. 3a and the combination-figure labels). "Rule" survives only for the
  pre-registered pass/fail rules, the tie rule, the pilot's choice rule and the e-prop rule.
- "Default settings commonly used in academic studies (Extended Data Table 1)" replaces "our default
  settings"; Methods defines the default settings as the values of Extended Data Table 1 and does
  not repeat the literature claim.
- New paragraph rule (plain-english rule 10): one or two ideas per paragraph; paragraphs under about
  five printed lines merged with a related neighbour, up to about fifteen lines.
- Introduction: Yang et al. sentence commented out; machine-learning paragraph rewritten around
  "dying ReLU"; techniques introduced by their purpose; no false "no term of the loss rewards
  activity" rationale; the obvious dimensionality sentence removed (dimensionality now stated once,
  in the Extended Data Fig. 2a caption: at most 1.1% higher over all units).

**The removal result (decision #161).** Pavel asked what the catch is in "remove the inactive units,
then train briefly". Entry 2026-10-31_1755 already held the answer: networks with as many random
units removed and trained for the same 3,000 iterations also recover (CDDM 0.821-0.828 against
0.825-0.829 intact; two of three DMTS networks; 3BFF 0.018-0.025 below inactive removal). Brief
retraining therefore does not single out the inactive units. The comparison without retraining
does, in all nine networks: inactive removal leaves R² at 0.26-0.52 (3BFF), unchanged within 0.003
(CDDM) and 0.75-0.99 (DMTS); random removal drops it to 0.08 or below. The Introduction and
Discussion now lead with that comparison; Results 1 reports both, with the control. Training before
removal was 20,500-67,000 iterations (manifest.tsv), not 75,000 as the round-10 brief first said
(#157).

**Clustering classes.** The "4 classes over all units, 3 over active" result is not used as an
example: entry 2026-11-01_0741 shows the extra 3BFF class is mostly inactive, non-selective units
(flat normalised profile), and CDDM silhouette margins are 0.01-0.07. The Extended Data Fig. 2e,f
caption states both caveats.
