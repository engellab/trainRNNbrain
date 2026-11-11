# Final review for density: Results cut to the main points, Figs. 3 and 5 decluttered

Pavel, 11 Nov: one last review. Every plot must be understandable at a glance, and the message must not
be lost in a numerical recital of results; a result that is incidental and visible in a figure is not
reported as a number in the text.

## What changed (dead-relu-paper branch wp/final-density, commits 83da538 and 37acd02)

- Results, five subsections: 2,977 to 2,480 words and 243 to 149 numerals (counted from the LaTeX source
  with references and macros stripped). No result, number or scope changed. Each subsection keeps its claim,
  one or two headline numbers and the figure pointer.
- Sentences whose numbers were stated only in Results or a caption moved to Methods: M04 (the 506 units
  below 2e-4 and the median participations of the Fig. 1b network), M06 (R^2 and success rates after removal
  and after further training), M07 (40 to 60 low-active CDDM units above the gradient threshold), M08 (the 17
  values with a direction on all three tasks; the weight-decay 1e-4 counts), M09 (one of three 3BFF
  prune-and-duplicate networks of 4,000 units has a floor crossing; the noisy-synapse verdict untested after
  the trainer correction), M12 (all nine CDDM networks flat, losing 34 to 64 active units per 100,000
  iterations).
- Disclosures kept in Results: the rule that counts only networks that learned (post hoc), the R^2
  tolerance and the one-value rule (post hoc), the flat-loss threshold timing, the failed DMTS test, the
  failed rate-penalty size test, the CDDM and Maze transfer tests without a verdict.
- Fig. 3 (`fig_paper_F3_tasks.py`, this branch 2dcd638): networks that did not learn are small grey crosses
  behind the means; the network-count numbers and the dashes for values not run are removed; a pale span in
  the task colour marks every value whose direction is "increase" under the existing `direction()` rule.
  33 slots are shaded (3BFF 16, CDDM 15, Maze 2); no value is shaded on all three tasks. Printout diff
  before/after: identical except the text-artist count and the new shaded list.
- Fig. 5a,b (`fig_paper_F5_tasks.py`, 2dcd638): the same grey crosses, counts removed, a pale span behind
  the prune-and-duplicate column. Panels c and d untouched.
- Captions of Figs. 1, 3 and 5 and the pointers in Supplementary Table 7 and the combination-test Note
  updated to match.

## Review

An independent reviewer (fresh agent, read-only, given the diff and not the author's conclusions) checked
every new sentence against the old text, Methods and tables, the homes of every removed number, the
captions against the redrawn figures, and the citation order. Five required changes, all applied in
37acd02: "most 3BFF networks" restored in the flat-loss topic sentence; "for five consecutive probes"
restored in the learned-point definition; "94 to 155 times" restored; the post-hoc sentence names the claim
it bears on; the CDDM and Maze removal changes (at most 0.003 and 0.03) and the two medians restored in
Methods. `tools/ed_order.py main.pdf` prints ORDER OK; the build has no undefined reference.

## Left as it is

Fig. 5d (performance after removing the k least participating active units against k random ones) is the
one main-figure panel that does not read at a glance. It was not redrawn.
