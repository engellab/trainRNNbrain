## 2026-10-31 22:10 — Round-2 decisions, and the status board brought in line with them

Work package board-r2 (dead-relu-paper#50, which absorbs #48 and #13 item 2). Branch `wp/board-r2`
from trainRNNbrain `main` 802f17f. No new runs and no new numbers: every number below is quoted from
the issue it names or from the notebook entry it names.

This entry records every decision the round-2 status board follows: for each, the issue, the
options, the choice and the reason, with the decision comment quoted. Decisions are the team's
(PROTOCOL.md, 31 Oct: Pavel delegated them and reviews only the finished paper).

### Round 2 (editor, 31 Oct, plan posted on dead-relu-paper#1 at 21:48)

**1. Population statistics: measure** ([#34](https://github.com/ptolmachev/dead-relu-paper/issues/34#issuecomment-6553682701)).
- Problem (committee, writing lens): the Introduction and Discussion say that inactive units change
  what a population analysis measures, and no Results item computes such a statistic.
- Options: (1) measure the statistics the Introduction names, over all N units and over active
  units only; (2) cut the claim to statistics that depend on N by construction (fractions and counts
  of units).
- Choice: option 1. Quoted: "Option 1 is chosen because the measurement is small (under one
  CPU-hour on nine existing networks, no training), it turns the paper's motivation from an
  assertion into a result, and either outcome is reportable: if a statistic does not move, the paper
  says which statistics are safe, which is also useful to a reader."
- Rule (fixed on #34 before any value was computed): four statistics (participation-ratio
  dimensionality; the same on rates normalised to unit variance; selective and mixed-selective unit
  counts; functional classes by k-means on selectivity profiles) on the nine N = 1000 default
  networks of Fig. 1d. Per statistic and task: *changes* if |S_all − S_act| ≥ 0.10 |S_act| in all
  three networks, *unchanged* if < 0.02 |S_act| in all three, *mixed* otherwise. The Introduction and
  Discussion keep a general claim only for statistics labelled *changes* on at least two of the three
  tasks.
- Board: new claim row C8, status Pending, filled by pop-stats (#56). The "Parked" list now names
  the four statistics as the one exception to "single-unit selectivity, clustering, cell types".

**2. The rate penalty is told once, in Results 5** ([#35, item 4](https://github.com/ptolmachev/dead-relu-paper/issues/35#issuecomment-6553683245)).
- Problem: on CDDM (Results 3) the rate penalty makes all 1,000 units active; on 3BFF (Results 5)
  it fails its pre-registered test. Neither section mentioned the other's result.
- Options: (a) tell both results once, in Results 5, with Results 3 keeping the CDDM values as a
  labelled reference; (b) move the CDDM result out of Figure 3 altogether; (c) train a CDDM
  rate-penalty size series.
- Choice: (a). Quoted: "Results 5 opens its rate-penalty paragraph with the CDDM result
  (λ_frm = 0.2, 75,000 iterations: all 1,000 units active, held-out R² 0.822-0.830 against
  0.831-0.834) and then gives the 3BFF result (λ_frm = 0.1: no floor crossing at N ≤ 1,000, loss
  bursts, fails its pre-registered test). It says that the two results differ in task and in
  strength and attributes the difference to neither." Reasons for rejecting the others, quoted:
  moving the CDDM result "loses a measured result from the only CDDM test"; a new size series costs
  training "and no claim depends on it".
- Board: C5 and C6 rows. The board's old open question "which biological penalty goes in the main
  text" (options: keep the rate penalty with the 3BFF failure stated; lower λ_frm; lead with
  synaptic noise) is closed by this decision: the rate penalty stays in the main text with both
  results stated. That reading is board-r2's; the editor may overrule it.

**3. One name per thing** ([#36](https://github.com/ptolmachev/dead-relu-paper/issues/36#issuecomment-6553683707)).
- Problem: work-package labels (B2, B3, B4, A3) in the paper, "read-out" with two meanings, and
  undefined project words ("floor read", "bars", "stability clause", "ladder", "standard network").
- Options: the committee's acceptance test allowed either a definition at first use or a plain
  replacement for each term. The editor chose replacements, fixed in the glossary of blueprint
  section 2.
- Choice, quoted: "'read-out' means the output weights only ... The measurement point is 'the floor
  crossing' ... or '75,000 iterations' ... B2, B3, B4, B5 and A3 become the rate-penalty size test,
  the combination test, the transfer test, the optimizer comparison, and removing the added units."
  Reason: a reader outside the team cannot decode the labels, and one word with two meanings two
  lines apart (R5) is ambiguous.
- Board: the Terms paragraph now names the paper's terms and says the board keeps its internal
  labels. The glossary binds paper files, not the board.

**4. No change to the floor-crossing code this round; mark the affected values** ([#18](https://github.com/ptolmachev/dead-relu-paper/issues/18#issuecomment-6553684172)).
- Problem: `common.noisy_floor` drops non-finite losses before counting bursts, and the R² at a
  floor crossing can come from a snapshot inside a loss burst.
- Options: (1) patch `common.noisy_floor` and re-measure every bursting network; (2) keep the code
  and mark the affected values.
- Choice: option 2. Quoted: "The affected R² values found so far are in the combination test and the
  rate-penalty networks, whose outcomes are already 'inconclusive' (#17) and 'fails' (the
  rate-penalty size test), so no claim rests on them, and option 1 would re-measure about 270
  networks for no change in a conclusion." If fig5-r2 (#55) finds an affected value elsewhere, it
  reopens #18 before changing any claim.
- Board: C6 row ("What remains"); the old open-decision bullet is now under "Decisions taken".

**5. Threshold sensitivity recomputed on the networks Fig. 2a fits** ([#21, comment 3 item 2](https://github.com/ptolmachev/dead-relu-paper/issues/21#issuecomment-6553684678)).
- Problem: Methods M04 gives the DMTS size exponent at the factor 0.05 as 0.58 (from N = 500-2000,
  entry 2026-10-31_1657) while Fig. 2a gives 0.55 (N = 500-4000).
- Options: recompute at the four sizes Fig. 2a uses, or say that the threshold row covers DMTS
  N = 500-2000 only.
- Choice: recompute. Quoted: "Chosen over 'say that the threshold row covers N = 500-2,000 for
  DMTS' because two exponents for one quantity is what the committee scored as an error of fact,
  and the recompute is a CPU job on stored participation traces." Owner: methods-r2 (#51).
- Board: C2 row ("What remains").

**6. Extended Data held to 10 items** (blueprint section 7; [editor's round-2 plan on #1](https://github.com/ptolmachev/dead-relu-paper/issues/1#issuecomment-6553707796)).
- Options, as the plan states them: the Nature-family format, which allows at most 10 Extended Data
  items, or the Nature Communications format that `main.tex`'s header comment names, which has no
  Extended Data.
- Choice: the Nature-family limit. Quoted: "Extended Data follows the Nature-family limit of 10
  items, as the reference does (9 items) ... the format the paper follows is the reference's." The
  task schematics, the list of every network, the per-unit-rule table at every size and the criteria
  table move to Supplementary Information; the optimizer comparison merges into the Extended Data
  figure of every standard-setting sweep.
- Board: Extended Data and Supplementary items are cited by content, never by number, because the
  printed numbers follow first citation and will move again when the Figure 4 and
  population-statistics floats are filled.

### Round 1 (manager on Pavel's delegation, 31 Oct), applied to the board now

**7. Rescaling and read-out dropout above N = 1000 reported at 75k** ([#16](https://github.com/ptolmachev/dead-relu-paper/issues/16#issuecomment-6552431217)).
Options: (1) report at 75k, labelled; (2) also report the 1.20× floor crossing; (3) continue the 16
runs past 200k. Choice, quoted: "report rescaling and read-out dropout at N = 2000 and 4000 at the
fixed 75,000-iteration read-out, labelled as such, with their 1.20x-floor crossing in Extended Data.
No continuation past 200k." Reason (from the issue's recommendation): for read-out dropout at
N = 4000 the counts at 75k and at 1.20× agree within 1%, so a longer run is unlikely to change the
fold change much. Board: the open-decision bullet moved to "Decisions taken".

**8. The combination test (B3) is inconclusive; no "several"** ([#17](https://github.com/ptolmachev/dead-relu-paper/issues/17#issuecomment-6552431376)).
Options: (1) take the literal verdict and say "several sources"; (2) treat B3 as between the
thresholds and keep "a drive of its own"; (3) continue rate alone and rescale alone to 300k (about
50 GPU-h) and re-read. Choice, quoted: "treat B3 as between the bars. The paper says a per-unit
participation drive is needed and does not claim that several sources are needed ... No stage 2."
Reason (from the issue): the only supporting comparison sets a combination with a 0.04 R² cost
against a single rule whose loss is still falling at 200k. Board: the summary paragraph now ends
"participation needs a drive of its own"; row C7 no longer says "likely several" or "waits on
Pavel"; the B3 line under "Open work" says no stage 2.

**9. The fourth 3BFF network at N = 1000 is counted** ([#20](https://github.com/ptolmachev/dead-relu-paper/issues/20#issuecomment-6552431018)).
Options: (1) keep the exclusion, 278 ± 33 (n = 3); (2) draw all four, 278 ± 27 (n = 4). Choice,
quoted: "option 2. Draw all four 3BFF networks at N = 1000 (278 ± 27, n = 4); the exclusion's stated
reason no longer holds under read-out 79932ef." Board: row and paragraph C1 now give 277.5 ± 26.6
(n = 4: 296, 277, 297, 240). I recomputed the mean and sample s.d. from the four active counts in
the paper's Supplementary table of every network (`floats/EDT2_runs.tex` on dead-relu-paper
`main` 2ceec93, lines 45-48): 277.5 and 26.64. C1 and C2 now agree. The 15 Oct decision "the
straggler stays excluded" is kept under "Decisions taken", marked as superseded.

**10. Results 3 is about the fall of the active count** ([#24](https://github.com/ptolmachev/dead-relu-paper/issues/24#issuecomment-6552952259)).
Options: (1) retitle to the fall; (2) keep "keeps units active" with the 3BFF input-scale gain as the
exception; (3) extend the input-scale sweep to 200k (about 60 GPU-h). Choice: option 1, because it
states what the data show at both measurement points and needs no new compute. Board: already
followed in C5; now listed under "Decisions taken".

### Board-r2's own edits, with reasons (the editor may overrule any)

- **The guide's health check** (#13 item 2, `docs/AGENT_GUIDE.md`). The check q50/q95 < 0.05 now
  applies only to networks with fewer than half their units active. More than half the units are
  active exactly when q50/q95 ≥ 0.05, so the check fails by construction on CDDM at N = 500
  (291-302 of 500 active, q50/q95 0.08-0.10; entry 2026-10-31_1657).
- **"Waits on Pavel" removed.** "Open decisions (Pavel)" is now "Open questions"; the Parked list
  says "do not restart without a `decision` issue" instead of "without Pavel"; the snapshot redo
  (~800 GPU-h) needs the editor's written go-ahead, as PROTOCOL.md requires for more than ~200 GPU-h.
  Reason: PROTOCOL.md (31 Oct) makes decisions the team's.
- **Still open, not decided here:** the C4 wording (owner fig4-no-return, #8) and whether the old
  correlated initialisation changes silencing (~45 GPU-h, not scheduled; the Discussion states the
  shared initial draw as a limitation, blueprint section 8 item 6).
- **Figures paragraph** now lists F1-F5 as the paper has them, from the captions on dead-relu-paper
  `main` 2ceec93 (F4 is still a placeholder there).
- Two stale pointers fixed on the way: row C2 said "Figure 1 synced to the paper 15 Oct" (C2 is
  Figure 2 since round 1), and the redraw rule in the catalogue pointed to Figure 2 (it is Fig. 3e).

**Acceptance checks run** (dead-relu-paper#50):
`grep -n "several\|stays excluded\|waits on Pavel\|ED Figs 8-9" docs/paper_audit.md` returns four
lines, all quoting a decision as history (B3's literal verdict in paragraph C7 and under "Decisions
taken"; the superseded 15 Oct exclusion). `grep -nE "ED ?[0-9]|ED Fig|Extended Data (Fig|Table)\.? ?[0-9]"`
returns nothing.
