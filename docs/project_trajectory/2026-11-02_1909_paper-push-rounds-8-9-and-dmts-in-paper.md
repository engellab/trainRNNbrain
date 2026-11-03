## 2026-11-02 19:09 — Paper push, rounds 8-9 and the DMTS result in the paper

**Round 8, abstract on Nature's slot template.** Pavel asked that every abstract sentence have a
concrete place and purpose, pointing to Nature's "How to construct a Nature summary paragraph"
(nature.com/documents/nature-summary-paragraph.pdf). New skill `nature-abstract`
(~/.claude/skills/nature-abstract, linked in ~/Documents/AI skills). The abstract (paper PR #148) is
255 words, every sentence labelled by slot (field, background, problem, here we show, reveals,
context). Reviews caught: "every unit contributes" (false: removing inactive units lowers R² in
3BFF and DMTS networks without retraining), an example rule (the rate penalty) that is unstable at
the stated scope (replaced by prune-and-duplicate), and a context sentence repeating slot 4.

**Round 9, topic sentences (Pavel item 26; plain-english rule 10).** Every paragraph now opens with
the sentence that states its one idea (PRs #149-#152). Paragraphs failing on main before: Intro 5/5,
Discussion 6/9, Results 24/30, Methods and supplement 81/122. Writing topic sentences surfaced
loose claims that reviewers corrected: "most changes that increased active units lowered R²"
(false: CDDM without weight decay, 3BFF input-weight scale); "under any swept setting" (now
"standard", excluding the rate-penalty networks); the silent-unit gradient under noise (only the
incoming-weight gradient was measured); "excluded from some analyses" (now names the two DMTS
networks that did not learn: excluded from every figure, count, fit and mean).

**DMTS setting sweeps in the paper (paper PR #155, decision #153).** Entry 2026-11-02_1822: the
sweeps fail the rule fixed in entry 2026-11-01_2124 (one of nine learned networks with a floor
crossing, σ_rec = 0.1, has 288 active units at the crossing and 308 at the end; three settings
untested). The abstract keeps "two of the three tasks"; Results 3 opens "The DMTS setting sweeps
fail the test that we set before training them"; Discussion fifth limit; Methods design and rule;
Supplementary Table 2. The reviewer recomputed every number from data/dmts_settings/*.pkl.

**Initial draws.** The 27 DMTS networks drew their three weight matrices independently; every other
network in the paper used the shared draw. Checked against every Della run config and from the
weights (correlation of the initial matrices in 33 networks: shared r 0.53-0.95, independent
|r| ≤ 0.04). Methods M01, the Discussion and the Supplementary Methods are scoped accordingly.
Spock-only data never copied to the Mac was not checked (ssh key refused).

**State.** Paper main 3ae8978, clean build. Issues all closed. Left for Pavel: data DOI, code
release, acknowledgements, author contributions.
