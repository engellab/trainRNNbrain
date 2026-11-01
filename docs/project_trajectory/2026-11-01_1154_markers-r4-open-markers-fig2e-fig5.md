## 2026-11-01 11:54 — markers-r4: Fig. 2e drawn open; every open marker of Figs 2 and 5 reads as open on the page (dead-relu-paper#104)

Branch `wp/markers-r4` (from trainRNNbrain main fda3868). Drawing only: the printouts of
`fig_paper_F2_size.py` and `fig_paper_F5_drive.py` are identical before and after (diffed line by
line), so no number in the paper changes.

**Fig. 2e.** Each marker is a least-squares slope over iterations 150,000 to the end of training,
not a value read at the floor crossing, so the markers are now open (filled means the floor
crossing and nothing else, `docs/figure_style.md` of the paper repo, #83). White inside, drawn
above the s.e. bar; an ink edge marks the two networks trained to 300,000 iterations. Triangles
(N = 4,000) are drawn 1.25 times larger: a matplotlib triangle has 64% of a circle's area at the
same size, and an open 3 pt triangle printed with its inside nearly closed.

**Fig. 5.** Every open marker has a white inside and is drawn above lines, filled markers and rings.
Networks of one rule whose open markers or rings would overlap are one marker at their mean, with
thin bars over their range and their count beside it (the treatment Figure 3 uses,
`fig_paper_F2_fails.draw_open_groups`). In panel c's lower axes the ringed and unringed networks of
one rule and size share a group, ringed if any member is, labelled "ringed/all" when only some are
(rescaling at N = 1,000: 2/4; N = 4,000: 4/6). The two open-marker rules (rescaling, read-out
dropout) take the outer columns of panel c and write their counts outward. Markers of two rules on
equal values are set side by side about their mean x: panel a (noisy synapses 572 and read-out
dropout 539 at 75,000 iterations), panel e at N = 1,000 (a rate-penalty and a read-out-dropout
network both at R² 0.13, 0.18), panel f's R² column (rate row, 0.931 ringed and 0.932). A count
label goes on the preferred side, else the other side, below or above, whichever touches no marker,
ring or other label. Off-scale groups in f carry their count past the arrow.

**Pixel scan.** `tests/scan_open_markers.py` runs each script with `paperstyle.save` wrapped, takes
every open-marker centre from the saved figure, renders the PDF with PyMuPDF at 300 dpi and reads
the pixels. Rules set before the first scan: (1) the darkest channel of the 3 x 3 block at each
open-marker centre is at least 200/255; (2) no two open markers of one axes closer than half a
marker diameter; (3) no two rings closer than 1.0 mm. A mapping check (every filled-marker centre
reads dark) guards the coordinates: 129 of 129 (Fig. 2) and 97 of 97 (Fig. 5). One change after the
first scan: rule 1 samples a triangle at its incentre, because matplotlib anchors a triangle at half
its height, under 0.5 pt from the stroke of a 3-4 pt open triangle; the darkest value at the anchor
is still printed ("anchor" column). Result, main (fda3868) against this branch:

| axes | open markers, main → branch | centre fails, main → branch | overlaps | ring pairs < 1 mm |
|---|---|---|---|---|
| Fig. 2e | 0 (filled) → 19 | – → 0 (darkest 255) | 0 | – |
| Fig. 5a | 6 → 6 | 1 → 0 | 0 | – |
| Fig. 5b | 19 → 16 | 2 → 0 | 1 → 0 | 0 |
| Fig. 5c top | 30 → 10 | 28 → 0 | 23 → 0 | – |
| Fig. 5c bottom | 30 → 10 | 28 → 0 | 30 → 0 | 13 → 0 |
| Fig. 5e, N = 1,000 | 16 → 14 | 2 → 0 | 3 → 0 | 0 |
| Fig. 5e, N = 4,000 | 20 → 14 | 4 → 0 | 1 → 0 | 1 → 0 |
| Fig. 5f active units | 22 → 13 | 7 → 0 | 5 → 0 | – |
| Fig. 5f R² | 22 → 10 | 14 → 0 | 11 → 0 | 1 → 0 |
| Fig. 5f participation ratio | 22 → 16 | 1 → 0 | 4 → 0 | – |
| Fig. 5c legend | 1 → 1 | 1 → 0 | – | – |

Main: 181 rule failures; branch: 0. The anchor of panel c's small read-out-dropout triangles
(2.65 pt scatter size before the 1.25 factor) still reads dark (15 and 22); their incentre reads 252
and 255. `tests/test_paperstyle.py` passes on both PDFs; no two text spans overlap (PyMuPDF).

Run: `python tests/scan_open_markers.py fig_paper_F2_size fig_paper_F5_drive` (repository root).
