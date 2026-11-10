# Round-22 polish: random removal drawn as a hexagon (Fig. 1d, Extended Data Fig. 1; dead-relu-paper #223)

Follows 2026-11-10_0151_r22-polish-figures.md, which changed the young networks of Fig. 4d,e from triangles to
pentagons. With only that change the triangle still meant two things: prune-and-duplicate (Fig. 5, Extended
Data Figs. 6, 9, 10) and random removal (Fig. 1d, Extended Data Fig. 1). Reviewer S's item (committee #217,
figures item 2) asks for one meaning per shape, so random removal is now a hexagon as well.

- `fig_paper_F1.py`: the "as many removed at random, then trained further" marks of panel d and their legend
  key, `"^"` to `"h"`; the size constant `D_TRI_MS` renamed `D_HEX_MS`, value 3.8 pt kept.
- `fig_supp_ablation.py` (Extended Data Fig. 1): the random-removal marks and legend key, `"^"` to `"h"`, size
  3.4 pt as before.

Both figures drawn on the Mac from the same inputs (Fig. 1 from `data/fig_paper_F1_collect_v2.pkl`); before
the change both reproduced main's PDFs pixel-identically at 200 dpi, and their printouts are unchanged by it.
Checks: `tests/scan_open_markers.py` fig_paper_F1 PASS; fig_supp_ablation prints the same 22 failures as on
main (intact-network bars drawn across the circles' centres, and three overlapping 3BFF markers at the -0.5
floor), none added. `tests/scan_marker_edges.py` PASS on both (88 and 76 markers). The independent layout check
gives the same counts as main, with 0 artists past the canvas edge.

After both changes: triangle = prune-and-duplicate only (plus the left and down triangles of rate + sparsity and
read-out dropout in Fig. 5, and the edge arrows); pentagon = young networks; hexagon = random removal.
