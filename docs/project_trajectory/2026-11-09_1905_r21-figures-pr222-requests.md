## Round-21 figures, PR #222 editor's requests (adds to 2026-11-09_1858_r21-figures-package.md)

- **Extended Data Fig. 3b-d,f:** once every task was drawn as a circle, an open Maze circle at the same values as a
  filled 3BFF circle hid that 3BFF network in panels c and f. Maze markers are now drawn last. A Maze marker whose
  centre is within one marker width of another network's marker is drawn 1.15 widths to the right, or else to the
  left, of its value.
  - Four Maze markers move: the mixed-selective cross at (23, 23) and the selective circle at (199, 200) in c, the
    cross at (0.055, 0.023) in d, and one class at (0.131, 0.151) in f.
  - The open-marker scan's mapping check now finds all 88 filled-marker centres dark (86 of 88 before).
- **Extended Data Fig. 6h:** each of the 45 networks is a thin line at alpha 0.35. Each value's networks, and the
  control networks, get a thick mean line up to the last update all of them reached. Learned points are small
  filled markers.
- **Methods M08:** the input-weight cap was run at three values on 3BFF only (0.1, 0.5 and 2.0; the Fig. 3 printout
  marks them "not run" on CDDM and Maze), not two.
- **Scans:** open-marker and marker-edge scans of both figures pass, and neither figure reports overlapping text.
  Code commit 4dbac3d.
