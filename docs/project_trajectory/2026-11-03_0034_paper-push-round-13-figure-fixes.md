## Paper push, round 13: Fig. 1a,b seeded, Extended Data Fig. 4 open markers (PR #179, #165, #176)

Follows `2026-11-02_2328_paper-push-rounds-11-12.md`. Package r13-figfix. An independent reviewer
approved it, using their own simulation and their own SVG parser.

- **Fig. 1a,b (#165).** The example batch is now drawn with task seed 11 (`cfg.task.seed =
  TRACE_SEED`). Network 3374117298 has 940 active units before training and 184 after 200,000
  iterations, against 945 and 183 from the unseeded batch. Two runs gave byte-identical records.
  The reviewer's independent simulation gives the same per-unit masks on all 1,000 units. The
  Results 1 numbers do not change (937 ± 14 and 188 ± 16, n = 10), because they come from the
  training traces. Entry 2341.
- **Extended Data Fig. 4 (#176).**
  - **Rejected first fix:** it moved markers along R². Reason: no marker moves along a measured
    continuous axis.
  - **Trace panels:** open markers that share the 75,000-iteration read point sit side by side,
    and the caption says so.
  - **R² panels and panel l:** every marker is at its value, within 0.000 mm on the reviewer's
    parser, and the rings are redrawn on top. Two near-coincident pairs, one in panel l and one in
    panel c, are drawn one inside the other, and the caption says so.
  - `tests/scan_open_markers.py`: an overlapping pair is excused only when the colours differ, and
    a nested pair only when it is fully inside with paper between the rings. With nesting switched
    off the scan fails, as it should.
  - Entries 2356, 2026-11-03_0002 and 0024.

**State.** Paper main e6f85dd, 62 pages, clean build. trainRNNbrain main 86376cb. No open issues on
dead-relu-paper.
