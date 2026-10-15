## 2026-10-15 13:54 — Repository cleanup: stale files removed, one agent guide

Pavel: "we do not need handoff anymore, and perhaps many other files and scripts are now stale …
The main point is for the new agents to be less confused by the conflicting information."

**How files were chosen.** Three read-only audits:
- **Code:** an AST import graph over every tracked `.py`, rooted at the training entry point, the
  tests, the live read-out and figure scripts, every script a launcher runs, every script whose
  output a paper branch includes, and the talk deck.
- **Launchers and configs:** each launcher's sweep root matched against the figure code and the
  status board; each config's references across launchers, tests, code and other configs.
- **Docs and memory:** every statement checked against the current rules.

A second, independent AST scan, run before deleting, found no kept file importing a deleted script.
That scan included `fig_cosyne2027.py` in the Cosyne session's worktree. Both live sessions were
asked first:
- trainrnnbrain-d1 (Cosyne figure) named eleven files it needs; all were already kept.
- trainrnnbrain-9d (reworking `common.noisy_floor`) owns the uncommitted `common.py` and
  `burst_stats.py`; neither was touched.

**Removed** (eec4a16, plus three unreferenced images in this commit):
- 100 analysis scripts: pre-project legacy, superseded figure builders, retired read-outs and
  measures, one-off pre-standard read-outs.
- 65 launchers; 38 configs; `jupyter/`; `scripts/`.
- Docs: HANDOFF, TODO, paper.md, paper_repo_setup, research_directions,
  hyperparameter_walkthrough, `docs/notes/agent-*.md`.

The untracked `docs/presentation_backup.md` (the 2 Oct deck) is in the macOS Trash as
`presentation_backup_2026-10-02.md`. Its HTML build was deleted.

**Kept on purpose:**
- Everything the paper's `main` still includes (F2-F5, the supplementary figures) and everything
  the talk deck builds from.
- Scripts behind numbers the supplementary text still quotes (old-story sections waiting on W1).
- `docs/experiments/`, because launchers copy those files into sweep folders.
- Launchers for every sweep a figure reads.

**Written:**
- `docs/AGENT_GUIDE.md`: one statement of how things are done now, from HANDOFF, the agent notes
  and the memory files, each fact checked.
- `docs/paper_audit.md`: rewritten as one current board (claims, remedies, parked topics, open work,
  open decisions).
- `trainRNNbrain/experiments_and_analysis/README.md`: rewritten around the live pipeline.
- `CLAUDE.md`: points at the guide; paths named in `git commit` too, because the index is shared.

**Added launchers** (`git add -f`, unchanged):
- `SilentReLU_f2_fill_{della,spock}.slurm` and `SilentReLU_std_penalties_spock.slurm`, which made
  remedy-grid and penalty sweeps but were never tracked.
- `slurm/b0_*.slurm` (11), copied from Della `~/trainRNNbrain_stdfix`: they produced the `_bias0`
  knob ladders behind Figure 2, and no copy was in git.

**Gap left open:** `SilentReLU_paper_grid_della.slurm` hard-codes `task=DMTS_d7`, so no committed
launcher regenerates `DMTS_paper_grid_d5`.

**Checks:** all 20 test scripts pass on the Mac. All 18 kept figure and read-out scripts import.
