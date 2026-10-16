# Working in this repo

Read next: `docs/AGENT_GUIDE.md` (how things are done: paths, clusters, read-out rules, figures),
then the status board `docs/paper_audit.md`.

## COMMIT ONLY YOUR OWN CHANGES

**Several agents work in this tree at the same time.** At any moment the working tree holds
uncommitted edits that are not yours, in files you have never opened. A commit that sweeps them up
puts someone else's half-finished work under your commit message, where neither of you will find it
again.

So, every time:

1. **Never stage everything.** No `git commit -a`, no `git add -A`, no `git add .`, no `git add -u`.
   Those four commands are how foreign work ends up in your commit.
2. **Name every path, in `git add` and again in `git commit`.** `git add path/one.py path/two.md`,
   then `git commit -m "..." -- path/one.py path/two.md`. The index is shared: a bare `git commit`
   also takes whatever another agent has staged.
3. **Read the diff of each path before you stage it.** `git diff <path>`. Every hunk must be one you
   recognise as your own. This takes seconds and it is the whole rule: staging a file you did not
   fully read is the same mistake as staging the tree.
4. **A file with hunks that are not yours does not get committed.** Interactive staging
   (`git add -p`, `git add -i`) does not work in this environment, so there is no partial-stage
   escape hatch. Leave the whole file alone, finish the rest of your commit, and say in your reply
   which file you left and why. The agent who owns those hunks will commit them.
5. **Check what you left behind.** Run `git status --short` after committing and report any of your
   own paths that are still modified.

Keep a running note of the paths you touch as you go. At commit time, "which files are mine" has to
be something you know, not something you reconstruct from `git status`.

## CODE RULES

- **All imports at the top of the file. Never inside a function, loop or conditional**, and never
  the same module twice under two names. `tests/test_no_function_imports.py` enforces it.
- **Every function gets a docstring** saying what it does, its inputs (types, shapes, units) and what
  it returns.
- **Config is the single source of truth.** Only a launcher overrides config values, and only the
  ones it sweeps; the run script never overrides config in code.

## THE LAB NOTEBOOK IS ONE FILE PER ENTRY

`docs/project_trajectory/` holds the record, one markdown file per entry, named
`YYYY-MM-DD_HHMM_<slug>.md`. Take the time from `date +%Y-%m-%d_%H%M`, never from memory. Write a
new file for every entry; never append to another entry's file, so two agents cannot collide in
one. A correction is a new entry that names the file it corrects. Details in its `README.md`.
Read the whole notebook with `cat docs/project_trajectory/README.md docs/project_trajectory/20*.md`.
`docs/project_trajectory.md` no longer exists; do not recreate it. A result that changes a claim
also updates the status board, `docs/paper_audit.md`.

### If you find foreign work already inside a commit you made

Say so plainly, in the reply and in a new entry in `docs/project_trajectory/`. Do not rewrite history to fix it —
another agent may already have branched from it. Name the commit and the paths that were not yours.

### Commit messages

End with the attribution line the session's instructions give you. Describe only what *you* changed.
If a file in your commit is one you only partly understand, that is a sign step 3 was skipped.
