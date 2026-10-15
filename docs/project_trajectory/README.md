# Project trajectory

A running, reproducibility-first log of experiments. Each entry records *why* the experiment
was run and *exactly how* it was run (commit, config, parameters, cluster, submission command,
output layout) so it can be regenerated after all working-memory context is lost.

---


## How entries are kept

One file per entry, named `YYYY-MM-DD_HHMM_<slug>.md`, so the folder listing is the chronology.

- **New entry:** a new file. Take the timestamp from the clock (`date +%Y-%m-%d_%H%M`), not from
  memory. Start it with `## YYYY-MM-DD HH:MM — <title>`.
- **Never append to another entry's file.** That is what lets agents work in parallel without
  colliding. A correction or a follow-up is a new entry that names the file it amends.
- **Read it end to end:** `cat docs/project_trajectory/README.md docs/project_trajectory/20*.md`.

**The files before 2026-10-15 were split from the single `project_trajectory.md`** (its last
version is `919fc92`). Concatenated in name order after this README's first eight lines, they
reproduce that file byte for byte. Their times come from the heading when it states one, else from
the commit that added the entry when that commit fell on the entry's date. Where no time is on
record the name carries `xxNN` (the entry's position within its day) instead of an invented time,
and entries that share one commit time carry `a`, `b`, ... after it. From 2026-10-13 15:23 on,
agents had filed their entries as `###` sections under one `##` heading; each of those is its own
file and keeps its `###` heading.
