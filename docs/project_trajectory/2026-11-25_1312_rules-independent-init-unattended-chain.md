## 2026-11-25 13:12 — Amendment to the independent-init rules: time limits and an unattended chain

Amends `2026-11-25_1303_rules-independent-init.md`. Written before any of its networks started training (all nine jobs,
18375654-18375656, were still pending). Firstmate (25 Nov): the Della connection may drop for long stretches, so the
chain is submitted with Slurm dependencies, and no job may go over the 80 GPU-hour cap unwatched. No bound, read point or
verdict rule changes.

- **Stage 1 time limits.** 3BFF 9:00, CDDM 8:00, Maze 9:30 hours, instead of 10, 9 and 11. Together the nine tasks
  can use at most 79.5 GPU-hours. They are 1.8 times the usual time of each network, not twice.
- **3BFF continuation to 200,000** (`slurm/independent_init_chain_della.slurm`, MODE=topup, array 1-3, after all nine
  stage 1 tasks have ended, whatever their state). Each task sums the elapsed hours of the nine stage 1 tasks (sacct, job
  name IndInit since 25 Nov 2026) and continues only if used + 27 ≤ 80, the rule of the rules entry. It then runs for at
  most (80 − used) / 3 hours, so the total cannot pass 80. If sacct does not show nine ended tasks, it does not train.
- **Read** (MODE=read, CPU, after the continuation, whatever its state): the reader, the Maze reader against a copy of the
  six suite networks' records (seeds 1-6, stage 1 and first continuation), the verdict, the figure and the two
  independent recount scripts, written to `data/trained_RNNs/independent_init_reads/` on Della.
- **A network killed by its wall clock** is not continued automatically. It is handled when the connection is up, under
  the cap rule of the rules entry.
- **Reader change.** A network continued to 200,000 iterations is read from its continuation, and any other from its own
  run. The 3BFF floor crossing is read only for a network trained to 200,000.
