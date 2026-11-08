## 2026-11-07 19:50 — Round 19 part 2: Della 17049802 tasks 94-96 (adamw) were a submission error and failed harmlessly; adamw is in Spock 7398207 (corrects 2026-11-07_1114_round19-settings-stage2-submitted.md)

**What happened.** The rules (`2026-11-07_1111_round19-settings-stage2-rules.md`) put adamw s1-s3 in stage 2:
seed 1 was not learned at update 2,000, and seeds 2 and 3 are continued so the arm shares one budget. The
submission placed adamw on Spock: 7398207 was submitted with `--array=7-9,13-15,19-21,26-30,37-39,43-45,49,61-63,67-72,79-90,94-96`,
and its last three indices are adamw. Della was meant to run met10 (indices 91-93) and the four SGD rates
(97-108), 15 networks. I submitted Della 17049802 with `--array=91-108` instead of `91-93,97-108`, so
it also held the three adamw indices. The submission entry's count of 15 networks for 17049802 is right for
the intended array and wrong for the one submitted, which had 18 tasks.

**Why nothing was harmed.** The adamw parents were never copied to Della. Each of the three tasks stopped at
the launcher's parent check, 1-6 s after starting ("FATAL: .../Maze_interventions/adamw/s<k> has no
*Params_Maze.npz to continue from"). That check comes before any folder is written, and Della has no
`adamw` or `adamw_cont` folder. The other 15 tasks of 17049802 are running (2.6-4.4 h in at the time of
this entry).

**Coverage check against the rules.** I expanded the arrays of the three part 2 jobs and compared them with the rules'
list of 63 networks. Every one of the 63 is in a job, and no job holds a network outside the list. The only
overlap between clusters is indices 94-96, which failed on Della. Nothing is resubmitted. Copying the adamw
parents to Della and resubmitting would run the same three continuations a second time, into a second tree.

**Not yet verified.** Spock is unreachable from the Mac (Pavel's ssh session has expired). So Spock 7398207's tasks
94-96 have not been checked since submission. When access returns, check their logs. If any of them failed,
resubmit only that task on Della: copy the parent from the Mac copy in
`data/trained_RNNs_rl_maze/Maze_interventions/adamw/s<k>`, then run from `~/trainRNNbrain_r19s` with 17049802's environment.
