## 2026-11-09 15:16 — Round 19: the Maze read-out dropout and rescaling reruns submitted (job ids for 2026-11-09_1513_round19-dropout-rescale-fz-rules.md)

Submitted on Della from `~/trainRNNbrain_r19fz` at fd564f2 (clean), with the commands of the rules entry
(`FREEZE=true`, `SAVE_TO=~/trainRNNbrain/data/trained_RNNs_rl_maze_fz`):

| Job | Arms | Indices |
|---|---|---|
| 17207738 (R19fzDrop) | dropout0.2_beta4, dropout0.05_beta4 | 19-21, 121-123 |
| 17207739 (R19fzResc) | rescale, rescale_mild, rescale_few, rescale_tgt4 | 22-30, 124-126 |

**Time limit lowered to 9 h after submission.** Della has a maintenance reservation from 10 Nov 06:00 to 18:00
Eastern, which began 11 h 45 min after submission, so jobs with the 12 h limit of the rules entry could not start
before it. Both arrays were changed to `TimeLimit=09:00:00` with `scontrol update` while pending. The longest
comparable runs took 6.1 h (the round 18 dropout runs) and 7.3 h (any Maze interventions run on Della in
round 19). Tasks that do not start before about 21:00 Eastern wait until after the maintenance. Nothing else
changed. Logs: `~/trainRNNbrain_logs/R19fzDrop.17207738_<index>.out` and `R19fzResc.17207739_<index>.out`.
