### B2/B3/B4 moved from Spock to Della (agent-move, merged; notes `docs/notes/agent-move.md`)

Spock had held all 25 runs pending for ~24 h; cancelled there (6929384-91, 6929407-09; none had
started). Resubmitted on Della at 00:55 on 15 Oct as 33 tasks, 3 seeds each, 200k, pinned at 709c17d
(= fcb3749, whose trainer/RNN/task code equals 7739a60's, plus `slurm/SilentReLU_b4_transfer_della.slurm`),
with the new default snapshots (every 1,000, float16) under `c1_snapshots/`: B3 pilot rate 0.01 /
0.03 (15598507-08), B2 N = 2000 / 1000 / 500 (15598509-11), B4 old / young / fresh (15598512-14),
B3 synaptic noise / rescaling / both (15598515-17); gpu-medium (36 h) or gpu-long (76-84 h). ~540
GPU-h. All ten launcher commands smoke-tested on the Mac (configs differ from the control only in
their remedy keys; B4 only in the task swap). B3's rate cells follow the pilot (~14 h of training).

