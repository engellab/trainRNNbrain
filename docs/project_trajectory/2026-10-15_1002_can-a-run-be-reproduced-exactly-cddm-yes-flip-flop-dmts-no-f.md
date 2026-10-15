### Can a run be reproduced exactly? CDDM yes, flip-flop/DMTS no - fixed for future runs (15 Oct)

Pavel asked whether rerunning a run would give the same trajectory with snapshots added. Test (pass =
relative loss difference ≤ 1e-6 at every one of 3,000 iterations): one claim-1 control per task rerun
from its saved config, same seed, same pinned code (b524ac7), on Della GPUs. **CDDM: bitwise identical
at all 3,000 iterations. Flip-flop: differs from iteration 1** (up to 7x by 3,000), because every
sampled-task config had `seed: null`, so its training trials came from an unseeded generator. So:
CDDM runs can be regenerated exactly with snapshots added (the trainer at 709c17d/HEAD equals
b524ac7's, and writing a snapshot consumes no random numbers); flip-flop and DMTS runs cannot - a
redo gives new, statistically equivalent networks. Fixed for the future: the flip-flop, hyper
flip-flop and DMTS configs now use `seed: ${seed}`; two local flip-flop runs with the same seed are
bitwise identical, a different seed differs. Runs already queued (B2/B3/B4, RMtop200 at 709c17d) keep
the unseeded tasks.

