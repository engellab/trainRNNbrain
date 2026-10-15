### C4: what "silent" means on each task (agent-c4, merged; notes `docs/notes/agent-c4.md`, figure `img/internal_figures/c4_gradient_mechanism.pdf`)

Claim-1 networks at their floor, N = 1000, 3 seeds per task, seeded batch. Oracle: a hand-written
float64 numpy BPTT, matching production `unit_gradients` per unit (median relative difference < 1e-6,
identical reached counts). Every bar was committed before its run.
- **Flip-flop: the "silent" units are not off.** Every one of the 145/273/195 silent-but-reached
  units crosses threshold: the median one is on for 2-4% of time steps, in 874-959 of 1024 trials,
  93-99% of its on-steps right after an input pulse. The participation rule cannot see them: below
  10% on-time q90(|r|) is 0. No flip-flop unit is never on; on CDDM 271-322 of 1000 are, and they get
  exactly zero gradient.
- Falsified: exact zeros leaking through the clamp (g_in bit-identical with f'(0) = 0 or 1), and
  training noise adding reach (with noise 0.79-0.84x as many reached).
- The bar matters both ways: moving it from 1% to 20% of q95 shrinks the flip-flop's reached count
  2.3-2.4-fold and the participation count 2.1-2.2-fold on both tasks, so "reached = active" on
  CDDM is also bar-dependent.
- Per on-step, flip-flop silent units have 4.1-4.9x the loss sensitivity of active units (CDDM
  0.8-1.1x); cause untested.
- Over the next 5,000 iterations (real snapshots) the flip-flop's silent-but-reached incoming rows
  change 1.2-1.7x as much as active rows, yet only 2-4% become active, and for 68-80% the drive at
  their firing moments FALLS (active units 47-54%): the gradient reaches them and training keeps
  them silent. On CDDM about a quarter become active (turnover at the bar; interpretation).
Proposed wording: CDDM "271-322 of 1000 units never cross threshold and receive exactly zero
gradient"; flip-flop "silence is not loss of gradient: silent units fire briefly after pulses,
their inputs keep learning, and training holds them silent". For claim 1, "silent" on the flip-flop
means "on for under 10% of the time". DMTS and other N not tested.

**B2/B3 queued on Spock (agent-b23, merged; notes `docs/notes/agent-b23.md`).** 24 runs, 3 seeds each,
200k, submitted 00:11 on 14 Oct (jobs 6929384-91), ~465 GPU-h: B3 pilot (rate penalty 0.01, 0.03 at
N = 1000), B3 synaptic noise / rescaling / noise + rescaling at N = 1000, B2 rate penalty 0.1 at
N = 500 / 1000 / 2000. Code 618fc9e = 7739a60 + one launcher (`SilentReLU_remedy_grid_spock.slurm`),
smoke-tested on the Mac for every arm (configs differ from the control only in their remedy keys).
Output `~/trainRNNbrain/data/trained_RNNs/NBitFlipFlop_remedy_grid/` on Spock; snapshots on the lab
share `/mnt/cup/labs/tengel/home/tolmach/trainRNNbrain_b23_snapshots`. B3's rate-containing cells wait
for the pilot (selection rule written in the notes before any result). B2 at N = 4000 cannot fit on
any Spock GPU (out of memory at 41.8 GB); passed to the agent to run on Della's 80 GB cards.
B2 at N = 4000 queued on Della (15527679, 3 seeds, 200k from scratch, λ 0.1, gpu80 on the 6-day
gpu-long queue, ~370 GPU-h; code 2816d14 = 618fc9e + a Della launcher; snapshots every 5,000 under
c1_snapshots/NBitFlipFlop_remedy_grid; smoke-tested locally and on a Della login node, config differs
from the N = 4000 control only in lambda_frm). B3's rate-containing cells wait for the pilot (~17 h
once it starts); commands and the selection rule are in agent-b23's notes §2.

