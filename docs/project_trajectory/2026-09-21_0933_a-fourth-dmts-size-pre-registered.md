## 2026-09-21 09:33 — a fourth DMTS size, pre-registered

Pavel asked whether Figure 1c should get a fourth DMTS point at N = 4000. It should, and the
argument is not "more data is nicer" — it is that three points do not determine an exponent. The
DMTS local slopes are 0.342 from N = 500 to 1000 and 0.757 from 1000 to 2000, so the pooled
b = 0.549 (which would put 1,000 active units at N ≈ 5,400) is whichever of those two the fit
happens to weight. Both four-point tasks behave the opposite way: their local slopes DECLINE with
size (flip-flop 0.580 / 0.493 / 0.294, CDDM 0.514 / 0.387 / 0.428), which is the mild saturation the
pooled exponent is meant to summarise. Either DMTS really does recruit faster than the other two
tasks, or its N = 1000 cell is low — it has the widest seed spread in the panel (± 55 units).

Submitted as Spock job **6309844_37–39**: DMTS_long, N = 4000, `pen=none`, 3 seeds, 100,000
iterations (the panel's matched-compute read-out; `anneal_noise: False`, so nothing in the
trajectory depends on `max_iter` and 100k is genuinely the first 100k of a 150k run). Only the
unpenalised arm — the scaling panel uses no other, and no penalty scaling is claimed anywhere.

**Code pin, which nearly went wrong.** The existing N = 500/1000/2000 cells were trained at
`800ee30`. `51237d6` later moved the DMTS decision cue off channel 2 onto its own input line, which
changes the input statistics of DMTS_long (5 channels, one of them unused in the old version). A
fourth point of the same fit has to be the same task, so the jobs run from a worktree pinned at
`800ee30` (`~/trainRNNbrain_dmts800`), passed in as `REPO=` — *not* by checking out `~/trainRNNbrain`,
which is at `c718b23` and is what ~50 pending array tasks will import when they start. Verified on
Spock that `import trainRNNbrain` under that `PYTHONPATH` resolves to the worktree and that its
`TaskDMTS` still has the old cue line. Smoke-tested the whole command locally first (N = 64,
5 iterations, from a local 800ee30 worktree): config resolves with dropout off and both λ at 0, and
the run lands in `DMTS_std_pen/EqType=h_N=4000_pen=none/`.

**Falsifier, fixed before the data exist** (also in `measured_facts.md` §1.1): a 2000→4000 local
slope of 0.44 ± 0.10 means DMTS agrees with the other two tasks and the single "N ≈ 14,000" figure
stands for all three; above 0.6 means the exponent is genuinely task-dependent and the paper must
quote a per-task extrapolation instead of one number. Either outcome is reported. If the jobs do
not finish, Figure 1c keeps three DMTS points and the caption is already correct — it says the DMTS
fit "is not extrapolated beyond its data".

Cost: N = 2000 ran 150k in 27–30 h (0.65–0.73 s/iter) and the cost is ~linear in N, so ~36–41 h
median at N = 4000 and ~70 h on the slow-node tail; requested 96 h, Spock's maximum, because the
participation trace is written only on completion.

