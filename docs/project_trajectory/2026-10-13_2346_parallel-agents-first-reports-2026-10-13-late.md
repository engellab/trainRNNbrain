### Parallel agents, first reports (2026-10-13, late)

**A5, task demand (agent-a5, merged; `fig_supp_task_demand.py`, `img/internal_figures/fig_supp_task_demand.png`,
notes `docs/notes/agent-a5.md`).** Answers "the task just doesn't need more units". All networks
PRE-STANDARD (bias trained in [-1, 1]); read at the floor (old rule, before the beta guard), end of
training as a second read-out. A harder task recruits more units, but no network that solves its
task fills: at N = 2000, plain flip-flop k = 1-8 bits 446-505 active (r² 0.95); parity flip-flop 3 /
15 / 63 / 255 targets 680 / 823 / 1174 / 1610 active at r² 0.94 / 0.88 / 0.80 / 0.70. The network
fills only where it fails the task (parity 255 at N = 1000: 968 active, r² 0.66). Yang 20 rules vs
the CDDM pair at N = 1000: 440 vs 378 active (1.16x). Independent numpy forward pass agrees with the
stored counts unit by unit (99.7-99.9% classified the same). Open: a standard-architecture parity
re-run; whether "solved" is defined (agent's bar r² ≥ 0.90).

**B2/B3/B4 plans (agent-bplan, merged; `docs/notes/agent-bplan.md`).** Nothing submitted. The 44-job
cap is the gpu-short QOS; gpu-medium (3 days, 20 GPUs) and gpu-long (6 days, 10 jobs) are nearly
empty. B2 (rate penalty alone, standard flip-flop, 200k): nothing on disk; minimal N = 500-2000
~195 GPU-h, full with N = 4000 ~570. B3 (2x2x2 at N = 1000 after a strength pilot): ~390 GPU-h; with
N = 4000 ~1,200. B4 (plasticity, transfer to the 3-bit hyper flip-flop): needs ~half a day of code
(warm start with a fresh read-out, iteration count and chain reset); minimal ~590-1,180 GPU-h.
Agent's recommendation: B2 full + B3 minimal now (~960 GPU-h); B4 code first.

**W2/W3 (paper repo branch `w2-w3-draft`, eca8741, not merged).** Discussion: "not brain-like"
pinned to three properties our silence lacks (a continuous, roughly lognormal rate distribution;
homeostatic return to a set point; recruitability), while saying plainly that cortex is sparse;
a criteria passage and Table 1 (local; acts continuously on a rate, synaptic strength or
excitability; a measured biological mechanism does the same job). 28 references, each verified and
annotated with the sentence it supports; compiles without undefined citations.

