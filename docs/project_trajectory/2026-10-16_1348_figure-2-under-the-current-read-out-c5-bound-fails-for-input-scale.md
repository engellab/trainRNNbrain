## 2026-10-16 13:48 — Figure 2 under the current read-out; C5's "~120 units" bound fails for input-weight scale

Figure 2 was collected again on Della (job 15695934, `~/trainRNNbrain_c1ana` at be17ac9) under the
current read-out: b121e02 (Hampel smoothing, sustained crossing) plus be17ac9 (multistart floor fit).
It was drawn on the Mac.

**What be17ac9 alone changes.** To isolate the fix, Figure 2 was also collected at 7ff3ccc, the
commit before it, in a separate checkout with its own data folder (job 15696350). Only two rungs
differ:
- Spectral radius 2.0: before, 0 of 2 runs read (both fits degenerate); now 399 and 413 active,
  R² 0.824-0.827.
- Recurrent-weight penalty λ = 0.05: before, 1 of 2 runs read; now 534 and 543.

Every other rung is identical, including every recurrent-noise rung, where all 5 runs were already
read.

**Active units at the floor → at the common budget** (means, CDDM N = 1000 unless stated):

| ladder | rungs |
|---|---|
| weight decay (200k) | none 483→412; default 1e-6 420→266; 1e-5 357→166; 1e-4 382→86 |
| recurrent noise (200k) | σ 0: 479→180; 0.01: 451→185; default 0.05: 404→264; 0.1: 427→308 |
| metabolic penalty (75k) | 0: 420→337; 0.01: 457→337; 0.1: 455→351; 1: 445→375; 10: 385→330 |
| spectral radius (75k) | 0.5: 452→361; default 1.2: 426→340; 2.0: 406→312 |
| W_rec penalty (75k) | 0: 426→340; 0.05: 538→438; 0.2: 527→448 |
| activation (200k) | ReLU 420→266; leaky 487→276; softplus 513→250; sigmoid 219→188 |
| input scale, CDDM (50k) | default 428→362; 0.5: 581 (1 run)→335; 2: 708→257; 5: 664→209; 20: 434→176 |
| input scale, 3BFF (150k) | default 280→206; 0.5: 392→231; 2: 485→298; 5: 609→302; 20: 712→329 |

**C5 needs revising.** The draft (entries 2207 and the 16 Oct second draft) says no hyperparameter
raises CDDM's active count at the floor by more than about 120. Under the current read-out, a larger
input-weight scale does:
- At scales 2 and 5, CDDM networks have 664-721 active units at the floor, against 416-437 for the
  default draw.
- They reach the floor at 4,300-6,900 iterations, against about 25,000 for the default. The
  snapshot oracle (entry 2054) puts their true crossing at 6-10k, so the rule reads them early.
- By 50,000 iterations they have 209-257 active units, fewer than the default's 362.

The other CDDM hyperparameters move the count at the floor by at most about 110 (W_rec penalty
λ = 0.05). Sigmoid now reads 219, not 235-255.

The paper's abstract (e6e13d6, 6ba644e) says "On the decision task no standard training setting
adds more than about 120 active units". That is not true under the current read-out. **Figure 2
and its text are not synced to the paper until Pavel chooses the C5 wording.**
