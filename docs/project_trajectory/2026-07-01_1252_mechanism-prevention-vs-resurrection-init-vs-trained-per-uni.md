## 2026-07-01 — mechanism: prevention vs resurrection (init vs trained, per-unit)

**Question (Pavel):** how many units are silent *at initialisation* (before any training), and — for any that
start silent — are they later rescued or do they stay silent? I.e. does `frm` work by **prevention** (keep
still-active units alive) or **resurrection** (revive dead ones)?

**Exact init reconstruction (no retraining needed).** Each net in these sweeps was trained as its own job
(`n_nets=1`), so the loop index is always `i=0` and the per-net RNN generator is seeded deterministically as
`rnn_seed = cfg.seed + (0·14653 + 65537³) mod 7309 = cfg.seed + 3508` (see `run_experiment.py:60`). Weight init
draws only from that generator (`get_connectivity_Dale`), and with `bias_range=[0,0]` there is no bias draw — so
the untrained weights are a **pure function of the saved config seed**, independent of training history. We
reconstruct each net's initial weights via `RNN_torch(...).get_params()` (the same effective/Dale-compliant
export used for trained weights), score them on the identical noise-free CDDM batch, and — since units are never
reordered during training — pair **the same unit index** before and after. Reconstruction is verified
deterministic, Dale-compliant, and (independently) reproduces the earlier "0% silent at init" now-check.
Analysis: [`plot_init_vs_trained.py`](../trainRNNbrain/experiments_and_analysis/plot_init_vs_trained.py).

**Result — nothing starts silent.** Init silent fraction is **0.0% in every condition, both metrics, both
activations** (softplus25 and leakyrelu; `init_vs_trained_silent.csv`). The trained columns reproduce the counts
reported above exactly. Representative (softplus-h / leaky-h):

| sweep · eq · penalty | init dead<0.01 | init silent<5%p95 | trained dead<0.01 | trained silent<5%p95 |
|----|----|----|----|----|
| softplus · h · none | 0.0% | 0.0% | 0.0% | 40.6% |
| softplus · h · rws  | 0.0% | 0.0% | 0.0% | 63.6% |
| softplus · h · fr / both | 0.0% | 0.0% | 0.0% | 0.0% |
| leaky · h · none | 0.0% | 0.0% | 32.5% | 45.2% |
| leaky · h · rws  | 0.0% | 0.0% | 49.5% | 56.2% |

![Init vs trained participation — softplus25 h](../img/internal_figures/init_vs_trained_hist_h_fb2792_g0_softplus25.png)
![Per-unit init→trained — softplus25 h](../img/internal_figures/init_vs_trained_scatter_h_fb2792_g0_softplus25.png)

**Mechanism — training *splits* a homogeneous population; `frm` prevents the downward branch.** At init every
unit sits in one narrow participation band (~0.05–0.08; black dashed in the histogram) — no silent units, no
active mode yet. Training then bifurcates it:

- Under `none`/`rws`, ~40–64% of units are driven **below their init level** into a near-zero silent mode
  (participation ~1e-3), the rest **up** into a broad active tail. The per-unit log-log scatter shows the init
  band fanning both ways; `corr(log init, log trained) ≈ 0.3–0.5` — a unit's (tiny) init activity only weakly
  predicts its fate, i.e. **the silencing is decided during training, not preset at init**.
- Under `fr`/`both`, **every** unit is pushed **up** into a single active mode (0.15–0.6, sharp `cap_fr` cutoff);
  0% end below the silent guide.

**Answer.** There are ~no units silent at initialisation, so `frm` cannot be *resurrecting* dead units — there
are none to revive. What it does is **prevention**: on the natural init it stops training from collapsing the
~half of units that `none`/`rws` would silence. (The stronger test — deliberately force-initialising units silent
and asking whether `frm` can pull *those* back — remains open; see `TODO.md`. It only matters for the adversarial
"can it resurrect?" question; on the real init the mechanism is prevention.)

