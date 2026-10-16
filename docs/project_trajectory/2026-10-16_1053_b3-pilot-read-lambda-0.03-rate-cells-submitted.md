## 2026-10-16 10:53 — B3 pilot read at the floor: λ = 0.03; the three rate cells submitted

The B3 pilot (Della 15598507-08) finished. Each run was read with `common.read_run` (main at
353dc13, the read-out approved on 15 Oct: a Hampel-smoothed curve and a sustained crossing at 1.05×
the fitted floor). The noisy loss of a penalised run includes the penalty. Della CPU job B3read,
script `~/f2_logs/b3_pilot_read.py`.

| cell (N = 1000) | runs read at the floor | active at the floor | mean | share | multiplier |
|---|---|---|---|---|---|
| control (`c1_snapshots_c200000`) | 3 of 3 | 345, 289, 265 | 299.7 | 0.30 | 1 |
| rate penalty λ = 0.01 | 2 of 3 (one never stays under the bar) | 766, 932 | 849 | 0.85 | 2.83 |
| rate penalty λ = 0.03 | 2 of 3 (one never stays under the bar) | 629, 810 | 720 | 0.72 | 2.40 |

**Rule** (pre-registered in entry 2026-10-15_1342):
- Both rungs have 2 of 3 read-outs, so both are eligible (rule 6).
- Only λ = 0.03 has an active share at or below 0.80 (rule 3), so it is chosen.
- Rules 4 and 5 do not apply.

λ = 0.01 leaves more units active than λ = 0.03, against the expected order. Each rung has only
two read-outs, and their values overlap (766-932 against 629-810), so the order is not established.
Both rungs score below the control by folder R² (0.933-0.941, against 0.952-0.953).

**Submitted** with the commands in the 1342 entry, from `~/trainRNNbrain_redo` (709c17d, clean),
LAMBDA_FRM = 0.03, N = 1000, 3 seeds each:
- 15689235: rate + synaptic noise (σ_w = 1.0);
- 15689236: rate + rescaling;
- 15689237: rate + synaptic noise + rescaling.

Each is about 17 GPU-h per run, about 150 GPU-h in all, under the B3 approval of 14 Oct. The
factorial's "rate" cell is the λ = 0.03 pilot rung, and its "none" cell is the control above.

**Other jobs at 10:51 on 16 Oct:**
- B4 N = 1000: old and young transfers done (6 of 6); fresh cell 1 of 3 running.
- B2 N = 500 and 1000 cells: 2 of 3 runs done each.
- B4 N = 4000 (15655301-03) and the new straggler seed FFseed4 (15657728): pending.
- Another session submitted FFstr300 (the straggler continued to 300k) and two DMTS jobs, DMd5n2k
  and DMd5n4k.
