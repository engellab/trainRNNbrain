## 2026-08-28 — ⚠️ COLLISION BUG: diverged runs were OVERWRITING each other

Found while auditing seed coverage. `frm k=5 N=1000` showed "1 done + 1 queued" against a target of 3
seeds — but `sacct` said tasks 40 and 41 both COMPLETED and 42 is RUNNING. All three were accounted
for; the disk showed one folder.

**Cause.** The per-network folder is named by its validation score, so **every diverged run of a cell
is called `nan_<taskname>_...`** and later ones silently overwrite earlier ones. Tasks 40 and 41 both
ended `r2 validation: nan` and collapsed into a single `nan_` folder.

**Consequences, project-wide** (quantified after the full log audit):

| cell | diverged jobs | `nan_` folders | overwritten |
|---|---|---|---|
| none k=2 N=2000 | 1 | **0** | — never wrote at all (a different failure) |
| none k=8 N=2000 | 1 | 1 | 0 |
| frm k=2 N=2000 (150k, retracted) | 2 | 1 | **1** |
| frm k=4 N=1000 (400k) | 1 | 1 | 0 |
| frm k=5 N=1000 (400k) | 2 | 1 | **1** |

**7 divergences across the project, 4 folders on disk, 2 genuinely overwritten.** Collisions only
occur when ≥2 runs of the SAME cell diverge, so the two `none` cases (different cells) were never at
risk. ⚠️ An earlier note in this entry said 3 were lost; that was wrong — one `none` case produced no
folder at all rather than being overwritten.

**What was actually lost is VISIBILITY, not much data.** The analyses already drop NaN runs, so they
were correctly using 2 (or 1) seeds all along. What could not be seen was WHY a cell was short: a
missing seed and a failed seed looked identical.

Usable seeds per affected cell: none k=2 N=2000 → 2; none k=8 N=2000 → 2; frm k=4 N=1000 → 2;
**frm k=5 N=1000 → 1** (pending task 42).
- The divergence count has been UNDER-REPORTED everywhere. Any cell with ≥2 diverged seeds looked
  like it had one.
- A cell short on seeds could not be distinguished from a cell whose seeds had failed.
- frm k=5 N=1000 in particular has **two of three seeds diverged**, not one seed missing.

**Fix.** The seed is now part of both the folder name and every file name:

```
0.9476656_s1234567_NBitFlipFlop_relu;N=1000;seed=1234567;LR=0.001;...   (folder)
0.9476656_s1234567_ParticipationTrace.pkl                                (files)
nan_s111_...  vs  nan_s222_...                                           (diverged siblings differ)
```

⚠️ The seed is APPENDED, never prefixed: `common.r2_from_dir` parses the score as
`float(basename.split("_")[0])`, so anything before it breaks every caller. Verified the score is
still the first underscore-delimited field, and that all four analysis globs
(`*ParticipationTrace.pkl`, `*TrainLosses.json`, `*_LastParams_*.npz`, `*_config.yaml`) still match —
they are prefix-agnostic by construction.

Deployed to Spock and Della. ⚠️ Existing folders keep the old names; the analysis globs handle both,
but any cell with historic `nan_` collisions has permanently lost the overwritten runs.

### Full seed audit (done + queued vs a target of 3)

| pen | N=500 | N=1000 | N=2000 |
|---|---|---|---|
| rws | k=1–8 ✓ | k=1–8 ✓ | k=1–8 ✓ |
| frm | k=1–8 ✓ | k=1–4 ✓, **k=5 SHORT**, k=6–8 ✓ | k=1–8 ✓ |
| both | k=1–8 ✓ | k=1–8 ✓ (Della) | k=1–8 ✓ |

Everything is submitted and accounted for except **frm k=5 N=1000**, which needs 2 replacement seeds
once the collision fix is in place — worth re-running now that a second divergence would no longer be
invisible.

---

