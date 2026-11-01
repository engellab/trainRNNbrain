## 2026-10-31 22:11 — Activity-threshold sensitivity on the networks Fig. 2a fits: the check passes; DMTS exponents 0.54 / 0.55 / 0.60

Work package `methods-r2` (ptolmachev/dead-relu-paper#51). Rule: entry `2026-10-31_2209`, written
before the run. Supersedes the exponents of entry `2026-10-31_1657` (whose DMTS row covered
N = 500-2000 only); the 3BFF and CDDM rows of that entry are unchanged.

**Job.** Della CPU job 16568716 (`sbatch --wrap`, 1 CPU, 8 GB), COMPLETED in 3 s:
`python threshold_sensitivity_f1ccfd0.py ~/trainRNNbrain_fig2size/data/fig_paper_F2_size_collect.pkl ~/trainRNNbrain_fig2size/data/trained_RNNs`,
the script at trainRNNbrain `wp/methods-r2` f1ccfd0, copied to `~/methods_r2_thresh/`. The collect
record on Della and on the Mac have the same MD5 (a7e9cdd15658eb2c241fa2f7fb0e472e).

**Check (pre-registered): PASS.** At f = 0.05 all 37 counts equal the record's `active_at[0.05]`,
and the exponents equal Fig. 2a's to three decimals (3BFF 0.478, CDDM 0.443, DMTS 0.551).

**Result** (37 networks: 3BFF 16, CDDM 12, DMTS 9; N = 500-4000)

| | 3BFF | CDDM | DMTS |
|---|---|---|---|
| count at 0.025 vs 0.05 | +4% to +25% | +9% to +15% | +3% to +15% |
| count at 0.10 vs 0.05 | −9% to −19% | −12% to −28% | −9% to −19% |
| exponent at 0.025 / 0.05 / 0.10 | 0.434 / 0.478 / 0.494 | 0.445 / 0.443 / 0.476 | 0.542 / 0.551 / 0.595 |
| largest q50/q95 at t\* | 0.038 | 0.099 (N = 500; at most 0.022 from N = 1000) | 0.015 |

Over all three tasks: halving the factor raises the count by 3-25%, doubling it lowers the count by
9-28%; the exponent at 0.025 or 0.10 differs from its value at 0.05 by at most 0.05 (3BFF 0.044 at 0.025). Adding DMTS N = 4000 (one network,
620 active) moved the DMTS exponents from 0.57 / 0.58 / 0.62 (entry 1657) to 0.54 / 0.55 / 0.60.

**Paper.** Methods M04 (dead-relu-paper PR for #51) gives these values.

**Output (verbatim)**

```
task  N     seed        t*      probe   q50/q95  f=0.025  f=0.05  f=0.10  record
3BFF  500    2489235792    60551    60600    0.008      215     179     162  179
3BFF  500     893469585    54466    54500    0.038      274     235     194  235
3BFF  500     822903475    54565    54600    0.025      250     206     178  206
3BFF  500     822939552    62857    62900    0.013      224     202     169  202
3BFF  500    2484557599    57573    57600    0.027      254     211     173  211
3BFF  500    2545439171    65570    65600    0.015      231     197     159  197
3BFF  1000   3374117298    58968    59000    0.000      328     277     231  277
3BFF  1000   1914910909    59917    59900    0.000      347     296     250  296
3BFF  1000   3866906883    62703    62700    0.000      348     297     251  297
3BFF  1000   4186074845    67871    67900    0.000      300     240     209  240
3BFF  2000   4145304148    65717    65700    0.000      463     385     337  385
3BFF  2000   2422919889    53562    53600    0.000      494     415     369  415
3BFF  2000    620846234    68814    68800    0.000      448     370     316  370
3BFF  4000   3341404959    66219    66200    0.000      552     517     441  517
3BFF  4000   3386387268    59334    59300    0.000      624     598     521  598
3BFF  4000   3346922328    69432    69400    0.001      580     540     469  540
CDDM  500    3291291930    26935    26900    0.078      333     291     227  291
CDDM  500    2251851164    27463    27500    0.099      329     302     250  302
CDDM  500    3565681970    25772    25800    0.084      346     302     218  302
CDDM  1000   3833537590    27259    27300    0.017      483     436     359  436
CDDM  1000   1272763220    26535    26500    0.014      466     420     352  420
CDDM  1000       457853    23929    23900    0.022      492     437     359  437
CDDM  2000   3640167262    26933    26900    0.000      620     543     433  543
CDDM  2000    575781518    31147    31100    0.000      587     522     441  522
CDDM  2000   4259132547    26511    26500    0.000      658     580     443  580
CDDM  4000   3046823201    28397    28400    0.000      881     776     635  776
CDDM  4000   3572448132    34015    34000    0.000      860     769     650  769
CDDM  4000    292757883    34943    34900    0.000      846     756     663  756
DMTS  500    2106510642    18205    18200    0.010      235     211     178  211
DMTS  500    1084904964    62088    62100    0.001      197     180     151  180
DMTS  500    2099578504   109348   109300    0.001      188     182     148  182
DMTS  1000     86371460    18599    18600    0.015      485     422     340  422
DMTS  1000   1167683015    75721    75700    0.000      232     218     192  218
DMTS  2000   2585017553    51598    51600    0.000      407     386     344  386
DMTS  2000    632794669    44102    44100    0.000      483     426     369  426
DMTS  2000   4124951173    73448    73400    0.000      499     465     406  465
DMTS  4000   2627187151    78569    78600    0.000      664     620     562  620
3BFF: 16 networks; f=0.025 vs 0.05 +4% to +25%; f=0.10 vs 0.05 -19% to -9%; exponent 0.434 / 0.478 / 0.494 (Fig. 2a 0.478); max q50/q95 0.038; q50/q95 >= 0.05 at []
CDDM: 12 networks; f=0.025 vs 0.05 +9% to +15%; f=0.10 vs 0.05 -28% to -12%; exponent 0.445 / 0.443 / 0.476 (Fig. 2a 0.443); max q50/q95 0.099; q50/q95 >= 0.05 at [(500, 0.078), (500, 0.099), (500, 0.084)]
DMTS: 9 networks; f=0.025 vs 0.05 +3% to +15%; f=0.10 vs 0.05 -19% to -9%; exponent 0.542 / 0.551 / 0.595 (Fig. 2a 0.551); max q50/q95 0.015; q50/q95 >= 0.05 at []
CHECK at f = 0.05: PASS
```
