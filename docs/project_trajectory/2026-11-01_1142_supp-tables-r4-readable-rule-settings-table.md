## 2026-11-01 11:42 — supp-tables-r4: the rule-settings table prints settings in words, not folder names

Work package supp-tables-r4, dead-relu-paper#103 (committee round 3, presentation). The paper side
renumbers the Supplementary tables by first citation; this entry records the code side, which
regenerates the rows of the rule-settings table (now Supplementary Table 2, formerly 3).

### What changed in the two table scripts

- `table_rule_settings.py` (block a): the first column was the sweep folder (`target_ladder`,
  `copy_perturb`, ...). It is now a sweep number, 1 to 18, assigned in the order the sweeps first
  appear in the table. Each row is printed after a source comment naming its catalogue cell folder.
  Within a rule, the shown setting comes first, then the other settings of its sweep, then the other
  sweeps. The combination-test rows say so in their Setting cell. The check list prints the
  number-to-folder map: 1 remedy_grid, 2 combination test, 3 combination-test pilot, 4 synnoise,
  5 synnoise_ext, 6 target_ladder, 7 norm_mode, 8 recgrow, 9 rescale_active, 10 rescale_revive,
  11 rescale_states, 12 revive_ops2, 13 revive_refr, 14 paper_grid, 15 copy_perturb,
  16 replace_variants_fix, 17 dropout_sizes, 18 std_bernoulli.
- `table_rule_screen.py` (block b): the "Sweep and cell" column printed the folder names. A new
  function `setting_of(cfg)` reads each cell's saved config and writes its parameters in the
  symbols of block a (for example `0.025N per check, wait 1,000`; `eta = 0.2, length not kept`;
  `m = 2.5, step 0.02, partners by rate`). Where a key is missing from the config, it takes the
  Trainer's default. Rows are grouped under one italic heading per rule, so the Rule column is gone.
  Each row is printed after a source comment naming its folder. The check list flags any two rows a
  reader could not tell apart (same rule, gamma, setting, training length and N); there are none.

### Validation (pass rule set before the comparison: every row keeps its numbers)

- Both scripts, run unchanged at the start, reproduced the rows in the paper's
  `floats/ST_rule_settings.tex` (origin/main 50ffc9c) exactly: 58 block-a rows and 64 block-b rows.
- After the change, per row: each of the 64 block-b folders keeps its rule label, its gamma mark and
  its five numbers (N, networks, trained, active, fold); 64 of 64 match. The 58 block-a rows have the
  same (setting, numbers) multiset as before, and each old sweep name maps to exactly one new number.
  The comparison scripts parse the old LaTeX by folder name, independently of the table scripts.
- The rows in the paper are the scripts' output verbatim (diff empty for both blocks).

### A fact the folder names hid

Every 3BFF cell of `NBitFlipFlop_std_dropout`, `std_pen` and `std_penlong` has
`model.bias_range: [-1.0, 1.0]`, a trainable bias. The control networks have a fixed zero bias.
Fourteen block-b rows come from these sweeps, and their fold change is against the zero-bias
controls. The table now prints "trainable bias in [-1, 1]" in those rows; no number changed.
Opened as dead-relu-paper#114 (label data) for the owners of Methods M08 and M09.
