# Diagnostic (after the data): the Maze "not decreasing" verdict comes from a threshold 12.5 times stricter per optimizer step than the supervised one

Follows `2026-11-17_*_maze-long-to-20000-result.md`. Pavel (17 Nov): the Maze result contradicts Dohare et al. (2024)
and the other two tasks; investigate. Everything below was computed after the verdict was known and is not a
pre-registered reading.

## 1. The two thresholds are in different units

- 3BFF/CDDM flat-loss test (M12): decreasing if at least 5 active units are lost per 100,000 iterations; one
  iteration is one Adam step.
- Maze flat-return test (G8, round 18; reused in the 12,000 and 20,000 rules): at least 5 units per 1,000 PPO updates.
  One PPO update is n_epochs x n_minibatches = 4 x 2 = 8 Adam steps, so 1,000 updates = 8,000 Adam steps and the
  Maze bar is 62.5 units per 100,000 Adam steps: 12.5 times the supervised bar.

## 2. Maze rates in the supervised unit (updates 15,000-20,000, the B window)

| Network | B flat | raw slope (units lost per 100,000 Adam steps) | with the two restart jumps removed |
|---|---|---|---|
| N=500 s1 (resumed) | yes | 57 | 60 |
| N=500 s2 (resumed) | yes | 44 | 30 |
| N=500 s3 (resumed) | yes | 49 | 60 |
| N=1000 s2 (resumed) | yes | 23 | 50 |
| N=1000 s3 | yes | 42 | 68 |
| N=1000 s4 | yes | -51 (gain) | 18 |
| N=1000 s6 (resumed) | yes | -69 (gain) | -20 (gain) |
| N=1000 s1 | no | 80 | 103 |
| N=1000 s5 | no | 19 | 65 |

CDDM networks with a flat loss lose 34 to 64 per 100,000 iterations (M12). At the supervised bar, 5 of the 7
flat-return Maze networks decrease (raw slopes), at CDDM-like rates; with the restart jumps removed, 6 of 7.
"Restart jump": the change of the active count across a stage boundary (median of 5 probes before and after), where
Adam restarts with fresh moments and a 50-update warm-up; it is mostly upward (section in the result entry). The raw
slope includes these reactivations; removing them overstates the loss a little (reactivated units decay again).

## 3. Other candidate causes checked

- Fallback restores in the resumed networks: each rolls back at most 200 updates and changes the active count by
  -26 to +10 units with no consistent sign; in N=1000 s6 the three restores inside the window sum to -5. They do not
  explain that network's gain, which stays after the restart correction (1 of 7).
- Learning rate: base 0.001 in both trainers; the Maze trainer scales it down with N (4.6e-4 at N = 1,000), which
  would make Maze lose fewer units per step, not more.

## 4. Dohare et al. (2024)

Their RL experiment is PPO on Ant with the friction changed every 2 million time steps: a non-stationary sequence of
tasks, where plasticity loss accumulates across changes. Our Maze is one stationary task, so their result does not
predict the size of the decrease here. (Sokar et al. 2023 report growing dormant fractions in DQN on single Atari
games; that setting has bootstrapped value targets, which our critic-free PPO does not; not checked further.)

## 5. Consequence

The pre-registered verdict stands as written ("not decreasing" at 5 units per 1,000 updates). The data are not in
conflict with 3BFF and CDDM: per optimizer step, most flat-return Maze networks lose units at CDDM-like rates. Any use
of the supervised-equivalent threshold in the paper must be marked as a choice made after the data. A clean test of
the restart effect would continue networks with the Adam state carried across stages (INIT_ADAM=true; the AdamState
files are on Della); not run.
