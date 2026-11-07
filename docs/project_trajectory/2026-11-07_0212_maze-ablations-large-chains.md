# Maze ablations of the N = 2,000 and 4,000 chains at their end weights

rl_maze_ablate.py on the Mac CPU, at the end of training (update 4,000), for the three N = 2,000
chains (2,000 + 2,000 updates) and the one learned N = 4,000 chain (seed 1, 1,000 + 3,000). The
score is the success rate under the training noise over 256 fixed drops and 4 draws, with units
removed by zeroing their outgoing weights, as for the smaller networks. The random column is the
median over 5 random orders at the inactive count.

| network | active | intact | inactive removed | as many random (median) | active removed |
|---|---|---|---|---|---|
| N = 2,000 s1 | 398 of 2,000 | 0.854 | 0.893 | 0.046 | 0.007 |
| N = 2,000 s2 | 508 of 2,000 | 0.869 | 0.896 | 0.190 | 0.011 |
| N = 2,000 s3 | 388 of 2,000 | 0.891 | 0.918 | 0.021 | 0.037 |
| N = 4,000 s1 | 682 of 4,000 | 0.756 | 0.792 | 0.014 | 0.141 |

In all four networks, removing every inactive unit leaves the success rate where it was or slightly
higher (+0.027 to +0.039), and removing as many units at random lowers it to 0.01 to 0.19. The
removal claim of Figure 1d holds at N = 2,000 and 4,000. The small rise in all four is within the
spread of the smaller sizes (-0.01 to +0.03) at N = 250 to 1,000, but it has the same sign in every
network here; it is reported, not interpreted. Two N = 4,000 networks did not learn and are not ablated.
