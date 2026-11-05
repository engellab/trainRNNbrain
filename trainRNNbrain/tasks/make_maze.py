"""Make the maze for configs/task/Maze.yaml, reproducibly.

A perfect maze (one route between any two cells, many dead ends) by depth-first search on a
`cells` x `cells` lattice, which gives a (2 cells + 1) x (2 cells + 1) grid with an all-wall border;
then `loops` interior walls between two free cells are removed at random, so a few detours exist
and the maze is not a single winding path; the cheese is the centre lattice cell. The rows are
printed as the YAML list to paste into the config, with the geodesic statistics (maximum and mean
cell distance to the cheese over free cells, number of dead ends).

Kept beside TaskMaze.py because it writes the one thing the task config holds that cannot be typed
by hand; it is a generator script, not a Task class, and nothing imports it.

Run: python trainRNNbrain/tasks/make_maze.py --cells 7 --seed 20 --loops 4   (the config's maze)
"""
import argparse
import numpy as np
from trainRNNbrain.tasks.TaskMaze import bfs_distance, STEPS_4


def dfs_maze(cells, rng):
    """A perfect maze on a cells x cells lattice by depth-first search from the centre cell.

    Args:
        cells: int, lattice size (odd, so the centre is a lattice cell); rng: numpy Generator.
    Returns:
        (wall, goal): wall (2 cells + 1, 2 cells + 1) bool array; goal the (row, col) of the
        centre lattice cell in grid coordinates.
    """
    n = 2 * cells + 1
    wall = np.ones((n, n), dtype=bool)
    start = (cells // 2, cells // 2)
    wall[2 * start[0] + 1, 2 * start[1] + 1] = False
    visited, stack = {start}, [start]
    while stack:
        i, j = stack[-1]
        nxt = [(i + di, j + dj) for di, dj in STEPS_4
               if 0 <= i + di < cells and 0 <= j + dj < cells and (i + di, j + dj) not in visited]
        if not nxt:
            stack.pop()
            continue
        ni, nj = nxt[rng.integers(len(nxt))]
        wall[2 * ni + 1, 2 * nj + 1] = False
        wall[i + ni + 1, j + nj + 1] = False          # the wall cell between the two lattice cells
        visited.add((ni, nj))
        stack.append((ni, nj))
    return wall, (2 * start[0] + 1, 2 * start[1] + 1)


def add_loops(wall, loops, rng):
    """Remove `loops` interior walls that separate two free cells (in place).

    Args:
        wall: (H, W) bool array from dfs_maze; loops: int; rng: numpy Generator.
    """
    H, W = wall.shape
    cand = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1) if wall[r, c]
            and ((not wall[r - 1, c] and not wall[r + 1, c] and wall[r, c - 1] and wall[r, c + 1])
                 or (not wall[r, c - 1] and not wall[r, c + 1] and wall[r - 1, c] and wall[r + 1, c]))]
    for k in rng.choice(len(cand), size=min(loops, len(cand)), replace=False):
        wall[cand[k]] = False


def maze_stats(wall, goal):
    """Geodesic statistics of a maze.

    Args:
        wall: (H, W) bool; goal: (row, col).
    Returns:
        dict: max_dist, mean_dist (cells, over free cells), dead_ends (free cells with one free
        neighbour), free (number of free cells).
    """
    dist = bfs_distance(wall, goal)
    free = ~wall
    nb = sum(np.roll(free, s, axis=(0, 1)) for s in STEPS_4)
    return {"max_dist": float(dist[free].max()), "mean_dist": float(dist[free].mean()),
            "dead_ends": int((free & (nb == 1)).sum()), "free": int(free.sum())}


def rows_of(wall, goal):
    """The maze as the list of strings the config takes ('#', '.', 'C')."""
    grid = np.where(wall, "#", ".")
    grid[goal] = "C"
    return ["".join(r) for r in grid]


def main():
    """Generate one maze, print its rows as YAML and its statistics."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loops", type=int, default=4)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    wall, goal = dfs_maze(args.cells, rng)
    add_loops(wall, args.loops, rng)
    for r in rows_of(wall, goal):
        print(f'    - "{r}"')
    print(maze_stats(wall, goal))


if __name__ == "__main__":
    main()
