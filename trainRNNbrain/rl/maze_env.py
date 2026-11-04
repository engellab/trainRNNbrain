"""A fixed maze run by a batch of point agents in parallel, in torch, on the policy's device.

The maze is an ASCII grid: '#' wall, '.' free, 'C' the cheese (exactly one cell). Every border
cell must be a wall, so an agent can never leave the grid and every sensor ray hits something.
Cells are 1 x 1 units: cell (row, col) covers x in [col, col + 1) and y in [row, row + 1). An agent
is a point (x, y) with a heading angle (radians, 0 along +x, counter-clockwise positive). Each
episode drops every agent in a random reachable free cell with a random heading.

Observation (the RNN's inputs): `n_rays` range sensors spread evenly over 360 degrees, egocentric
(ray 0 points along the heading). Input i is 1 - d_i / ray_max, with d_i the distance to the first
wall along ray i (ray marching with step `ray_step`), so a wall at the nose reads 1 and open space
reads 0. With `goal_cue` two more inputs give cos and sin of the bearing to the cheese relative to
the heading (a "scent"); without it the agent sees walls only and has to localise from memory.

Action (2 raw policy outputs per agent): a[0] turn, clipped to [-1, 1] and scaled by `turn_max`
(radians per step); a[1] speed, clipped to [0, 1] and scaled by `v_max` (cells per step). A move
whose end point lies in a wall cell, or which cuts a corner between two wall cells, is cancelled:
the agent keeps its position (the new heading is kept) and pays `reward["wall"]`.

Reward per step: `reward["step"]` (a cost, negative), `reward["wall"]` if the move was blocked,
`reward["goal"]` once, on coming within `goal_radius` of the cheese centre (the episode ends for that
agent), and the potential-based shaping `reward["shape"] * (d - gamma * d')`, with d and d' the
breadth-first geodesic cell distance to the cheese before and after the move. Potential-based
shaping leaves the optimal policy unchanged (Ng, Harada and Russell, 1999); it only makes the
reward dense.

Every tensor has the batch as its LAST axis for observations (n_inputs, B) and actions (2, B), the
RNN's convention; the internal state uses (B, ...) because torch indexing is easier that way.
"""
import collections
import math
import numpy as np
import torch

STEPS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def parse_maze(rows):
    """Wall mask and cheese cell of an ASCII maze.

    Args:
        rows: list of equal-length strings; '#' wall, '.' free, 'C' the cheese (exactly one).
    Returns:
        (wall, goal): wall an (H, W) bool array, goal the (row, col) of the cheese.
    Raises:
        ValueError if rows differ in length, the border is not all wall, 'C' is not unique, or
        another character appears.
    """
    if len(set(len(r) for r in rows)) != 1:
        raise ValueError("maze rows must have equal length")
    grid = np.array([list(r) for r in rows])
    if not set(np.unique(grid)) <= {"#", ".", "C"}:
        raise ValueError(f"maze may contain only '#', '.', 'C'; got {sorted(np.unique(grid))}")
    wall = grid == "#"
    if not (wall[0].all() and wall[-1].all() and wall[:, 0].all() and wall[:, -1].all()):
        raise ValueError("the maze border must be all wall")
    goal = np.argwhere(grid == "C")
    if len(goal) != 1:
        raise ValueError(f"the maze needs exactly one 'C', found {len(goal)}")
    return wall, (int(goal[0][0]), int(goal[0][1]))


def bfs_distance(wall, goal):
    """Geodesic distance, in 4-neighbour cell steps, from every cell to the goal.

    Args:
        wall: (H, W) bool array with an all-wall border; goal: (row, col).
    Returns:
        (H, W) float array: 0 at the goal, inf in walls and in cells that cannot reach the goal.
    """
    dist = np.full(wall.shape, np.inf)
    dist[goal] = 0.0
    queue = collections.deque([goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS_4:   # the border is wall, so every neighbour of a free cell is in range
            rr, cc = r + dr, c + dc
            if not wall[rr, cc] and np.isinf(dist[rr, cc]):
                dist[rr, cc] = dist[r, c] + 1.0
                queue.append((rr, cc))
    return dist


class MazeEnv:
    """A batch of agents in one fixed maze. See the module docstring for the task definition."""

    def __init__(self, maze, n_rays, ray_max, ray_step, v_max, turn_max, goal_radius, goal_cue,
                 reward, gamma, device, seed):
        """Build the maze tables and the drop generator; no agents exist until `reset`.

        Args:
            maze: list of str, the ASCII grid (see the module docstring).
            n_rays: int, range sensors per agent.
            ray_max: float, cells, sensor range; ray_step: float, cells, ray-marching step.
            v_max: float, cells per step at speed 1; turn_max: float, radians per step at turn 1.
            goal_radius: float, cells, distance from the cheese centre that ends the episode.
            goal_cue: bool, add the two bearing-to-cheese inputs.
            reward: mapping with float entries goal, step, wall, shape.
            gamma: float, the discount factor, used in the shaping term.
            device: torch.device; seed: int, seeds the drop generator.
        """
        wall, goal = parse_maze(maze)
        dist = bfs_distance(wall, goal)
        self.wall = torch.as_tensor(wall, device=device)
        self.dist = torch.as_tensor(dist, dtype=torch.float32, device=device)
        self.goal_xy = torch.tensor([goal[1] + 0.5, goal[0] + 0.5], device=device)
        start = np.argwhere(~wall & np.isfinite(dist))
        start = start[(start[:, 0] != goal[0]) | (start[:, 1] != goal[1])]
        self.start_cells = torch.as_tensor(start, device=device)          # (n_free, 2) as (row, col)
        self.n_rays, self.ray_max, self.ray_step = int(n_rays), float(ray_max), float(ray_step)
        self.ray_offsets = torch.arange(self.n_rays, device=device) * (2 * math.pi / self.n_rays)
        n_samples = int(round(self.ray_max / self.ray_step))
        self.ray_samples = torch.arange(1, n_samples + 1, device=device, dtype=torch.float32) * self.ray_step
        self.v_max, self.turn_max, self.goal_radius = float(v_max), float(turn_max), float(goal_radius)
        self.goal_cue = bool(goal_cue)
        self.reward = {k: float(reward[k]) for k in ("goal", "step", "wall", "shape")}
        self.gamma = float(gamma)
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))
        self.n_inputs = self.n_rays + (2 if self.goal_cue else 0)
        self.n_actions = 2
        self.xy = self.heading = self.done = None

    def cell(self, xy):
        """(row, col) long tensors of the cells containing the points xy (n, 2), clamped into the grid."""
        H, W = self.wall.shape
        col = xy[:, 0].floor().long().clamp(0, W - 1)
        row = xy[:, 1].floor().long().clamp(0, H - 1)
        return row, col

    def reset(self, n, generator=None):
        """Drop n agents: a random reachable free cell (never the cheese), a uniform position inside
        it, a uniform heading.

        Args:
            n: int, number of agents (the batch); generator: torch.Generator on this env's device,
            or None for the env's own (a fixed generator gives the same drops every call).
        Returns:
            observations, (n_inputs, n) float tensor.
        """
        g = self.generator if generator is None else generator
        idx = torch.randint(len(self.start_cells), (n,), generator=g, device=self.device)
        cells = self.start_cells[idx].float()
        offset = torch.rand((n, 2), generator=g, device=self.device)
        self.xy = torch.stack([cells[:, 1] + offset[:, 0], cells[:, 0] + offset[:, 1]], dim=1)
        self.heading = torch.rand((n,), generator=g, device=self.device) * (2 * math.pi)
        self.done = torch.zeros(n, dtype=torch.bool, device=self.device)
        return self.observe()

    def observe(self):
        """Sensor inputs for the current positions and headings: (n_inputs, B) float tensor."""
        angles = self.heading[:, None] + self.ray_offsets[None, :]                     # (B, R)
        direction = torch.stack([angles.cos(), angles.sin()], dim=-1)                  # (B, R, 2)
        points = (self.xy[:, None, None, :]
                  + direction[:, :, None, :] * self.ray_samples[None, None, :, None])  # (B, R, S, 2)
        row, col = self.cell(points.reshape(-1, 2))
        hit = self.wall[row, col].reshape(points.shape[:3])                            # (B, R, S)
        first = hit.int().argmax(dim=2)         # index of the first wall sample (argmax = first max)
        dist = (first + 1).float() * self.ray_step
        dist = torch.where(hit.any(dim=2), dist, torch.full_like(dist, self.ray_max))
        obs = 1.0 - dist / self.ray_max                                                # (B, R)
        if self.goal_cue:
            to_goal = self.goal_xy[None, :] - self.xy
            bearing = torch.atan2(to_goal[:, 1], to_goal[:, 0]) - self.heading
            obs = torch.cat([obs, bearing.cos()[:, None], bearing.sin()[:, None]], dim=1)
        return obs.T.contiguous()

    def step(self, action):
        """Advance every agent whose episode is still running by one step.

        Args:
            action: (2, B) raw policy outputs, turn and speed (see the module docstring).
        Returns:
            (obs, reward, done): obs (n_inputs, B); reward (B,) float, 0 for finished agents;
            done (B,) bool, True once the agent has reached the cheese at this step or earlier.
        """
        live = ~self.done
        turn = action[0].clamp(-1.0, 1.0) * self.turn_max * live
        speed = action[1].clamp(0.0, 1.0) * self.v_max * live
        heading = self.heading + turn
        new_xy = self.xy + torch.stack([heading.cos(), heading.sin()], dim=1) * speed[:, None]
        # blocked if the end point is in a wall, or if the step cuts a corner between two wall
        # cells (a step is shorter than a cell, so it crosses at most one row and one column line)
        r0, c0 = self.cell(self.xy)
        r1, c1 = self.cell(new_xy)
        corner = (r0 != r1) & (c0 != c1)
        blocked = self.wall[r1, c1] | (corner & (self.wall[r0, c1] | self.wall[r1, c0]))
        d_before = self.dist[r0, c0]
        self.xy = torch.where(blocked[:, None], self.xy, new_xy)
        self.heading = heading
        d_after = self.dist[self.cell(self.xy)]
        reached = live & ((self.xy - self.goal_xy[None, :]).norm(dim=1) < self.goal_radius)
        reward = (self.reward["step"]
                  + self.reward["wall"] * blocked.float()
                  + self.reward["shape"] * (d_before - self.gamma * d_after)
                  + self.reward["goal"] * reached.float()) * live.float()
        self.done = self.done | reached
        return self.observe(), reward, self.done.clone()
