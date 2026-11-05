"""The maze task: a fixed maze run by a batch of point agents in parallel, in torch, on the RNN's device.

The reinforcement-learning task of the project, in the place of a supervised Task. It has no input and
target batches (`get_batch` raises): the agent's observations depend on what it did, so the trainer
(trainRNNbrain/trainer/TrainerPPO.py) plays episodes through `reset`, `observe` and `step` and trains
the RNN on what it saw and did. Built from configs/task/Maze.yaml. Until 5 Nov 2026 this class was
`MazeEnv` in trainRNNbrain/rl/maze_env.py; the tables, the sensors, the move, the slide and the reward
are unchanged.

The maze is an ASCII grid: '#' wall, '.' free, 'C' the cheese (exactly one cell). Every border
cell must be a wall, so an agent can never leave the grid and every sensor ray hits something.
Cells are 1 x 1 units: cell (row, col) covers x in [col, col + 1) and y in [row, row + 1). An agent
is a point (x, y) with a heading angle (radians, 0 along +x, counter-clockwise positive). Each
episode drops every agent in a random reachable free cell with a random heading.

Observation (the RNN's inputs): `n_rays` range sensors spread evenly over 360 degrees, egocentric
(ray 0 points along the heading). Input i is 1 - d_i / ray_max, with d_i the distance to the first
wall along ray i (ray marching with step `ray_step`), so a wall at the nose reads 1 and open space
reads 0. With `action_feedback` two more inputs carry the agent's own previous executed action,
turn / turn_max in [-1, 1] and speed / v_max in [0, 1] (zeros at the first step): an efference
copy, so the network knows what it actually did, exploration noise included, and can dead-reckon
(Pavel, 4 Nov 2026). With `goal_cue` two more inputs give cos and sin of the bearing to the cheese
relative to the heading (a "scent"); without it the agent has no information about the cheese and
has to localise from the walls and its memory.

Action (2 raw policy outputs per agent, squashed here, not clipped, so no sample is ever stuck
at a bound): turn = turn_max * tanh(a[0]) (radians per step); speed = v_max * sigmoid(a[1])
(cells per step; an untrained policy with a[1] near 0 moves at v_max / 2). A move whose end
point lies in a wall cell, or which cuts a corner between two wall cells, is not cancelled
outright: the agent SLIDES, keeping the component of the move that stays free (x first, then y),
as a body pressed against a wall does; when neither component is free it stays put (the new
heading is kept). Every bump, a move whose intended end point was blocked, slid or not, costs
`reward["wall"]` (Pavel, 4 Nov 2026: a small cost for bumping into walls). Without sliding, an
untrained agent in a one-cell-wide corridor stood wedged against a wall in 77 to 93 of 100 steps
(4 Nov 2026) and learned only not to bump.

Reward per step: `reward["step"]` (a cost, negative), `reward["wall"]` on a bump,
`reward["goal"]` once, on coming within `goal_radius` of the cheese centre (the episode ends for that
agent), and the shaping `reward["shape"] * (d - d')`, with d and d' the breadth-first geodesic cell
distance to the cheese before and after the move, so every cell of progress pays and standing
still pays nothing. The shaping is deliberately NOT the discounted potential-based form
`d - gamma * d'` (Ng, Harada and Russell, 1999): with the discount inside, a stationary agent earns
shape * (1 - gamma) * d per step, which at 10 cells from the cheese outpaces the step cost, and the
first PPO runs (Della array 16790695, 3 Nov 2026) converged to standing still.

Every tensor has the batch as its LAST axis for observations (n_inputs, B) and actions (2, B), the
RNN's convention; the internal state uses (B, ...) because torch indexing is easier that way.
"""
import collections
import math
import numpy as np
import torch
from trainRNNbrain.tasks.TaskBase import Task

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


class TaskMaze(Task):
    """A batch of agents in one fixed maze. See the module docstring for the task definition.

    Attributes the trainer and the analysis read: n_inputs (sensors plus the optional inputs),
    n_outputs and n_actions (2, turn and speed), device, goal_cue, action_feedback, n_rays, ray_max,
    ray_offsets, wall (H, W) bool tensor, goal_xy (2,) the cheese centre as (x, y), and after a
    `reset` the state xy (B, 2), heading (B,), done (B,) and last_action (B, 2).
    """

    def __init__(self, maze, n_rays, ray_max, ray_step, v_max, turn_max, goal_radius, goal_cue,
                 action_feedback, reward, seed, n_inputs=None, n_outputs=None, device=None):
        """Build the maze tables and the drop generator; no agents exist until `reset`.

        Args:
            maze: list of str, the ASCII grid (see the module docstring).
            n_rays: int, range sensors per agent.
            ray_max: float, cells, sensor range; ray_step: float, cells, ray-marching step.
            v_max: float, cells per step at full speed; turn_max: float, radians per step at full turn.
            goal_radius: float, cells, distance from the cheese centre that ends the episode.
            goal_cue: bool, add the two bearing-to-cheese inputs.
            action_feedback: bool, add the two previous-executed-action inputs.
            reward: mapping with float entries goal, step, wall, shape.
            seed: int, the run seed. The drop generator is seeded with seed + 1, so its stream
                differs from the RNN's, which the run script seeds with the run seed itself (the
                seeding of the maze runs before 5 Nov 2026, so a rerun reproduces their drops).
            n_inputs: int or None. The number of inputs is derived from the sensors, n_rays + 2 with
                action_feedback + 2 with goal_cue; a value given here is checked against it, so a
                saved config cannot disagree with the task it describes. None accepts the derived
                number (the run script records it in the saved config).
            n_outputs: int or None, likewise for the 2 action outputs (turn and speed).
            device: torch.device, or None for CUDA when available, else the CPU (the RNN's choice).
        Raises:
            ValueError if n_inputs or n_outputs is given and does not match the derived number.
        """
        wall, goal = parse_maze(maze)
        dist = bfs_distance(wall, goal)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
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
        self.action_feedback = bool(action_feedback)
        self.reward = {k: float(reward[k]) for k in ("goal", "step", "wall", "shape")}
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed) + 1)
        derived_inputs = self.n_rays + (2 if self.action_feedback else 0) + (2 if self.goal_cue else 0)
        self.n_actions = 2
        if n_inputs is not None and int(n_inputs) != derived_inputs:
            raise ValueError(f"task.n_inputs is {n_inputs}, but the sensors give {derived_inputs} inputs "
                             f"(n_rays {self.n_rays}, action_feedback {self.action_feedback}, "
                             f"goal_cue {self.goal_cue}); leave it null to record the derived number")
        if n_outputs is not None and int(n_outputs) != self.n_actions:
            raise ValueError(f"task.n_outputs is {n_outputs}, but the maze has {self.n_actions} actions "
                             f"(turn and speed); leave it null")
        # no trial length: an episode's length in decisions is the trainer's T
        Task.__init__(self, n_steps=None, n_inputs=derived_inputs, n_outputs=self.n_actions, seed=seed)
        self.xy = self.heading = self.done = self.last_action = None

    def get_batch(self, **kwargs):
        """The maze has no input and target batches: TrainerPPO.rollout plays episodes instead."""
        raise NotImplementedError("the maze has no input/target batches; TrainerPPO.rollout plays episodes")

    def cell(self, xy):
        """(row, col) long tensors of the cells containing the points xy (n, 2), clamped into the grid."""
        H, W = self.wall.shape
        col = xy[:, 0].floor().long().clamp(0, W - 1)
        row = xy[:, 1].floor().long().clamp(0, H - 1)
        return row, col

    def blocked(self, xy, new_xy):
        """Which moves from xy to new_xy (both (n, 2)) end in a wall or cut a corner between two
        wall cells (a step is shorter than a cell, so it crosses at most one row and one column
        line). Returns an (n,) bool tensor."""
        r0, c0 = self.cell(xy)
        r1, c1 = self.cell(new_xy)
        corner = (r0 != r1) & (c0 != c1)
        return self.wall[r1, c1] | (corner & (self.wall[r0, c1] | self.wall[r1, c0]))

    def reset(self, n, generator=None):
        """Drop n agents: a random reachable free cell (never the cheese), a uniform position inside
        it, a uniform heading.

        Args:
            n: int, number of agents (the batch); generator: torch.Generator on this task's device,
            or None for the task's own (a fixed generator gives the same drops every call).
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
        self.last_action = torch.zeros((n, 2), device=self.device)      # (turn, speed), scaled
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
        if self.action_feedback:
            obs = torch.cat([obs, self.last_action], dim=1)
        if self.goal_cue:
            to_goal = self.goal_xy[None, :] - self.xy
            bearing = torch.atan2(to_goal[:, 1], to_goal[:, 0]) - self.heading
            obs = torch.cat([obs, bearing.cos()[:, None], bearing.sin()[:, None]], dim=1)
        return obs.T.contiguous()

    def step(self, action):
        """Advance every agent whose episode is still running by one step.

        Args:
            action: (2, B) raw policy outputs, squashed here to turn and speed (module docstring).
        Returns:
            (obs, reward, done): obs (n_inputs, B); reward (B,) float, 0 for finished agents;
            done (B,) bool, True once the agent has reached the cheese at this step or earlier.
        """
        live = ~self.done
        turn = torch.tanh(action[0]) * self.turn_max * live
        speed = torch.sigmoid(action[1]) * self.v_max * live
        self.last_action = torch.stack([turn / self.turn_max, speed / self.v_max], dim=1)
        heading = self.heading + turn
        full = self.xy + torch.stack([heading.cos(), heading.sin()], dim=1) * speed[:, None]
        hit = self.blocked(self.xy, full)
        # slide: keep the free component of a blocked move, x first, then y; an axis-aligned
        # partial move crosses one grid line at most, so it cannot cut a corner
        slide_x = torch.stack([full[:, 0], self.xy[:, 1]], dim=1)
        slide_y = torch.stack([self.xy[:, 0], full[:, 1]], dim=1)
        ok_x = ~self.blocked(self.xy, slide_x) & (slide_x[:, 0] != self.xy[:, 0])
        ok_y = ~self.blocked(self.xy, slide_y) & (slide_y[:, 1] != self.xy[:, 1])
        slid = torch.where(ok_x[:, None], slide_x, torch.where(ok_y[:, None], slide_y, self.xy))
        d_before = self.dist[self.cell(self.xy)]
        self.xy = torch.where(hit[:, None], slid, full)
        self.heading = heading
        d_after = self.dist[self.cell(self.xy)]
        reached = live & ((self.xy - self.goal_xy[None, :]).norm(dim=1) < self.goal_radius)
        reward = (self.reward["step"]
                  + self.reward["wall"] * hit.float()
                  + self.reward["shape"] * (d_before - d_after)
                  + self.reward["goal"] * reached.float()) * live.float()
        self.done = self.done | reached
        return self.observe(), reward, self.done.clone()
