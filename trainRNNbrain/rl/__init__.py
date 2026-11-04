"""Reinforcement-learning path: the maze task, a policy wrapper around RNN_torch, and PPO.

Nothing here is imported by the supervised path (trainer/, training/, tasks/), and nothing here
changes it: the RNN is RNN_torch as it is, run one step at a time through its own `rhs` and weights.
"""
