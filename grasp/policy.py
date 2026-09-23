"""The small MLP grasp policy shared by the training and visualisation scripts.

:mod:`grasp.supervised` / :mod:`grasp.train_live` train it, :mod:`grasp.visualize`
/ :mod:`grasp.view` replay it.  Keeping it (and its observation/action
dimensions) in one module means a checkpoint always loads into the same
architecture, no matter which script produced it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

#: Observation / action sizes of ``env.RokaePro7Pick``.
OBS_DIM = 30
#: Observation size of ``env.RokaePro7PickPlace`` (the pick & place task adds
#: the holding flag, the sub-goal error, the cube-to-pad error and the plan's
#: "parked on the pad" flag).
OBS_DIM_PLACE = 38
ACT_DIM = 8


class Policy(nn.Module):
    """Two hidden layers + tanh head: bounded actions in ``[-1, 1]``."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def action(policy: Policy, obs: np.ndarray) -> np.ndarray:
    """Deterministic action for a single observation."""
    with torch.no_grad():
        return policy(torch.as_tensor(obs[None], dtype=torch.float32)).numpy()[0]


def load_policy(path: str, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM) -> Policy:
    """Load a saved policy and switch it to eval mode."""
    policy = Policy(obs_dim=obs_dim, act_dim=act_dim)
    policy.load_state_dict(torch.load(path, map_location="cpu"))
    policy.eval()
    return policy


def rollout(env, policy: Policy, seed: int, on_step=None) -> tuple[bool, float]:
    """Run the learned policy on one cube; return ``(grasped, final_dist)``.

    ``on_step`` (if given) is called after every environment step, which the
    live-visualisation trainer uses to keep its window updating during
    evaluation.
    """
    info = rollout_info(env, policy, seed, on_step=on_step)
    return bool(info["grasped"]), float(info["dist_to_cube"])


def rollout_info(env, policy: Policy, seed: int, on_step=None) -> dict:
    """Run the learned policy for one episode; return the final ``info`` dict.

    Works for both the grasp and the pick & place environment: the place task
    simply adds ``placed`` / ``place_dist`` to the metrics.
    """
    obs, info = env.reset(seed=seed)
    for _ in range(env.max_steps):
        obs, _r, terminated, truncated, info = env.step(action(policy, obs))
        if on_step is not None:
            on_step()
        if terminated or truncated:
            break
    return info
