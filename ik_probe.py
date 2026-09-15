"""Feasibility baseline: an analytic 2-link IK + PD torque controller.

Confirms the target radius is reachable and gives a benchmark distance: if a
hand-written IK/PD controller cannot solve the task, the task itself (dynamics,
contacts or reward) is broken rather than the learning algorithm.
"""

from __future__ import annotations

import numpy as np

from env import make_env


L1, L2 = 0.55, 0.50
EPISODES = 200


def ik_target(x, y):
    """Closed-form IK for the planar two-link arm."""
    r = np.clip(np.hypot(x, y), 1e-6, L1 + L2 - 1e-6)
    c2 = np.clip((r ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1)
    q2 = np.arccos(c2)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return q1, q2


def pd_controller(env):
    """Run one episode of saturated PD force control towards the IK solution."""
    _obs, info = env.reset()
    target = env.target
    q1, q2 = ik_target(target[0], target[1])
    d0 = info["dist_to_target"]
    for _ in range(env.max_steps):
        cur1, cur2 = env.data.qpos[0], env.data.qpos[1]
        v1, v2 = env.data.qvel[0], env.data.qvel[1]
        action = np.clip([6.0 * (q1 - cur1) - 1.2 * v1,
                          6.0 * (q2 - cur2) - 1.2 * v2], -1, 1)
        _obs, _r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return d0, info["dist_to_target"], info["success"]


def main():
    env = make_env("two_joint")
    d0s, dfs, succ = [], [], []
    for s in range(EPISODES):
        env.reset(seed=s)
        d0, df, ok = pd_controller(env)
        d0s.append(d0)
        dfs.append(df)
        succ.append(ok)
    env.close()

    print(f"IK/PD baseline over {EPISODES} episodes")
    print(f"  initial dist mean   : {np.mean(d0s):.3f}")
    print(f"  final   dist mean   : {np.mean(dfs):.3f}  (median {np.median(dfs):.3f})")
    print(f"  final   dist max    : {np.max(dfs):.3f}")
    print(f"  success (<0.08) rate: {np.mean(succ):.1%}")


if __name__ == "__main__":
    main()
