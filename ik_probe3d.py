"""Feasibility baseline for the 3-DOF arm: analytic yaw + 2-link planar IK + PD."""

from __future__ import annotations

import numpy as np

from env import make_env


L1, L2 = 0.50, 0.45
EPISODES = 200


def ik3(tx, ty, tz):
    """Yaw + vertical-plane 2-link IK for the 3R arm."""
    q1 = np.arctan2(ty, tx)
    r_h = np.hypot(tx, ty)
    d = np.clip(np.hypot(r_h, tz), 1e-6, L1 + L2 - 1e-6)
    c2 = np.clip((d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1)
    q3 = np.arccos(c2)
    q2 = np.arctan2(tz, r_h) - np.arctan2(L2 * np.sin(q3), L1 + L2 * np.cos(q3))
    # MuJoCo hinge joints rotate the OPPOSITE way about +y, so negate q2/q3.
    return np.array([q1, -q2, -q3])


def run_one(env, seed):
    _obs, _info = env.reset(seed=seed)
    q = ik3(*env.target)
    for _ in range(env.max_steps):
        cur = env.data.qpos[:3]
        vel = env.data.qvel[:3]
        action = np.clip(6.0 * (q - cur) - 1.5 * vel, -1, 1)
        _obs, _r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info["dist_to_target"], info["success"]


def main():
    env = make_env("three_joint")
    dists, succ = [], []
    for s in range(EPISODES):
        d, ok = run_one(env, s)
        dists.append(d)
        succ.append(ok)
    env.close()

    dists = np.asarray(dists)
    print(f"IK/PD baseline over {EPISODES} episodes (3D arm)")
    print(f"  final dist mean/median : {dists.mean():.3f} / {np.median(dists):.3f}")
    print(f"  final dist max         : {dists.max():.3f}")
    print(f"  success (<0.10) rate   : {np.mean(succ):.1%}")


if __name__ == "__main__":
    main()
