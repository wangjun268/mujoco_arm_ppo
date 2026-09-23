"""Run a trained take_box policy in a live MuJoCo window.

    python3 viewer_demo.py --env take_box --episodes 5

Close the window (or ESC) to stop.  The window shows exactly the model the
policy was trained on, so it doubles as a visual check of the joint torque /
grip commands before they are mapped onto the real controller.
"""

from __future__ import annotations

import argparse
import os
import time

import mujoco.viewer
from stable_baselines3 import PPO

from take_box_env import env_names, make_env

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="take_box", choices=env_names())
    parser.add_argument("--model", default=None)
    parser.add_argument("--episodes", type=int, default=None,
                        help="stop after N episodes (default: until the window closes)")
    parser.add_argument("--fps", type=float, default=50.0)
    args = parser.parse_args()

    model = PPO.load(args.model or os.path.join(RUNS, f"ppo_{args.env}"), device="cpu")
    env = make_env(args.env)
    obs, _ = env.reset(seed=0)

    episodes = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.25]
        viewer.cam.distance = 1.9
        viewer.cam.azimuth = 140.0
        viewer.cam.elevation = -18.0
        while viewer.is_running():
            step_start = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            viewer.sync()
            if terminated or truncated:
                episodes += 1
                obs, _ = env.reset()
                if args.episodes is not None and episodes >= args.episodes:
                    break
            elapsed = time.time() - step_start
            time.sleep(max(0.0, env.model.opt.timestep - elapsed))
    env.close()
    print(f"viewer closed after {episodes} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
