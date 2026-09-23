"""Render a grasp rollout to a GIF, with the vision-localised point marker.

Drives the Pro7 pick scene (expert servo or a supervised/learned policy) and
records the overview camera.  A translucent green marker tracks the detected cube
point, so you can watch detection -> approach -> grasp.  Output is a GIF you can
view anywhere (no GUI window needed).

Run::

    python3 grasp/visualize.py --mode expert
    python3 grasp/visualize.py --mode policy --policy-path results/pro7_pick/grasp_policy_online.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

# Allow `python3 grasp/visualize.py` as well as `python3 -m grasp.visualize`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import make_env
from grasp.common import set_marker, teacher_action
from grasp.policy import action, load_policy
from paths import ensure_dir, results_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="expert", choices=["expert", "policy"])
    ap.add_argument("--obs-target", default="true", choices=["true", "vision"])
    ap.add_argument("--policy-path",
                    default=results_path("pro7_pick", "grasp_policy_online.pt"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default=results_path("pro7_pick", "grasp_rollout.gif"))
    args = ap.parse_args()

    env = make_env("pro7_pick", obs_target=args.obs_target)
    policy = load_policy(args.policy_path) if args.mode == "policy" else None

    renderer = mujoco.Renderer(env.model, height=360, width=480)

    frames = []
    grasped = 0
    dists = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        for _ in range(env.max_steps):
            act = teacher_action(env) if args.mode == "expert" else action(policy, obs)
            obs, _r, terminated, truncated, info = env.step(act)

            target, _seen = env.detected_world()
            set_marker(env.data, env.ids, target)
            renderer.update_scene(env.data, camera="cam_iso")
            frames.append(renderer.render().copy())
            if terminated or truncated:
                break
        grasped += int(info["grasped"])
        dists.append(info["dist_to_cube"])

    renderer.close()
    env.close()
    ensure_dir(os.path.dirname(args.out))
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"grasped {grasped}/{args.episodes}  mean final dist={np.mean(dists):.3f} m")
    print(f"saved {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
