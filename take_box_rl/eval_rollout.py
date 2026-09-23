"""Evaluate a take_box policy: success rate, lift height, tilt, optional video.

    python3 eval_rollout.py --env take_box --episodes 100
    python3 eval_rollout.py --env take_box --episodes 20 --video
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from stable_baselines3 import PPO

from take_box_env import env_names, make_env

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")


def evaluate(env, model, episodes: int):
    records = []
    for seed in range(episodes):
        obs, _ = env.reset(seed=seed)
        info = {}
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        records.append({
            "success": bool(info.get("success", False)),
            "grasped": bool(info.get("grasped", False)),
            "lift": float(info.get("lift", 0.0)),
            "tilt_deg": float(info.get("tilt_deg", 0.0)),
            "dist": float(info.get("dist_to_target", 0.0)),
            "steps": int(info.get("steps", 0)),
        })
    return records


def render_video(env, model, episodes: int, path: str, fps: int = 25) -> str:
    import imageio.v2 as imageio

    frames = []
    for seed in range(episodes):
        obs, _ = env.reset(seed=seed)
        for _ in range(env.max_steps):
            frames.append(env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
    imageio.mimsave(path, frames, fps=fps)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="take_box", choices=env_names())
    parser.add_argument("--model", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-episodes", type=int, default=1)
    args = parser.parse_args()

    model_path = args.model or os.path.join(RUNS, f"ppo_{args.env}")
    model = PPO.load(model_path, device="cpu")
    env = make_env(args.env)
    records = evaluate(env, model, args.episodes)
    env.close()

    success = np.mean([r["success"] for r in records])
    grasped = np.mean([r["grasped"] for r in records])
    lift = np.asarray([r["lift"] for r in records])
    print("=== PPO evaluation ===")
    print(f"episodes        : {args.episodes}")
    print(f"success rate    : {success:.1%}")
    print(f"grasp rate      : {grasped:.1%}")
    print(f"final lift      : mean {lift.mean():.3f} m  median {np.median(lift):.3f} m")
    print(f"final tilt      : median {np.median([r['tilt_deg'] for r in records]):.1f} deg")
    print(f"episode steps   : {np.mean([r['steps'] for r in records]):.0f}")

    if args.video:
        out_dir = os.path.join(RUNS, args.env)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "rollout.mp4")
        video_env = make_env(args.env, render_mode="rgb_array")
        render_video(video_env, model, args.video_episodes, path)
        video_env.close()
        print(f"saved {path}")


if __name__ == "__main__":
    main()
