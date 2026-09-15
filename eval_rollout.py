"""Evaluate a trained PPO policy and render a demo video + learning curve."""

from __future__ import annotations

import argparse
import glob
import os

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cli import add_env_arg, add_model_arg, add_out_arg, apply_defaults, load_policy
from env import make_env as build_env
from paths import default_tb_dir, ensure_dir


def load_curve(tb_dir):
    """Return ``[(steps, ep_rew_mean)]`` series from the TensorBoard events."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    series = []
    for f in sorted(glob.glob(os.path.join(tb_dir, "**", "events*"), recursive=True)):
        acc = EventAccumulator(f)
        acc.Reload()
        if "rollout/ep_rew_mean" not in acc.Tags().get("scalars", []):
            continue
        events = acc.Scalars("rollout/ep_rew_mean")
        series.append(([e.step for e in events], [e.value for e in events]))
    return series or None


def plot_curve(curve, env_name, out_path):
    """Plot the concatenated episodic-reward history (across resumed runs)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    offset = 0
    xs, ys = [], []
    for steps, values in curve:
        xs.extend([s + offset for s in steps])
        ys.extend(values)
        if steps:
            offset += steps[-1]
    order = np.argsort(xs)
    ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], lw=2, color="#1f77b4",
            label="episodic mean reward")
    ax.set_xlabel("total timesteps")
    ax.set_ylabel("episodic mean reward")
    ax.set_title(f"PPO learning curve ({env_name} reacher)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def evaluate(env, model, episodes):
    """Roll out the deterministic policy; return the per-episode metrics."""
    final_dists, success, rewards, steps = [], [], [], []
    for s in range(episodes):
        obs, _ = env.reset(seed=s)
        ep_reward = 0.0
        info = {}
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        final_dists.append(info["dist_to_target"])
        success.append(info["success"])
        rewards.append(ep_reward)
        steps.append(info["steps"])
    return np.asarray(final_dists), np.asarray(success), np.asarray(rewards), np.asarray(steps)


def render_video(env, model, episodes, fps):
    frames = []
    for s in range(episodes):
        obs, _ = env.reset(seed=s)
        for _ in range(env.max_steps):
            frames.append(env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_env_arg(ap)
    add_model_arg(ap)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--video-episodes", type=int, default=3)
    add_out_arg(ap)
    ap.add_argument("--viewer", action="store_true",
                    help="open a live MuJoCo viewer window (run trained policy)")
    ap.add_argument("--fps", type=float, default=50.0,
                    help="playback fps of the demo video / live viewer")
    args = apply_defaults(ap.parse_args())

    if args.viewer:
        from viewer_demo import run_live_viewer

        print("Opening live MuJoCo viewer; close the window to stop. T = new target.")
        run_live_viewer(args.model, env_name=args.env, seed=0,
                        fps=args.fps, n_episodes=None)
        print("viewer closed.")
        return

    ensure_dir(args.out)
    model = load_policy(args.model)

    # ---- quantitative evaluation --------------------------------------
    env = build_env(args.env, render_mode=None)
    final_dists, success, rewards, steps = evaluate(env, model, args.episodes)
    env.close()

    print("=== PPO evaluation ===")
    print(f"episodes              : {args.episodes}")
    print(f"episode success rate  : {success.mean():.1%}")
    print(f"final dist mean/median: {final_dists.mean():.3f} / {np.median(final_dists):.3f}")
    print(f"final dist <= 0.10    : {(final_dists < 0.10).mean():.1%}")
    print(f"episode reward mean   : {np.mean(rewards):.1f}")
    print(f"episode steps mean    : {np.mean(steps):.1f}")

    # ---- learning curve ------------------------------------------------
    curve = load_curve(default_tb_dir(args.env))
    if curve:
        plot_curve(curve, args.env, os.path.join(args.out, "learning_curve.png"))
        print("saved learning_curve.png")

    # ---- demo video ----------------------------------------------------
    vid_env = build_env(args.env, render_mode="rgb_array")
    frames = render_video(vid_env, model, args.video_episodes, args.fps)
    vid_env.close()

    if not frames:
        print("no demo video requested (--video-episodes 0)")
        return
    gif = os.path.join(args.out, "rollout.gif")
    mp4 = os.path.join(args.out, "rollout.mp4")
    imageio.mimsave(gif, frames, fps=args.fps)
    print(f"saved {gif} ({len(frames)} frames)")
    try:
        imageio.mimsave(mp4, frames, fps=args.fps, format="FFMPEG")
        print(f"saved {mp4}")
    except Exception as e:  # ffmpeg is optional
        print("mp4 skipped:", e)


if __name__ == "__main__":
    main()
