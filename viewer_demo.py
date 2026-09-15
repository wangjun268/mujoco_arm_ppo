"""Live MuJoCo viewer that drives the trained PPO policy.

Opens the native MuJoCo ``launch_passive`` window (Simulate GUI) and runs the
learned arm policy in a loop, re-randomising the target after each episode.
Close the window to stop; press ``T`` while running to jump to a new target
immediately.

Run::

    python3 viewer_demo.py --env three_joint
    python3 viewer_demo.py --env pro7_joint --episodes 5

Notes
-----
* In a headless-ish box the GL context is created with the software (llvmpipe)
  renderer via ``LIBGL_ALWAYS_SOFTWARE``.  On a normal GPU desktop you can drop
  that; it is harmless to keep.
* MuJoCo's viewer runs a separate daemon UI thread, so we must ``close()`` and
  then let that thread finish before the process exits, otherwise GLFW is torn
  down concurrently and the interpreter segfaults.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")

from cli import add_env_arg, add_model_arg, apply_defaults, load_policy  # noqa: E402
from env import make_env as build_env  # noqa: E402
from live_viewer import KEY_T, LiveViewer  # noqa: E402


def run_live_viewer(
    model_path: str,
    env_name: str = "two_joint",
    seed: int = 0,
    fps: float = 50.0,
    n_episodes: int | None = None,
) -> int:
    """Run the trained policy in a live MuJoCo viewer window.

    Returns the number of completed episodes.
    """
    model = load_policy(model_path)
    env = build_env(env_name)

    reset_requested = {"flag": False}

    def key_callback(keycode: int) -> None:
        # Runs on the viewer UI thread; just flag the request.
        if keycode == KEY_T:
            reset_requested["flag"] = True

    obs, _ = env.reset(seed=seed)
    viewer = LiveViewer(env.model, env.data, key_callback=key_callback, fps=fps)
    viewer.set_camera("cam_xy", "cam_iso")  # the model's dedicated camera

    episodes = 0
    next_frame = time.time()
    try:
        while viewer.running and (
            n_episodes is None or episodes < n_episodes
        ):
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, _info = env.step(action)
            viewer.sync(force=True)  # the loop already paces itself below

            next_frame += 1.0 / fps
            sleep = next_frame - time.time()
            if sleep > 0:
                time.sleep(sleep)

            if terminated or truncated or reset_requested["flag"]:
                reset_requested["flag"] = False
                episodes += 1
                obs, _ = env.reset()
                if n_episodes is not None and episodes >= n_episodes:
                    break
    finally:
        # Stop the UI thread and give it time to tear down GLFW by itself so
        # the main thread doesn't race with it during interpreter shutdown.
        viewer.close()
    return episodes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_env_arg(ap)
    add_model_arg(ap)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--episodes", type=int, default=None,
                    help="auto-close after N episodes (default: run until window closed)")
    args = apply_defaults(ap.parse_args())

    print("Opening MuJoCo viewer; close the window (or ESC) to stop. T = new target.")
    n = run_live_viewer(args.model, env_name=args.env, seed=args.seed,
                        fps=args.fps, n_episodes=args.episodes)
    print(f"viewer closed after {n} episodes.")


if __name__ == "__main__":
    main()
