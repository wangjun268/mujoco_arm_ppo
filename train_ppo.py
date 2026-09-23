"""Train a PPO agent (Stable-Baselines3) on a MuJoCo reacher environment."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from cli import add_env_arg, add_model_arg, apply_defaults
from env import make_env as build_env
from live_viewer import LiveViewer
from paths import default_tb_dir, ensure_dir


class EpisodeStats(BaseCallback):
    """Track mean episode reward and success rate for logging."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_rewards = []
        self._success = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
                self._success.append(info.get("success", False))
        return True

    def stats(self):
        if not self._ep_rewards:
            return {}
        return {
            "ep_mean_reward": float(np.mean(self._ep_rewards)),
            "ep_success_rate": float(np.mean(self._success)),
        }


class LiveViewerCallback(BaseCallback):
    """Publish the training state to a :class:`LiveViewer` every step.

    The viewer throttles itself, so calling ``sync()`` thousands of times a
    second is cheap.  Metrics come from the SB3 logger and are drawn in the
    window's four corners; closing the window stops the rendering but lets the
    training run to completion.
    """

    def __init__(self, viewer: LiveViewer, verbose: int = 0):
        super().__init__(verbose)
        self.viewer = viewer
        self._closed = False
        self._ticks = 0

    def _on_step(self) -> bool:
        if self._closed:
            return True
        self._ticks += 1
        if self._ticks % 8 == 0:  # text refresh is much cheaper when not per step
            self.viewer.set_status(self.status_lines())
        if not self.viewer.sync():
            self._closed = True
            print("\n[live viewer] window closed - training continues headless.")
        return True

    def status_lines(self) -> list:
        # Defensive: the callback can be exercised without a model attached
        # (unit tests), in which case there are simply no metrics yet.
        model = getattr(self, "model", None)
        logger = getattr(model, "logger", None)
        values = getattr(logger, "name_to_value", None) or {}

        def value(key, fmt="{:.2f}", fallback="-"):
            raw = values.get(key)
            return fallback if raw is None else fmt.format(raw)

        return [
            f"step {self.num_timesteps}",
            f"fps {value('time/fps', '{:.0f}')}",
            f"ep_rew {value('rollout/ep_rew_mean')}",
            f"ep_len {value('rollout/ep_len_mean', '{:.0f}')}",
        ]


def attach_live_viewer(venv, env_index: int = 0, fps: float = 60.0) -> LiveViewer:
    """Open a MuJoCo window on one of the vectorised training environments."""
    base_env = venv.envs[env_index].unwrapped
    viewer = LiveViewer(base_env.model, base_env.data, fps=fps)
    candidates = [
        name for name in (getattr(base_env, "CAMERA", None), "cam_iso")
        if name
    ]
    viewer.set_camera(*candidates)
    return viewer


def resolve_device(requested: str) -> str:
    """``auto`` -> CUDA when available, otherwise CPU."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_env_arg(parser)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--viewer", action="store_true",
                        help="open a live MuJoCo window and watch one env train")
    parser.add_argument("--viewer-env", type=int, default=0,
                        help="which vectorised sub-environment the window shows")
    parser.add_argument("--viewer-fps", type=float, default=60.0,
                        help="UI refresh rate of the live training window")
    add_model_arg(parser, help="model path (default: results/ppo_<env>)")
    parser.add_argument("--init-model", default=None,
                        help="path to a .zip checkpoint to resume from")
    args = apply_defaults(parser.parse_args())

    device = resolve_device(args.device)
    ensure_dir(os.path.dirname(os.path.abspath(args.model)))
    tb_dir = default_tb_dir(args.env)

    venv = DummyVecEnv(
        [lambda: Monitor(build_env(args.env)) for _ in range(args.n_envs)]
    )

    print(f"env={args.env}  device -> {device}")
    if args.init_model:
        model = PPO.load(args.init_model, env=venv, device=device,
                         tensorboard_log=tb_dir, verbose=1,
                         learning_rate=args.lr)
        print(f"Resumed from {args.init_model}")
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=args.lr,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=tb_dir,
            device=device,
            seed=args.seed,
        )

    stats = EpisodeStats()
    callbacks = [stats]
    viewer = None
    if args.viewer:
        try:
            viewer = attach_live_viewer(venv, args.viewer_env, args.viewer_fps)
            callbacks.append(LiveViewerCallback(viewer))
            print(f"live MuJoCo window: env #{args.viewer_env}"
                  " (close it and training keeps running)")
        except Exception as ex:  # no display / GL context
            print("could not open the MuJoCo viewer:", ex)
            viewer = None

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=True,
        )
    finally:
        if viewer is not None:
            viewer.close()
    dt = time.time() - t0
    print(f"\nTraining finished in {dt:.1f}s ({args.steps / dt:.0f} steps/s)")
    print("Last stats:", stats.stats())

    model.save(args.model)
    print(f"Saved model -> {args.model}.zip")


if __name__ == "__main__":
    main()
