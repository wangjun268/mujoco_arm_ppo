"""PPO training for the take_box MuJoCo environments (Stable-Baselines3).

    python3 train_ppo.py --env take_box_reach --steps 300000   # 课程 1：两臂贴到抓取面
    python3 train_ppo.py --env take_box --steps 1500000        # 课程 2：抓起并端平
    python3 train_ppo.py --env take_box --viewer               # 边训边看

Checkpoints land in ``runs/ppo_<env>.zip`` and TensorBoard events in
``runs/tb_<env>/``.  The trained policy is a joint-torque + grip controller for
the 14 arm joints and 2 grasp channels of the real cell.
"""

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

from take_box_env import env_names, make_env

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")


class EpisodeStats(BaseCallback):
    """Mean reward / success rate / lift of the finished episodes."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.rewards, self.success, self.lifted = [], [], []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
                self.success.append(info.get("success", False))
                self.lifted.append(info.get("lift", 0.0))
        return True

    def stats(self) -> dict:
        if not self.rewards:
            return {}
        return {
            "ep_mean_reward": float(np.mean(self.rewards)),
            "ep_success_rate": float(np.mean(self.success)),
            "ep_lift_mean": float(np.mean(self.lifted)),
        }


class LiveViewerCallback(BaseCallback):
    """Optional MuJoCo window on one of the vectorised environments."""

    def __init__(self, viewer, verbose: int = 0):
        super().__init__(verbose)
        self.viewer = viewer

    def _on_step(self) -> bool:
        self.viewer.sync()
        # mujoco's passive viewer exposes is_running(); closing the window stops
        # training instead of raising AttributeError on the next step
        if not self.viewer.is_running():
            print("MuJoCo viewer closed - stopping training")
            return False
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="take_box", choices=env_names())
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--model", default=None, help="checkpoint path, default runs/ppo_<env>")
    parser.add_argument("--init-model", default=None, help="resume from this .zip")
    parser.add_argument("--viewer", action="store_true", help="open a live MuJoCo window")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model_path = args.model or os.path.join(RUNS, f"ppo_{args.env}")
    tb_dir = os.path.join(RUNS, f"tb_{args.env}")
    os.makedirs(RUNS, exist_ok=True)

    venv = DummyVecEnv([lambda: Monitor(make_env(args.env)) for _ in range(args.n_envs)])
    print(f"env={args.env}  device={device}  n_envs={args.n_envs}")

    if args.init_model:
        model = PPO.load(args.init_model, env=venv, device=device,
                         tensorboard_log=tb_dir, verbose=1, learning_rate=args.lr)
        print(f"resumed from {args.init_model}")
    else:
        model = PPO(
            "MlpPolicy", venv, learning_rate=args.lr, n_steps=1024, batch_size=256,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.001, vf_coef=0.5, max_grad_norm=0.5,
            verbose=1, tensorboard_log=tb_dir, device=device, seed=args.seed,
        )

    stats = EpisodeStats()
    callbacks = [stats]
    viewer = None
    if args.viewer:
        try:
            import mujoco.viewer

            base = venv.envs[0].unwrapped
            viewer = mujoco.viewer.launch_passive(base.model, base.data)
            callbacks.append(LiveViewerCallback(viewer))
            print("live MuJoCo window: env #0 (close it and training continues)")
        except Exception as error:  # no display / GL context
            print("could not open the MuJoCo viewer:", error)
            viewer = None

    started = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
    elapsed = time.time() - started
    print(f"finished {args.steps} steps in {elapsed:.1f}s ({args.steps / elapsed:.0f} steps/s)")
    print("last stats:", stats.stats())

    model.save(model_path)
    print(f"saved -> {model_path}.zip")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
