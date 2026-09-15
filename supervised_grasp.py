"""Real-time supervised (DAgger / behaviour-cloning) training for the 7-DOF grasp.

An expert teacher — the resolved-rate servo from :mod:`grasp_common` — emits the
*correct* action for whoever visits a state.  Early rounds mostly follow the
expert (behaviour cloning); later rounds increasingly follow the learned policy
while still being labelled by the expert (DAgger).  Covering the policy's own
state distribution removes the compounding error that pure BC suffers from.
Every few rounds the learned policy is rolled out on fresh cubes and the grasp
success + final distance are printed live, so you can watch it learn in real
time.  ``--obs-target true`` (default) feeds the ground-truth cube position,
which makes this supervised stage learn reliably; ``--obs-target vision`` runs it
on the RGB-D estimate from ``detect_red_cube`` (harder, since imitation of a
stiff servo is sensitive to observation noise).  Imitation reduces the reaching
error and the grasp success grows (observed up to ~40 % with low train MSE
~0.01), but a stiff feedback grasp is best completed by the expert servo or
further RL.

Run::

    python3 supervised_grasp.py --rounds 40 --eval-every 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from env import make_env
from grasp_common import teacher_action
from grasp_policy import Policy, action, rollout
from paths import ensure_dir, results_path


def main():
    ap = argparse.ArgumentParser(description="Online supervised grasp training.")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--collect-episodes", type=int, default=3)
    ap.add_argument("--updates-per-round", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--buffer-size", type=int, default=60000)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--beta-min", type=float, default=0.2)
    ap.add_argument("--obs-target", default="true", choices=["true", "vision"],
                    help="what the policy observes: true cube position (learnable, "
                         "privileged) or the RGB-D vision estimate")
    ap.add_argument("--visualize", action="store_true",
                    help="after training, open the live MuJoCo viewer with the learned policy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=results_path("pro7_pick"))
    args = ap.parse_args()

    ensure_dir(args.out)
    env = make_env("pro7_pick", obs_target=args.obs_target)
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    buffer = deque(maxlen=args.buffer_size)

    train_losses, succ = [], []
    print(f"obs_target={args.obs_target}")
    print("round | data  | train_mse | eval_success | note")
    for r in range(args.rounds):
        # DAgger mixing: follow the expert with prob beta, else follow the policy,
        # but *always* label the visited state with the expert action.
        beta = args.beta_min + (1.0 - args.beta_min) * max(
            0.0, 1.0 - r / max(1, args.rounds)
        )
        rng = np.random.default_rng(r + 137)
        for ep in range(args.collect_episodes):
            obs, _ = env.reset(seed=args.seed * 1000 + r * 100 + ep)
            for _ in range(env.max_steps):
                expert = teacher_action(env)
                buffer.append((obs.copy(), expert.copy()))
                act = expert
                if rng.random() >= beta:  # let the policy drive, expert still labels
                    act = action(policy, obs)
                obs, _, terminated, truncated, _ = env.step(act)
                if terminated or truncated:
                    break

        # ---- online supervised update ------------------------------------
        loss_running = []
        for _ in range(args.updates_per_round):
            idx = np.random.randint(0, len(buffer), size=args.batch_size)
            obs_b = torch.tensor(
                np.stack([buffer[i][0] for i in idx]), dtype=torch.float32
            )
            act_b = torch.tensor(
                np.stack([buffer[i][1] for i in idx]), dtype=torch.float32
            )
            opt.zero_grad()
            loss = loss_fn(policy(obs_b), act_b)
            loss.backward()
            opt.step()
            loss_running.append(float(loss.item()))
        train_losses.append(float(np.mean(loss_running)))

        # ---- periodic live evaluation ------------------------------------
        msg = ""
        if (r + 1) % args.eval_every == 0 or r == args.rounds - 1:
            ok = 0
            dists_s = []
            for s in range(args.eval_episodes):
                s_ok, s_d = rollout(env, policy, seed=args.seed * 1000 + r * 1000 + s)
                ok += int(s_ok)
                dists_s.append(s_d)
            succ.append(100.0 * ok / args.eval_episodes)
            msg = (f"  success={ok}/{args.eval_episodes}"
                   f" ({100.0 * ok / args.eval_episodes:.0f}%)"
                   f"  dist={np.mean(dists_s):.3f}")
            print(f"{r + 1:5d} | {len(buffer):5d} | {train_losses[-1]:.4f} | {succ[-1]:6.2f}{msg}")
        else:
            print(f"{r + 1:5d} | {len(buffer):5d} | {train_losses[-1]:.4f} |    -")

    env.close()

    # ---- save artifacts ---------------------------------------------------
    policy_path = os.path.join(args.out, "grasp_policy_online.pt")
    torch.save(policy.state_dict(), policy_path)
    if len(train_losses) > 1 and succ:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(train_losses, color="#1f77b4", lw=2, label="train MSE")
        ax1.set_xlabel("round")
        ax1.set_ylabel("train MSE", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax2 = ax1.twinx()
        xs = [i * args.eval_every for i in range(len(succ))]
        ax2.plot(xs, succ, color="#d62728", lw=2, label="grasp success %")
        ax2.set_ylabel("grasp success %", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "supervised_learning_curve.png"), dpi=140)
        print("saved supervised_learning_curve.png + grasp_policy_online.pt")

    if args.visualize:
        viewer_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "view_pick.py"
        )
        cmd = [
            sys.executable, viewer_script,
            "--mode", "policy", "--policy-path", policy_path,
            "--obs-target", args.obs_target,
        ]
        print("opening live viewer:", " ".join(cmd))
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
