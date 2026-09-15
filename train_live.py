"""Real-time training-process visualisation for the supervised pick & place policy.

Runs the online DAgger training and, every evaluation round, renders the current
policy *doing the task* - the whole storyboard, approach -> pinch -> lift ->
carry -> place on the pad - plus a live metric panel (train MSE, pick and place
success).  The composited frames are appended to ``training_process.gif`` so you
can watch the policy go from failing to placing in real time.  With ``--live``
it also opens the native MuJoCo window (best effort, needs a display).

The default task is ``pick_place``: ``env.RokaePro7PickPlace`` keeps the episode
running past the pinch, so the policy is trained (and scored) on the *whole*
cell instead of stopping the moment the cube is grabbed.  ``--task grasp`` runs
the original grasp-only environment (episode over once the jaws close).

The carry is gentle by design (the pinch holds ~0.3 N, so anything above
~2 cm/s squeezes the cube out of the jaws - see :mod:`grasp_common`), which
makes one episode ~2200-3000 steps: at ``--live-speed 1`` that is ~50 s per
round.  Use ``--live-speed 4``, or fewer ``--rounds``, to get through the
training faster.

Run::

    python3 train_live.py --rounds 8
    python3 train_live.py --rounds 24 --live
    python3 train_live.py --task grasp --rounds 24 --live
"""

from __future__ import annotations

import argparse
import os
from collections import deque

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import torch.nn as nn

from env import make_env
from grasp_common import teacher_action, teacher_action_place
from grasp_policy import Policy, action, rollout_info
from live_viewer import LiveViewer, Pacer
from paths import ensure_dir, results_path

#: ``task -> (environment, expert teacher, default results directory)``.
TASKS = {
    "pick_place": ("pro7_pick_place", teacher_action_place, "pro7_pick_place"),
    "grasp": ("pro7_pick", teacher_action, "pro7_pick"),
}

#: Per-task defaults for the knobs whose sensible value depends on the task.
#: The pick & place carry is quasi-static (~2 cm/s) and an episode runs for
#: ~2500 steps, so 1x playback is already slow - and a 24-round run is ~20 min.
#: The grasp task takes <1 s of real time per episode, so it plays back at half
#: speed to stay watchable.
TASK_DEFAULTS = {
    # One pick & place episode already fills the buffer with ~2500 samples, and
    # that gentle transport needs ~400 updates to imitate, hence the very
    # different numbers from the (60-step) grasp task.
    "pick_place": {"collect_episodes": 1, "eval_episodes": 10,
                   "updates_per_round": 400, "live_speed": 1.0,
                   "beta_min": 0.5, "min_collect_steps": 2500},
    "grasp": {"collect_episodes": 3, "eval_episodes": 20,
              "updates_per_round": 50, "live_speed": 0.5,
              "beta_min": 0.2, "min_collect_steps": 0},
}


def compose_frame(scene_imgs, history, task):
    """Composite the policy storyboard (left) with the live metrics (right)."""
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(3, 4)
    for i, img in enumerate(scene_imgs[:6]):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.set_title(f"t={i}", fontsize=8)
        ax.axis("off")

    rounds = history["round"]
    ax = fig.add_subplot(gs[:, 2:])
    ax.plot(rounds, history["mse"], color="#1f77b4", lw=2, label="train MSE")
    ax.set_xlabel("round")
    ax.set_ylabel("train MSE", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    if task == "pick_place":
        ax2.plot(rounds, history["pick"], color="#2ca02c", lw=2, label="pick %")
        ax2.plot(rounds, history["place"], color="#d62728", lw=2, label="place %")
    else:
        ax2.plot(rounds, history["pick"], color="#d62728", lw=2, label="grasp %")
    ax2.set_ylabel("success %", color="#d62728")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def capture_rollout_frames(env, policy, renderer, seed, n_frames=6):
    """Run the policy on one cube, capture ``n_frames`` frames of the episode.

    For the pick & place task the frames are taken when the *phase* changes, so
    the storyboard really shows approach -> pinch -> lift -> carry -> lower ->
    release instead of six stills of the long quasi-static carry.  The grasp
    task has no phases, so it samples the episode evenly.
    """
    obs, _info = env.reset(seed=seed)

    def shot() -> np.ndarray:
        renderer.update_scene(env.data, camera="cam_iso")
        return renderer.render().copy()

    phased = hasattr(env, "phase")
    stride = max(1, env.max_steps // 60)  # ~30 MB of frames, whatever the length
    phase = getattr(env, "phase", None)
    all_frames = [shot()]
    for t in range(env.max_steps):
        if not phased and t % stride == 0:
            all_frames.append(shot())
        obs, _r, terminated, truncated, info = env.step(action(policy, obs))
        if phased and env.phase != phase:
            phase = env.phase
            all_frames.append(shot())
        if terminated or truncated:
            break
    all_frames.append(shot())
    while len(all_frames) < n_frames:  # failed early: repeat the last frame
        all_frames.insert(0, all_frames[0])
    idx = np.round(np.linspace(0, len(all_frames) - 1, n_frames)).astype(int)
    return [all_frames[i] for i in idx], info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="pick_place", choices=sorted(TASKS),
                    help="pick_place trains the whole cell (default); grasp "
                         "stops the episode on the pinch")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--collect-episodes", type=int, default=None,
                    help="episodes per round (default: 1 pick & place / 3 grasp)")
    ap.add_argument("--min-collect-steps", type=int, default=None,
                    help="keep starting episodes until this many steps have been "
                         "collected (a failed pick ends the episode early, which "
                         "would otherwise starve the buffer). "
                         "Default: 2500 pick & place / 0 grasp")
    ap.add_argument("--beta-min", type=float, default=None,
                    help="DAgger floor: probability of following the expert "
                         "(default: 0.5 pick & place / 0.2 grasp)")
    ap.add_argument("--updates-per-round", type=int, default=None,
                    help="supervised updates per round "
                         "(default: 400 pick & place / 50 grasp)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--buffer-size", type=int, default=60000)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-episodes", type=int, default=None,
                    help="rollouts per evaluation (default: 10 pick & place / 20 grasp)")
    ap.add_argument("--obs-target", default="true", choices=["true", "vision"])
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--live", action="store_true", help="open the MuJoCo window too")
    ap.add_argument("--live-fps", type=float, default=60.0,
                    help="UI refresh rate of the live window")
    ap.add_argument("--live-speed", type=float, default=None,
                    help="playback speed of the live window (1 = real time, "
                         "0 = train as fast as possible; < 1 shows the motion "
                         "in slow motion). Default: 0.5 grasp / 1 pick & place")
    ap.add_argument("--out", default=None,
                    help="output directory (default: results/<task>)")
    args = ap.parse_args()

    env_name, expert, out_default = TASKS[args.task]
    defaults = TASK_DEFAULTS[args.task]
    for key, value in defaults.items():  # per-task default, CLI wins
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.out is None:
        args.out = results_path(out_default)

    ensure_dir(args.out)
    env = make_env(env_name, obs_target=args.obs_target)
    policy = Policy(obs_dim=int(np.prod(env.observation_space.shape)))
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    buffer = deque(maxlen=args.buffer_size)
    renderer = mujoco.Renderer(env.model, height=360, width=480)

    viewer = None
    pacer = None
    if args.live:
        try:
            viewer = LiveViewer(env.model, env.data, camera="cam_iso",
                                fps=args.live_fps)
            pacer = Pacer(args.live_speed, sim_dt=env.model.opt.timestep)
            pace = ("as fast as possible" if not pacer.paced
                    else f"{args.live_speed:g}x real time")
            print(f"live MuJoCo window opened - playback {pace}"
                  "  (close it and training carries on headless)")
        except Exception as ex:
            print("could not open live viewer:", ex)
            viewer = None

    print(f"task={args.task}  env={env_name}")
    history = {"round": [], "mse": [], "pick": [], "place": [], "dist": []}
    gif_frames = []
    # DAgger on this plant is noisy round to round (the pinch is knife-edge), so
    # the checkpoint that is kept is the best evaluation, not the last one.
    best = {"score": None, "state": None}
    if args.task == "pick_place":
        print("round | data  | train_mse | pick % | place % | final_dist")
    else:
        print("round | data  | train_mse | grasp % | final_dist")
    try:
        train(env, policy, opt, loss_fn, buffer, args, expert, viewer, pacer,
              renderer, history, gif_frames, best)
    finally:
        # Always tear the window down (an exception in the loop would otherwise
        # leave the UI thread alive and the process hanging on exit).
        if viewer is not None:
            viewer.close()

    env.close()
    renderer.close()

    state = policy.state_dict() if best["state"] is None else best["state"]
    torch.save(state, os.path.join(args.out, "grasp_policy_online.pt"))
    if gif_frames:
        gif = os.path.join(args.out, "training_process.gif")
        imageio.mimsave(gif, gif_frames, fps=2)
        print(f"saved {gif} ({len(gif_frames)} process frames)")
    print("saved grasp_policy_online.pt")


def train(env, policy, opt, loss_fn, buffer, args, expert, viewer, pacer,
          renderer, history, gif_frames, best=None):
    """Run the DAgger rounds; extracted so ``main`` can always close the viewer."""
    lost_window = False

    def live_sync():
        """Keep the window fresh and report the first time it disappears."""
        nonlocal lost_window
        if viewer is None or lost_window:
            return
        if not viewer.sync():
            lost_window = True
            print("[live] window closed - training continues headless.")

    for r in range(args.rounds):
        beta = args.beta_min + (1.0 - args.beta_min) * max(
            0.0, 1.0 - r / max(1, args.rounds)
        )
        rng = np.random.default_rng(r + 137)
        added = 0
        episodes = 0
        # A pick that fails knocks the cube off the bench and the episode ends
        # after ~150 steps, so a fixed episode count can collect almost nothing;
        # keep going until the round has enough fresh data (with a hard cap so a
        # hopeless policy cannot spin here forever).
        while (episodes < args.collect_episodes
               or (added < args.min_collect_steps
                   and episodes < 6 * args.collect_episodes)):
            episodes += 1
            obs, _ = env.reset(seed=int(args.seed * 1000 + r * 100 + rng.integers(0, 1 << 30)))
            for _ in range(env.max_steps):
                label = expert(env)
                buffer.append((obs.copy(), label.copy()))
                added += 1
                act = label
                if rng.random() >= beta:
                    act = action(policy, obs)
                obs, _r, terminated, truncated, _ = env.step(act)
                if viewer is not None:
                    if pacer is not None:
                        pacer.tick()
                    live_sync()
                if terminated or truncated:
                    break

        loss_run = []
        for _ in range(args.updates_per_round):
            idx = np.random.randint(0, len(buffer), size=args.batch_size)
            ob = torch.tensor(np.stack([buffer[i][0] for i in idx]), dtype=torch.float32)
            ac = torch.tensor(np.stack([buffer[i][1] for i in idx]), dtype=torch.float32)
            opt.zero_grad()
            loss = loss_fn(policy(ob), ac)
            loss.backward()
            opt.step()
            loss_run.append(float(loss.item()))
        mse = float(np.mean(loss_run))

        if (r + 1) % args.eval_every == 0 or r == args.rounds - 1:
            picks = 0
            places = 0
            dists = []
            for s in range(args.eval_episodes):
                # Evaluation is a metric, not something to watch: run it at full
                # speed but keep publishing frames so the window never freezes.
                info = rollout_info(env, policy, seed=args.seed * 1000 + r * 1000 + s,
                                    on_step=live_sync)
                picks += int(info.get("picked", info.get("grasped", False)))
                places += int(info.get("placed", False))
                dists.append(float(info.get("place_dist", info["dist_to_cube"])))
            pick_pct = 100.0 * picks / args.eval_episodes
            place_pct = 100.0 * places / args.eval_episodes
            dist = float(np.mean(dists))
            history["round"].append(r + 1)
            history["mse"].append(mse)
            history["pick"].append(pick_pct)
            history["place"].append(place_pct)
            history["dist"].append(dist)
            if viewer is not None and not lost_window:
                viewer.set_status([
                    f"round {r + 1}/{args.rounds}",
                    f"train MSE {mse:.4f}",
                    f"pick {pick_pct:.0f}%  place {place_pct:.0f}%",
                    f"dist {dist:.3f} m",
                ])

            if best is not None:
                score = (place_pct, pick_pct, -dist)
                if best["score"] is None or score > best["score"]:
                    best["score"] = score
                    best["state"] = {
                        k: v.detach().clone() for k, v in policy.state_dict().items()
                    }
                    if args.task == "pick_place":
                        print(f"      * best so far: place {place_pct:.0f}%"
                              f"  pick {pick_pct:.0f}%")
                    else:
                        print(f"      * best so far: grasp {pick_pct:.0f}%")

            # Capture the current policy doing the whole task.
            frames, _info = capture_rollout_frames(env, policy, renderer,
                                                   seed=args.seed * 5 + r)
            gif_frames.append(compose_frame(frames, history, args.task))
            if viewer is not None and not lost_window:
                viewer.sync(force=True)
            if args.task == "pick_place":
                print(
                    f"{r + 1:5d} | {len(buffer):5d} | {mse:8.4f} |"
                    f" {pick_pct:5.0f}% | {place_pct:6.0f}% | {dist:.3f}"
                )
            else:
                print(
                    f"{r + 1:5d} | {len(buffer):5d} | {mse:8.4f} |"
                    f" {pick_pct:5.0f}% | {dist:.3f}"
                )


if __name__ == "__main__":
    main()
