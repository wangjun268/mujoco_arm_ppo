"""Live MuJoCo viewer for the Pro7 grasp / pick & place scene.

Opens the native MuJoCo ``launch_passive`` window and drives the arm on red
cubes in real time.  Two tasks (``--task grasp`` / ``--task pick_place``) and
two drivers:
  * ``--mode expert``  — the resolved-rate servo teacher (reliable, ~90 %).
  * ``--mode policy`` — a supervised (DAgger) policy trained by
    ``supervised_grasp.py`` / ``train_live.py`` (load ``--policy-path``).

A translucent green marker tracks the vision-localised cube point.  Close the
window to stop; press ``T`` for a new random cube.

Run::

    python3 view_pick.py --mode expert
    python3 view_pick.py --mode policy --policy-path results/pro7_pick/grasp_policy_online.pt
    python3 view_pick.py --task pick_place --mode policy
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")

from env import make_env  # noqa: E402
from grasp_common import set_marker, teacher_action, teacher_action_place  # noqa: E402
from grasp_policy import OBS_DIM, OBS_DIM_PLACE, action, load_policy  # noqa: E402
from live_viewer import KEY_T, LiveViewer  # noqa: E402
from paths import results_path  # noqa: E402

#: ``task -> (env name, teacher, observation size, default results directory)``.
TASKS = {
    "grasp": ("pro7_pick", teacher_action, OBS_DIM, "pro7_pick"),
    "pick_place": ("pro7_pick_place", teacher_action_place, OBS_DIM_PLACE,
                   "pro7_pick_place"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="grasp", choices=sorted(TASKS))
    ap.add_argument("--mode", default="expert", choices=["expert", "policy"])
    ap.add_argument("--obs-target", default="true", choices=["true", "vision"])
    ap.add_argument("--policy-path", default=None,
                    help="default: results/<task>/grasp_policy_online.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=60.0)
    args = ap.parse_args()

    env_name, expert, obs_dim, out_dir = TASKS[args.task]
    if args.policy_path is None:
        args.policy_path = results_path(out_dir, "grasp_policy_online.pt")
    env = make_env(env_name, obs_target=args.obs_target)
    policy = (load_policy(args.policy_path, obs_dim=obs_dim)
              if args.mode == "policy" else None)

    assert env.ids.marker_mocap >= 0, "marker is not a mocap body"

    reset_flag = {"on": False}

    def key_cb(keycode):
        if keycode == KEY_T:
            reset_flag["on"] = True

    obs, _ = env.reset(seed=args.seed)
    viewer = LiveViewer(env.model, env.data, key_callback=key_cb, fps=args.fps)
    viewer.set_camera("cam_iso")

    episodes = 0
    next_frame = time.time()
    print(f"Opening viewer ({args.mode}); close the window to stop. T = new cube.")
    try:
        while viewer.running:
            act = expert(env) if args.mode == "expert" else action(policy, obs)
            obs, _r, terminated, truncated, _info = env.step(act)

            # Put the detection marker at the localised cube point.
            target, _seen = env.detected_world()
            set_marker(env.data, env.ids, target)
            viewer.sync(force=True)

            next_frame += 1.0 / args.fps
            sleep = next_frame - time.time()
            if sleep > 0:
                time.sleep(sleep)

            if terminated or truncated or reset_flag["on"]:
                reset_flag["on"] = False
                episodes += 1
                obs, _ = env.reset(seed=args.seed + episodes)
    finally:
        viewer.close()
    print(f"viewer closed after {episodes} grasps.")


if __name__ == "__main__":
    main()
