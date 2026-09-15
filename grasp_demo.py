"""End-to-end validation: detect a red cube, localise it in 3D, grasp it.

Pipeline
--------
1. A red cube sits on a workbench in the MuJoCo scene built by
   :mod:`grasp_common` around the Rokae xMate Pro7 7-DOF arm with an
   eye-in-hand gripper + RGB-D camera.
2. ``detect_red_cube`` segments the red blob and back-projects its depth to
   estimate the cube's world (x, y, z).
3. The resolved-rate expert from :mod:`grasp_common` servos the gripper centre
   onto that point, then closes the pinch gripper.

Run::

    python3 grasp_demo.py                     # 5 random cubes, no video
    python3 grasp_demo.py --episodes 8 --video
"""

from __future__ import annotations

import argparse

import imageio.v2 as imageio
import mujoco
import numpy as np

from grasp_common import (
    GRIP_CTRL_CLOSED,
    GRIP_CTRL_OPEN,
    cube_world,
    detect_cube,
    grasp_world,
    gripper_gap,
    is_grasped,
    make_scene,
    scene_ids,
    set_hand,
    servo_command,
)
from paths import ensure_dir, results_path


def run_grasp(model, data, renderer, last_target=None, max_steps=250):
    """Servo the hand's grasp centre to the detected cube and close the hand.

    Returns ``(grasped, steps_used, last_target)``.
    """
    ids = scene_ids(model)
    target = last_target
    closed_for = 0

    for step in range(max_steps):
        estimate = detect_cube(model, data, renderer)
        if estimate is not None:
            target = estimate
        if target is None:
            target = grasp_world(data, ids)

        torques, close, _ = servo_command(model, data, target, ids)
        data.ctrl[: ids.n_arm] = torques
        set_hand(data, ids, GRIP_CTRL_CLOSED if close else GRIP_CTRL_OPEN)
        if close:
            closed_for += 1
        mujoco.mj_step(model, data)

        # Success: enough fingers contact the cube while closed & near it.  The
        # gate counts *cumulative* close-range steps (matching the original
        # demo), so leaving it un-reset keeps the reported success rate stable.
        if closed_for >= 15:
            dist = float(np.linalg.norm(grasp_world(data, ids) - cube_world(data, ids)))
            if is_grasped(data, ids, dist, gripper_gap(data, ids)):
                return True, step, target
    return False, max_steps, target


def main():
    ap = argparse.ArgumentParser(description="Red-cube 3D detection + 7-DOF grasp.")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true", help="save a rollout gif")
    ap.add_argument("--out", default=results_path("pro7_pick"))
    args = ap.parse_args()

    ensure_dir(args.out)
    successes = 0
    det_errs = []
    frames = []
    for ep in range(args.episodes):
        model, data, cube_true = make_scene(seed=args.seed + ep)
        renderer = mujoco.Renderer(model, height=240, width=320)
        # Report detection accuracy at the start.
        est0 = detect_cube(model, data, renderer)
        det_err = float(np.linalg.norm(est0 - cube_true)) if est0 is not None else None

        ok, steps, last_target = run_grasp(model, data, renderer, last_target=est0)
        successes += int(ok)
        if det_err is not None:
            det_errs.append(det_err)
        print(
            f"ep {ep}: cube_true={np.round(cube_true, 3)}"
            f" detected={np.round(est0, 3) if est0 is not None else None}"
            f"  det_err={det_err if det_err is None else round(det_err, 3)}"
            f"  grasp_ok={ok} in {steps} steps"
        )
        if args.video:
            vid = mujoco.Renderer(model, height=360, width=480)
            for _ in range(120):
                vid.update_scene(data, camera="cam_iso")
                frames.append(vid.render())
                mujoco.mj_step(model, data)
            vid.close()
        renderer.close()

    if det_errs:
        print(
            f"\ndetection 3D error: mean {np.mean(det_errs):.3f} m"
            f"  max {np.max(det_errs):.3f} m"
        )
    print(f"grasp success: {successes}/{args.episodes}")
    if args.video and frames:
        gif = f"{args.out}/grasp_demo.gif"
        imageio.mimsave(gif, frames, fps=30)
        print("saved", gif)


if __name__ == "__main__":
    main()
