"""Render an approach -> grasp montage of the Pro7 pick scene.

Run::

    python3 grasp/montage.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

# Allow `python3 grasp/montage.py` as well as `python3 -m grasp.montage`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grasp.common import (
    GRIP_CTRL_CLOSED,
    GRIP_CTRL_OPEN,
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=results_path("pro7_pick"))
    ap.add_argument("--ncols", type=int, default=3)
    args = ap.parse_args()

    model, data, _cube_true = make_scene(seed=args.seed)
    ids = scene_ids(model)
    renderer = mujoco.Renderer(model, height=240, width=320)
    overview = mujoco.Renderer(model, height=360, width=480)

    target = detect_cube(model, data, renderer)
    if target is None:
        target = grasp_world(data, ids)

    shots = {}
    want = [0, 12, 30, 55, 80]
    for step in range(140):
        estimate = detect_cube(model, data, renderer)
        if estimate is not None:
            target = estimate

        torques, close, _ = servo_command(model, data, target, ids)
        data.ctrl[: ids.n_arm] = torques
        set_hand(data, ids, GRIP_CTRL_CLOSED if close else GRIP_CTRL_OPEN)
        mujoco.mj_step(model, data)

        if step in want:
            overview.update_scene(data, camera="cam_iso")
            shots[step] = overview.render().copy()

        # Stop on the pinch; this montage only cares about jaws + gap.
        if is_grasped(
            data, ids, dist=0.0, gap=gripper_gap(data, ids), max_dist=np.inf
        ):
            overview.update_scene(data, camera="cam_iso")
            shots[step] = overview.render().copy()
            break

    overview.close()
    renderer.close()

    keys = sorted(shots)
    ncols = min(args.ncols, len(keys))
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = axes.flat if nrows * ncols > 1 else [axes]
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(keys):
            ax.imshow(shots[keys[i]])
            ax.set_title(f"t = {keys[i]}", fontsize=11)
    fig.suptitle("Pro7 7-DOF arm: vision-localised red cube grasp", fontsize=14)
    fig.tight_layout()
    ensure_dir(args.out)
    out = os.path.join(args.out, "grasp_montage.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
