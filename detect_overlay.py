"""Visualise the red-cube detection: segmentation + 3D localisation.

Produces a figure with (a) the raw eye-in-hand camera frame, (b) the red-mask /
centroid overlay, and (c) a 3D plot of the arm, camera, gripper, the *detected*
cube point (green) and the ground-truth cube (red), so you can see how well the
vision localisation matches reality.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from detect_red_cube import estimate_cube_world_rgbd, project_world, red_mask
from grasp_common import CUBE_SIDE, TABLE_Z, make_scene
from paths import ensure_dir, results_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=results_path("pro7_pick", "detection_overlay.png"))
    args = ap.parse_args()

    model, data, cube_true = make_scene(seed=args.seed)
    renderer = mujoco.Renderer(model, height=240, width=320)
    renderer.update_scene(data, camera="cam_hand")
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="cam_hand")
    depth = renderer.render()
    renderer.close()

    estimate = estimate_cube_world_rgbd(
        model, data, cam_name="cam_hand", plane_z=TABLE_Z, cube_side=CUBE_SIDE,
        width=320, height=240, img=rgb, depth_img=depth,
    )

    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_hand")
    cam_pos = data.cam_xpos[cam]

    mask = red_mask(rgb)
    blob = None
    if mask.any():
        ys, xs = np.nonzero(mask)
        blob = (float(xs.mean()), float(ys.mean()))

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(rgb)
    ax.set_title("(a) eye-in-hand camera")
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 2)
    overlay = (rgb * 0.35).astype(np.uint8)
    overlay[mask] = rgb[mask]
    ax.imshow(overlay)
    if blob is not None:
        ax.plot([blob[0]], [blob[1]], "x", color="cyan", ms=12, mew=2, label="centroid")
    if estimate is not None:
        ue, ve = project_world(model, data, estimate, "cam_hand",
                               width=rgb.shape[1], height=rgb.shape[0])
        ax.plot([ue], [ve], "o", color="lime", ms=6, label="localised 3D proj.")
    ax.set_title("(b) red mask + centroid + 3D proj.")
    ax.legend(loc="lower right", fontsize=7)
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    gripper_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    ax.scatter(*cam_pos, c="k", s=40, label="camera")
    ax.scatter(*data.xpos[gripper_body][:3], c="b", s=40, label="gripper")
    ax.scatter(*cube_true, c="r", s=120, marker="s", label="true cube")
    if estimate is not None:
        ax.scatter(*estimate, c="lime", s=120, marker="o", label="detected 3D")
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    ax.set_title("(c) detected vs true cube (m)")
    ax.legend(fontsize=7)
    ax.set_box_aspect((1, 1, 1))

    ensure_dir(os.path.dirname(args.out))
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    if estimate is not None:
        print(f"detection error: {np.linalg.norm(estimate - cube_true):.4f} m")
    print("saved", args.out)


if __name__ == "__main__":
    main()
