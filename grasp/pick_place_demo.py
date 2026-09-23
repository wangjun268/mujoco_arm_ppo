"""Pick-and-place demo on the two-bench cell.

The arm grasps the red cube on the source bench (with green/blue/yellow
distractor blocks around it) and puts it down on the drop-off pad of the
destination bench.  Physics + vision are the same code paths as
:mod:`grasp.demo` (:mod:`grasp.common`), only the task is different: after pinching the cube the
expert lifts it, carries it across and releases it.

Run::

    python3 grasp/pick_place_demo.py --episodes 5
    python3 grasp/pick_place_demo.py --episodes 3 --video --montage
"""

from __future__ import annotations

import argparse
import os
import sys

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

# Allow `python3 grasp/pick_place_demo.py` as well as `python3 -m grasp.pick_place_demo`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grasp.common as gc
from paths import ensure_dir, results_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true",
                    help="record the first episode to pick_place_demo.gif")
    ap.add_argument("--montage", action="store_true",
                    help="save a phase storyboard to pick_place_montage.png")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default=results_path("pro7_pick_place"))
    args = ap.parse_args()
    ensure_dir(args.out)

    successes = 0
    final_dists = []
    frames, shots = [], {}

    for episode in range(args.episodes):
        model, data, cube_start = gc.make_scene(seed=args.seed + episode)
        ids = gc.scene_ids(model)
        detect_renderer = mujoco.Renderer(model, height=240, width=320)
        view = mujoco.Renderer(model, height=360, width=480)
        record = args.video and episode == 0

        def snapshot() -> np.ndarray:
            view.update_scene(data, camera="cam_iso")
            return view.render().copy()

        counter = {"n": 0}

        def on_step() -> None:
            if not record:
                return
            counter["n"] += 1
            if counter["n"] % 3 == 0:  # thin the movie down to a sane length
                frames.append(snapshot())

        def on_phase(name: str) -> None:
            if args.montage and episode == 0:
                shots[name] = snapshot()

        estimate = gc.detect_cube(model, data, detect_renderer)
        result = gc.pick_and_place(
            model, data, ids, detect_renderer,
            cube_hint=estimate, on_step=on_step, on_phase=on_phase,
        )
        if args.montage and episode == 0:
            shots["done"] = snapshot()
        if record:
            frames.append(snapshot())

        cube_end = gc.cube_world(data, ids)
        miss = float(np.linalg.norm(cube_end - gc.PLACE_TARGET))
        final_dists.append(miss)
        successes += int(result.success)
        print(
            f"ep {episode}: 抓取={result.picked} 放置={result.placed}"
            f"  步数={result.steps:4d}  方块 {np.round(cube_start, 3)} ->"
            f" {np.round(cube_end, 3)}  距放置点 {miss * 1000:5.1f} mm  ({result.reason})"
        )
        detect_renderer.close()
        view.close()

    print(f"\n取放成功 {successes}/{args.episodes}"
          f"  平均落点误差 {np.mean(final_dists) * 1000:.1f} mm"
          f"  （判定阈值 {gc.PLACE_TOLERANCE * 1000:.0f} mm）")

    if frames:
        gif = f"{args.out}/pick_place_demo.gif"
        imageio.mimsave(gif, frames, fps=args.fps)
        print(f"saved {gif} ({len(frames)} frames)")

    if shots:
        order = ["approach", "lift", "transport", "lower", "release", "done"]
        keys = [k for k in order if k in shots]
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.6))
        axes = np.atleast_1d(axes)
        for ax, key in zip(axes, keys):
            ax.imshow(shots[key])
            ax.set_title(key, fontsize=12)
            ax.axis("off")
        fig.suptitle("Pro7 pick & place: source bench -> destination bench", fontsize=14)
        fig.tight_layout()
        png = f"{args.out}/pick_place_montage.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        print(f"saved {png}")


if __name__ == "__main__":
    main()
