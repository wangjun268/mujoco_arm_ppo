"""Render a montage of a trained arm approaching & hitting the target."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cli import add_env_arg, add_model_arg, add_out_arg, apply_defaults, load_policy
from env import make_env as build_env
from paths import ensure_dir

# Steps (in policy time) captured for the montage.
WANTED_STEPS = [0, 2, 6, 12, 20, 40, 60, 90, 120]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_env_arg(ap)
    add_model_arg(ap)
    ap.add_argument("--seed", type=int, default=7)
    add_out_arg(ap)
    args = apply_defaults(ap.parse_args())

    ensure_dir(args.out)
    model = load_policy(args.model)
    env = build_env(args.env, render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)

    shots = {}
    step = 0
    while step < env.max_steps:
        if step in WANTED_STEPS:
            shots[step] = env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, _info = env.step(action)
        step += 1
        if terminated or truncated:
            break
    shots[step] = env.render()
    env.close()

    keys = sorted(shots)
    ncols = min(3, len(keys))
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.4 * nrows))
    axes = axes.flat if nrows * ncols > 1 else [axes]
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(keys):
            ax.imshow(shots[keys[i]])
            ax.set_title(f"t = {keys[i]}", fontsize=11)
    fig.suptitle(f"{args.env} arm reaching the target (trained PPO policy)", fontsize=14)
    fig.tight_layout()
    out = os.path.join(args.out, "montage.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
