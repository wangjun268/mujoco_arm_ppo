"""Shared command-line plumbing for the entry-point scripts.

Keeps ``--env`` / ``--model`` / ``--out`` identical (and identically
defaulted) across train / eval / viewer / montage, so adding a new script means
adding its own extras rather than copying boilerplate.
"""

from __future__ import annotations

import argparse

from env import env_names
from paths import DEFAULT_ENV, default_model_path, default_out_dir, resolve_model


def add_env_arg(
    parser: argparse.ArgumentParser, default: str = DEFAULT_ENV
) -> argparse.ArgumentParser:
    """Add ``--env`` restricted to the registered environments."""
    parser.add_argument(
        "--env", default=default, choices=env_names(),
        help=f"environment to use (default: {default})",
    )
    return parser


def add_model_arg(
    parser: argparse.ArgumentParser, help: str | None = None
) -> argparse.ArgumentParser:
    """Add ``--model`` for reading/writing ``results/ppo_<env>.zip``."""
    parser.add_argument(
        "--model", default=None,
        help=help or "model path (default: results/ppo_<env>)",
    )
    return parser


def add_out_arg(
    parser: argparse.ArgumentParser, help: str | None = None
) -> argparse.ArgumentParser:
    """Add ``--out`` for the generated artefacts of an environment."""
    parser.add_argument(
        "--out", default=None,
        help=help or "output dir (default: results/<env>)",
    )
    return parser


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill ``--model`` / ``--out`` defaults derived from ``--env``."""
    if getattr(args, "model", None) is None:
        args.model = default_model_path(args.env)
    if hasattr(args, "out") and args.out is None:
        args.out = default_out_dir(args.env)
    return args


def load_policy(model_path: str, device: str = "cpu"):
    """Load a PPO checkpoint for evaluation.

    MLP policies are evaluated on the CPU by default: it matches the saved
    device-agnostic weights, avoids a per-call GPU transfer and silences the
    Stable-Baselines3 "PPO on GPU without a CNN" warning.
    """
    from stable_baselines3 import PPO

    return PPO.load(resolve_model(model_path), device=device)
