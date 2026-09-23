"""Central project paths, workspace defaults and checkpoint resolution.

Every entry point locates the repository, the MuJoCo assets and the generated
results through this module.  Nothing else in the project should hard-code an
absolute path, so the checkout can be moved or renamed (and the scripts run
from any working directory) without editing a single line of logic.

This file therefore stays at the checkout root: ``PROJECT_ROOT`` *is* the
directory holding it, which is also how ``ros2_ws`` finds the project and how
``grasp/`` and ``tools/`` scripts (run as plain scripts) get the root on
``sys.path``.
"""

from __future__ import annotations

import os
import sys

#: Root of the checkout (the directory holding this file).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
#: MuJoCo XML models live here.
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
#: Models, logs, videos and curves are written here.
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

#: Environment used when a script is called without ``--env``.
DEFAULT_ENV = "pro7_urdf"


def asset_path(name: str) -> str:
    """Absolute path of a MuJoCo model inside ``assets/``."""
    return os.path.join(ASSETS_DIR, name)


def results_path(*parts: str) -> str:
    """Absolute path inside the shared ``results/`` directory."""
    return os.path.join(RESULTS_DIR, *parts)


def default_model_path(env: str) -> str:
    """Where ``train_ppo.py`` saves (and every other script loads) ``env``."""
    return results_path(f"ppo_{env}")


def default_tb_dir(env: str) -> str:
    """TensorBoard event directory for ``env``."""
    return results_path(f"tb_{env}")


def default_out_dir(env: str) -> str:
    """Directory holding the videos / curves / montages of ``env``."""
    return results_path(env)


def ensure_dir(path: str) -> str:
    """Create ``path`` as a directory (if needed) and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def resolve_model(model_path: str) -> str:
    """Return an absolute ``.zip`` model path or exit with a clear message.

    Turns the two common mistakes (passing a directory, or a checkpoint that was
    never trained) into an actionable error instead of an ``IsADirectoryError``
    or a bare ``FileNotFoundError``.
    """
    p = str(model_path)
    if os.path.isdir(p):
        sys.exit(
            f"ERROR: the model path is a directory, not a model file: {p!r}\n"
            f"       Pass a model name, e.g.  --model results/ppo_pro7_urdf\n"
            f"       (or the full file results/ppo_pro7_urdf.zip)."
        )
    zp = p if p.endswith(".zip") else p + ".zip"
    if not os.path.isfile(zp):
        sys.exit(
            f"ERROR: model file not found: {zp}\n"
            f"       Train it first:  python3 train_ppo.py --env <env> --steps 600000\n"
            f"       or point --model at an existing checkpoint."
        )
    return os.path.abspath(zp)
