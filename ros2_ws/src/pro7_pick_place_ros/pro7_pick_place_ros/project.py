"""Locate the ``mujoco_arm_ppo`` checkout and expose its Python modules.

This package is a *driver*, not a re-implementation: the scene, the vision
pipeline and the scripted expert all live in the plain-Python project at the
checkout (``grasp/common.py``, ``grasp/detect.py``, ``paths.py``, ...).  The
node therefore only has to find that root and put it on ``sys.path``; no copy
of the scene, the controller or the hand calibration lives in the ROS package,
so tuning the project keeps working for the node.

Resolution order:

1. the ``project_root`` parameter / explicit argument,
2. the ``MUJOCO_ARM_PPO_ROOT`` environment variable,
3. walking up from this file (true for an in-tree checkout *and* for the
   ``install/`` copy, because ``ros2_ws`` sits inside the checkout).
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Iterator, Optional, Tuple

#: Files that must exist for a directory to be the checkout root.
_MARKERS = ("grasp/common.py", "grasp/detect.py", "paths.py")


def looks_like_project(path: str) -> bool:
    """True when ``path`` holds the project's Python modules."""
    return bool(path) and all(
        os.path.isfile(os.path.join(path, marker)) for marker in _MARKERS
    )


def _candidates(explicit: Optional[str]) -> Iterator[str]:
    if explicit:
        yield explicit
    env = os.environ.get("MUJOCO_ARM_PPO_ROOT")
    if env:
        yield env
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        yield current
        parent = os.path.dirname(current)
        if parent == current:
            return
        current = parent


def find_project_root(explicit: Optional[str] = None) -> str:
    """Absolute path of the checkout, or ``RuntimeError`` when it cannot be found."""
    seen = set()
    for candidate in _candidates(explicit):
        candidate = os.path.abspath(os.path.expanduser(candidate))
        if candidate in seen:
            continue
        seen.add(candidate)
        if looks_like_project(candidate):
            return candidate
    raise RuntimeError(
        "mujoco_arm_ppo checkout not found; pass the 'project_root' parameter "
        "(or set MUJOCO_ARM_PPO_ROOT) to the directory that holds the grasp/ package"
    )


def import_project(project_root: str) -> Tuple[object, object]:
    """Put ``project_root`` on ``sys.path``; return ``(grasp.common, paths)``."""
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return importlib.import_module("grasp.common"), importlib.import_module("paths")


def import_module(name: str, project_root: str):
    """Import ``name`` from the checkout (``grasp.detect``, ``env``, ...)."""
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return importlib.import_module(name)
